"""OpenSpec task execution with persisted, same-tier fallback decisions."""

import asyncio
import time
import os
from dataclasses import replace
from collections.abc import AsyncIterator
from collections.abc import Awaitable, Callable

import providers

from agent.prompt import (CoderPromptContext, InlineCoderPromptBuilder,
                          InlineCoderPromptContext, PromptFactory, PromptRole)
from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.evidence import is_verification_call, verification_succeeded
from agent.routing.gateway import brief_failure, gateway_attempts, same_credential_limit
from agent.routing.inline_execution import InlineCoder, InlineOutcome, InlineRequest
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.models import ExecutionContext, ExecutionDecision
from agent.routing.persistence import RoutingPersistence
from agent.routing.rollback import RollbackJournal
from agent.routing.service import ModelRoutingService
from agent.spec.models import Task
from agent.workspace import (NATIVE_SHELL_TOOL, READ_ONLY_CAPABILITIES, NativeRun,
                             RegisteredProject, command_result, plan_native,
                             project_inventory, read_only_session, toolchain)
from mcp_bridge.models import (
    DoneEvent, ErrorEvent, Event, MutationEvent, ProgressEvent, Run, TextEvent,
    ToolCallEvent,
)
from mcp_bridge.registry import ToolSpec
from providers import subagents
from providers.compaction import compaction_brief
from telemetry import tracing


class TaskExecutionService:
    strict_handoff = True

    def __init__(self, config: RoutingConfig | None = None,
                 router: ModelRoutingService | None = None,
                 persistence: RoutingPersistence | None = None,
                 executors: ExecutorRegistry | None = None):
        self.config = config or load_routing_config()
        self.router = router or ModelRoutingService(self.config)
        self.persistence = persistence or RoutingPersistence()
        self.executors = executors or ExecutorRegistry()

    async def execute(self, task: Task, change_id: str, messages: list[dict],
                      specs: list[ToolSpec],
                      on_decision: Callable[[ExecutionDecision], Awaitable[None]] | None = None,
                      execution_instruction: str = "",
                      direct_tool_handlers: dict | None = None,
                      exclude_models: frozenset[str] = frozenset(),
                      contract=None, owner_metadata: dict | None = None,
                      plan_context: str = "",
                      handoff_context: str = "",
                      exploration_scope: tuple[str, ...] = (),
                      exploration_read_only: tuple[str, ...] = (),
                      inline: InlineRequest | None = None,
                      ) -> AsyncIterator[Event]:
        # The active project belongs to the IDE request that advertised `specs`.
        # Never route through a process-wide IDE/project setting.
        with tracing.routing_span("coder", task.complexity, exclude_models) as span:
            decision = await self.router.route(task, change_id, "bridge-request",
                                               exclude=exclude_models)
            tracing.record_routing(
                span, source=decision.routing_source, provider=decision.recommended_provider,
                model=decision.recommended_model, candidates=decision.candidates,
                fallbacks=decision.fallback_models, task_id=decision.task_id,
                decision_id=decision.decision_id)
        await self.persistence.record_decision(decision)
        if on_decision is not None:
            await on_decision(decision)
        excluded_files = tuple(getattr(contract, "excluded_files", ()) or ())
        if inline is not None:
            outcome = InlineOutcome()
            prompt = InlineCoderPromptBuilder().with_context(InlineCoderPromptContext(
                task, inline.source.text, inline.edit_files, execution_instruction, contract,
                plan_context, inline.checks)).build().user_message()
            prefix = [message for message in messages[:-1] if message.get("role") == "system"]
            async for event in InlineCoder(self.config, self.persistence, self.executors,
                                           self.router.health).run(
                    task, decision, inline, [*prefix, prompt], owner_metadata or {},
                    excluded_files, outcome):
                yield event
            if outcome.fallback is None:
                return
            yield ProgressEvent(f"- ↩️ inline: {outcome.fallback}; continuing as a "
                                "tool-using session from the current files\n")
        discovery = getattr(self.config, "discovery", None)
        policy = getattr(discovery, "policy", "selective")
        # Native attempts delegate exploration (within the planner-assigned files) and long
        # test runs to subagents on the configured cheap model.
        subagent_request = subagents.request(
            "task", scope=exploration_scope, models=self.config.subagent_models(),
            max_turns=self.config.subagents.max_turns, read_only=exploration_read_only)
        async for event in self._run(
                task, decision, self._execution_messages(task, messages, execution_instruction,
                                                         contract, plan_context, handoff_context,
                                                         kb_policy=policy),
                specs, direct_tool_handlers or {},
                {**(owner_metadata or {}),
                 "enforce_scope": getattr(self.config.subagents, "enforce_scope", True),
                 # Custom instructions for the CLI's own compaction, not a summarizer.
                 "compaction_brief": compaction_brief(task, contract),
                 # The plan's files: the edits' preimages are kept for the review diff,
                 # and reads outside the scope are counted.
                 **({"edit_files": list(exploration_scope),
                     "scope_files": list(dict.fromkeys(
                         (*exploration_scope, *exploration_read_only)))}
                    if exploration_scope else {})},
                native_messages=self._execution_messages(
                    task, messages, execution_instruction, contract, plan_context,
                    handoff_context,
                    native=True, kb_policy=policy, subagent_request=subagent_request),
                excluded_files=excluded_files,
                subagent_request=subagent_request):
            yield event

    @staticmethod
    def _execution_messages(task: Task, messages: list[dict],
                            execution_instruction: str = "", contract=None,
                            plan_context: str = "", handoff_context: str = "",
                            native: bool = False,
                            kb_policy: str = "selective",
                            subagent_request: subagents.Request | None = None,
                            aci_enabled: bool | None = None) -> list[dict]:
        """Replace the transport command with the actual OpenSpec task objective."""
        if aci_enabled is None:
            aci_enabled = os.getenv("CODEPARTNER_ACI_ENABLED", "true").lower() != "false"
        prompt = PromptFactory.create(PromptRole.CODER).with_context(CoderPromptContext(
            task, execution_instruction, contract, plan_context, handoff_context,
            native=native, kb_policy=kb_policy, subagent_request=subagent_request,
            aci_enabled=aci_enabled)).build()
        prefix = [message for message in messages[:-1] if message.get("role") == "system"]
        return [*prefix, prompt.user_message()]

    def _native_run(self, project: RegisteredProject, owner_metadata: dict, task=None,
                    excluded_files: tuple[str, ...] = ()) -> NativeRun:
        """Native settings for one attempt. Writing attempts are journaled for review
        and evidence; read-only tasks, and attempts without an implementation job
        position to journal against, run as read-only sessions."""
        job_id = owner_metadata.get("implementation_job_id")
        position = owner_metadata.get("implementation_position")
        capability = str(getattr(task, "preferred_capability", "") or "").lower()
        if job_id is None or position is None or capability in READ_ONLY_CAPABILITIES:
            return read_only_session(project)
        journal = RollbackJournal(self.persistence, job_id, position, [], {})
        env, notes = toolchain(project)

        async def on_command(command: str, output: str, exit_code: int | None) -> None:
            if is_verification_call({"command": command}):
                result = command_result(output, exit_code)
                await journal.record_verification(
                    NATIVE_SHELL_TOOL, {"command": command}, result,
                    exit_code == 0 if exit_code is not None
                    else verification_succeeded(result))

        return NativeRun(
            workspace=project.workspace, host_path=project.host_path, env=env, notes=notes,
            start=lambda: journal.record_git_baseline(
                project.workspace, tuple(owner_metadata.get("edit_files") or ())),
            finish=lambda: journal.record_native_changes(excluded_files),
            on_command=on_command)

    async def resume(self, run: Run, messages: list[dict],
                     specs: list[ToolSpec]) -> AsyncIterator[Event]:
        provider = providers.get(run.metadata.get("provider"))
        started = time.monotonic()
        failure: str | None = None
        terminal = False
        paused = False
        try:
            async for event in provider.tools(messages, specs):
                if isinstance(event, ErrorEvent):
                    failure = event.message
                elif isinstance(event, ToolCallEvent):
                    paused = True
                if isinstance(event, DoneEvent):
                    terminal = True
                    if event.terminal in {"failed", "error"} and failure is None:
                        failure = f"coder finished with {event.terminal}"
                        yield ErrorEvent(failure)
                    await self._finalize_metadata(run, event, started, failure)
                yield event
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            yield ErrorEvent(failure)
        if not terminal and not paused:
            failure = failure or "coder stopped without a terminal result"
            await self._finalize_metadata(run, DoneEvent("failed"), started, failure)
            if failure == "coder stopped without a terminal result":
                yield ErrorEvent(failure)

    async def _run(self, task, decision, messages, specs,
                   direct_tool_handlers: dict,
                   owner_metadata: dict | None = None,
                   native_messages: list[dict] | None = None,
                   excluded_files: tuple[str, ...] = (),
                   subagent_request=None) -> AsyncIterator[Event]:
        attempts = (decision.recommended_model, *decision.fallback_models)
        last_error: str | None = None
        for index, model_name in enumerate(attempts):
            model = self.config.models[model_name]
            # `auto` falls back to the IDE path; an explicit `native` mode fails instead.
            plan = plan_native(model.provider, excluded_files)
            native_project = plan.project
            for gateway_used in gateway_attempts(model.gateway_mode):
                native = (self._native_run(native_project, owner_metadata or {}, task,
                                           excluded_files)
                          if native_project is not None else None)
                if native is None and plan.mode == "native":
                    reason = plan.reason or "native execution is unavailable"
                    yield ErrorEvent(f"Native execution was requested but is unavailable: "
                                     f"{reason}. Nothing was executed.")
                    return
                if native is not None:
                    yield ProgressEvent(f"- 📂 {model.name} works natively in project "
                                        f"{native_project.id}\n"
                                        + "".join(f"- ⚠️ {note}\n" for note in native.notes))
                started = time.monotonic()
                await self.persistence.record_running_model(
                    decision, model.name, model.provider, gateway_used)
                context = ExecutionContext(
                    messages=messages, specs=specs, decision=decision,
                    gateway_url=self.config.agentgateway_url, use_gateway=gateway_used,
                    started_at=started,
                    direct_tool_handlers=direct_tool_handlers,
                    owner_metadata={**(owner_metadata or {}),
                                    **({"execution_mode": "native",
                                        "workspace_project": native_project.id}
                                       if native is not None else {"execution_mode": "ide"})},
                    native=native,
                    subagents=subagent_request if native is not None else None,
                    prompt_cache_ttl=self.config.execution_cache_ttl,
                )
                if native is not None:
                    inventory = await asyncio.to_thread(project_inventory,
                                                        native_project.workspace)
                    context = replace(context, messages=PromptFactory.create(
                        PromptRole.CODER).with_workspace(
                        native_messages or messages, native_project.workspace, inventory))
                executor = self.executors.get(model.executor)
                attempt_error: str | None = None
                mutation_seen = False
                text_seen = False
                paused_for_tool = False
                terminal: DoneEvent | None = None
                try:
                    async for event in executor.execute(task, model, context):
                        if isinstance(event, ErrorEvent):
                            attempt_error = event.message
                        elif isinstance(event, MutationEvent):
                            mutation_seen = True
                        elif isinstance(event, ToolCallEvent):
                            paused_for_tool = True
                        elif isinstance(event, DoneEvent):
                            terminal = event
                        elif isinstance(event, TextEvent):
                            text_seen = True
                            yield event
                        elif isinstance(event, ProgressEvent):
                            yield event
                        if isinstance(event, (MutationEvent, ToolCallEvent)):
                            yield event
                except Exception as exc:
                    attempt_error = f"{type(exc).__name__}: {exc}"
                if paused_for_tool:
                    # The provider is still waiting for the IDE result. A later
                    # request resumes this exact run before another agent may start.
                    return
                if terminal is None and attempt_error is None:
                    attempt_error = "coder stopped without a terminal result"
                elif (terminal is not None and terminal.terminal in {"failed", "error"}
                      and attempt_error is None):
                    attempt_error = f"coder finished with {terminal.terminal}"
                latency = int((time.monotonic() - started) * 1000)
                status = "failed" if attempt_error else "completed"
                await self.persistence.record_attempt(
                    decision, model.name, model.provider, model.executor,
                    gateway_used, status, latency, attempt_error,
                    terminal.usage if terminal else None)
                await self.persistence.finalize(
                    decision, model.name, model.provider, gateway_used,
                    status, latency, "executor-fallback" if index else None)
                self.router.health.record(model.name, model.executor, gateway_used,
                                          status == "completed")
                if status == "completed":
                    yield terminal
                    return
                last_error = attempt_error
                if mutation_seen or text_seen:
                    # The implementation service persists the partial report and
                    # checks the journal before starting a fresh agent. Retrying
                    # here would discard that context from the replacement prompt.
                    yield ErrorEvent(attempt_error)
                    return
                yield ProgressEvent(
                    f"- 🔁 {model.name} stopped: {brief_failure(attempt_error)}. "
                    "Trying another route.\n")
                if gateway_used and model.provider == "claude" and same_credential_limit(attempt_error):
                    # The gateway and direct Claude CLI share this subscription.
                    # Do not spend another attempt on a known account-wide limit.
                    break
        yield ErrorEvent(f"all {decision.complexity} routing candidates failed: {last_error or 'unknown error'}")

    @staticmethod
    def _gateway_attempts(mode: str) -> tuple[bool, ...]:
        return gateway_attempts(mode)

    async def _finalize_metadata(self, run: Run, event: DoneEvent, started: float,
                                 failure: str | None = None) -> None:
        # The database row can be finalized by the initial request; this guard keeps
        # normal tool resumes from failing if the service was recreated in tests.
        decision_id = run.metadata.get("decision_id")
        model_name = run.metadata.get("model")
        if not decision_id or not model_name:
            return
        provider = run.metadata.get("provider")
        executor = "claude-cli" if provider == "claude" else "codex-cli"
        latency = int((time.monotonic() - (run.metadata.get("attempt_started_at") or started)) * 1000)
        await self.persistence.finalize_by_id(
            decision_id, model_name, provider or "unknown", executor,
            bool(run.metadata.get("gateway_used")), "failed" if failure else "completed", latency,
            event.usage, failure,
        )

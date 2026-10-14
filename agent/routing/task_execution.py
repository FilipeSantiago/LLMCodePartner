"""OpenSpec task execution with persisted, same-tier fallback decisions."""

import time
import os
from collections.abc import AsyncIterator
from collections.abc import Awaitable, Callable

import providers

from agent.planning.knowledge_base import KB_READING_GUIDANCE
from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.gateway import brief_failure, gateway_attempts, same_credential_limit
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.models import ExecutionContext, ExecutionDecision
from agent.routing.persistence import RoutingPersistence
from agent.routing.service import ModelRoutingService
from agent.spec.models import Task
from mcp_bridge.models import (
    DoneEvent, ErrorEvent, Event, MutationEvent, ProgressEvent, Run, TextEvent,
    ToolCallEvent,
)
from mcp_bridge.registry import ToolSpec


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
                      kb_first=None,
                      ) -> AsyncIterator[Event]:
        # The active project belongs to the IDE request that advertised `specs`.
        # Never route through a process-wide IDE/project setting.
        decision = await self.router.route(task, change_id, "bridge-request",
                                           exclude=exclude_models)
        await self.persistence.record_decision(decision)
        if on_decision is not None:
            await on_decision(decision)
        async for event in self._run(
                task, decision, self._execution_messages(task, messages, execution_instruction,
                                                         contract, plan_context),
                specs, direct_tool_handlers or {}, owner_metadata or {}, kb_first):
            yield event

    @staticmethod
    def _execution_messages(task: Task, messages: list[dict],
                            execution_instruction: str = "", contract=None,
                            plan_context: str = "") -> list[dict]:
        """Replace the transport command with the actual OpenSpec task objective."""
        task_prompt = "\n".join(filter(None, [
            f"Implement OpenSpec task {task.id}: {task.title}",
            f"Description: {task.description}",
            f"Acceptance criteria: {task.acceptance_criteria or [task.description]}",
            f"Context: {task.context}",
            f"Preferred capability: {task.preferred_capability}",
            (f"Additional execution instruction: {execution_instruction}"
             if execution_instruction else ""),
            (f"TaskContract revision {contract.revision}: {contract.goal}. "
             f"Expected files: {', '.join(contract.primary_files) or 'none named'}. "
             "These are guidance, not a file allowlist."
             if contract is not None else ""),
            plan_context,
            ("A planner already investigated this work package, and earlier tasks' results "
             "are listed above; treat both as established. Do not repeat that discovery: read "
             "only the source regions you will change or call, starting from the listed files "
             "and line ranges, then implement. Search the repository more broadly only if that "
             "context proves wrong or insufficient, and report any such deviation from the "
             "plan in the summary, or in remaining_gaps if it leaves this task incomplete "
             "(use execution_dispute when it concerns execution). "
             "Respect the do-not-modify areas."
             if plan_context else
             "Use the IDE MCP tools as needed to complete the task's actual objective."),
            KB_READING_GUIDANCE,
            "Use the approved plan's files and symbols as KB search and outline seeds. "
            "Use current IDE reads for the exact signatures, edit regions and tests; "
            "indexed line numbers may have shifted. Broaden discovery only for a concrete "
            "question the plan does not settle.",
            ("Use codepartner_read_file with explicit line ranges for targeted source "
             "reads; a full-file read is available when context requires it. "
             "Repeated reads use a run-local cache; pass "
             "refresh=true when you need a fresh IDE read. Use codepartner_search_in_file "
             "to locate text within one file. Tool results include observation IDs that "
             "codepartner_get_observation can expand when diagnostics require more detail."
             if os.getenv("CODEPARTNER_ACI_ENABLED", "true").lower() != "false"
             else "Use the advertised IDE file tools to inspect relevant source."),
            "Change existing files with focused patches (apply_patch) instead of re-creating "
            "whole files, and keep each tool call's arguments small; very large tool-call "
            "payloads can fail to parse and stop the task.",
            "If only the universal `execute_tool` is available, it runs IDE tools, not shell "
            "commands: run tests as `execute_terminal_command --command \"uv run pytest "
            "tests/test_x.py -v\"` (named `--param value` arguments only).",
            "If this task requires Python verification, inspect the project metadata and "
            "configured IDE interpreter or run configurations before declaring the checks "
            "unavailable. Prefer the project-native environment (for example `uv run pytest` "
            "or `.venv/bin/python -m pytest` when those are actually present) over a bare "
            "system `python` or `pytest`. Do not add test execution to a research or inspection "
            "task unless its own description or acceptance criteria requires it.",
            "Finish with exactly one JSON object using this handoff schema: "
            '{"schema_version":1,"task_id":"<task id>","summary":"<work done>",'
            '"changed_files":[{"path":"project/relative/path","change":'
            '"created|modified|deleted|renamed|moved","purpose":"<why>"}],'
            '"criteria":[{"criterion":"<requirement>","status":'
            '"satisfied|blocked","paths":["project/relative/path"],'
            '"evidence_ids":[]}],"remaining_gaps":[],"execution_dispute":null}. '
            "Use remaining_gaps only for unfinished work required by this task's own "
            "acceptance criteria. Mention planned work assigned to later tasks in the summary; "
            "it does not make this task incomplete. "
            "List every created or changed file. Write every path relative to the repository "
            "root and include its directories (for example, tests/test_trainer.py, never only "
            "test_trainer.py). This manifest helps discovery but does not replace the mutation "
            "journal or captured execution evidence. "
            "Evidence IDs must be numeric IDs actually issued by CodePartner; descriptive labels "
            "are claims, not evidence. If you disagree with a reviewer statement that code or "
            "tests do not compile, collect, or run, replace execution_dispute:null with an object "
            "containing claim, source_paths, working_directory, and steps. Each step must name an "
            "advertised tool and its exact arguments so CodePartner can replay it for the reviewer. "
            "Do not create a project-local runner to support a dispute. "
            "Do not claim a test ran unless it did. Do not create summary, completion, "
            "evidence, verification, or test-result files solely to communicate with the "
            "reviewer; return those details in your final response instead.",
        ]))
        prefix = [message for message in messages[:-1] if message.get("role") == "system"]
        return [*prefix, {"role": "user", "content": task_prompt}]

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
                   kb_first=None) -> AsyncIterator[Event]:
        attempts = (decision.recommended_model, *decision.fallback_models)
        last_error: str | None = None
        for index, model_name in enumerate(attempts):
            model = self.config.models[model_name]
            for gateway_used in gateway_attempts(model.gateway_mode):
                started = time.monotonic()
                await self.persistence.record_running_model(
                    decision, model.name, model.provider, gateway_used)
                context = ExecutionContext(
                    messages=messages, specs=specs, decision=decision,
                    gateway_url=self.config.agentgateway_url, use_gateway=gateway_used,
                    started_at=started,
                    direct_tool_handlers=direct_tool_handlers,
                    owner_metadata=owner_metadata or {},
                    kb_first=kb_first.fork() if kb_first else None,
                )
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

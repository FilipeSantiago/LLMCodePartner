"""OpenSpec task execution with persisted, same-tier fallback decisions."""

import time
import os
from dataclasses import replace
from collections.abc import AsyncIterator
from collections.abc import Awaitable, Callable

import providers

from agent.kb_first import KbFirstPolicy, enforced_reads, native_gate_metadata
from agent.planning.knowledge_base import kb_guidance
from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.evidence import is_verification_call, verification_succeeded
from agent.routing.gateway import brief_failure, gateway_attempts, same_credential_limit
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.models import ExecutionContext, ExecutionDecision
from agent.routing.persistence import RoutingPersistence
from agent.routing.rollback import RollbackJournal
from agent.routing.service import ModelRoutingService
from agent.spec.models import Task
from agent.workspace import (NATIVE_SHELL_TOOL, READ_ONLY_CAPABILITIES, NativeRun,
                             RegisteredProject, command_result, plan_native,
                             read_only_session, toolchain)
from mcp_bridge.models import (
    DoneEvent, ErrorEvent, Event, MutationEvent, ProgressEvent, Run, TextEvent,
    ToolCallEvent,
)
from mcp_bridge.registry import ToolSpec
from providers.compaction import compaction_brief

# Applies to every executor and model. Minimizes unnecessary work; never forbids the
# reads, searches or checks needed to resolve uncertainty, failures or verification.
EXPLORATION_RULES = (
    "EXECUTION RULES: Never read the whole repository. Start from the files, symbols and "
    "line ranges named above; prefer targeted searches and reading specific functions, "
    "classes or line ranges over whole files; widen the scope only to resolve a "
    "dependency or unexpected behavior, and skip unrelated modules, tests, docs and "
    "generated files. Reuse what earlier tool calls returned: do not re-read an unchanged "
    "file region, repeat a search that already answered the question, fetch the same "
    "information with several tools, or re-query context already given here; batch "
    "related reads and searches when the tools allow it. Treat verified planner findings "
    "and earlier task results as navigation and do not repeat their investigation; when "
    "current source contradicts them, current source wins. Settle a minimal approach "
    "before editing, make small focused changes, reuse existing utilities and interfaces, "
    "and avoid unrelated refactoring, speculative improvements and new abstractions the "
    "task does not need. Validate with the targeted tests first, broaden only when needed "
    "or required, do not rerun unchanged tests without a reason, and run the required "
    "lint, format and type checks. These rules never limit necessary work: read more, "
    "re-inspect or validate more broadly whenever that is needed to resolve uncertainty, "
    "investigate a failure or verify the result, and if this context proves insufficient, "
    "retrieve more with the available search, KB and file tools.")

_HANDOFF_SCHEMA = (
    "Finish with exactly one JSON object using this handoff schema: "
    '{"schema_version":1,"task_id":"<task id>","summary":"<work done>",'
    '"changed_files":[{"path":"project/relative/path","change":'
    '"created|modified|deleted|renamed|moved","purpose":"<why>"}],'
    '"criteria":[{"criterion":"<requirement>","status":'
    '"satisfied|blocked","paths":["project/relative/path"],'
    '"evidence_ids":[]}],"remaining_gaps":[],"execution_dispute":null}. '
    "Use remaining_gaps only for unfinished work required by this task's own "
    "acceptance criteria. ")


def _task_contract(task: Task, execution_instruction: str, contract) -> list[str]:
    """The task's own contract, each fact stated once."""
    goal = getattr(contract, "goal", "") if contract is not None else ""
    return [
        f"Implement OpenSpec task {task.id}: {task.title}",
        f"Description: {task.description}",
        f"Acceptance criteria: {task.acceptance_criteria or [task.description]}",
        f"Context: {task.context}" if task.context else "",
        f"Preferred capability: {task.preferred_capability}",
        (f"Additional execution instruction: {execution_instruction}"
         if execution_instruction else ""),
        (f"TaskContract revision {contract.revision}"
         + (f": {goal}" if goal and goal.strip() != task.description.strip() else "")
         + f". Expected files: {', '.join(contract.primary_files) or 'none named'}. "
         "These are guidance, not a file allowlist."
         if contract is not None else ""),
    ]


def _planned(plan_context: str, ide: bool) -> str:
    if not plan_context:
        return ""
    return ("A planner already investigated this work package; the context above is "
            "scoped to this task" + (" and its line ranges are leads" if ide else "")
            + ". Report any deviation from it in the summary, or in remaining_gaps if it "
            "leaves this task incomplete" + (" (use execution_dispute when it concerns "
                                            "execution)" if ide else "")
            + ". Respect the do-not-modify areas.")


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
        discovery = getattr(self.config, "discovery", None)
        policy = getattr(discovery, "policy", "selective")
        if enforced_reads(discovery) and kb_first is None:
            kb_first = KbFirstPolicy(
                role="coder", tool_names=[spec.name for spec in specs],
                outline_gate_lines=discovery.outline_gate_lines,
                search_gate_max=discovery.search_gate_max,
                excluded_files=getattr(contract, "excluded_files", ()),
                enforce_reads=True)
        async for event in self._run(
                task, decision, self._execution_messages(task, messages, execution_instruction,
                                                         contract, plan_context,
                                                         kb_policy=policy),
                specs, direct_tool_handlers or {},
                {**(owner_metadata or {}),
                 **native_gate_metadata(getattr(self.config, "discovery", None)),
                 # Custom instructions for the CLI's own compaction, not a summarizer.
                 "compaction_brief": compaction_brief(task, contract)}, kb_first,
                native_messages=self._execution_messages(
                    task, messages, execution_instruction, contract, plan_context,
                    native=True, kb_policy=policy),
                excluded_files=tuple(getattr(contract, "excluded_files", ()) or ())):
            yield event

    @staticmethod
    def _execution_messages(task: Task, messages: list[dict],
                            execution_instruction: str = "", contract=None,
                            plan_context: str = "", native: bool = False,
                            kb_policy: str = "selective") -> list[dict]:
        """Replace the transport command with the actual OpenSpec task objective."""
        if native:
            return TaskExecutionService._native_messages(
                task, messages, execution_instruction, contract, plan_context, kb_policy)
        task_prompt = "\n".join(filter(None, [
            *_task_contract(task, execution_instruction, contract),
            plan_context,
            _planned(plan_context, ide=True),
            EXPLORATION_RULES,
            ("" if plan_context else
             "Use the IDE MCP tools as needed to complete the task's actual objective."),
            kb_guidance(kb_policy),
            ("Use the approved plan's files and symbols as KB search and outline seeds. "
             "Use current IDE reads for the exact signatures, edit regions and tests; "
             "indexed line numbers may have shifted. Broaden discovery only for a concrete "
             "question the plan does not settle." if kb_policy == "kb_first" else ""),
            ("Use codepartner_read_file with explicit line ranges for targeted source "
             "reads. Under kb_first, its first call for each file returns KB navigation "
             "and later reads must stay within the reported range limit. "
             "Repeated reads use a run-local cache; pass "
             "refresh=true when you need a fresh IDE read. Use codepartner_search_in_file "
             "to locate text within one file. Tool results include observation IDs that "
             "codepartner_get_observation can expand when diagnostics require more detail."
             if os.getenv("CODEPARTNER_ACI_ENABLED", "true").lower() != "false"
             else "Use the advertised IDE file tools to inspect relevant source."),
            "Edit existing files with apply_patch; use create_new_file only for files "
            "that do not exist yet. Keep each tool call's arguments small; very large tool-call "
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
            _HANDOFF_SCHEMA +
            "Mention planned work assigned to later tasks in the summary; "
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

    @staticmethod
    def _native_messages(task: Task, messages: list[dict], execution_instruction: str = "",
                         contract=None, plan_context: str = "",
                         kb_policy: str = "selective") -> list[dict]:
        """The task prompt for native execution: the CLI's own file and shell tools in
        the project's working directory; IDE tools only for IDE-specific work."""
        task_prompt = "\n".join(filter(None, [
            *_task_contract(task, execution_instruction, contract),
            plan_context,
            _planned(plan_context, ide=False),
            EXPLORATION_RULES,
            "You are working directly in the project's working directory. Use your own file "
            "tools to read, search and edit files, and your shell to run the project's own "
            "commands (for example `uv run pytest tests/test_x.py`). Current files are "
            "authoritative. Name files in your report by their project-relative paths.",
            kb_guidance(kb_policy, native=True),
            "Bridged IDE tools, when offered, are for IDE-specific work such as inspections, "
            "symbol lookups and refactorings, not for ordinary reads and edits.",
            "Do not commit, switch branches, stash or rewrite git history; CodePartner "
            "journals your file changes and the reviewer inspects the work tree. Only change "
            "files this task needs.",
            "Do not claim a test ran unless it did, and say so when a command could not run "
            "in this environment.",
            _HANDOFF_SCHEMA +
            "List every created or changed file, with paths relative to "
            "the repository root. Evidence IDs must be numeric IDs actually issued by "
            "CodePartner; leave them empty otherwise. Do not create summary, completion, "
            "evidence, verification, or test-result files solely to communicate with the "
            "reviewer; return those details in your final response instead.",
        ]))
        prefix = [message for message in messages[:-1] if message.get("role") == "system"]
        return [*prefix, {"role": "user", "content": task_prompt}]

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
            workspace=project.workspace, env=env, notes=notes,
            start=lambda: journal.record_git_baseline(project.workspace),
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
                   kb_first=None, native_messages: list[dict] | None = None,
                   excluded_files: tuple[str, ...] = ()) -> AsyncIterator[Event]:
        attempts = (decision.recommended_model, *decision.fallback_models)
        last_error: str | None = None
        for index, model_name in enumerate(attempts):
            model = self.config.models[model_name]
            # `auto` falls back to the IDE path; an explicit `native` mode fails instead.
            plan = plan_native(model.provider, excluded_files,
                               controlled_reads=enforced_reads(self.config.discovery))
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
                    # Native runs use the KB on demand: no KB-first interception.
                    kb_first=kb_first.fork() if kb_first and native is None else None,
                    native=native,
                )
                if native is not None:
                    context = replace(context, messages=native_messages or messages)
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

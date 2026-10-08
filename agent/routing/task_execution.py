"""OpenSpec task execution with persisted, same-tier fallback decisions."""

import time
import os
from collections.abc import AsyncIterator
from collections.abc import Awaitable, Callable

import providers

from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.models import ExecutionContext, ExecutionDecision
from agent.routing.persistence import RoutingPersistence
from agent.routing.service import ModelRoutingService
from agent.spec.models import Task
from mcp_bridge.models import (
    DoneEvent, ErrorEvent, Event, MutationEvent, Run, TextEvent, ToolCallEvent,
)
from mcp_bridge.registry import ToolSpec


class TaskExecutionService:
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
                                                         contract),
                specs, direct_tool_handlers or {}, owner_metadata or {}):
            yield event

    @staticmethod
    def _execution_messages(task: Task, messages: list[dict],
                            execution_instruction: str = "", contract=None) -> list[dict]:
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
            "Use the IDE MCP tools as needed to complete the task's actual objective.",
            ("Use codepartner_read_file for 100-line windows and codepartner_search_in_file "
             "to locate text within one file. Repository searches return locations; request "
             "only the source windows you need. Tool results include observation IDs that "
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
        async for event in provider.tools(messages, specs):
            if isinstance(event, ErrorEvent):
                failure = event.message
            if isinstance(event, DoneEvent):
                # The initial execution already created the audit record. Resume
                # rows are represented as attempts only when a terminal turn ends.
                await self._finalize_metadata(run, event, started, failure)
            yield event

    async def _run(self, task, decision, messages, specs,
                   direct_tool_handlers: dict,
                   owner_metadata: dict | None = None) -> AsyncIterator[Event]:
        attempts = (decision.recommended_model, *decision.fallback_models)
        last_error = None
        for index, model_name in enumerate(attempts):
            model = self.config.models[model_name]
            gateway_attempts = self._gateway_attempts(model.gateway_mode)
            for gateway_used in gateway_attempts:
                started = time.monotonic()
                context = ExecutionContext(
                    messages=messages, specs=specs, decision=decision,
                    gateway_url=self.config.agentgateway_url, use_gateway=gateway_used,
                    started_at=started,
                    direct_tool_handlers=direct_tool_handlers,
                    owner_metadata=owner_metadata or {},
                )
                executor = self.executors.get(model.executor)
                failed_before_write = False
                emitted_material = False
                paused_for_tool = False
                async for event in executor.execute(task, model, context):
                    if isinstance(event, ErrorEvent):
                        failed_before_write = True
                        last_error = event.message
                    if isinstance(event, (TextEvent, ToolCallEvent, MutationEvent)):
                        emitted_material = True
                    if isinstance(event, ToolCallEvent):
                        paused_for_tool = True
                    # Do not close a stream with a provisional failure when the
                    # executor has produced nothing observable yet: the next
                    # gateway/direct attempt is safe and can become the response.
                    # Terminal events are emitted only after operational state is
                    # durable; SSE stops consuming at DoneEvent.
                    if not isinstance(event, DoneEvent) and not (failed_before_write and not emitted_material):
                        yield event
                    if isinstance(event, DoneEvent):
                        latency = int((time.monotonic() - started) * 1000)
                        status = "failed" if failed_before_write else "completed"
                        await self.persistence.record_attempt(
                            decision, model.name, model.provider, model.executor,
                            gateway_used, status, latency, last_error, event.usage,
                        )
                        await self.persistence.finalize(
                            decision, model.name, model.provider, gateway_used,
                            status, latency,
                            "executor-fallback" if index else None,
                        )
                        self.router.health.record(model.name, model.executor, gateway_used,
                                                  status == "completed")
                        if status == "completed":
                            yield event
                            return
                        # Once text or a tool call has reached the client, another
                        # model would risk duplicated work or conflicting output.
                        if emitted_material:
                            yield event
                            return
                # An executor with a tool call remains alive; fallback must wait for
                # the resumed terminal event, never start a competing run.
                if paused_for_tool:
                    return
            if last_error:
                continue
        yield ErrorEvent(f"all {decision.complexity} routing candidates failed: {last_error or 'unknown error'}")

    @staticmethod
    def _gateway_attempts(mode: str) -> tuple[bool, ...]:
        if mode == "required":
            return (True,)
        if mode == "disabled":
            return (False,)
        return (True, False)

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

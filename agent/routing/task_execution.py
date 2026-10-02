"""OpenSpec task execution with persisted, same-tier fallback decisions."""

import os
import time
from collections.abc import AsyncIterator
from collections.abc import Awaitable, Callable

import providers

from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.models import ExecutionContext, ExecutionDecision
from agent.routing.persistence import RoutingPersistence
from agent.routing.service import ModelRoutingService
from agent.spec.models import Task
from mcp_bridge.models import DoneEvent, ErrorEvent, Event, Run, TextEvent, ToolCallEvent
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
                      ) -> AsyncIterator[Event]:
        project_path = os.getenv("JETBRAINS_MCP_PROJECT_PATH", "unknown-project")
        decision = await self.router.route(task, change_id, project_path)
        await self.persistence.record_decision(decision)
        if on_decision is not None:
            await on_decision(decision)
        async for event in self._run(task, decision, self._execution_messages(task, messages), specs):
            yield event

    @staticmethod
    def _execution_messages(task: Task, messages: list[dict]) -> list[dict]:
        """Replace the transport command with the actual OpenSpec task objective."""
        task_prompt = "\n".join(filter(None, [
            f"Implement OpenSpec task {task.id}: {task.title}",
            f"Description: {task.description}",
            f"Context: {task.context}",
            f"Preferred capability: {task.preferred_capability}",
            "Use the IDE tools to inspect relevant code, apply the change, and summarize it.",
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
            yield event
            if isinstance(event, DoneEvent):
                # The initial execution already created the audit record. Resume
                # rows are represented as attempts only when a terminal turn ends.
                await self._finalize_metadata(run, event, started, failure)

    async def _run(self, task, decision, messages, specs) -> AsyncIterator[Event]:
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
                )
                executor = self.executors.get(model.executor)
                failed_before_write = False
                emitted_material = False
                paused_for_tool = False
                async for event in executor.execute(task, model, context):
                    if isinstance(event, ErrorEvent):
                        failed_before_write = True
                        last_error = event.message
                    if isinstance(event, (TextEvent, ToolCallEvent)):
                        emitted_material = True
                    if isinstance(event, ToolCallEvent):
                        paused_for_tool = True
                    # Do not close a stream with a provisional failure when the
                    # executor has produced nothing observable yet: the next
                    # gateway/direct attempt is safe and can become the response.
                    if not (failed_before_write and not emitted_material):
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
                            return
                        # Once text or a tool call has reached the client, another
                        # model would risk duplicated work or conflicting output.
                        if emitted_material:
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

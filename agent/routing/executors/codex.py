"""Codex task executor backed by the existing authenticated CLI process."""

from collections.abc import AsyncIterator

from agent.routing.models import ExecutionContext, ModelRef
from agent.spec.models import Task
from mcp_bridge.models import Event
from mcp_bridge.registry import ROLE_CODER
from providers.codex import CodexProvider


class CodexCLIExecutor:
    name = "codex-cli"

    def __init__(self, provider: CodexProvider | None = None):
        self._provider = provider or CodexProvider()

    async def execute(self, task: Task, model: ModelRef,
                      context: ExecutionContext) -> AsyncIterator[Event]:
        async for event in self._provider.tools(
            context.messages, context.specs, model=model.cli_model,
            role=ROLE_CODER,
            run_metadata={
                "task_execution": True,
                "provider": "codex",
                "decision_id": context.decision.decision_id,
                "model": model.name,
                "gateway_used": context.use_gateway,
                "attempt_started_at": context.started_at,
                "required_operations": task.required_operations,
            },
             gateway_url=context.gateway_url if context.use_gateway else None,
             concrete_model=True,
             reasoning_effort=model.reasoning_effort,
             direct_tool_handlers=context.direct_tool_handlers,
        ):
            yield event

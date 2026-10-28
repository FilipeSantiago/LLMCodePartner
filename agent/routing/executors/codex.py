"""Codex task executor backed by the existing authenticated CLI process."""

from collections.abc import AsyncIterator

from agent.routing.models import ExecutionContext, ModelRef
from agent.spec.models import Task
from agent.workspace import READ_ONLY_CAPABILITIES
from mcp_bridge.models import Event
from mcp_bridge.registry import ROLE_CODER, ROLE_OPTIMIZER
from providers.codex import CodexProvider


class CodexCLIExecutor:
    name = "codex-cli"

    def __init__(self, provider: CodexProvider | None = None):
        self._provider = provider or CodexProvider()

    async def execute(self, task: Task, model: ModelRef,
                      context: ExecutionContext) -> AsyncIterator[Event]:
        async for event in self._provider.tools(
            context.messages, context.specs, model=model.cli_model,
            # The coder role adds the agent tool interface; a tool-free call gets none.
            role=ROLE_OPTIMIZER if context.tool_free else ROLE_CODER,
            run_metadata={
                "task_execution": True,
                "provider": "codex",
                "decision_id": context.decision.decision_id,
                "task_id": task.id,
                "model": model.name,
                "gateway_used": context.use_gateway,
                "attempt_started_at": context.started_at,
                "required_operations": task.required_operations,
                "read_only": task.preferred_capability.lower() in READ_ONLY_CAPABILITIES,
                "excluded_files": tuple(getattr(task, "excluded_files", ())),
                **context.owner_metadata,
                **({"tool_free": True} if context.tool_free else {}),
            },
             gateway_url=context.gateway_url if context.use_gateway else None,
             concrete_model=True,
             reasoning_effort=model.reasoning_effort,
             direct_tool_handlers=context.direct_tool_handlers,
             native=context.native,
             subagents=context.subagents,
             preapprove_ide_tools=bool(context.owner_metadata.get("implementation_job_id")),
        ):
            yield event

"""Claude task executor backed by the existing authenticated Agent SDK."""

from collections.abc import AsyncIterator

from agent.routing.models import ExecutionContext, ModelRef
from agent.spec.models import Task
from agent.workspace import READ_ONLY_CAPABILITIES
from mcp_bridge.models import Event
from mcp_bridge.registry import ROLE_CODER, ROLE_OPTIMIZER
from providers.claude import ClaudeProvider


class ClaudeCLIExecutor:
    name = "claude-cli"

    def __init__(self, provider: ClaudeProvider | None = None):
        self._provider = provider or ClaudeProvider()

    async def execute(self, task: Task, model: ModelRef,
                      context: ExecutionContext) -> AsyncIterator[Event]:
        env = {}
        if context.use_gateway and context.gateway_url:
            env["ANTHROPIC_BASE_URL"] = context.gateway_url
        if context.prompt_cache_ttl == "5m":
            env["FORCE_PROMPT_CACHING_5M"] = "1"
        async for event in self._provider.tools(
            context.messages, context.specs, model=model.cli_model,
            # The coder role adds the agent tool interface; a tool-free call gets none.
            role=ROLE_OPTIMIZER if context.tool_free else ROLE_CODER,
            run_metadata={
                "task_execution": True,
                "provider": "claude",
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
            execution_env=env,
            direct_tool_handlers=context.direct_tool_handlers,
            native=context.native,
            subagents=context.subagents,
        ):
            yield event

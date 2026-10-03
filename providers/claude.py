"""Claude provider (local `claude` CLI via the Agent SDK).

`ClaudeProvider` implements the `Provider` surface:
  - plain chat: `stream` (and inherited `complete`) — no tools, no filesystem.
  - tool calling: `tools(messages, specs)` exposes the JetBrains tools as an
    in-process MCP server (each handler bridges out via `bridge.call_tool`) and
    drives a background Run; the stateful start-vs-resume decision is internal.

This is the only module that imports `claude_agent_sdk`.
"""
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from mcp_bridge import bridge
from mcp_bridge import server as engine
from mcp_bridge.models import Event, Run, TextEvent
from mcp_bridge.registry import ToolSpec, allowed_for
from providers.base import Provider

log = logging.getLogger("codepartner.providers.claude")

SERVER_NAME = "jetbrains"


class ClaudeProvider(Provider):
    NAME = "claude"

    async def stream(self, prompt: str, system: str | None = None
                     ) -> AsyncIterator[tuple[str, str | None]]:
        async for message in query(prompt=prompt, options=self._plain_options(system)):
            if isinstance(message, AssistantMessage):
                for b in message.content:
                    if isinstance(b, TextBlock):
                        yield b.text, None
            elif isinstance(message, ResultMessage):
                yield "", message.terminal_reason

    async def tools(self, messages: list[dict], specs: list[ToolSpec],
                    model: str | None = None, role: str | None = None,
                    run_metadata: dict | None = None,
                    execution_env: dict[str, str] | None = None,
                    direct_tool_handlers: dict | None = None, **kwargs) -> AsyncIterator[Event]:
        """Stateful, but the start-vs-resume decision is private: a history carrying
        tool results resolves the pending futures and continues the SAME background
        run; otherwise a fresh run is started. Either way, yields the same neutral
        events every provider's `tools()` emits. `model` selects the Claude tier
        ("haiku"/"sonnet"/"opus"/full id) for a fresh run; None = CLI default. `role`
        selects the tool allowlist for a fresh run; a resume inherits the role already
        baked into the Run."""
        results = self._trailing_tool_results(messages)
        if results:
            run: Run | None = None
            for m in results:
                run = engine.resume(m["tool_call_id"], m.get("content") or "") or run
            if run is None:
                yield TextEvent("No pending tool call matched this result (it may have timed out).")
                return
        else:
            prompt, system = self._flatten(messages)

            async def strategy(run: Run, prompt: str, system: str | None,
                               specs: list[ToolSpec]) -> str:
                return await self._run_strategy(run, prompt, system, specs, model, execution_env)

            run = engine.start_run(prompt, system, specs, strategy=strategy,
                                   allowed=allowed_for(role),
                                   direct_tool_handlers=direct_tool_handlers)
            run.metadata.update(run_metadata or {})

        async for event in self._drain_queue(run):
            yield event

    # --- internals -----------------------------------------------------------

    def _plain_options(self, system: str | None) -> ClaudeAgentOptions:
        # `tools=[]` removes built-in Read/Bash/etc. — no backend filesystem access.
        return ClaudeAgentOptions(system_prompt=system, tools=[], allowed_tools=[], mcp_servers={})

    def _make_tool(self, run: Run, spec: ToolSpec):
        @tool(spec.name, spec.description, spec.schema)
        async def _handler(args: dict[str, Any], _run: Run = run, _name: str = spec.name):
            try:
                text = await bridge.call_tool(_run, _name, args)
            except TimeoutError:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: tool '{_name}' timed out after {bridge.TOOL_TIMEOUT}s",
                    }],
                    "is_error": True,
                }
            return {"content": [{"type": "text", "text": text}]}

        return _handler

    async def _run_strategy(self, run: Run, prompt: str, system: str | None,
                            specs: list[ToolSpec], model: str | None = None,
                            execution_env: dict[str, str] | None = None) -> str:
        """The engine `Strategy`: drive the model in-process, pushing TextEvents onto
        the Run and (via `_make_tool` → `bridge.call_tool`) parking a future per tool
        call. Returns the terminal reason. Runs as the Run's background task. `model`
        selects the Claude tier (None = CLI default). Captures token/cost usage from the
        terminal `ResultMessage` onto the Run for the engine's DoneEvent."""
        server = create_sdk_mcp_server(
            name=SERVER_NAME, version="1.0.0",
            tools=[self._make_tool(run, s) for s in specs],
        )
        allowed = [f"mcp__{SERVER_NAME}__{s.name}" for s in specs]
        options = ClaudeAgentOptions(
            system_prompt=system, tools=[], allowed_tools=allowed,
            mcp_servers={SERVER_NAME: server}, model=model,
            # The Agent SDK may treat ``env`` as the complete subprocess
            # environment. Preserve the authenticated CLI session while adding
            # a gateway override.
            env={**os.environ, **(execution_env or {})},
        )
        terminal = "endTurn"
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for b in message.content:
                    if isinstance(b, TextBlock) and b.text:
                        await run.put(TextEvent(b.text))
            elif isinstance(message, ResultMessage):
                terminal = message.terminal_reason
                run.usage = message.usage
                run.model_usage = message.model_usage
                log.info(
                    "claude coder usage: model=%s cost_usd=%s model_usage=%s",
                    model or "default", message.total_cost_usd, message.model_usage,
                )
        return terminal

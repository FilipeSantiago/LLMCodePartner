"""Claude provider (local `claude` CLI via the Agent SDK).

`ClaudeProvider` implements the `Provider` surface:
  - plain chat: `stream` (and inherited `complete`) — no tools, no filesystem.
  - tool calling: `tools(messages, specs)` exposes the JetBrains tools as an
    in-process MCP server (each handler bridges out via `bridge.call_tool`) and
    drives a background Run; the stateful start-vs-resume decision is internal.

This is the only module that imports `claude_agent_sdk`.
"""
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
from mcp_bridge.models import DoneEvent, Event, Run, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec
from providers.base import Provider

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

    async def tools(self, messages: list[dict], specs: list[ToolSpec]) -> AsyncIterator[Event]:
        """Stateful, but the start-vs-resume decision is private: a history carrying
        tool results resolves the pending futures and continues the SAME background
        run; otherwise a fresh run is started. Either way, yields the same neutral
        events every provider's `tools()` emits."""
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
            run = engine.start_run(prompt, system, specs, strategy=self._run_strategy)

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
                            specs: list[ToolSpec]) -> str:
        """The engine `Strategy`: drive the model in-process, pushing TextEvents onto
        the Run and (via `_make_tool` → `bridge.call_tool`) parking a future per tool
        call. Returns the terminal reason. Runs as the Run's background task."""
        server = create_sdk_mcp_server(
            name=SERVER_NAME, version="1.0.0",
            tools=[self._make_tool(run, s) for s in specs],
        )
        allowed = [f"mcp__{SERVER_NAME}__{s.name}" for s in specs]
        options = ClaudeAgentOptions(
            system_prompt=system, tools=[], allowed_tools=allowed,
            mcp_servers={SERVER_NAME: server},
        )
        terminal = "endTurn"
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for b in message.content:
                    if isinstance(b, TextBlock) and b.text:
                        await run.put(TextEvent(b.text))
            elif isinstance(message, ResultMessage):
                terminal = message.terminal_reason
        return terminal

    def _flatten(self, messages: list[dict]) -> tuple[str, str | None]:
        """Neutral history → (prompt, system) for a fresh turn. Tool-plumbing turns
        (assistant `tool_calls`, `role:"tool"` results) are dropped — the model gets
        tool results via the resumed MCP handler, not the prompt."""
        system = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system" and m.get("content")
        ) or None
        convo = [
            m for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content") and not m.get("tool_calls")
        ]
        if len(convo) == 1:
            prompt = convo[0]["content"]
        else:
            prompt = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in convo)
        return prompt, system

    def _trailing_tool_results(self, messages: list[dict]) -> list[dict]:
        """`role:"tool"` messages carrying a tool_call_id (client tool results)."""
        return [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id")]

    async def _drain_queue(self, run: Run):
        """Yield a Run's neutral events until it stops for this request: a ToolCallEvent
        (background task now blocked on the future — resumed by a later request) or a
        DoneEvent (turn finished)."""
        while True:
            event = await run.next_event()
            yield event
            if isinstance(event, (ToolCallEvent, DoneEvent)):
                return

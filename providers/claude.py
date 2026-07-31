"""Claude provider (local `claude` CLI via the Agent SDK).

Two capabilities:
  - plain chat: `stream` / `complete` (no tools, no filesystem access).
  - tool strategy: `run(...)` exposes the JetBrains tools as an in-process MCP
    server; each tool handler bridges out via `bridge.call_tool`.

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
from mcp_bridge.models import Run, TextEvent
from mcp_bridge.registry import ToolSpec

NAME = "claude"

SERVER_NAME = "jetbrains"


# --- plain chat (no tools) ---------------------------------------------------

def _plain_options(system: str | None) -> ClaudeAgentOptions:
    # `tools=[]` removes built-in Read/Bash/etc. — no backend filesystem access.
    return ClaudeAgentOptions(system_prompt=system, tools=[], allowed_tools=[], mcp_servers={})


async def complete(prompt: str, system: str | None = None) -> tuple[str, str]:
    text, terminal = "", "endTurn"
    async for message in query(prompt=prompt, options=_plain_options(system)):
        if isinstance(message, AssistantMessage):
            text += "".join(b.text for b in message.content if isinstance(b, TextBlock))
        elif isinstance(message, ResultMessage):
            terminal = message.terminal_reason
    return text, terminal


async def stream(prompt: str, system: str | None = None) -> AsyncIterator[tuple[str, str | None]]:
    async for message in query(prompt=prompt, options=_plain_options(system)):
        if isinstance(message, AssistantMessage):
            for b in message.content:
                if isinstance(b, TextBlock):
                    yield b.text, None
        elif isinstance(message, ResultMessage):
            yield "", message.terminal_reason


# --- tool strategy (in-process MCP server) -----------------------------------

def _make_tool(run: Run, spec: ToolSpec):
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


async def run(run: Run, prompt: str, system: str | None, specs: list[ToolSpec]) -> str:
    server = create_sdk_mcp_server(
        name=SERVER_NAME, version="1.0.0",
        tools=[_make_tool(run, s) for s in specs],
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

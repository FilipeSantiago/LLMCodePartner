"""MCP engine — run state and neutral events (model-agnostic).

Events are wire-format-agnostic; the OpenAI SSE translation lives in
`conversation/streaming_responder.py`, not here.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any

from mcp_bridge.registry import ToolRegistry


# --- Events placed on a Run's queue, consumed by the OpenAI SSE translator ---

@dataclass
class TextEvent:
    text: str


@dataclass
class ToolCallEvent:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ErrorEvent:
    message: str


@dataclass
class DoneEvent:
    terminal: str


# The neutral events a provider's `tools()` may yield, as one name.
Event = TextEvent | ToolCallEvent | ErrorEvent | DoneEvent


@dataclass
class Run:
    """One in-flight Claude/Codex execution, isolated per request.

    Holds its OWN tool registry (never global) plus the event queue, background
    task, and the pending tool-call futures for this execution.
    """

    registry: ToolRegistry = field(default_factory=ToolRegistry)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    pending: dict[str, asyncio.Future] = field(default_factory=dict)

    async def put(self, event: Any) -> None:
        await self.queue.put(event)

    async def next_event(self) -> Any:
        return await self.queue.get()

"""MCP engine — run state and neutral events (model-agnostic).

Events are wire-format-agnostic; the OpenAI SSE translation lives in
`conversation/streaming_responder.py`, not here.
"""
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import uuid4
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
class MutationEvent:
    """A direct standalone-IDE mutation that succeeded inside the backend."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ErrorEvent:
    message: str


@dataclass
class DoneEvent:
    terminal: str
    # Optional token/cost usage for this turn (None when the provider doesn't report it).
    # `usage` is the provider's aggregate token dict; `model_usage` is a per-model
    # breakdown (Claude's ResultMessage.model_usage shape) for attributing spend.
    usage: dict | None = None
    model_usage: dict | None = None


# The neutral events a provider's `tools()` may yield, as one name.
Event = TextEvent | ToolCallEvent | MutationEvent | ErrorEvent | DoneEvent


@dataclass
class Run:
    """One in-flight Claude/Codex execution, isolated per request.

    Holds its OWN tool registry (never global) plus the event queue, background
    task, and the pending tool-call futures for this execution. `start_run` builds
    the registry with the calling role's allowlist, so the role is fixed for the
    whole run — including later tool-result resumes. The default factory here is the
    un-roled full ceiling.
    """

    registry: ToolRegistry = field(default_factory=ToolRegistry)
    # ##DELETE AFTER CORRECTION## Correlates temporary DEBUG protocol traces.
    run_id: str = field(default_factory=lambda: "run_" + uuid4().hex)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    # Token/cost usage captured by the strategy from the terminal result, drained onto
    # the engine's DoneEvent. None until the strategy sets them.
    usage: dict | None = None
    model_usage: dict | None = None
    # Optional owner metadata for specialised run entrypoints.  The generic bridge
    # never interprets it; the HTTP controller uses it only to resume the same
    # executor after JetBrains returns a tool result.
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_names: dict[str, str] = field(default_factory=dict)
    mutating_tool_succeeded: bool = False
    # Direct tools are only attached to implementation runs. Unlike ordinary
    # bridge tools, they execute against PyCharm's standalone MCP endpoint and
    # therefore do not require an AI Chat OpenAI tool-call round trip.
    direct_tool_handlers: dict[str, Callable[[dict], Awaitable[str]]] = field(default_factory=dict)

    async def put(self, event: Any) -> None:
        await self.queue.put(event)

    async def next_event(self) -> Any:
        return await self.queue.get()

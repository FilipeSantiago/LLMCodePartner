"""MCP engine — generic run lifecycle (provider-agnostic, pure stdlib).

Owns the background-task lifecycle and cleanup, but not *how* the model runs.
The caller injects a provider **strategy**: an `async def run(run, prompt, system,
specs) -> terminal` that drives the model and, on each tool call, uses
`bridge.call_tool`. Claude implements it with an in-process MCP server; Ollama
with a native tool loop. This module imports no LLM SDK.
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable

from mcp_bridge import bridge
from mcp_bridge.models import DoneEvent, ErrorEvent, Run
from mcp_bridge.registry import ToolSpec

log = logging.getLogger("mcp_bridge")

# A provider run strategy: drive the model with these tool specs, pushing events
# onto run.queue (and calling bridge.call_tool per tool call); return the
# terminal reason ("endTurn" / "maxTurnsReached").
Strategy = Callable[[Run, str, str | None, list[ToolSpec]], Awaitable[str]]


async def _run(run: Run, prompt: str, system: str | None,
               specs: list[ToolSpec], strategy: Strategy) -> None:
    terminal = "endTurn"
    try:
        terminal = await strategy(run, prompt, system, specs)
    except asyncio.CancelledError:
        bridge.discard(run)
        raise
    except Exception as exc:  # surface provider/model failures into the stream
        await run.put(ErrorEvent(str(exc)))
    finally:
        await run.put(DoneEvent(terminal))
        bridge.discard(run)


def start_run(prompt: str, system: str | None, specs: list[ToolSpec],
              strategy: Strategy) -> Run:
    """Begin a bridged execution: register the supported tools and run the
    injected provider strategy in the background."""
    run = Run()
    registered = run.registry.register(specs)
    log.info("mcp tool registration: %s", [s.name for s in registered])
    run.task = asyncio.create_task(_run(run, prompt, system, run.registry.tools(), strategy))
    return run


def resume(tool_call_id: str, content: str) -> Run | None:
    """Deliver a client tool result; return the Run to keep streaming."""
    return bridge.resolve(tool_call_id, content)

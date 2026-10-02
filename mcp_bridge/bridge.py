import asyncio
import logging
import os
from uuid import uuid4

from mcp_bridge.models import Run, ToolCallEvent

log = logging.getLogger("mcp_bridge")

# Configurable via .env; how long a pending tool call waits for JetBrains.
TOOL_TIMEOUT = int(os.getenv("MCP_TOOL_TIMEOUT", "120"))

# Cross-request routing only (tool_call_id -> Run). Ids are unique, so this never
# mixes concurrent requests; the per-request *tool registry* lives on each Run.
_futures: dict[str, asyncio.Future] = {}
_runs: dict[str, Run] = {}


def register(tool_call_id: str, run: Run, name: str, arguments: dict) -> asyncio.Future:
    """Record a pending tool call and return the Future its handler awaits."""
    fut = asyncio.get_running_loop().create_future()
    _futures[tool_call_id] = fut
    _runs[tool_call_id] = run
    run.pending[tool_call_id] = fut
    run.tool_names[tool_call_id] = name
    log.info("tool call emitted id=%s name=%s args=%s", tool_call_id, name, arguments)
    return fut


async def await_result(tool_call_id: str, fut: asyncio.Future) -> str:
    """Await the JetBrains result, bounded by TOOL_TIMEOUT."""
    try:
        return await asyncio.wait_for(fut, timeout=TOOL_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("tool timeout id=%s after %ss", tool_call_id, TOOL_TIMEOUT)
        raise


async def call_tool(run: Run, name: str, arguments: dict) -> str:
    """Surface one tool call to the client and await its result.

    The shared primitive every provider's run strategy uses: mint an id, emit a
    `ToolCallEvent` (which `drain` turns into an OpenAI tool_call), and block on
    the future until the client returns a `role:"tool"` result. Raises
    `asyncio.TimeoutError` if the client doesn't answer within `TOOL_TIMEOUT`.
    """
    tool_call_id = "call_" + uuid4().hex[:24]
    fut = register(tool_call_id, run, name, arguments)
    await run.put(ToolCallEvent(tool_call_id, name, arguments))
    return await await_result(tool_call_id, fut)


def resolve(tool_call_id: str, content: str) -> Run | None:
    """Deliver a JetBrains tool result; return the Run to resume, or None if the
    id is unknown/already resolved (e.g. timed out)."""
    fut = _futures.get(tool_call_id)
    if fut is None or fut.done():
        log.warning("unknown/late tool result id=%s", tool_call_id)
        return None
    run = _runs.get(tool_call_id)
    # A failed IDE write is normally returned as an error-shaped tool result. Do
    # not declare a project mutation in that case: fallback is only unsafe after
    # a write actually succeeded.
    tool_name = (run.tool_names.get(tool_call_id, "") if run else "").lower()
    is_write = any(token in tool_name for token in ("create", "write", "edit", "replace", "rename", "delete"))
    if run is not None and is_write and "error" not in (content or "").lower():
        run.mutating_tool_succeeded = True
    fut.set_result(content)
    log.info("tool result received id=%s len=%d", tool_call_id, len(content or ""))
    return run


def run_for(tool_call_id: str) -> Run | None:
    """Return the active run for a pending IDE tool call without resolving it."""
    return _runs.get(tool_call_id)


def discard(run: Run) -> None:
    """Clean up a finished/cancelled/timed-out run: cancel leftover futures and
    drop all of its tool_call_ids from the routing tables."""
    ids = [tid for tid, r in _runs.items() if r is run]
    for tid in ids:
        fut = _futures.pop(tid, None)
        if fut is not None and not fut.done():
            fut.cancel()
        _runs.pop(tid, None)
    run.pending.clear()
    if ids:
        log.info("cleanup run tool_calls=%d", len(ids))

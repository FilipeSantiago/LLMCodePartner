import asyncio
import logging
import os
from uuid import uuid4

from mcp_bridge.models import MutationEvent, Run, ToolCallEvent
from mcp_bridge.registry import is_mutating_tool, operations_for
from logger.diagnostic import debug

log = logging.getLogger("codepartner.bridge")

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
    debug(log, "bridge.tool_call_registered", trace_id=run.metadata.get("trace_id"),
          run_id=run.run_id, tool_call_id=tool_call_id, tool_name=name,
          arguments=arguments, pending_call_ids=sorted(run.pending))
    log.info("tool_call_emitted id=%s name=%s argument_keys=%s", tool_call_id, name,
             sorted(arguments) if isinstance(arguments, dict) else [])
    return fut


async def await_result(tool_call_id: str, fut: asyncio.Future) -> str:
    """Await the JetBrains result, bounded by TOOL_TIMEOUT."""
    try:
        debug(log, "bridge.tool_result_wait_started", tool_call_id=tool_call_id,
              timeout_seconds=TOOL_TIMEOUT)
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
    # Every route, including direct standalone tools, goes through the per-run
    # catalog adapter.  A provider cannot bypass dispatcher/schema restrictions.
    try:
        name, arguments = run.registry.prepare_advertised_call(name, arguments)
    except (LookupError, ValueError) as exc:
        return f"Error: invalid MCP tool call: {exc}"
    handler = run.direct_tool_handlers.get(name)
    if handler is not None:
        try:
            debug(log, "bridge.direct_tool_started", trace_id=run.metadata.get("trace_id"),
                  run_id=run.run_id, tool_name=name, arguments=arguments)
            text = await handler(arguments)
        except Exception as exc:
            log.warning("direct IDE MCP tool failed name=%s error=%s", name, exc)
            return f"Error: direct PyCharm MCP tool '{name}' failed: {exc}"
        required = set(run.metadata.get("required_operations", ()))
        if is_mutating_tool(name) and (not required or required.intersection(operations_for(name))):
            run.mutating_tool_succeeded = True
            await run.put(MutationEvent(name, arguments))
        log.info("direct IDE MCP mutation succeeded name=%s", name)
        debug(log, "bridge.direct_tool_result", trace_id=run.metadata.get("trace_id"),
              run_id=run.run_id, tool_name=name, result=text)
        return text
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
        # ##DELETE AFTER CORRECTION## Full late-result diagnostics.
        debug(log, "bridge.tool_result_unmatched", tool_call_id=tool_call_id, content=content,
              known_call_ids=sorted(_futures))
        return None
    run = _runs.get(tool_call_id)
    debug(log, "bridge.tool_result_received", trace_id=run.metadata.get("trace_id") if run else None,
          run_id=run.run_id if run else None, tool_call_id=tool_call_id,
          tool_name=run.tool_names.get(tool_call_id) if run else None, content=content)
    # A failed IDE write is normally returned as an error-shaped tool result. Do
    # not declare a project mutation in that case: fallback is only unsafe after
    # a write actually succeeded.
    tool_name = run.tool_names.get(tool_call_id, "") if run else ""
    required = set(run.metadata.get("required_operations", ())) if run else set()
    if (run is not None and is_mutating_tool(tool_name) and "error" not in (content or "").lower()
            and (not required or required.intersection(operations_for(tool_name)))):
        run.mutating_tool_succeeded = True
    fut.set_result(content)
    log.info("tool_result_received id=%s content_length=%d", tool_call_id, len(content or ""))
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
    debug(log, "bridge.run_discarded", trace_id=run.metadata.get("trace_id"),
          run_id=run.run_id, discarded_call_ids=ids)

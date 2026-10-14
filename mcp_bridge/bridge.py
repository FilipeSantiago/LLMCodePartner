import asyncio
import logging
import os
import shlex
from uuid import uuid4

from mcp_bridge.activity import describe_call
from mcp_bridge.models import MutationEvent, ProgressEvent, Run, ToolCallEvent
from mcp_bridge.agent_tools import tool_failed
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
    run.tool_arguments[tool_call_id] = arguments
    debug(log, "bridge.tool_call_registered", trace_id=run.metadata.get("trace_id"),
          run_id=run.run_id, tool_call_id=tool_call_id, tool_name=name,
          arguments=arguments, pending_call_ids=sorted(run.pending))
    log.info("mcp_tool_route route=ide_bridge trace_id=%s run_id=%s id=%s name=%s",
             run.metadata.get("trace_id"), run.run_id, tool_call_id, name)
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
    async def dispatch(tool_name: str, tool_args: dict) -> str:
        if run.tool_interface is None:
            return await _call_backend(run, tool_name, tool_args)
        result = await run.tool_interface.execute(
            tool_name, tool_args,
            lambda actual_name, actual_args: _call_backend(run, actual_name, actual_args))
        run.tool_interface.metrics["model_output_bytes"] += len(result.encode())
        run.tool_interface.record_delivered_result(tool_name, tool_args, result)
        return result

    if run.kb_first is not None:
        return await run.kb_first.call(name, arguments, dispatch, run=run)
    return await dispatch(name, arguments)


async def _call_backend(run: Run, name: str, arguments: dict) -> str:
    if run.metadata.get("stop_requested"):
        return "Error: this job is being stopped; no further tool calls are allowed."
    if run.tool_interface is not None:
        run.tool_interface.metrics["backend_calls"] += 1
    # Every route, including direct standalone tools, goes through the per-run
    # catalog adapter.  A provider cannot bypass dispatcher/schema restrictions.
    try:
        name, arguments = run.registry.prepare_advertised_call(name, arguments)
    except (LookupError, ValueError) as exc:
        return f"Error: invalid MCP tool call: {exc}"
    handler = run.direct_tool_handlers.get(name)
    effective_name = _effective_mutation_name(name, arguments)
    if handler is not None and effective_name:
        # A stop must wait for this write to finish; see `has_inflight_mutation`.
        run.metadata["inflight_direct_mutations"] = (
            run.metadata.get("inflight_direct_mutations", 0) + 1)
        try:
            return await _call_direct(run, handler, name, arguments, effective_name)
        finally:
            run.metadata["inflight_direct_mutations"] -= 1
    if handler is not None:
        return await _call_direct(run, handler, name, arguments, effective_name)
    tool_call_id = "call_" + uuid4().hex[:24]
    fut = register(tool_call_id, run, name, arguments)
    await run.put(ToolCallEvent(tool_call_id, name, arguments))
    result = await await_result(tool_call_id, fut)
    if run.tool_interface is not None:
        run.tool_interface.metrics["backend_output_bytes"] += len(result.encode())
    return result


async def _call_direct(run: Run, handler, name: str, arguments: dict,
                       effective_name: str | None) -> str:
    log.info("mcp_tool_route route=direct_ide_mcp trace_id=%s run_id=%s name=%s",
             run.metadata.get("trace_id"), run.run_id, name)
    if run.metadata.get("task_execution"):
        icon = "✏️" if effective_name else "🔎"
        await run.put(ProgressEvent(f"- {icon} {describe_call(name, arguments)}\n"))
    try:
        debug(log, "bridge.direct_tool_started", trace_id=run.metadata.get("trace_id"),
              run_id=run.run_id, tool_name=name, arguments=arguments)
        text = await handler(arguments)
    except Exception as exc:
        log.warning("mcp_tool_failed route=direct_ide_mcp trace_id=%s run_id=%s name=%s error_type=%s",
                    run.metadata.get("trace_id"), run.run_id, name, type(exc).__name__)
        return f"Error: direct PyCharm MCP tool '{name}' failed: {exc}"
    required = set(run.metadata.get("required_operations", ()))
    if (effective_name and not tool_failed(text)
            and (not required or required.intersection(operations_for(effective_name)))):
        run.mutating_tool_succeeded = True
        await run.put(MutationEvent(name, arguments))
    log.info("mcp_tool_succeeded route=direct_ide_mcp trace_id=%s run_id=%s name=%s",
             run.metadata.get("trace_id"), run.run_id, name)
    debug(log, "bridge.direct_tool_result", trace_id=run.metadata.get("trace_id"),
          run_id=run.run_id, tool_name=name, result_length=len(text or ""))
    if run.tool_interface is not None:
        run.tool_interface.metrics["backend_output_bytes"] += len(text.encode())
    return text


def resolve(tool_call_id: str, content: str) -> Run | None:
    """Deliver a JetBrains tool result; return the Run to resume, or None if the
    id is unknown/already resolved (e.g. timed out)."""
    fut = _futures.get(tool_call_id)
    if fut is None or fut.done():
        log.warning("unknown/late tool result id=%s", tool_call_id)
        # ##DELETE AFTER CORRECTION## Full late-result diagnostics.
        debug(log, "bridge.tool_result_unmatched", tool_call_id=tool_call_id,
              result_length=len(content or ""),
              known_call_ids=sorted(_futures))
        return None
    run = _runs.get(tool_call_id)
    debug(log, "bridge.tool_result_received", trace_id=run.metadata.get("trace_id") if run else None,
          run_id=run.run_id if run else None, tool_call_id=tool_call_id,
          tool_name=run.tool_names.get(tool_call_id) if run else None,
          result_length=len(content or ""))
    # A failed IDE write is normally returned as an error-shaped tool result. Do
    # not declare a project mutation in that case: fallback is only unsafe after
    # a write actually succeeded.
    tool_name = run.tool_names.get(tool_call_id, "") if run else ""
    effective_name = _effective_mutation_name(
        tool_name, run.tool_arguments.get(tool_call_id, {}) if run else {})
    required = set(run.metadata.get("required_operations", ())) if run else set()
    if (run is not None and effective_name
            and not tool_failed(content)
            and (not required or required.intersection(operations_for(effective_name)))):
        run.mutating_tool_succeeded = True
    fut.set_result(content)
    log.info("mcp_tool_result_received route=ide_bridge trace_id=%s run_id=%s id=%s name=%s",
             run.metadata.get("trace_id") if run else None,
             run.run_id if run else None, tool_call_id, tool_name)
    return run


def run_for(tool_call_id: str) -> Run | None:
    """Return the active run for a pending IDE tool call without resolving it."""
    return _runs.get(tool_call_id)


def has_inflight_mutation(run: Run) -> bool:
    """True while a write may still be applied: a direct write in progress, or a
    mutating IDE tool call whose result has not come back yet."""
    if run.metadata.get("inflight_direct_mutations", 0) > 0:
        return True
    return any(not fut.done() and _effective_mutation_name(
                   run.tool_names.get(call_id, ""), run.tool_arguments.get(call_id, {}))
               for call_id, fut in run.pending.items())


def _effective_mutation_name(name: str, arguments: dict) -> str | None:
    if is_mutating_tool(name):
        return name
    if name.rsplit("__", 1)[-1] != "execute_tool":
        return None
    command = arguments.get("command")
    if not isinstance(command, str):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    return tokens[0] if tokens and is_mutating_tool(tokens[0]) else None


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
    run.tool_arguments.clear()
    if ids:
        log.info("cleanup run tool_calls=%d", len(ids))
    debug(log, "bridge.run_discarded", trace_id=run.metadata.get("trace_id"),
          run_id=run.run_id, discarded_call_ids=ids)

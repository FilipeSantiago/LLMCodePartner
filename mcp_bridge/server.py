"""MCP engine — generic run lifecycle (provider-agnostic, pure stdlib).

Owns the background-task lifecycle and cleanup, but not *how* the model runs.
The caller injects a provider **strategy**: an `async def run(run, prompt, system,
specs) -> terminal` that drives the model and, on each tool call, uses
`bridge.call_tool`. Claude implements it with an in-process MCP server; Ollama
with a native tool loop. This module imports no LLM SDK.
"""
import asyncio
import logging
import os
from collections.abc import Awaitable, Callable

from mcp_bridge import bridge
from mcp_bridge.models import DoneEvent, ErrorEvent, Run
from mcp_bridge.registry import ToolRegistry, ToolSpec
from logger.diagnostic import debug

log = logging.getLogger("codepartner.bridge.run")

# A provider run strategy: drive the model with these tool specs, pushing events
# onto run.queue (and calling bridge.call_tool per tool call); return the
# terminal reason ("endTurn" / "maxTurnsReached").
Strategy = Callable[[Run, str, str | None, list[ToolSpec]], Awaitable[str]]

# Every run whose background task has not finished, including runs that only use
# direct IDE tools and so never appear in the bridge's pending-call table.
_live_runs: dict[str, Run] = {}


def live_runs_for_job(job_id: str) -> list[Run]:
    """Runs currently executing an implementation job in this process."""
    return [run for run in list(_live_runs.values())
            if run.metadata.get("implementation_job_id") == job_id
            and run.task is not None and not run.task.done()]


async def _run(run: Run, prompt: str, system: str | None,
               specs: list[ToolSpec], strategy: Strategy) -> None:
    terminal = "endTurn"
    try:
        if run.metadata.get("reviewer"):
            # Reviewer prompts contain IDE MCP file snapshots. Keep diagnostic
            # visibility without copying project contents into logs.
            debug(log, "run.strategy_started", trace_id=run.metadata.get("trace_id"),
                  run_id=run.run_id, reviewer_model=run.metadata.get("reviewer_model"),
                  prompt_length=len(prompt), has_system=bool(system),
                  registered_tools=[vars(spec) for spec in specs])
        else:
            debug(log, "run.strategy_started", trace_id=run.metadata.get("trace_id"),
                  run_id=run.run_id, prompt=prompt, system=system,
                  registered_tools=[vars(spec) for spec in specs])
        terminal = await strategy(run, prompt, system, specs)
    except asyncio.CancelledError:
        bridge.discard(run)
        raise
    except Exception as exc:  # surface provider/model failures into the stream
        log.exception("run strategy failed run_id=%s", run.run_id)
        await run.put(ErrorEvent(str(exc)))
    finally:
        debug(log, "run.strategy_finished", trace_id=run.metadata.get("trace_id"),
              run_id=run.run_id, terminal=terminal)
        if run.tool_interface is not None:
            try:
                from telemetry.storage import TelemetryStorage
                TelemetryStorage().record_aci_attempt(
                    run.run_id, run.metadata, run.tool_interface.summary(), run.usage)
            except Exception:
                log.exception("could not record ACI metrics run_id=%s", run.run_id)
        await run.put(DoneEvent(terminal, usage=run.usage, model_usage=run.model_usage))
        bridge.discard(run)


def start_run(prompt: str, system: str | None, specs: list[ToolSpec],
              strategy: Strategy, allowed: frozenset[str] | None = None,
              allowed_operations: frozenset[str] | None = None,
              direct_tool_handlers: dict[str, Callable[[dict], Awaitable[str]]] | None = None,
              agent_tool_interface: bool = False) -> Run:
    """Begin a bridged execution: register the tools this run's role allows and run
    the injected provider strategy in the background.

    `allowed` is the caller's role allowlist (`registry.allowed_for`); None = the full
    ceiling. It is baked into the Run, so later resumes stay inside the same role.
    """
    run = Run(registry=ToolRegistry(allowed, allowed_operations))
    run.direct_tool_handlers = dict(direct_tool_handlers or {})
    if agent_tool_interface:
        from mcp_bridge.agent_tools import AgentToolInterface, tool_specs
        compact = os.getenv("CODEPARTNER_ACI_ENABLED", "true").lower() != "false"
        run.tool_interface = AgentToolInterface(run, compact=compact)
        if compact:
            shaped = []
            for spec in specs:
                bare = spec.name.rsplit("__", 1)[-1]
                if bare == "get_git_diff_all":
                    continue
                if bare == "read_file" and isinstance(spec.schema, dict):
                    schema = dict(spec.schema)
                    original_props = schema.get("properties")
                    props = dict(original_props) if isinstance(original_props, dict) else {}
                    limit = props.get("limit")
                    props["limit"] = {**(limit if isinstance(limit, dict) else {}),
                                      "default": 100, "maximum": 200}
                    schema["properties"] = props
                    spec = ToolSpec(spec.name,
                                    "Read a project file in a contiguous window. Default 100 lines, "
                                    "maximum 200; use offset to continue. Returns total lines.", schema)
                shaped.append(spec)
            specs = [*shaped, *tool_specs()]
    registered = run.registry.register(specs)
    log.info("mcp tool registration: %s", [s.name for s in registered])
    debug(log, "run.started", run_id=run.run_id, allowed_tools=sorted(allowed) if allowed else None,
          allowed_operations=sorted(allowed_operations) if allowed_operations else None,
          advertised_tools=[vars(spec) for spec in specs], registered_tools=[vars(spec) for spec in registered],
          direct_tool_names=sorted(run.direct_tool_handlers))
    run.task = asyncio.create_task(_run(run, prompt, system, run.registry.tools(), strategy))
    _live_runs[run.run_id] = run
    run.task.add_done_callback(lambda _task: _live_runs.pop(run.run_id, None))
    return run


def resume(tool_call_id: str, content: str) -> Run | None:
    """Deliver a client tool result; return the Run to keep streaming."""
    return bridge.resolve(tool_call_id, content)

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
from mcp_bridge.models import DoneEvent, ErrorEvent, MutationEvent, ProgressEvent, Run
from mcp_bridge.registry import ToolRegistry, ToolSpec
from logger.diagnostic import debug
from telemetry import tracing

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


async def _finish_native(run: Run) -> bool:
    """Journal (or, for read-only runs, check) a native run's changes
    once, after its CLI has exited. False when the run must be reported as failed."""
    native, finish = run.native, getattr(run.native, "finish", None)
    if finish is None:
        return True
    native.finish = None
    try:
        # Shielded: a second cancellation must not interrupt journaling or checking.
        paths = await asyncio.shield(finish())
    except asyncio.CancelledError:
        log.error("native finish interrupted run_id=%s; its changes may be unjournaled",
                  run.run_id)
        raise
    except Exception as exc:
        from agent.workspace import NativeSessionError
        if isinstance(exc, NativeSessionError):
            log.error("native run failed run_id=%s: %s", run.run_id, exc)
            await run.put(ErrorEvent(str(exc)))
        else:
            log.exception("native change journal failed run_id=%s", run.run_id)
            await run.put(ErrorEvent(f"Native changes could not be journaled: {exc}"))
        return False
    for warning in native.warnings:
        log.warning("native run warning run_id=%s: %s", run.run_id, warning)
        await run.put(ProgressEvent(f"- ⚠️ {warning}\n"))
    for path in paths:
        run.mutating_tool_succeeded = True
        await run.put(MutationEvent("native_cli", {"path": path}))
    return True


async def _run(run: Run, prompt: str, system: str | None,
               specs: list[ToolSpec], strategy: Strategy) -> None:
    terminal = "endTurn"
    outcome, failure = "completed", None
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
        if run.native is not None and run.native.start is not None:
            # No baseline, no native run: a failure here precedes any write.
            await run.native.start()
        terminal = await strategy(run, prompt, system, specs)
    except asyncio.CancelledError:
        outcome = "cancelled"
        await _finish_native(run)
        bridge.discard(run)
        raise
    except Exception as exc:  # surface provider/model failures into the stream
        outcome, failure = "failed", exc
        log.exception("run strategy failed run_id=%s", run.run_id)
        await run.put(ErrorEvent(str(exc)))
    finally:
        try:
            if not await _finish_native(run):
                terminal = "failed"
            debug(log, "run.strategy_finished", trace_id=run.metadata.get("trace_id"),
                  run_id=run.run_id, terminal=terminal)
            if run.native is not None and run.metadata.get("implementation_job_id"):
                try:
                    from telemetry.storage import TelemetryStorage
                    TelemetryStorage().record_aci_attempt(
                        run.run_id, run.metadata,
                        {"execution_mode": "native",
                         "exploration_mode": run.metadata.get("exploration_mode", "off"),
                         **(run.metadata.get("native_tool_metrics")
                            or run.native.metrics or {"tool_calls": 0})},
                        run.usage)
                except Exception:
                    log.exception("could not record native metrics run_id=%s", run.run_id)
            await run.put(DoneEvent(terminal, usage=run.usage, model_usage=run.model_usage))
            bridge.discard(run)
        finally:
            tracing.finish_run(run, terminal, outcome, failure)


def start_run(prompt: str, system: str | None, specs: list[ToolSpec],
              strategy: Strategy, allowed: frozenset[str] | None = None,
              allowed_operations: frozenset[str] | None = None,
              direct_tool_handlers: dict[str, Callable[[dict], Awaitable[str]]] | None = None,
              *, trace_kind: str = "execution",
              trace_provider: str | None = None, trace_model: str | None = None,
              trace_reasoning_effort: str | None = None) -> Run:
    """Begin a bridged execution: register the tools this run's role allows and run
    the injected provider strategy in the background.

    `allowed` is the caller's role allowlist (`registry.allowed_for`); None = the full
    ceiling. It is baked into the Run, so later resumes stay inside the same role.

    `trace_*` describe the Run for tracing only: providers pass `trace_kind="llm"`
    and their provider/model; orchestration Runs keep the default.
    """
    run = Run(registry=ToolRegistry(allowed, allowed_operations))
    run.direct_tool_handlers = dict(direct_tool_handlers or {})
    registered = run.registry.register(specs)
    log.info("mcp tool registration: %s", [s.name for s in registered])
    debug(log, "run.started", run_id=run.run_id, allowed_tools=sorted(allowed) if allowed else None,
          allowed_operations=sorted(allowed_operations) if allowed_operations else None,
          advertised_tools=[vars(spec) for spec in specs], registered_tools=[vars(spec) for spec in registered],
          direct_tool_names=sorted(run.direct_tool_handlers))
    run.trace = tracing.start_run(run, trace_kind, provider=trace_provider, model=trace_model,
                                  reasoning_effort=trace_reasoning_effort)
    with tracing.task_context(run.trace):
        run.task = asyncio.create_task(_run(run, prompt, system, run.registry.tools(), strategy))
    _live_runs[run.run_id] = run
    run.task.add_done_callback(lambda _task: _live_runs.pop(run.run_id, None))
    return run


def resume(tool_call_id: str, content: str) -> Run | None:
    """Deliver a client tool result; return the Run to keep streaming."""
    return bridge.resolve(tool_call_id, content)

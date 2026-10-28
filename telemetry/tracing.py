"""OpenTelemetry tracing for Code Partner executions (exported via the OTel Collector).

The only module that touches the OpenTelemetry SDK. Everything else calls the helpers
below, which never raise and never block an execution: spans are exported by a
background thread with a bounded queue, and a failure here only logs.

Trace shape (docs/observability.md):

    codepartner.execution            one per top-level Run (`mcp_bridge.server.start_run`)
    ├── codepartner.model.routing    coder / planner model selection
    ├── codepartner.llm.execution    one per provider Run (Claude Code / Codex CLI)
    │   └── codepartner.mcp.tool_call    one per bridged MCP backend call
    └── codepartner.mcp.tool_call    calls made by an orchestration Run itself

A Run outlives HTTP requests (it pauses for IDE tool round-trips), so spans belong to
Runs, not requests. Each Run holds its own span context (`Run.trace`); tool-call spans
are parented on the Run that issued them explicitly, never on whatever context happens
to be current, so Codex HTTP callbacks and concurrent runs cannot mix traces.

Privacy: attributes pass a strict schema (`_SCHEMA`): unknown keys are dropped, every
value is type- and pattern-checked, and secret-looking values are dropped as a second
layer. No prompts, tool arguments, tool results, file contents, routing reasons or
exception messages are ever recorded.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

log = logging.getLogger("codepartner.tracing")

TRACER_NAME = "codepartner"

EXECUTION_SPAN = "codepartner.execution"
LLM_SPAN = "codepartner.llm.execution"
TURN_SPAN = "codepartner.llm.turn"
ROUTING_SPAN = "codepartner.model.routing"
TOOL_SPAN = "codepartner.mcp.tool_call"

_provider: Any | None = None       # the SDK TracerProvider when enabled, else None
_tracer: trace.Tracer | None = None
_logged_failures: set[str] = set()

# The conversation (Langfuse session) of the current request, and the execution a
# new Run belongs to. Both are copied into a Run's background task by create_task.
_session: ContextVar[str | None] = ContextVar("codepartner_trace_session", default=None)
_execution: ContextVar["_Execution | None"] = ContextVar("codepartner_trace_execution",
                                                         default=None)


# --- Attribute schema: strict allowlist first, secret scrubbing second ---------------

_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
_NAME = re.compile(r"^[A-Za-z0-9_.:/@\-\[\]]{1,128}$")
_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|pk-lf-|sk-lf-|gh[pousr]_[A-Za-z0-9]{8,}|xox[abpr]-|AKIA[0-9A-Z]{12,}"
    r"|eyJ[A-Za-z0-9_\-]{8,}\.|bearer|password|passwd|secret|api[_-]?key|token=)",
    re.IGNORECASE)

_STATUS = frozenset({"completed", "failed", "cancelled"})
_KIND = frozenset({"execution", "llm"})
_PROVIDERS = frozenset({"claude", "codex", "ollama", "anthropic", "openai", "unknown"})
_COMPLEXITY = frozenset({"simple", "medium", "complex", "low", "high", "unknown"})
_EFFORT = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})
_ROUTING_KIND = frozenset({"coder", "planner"})
_ROUTING_SOURCE = frozenset({"semantic-router", "deterministic-fallback", "planner-tier"})
_TOOL_ROUTE = frozenset({"direct", "ide_bridge", "refused"})
_TOOL_SERVER = frozenset({"jetbrains-ide", "jetbrains-standalone", "codebase-memory",
                          "codepartner-aci"})
_TOOL_STATUS = frozenset({"ok", "error"})
_ERROR_KINDS = frozenset({"tool_error", "policy_refused", "timeout", "cancelled",
                          "invalid_call", "stopped", "terminal_failed", "error_event",
                          "routing_error"})


def _enum(values: frozenset[str]):
    return lambda value: value if isinstance(value, str) and value in values else None


def _pattern(regex: re.Pattern):
    return lambda value: value if isinstance(value, str) and regex.match(value) else None


def _int(value):
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value >= 0 else None


def _float(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value >= 0 else None


def _bool(value):
    return value if isinstance(value, bool) else None


def _error_type(value):
    # A classification from _ERROR_KINDS or an exception class name; never a message.
    if not isinstance(value, str):
        return None
    if value in _ERROR_KINDS:
        return value
    return value if re.match(r"^[A-Z][A-Za-z0-9_]{0,63}$", value) else None


def _list_of(check, limit: int = 32):
    def validate(value):
        if not isinstance(value, (list, tuple, frozenset, set)):
            return None
        items = [checked for checked in (check(item) for item in value) if checked is not None]
        return items[:limit] or None
    return validate


_id, _name = _pattern(_ID), _pattern(_NAME)

_SCHEMA: dict[str, Any] = {
    # Execution identity and result.
    "codepartner.execution.id": _id,
    "codepartner.execution.kind": _enum(_KIND),
    "codepartner.execution.status": _enum(_STATUS),
    "codepartner.execution.terminal": _id,
    "codepartner.execution.duration_ms": _float,
    "codepartner.execution.tool_calls": _int,
    "codepartner.execution.tool_errors": _int,
    "codepartner.execution.errors": _int,
    "codepartner.run.id": _id,
    "codepartner.diagnostic_trace_id": _id,
    "codepartner.session.id": _id,
    "codepartner.command": _id,
    "codepartner.task.id": _id,
    "codepartner.job.id": _id,
    "codepartner.decision.id": _id,
    "codepartner.change.id": _id,
    "codepartner.plan.id": _id,
    "codepartner.provider": _enum(_PROVIDERS),
    "codepartner.model": _name,
    "codepartner.routing.strategy": _enum(_ROUTING_SOURCE),
    "codepartner.reviewer": _bool,
    "codepartner.gateway": _bool,
    "codepartner.native": _bool,
    "codepartner.native.tool_calls": _int,
    "codepartner.native.kb_calls": _int,
    "codepartner.native.out_of_scope_reads": _int,
    "codepartner.native.scope_exceptions": _int,
    "codepartner.cost_usd": _float,
    "codepartner.turn.sequence": _int,
    "codepartner.turn.tool_calls": _int,
    "codepartner.turn.native_tool_calls": _int,
    "codepartner.turn.native_tools": _list_of(_name),
    "codepartner.turn.context_tokens": _int,
    "codepartner.turn.context_measurement": _enum(frozenset({"input_proxy", "unavailable"})),
    "codepartner.aggregate.input_tokens": _int,
    "codepartner.aggregate.cache_read_tokens": _int,
    "codepartner.aggregate.cache_write_tokens": _int,
    "codepartner.aggregate.output_tokens": _int,
    "codepartner.source.first_read_bytes": _int,
    "codepartner.source.preview_reads": _int,
    "codepartner.source.preview_bytes": _int,
    "codepartner.source.preview_followup_reads": _int,
    "codepartner.source.preview_observation_expansions": _int,
    "codepartner.source.preview_observation_bytes": _int,
    "codepartner.source.explicit_full_reads": _int,
    "codepartner.source.explicit_full_bytes": _int,
    "codepartner.source.explicit_range_reads": _int,
    "codepartner.source.explicit_range_bytes": _int,
    # Routing.
    "codepartner.routing.kind": _enum(_ROUTING_KIND),
    "codepartner.routing.complexity": _enum(_COMPLEXITY),
    "codepartner.routing.candidates": _list_of(_name),
    "codepartner.routing.excluded": _list_of(_name),
    "codepartner.routing.fallbacks": _list_of(_name),
    "codepartner.routing.reasoning_effort": _enum(_EFFORT),
    "codepartner.routing.duration_ms": _float,
    # MCP tool calls.
    "codepartner.mcp.server": _enum(_TOOL_SERVER),
    "codepartner.mcp.route": _enum(_TOOL_ROUTE),
    "codepartner.mcp.status": _enum(_TOOL_STATUS),
    "codepartner.mcp.mutating": _bool,
    "codepartner.mcp.argument_keys": _list_of(_id),
    "codepartner.mcp.argument_bytes": _int,
    "codepartner.mcp.result_bytes": _int,
    "codepartner.mcp.duration_ms": _float,
    "codepartner.mcp.attempt": _int,
    # OpenTelemetry GenAI / MCP semantic conventions.
    "gen_ai.operation.name": _enum(frozenset({"invoke_agent", "execute_tool"})),
    "gen_ai.provider.name": _enum(frozenset({"anthropic", "openai", "ollama"})),
    "gen_ai.request.model": _name,
    "gen_ai.response.model": _name,
    "gen_ai.usage.input_tokens": _int,
    "gen_ai.usage.output_tokens": _int,
    "gen_ai.usage.cache_read.input_tokens": _int,
    "gen_ai.usage.cache_creation.input_tokens": _int,
    "gen_ai.tool.name": _name,
    "mcp.method.name": _enum(frozenset({"tools/call"})),
    "error.type": _error_type,
    # Langfuse mapping (trace-level filters and observation type).
    "langfuse.session.id": _id,
    "langfuse.trace.name": _id,
    "langfuse.trace.tags": _list_of(_id, limit=8),
    "langfuse.observation.type": _enum(frozenset({"span", "generation", "tool", "agent"})),
    "langfuse.trace.metadata.execution_id": _id,
    "langfuse.trace.metadata.status": _enum(_STATUS),
    "langfuse.trace.metadata.command": _id,
    "langfuse.trace.metadata.task_id": _id,
    "langfuse.trace.metadata.job_id": _id,
    "langfuse.trace.metadata.decision_id": _id,
    "langfuse.trace.metadata.provider": _enum(_PROVIDERS),
    "langfuse.trace.metadata.model": _name,
    "langfuse.trace.metadata.diagnostic_trace_id": _id,
}
# Reserved for a future Security Gateway's policy decisions; nothing writes it yet.
_RESERVED_PREFIXES = ("codepartner.security.",)


def clean_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """The subset of `attributes` the schema accepts, after validation and scrubbing."""
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        check = _SCHEMA.get(key)
        if check is None or value is None:
            continue
        value = check(value)
        if value is None:
            continue
        if isinstance(value, str) and _SECRET.search(value):
            continue
        if isinstance(value, list):
            value = [item for item in value
                     if not (isinstance(item, str) and _SECRET.search(item))]
            if not value:
                continue
        cleaned[key] = value
    return cleaned


def _set(span: Span | None, attributes: dict[str, Any]) -> None:
    """The only way instrumentation sets attributes: through the schema. Never raises."""
    if span is None:
        return
    try:
        cleaned = clean_attributes(attributes)
        if cleaned:
            span.set_attributes(cleaned)
    except Exception:
        _failed("set_attributes")


def mark_error(span: Span | None, kind: str | BaseException) -> None:
    """Mark the span failed with a classification only: never a message or traceback."""
    if span is None:
        return
    try:
        error_type = type(kind).__name__ if isinstance(kind, BaseException) else kind
        cleaned = clean_attributes({"error.type": error_type})
        span.set_attributes(cleaned)
        span.set_status(Status(StatusCode.ERROR, cleaned.get("error.type", "error")))
    except Exception:
        _failed("mark_error")


def _failed(where: str) -> None:
    # Log each kind of tracing failure once; telemetry must stay quiet and harmless.
    if where not in _logged_failures:
        _logged_failures.add(where)
        log.warning("tracing %s failed; telemetry for it is skipped", where, exc_info=True)


# --- Setup and shutdown ---------------------------------------------------------------

def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def configure_tracing(exporter: Any | None = None, *, force: bool = False) -> bool:
    """Install the tracer provider when OTEL_ENABLED is true (or `force`).

    `exporter` replaces the OTLP exporter (tests pass an in-memory exporter, exported
    synchronously). Returns whether tracing is active. Never raises.
    """
    global _provider, _tracer
    if _provider is not None:
        return True
    if not (force or _env_true("OTEL_ENABLED")):
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        os.environ.setdefault("OTEL_SERVICE_NAME", "code-partner")
        # shutdown_on_exit=False: shutdown is ours, bounded (see shutdown_tracing).
        provider = TracerProvider(resource=Resource.create(), shutdown_on_exit=False)
        if exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            os.environ.setdefault("OTEL_EXPORTER_OTLP_TIMEOUT", "5")
            # Background thread, bounded queue (spans are dropped when it is full).
            provider.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(),
                max_queue_size=int(os.getenv("OTEL_BSP_MAX_QUEUE_SIZE", "2048")),
                schedule_delay_millis=int(os.getenv("OTEL_BSP_SCHEDULE_DELAY", "2000")),
                export_timeout_millis=int(os.getenv("OTEL_BSP_EXPORT_TIMEOUT", "5000"))))
        _provider = provider
        _tracer = provider.get_tracer(TRACER_NAME)
        log.info("tracing enabled service=%s endpoint=%s", os.environ["OTEL_SERVICE_NAME"],
                 "in-memory" if exporter is not None
                 else os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"))
        return True
    except Exception:
        log.warning("tracing could not be configured; continuing without it", exc_info=True)
        _provider = _tracer = None
        return False


def shutdown_tracing(timeout: float | None = None) -> bool:
    """Flush and shut down within a hard wall-clock bound. Never raises.

    `force_flush` honours its timeout, but `shutdown` (processor thread join, exporter
    close) does not promise one, so both run in a daemon thread that is abandoned if it
    overruns. The daemon thread cannot keep the process alive.
    """
    global _provider, _tracer
    provider, _provider, _tracer = _provider, None, None
    if provider is None:
        return True
    bound = timeout if timeout is not None else float(
        os.getenv("OTEL_SHUTDOWN_TIMEOUT_SECONDS", "5"))

    def close() -> None:
        try:
            provider.force_flush(timeout_millis=int(max(bound * 0.6, 0.1) * 1000))
            provider.shutdown()
        except Exception:
            log.warning("tracing shutdown failed", exc_info=True)

    worker = threading.Thread(target=close, name="otel-shutdown", daemon=True)
    worker.start()
    worker.join(bound)
    if worker.is_alive():
        log.warning("tracing shutdown exceeded %.1fs; unexported spans are dropped", bound)
        return False
    return True


def force_flush(timeout_millis: int = 5000) -> bool:
    """Export queued spans now (smoke tests). Never raises."""
    try:
        return bool(_provider.force_flush(timeout_millis)) if _provider is not None else True
    except Exception:
        _failed("force_flush")
        return False


# --- Request scope ------------------------------------------------------------------

def set_session(conversation_id: str | None) -> None:
    """Remember the conversation of the current request for Runs it starts."""
    if _tracer is not None:
        _session.set(conversation_id)


# --- Run spans ----------------------------------------------------------------------

@dataclass
class _Execution:
    """One trace: the root `codepartner.execution` span and its accumulated results."""

    span: Span
    context: Any
    run_id: str
    session_id: str | None
    started: float = field(default_factory=time.monotonic)
    tool_calls: int = 0
    tool_errors: int = 0
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTrace:
    """A Run's own span. `context` is what this Run's tool calls are parented on: the
    LLM span for a provider Run, the root span for an orchestration Run."""

    span: Span
    context: Any
    kind: str
    execution: _Execution
    owns_root: bool
    started: float = field(default_factory=time.monotonic)
    tool_calls: int = 0
    tool_errors: int = 0
    errors: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)
    turn: "TurnTrace | None" = None
    turn_sequence: int = 0


@dataclass
class TurnTrace:
    span: Span
    context: Any
    tool_calls_at_start: int
    native_calls_at_start: int
    native_tools_at_start: dict[str, int]


def start_run(run: Any, kind: str = "execution", *, provider: str | None = None,
              model: str | None = None, reasoning_effort: str | None = None
              ) -> RunTrace | None:
    """Open the span(s) for a new Run, in the caller's context. Never raises.

    A Run started inside an execution (its parent's task) joins that trace: a provider
    Run becomes a `codepartner.llm.execution` child, an orchestration Run a nested
    execution span. Otherwise this Run is the root of a new trace; a provider Run with
    no execution around it gets both a root and an LLM span.
    """
    if _tracer is None:
        return None
    try:
        execution = _execution.get()
        owns_root = execution is None
        if owns_root:
            session_id = _session.get()
            root = _tracer.start_span(EXECUTION_SPAN, context=otel_context.Context(),
                                      kind=SpanKind.INTERNAL)
            execution = _Execution(root, trace.set_span_in_context(root), run.run_id,
                                   session_id)
            _set(root, {
                "codepartner.execution.id": run.run_id,
                "codepartner.execution.kind": "execution",
                "codepartner.session.id": session_id,
                "langfuse.session.id": session_id,
                "langfuse.trace.name": EXECUTION_SPAN,
                "langfuse.trace.metadata.execution_id": run.run_id,
                "langfuse.observation.type": "agent",
            })
        if kind == "llm":
            # Parent: the span current in the caller (a nested orchestration Run's own
            # span) when it belongs to this trace, else the root.
            current = trace.get_current_span()
            same_trace = (not owns_root and current.get_span_context().is_valid
                          and current.get_span_context().trace_id
                          == execution.span.get_span_context().trace_id)
            parent = otel_context.get_current() if same_trace else execution.context
            span = _tracer.start_span(LLM_SPAN, context=parent, kind=SpanKind.CLIENT)
            gen_ai_provider = {"claude": "anthropic", "codex": "openai"}.get(provider or "")
            _set(span, {
                "codepartner.run.id": run.run_id,
                "codepartner.execution.id": execution.run_id,
                "codepartner.execution.kind": "llm",
                "codepartner.provider": provider,
                "codepartner.model": model,
                "codepartner.routing.reasoning_effort": reasoning_effort,
                # No gen_ai.operation.name=invoke_agent: Langfuse (v4) drops model,
                # usage and cost from invoke_agent spans, and this span is the only one
                # carrying the CLI session's usage, so nothing is double counted.
                "gen_ai.provider.name": gen_ai_provider,
                "gen_ai.request.model": model,
                "langfuse.session.id": execution.session_id,
                "langfuse.observation.type": "agent",
            })
            execution.facts.setdefault("provider", provider)
            execution.facts.setdefault("model", model)
        elif owns_root:
            span = execution.span
        else:
            span = _tracer.start_span(EXECUTION_SPAN, context=otel_context.get_current(),
                                      kind=SpanKind.INTERNAL)
            _set(span, {"codepartner.run.id": run.run_id,
                        "codepartner.execution.id": execution.run_id,
                        "codepartner.execution.kind": "execution",
                        "langfuse.session.id": execution.session_id})
        run_trace = RunTrace(span, trace.set_span_in_context(span), kind, execution, owns_root)
        run_trace.attrs.update({"provider": provider, "model": model})
        trace_id = format(span.get_span_context().trace_id, "032x")
        run.metadata["otel_trace_id"] = trace_id
        # Correlates Code Partner logs (and the run_id-keyed SQLite telemetry) with traces.
        log.info("trace_started kind=%s run_id=%s execution_id=%s otel_trace_id=%s",
                 kind, run.run_id, execution.run_id, trace_id)
        return run_trace
    except Exception:
        _failed("start_run")
        return None


@contextmanager
def task_context(run_trace: RunTrace | None) -> Iterator[None]:
    """Make the Run's span and execution current around `asyncio.create_task`, so the
    task's copied context carries them; the caller's context is restored after."""
    if run_trace is None:
        yield
        return
    try:
        token = otel_context.attach(run_trace.context)
        execution_token = _execution.set(run_trace.execution)
    except Exception:
        _failed("task_context")
        yield
        return
    try:
        yield
    finally:
        try:
            _execution.reset(execution_token)
            otel_context.detach(token)
        except Exception:
            _failed("task_context")


_METADATA_ATTRIBUTES = {
    "decision_id": "codepartner.decision.id",
    "implementation_job_id": "codepartner.job.id",
    "job_id": "codepartner.job.id",
    "task_id": "codepartner.task.id",
    "implementation_task_id": "codepartner.task.id",
    "change_id": "codepartner.change.id",
    "plan_id": "codepartner.plan.id",
    "trace_id": "codepartner.diagnostic_trace_id",
}


def _command_of(metadata: dict) -> str:
    for key in ("bridge_command", "artifact_command_kind"):
        if isinstance(metadata.get(key), str):
            return metadata[key]
    for flag, name in (("mcp_bootstrap", "mcp_bootstrap"), ("artifact_command", "artifact"),
                       ("reviewer", "review"), ("task_execution", "task_execution"),
                       ("planner", "planning")):
        if metadata.get(flag):
            return name
    return "agent_turn"


def _usage_attributes(usage: dict | None) -> dict[str, Any]:
    return {key: value for key, value in _turn_usage_attributes(usage, "").items()
            if key.startswith("gen_ai.usage.")}


def _turn_usage_attributes(usage: dict | None, provider: str) -> dict[str, Any]:
    """Preserve missing provider fields as unavailable, never as measured zero."""
    if not isinstance(usage, dict):
        return {}
    from telemetry.aggregation import normalize_usage
    normalized = normalize_usage(usage)
    keys = {"gen_ai.usage.input_tokens": "total_input_tokens",
            "gen_ai.usage.output_tokens": "output_tokens",
            "gen_ai.usage.cache_read.input_tokens": "cache_read_tokens",
            "gen_ai.usage.cache_creation.input_tokens": "cache_write_tokens"}
    present = {"gen_ai.usage.input_tokens": ("input_tokens", "inputTokens"),
               "gen_ai.usage.output_tokens": ("output_tokens", "outputTokens"),
               "gen_ai.usage.cache_read.input_tokens":
                   ("cache_read_input_tokens", "cached_input_tokens"),
               "gen_ai.usage.cache_creation.input_tokens":
                   ("cache_creation_input_tokens", "cache_write_input_tokens")}
    result = {key: normalized[value] for key, value in keys.items()
              if any(isinstance(usage.get(name), (int, float)) for name in present[key])}
    if provider == "claude" and "gen_ai.usage.input_tokens" in result:
        result["codepartner.turn.context_tokens"] = normalized["total_input_tokens"]
        result["codepartner.turn.context_measurement"] = "input_proxy"
    else:
        result["codepartner.turn.context_measurement"] = "unavailable"
    return result


def start_turn(run: Any, model: str | None = None) -> None:
    """Start one provider-exposed turn; tools started meanwhile become its children."""
    rt: RunTrace | None = getattr(run, "trace", None)
    if _tracer is None or rt is None or rt.kind != "llm":
        return
    try:
        finish_turn(run)
        rt.turn_sequence += 1
        native = run.metadata.get("native_tool_metrics", {}) if isinstance(run.metadata, dict) else {}
        span = _tracer.start_span(TURN_SPAN, context=rt.context, kind=SpanKind.CLIENT)
        context = trace.set_span_in_context(span)
        rt.turn = TurnTrace(span, context, rt.tool_calls, native.get("tool_calls", 0),
                            dict(native.get("tools", {})))
        ids = {attribute: run.metadata.get(key) for key, attribute in _METADATA_ATTRIBUTES.items()
               if isinstance(run.metadata.get(key), str)}
        _set(span, {**ids, "codepartner.run.id": run.run_id,
                    "codepartner.execution.id": rt.execution.run_id,
                    "codepartner.turn.sequence": rt.turn_sequence,
                    "codepartner.turn.context_measurement": "unavailable",
                    "codepartner.provider": rt.attrs.get("provider"),
                    "codepartner.model": model or rt.attrs.get("model"),
                    "gen_ai.provider.name": {"claude": "anthropic", "codex": "openai"}.get(
                        rt.attrs.get("provider")),
                    "gen_ai.request.model": model or rt.attrs.get("model"),
                    "gen_ai.response.model": model if rt.attrs.get("provider") == "claude"
                    else None,
                    "langfuse.session.id": rt.execution.session_id,
                    "langfuse.observation.type": "generation"})
    except Exception:
        _failed("start_turn")


def record_turn_usage(run: Any, usage: dict | None, model: str | None = None) -> None:
    rt: RunTrace | None = getattr(run, "trace", None)
    if rt is None or rt.turn is None:
        return
    _set(rt.turn.span, {**_turn_usage_attributes(usage, rt.attrs.get("provider", "")),
                        "gen_ai.response.model": model})


def finish_turn(run: Any) -> None:
    rt: RunTrace | None = getattr(run, "trace", None)
    if rt is None or rt.turn is None:
        return
    try:
        turn = rt.turn
        native = run.metadata.get("native_tool_metrics", {}) if isinstance(run.metadata, dict) else {}
        native_tools = native.get("tools", {})
        names = [f"{name}:{count - turn.native_tools_at_start.get(name, 0)}"
                 for name, count in native_tools.items()
                 if isinstance(count, int) and count > turn.native_tools_at_start.get(name, 0)]
        _set(turn.span, {"codepartner.turn.tool_calls": rt.tool_calls - turn.tool_calls_at_start,
                         "codepartner.turn.native_tool_calls": max(
                             0, native.get("tool_calls", 0) - turn.native_calls_at_start),
                         "codepartner.turn.native_tools": names})
        turn.span.set_status(Status(StatusCode.OK))
        turn.span.end()
    except Exception:
        _failed("finish_turn")
    finally:
        rt.turn = None


def _main_model(model_usage: Any) -> str | None:
    """The model that did most of the work, from Claude's per-model usage breakdown
    (the CLI also reports small helper-model calls)."""
    if not isinstance(model_usage, dict) or not model_usage:
        return None

    def tokens(usage: Any) -> int:
        if not isinstance(usage, dict):
            return 0
        return sum(value for key, value in usage.items()
                   if key in {"inputTokens", "outputTokens", "cacheReadInputTokens",
                              "cacheCreationInputTokens", "input_tokens", "output_tokens"}
                   and isinstance(value, int))

    return max(model_usage, key=lambda name: tokens(model_usage[name]))


def finish_run(run: Any, terminal: str, outcome: str, error: BaseException | None = None
               ) -> None:
    """End the Run's span (and the root, if this Run owns it). Never raises."""
    run_trace: RunTrace | None = getattr(run, "trace", None)
    if run_trace is None:
        return
    try:
        finish_turn(run)
        metadata = run.metadata if isinstance(run.metadata, dict) else {}
        status = outcome
        if status == "completed" and (terminal == "failed" or run_trace.errors):
            status = "failed"
        ids = {attribute: metadata.get(key) for key, attribute in _METADATA_ATTRIBUTES.items()
               if isinstance(metadata.get(key), str)}
        native = metadata.get("native_tool_metrics") if isinstance(
            metadata.get("native_tool_metrics"), dict) else {}
        source = (run.tool_interface.summary() if run.tool_interface is not None else {})
        execution = run_trace.execution
        execution.facts.update({key: value for key, value in ids.items() if value})
        common = {
            **ids,
            "codepartner.command": _command_of(metadata),
            "codepartner.reviewer": bool(metadata.get("reviewer")),
            "codepartner.native": run.native is not None,
            "codepartner.native.tool_calls": native.get("tool_calls"),
            "codepartner.native.kb_calls": native.get("kb_calls"),
            "codepartner.native.out_of_scope_reads": len(native.get("out_of_scope_reads") or ()),
            "codepartner.native.scope_exceptions": native.get("scope_exceptions"),
            "codepartner.execution.status": status,
            "codepartner.execution.terminal": terminal,
            "codepartner.execution.duration_ms": (time.monotonic() - run_trace.started) * 1000,
            "codepartner.execution.tool_calls": run_trace.tool_calls,
            "codepartner.execution.tool_errors": run_trace.tool_errors,
            "codepartner.execution.errors": run_trace.errors,
            "codepartner.source.first_read_bytes": source.get("first_source_read_bytes"),
            "codepartner.source.preview_reads": source.get("bounded_preview_reads"),
            "codepartner.source.preview_bytes": source.get("bounded_preview_bytes"),
            "codepartner.source.preview_followup_reads": source.get("preview_followup_reads"),
            "codepartner.source.preview_observation_expansions": source.get(
                "preview_observation_expansions"),
            "codepartner.source.preview_observation_bytes": source.get(
                "preview_observation_bytes"),
            "codepartner.source.explicit_full_reads": source.get("explicit_full_file_reads"),
            "codepartner.source.explicit_full_bytes": source.get("explicit_full_file_bytes"),
            "codepartner.source.explicit_range_reads": source.get("explicit_range_reads"),
            "codepartner.source.explicit_range_bytes": source.get("explicit_range_bytes"),
        }
        if run_trace.kind == "llm":
            response_model = _main_model(getattr(run, "model_usage", None))
            usage = _usage_attributes(getattr(run, "usage", None))
            aggregate = {
                "codepartner.aggregate.input_tokens": usage.get("gen_ai.usage.input_tokens"),
                "codepartner.aggregate.output_tokens": usage.get("gen_ai.usage.output_tokens"),
                "codepartner.aggregate.cache_read_tokens": usage.get(
                    "gen_ai.usage.cache_read.input_tokens"),
                "codepartner.aggregate.cache_write_tokens": usage.get(
                    "gen_ai.usage.cache_creation.input_tokens"),
            }
            _set(run_trace.span, {**common, **aggregate,
                                  "gen_ai.response.model": response_model,
                                  "codepartner.cost_usd": metadata.get("cost_usd")})
            _end(run_trace.span, status, terminal, error)
        elif not run_trace.owns_root:
            _set(run_trace.span, common)
            _end(run_trace.span, status, terminal, error)
        # The root ends last, after every attribute it summarizes is known.
        if run_trace.owns_root:
            _finish_root(execution, common, status, terminal, error, metadata)
    except Exception:
        _failed("finish_run")


def _end(span: Span, status: str, terminal: str, error: BaseException | None) -> None:
    if status == "cancelled":
        mark_error(span, "cancelled")
    elif error is not None:
        mark_error(span, error)
    elif status == "failed":
        mark_error(span, "terminal_failed" if terminal == "failed" else "error_event")
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


def _finish_root(execution: _Execution, common: dict, status: str, terminal: str,
                 error: BaseException | None, metadata: dict) -> None:
    facts = execution.facts
    command = common["codepartner.command"]
    _set(execution.span, {
        **common,
        "codepartner.task.id": facts.get("codepartner.task.id"),
        "codepartner.job.id": facts.get("codepartner.job.id"),
        "codepartner.decision.id": facts.get("codepartner.decision.id"),
        "codepartner.provider": facts.get("provider"),
        "codepartner.model": facts.get("model"),
        "codepartner.routing.strategy": facts.get("routing_source"),
        "codepartner.execution.duration_ms": (time.monotonic() - execution.started) * 1000,
        "codepartner.execution.tool_calls": execution.tool_calls,
        "codepartner.execution.tool_errors": execution.tool_errors,
        "langfuse.trace.tags": [command, status],
        "langfuse.trace.metadata.status": status,
        "langfuse.trace.metadata.command": command,
        "langfuse.trace.metadata.task_id": facts.get("codepartner.task.id"),
        "langfuse.trace.metadata.job_id": facts.get("codepartner.job.id"),
        "langfuse.trace.metadata.decision_id": facts.get("codepartner.decision.id"),
        "langfuse.trace.metadata.provider": facts.get("provider"),
        "langfuse.trace.metadata.model": facts.get("model"),
        "langfuse.trace.metadata.diagnostic_trace_id": metadata.get("trace_id"),
    })
    _end(execution.span, status, terminal, error)


# --- Routing spans ------------------------------------------------------------------

@contextmanager
def routing_span(kind: str, complexity: str | None,
                 excluded: frozenset[str] | tuple = ()) -> Iterator[Span | None]:
    """Span around one model-routing decision. Exceptions propagate unchanged."""
    span = None
    if _tracer is not None:
        try:
            span = _tracer.start_span(ROUTING_SPAN)
            execution = _execution.get()
            _set(span, {"langfuse.session.id": execution.session_id if execution else None,
                        "codepartner.execution.id": execution.run_id if execution else None,
                        "codepartner.routing.kind": kind,
                        "codepartner.routing.complexity": complexity,
                        "codepartner.routing.excluded": sorted(excluded)})
        except Exception:
            _failed("routing_span")
            span = None
    started = time.monotonic()
    try:
        yield span
    except BaseException as exc:
        mark_error(span, exc)
        _end_routing(span, started)
        raise
    _end_routing(span, started)


def _end_routing(span: Span | None, started: float) -> None:
    if span is None:
        return
    try:
        _set(span, {"codepartner.routing.duration_ms": (time.monotonic() - started) * 1000})
        span.end()
    except Exception:
        _failed("routing_span")


def record_routing(span: Span | None, *, source: str, provider: str, model: str,
                   candidates=(), fallbacks=(), task_id: str | None = None,
                   decision_id: str | None = None, reasoning_effort: str | None = None
                   ) -> None:
    """Attach a routing decision (no free-text reason) and feed the execution root."""
    if span is None:
        return
    _set(span, {"codepartner.routing.strategy": source, "codepartner.provider": provider,
                "codepartner.model": model, "codepartner.routing.candidates": list(candidates),
                "codepartner.routing.fallbacks": list(fallbacks),
                "codepartner.routing.reasoning_effort": reasoning_effort,
                "codepartner.task.id": task_id, "codepartner.decision.id": decision_id})
    span.set_status(Status(StatusCode.OK))
    execution = _execution.get()
    if execution is not None:
        facts = {"routing_source": source, "provider": provider, "model": model,
                 "codepartner.task.id": task_id, "codepartner.decision.id": decision_id}
        execution.facts.update({key: value for key, value in facts.items() if value})


# --- MCP tool-call spans --------------------------------------------------------------

@dataclass
class ToolSpan:
    span: Span
    run_trace: RunTrace
    started: float = field(default_factory=time.monotonic)
    route: str | None = None
    refusal: str | None = None


def _bytes(arguments: Any) -> int:
    try:
        import json
        return len(json.dumps(arguments, default=str).encode())
    except Exception:
        return 0


def _server_for(bare: str, route: str | None) -> str:
    if bare.startswith("kb_"):
        return "codebase-memory"
    if bare.startswith("codepartner_"):
        return "codepartner-aci"
    return "jetbrains-standalone" if route == "direct" else "jetbrains-ide"


def start_tool(run: Any, name: str, arguments: Any) -> ToolSpan | None:
    """Open a tool-call span parented on the Run that issued the call. Never raises."""
    run_trace: RunTrace | None = getattr(run, "trace", None)
    if _tracer is None or run_trace is None:
        return None
    try:
        span = _tracer.start_span(TOOL_SPAN,
                                  context=run_trace.turn.context if run_trace.turn else run_trace.context,
                                  kind=SpanKind.CLIENT)
        bare = name.rsplit("__", 1)[-1]
        run_trace.tool_calls += 1
        run_trace.execution.tool_calls += 1
        _set(span, {"gen_ai.operation.name": "execute_tool", "mcp.method.name": "tools/call",
                    "gen_ai.tool.name": bare,
                    "codepartner.run.id": getattr(run, "run_id", None),
                    "codepartner.execution.id": run_trace.execution.run_id,
                    "codepartner.turn.sequence": run_trace.turn_sequence
                    if run_trace.turn else None,
                    "codepartner.mcp.argument_keys": sorted(arguments)
                    if isinstance(arguments, dict) else None,
                    "codepartner.mcp.argument_bytes": _bytes(arguments),
                    "langfuse.session.id": run_trace.execution.session_id,
                    "langfuse.observation.type": "tool"})
        return ToolSpan(span, run_trace)
    except Exception:
        _failed("start_tool")
        return None


def tool_route(tool: ToolSpan | None, route: str, *, mutating: bool = False,
               refusal: str = "policy_refused") -> None:
    """Record how the call was dispatched (direct / ide_bridge / refused, and why)."""
    if tool is None:
        return
    tool.route = route
    tool.refusal = refusal if route == "refused" else None
    _set(tool.span, {"codepartner.mcp.route": route, "codepartner.mcp.mutating": mutating})


def finish_tool(tool: ToolSpan | None, name: str, *, result: Any = None,
                error: BaseException | None = None) -> None:
    """End a tool-call span from the already-computed outcome. Never raises and never
    touches the result or the exception, which the caller passes through unchanged."""
    if tool is None:
        return
    try:
        from mcp_bridge.agent_tools import tool_failed
        bare = name.rsplit("__", 1)[-1]
        failure: str | BaseException | None = None
        if error is not None:
            import asyncio
            failure = ("timeout" if isinstance(error, (TimeoutError, asyncio.TimeoutError))
                       else "cancelled" if isinstance(error, asyncio.CancelledError) else error)
        elif tool.refusal is not None:
            failure = tool.refusal
        elif isinstance(result, str) and tool_failed(result):
            failure = "tool_error"
        _set(tool.span, {
            "codepartner.mcp.server": _server_for(bare, tool.route),
            "codepartner.mcp.status": "error" if failure is not None else "ok",
            "codepartner.mcp.result_bytes": len(result.encode())
            if isinstance(result, str) else None,
            "codepartner.mcp.duration_ms": (time.monotonic() - tool.started) * 1000,
        })
        if failure is not None:
            tool.run_trace.tool_errors += 1
            tool.run_trace.execution.tool_errors += 1
            mark_error(tool.span, failure)
        else:
            tool.span.set_status(Status(StatusCode.OK))
        tool.span.end()
    except Exception:
        _failed("finish_tool")


def _reset_for_tests() -> None:
    """Drop the provider without exporting (tests install their own)."""
    global _provider, _tracer
    _provider = _tracer = None
    _logged_failures.clear()

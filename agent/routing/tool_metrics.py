"""Read-only tool instrumentation for roles that run with direct read/search handlers
(planner and similar): those handlers are wrapped here.
"""

from dataclasses import dataclass, field

from mcp_bridge.read_metrics import SourceReadStats
from mcp_bridge.registry import (DOMAIN_SOURCE, OP_FILE_READ, OP_FILE_SEARCH,
                                 ROLE_OPTIMIZER, ToolRegistry, ToolSpec, allowed_for)

BUDGET_EXHAUSTED = ("Tool budget exhausted. Do not call more tools; conclude now with "
                    "the findings you already have.")
READ_BUDGET_EXHAUSTED = ("File-read budget exhausted. Conclude with the findings you have; "
                         "use outlines or searches only for a specific open question.")


@dataclass
class ToolMetrics:
    calls: int = 0
    reads: int = 0
    searches: int = 0
    other: int = 0
    failures: int = 0
    refused_over_budget: int = 0
    refused_reads: int = 0
    kb_calls: int = 0
    files: set[str] = field(default_factory=set)
    source: SourceReadStats = field(default_factory=SourceReadStats)

    def summary(self) -> dict:
        return {
            "tool_calls": self.calls, "file_reads": self.reads, "searches": self.searches,
            "other_calls": self.other, "failures": self.failures,
            "refused_over_budget": self.refused_over_budget,
            "refused_reads": self.refused_reads, "kb_calls": self.kb_calls,
            "unique_files_read": len(self.files), **self.source.summary(),
        }


def read_only_handlers(specs: list[ToolSpec], handlers: dict
                       ) -> tuple[dict, frozenset[str], frozenset[str]]:
    """Keep only optimizer-approved (read/search) handlers, as the reviewer does."""
    registry = ToolRegistry(allowed_for(ROLE_OPTIMIZER))
    registry.register(specs)
    allowed = set(registry.names())
    reads = frozenset(spec.name for spec in registry.candidates(OP_FILE_READ))
    searches = frozenset(spec.name for spec in registry.candidates(OP_FILE_SEARCH, DOMAIN_SOURCE))
    return ({name: handler for name, handler in handlers.items() if name in allowed},
            reads, searches)


def instrument_handlers(handlers: dict, reads: frozenset[str], searches: frozenset[str],
                        max_calls: int | None = None,
                        on_call=None, max_reads: int | None = None,
                        tracker=None) -> tuple[dict, ToolMetrics]:
    """Wrap direct handlers to count calls; past a budget, refuse instead of calling.

    Codebase Memory (`kb_`) calls are graph lookups and do not consume `max_calls`;
    file reads additionally stop at `max_reads`. `on_call(name, arguments, refused)` is
    awaited before each call for progress display, with `refused` set to None, "calls"
    or "reads". An optional `tracker` (planner discovery telemetry) observes each
    response; only in advise mode may it append a one-line advisory to a result.
    """
    metrics = ToolMetrics()

    def wrap(name, handler):
        async def call(arguments):
            kb = name.startswith("kb_")
            refused = ("calls" if not kb and max_calls is not None
                       and metrics.calls >= max_calls else
                       "reads" if name in reads and max_reads is not None
                       and metrics.reads >= max_reads else None)
            if on_call is not None:
                await on_call(name, arguments, refused)
            if refused == "calls":
                metrics.refused_over_budget += 1
                if tracker is not None:
                    tracker.delivered(BUDGET_EXHAUSTED)
                return BUDGET_EXHAUSTED
            if refused == "reads":
                metrics.refused_reads += 1
                if tracker is not None:
                    tracker.delivered(READ_BUDGET_EXHAUSTED)
                return READ_BUDGET_EXHAUSTED
            if kb:
                metrics.kb_calls += 1
            else:
                metrics.calls += 1
            if name in reads:
                metrics.reads += 1
                path = arguments.get("file_path") or arguments.get("path")
                if isinstance(path, str) and path:
                    metrics.files.add(path)
            elif name in searches:
                metrics.searches += 1
            elif not kb:
                metrics.other += 1
            try:
                result = await handler(arguments)
                repeated_before = metrics.source.repeated_source_reads
                metrics.source.observe(name, arguments, result, read=name in reads)
                if tracker is not None:
                    advisory = tracker.observe(
                        name, arguments, result, read=name in reads,
                        repeated=metrics.source.repeated_source_reads > repeated_before)
                    if advisory and isinstance(result, str):
                        result = result + "\n\n" + advisory
                return result
            except Exception:
                metrics.failures += 1
                raise
        return call

    return {name: wrap(name, handler) for name, handler in handlers.items()}, metrics

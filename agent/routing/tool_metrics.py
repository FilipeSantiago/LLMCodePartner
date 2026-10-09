"""Read-only tool instrumentation for non-coder roles (planner and similar).

Coder runs are measured by `AgentToolInterface`; roles that run with direct
read/search handlers wrap those handlers here instead.
"""

from dataclasses import dataclass, field

from mcp_bridge.registry import (DOMAIN_SOURCE, OP_FILE_READ, OP_FILE_SEARCH,
                                 ROLE_OPTIMIZER, ToolRegistry, ToolSpec, allowed_for)

BUDGET_EXHAUSTED = ("Tool budget exhausted. Do not call more tools; conclude now with "
                    "the findings you already have.")


@dataclass
class ToolMetrics:
    calls: int = 0
    reads: int = 0
    searches: int = 0
    other: int = 0
    failures: int = 0
    refused_over_budget: int = 0
    files: set[str] = field(default_factory=set)

    def summary(self) -> dict:
        return {
            "tool_calls": self.calls, "file_reads": self.reads, "searches": self.searches,
            "other_calls": self.other, "failures": self.failures,
            "refused_over_budget": self.refused_over_budget,
            "unique_files_read": len(self.files),
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
                        on_call=None) -> tuple[dict, ToolMetrics]:
    """Wrap direct handlers to count calls; past `max_calls`, refuse instead of calling.

    `on_call(name, arguments, refused)` is awaited before each call, for progress display.
    """
    metrics = ToolMetrics()

    def wrap(name, handler):
        async def call(arguments):
            refused = max_calls is not None and metrics.calls >= max_calls
            if on_call is not None:
                await on_call(name, arguments, refused)
            if refused:
                metrics.refused_over_budget += 1
                return BUDGET_EXHAUSTED
            metrics.calls += 1
            if name in reads:
                metrics.reads += 1
                path = arguments.get("file_path") or arguments.get("path")
                if isinstance(path, str) and path:
                    metrics.files.add(path)
            elif name in searches:
                metrics.searches += 1
            else:
                metrics.other += 1
            try:
                return await handler(arguments)
            except Exception:
                metrics.failures += 1
                raise
        return call

    return {name: wrap(name, handler) for name, handler in handlers.items()}, metrics

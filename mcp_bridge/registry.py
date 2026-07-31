"""MCP engine — tool registry (model-agnostic).

Knows nothing about OpenAI request shapes or pydantic models. Callers hand it
neutral `ToolSpec`s; it keeps a per-run, isolated set of the tools it supports.
"""
from dataclasses import dataclass
from typing import Any

# The tools this engine knows how to bridge. Anything else is ignored.
SUPPORTED = {
    "list_directory_tree",
    "get_file_text_by_path",
    "search_in_files_by_text",
}


@dataclass
class ToolSpec:
    """A neutral tool definition: what to expose to the LLM as an MCP tool."""
    name: str
    description: str
    schema: dict[str, Any]


class ToolRegistry:
    """Per-request registry of the tools exposed to the LLM for one execution.
    Isolated per Run — never shared across concurrent requests."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        """Keep the supported specs (dedup by name); return the accepted set."""
        for spec in specs:
            if spec.name in SUPPORTED and spec.name not in self._tools:
                self._tools[spec.name] = spec
        return list(self._tools.values())

    def tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def is_empty(self) -> bool:
        return not self._tools

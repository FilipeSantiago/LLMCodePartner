"""MCP engine — tool registry and the per-role tool policy (model-agnostic).

Knows nothing about OpenAI request shapes or pydantic models. Callers hand it
neutral `ToolSpec`s; it keeps a per-run, isolated set of the tools it supports.

The allowlist is **per agent role**, not global: the bridge owns the policy (below)
and every agent asks for its own slice via `allowed_for`. A tool outside an agent's
role never becomes callable for it — the Optimizer cannot edit code even if the IDE
advertises the write tools.
"""
from dataclasses import dataclass
from typing import Any

# --- capability groups ------------------------------------------------------
#
# Each group lists name variants across JetBrains products/versions; that is safe
# because `ToolRegistry.register` keeps only the intersection of the role's allowlist
# and the tools JetBrains actually advertises per request — an unadvertised name
# simply never registers.

READ = frozenset({
    "list_directory_tree",
    "get_file_text_by_path",
    "search_in_files_by_text",
})

WRITE = frozenset({
    # --- create a file (name varies by IDE/version) ---
    "create_new_file",
    "create_new_file_with_text",
    # --- edit / replace file text (name varies by IDE/version) ---
    "replace_file_text_by_path",
    "replace_specific_text",
    "replace_text_in_file",
    "apply_patch",
})

REFACTOR = frozenset({
    # --- IDE-safe refactor + tidy + self-check on edited files ---
    "rename_refactoring",
    "reformat_file",
    "get_file_problems",
})

# --- role policy ------------------------------------------------------------

ROLE_OPTIMIZER = "optimizer"
ROLE_CODER = "coder"

# What each agent role may call. The Optimizer only inspects the project to reason
# about the request; it must never modify it. The Coder applies the accepted change.
ROLES = {
    ROLE_OPTIMIZER: READ,
    ROLE_CODER: READ | WRITE | REFACTOR,
}

# The engine ceiling: every tool this bridge knows how to speak, across all roles.
# Also the un-roled default, so a caller that names no role behaves as before.
SUPPORTED = READ | WRITE | REFACTOR


def allowed_for(role: str | None) -> frozenset[str]:
    """The tool names `role` may call.

    `None` means un-roled and yields the full ceiling. An *unrecognized* role fails
    closed to READ rather than silently granting writes — a typo'd role name should
    cost capability, not safety.
    """
    if role is None:
        return SUPPORTED
    return ROLES.get(role, READ)


@dataclass
class ToolSpec:
    """A neutral tool definition: what to expose to the LLM as an MCP tool."""
    name: str
    description: str
    schema: dict[str, Any]


class ToolRegistry:
    """Per-request registry of the tools exposed to the LLM for one execution.
    Isolated per Run — never shared across concurrent requests.

    `allowed` is the role's allowlist (see `allowed_for`); None = the full ceiling.
    This is the enforcement point: a spec outside `allowed` never registers, so the
    provider never builds a handler for it.
    """

    def __init__(self, allowed: frozenset[str] | None = None) -> None:
        self._allowed = SUPPORTED if allowed is None else allowed
        self._tools: dict[str, ToolSpec] = {}

    def register(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        """Keep the specs this role allows (dedup by name); return the accepted set."""
        for spec in specs:
            if spec.name in self._allowed and spec.name not in self._tools:
                self._tools[spec.name] = spec
        return list(self._tools.values())

    def tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def is_empty(self) -> bool:
        return not self._tools

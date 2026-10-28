"""Per-run native tool policy shared by Claude and Codex hook adapters.

The provider must also restrict the parent's tool surface. Hook decisions alone
cannot confine arbitrary shell commands or every future CLI tool path.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


log = logging.getLogger("codepartner.providers.scope_guard")

# This module also runs as a standalone hook script, so it imports nothing from the project.
AGENT_TOOLS = frozenset({"Agent", "spawn_agent", "collaborationspawn_agent"})
SCOPE_EXCEPTION = "SCOPE EXCEPTION:"


@dataclass(frozen=True)
class ScopePolicy:
    role: str
    workspace: str
    own_files: tuple[str, ...] = ()
    reader_offered: bool = False
    own_ranges: dict[str, list[list[int]]] = field(default_factory=dict)
    ranges_known: bool = False
    hosted_reader: bool = False
    # The subagent types this run offers; any other Agent type is refused.
    subagent_types: tuple[str, ...] = ()
    # Bridged KB tools that return only paths, symbols and line ranges.
    navigation_tools: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return self.reader_offered and self.role in {"coder", "reviewer"}


def _decision(verdict: str, reason: str, updated: dict | None = None) -> dict:
    output: dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": verdict,
        "permissionDecisionReason": reason,
    }
    if updated is not None:
        output["updatedInput"] = updated
    return {"hookSpecificOutput": output}


def _inside(path: object, policy: ScopePolicy) -> tuple[bool, bool]:
    if not isinstance(path, str) or not path:
        return False, False
    root = Path(policy.workspace).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve(strict=False)
    if not target.is_relative_to(root):
        return False, False
    own = set()
    for entry in policy.own_files:
        candidate = Path(entry)
        own.add((candidate if candidate.is_absolute() else root / candidate).resolve(
            strict=False))
    return True, target in own


def decide(event: dict, policy: ScopePolicy, state: dict | None = None) -> dict | None:
    """Return a PreToolUse decision, or None for an allowed call. `state` persists across
    one run's calls when the caller keeps it (the in-process Claude hook does)."""
    name = str(event.get("tool_name") or "")
    args = event.get("tool_input") or {}
    if not isinstance(args, dict):
        args = {}
    if event.get("agent_id"):
        return None
    if name in AGENT_TOOLS:
        role = args.get("subagent_type") or args.get("agent_type")
        if policy.subagent_types and role not in policy.subagent_types:
            return _decision("deny", "This run's only subagents are "
                             + ", ".join(policy.subagent_types) + ".")
        if role == "reader" and policy.role == "coder" and state is not None:
            if state.get("reader_calls") and SCOPE_EXCEPTION not in str(
                    args.get("prompt") or ""):
                return _decision(
                    "deny", "Send every context question in one reader request. A further "
                    f"request needs a `{SCOPE_EXCEPTION} <question>` line in its prompt.")
            state["reader_calls"] = state.get("reader_calls", 0) + 1
        if name == "Agent" and args.get("run_in_background") is not False:
            # Left unset, the CLI starts the subagent in the background and the brief
            # arrives as a later notification that ends the parent's turn.
            return _decision("allow", "Subagents run in the foreground; the brief is this "
                             "call's result.", {**args, "run_in_background": False})
        return None
    if policy.role == "planner" and policy.hosted_reader and name in {
            "Bash", "exec_command", "functions.exec", "Read", "Grep", "Glob",
            "apply_patch", "Edit", "Write", "MultiEdit", "NotebookEdit"}:
        return _decision("deny", "Planner source inspection goes through planner_reader.")
    if not policy.active:
        return None
    if name in {"Bash", "exec_command", "functions.exec", "Grep", "Glob",
                "search_code", "search_graph", "get_code_snippet", "query_graph"}:
        return _decision("deny", "Ask the reader for source or use the restricted tools.")
    if name.startswith("mcp__") and name.rsplit("__", 1)[-1] in policy.navigation_tools:
        return None
    if name.startswith("mcp__"):
        return _decision("deny", "This parent cannot call unrestricted MCP tools; ask the reader.")
    if name in {"Read", "Edit", "MultiEdit", "Write", "NotebookEdit", "apply_patch"}:
        if name == "apply_patch":
            return _decision("deny", "Use a restricted edit tool; patches are not scoped.")
        if name in {"Edit", "MultiEdit", "Write", "NotebookEdit"} and policy.role == "reviewer":
            return _decision("deny", "Reviewer runs are read-only.")
        path = args.get("file_path") or args.get("path")
        inside, own = _inside(path, policy)
        if not inside:
            return _decision("deny", f"The project is at {policy.workspace}; use a path under it.")
        if not own:
            return _decision("deny", f"{path} is outside this role's files; ask the reader.")
        if name == "Read":
            target = Path(path)
            if not target.is_absolute():
                target = Path(policy.workspace) / target
            try:
                with target.open(encoding="utf-8", errors="replace") as handle:
                    long_file = sum(1 for _ in handle) > 200
            except OSError:
                long_file = False
            intervals = []
            if isinstance(path, str):
                relative = str(target.resolve(strict=False).relative_to(
                    Path(policy.workspace).resolve()))
                intervals = policy.own_ranges.get(relative, [])
            if policy.ranges_known and not intervals and policy.role == "coder":
                return _decision("deny", "No planned edit range exists for this file; ask the reader.")
            needs_range = long_file or bool(intervals)
            if needs_range and (not isinstance(args.get("offset"), int)
                              or args["offset"] < 1
                              or not isinstance(args.get("limit"), int)
                              or not 1 <= args["limit"] <= 200):
                return _decision("deny", "Read a planned range with offset and limit (at most 200 lines).")
            if intervals and not any(start <= args["offset"] and
                                     args["offset"] + args["limit"] - 1 <= end
                                     for start, end in intervals):
                return _decision("deny", "Read only a planned edit range; ask the reader for other content.")
    return None


def policy_from_run(run, children: list) -> ScopePolicy | None:
    native = getattr(run, "native", None)
    metadata = getattr(run, "metadata", {}) or {}
    if (native is None or not metadata.get("enforce_scope")
            or not children and not metadata.get("hosted_reader")):
        return None
    role = ("planner" if metadata.get("planner") else
            "reviewer" if metadata.get("reviewer") else
            "coder" if metadata.get("task_execution") else "other")
    if role in {"coder", "reviewer"} and "edit_files" not in metadata:
        return None
    ranges: dict[str, list[list[int]]] = {}
    for item in metadata.get("edit_ranges") or ():
        if not isinstance(item, dict):
            continue
        path, start, end = item.get("file"), item.get("start_line"), item.get("end_line")
        if isinstance(path, str) and isinstance(start, int) and isinstance(end, int):
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = Path(native.workspace) / candidate
            candidate = candidate.resolve(strict=False)
            root = Path(native.workspace).resolve()
            if candidate.is_relative_to(root):
                ranges.setdefault(str(candidate.relative_to(root)), []).append([start, end])
    from mcp_bridge.registry import KB_NAVIGATION_TOOLS

    return ScopePolicy(role, native.workspace,
                       tuple(metadata.get("edit_files") or ()),
                       any(getattr(child, "name", None) == "reader" for child in children),
                       ranges, "edit_ranges" in metadata,
                       bool(metadata.get("hosted_reader")),
                       subagent_types(children), tuple(sorted(KB_NAVIGATION_TOOLS)))


def subagent_types(children: list) -> tuple[str, ...]:
    return tuple(name for name in (getattr(child, "name", None) for child in children) if name)


def record_denial(metrics: dict, event: dict, decision: dict | None) -> None:
    if not decision:
        return
    output = decision.get("hookSpecificOutput") or {}
    if output.get("permissionDecision") != "deny":
        return
    name = str(event.get("tool_name") or "unknown")
    counts = metrics.setdefault("scope_denials", {})
    counts[name] = counts.get(name, 0) + 1
    log.warning("native scope denied tool=%s reason=%s", name,
                output.get("permissionDecisionReason"))


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    policy_path = Path(sys.argv[1])
    policy = ScopePolicy(**json.loads(policy_path.read_text()))
    event = json.load(sys.stdin)
    result = decide(event, policy)
    if result is not None:
        print(json.dumps(result))
        if result["hookSpecificOutput"]["permissionDecision"] == "deny":
            log.warning("native scope denied tool=%s", event.get("tool_name"))
            with (policy_path.parent / "scope_denials.jsonl").open("a") as handle:
                handle.write(json.dumps({"tool": event.get("tool_name"),
                                         "reason": result["hookSpecificOutput"]["permissionDecisionReason"]})
                             + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

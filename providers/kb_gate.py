"""KB-first nudge for native agents' own search tools.

Native agents search with Claude Code's Grep/Glob/Bash, which the bridge's IDE gates never
see. `NativeKbGate` decides, per call, whether a broad symbol search should first be
answered by Codebase Memory. It only advises: a gated call returns a reason to the model,
and the identical call repeated runs unchanged. Literal, import, usage, single-file and
tests/docs searches always run, and nothing is gated once the agent has used the KB.
"""

import json
import os
import re
import shlex

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_SCOPED_PATH = re.compile(r"(?:test|docs?|\.md|config|\.ya?ml|\.json)")
_SEARCH_COMMANDS = {"grep", "egrep", "rg", "ag"}
_SHELL_OPERATORS = ("|", "&&", "||", ";", ">", "<", "`", "$(")


def classify_search(arguments: dict) -> tuple[list[str] | None, str | None]:
    """Identifier names a text search looks for, or (None, why it is not a symbol search)."""
    query = next((arguments.get(key) for key in ("q", "query", "text", "searchText", "regex",
                                                 "pattern")
                  if isinstance(arguments.get(key), str)), None)
    if not query:
        return None, "literal"
    raw = re.sub(r"^(?:\\b|\^)+", "", query.strip())
    raw = re.sub(r"(?:\\b)+$", "", raw)
    names = [part.strip() for part in raw.split("|")]
    if not names or not all(_IDENTIFIER.fullmatch(part) for part in names):
        if re.search(r"\b(?:import|from)\b", query):
            return None, "intent_import"
        if re.search(r"(?:\w\s*\(|\.\w|=\s*\w)", query):
            return None, "intent_usage"
        return None, "literal"
    for key in ("paths", "path", "file_path", "fileMask", "file_mask", "directory"):
        value = str(arguments.get(key) or "").lower()
        if value and _SCOPED_PATH.search(value):
            return None, "intent_tests_docs"
        if value and key in {"paths", "path", "file_path", "directory"}:
            return None, "intent_tests_docs"
    return names, None


def _shell_search(command: str) -> tuple[str, str | None] | None:
    """(pattern, path) of a simple recursive grep/rg command, else None."""
    if any(op in command for op in _SHELL_OPERATORS):
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if words[:2] == ["git", "grep"]:
        words, recursive = words[2:], True
    elif words and os.path.basename(words[0]) in _SEARCH_COMMANDS:
        recursive = os.path.basename(words[0]) != "grep" and os.path.basename(words[0]) != "egrep"
        words = words[1:]
    else:
        return None
    positional: list[str] = []
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if word.startswith("-"):
            if re.fullmatch(r"-[A-Za-z]*[rR][A-Za-z]*|--recursive", word):
                recursive = True
            if word in {"-e", "--regexp", "-g", "--glob", "-t", "--type", "--include",
                        "-A", "-B", "-C", "-m"}:
                skip = word not in {"-e", "--regexp"}
            continue
        positional.append(word)
    if not recursive or not positional:
        return None
    return positional[0], positional[1] if len(positional) > 1 else None


class NativeKbGate:
    def __init__(self, kb_tool: str, max_gates: int = 5):
        self.kb_tool = kb_tool  # e.g. mcp__jetbrains__kb_search_graph
        self.max_gates = max_gates
        self.kb_used = False
        self.gated: set[str] = set()
        self.repeated_anyway = 0
        self.glob_advised = False

    def observe(self, tool_name: str) -> None:
        """Every tool call the agent makes; any KB call turns the gate off."""
        if tool_name.rsplit("__", 1)[-1].startswith("kb_"):
            self.kb_used = True

    def decide(self, tool_name: str, tool_input: dict, workspace: str | None = None) -> str | None:
        """A reason to return instead of running this call, or None to let it run."""
        if self.kb_used:
            return None
        signature = json.dumps([tool_name, tool_input], sort_keys=True, default=str)
        if signature in self.gated:
            self.repeated_anyway += 1
            return None
        names = self._names(tool_name, tool_input or {}, workspace)
        if names is None or len(self.gated) >= self.max_gates:
            return None
        self.gated.add(signature)
        if tool_name == "Glob":
            return ("Search not run: map the project with Codebase Memory before scanning the "
                    "whole repository. Call mcp__jetbrains__kb_get_architecture, or "
                    f"{self.kb_tool} with a name_pattern for the symbols you need; KB results "
                    "are hints, verify them in files. To run this search anyway, repeat the "
                    "identical call.")
        return (f"Search not run: for symbol/architecture questions call {self.kb_tool} with "
                f"name_pattern \"{'|'.join(names)}\" (or mcp__jetbrains__kb_trace_path for "
                "dependencies) first; KB results are hints, verify them in files. To run "
                "this search anyway, repeat the identical call.")

    def summary(self) -> dict:
        return {"advised": len(self.gated), "repeated_anyway": self.repeated_anyway,
                "kb_used": self.kb_used}

    def _names(self, tool_name: str, tool_input: dict, workspace: str | None) -> list[str] | None:
        if tool_name == "Grep":
            path = tool_input.get("path")
            if path and not self._directory(path, workspace):
                return None  # a single file
            arguments = {"pattern": tool_input.get("pattern")}
            if tool_input.get("glob") or tool_input.get("type"):
                arguments["fileMask"] = str(tool_input.get("glob") or tool_input.get("type"))
            if path and _SCOPED_PATH.search(str(path).lower()):
                return None
            return classify_search(arguments)[0]
        if tool_name == "Bash":
            search = _shell_search(str(tool_input.get("command") or ""))
            if search is None:
                return None
            pattern, path = search
            if path and (not self._directory(path, workspace)
                         or _SCOPED_PATH.search(path.lower())):
                return None
            return classify_search({"pattern": pattern})[0]
        if tool_name == "Glob":
            pattern = str(tool_input.get("pattern") or "")
            if self.glob_advised or tool_input.get("path") or not pattern.startswith("**/"):
                return None
            self.glob_advised = True
            return []
        return None

    @staticmethod
    def _directory(path: str, workspace: str | None) -> bool:
        if path in {".", "./", ""}:
            return True
        full = path if os.path.isabs(path) or not workspace else os.path.join(workspace, path)
        return os.path.isdir(full)

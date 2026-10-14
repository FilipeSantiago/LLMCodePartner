"""One short progress line per IDE tool call, e.g. "Reading `src/app.py`"."""

import re
import shlex

from mcp_bridge.registry import (
    OP_FILE_CREATE, OP_FILE_DELETE, OP_FILE_FORMAT, OP_FILE_PATCH, OP_FILE_READ,
    OP_FILE_REPLACE, OP_FILE_SEARCH, OP_SHELL_EXECUTE, OP_SYMBOL_RENAME, operations_for)

_PATH_KEYS = ("file_path", "filePath", "pathInProject", "path", "directoryPath")
_SEARCH_KEYS = ("q", "query", "pattern", "searchText", "text", "nameKeyword", "regex")
_PATCH_HEADER = re.compile(
    r"^(?:\*\*\* (Add|Update|Delete) File: (.+)|\+\+\+ (?:b/)?(.+))$", re.MULTILINE)
_VERBS = {
    OP_FILE_CREATE: "Writing", OP_FILE_REPLACE: "Editing", OP_FILE_DELETE: "Deleting",
    OP_SYMBOL_RENAME: "Renaming in", OP_FILE_FORMAT: "Formatting",
    "notebook_content_read": "Reading", "notebook_replace": "Editing",
    "notebook_patch": "Editing",
}


def _path(arguments: dict) -> str:
    for key in _PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    files = arguments.get("files")
    if isinstance(files, list) and files and all(isinstance(item, str) for item in files):
        return ", ".join(files)
    return ""


def _dispatched(arguments: dict) -> tuple[str, dict] | None:
    """`execute_tool` carries the real tool and its `--key value` flags in one string."""
    command = arguments.get("command")
    try:
        tokens = shlex.split(command) if isinstance(command, str) else []
    except ValueError:
        return None
    if not tokens:
        return None
    flags = {}
    for key, value in zip(tokens[1::2], tokens[2::2]):
        if key.startswith("--"):
            flags[key[2:]] = value
    return tokens[0], flags


def _patch_line(arguments: dict) -> str:
    text = arguments.get("input") or arguments.get("patch") or ""
    verbs: dict[str, str] = {}
    for kind, codex_path, diff_path in _PATCH_HEADER.findall(text if isinstance(text, str) else ""):
        path = (codex_path or diff_path).strip()
        if path and path != "/dev/null":
            verbs.setdefault(path, {"Add": "Writing", "Delete": "Deleting"}.get(kind, "Editing"))
    if not verbs:
        return "Editing files (patch)"
    groups: dict[str, list[str]] = {}
    for path, verb in verbs.items():
        groups.setdefault(verb, []).append(f"`{path}`")
    return "; ".join(f"{verb} {', '.join(paths)}" for verb, paths in groups.items())


def describe_call(name: str, arguments: dict) -> str:
    """What one tool call does, in a few words (no model tokens involved)."""
    bare = name.rsplit("__", 1)[-1]
    if bare == "execute_tool" and (inner := _dispatched(arguments)):
        bare, arguments = inner
    kb_labels = {
        "kb_get_architecture": "Reading Codebase Memory architecture",
        "kb_search_graph": "Searching Codebase Memory graph",
        "kb_trace_path": "Tracing Codebase Memory dependencies",
        "kb_index_status": "Checking Codebase Memory index status",
    }
    if bare in kb_labels:
        return kb_labels[bare]
    if bare == "kb_get_file_outline":
        path = arguments.get("file_path")
        return (f"Reading indexed outline of `{path}`" if isinstance(path, str) and path
                else "Reading a Codebase Memory file outline")
    operations = operations_for(bare)
    path = _path(arguments)
    if OP_FILE_PATCH in operations:
        return _patch_line(arguments)
    verb = next((_VERBS[op] for op in sorted(operations) if op in _VERBS), None)
    if verb:
        return f"{verb} `{path}`" if path else f"{verb} a file"
    if OP_FILE_READ in operations:
        offset, limit = arguments.get("offset"), arguments.get("limit")
        span = (f" (lines {offset}–{offset + limit - 1})"
                if isinstance(offset, int) and isinstance(limit, int) else "")
        return f"Reading `{path or 'a file'}`{span}"
    if bare in {"list_directory_tree", "list_files_in_folder"}:
        return f"Listing `{path or '.'}`"
    query = next((arguments[key] for key in _SEARCH_KEYS
                  if isinstance(arguments.get(key), str) and arguments[key]), None)
    if query and (operations & {OP_FILE_SEARCH, "file_text_search"} or "search" in bare
                  or "find" in bare):
        return f"Searching for `{query}`"
    if OP_SHELL_EXECUTE in operations or bare == "execute_terminal_command":
        command = str(arguments.get("command", ""))
        return f"Running `{command[:60]}{'…' if len(command) > 60 else ''}`"
    return f"`{bare}`" + (f" on `{path}`" if path else "")

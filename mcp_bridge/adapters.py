"""Schema-checked adapters for supported IDE tool families."""
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
import shlex
from typing import Any
@dataclass(frozen=True)
class AdapterCheck: ok: bool; reason: str
_REQUIRED = {"native_read": ("file_path",), "legacy_read": ("file_path",), "file_search": ("q",), "search": ("q",), "inspect": (), "dispatcher": ("command",), "create": ("pathInProject", "text"), "replace": (), "patch": (), "delete": (), "rename": (), "format": (), "shell": (), "notebook_read": (), "notebook_edit": ()}
def validate_adapter(adapter: str, schema: dict) -> AdapterCheck:
    props = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(x, str) for x in required): return AdapterCheck(False, "schema_required_not_string_list")
    needed = ("path",) if adapter == "legacy_read" else _REQUIRED.get(adapter, ())
    if props and any(x not in props for x in needed): return AdapterCheck(False, "missing_required_adapter_fields:" + ",".join(x for x in needed if x not in props))
    if adapter == "native_read" and props and "limit" in props and props["limit"].get("maximum", 5000) > 5000: return AdapterCheck(False, "read_limit_exceeds_5000")
    return AdapterCheck(True, "supported")
def prepare_arguments(adapter: str, schema: dict, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict): raise ValueError("tool arguments must be an object")
    missing = [x for x in _REQUIRED.get(adapter, ()) if x not in arguments]
    if missing: raise ValueError("missing required arguments: " + ", ".join(missing))
    out = dict(arguments)
    if adapter == "native_read": out.pop("max_lines", None); out["limit"] = min(int(out.get("limit", 5000)), 5000)
    if adapter == "legacy_read": out = {"path": out["file_path"]}
    return out


def prepare_dispatcher_read(arguments: dict[str, Any]) -> dict[str, str]:
    """Encode the documented nested ``read_file`` call for ``execute_tool``.

    The outer tool remains the exact advertised dispatcher.  A supplied
    ``limit`` is preserved so reads can use distinct deterministic arguments.
    """
    if not isinstance(arguments, dict) or not isinstance(arguments.get("file_path"), str):
        raise ValueError("dispatcher read requires string file_path")
    parts = ["read_file", "--file_path", shlex.quote(arguments["file_path"])]
    if "limit" in arguments:
        limit = arguments["limit"]
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5000:
            raise ValueError("dispatcher read limit must be an integer from 1 to 5000")
        parts.extend(("--limit", str(limit)))
    if "offset" in arguments:
        offset = arguments["offset"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset <= 0:
            raise ValueError("dispatcher read offset must be a positive, 1-based integer")
        parts.extend(("--offset", str(offset)))
    return {"command": " ".join(parts)}


def prepare_dispatcher_terminal(command: str, timeout_ms: int) -> dict[str, str]:
    """Encode a shell command as ``execute_tool``'s nested terminal call.

    The dispatcher only accepts ``<ide tool> --paramName value``; a bare shell
    command or a positional argument is rejected before anything runs.  The
    user's shell is used so project tools such as ``uv`` resolve on PATH.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("dispatcher terminal call requires a command")
    return {"command": " ".join((
        "execute_terminal_command", "--command", shlex.quote(command.strip()),
        "--executeInShell", "true", "--timeout", str(int(timeout_ms)),
    ))}


_PATH_FIELDS = ("file_path", "filePath", "pathInProject", "path")
_SOURCE_PATH_FIELDS = ("oldPath", "old_path", "source", "sourcePath", "source_path")
_DESTINATION_PATH_FIELDS = (
    "newPath", "new_path", "destination", "destinationPath", "destination_path",
)
_PATH_ARRAY_FIELDS = ("files", "paths", "file_paths", "filesToRebuild")
_PATCH_HEADER = re.compile(
    r"^(?:\*\*\* (?:Add|Update|Delete) File:|\*\*\* Move to:|"
    r"--- |\+\+\+ |rename from |rename to |copy from |copy to )(?P<path>.+?)\s*$"
)


def _project_path(value: object, *, unified_diff: bool = False) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        return None
    value = value.strip().split("\t", 1)[0]
    if value == "/dev/null":
        return None
    if unified_diff and value.startswith(("a/", "b/")):
        value = value[2:]
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", "..", ".git"} for part in path.parts):
        return None
    return path.as_posix()


def _append_path(paths: list[str], value: object, *, unified_diff: bool = False) -> None:
    path = _project_path(value, unified_diff=unified_diff)
    if path and path not in paths:
        paths.append(path)


def _field_paths(arguments: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for field in fields:
        _append_path(paths, arguments.get(field))
    return paths


def _patch_paths(arguments: dict[str, Any]) -> list[str]:
    patch = next((arguments.get(key) for key in ("input", "patch", "patch_text", "text")
                  if isinstance(arguments.get(key), str)), None)
    if patch is None:
        return []
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            try:
                tokens = shlex.split(line)
            except ValueError:
                continue
            if len(tokens) >= 4:
                _append_path(paths, tokens[2], unified_diff=True)
                _append_path(paths, tokens[3], unified_diff=True)
            continue
        match = _PATCH_HEADER.match(line)
        if match:
            _append_path(
                paths, match.group("path"),
                unified_diff=line.startswith(("--- ", "+++ ")),
            )
    return paths


def mutation_targets(adapter: str, arguments: dict[str, Any]) -> tuple[str, ...]:
    """Return canonical project-relative targets for one validated mutation call.

    An empty tuple is deliberately a validation failure for a mutating adapter;
    callers must reject the operation instead of guessing after it executes.
    """
    if not isinstance(arguments, dict):
        return ()
    if adapter in {"create", "replace", "delete", "notebook_edit"}:
        paths = _field_paths(arguments, _PATH_FIELDS)
    elif adapter == "patch":
        paths = _patch_paths(arguments)
    elif adapter == "rename":
        paths = _field_paths(
            arguments, _SOURCE_PATH_FIELDS + _DESTINATION_PATH_FIELDS + _PATH_FIELDS)
    elif adapter == "format":
        paths = _field_paths(arguments, _PATH_FIELDS)
        for field in _PATH_ARRAY_FIELDS:
            values = arguments.get(field)
            if isinstance(values, (list, tuple)):
                for value in values:
                    _append_path(paths, value)
    else:
        return ()
    return tuple(paths)

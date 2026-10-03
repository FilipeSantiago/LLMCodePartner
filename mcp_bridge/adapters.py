"""Schema-checked adapters for supported IDE tool families."""
from dataclasses import dataclass
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
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("dispatcher read offset must be a non-negative integer")
        parts.extend(("--offset", str(offset)))
    return {"command": " ".join(parts)}

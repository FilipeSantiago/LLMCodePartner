"""Run-local, provider-neutral observations for coding agents.

The IDE transport and mutation journal still own execution. This module controls
only calls made by a coder and the text delivered back to that coder.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from collections import Counter, OrderedDict
from pathlib import PurePosixPath
from typing import Awaitable, Callable

from mcp_bridge.adapters import mutation_targets
from mcp_bridge.edit_validation import validate_postimages, validator_for
from mcp_bridge.registry import DEFINITIONS, ToolSpec, is_mutating_tool

WINDOW = 100
MAX_WINDOW = 200
MAX_BYTES = 4000
SEARCH_MATCHES = 30
MAX_SAVED = 128
MAX_SAVED_BYTES = 2_000_000
_NUMBERED = re.compile(r"^L(\d+): ?(.*)$")
_TRUNCATED = re.compile(r"…\d+ lines truncated…")
_PATH_KEYS = frozenset({
    "file_path", "filePath", "path", "pathInProject", "oldPath", "newPath",
    "sourcePath", "targetPath", "directoryPath", "directory", "folder",
    "paths", "files", "file_paths",
})
_READ = frozenset({"read_file", "get_file_text_by_path", "codepartner_read_file"})
_SEARCH = frozenset({"search_file", "search_text", "search_regex", "find_text",
                     "exact_search", "search_in_files_by_text", "codepartner_search_in_file"})
_DISPATCHER_ALLOWED = (frozenset(alias for definition in DEFINITIONS
                                 for alias in definition.aliases
                                 if not alias.startswith("codepartner_"))
                       | {"execute_terminal_command",
                          "get_python_environment", "get_run_configurations", "build_project"}) - {"execute_tool"}


def tool_failed(raw: str | None) -> bool:
    text = (raw or "").lstrip()
    if text.lower().startswith(("error", "failed", "access denied")):
        return True
    try:
        value = json.loads(text)
    except ValueError:
        return False
    if isinstance(value, dict):
        return bool(value.get("isError") or value.get("is_error") or value.get("error")) \
            or value.get("status") in {"failed", "error"}
    return False


def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec("codepartner_read_file",
                 "Read a contiguous, numbered window of a project file. Defaults to 100 lines; "
                 "returns total lines and a continuation. Search first for large files.",
                 {"type": "object", "properties": {
                     "path": {"type": "string"},
                     "start_line": {"type": "integer", "minimum": 1},
                     "end_line": {"type": "integer", "minimum": 1},
                 }, "required": ["path"]}),
        ToolSpec("codepartner_search_in_file",
                 "Find a literal in one project file. Returns line numbers and short previews.",
                 {"type": "object", "properties": {
                     "path": {"type": "string"}, "query": {"type": "string"},
                 }, "required": ["path", "query"]}),
        ToolSpec("codepartner_get_observation",
                 "Reload a bounded page of an earlier authorized result by id. "
                 "Use observation_id='index' to list observed ranges and edits.",
                 {"type": "object", "properties": {
                     "observation_id": {"type": "string"},
                     "offset": {"type": "integer", "minimum": 0},
                 }, "required": ["observation_id"]}),
    ]


def _bare(name: str) -> str:
    return name.rsplit("__", 1)[-1]


def _clip(text: str, limit: int = MAX_BYTES) -> tuple[str, bool]:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text, False
    return data[:limit].decode("utf-8", errors="ignore"), True


def _unwrap(raw: str) -> str:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return raw or ""
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        blocks = value["content"]
        if all(isinstance(block, dict) and isinstance(block.get("text"), str)
               for block in blocks):
            return "\n".join(block["text"] for block in blocks)
    return raw or ""


def _path_reason(path: object, *, write: bool = False, exclusions: tuple[str, ...] = ()) -> str | None:
    if (not isinstance(path, str) or not path or len(path) > 512 or "\\" in path or ":" in path
            or "!" in path and not path.startswith("!")):
        return "a project-relative path is required"
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(part in {"..", "."} for part in path.split("/")):
        return "path escapes the project"
    parts = parsed.parts
    if ".git" in parts:
        return ".git is protected"
    if any(part == ".codepartner" for part in parts):
        return "CodePartner internal files are protected"
    if any(part.startswith(".env") and part != ".env.example" for part in parts):
        return "environment files may contain secrets; use .env.example or configuration source"
    if write and parts and parts[0] == "openspec":
        return "OpenSpec artifacts are protected during coding"
    if any(parsed.as_posix() == item or parsed.as_posix().startswith(item.rstrip("/") + "/")
           for item in exclusions):
        return "path was excluded by the user"
    return None


def _dispatcher(arguments: dict) -> tuple[str, dict]:
    command = arguments.get("command")
    if not isinstance(command, str):
        return "", {}
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "", {}
    if not tokens:
        return "", {}
    params: dict = {}
    for index, token in enumerate(tokens[:-1]):
        if token.startswith("--"):
            params[token[2:]] = tokens[index + 1]
    return tokens[0], params


def _paths(arguments: dict) -> tuple[str, ...]:
    found: list[str] = []
    def visit(value, key=""):
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key in _PATH_KEYS:
            found.append(value)
    visit(arguments)
    return tuple(dict.fromkeys(found))


def _read_result(raw: str, path: str, start: int, count: int) -> tuple[str, int, str]:
    text = _unwrap(raw)
    if (text.lstrip().lower().startswith("error")
            or re.search(r"\b(?:does not exist|doesn't exist|cannot be opened|can't be opened|not found)\b",
                         text, re.I)):
        shown, _ = _clip(text)
        return f"READ FAILED\n{path}\n{shown}", 0, ""
    numbered: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = _NUMBERED.match(line)
        if match:
            numbered.append((int(match.group(1)), match.group(2)))
    if not numbered:
        if not text.strip():
            return "READ COMPLETE\n" + path + " — empty file (0 lines).\ntruncated=false", 0, ""
        if _TRUNCATED.search(text):
            return "READ ERROR\nIDE response contained no verifiable numbered lines.", 0, ""
        # Legacy readers can provide plain whole-file text. Their observations
        # are still windowed, though the IDE may have sent more internally.
        lines = text.splitlines()
        numbered = list(enumerate(lines, 1))
    total = numbered[-1][0]
    by_number = {number: value for number, value in numbered}
    stop = min(start + count - 1, total)
    if start > total:
        return (f"READ COMPLETE\n{path} — 0 lines returned; file has {total} lines.\n"
                "truncated=false", 0, "")
    missing = [number for number in range(start, stop + 1) if number not in by_number]
    if missing:
        return (f"READ ERROR\n{path}: IDE omitted requested line {missing[0]}; "
                "a contiguous window could not be verified.", 0, "")
    rendered = [f"{number:>5} | {by_number[number]}" for number in range(start, stop + 1)]
    displayed = []
    used = 0
    long_line = False
    for line in rendered:
        encoded = line.encode()
        additional = len(encoded) + (1 if displayed else 0)
        if used + additional > MAX_BYTES:
            if not displayed:
                displayed.append(encoded[:MAX_BYTES].decode("utf-8", errors="ignore"))
                long_line = True
            break
        displayed.append(line)
        used += additional
    body = "\n".join(displayed)
    clipped = len(displayed) < len(rendered) or long_line
    returned = len(displayed)
    actual_end = start + returned - 1 if returned else start - 1
    more = actual_end < total
    header = (f"READ COMPLETE\n{path} — lines {start}–{actual_end} of {total}\n"
              f"truncated={'true' if more or clipped else 'false'}\n")
    if more:
        header += (f"Remaining: {actual_end + 1}–{total}\nSuggested continuation: "
                   f"codepartner_read_file(path={path!r}, start_line={actual_end + 1}, "
                   f"end_line={min(actual_end + WINDOW, total)})\n")
    if long_line:
        header += "The displayed line exceeds the byte limit; its text is clipped. Use observation expansion for more.\n"
    return header + body, returned, body


def _search_result(raw: str, query: str, exclusions: tuple[str, ...]) -> tuple[str, int]:
    text = _unwrap(raw)
    query = query[:160]
    if text.lstrip().lower().startswith("error"):
        shown, _ = _clip(text)
        return f"SEARCH FAILED\nquery={query!r}\n{shown}", 0
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
        safe = [item for item in items if isinstance(item, dict)
                and not _path_reason(item.get("filePath"), exclusions=exclusions)]
        output = []
        for item in safe[:SEARCH_MATCHES]:
            path = item["filePath"]
            line = item.get("startLine") or item.get("line")
            output.append(f"{path}:{line}" if line else path)
        more = bool(payload.get("more")) or len(safe) > SEARCH_MATCHES
        listing, clipped = _clip("\n".join(output))
        more = more or clipped
        heading = (f"SEARCH COMPLETE\nquery={query!r}\n{len(output)} matches returned"
                   + ("; more matches available" if more else "")
                   + f"\ntruncated={'true' if more else 'false'}")
        if len(safe) != len(items):
            heading += f"\n{len(items) - len(safe)} protected/excluded result(s) omitted"
        return heading + ("\n" + listing if output else "\n0 visible matches."), len(output)
    lines = [line for line in text.splitlines()
             if not _path_reason(line.split(":", 1)[0], exclusions=exclusions)]
    output = "\n".join(lines[:SEARCH_MATCHES])
    output, clipped = _clip(output)
    more = clipped or len(lines) > SEARCH_MATCHES
    return (f"SEARCH COMPLETE\nquery={query!r}\n{len(lines[:SEARCH_MATCHES])} matches returned\n"
            f"truncated={'true' if more else 'false'}\n" + (output or "0 matches found.")), len(lines[:SEARCH_MATCHES])


def _safe_search_raw(raw: str, exclusions: tuple[str, ...]) -> str:
    text = _unwrap(raw)
    try:
        payload = json.loads(text)
    except ValueError:
        return "\n".join(line for line in text.splitlines()
                         if not _path_reason(line.split(":", 1)[0], exclusions=exclusions))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return ""
    payload["items"] = [item for item in payload["items"]
                        if isinstance(item, dict)
                        and not _path_reason(item.get("filePath"), exclusions=exclusions)]
    return json.dumps(payload, ensure_ascii=False)


def _redact_command_output(raw: str) -> str:
    lines = []
    for line in _unwrap(raw).splitlines():
        if re.search(r"(?:^|[ /])\.env(?!\.example)(?:[. /:]|$)", line):
            lines.append("[protected environment file output omitted]")
            continue
        line = re.sub(
            r"(?i)\b(password|api[_-]?key|secret|token)\b(\s*[:=]\s*)\S+",
            r"\1\2[REDACTED]", line)
        lines.append(line)
    return "\n".join(lines)


def _check_command(command: str) -> str | None:
    if re.search(r"(?:[;|`<>]|\$\(|[\r\n])", command):
        return "shell control or redirection is not allowed"
    if re.search(r"(?:^|[\s/])\.env(?!\.example)(?:[.\s/]|$)", command):
        return "environment files may contain secrets"
    try:
        words = shlex.split(command)
    except ValueError:
        return "command could not be parsed"
    if not words:
        return "empty command"
    if words[:2] == ["uv", "run"]:
        words = words[2:]
    if (words[0].rsplit("/", 1)[-1] in {"python", "python3"}
            and words[1:3] == ["-m", "pytest"]):
        return None
    if words[0] in {"pytest", "py.test"}:
        return None
    if words[:2] in (["ruff", "check"], ["cargo", "test"], ["cargo", "check"],
                     ["go", "test"], ["node", "--test"]):
        return None
    if words[0] in {"npm", "pnpm", "yarn"} and words[1:3] in (
            ["test"], ["run", "test"], ["run", "build"], ["run", "lint"]):
        return None
    return "only test, build, and file-local static-check commands are allowed"


def _command_result(raw: str) -> str:
    text = _unwrap(raw)
    if not text.strip():
        return "COMMAND COMPLETE\nexit_code=unknown\nNo stdout was produced.\ntruncated=false"
    exit_match = re.search(r"(?:exit[_ ]?code|returncode)\D{0,8}(-?\d+)", text, re.I)
    exit_code = int(exit_match.group(1)) if exit_match else None
    totals = re.findall(r"\b\d+\s+(?:failed|passed|skipped|error|errors)\b", text, re.I)
    failed = text.lstrip().lower().startswith("error") or exit_code not in (None, 0) or any(re.match(r"[1-9]\d*\s+(?:failed|error)", item, re.I)
                                                 for item in totals)
    lines = text.splitlines()
    if totals:
        detail = [line for line in lines if line.startswith(("FAILED ", "ERROR ", "E ", "> "))]
        shown = "\n".join((totals[-5:] + detail[:12]))
    else:
        shown = "\n".join(lines[:35] + (lines[-8:] if len(lines) > 45 else []))
    shown, clipped = _clip(shown)
    status = "FAILED" if failed else "SUCCESS" if exit_code == 0 else "COMPLETE"
    return (f"COMMAND {status}\nexit_code={exit_code if exit_code is not None else 'unknown'}\n"
            f"truncated={'true' if clipped or shown != text else 'false'}\n" + shown)


def _complete_source(raw: str) -> str | None:
    """Take a complete numbered IDE read; refuse any head/tail or partial read."""
    text = _unwrap(raw)
    if text.lstrip().lower().startswith("error") or _TRUNCATED.search(text):
        return None
    lines = text.splitlines()
    numbered = [_NUMBERED.match(line) for line in lines]
    if lines and all(numbered):
        indexes = [int(match.group(1)) for match in numbered if match]
        if indexes != list(range(1, indexes[-1] + 1)):
            return None
        return "\n".join(match.group(2) for match in numbered if match)
    return None


def _listing_result(raw: str, exclusions: tuple[str, ...]) -> str:
    text = _unwrap(raw)
    lines = text.splitlines()
    protected = re.compile(r"(?:^|[/\"' ])\.(?:git|codepartner)(?:[/\"' ,]|$)|"
                           r"(?:^|[/\"' ])\.env(?!\.example)")
    safe = [line for line in lines if not protected.search(line)
            and not any(item in line for item in exclusions)]
    # A JSON tree can put every entry on one line. Hide that line if it names
    # a protected resource rather than forwarding its surrounding content.
    visible = "\n".join(safe[:50])
    visible, clipped = _clip(visible)
    more = clipped or len(safe) > 50
    return (f"LIST COMPLETE\n{len(safe[:50])} entries returned\n"
            f"truncated={'true' if more else 'false'}\n"
            + (visible or "No visible entries.")
            + (f"\n{len(lines) - len(safe)} protected entry line(s) omitted"
               if len(lines) != len(safe) else ""))


def _safe_generic_text(raw: str, exclusions: tuple[str, ...]) -> str:
    text = _unwrap(raw)
    protected = re.compile(r"(?:^|[/\"' ])\.(?:git|codepartner)(?:[/\"' ,]|$)|"
                           r"(?:^|[/\"' ])\.env(?!\.example)")
    return "\n".join(line for line in text.splitlines()
                     if not protected.search(line)
                     and not any(item and item in line for item in exclusions))


class AgentToolInterface:
    def __init__(self, run, *, compact: bool = True):
        self.run = run
        self.compact = compact
        self.observations: OrderedDict[str, str] = OrderedDict()
        self.prior: dict[tuple, tuple[str, str]] = {}
        self.prior_search: dict[tuple, str] = {}
        self.counter = 0
        self.metrics: Counter[str] = Counter()
        self.files_read: set[str] = set()
        self.observed: list[str] = []
        self.started_at = time.monotonic()

    def summary(self) -> dict:
        return {**dict(self.metrics), "unique_files_read": sorted(self.files_read),
                "elapsed_ms": int((time.monotonic() - self.started_at) * 1000),
                "observation_mode": "compact" if self.compact else "baseline"}

    def _remember(self, result: str, raw: str, *, category: str) -> str:
        self.counter += 1
        key = f"obs-{self.counter}"
        self.metrics[f"{category}_calls"] += 1
        if raw and raw != result:
            if len(raw.encode()) > MAX_SAVED_BYTES:
                suffix = "\nFull output unavailable: it exceeded the observation retention limit."
                return result + suffix
            self.observations[key] = raw
            while len(self.observations) > MAX_SAVED or sum(map(len, self.observations.values())) > MAX_SAVED_BYTES:
                self.observations.popitem(last=False)
            suffix = f"\nFull authorized output: codepartner_get_observation(observation_id={key!r})"
            return result + suffix
        return result

    def _reload(self, arguments: dict) -> str:
        key = arguments.get("observation_id")
        if key == "index":
            listing = "\n".join(self.observed[-50:]) or "No observations recorded yet."
            shown, truncated = _clip(listing)
            return (f"OBSERVATION INDEX\n{len(self.observed)} entries recorded\n"
                    f"truncated={'true' if truncated or len(self.observed) > 50 else 'false'}\n"
                    + shown)
        raw = self.observations.get(key)
        if raw is None:
            return "OBSERVATION UNAVAILABLE\nThis observation expired or belongs to another execution."
        try:
            offset = int(arguments.get("offset", 0))
        except (TypeError, ValueError):
            return "OBSERVATION ERROR\noffset must be a nonnegative integer."
        if offset < 0:
            return "OBSERVATION ERROR\noffset must be a nonnegative integer."
        data = raw.encode()
        segment = data[offset:offset + MAX_BYTES].decode("utf-8", errors="replace")
        next_offset = min(offset + MAX_BYTES, len(data))
        return (f"OBSERVATION COMPLETE\nid={key}\nbytes={offset}–{next_offset} of {len(data)}\n"
                f"truncated={'true' if next_offset < len(data) else 'false'}\n" + segment
                + (f"\nContinue with offset={next_offset}" if next_offset < len(data) else ""))

    async def _validate_mutation(self, paths: list[str],
                                 backend: Callable[[str, dict], Awaitable[str]]) -> str:
        source_paths = [path for path in dict.fromkeys(paths)
                        if validator_for(path) is not None]
        if not source_paths:
            return "Validation:\nNo configured parser for edited files; check skipped"
        candidate = next((spec for spec in self.run.registry.tools()
                          if _bare(spec.name) == "read_file"), None)
        if candidate is None:
            return "Validation:\nIDE source reader unavailable; check skipped"
        postimages = {}
        unavailable = []
        for path in source_paths:
            try:
                raw = await backend(candidate.name, {"file_path": path, "offset": 1, "limit": 5000})
            except Exception:
                unavailable.append(path)
                continue
            content = _complete_source(raw)
            if content is None:
                unavailable.append(path)
            else:
                postimages[path] = content
        result = validate_postimages(postimages) if postimages else "Validation:"
        if unavailable:
            result += "\n" + "\n".join(
                f"{path}: complete postimage unavailable; parser check skipped"
                for path in unavailable)
        return result

    async def _inspect_mutation(self, paths: list[str],
                                backend: Callable[[str, dict], Awaitable[str]]) -> str:
        candidate = next((spec for spec in self.run.registry.tools()
                          if _bare(spec.name) == "get_file_problems"), None)
        python_paths = [path for path in dict.fromkeys(paths) if path.endswith(".py")]
        if not python_paths:
            return ""
        if candidate is None:
            return "IDE inspection: unavailable; no lint result claimed"
        messages = []
        for path in python_paths:
            try:
                raw = await backend(candidate.name, {"filePath": path, "errorsOnly": True,
                                                     "timeout": 5000})
                payload = json.loads(_unwrap(raw))
                problems = payload.get("errors") if isinstance(payload, dict) else None
                if not isinstance(problems, list) or payload.get("timedOut"):
                    messages.append(f"{path}: IDE inspection incomplete")
                elif not problems:
                    messages.append(f"{path}: no IDE-reported errors")
                else:
                    shown = "; ".join(
                        f"line {item.get('line', '?')}: {str(item.get('description', 'error'))[:120]}"
                        for item in problems[:3] if isinstance(item, dict))
                    messages.append(f"{path}: {len(problems)} IDE-reported error(s); {shown}")
            except Exception:
                messages.append(f"{path}: IDE inspection unavailable")
        return "IDE inspection:\n" + "\n".join(messages)

    async def execute(self, name: str, arguments: dict,
                      backend: Callable[[str, dict], Awaitable[str]]) -> str:
        bare = _bare(name)
        effective, inner = _dispatcher(arguments) if bare == "execute_tool" else (bare, arguments)
        self.metrics["agent_calls"] += 1
        if not effective:
            return "TOOL ERROR\nInvalid or empty tool command."
        if bare == "execute_tool" and effective not in _DISPATCHER_ALLOWED:
            return "ACCESS DENIED\nNested IDE tool is not supported by the agent interface."
        if effective == "codepartner_get_observation":
            return self._reload(arguments)
        if bare == "codepartner_search_in_file":
            path, query = arguments.get("path"), arguments.get("query")
            if not isinstance(query, str) or not query:
                return "SEARCH ERROR\nquery must be nonempty."
            reason = _path_reason(path, exclusions=tuple(self.run.metadata.get("excluded_files", ())))
            if reason:
                return f"ACCESS DENIED\n{reason}. Do not retry this path."
            candidate = next((s for s in self.run.registry.tools() if _bare(s.name) in
                              {"search_text", "find_text", "exact_search", "search_in_files_by_text"}), None)
            if candidate is None:
                return "SEARCH UNAVAILABLE\nThe IDE advertised no literal content search."
            search_args = {"q": query, "limit": SEARCH_MATCHES + 1, "paths": [path]}
            props = candidate.schema.get("properties", {})
            if props and "paths" not in props:
                return "SEARCH UNAVAILABLE\nThe IDE search tool cannot restrict results to one file."
            if props and "limit" not in props:
                search_args.pop("limit")
            raw = await backend(candidate.name, search_args)
            result, _ = _search_result(raw, query, tuple(self.run.metadata.get("excluded_files", ())))
            self.observed.append(f"search {path}: {query!r}")
            safe_raw = _safe_search_raw(raw, tuple(self.run.metadata.get("excluded_files", ())))
            return self._remember(result, safe_raw, category="search")
        paths = list(_paths(inner))
        if bare == "codepartner_read_file" and isinstance(arguments.get("path"), str):
            paths = [arguments["path"]]
        mutating = is_mutating_tool(name) or is_mutating_tool(effective)
        if not mutating and not self.metrics["edits"]:
            self.metrics["calls_before_first_edit"] += 1
        if mutating:
            if bare == "execute_tool":
                adapter = ({"apply_patch": "patch", "create_new_file": "create",
                            "create_new_file_with_text": "create", "rename_refactoring": "rename",
                            "replace_file_text_by_path": "replace",
                            "replace_specific_text": "replace", "replace_text_in_file": "replace",
                            "delete_file": "delete", "delete_file_by_path": "delete",
                            "reformat_file": "format", "edit_notebook": "notebook_edit"}
                           .get(effective))
                if adapter is None:
                    return "ACCESS DENIED\nDispatcher mutation has no journal adapter."
                targets = mutation_targets(adapter, inner) if adapter else ()
            else:
                try:
                    targets = self.run.registry.mutation_targets(name, arguments)
                except LookupError:
                    targets = ()
            paths = list(dict.fromkeys([*paths, *targets]))
            if effective == "apply_patch":
                patch = next((inner.get(key) for key in ("input", "patch", "patch_text", "text")
                              if isinstance(inner.get(key), str)), "")
                for header in re.findall(
                    r"(?m)^(?:\*\*\* (?:Add|Update|Delete) File:|\*\*\* Move to:|--- |\+\+\+ |rename from |rename to |copy from |copy to )(.+)$",
                    patch):
                    path = header.strip().split("\t", 1)[0]
                    if path == "/dev/null":
                        continue
                    if path.startswith(("a/", "b/")):
                        path = path[2:]
                    paths.append(path)
                for line in patch.splitlines():
                    if line.startswith("diff --git "):
                        try:
                            tokens = shlex.split(line)
                        except ValueError:
                            return "ACCESS DENIED\nInvalid diff header."
                        if len(tokens) < 4:
                            return "ACCESS DENIED\nInvalid diff header."
                        paths.extend(token[2:] if token.startswith(("a/", "b/")) else token
                                     for token in tokens[2:4])
            if not paths:
                return "ACCESS DENIED\nMutation targets could not be identified."
            if (self.run.metadata.get("task_execution")
                    and name not in self.run.direct_tool_handlers):
                return ("ACCESS DENIED\nThis IDE write is unavailable through the "
                        "mutation journal; use an advertised direct IDE editor.")
        if bare == "execute_tool" and "terminal" in effective.lower():
            command = inner.get("command", "")
            reason = _check_command(command) if isinstance(command, str) else "missing command"
            if reason:
                return f"ACCESS DENIED\n{reason}."
        if mutating and self.run.metadata.get("read_only"):
            return "ACCESS DENIED\nThis execution is read-only."
        for path in paths:
            reason = _path_reason(path, write=mutating,
                                  exclusions=tuple(self.run.metadata.get("excluded_files", ())))
            if reason:
                return f"ACCESS DENIED\n{path}: {reason}. Do not retry this path."
        if effective == "get_git_diff_all":
            return "ACCESS DENIED\nFull project diffs may include protected files; inspect specific source files instead."
        if not self.compact and effective in _READ:
            raw = await backend(name, arguments)
            path = inner.get("file_path") or inner.get("path")
            if isinstance(path, str):
                self.files_read.add(path)
            self.metrics["lines_returned"] += sum(
                bool(_NUMBERED.match(line)) for line in _unwrap(raw).splitlines())
            return self._remember(raw or "READ COMPLETE\nNo output was produced.", raw,
                                  category="read")
        if effective in _READ:
            try:
                if bare == "codepartner_read_file":
                    path = arguments["path"]
                    start = int(arguments.get("start_line", 1))
                    end = int(arguments.get("end_line", start + WINDOW - 1))
                else:
                    path = inner.get("file_path") or inner.get("path")
                    start = int(inner.get("offset", 1))
                    end = start + min(int(inner.get("limit", WINDOW)), MAX_WINDOW) - 1
            except (KeyError, TypeError, ValueError):
                return "READ ERROR\nInvalid file path or line range."
            if not isinstance(path, str) or start < 1 or end < start or end - start + 1 > MAX_WINDOW:
                return f"READ ERROR\nRequest a range of 1–{MAX_WINDOW} lines."
            reason = _path_reason(path, exclusions=tuple(self.run.metadata.get("excluded_files", ())))
            if reason:
                return f"ACCESS DENIED\n{path}: {reason}. Do not retry this path."
            count = end - start + 1
            if bare == "codepartner_read_file":
                candidate = next((s for s in self.run.registry.tools() if _bare(s.name) == "read_file"), None)
                if candidate is None:
                    return "READ UNAVAILABLE\nThe IDE advertised no window-capable file reader."
                name = candidate.name
            if effective == "get_file_text_by_path":
                candidate = next((s for s in self.run.registry.tools() if _bare(s.name) == "read_file"), None)
                if candidate is not None:
                    raw = await backend(candidate.name, {"file_path": path, "offset": start,
                                                         "limit": min(5000, count * 2 + 2)})
                else:
                    raw = await backend(name, inner)
            elif bare == "execute_tool":
                command = " ".join(("read_file", "--file_path", shlex.quote(path),
                                    "--offset", str(start), "--limit", str(min(5000, count * 2 + 2))))
                raw = await backend(name, {"command": command})
            else:
                raw = await backend(name, {"file_path": path, "offset": start,
                                           "limit": min(5000, count * 2 + 2)})
            result, returned, body = _read_result(raw, path, start, count)
            self.files_read.add(path)
            if result.startswith("READ COMPLETE"):
                self.observed.append(f"read {path} lines {start}–{start + returned - 1}")
            signature = ("read", path, start, end)
            if result.startswith("READ COMPLETE"):
                previous = self.prior.get(signature)
                if previous and previous[1] == result:
                    self.metrics["duplicate_calls"] += 1
                    return self._remember(
                        f"DUPLICATE READ\n{path} lines {start}–{start + returned - 1} "
                        "were already returned and the IDE returned the same content again. "
                        "Use the previous observation or request a different range.\ntruncated=false",
                        raw, category="read")
                self.prior[signature] = (body, result)
                self.metrics["lines_returned"] += returned
            return self._remember(result, raw, category="read")
        dispatch_arguments = arguments
        if effective in _SEARCH:
            if bare == "execute_tool":
                try:
                    tokens = shlex.split(arguments["command"])
                    bounded = []
                    index = 0
                    while index < len(tokens):
                        if tokens[index] == "--limit" and index + 1 < len(tokens):
                            index += 2
                            continue
                        bounded.append(tokens[index])
                        index += 1
                    bounded.extend(("--limit", str(SEARCH_MATCHES + 1)))
                    dispatch_arguments = {**arguments, "command": shlex.join(bounded)}
                except (KeyError, ValueError):
                    return "SEARCH ERROR\nInvalid dispatcher arguments."
            else:
                spec = next((item for item in self.run.registry.tools() if item.name == name), None)
                props = spec.schema.get("properties", {}) if spec and isinstance(spec.schema, dict) else {}
                if not props or "limit" in props:
                    dispatch_arguments = {**arguments, "limit": SEARCH_MATCHES + 1}
        raw = await backend(name, dispatch_arguments)
        if mutating:
            self.prior.clear()
            self.prior_search.clear()
            status = ("EDIT FAILED OR UNCERTAIN" if tool_failed(raw)
                      else "EDIT OUTCOME UNCERTAIN" if not raw.strip() and not getattr(raw, "validation", None)
                      else "EDIT APPLIED")
            if status == "EDIT APPLIED":
                self.observed.append("edit " + ", ".join(paths))
                self.metrics["edits"] += 1
                if self.metrics["edits"] == 1:
                    self.metrics["time_to_first_edit_ms"] = int((time.monotonic() - self.started_at) * 1000)
            clipped, trunc = _clip(_unwrap(raw))
            resources, resources_clipped = _clip(", ".join(paths) or "unknown", 800)
            result = (f"{status}\nresource={resources}\n"
                      f"truncated={'true' if trunc or resources_clipped else 'false'}\n"
                      + (clipped or "No stdout was produced."))
            validation = getattr(raw, "validation", None)
            if not validation and status != "EDIT FAILED OR UNCERTAIN":
                validation = await self._validate_mutation(paths, backend)
            if validation:
                if "VALIDATION FAILED" in validation:
                    result = result.replace("EDIT APPLIED", "EDIT APPLIED BUT VALIDATION FAILED", 1)
                result += "\n" + validation
            if status != "EDIT FAILED OR UNCERTAIN" and "VALIDATION FAILED" not in (validation or ""):
                inspection = await self._inspect_mutation(paths, backend)
                if inspection:
                    result += "\n" + inspection
            return self._remember(result, raw, category="edit")
        if effective in _SEARCH:
            result, _ = _search_result(raw, str(inner.get("q", "")),
                                       tuple(self.run.metadata.get("excluded_files", ())))
            self.observed.append(f"search {inner.get('q', '')!r}")
            signature = (effective, json.dumps(inner, sort_keys=True, default=str))
            if self.prior_search.get(signature) == result and result.startswith("SEARCH COMPLETE"):
                self.metrics["duplicate_calls"] += 1
                return self._remember(
                    "DUPLICATE SEARCH\nThe IDE returned the same matches again. "
                    "Refine the query or use the earlier locations.\ntruncated=false",
                    "", category="search")
            self.prior_search[signature] = result
            safe_raw = _safe_search_raw(raw, tuple(self.run.metadata.get("excluded_files", ())))
            return self._remember(result, safe_raw, category="search")
        if effective in {"list_files_in_folder", "list_directory_tree"}:
            result = _listing_result(raw, tuple(self.run.metadata.get("excluded_files", ())))
            return self._remember(result, result, category="listing")
        if ("terminal" in effective.lower() or "run_configuration" in effective.lower()
                or effective == "build_project"):
            self.metrics["test_executions"] += 1
            safe_raw = _redact_command_output(raw)
            return self._remember(_command_result(safe_raw), safe_raw, category="command")
        text = _safe_generic_text(raw, tuple(self.run.metadata.get("excluded_files", ())))
        shown, truncated = _clip(text)
        result = (f"TOOL {'FAILED' if text.lstrip().lower().startswith('error') else 'COMPLETE'}\n"
                  f"tool={effective}\ntruncated={'true' if truncated else 'false'}\n"
                  + (shown or "No output was produced."))
        return self._remember(result, text, category="other")

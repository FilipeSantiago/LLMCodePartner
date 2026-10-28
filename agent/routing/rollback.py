"""Task-scoped mutation journal for review and execution evidence.

IDE mutations are journaled per tool call. Native (CLI filesystem) execution is
journaled from a git baseline taken before the run and compared after it. The
journal records which files changed and how; it keeps no preimages and restores
nothing.
"""

import asyncio
import hashlib
import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

from agent.routing.persistence import RoutingPersistence
from agent.routing.evidence import (is_verification_call, normalize_dispatcher_call,
                                    store_execution_payload, verification_succeeded)
from mcp_bridge.edit_validation import ValidatedResult, validate_postimages
from agent.spec.bridge_client import BridgeMcpFileClient
from mcp_bridge.adapters import mutation_targets
from mcp_bridge.registry import (
    DOMAIN_SOURCE,
    OP_FILE_READ,
    ToolRegistry,
    ToolSpec,
    is_mutating_tool,
)

_MISSING = re.compile(r"(?:doesn't exist|not found|cannot find|can't be opened)", re.I)
_TRUNCATED = re.compile(r"…\d+ lines truncated…")
_MISSING_HASH = "__missing__"
_DECLARED_ARTIFACT = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\."
    r"(?:md|txt|ipynb|py)(?![A-Za-z0-9_.-])", re.IGNORECASE)
_COMMUNICATION_ARTIFACT = re.compile(
    r"(?:summary|completion|report|evidence|verification|test[_-]?mapping|"
    r"test[_-]?results?|^run_.*tests?)", re.IGNORECASE)
_TEMPORARY_CODE_ARTIFACT = re.compile(
    r"^(?:(?:verify|validate|check|diagnose|debug|quick|manual|tmp|scratch)[_-].*|"
    r"run[_-].*tests?.*|test[_-].*(?:runner|execution|validation|verification|syntax).*|"
    r".*[_-](?:runner|validation|verification))\.(?:py|ipynb)$",
    re.IGNORECASE,
)
_TEMPORARY_CODE_CONTENT = re.compile(
    r"(?:pytest\.main|subprocess\.(?:run|Popen)|py_compile|ast\.parse|"
    r"compileall|--collect-only|python\s+-m\s+pytest)",
    re.IGNORECASE,
)


def declared_artifacts(text: str) -> frozenset[str]:
    return frozenset(match.group() for match in _DECLARED_ARTIFACT.finditer(text or ""))


def _mutation_text(arguments: dict) -> str:
    if not isinstance(arguments, dict):
        return ""
    return next((value for key in (
        "text", "content", "newText", "new_source", "input", "patch", "patch_text",
    ) if isinstance((value := arguments.get(key)), str)), "")


def _communication_artifact(path: str, content: str = "") -> bool:
    name = path.rsplit("/", 1)[-1]
    lower = name.lower()
    if lower.endswith((".py", ".ipynb")):
        if _TEMPORARY_CODE_ARTIFACT.match(lower):
            return True
        # A new root-level script whose body is predominantly an execution wrapper is
        # temporary evidence even when its filename is deliberately generic.
        return "/" not in path and bool(_TEMPORARY_CODE_CONTENT.search(content))
    return lower.endswith((".md", ".txt")) and bool(
        _COMMUNICATION_ARTIFACT.search(name))


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _failed(result: object) -> bool:
    return not isinstance(result, str) or result.lstrip().lower().startswith("error")


def _normalize_read(result: str) -> str:
    text, _ = BridgeMcpFileClient._text_from_tool_response(result)
    text, _, _ = BridgeMcpFileClient._strip_numbered_lines(text)
    return text


_DISPATCHER_MUTATIONS = {
    "apply_patch": "patch", "create_new_file": "create",
    "create_new_file_with_text": "create", "replace_file_text_by_path": "replace",
    "replace_specific_text": "replace", "replace_text_in_file": "replace",
    "delete_file": "delete", "delete_file_by_path": "delete",
    "rename_refactoring": "rename", "reformat_file": "format",
    "edit_notebook": "notebook_edit",
}


def _nested_mutation(name: str, arguments: dict) -> tuple[str | None, dict]:
    if name.rsplit("__", 1)[-1] != "execute_tool":
        return None, {}
    command = arguments.get("command")
    if not isinstance(command, str):
        return None, {}
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None, {}
    if not tokens or tokens[0] not in _DISPATCHER_MUTATIONS:
        return None, {}
    parsed = {}
    for index, token in enumerate(tokens[:-1]):
        if token.startswith("--"):
            parsed[token[2:]] = tokens[index + 1]
    return _DISPATCHER_MUTATIONS[tokens[0]], parsed


# --- native git baseline -------------------------------------------------------
#
# Captures the pre-task state for change detection without writing anything to the
# repository: file oids come from `hash-object --no-filters` (no `-w`), file modes
# from `lstat`, and staged entries from `ls-files -s`. Only paths the run changed are
# journaled; nothing is restored.

_GIT_TIMEOUT = 300


def _git(root: str, *args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "-C", root, *args], input=data,
                            capture_output=True, timeout=_GIT_TIMEOUT)
    if result.returncode:
        raise RuntimeError(f"git {args[0]} failed: "
                           f"{result.stderr.decode(errors='replace').strip()[:300]}")
    return result.stdout


def _nul_paths(output: bytes) -> list[str]:
    return [os.fsdecode(item) for item in output.split(b"\0") if item]


def _index_entries(root: str) -> dict[str, list[list[str]]]:
    """path -> [[mode, oid, stage], ...] (several stages while a merge conflict is open)."""
    entries: dict[str, list[list[str]]] = {}
    for record in _git(root, "ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        entries.setdefault(os.fsdecode(path), []).append(meta.decode().split())
    return entries


def _candidates(root: str) -> set[str]:
    """Tracked and untracked, non-ignored paths (ignored files are not journaled)."""
    return set(_nul_paths(_git(root, "ls-files", "-z", "-c", "-o", "--exclude-standard")))


def _lstat(root: str, path: str):
    try:
        info = os.lstat(os.path.join(root, path))
    except FileNotFoundError:
        return None
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        return None  # directories (submodules) and special files are not journaled
    return info


def _signature(info) -> list[int] | None:
    return None if info is None else [info.st_mode, info.st_size, info.st_mtime_ns,
                                      info.st_ctime_ns, info.st_ino]


def _file_bytes(root: str, path: str, info) -> bytes:
    full = os.path.join(root, path)
    if stat.S_ISLNK(info.st_mode):
        return os.fsencode(os.readlink(full))
    with open(full, "rb") as handle:
        return handle.read()


def _oid(root: str, data: bytes) -> str:
    return _git(root, "hash-object", "--no-filters", "--stdin", data=data).decode().strip()


# Preimages above this size are not kept; the reviewer then sees the whole file.
_MAX_PREIMAGE_CHARS = 512_000


def _preimage(content: str | None) -> str | None:
    return content if content is not None and len(content) <= _MAX_PREIMAGE_CHARS else None


def _read_preimages(root: str, paths: tuple[str, ...]) -> dict[str, str]:
    """The current text of each existing regular file in `paths`."""
    preimages = {}
    for path in dict.fromkeys(paths):
        info = _lstat(root, path)
        if info is None or stat.S_ISLNK(info.st_mode) or info.st_size > _MAX_PREIMAGE_CHARS:
            continue
        try:
            preimages[path] = _file_bytes(root, path, info).decode()
        except (OSError, UnicodeDecodeError):
            continue
    return preimages


def _baseline_preimage(baseline: dict, path: str) -> str | None:
    """A changed file's content at the baseline: captured up front for planned files,
    otherwise its baseline blob when git already stores it (clean tracked files)."""
    if path in baseline.get("preimages", {}):
        return baseline["preimages"][path]
    oid = (baseline["files"].get(path) or {}).get("oid")
    if not oid:
        return None
    try:
        data = _git(baseline["root"], "cat-file", "blob", oid)
        return _preimage(data.decode())
    except Exception:
        return None


def _capture_baseline(root: str) -> dict:
    head = subprocess.run(["git", "-C", root, "rev-parse", "--verify", "-q", "HEAD"],
                          capture_output=True, text=True, timeout=_GIT_TIMEOUT).stdout.strip()
    index = _index_entries(root)
    files: dict[str, dict] = {}
    regular: list[str] = []
    for path in sorted(_candidates(root) | set(index)):
        info = _lstat(root, path)
        if info is None:
            continue
        files[path] = {"stat": _signature(info), "perm": stat.S_IMODE(info.st_mode),
                       "link": stat.S_ISLNK(info.st_mode)}
        if files[path]["link"]:
            files[path]["oid"] = _oid(root, _file_bytes(root, path, info))
        else:
            regular.append(path)
    if regular:
        oids = _git(root, "hash-object", "--no-filters", "--stdin-paths",
                    data=os.fsencode("\n".join(regular) + "\n")).decode().split()
        for path, oid in zip(regular, oids):
            files[path]["oid"] = oid
    return {"root": root, "head": head or None, "index": index, "files": files}


def _detect_changes(baseline: dict) -> tuple[list[dict], str | None]:
    """Paths whose content, mode, existence or index entry differ from the baseline."""
    root = baseline["root"]
    index = _index_entries(root)
    before_files = baseline["files"]
    changes = []
    for path in sorted(_candidates(root) | set(index) | set(before_files)
                       | set(baseline["index"])):
        before = before_files.get(path)
        info = _lstat(root, path)
        content_changed = False
        if (before is None) != (info is None):
            content_changed = True
        elif info is not None and _signature(info) != before["stat"]:
            content_changed = (_oid(root, _file_bytes(root, path, info)) != before["oid"]
                               or stat.S_IMODE(info.st_mode) != before["perm"]
                               or stat.S_ISLNK(info.st_mode) != before["link"])
        index_changed = index.get(path) != baseline["index"].get(path)
        if not (content_changed or index_changed):
            continue
        current_hash = (_MISSING_HASH if info is None
                        else hashlib.sha256(_file_bytes(root, path, info)).hexdigest())
        changes.append({"path": path, "before_exists": before is not None,
                        "expected_hash": current_hash})
    head = subprocess.run(["git", "-C", root, "rev-parse", "--verify", "-q", "HEAD"],
                          capture_output=True, text=True, timeout=_GIT_TIMEOUT).stdout.strip()
    moved = None if (head or None) == baseline["head"] else (
        f"HEAD moved from {baseline['head'] or 'none'} to {head or 'none'} during the task")
    return changes, moved


class RollbackJournal:
    def __init__(self, persistence: RoutingPersistence, job_id: str, position: int,
                 specs: list[ToolSpec], handlers: dict,
                 allowed_artifacts: frozenset[str] = frozenset()):
        self.persistence = persistence
        self.job_id = job_id
        self.position = position
        self.registry = ToolRegistry()
        self.registry.register(specs)
        self.handlers = handlers
        self.allowed_artifacts = allowed_artifacts
        self._last_postimages: dict[str, str | None] = {}
        self.read_spec = next((spec for spec in self.registry.candidates(
            OP_FILE_READ, DOMAIN_SOURCE) if spec.name in handlers), None)

    async def _read(self, path: str) -> tuple[bool, str | None, str | None]:
        if self.read_spec is None:
            return False, None, "no IDE source reader"
        try:
            supplied = {"file_path": path, "limit": 5000}
            arguments = self.registry.prepare_advertised_call(
                self.read_spec.name, supplied)[1]
            result = await self.handlers[self.read_spec.name](arguments)
        except Exception as exc:
            detail = str(exc)
            return ((False, None, None) if _MISSING.search(detail)
                    else (False, None, f"{type(exc).__name__}: {detail}"))
        if not isinstance(result, str):
            return False, None, "IDE reader returned non-text content"
        if _MISSING.search(result):
            return False, None, None
        if _failed(result):
            return False, None, result[:200]
        content = _normalize_read(result)
        if _TRUNCATED.search(content):
            return False, None, "IDE reader truncated the file"
        return True, content, None

    async def before_mutation(self, name: str, arguments: dict) \
            -> tuple[int, list[str], list[tuple[str, dict]]]:
        effective, nested = _nested_mutation(name, arguments)
        paths = (list(mutation_targets(effective, nested)) if effective
                 else list(self.registry.mutation_targets(name, arguments)))
        mutation_id = await self.persistence.plan_mutation(
            self.job_id, self.position, name,
            effective or self.registry.mutation_operation(name), paths)
        if not paths:
            await self.persistence.advance_mutation(
                mutation_id, "planned", "target_resolution",
                "no canonical project-relative mutation targets")
            return mutation_id, [], [("rollback_untracked_mutation", {"tool": name})]
        tracked = []
        issues: list[tuple[str, dict]] = []
        pending: list[tuple[str, bool, str | None]] = []
        mutation_text = _mutation_text(arguments)
        for path in paths:
            existing_entry = await self.persistence.rollback_file(
                self.job_id, self.position, path)
            if existing_entry:
                if (not existing_entry["before_exists"]
                        and _communication_artifact(path, mutation_text)
                        and path not in self.allowed_artifacts):
                    issues.append(("project_artifact_policy", {
                        "tool": name,
                        "path": path,
                        "error": ("task-created temporary validation, runner, or communication "
                                  "artifact is quarantined; modify the requested deliverable and "
                                  "use captured execution evidence instead"),
                    }))
                    continue
                tracked.append(path)
                continue
            # The preimage is kept only for the reviewer's diff; nothing is restored from
            # it. An unreadable file is assumed to exist rather than blocking the write.
            exists, before, error = await self._read(path)
            if error:
                exists, before = True, None
            if (not exists and _communication_artifact(path, mutation_text)
                    and path not in self.allowed_artifacts):
                issues.append(("project_artifact_policy", {
                    "tool": name,
                    "path": path,
                    "error": ("temporary validation, runner, or communication artifacts "
                              "belong in the CodePartner evidence workspace, not the target "
                              "repository; run the check through the execution capability"),
                }))
                continue
            pending.append((path, exists, _preimage(before)))
        if issues:
            await self.persistence.advance_mutation(
                mutation_id, "planned", "snapshot",
                "; ".join(str(payload.get("error") or payload.get("path") or action)
                          for action, payload in issues))
            return mutation_id, [], issues
        for path, exists, before in pending:
            await self.persistence.record_rollback_file(
                self.job_id, self.position, path, exists, before_content=before)
            tracked.append(path)
        await self.persistence.advance_mutation(mutation_id, "snapshotted")
        return mutation_id, tracked, issues

    async def after_mutation(self, mutation_id: int, paths: list[str],
                             issues: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        await self.persistence.advance_mutation(mutation_id, "applied")
        self._last_postimages = {}
        postimage_issues = list(issues)
        for path in paths:
            exists, content, error = await self._read(path)
            if error:
                await self.persistence.record_resolution_action(
                    self.job_id, self.position, "rollback_postimage_failed",
                    {"path": path, "error": error})
                postimage_issues.append(("rollback_postimage_failed", {
                    "path": path, "error": error,
                }))
                continue
            await self.persistence.update_rollback_hash(
                self.job_id, self.position, path,
                _hash(content) if exists and content is not None else _MISSING_HASH)
            self._last_postimages[path] = content if exists else None
        if postimage_issues:
            await self.persistence.advance_mutation(
                mutation_id, "applied", "postimage",
                "; ".join(str(payload.get("error") or payload.get("path") or action)
                          for action, payload in postimage_issues))
        else:
            await self.persistence.advance_mutation(
                mutation_id, "postimage_verified")
        return postimage_issues

    async def _record_blocked(self, name: str,
                              issues: list[tuple[str, dict]]) -> str:
        details = {
            "tool": name,
            "reasons": [{"kind": action, **payload} for action, payload in issues],
        }
        await self.persistence.record_resolution_action(
            self.job_id, self.position, "rollback_mutation_blocked", details)
        reasons = "; ".join(
            str(payload.get("error") or payload.get("path") or action)
            for action, payload in issues)
        if any(action == "project_artifact_policy" for action, _ in issues):
            return ("Error: mutation blocked before execution by the project artifact policy "
                    f"({reasons}). No files were changed.")
        return ("Error: mutation blocked before execution because its target files could "
                f"not be identified for review ({reasons}). No files were changed.")

    async def record_verification(self, name: str, arguments: dict,
                                  result: object, success: bool,
                                  normalized_from: str | None = None) -> None:
        evidence_id = await self.persistence.reserve_evidence(
            self.job_id, self.position, "verification", name, arguments, success)
        manifest = store_execution_payload(
            self.job_id, self.position, evidence_id, name, arguments, result)
        metadata = dict(manifest["metadata"])
        if normalized_from is not None:
            metadata["normalized_from"] = normalized_from
        await self.persistence.complete_evidence(
            evidence_id, manifest["path"], manifest["sha256"],
            manifest["result_excerpt"], metadata)

    def wrapped_handlers(self) -> dict:
        wrapped = dict(self.handlers)
        for name, handler in self.handlers.items():
            if name.rsplit("__", 1)[-1] == "execute_tool":
                async def execute(arguments, _name=name, _handler=handler):
                    # Coders often send `pytest ...` or a positional terminal command;
                    # the dispatcher rejects both before running anything.
                    arguments, normalized_from = normalize_dispatcher_call(arguments)
                    effective, _ = _nested_mutation(_name, arguments)
                    if effective:
                        existing = await self.integrity_issues()
                        if existing:
                            return "Error: mutation blocked because journal recovery is required."
                        mutation_id, paths, issues = await self.before_mutation(_name, arguments)
                        if issues:
                            return await self._record_blocked(_name, issues)
                        try:
                            result = await _handler(arguments)
                        except Exception:
                            await self.after_mutation(mutation_id, paths, issues)
                            raise
                        postimage_issues = await self.after_mutation(mutation_id, paths, issues)
                        if postimage_issues:
                            return "Error: mutation postimage could not be verified; journal recovery is required."
                        return ValidatedResult(str(result), validate_postimages(self._last_postimages))
                    if not is_verification_call(arguments):
                        return await _handler(arguments)
                    try:
                        result = await _handler(arguments)
                    except Exception as exc:
                        await self.record_verification(
                            _name, arguments, f"{type(exc).__name__}: {exc}", False,
                            normalized_from)
                        raise
                    await self.record_verification(
                        _name, arguments, result, verification_succeeded(result),
                        normalized_from)
                    return result

                wrapped[name] = execute
                continue
            if not is_mutating_tool(name):
                continue

            async def call(arguments, _name=name, _handler=handler):
                existing = await self.integrity_issues()
                if existing:
                    return ("Error: mutation blocked because this task requires journal "
                            "recovery before additional files may be changed.")
                mutation_id, paths, issues = await self.before_mutation(_name, arguments)
                if issues:
                    return await self._record_blocked(_name, issues)
                try:
                    result = await _handler(arguments)
                except Exception as exc:
                    postimage_issues = await self.after_mutation(
                        mutation_id, paths, issues)
                    if postimage_issues:
                        return ("Error: the mutation may have been applied, but its post-image "
                                "could not be verified. Journal recovery is required before review.")
                    await self.persistence.advance_mutation(
                        mutation_id, "postimage_verified", "handler_exception",
                        f"{type(exc).__name__}: {exc}")
                    raise
                # Tool-level failures can still arrive after a partial write. Always
                # capture the observed post-image, regardless of the returned status.
                postimage_issues = await self.after_mutation(
                    mutation_id, paths, issues)
                if postimage_issues:
                    return ("Error: the mutation may have been applied, but its post-image "
                            "could not be verified. Journal recovery is required before review.")
                if _failed(result):
                    await self.persistence.advance_mutation(
                        mutation_id, "postimage_verified", "handler_result",
                        str(result)[:500])
                return ValidatedResult(str(result), validate_postimages(self._last_postimages))

            wrapped[name] = call
        return wrapped

    async def record_git_baseline(self, root: str, preimage_paths: tuple[str, ...] = ()) -> None:
        """Snapshot a native run's work tree, index and HEAD before it starts, keeping
        the content of `preimage_paths` (the files the plan edits) for the review diff."""
        self._native_baseline = await asyncio.to_thread(_capture_baseline, root)
        self._native_baseline["preimages"] = await asyncio.to_thread(
            _read_preimages, root, preimage_paths)

    async def record_native_changes(self, excluded_files: tuple[str, ...] = ()) -> list[str]:
        """Journal every path the native run changed since `record_git_baseline`.

        Native tools cannot be stopped from touching the contract's excluded files:
        such changes are journaled like any other, and the run then fails
        (`NativeSessionError`) naming them. Nothing is restored.
        """
        baseline = getattr(self, "_native_baseline", None)
        if baseline is None:
            return []
        self._native_baseline = None
        changes, moved = await asyncio.to_thread(_detect_changes, baseline)
        for change in changes:
            path = change["path"]
            operation = ("file_create" if not change["before_exists"] else
                         "file_delete" if change["expected_hash"] == _MISSING_HASH else
                         "file_replace")
            mutation_id = await self.persistence.plan_mutation(
                self.job_id, self.position, "native_cli", operation, [path])
            before = (await asyncio.to_thread(_baseline_preimage, baseline, path)
                      if change["before_exists"] else None)
            await self.persistence.record_rollback_file(
                self.job_id, self.position, path, change["before_exists"],
                change["expected_hash"], before)
            await self.persistence.update_rollback_hash(
                self.job_id, self.position, path, change["expected_hash"])
            await self.persistence.advance_mutation(mutation_id, "postimage_verified")
        if moved:
            await self.persistence.record_resolution_action(
                self.job_id, self.position, "rollback_untracked_mutation",
                {"tool": "native_cli", "error": moved})
        paths = [change["path"] for change in changes]
        if excluded_files:
            from agent.task_contracts.policy import path_excluded
            excluded = [path for path in paths if path_excluded(path, excluded_files)]
            if excluded:
                from agent.workspace import NativeSessionError
                raise NativeSessionError(
                    "The attempt changed files the task contract excludes: "
                    + ", ".join(excluded) + ". They were left as changed; review them "
                    "before continuing.")
        return paths

    async def changed_paths(self) -> list[str]:
        entries = await self.persistence.rollback_files(self.job_id, self.position)
        return [entry["path"] for entry in entries if entry["expected_hash"]]

    async def review_manifest(self) -> list[dict]:
        """Bounded metadata for review; rollback preimages never leave persistence."""
        entries = await self.persistence.rollback_files(self.job_id, self.position)
        mutations = await self.persistence.mutations(self.job_id, self.position)
        operations: dict[str, list[str]] = {}
        rename_peers: dict[str, str] = {}
        for mutation in mutations:
            if mutation["stage"] not in {"applied", "postimage_verified"}:
                continue
            for path in mutation["targets"]:
                operations.setdefault(path, []).append(mutation["operation"])
            if mutation["operation"] == "rename" and len(mutation["targets"]) == 2:
                source, destination = mutation["targets"]
                rename_peers[source] = destination
                rename_peers[destination] = source
        manifest = []
        for entry in entries:
            if not entry["expected_hash"]:
                continue
            before = entry["before_exists"]
            after = entry["expected_hash"] != _MISSING_HASH
            change = ("created" if not before and after else
                      "deleted" if before and not after else
                      "modified" if before and after else "absent_after_mutation")
            item = {
                "path": entry["path"], "change": change,
                "before_exists": before, "after_exists": after,
                "operations": list(dict.fromkeys(operations.get(entry["path"], ()))),
                "postimage_verified": True,
            }
            if entry["path"] in rename_peers:
                item["rename_peer"] = rename_peers[entry["path"]]
            manifest.append(item)
        return manifest

    async def integrity_issues(self) -> list[dict]:
        return await self.persistence.rollback_integrity_issues(
            self.job_id, self.position)


def journal_handlers(persistence: RoutingPersistence, job_id: str, position: int,
                     specs: list[ToolSpec], handlers: dict,
                     allowed_artifacts: frozenset[str] = frozenset()) -> dict:
    return RollbackJournal(
        persistence, job_id, position, specs, handlers,
        allowed_artifacts).wrapped_handlers()

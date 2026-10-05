"""Task-scoped IDE mutation journal and conflict-safe rollback."""

import hashlib
import re

from agent.routing.persistence import RoutingPersistence
from agent.routing.evidence import (is_verification_call, normalize_dispatcher_call,
                                    store_execution_payload, verification_succeeded)
from agent.spec.bridge_client import BridgeMcpFileClient
from mcp_bridge.registry import (
    DOMAIN_SOURCE,
    OP_FILE_CREATE,
    OP_FILE_DELETE,
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


def _arguments_for_path(spec: ToolSpec, path: str, content: str | None = None) -> dict:
    properties = spec.schema.get("properties", {}) if isinstance(spec.schema, dict) else {}
    key = next((name for name in ("file_path", "filePath", "pathInProject", "path")
                if not properties or name in properties), "file_path")
    arguments = {key: path}
    if content is not None:
        text_key = next((name for name in ("text", "content", "newText")
                         if not properties or name in properties), "text")
        arguments[text_key] = content
        if not properties or "overwrite" in properties:
            arguments["overwrite"] = True
    return arguments


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
        paths = list(self.registry.mutation_targets(name, arguments))
        mutation_id = await self.persistence.plan_mutation(
            self.job_id, self.position, name,
            self.registry.mutation_operation(name), paths)
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
            exists, content, error = await self._read(path)
            if error:
                issues.append(("rollback_snapshot_failed", {
                    "tool": name, "path": path, "error": error,
                }))
                continue
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
            pending.append((path, exists, content))
        if issues:
            await self.persistence.advance_mutation(
                mutation_id, "planned", "snapshot",
                "; ".join(str(payload.get("error") or payload.get("path") or action)
                          for action, payload in issues))
            return mutation_id, [], issues
        for path, exists, content in pending:
            await self.persistence.record_rollback_file(
                self.job_id, self.position, path, exists, content)
            tracked.append(path)
        await self.persistence.advance_mutation(mutation_id, "snapshotted")
        return mutation_id, tracked, issues

    async def after_mutation(self, mutation_id: int, paths: list[str],
                             issues: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        await self.persistence.advance_mutation(mutation_id, "applied")
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
        return ("Error: mutation blocked before execution because a complete rollback "
                f"snapshot could not be established ({reasons}). No files were changed.")

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
                return result

            wrapped[name] = call
        return wrapped

    async def restore(self) -> tuple[list[str], list[str]] | None:
        issues = await self.persistence.rollback_integrity_issues(
            self.job_id, self.position)
        if issues:
            labels = [str(issue.get("path") or issue.get("tool") or issue["action"])
                      for issue in issues]
            return [], labels
        entries = await self.persistence.rollback_files(self.job_id, self.position)
        if not entries:
            return None
        create = next((spec for spec in self.registry.candidates(
            OP_FILE_CREATE, DOMAIN_SOURCE) if spec.name in self.handlers), None)
        delete = next((spec for spec in self.registry.candidates(
            OP_FILE_DELETE, DOMAIN_SOURCE) if spec.name in self.handlers), None)
        ready: list[tuple[dict, ToolSpec]] = []
        conflicts: list[str] = []
        for entry in entries:
            path = entry["path"]
            exists, content, error = await self._read(path)
            current_hash = (_hash(content) if exists and content is not None
                            else _MISSING_HASH)
            if error or not entry["expected_hash"] or current_hash != entry["expected_hash"]:
                conflicts.append(path)
                continue
            spec = create if entry["before_exists"] else delete
            if spec is None:
                conflicts.append(path)
                continue
            ready.append((entry, spec))
        # Validate the complete set before changing any file. This prevents a
        # known conflict from producing a partial rollback.
        if conflicts:
            return [], conflicts
        restored: list[str] = []
        for entry, spec in ready:
            path = entry["path"]
            supplied = _arguments_for_path(
                spec, path, entry["before_content"] if entry["before_exists"] else None)
            try:
                arguments = self.registry.prepare_advertised_call(spec.name, supplied)[1]
                result = await self.handlers[spec.name](arguments)
            except Exception:
                conflicts.append(path)
                continue
            if _failed(result):
                conflicts.append(path)
                continue
            restored.append(path)
        return restored, conflicts

    async def changed_paths(self) -> list[str]:
        entries = await self.persistence.rollback_files(self.job_id, self.position)
        return [entry["path"] for entry in entries if entry["expected_hash"]]

    async def integrity_issues(self) -> list[dict]:
        return await self.persistence.rollback_integrity_issues(
            self.job_id, self.position)


def journal_handlers(persistence: RoutingPersistence, job_id: str, position: int,
                     specs: list[ToolSpec], handlers: dict,
                     allowed_artifacts: frozenset[str] = frozenset()) -> dict:
    return RollbackJournal(
        persistence, job_id, position, specs, handlers,
        allowed_artifacts).wrapped_handlers()

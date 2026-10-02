"""Detection and orchestration for additive /spec and scoped /update commands."""

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from agent.spec.artifact_store import ArtifactSelectionError, ArtifactStore
from agent.spec.models import render_response
from agent.spec.openspec_adapter import OpenSpecAdapter

_SPEC_COMMAND = re.compile(r"^\s*/spec(?:\s+|$)", re.IGNORECASE)
_UPDATE_COMMAND = re.compile(r"^\s*/update(?:\s+|$)", re.IGNORECASE)
_RUN_COMMAND = re.compile(r"^\s*/run(?:\s+|$)", re.IGNORECASE)
_IMPLEMENT_COMMAND = re.compile(r"^\s*/(?:implement|execute)(?:\s+|$)", re.IGNORECASE)
_TASK_TARGET = re.compile(
    r"\b(?:WP|US)\s*(\d+)\s*-\s*T(?:ASK)?\s*(\d+)\b",
    re.IGNORECASE,
)
_PACKAGE_TARGET = re.compile(r"(?:WP|US)\s*(\d+)", re.IGNORECASE)


class SpecCommandError(ValueError):
    """A planning command is malformed or names unavailable work."""


@dataclass(frozen=True)
class SpecResult:
    change_id: str
    content: str
    path: str
    updated_ids: tuple[str, ...] = ()


def is_spec_command(content: str | None) -> bool:
    return bool(content and _SPEC_COMMAND.match(content))


def is_update_command(content: str | None) -> bool:
    return bool(content and _UPDATE_COMMAND.match(content))


def is_run_command(content: str | None) -> bool:
    return bool(content and _RUN_COMMAND.match(content))


def is_implementation_command(content: str | None) -> bool:
    return bool(content and _IMPLEMENT_COMMAND.match(content))


def extract_spec_request(content: str) -> str:
    match = _SPEC_COMMAND.match(content)
    if not match:
        raise SpecCommandError("message is not a /spec command")
    request = content[match.end():].strip()
    if len(request) >= 2 and request[0] == request[-1] and request[0] in "\"'":
        request = request[1:-1].strip()
    if not request:
        raise SpecCommandError("/spec requires a request")
    return request


def extract_update_request(content: str) -> tuple[list[str], str]:
    command = _UPDATE_COMMAND.match(content)
    if not command:
        raise SpecCommandError("message is not an /update command")
    body = content[command.end():].strip()
    matches = list(_TASK_TARGET.finditer(body))
    if not matches:
        raise SpecCommandError(
            "/update requires at least one task ID such as WP1-T1"
        )
    task_ids = list(dict.fromkeys(
        f"WP{match.group(1)}-T{match.group(2)}" for match in matches
    ))
    if ":" in body:
        instruction = body.split(":", 1)[1].strip()
    else:
        instruction = body[matches[-1].end():].strip(" ,;-")
    if not instruction:
        raise SpecCommandError("/update requires a refinement instruction")
    return task_ids, instruction


def extract_run_task_id(content: str) -> str:
    command = _RUN_COMMAND.match(content)
    if not command:
        raise SpecCommandError("message is not a /run command")
    body = content[command.end():].strip()
    match = _TASK_TARGET.fullmatch(body)
    if not match:
        raise SpecCommandError("/run requires exactly one task ID such as WP1-T1")
    return f"WP{match.group(1)}-T{match.group(2)}"


def extract_implementation_selectors(content: str) -> list[str]:
    """Parse ordered task/work-package selectors for /implement and /execute."""
    command = _IMPLEMENT_COMMAND.match(content)
    if not command:
        raise SpecCommandError("message is not an /implement or /execute command")
    body = content[command.end():].strip()
    if not body:
        raise SpecCommandError("/implement requires task IDs such as WP1-T1 or work packages such as WP1")
    selectors: list[str] = []
    for token in re.split(r"[\s,]+", body):
        task = _TASK_TARGET.fullmatch(token)
        if task:
            selectors.append(f"WP{task.group(1)}-T{task.group(2)}")
            continue
        package = _PACKAGE_TARGET.fullmatch(token)
        if package:
            selectors.append(f"WP{package.group(1)}")
            continue
        raise SpecCommandError(
            f"invalid implementation selector {token!r}; use WP1-T1 or WP1"
        )
    return list(dict.fromkeys(selectors))


def create_change_id(request: str, max_length: int = 48) -> str:
    normalized = unicodedata.normalize("NFKD", request).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    if not slug:
        slug = "change-" + hashlib.sha256(request.encode()).hexdigest()[:10]
    return slug[:max_length].rstrip("-")


class SpecCommandHandler:
    def __init__(self, adapter: OpenSpecAdapter, store: ArtifactStore):
        self._adapter = adapter
        self._store = store
        self._lock = asyncio.Lock()

    async def handle(self, messages: list[dict[str, Any]]) -> SpecResult:
        last = messages[-1] if messages else {}
        if last.get("role") != "user":
            raise SpecCommandError("no active planning command")
        content = last.get("content") or ""
        async with self._lock:
            if is_spec_command(content):
                request = extract_spec_request(content)
                draft = await self._adapter.generate_change(request)
                change = await self._store.create_change(
                    draft, create_change_id(draft.title)
                )
                return SpecResult(
                    change.id,
                    render_response(change),
                    change.path,
                )
            if is_update_command(content):
                task_ids, instruction = extract_update_request(content)
                try:
                    change = await self._store.change_for_tasks(task_ids)
                except ArtifactSelectionError as exc:
                    raise SpecCommandError(str(exc)) from exc
                replacements = await self._adapter.refine_tasks(
                    change, task_ids, instruction
                )
                updated = await self._store.replace_tasks(change, replacements)
                return SpecResult(
                    updated.id,
                    render_response(updated, task_ids),
                    updated.path,
                    tuple(task_ids),
                )
        raise SpecCommandError("no active planning command")

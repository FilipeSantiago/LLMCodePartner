"""Detection and orchestration for additive /spec and scoped /update commands."""

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from agent.spec.artifact_store import ArtifactSelectionError, ArtifactStore
from agent.spec.models import OpenSpecChange, Task, render_response
from agent.spec.openspec_adapter import OpenSpecAdapter

_SPEC_COMMAND = re.compile(r"^\s*/spec(?:\s+|$)", re.IGNORECASE)
_UPDATE_COMMAND = re.compile(r"^\s*/update(?:\s+|$)", re.IGNORECASE)
_RUN_COMMAND = re.compile(r"^\s*/run(?:\s+|$)", re.IGNORECASE)
_IMPLEMENT_COMMAND = re.compile(r"^\s*/(?:implement|execute)(?:\s+|$)", re.IGNORECASE)
_REVIEW_COMMAND = re.compile(r"^\s*/review(?:\s+|$)", re.IGNORECASE)
_REWORK_COMMAND = re.compile(r"^\s*/rework(?:\s+|$)", re.IGNORECASE)
_HELP_COMMAND = re.compile(r"^\s*/(?:help|man)(?:\s+|$)", re.IGNORECASE)
_TASK_TARGET = re.compile(
    r"\b(?:WP|US)\s*(\d+)\s*-\s*T(?:ASK)?\s*(\d+)\b",
    re.IGNORECASE,
)
_PACKAGE_TARGET = re.compile(r"(?:WP|US)\s*(\d+)", re.IGNORECASE)


class SpecCommandError(ValueError):
    """A planning command is malformed or names unavailable work."""


@dataclass(frozen=True)
class CommandInfo:
    name: str
    usage: str
    summary: str
    details: str


COMMANDS = (
    CommandInfo("spec", "/spec <request>", "Create an additive OpenSpec change.",
                "Creates a new change and work packages; it never replaces earlier changes."),
    CommandInfo("update", "/update WP1-T1 [WP1-T2 ...]: <instruction>", "Refine planned tasks.",
                "Updates only selected tasks in one existing change. Use before implementation."),
    CommandInfo("review", "/review WP1|WP1-T1[: <question>]", "Read-only implementation review.",
                "Inspects selected work through read-only IDE tools and reports evidence, gaps, and tests. It never writes files."),
    CommandInfo("rework", "/rework WP1|WP1-T1[: <feedback>]", "Create corrective follow-up work.",
                "Creates a separate additive OpenSpec change linked to the selected source tasks; completed history is preserved."),
    CommandInfo("implement", "/implement WP1|WP1-T1 [more selectors][: <execution instruction>]", "Route and implement tasks sequentially.",
                "Expands work packages into incomplete tasks, routes each task at start time, executes sequentially, and stops on failure. Put optional execution guidance after ':'."),
    CommandInfo("execute", "/execute WP1|WP1-T1 [more selectors][: <execution instruction>]", "Alias for /implement.",
                "Uses exactly the same persisted sequential implementation-job flow as /implement."),
    CommandInfo("run", "/run WP1-T1", "Run one task through the routing pipeline.",
                "Use /implement for new work; /run remains available for a single-task execution."),
    CommandInfo("help", "/help", "List available commands.", "Use /man <command> for detailed syntax and behavior."),
    CommandInfo("man", "/man <command>", "Show command documentation.", "For example: /man implement or /man rework."),
)


def help_text() -> str:
    lines = ["# Code Partner commands", ""]
    lines.extend(f"- `{item.usage}` — {item.summary}" for item in COMMANDS)
    lines.extend(["", "Use `/man <command>` for details."])
    return "\n".join(lines)


def manual_text(command: str) -> str:
    target = command.strip().lower().lstrip("/")
    item = next((item for item in COMMANDS if item.name == target), None)
    if item is None:
        raise SpecCommandError(f"unknown command {command!r}; use /help")
    return f"# /{item.name}\n\n**Usage:** `{item.usage}`\n\n{item.details}"


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


def is_review_command(content: str | None) -> bool:
    return bool(content and _REVIEW_COMMAND.match(content))


def is_rework_command(content: str | None) -> bool:
    return bool(content and _REWORK_COMMAND.match(content))


def is_help_command(content: str | None) -> bool:
    return bool(content and _HELP_COMMAND.match(content))


def extract_help_response(content: str) -> str:
    command = _HELP_COMMAND.match(content)
    if not command:
        raise SpecCommandError("message is not a /help or /man command")
    verb = content.strip().split(maxsplit=1)[0].lower()
    argument = content[command.end():].strip()
    if verb == "/help":
        if argument:
            raise SpecCommandError("/help takes no arguments; use /man <command>")
        return help_text()
    if not argument:
        raise SpecCommandError("/man requires a command name; for example /man implement")
    return manual_text(argument)


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
    """Parse ordered selectors, preserving the legacy selector-only API."""
    selectors, _ = extract_implementation_request(content)
    return selectors


def extract_implementation_request(content: str) -> tuple[list[str], str]:
    """Parse /implement or /execute selectors and optional guidance.

    The colon is intentionally required before free-form guidance.  That keeps
    selector parsing deterministic and prevents a sentence appended by a client
    from silently being treated as another selector.
    """
    command = _IMPLEMENT_COMMAND.match(content)
    if not command:
        raise SpecCommandError("message is not an /implement or /execute command")
    body = content[command.end():].strip()
    selectors_text, separator, instruction = body.partition(":")
    selectors = _parse_implementation_selectors(selectors_text)
    instruction = instruction.strip()
    if separator and not instruction:
        raise SpecCommandError("/implement requires an execution instruction after ':'")
    return selectors, instruction


def _parse_implementation_selectors(body: str) -> list[str]:
    if not body.strip():
        raise SpecCommandError("/implement requires task IDs such as WP1-T1 or work packages such as WP1")
    selectors: list[str] = []
    for token in re.split(r"[\s,]+", body.strip()):
        task = _TASK_TARGET.fullmatch(token)
        if task:
            selectors.append(f"WP{task.group(1)}-T{task.group(2)}")
            continue
        package = _PACKAGE_TARGET.fullmatch(token)
        if package:
            selectors.append(f"WP{package.group(1)}")
            continue
        if selectors:
            raise SpecCommandError(
                f"non-selector text {token!r} follows an implementation selector; "
                "put execution guidance after ':' (for example, /implement WP1-T1: add tests)"
            )
        raise SpecCommandError(
            f"invalid implementation selector {token!r}; use WP1-T1 or WP1"
        )
    return list(dict.fromkeys(selectors))


def extract_scoped_request(content: str, verb: str, require_instruction: bool = False) -> tuple[list[str], str]:
    """Extract task/work-package targets and optional feedback from a scoped verb."""
    pattern = _REVIEW_COMMAND if verb == "review" else _REWORK_COMMAND
    command = pattern.match(content)
    if not command:
        raise SpecCommandError(f"message is not a /{verb} command")
    body = content[command.end():].strip()
    selectors_text, separator, instruction = body.partition(":")
    selectors = _parse_selectors(selectors_text, verb)
    instruction = instruction.strip()
    if require_instruction and not instruction:
        raise SpecCommandError(f"/{verb} requires feedback after ':'")
    return selectors, instruction


def _parse_selectors(body: str, verb: str) -> list[str]:
    if not body.strip():
        raise SpecCommandError(f"/{verb} requires WP1-T1 or WP1")
    selectors: list[str] = []
    for token in re.split(r"[\s,]+", body.strip()):
        task = _TASK_TARGET.fullmatch(token)
        if task:
            selectors.append(f"WP{task.group(1)}-T{task.group(2)}")
            continue
        package = _PACKAGE_TARGET.fullmatch(token)
        if package:
            selectors.append(f"WP{package.group(1)}")
            continue
        raise SpecCommandError(f"invalid {verb} selector {token!r}; use WP1-T1 or WP1")
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

    async def rework(self, targets: list[tuple[OpenSpecChange, Task]],
                     feedback: str) -> SpecResult:
        """Create a new, traceable change rather than modifying completed work."""
        source = [
            {"change_id": change.id, "task": vars(task)}
            for change, task in targets
        ]
        request = (
            "Create corrective follow-up work for the completed OpenSpec tasks below. "
            "Preserve their history: this must be a new additive change, not a rewrite. "
            f"User feedback: {feedback}\n\nSource tasks:\n{source}"
        )
        async with self._lock:
            draft = await self._adapter.generate_change(request)
            task_ids = [task.id for _, task in targets]
            change = await self._store.create_change(
                OpenSpecChange(**{**vars(draft), "follow_up_for": task_ids}),
                create_change_id(draft.title),
            )
        return SpecResult(
            change.id,
            f"Created corrective follow-up {change.id} for {', '.join(task_ids)}.\n\n" + render_response(change),
            change.path,
        )

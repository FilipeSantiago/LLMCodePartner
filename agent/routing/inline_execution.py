"""Inline task execution: the plan's source in the prompt, one reply, CodePartner edits.

An agentic coder reads the files the planner already located, one tool call per turn,
and every turn re-reads the whole conversation. An inline coder gets that source in its
first prompt (`agent.planning.snippets`), has no tools at all, and replies with exact
replacements (`agent.routing.patch_apply`). CodePartner writes them, runs the task's
checks, and sends a rejected reply or a failing check back once per fix round. A task is
then a few single-response calls instead of an open-ended session.

The attempt is journaled like a native run (a git baseline before the first write,
`RollbackJournal.record_native_changes` after), its checks become execution evidence,
and it ends with the same versioned coder handoff, so review and the job continue
unchanged. When the plan's source is not enough (a second `need_context`, or edits that
stay unusable), `InlineOutcome.fallback` asks the caller to continue as an agentic session
from the files as they are.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shlex
import subprocess
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from agent.planning.models import ReferenceRange, TaskContext
from agent.planning.rebase import rebase_task
from agent.planning.snippets import Snippets, materialize
from agent.prompt import InlineCoderPromptBuilder, InlineCoderPromptContext
from agent.routing import patch_apply
from agent.routing.evidence import verification_succeeded
from agent.routing.gateway import brief_failure, gateway_attempts, same_credential_limit
from agent.routing.models import ExecutionContext
from agent.routing.rollback import RollbackJournal
from agent.task_contracts.policy import path_excluded
from agent.workspace import NATIVE_SHELL_TOOL, RegisteredProject, command_result, run_native_command
from mcp_bridge.models import (DoneEvent, ErrorEvent, Event, ProgressEvent, TextEvent,
                               ToolCallEvent)

log = logging.getLogger("codepartner.inline")
# The tail of a failing check's output returned to the coder.
_CHECK_OUTPUT_CHARS = 4000
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class InlineRequest:
    """One planned task, fitted to the current files, for inline execution."""
    project: RegisteredProject
    task_context: TaskContext
    source: Snippets
    checks: tuple[str, ...] = ()
    fix_rounds: int = 2
    margin: int = 10
    lint_command: str = ""

    @property
    def root(self) -> Path:
        return Path(self.project.workspace)

    @property
    def edit_files(self) -> tuple[str, ...]:
        return tuple(self.task_context.edit_files())


@dataclass
class InlineOutcome:
    # Why the caller should continue as an agentic session; None when inline finished.
    fallback: str | None = None
    stats: dict = field(default_factory=lambda: {
        "llm_calls": 0, "fix_rounds": 0, "patch_apply_failures": 0,
        "need_context_requests": 0, "checks_failed_final": False,
        "fallback_to_agentic": False})


def _add_usage(total: dict | None, part: dict | None) -> dict | None:
    if not part:
        return total
    combined = dict(total or {})
    for key, value in part.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            previous = combined.get(key)
            combined[key] = (previous if isinstance(previous, (int, float))
                             and not isinstance(previous, bool) else 0) + value
        elif key not in combined:
            combined[key] = value
    return combined


def _read_all(root: Path, paths) -> dict[str, str]:
    contents = {}
    for path in paths:
        try:
            contents[path] = (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return contents


def _find_symbol(root: Path, name: str) -> tuple[str, int] | None:
    """The file and line where a top-level or nested `def`/`class name` is declared."""
    pattern = rf"^\s*(async\s+def|def|class)\s+{name}\b"
    try:
        result = subprocess.run(["git", "-C", str(root), "grep", "-n", "-E", pattern, "--",
                                 "*.py"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        path, _, rest = line.partition(":")
        number, _, _ = rest.partition(":")
        if number.isdigit():
            return path, int(number)
    return None


def _definition_span(text: str, line: int) -> tuple[int, int]:
    """The lines of the definition declared on `line`, decorators included."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return line, line
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.lineno == line):
            return (min([line, *(item.lineno for item in node.decorator_list)]),
                    node.end_lineno or line)
    return line, line


def _requested_references(root: Path, items: tuple[dict, ...]
                          ) -> tuple[list[ReferenceRange], list[str]]:
    """References for an inline coder's `need_context` items, and those not found."""
    references: list[ReferenceRange] = []
    unresolved: list[str] = []
    for item in items:
        why = str(item.get("why") or "requested by the coder")[:200]
        path, symbol = item.get("file"), item.get("symbol")
        if isinstance(path, str) and path.strip():
            path = path.strip()
            if (root / path).is_file() and ".." not in Path(path).parts:
                lines = item.get("lines")
                match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", lines) \
                    if isinstance(lines, str) else None
                references.append(ReferenceRange(
                    path, why, *((int(match.group(1)), int(match.group(2)))
                                 if match else (None, None))))
            else:
                unresolved.append(path)
        elif isinstance(symbol, str) and symbol.strip():
            name = symbol.strip().rsplit("::", 1)[-1].rsplit(".", 1)[-1]
            found = _find_symbol(root, name) if _SYMBOL.match(name) else None
            if found is None:
                unresolved.append(symbol)
                continue
            path, line = found
            text = (root / path).read_text(encoding="utf-8", errors="replace")
            references.append(ReferenceRange(path, why, *_definition_span(text, line)))
    return references, unresolved


def _checks(request: InlineRequest, changed: list[str]) -> list[str]:
    commands = list(request.checks)
    if request.lint_command and changed:
        commands.append(request.lint_command.replace(
            "{files}", " ".join(shlex.quote(path) for path in changed)))
    return commands


class InlineCoder:
    """Runs one task inline on the routed model, falling back to the routed fallbacks
    only while nothing has been written."""

    def __init__(self, config, persistence, executors, health):
        self.config = config
        self.persistence = persistence
        self.executors = executors
        self.health = health

    async def _call(self, task, model, decision, conversation: list[dict],
                    owner_metadata: dict) -> tuple[str | None, dict | None, str | None]:
        """One tool-free reply: (text, usage, error)."""
        error: str | None = None
        for gateway_used in gateway_attempts(model.gateway_mode):
            started = time.monotonic()
            await self.persistence.record_running_model(
                decision, model.name, model.provider, gateway_used)
            context = ExecutionContext(
                messages=conversation, specs=[], decision=decision,
                gateway_url=self.config.agentgateway_url, use_gateway=gateway_used,
                started_at=started, owner_metadata={**owner_metadata,
                                                    "execution_mode": "inline"},
                native=None, prompt_cache_ttl=self.config.execution_cache_ttl,
                tool_free=True)
            parts: list[str] = []
            done: DoneEvent | None = None
            error = None
            try:
                async for event in self.executors.get(model.executor).execute(
                        task, model, context):
                    if isinstance(event, TextEvent):
                        parts.append(event.text)
                    elif isinstance(event, ErrorEvent):
                        error = event.message
                    elif isinstance(event, ToolCallEvent):
                        error = "the inline coder requested a tool, which it does not have"
                    elif isinstance(event, DoneEvent):
                        done = event
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if done is None and error is None:
                error = "inline coder stopped without a terminal result"
            elif done is not None and done.terminal in {"failed", "error"} and error is None:
                error = f"inline coder finished with {done.terminal}"
            latency = int((time.monotonic() - started) * 1000)
            status = "failed" if error else "completed"
            await self.persistence.record_attempt(
                decision, model.name, model.provider, model.executor, gateway_used, status,
                latency, error, done.usage if done else None)
            self.health.record(model.name, model.executor, gateway_used,
                               status == "completed")
            if status == "completed":
                await self.persistence.finalize(decision, model.name, model.provider,
                                                gateway_used, status, latency, None)
                return "".join(parts), done.usage if done else None, None
            if gateway_used and model.provider == "claude" and same_credential_limit(error):
                break
        return None, None, error

    async def run(self, task, decision, request: InlineRequest, messages: list[dict],
                  owner_metadata: dict, excluded_files: tuple[str, ...],
                  outcome: InlineOutcome) -> AsyncIterator[Event]:
        """`messages`: the system messages and the inline prompt."""
        stats = outcome.stats
        job_id = owner_metadata.get("implementation_job_id")
        position = owner_metadata.get("implementation_position")
        journal = (RollbackJournal(self.persistence, job_id, position, [], {})
                   if job_id is not None and position is not None else None)
        root = request.root
        allowed = frozenset(path for path in request.edit_files
                            if not (excluded_files and path_excluded(path, excluded_files)))
        originals = _read_all(root, request.edit_files)
        written: set[str] = set()
        created: set[str] = set()
        usage: dict | None = None
        conversation = list(messages)
        response: patch_apply.PatchResponse | None = None
        last_failures: list[tuple[str, str]] = []
        builder = InlineCoderPromptBuilder()
        attempts = [self.config.models[name] for name in
                    (decision.recommended_model, *decision.fallback_models)]
        model = attempts.pop(0)
        if journal is not None:
            await journal.record_git_baseline(str(root), request.edit_files)
        started = time.monotonic()
        # Replies on the current model; a failed first call moves to the next model.
        replies = 0
        try:
            while True:
                stats["llm_calls"] += 1
                text, part, error = await self._call(task, model, decision, conversation,
                                                     owner_metadata)
                if error is not None:
                    if not replies and attempts:
                        yield ProgressEvent(f"- 🔁 {model.name} stopped: "
                                            f"{brief_failure(error)}. Trying another route.\n")
                        model = attempts.pop(0)
                        continue
                    yield ErrorEvent(error)
                    return
                replies += 1
                usage = _add_usage(usage, part)
                conversation = [*conversation, {"role": "assistant", "content": text}]
                try:
                    response = patch_apply.parse_response(text)
                except patch_apply.PatchFormatError as exc:
                    stats["patch_apply_failures"] += 1
                    if stats["fix_rounds"] >= request.fix_rounds:
                        outcome.fallback = f"the inline reply stayed unusable ({exc})"
                        return
                    stats["fix_rounds"] += 1
                    conversation.append({"role": "user", "content": builder.repair(exc)})
                    continue
                if response.need_context and response.empty:
                    stats["need_context_requests"] += 1
                    if stats["need_context_requests"] > 1:
                        outcome.fallback = ("the inline coder needed context beyond the "
                                            "plan twice")
                        return
                    references, unresolved = _requested_references(root, response.need_context)
                    given = materialize(TaskContext(task.id, "", references=references), root,
                                        budget=request.source.budget)
                    yield ProgressEvent(f"- 📎 inline: {len(references)} requested context "
                                        f"region(s) added\n")
                    conversation.append({"role": "user", "content": builder.context_given(
                        given.text, unresolved)})
                    continue
                contents, errors = patch_apply.resolve(response, root, allowed)
                if errors:
                    stats["patch_apply_failures"] += 1
                    if stats["fix_rounds"] >= request.fix_rounds:
                        outcome.fallback = ("the inline edits could not be applied: "
                                            + "; ".join(errors[:3]))
                        return
                    stats["fix_rounds"] += 1
                    yield ProgressEvent(f"- ✏️ inline: {len(errors)} edit(s) rejected; "
                                        "asking for a corrected reply\n")
                    conversation.append({"role": "user", "content": builder.apply_failed(
                        errors, self._fresh_source(request, originals))})
                    continue
                for path in patch_apply.write(root, contents):
                    if path not in originals:
                        created.add(path)
                    written.add(path)
                yield ProgressEvent(f"- ✏️ inline: {len(response.edits)} edit(s) and "
                                    f"{len(response.new_files)} new file(s) applied\n")
                last_failures = await self._run_checks(
                    request, sorted(written), journal)
                if not last_failures:
                    break
                yield ProgressEvent(f"- 🧪 inline: {len(last_failures)} check(s) failed\n")
                if stats["fix_rounds"] >= request.fix_rounds:
                    stats["checks_failed_final"] = True
                    break
                stats["fix_rounds"] += 1
                conversation.append({"role": "user", "content": builder.checks_failed(
                    last_failures, self._fresh_source(request, originals))})
        finally:
            if journal is not None:
                await journal.record_native_changes(excluded_files)
            stats["fallback_to_agentic"] = outcome.fallback is not None
            self._record_telemetry(task, model, decision, owner_metadata, stats, usage)
            log.info("inline_execution task_id=%s model=%s stats=%s latency_ms=%d",
                     task.id, model.name, stats, int((time.monotonic() - started) * 1000))
        yield TextEvent(self._report(task, response, written, created, last_failures))
        yield DoneEvent("endTurn", usage)

    @staticmethod
    def _fresh_source(request: InlineRequest, originals: dict[str, str]) -> str:
        """The edit regions as they are now, refitted past the edits already written."""
        current = _read_all(request.root, request.edit_files)
        fitted = rebase_task(request.task_context, originals, current)
        return materialize(replace(fitted, references=[]), request.root,
                           margin=request.margin, budget=request.source.budget).text

    async def _run_checks(self, request: InlineRequest, changed: list[str],
                          journal: RollbackJournal | None) -> list[tuple[str, str]]:
        failures = []
        for command in _checks(request, changed):
            output, exit_code = await run_native_command(request.project, command)
            result = command_result(output, exit_code)
            success = (exit_code == 0 if exit_code is not None
                       else verification_succeeded(result))
            if journal is not None:
                await journal.record_verification(NATIVE_SHELL_TOOL, {"command": command},
                                                  result, success)
            if not success:
                failures.append((command, result[-_CHECK_OUTPUT_CHARS:]))
        return failures

    @staticmethod
    def _report(task, response: patch_apply.PatchResponse | None, written: set[str],
                created: set[str], failures: list[tuple[str, str]]) -> str:
        """The versioned coder handoff the job and the reviewer expect."""
        response = response or patch_apply.PatchResponse()
        criteria = [{"criterion": str(item.get("criterion") or ""),
                     "status": item.get("status") if item.get("status") in
                     {"satisfied", "blocked"} else "satisfied",
                     "paths": [path for path in item.get("paths") or []
                               if isinstance(path, str)],
                     "evidence_ids": []}
                    for item in response.criteria if item.get("criterion")]
        if not criteria:
            criteria = [{"criterion": criterion, "status": "satisfied",
                         "paths": sorted(written), "evidence_ids": []}
                        for criterion in (task.acceptance_criteria or [task.description])]
        summary = response.summary or "Applied the planned changes inline."
        if failures:
            summary += (" These checks still fail after the last fix round: "
                        + ", ".join(command for command, _ in failures) + ".")
        handoff = {
            "schema_version": 1, "task_id": task.id, "summary": summary,
            "changed_files": [{"path": path,
                               "change": "created" if path in created else "modified",
                               "purpose": "inline edit"} for path in sorted(written)],
            "criteria": criteria, "remaining_gaps": list(response.remaining_gaps),
            "execution_dispute": None, "context_handoff": response.context_handoff,
        }
        return f"{summary}\n\n{json.dumps(handoff, ensure_ascii=False)}"

    @staticmethod
    def _record_telemetry(task, model, decision, owner_metadata: dict, stats: dict,
                          usage: dict | None) -> None:
        try:
            from telemetry.storage import TelemetryStorage
            TelemetryStorage().record_aci_attempt(
                "inline_" + uuid4().hex,
                {"task_id": task.id, "provider": model.provider, "model": model.name,
                 "decision_id": decision.decision_id,
                 "implementation_job_id": owner_metadata.get("implementation_job_id"),
                 "implementation_task_id": task.id,
                 # Claude reports cost; Codex tokens only, so None keeps the sum honest.
                 "cost_usd": (usage or {}).get("total_cost_usd")},
                {"role": "coder", "execution_mode": "inline", **stats}, usage)
        except Exception:
            log.exception("could not record inline telemetry task_id=%s", task.id)


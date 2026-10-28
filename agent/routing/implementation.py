"""Sequential, durable execution and outcome review of OpenSpec tasks."""

import asyncio
import json
import logging
import re
import shlex
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

from agent.planning.models import ApprovedWPPlan
from agent.planning.rebase import rebase_task, with_package_markers
from agent.planning.render import (DEFAULT_MARGIN, coder_context, prior_task_context,
                                    reviewer_context, task_files, task_read_only_files)
from agent.routing.config import load_routing_config
from agent.routing.gateway import brief_failure
from conversation import presentation as ui
from agent.routing.models import HALTED_JOB_STATUSES, ImplementationJob, ImplementationJobTask
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import (ReviewDecision, ReviewError, ReviewFinding,
                                  ReviewFormatError, ReviewService, _finding_owner,
                                  model_identity)
from agent.routing.rollback import (RollbackJournal, declared_artifacts,
                                    journal_handlers)
from agent.routing.evidence import (has_passing_test_run, is_reproducible_verification_call,
                                    shell_test_command, verification_succeeded)
from agent.routing.task_execution import TaskExecutionService
from agent.spec.artifact_store import ArtifactStore
from agent.spec.models import (CompletionReview, Task, ValidationContract,
                               WorkPackage, WorkPackageReview)
from agent.task_contracts.contractor import TaskContractor, requested_paths, USER_RESOLUTION_REASON
from agent.task_contracts.persistence import TaskContractPersistence
from agent.workspace import (NATIVE_SHELL_TOOL, command_result, native_project,
                             run_native_command)
from mcp_bridge import bridge
from mcp_bridge.activity import describe_call
from mcp_bridge.models import (DoneEvent, ErrorEvent, Event, ProgressEvent, Run,
                               TextEvent, ToolCallEvent)
from mcp_bridge.registry import ToolSpec


log = logging.getLogger("codepartner.implementation")
REVISION_BUDGET = 3
_REVIEWER_CONTEXT_MARKER = "BEGIN CODEPARTNER REVIEWER REVISION CONTEXT"
_REVIEWER_CONTEXT_END = "END CODEPARTNER REVIEWER REVISION CONTEXT"
# A failed provider run stays attached to its implementation job. The next
# attempt starts from the journaled project state, not from a chat-only answer.
CODER_RECOVERY_STATUS = "awaiting_coder_recovery"
_REQUIRES_TESTS = re.compile(r"\b(?:tests?|pytest|unit[- ]tests?)\b", re.IGNORECASE)
_CONTEXT_HANDOFF_LIMITS = {"files": 12, "symbols": 10, "decisions": 4,
                           "dependencies": 6, "verification": 4, "constraints": 4}


def _compact_context_handoff(value: object, changed: list[str]) -> dict[str, list[str]]:
    """Bound coder-provided navigation hints before persisting or prompting with them."""
    source = value if isinstance(value, dict) else {}
    result: dict[str, list[str]] = {}
    for kind, limit in _CONTEXT_HANDOFF_LIMITS.items():
        supplied = source.get(kind)
        supplied = supplied if isinstance(supplied, list) else []
        candidates = [*changed, *supplied] if kind == "files" else supplied
        if not isinstance(candidates, list):
            continue
        items: list[str] = []
        for item in candidates:
            if not isinstance(item, str):
                continue
            text = " ".join(item.split())[:160]
            if not text:
                continue
            if kind == "files":
                path = PurePosixPath(text)
                if path.is_absolute() or any(part in {"..", "."} for part in text.split("/")):
                    continue
            if text not in items:
                items.append(text)
            if len(items) >= limit:
                break
        if items:
            result[kind] = items
    return result


def handoff_blocker(report: str, task_id: str) -> str | None:
    """Reject invalid handoffs and coder-declared blocked task criteria."""
    handoff = _handoff(report)
    if (handoff.get("schema_version") != 1 or handoff.get("task_id") != task_id
            or not isinstance(handoff.get("summary"), str)
            or not handoff["summary"].strip()):
        return "Coder handoff invalid: missing the versioned result for this task"
    criteria = handoff.get("criteria")
    if (not isinstance(criteria, list) or not criteria
            or any(not isinstance(item, dict)
                   or item.get("status") not in {"satisfied", "blocked"}
                   for item in criteria)):
        return "Coder handoff invalid: criteria must report satisfied or blocked"
    gaps = handoff.get("remaining_gaps")
    if not isinstance(gaps, list) or any(not isinstance(gap, str) for gap in gaps):
        return "Coder handoff invalid: remaining_gaps must be an array of strings"
    # A gap can describe work assigned to a later task. Let the reviewer judge
    # its relevance to this task instead of stopping the sequential job here.
    if any(item["status"] == "blocked" for item in criteria):
        return "Coder handoff blocked: active-task acceptance criteria are unresolved"
    return None


def _requires_test_run(package) -> bool:
    validation = getattr(package, "validation", None)
    return bool(validation and _REQUIRES_TESTS.search(" ".join(validation.required)))


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


class ImplementationService:
    """Runs an implementation job one task at a time, including tool resumes."""

    def __init__(self, artifacts: ArtifactStore,
                 task_execution: TaskExecutionService | None = None,
                 persistence: RoutingPersistence | None = None,
                 reviewer: ReviewService | None = None,
                 contract_persistence: TaskContractPersistence | None = None,
                 kb_refresh=None, kb_catch_up=None):
        self.artifacts = artifacts
        self.task_execution = task_execution or TaskExecutionService()
        self.persistence = persistence or self.task_execution.persistence
        self.reviewer = reviewer or ReviewService(getattr(self.task_execution, "config", None))
        self.contract_persistence = contract_persistence or TaskContractPersistence(
            self.persistence.path)
        self.contractor = TaskContractor()
        self.config = (getattr(self.reviewer, "config", None)
                       or getattr(self.task_execution, "config", None)
                       or load_routing_config())
        self.kb_refresh = kb_refresh
        # Reindexes first when accepted edits are newer than the KB index.
        self.kb_catch_up = kb_catch_up

    def _margin(self) -> int:
        """Lines added around each planner edit range in coder and reviewer context."""
        planner = getattr(self.config, "planner", None)
        return getattr(planner, "edit_margin_lines", DEFAULT_MARGIN)

    def _coder_line(self, model: str | None, fallbacks: tuple[str, ...] = ()) -> str:
        """Which model routing picked for the coder."""
        try:
            label = model_identity(self.config, model).label if model else "unknown"
        except ReviewError:
            label = model or "unknown"
        return ui.block(f"_coder: {label}"
                        + (f" (fallbacks: {', '.join(fallbacks)})" if fallbacks else "") + "_")

    async def _fitted_plan(self, plan: ApprovedWPPlan, job_id: str,
                           task_ids: list[str] | None) -> ApprovedWPPlan:
        """`plan` with the given tasks (all when None) fitted to the current files: their
        ranges moved past what earlier tasks changed, and package markers added."""
        project = native_project()
        preimages = getattr(self.persistence, "rollback_preimages", None)
        if project is None or preimages is None:
            return plan
        root = Path(project.workspace)
        # Each file's first preimage in this job is the source the plan was made against.
        before = {path: value["before_content"]
                  for path, value in (await preimages(job_id)).items()
                  if value.get("before_content") is not None}
        current = {}
        for path in before:
            try:
                current[path] = (root / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        planned = {edit.file for context in plan.tasks.values() for edit in context.edits}
        tasks = {task_id: with_package_markers(rebase_task(context, before, current),
                                               planned, root)
                 if task_ids is None or task_id in task_ids else context
                 for task_id, context in plan.tasks.items()}
        return replace(plan, tasks=tasks)

    async def _approved_plan(self, job_id: str) -> ApprovedWPPlan | None:
        """The approved WP plan this job executes, if it came from the planner."""
        lookup = getattr(self.persistence, "job_plan_id", None)
        plan_id = await lookup(job_id) if lookup else None
        value = await self.persistence.approved_plan(plan_id) if plan_id else None
        return ApprovedWPPlan.from_dict(value) if value else None

    async def _unplanned_prior_context(self, item: ImplementationJobTask, task: Task,
                                       contract) -> str:
        """Use saved reports from earlier tasks of this package when no plan exists."""
        _, package = await self._package_for_task(item.task_id)
        events: list[dict] = []
        for row in await self._package_rows(item.job_id, package.id):
            if row.position >= item.position or row.status not in {"implemented", "completed"}:
                continue
            report = await self.persistence.coder_report_for(item.job_id, row.position)
            handoff = _handoff(report or "")
            if not handoff:
                continue
            changed = [entry.get("path") for entry in handoff.get("changed_files") or []
                       if isinstance(entry, dict) and isinstance(entry.get("path"), str)]
            compact = _compact_context_handoff(handoff.get("context_handoff"), changed)
            if not compact:
                continue
            events.append({"task_id": row.task_id, "kind": "task_handoff",
                           "summary": "Structured task context", "payload": compact})
            events.append({"task_id": row.task_id, "kind": "decisions",
                           "summary": str(handoff.get("summary") or "")[:350]})
        return prior_task_context(events, item.task_id, task.depends_on,
                                  tuple(getattr(contract, "primary_files", ()) or ()))

    async def _record_plan_events(self, plan: ApprovedWPPlan, item: ImplementationJobTask,
                                  task_id: str, report: str, failure: str | None,
                                  contract_revision: int, retried: bool) -> None:
        """Enrich the reviewer context with what actually happened for this task."""
        events: list[tuple[str, str, object]] = []
        handoff = _handoff(report)
        if failure:
            events.append(("failure", failure[:500], None))
        if handoff.get("summary"):
            events.append(("decisions", str(handoff["summary"])[:3000], None))
        changed = [entry.get("path") for entry in handoff.get("changed_files") or []
                   if isinstance(entry, dict) and isinstance(entry.get("path"), str)]
        compact = _compact_context_handoff(handoff.get("context_handoff"), changed)
        if compact:
            events.append(("task_handoff", "Structured task context", compact))
        if changed:
            events.append(("changed_files", ", ".join(changed), changed))
            context = plan.tasks.get(task_id)
            planned = {*plan.shared.relevant_files,
                       *(context.relevant_files if context else ())}
            outside = [path for path in changed if path not in planned]
            if outside:
                events.append(("deviations", "changed files not named in the plan: "
                               + ", ".join(outside), outside))
        if handoff.get("execution_dispute"):
            events.append(("deviations", "coder raised an execution dispute",
                           handoff["execution_dispute"]))
        gaps = [str(gap) for gap in handoff.get("remaining_gaps") or [] if str(gap).strip()]
        if gaps:
            events.append(("unresolved", "; ".join(gaps)[:1000], gaps))
        evidence = await self.persistence.verification_evidence(item.job_id, item.position)
        if evidence:
            failed = sum(1 for entry in evidence if not entry.get("success"))
            events.append(("tests", f"{len(evidence)} captured executions, {failed} failed",
                           [{"tool": entry.get("tool"), "success": entry.get("success")}
                            for entry in evidence]))
        if contract_revision > 1 or retried:
            events.append(("retries", f"contract revision {contract_revision}"
                           + ("; retried after an interrupted coder" if retried else ""), None))
        for kind, summary, payload in events:
            await self.persistence.record_reviewer_context_event(
                plan.plan_id, item.job_id, item.position, task_id, kind, summary, payload)

    @staticmethod
    def _findings_json(findings: tuple[ReviewFinding, ...]) -> list[str]:
        if not findings:
            return []
        payload = [{
            "problem": finding.problem,
            "suggested_task": finding.suggested_task,
            "approval_criteria": list(finding.approval_criteria),
            "artifact_paths": list(finding.artifact_paths),
        } for finding in findings]
        return ["```json", json.dumps(payload, ensure_ascii=False, indent=2), "```"]

    @staticmethod
    def _resolution_options(task_id: str, verdict: str,
                            job_id: str | None = None) -> str:
        suffix = f" (job {ui.short_job(job_id)})" if job_id else ""
        status = ("The reviewer still reports blocking findings."
                  if verdict == "needs_work" else "The review is ready.")
        return ui.block(f"{task_id}{suffix}: {status}") + ui.menu((
            "1. Reply `accept` to keep the implementation and continue.",
            "2. Reply `submit to coder again: <additional prompt>` to keep the current "
            "work and start a fresh coder/reviewer cycle.",
        ))

    @staticmethod
    def _decision_text(task_id: str, verdict: str, decision: ReviewDecision,
                       review: CompletionReview) -> str:
        state = ui.status("approved" if verdict == "done" else verdict)
        implementation = tuple(finding for finding in decision.findings
                               if finding.kind == "implementation")
        verification = tuple(finding for finding in decision.findings
                             if finding.kind == "verification")
        return "".join((
            ui.heading("🔍", f"Review · {task_id} — {state}",
                       f"coder: {review.coder.label} → reviewer: {review.reviewer.label}"),
            ui.paragraph("Why", decision.rationale),
            ui.validation_table(decision.validation_results),
            ui.paragraph("Inspected", ui.paths(decision.inspected_paths)),
            ui.section("Evidence and limitations", decision.evidence),
            (ui.block("**Implementation findings**") + ui.findings_list(implementation)
             if implementation else ""),
            (ui.block("**Verification incomplete**") + ui.findings_list(verification)
             if verification else ""),
        ))

    @staticmethod
    def _revision_context(task_id: str, findings: tuple[ReviewFinding, ...]) -> str:
        payload = [finding.as_dict() for finding in findings
                   if finding.owner == "coder"]
        return "\n".join([
            _REVIEWER_CONTEXT_MARKER,
            f"Reviewer findings for {task_id}:",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "Treat each suggested_task as advisory. You may choose a better implementation, "
            "but you must resolve every problem and satisfy every approval_criteria item. "
            "Edit every reviewer-named artifact_path directly unless it is demonstrably not "
            "applicable; explain any such exception. Do not create substitute reports or "
            "parallel documentation instead of correcting the named artifacts. If verification "
            "is requested, actually run it with the available project toolchain and report its "
            "captured result, or report the concrete blocker. When pytest evidence is required, "
            "capture collection first, then execution totals, and derive documented category counts "
            "from the collected node IDs rather than regexes or manual source inspection. Never "
            "claim an unexecuted check passed. "
            "In your final report, map each approval criterion to concrete project-relative "
            "artifacts and state accurately which tests ran. Return the versioned JSON coder "
            "handoff requested by the task prompt, including every changed_files path; the "
            "reviewer will independently reconcile it with the mutation journal and project. "
            "Communicate status and execution output only in that final response; never create "
            "summary, completion, evidence, "
            "verification, or test-result files solely to communicate with the reviewer.",
            "If you disagree with a finding that code or tests do not compile, collect, or run, "
            "include an execution_dispute object in the final JSON. It must contain the claim, "
            "source_paths, working_directory, and executable steps with the exact advertised "
            "tool name and arguments. CodePartner will replay those steps for the reviewer. "
            "A prose assertion or a repository-local runner is not a valid dispute.",
            _REVIEWER_CONTEXT_END,
        ])

    @staticmethod
    def _coder_manifest(report: str) -> dict | None:
        decoder = json.JSONDecoder()
        for index, char in enumerate(report):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(report[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("schema_version") == 1:
                return value
        return None

    async def _capture_execution_dispute(
            self, journal: RollbackJournal, report: str,
            direct_tool_handlers: dict, existing_evidence: list[dict]) -> list[str]:
        """Replay a coder's exact safe verification procedure for reviewer evidence."""
        manifest = self._coder_manifest(report)
        dispute = manifest.get("execution_dispute") if manifest else None
        if dispute in (None, {}):
            return []
        if not isinstance(dispute, dict) or not isinstance(dispute.get("claim"), str):
            return ["execution_dispute must contain a non-empty claim"]
        source_paths = dispute.get("source_paths")
        working_directory = dispute.get("working_directory")
        if (not isinstance(source_paths, list) or not source_paths
                or not all(isinstance(path, str) and path and not path.startswith("/")
                           and ".." not in path.split("/") for path in source_paths)):
            return ["execution_dispute must contain safe project-relative source_paths"]
        if (not isinstance(working_directory, str) or not working_directory
                or working_directory.startswith("/")
                or ".." in working_directory.split("/")):
            return ["execution_dispute must contain a project-relative working_directory"]
        steps = dispute.get("steps")
        if not isinstance(steps, list) or not steps:
            return ["execution_dispute must contain at least one executable step"]
        seen = {(item["tool"], json.dumps(item["arguments"], sort_keys=True))
                for item in existing_evidence}
        errors = []
        for number, step in enumerate(steps, 1):
            tool = step.get("tool") if isinstance(step, dict) else None
            arguments = step.get("arguments") if isinstance(step, dict) else None
            project = native_project() if tool == NATIVE_SHELL_TOOL else None
            if (not isinstance(tool, str)
                    or (tool not in direct_tool_handlers and project is None)
                    or not isinstance(arguments, dict)
                    or not is_reproducible_verification_call(arguments)):
                errors.append(
                    f"execution_dispute step {number} is not an available verification call")
                continue
            signature = (tool, json.dumps(arguments, sort_keys=True))
            if signature in seen:
                continue
            try:
                if project is not None:
                    output, exit_code = await run_native_command(project, arguments["command"])
                    result = command_result(output, exit_code)
                else:
                    result = await direct_tool_handlers[tool](arguments)
                success = verification_succeeded(result)
            except Exception as exc:
                result = f"{type(exc).__name__}: {exc}"
                success = False
            await journal.record_verification(tool, arguments, result, success)
            seen.add(signature)
        return errors

    @classmethod
    def _handoff_evidence_errors(cls, report: str,
                                 execution_evidence: list[dict]) -> list[str]:
        manifest = cls._coder_manifest(report)
        if manifest is None:
            return ["coder did not return the required versioned JSON handoff"]
        valid_ids = {item["evidence_id"] for item in execution_evidence}
        errors = []
        criteria = manifest.get("criteria", [])
        if not isinstance(criteria, list):
            return ["coder handoff criteria must be an array"]
        for index, criterion in enumerate(criteria, 1):
            references = criterion.get("evidence_ids", []) \
                if isinstance(criterion, dict) else []
            if not isinstance(references, list):
                errors.append(f"criterion {index} evidence_ids must be an array")
                continue
            invalid = [reference for reference in references
                       if not isinstance(reference, int) or reference not in valid_ids]
            if invalid:
                errors.append(
                    f"criterion {index} cites non-CodePartner evidence IDs: {invalid[:3]}")
        return errors

    async def _failed_text(self, job_id: str, position: int, error: str) -> str:
        item = await self.persistence.implementation_task(job_id, position)
        coder_model = await self.persistence.actual_model_for(item.decision_id) if item else None
        try:
            coder_label = model_identity(self.config, coder_model).label if coder_model else "unknown"
        except ReviewError:
            coder_label = coder_model or "unknown"
        task_id = item.task_id if item else f"position {position}"
        return "".join((
            ui.heading("⛔", f"{task_id} — Coding failed",
                       f"coder: {coder_label} · reviewer: not run"),
            ui.quote(f"Reason: {brief_failure(error)}"),
            ui.block(f"Implementation job {ui.short_job(job_id)} stopped."),
        ))

    async def _coder_failed(self, job_id: str, position: int, error: str) -> str:
        """Keep an exhausted operational failure available for a later chat retry."""
        await self.persistence.update_implementation_task(
            job_id, position, CODER_RECOVERY_STATUS, error=error)
        item = await self.persistence.implementation_task(job_id, position)
        coder_model = await self.persistence.actual_model_for(item.decision_id) if item else None
        try:
            coder_label = model_identity(self.config, coder_model).label if coder_model else "unknown"
        except ReviewError:
            coder_label = coder_model or "unknown"
        task_id = item.task_id if item else f"position {position}"
        if error.startswith("Coder handoff blocked"):
            # The coder finished and reported gaps; the Reviewer can still judge it.
            report = await self.persistence.coder_report_for(job_id, position)
            gaps = [gap for gap in _handoff(report or "").get("remaining_gaps") or []
                    if isinstance(gap, str) and gap.strip()]
            return "".join((
                ui.heading("⏸", f"{task_id} — Coder reported unresolved gaps",
                           f"coder: {coder_label} · reviewer: not run yet"),
                ui.quote(f"Reason: {brief_failure(error)}"),
                *(ui.quote("Gap: " + gap) for gap in gaps),
                ui.block(f"{task_id} (job {ui.short_job(job_id)}): choose what to do next."),
                ui.menu((
                    "1. Reply `review anyway` to send the coder's report to the Reviewer, "
                    "which judges the acceptance criteria; the job continues if it passes.",
                    "2. Reply `try coder again: <guidance>` to rerun the coder under the same "
                    "task contract.",
                )),
            ))
        return "".join((
            ui.heading("⏸", f"{task_id} — Coding interrupted",
                       f"coder: {coder_label} · reviewer: not run (the coder did not finish)"),
            ui.quote(f"Reason: {brief_failure(error)}"),
            ui.block("Files written before the interruption are still in the project."),
            ui.block(f"{task_id} (job {ui.short_job(job_id)}): choose what to do next."),
            ui.menu((
                "1. Reply `try coder again` (optionally `try coder again: <guidance>`) to "
                "continue this task from the saved files.",
            )),
        ))

    def _pending_options(self, item: ImplementationJobTask, latest: dict | None,
                         job_id: str) -> str:
        if item.status == CODER_RECOVERY_STATUS or latest is None:
            return (f"{item.task_id} for job {job_id}: reply `try coder again` to continue on a "
                    "different model.")
        return self._resolution_options(item.task_id, latest["verdict"], job_id)

    async def create_job(self, selectors: list[str], instruction: str = "",
                         conversation_id: str | None = None,
                         project_id: str | None = None) -> ImplementationJob:
        targets = await self.artifacts.tasks_for_selectors(selectors)
        job = ImplementationJob.create(selectors, [task.id for _, task in targets], instruction)
        queued = [
            ImplementationJobTask(job.job_id, position, change.id, task.id)
            for position, (change, task) in enumerate(targets)
        ]
        await self.persistence.create_implementation_job(job, queued, conversation_id,
                                                         project_id)
        return job

    async def _halted(self, job_id: str) -> bool:
        """A stopped (or stopping) job must not advance, review, or record failures."""
        return await self.persistence.implementation_job_status(job_id) in HALTED_JOB_STATUSES

    async def start(self, job: ImplementationJob, messages: list[dict],
                    specs: list[ToolSpec], direct_tool_handlers: dict | None = None) -> AsyncIterator[Event]:
        yield TextEvent(ui.block(
            f"**Job {ui.short_job(job.job_id)}** queued {len(job.task_ids)} task(s): "
            f"{', '.join(job.task_ids)}"))
        if self.kb_catch_up is not None:
            caught_up = await self.kb_catch_up(
                await self.persistence.implementation_job_project_id(job.job_id))
            if caught_up:
                yield TextEvent(ui.block(caught_up))
        async for event in self._advance(
                job.job_id, messages, specs, job.instruction, direct_tool_handlers or {}):
            yield event

    async def resume(self, run: Run, messages: list[dict],
                     specs: list[ToolSpec]) -> AsyncIterator[Event]:
        job_id = run.metadata.get("implementation_job_id")
        position = run.metadata.get("implementation_position")
        if not job_id or position is None:
            raise RuntimeError("routed run is not attached to an implementation job")
        report = await self.persistence.coder_report_for(job_id, position)
        failed: str | None = None
        paused = False
        terminal: DoneEvent | None = None
        async for event in self.task_execution.resume(run, messages, specs):
            if isinstance(event, TextEvent):
                report += event.text
            if isinstance(event, ErrorEvent):
                failed = event.message
            if isinstance(event, ToolCallEvent):
                paused = True
                yield ProgressEvent(
                    ui.progress_line("tool", describe_call(event.name, event.arguments)))
            if isinstance(event, DoneEvent):
                terminal = event
            elif not isinstance(event, TextEvent):
                # Coder prose is kept in the saved report, not echoed to chat.
                yield event
        await self.persistence.save_coder_report(job_id, position, report)
        if paused:
            return
        if await self._halted(job_id):
            yield TextEvent(ui.block(
                f"⏹ Job {ui.short_job(job_id)} was stopped; "
                f"{run.metadata.get('implementation_task_id') or 'the task'} "
                "was not reviewed and no further tasks will run."))
            if terminal is not None:
                yield terminal
            return
        if not failed and getattr(self.task_execution, "strict_handoff", False):
            blocker = handoff_blocker(
                report, run.metadata.get("implementation_task_id", ""))
            if blocker:
                if blocker.startswith("Coder handoff invalid"):
                    item = await self.persistence.implementation_task(job_id, position)
                    async for event in self._continue_after_coder_failure(
                            item, messages, specs,
                            run.metadata.get("implementation_instruction", ""),
                            run.direct_tool_handlers, blocker):
                        yield event
                else:
                    yield TextEvent(await self._coder_failed(job_id, position, blocker))
                return
        if failed:
            item = await self.persistence.implementation_task(job_id, position)
            async for event in self._continue_after_coder_failure(
                    item, messages, specs, run.metadata.get("implementation_instruction", ""),
                    run.direct_tool_handlers, failed):
                yield event
        else:
            async for event in self._task_finished(
                    job_id, position, messages, specs,
                    run.metadata.get("implementation_instruction", ""),
                    run.direct_tool_handlers, report, run.metadata.get("model")):
                yield event
        if terminal is not None and not failed:
            yield terminal

    async def finish_child(self, run: Run, messages: list[dict],
                           specs: list[ToolSpec]) -> AsyncIterator[Event]:
        job_id = run.metadata["implementation_job_id"]
        position = run.metadata["implementation_position"]
        report = await self.persistence.coder_report_for(job_id, position)
        async for event in self._task_finished(
                job_id, position, messages, specs,
                run.metadata.get("implementation_instruction", ""),
                run.direct_tool_handlers, report, run.metadata.get("model")):
            yield event

    async def _advance(self, job_id: str, messages: list[dict], specs: list[ToolSpec],
                       instruction: str = "", direct_tool_handlers: dict | None = None
                       ) -> AsyncIterator[Event]:
        if await self._halted(job_id):
            yield TextEvent(ui.block(f"⏹ Job {ui.short_job(job_id)} was stopped; "
                                     "no further tasks will run."))
            return
        item = await self.persistence.next_implementation_task(job_id)
        if item is None:
            await self.persistence.finish_implementation_job(job_id)
            yield TextEvent(ui.heading("✅", f"Job {ui.short_job(job_id)} completed"))
            return
        async for event in self._run_item(item, messages, specs, instruction,
                                          direct_tool_handlers or {}):
            yield event

    async def _run_item(self, item: ImplementationJobTask, messages: list[dict],
                        specs: list[ToolSpec], instruction: str,
                        direct_tool_handlers: dict,
                        exclude_models: frozenset[str] = frozenset()) -> AsyncIterator[Event]:
        change, task = await self.artifacts.task_by_id(item.task_id)
        plan = await self._approved_plan(item.job_id)
        if plan is not None and task.id in plan.tasks:
            plan = await self._fitted_plan(plan, item.job_id, [task.id])
        prior_events = [event for event in await self.persistence.reviewer_context_events(
                            plan.plan_id, item.job_id)
                        if event["position"] < item.position] if plan else []
        contract = await self.contract_persistence.latest(item.job_id, item.position)
        if contract is None:
            contract = self.contractor.initial_contract(
                task, specs, instruction,
                task_context=plan.tasks.get(task.id) if plan else None)
            await self.contract_persistence.save(item.job_id, item.position, contract)
        if contract.excluded_files:
            specs = [spec for spec in specs if not spec.name.startswith("kb_")]
            direct_tool_handlers = {name: handler for name, handler in
                                      direct_tool_handlers.items()
                                      if not name.startswith("kb_")}
        unplanned_context = (await self._unplanned_prior_context(item, task, contract)
                             if plan is None else "")
        await self.persistence.update_implementation_task(item.job_id, item.position, "running")
        yield TextEvent(ui.heading(
            "🛠", f"{task.id} · {task.title}",
            f"task complexity {task.complexity} · {change.id} · contract r{contract.revision}"))
        failure: str | None = None
        paused = False
        report_parts: list[str] = []
        decision_id: str | None = None
        routed = None
        announced = False
        async def record_decision(decision) -> None:
            nonlocal decision_id, routed
            decision_id = decision.decision_id
            routed = decision
            await self.persistence.update_implementation_task(
                item.job_id, item.position, "running", decision_id=decision_id
            )

        tracked_handlers = journal_handlers(
            self.persistence, item.job_id, item.position, specs, direct_tool_handlers,
            declared_artifacts("\n".join((
                task.title, task.description, task.context,
            ))))
        async for event in self.task_execution.execute(
                task, change.id, messages, specs, on_decision=record_decision,
                execution_instruction=instruction,
                direct_tool_handlers=tracked_handlers, exclude_models=exclude_models,
                contract=contract, owner_metadata={
                    "implementation_job_id": item.job_id,
                    "implementation_position": item.position,
                    "implementation_task_id": task.id,
                    **({"plan_id": plan.plan_id} if plan else {}),
                    **({"edit_ranges": [
                        {"file": edit.file,
                         "start_line": max(1, edit.start_line - self._margin()),
                         "end_line": edit.end_line + self._margin()}
                        for edit in plan.tasks[task.id].edits
                        if edit.start_line is not None and edit.end_line is not None]}
                       if plan and task.id in plan.tasks else {}),
                  }, **({"plan_context": coder_context(
                         plan, task.id, prior_events, task.description, self._margin()),
                         "exploration_scope": tuple(task_files(plan, task.id)),
                         "exploration_read_only": tuple(task_read_only_files(plan, task.id))}
                     if plan else {"handoff_context": unplanned_context})):
            if routed is not None and not announced:
                announced = True
                yield TextEvent(self._coder_line(
                    routed.recommended_model, tuple(routed.fallback_models)))
            if isinstance(event, TextEvent):
                report_parts.append(event.text)
            if isinstance(event, ToolCallEvent):
                yield ProgressEvent(ui.progress_line(
                    "tool", describe_call(event.name, event.arguments)))
                paused = True
                run = bridge.run_for(event.tool_call_id)
                if run is not None:
                    run.metadata.update({
                        "implementation_job_id": item.job_id,
                        "implementation_position": item.position,
                        "implementation_task_id": task.id,
                        "implementation_instruction": instruction,
                        "implementation_service": self,
                    })
                await self.persistence.update_implementation_task(item.job_id, item.position, "waiting")
            if isinstance(event, ErrorEvent):
                failure = event.message
            # Coder prose stays in the saved report; the chat shows one line per
            # tool call (from the bridge) and the summarized handoff below.
            if not isinstance(event, (TextEvent, DoneEvent)):
                yield event
        report = "".join(report_parts)
        await self.persistence.save_coder_report(item.job_id, item.position, report)
        if paused:
            return
        yield TextEvent(ui.coder_handoff(_handoff(report)))
        if plan is not None:
            await self._record_plan_events(plan, item, task.id, report, failure,
                                           contract.revision, bool(exclude_models))
        if await self._halted(item.job_id):
            yield TextEvent(ui.block(f"⏹ Job {ui.short_job(item.job_id)} was stopped; "
                                     f"{task.id} was not reviewed and no further tasks will run."))
            return
        if failure:
            async for event in self._continue_after_coder_failure(
                    item, messages, specs, instruction, direct_tool_handlers, failure,
                    exclude_models):
                yield event
        else:
            blocker = (handoff_blocker(report, task.id)
                       if getattr(self.task_execution, "strict_handoff", False) else None)
            if blocker:
                if blocker.startswith("Coder handoff invalid"):
                    async for event in self._continue_after_coder_failure(
                            item, messages, specs, instruction, direct_tool_handlers,
                            blocker, exclude_models):
                        yield event
                else:
                    yield TextEvent(await self._coder_failed(
                        item.job_id, item.position, blocker))
                return
            coder_model = await self.persistence.actual_model_for(decision_id)
            async for event in self._task_finished(
                    item.job_id, item.position, messages, specs, instruction,
                    direct_tool_handlers, report, coder_model):
                yield event

    async def _continue_after_coder_failure(
            self, item: ImplementationJobTask, messages: list[dict], specs: list[ToolSpec],
            instruction: str, direct_tool_handlers: dict, error: str,
            excluded: frozenset[str] = frozenset()) -> AsyncIterator[Event]:
        """Start one untried agent after verifying earlier writes are journaled."""
        item = await self.persistence.implementation_task(item.job_id, item.position)
        attempted_lookup = getattr(self.persistence, "attempted_models_for", None)
        attempted = (await attempted_lookup(item.decision_id)
                     if attempted_lookup else frozenset())
        failed_model = await self.persistence.actual_model_for(item.decision_id)
        attempted = frozenset(excluded | attempted |
                              ({failed_model} if failed_model else set()))
        _, task = await self.artifacts.task_by_id(item.task_id)
        remaining = [name for name in self.config.candidates_for(task.complexity)
                     if name not in attempted]
        if failed_model in self.config.models:
            provider = self.config.models[failed_model].provider
            remaining.sort(key=lambda name: self.config.models[name].provider == provider)
        if await self._halted(item.job_id):
            return
        if not remaining:
            yield TextEvent(await self._coder_failed(item.job_id, item.position, error))
            return
        journal = RollbackJournal(self.persistence, item.job_id, item.position,
                                  specs, direct_tool_handlers)
        manifest = await journal.review_manifest()
        paths = [entry["path"] for entry in manifest]
        if not await self.persistence.claim_auto_coder_fallback(
                item.job_id, item.position, item.decision_id):
            return
        await self.persistence.record_resolution_action(
            item.job_id, item.position, "coder_auto_fallback", {
                "failed_model": failed_model, "error": error,
                "next_candidates": remaining, "changed_paths": paths,
            })
        yield TextEvent(ui.progress_line(
            "note", f"{failed_model or 'Coder'} stopped; continuing {item.task_id} "
                    f"with {remaining[0]} from the saved project state"))
        resume_instruction = "\n".join(filter(None, (
            instruction,
            f"The previous coder stopped: {error}. Inspect the current project before "
            "editing. Keep valid changes already made and finish the remaining acceptance "
            "criteria; do not replay completed tool calls.",
            f"Journaled changed paths: {', '.join(paths) if paths else 'none'}.",
        )))
        async for event in self._run_item(
                item, messages, specs, resume_instruction, direct_tool_handlers,
                exclude_models=attempted):
            yield event

    async def _task_finished(self, job_id: str, position: int,
                             messages: list[dict], specs: list[ToolSpec],
                             instruction: str, direct_tool_handlers: dict,
                             report: str, coder_model: str | None) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None:
            return
        change, package = await self._package_for_task(item.task_id)
        selectors = await self.persistence.implementation_job_selectors(job_id)
        wp_boundary = package.id in selectors or bool(
            getattr(change, "epic", None) and change.epic.id in selectors)
        if not wp_boundary:
            async for event in self._review_and_advance(
                    job_id, position, messages, specs, instruction,
                    direct_tool_handlers, report, coder_model, wp_boundary=False):
                yield event
            return

        await self.persistence.update_implementation_task(job_id, position, "implemented")
        rows = await self._package_rows(job_id, package.id)
        if any(row.status == "queued" for row in rows):
            yield TextEvent(ui.block(
                f"⏳ {item.task_id} implemented; review deferred until {package.id} is assembled."))
            async for event in self._advance(
                    job_id, messages, specs, instruction, direct_tool_handlers):
                yield event
            return
        async for event in self._review_and_advance(
                job_id, position, messages, specs, instruction,
                direct_tool_handlers, report, coder_model):
            yield event

    async def _package_rows(self, job_id: str, package_id: str):
        rows = await self.persistence.implementation_job_tasks(job_id)
        selected = []
        for row in rows:
            try:
                _, package = await self._package_for_task(row.task_id)
            except Exception:
                continue
            if package.id == package_id:
                selected.append(row)
        return selected

    async def _package_for_task(self, task_id: str):
        try:
            return await self.artifacts.package_for_task(task_id)
        except (AttributeError, NotImplementedError):
            change, task = await self.artifacts.task_by_id(task_id)
            packages = getattr(change, "work_packages", None)
            if packages:
                package = next(package for package in packages
                               if any(candidate.id == task_id for candidate in package.tasks))
            else:
                package_id = task_id.split("-T", 1)[0]
                known_tasks = getattr(self.artifacts, "tasks", {})
                tasks = [candidate for candidate in known_tasks.values()
                         if candidate.id.startswith(package_id + "-T")] or [task]
                package = WorkPackage(
                    id=package_id, title=package_id, kind="legacy",
                    objective=getattr(change, "title", package_id),
                    acceptance_criteria=list(task.acceptance_criteria),
                    requirement_refs=[], tasks=tasks,
                    validation=ValidationContract(),
                )
            return change, package

    async def _record_wp_review(self, package, review: WorkPackageReview,
                                completed: bool) -> None:
        try:
            await self.artifacts.record_work_package_review(
                package.id, review, completed)
        except (AttributeError, NotImplementedError):
            if completed:
                for task in package.tasks:
                    try:
                        await self.artifacts.mark_task_completed(task.id)
                    except TypeError:
                        await self.artifacts.mark_task_completed(task.id, None)

    async def _review_and_advance(self, job_id: str, position: int,
                                  messages: list[dict], specs: list[ToolSpec],
                                  instruction: str, direct_tool_handlers: dict,
                                  report: str, coder_model: str | None,
                                  allow_confirmation_retry: bool = False,
                                  wp_boundary: bool | None = None,
                                  ) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or not await self.persistence.claim_review(
                job_id, position, allow_confirmation_retry):
            return
        change, original_task = await self.artifacts.task_by_id(item.task_id)
        _, package = await self._package_for_task(item.task_id)
        if wp_boundary is None:
            selectors = await self.persistence.implementation_job_selectors(job_id)
            wp_boundary = package.id in selectors or bool(
                getattr(change, "epic", None) and change.epic.id in selectors)
        if not wp_boundary:
            task = original_task
            package_rows = [item]
            yield TextEvent(ui.block(f"🔍 _Reviewing {task.id}…_"))
        else:
            package_rows = await self._package_rows(job_id, package.id)
            reports = [await self.persistence.coder_report_for(job_id, row.position)
                       for row in package_rows]
            validation = package.validation
            task = Task(
                id=package.id,
                title=package.title,
                description=package.objective,
                complexity=max(
                    (candidate.complexity for candidate in package.tasks),
                    key={"simple": 0, "medium": 1, "complex": 2}.get,
                    default=original_task.complexity,
                ),
                reasoning="WP-level assembled feature review",
                context="\n".join(filter(None, [
                    "BEGIN WORK PACKAGE REVIEW CONTRACT",
                    f"Required validation: {json.dumps(validation.required)}",
                    f"Preferred validation: {json.dumps(validation.preferred)}",
                    f"Optional validation: {json.dumps(validation.optional)}",
                    f"Explicitly not required: {json.dumps(validation.not_required)}",
                    "The reviewer must not invent requirements outside this contract.",
                    "END WORK PACKAGE REVIEW CONTRACT",
                ])),
                preferred_capability="write",
                acceptance_criteria=list(package.acceptance_criteria),
            )
            report = "\n\n".join(
                f"Task {row.task_id} coder handoff:\n{task_report}"
                for row, task_report in zip(package_rows, reports)
            )
        contracts = [await self.contract_persistence.latest(job_id, row.position)
                     for row in package_rows]
        if any(contract and contract.excluded_files for contract in contracts):
            specs = [spec for spec in specs if not spec.name.startswith("kb_")]
            direct_tool_handlers = {name: handler for name, handler in
                                    direct_tool_handlers.items()
                                    if not name.startswith("kb_")}
        if wp_boundary:
            yield TextEvent(ui.block(
                f"🔍 _Reviewing work package {package.id} against its "
                "required/preferred/optional contract…_"))
        coder_model = coder_model or await self.persistence.actual_model_for(item.decision_id)
        try:
            if not coder_model:
                raise ReviewError("actual coder model was not recorded")
            journal = RollbackJournal(
                self.persistence, job_id, position, specs, direct_tool_handlers)
            integrity_issues = await journal.integrity_issues()
            if integrity_issues:
                # Not blocking: the reviewer judges the project as it is, told which
                # changes the journal could not verify.
                details = "; ".join(
                    str(issue.get("path") or issue.get("tool") or issue["action"])
                    + (f" ({issue['error']})" if issue.get("error") else "")
                    for issue in integrity_issues)
                yield TextEvent(ui.block(
                    f"⚠️ {task.id}: some changes are not fully recorded in the change "
                    f"journal ({details}); the reviewer is told."))
                report = (f"{report}\n\nCodePartner journal warning: these changes could "
                          f"not be fully verified in the change journal: {details}.")
            journal_manifest = await journal.review_manifest()
            changed_paths = [entry["path"] for entry in journal_manifest]
            if wp_boundary and _requires_test_run(package):
                async for event in self._ensure_required_test_run(
                        job_id, package_rows, journal, specs, direct_tool_handlers):
                    yield event
            execution_evidence = await self.persistence.verification_evidence(
                job_id, position)
            dispute_errors = await self._capture_execution_dispute(
                journal, report, direct_tool_handlers, execution_evidence)
            execution_evidence = await self.persistence.verification_evidence(
                job_id, position)
            handoff_errors = self._handoff_evidence_errors(report, execution_evidence)
            review_task = task
            extra_context = []
            if changed_paths:
                extra_context.append("\n".join((
                    "Authoritative task mutation journal (all paths are required review evidence):",
                    *[f"Required mutation artifact: {path}" for path in changed_paths],
                )))
            if execution_evidence:
                safe_evidence = [{
                    "evidence_id": item["evidence_id"],
                    "tool": item["tool"],
                    "arguments": item["arguments"],
                    "success": item["success"],
                    "sha256": item["sha256"],
                    "result_excerpt": item["result_excerpt"],
                    "metadata": item["metadata"],
                } for item in execution_evidence]
                extra_context.append("\n".join((
                    "BEGIN CODEPARTNER EXECUTION EVIDENCE",
                    json.dumps(safe_evidence, ensure_ascii=False),
                    "END CODEPARTNER EXECUTION EVIDENCE",
                )))
            if dispute_errors:
                extra_context.append(
                    "Invalid coder execution dispute: " + "; ".join(dispute_errors))
            if handoff_errors:
                extra_context.append(
                    "Unverified coder handoff claims: " + "; ".join(handoff_errors))
            if extra_context:
                review_task = replace(
                    task, context="\n".join(filter(None, [task.context, *extra_context])))
            # Keep authoritative metadata out of task text: OpenSpec context is
            # untrusted and can mention files, but cannot fabricate journal records.
            object.__setattr__(review_task, "_review_journal_manifest", journal_manifest)
            preimages = getattr(self.persistence, "rollback_preimages", None)
            if preimages is not None:
                # The reviewer sees each change as a diff against its first preimage.
                object.__setattr__(review_task, "_review_preimages", await preimages(
                    job_id, [row.position for row in package_rows]))
            plan = await self._approved_plan(job_id)
            if plan is not None:
                reviewed = None if wp_boundary else [item.task_id]
                plan = await self._fitted_plan(plan, job_id, reviewed)
                events = await self.persistence.reviewer_context_events(plan.plan_id, job_id)
                planned_files = [*plan.reviewer.relevant_files, *plan.shared.relevant_files,
                                 *(path for task_id, context in plan.tasks.items()
                                   if reviewed is None or task_id in reviewed
                                   for path in (*context.relevant_files, *context.edit_files(),
                                                *context.reference_files()))]
                object.__setattr__(review_task, "_planner_review_context",
                                   reviewer_context(plan, events, reviewed, self._margin()))
                object.__setattr__(review_task, "_planner_edit_ranges", tuple(
                    edit for task_id, context in plan.tasks.items()
                    if reviewed is None or task_id in reviewed for edit in context.edits))
                object.__setattr__(review_task, "_planner_relevant_files",
                                   tuple(dict.fromkeys(planned_files)))
            if isinstance(self.reviewer, ReviewService):
                progress_queue: asyncio.Queue[str | None] = asyncio.Queue()
                outcome: dict = {}

                async def review_progress(message: str) -> None:
                    await progress_queue.put(message)

                async def run_review() -> None:
                    try:
                        outcome["decision"] = await self.reviewer.review(
                            change, review_task, report, coder_model, specs,
                            direct_tool_handlers, progress=review_progress,
                            plan_id=plan.plan_id if plan else None, job_id=job_id)
                    except Exception as exc:
                        outcome["error"] = exc
                    finally:
                        await progress_queue.put(None)

                review_run = asyncio.create_task(run_review())
                try:
                    while (message := await progress_queue.get()) is not None:
                        yield ProgressEvent(ui.progress_line("tool", message))
                    await review_run
                finally:
                    if not review_run.done():
                        review_run.cancel()
                        try:
                            await review_run
                        except asyncio.CancelledError:
                            pass
                if "error" in outcome:
                    raise outcome["error"]
                decision = outcome["decision"]
            else:
                decision = await self.reviewer.review(
                    change, review_task, report, coder_model, specs, direct_tool_handlers)
            review = CompletionReview(
                coder=model_identity(self.config, coder_model),
                reviewer=model_identity(self.config, decision.reviewer_model, review=True),
                reason=decision.rationale,
                evidence=list(decision.evidence),
            )
        except ReviewError as exc:
            await self.persistence.update_implementation_task(
                job_id, position, "review_pending", error=str(exc))
            try:
                coder_label = model_identity(self.config, coder_model).label if coder_model else "unknown"
            except ReviewError:
                coder_label = coder_model or "unknown"
            if exc.reviewer_model:
                try:
                    reviewer_label = model_identity(
                        self.config, exc.reviewer_model, review=True).label
                except ReviewError:
                    reviewer_label = exc.reviewer_model
                reviewer_status = f"{reviewer_label} (ran, verification incomplete)"
            else:
                reviewer_status = "not run"
            if exc.reviewer_model and coder_model:
                owners = {finding.owner for finding in exc.findings}
                await self.persistence.record_review(
                    job_id, position, coder_model, exc.reviewer_model,
                    "review_pending", str(exc), None, {
                        "run_status": "ran_incomplete",
                        "reviewer_routing_attempts": list(exc.routing_attempts),
                        "findings_schema_version": 1,
                        "findings": [finding.as_dict() for finding in exc.findings],
                        "retry_requires_evidence_change": owners == {"evidence_service"},
                        "evidence_fingerprint": await self.persistence.evidence_fingerprint(
                            job_id, position),
                    })
            if wp_boundary:
                await self._record_wp_review(
                    package,
                    WorkPackageReview(
                        outcome="UNVERIFIED", reason=str(exc),
                        evidence=[finding.problem for finding in exc.findings],
                        failure_type=("infrastructure_failure" if not exc.findings
                                      else "review_failure"),
                        reviewer=(model_identity(self.config, exc.reviewer_model, review=True)
                                  if exc.reviewer_model else None),
                        validation_results=[],
                    ),
                    completed=False,
                )
            title = (f"{package.id} — UNVERIFIED" if wp_boundary
                     else f"{task.id} — Review pending")
            lines = [ui.heading("⚠️", title,
                                f"coder: {coder_label} · reviewer: {reviewer_status}"),
                     ui.quote(f"Reason: {exc}.")]
            if exc.findings:
                lines.extend([ui.block("**Verification incomplete**"),
                              ui.findings_list(exc.findings)])
            owners = {finding.owner for finding in exc.findings}
            test_evidence = (await self.persistence.verification_evidence(job_id, position)
                             if wp_boundary and _requires_test_run(package) else None)
            if isinstance(exc, ReviewFormatError):
                lines.append(
                    "The reviewer's answer did not match the required JSON format, even after "
                    "one correction request. The project evidence is not the problem; reply "
                    "`retry review` to run the reviewer again.")
            elif test_evidence is not None and not has_passing_test_run(test_evidence):
                lines.extend(self._test_blocker_lines(test_evidence))
            elif owners == {"evidence_service"}:
                lines.append(
                    "CodePartner exhausted its safe read-only discovery and evidence fallbacks. "
                    "Repair the reported internal reader/evidence condition or restore the "
                    "required artifact before using `retry review`.")
            else:
                lines.append(
                    "The reviewer exhausted the safe evidence available to this request. "
                    "Retry only after the reported condition or project evidence has changed.")
            yield TextEvent("".join(ui.block(line) for line in lines if line))
            return
        verdict = ("ask_user" if task.preferred_capability.lower() in {"research", "verification", "review"}
                   and decision.verdict == "done" else decision.verdict)
        if verdict == "ask_user":
            if not wp_boundary:
                await self.persistence.record_review(
                    job_id, position, coder_model, decision.reviewer_model,
                    verdict, decision.rationale, decision.question, {
                        "completion_review": asdict(review),
                        "evidence": list(decision.evidence),
                        "findings_schema_version": 1,
                        "findings": [finding.as_dict() for finding in decision.findings],
                        "unresolved_references": list(decision.unresolved_references),
                        "reviewer_routing_reason": decision.routing_reason,
                        "reviewer_routing_attempts": list(decision.routing_attempts),
                        "reviewer_gateway_used": decision.gateway_used,
                    })
                await self.persistence.update_implementation_task(
                    job_id, position, "awaiting_user_decision")
                yield TextEvent(self._decision_text(task.id, verdict, decision, review))
                question = decision.question or f"Do you consider {task.id} complete?"
                yield TextEvent(ui.block(f"❓ **Reviewer question:** {question}"))
                yield TextEvent(self._resolution_options(task.id, verdict, job_id))
                return
            await self.persistence.record_review(
                job_id, position, coder_model, decision.reviewer_model,
                "unverified", decision.rationale, decision.question, {
                    "evidence": list(decision.evidence),
                    "findings_schema_version": 1,
                    "findings": [finding.as_dict() for finding in decision.findings],
                    "unresolved_references": list(decision.unresolved_references),
                    "reviewer_routing_reason": decision.routing_reason,
                    "reviewer_routing_attempts": list(decision.routing_attempts),
                    "reviewer_gateway_used": decision.gateway_used,
                })
            await self.persistence.update_implementation_task(
                job_id, position, "review_pending", error=decision.rationale)
            await self._record_wp_review(
                package,
                WorkPackageReview(
                    outcome="UNVERIFIED", reason=decision.rationale,
                    evidence=list(decision.evidence), failure_type="review_failure",
                    reviewer=model_identity(
                        self.config, decision.reviewer_model, review=True),
                    validation_results=list(decision.validation_results),
                ),
                completed=False,
            )
            yield TextEvent("".join((
                ui.heading("⚠️", f"{package.id} — UNVERIFIED",
                           "the reviewer could not obtain enough evidence"),
                ui.block("This is a review/validation problem, not an implementation failure. "
                         "Use `retry review` after the evidence or infrastructure condition "
                         "changes."),
            )))
            return
        await self.persistence.record_review(
            job_id, position, coder_model, decision.reviewer_model,
            verdict, decision.rationale, decision.question, {
                "completion_review": asdict(review),
                "evidence": list(decision.evidence),
                "inspected_paths": list(decision.inspected_paths),
                "gaps": list(decision.gaps),
                "feedback": decision.feedback,
                "findings_schema_version": 1,
                "findings": [finding.as_dict() for finding in decision.findings],
                "unresolved_references": list(decision.unresolved_references),
                "reviewer_routing_reason": decision.routing_reason,
                "reviewer_routing_attempts": list(decision.routing_attempts),
                "reviewer_gateway_used": decision.gateway_used,
            })
        log.info("implementation_review job_id=%s task_id=%s coder_model=%s "
                 "reviewer_model=%s verdict=%s", job_id, task.id, coder_model,
                 decision.reviewer_model, verdict)
        yield TextEvent(self._decision_text(task.id, verdict, decision, review))
        if verdict == "needs_work":
            if wp_boundary:
                await self._record_wp_review(
                    package,
                    WorkPackageReview(
                        outcome="BLOCK", reason=decision.rationale,
                        evidence=list(decision.evidence),
                        failure_type="implementation_failure",
                        reviewer=model_identity(
                            self.config, decision.reviewer_model, review=True),
                        validation_results=list(decision.validation_results),
                    ),
                    completed=False,
                )
            implementation_findings = tuple(
                finding for finding in decision.findings
                if finding.owner == "coder")
            feedback = decision.feedback or decision.rationale
            if not await self.persistence.consume_revision(job_id, position):
                await self.persistence.update_implementation_task(
                    job_id, position, "awaiting_user_decision", error=feedback)
                yield TextEvent(self._resolution_options(task.id, verdict, job_id))
                return
            attempts, budget = await self.persistence.revision_state(job_id, position)
            yield TextEvent(ui.block(f"🔁 Revising {task.id}: attempt {attempts}/{budget}."))
            revised_instruction = "\n".join(filter(None, [
                instruction, self._revision_context(task.id, implementation_findings)
            ]))
            async for event in self._run_item(item, messages, specs,
                                              revised_instruction, direct_tool_handlers):
                yield event
            return
        if not wp_boundary:
            await self.persistence.update_implementation_task(
                job_id, position, "awaiting_user_decision")
            yield TextEvent(self._resolution_options(task.id, verdict, job_id))
            return
        await self._record_wp_review(
            package,
            WorkPackageReview(
                outcome="PASS", reason=decision.rationale,
                evidence=list(decision.evidence),
                reviewer=model_identity(self.config, decision.reviewer_model, review=True),
                validation_results=list(decision.validation_results),
            ),
            completed=True,
        )
        if self.kb_refresh is not None:
            try:
                refresh_message = await self.kb_refresh(job_id, package.id)
                if refresh_message:
                    yield TextEvent(ui.block(refresh_message))
            except Exception as exc:
                log.exception("kb_refresh_hook_failed job_id=%s wp_id=%s", job_id, package.id)
                yield TextEvent(ui.block(
                    f"KB refresh failed for {package.id}: {exc}. The accepted review remains valid."))
        for row in package_rows:
            await self.persistence.update_implementation_task(
                job_id, row.position, "completed")
        yield TextEvent(ui.block(f"✅ **{package.id} — PASS.** The completed WP satisfies its "
                                 "OpenSpec contract."))
        async for event in self._advance(
                job_id, messages, specs, instruction, direct_tool_handlers):
            yield event

    async def confirm(self, job_id: str, position: int, approved: bool,
                      feedback: str, messages: list[dict], specs: list[ToolSpec],
                      direct_tool_handlers: dict | None = None,
                      instruction: str = "") -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status not in {"awaiting_confirmation",
                                               "awaiting_human_confirmation"}:
            yield ErrorEvent("This task is no longer awaiting confirmation.")
            return
        if approved:
            latest = await self.persistence.latest_review(job_id, position)
            if latest is None:
                yield ErrorEvent("The review decision for this task was not found.")
                return
            details = latest["details"]
            review = (CompletionReview.from_dict(details["completion_review"])
                      if details and "completion_review" in details else CompletionReview(
                coder=model_identity(self.config, latest["coder_model"]),
                reviewer=model_identity(self.config, latest["reviewer_model"], review=True),
                reason=latest["rationale"], evidence=[],
            ))
            await self._complete(job_id, position, item.task_id, review)
            yield TextEvent(ui.block(f"{item.task_id} confirmed complete."))
            async for event in self._advance(job_id, messages, specs, instruction,
                                             direct_tool_handlers or {}):
                yield event
        else:
            yield TextEvent(ui.block(f"Continuing {item.task_id} with your feedback."))
            revised_instruction = "\n".join(filter(None, [instruction, feedback]))
            async for event in self._run_item(
                    item, messages, specs, revised_instruction, direct_tool_handlers or {}):
                yield event

    async def _ensure_required_test_run(
            self, job_id: str, package_rows: list[ImplementationJobTask],
            journal: RollbackJournal, specs: list[ToolSpec],
            direct_tool_handlers: dict) -> AsyncIterator[Event]:
        """Run the WP's required tests once when no captured run has passed.

        The reviewer is read-only, so a required "tests pass" item can only be met
        by CodePartner-captured execution. The run goes through the journaled
        ``execute_tool`` wrapper and is recorded as evidence for the reviewer.
        """
        evidence = []
        for row in package_rows:
            evidence.extend(await self.persistence.verification_evidence(job_id, row.position))
        if has_passing_test_run(evidence):
            return
        handlers = journal.wrapped_handlers()
        dispatcher = next((name for name in handlers
                           if name.rsplit("__", 1)[-1] == "execute_tool"), None)
        project = native_project()
        if dispatcher is None and project is None:
            yield TextEvent(ui.block("The required tests have no passing run, and this IDE exposes no "
                            "`execute_tool` for CodePartner to run them."))
            return
        command = next((found for item in reversed(evidence)
                        if (found := shell_test_command(
                            (item.get("arguments") or {}).get("command")))), None)
        if command is None:
            test_paths: list[str] = []
            for row in package_rows:
                row_journal = RollbackJournal(
                    self.persistence, job_id, row.position, specs, direct_tool_handlers)
                test_paths.extend(path for path in await row_journal.changed_paths()
                                  if _is_test_path(path) and path not in test_paths)
            if not test_paths:
                yield TextEvent(ui.block("The required tests have no passing run, and no test command "
                                "or changed test file was found to run."))
                return
            command = "pytest " + " ".join(shlex.quote(path) for path in test_paths)
        command = await self._project_test_command(journal, command, project)
        yield TextEvent(ui.block(f"Running the required tests through CodePartner: `{command}`"))
        if project is not None:
            output, exit_code = await run_native_command(project, command)
            result = command_result(output, exit_code)
            await journal.record_verification(
                NATIVE_SHELL_TOOL, {"command": command}, result,
                exit_code == 0 if exit_code is not None else verification_succeeded(result))
            return
        try:
            await handlers[dispatcher]({"command": command})
        except Exception as exc:
            log.warning("required_test_run_failed job_id=%s error=%s", job_id, exc)

    @staticmethod
    def _test_blocker_lines(evidence: list[dict]) -> list[str]:
        """Explain a required-tests blocker with the actual command and its outcome."""
        if not evidence:
            return ["No test run was captured for this work package.",
                    "Reply `retry review`: CodePartner will run the required tests itself "
                    "and the reviewer will judge their output."]
        last = evidence[-1]
        command = (last.get("arguments") or {}).get("command", "")
        pytest = (last.get("metadata") or {}).get("pytest") or {}
        excerpt = " ".join(str(last.get("result_excerpt") or "").split())[:300]
        if pytest.get("totals"):
            totals = ", ".join(f"{count} {label}" for label, count in pytest["totals"].items())
            return [f"The required tests ran and did not pass ({totals}).",
                    f"Command: `{command}`",
                    "Fix the failing tests, then reply `retry review`."]
        return [f"The last test run never started pytest: `{command}`",
                f"Error: {excerpt}",
                "Reply `retry review`: CodePartner will run the required tests itself "
                "and the reviewer will judge their output."]

    @staticmethod
    async def _project_test_command(journal: RollbackJournal, command: str,
                                    project=None) -> str:
        """Run a bare pytest through the project's environment rather than PATH."""
        if not re.match(r"(?:pytest|py\.test)(?:\s|$)", command):
            return command
        if project is not None:
            # A host .venv is unusable in the container; uv builds its own environment.
            root = Path(project.workspace)
            if (root / "uv.lock").exists() or (root / "pyproject.toml").exists():
                return f"uv run {command}"
            return f"python3 -m {command}"
        if (await journal._read("uv.lock"))[0]:
            return f"uv run {command}"
        if (await journal._read(".venv/pyvenv.cfg"))[0]:
            return f".venv/bin/python -m {command}"
        return f"python -m {command}"

    async def retry_review(self, job_id: str, position: int,
                           messages: list[dict], specs: list[ToolSpec],
                           instruction: str, direct_tool_handlers: dict
                           ) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status not in {
                "review_pending", "awaiting_evidence_resolution", "reviewing",
                "awaiting_confirmation", "awaiting_human_confirmation"}:
            yield ErrorEvent("This task is no longer pending review.")
            return
        latest = await self.persistence.latest_review(job_id, position)
        details = latest.get("details") if latest else None
        _, package = await self._package_for_task(item.task_id)
        acquires_tests = (_requires_test_run(package) and not has_passing_test_run(
            await self.persistence.verification_evidence(job_id, position)))
        if (isinstance(details, dict) and not acquires_tests
                and details.get("retry_requires_evidence_change")
                and details.get("evidence_fingerprint")
                == await self.persistence.evidence_fingerprint(job_id, position)):
            yield ErrorEvent(
                "Review evidence is unchanged. CodePartner will not repeat the same "
                "incomplete review; acquire the missing evidence first.")
            return
        report = await self.persistence.coder_report_for(job_id, position)
        coder_model = await self.persistence.actual_model_for(item.decision_id)
        async for event in self._review_and_advance(
                job_id, position, messages, specs, instruction,
                direct_tool_handlers, report, coder_model,
                allow_confirmation_retry=True):
            yield event

    async def _record_contract_escalation(self, job_id: str, position: int,
                                          contract, reason: str,
                                          exhausted_limit: str) -> None:
        await self.contract_persistence.save_escalation(job_id, position, {
            "task_id": contract.task_id, "contract_id": contract.contract_id,
            "reason": reason, "exhausted_limit": exhausted_limit,
        })

    async def retry_contract(self, job_id: str, position: int,
                             messages: list[dict], specs: list[ToolSpec],
                             instruction: str, direct_tool_handlers: dict,
                             guidance: str, add_files: tuple[str, ...] = ()
                             ) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status != "awaiting_contract_resolution":
            yield ErrorEvent("This task is no longer awaiting contract resolution.")
            return
        previous = await self.contract_persistence.latest(job_id, position)
        if previous is None:
            yield ErrorEvent("The previous TaskContract was not found.")
            return
        revised = self.contractor.revise_after_escalation(
            previous, guidance, specs, add_files,
            requested_paths(item.error or ""))
        await self.contract_persistence.save(job_id, position, revised)
        scope_decision = revised.known_context[-1].attributes.get("scope_decision")
        await self.persistence.record_resolution_action(
            job_id, position, "retry_contract", {
                "guidance": guidance, "scope_decision": scope_decision,
                "contract_id": revised.contract_id,
            })
        yield TextEvent(ui.block(f"TaskContract revision {revised.revision} confirmed for {item.task_id}."))
        async for event in self._run_item(
                item, messages, specs, "\n".join(filter(None, (instruction, guidance))),
                direct_tool_handlers):
            yield event

    async def stop_job(self, job_id: str, position: int,
                       expected_status: str) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status != expected_status:
            yield ErrorEvent("The job state changed; it was not stopped.")
            return
        await self.persistence.update_implementation_task(job_id, position, "stopped")
        await self.persistence.finish_implementation_job(job_id, "stopped")
        yield TextEvent(ui.block(f"Stopped job {job_id} at {item.task_id}. Files were left as they "
                        "are; nothing was rolled back."))

    @staticmethod
    def _stored_findings(latest: dict | None) -> tuple[ReviewFinding, ...]:
        details = latest.get("details") if latest else None
        raw_findings = details.get("findings", []) if isinstance(details, dict) else []
        findings = []
        for raw in raw_findings:
            if not isinstance(raw, dict) or raw.get("kind") not in {
                    "implementation", "verification"}:
                continue
            owner, reason = _finding_owner(
                raw["kind"], str(raw.get("problem", "")))
            findings.append(ReviewFinding(
                raw["kind"], str(raw.get("problem", "")),
                str(raw.get("suggested_task", "")),
                tuple(str(item) for item in raw.get("approval_criteria", [])),
                tuple(str(path) for path in raw.get("artifact_paths", [])),
                str(raw.get("owner", owner)),
                str(raw.get("reason", reason)),
            ))
        return tuple(findings)

    async def retry_coder(self, job_id: str, position: int,
                          messages: list[dict], specs: list[ToolSpec],
                          instruction: str, direct_tool_handlers: dict,
                          guidance: str = "") -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is not None and item.status == CODER_RECOVERY_STATUS:
            async for event in self._retry_interrupted_coder(
                    item, messages, specs, instruction, direct_tool_handlers, guidance):
                yield event
            return
        latest = await self.persistence.latest_review(job_id, position)
        if item is None or latest is None or latest["verdict"] not in {
                "done", "needs_work", "ask_user"}:
            yield ErrorEvent("This task has no completed review to submit again.")
            return
        if not guidance.strip():
            yield ErrorEvent(
                "Submit to coder again requires an additional prompt after the colon.")
            return
        findings = tuple(finding for finding in self._stored_findings(latest)
                         if finding.kind == "implementation")
        if not await self.persistence.restart_revision_cycle(
                job_id, position, REVISION_BUDGET):
            yield ErrorEvent("This task is no longer awaiting a review decision.")
            return
        await self.persistence.record_resolution_action(
            job_id, position, "submit_again", {
                "revision_budget": REVISION_BUDGET,
                "guidance": guidance.strip(),
                "reviewer_model": latest["reviewer_model"],
            })
        yield TextEvent(ui.block(
            f"Submitting {item.task_id} to the coder again with a fresh "
            f"{REVISION_BUDGET}-revision review cycle."))
        revised_instruction = "\n".join(filter(None, [
            instruction, guidance, self._revision_context(item.task_id, findings),
        ]))
        async for event in self._run_item(
                item, messages, specs, revised_instruction, direct_tool_handlers):
            yield event

    async def _retry_interrupted_coder(self, item: ImplementationJobTask,
                                       messages: list[dict], specs: list[ToolSpec],
                                       instruction: str, direct_tool_handlers: dict,
                                       guidance: str) -> AsyncIterator[Event]:
        failed_model = await self.persistence.actual_model_for(item.decision_id)
        attempted_lookup = getattr(self.persistence, "attempted_models_for", None)
        attempted = (await attempted_lookup(item.decision_id)
                     if attempted_lookup else frozenset())
        attempted = frozenset(attempted | ({failed_model} if failed_model else set()))
        candidates = self.config.candidates_for(
            (await self.artifacts.task_by_id(item.task_id))[1].complexity)
        if (item.error or "").startswith(("Coder handoff blocked", "Coder handoff invalid")):
            # A repaired IDE tool policy makes the same direct Codex model usable.
            attempted = (frozenset(name for name in candidates if name != failed_model)
                         if failed_model in candidates else frozenset())
        if not any(name not in attempted for name in candidates):
            # An explicit retry is a fresh attempt even when all configured
            # models have previously failed; prefer the other provider.
            attempted = (frozenset({failed_model})
                         if any(name != failed_model for name in candidates)
                         else frozenset())
        journal = RollbackJournal(self.persistence, item.job_id, item.position,
                                  specs, direct_tool_handlers)
        manifest = await journal.review_manifest()
        if not await self.persistence.claim_coder_recovery(item.job_id, item.position):
            yield ErrorEvent("This task is no longer awaiting a coder recovery decision.")
            return
        await self.persistence.record_resolution_action(
            item.job_id, item.position, "coder_retry", {
                "failed_model": failed_model, "error": item.error,
                "guidance": guidance.strip(),
            })
        yield TextEvent(ui.block(
            f"Continuing {item.task_id} from the saved project state; "
            "files written so far are kept."))
        revised_instruction = "\n".join(filter(None, [
            instruction, guidance.strip(),
            "A previous coder attempt on this task was interrupted before finishing "
            f"({item.error}). Inspect the current state of the files it touched and "
            "continue from there rather than starting over. Do not replay completed tool calls.",
            "Journaled changed paths: " + ", ".join(
                entry["path"] for entry in manifest),
        ]))
        async for event in self._run_item(
                item, messages, specs, revised_instruction, direct_tool_handlers,
                exclude_models=attempted):
            yield event

    async def review_blocked_handoff(self, job_id: str, position: int,
                                     messages: list[dict], specs: list[ToolSpec],
                                     instruction: str, direct_tool_handlers: dict,
                                     guidance: str = "") -> AsyncIterator[Event]:
        """Send a complete handoff that reported gaps to the Reviewer as it stands."""
        item = await self.persistence.implementation_task(job_id, position)
        report = await self.persistence.coder_report_for(job_id, position) if item else ""
        blocker = handoff_blocker(report or "", item.task_id) if item else None
        if (item is None or item.status != CODER_RECOVERY_STATUS
                or not (item.error or "").startswith("Coder handoff blocked")
                or (blocker is not None and not blocker.startswith("Coder handoff blocked"))):
            yield ErrorEvent("This task has no blocked coder handoff to review; "
                             "only `try coder again` applies.")
            return
        if not await self.persistence.claim_coder_recovery(job_id, position):
            yield ErrorEvent("This task is no longer awaiting a coder recovery decision.")
            return
        handoff = _handoff(report)
        await self.persistence.record_resolution_action(
            job_id, position, "review_blocked_handoff", {
                "error": item.error,
                "remaining_gaps": handoff.get("remaining_gaps") or [],
                "guidance": guidance.strip(),
            })
        yield TextEvent(ui.block(
            f"Sending {item.task_id}'s coder report to the Reviewer with its reported gaps."))
        coder_model = await self.persistence.actual_model_for(item.decision_id)
        async for event in self._task_finished(
                job_id, position, messages, specs, instruction,
                direct_tool_handlers, report, coder_model):
            yield event

    async def accept_anyway(self, job_id: str, position: int,
                            messages: list[dict], specs: list[ToolSpec],
                            instruction: str, direct_tool_handlers: dict,
                            reason: str = "") -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        latest = await self.persistence.latest_review(job_id, position)
        if item is None or latest is None or latest["verdict"] not in {
                "done", "needs_work", "ask_user"}:
            yield ErrorEvent("This task has no completed review to accept.")
            return
        override = latest["verdict"] == "needs_work"
        action = "accept_override" if override else "accept"
        if not await self.persistence.claim_resolution(job_id, position, action):
            yield ErrorEvent("This task is no longer awaiting a review decision.")
            return
        details = latest.get("details") or {}
        if "completion_review" in details:
            prior = CompletionReview.from_dict(details["completion_review"])
        else:
            prior = CompletionReview(
                coder=model_identity(self.config, latest["coder_model"]),
                reviewer=model_identity(self.config, latest["reviewer_model"], review=True),
                reason=latest["rationale"], evidence=[],
            )
        if override:
            explanation = (reason.strip()
                           or "User accepted the implementation despite reviewer findings.")
            review = CompletionReview(
                coder=prior.coder,
                reviewer=prior.reviewer,
                reason=(f"User override: {explanation} Reviewer verdict was needs_work: "
                        f"{latest['rationale']}"),
                evidence=[*prior.evidence,
                          "Completion was explicitly accepted by the user as an override."],
            )
        else:
            review = prior
        await self.persistence.record_resolution_action(
            job_id, position, action, {
                "reason": reason.strip() or None,
                "reviewer_model": latest["reviewer_model"],
                "reviewer_verdict": latest["verdict"],
            })
        await self._complete(job_id, position, item.task_id, review)
        message = (f"{item.task_id} accepted by user override. "
                   "The reviewer verdict remains needs_work."
                   if override else f"{item.task_id} accepted after review.")
        yield TextEvent(ui.block(message))
        async for event in self._advance(
                job_id, messages, specs, instruction, direct_tool_handlers):
            yield event

    async def _complete(self, job_id: str, position: int, task_id: str,
                        review: CompletionReview) -> None:
        await self.artifacts.mark_task_completed(task_id, review)
        await self.persistence.update_implementation_task(job_id, position, "completed")

    async def _fail(self, job_id: str, position: int, error: str) -> None:
        await self.persistence.update_implementation_task(job_id, position, "failed", error=error)
        await self.persistence.finish_implementation_job(job_id, "failed", error)


def _handoff(report: str) -> dict:
    """The coder's final JSON handoff (last object with a summary), or {}."""
    decoder = json.JSONDecoder()
    found: dict = {}
    for index, char in enumerate(report):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(report[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "summary" in value:
            found = value
    return found

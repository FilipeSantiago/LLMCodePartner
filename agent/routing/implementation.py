"""Sequential, durable execution and outcome review of OpenSpec tasks."""

import json
import logging
import re
import shlex
from collections.abc import AsyncIterator
from dataclasses import asdict, replace

from agent.routing.config import load_routing_config
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
from mcp_bridge import bridge
from mcp_bridge.models import DoneEvent, ErrorEvent, Event, Run, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec


log = logging.getLogger("codepartner.implementation")
REVISION_BUDGET = 3
_REVIEWER_CONTEXT_MARKER = "BEGIN CODEPARTNER REVIEWER REVISION CONTEXT"
_REVIEWER_CONTEXT_END = "END CODEPARTNER REVIEWER REVISION CONTEXT"
# Coder failures where the model itself broke down mid-task. Files written so far
# stay in place, so the user chooses between another model and undo.
_RECOVERABLE_CODER_FAILURE = re.compile(r"tool call could not be parsed", re.IGNORECASE)
CODER_RECOVERY_STATUS = "awaiting_coder_recovery"
_REQUIRES_TESTS = re.compile(r"\b(?:tests?|pytest|unit[- ]tests?)\b", re.IGNORECASE)


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
                 contract_persistence: TaskContractPersistence | None = None):
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
        suffix = f" for job {job_id}" if job_id else ""
        status = ("The reviewer still reports blocking findings. "
                  if verdict == "needs_work" else "The review is ready. ")
        return "\n".join((
            f"{task_id}{suffix}: {status}Choose what to do next:",
            "1. Reply `accept` to keep the implementation and continue.",
            "2. Reply `undo` to restore only this task's changes and stop the job.",
            "3. Reply `submit to coder again: <additional prompt>` to keep the current "
            "work and start a fresh coder/reviewer cycle.",
        ))

    @staticmethod
    def _decision_text(task_id: str, verdict: str, decision: ReviewDecision,
                       review: CompletionReview) -> str:
        status = {"done": "Approved", "needs_work": "Needs work",
                  "ask_user": "Needs your confirmation"}[verdict]
        lines = [
            f"{task_id} — {status}",
            f"Coded by: {review.coder.label}",
            f"Reviewed by: {review.reviewer.label}",
            f"Reviewer decision: {decision.rationale}",
        ]
        if decision.inspected_paths:
            lines.extend(["Artifacts reviewed:",
                          *[f"- {path}" for path in decision.inspected_paths]])
        lines.extend(["Evidence and limitations:",
                      *[f"- {item}" for item in decision.evidence]])
        implementation = tuple(finding for finding in decision.findings
                               if finding.kind == "implementation")
        verification = tuple(finding for finding in decision.findings
                             if finding.kind == "verification")
        if implementation:
            lines.extend(["Implementation findings:",
                          *ImplementationService._findings_json(implementation)])
        if verification:
            lines.extend(["Verification incomplete:",
                          *ImplementationService._findings_json(verification)])
        return "\n".join(lines)

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
            if (not isinstance(tool, str) or tool not in direct_tool_handlers
                    or not isinstance(arguments, dict)
                    or not is_reproducible_verification_call(arguments)):
                errors.append(
                    f"execution_dispute step {number} is not an available verification call")
                continue
            signature = (tool, json.dumps(arguments, sort_keys=True))
            if signature in seen:
                continue
            try:
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
        return (f"{task_id} — Coding failed\nCoded by: {coder_label}\n"
                f"Reviewer: not run\nReason: {error}\nImplementation job {job_id} stopped.")

    async def _coder_failed(self, job_id: str, position: int, error: str) -> str:
        """Record a coder failure; recoverable ones wait for the user instead of stopping."""
        if not _RECOVERABLE_CODER_FAILURE.search(error):
            await self._fail(job_id, position, error)
            return await self._failed_text(job_id, position, error)
        await self.persistence.update_implementation_task(
            job_id, position, CODER_RECOVERY_STATUS, error=error)
        item = await self.persistence.implementation_task(job_id, position)
        coder_model = await self.persistence.actual_model_for(item.decision_id) if item else None
        try:
            coder_label = model_identity(self.config, coder_model).label if coder_model else "unknown"
        except ReviewError:
            coder_label = coder_model or "unknown"
        task_id = item.task_id if item else f"position {position}"
        return "\n".join((
            f"{task_id} — Coding interrupted",
            f"Coded by: {coder_label}",
            "Reviewer: not run (the coder did not finish)",
            f"Reason: {error}",
            "Files written before the interruption are still in the project.",
            f"{task_id} for job {job_id}: Choose what to do next:",
            "1. Reply `try coder again` (optionally `try coder again: <guidance>`) to continue "
            "this task on a different model, keeping the current files.",
            "2. Reply `undo` to restore only this task's changes and stop the job.",
        ))

    def _pending_options(self, item: ImplementationJobTask, latest: dict | None,
                         job_id: str) -> str:
        if item.status == CODER_RECOVERY_STATUS or latest is None:
            return (f"{item.task_id} for job {job_id}: reply `try coder again` to continue on a "
                    "different model, or `undo` to restore this task's changes.")
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
        yield TextEvent(
            f"Implementation job {job.job_id} queued {len(job.task_ids)} task(s): "
            f"{', '.join(job.task_ids)}."
        )
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
            if isinstance(event, DoneEvent):
                terminal = event
            else:
                yield event
        await self.persistence.save_coder_report(job_id, position, report)
        if paused:
            return
        if await self._halted(job_id):
            yield TextEvent(f"Job {job_id} was stopped; {run.metadata.get('implementation_task_id') or 'the task'} "
                            "was not reviewed and no further tasks will run.")
            if terminal is not None:
                yield terminal
            return
        if failed:
            yield TextEvent(await self._coder_failed(job_id, position, failed))
        else:
            async for event in self._task_finished(
                    job_id, position, messages, specs,
                    run.metadata.get("implementation_instruction", ""),
                    run.direct_tool_handlers, report, run.metadata.get("model")):
                yield event
        if terminal is not None:
            yield terminal

    async def finish_child(self, run: Run, messages: list[dict],
                           specs: list[ToolSpec]) -> AsyncIterator[Event]:
        job_id = run.metadata["implementation_job_id"]
        position = run.metadata["implementation_position"]
        if run.metadata.get("implementation_action") == "undo":
            async for event in self._complete_undo(job_id, position):
                yield event
            return
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
            yield TextEvent(f"Job {job_id} was stopped; no further tasks will run.")
            return
        item = await self.persistence.next_implementation_task(job_id)
        if item is None:
            await self.persistence.finish_implementation_job(job_id)
            yield TextEvent(f"Implementation job {job_id} completed.")
            return
        async for event in self._run_item(item, messages, specs, instruction,
                                          direct_tool_handlers or {}):
            yield event

    async def _run_item(self, item: ImplementationJobTask, messages: list[dict],
                        specs: list[ToolSpec], instruction: str,
                        direct_tool_handlers: dict,
                        exclude_models: frozenset[str] = frozenset()) -> AsyncIterator[Event]:
        change, task = await self.artifacts.task_by_id(item.task_id)
        contract = await self.contract_persistence.latest(item.job_id, item.position)
        if contract is None:
            contract = self.contractor.initial_contract(task, specs, instruction)
            await self.contract_persistence.save(item.job_id, item.position, contract)
        await self.persistence.update_implementation_task(item.job_id, item.position, "running")
        yield TextEvent(f"Starting {task.id} ({task.complexity}) from {change.id}.")
        failure: str | None = None
        paused = False
        report_parts: list[str] = []
        decision_id: str | None = None
        async def record_decision(decision) -> None:
            nonlocal decision_id
            decision_id = decision.decision_id
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
                }):
            if isinstance(event, TextEvent):
                report_parts.append(event.text)
            if isinstance(event, ToolCallEvent):
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
            if isinstance(event, TextEvent):
                yield TextEvent("Coder output (untrusted claim):\n" + event.text)
            elif not isinstance(event, DoneEvent):
                yield event
        report = "".join(report_parts)
        await self.persistence.save_coder_report(item.job_id, item.position, report)
        if paused:
            return
        if await self._halted(item.job_id):
            yield TextEvent(f"Job {item.job_id} was stopped; {task.id} was not reviewed "
                            "and no further tasks will run.")
            return
        if failure:
            yield TextEvent(await self._coder_failed(item.job_id, item.position, failure))
        else:
            coder_model = await self.persistence.actual_model_for(decision_id)
            async for event in self._task_finished(
                    item.job_id, item.position, messages, specs, instruction,
                    direct_tool_handlers, report, coder_model):
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
            yield TextEvent(
                f"{item.task_id} implemented; deferring review until {package.id} is assembled.")
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
            yield TextEvent(f"Reviewing explicitly selected task {task.id}.")
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
        yield TextEvent(f"Reviewing completed work package {package.id}.")
        if wp_boundary:
            yield TextEvent(
                f"Validating {package.id} using its required/preferred/optional contract.")
        coder_model = coder_model or await self.persistence.actual_model_for(item.decision_id)
        try:
            if not coder_model:
                raise ReviewError("actual coder model was not recorded")
            journal = RollbackJournal(
                self.persistence, job_id, position, specs, direct_tool_handlers)
            integrity_issues = await journal.integrity_issues()
            if integrity_issues:
                await self.persistence.update_implementation_task(
                    job_id, position, "recovery_required",
                    error="mutation journal integrity could not be verified")
                details = "; ".join(
                    str(issue.get("path") or issue.get("tool") or issue["action"])
                    for issue in integrity_issues)
                yield TextEvent("\n".join((
                    f"{task.id} — Recovery required",
                    "Reviewer: not run",
                    "A mutation was applied without a complete verified journal, so safe undo "
                    "and independent review cannot be guaranteed.",
                    f"Journal evidence: {details}.",
                    "Reply `inspect recovery` for the mutation audit, "
                    "`establish new baseline: <prompt>` to explicitly accept the current "
                    "project state as the new undo baseline and rerun the coder/reviewer "
                    "cycle, or `stop and repair manually` to end this job. "
                    "`retry review` is unavailable.",
                )))
                return
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
            heading = (f"{package.id} — UNVERIFIED" if wp_boundary
                       else f"{task.id} — Review pending")
            lines = [heading, f"Coded by: {coder_label}",
                     f"Reviewer: {reviewer_status}", f"Reason: {exc}."]
            if exc.findings:
                lines.extend(["Verification incomplete:",
                              *self._findings_json(exc.findings)])
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
            yield TextEvent("\n".join(lines))
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
                    })
                await self.persistence.update_implementation_task(
                    job_id, position, "awaiting_user_decision")
                yield TextEvent(self._decision_text(task.id, verdict, decision, review))
                question = decision.question or f"Do you consider {task.id} complete?"
                yield TextEvent(f"Reviewer question: {question}")
                yield TextEvent(self._resolution_options(task.id, verdict, job_id))
                return
            await self.persistence.record_review(
                job_id, position, coder_model, decision.reviewer_model,
                "unverified", decision.rationale, decision.question, {
                    "evidence": list(decision.evidence),
                    "findings_schema_version": 1,
                    "findings": [finding.as_dict() for finding in decision.findings],
                    "unresolved_references": list(decision.unresolved_references),
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
            yield TextEvent(
                f"{package.id} — UNVERIFIED. The reviewer could not obtain enough evidence. "
                "This is a review/validation problem, not an implementation failure. "
                "Use `retry review` after the evidence or infrastructure condition changes.")
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
            yield TextEvent(f"Revising {task.id}: attempt {attempts}/{budget}.")
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
        for row in package_rows:
            await self.persistence.update_implementation_task(
                job_id, row.position, "completed")
        yield TextEvent(f"{package.id} — PASS. The completed WP satisfies its OpenSpec contract.")
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
            yield TextEvent(f"{item.task_id} confirmed complete.")
            async for event in self._advance(job_id, messages, specs, instruction,
                                             direct_tool_handlers or {}):
                yield event
        else:
            yield TextEvent(f"Continuing {item.task_id} with your feedback.")
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
        if dispatcher is None:
            yield TextEvent("The required tests have no passing run, and this IDE exposes no "
                            "`execute_tool` for CodePartner to run them.")
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
                yield TextEvent("The required tests have no passing run, and no test command "
                                "or changed test file was found to run.")
                return
            command = "pytest " + " ".join(shlex.quote(path) for path in test_paths)
        command = await self._project_test_command(journal, command)
        yield TextEvent(f"Running the required tests through CodePartner: `{command}`")
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
    async def _project_test_command(journal: RollbackJournal, command: str) -> str:
        """Run a bare pytest through the project's environment rather than PATH."""
        if not re.match(r"(?:pytest|py\.test)(?:\s|$)", command):
            return command
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
        yield TextEvent(f"TaskContract revision {revised.revision} confirmed for {item.task_id}.")
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
        yield TextEvent(f"Stopped job {job_id} at {item.task_id}. Files were left as they "
                        "are; nothing was rolled back.")

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
        yield TextEvent(
            f"Submitting {item.task_id} to the coder again with a fresh "
            f"{REVISION_BUDGET}-revision review cycle.")
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
        if not await self.persistence.restart_revision_cycle(
                item.job_id, item.position, REVISION_BUDGET):
            yield ErrorEvent("This task is no longer awaiting a coder recovery decision.")
            return
        await self.persistence.record_resolution_action(
            item.job_id, item.position, "coder_retry", {
                "failed_model": failed_model, "error": item.error,
                "guidance": guidance.strip(),
            })
        yield TextEvent(
            f"Continuing {item.task_id} on a different model than "
            f"{failed_model or 'the interrupted coder'}; files written so far are kept.")
        revised_instruction = "\n".join(filter(None, [
            instruction, guidance.strip(),
            "A previous coder attempt on this task was interrupted before finishing "
            f"({item.error}). Inspect the current state of the files it touched and "
            "continue from there rather than starting over.",
        ]))
        async for event in self._run_item(
                item, messages, specs, revised_instruction, direct_tool_handlers,
                exclude_models=frozenset({failed_model}) if failed_model else frozenset()):
            yield event

    async def establish_new_baseline(self, job_id: str, position: int,
                                     messages: list[dict], specs: list[ToolSpec],
                                     instruction: str, direct_tool_handlers: dict,
                                     guidance: str) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status != "recovery_required":
            yield ErrorEvent("This task does not require journal recovery.")
            return
        if not guidance.strip():
            yield ErrorEvent(
                "Establish new baseline requires a recovery prompt after the colon.")
            return
        if not await self.persistence.establish_recovery_baseline(
                job_id, position, REVISION_BUDGET, guidance.strip()):
            yield ErrorEvent("This task is no longer awaiting journal recovery.")
            return
        yield TextEvent(
            f"Established the current project state as the new undo baseline for "
            f"{item.task_id}. Prior uncertain mutations cannot be undone automatically. "
            f"Starting a fresh coder/reviewer cycle.")
        recovery_instruction = "\n".join(filter(None, [
            instruction,
            guidance.strip(),
            "Recovery run: inspect and directly repair the named original artifacts. "
            "All new mutations must be journaled. Run every requested verification and "
            "report only captured results.",
        ]))
        async for event in self._run_item(
                item, messages, specs, recovery_instruction, direct_tool_handlers):
            yield event

    async def inspect_recovery(self, job_id: str, position: int) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status != "recovery_required":
            yield ErrorEvent("This task does not require journal recovery.")
            return
        mutations = await self.persistence.mutations(job_id, position)
        issues = await self.persistence.rollback_integrity_issues(job_id, position)
        lines = [
            f"{item.task_id} — Recovery audit",
            "Undo remains disabled until a new baseline is explicitly established.",
        ]
        if mutations:
            lines.append("Mutation lifecycle:")
            for mutation in mutations:
                targets = ", ".join(mutation["targets"]) or "unresolved targets"
                failure = (f"; failure={mutation['failure_stage']}: "
                           f"{mutation['failure_detail']}"
                           if mutation["failure_stage"] else "")
                lines.append(
                    f"- #{mutation['id']} {mutation['tool']} ({mutation['operation']}): "
                    f"{mutation['stage']} — {targets}{failure}")
        else:
            lines.append("Mutation lifecycle: no structured lifecycle records are available.")
        if issues:
            lines.append("Integrity issues:")
            lines.extend(
                f"- {issue['action']}: "
                f"{issue.get('path') or issue.get('tool') or issue.get('error') or 'unknown'}"
                for issue in issues)
        yield TextEvent("\n".join(lines))

    async def stop_recovery(self, job_id: str, position: int) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status != "recovery_required":
            yield ErrorEvent("This task does not require journal recovery.")
            return
        if not await self.persistence.stop_recovery(job_id, position):
            yield ErrorEvent("This task is no longer awaiting journal recovery.")
            return
        yield TextEvent(
            f"Stopped implementation job {job_id} at {item.task_id}. No automatic undo "
            "was attempted; inspect and repair the current project state manually.")

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
        yield TextEvent(message)
        async for event in self._advance(
                job_id, messages, specs, instruction, direct_tool_handlers):
            yield event

    async def undo_code(self, job_id: str, position: int,
                        messages: list[dict], specs: list[ToolSpec],
                        direct_tool_handlers: dict) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        latest = await self.persistence.latest_review(job_id, position)
        interrupted = item is not None and item.status == CODER_RECOVERY_STATUS
        if item is None or not interrupted and (latest is None or latest["verdict"] not in {
                "done", "needs_work", "ask_user"}):
            yield ErrorEvent("This task has no reviewed implementation to undo.")
            return
        waiting_status = CODER_RECOVERY_STATUS if interrupted else "awaiting_review_resolution"
        if not await self.persistence.claim_resolution(job_id, position, "undo"):
            yield ErrorEvent("This task is no longer awaiting a review decision.")
            return
        journal = RollbackJournal(
            self.persistence, job_id, position, specs, direct_tool_handlers)
        restored = await journal.restore()
        if restored is not None:
            restored_paths, conflicts = restored
            if conflicts:
                await self.persistence.update_implementation_task(
                    job_id, position, waiting_status,
                    error="Undo conflicts: " + ", ".join(conflicts))
                await self.persistence.record_resolution_action(
                    job_id, position, "undo_conflict", {
                        "restored_paths": restored_paths,
                        "conflict_paths": conflicts,
                    })
                yield TextEvent(
                    "Undo stopped to protect files changed after the coder run. "
                    f"Conflicts: {', '.join(conflicts)}. "
                    "Already restored: " + (", ".join(restored_paths) or "none") + ".")
                yield TextEvent(self._pending_options(item, latest, job_id))
                return
            await self.persistence.record_resolution_action(
                job_id, position, "undo_completed", {
                    "mode": "journal", "restored_paths": restored_paths,
                })
            await self.persistence.update_implementation_task(
                job_id, position, "undone", error=None)
            await self.persistence.finish_implementation_job(job_id, "undone")
            yield TextEvent(
                f"{item.task_id} code undo completed for {len(restored_paths)} file(s). "
                "The task remains incomplete and the job stopped.")
            return
        report = await self.persistence.coder_report_for(job_id, position)
        findings = self._stored_findings(latest)
        claimed_paths = sorted({path for finding in findings
                                for path in finding.artifact_paths})
        await self.persistence.record_resolution_action(
            job_id, position, "undo_requested", {
                "claimed_paths": claimed_paths,
                "legacy_without_snapshot": True,
            })
        change, queued_task = await self.artifacts.task_by_id(item.task_id)
        undo_instruction = "\n".join([
            "Undo only the project-code changes introduced while implementing this task.",
            "Use the saved coder report and reviewer artifact paths as claims to inspect, not as proof.",
            "Preserve unrelated user changes. Do not modify OpenSpec or .codepartner tracking artifacts.",
            "If a change cannot be safely attributed to this task, leave it untouched and report it.",
            f"Task change: {change.id}",
            f"Reviewer artifact paths: {json.dumps(claimed_paths)}",
            f"Saved coder report:\n{report}",
        ])
        yield TextEvent(f"Undoing code for {item.task_id} with a routed coder recovery run.")
        failure: str | None = None
        paused = False
        decision_id: str | None = None

        async def record_decision(decision) -> None:
            nonlocal decision_id
            decision_id = decision.decision_id
            await self.persistence.update_implementation_task(
                job_id, position, "undoing", decision_id=decision_id)

        async for event in self.task_execution.execute(
                queued_task, change.id, messages, specs, on_decision=record_decision,
                execution_instruction=undo_instruction,
                direct_tool_handlers=direct_tool_handlers, owner_metadata={
                    "implementation_job_id": job_id,
                    "implementation_position": position,
                    "implementation_task_id": item.task_id,
                    "implementation_action": "undo",
                }):
            if isinstance(event, ErrorEvent):
                failure = event.message
            if isinstance(event, ToolCallEvent):
                paused = True
                run = bridge.run_for(event.tool_call_id)
                if run is not None:
                    run.metadata.update({
                        "implementation_job_id": job_id,
                        "implementation_position": position,
                        "implementation_task_id": item.task_id,
                        "implementation_action": "undo",
                        "implementation_service": self,
                    })
            if not isinstance(event, DoneEvent):
                yield event
        if paused:
            return
        if failure:
            await self.persistence.update_implementation_task(
                job_id, position, waiting_status, error=failure)
            await self.persistence.record_resolution_action(
                job_id, position, "undo_failed", {"error": failure})
            yield TextEvent(f"Undo failed for {item.task_id}: {failure}\n"
                            + self._pending_options(item, latest, job_id))
            return
        async for event in self._complete_undo(job_id, position):
            yield event

    async def _complete_undo(self, job_id: str, position: int) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None:
            yield ErrorEvent("The implementation task was not found after undo.")
            return
        await self.persistence.record_resolution_action(
            job_id, position, "undo_completed", {"mode": "coder_assisted"})
        await self.persistence.update_implementation_task(
            job_id, position, "undone", error=None)
        await self.persistence.finish_implementation_job(job_id, "undone")
        yield TextEvent(
            f"{item.task_id} code undo completed. The task remains incomplete and the job stopped.")

    async def _complete(self, job_id: str, position: int, task_id: str,
                        review: CompletionReview) -> None:
        await self.artifacts.mark_task_completed(task_id, review)
        await self.persistence.update_implementation_task(job_id, position, "completed")

    async def _fail(self, job_id: str, position: int, error: str) -> None:
        await self.persistence.update_implementation_task(job_id, position, "failed", error=error)
        await self.persistence.finish_implementation_job(job_id, "failed", error)

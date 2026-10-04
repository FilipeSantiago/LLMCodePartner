"""Sequential, durable execution and outcome review of OpenSpec tasks."""

import logging
from collections.abc import AsyncIterator

from agent.routing.models import ImplementationJob, ImplementationJobTask
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import ReviewError, ReviewService
from agent.routing.task_execution import TaskExecutionService
from agent.spec.artifact_store import ArtifactStore
from mcp_bridge import bridge
from mcp_bridge.models import DoneEvent, ErrorEvent, Event, Run, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec


log = logging.getLogger("codepartner.implementation")
MAX_REVISIONS = 2


class ImplementationService:
    """Runs an implementation job one task at a time, including tool resumes."""

    def __init__(self, artifacts: ArtifactStore,
                 task_execution: TaskExecutionService | None = None,
                 persistence: RoutingPersistence | None = None,
                 reviewer: ReviewService | None = None):
        self.artifacts = artifacts
        self.task_execution = task_execution or TaskExecutionService()
        self.persistence = persistence or self.task_execution.persistence
        self.reviewer = reviewer or ReviewService(getattr(self.task_execution, "config", None))

    async def create_job(self, selectors: list[str], instruction: str = "",
                         conversation_id: str | None = None) -> ImplementationJob:
        targets = await self.artifacts.tasks_for_selectors(selectors)
        job = ImplementationJob.create(selectors, [task.id for _, task in targets], instruction)
        queued = [
            ImplementationJobTask(job.job_id, position, change.id, task.id)
            for position, (change, task) in enumerate(targets)
        ]
        await self.persistence.create_implementation_job(job, queued, conversation_id)
        return job

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
        if failed:
            await self._fail(job_id, position, failed)
            yield TextEvent(f"Implementation job {job_id} stopped: {failed}")
        else:
            async for event in self._review_and_advance(
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
        report = await self.persistence.coder_report_for(job_id, position)
        async for event in self._review_and_advance(
                job_id, position, messages, specs,
                run.metadata.get("implementation_instruction", ""),
                run.direct_tool_handlers, report, run.metadata.get("model")):
            yield event

    async def _advance(self, job_id: str, messages: list[dict], specs: list[ToolSpec],
                       instruction: str = "", direct_tool_handlers: dict | None = None
                       ) -> AsyncIterator[Event]:
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
                        direct_tool_handlers: dict) -> AsyncIterator[Event]:
        change, task = await self.artifacts.task_by_id(item.task_id)
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

        async for event in self.task_execution.execute(
                task, change.id, messages, specs, on_decision=record_decision,
                execution_instruction=instruction,
                direct_tool_handlers=direct_tool_handlers):
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
            if not isinstance(event, DoneEvent):
                yield event
        report = "".join(report_parts)
        await self.persistence.save_coder_report(item.job_id, item.position, report)
        if paused:
            return
        if failure:
            await self._fail(item.job_id, item.position, failure)
            yield TextEvent(f"Implementation job {item.job_id} stopped: {failure}")
        else:
            coder_model = await self.persistence.actual_model_for(decision_id)
            async for event in self._review_and_advance(
                    item.job_id, item.position, messages, specs, instruction,
                    direct_tool_handlers, report, coder_model):
                yield event

    async def _review_and_advance(self, job_id: str, position: int,
                                  messages: list[dict], specs: list[ToolSpec],
                                  instruction: str, direct_tool_handlers: dict,
                                  report: str, coder_model: str | None
                                  ) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or not await self.persistence.claim_review(job_id, position):
            return
        change, task = await self.artifacts.task_by_id(item.task_id)
        coder_model = coder_model or await self.persistence.actual_model_for(item.decision_id)
        yield TextEvent(f"Reviewing {task.id}.")
        try:
            if not coder_model:
                raise ReviewError("actual coder model was not recorded")
            decision = await self.reviewer.review(
                change, task, report, coder_model, specs, direct_tool_handlers)
        except ReviewError as exc:
            await self.persistence.update_implementation_task(
                job_id, position, "review_pending", error=str(exc))
            yield TextEvent(f"Review of {task.id} is pending: {exc}. "
                            "Reply 'retry review' in this chat to try again.")
            return
        verdict = ("ask_user" if task.preferred_capability.lower() in {"research", "verification", "review"}
                   and decision.verdict == "done" else decision.verdict)
        await self.persistence.record_review(
            job_id, position, coder_model, decision.reviewer_model,
            verdict, decision.rationale, decision.question)
        log.info("implementation_review job_id=%s task_id=%s coder_model=%s "
                 "reviewer_model=%s verdict=%s", job_id, task.id, coder_model,
                 decision.reviewer_model, verdict)
        if verdict == "ask_user":
            await self.persistence.update_implementation_task(
                job_id, position, "awaiting_confirmation")
            question = decision.question or f"Do you consider {task.id} complete?"
            yield TextEvent(f"{task.id} review: {decision.rationale}\n\n{question} "
                            f"Reply yes or no for job {job_id}.")
            return
        if verdict == "needs_work":
            review_count = await self.persistence.review_count(job_id, position)
            feedback = decision.feedback or decision.rationale
            if review_count > MAX_REVISIONS:
                await self.persistence.update_implementation_task(
                    job_id, position, "review_pending", error=feedback)
                yield TextEvent(f"{task.id} still needs work: {feedback}. "
                                "Reply 'retry review' after addressing the gaps.")
                return
            yield TextEvent(f"{task.id} needs work: {feedback} "
                            f"Revision {review_count}/{MAX_REVISIONS}.")
            revised_instruction = "\n".join(filter(None, [
                instruction, f"Reviewer feedback for {task.id}: {feedback}"
            ]))
            async for event in self._run_item(item, messages, specs,
                                              revised_instruction, direct_tool_handlers):
                yield event
            return
        await self._complete(job_id, position, task.id)
        yield TextEvent(f"{task.id} completed after review: {decision.rationale}")
        async for event in self._advance(job_id, messages, specs, instruction,
                                         direct_tool_handlers):
            yield event

    async def confirm(self, job_id: str, position: int, approved: bool,
                      feedback: str, messages: list[dict], specs: list[ToolSpec],
                      direct_tool_handlers: dict | None = None,
                      instruction: str = "") -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status != "awaiting_confirmation":
            yield ErrorEvent("This task is no longer awaiting confirmation.")
            return
        if approved:
            await self._complete(job_id, position, item.task_id)
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

    async def retry_review(self, job_id: str, position: int,
                           messages: list[dict], specs: list[ToolSpec],
                           instruction: str, direct_tool_handlers: dict
                           ) -> AsyncIterator[Event]:
        item = await self.persistence.implementation_task(job_id, position)
        if item is None or item.status not in {"review_pending", "reviewing"}:
            yield ErrorEvent("This task is no longer pending review.")
            return
        report = await self.persistence.coder_report_for(job_id, position)
        coder_model = await self.persistence.actual_model_for(item.decision_id)
        async for event in self._review_and_advance(
                job_id, position, messages, specs, instruction,
                direct_tool_handlers, report, coder_model):
            yield event

    async def _complete(self, job_id: str, position: int, task_id: str) -> None:
        await self.artifacts.mark_task_completed(task_id)
        await self.persistence.update_implementation_task(job_id, position, "completed")

    async def _fail(self, job_id: str, position: int, error: str) -> None:
        await self.persistence.update_implementation_task(job_id, position, "failed", error=error)
        await self.persistence.finish_implementation_job(job_id, "failed", error)

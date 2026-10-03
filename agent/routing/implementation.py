"""Sequential, durable execution of selected OpenSpec tasks."""

from collections.abc import AsyncIterator

from agent.routing.models import ImplementationJob, ImplementationJobTask
from agent.routing.persistence import RoutingPersistence
from agent.routing.task_execution import TaskExecutionService
from agent.spec.artifact_store import ArtifactStore
from mcp_bridge import bridge
from mcp_bridge.models import DoneEvent, ErrorEvent, Event, Run, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec


class ImplementationService:
    """Runs an implementation job one task at a time, including tool resumes."""

    def __init__(self, artifacts: ArtifactStore,
                 task_execution: TaskExecutionService | None = None,
                 persistence: RoutingPersistence | None = None):
        self.artifacts = artifacts
        self.task_execution = task_execution or TaskExecutionService()
        self.persistence = persistence or self.task_execution.persistence

    async def create_job(self, selectors: list[str], instruction: str = "") -> ImplementationJob:
        targets = await self.artifacts.tasks_for_selectors(selectors)
        job = ImplementationJob.create(selectors, [task.id for _, task in targets], instruction)
        queued = [
            ImplementationJobTask(job.job_id, position, change.id, task.id)
            for position, (change, task) in enumerate(targets)
        ]
        await self.persistence.create_implementation_job(job, queued)
        return job

    async def start(self, job: ImplementationJob, messages: list[dict],
                    specs: list[ToolSpec]) -> AsyncIterator[Event]:
        yield TextEvent(
            f"Implementation job {job.job_id} queued {len(job.task_ids)} task(s): "
            f"{', '.join(job.task_ids)}."
        )
        async for event in self._advance(job.job_id, messages, specs, job.instruction):
            yield event

    async def resume(self, run: Run, messages: list[dict],
                     specs: list[ToolSpec]) -> AsyncIterator[Event]:
        job_id = run.metadata.get("implementation_job_id")
        position = run.metadata.get("implementation_position")
        if not job_id or position is None:
            raise RuntimeError("routed run is not attached to an implementation job")
        failed: str | None = None
        paused = False
        terminal: DoneEvent | None = None
        async for event in self.task_execution.resume(run, messages, specs):
            if isinstance(event, ErrorEvent):
                failed = event.message
            if isinstance(event, ToolCallEvent):
                paused = True
            if isinstance(event, DoneEvent):
                terminal = event
            else:
                yield event
        if paused:
            return
        if failed:
            await self._fail(job_id, position, failed)
            yield TextEvent(f"Implementation job {job_id} stopped: {failed}")
            if terminal is not None:
                yield terminal
            return
        if not run.mutating_tool_succeeded:
            failure = self._missing_mutation_error(run.metadata["implementation_task_id"])
            await self._fail(job_id, position, failure)
            yield ErrorEvent(failure)
            yield TextEvent(f"Implementation job {job_id} stopped: {failure}")
            if terminal is not None:
                yield terminal
            return
        await self._complete(job_id, position, run.metadata["implementation_task_id"])
        async for event in self._advance(
                job_id, messages, specs, run.metadata.get("implementation_instruction", "")):
            yield event
        if terminal is not None:
            yield terminal

    async def _advance(self, job_id: str, messages: list[dict], specs: list[ToolSpec],
                       instruction: str = "") -> AsyncIterator[Event]:
        item = await self.persistence.next_implementation_task(job_id)
        if item is None:
            await self.persistence.finish_implementation_job(job_id)
            yield TextEvent(f"Implementation job {job_id} completed.")
            return
        change, task = await self.artifacts.task_by_id(item.task_id)
        await self.persistence.update_implementation_task(job_id, item.position, "running")
        yield TextEvent(f"Starting {task.id} ({task.complexity}) from {change.id}.")
        failure: str | None = None
        paused = False
        terminal: DoneEvent | None = None
        async def record_decision(decision) -> None:
            await self.persistence.update_implementation_task(
                job_id, item.position, "running", decision_id=decision.decision_id
            )

        async for event in self.task_execution.execute(
                task, change.id, messages, specs, on_decision=record_decision,
                execution_instruction=instruction):
            if isinstance(event, ToolCallEvent):
                paused = True
                run = bridge.run_for(event.tool_call_id)
                if run is not None:
                    run.metadata.update({
                        "implementation_job_id": job_id,
                        "implementation_position": item.position,
                        "implementation_task_id": task.id,
                        "implementation_instruction": instruction,
                        "implementation_service": self,
                    })
                await self.persistence.update_implementation_task(job_id, item.position, "waiting")
            if isinstance(event, ErrorEvent):
                failure = event.message
            if isinstance(event, DoneEvent):
                terminal = event
            else:
                yield event
        if paused:
            return
        if failure:
            await self._fail(job_id, item.position, failure)
            yield TextEvent(f"Implementation job {job_id} stopped: {failure}")
            if terminal is not None:
                yield terminal
            return
        failure = self._missing_mutation_error(task.id)
        await self._fail(job_id, item.position, failure)
        yield ErrorEvent(failure)
        yield TextEvent(f"Implementation job {job_id} stopped: {failure}")
        if terminal is not None:
            yield terminal

    async def _complete(self, job_id: str, position: int, task_id: str) -> None:
        await self.artifacts.mark_task_completed(task_id)
        await self.persistence.update_implementation_task(job_id, position, "completed")

    async def _fail(self, job_id: str, position: int, error: str) -> None:
        await self.persistence.update_implementation_task(job_id, position, "failed", error=error)
        await self.persistence.finish_implementation_job(job_id, "failed", error)

    @staticmethod
    def _missing_mutation_error(task_id: str) -> str:
        return (
            f"{task_id} ended without a successful source-file write MCP tool call; "
            "the task remains incomplete."
        )

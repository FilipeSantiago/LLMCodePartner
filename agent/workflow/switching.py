"""Stopping one implementation job so another flow can start in the same project.

A switch cannot be one transaction: the old job runs in a provider task (and its
IDE may still be applying a write) while the new flow is only a database row.
The claim records a durable `workflow_switches` row, then this module moves it
through `switch_requested → stopping → stopped → started`. Any failure ends in
`stop_failed` without starting the new flow, and `recover_switches` finishes
switches a crash left behind. Nothing here rolls back files.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass

from agent.routing.models import ACTIVE_JOB_STATUSES, TERMINAL_JOB_STATUSES
from agent.routing.persistence import RoutingPersistence
from agent.workflow.states import STATE_DESCRIPTIONS, WorkflowState
from mcp_bridge import bridge
from mcp_bridge import server as bridge_server

log = logging.getLogger("codepartner.workflow.switch")

STOP_WAIT_SECONDS = float(os.getenv("CODEPARTNER_STOP_WAIT_SECONDS", "20"))
_POLL_SECONDS = 0.05
# A just-claimed switch is about to be executed by its own request; only switches
# untouched for longer than a full stop could take are treated as abandoned.
RECOVERY_GRACE_SECONDS = 2 * STOP_WAIT_SECONDS + 10
# Switches being executed by a request in this process; recovery leaves them alone.
_executing: set[str] = set()


@dataclass(frozen=True)
class JobActivity:
    """One unfinished job in the project, as the user should hear about it."""

    job_id: str
    conversation_id: str | None
    status: str
    selectors: tuple[str, ...]
    task_id: str | None
    # live: a run is executing in this process; orphaned: recorded as active but
    # nothing executes it (for example after a restart); paused: awaiting a decision.
    liveness: str

    @property
    def active(self) -> bool:
        return self.liveness in {"live", "orphaned"}

    @property
    def label(self) -> str:
        return join_names(self.selectors) or self.task_id or f"job {self.job_id[:8]}"

    def payload(self) -> dict:
        return {"job_id": self.job_id, "status": self.status, "liveness": self.liveness,
                "label": self.label, "conversation_id": self.conversation_id}


def join_names(names) -> str:
    names = [name for name in names if name]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


async def project_activity(persistence: RoutingPersistence, project_id: str | None,
                           conversation_id: str | None) -> tuple[JobActivity, ...]:
    rows = await persistence.project_jobs(project_id, conversation_id)
    found = []
    for row in rows:
        if row["status"] in ACTIVE_JOB_STATUSES:
            liveness = "live" if bridge_server.live_runs_for_job(row["job_id"]) else "orphaned"
        else:
            liveness = "paused"
        found.append(JobActivity(row["job_id"], row["conversation_id"], row["status"],
                                 row["selectors"], row["task_id"], liveness))
    return tuple(found)


def switch_preamble(displaced: list[dict], conversation_id: str, start_phrase: str) -> str:
    """The single question that asks to stop the active work and start the new flow."""
    lines = []
    for item in displaced:
        origin = ("" if item.get("conversation_id") in {None, conversation_id}
                  else " (started from another chat)")
        if item["liveness"] == "live":
            lines.append(f"CodePartner is currently working on {item['label']}{origin}.")
        else:
            lines.append(f"{item['label']}{origin} is recorded as {item['status']}, but "
                         "nothing is executing it; it was probably interrupted.")
    names = join_names([item["label"] for item in displaced])
    action = (f"stop {names}" if any(item["liveness"] == "live" for item in displaced)
              else f"mark {names} as stopped")
    lines.append(f"Would you like to {action} and {start_phrase}? "
                 f"{names}'s existing file changes will remain; they will not be rolled back.")
    return " ".join(lines)


def paused_note(paused: list[dict]) -> str:
    notes = []
    for item in paused:
        try:
            meaning = STATE_DESCRIPTIONS[WorkflowState(item["status"])]
        except ValueError:
            meaning = item["status"]
        notes.append(f"{item['label']} is paused ({meaning}). It will stay paused; "
                     "you can return to it later.")
    return " ".join(notes)


@dataclass(frozen=True)
class StopResult:
    stopped: bool
    message: str


async def stop_job(persistence: RoutingPersistence, job_id: str, label: str,
                   reason: str, wait: float | None = None) -> StopResult:
    """Stop one job: block new tool calls, let in-flight writes land, cancel, record.

    A write that the IDE may still apply would be lost from the mutation journal if
    the run were cancelled underneath it, so cancellation waits for those first.
    """
    wait = STOP_WAIT_SECONDS if wait is None else wait
    loop = asyncio.get_running_loop()
    runs = bridge_server.live_runs_for_job(job_id)
    for run in runs:
        run.metadata["stop_requested"] = True
    deadline = loop.time() + wait
    while any(bridge.has_inflight_mutation(run) for run in runs):
        if loop.time() >= deadline:
            for run in runs:
                run.metadata.pop("stop_requested", None)
            return StopResult(False, (
                f"{label} could not be stopped safely: a file change is still being applied "
                f"by the IDE. {label} continues to run."))
        await asyncio.sleep(_POLL_SECONDS)

    previous = await persistence.mark_job_stopping(job_id)
    if previous is None:
        return StopResult(False, f"{label} no longer exists.")
    if previous in TERMINAL_JOB_STATUSES:
        return StopResult(True, f"{label} had already finished ({previous}).")
    if previous not in ACTIVE_JOB_STATUSES:
        # It paused while the user was deciding; it is no longer running.
        return StopResult(True, f"{label} had already paused and was left unchanged.")

    tasks = [run.task for run in runs if run.task is not None and not run.task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=wait)
        if pending:
            # The job stays `stopping`, which already prevents it from advancing.
            return StopResult(False, (
                f"{label} did not confirm that it stopped. It will not start new tasks, "
                "but it may still be finishing its current step."))
    await persistence.finalize_stopped_job(job_id, reason)
    return StopResult(True, f"Stopped {label}. Its file changes were kept.")


@dataclass(frozen=True)
class SwitchOutcome:
    ok: bool
    message: str


async def run_switch(persistence: RoutingPersistence, switch: dict, start_label: str,
                     wait: float | None = None) -> SwitchOutcome:
    """Stop every displaced job; release the new job only if all of them stopped."""
    switch_id = switch["switch_id"]
    if switch_id in _executing:
        return SwitchOutcome(False, "This switch is already in progress.")
    _executing.add(switch_id)
    try:
        if switch["state"] not in {"switch_requested", "stopping"}:
            return SwitchOutcome(False, f"This switch is already {switch['state']}.")
        await persistence.advance_switch(switch_id, ("switch_requested",), "stopping")
        messages = []
        for item in switch["displaced"]:
            result = await stop_job(persistence, item["job_id"], item["label"],
                                    f"stopped to {start_label}", wait)
            if not result.stopped:
                await persistence.advance_switch(switch_id, ("stopping",), "stop_failed",
                                                 result.message)
                if switch.get("new_job_id"):
                    await persistence.set_switched_job_status(switch["new_job_id"], "cancelled")
                log.info("workflow_switch switch_id=%s outcome=stop_failed", switch_id)
                return SwitchOutcome(False, (
                    f"{' '.join((*messages, result.message))} Nothing new was started; "
                    "send your request again to choose what to do next."))
            messages.append(result.message)
        await persistence.advance_switch(switch_id, ("stopping",), "stopped")
        if switch.get("new_job_id"):
            await persistence.set_switched_job_status(switch["new_job_id"], "queued")
        log.info("workflow_switch switch_id=%s outcome=stopped", switch_id)
        return SwitchOutcome(True, " ".join(messages))
    finally:
        _executing.discard(switch_id)


async def mark_switch_started(persistence: RoutingPersistence, switch_id: str) -> None:
    await persistence.advance_switch(switch_id, ("stopped",), "started")


async def recover_switches(persistence: RoutingPersistence) -> None:
    """Finish switches interrupted by a crash or a dropped request (idempotent).

    Jobs flagged `stopping` are recorded as stopped once nothing executes them;
    a new job that never started is cancelled rather than started unattended.
    """
    for switch in await persistence.unfinished_switches():
        if (switch["switch_id"] in _executing
                or time.time() - switch["updated_at"] < RECOVERY_GRACE_SECONDS):
            continue
        if any(bridge_server.live_runs_for_job(item["job_id"])
               for item in switch["displaced"]):
            continue
        for item in switch["displaced"]:
            if await persistence.implementation_job_status(item["job_id"]) == "stopping":
                await persistence.finalize_stopped_job(
                    item["job_id"], "stopped by an interrupted switch")
        if switch.get("new_job_id"):
            await persistence.set_switched_job_status(switch["new_job_id"], "cancelled")
        await persistence.advance_switch(
            switch["switch_id"], ("switch_requested", "stopping", "stopped"), "interrupted",
            "recovered after the switch was interrupted")
        log.info("workflow_switch switch_id=%s outcome=interrupted", switch["switch_id"])


def interruption_notice(switches: list[dict]) -> str:
    lines = []
    for switch in switches:
        names = join_names([item["label"] for item in switch["displaced"]])
        lines.append(f"An earlier request to stop {names} and start new work was "
                     f"interrupted. {names} is no longer running and its file changes "
                     "were kept; the new work was not started.")
    return " ".join(lines)

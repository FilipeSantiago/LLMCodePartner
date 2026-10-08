"""Failure-isolated telemetry recording API."""

import logging

from telemetry.models import AttemptSummary, ToolActivity, TurnEvent, WorkflowEvent
from telemetry.storage import TelemetryStorage

log = logging.getLogger("codepartner.telemetry")


class TelemetryRecorder:
    """Records local metrics without ever affecting execution policy."""

    def __init__(self, storage: TelemetryStorage | None = None):
        self.storage = storage or TelemetryStorage()

    def tool_started(self, activity: ToolActivity) -> None:
        try:
            self.storage.start_tool_call(activity)
        except Exception:
            log.exception("telemetry tool-start recording failed")

    def tool_finished(self, call_id: str, success: bool, output_chars: int,
                      duration_ms: int, meaningful_progress: bool,
                      error_type: str | None = None) -> None:
        try:
            self.storage.finish_tool_call(
                call_id, success, output_chars, duration_ms,
                meaningful_progress, error_type,
            )
        except Exception:
            log.exception("telemetry tool-result recording failed")

    def attempt_finished(self, summary: AttemptSummary) -> None:
        try:
            self.storage.record_summary(summary)
        except Exception:
            log.exception("telemetry attempt recording failed")

    def workflow_event(self, event: WorkflowEvent) -> None:
        try:
            self.storage.record_workflow_event(event)
        except Exception:
            log.exception("telemetry workflow-event recording failed")

    def turn_event(self, event: TurnEvent) -> None:
        try:
            self.storage.record_turn_event(event)
        except Exception:
            log.exception("telemetry turn-event recording failed")

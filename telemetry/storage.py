"""SQLite storage owned by the telemetry package."""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from telemetry.models import AttemptSummary, ToolActivity, TurnEvent, WorkflowEvent


class TelemetryStorage:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("ROUTING_DB_PATH")
                         or Path.home() / ".codepartner" / "routing.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _database(self):
        db = sqlite3.connect(self.path)
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _initialize(self) -> None:
        with self._database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS telemetry_tool_activity (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT, task_id TEXT, contract_id TEXT,
                  contract_revision INTEGER, attempt INTEGER, role TEXT NOT NULL,
                  call_id TEXT NOT NULL, tool_name TEXT NOT NULL,
                  category TEXT NOT NULL, signature TEXT NOT NULL,
                  paths_json TEXT NOT NULL, repeated INTEGER NOT NULL,
                  calls_before_first_edit INTEGER NOT NULL,
                  success INTEGER, error_type TEXT, output_chars INTEGER,
                  duration_ms INTEGER, meaningful_progress INTEGER NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL, completed_at REAL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS telemetry_tool_call_id
                  ON telemetry_tool_activity(call_id);
                CREATE TABLE IF NOT EXISTS telemetry_attempt_summaries (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT, task_id TEXT, contract_id TEXT,
                  contract_revision INTEGER, attempt INTEGER, role TEXT NOT NULL,
                  outcome TEXT NOT NULL, tool_calls INTEGER NOT NULL,
                  calls_by_category_json TEXT NOT NULL,
                  unique_files_read_json TEXT NOT NULL,
                  output_chars INTEGER NOT NULL, repeated_calls INTEGER NOT NULL,
                  calls_before_first_edit INTEGER NOT NULL,
                  meaningful_progress INTEGER NOT NULL,
                  usage_json TEXT, escalation_reason TEXT, extra_json TEXT NOT NULL,
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS telemetry_workflow_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  proposal_id TEXT, stage TEXT NOT NULL, action TEXT,
                  source_state TEXT, interpretation_source TEXT, model TEXT,
                  confidence TEXT, ambiguity_count INTEGER NOT NULL,
                  selector_count INTEGER NOT NULL, guidance_present INTEGER NOT NULL,
                  latency_ms INTEGER, confirmation_latency_ms INTEGER,
                  input_tokens INTEGER, output_tokens INTEGER,
                  transition_result TEXT, message_fingerprint TEXT,
                  detail TEXT, created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS telemetry_workflow_proposal
                  ON telemetry_workflow_events(proposal_id);
                CREATE TABLE IF NOT EXISTS telemetry_turn_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conversation_id TEXT, project_id TEXT, turn_type TEXT NOT NULL,
                  route TEXT NOT NULL, free_json TEXT, contextual_json TEXT,
                  proposal_id TEXT, outcome TEXT, message_fingerprint TEXT,
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS telemetry_aci_attempts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL UNIQUE,
                  job_id TEXT, task_id TEXT, decision_id TEXT, provider TEXT, model TEXT,
                  metrics_json TEXT NOT NULL, usage_json TEXT,
                  cost_usd REAL, created_at REAL NOT NULL
                );
            """)

    def record_aci_attempt(self, run_id: str, metadata: dict,
                           metrics: dict, usage: dict | None) -> None:
        with self._database() as db:
            db.execute("""INSERT OR REPLACE INTO telemetry_aci_attempts
              (run_id, job_id, task_id, decision_id, provider, model, metrics_json,
               usage_json, cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                run_id, metadata.get("implementation_job_id"),
                metadata.get("implementation_task_id") or metadata.get("task_id"),
                metadata.get("decision_id"), metadata.get("provider"),
                metadata.get("model"), json.dumps(metrics, sort_keys=True),
                json.dumps(usage, sort_keys=True) if usage else None,
                metadata.get("cost_usd"), time.time(),
            ))

    def start_tool_call(self, activity: ToolActivity) -> None:
        c = activity.context
        with self._database() as db:
            db.execute("""INSERT OR IGNORE INTO telemetry_tool_activity
              (job_id, task_id, contract_id, contract_revision, attempt, role,
               call_id, tool_name, category, signature, paths_json, repeated,
               calls_before_first_edit, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                c.job_id, c.task_id, c.contract_id, c.contract_revision, c.attempt, c.role,
                activity.call_id, activity.tool_name, activity.category, activity.signature,
                json.dumps(activity.paths), int(activity.repeated),
                activity.calls_before_first_edit, time.time(),
            ))

    def finish_tool_call(self, call_id: str, success: bool, output_chars: int,
                         duration_ms: int, meaningful_progress: bool,
                         error_type: str | None = None) -> None:
        with self._database() as db:
            db.execute("""UPDATE telemetry_tool_activity SET success=?, error_type=?,
              output_chars=?, duration_ms=?, meaningful_progress=?, completed_at=?
              WHERE call_id=?""", (
                int(success), error_type, output_chars, duration_ms,
                int(meaningful_progress), time.time(), call_id,
            ))

    def record_summary(self, summary: AttemptSummary) -> None:
        c = summary.context
        with self._database() as db:
            db.execute("""INSERT INTO telemetry_attempt_summaries
              (job_id, task_id, contract_id, contract_revision, attempt, role,
               outcome, tool_calls, calls_by_category_json, unique_files_read_json,
               output_chars, repeated_calls, calls_before_first_edit,
               meaningful_progress, usage_json, escalation_reason, extra_json, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                c.job_id, c.task_id, c.contract_id, c.contract_revision, c.attempt, c.role,
                summary.outcome, summary.tool_calls,
                json.dumps(summary.calls_by_category, sort_keys=True),
                json.dumps(summary.unique_files_read), summary.output_chars,
                summary.repeated_calls, summary.calls_before_first_edit,
                summary.meaningful_progress,
                json.dumps(summary.usage, sort_keys=True) if summary.usage else None,
                summary.escalation_reason,
                json.dumps(summary.extra, sort_keys=True), time.time(),
            ))

    def record_turn_event(self, event: TurnEvent) -> None:
        with self._database() as db:
            db.execute("""INSERT INTO telemetry_turn_events
              (conversation_id, project_id, turn_type, route, free_json, contextual_json,
               proposal_id, outcome, message_fingerprint, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                event.conversation_id, event.project_id, event.turn_type, event.route,
                json.dumps(event.free_classification, sort_keys=True)
                if event.free_classification else None,
                json.dumps(event.contextual_classification, sort_keys=True)
                if event.contextual_classification else None,
                event.proposal_id, event.outcome, event.message_fingerprint, time.time(),
            ))

    def record_workflow_event(self, event: WorkflowEvent) -> None:
        with self._database() as db:
            db.execute("""INSERT INTO telemetry_workflow_events
              (proposal_id, stage, action, source_state, interpretation_source,
               model, confidence, ambiguity_count, selector_count, guidance_present,
               latency_ms, confirmation_latency_ms, input_tokens, output_tokens,
               transition_result, message_fingerprint, detail, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                event.proposal_id, event.stage, event.action, event.source_state,
                event.interpretation_source, event.model, event.confidence,
                event.ambiguity_count, event.selector_count,
                int(event.guidance_present), event.latency_ms,
                event.confirmation_latency_ms, event.input_tokens,
                event.output_tokens, event.transition_result,
                event.message_fingerprint, event.detail, time.time(),
            ))

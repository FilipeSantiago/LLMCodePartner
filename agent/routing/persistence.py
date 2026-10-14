"""Durable operational audit records, separate from OpenSpec artifacts."""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from agent.routing.models import (
    ACTIVE_JOB_STATUSES, HALTED_JOB_STATUSES, TERMINAL_JOB_STATUSES,
    ExecutionDecision, ImplementationJob,
    ImplementationJobTask,
)
from agent.routing.gateway import operational_failure

from agent.workflow.states import PAUSED_STATES

_ACTIVE_SQL = ", ".join(f"'{status}'" for status in sorted(ACTIVE_JOB_STATUSES))
_HALTED_SQL = ", ".join(f"'{status}'" for status in sorted(HALTED_JOB_STATUSES))
# Canonical proposal schema. Rows are always read by these column names, so a
# table created by an older version cannot be mis-mapped by position.
_PROPOSAL_COLUMNS = (
    "proposal_id", "confirmation_token", "conversation_id", "action",
    "target_job_id", "target_position", "target_task_id", "expected_status",
    "expected_revision", "payload_json", "request_fingerprint", "confidence",
    "ambiguities_json", "interpreter_model", "interpretation_source",
    "corrected_from", "status", "created_at", "expires_at", "transition_result",
)
_PROPOSAL_SELECT = "SELECT " + ", ".join(_PROPOSAL_COLUMNS) + " FROM workflow_action_proposals"
_PROPOSALS_DDL = """
                CREATE TABLE IF NOT EXISTS workflow_action_proposals (
                  proposal_id TEXT PRIMARY KEY, confirmation_token TEXT UNIQUE NOT NULL,
                  conversation_id TEXT NOT NULL, action TEXT NOT NULL,
                  target_job_id TEXT, target_position INTEGER, target_task_id TEXT,
                  expected_status TEXT, expected_revision INTEGER,
                  payload_json TEXT NOT NULL, request_fingerprint TEXT NOT NULL,
                  confidence TEXT NOT NULL, ambiguities_json TEXT NOT NULL,
                  interpreter_model TEXT, interpretation_source TEXT NOT NULL,
                  corrected_from TEXT, status TEXT NOT NULL,
                  created_at REAL NOT NULL, expires_at REAL NOT NULL,
                  transition_result TEXT
                );
                CREATE INDEX IF NOT EXISTS workflow_proposals_conversation
                  ON workflow_action_proposals(conversation_id, status, created_at);
"""


def _migrate_workflow_proposals(db: sqlite3.Connection) -> None:
    """Rebuild a proposals table created by an older version (idempotent).

    Earlier builds added NOT NULL columns the current insert does not fill
    (`updated_at`) and used another column order. SQLite cannot drop a NOT NULL
    constraint, so the table is copied into the canonical schema in one savepoint.
    """
    existing = [row[1] for row in db.execute("PRAGMA table_info(workflow_action_proposals)")]
    if not existing or tuple(existing) == _PROPOSAL_COLUMNS:
        return
    present = set(existing)
    defaults = {"interpretation_source": "'deterministic'", "ambiguities_json": "'[]'"}
    select = ", ".join(
        (f"COALESCE({column}, {defaults[column]})" if column in defaults else column)
        if column in present else defaults.get(column, "NULL")
        for column in _PROPOSAL_COLUMNS)
    db.execute("SAVEPOINT migrate_workflow_proposals")
    try:
        db.execute("DROP INDEX IF EXISTS workflow_proposals_conversation")
        db.execute("ALTER TABLE workflow_action_proposals "
                   "RENAME TO workflow_action_proposals_legacy")
        for statement in filter(str.strip, _PROPOSALS_DDL.split(";")):
            db.execute(statement)
        db.execute(f"""INSERT INTO workflow_action_proposals ({", ".join(_PROPOSAL_COLUMNS)})
          SELECT {select} FROM workflow_action_proposals_legacy""")
        db.execute("DROP TABLE workflow_action_proposals_legacy")
    except Exception:
        db.execute("ROLLBACK TO migrate_workflow_proposals")
        db.execute("RELEASE migrate_workflow_proposals")
        raise
    db.execute("RELEASE migrate_workflow_proposals")


# Switch states that have not reached a final outcome yet.
UNFINISHED_SWITCH_STATES = ("switch_requested", "stopping", "stopped")


class RoutingPersistence:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("ROUTING_DB_PATH") or Path.home() / ".codepartner" / "routing.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.path)

    @contextmanager
    def _database(self):
        db = self._connect()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _initialize(self) -> None:
        with self._database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS routing_decisions (
                  decision_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                  change_id TEXT NOT NULL, task_id TEXT NOT NULL,
                  complexity TEXT NOT NULL, candidates_json TEXT NOT NULL,
                  recommended_model TEXT NOT NULL, recommended_provider TEXT NOT NULL,
                  executor TEXT NOT NULL, routing_reason TEXT, routing_source TEXT NOT NULL,
                  actual_model TEXT, actual_provider TEXT, fallback_reason TEXT,
                  gateway_used INTEGER, routing_timestamp REAL NOT NULL,
                  execution_status TEXT, latency_ms INTEGER
                );
                CREATE TABLE IF NOT EXISTS execution_attempts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL,
                  model TEXT NOT NULL, provider TEXT NOT NULL, executor TEXT NOT NULL,
                  gateway_used INTEGER NOT NULL, status TEXT NOT NULL,
                  latency_ms INTEGER, error TEXT, usage_json TEXT, created_at REAL NOT NULL,
                  FOREIGN KEY(decision_id) REFERENCES routing_decisions(decision_id)
                );
                CREATE TABLE IF NOT EXISTS implementation_jobs (
                  job_id TEXT PRIMARY KEY, selectors_json TEXT NOT NULL,
                  instruction TEXT NOT NULL DEFAULT '',
                  conversation_id TEXT,
                  status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                  error TEXT
                );
                CREATE TABLE IF NOT EXISTS implementation_job_tasks (
                  job_id TEXT NOT NULL, position INTEGER NOT NULL, change_id TEXT NOT NULL,
                  task_id TEXT NOT NULL, status TEXT NOT NULL, decision_id TEXT, error TEXT,
                  coder_report TEXT, review_question TEXT, review_started_at REAL,
                  revision_attempts INTEGER NOT NULL DEFAULT 0,
                  revision_budget INTEGER NOT NULL DEFAULT 3,
                  PRIMARY KEY(job_id, position),
                  FOREIGN KEY(job_id) REFERENCES implementation_jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS implementation_reviews (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                  position INTEGER NOT NULL, coder_model TEXT NOT NULL,
                  reviewer_model TEXT NOT NULL, verdict TEXT NOT NULL,
                  rationale TEXT NOT NULL, question TEXT, details_json TEXT,
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_refreshes (
                  job_id TEXT NOT NULL, wp_id TEXT NOT NULL, kb_project_id TEXT NOT NULL,
                  status TEXT NOT NULL, details_json TEXT, created_at REAL NOT NULL,
                  updated_at REAL NOT NULL, PRIMARY KEY(job_id, wp_id)
                );
                CREATE TABLE IF NOT EXISTS blocked_operations (
                  operation_id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                  status TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS refinement_sessions (
                  session_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                  target_id TEXT NOT NULL, target_kind TEXT NOT NULL,
                  change_id TEXT NOT NULL, base_fingerprint TEXT NOT NULL,
                  base_json TEXT NOT NULL, draft_json TEXT NOT NULL,
                  status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS refinement_conversation_status
                  ON refinement_sessions(conversation_id, status, updated_at);
                CREATE TABLE IF NOT EXISTS implementation_resolution_actions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                  position INTEGER NOT NULL, action TEXT NOT NULL,
                  details_json TEXT, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS implementation_rollback_files (
                  job_id TEXT NOT NULL, position INTEGER NOT NULL, path TEXT NOT NULL,
                  before_exists INTEGER NOT NULL, before_content TEXT,
                  expected_hash TEXT, PRIMARY KEY(job_id, position, path)
                );
                CREATE TABLE IF NOT EXISTS implementation_mutations (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL, position INTEGER NOT NULL,
                  tool_name TEXT NOT NULL, operation TEXT NOT NULL,
                  targets_json TEXT NOT NULL, stage TEXT NOT NULL,
                  failure_stage TEXT, failure_detail TEXT,
                  created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS implementation_evidence (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL, position INTEGER NOT NULL,
                  attempt INTEGER NOT NULL DEFAULT 0,
                  kind TEXT NOT NULL, tool_name TEXT NOT NULL,
                  arguments_json TEXT NOT NULL, success INTEGER NOT NULL,
                  payload_path TEXT, payload_hash TEXT, result_excerpt TEXT,
                  metadata_json TEXT,
                  created_at REAL NOT NULL
                );
""" + _PROPOSALS_DDL + """
                CREATE TABLE IF NOT EXISTS workflow_switches (
                  switch_id TEXT PRIMARY KEY, proposal_id TEXT NOT NULL UNIQUE,
                  conversation_id TEXT NOT NULL, project_id TEXT,
                  displaced_json TEXT NOT NULL, new_job_id TEXT,
                  state TEXT NOT NULL, detail TEXT,
                  reported INTEGER NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS wp_planning_sessions (
                  session_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                  project_id TEXT, change_id TEXT NOT NULL, wp_id TEXT NOT NULL,
                  wp_fingerprint TEXT NOT NULL, wp_complexity TEXT NOT NULL,
                  complexity_source TEXT NOT NULL, planner_model TEXT NOT NULL,
                  planner_effort TEXT, status TEXT NOT NULL, revision INTEGER NOT NULL,
                  draft_json TEXT, discussion_json TEXT NOT NULL DEFAULT '[]',
                  gaps_json TEXT NOT NULL DEFAULT '[]',
                  queued_wp_ids_json TEXT NOT NULL DEFAULT '[]', request TEXT,
                  created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS wp_planning_conversation_status
                  ON wp_planning_sessions(conversation_id, status, updated_at);
                CREATE TABLE IF NOT EXISTS wp_planning_turns (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                  revision INTEGER NOT NULL, kind TEXT NOT NULL, user_message TEXT,
                  output_json TEXT, model TEXT NOT NULL, usage_json TEXT, cost_usd REAL,
                  tool_metrics_json TEXT, latency_ms INTEGER, error TEXT,
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS wp_approved_plans (
                  plan_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                  change_id TEXT NOT NULL, wp_id TEXT NOT NULL, plan_json TEXT NOT NULL,
                  fingerprint TEXT NOT NULL, approved_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS wp_reviewer_context_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, plan_id TEXT NOT NULL,
                  job_id TEXT NOT NULL, position INTEGER NOT NULL, task_id TEXT NOT NULL,
                  kind TEXT NOT NULL, summary TEXT NOT NULL, payload_json TEXT,
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_offered_choices (
                  conversation_id TEXT NOT NULL, option_number INTEGER NOT NULL,
                  choice_text TEXT NOT NULL, status TEXT NOT NULL,
                  created_at REAL NOT NULL, expires_at REAL NOT NULL,
                  PRIMARY KEY(conversation_id, option_number)
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(implementation_jobs)")}
            if "instruction" not in columns:
                db.execute("ALTER TABLE implementation_jobs ADD COLUMN instruction TEXT NOT NULL DEFAULT ''")
            if "conversation_id" not in columns:
                db.execute("ALTER TABLE implementation_jobs ADD COLUMN conversation_id TEXT")
            if "project_id" not in columns:
                db.execute("ALTER TABLE implementation_jobs ADD COLUMN project_id TEXT")
            if "plan_id" not in columns:
                db.execute("ALTER TABLE implementation_jobs ADD COLUMN plan_id TEXT")
            planning_columns = {row[1] for row in db.execute(
                "PRAGMA table_info(wp_planning_sessions)")}
            if "queued_wp_ids_json" not in planning_columns:
                db.execute("""ALTER TABLE wp_planning_sessions ADD COLUMN
                  queued_wp_ids_json TEXT NOT NULL DEFAULT '[]'""")
            if "request" not in planning_columns:
                db.execute("ALTER TABLE wp_planning_sessions ADD COLUMN request TEXT")
            task_columns = {row[1] for row in db.execute("PRAGMA table_info(implementation_job_tasks)")}
            if "coder_report" not in task_columns:
                db.execute("ALTER TABLE implementation_job_tasks ADD COLUMN coder_report TEXT")
            if "review_question" not in task_columns:
                db.execute("ALTER TABLE implementation_job_tasks ADD COLUMN review_question TEXT")
            if "review_started_at" not in task_columns:
                db.execute("ALTER TABLE implementation_job_tasks ADD COLUMN review_started_at REAL")
            added_revision_attempts = "revision_attempts" not in task_columns
            if added_revision_attempts:
                db.execute("""ALTER TABLE implementation_job_tasks ADD COLUMN
                  revision_attempts INTEGER NOT NULL DEFAULT 0""")
            if "revision_budget" not in task_columns:
                db.execute("""ALTER TABLE implementation_job_tasks ADD COLUMN
                  revision_budget INTEGER NOT NULL DEFAULT 3""")
            if added_revision_attempts:
                # Preserve the exhausted state of jobs created before explicit
                # revision counters existed. A running/waiting row means every
                # needs_work review already launched a revision. Only the old
                # exhausted review_pending state (three needs_work reviews under
                # the former two-revision cap) has one review that did not launch.
                db.execute("""UPDATE implementation_job_tasks AS task SET revision_attempts =
                  MIN(3, MAX(0,
                    (SELECT COUNT(*) FROM implementation_reviews AS review
                     WHERE review.job_id=task.job_id AND review.position=task.position
                       AND review.verdict='needs_work')
                    - CASE WHEN task.status='review_pending' AND
                        (SELECT COUNT(*) FROM implementation_reviews AS review
                         WHERE review.job_id=task.job_id AND review.position=task.position
                           AND review.verdict='needs_work') >= 3
                      THEN 1 ELSE 0 END))""")
                # Old exhausted jobs used review_pending for both retryable review
                # failures and exhausted needs_work outcomes. Only the latter is a
                # user-decision state; make that distinction durable before normal
                # resolution queries deliberately exclude review_pending.
                db.execute("""UPDATE implementation_job_tasks AS task
                  SET status='awaiting_review_resolution'
                  WHERE task.status='review_pending' AND
                    (SELECT COUNT(*) FROM implementation_reviews AS review
                     WHERE review.job_id=task.job_id AND review.position=task.position
                       AND review.verdict='needs_work') >= 3""")
                db.execute("""UPDATE implementation_jobs SET status='awaiting_review_resolution'
                  WHERE job_id IN (SELECT job_id FROM implementation_job_tasks
                                   WHERE status='awaiting_review_resolution')
                    AND status='review_pending'""")
            evidence_columns = {row[1] for row in db.execute(
                "PRAGMA table_info(implementation_evidence)")}
            if "metadata_json" not in evidence_columns:
                db.execute("ALTER TABLE implementation_evidence ADD COLUMN metadata_json TEXT")
            if "attempt" not in evidence_columns:
                db.execute("""ALTER TABLE implementation_evidence ADD COLUMN
                  attempt INTEGER NOT NULL DEFAULT 0""")
            # Upgrade unfinished jobs from the previous two-revision default.
            # Larger values were explicitly extended by a user and are preserved.
            db.execute("""UPDATE implementation_job_tasks SET revision_budget=3
              WHERE revision_budget=2 AND status NOT IN ('completed', 'undone', 'failed')""")
            review_columns = {row[1] for row in db.execute("PRAGMA table_info(implementation_reviews)")}
            if "details_json" not in review_columns:
                db.execute("ALTER TABLE implementation_reviews ADD COLUMN details_json TEXT")
            _migrate_workflow_proposals(db)
            from agent.workflow.migration import migrate_paused_states
            migrate_paused_states(db)
            # Older jobs treated all coder errors as terminal. Expose only
            # identifiable operational failures as resumable paused work.
            rows = db.execute("""SELECT job_id, position, error
              FROM implementation_job_tasks WHERE status='failed'""").fetchall()
            for job_id, position, error in rows:
                if operational_failure(error or ""):
                    db.execute("""UPDATE implementation_job_tasks
                      SET status='awaiting_coder_recovery' WHERE job_id=? AND position=?""",
                      (job_id, position))
                    db.execute("""UPDATE implementation_jobs
                      SET status='awaiting_coder_recovery' WHERE job_id=? AND status='failed'""",
                      (job_id,))

    async def record_decision(self, decision: ExecutionDecision) -> None:
        await asyncio.to_thread(self._record_decision, decision)

    def _record_decision(self, decision: ExecutionDecision) -> None:
        with self._database() as db:
            db.execute("""INSERT OR REPLACE INTO routing_decisions
              (decision_id, project_id, change_id, task_id, complexity, candidates_json,
               recommended_model, recommended_provider, executor, routing_reason,
               routing_source, routing_timestamp)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                decision.decision_id, decision.project_id, decision.change_id, decision.task_id,
                decision.complexity, json.dumps(decision.candidates), decision.recommended_model,
                decision.recommended_provider, decision.executor, decision.routing_reason,
                decision.routing_source, decision.routed_at.timestamp(),
            ))

    async def record_attempt(self, decision: ExecutionDecision, model: str, provider: str,
                             executor: str, gateway_used: bool, status: str,
                             latency_ms: int | None = None, error: str | None = None,
                             usage: dict | None = None) -> None:
        await asyncio.to_thread(self._record_attempt, decision, model, provider, executor,
                                gateway_used, status, latency_ms, error, usage)

    def _record_attempt(self, decision, model, provider, executor, gateway_used, status, latency_ms, error, usage):
        with self._database() as db:
            db.execute("""INSERT INTO execution_attempts
              (decision_id, model, provider, executor, gateway_used, status, latency_ms, error, usage_json, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                decision.decision_id, model, provider, executor, int(gateway_used), status,
                latency_ms, error, json.dumps(usage) if usage else None, time.time(),
            ))

    async def attempted_models_for(self, decision_id: str | None) -> frozenset[str]:
        if not decision_id:
            return frozenset()
        return await asyncio.to_thread(self._attempted_models_for, decision_id)

    def _attempted_models_for(self, decision_id):
        with self._database() as db:
            rows = db.execute(
                "SELECT DISTINCT model FROM execution_attempts WHERE decision_id=?",
                (decision_id,)).fetchall()
            current = db.execute(
                "SELECT actual_model FROM routing_decisions WHERE decision_id=?",
                (decision_id,)).fetchone()
        models = {row[0] for row in rows}
        if current and current[0]:
            models.add(current[0])
        return frozenset(models)

    async def record_running_model(self, decision: ExecutionDecision, model: str,
                                   provider: str, gateway_used: bool) -> None:
        await asyncio.to_thread(self._record_running_model, decision, model, provider,
                                gateway_used)

    def _record_running_model(self, decision, model, provider, gateway_used):
        with self._database() as db:
            db.execute("""UPDATE routing_decisions SET actual_model=?, actual_provider=?,
              gateway_used=?, execution_status='running' WHERE decision_id=?""",
              (model, provider, int(gateway_used), decision.decision_id))

    async def finalize(self, decision: ExecutionDecision, model: str, provider: str,
                       gateway_used: bool, status: str, latency_ms: int,
                       fallback_reason: str | None = None) -> None:
        await asyncio.to_thread(self._finalize, decision, model, provider, gateway_used,
                                status, latency_ms, fallback_reason)

    def _finalize(self, decision, model, provider, gateway_used, status, latency_ms, fallback_reason):
        with self._database() as db:
            db.execute("""UPDATE routing_decisions SET actual_model=?, actual_provider=?,
              fallback_reason=?, gateway_used=?, execution_status=?, latency_ms=? WHERE decision_id=?""", (
                model, provider, fallback_reason, int(gateway_used), status, latency_ms,
                decision.decision_id,
            ))

    async def finalize_by_id(self, decision_id: str, model: str, provider: str,
                             executor: str, gateway_used: bool, status: str,
                             latency_ms: int, usage: dict | None = None,
                             error: str | None = None) -> None:
        await asyncio.to_thread(self._finalize_by_id, decision_id, model, provider, executor,
                                gateway_used, status, latency_ms, usage, error)

    def _finalize_by_id(self, decision_id, model, provider, executor, gateway_used,
                        status, latency_ms, usage, error):
        with self._database() as db:
            db.execute("""INSERT INTO execution_attempts
              (decision_id, model, provider, executor, gateway_used, status, latency_ms, error, usage_json, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                decision_id, model, provider, executor, int(gateway_used), status,
                latency_ms, error, json.dumps(usage) if usage else None, time.time(),
            ))
            db.execute("""UPDATE routing_decisions SET actual_model=?, actual_provider=?,
              gateway_used=?, execution_status=?, latency_ms=? WHERE decision_id=?""", (
                model, provider, int(gateway_used), status, latency_ms, decision_id,
            ))

    async def create_implementation_job(self, job: ImplementationJob,
                                        tasks: list[ImplementationJobTask],
                                        conversation_id: str | None = None,
                                        project_id: str | None = None) -> None:
        await asyncio.to_thread(self._create_implementation_job, job, tasks,
                                conversation_id, project_id)

    def _create_implementation_job(self, job, tasks, conversation_id, project_id=None) -> None:
        now = time.time()
        with self._database() as db:
            db.execute("""INSERT INTO implementation_jobs
              (job_id, selectors_json, instruction, conversation_id, project_id, status,
               created_at, updated_at, plan_id)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                job.job_id, json.dumps(job.selectors), job.instruction, conversation_id,
                project_id, job.status, now, now, getattr(job, "plan_id", None),
            ))
            db.executemany("""INSERT INTO implementation_job_tasks
              (job_id, position, change_id, task_id, status, decision_id, error,
               revision_budget)
              VALUES (?, ?, ?, ?, ?, ?, ?, 3)""", [
                (task.job_id, task.position, task.change_id, task.task_id,
                 task.status, task.decision_id, task.error)
                for task in tasks
            ])

    async def next_implementation_task(self, job_id: str) -> ImplementationJobTask | None:
        return await asyncio.to_thread(self._next_implementation_task, job_id)

    def _next_implementation_task(self, job_id):
        with self._database() as db:
            row = db.execute("""SELECT job_id, position, change_id, task_id, status, decision_id, error
              FROM implementation_job_tasks WHERE job_id=? AND status='queued'
              ORDER BY position LIMIT 1""", (job_id,)).fetchone()
        return ImplementationJobTask(*row) if row else None

    async def implementation_task(self, job_id: str, position: int) -> ImplementationJobTask | None:
        return await asyncio.to_thread(self._implementation_task, job_id, position)

    def _implementation_task(self, job_id, position):
        with self._database() as db:
            row = db.execute("""SELECT job_id, position, change_id, task_id, status, decision_id, error
              FROM implementation_job_tasks WHERE job_id=? AND position=?""",
              (job_id, position)).fetchone()
        return ImplementationJobTask(*row) if row else None

    async def implementation_job_tasks(self, job_id: str) -> list[ImplementationJobTask]:
        return await asyncio.to_thread(self._implementation_job_tasks, job_id)

    def _implementation_job_tasks(self, job_id):
        with self._database() as db:
            rows = db.execute("""SELECT job_id, position, change_id, task_id, status,
              decision_id, error FROM implementation_job_tasks WHERE job_id=?
              ORDER BY position""", (job_id,)).fetchall()
        return [ImplementationJobTask(*row) for row in rows]

    async def implementation_job_selectors(self, job_id: str) -> tuple[str, ...]:
        return await asyncio.to_thread(self._implementation_job_selectors, job_id)

    def _implementation_job_selectors(self, job_id):
        with self._database() as db:
            row = db.execute("SELECT selectors_json FROM implementation_jobs WHERE job_id=?",
                             (job_id,)).fetchone()
        return tuple(json.loads(row[0])) if row else ()

    async def update_implementation_task(self, job_id: str, position: int, status: str,
                                         decision_id: str | None = None,
                                         error: str | None = None) -> None:
        await asyncio.to_thread(self._update_implementation_task, job_id, position,
                                status, decision_id, error)

    def _update_implementation_task(self, job_id, position, status, decision_id, error):
        with self._database() as db:
            db.execute("""UPDATE implementation_job_tasks SET status=?,
              decision_id=COALESCE(?, decision_id), error=?, review_started_at=NULL
              WHERE job_id=? AND position=?""", (
                status, decision_id, error, job_id, position,
            ))
            # A halted job keeps its status: late writes from a cancelled run's
            # request must not make it look active again.
            db.execute(f"""UPDATE implementation_jobs SET status=?, updated_at=?, error=?
              WHERE job_id=? AND status NOT IN ({_HALTED_SQL})""", (
                status if status in {"failed", "awaiting_confirmation", "review_pending",
                                     "awaiting_user_decision", "undone",
                                     *PAUSED_STATES} else "running",
                time.time(), error, job_id,
            ))

    async def claim_review(self, job_id: str, position: int,
                           allow_confirmation_retry: bool = False) -> bool:
        return await asyncio.to_thread(self._claim_review, job_id, position,
                                       allow_confirmation_retry)

    def _claim_review(self, job_id, position, allow_confirmation_retry=False):
        now = time.time()
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks
              SET status='reviewing', review_started_at=?, error=NULL, review_question=NULL
              WHERE job_id=? AND position=? AND (
                status IN ('running', 'waiting', 'implemented', 'review_pending',
                           'awaiting_evidence_resolution') OR
                (status IN ('awaiting_confirmation', 'awaiting_human_confirmation') AND ?) OR
                (status='reviewing' AND review_started_at<?))""",
                (now, job_id, position, int(allow_confirmation_retry), now - 15 * 60))
            if result.rowcount:
                db.execute("UPDATE implementation_jobs SET status='running', updated_at=?, error=NULL WHERE job_id=?",
                           (now, job_id))
            return bool(result.rowcount)

    async def actual_model_for(self, decision_id: str | None) -> str | None:
        if not decision_id:
            return None
        return await asyncio.to_thread(self._actual_model_for, decision_id)

    def _actual_model_for(self, decision_id):
        with self._database() as db:
            row = db.execute("SELECT actual_model FROM routing_decisions WHERE decision_id=?",
                             (decision_id,)).fetchone()
        return row[0] if row else None

    async def save_coder_report(self, job_id: str, position: int, report: str) -> None:
        await asyncio.to_thread(self._save_coder_report, job_id, position, report)

    def _save_coder_report(self, job_id, position, report):
        with self._database() as db:
            db.execute("UPDATE implementation_job_tasks SET coder_report=? WHERE job_id=? AND position=?",
                       (report, job_id, position))

    async def coder_report_for(self, job_id: str, position: int) -> str:
        return await asyncio.to_thread(self._coder_report_for, job_id, position)

    def _coder_report_for(self, job_id, position):
        with self._database() as db:
            row = db.execute("SELECT coder_report FROM implementation_job_tasks WHERE job_id=? AND position=?",
                             (job_id, position)).fetchone()
        return (row[0] or "") if row else ""

    async def review_count(self, job_id: str, position: int) -> int:
        return await asyncio.to_thread(self._review_count, job_id, position)

    def _review_count(self, job_id, position):
        with self._database() as db:
            row = db.execute("SELECT COUNT(*) FROM implementation_reviews WHERE job_id=? AND position=?",
                             (job_id, position)).fetchone()
        return row[0]

    async def revision_count(self, job_id: str, position: int) -> int:
        return await asyncio.to_thread(self._revision_count, job_id, position)

    def _revision_count(self, job_id, position):
        with self._database() as db:
            row = db.execute("""SELECT COUNT(*) FROM implementation_reviews
              WHERE job_id=? AND position=? AND verdict='needs_work'""",
                             (job_id, position)).fetchone()
        return row[0]

    async def revision_state(self, job_id: str, position: int) -> tuple[int, int]:
        return await asyncio.to_thread(self._revision_state, job_id, position)

    def _revision_state(self, job_id, position):
        with self._database() as db:
            row = db.execute("""SELECT revision_attempts, revision_budget
              FROM implementation_job_tasks WHERE job_id=? AND position=?""",
                             (job_id, position)).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    async def consume_revision(self, job_id: str, position: int) -> bool:
        return await asyncio.to_thread(self._consume_revision, job_id, position)

    def _consume_revision(self, job_id, position):
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks
              SET revision_attempts=revision_attempts+1, status='running', error=NULL
              WHERE job_id=? AND position=? AND revision_attempts < revision_budget""",
                                (job_id, position))
            return bool(result.rowcount)

    async def extend_revisions(self, job_id: str, position: int,
                               amount: int = 2) -> bool:
        return await asyncio.to_thread(self._extend_revisions, job_id, position, amount)

    def _extend_revisions(self, job_id, position, amount):
        now = time.time()
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks
              SET revision_budget=revision_budget+?, revision_attempts=revision_attempts+1,
                  status='running', error=NULL, review_started_at=NULL
              WHERE job_id=? AND position=?
                AND status IN ('awaiting_review_resolution', 'review_pending')""",
                                (amount, job_id, position))
            if result.rowcount:
                db.execute("""DELETE FROM implementation_evidence
                  WHERE job_id=? AND position=?""", (job_id, position))
                db.execute("""UPDATE implementation_jobs SET status='running',
                  updated_at=?, error=NULL WHERE job_id=?""", (now, job_id))
            return bool(result.rowcount)

    async def restart_revision_cycle(self, job_id: str, position: int,
                                     budget: int = 3) -> bool:
        return await asyncio.to_thread(
            self._restart_revision_cycle, job_id, position, budget)

    async def claim_coder_recovery(self, job_id: str, position: int) -> bool:
        return await asyncio.to_thread(self._claim_coder_recovery, job_id, position)

    def _claim_coder_recovery(self, job_id, position):
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks
              SET status='running', error=NULL WHERE job_id=? AND position=?
              AND status='awaiting_coder_recovery'""", (job_id, position))
            if result.rowcount:
                db.execute("""UPDATE implementation_jobs SET status='running',
                  error=NULL, updated_at=? WHERE job_id=?""", (time.time(), job_id))
            return bool(result.rowcount)

    async def claim_auto_coder_fallback(self, job_id: str, position: int,
                                        decision_id: str | None) -> bool:
        return await asyncio.to_thread(
            self._claim_auto_coder_fallback, job_id, position, decision_id)

    def _claim_auto_coder_fallback(self, job_id, position, decision_id):
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks
              SET status='retrying' WHERE job_id=? AND position=?
              AND decision_id IS ? AND status IN ('running', 'waiting')""",
              (job_id, position, decision_id))
            return bool(result.rowcount)

    def _restart_revision_cycle(self, job_id, position, budget):
        now = time.time()
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks
              SET revision_budget=?, revision_attempts=0, status='running', error=NULL,
                  review_started_at=NULL
              WHERE job_id=? AND position=? AND status IN
                ('awaiting_user_decision', 'awaiting_review_resolution',
                 'awaiting_coder_recovery')""",
                                (budget, job_id, position))
            if result.rowcount:
                db.execute("""DELETE FROM implementation_evidence
                  WHERE job_id=? AND position=?""", (job_id, position))
                db.execute("""UPDATE implementation_jobs SET status='running',
                  updated_at=?, error=NULL WHERE job_id=?""", (now, job_id))
            return bool(result.rowcount)

    async def claim_resolution(self, job_id: str, position: int, action: str) -> bool:
        return await asyncio.to_thread(self._claim_resolution, job_id, position, action)

    def _claim_resolution(self, job_id, position, action):
        target = ("accepting_override" if action == "accept_override" else
                  "accepting" if action == "accept" else "undoing")
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks SET status=?, error=NULL
              WHERE job_id=? AND position=?
                AND status IN ('awaiting_user_decision', 'awaiting_review_resolution',
                               'awaiting_coder_recovery')""",
                                (target, job_id, position))
            return bool(result.rowcount)

    async def record_resolution_action(self, job_id: str, position: int,
                                       action: str, details: dict | None = None) -> None:
        await asyncio.to_thread(self._record_resolution_action, job_id, position,
                                action, details)

    def _record_resolution_action(self, job_id, position, action, details):
        with self._database() as db:
            db.execute("""INSERT INTO implementation_resolution_actions
              (job_id, position, action, details_json, created_at)
              VALUES (?, ?, ?, ?, ?)""", (
                job_id, position, action,
                json.dumps(details) if details is not None else None, time.time(),
            ))

    async def plan_mutation(self, job_id: str, position: int, tool_name: str,
                            operation: str, targets: list[str]) -> int:
        return await asyncio.to_thread(
            self._plan_mutation, job_id, position, tool_name, operation, targets)

    def _plan_mutation(self, job_id, position, tool_name, operation, targets):
        now = time.time()
        with self._database() as db:
            cursor = db.execute("""INSERT INTO implementation_mutations
              (job_id, position, tool_name, operation, targets_json, stage,
               created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'planned', ?, ?)""", (
                job_id, position, tool_name, operation, json.dumps(targets), now, now,
            ))
            return int(cursor.lastrowid)

    async def advance_mutation(self, mutation_id: int, stage: str,
                               failure_stage: str | None = None,
                               failure_detail: str | None = None) -> None:
        await asyncio.to_thread(
            self._advance_mutation, mutation_id, stage, failure_stage, failure_detail)

    def _advance_mutation(self, mutation_id, stage, failure_stage, failure_detail):
        with self._database() as db:
            db.execute("""UPDATE implementation_mutations
              SET stage=?, failure_stage=?, failure_detail=?, updated_at=? WHERE id=?""",
                       (stage, failure_stage, failure_detail, time.time(), mutation_id))

    async def mutations(self, job_id: str, position: int) -> list[dict]:
        return await asyncio.to_thread(self._mutations, job_id, position)

    async def recent_mutations(self, project_id: str, since: float,
                               limit: int = 200) -> list[dict]:
        """Read-only, bounded mutation hints newer than the KB index."""
        return await asyncio.to_thread(self._recent_mutations, project_id, since, limit)

    def _recent_mutations(self, project_id, since, limit):
        with self._database() as db:
            rows = db.execute("""SELECT m.job_id, m.position, m.targets_json,
                m.updated_at FROM implementation_mutations m
                JOIN implementation_jobs j ON j.job_id=m.job_id
                WHERE j.project_id=? AND m.updated_at>?
                  AND m.stage IN ('applied', 'postimage_verified')
                ORDER BY m.updated_at DESC LIMIT ?""",
                (project_id, since, max(1, min(int(limit), 1000)))).fetchall()
        return [{"job_id": row[0], "position": row[1],
                 "targets": json.loads(row[2]), "updated_at": row[3]}
                for row in rows]

    async def implementation_job_project_id(self, job_id: str) -> str | None:
        return await asyncio.to_thread(self._implementation_job_project_id, job_id)

    def _implementation_job_project_id(self, job_id):
        with self._database() as db:
            row = db.execute("SELECT project_id FROM implementation_jobs WHERE job_id=?",
                             (job_id,)).fetchone()
        return row[0] if row else None

    def _mutations(self, job_id, position):
        with self._database() as db:
            rows = db.execute("""SELECT id, tool_name, operation, targets_json, stage,
              failure_stage, failure_detail FROM implementation_mutations
              WHERE job_id=? AND position=? ORDER BY id""",
                              (job_id, position)).fetchall()
        return [{
            "id": row[0], "tool": row[1], "operation": row[2],
            "targets": json.loads(row[3]), "stage": row[4],
            "failure_stage": row[5], "failure_detail": row[6],
        } for row in rows]

    async def reserve_evidence(self, job_id: str, position: int, kind: str,
                               tool_name: str, arguments: dict,
                               success: bool) -> int:
        return await asyncio.to_thread(
            self._reserve_evidence, job_id, position, kind, tool_name,
            arguments, success)

    def _reserve_evidence(self, job_id, position, kind, tool_name, arguments, success):
        with self._database() as db:
            row = db.execute("""SELECT revision_attempts
              FROM implementation_job_tasks WHERE job_id=? AND position=?""",
                             (job_id, position)).fetchone()
            attempt = int(row[0]) if row else 0
            cursor = db.execute("""INSERT INTO implementation_evidence
              (job_id, position, attempt, kind, tool_name, arguments_json, success, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
                job_id, position, attempt, kind, tool_name, json.dumps(arguments),
                int(success), time.time(),
            ))
            return int(cursor.lastrowid)

    async def complete_evidence(self, evidence_id: int, payload_path: str,
                                payload_hash: str, result_excerpt: str,
                                metadata: dict | None = None) -> None:
        await asyncio.to_thread(
            self._complete_evidence, evidence_id, payload_path,
            payload_hash, result_excerpt, metadata)

    def _complete_evidence(self, evidence_id, payload_path, payload_hash,
                           result_excerpt, metadata):
        with self._database() as db:
            db.execute("""UPDATE implementation_evidence
              SET payload_path=?, payload_hash=?, result_excerpt=?, metadata_json=?
              WHERE id=?""", (payload_path, payload_hash, result_excerpt,
                               json.dumps(metadata) if metadata is not None else None,
                               evidence_id))

    async def verification_evidence(self, job_id: str, position: int) -> list[dict]:
        return await asyncio.to_thread(self._verification_evidence, job_id, position)

    def _verification_evidence(self, job_id, position):
        with self._database() as db:
            rows = db.execute("""SELECT id, tool_name, arguments_json, success,
              payload_path, payload_hash, result_excerpt, metadata_json, created_at
              FROM implementation_evidence WHERE job_id=? AND position=?
                AND attempt=COALESCE((SELECT revision_attempts
                  FROM implementation_job_tasks WHERE job_id=? AND position=?), 0)
                AND kind='verification' ORDER BY id""",
                              (job_id, position, job_id, position)).fetchall()
        return [{
            "evidence_id": row[0], "tool": row[1],
            "arguments": json.loads(row[2]), "success": bool(row[3]),
            "payload_path": row[4], "sha256": row[5],
            "result_excerpt": row[6],
            "metadata": json.loads(row[7]) if row[7] else {},
            "created_at": row[8],
        } for row in rows]

    async def evidence_fingerprint(self, job_id: str, position: int) -> str:
        return await asyncio.to_thread(self._evidence_fingerprint, job_id, position)

    def _evidence_fingerprint(self, job_id, position):
        with self._database() as db:
            files = db.execute("""SELECT path, expected_hash
              FROM implementation_rollback_files WHERE job_id=? AND position=?
              ORDER BY path""", (job_id, position)).fetchall()
            evidence = db.execute("""SELECT id, success, payload_hash
              FROM implementation_evidence WHERE job_id=? AND position=?
                AND attempt=COALESCE((SELECT revision_attempts
                  FROM implementation_job_tasks WHERE job_id=? AND position=?), 0)
              ORDER BY id""", (job_id, position, job_id, position)).fetchall()
        encoded = json.dumps({"files": files, "evidence": evidence}, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def rollback_file(self, job_id: str, position: int, path: str) -> dict | None:
        return await asyncio.to_thread(self._rollback_file, job_id, position, path)

    def _rollback_file(self, job_id, position, path):
        with self._database() as db:
            row = db.execute("""SELECT before_exists, before_content, expected_hash
              FROM implementation_rollback_files
              WHERE job_id=? AND position=? AND path=?""",
                             (job_id, position, path)).fetchone()
        return ({"path": path, "before_exists": bool(row[0]),
                 "before_content": row[1], "expected_hash": row[2]}
                if row else None)

    async def record_rollback_file(self, job_id: str, position: int, path: str,
                                   before_exists: bool, before_content: str | None,
                                   expected_hash: str | None = None) -> None:
        await asyncio.to_thread(self._record_rollback_file, job_id, position, path,
                                before_exists, before_content, expected_hash)

    def _record_rollback_file(self, job_id, position, path, before_exists,
                              before_content, expected_hash):
        with self._database() as db:
            db.execute("""INSERT OR IGNORE INTO implementation_rollback_files
              (job_id, position, path, before_exists, before_content, expected_hash)
              VALUES (?, ?, ?, ?, ?, ?)""", (
                job_id, position, path, int(before_exists), before_content, expected_hash,
            ))

    async def update_rollback_hash(self, job_id: str, position: int, path: str,
                                   expected_hash: str | None) -> None:
        await asyncio.to_thread(self._update_rollback_hash, job_id, position,
                                path, expected_hash)

    def _update_rollback_hash(self, job_id, position, path, expected_hash):
        with self._database() as db:
            db.execute("""UPDATE implementation_rollback_files SET expected_hash=?
              WHERE job_id=? AND position=? AND path=?""",
                       (expected_hash, job_id, position, path))

    async def rollback_files(self, job_id: str, position: int) -> list[dict]:
        return await asyncio.to_thread(self._rollback_files, job_id, position)

    def _rollback_files(self, job_id, position):
        with self._database() as db:
            rows = db.execute("""SELECT path, before_exists, before_content, expected_hash
              FROM implementation_rollback_files WHERE job_id=? AND position=?
              ORDER BY rowid DESC""", (job_id, position)).fetchall()
        return [{"path": row[0], "before_exists": bool(row[1]),
                 "before_content": row[2], "expected_hash": row[3]}
                for row in rows]

    async def rollback_integrity_issues(self, job_id: str, position: int) -> list[dict]:
        return await asyncio.to_thread(
            self._rollback_integrity_issues, job_id, position)

    def _rollback_integrity_issues(self, job_id, position):
        actions = ("rollback_untracked_mutation", "rollback_snapshot_failed",
                   "rollback_postimage_failed")
        placeholders = ",".join("?" for _ in actions)
        with self._database() as db:
            rows = db.execute(f"""SELECT action, details_json
              FROM implementation_resolution_actions
              WHERE job_id=? AND position=? AND action IN ({placeholders})
                AND id > COALESCE((SELECT MAX(id) FROM implementation_resolution_actions
                  WHERE job_id=? AND position=?
                    AND action='recovery_baseline_established'), 0)
              ORDER BY id""", (job_id, position, *actions, job_id, position)).fetchall()
        return [{"action": action,
                 **(json.loads(details) if details else {})}
                for action, details in rows]

    async def establish_recovery_baseline(self, job_id: str, position: int,
                                          budget: int, guidance: str) -> bool:
        return await asyncio.to_thread(
            self._establish_recovery_baseline, job_id, position, budget, guidance)

    def _establish_recovery_baseline(self, job_id, position, budget, guidance):
        now = time.time()
        with self._database() as db:
            row = db.execute("""SELECT task_id FROM implementation_job_tasks
              WHERE job_id=? AND position=? AND status='recovery_required'""",
                             (job_id, position)).fetchone()
            if row is None:
                return False
            db.execute("""INSERT INTO implementation_resolution_actions
              (job_id, position, action, details_json, created_at)
              VALUES (?, ?, 'recovery_baseline_established', ?, ?)""", (
                job_id, position, json.dumps({
                    "guidance": guidance,
                    "warning": "The prior uncertain mutations were accepted as the new undo baseline.",
                }), now,
            ))
            db.execute("""DELETE FROM implementation_rollback_files
              WHERE job_id=? AND position=?""", (job_id, position))
            db.execute("""DELETE FROM implementation_evidence
              WHERE job_id=? AND position=?""", (job_id, position))
            db.execute("""UPDATE implementation_job_tasks
              SET status='running', error=NULL, review_started_at=NULL,
                  revision_attempts=0, revision_budget=?
              WHERE job_id=? AND position=?""", (budget, job_id, position))
            db.execute("""UPDATE implementation_jobs
              SET status='running', error=NULL, updated_at=? WHERE job_id=?""",
                       (now, job_id))
            return True

    async def stop_recovery(self, job_id: str, position: int) -> bool:
        return await asyncio.to_thread(self._stop_recovery, job_id, position)

    def _stop_recovery(self, job_id, position):
        now = time.time()
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks
              SET status='recovery_stopped', error='manual recovery requested'
              WHERE job_id=? AND position=? AND status='recovery_required'""",
                                (job_id, position))
            if not result.rowcount:
                return False
            db.execute("""UPDATE implementation_jobs
              SET status='stopped', error='manual recovery requested', updated_at=?
              WHERE job_id=?""", (now, job_id))
            db.execute("""INSERT INTO implementation_resolution_actions
              (job_id, position, action, details_json, created_at)
              VALUES (?, ?, 'recovery_stopped', ?, ?)""", (
                job_id, position, json.dumps({
                    "warning": "No automatic undo was attempted; manual repair is required.",
                }), now,
            ))
            return True

    async def record_review(self, job_id: str, position: int, coder_model: str,
                            reviewer_model: str, verdict: str, rationale: str,
                            question: str | None = None,
                            details: dict | None = None) -> None:
        await asyncio.to_thread(self._record_review, job_id, position, coder_model,
                                reviewer_model, verdict, rationale, question, details)

    async def claim_kb_refresh(self, job_id: str, wp_id: str, kb_project_id: str) -> bool:
        return await asyncio.to_thread(self._claim_kb_refresh, job_id, wp_id, kb_project_id)

    def _claim_kb_refresh(self, job_id, wp_id, kb_project_id):
        with self._database() as db:
            now = time.time()
            cursor = db.execute("""INSERT OR IGNORE INTO kb_refreshes
              (job_id, wp_id, kb_project_id, status, created_at, updated_at)
              VALUES (?, ?, ?, 'running', ?, ?)""",
              (job_id, wp_id, kb_project_id, now, now))
            return cursor.rowcount == 1

    async def finish_kb_refresh(self, job_id: str, wp_id: str, status: str,
                                details: dict) -> None:
        await asyncio.to_thread(self._finish_kb_refresh, job_id, wp_id, status, details)

    def _finish_kb_refresh(self, job_id, wp_id, status, details):
        with self._database() as db:
            db.execute("""UPDATE kb_refreshes SET status=?, details_json=?, updated_at=?
              WHERE job_id=? AND wp_id=?""", (
                status, json.dumps(details, ensure_ascii=False), time.time(), job_id, wp_id))

    def _record_review(self, job_id, position, coder_model, reviewer_model, verdict,
                       rationale, question, details):
        with self._database() as db:
            db.execute("""INSERT INTO implementation_reviews
              (job_id, position, coder_model, reviewer_model, verdict, rationale, question,
               details_json, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                job_id, position, coder_model, reviewer_model, verdict, rationale, question,
                json.dumps(details) if details is not None else None, time.time(),
            ))
            db.execute("UPDATE implementation_job_tasks SET review_question=? WHERE job_id=? AND position=?",
                       (question, job_id, position))

    async def latest_review(self, job_id: str, position: int) -> dict | None:
        return await asyncio.to_thread(self._latest_review, job_id, position)

    def _latest_review(self, job_id, position):
        with self._database() as db:
            row = db.execute("""SELECT coder_model, reviewer_model, verdict, rationale,
              question, details_json FROM implementation_reviews WHERE job_id=? AND position=?
              ORDER BY id DESC LIMIT 1""", (job_id, position)).fetchone()
        if row is None:
            return None
        return {
            "coder_model": row[0], "reviewer_model": row[1], "verdict": row[2],
            "rationale": row[3], "question": row[4],
            "details": json.loads(row[5]) if row[5] else None,
        }

    async def pending_confirmation(self, conversation_id: str | None):
        if not conversation_id:
            return None
        return await asyncio.to_thread(self._pending_confirmation, conversation_id)

    def _pending_confirmation(self, conversation_id):
        with self._database() as db:
            return db.execute("""SELECT t.job_id, t.position, t.task_id, t.review_question,
              j.instruction FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id WHERE j.conversation_id=? AND t.status='awaiting_confirmation'
              ORDER BY j.updated_at DESC LIMIT 1""", (conversation_id,)).fetchone()

    async def pending_confirmation_for_job(self, job_id: str):
        return await asyncio.to_thread(self._pending_confirmation_for_job, job_id)

    def _pending_confirmation_for_job(self, job_id):
        with self._database() as db:
            return db.execute("""SELECT t.job_id, t.position, t.task_id, t.review_question,
              j.instruction FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id WHERE t.job_id=? AND t.status='awaiting_confirmation'
              ORDER BY t.position LIMIT 1""", (job_id,)).fetchone()

    async def pending_review(self, conversation_id: str | None):
        if not conversation_id:
            return None
        return await asyncio.to_thread(self._pending_review, conversation_id)

    def _pending_review(self, conversation_id):
        with self._database() as db:
            return db.execute("""SELECT t.job_id, t.position, t.task_id, j.instruction
              FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id WHERE j.conversation_id=? AND (
                t.status IN ('review_pending', 'awaiting_confirmation') OR
                (t.status='reviewing' AND t.review_started_at<?))
              ORDER BY j.updated_at DESC LIMIT 1""",
              (conversation_id, time.time() - 15 * 60)).fetchone()

    async def pending_review_for_job(self, job_id: str):
        return await asyncio.to_thread(self._pending_review_for_job, job_id)

    def _pending_review_for_job(self, job_id):
        with self._database() as db:
            return db.execute("""SELECT t.job_id, t.position, t.task_id, j.instruction
              FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id WHERE t.job_id=? AND (
                t.status IN ('review_pending', 'awaiting_confirmation') OR
                (t.status='reviewing' AND t.review_started_at<?))
              ORDER BY t.position LIMIT 1""",
              (job_id, time.time() - 15 * 60)).fetchone()

    async def pending_recovery(self, conversation_id: str | None):
        if not conversation_id:
            return None
        return await asyncio.to_thread(self._pending_recovery, conversation_id)

    def _pending_recovery(self, conversation_id):
        with self._database() as db:
            return db.execute("""SELECT t.job_id, t.position, t.task_id, j.instruction
              FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id WHERE j.conversation_id=?
                AND t.status='recovery_required'
              ORDER BY j.updated_at DESC LIMIT 1""", (conversation_id,)).fetchone()

    async def pending_recovery_for_job(self, job_id: str):
        return await asyncio.to_thread(self._pending_recovery_for_job, job_id)

    def _pending_recovery_for_job(self, job_id):
        with self._database() as db:
            return db.execute("""SELECT t.job_id, t.position, t.task_id, j.instruction
              FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id WHERE t.job_id=?
                AND t.status='recovery_required'
              ORDER BY t.position LIMIT 1""", (job_id,)).fetchone()

    async def pending_resolution(self, conversation_id: str | None):
        if not conversation_id:
            return None
        return await asyncio.to_thread(self._pending_resolution, conversation_id)

    def _pending_resolution(self, conversation_id):
        with self._database() as db:
            return db.execute("""SELECT t.job_id, t.position, t.task_id, j.instruction
              FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id WHERE j.conversation_id=?
                AND t.status IN ('awaiting_user_decision', 'awaiting_review_resolution',
                                 'awaiting_coder_recovery')
              ORDER BY j.updated_at DESC LIMIT 1""", (conversation_id,)).fetchone()

    async def pending_resolution_for_job(self, job_id: str):
        return await asyncio.to_thread(self._pending_resolution_for_job, job_id)

    def _pending_resolution_for_job(self, job_id):
        with self._database() as db:
            return db.execute("""SELECT t.job_id, t.position, t.task_id, j.instruction
              FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id WHERE t.job_id=?
                AND t.status IN ('awaiting_user_decision', 'awaiting_review_resolution',
                                 'awaiting_coder_recovery')
              ORDER BY t.position LIMIT 1""", (job_id,)).fetchone()

    async def finish_implementation_job(self, job_id: str, status: str = "completed",
                                        error: str | None = None) -> None:
        await asyncio.to_thread(self._finish_implementation_job, job_id, status, error)

    def _finish_implementation_job(self, job_id, status, error):
        with self._database() as db:
            db.execute("UPDATE implementation_jobs SET status=?, updated_at=?, error=? WHERE job_id=?", (
                status, time.time(), error, job_id,
            ))

    async def record_blocked_operation(self, operation_id: str, operation: str, reason: str) -> None:
        await asyncio.to_thread(self._record_blocked_operation, operation_id, operation, reason)

    def _record_blocked_operation(self, operation_id, operation, reason):
        with self._database() as db:
            db.execute("""INSERT OR REPLACE INTO blocked_operations
              (operation_id, operation, status, reason, created_at) VALUES (?, ?, 'blocked', ?, ?)""", (
                operation_id, operation, reason, time.time(),
            ))

    async def create_refinement(self, session_id: str, conversation_id: str,
                                target_id: str, target_kind: str,
                                change_id: str, change: dict) -> None:
        await asyncio.to_thread(
            self._create_refinement, session_id, conversation_id, target_id,
            target_kind, change_id, change)

    def _create_refinement(self, session_id, conversation_id, target_id,
                           target_kind, change_id, change):
        serialized = json.dumps(change, sort_keys=True, ensure_ascii=False)
        fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
        now = time.time()
        with self._database() as db:
            db.execute("UPDATE refinement_sessions SET status='superseded', updated_at=? "
                       "WHERE conversation_id=? AND status='active'", (now, conversation_id))
            db.execute("""INSERT INTO refinement_sessions
              (session_id, conversation_id, target_id, target_kind, change_id,
               base_fingerprint, base_json, draft_json, status, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""", (
                session_id, conversation_id, target_id, target_kind, change_id,
                fingerprint, serialized, serialized, now, now,
            ))

    async def active_refinement(self, conversation_id: str | None) -> dict | None:
        if not conversation_id:
            return None
        return await asyncio.to_thread(self._active_refinement, conversation_id)

    def _active_refinement(self, conversation_id):
        with self._database() as db:
            row = db.execute("""SELECT session_id, target_id, target_kind, change_id,
              base_fingerprint, base_json, draft_json FROM refinement_sessions
              WHERE conversation_id=? AND status='active' ORDER BY updated_at DESC LIMIT 1""",
                             (conversation_id,)).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0], "target_id": row[1], "target_kind": row[2],
            "change_id": row[3], "base_fingerprint": row[4],
            "base": json.loads(row[5]), "draft": json.loads(row[6]),
        }

    async def update_refinement(self, session_id: str, draft: dict) -> None:
        await asyncio.to_thread(self._update_refinement, session_id, draft)

    def _update_refinement(self, session_id, draft):
        with self._database() as db:
            db.execute("UPDATE refinement_sessions SET draft_json=?, updated_at=? "
                       "WHERE session_id=? AND status='active'", (
                json.dumps(draft, sort_keys=True, ensure_ascii=False), time.time(), session_id,
            ))

    async def finish_refinement(self, session_id: str, status: str = "applied") -> None:
        await asyncio.to_thread(self._finish_refinement, session_id, status)

    def _finish_refinement(self, session_id, status):
        with self._database() as db:
            db.execute("UPDATE refinement_sessions SET status=?, updated_at=? WHERE session_id=?",
                       (status, time.time(), session_id))

    # --- WP planning ---------------------------------------------------------

    _PLANNING_COLUMNS = (
        "session_id", "conversation_id", "project_id", "change_id", "wp_id",
        "wp_fingerprint", "wp_complexity", "complexity_source", "planner_model",
        "planner_effort", "status", "revision", "draft_json", "discussion_json",
        "gaps_json", "queued_wp_ids_json", "request", "created_at", "updated_at",
    )

    async def create_planning(self, session: dict) -> None:
        """Start a planning session; any other active one in the conversation is superseded."""
        await asyncio.to_thread(self._create_planning, session)

    def _create_planning(self, session):
        now = time.time()
        with self._database() as db:
            db.execute("UPDATE wp_planning_sessions SET status='superseded', updated_at=? "
                       "WHERE conversation_id=? AND status='active'",
                       (now, session["conversation_id"]))
            db.execute("""INSERT INTO wp_planning_sessions
              (session_id, conversation_id, project_id, change_id, wp_id, wp_fingerprint,
               wp_complexity, complexity_source, planner_model, planner_effort, status,
               revision, draft_json, discussion_json, gaps_json, queued_wp_ids_json, request,
               created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 0, NULL, '[]', '[]', ?, ?, ?, ?)""", (
                session["session_id"], session["conversation_id"], session.get("project_id"),
                session["change_id"], session["wp_id"], session["wp_fingerprint"],
                session["wp_complexity"], session["complexity_source"],
                session["planner_model"], session.get("planner_effort"),
                json.dumps(list(session.get("queued_wp_ids") or ())), session.get("request"),
                now, now,
            ))

    def _planning_row(self, row) -> dict | None:
        if row is None:
            return None
        value = dict(zip(self._PLANNING_COLUMNS, row))
        value["draft"] = json.loads(value.pop("draft_json")) if value["draft_json"] else None
        value["discussion"] = json.loads(value.pop("discussion_json") or "[]")
        value["gaps"] = json.loads(value.pop("gaps_json") or "[]")
        value["queued_wp_ids"] = json.loads(value.pop("queued_wp_ids_json") or "[]")
        return value

    async def active_planning(self, conversation_id: str | None) -> dict | None:
        if not conversation_id:
            return None
        return await asyncio.to_thread(self._active_planning, conversation_id)

    def _active_planning(self, conversation_id):
        with self._database() as db:
            row = db.execute(
                f"SELECT {', '.join(self._PLANNING_COLUMNS)} FROM wp_planning_sessions "
                "WHERE conversation_id=? AND status='active' "
                "ORDER BY updated_at DESC LIMIT 1", (conversation_id,)).fetchone()
        return self._planning_row(row)

    async def planning_session(self, session_id: str) -> dict | None:
        return await asyncio.to_thread(self._planning_session, session_id)

    def _planning_session(self, session_id):
        with self._database() as db:
            row = db.execute(
                f"SELECT {', '.join(self._PLANNING_COLUMNS)} FROM wp_planning_sessions "
                "WHERE session_id=?", (session_id,)).fetchone()
        return self._planning_row(row)

    async def update_planning(self, session_id: str, revision: int, draft: dict,
                              discussion: list[dict], gaps: list[str]) -> None:
        await asyncio.to_thread(self._update_planning, session_id, revision, draft,
                                discussion, gaps)

    def _update_planning(self, session_id, revision, draft, discussion, gaps):
        with self._database() as db:
            db.execute("""UPDATE wp_planning_sessions SET revision=?, draft_json=?,
              discussion_json=?, gaps_json=?, updated_at=?
              WHERE session_id=? AND status='active'""", (
                revision, json.dumps(draft, ensure_ascii=False),
                json.dumps(discussion, ensure_ascii=False), json.dumps(gaps),
                time.time(), session_id,
            ))

    async def update_planning_model(self, session_id: str, previous: str,
                                    model: str, effort: str | None) -> None:
        """Save the provider that actually produced the plan before a repair turn."""
        changed = await asyncio.to_thread(self._update_planning_model, session_id,
                                          previous, model, effort)
        if not changed:
            raise RuntimeError("planning session changed while switching planner models")

    def _update_planning_model(self, session_id, previous, model, effort):
        with self._database() as db:
            result = db.execute("""UPDATE wp_planning_sessions
              SET planner_model=?, planner_effort=?, updated_at=?
              WHERE session_id=? AND status='active' AND planner_model=?""",
                                (model, effort, time.time(), session_id, previous))
            return result.rowcount == 1

    async def finish_planning(self, session_id: str, status: str) -> None:
        await asyncio.to_thread(self._finish_planning, session_id, status)

    def _finish_planning(self, session_id, status):
        with self._database() as db:
            db.execute("UPDATE wp_planning_sessions SET status=?, updated_at=? "
                       "WHERE session_id=?", (status, time.time(), session_id))

    async def record_planning_turn(self, turn: dict) -> None:
        await asyncio.to_thread(self._record_planning_turn, turn)

    def _record_planning_turn(self, turn):
        with self._database() as db:
            db.execute("""INSERT INTO wp_planning_turns
              (session_id, revision, kind, user_message, output_json, model, usage_json,
               cost_usd, tool_metrics_json, latency_ms, error, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                turn["session_id"], turn["revision"], turn["kind"], turn.get("user_message"),
                turn.get("output"), turn["model"],
                json.dumps(turn["usage"]) if turn.get("usage") is not None else None,
                turn.get("cost_usd"), json.dumps(turn.get("tool_metrics") or {}),
                turn.get("latency_ms"), turn.get("error"), time.time(),
            ))

    async def planning_turns(self, session_id: str) -> list[dict]:
        return await asyncio.to_thread(self._planning_turns, session_id)

    def _planning_turns(self, session_id):
        with self._database() as db:
            rows = db.execute("""SELECT revision, kind, model, usage_json, cost_usd,
              tool_metrics_json, latency_ms, error FROM wp_planning_turns
              WHERE session_id=? ORDER BY id""", (session_id,)).fetchall()
        return [{"revision": row[0], "kind": row[1], "model": row[2],
                 "usage": json.loads(row[3]) if row[3] else None, "cost_usd": row[4],
                 "tool_metrics": json.loads(row[5] or "{}"), "latency_ms": row[6],
                 "error": row[7]} for row in rows]

    async def save_approved_plan(self, plan: dict) -> None:
        await asyncio.to_thread(self._save_approved_plan, plan)

    def _save_approved_plan(self, plan):
        with self._database() as db:
            db.execute("""INSERT OR REPLACE INTO wp_approved_plans
              (plan_id, session_id, change_id, wp_id, plan_json, fingerprint, approved_at)
              VALUES (?, ?, ?, ?, ?, ?, ?)""", (
                plan["plan_id"], plan["session_id"], plan["change_id"], plan["wp_id"],
                json.dumps(plan, ensure_ascii=False), plan["fingerprint"], plan["approved_at"],
            ))

    async def approved_plan(self, plan_id: str | None) -> dict | None:
        if not plan_id:
            return None
        return await asyncio.to_thread(self._approved_plan, plan_id)

    def _approved_plan(self, plan_id):
        with self._database() as db:
            row = db.execute("SELECT plan_json FROM wp_approved_plans WHERE plan_id=?",
                             (plan_id,)).fetchone()
        return json.loads(row[0]) if row else None

    async def job_plan_id(self, job_id: str | None) -> str | None:
        if not job_id:
            return None
        return await asyncio.to_thread(self._job_plan_id, job_id)

    def _job_plan_id(self, job_id):
        with self._database() as db:
            row = db.execute("SELECT plan_id FROM implementation_jobs WHERE job_id=?",
                             (job_id,)).fetchone()
        return row[0] if row else None

    async def record_reviewer_context_event(self, plan_id: str, job_id: str, position: int,
                                            task_id: str, kind: str, summary: str,
                                            payload=None) -> None:
        await asyncio.to_thread(self._record_reviewer_context_event, plan_id, job_id,
                                position, task_id, kind, summary, payload)

    def _record_reviewer_context_event(self, plan_id, job_id, position, task_id, kind,
                                       summary, payload):
        with self._database() as db:
            db.execute("""INSERT INTO wp_reviewer_context_events
              (plan_id, job_id, position, task_id, kind, summary, payload_json, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
                plan_id, job_id, position, task_id, kind, summary,
                json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                time.time(),
            ))

    async def reviewer_context_events(self, plan_id: str, job_id: str) -> list[dict]:
        return await asyncio.to_thread(self._reviewer_context_events, plan_id, job_id)

    def _reviewer_context_events(self, plan_id, job_id):
        with self._database() as db:
            rows = db.execute("""SELECT position, task_id, kind, summary, payload_json
              FROM wp_reviewer_context_events WHERE plan_id=? AND job_id=?
              ORDER BY id""", (plan_id, job_id)).fetchall()
        return [{"position": row[0], "task_id": row[1], "kind": row[2], "summary": row[3],
                 "payload": json.loads(row[4]) if row[4] else None} for row in rows]

    async def paused_tasks(self, conversation_id: str | None,
                           referenced_job_ids: tuple[str, ...] = ()) -> list[dict]:
        return await asyncio.to_thread(self._paused_tasks, conversation_id,
                                       referenced_job_ids)

    async def running_jobs(self, conversation_id: str | None) -> tuple[str, ...]:
        if not conversation_id:
            return ()
        return await asyncio.to_thread(self._running_jobs, conversation_id)

    async def record_offered_choices(self, conversation_id: str,
                                     choices: dict[int, str]) -> None:
        if choices:
            await asyncio.to_thread(self._record_offered_choices, conversation_id, choices)

    def _record_offered_choices(self, conversation_id, choices):
        now = time.time()
        with self._database() as db:
            db.execute("UPDATE conversation_offered_choices SET status='superseded' "
                       "WHERE conversation_id=? AND status='active'", (conversation_id,))
            db.executemany("""INSERT INTO conversation_offered_choices
              (conversation_id, option_number, choice_text, status, created_at, expires_at)
              VALUES (?, ?, ?, 'active', ?, ?)
              ON CONFLICT(conversation_id, option_number) DO UPDATE SET
                choice_text=excluded.choice_text, status='active',
                created_at=excluded.created_at, expires_at=excluded.expires_at""",
              [(conversation_id, number, label[:500], now, now + 30 * 60)
               for number, label in choices.items()])

    async def offered_choice(self, conversation_id: str, number: int) -> str | None:
        return await asyncio.to_thread(self._offered_choice, conversation_id, number)

    def _offered_choice(self, conversation_id, number):
        with self._database() as db:
            row = db.execute("""SELECT choice_text FROM conversation_offered_choices
              WHERE conversation_id=? AND option_number=? AND status='active'
                AND expires_at>?""", (conversation_id, number, time.time())).fetchone()
        return row[0] if row else None

    async def implementation_job_status(self, job_id: str) -> str | None:
        return await asyncio.to_thread(self._implementation_job_status, job_id)

    def _implementation_job_status(self, job_id):
        with self._database() as db:
            row = db.execute("SELECT status FROM implementation_jobs WHERE job_id=?",
                             (job_id,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _scope_clause(project_id: str | None) -> str:
        """Jobs in this project, plus this conversation's jobs from before projects."""
        if project_id:
            return "(j.project_id=? OR (j.project_id IS NULL AND j.conversation_id=?))"
        return "j.conversation_id=?"

    @staticmethod
    def _scope_args(project_id: str | None, conversation_id: str | None) -> tuple:
        return (project_id, conversation_id) if project_id else (conversation_id,)

    async def project_jobs(self, project_id: str | None,
                           conversation_id: str | None) -> list[dict]:
        """Unfinished jobs in the project (active or paused), from any conversation."""
        if not project_id and not conversation_id:
            return []
        return await asyncio.to_thread(self._project_jobs, project_id, conversation_id)

    async def unfinished_job_for_wp(self, project_id: str | None,
                                    conversation_id: str | None,
                                    wp_id: str) -> str | None:
        """Find an unfinished job containing any task owned by this work package."""
        if not project_id and not conversation_id:
            raise ValueError("a project or conversation is required to check active jobs")
        return await asyncio.to_thread(
            self._unfinished_job_for_wp, project_id, conversation_id, wp_id)

    def _unfinished_job_for_wp(self, project_id, conversation_id, wp_id):
        terminal = tuple(sorted(TERMINAL_JOB_STATUSES))
        with self._database() as db:
            row = db.execute(f"""SELECT j.job_id FROM implementation_jobs j
              WHERE j.status NOT IN ({','.join('?' for _ in terminal)})
                AND {self._scope_clause(project_id)}
                AND EXISTS (SELECT 1 FROM implementation_job_tasks t
                            WHERE t.job_id=j.job_id AND t.task_id LIKE ?)
              ORDER BY j.updated_at DESC LIMIT 1""",
              (*terminal, *self._scope_args(project_id, conversation_id),
               wp_id + "-T%")).fetchone()
        return row[0] if row else None

    def _project_jobs(self, project_id, conversation_id):
        statuses = (*ACTIVE_JOB_STATUSES, *PAUSED_STATES, "review_pending",
                    "awaiting_confirmation", "awaiting_user_decision")
        with self._database() as db:
            rows = db.execute(f"""SELECT j.job_id, j.conversation_id, j.project_id, j.status,
                j.selectors_json, j.updated_at,
                (SELECT t.task_id FROM implementation_job_tasks t
                 WHERE t.job_id=j.job_id AND t.status NOT IN ('completed', 'queued', 'undone')
                 ORDER BY t.position LIMIT 1)
              FROM implementation_jobs j
              WHERE j.status IN ({",".join("?" for _ in statuses)})
                AND {self._scope_clause(project_id)}
              ORDER BY j.updated_at DESC""",
              (*statuses, *self._scope_args(project_id, conversation_id))).fetchall()
        return [{"job_id": row[0], "conversation_id": row[1], "project_id": row[2],
                 "status": row[3], "selectors": tuple(json.loads(row[4])),
                 "updated_at": row[5], "task_id": row[6]} for row in rows]

    def _active_job_ids(self, db, project_id, conversation_id, exclude_job_id=None):
        rows = db.execute(f"""SELECT j.job_id FROM implementation_jobs j
          WHERE j.status IN ({_ACTIVE_SQL}) AND j.job_id != ?
            AND {self._scope_clause(project_id)}""",
          (exclude_job_id or "", *self._scope_args(project_id, conversation_id))).fetchall()
        return {row[0] for row in rows}

    # --- flow switches (stop one job, then start another) -------------------

    async def switch_for_proposal(self, proposal_id: str) -> dict | None:
        return await asyncio.to_thread(self._switch_for_proposal, proposal_id)

    def _switch_for_proposal(self, proposal_id):
        with self._database() as db:
            row = db.execute("SELECT * FROM workflow_switches WHERE proposal_id=?",
                             (proposal_id,)).fetchone()
            columns = [item[0] for item in db.execute(
                "SELECT * FROM workflow_switches LIMIT 0").description]
        return self._switch_dict(columns, row)

    @staticmethod
    def _switch_dict(columns, row) -> dict | None:
        if row is None:
            return None
        value = dict(zip(columns, row))
        value["displaced"] = json.loads(value.pop("displaced_json"))
        return value

    async def advance_switch(self, switch_id: str, from_states: tuple[str, ...],
                             state: str, detail: str | None = None) -> bool:
        """Move a switch forward only from an expected state, so retries are idempotent."""
        return await asyncio.to_thread(self._advance_switch, switch_id, from_states,
                                       state, detail)

    def _advance_switch(self, switch_id, from_states, state, detail):
        with self._database() as db:
            result = db.execute(f"""UPDATE workflow_switches
              SET state=?, detail=COALESCE(?, detail), updated_at=?
              WHERE switch_id=? AND state IN ({",".join("?" for _ in from_states)})""",
              (state, detail, time.time(), switch_id, *from_states))
        return bool(result.rowcount)

    async def unfinished_switches(self) -> list[dict]:
        return await asyncio.to_thread(self._unfinished_switches)

    def _unfinished_switches(self):
        with self._database() as db:
            rows = db.execute(f"""SELECT * FROM workflow_switches WHERE state IN
              ({",".join("?" for _ in UNFINISHED_SWITCH_STATES)})""",
              UNFINISHED_SWITCH_STATES).fetchall()
            columns = [item[0] for item in db.execute(
                "SELECT * FROM workflow_switches LIMIT 0").description]
        return [self._switch_dict(columns, row) for row in rows]

    async def take_switch_notices(self, conversation_id: str) -> list[dict]:
        """Interrupted switches this conversation has not been told about yet."""
        return await asyncio.to_thread(self._take_switch_notices, conversation_id)

    def _take_switch_notices(self, conversation_id):
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("""SELECT * FROM workflow_switches WHERE conversation_id=?
              AND state='interrupted' AND reported=0""", (conversation_id,)).fetchall()
            columns = [item[0] for item in db.execute(
                "SELECT * FROM workflow_switches LIMIT 0").description]
            db.execute("""UPDATE workflow_switches SET reported=1 WHERE conversation_id=?
              AND state='interrupted' AND reported=0""", (conversation_id,))
        return [self._switch_dict(columns, row) for row in rows]

    async def mark_job_stopping(self, job_id: str) -> str | None:
        """Mark an active job as stopping; return its status before the change."""
        return await asyncio.to_thread(self._mark_job_stopping, job_id)

    def _mark_job_stopping(self, job_id):
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM implementation_jobs WHERE job_id=?",
                             (job_id,)).fetchone()
            if row is None:
                return None
            if row[0] in ACTIVE_JOB_STATUSES and row[0] != "stopping":
                db.execute("UPDATE implementation_jobs SET status='stopping', updated_at=? "
                           "WHERE job_id=?", (time.time(), job_id))
            return row[0]

    async def finalize_stopped_job(self, job_id: str, reason: str) -> None:
        """Record a stopped job. Files are left as they are; nothing is rolled back."""
        await asyncio.to_thread(self._finalize_stopped_job, job_id, reason)

    def _finalize_stopped_job(self, job_id, reason):
        now = time.time()
        with self._database() as db:
            db.execute("""UPDATE implementation_job_tasks SET status='stopped', error=?
              WHERE job_id=? AND status IN ('running', 'waiting', 'reviewing', 'undoing')""",
              (reason, job_id))
            db.execute(f"""UPDATE implementation_jobs SET status='stopped', error=?, updated_at=?
              WHERE job_id=? AND status IN ({_ACTIVE_SQL})""", (reason, now, job_id))

    async def set_switched_job_status(self, job_id: str, status: str) -> None:
        """Release (`queued`) or cancel (`cancelled`) a job created by a switch."""
        await asyncio.to_thread(self._set_switched_job_status, job_id, status)

    def _set_switched_job_status(self, job_id, status):
        with self._database() as db:
            db.execute("""UPDATE implementation_jobs SET status=?, updated_at=?
              WHERE job_id=? AND status='pending_switch'""", (status, time.time(), job_id))
            if status == "cancelled":
                db.execute("""UPDATE implementation_job_tasks SET status='cancelled'
                  WHERE job_id=? AND status='queued'""", (job_id,))

    def _running_jobs(self, conversation_id):
        with self._database() as db:
            rows = db.execute("""SELECT job_id FROM implementation_jobs
              WHERE conversation_id=? AND status IN ('queued', 'running', 'waiting',
                'reviewing', 'implemented') ORDER BY updated_at DESC""",
              (conversation_id,)).fetchall()
        return tuple(row[0] for row in rows)

    def _paused_tasks(self, conversation_id, referenced_job_ids):
        from agent.workflow.states import PAUSED_STATES
        if not conversation_id and not referenced_job_ids:
            return []
        statuses = (*PAUSED_STATES, "review_pending", "awaiting_confirmation")
        placeholders = ",".join("?" for _ in statuses)
        with self._database() as db:
            rows = db.execute(f"""SELECT t.job_id, t.position, t.task_id,
              CASE t.status WHEN 'review_pending' THEN 'awaiting_evidence_resolution'
                WHEN 'reviewing' THEN 'awaiting_evidence_resolution'
                WHEN 'awaiting_confirmation' THEN 'awaiting_human_confirmation'
                ELSE t.status END,
              t.error, t.review_question, j.instruction,
              (SELECT r.verdict FROM implementation_reviews r
               WHERE r.job_id=t.job_id AND r.position=t.position
               ORDER BY r.id DESC LIMIT 1) AS latest_verdict
              FROM implementation_job_tasks t JOIN implementation_jobs j
              ON j.job_id=t.job_id
              WHERE (t.status IN ({placeholders}) OR
                     (t.status='reviewing' AND t.review_started_at<?)) AND
              (j.conversation_id=? OR t.job_id IN ({','.join('?' for _ in referenced_job_ids) or "''"}))
              ORDER BY j.updated_at DESC, t.position""",
              (*statuses, time.time() - 15 * 60, conversation_id,
               *referenced_job_ids)).fetchall()
        return [dict(zip(("job_id", "position", "task_id", "status", "error",
                          "review_question", "instruction", "latest_verdict"), row))
                for row in rows]

    async def implementation_job_instruction(self, job_id: str | None) -> str:
        if not job_id:
            return ""
        return await asyncio.to_thread(self._implementation_job_instruction, job_id)

    def _implementation_job_instruction(self, job_id):
        with self._database() as db:
            row = db.execute("SELECT instruction FROM implementation_jobs WHERE job_id=?",
                             (job_id,)).fetchone()
        return row[0] if row else ""

    @staticmethod
    def _proposal_dict(row):
        if row is None:
            return None
        columns = ("proposal_id", "confirmation_token", "conversation_id", "action",
                   "target_job_id", "target_position", "target_task_id",
                   "expected_status", "expected_revision", "payload", "request_fingerprint",
                   "confidence", "ambiguities", "interpreter_model",
                   "interpretation_source", "corrected_from", "status", "created_at",
                   "expires_at", "transition_result")
        result = dict(zip(columns, row))
        result["payload"] = json.loads(result["payload"])
        result["ambiguities"] = json.loads(result["ambiguities"])
        return result

    async def create_workflow_proposal(self, value: dict) -> list[dict]:
        return await asyncio.to_thread(self._create_workflow_proposal, value)

    def _create_workflow_proposal(self, value):
        now = time.time()
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(_PROPOSAL_SELECT + " WHERE conversation_id=? "
                              "AND status='pending'", (value["conversation_id"],)).fetchall()
            db.execute("UPDATE workflow_action_proposals SET status='superseded' "
                       "WHERE conversation_id=? AND status='pending'",
                       (value["conversation_id"],))
            db.execute("""INSERT INTO workflow_action_proposals
              (proposal_id, confirmation_token, conversation_id, action,
               target_job_id, target_position, target_task_id, expected_status,
               expected_revision, payload_json, request_fingerprint, confidence,
               ambiguities_json, interpreter_model, interpretation_source,
               corrected_from, status, created_at, expires_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
              (value["proposal_id"], value["confirmation_token"],
               value["conversation_id"], value["action"], value["target_job_id"],
               value["target_position"], value["target_task_id"],
               value["expected_status"], value["expected_revision"],
               json.dumps(value["payload"]), value["request_fingerprint"],
               value["confidence"], json.dumps(value["ambiguities"]),
               value["interpreter_model"], value["interpretation_source"],
               value["corrected_from"], now, value["expires_at"]))
        return [self._proposal_dict(row) for row in rows]

    async def workflow_proposal_by_token(self, conversation_id: str | None,
                                         token: str) -> dict | None:
        if not conversation_id:
            return None
        return await asyncio.to_thread(self._workflow_proposal_by_token,
                                       conversation_id, token)

    def _workflow_proposal_by_token(self, conversation_id, token):
        with self._database() as db:
            row = db.execute(_PROPOSAL_SELECT + " "
                             "WHERE conversation_id=? AND confirmation_token=?",
                             (conversation_id, token)).fetchone()
        return self._proposal_dict(row)

    async def pending_workflow_proposals(self, conversation_id: str | None) -> list[dict]:
        if not conversation_id:
            return []
        return await asyncio.to_thread(self._pending_workflow_proposals, conversation_id)

    def _pending_workflow_proposals(self, conversation_id):
        with self._database() as db:
            rows = db.execute(_PROPOSAL_SELECT + " "
                              "WHERE conversation_id=? AND status='pending' "
                              "ORDER BY created_at DESC", (conversation_id,)).fetchall()
        return [self._proposal_dict(row) for row in rows]

    async def expire_workflow_proposals(self, conversation_id: str | None) -> list[dict]:
        return await asyncio.to_thread(self._expire_workflow_proposals, conversation_id)

    def _expire_workflow_proposals(self, conversation_id):
        if not conversation_id:
            return []
        with self._database() as db:
            rows = db.execute(_PROPOSAL_SELECT + " "
                              "WHERE conversation_id=? AND status='pending' AND expires_at<=?",
                              (conversation_id, time.time())).fetchall()
            db.execute("UPDATE workflow_action_proposals SET status='expired' "
                       "WHERE conversation_id=? AND status='pending' AND expires_at<=?",
                       (conversation_id, time.time()))
        return [self._proposal_dict(row) for row in rows]

    async def resolve_workflow_proposal(self, proposal_id: str, status: str) -> bool:
        return await asyncio.to_thread(self._resolve_workflow_proposal,
                                       proposal_id, status)

    def _resolve_workflow_proposal(self, proposal_id, status):
        with self._database() as db:
            result = db.execute("UPDATE workflow_action_proposals SET status=? "
                                "WHERE proposal_id=? AND status='pending'",
                                (status, proposal_id))
        return bool(result.rowcount)

    async def claim_workflow_proposal(self, proposal_id: str,
                                      job: ImplementationJob | None = None,
                                      tasks: list[ImplementationJobTask] | None = None) -> str:
        """Claim a proposal once. Mutating proposals also re-check, in the same
        transaction, that the project's active jobs are exactly the ones the user
        agreed to stop; a displacing claim records a durable switch."""
        return await asyncio.to_thread(self._claim_workflow_proposal,
                                       proposal_id, job, tasks)

    def _claim_workflow_proposal(self, proposal_id, job, tasks):
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(_PROPOSAL_SELECT + " WHERE proposal_id=?",
                             (proposal_id,)).fetchone()
            proposal = self._proposal_dict(row)
            if proposal is None:
                return "missing"
            if proposal["status"] != "pending":
                return proposal["status"]
            if proposal["expires_at"] <= time.time():
                db.execute("UPDATE workflow_action_proposals SET status='expired' "
                           "WHERE proposal_id=?", (proposal_id,))
                return "expired"
            if proposal["target_job_id"] is not None:
                task = db.execute("SELECT status, review_started_at FROM implementation_job_tasks "
                                  "WHERE job_id=? AND position=?",
                                  (proposal["target_job_id"],
                                   proposal["target_position"])).fetchone()
                actual_status = ({"review_pending": "awaiting_evidence_resolution",
                                  "reviewing": "awaiting_evidence_resolution",
                                  "awaiting_confirmation": "awaiting_human_confirmation"}
                                 .get(task[0], task[0]) if task else None)
                stale_review = (task is not None and task[0] == "reviewing" and
                                (task[1] is None or task[1] >= time.time() - 15 * 60))
                if task is None or actual_status != proposal["expected_status"] or stale_review:
                    db.execute("UPDATE workflow_action_proposals SET status='stale' "
                               "WHERE proposal_id=?", (proposal_id,))
                    return "stale"
                if proposal["expected_revision"] is not None:
                    revision = db.execute("SELECT MAX(revision) FROM task_contracts "
                                          "WHERE job_id=? AND position=?",
                                          (proposal["target_job_id"],
                                           proposal["target_position"])).fetchone()[0]
                    if revision != proposal["expected_revision"]:
                        db.execute("UPDATE workflow_action_proposals SET status='stale' "
                                   "WHERE proposal_id=?", (proposal_id,))
                        return "stale"
            refinement_id = proposal["payload"].get("refinement_session_id")
            if refinement_id:
                draft = db.execute("SELECT status FROM refinement_sessions "
                                   "WHERE session_id=? AND conversation_id=?",
                                   (refinement_id, proposal["conversation_id"])).fetchone()
                if draft is None or draft[0] != "active":
                    db.execute("UPDATE workflow_action_proposals SET status='stale' "
                               "WHERE proposal_id=?", (proposal_id,))
                    return "stale"
            parameters = proposal["payload"].get("parameters") or {}
            planning_id = parameters.get("planning_session_id")
            if planning_id:
                # Approval binds to one plan revision; a later revision or a closed
                # session makes it stale.
                planning = db.execute("SELECT status, revision FROM wp_planning_sessions "
                                      "WHERE session_id=? AND conversation_id=?",
                                      (planning_id, proposal["conversation_id"])).fetchone()
                if (planning is None or planning[0] != "active"
                        or planning[1] != parameters.get("planning_revision")):
                    db.execute("UPDATE workflow_action_proposals SET status='stale' "
                               "WHERE proposal_id=?", (proposal_id,))
                    return "stale"
            project_id = parameters.get("project_id")
            displaced = parameters.get("displace") or []
            if parameters.get("conflict_checked"):
                active = self._active_job_ids(db, project_id, proposal["conversation_id"],
                                              proposal["target_job_id"])
                if active != {item["job_id"] for item in displaced}:
                    db.execute("UPDATE workflow_action_proposals SET status='stale' "
                               "WHERE proposal_id=?", (proposal_id,))
                    return "stale"
            now = time.time()
            if job is not None:
                db.execute("""INSERT INTO implementation_jobs
                  (job_id, selectors_json, instruction, conversation_id, project_id, status,
                   created_at, updated_at, plan_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (job.job_id, json.dumps(job.selectors), job.instruction,
                   proposal["conversation_id"], project_id,
                   "pending_switch" if displaced else job.status, now, now,
                   getattr(job, "plan_id", None)))
                db.executemany("""INSERT INTO implementation_job_tasks
                  (job_id, position, change_id, task_id, status, decision_id, error,
                   revision_budget) VALUES (?, ?, ?, ?, ?, ?, ?, 3)""",
                  [(task.job_id, task.position, task.change_id, task.task_id,
                    task.status, task.decision_id, task.error) for task in tasks or []])
            if displaced:
                db.execute("""INSERT INTO workflow_switches
                  (switch_id, proposal_id, conversation_id, project_id, displaced_json,
                   new_job_id, state, created_at, updated_at)
                  VALUES (?, ?, ?, ?, ?, ?, 'switch_requested', ?, ?)""",
                  ("switch_" + proposal_id, proposal_id, proposal["conversation_id"],
                   project_id, json.dumps(displaced), job.job_id if job else None, now, now))
            db.execute("UPDATE workflow_action_proposals SET status='confirmed' "
                       "WHERE proposal_id=?", (proposal_id,))
            return "claimed"

    async def record_workflow_transition(self, proposal_id: str, result: str) -> None:
        await asyncio.to_thread(self._record_workflow_transition, proposal_id, result)

    def _record_workflow_transition(self, proposal_id, result):
        with self._database() as db:
            db.execute("UPDATE workflow_action_proposals SET transition_result=? "
                       "WHERE proposal_id=?", (result, proposal_id))

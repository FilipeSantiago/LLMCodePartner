"""Durable operational audit records, separate from OpenSpec artifacts."""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from agent.routing.models import ExecutionDecision, ImplementationJob, ImplementationJobTask


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
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(implementation_jobs)")}
            if "instruction" not in columns:
                db.execute("ALTER TABLE implementation_jobs ADD COLUMN instruction TEXT NOT NULL DEFAULT ''")
            if "conversation_id" not in columns:
                db.execute("ALTER TABLE implementation_jobs ADD COLUMN conversation_id TEXT")
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
                                        conversation_id: str | None = None) -> None:
        await asyncio.to_thread(self._create_implementation_job, job, tasks, conversation_id)

    def _create_implementation_job(self, job, tasks, conversation_id) -> None:
        now = time.time()
        with self._database() as db:
            db.execute("""INSERT INTO implementation_jobs
              (job_id, selectors_json, instruction, conversation_id, status, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, ?)""", (
                job.job_id, json.dumps(job.selectors), job.instruction, conversation_id,
                job.status, now, now,
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
            db.execute("UPDATE implementation_jobs SET status=?, updated_at=?, error=? WHERE job_id=?", (
                status if status in {"failed", "awaiting_confirmation", "review_pending",
                                     "awaiting_review_resolution", "awaiting_user_decision",
                                     "awaiting_coder_recovery", "recovery_required",
                                     "undone"} else "running",
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
                status IN ('running', 'waiting', 'implemented', 'review_pending') OR
                (status='awaiting_confirmation' AND ?) OR
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

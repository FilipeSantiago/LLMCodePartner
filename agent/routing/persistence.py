"""Durable operational audit records, separate from OpenSpec artifacts."""

import asyncio
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
                  PRIMARY KEY(job_id, position),
                  FOREIGN KEY(job_id) REFERENCES implementation_jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS implementation_reviews (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                  position INTEGER NOT NULL, coder_model TEXT NOT NULL,
                  reviewer_model TEXT NOT NULL, verdict TEXT NOT NULL,
                  rationale TEXT NOT NULL, question TEXT,
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS blocked_operations (
                  operation_id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                  status TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL
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
              (job_id, position, change_id, task_id, status, decision_id, error)
              VALUES (?, ?, ?, ?, ?, ?, ?)""", [
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
                status if status in {"failed", "awaiting_confirmation", "review_pending"} else "running",
                time.time(), error, job_id,
            ))

    async def claim_review(self, job_id: str, position: int) -> bool:
        return await asyncio.to_thread(self._claim_review, job_id, position)

    def _claim_review(self, job_id, position):
        now = time.time()
        with self._database() as db:
            result = db.execute("""UPDATE implementation_job_tasks
              SET status='reviewing', review_started_at=?, error=NULL
              WHERE job_id=? AND position=? AND (
                status IN ('running', 'waiting', 'review_pending') OR
                (status='reviewing' AND review_started_at<?))""",
                (now, job_id, position, now - 15 * 60))
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

    async def record_review(self, job_id: str, position: int, coder_model: str,
                            reviewer_model: str, verdict: str, rationale: str,
                            question: str | None = None) -> None:
        await asyncio.to_thread(self._record_review, job_id, position, coder_model,
                                reviewer_model, verdict, rationale, question)

    def _record_review(self, job_id, position, coder_model, reviewer_model, verdict, rationale, question):
        with self._database() as db:
            db.execute("""INSERT INTO implementation_reviews
              (job_id, position, coder_model, reviewer_model, verdict, rationale, question, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
                job_id, position, coder_model, reviewer_model, verdict, rationale, question, time.time(),
            ))
            db.execute("UPDATE implementation_job_tasks SET review_question=? WHERE job_id=? AND position=?",
                       (question, job_id, position))

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
                t.status='review_pending' OR
                (t.status='reviewing' AND t.review_started_at<?))
              ORDER BY j.updated_at DESC LIMIT 1""",
              (conversation_id, time.time() - 15 * 60)).fetchone()

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

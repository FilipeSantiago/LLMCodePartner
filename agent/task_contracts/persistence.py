"""Durable TaskContract and execution-cycle state."""

import asyncio
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

from agent.task_contracts.models import (
    AcceptanceCriterion, ContextFact, ExecutionLimits, TaskContract,
)
from agent.task_contracts.contractor import USER_RESOLUTION_REASON


class TaskContractPersistence:
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
                CREATE TABLE IF NOT EXISTS task_contracts (
                  contract_id TEXT PRIMARY KEY, parent_contract_id TEXT,
                  job_id TEXT NOT NULL, position INTEGER NOT NULL,
                  task_id TEXT NOT NULL, revision INTEGER NOT NULL,
                  contract_json TEXT NOT NULL, planning_reason TEXT NOT NULL,
                  created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_contract_job_task
                  ON task_contracts(job_id, position, revision);
                CREATE TABLE IF NOT EXISTS task_contract_results (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  contract_id TEXT NOT NULL, outcome TEXT NOT NULL,
                  result_json TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_contract_escalations (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL, position INTEGER NOT NULL,
                  task_id TEXT NOT NULL, contract_id TEXT NOT NULL,
                  reason TEXT NOT NULL, escalation_json TEXT NOT NULL,
                  created_at REAL NOT NULL
                );
            """)

    async def save(self, job_id: str, position: int, contract: TaskContract) -> None:
        await asyncio.to_thread(self._save, job_id, position, contract)

    def _save(self, job_id, position, contract):
        with self._database() as db:
            db.execute("""INSERT OR REPLACE INTO task_contracts
              (contract_id, parent_contract_id, job_id, position, task_id, revision,
               contract_json, planning_reason, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                contract.contract_id, contract.parent_contract_id, job_id, position,
                contract.task_id, contract.revision,
                json.dumps(contract.as_dict(), sort_keys=True),
                contract.planning_reason, time.time(),
            ))

    async def latest(self, job_id: str, position: int) -> TaskContract | None:
        return await asyncio.to_thread(self._latest, job_id, position)

    def _latest(self, job_id, position):
        with self._database() as db:
            row = db.execute("""SELECT contract_json FROM task_contracts
              WHERE job_id=? AND position=? ORDER BY revision DESC LIMIT 1""",
                             (job_id, position)).fetchone()
        return _contract_from_dict(json.loads(row[0])) if row else None

    async def count(self, job_id: str, position: int) -> int:
        return await asyncio.to_thread(self._count, job_id, position)

    def _count(self, job_id, position):
        with self._database() as db:
            row = db.execute("""SELECT COUNT(*) FROM task_contracts
              WHERE job_id=? AND position=?""", (job_id, position)).fetchone()
        return int(row[0])

    async def execution_cycle_count(self, job_id: str, position: int) -> int:
        """Count attempts since the latest explicitly user-confirmed retry."""
        return await asyncio.to_thread(self._execution_cycle_count, job_id, position)

    def _execution_cycle_count(self, job_id, position):
        with self._database() as db:
            boundary = db.execute("""SELECT MAX(revision) FROM task_contracts
              WHERE job_id=? AND position=? AND planning_reason=?""",
                                  (job_id, position, USER_RESOLUTION_REASON)).fetchone()[0]
            if boundary is None:
                return self._count(job_id, position)
            row = db.execute("""SELECT COUNT(*) FROM task_contracts
              WHERE job_id=? AND position=? AND revision>=?""",
                             (job_id, position, boundary)).fetchone()
        return int(row[0])

    async def save_result(self, contract_id: str, outcome: str, result: dict) -> None:
        await asyncio.to_thread(self._save_result, contract_id, outcome, result)

    def _save_result(self, contract_id, outcome, result):
        with self._database() as db:
            db.execute("""INSERT INTO task_contract_results
              (contract_id, outcome, result_json, created_at) VALUES (?, ?, ?, ?)""",
                       (contract_id, outcome, json.dumps(result, sort_keys=True), time.time()))

    async def save_escalation(self, job_id: str, position: int,
                              escalation: dict) -> None:
        await asyncio.to_thread(self._save_escalation, job_id, position, escalation)

    def _save_escalation(self, job_id, position, escalation):
        with self._database() as db:
            db.execute("""INSERT INTO task_contract_escalations
              (job_id, position, task_id, contract_id, reason,
               escalation_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""", (
                job_id, position, escalation["task_id"], escalation["contract_id"],
                escalation["reason"], json.dumps(escalation, sort_keys=True), time.time(),
            ))

    async def latest_escalation(self, job_id: str, position: int) -> dict | None:
        return await asyncio.to_thread(self._latest_escalation, job_id, position)

    def _latest_escalation(self, job_id, position):
        with self._database() as db:
            row = db.execute("""SELECT escalation_json FROM task_contract_escalations
              WHERE job_id=? AND position=? ORDER BY id DESC LIMIT 1""",
                             (job_id, position)).fetchone()
        return json.loads(row[0]) if row else None


_LIMIT_FIELDS = {field.name for field in fields(ExecutionLimits)}


def _contract_from_dict(value: dict) -> TaskContract:
    return TaskContract(
        **{
            **value,
            "acceptance_criteria": tuple(
                AcceptanceCriterion(**item) for item in value["acceptance_criteria"]),
            "known_context": tuple(ContextFact(
                **{**item, "paths": tuple(item.get("paths", ()))})
                for item in value["known_context"]),
            # Contracts saved before call budgets were removed carry extra keys.
            "execution_limits": ExecutionLimits(**{
                key: item for key, item in value["execution_limits"].items()
                if key in _LIMIT_FIELDS}),
            "primary_files": tuple(value["primary_files"]),
            "primary_symbols": tuple(value["primary_symbols"]),
            "reference_files": tuple(value["reference_files"]),
            "allowed_tools": tuple(value["allowed_tools"]),
            "allowed_operations": tuple(value["allowed_operations"]),
            "escalation_conditions": tuple(value["escalation_conditions"]),
            "excluded_files": tuple(value.get("excluded_files", ())),
        }
    )

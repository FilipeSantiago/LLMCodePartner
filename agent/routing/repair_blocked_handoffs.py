"""Preview or repair the WP3 job advanced after blocked coder handoffs.

Run with ``--list`` to find candidates, ``--job-id ID`` to preview one,
and ``--job-id ID --apply`` to perform the guarded, audited repair.
"""

import argparse
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from agent.routing.implementation import handoff_blocker
from agent.routing.models import TERMINAL_JOB_STATUSES

_AFFECTED = ("WP3-T1", "WP3-T2", "WP3-T3")


def _connect(path: Path, writable: bool = False) -> sqlite3.Connection:
    mode = "rw" if writable else "ro"
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode={mode}", uri=True)


def candidates(path: Path) -> list[dict]:
    with closing(_connect(path)) as db:
        rows = db.execute("""SELECT j.job_id, j.status, j.updated_at,
          t.position, t.coder_report
          FROM implementation_jobs j JOIN implementation_job_tasks t ON t.job_id=j.job_id
          WHERE t.task_id='WP3-T1' ORDER BY j.updated_at DESC""").fetchall()
    return [{"job_id": job_id, "status": status, "updated_at": updated_at,
             "first_position": position}
            for job_id, status, updated_at, position, report in rows
            if (report and (handoff_blocker(report, "WP3-T1") or "").startswith(
                "Coder handoff blocked"))]


def _inspect(db: sqlite3.Connection, job_id: str) -> dict:
    job = db.execute("SELECT status FROM implementation_jobs WHERE job_id=?",
                     (job_id,)).fetchone()
    if job is None:
        raise ValueError("job does not exist")
    rows = db.execute("""SELECT position, task_id, status, coder_report
      FROM implementation_job_tasks WHERE job_id=? ORDER BY position""",
      (job_id,)).fetchall()
    affected = [row for row in rows if row[1] in _AFFECTED]
    if tuple(row[1] for row in affected) != _AFFECTED:
        raise ValueError("job does not contain WP3-T1, WP3-T2, WP3-T3 in order")
    if job[0] == "awaiting_coder_recovery" and affected[0][2] == "awaiting_coder_recovery" \
            and all(row[2] == "queued" for row in affected[1:]):
        return {"job_id": job_id, "status": job[0], "result": "already_repaired"}
    if job[0] not in TERMINAL_JOB_STATUSES:
        raise ValueError(f"job is not quiescent: {job[0]}")
    if any(row[2] != "implemented" or not (row[3] and
           (handoff_blocker(row[3], row[1]) or "").startswith("Coder handoff blocked"))
           for row in affected):
        raise ValueError("all three WP3 tasks must be implemented with blocked handoffs")
    first_position = affected[0][0]
    if any(row[2] != "queued" for row in rows if row[0] > affected[-1][0]):
        raise ValueError("a later task progressed; manual recovery is required")
    for table in ("implementation_mutations", "implementation_rollback_files",
                  "implementation_reviews"):
        count = db.execute(f"SELECT COUNT(*) FROM {table} WHERE job_id=? AND position>=?",
                           (job_id, first_position)).fetchone()[0]
        if count:
            raise ValueError(f"{table} contains {count} later records; manual recovery is required")
    return {"job_id": job_id, "status": job[0], "result": "ready",
            "first_position": first_position,
            "changes": [{"position": row[0], "task_id": row[1], "from": row[2],
                         "to": "awaiting_coder_recovery" if index == 0 else "queued"}
                        for index, row in enumerate(affected)]}


def preview(path: Path, job_id: str) -> dict:
    with closing(_connect(path)) as db:
        return _inspect(db, job_id)


def apply(path: Path, job_id: str) -> dict:
    db = _connect(path, writable=True)
    try:
        db.execute("BEGIN IMMEDIATE")
        result = _inspect(db, job_id)
        if result["result"] == "already_repaired":
            db.rollback()
            return result
        now = time.time()
        for index, change in enumerate(result["changes"]):
            db.execute("""UPDATE implementation_job_tasks SET status=?, error=?,
              review_started_at=NULL WHERE job_id=? AND position=? AND status='implemented'""",
              (change["to"],
               "Coder handoff blocked: retry after IDE tool policy repair" if index == 0 else None,
               job_id, change["position"]))
            db.execute("""INSERT INTO implementation_resolution_actions
              (job_id, position, action, details_json, created_at) VALUES (?, ?, ?, ?, ?)""",
              (job_id, change["position"], "repair_blocked_handoff",
               json.dumps({"previous_status": change["from"], "new_status": change["to"]}), now))
        db.execute("""UPDATE implementation_jobs SET status='awaiting_coder_recovery',
          error='WP3 coder handoff blocked', updated_at=? WHERE job_id=?""", (now, job_id))
        db.commit()
        return {**result, "result": "repaired"}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(
        os.getenv("ROUTING_DB_PATH") or Path.home() / ".codepartner" / "routing.sqlite3"))
    parser.add_argument("--list", action="store_true", help="list likely affected jobs")
    parser.add_argument("--job-id", help="preview or repair this exact job")
    parser.add_argument("--apply", action="store_true", help="perform the guarded repair")
    args = parser.parse_args()
    if args.apply and not args.job_id:
        parser.error("--apply requires --job-id")
    if args.list == bool(args.job_id):
        parser.error("choose exactly one of --list or --job-id")
    try:
        result = (candidates(args.db) if args.list else
                  apply(args.db, args.job_id) if args.apply else preview(args.db, args.job_id))
    except (ValueError, sqlite3.Error, OSError) as exc:
        parser.exit(2, f"repair refused: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

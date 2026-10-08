"""Reclassify paused implementation tasks into explicit workflow states.

Older releases used `awaiting_user_decision` for pre-review contract
escalations, post-review decisions, and interrupted Coder runs. They also used
`awaiting_confirmation` and the `review_pending` *status* for what are now
`awaiting_human_confirmation` and `awaiting_evidence_resolution`. The
`review_pending` review *verdict* is unrelated and is left untouched.

The migration is idempotent and runs from every initializer that owns one of
the involved tables, so it completes no matter which one opens the database
first.
"""

import sqlite3

_RENAMES = (
    ("awaiting_confirmation", "awaiting_human_confirmation"),
    ("review_pending", "awaiting_evidence_resolution"),
)

_COMPLETED_REVIEW = """SELECT MAX(review.created_at) FROM implementation_reviews AS review
  WHERE review.job_id=task.job_id AND review.position=task.position
    AND review.verdict IN ('done', 'needs_work', 'ask_user')"""


def migrate_paused_states(db: sqlite3.Connection) -> None:
    tables = {row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"implementation_job_tasks", "implementation_jobs"}.issubset(tables):
        return
    for old, new in _RENAMES:
        db.execute("UPDATE implementation_job_tasks SET status=? WHERE status=?", (new, old))
        db.execute("UPDATE implementation_jobs SET status=? WHERE status=?", (new, old))

    has_reviews = "implementation_reviews" in tables
    if "task_contract_escalations" in tables:
        # The latest escalation must be a pre-review execution escalation that is
        # newer than any completed review. Exhausted review cycles also record an
        # escalation, but those belong to review resolution.
        newer_than_review = (f"AND latest.created_at > COALESCE(({_COMPLETED_REVIEW}), 0)"
                             if has_reviews else "")
        db.execute(f"""UPDATE implementation_job_tasks AS task
          SET status='awaiting_contract_resolution'
          WHERE task.status='awaiting_user_decision' AND EXISTS (
            SELECT 1 FROM task_contract_escalations AS latest
            WHERE latest.id=(SELECT MAX(id) FROM task_contract_escalations AS e
                             WHERE e.job_id=task.job_id AND e.position=task.position)
              AND COALESCE(json_extract(latest.escalation_json, '$.exhausted_limit'), '')
                  != 'review_failures'
              {newer_than_review})""")
    if has_reviews:
        db.execute(f"""UPDATE implementation_job_tasks AS task
          SET status='awaiting_review_resolution'
          WHERE task.status='awaiting_user_decision'
            AND ({_COMPLETED_REVIEW}) IS NOT NULL""")
    interrupted = ["task.error LIKE '%tool call could not be parsed%'"]
    if "implementation_resolution_actions" in tables:
        interrupted.append("""EXISTS (SELECT 1 FROM implementation_resolution_actions AS action
          WHERE action.job_id=task.job_id AND action.position=task.position
            AND action.action='coder_retry')""")
    db.execute(f"""UPDATE implementation_job_tasks AS task
      SET status='awaiting_coder_recovery'
      WHERE task.status='awaiting_user_decision' AND ({' OR '.join(interrupted)})""")
    db.execute("""UPDATE implementation_job_tasks SET status='awaiting_manual_resolution'
      WHERE status='awaiting_user_decision'""")
    # Keep job rows consistent with the reclassified task that paused them.
    db.execute("""UPDATE implementation_jobs AS job SET status=(
        SELECT task.status FROM implementation_job_tasks AS task
        WHERE task.job_id=job.job_id AND task.status IN (
          'awaiting_contract_resolution', 'awaiting_review_resolution',
          'awaiting_coder_recovery', 'awaiting_manual_resolution')
        ORDER BY task.position LIMIT 1)
      WHERE job.status='awaiting_user_decision' AND EXISTS (
        SELECT 1 FROM implementation_job_tasks AS task
        WHERE task.job_id=job.job_id AND task.status IN (
          'awaiting_contract_resolution', 'awaiting_review_resolution',
          'awaiting_coder_recovery', 'awaiting_manual_resolution'))""")

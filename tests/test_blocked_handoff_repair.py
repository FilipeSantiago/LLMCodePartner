"""Regression tests for subscription-only routing and guarded WP3 recovery."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.routing.config import load_routing_config
from agent.routing.gateway import same_credential_limit
from agent.routing.implementation import handoff_blocker
from agent.routing.models import ImplementationJob, ImplementationJobTask
from agent.routing.persistence import RoutingPersistence
from agent.routing.repair_blocked_handoffs import apply, candidates, preview


def _handoff(task_id: str, blocked: bool = True) -> str:
    return json.dumps({
        "schema_version": 1, "task_id": task_id,
        "summary": "Inspection blocked" if blocked else "Inspection complete",
        "changed_files": [],
        "criteria": [{"criterion": "Inspect sources", "status":
                      "blocked" if blocked else "satisfied"}],
        "remaining_gaps": ["IDE reader denied"] if blocked else [],
    })


class HandoffAndRoutingTests(unittest.TestCase):
    def test_blocked_handoff_is_not_success(self):
        self.assertIn("blocked", handoff_blocker(_handoff("WP3-T1"), "WP3-T1"))
        self.assertIsNone(handoff_blocker(_handoff("WP3-T1", False), "WP3-T1"))
        self.assertIn("invalid", handoff_blocker("nothing", "WP3-T1"))
        self.assertIn("invalid", handoff_blocker(_handoff("WP3-T1", False), "WP3-T2"))

    def test_codex_uses_subscription_directly(self):
        config = load_routing_config()
        self.assertTrue(all(model.gateway_mode == "disabled"
                            for model in config.models.values() if model.provider == "codex"))

    def test_session_limit_skips_same_credential_retry(self):
        self.assertTrue(same_credential_limit("You've hit your session limit (API status 429)"))
        self.assertFalse(same_credential_limit("HTTP 503 service unavailable"))


class RepairTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "routing.sqlite3"
        self.store = RoutingPersistence(self.path)
        self.job = ImplementationJob.create(["WP3"], ["WP3-T1", "WP3-T2", "WP3-T3"])
        await self.store.create_implementation_job(self.job, [
            ImplementationJobTask(self.job.job_id, index, "CHG1", task_id)
            for index, task_id in enumerate(("WP3-T1", "WP3-T2", "WP3-T3"))])
        for index, task_id in enumerate(("WP3-T1", "WP3-T2", "WP3-T3")):
            await self.store.save_coder_report(self.job.job_id, index, _handoff(task_id))
            await self.store.update_implementation_task(self.job.job_id, index, "implemented")
        await self.store.finish_implementation_job(self.job.job_id)

    async def test_preview_apply_and_idempotence(self):
        self.assertEqual(candidates(self.path)[0]["job_id"], self.job.job_id)
        self.assertEqual(preview(self.path, self.job.job_id)["result"], "ready")
        self.assertEqual((await self.store.implementation_task(self.job.job_id, 0)).status,
                         "implemented")
        self.assertEqual(apply(self.path, self.job.job_id)["result"], "repaired")
        self.assertEqual(apply(self.path, self.job.job_id)["result"], "already_repaired")
        rows = await self.store.implementation_job_tasks(self.job.job_id)
        self.assertEqual([row.status for row in rows],
                         ["awaiting_coder_recovery", "queued", "queued"])
        self.assertEqual(await self.store.implementation_job_status(self.job.job_id),
                         "awaiting_coder_recovery")
        self.assertEqual(await self.store.coder_report_for(self.job.job_id, 1),
                         _handoff("WP3-T2"))

    async def test_refuses_mutation_or_review_history(self):
        with sqlite3.connect(self.path) as db:
            db.execute("""INSERT INTO implementation_mutations
              (job_id, position, tool_name, operation, targets_json, stage, created_at, updated_at)
              VALUES (?, 2, 'apply_patch', 'patch', '[]', 'committed', 1, 1)""",
              (self.job.job_id,))
        with self.assertRaisesRegex(ValueError, "implementation_mutations"):
            apply(self.path, self.job.job_id)
        self.assertEqual((await self.store.implementation_task(self.job.job_id, 0)).status,
                         "implemented")

    async def test_refuses_running_job(self):
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE implementation_jobs SET status='running' WHERE job_id=?",
                       (self.job.job_id,))
        with self.assertRaisesRegex(ValueError, "not quiescent"):
            apply(self.path, self.job.job_id)

    async def test_refuses_review_history(self):
        with sqlite3.connect(self.path) as db:
            db.execute("""INSERT INTO implementation_reviews
              (job_id, position, coder_model, reviewer_model, verdict, rationale, created_at)
              VALUES (?, 2, 'codex-luna-medium', 'claude-sonnet', 'done', 'accepted', 1)""",
              (self.job.job_id,))
        with self.assertRaisesRegex(ValueError, "implementation_reviews"):
            apply(self.path, self.job.job_id)


if __name__ == "__main__":
    unittest.main()

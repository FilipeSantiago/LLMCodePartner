"""Regression tests for blocked handoffs and subscription-only routing."""

import json
import unittest

from agent.routing.config import load_routing_config
from agent.routing.gateway import same_credential_limit
from agent.routing.implementation import handoff_blocker


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

if __name__ == "__main__":
    unittest.main()

"""Proposal tables created by older builds are upgraded instead of breaking proposals."""

import contextlib
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.routing.persistence import _PROPOSAL_COLUMNS, RoutingPersistence
from agent.workflow.controller import ConversationWorkflow
from agent.workflow.states import WorkflowAction as A
from agent.workflow.intent import WorkflowIntent
from api import dialog_controller
from model.chat import ChatCompletionRequest, ChatMessage

# The schema found in a deployed database (an earlier, uncommitted build).
LEGACY_DDL = """CREATE TABLE workflow_action_proposals (
  proposal_id TEXT PRIMARY KEY, confirmation_token TEXT NOT NULL UNIQUE,
  conversation_id TEXT NOT NULL, action TEXT NOT NULL,
  target_job_id TEXT, target_position INTEGER,
  expected_status TEXT, payload_json TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL, confidence TEXT NOT NULL,
  ambiguities_json TEXT NOT NULL, interpreter_model TEXT,
  status TEXT NOT NULL, created_at REAL NOT NULL,
  updated_at REAL NOT NULL, expires_at REAL NOT NULL,
  resolved_at REAL, target_task_id TEXT, expected_revision INTEGER,
  interpretation_source TEXT, job_id TEXT, corrected_from TEXT,
  transition_result TEXT
)"""


def _intent(conversation="conv"):
    return WorkflowIntent(conversation_id=conversation, message_fingerprint="f",
                          action=A.EXECUTE_COMMAND, expected_effects=("Run `/review`",),
                          parameters={"command": "/review WP4"})


class LegacyProposalTableTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "routing.sqlite3"
        now = time.time()
        with contextlib.closing(sqlite3.connect(self.path)) as db, db:
            db.execute(LEGACY_DDL)
            db.execute("""INSERT INTO workflow_action_proposals
              (proposal_id, confirmation_token, conversation_id, action, payload_json,
               request_fingerprint, confidence, ambiguities_json, status, created_at,
               updated_at, expires_at, job_id)
              VALUES ('old', 'OLDTOK', 'conv', 'STOP_JOB', '{}', 'f', 'high', '[]',
                      'rejected', ?, ?, ?, 'job-1')""", (now, now, now + 60))

    def columns(self):
        with contextlib.closing(sqlite3.connect(self.path)) as db:
            return tuple(row[1] for row in db.execute(
                "PRAGMA table_info(workflow_action_proposals)"))

    async def test_legacy_table_is_rebuilt_and_proposals_work(self):
        persistence = RoutingPersistence(self.path)
        self.assertEqual(self.columns(), _PROPOSAL_COLUMNS)

        workflow = ConversationWorkflow(persistence)
        old = await workflow.find("conv", "OLDTOK")
        self.assertEqual(old["status"], "rejected")
        self.assertEqual(old["interpretation_source"], "deterministic")

        proposal = await workflow.propose(_intent())
        pending = await workflow.pending("conv")
        self.assertEqual([row["proposal_id"] for row in pending], [proposal.proposal_id])
        self.assertEqual(pending[0]["payload"]["parameters"]["command"], "/review WP4")
        claimed = await workflow.claim(pending[0])
        self.assertTrue(claimed.claimed)

    def test_migration_is_idempotent(self):
        RoutingPersistence(self.path)
        RoutingPersistence(self.path)
        self.assertEqual(self.columns(), _PROPOSAL_COLUMNS)
        with contextlib.closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM workflow_action_proposals").fetchone()[0], 1)
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}
        self.assertNotIn("workflow_action_proposals_legacy", tables)
        self.assertIn("workflow_proposals_conversation", tables)


class ProposalErrorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.persistence = RoutingPersistence(Path(directory.name) / "routing.sqlite3")

    async def test_other_integrity_errors_are_not_called_token_collisions(self):
        workflow = ConversationWorkflow(self.persistence)
        failure = sqlite3.IntegrityError(
            "NOT NULL constraint failed: workflow_action_proposals.updated_at")
        with mock.patch.object(self.persistence, "create_workflow_proposal",
                               side_effect=failure):
            with self.assertRaisesRegex(sqlite3.IntegrityError, "updated_at"):
                await workflow.propose(_intent())

    async def test_token_collisions_are_retried(self):
        workflow = ConversationWorkflow(self.persistence)
        collision = sqlite3.IntegrityError(
            "UNIQUE constraint failed: workflow_action_proposals.confirmation_token")
        original = self.persistence.create_workflow_proposal
        calls = []

        async def flaky(value):
            calls.append(value["confirmation_token"])
            if len(calls) == 1:
                raise collision
            return await original(value)
        with mock.patch.object(self.persistence, "create_workflow_proposal", flaky):
            proposal = await workflow.propose(_intent())
        self.assertEqual(len(calls), 2)
        self.assertEqual(proposal.confirmation_token, calls[1])

    async def test_workflow_failure_is_answered_in_band(self):
        session = SimpleNamespace(client=object(), tools=SimpleNamespace(specs=[], handlers={}),
                                  project_id="proj")
        with (mock.patch.object(dialog_controller, "get_session", return_value=session),
              mock.patch.object(dialog_controller, "task_execution",
                                SimpleNamespace(persistence=self.persistence)),
              mock.patch.object(ConversationWorkflow, "route_turn",
                                side_effect=RuntimeError("boom")),
              self.assertLogs("codepartner.api", "ERROR")):
            response = await dialog_controller.chat_completions(ChatCompletionRequest(
                model="test", conversation_id="conv",
                messages=[ChatMessage(role="user", content="implement WP5")]))
        text = response.choices[0].message.content
        self.assertIn("internal error", text)
        self.assertRegex(text, r"trace \S+")


if __name__ == "__main__":
    unittest.main()

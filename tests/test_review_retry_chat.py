"""Conversational retry-review proposals in implement-mode chat."""

import re
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.routing.models import ImplementationJob, ImplementationJobTask
from agent.routing.persistence import RoutingPersistence
from agent.workflow.controller import ConversationWorkflow
from agent.workflow.review_retry import offered_choice
from api import dialog_controller
from mcp_bridge.models import TextEvent
from model.chat import ChatCompletionRequest, ChatMessage


class ReviewRetryChatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.persistence = RoutingPersistence(Path(self.directory.name) / "routing.sqlite3")
        job = ImplementationJob.create(["WP4"], ["WP4-T5"])
        await self.persistence.create_implementation_job(
            job, [ImplementationJobTask(job.job_id, 0, "CHG1", "WP4-T5")], "conv")
        await self.persistence.update_implementation_task(
            job.job_id, 0, "review_pending", error="unsupported project artifact path(s): /Dockerfile")
        self.job = job
        session = SimpleNamespace(client=object(), tools=SimpleNamespace(specs=[], handlers={}))
        self.patches = [
            mock.patch.object(dialog_controller, "get_session", return_value=session),
            mock.patch.object(dialog_controller, "task_execution",
                              SimpleNamespace(persistence=self.persistence)),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    async def say(self, content, previous=(), stream=False, model="codepartner-implement"):
        messages = [ChatMessage(role=role, content=text) for role, text in previous]
        messages.append(ChatMessage(role="user", content=content))
        result = await dialog_controller.chat_completions(ChatCompletionRequest(
            model=model, conversation_id="conv", messages=messages,
            stream=stream))
        if not stream:
            return result.choices[0].message.content
        data = b"".join([item if isinstance(item, bytes) else item.encode()
                         async for item in result.body_iterator]).decode()
        return data

    async def test_natural_request_proposes_retry_without_running_it(self):
        with mock.patch.object(dialog_controller, "ImplementationService") as start:
            answer = await self.say("Retry the WP4 review, forget the Dockerfile")
        self.assertIn("Run the Reviewer again for WP4-T5", answer)
        self.assertIn("Nothing has executed yet", answer)
        self.assertIn("mutation journal before other project evidence", answer)
        self.assertIn("Do not discard a path claim", answer)
        self.assertRegex(answer, r"confirm [A-Z0-9]{6}")
        start.assert_not_called()

    async def test_numbered_reply_uses_immediately_previous_menu(self):
        menu = ("Choose one:\n1. **Retry the WP4 review** (check the evidence again).\n"
                "2. Skip review and start WP5.")
        answer = await self.say("1", previous=(("assistant", menu),))
        self.assertIn("Run the Reviewer again for WP4-T5", answer)
        self.assertIsNone(offered_choice("1", [{"role": "user", "content": menu}]))

    async def test_unbound_number_does_not_create_a_proposal(self):
        answer = await self.say("1")
        self.assertIn("cannot link that number", answer)
        with sqlite3.connect(self.persistence.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM workflow_action_proposals").fetchone()[0], 0)

    async def test_persisted_menu_choice_survives_omitted_history(self):
        await self.persistence.record_offered_choices("conv", {
            1: "Retry the WP4 review", 2: "Stop the job"})
        answer = await self.say("1")
        self.assertIn("Run the Reviewer again for WP4-T5", answer)

    async def test_negated_retry_does_not_create_a_proposal(self):
        answer = await self.say("Do not retry the WP4 review")
        self.assertIn("do not want to retry", answer)
        with sqlite3.connect(self.persistence.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM workflow_action_proposals").fetchone()[0], 0)

    async def test_ambiguous_work_package_requires_a_task(self):
        other = ImplementationJob.create(["WP4"], ["WP4-T4"])
        await self.persistence.create_implementation_job(
            other, [ImplementationJobTask(other.job_id, 0, "CHG1", "WP4-T4")], "conv")
        await self.persistence.update_implementation_task(other.job_id, 0, "review_pending")
        answer = await self.say("Retry the WP4 review")
        self.assertIn("More than one paused task", answer)
        specific = await self.say("Retry the WP4-T5 review")
        self.assertIn("Run the Reviewer again for WP4-T5", specific)
        by_job = await self.say(f"Retry review for job {self.job.job_id}")
        self.assertIn("Run the Reviewer again for WP4-T5", by_job)

    async def test_confirmation_rechecks_state_and_runs_only_once(self):
        answer = await self.say("Retry the WP4 review")
        token = re.search(r"confirm ([A-Z0-9]{6})", answer).group(1)

        async def events(*_args):
            yield TextEvent("review started")

        service = SimpleNamespace(retry_review=events)
        with mock.patch.object(dialog_controller, "ImplementationService",
                               return_value=service) as start:
            result = await self.say(f"confirm {token}", stream=True)
            self.assertIn("review started", result)
            self.assertEqual(start.call_count, 1)
            repeated = await self.say(f"confirm {token}", stream=True)
            self.assertIn("confirmed", repeated.lower())
            self.assertEqual(start.call_count, 1)

    async def test_stale_confirmation_executes_nothing(self):
        answer = await self.say("Retry the WP4 review")
        token = re.search(r"confirm ([A-Z0-9]{6})", answer).group(1)
        await self.persistence.update_implementation_task(self.job.job_id, 0, "completed")
        with mock.patch.object(dialog_controller, "ImplementationService") as start:
            result = await self.say(f"confirm {token}", stream=True)
        self.assertIn("workflow changed", result)
        start.assert_not_called()

    async def test_retry_without_paused_review_explains_missing_state(self):
        await self.persistence.update_implementation_task(self.job.job_id, 0, "completed")
        answer = await self.say("Retry the WP4 review")
        self.assertIn("no paused review was found", answer)
        self.assertNotIn("needs an explicit scope", answer)

    async def test_review_mode_cannot_confirm_a_mutating_command(self):
        answer = await self.say("/implement WP4", model="codepartner-review")
        self.assertIn("read-only", answer)
        with sqlite3.connect(self.persistence.path) as db:
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM workflow_action_proposals").fetchone()[0], 0)

    async def test_stale_reviewing_task_can_be_proposed_again(self):
        await self.persistence.update_implementation_task(self.job.job_id, 0, "reviewing")
        with sqlite3.connect(self.persistence.path) as db:
            db.execute("""UPDATE implementation_job_tasks SET review_started_at=?
              WHERE job_id=? AND position=0""", (time.time() - 20 * 60, self.job.job_id))
        answer = await self.say("Retry the WP4 review")
        self.assertIn("Run the Reviewer again for WP4-T5", answer)
        token = re.search(r"confirm ([A-Z0-9]{6})", answer).group(1)
        workflow = ConversationWorkflow(self.persistence)
        claimed = await workflow.claim(await workflow.find("conv", token))
        self.assertTrue(claimed.claimed)

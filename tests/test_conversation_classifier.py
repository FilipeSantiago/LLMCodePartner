"""Free-form intent is advisory; only a confirmed, validated action may run."""

import unittest
import tempfile
from pathlib import Path
from unittest import mock

from agent.routing.persistence import RoutingPersistence
from agent.workflow.controller import ConversationWorkflow
from agent.workflow.interpreter import ConversationIntentInterpreter
from tests import native_mode


def setUpModule():
    native_mode.pin()


def tearDownModule():
    native_mode.unpin()



class _Provider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def emit_json(self, messages, schema, model=None):
        if self.error:
            raise self.error
        return self.result


class ConversationClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_can_classify_free_form_review_request(self):
        result = {"action": "REVIEW", "selectors": ["WP4"],
                  "instruction": "Check whether the tests are complete",
                  "confidence": "high", "ambiguities": []}
        intent = await ConversationIntentInterpreter(_Provider(result)).interpret(
            "Could you look over WP4 and check whether the tests are complete?")
        self.assertEqual(intent["action"], "REVIEW")
        self.assertEqual(intent["selectors"], ("WP4",))

    async def test_model_cannot_invent_a_task_selector(self):
        result = {"action": "START_IMPLEMENTATION", "selectors": ["WP5"],
                  "instruction": "Start work", "confidence": "high", "ambiguities": []}
        intent = await ConversationIntentInterpreter(_Provider(result)).interpret(
            "Tell me about WP4")
        self.assertEqual(intent["action"], "ASK_CLARIFICATION")

    async def test_model_failure_falls_back_to_safe_answer(self):
        intent = await ConversationIntentInterpreter(
            _Provider(error=TimeoutError())).interpret("What happened in WP4?")
        self.assertEqual(intent["action"], "ANSWER")

    async def test_implement_mode_is_a_hint_not_a_command(self):
        intent = await ConversationIntentInterpreter(
            _Provider(error=TimeoutError())).interpret("WP4", "implement")
        self.assertEqual(intent["action"], "START_IMPLEMENTATION")

    async def test_ad_hoc_coding_is_proposed_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = ConversationWorkflow(RoutingPersistence(
                Path(directory) / "routing.sqlite3"))
            classification = {"action": "CODING", "selectors": (),
                              "instruction": "Fix the parser", "confidence": "high"}
            with mock.patch.object(ConversationIntentInterpreter, "interpret",
                                   return_value=classification):
                proposal = await workflow.interpret_fresh(
                    "conv", "Fix the parser", "")
            self.assertIn("Ask the coding agent to change the project", proposal.render())
            self.assertIn("Nothing has executed yet", proposal.render())
            pending = await workflow.pending("conv")
            self.assertEqual(pending[0]["action"], "CODING")


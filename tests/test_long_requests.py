"""User text has no length limit; only the small classifier sees an excerpt."""

import json
import unittest

from agent.workflow.intent import WorkflowContext
from agent.workflow.interpreter import (ConversationIntentInterpreter, WorkflowInterpreter,
                                        classifier_excerpt)
from tests.test_turn_routing import _WorldTest

LAST_PARAGRAPH = "Final section: log the piano-roll video artifact to MLflow."


def _book(chars: int) -> str:
    body = "Implement the experiment pipeline. " * (chars // 35)
    return f"{body}\n\n{LAST_PARAGRAPH}"


class _RecordingProvider:
    def __init__(self, result):
        self.result = result
        self.messages = None

    async def emit_json(self, messages, schema, model=None):
        self.messages = messages
        return self.result


class ClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_spec_request_is_classified_not_rejected(self):
        provider = _RecordingProvider({"action": "SPEC_DRAFT", "selectors": [],
                                       "instruction": "Pipeline spec", "confidence": "high",
                                       "ambiguities": []})
        message = _book(10_000)
        result = await ConversationIntentInterpreter(provider).interpret(message, "spec")
        self.assertEqual(result["action"], "SPEC_DRAFT")
        sent = provider.messages[-1]["content"]
        self.assertIn("characters omitted", sent)
        self.assertIn(LAST_PARAGRAPH, sent)
        self.assertLess(len(sent), 4000)

    async def test_empty_message_still_asks_for_clarification(self):
        result = await ConversationIntentInterpreter(_RecordingProvider({})).interpret("  ")
        self.assertEqual(result["action"], "ASK_CLARIFICATION")

    async def test_instruction_is_not_truncated(self):
        instruction = "x" * 5000
        provider = _RecordingProvider({"action": "CODING", "selectors": [],
                                       "instruction": instruction, "confidence": "high",
                                       "ambiguities": []})
        result = await ConversationIntentInterpreter(provider).interpret("fix it")
        self.assertEqual(result["instruction"], instruction)

    async def test_long_reply_to_workflow_is_not_rejected_for_length(self):
        intent = await WorkflowInterpreter(_RecordingProvider({}), "m").interpret(
            _book(50_000), WorkflowContext("conv"), "fp")
        self.assertNotIn("characters", intent.clarification)
        self.assertIn("No paused workflow", intent.clarification)

    def test_short_messages_are_sent_unchanged(self):
        self.assertEqual(classifier_excerpt("Let's do WP4"), "Let's do WP4")


class SpecModeTurnTests(_WorldTest):
    async def test_long_spec_request_drafts_from_the_full_text(self):
        message = _book(12_000)
        answer = await self.world.say(message, model="codepartner-spec")
        self.assertIn("Draft a new OpenSpec change from your request", answer)
        self.assertIn("Starts:", answer)
        self.assertNotIn(LAST_PARAGRAPH, answer)
        payload = json.loads(self.world.rows(
            "SELECT payload_json FROM workflow_action_proposals")[0][0])
        command = payload["parameters"]["command"]
        self.assertTrue(command.startswith("/__workflow_spec_draft__ "))
        self.assertTrue(command.endswith(LAST_PARAGRAPH))


if __name__ == "__main__":
    unittest.main()

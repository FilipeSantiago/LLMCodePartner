import unittest

from conversation.openai_request import safe_request_structure, tool_specs
from model.chat import ChatCompletionRequest, ChatMessage


class OpenAiRequestTests(unittest.TestCase):
    def test_mcp_input_schema_is_preserved(self):
        request = ChatCompletionRequest(
            model="test", messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {
                "name": "apply_patch", "inputSchema": {
                    "type": "object", "properties": {"patch": {"type": "string"}},
                },
            }}],
        )
        [spec] = tool_specs(request)
        self.assertEqual(spec.name, "apply_patch")
        self.assertIn("patch", spec.schema["properties"])

    def test_safe_request_structure_classifies_without_retaining_content(self):
        secret = "do-not-log-this-message"
        request = ChatCompletionRequest(
            model="test", messages=[ChatMessage(role="user", content=f"/implement WP1-T1 {secret}")],
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )
        audit = safe_request_structure(request)
        self.assertEqual(audit["origin"], "primary_user_command")
        self.assertEqual(audit["command"], "/implement")
        self.assertEqual(audit["tool_names"], ["apply_patch"])
        self.assertNotIn(secret, repr(audit))


if __name__ == "__main__":
    unittest.main()

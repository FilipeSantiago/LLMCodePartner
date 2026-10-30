"""KB-first for every agent is a prompt rule: guidance per policy, and KB-call metrics."""

import unittest
from types import SimpleNamespace

from agent.prompt import PromptFactory, PromptRole
from agent.routing.task_execution import TaskExecutionService
from tests.test_routing import task


class GuidanceTests(unittest.TestCase):
    def test_coder_and_reviewer_own_their_read_pipelines(self):
        self.assertNotEqual(PromptFactory.create(PromptRole.CODER).read_pipeline(),
                            PromptFactory.create(PromptRole.REVIEWER).read_pipeline())

    def test_coder_prompt_has_the_read_pipeline(self):
        prompt = TaskExecutionService._execution_messages(task(), [], "")[0]["content"]
        self.assertIn(PromptFactory.create(PromptRole.CODER).read_pipeline(), prompt)
        self.assertNotIn("first call for each file returns KB navigation", prompt)


class MetricsTests(unittest.TestCase):
    def test_kb_calls_are_counted(self):
        from providers.claude import record_native_tool
        from providers.codex import record_native_item
        run = SimpleNamespace(metadata={}, native=None)
        record_native_tool(run, "mcp__jetbrains__kb_search_graph", {})
        record_native_tool(run, "Read", {"file_path": "a.py"})
        self.assertEqual(run.metadata["native_tool_metrics"]["kb_calls"], 1)
        run = SimpleNamespace(metadata={}, native=None)
        record_native_item(run, {"type": "mcp_tool_call", "tool": "kb_trace_path"})
        self.assertEqual(run.metadata["native_tool_metrics"]["kb_calls"], 1)


if __name__ == "__main__":
    unittest.main()

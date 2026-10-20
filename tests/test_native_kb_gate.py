"""KB-first discovery for native agents: guidance, the Claude search nudge, metrics."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from agent.kb_first import native_gate_metadata
from agent.planning.knowledge_base import (KB_NATIVE_EXECUTION_GUIDANCE, KB_READING_GUIDANCE,
                                           KB_SELECTIVE_GUIDANCE, kb_guidance)
from agent.routing.task_execution import TaskExecutionService
from mcp_bridge.registry import ToolSpec
from providers.kb_gate import NativeKbGate, classify_search
from tests.test_routing import task

KB = "mcp__jetbrains__kb_search_graph"


class ClassifierTests(unittest.TestCase):
    def test_identifier_searches_versus_everything_else(self):
        self.assertEqual(classify_search({"pattern": "run_experiment"}), (["run_experiment"], None))
        self.assertEqual(classify_search({"pattern": r"\bFoo|Bar\b"}), (["Foo", "Bar"], None))
        self.assertEqual(classify_search({"pattern": "error:"})[1], "literal")
        self.assertEqual(classify_search({"pattern": "from x import y"})[1], "intent_import")
        self.assertEqual(classify_search({"pattern": "foo("})[1], "intent_usage")
        self.assertEqual(classify_search({"pattern": "Foo", "fileMask": "*.yaml"})[1],
                         "intent_tests_docs")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "pkg"))
        open(os.path.join(self.root, "pkg", "a.py"), "w").close()
        self.gate = NativeKbGate(KB, max_gates=3)

    def decide(self, name, arguments):
        return self.gate.decide(name, arguments, self.root)

    def test_broad_symbol_searches_are_advised_once(self):
        reason = self.decide("Grep", {"pattern": "run_experiment", "path": "."})
        self.assertIn(KB, reason)
        self.assertIn('name_pattern "run_experiment"', reason)
        self.assertIn("repeat the identical call", reason)
        # The identical repeat runs.
        self.assertIsNone(self.decide("Grep", {"pattern": "run_experiment", "path": "."}))
        self.assertIsNotNone(self.decide("Grep", {"pattern": "Trainer", "path": "pkg"}))
        self.assertIsNotNone(self.decide("Bash", {"command": "grep -rn ExperimentConfig ."}))
        self.assertEqual(self.gate.summary(),
                         {"advised": 3, "repeated_anyway": 1, "kb_used": False})

    def test_targeted_and_non_symbol_searches_run(self):
        self.assertIsNone(self.decide("Grep", {"pattern": "Foo", "path": "pkg/a.py"}))
        self.assertIsNone(self.decide("Grep", {"pattern": "error:"}))
        self.assertIsNone(self.decide("Grep", {"pattern": "Foo", "path": "tests"}))
        self.assertIsNone(self.decide("Bash", {"command": "grep -rn Foo . | head"}))
        self.assertIsNone(self.decide("Bash", {"command": "grep Foo pkg/a.py"}))
        self.assertIsNone(self.decide("Bash", {"command": "uv run pytest -q"}))
        self.assertIsNone(self.decide("Read", {"file_path": "pkg/a.py"}))
        self.assertIsNone(self.decide("Glob", {"pattern": "pkg/*.py"}))

    def test_repo_wide_glob_is_advised_only_the_first_time(self):
        self.assertIn("kb_get_architecture", self.decide("Glob", {"pattern": "**/*.py"}))
        self.assertIsNone(self.decide("Glob", {"pattern": "**/*.md"}))

    def test_kb_use_and_the_cap_turn_the_gate_off(self):
        self.gate.observe("mcp__jetbrains__kb_trace_path")
        self.assertIsNone(self.decide("Grep", {"pattern": "Foo"}))
        capped = NativeKbGate(KB, max_gates=1)
        self.assertIsNotNone(capped.decide("Grep", {"pattern": "Foo"}, self.root))
        self.assertIsNone(capped.decide("Grep", {"pattern": "Bar"}, self.root))


class GuidanceTests(unittest.TestCase):
    def test_guidance_follows_policy_and_mode(self):
        self.assertEqual(kb_guidance("kb_first", native=True), KB_NATIVE_EXECUTION_GUIDANCE)
        self.assertEqual(kb_guidance("kb_first"), KB_READING_GUIDANCE)
        self.assertEqual(kb_guidance("selective", native=True), KB_SELECTIVE_GUIDANCE)

    def test_native_coder_prompt_follows_the_policy(self):
        native = TaskExecutionService._execution_messages(
            task(), [], "", native=True, kb_policy="kb_first")[0]["content"]
        self.assertIn(KB_NATIVE_EXECUTION_GUIDANCE, native)
        self.assertNotIn("IDE source", native)
        selective = TaskExecutionService._execution_messages(
            task(), [], "", native=True)[0]["content"]
        self.assertIn(KB_SELECTIVE_GUIDANCE, selective)

    def test_gate_metadata_follows_the_policy(self):
        kb_first = SimpleNamespace(enabled=True, policy="kb_first", search_gate_max=4)
        selective = SimpleNamespace(enabled=True, policy="selective", search_gate_max=4)
        self.assertEqual(native_gate_metadata(kb_first), {"kb_gate": True, "kb_gate_max": 4})
        self.assertEqual(native_gate_metadata(selective), {})
        self.assertEqual(native_gate_metadata(selective, always=True)["kb_gate"], True)
        self.assertEqual(native_gate_metadata(
            SimpleNamespace(enabled=False, policy="kb_first")), {})


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def _options(self, metadata, specs):
        from providers import claude
        seen = {}

        async def fake_query(prompt, options):
            seen["options"] = options
            return
            yield

        async def put(_event):
            return None

        native = SimpleNamespace(read_only=True, workspace="/work", env={}, metrics={
            "tool_calls": 0, "tools": {}, "files": []}, agent_can_write=False)
        run = SimpleNamespace(metadata=metadata, native=native, put=put)
        with mock.patch.object(claude, "query", fake_query):
            await claude.ClaudeProvider()._run_strategy(run, "p", None, specs)
        return seen["options"]

    async def test_hook_is_installed_only_when_asked_and_the_kb_is_offered(self):
        kb = [ToolSpec("kb_search_graph", "KB", {"type": "object"})]
        options = await self._options({"kb_gate": True}, kb)
        hook = options.hooks["PreToolUse"][0].hooks[0]
        denied = await hook({"tool_name": "Grep", "tool_input": {"pattern": "Foo"}}, "t1", None)
        self.assertEqual(denied["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(await hook({"tool_name": "Grep", "tool_input": {"pattern": "Foo"}},
                                    "t2", None), {})
        self.assertIsNone((await self._options({}, kb)).hooks)
        self.assertIsNone((await self._options({"kb_gate": True}, [])).hooks)

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

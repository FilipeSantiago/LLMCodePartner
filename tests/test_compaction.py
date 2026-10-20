"""Native CLI auto-compaction settings: configured per invocation, never globally."""

import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from claude_agent_sdk import SystemMessage

from providers import compaction
from providers.codex import COMPACTION_ITEMS, CodexProvider
from tests.test_routing import task


class ThresholdTests(unittest.TestCase):
    def test_default_and_overrides(self):
        with mock.patch.dict(os.environ, {compaction.ENV: ""}):
            self.assertEqual(compaction.auto_compact_tokens(), 250_000)
        with mock.patch.dict(os.environ, {compaction.ENV: "400000"}):
            self.assertEqual(compaction.auto_compact_tokens(), 400_000)
        with mock.patch.dict(os.environ, {compaction.ENV: "off"}):
            self.assertIsNone(compaction.auto_compact_tokens())
            self.assertEqual(compaction.codex_args(), [])
            self.assertIsNone(compaction.claude_settings())
        with mock.patch.dict(os.environ, {compaction.ENV: "5000"}), \
                self.assertLogs("codepartner.providers.compaction", "WARNING"):
            self.assertEqual(compaction.auto_compact_tokens(), 250_000)

    def test_brief_carries_the_task_contract(self):
        contract = SimpleNamespace(goal="Inject the tracker", excluded_files=(".env",))
        brief = compaction.compaction_brief(task(), contract)
        self.assertIn(task().id, brief)
        self.assertIn("Inject the tracker", brief)
        self.assertIn("Never modify: .env", brief)
        self.assertIn("every file modified so far", brief)
        self.assertIn("remain authoritative", brief)


class CodexCompactionTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {compaction.ENV: "", "CODEX_ARGS": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ide_and_native_runs_set_the_limit_per_invocation(self):
        native = SimpleNamespace(workspace="/work")
        for args in (CodexProvider()._base_args(),
                     CodexProvider()._base_args(native=native)):
            index = args.index("model_auto_compact_token_limit=250000")
            self.assertEqual(args[index - 1], "-c")
            self.assertIn("--ignore-user-config", args)

    def test_operator_codex_args_still_win(self):
        with mock.patch.dict(os.environ, {
                "CODEX_ARGS": "-c model_auto_compact_token_limit=300000"}):
            args = CodexProvider()._base_args()
        self.assertLess(args.index("model_auto_compact_token_limit=250000"),
                        args.index("model_auto_compact_token_limit=300000"))

    def test_compaction_item_type(self):
        self.assertIn("context_compaction", COMPACTION_ITEMS)


class ClaudeCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, metadata):
        from providers import claude
        seen = {}

        async def fake_query(prompt, options):
            seen["options"] = options
            yield SystemMessage(subtype="compact_boundary", data={
                "compact_metadata": {"trigger": "auto", "pre_tokens": 251_000}})

        async def put(event):
            return None

        run = SimpleNamespace(metadata=metadata, native=None, put=put)
        with tempfile.TemporaryDirectory() as home, \
                mock.patch.dict(os.environ, {compaction.ENV: "", "HOME": home}), \
                mock.patch.object(claude, "query", fake_query):
            await claude.ClaudeProvider()._run_strategy(run, "p", None, [])
            self.assertEqual(os.listdir(home), [])  # no user-global config written
        return run, seen["options"]

    async def test_settings_request_250k_and_preserve_the_task(self):
        brief = compaction.compaction_brief(task())
        run, options = await self._run({"compaction_brief": brief})
        settings = json.loads(options.settings)
        self.assertEqual(settings["autoCompactWindow"], 250_000)
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", options.env or {})
        hook = settings["hooks"]["PreCompact"][0]["hooks"][0]
        self.assertEqual(hook["type"], "command")
        printed = subprocess.run(["sh", "-c", hook["command"]], capture_output=True,
                                 text=True, check=True).stdout
        self.assertEqual(printed, brief)
        self.assertEqual(run.metadata["context_compaction"], {
            "requested_auto_compact_tokens": 250_000,
            "mechanism": "claude autoCompactWindow"})
        self.assertEqual(run.metadata["compactions"],
                         [{"trigger": "auto", "pre_tokens": 251_000}])

    async def test_runs_without_a_brief_get_no_hook(self):
        _, options = await self._run({})
        self.assertEqual(json.loads(options.settings), {"autoCompactWindow": 250_000})


class ExecutionBriefTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_execution_hands_the_brief_to_the_executor(self):
        from pathlib import Path
        from agent.routing.config import load_routing_config
        from agent.routing.executors.registry import ExecutorRegistry
        from agent.routing.persistence import RoutingPersistence
        from agent.routing.service import ModelRoutingService
        from agent.routing.task_execution import TaskExecutionService
        from mcp_bridge.models import DoneEvent
        from tests.test_routing import Router
        seen = []

        class Capture:
            async def execute(self, _task, _model, context):
                seen.append(context.owner_metadata)
                yield DoneEvent("endTurn")

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {"CODEPARTNER_EXECUTION_MODE": "ide"}):
            config = load_routing_config()
            service = TaskExecutionService(
                config=config,
                router=ModelRoutingService(config, Router("claude-haiku")),
                persistence=RoutingPersistence(Path(directory) / "routing.sqlite3"),
                executors=ExecutorRegistry({"claude-cli": Capture(), "codex-cli": Capture()}))
            [event async for event in service.execute(
                task("simple"), "CHG1", [], [], owner_metadata={"plan_id": "p"})]
        self.assertEqual(seen[0]["plan_id"], "p")
        self.assertIn(task("simple").id, seen[0]["compaction_brief"])


if __name__ == "__main__":
    unittest.main()

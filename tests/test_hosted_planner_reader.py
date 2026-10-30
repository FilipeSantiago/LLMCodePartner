"""A Codex planner receives briefs without native source-read tools."""

import contextlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.planning.service import WPPlannerService
from agent.planning.hosted_reader import (MAX_BRIEF_CHARS, _source_batch,
                                          declarations_query, read_brief)
from agent.routing.config import load_routing_config
from agent.workspace import NativeRun
from mcp_bridge.models import DoneEvent, TextEvent
from mcp_bridge.models import Run
from mcp_bridge.registry import ROLE_OPTIMIZER, ToolRegistry, ToolSpec, allowed_for
from providers.codex import CodexProvider
from providers.scope_guard import policy_from_run, decide


class SourceBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "a.py").write_text("one\ntwo\nthree\n")

    def request(self, path="a.py", start=1, end=2):
        return {"objective": "Which function handles this?", "files": [
            {"path": path, "start_line": start, "end_line": end,
             "question": "What does this do?"}]}

    def test_exact_lines_are_sent_to_reader_model(self):
        source = _source_batch(str(self.root), self.request(start=2, end=3))
        self.assertIn("a.py:2 two", source)
        self.assertIn("a.py:3 three", source)
        self.assertNotIn("a.py:1 one", source)

    def test_traversal_symlink_and_oversized_ranges_are_rejected(self):
        (self.root / "escape.py").symlink_to(self.root.parent / "outside.py")
        for request in (self.request("../outside.py"), self.request("escape.py"),
                        self.request(end=201), self.request(start=0),
                        {"objective": "", "files": self.request()["files"]}):
            with self.subTest(request=request), self.assertRaises((ValueError, OSError)):
                _source_batch(str(self.root), request)

    def test_missing_path_blank_question_and_total_limit_are_rejected(self):
        blank = self.request()
        blank["files"][0]["question"] = "   "
        many = {"objective": "Read everything", "files": [
            {"path": "a.py", "start_line": 1, "end_line": 200, "question": "Why?"}] * 7}
        for request in (self.request("missing.py"), blank, self.request(start=3, end=2),
                        many):
            with self.subTest(request=request), self.assertRaises((ValueError, OSError)):
                _source_batch(str(self.root), request)

    def test_bulk_declaration_query_has_fixed_projection(self):
        query = declarations_query({"files": ["a.py", "b.py"]})["query"]
        self.assertIn('IN ["a.py", "b.py"]', query)
        self.assertIn("RETURN n.file_path, n.name, n.start_line, n.end_line", query)
        with self.assertRaises(ValueError):
            declarations_query({"files": ["../private.py"]})


class HostedReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_receives_source_but_parent_receives_only_brief(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "a.py").write_text("SECRET_SOURCE_LINE\n")
            calls = []

            class Provider:
                async def tools(self, messages, specs, **kwargs):
                    calls.append((messages, specs, kwargs))
                    yield TextEvent("a.py:1 contains the answer")
                    yield DoneEvent("endTurn", {"input_tokens": 42})

            metrics = {}
            result = await read_brief(root, {
                "objective": "Explain the file", "files": [{"path": "a.py",
                "start_line": 1, "end_line": 1, "question": "What is here?"}]},
                Provider(), "reader-model", "low", metrics)
            self.assertEqual(result, "a.py:1 contains the answer")
            self.assertNotIn("SECRET_SOURCE_LINE", result)
            self.assertIn("SECRET_SOURCE_LINE", calls[0][0][0]["content"])
            self.assertEqual(calls[0][1], [])
            self.assertEqual(metrics["hosted_reader"]["usage"]["input_tokens"], 42)

    async def _brief(self, reply: str, metrics: dict) -> str:
        with tempfile.TemporaryDirectory() as root:
            Path(root, "a.py").write_text("x = 1\n")

            class Provider:
                async def tools(self, messages, specs, **kwargs):
                    yield TextEvent(reply)
                    yield DoneEvent("endTurn", {})

            return await read_brief(root, {
                "objective": "Explain", "files": [{"path": "a.py",
                "start_line": 1, "end_line": 1, "question": "What is here?"}]},
                Provider(), "reader-model", None, metrics)

    async def test_brief_is_capped(self):
        result = await self._brief("a.py:1 " + "x" * (MAX_BRIEF_CHARS * 2), {})
        self.assertEqual(len(result), MAX_BRIEF_CHARS)
        self.assertTrue(result.startswith("a.py:1"))

    async def test_brief_without_citation_is_a_failure(self):
        metrics = {}
        result = await self._brief("It sets a variable.", metrics)
        self.assertIn("lacked a path:line citation", result)
        self.assertEqual(metrics["hosted_reader"]["failures"], 1)

    async def test_model_failure_returns_no_source(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "a.py").write_text("SECRET_SOURCE_LINE\n")

            class Provider:
                async def tools(self, messages, specs, **kwargs):
                    raise RuntimeError("failed")
                    yield  # Keep this an async generator.

            result = await read_brief(root, {
                "objective": "Explain", "files": [{"path": "a.py",
                "start_line": 1, "end_line": 1, "question": "What is here?"}]},
                Provider(), "reader-model", None, {})
            self.assertIn("hosted reader failed", result)
            self.assertNotIn("SECRET_SOURCE_LINE", result)


class PlannerPolicyTests(unittest.TestCase):
    def test_codex_planner_denies_shell_without_cli_child(self):
        run = SimpleNamespace(native=SimpleNamespace(workspace="/project"),
                              metadata={"planner": True, "enforce_scope": True,
                                        "hosted_reader": True})
        policy = policy_from_run(run, [])
        self.assertIsNotNone(policy)
        for name in ("Bash", "exec_command", "Read"):
            with self.subTest(tool=name):
                result = decide({"tool_name": name, "tool_input": {"command": "cat a.py",
                                                                  "file_path": "a.py"}},
                                policy)
                self.assertEqual(result["hookSpecificOutput"]["permissionDecision"],
                                 "deny")

    def test_claude_planner_keeps_direct_read_tools(self):
        run = SimpleNamespace(native=SimpleNamespace(workspace="/project"),
                              metadata={"planner": True, "enforce_scope": True})
        policy = policy_from_run(run, [SimpleNamespace(name="reader")])
        self.assertIsNone(decide({"tool_name": "Read",
                                  "tool_input": {"file_path": "a.py"}}, policy))


class PlannerSurfaceTests(unittest.IsolatedAsyncioTestCase):
    async def _attempt(self, root, provider, model_name, config=None):
        records = []

        class Persistence:
            async def record_planning_turn(self, record):
                records.append(record)

        service = WPPlannerService(None, Persistence(),
                                   config=config or load_routing_config(),
                                   provider_lookup=lambda _: provider)
        service._router = SimpleNamespace(health=SimpleNamespace(record=lambda *a: None))
        specs = [ToolSpec(name, name, {"type": "object", "properties": {}})
                 for name in ("read_file", "kb_search_graph", "kb_query_graph",
                              "kb_get_file_outline")]
        handlers = {spec.name: self._handler for spec in specs}
        native = NativeRun(workspace=root, read_only=True)
        plan = SimpleNamespace(native=True, project=SimpleNamespace(workspace=root),
                               mode="native")
        with mock.patch("agent.planning.service.plan_native", return_value=plan), \
             mock.patch("agent.planning.service.read_only_session", return_value=native):
            result = await service._attempt_planner(
                {"session_id": "test", "wp_id": "WP1", "wp_complexity": "high"},
                1, "plan", "go", [{"role": "user", "content": "plan"}], specs,
                self._progress, handlers, frozenset({"read_file"}), frozenset(),
                model_name, service.config.models[model_name],
                "high", False, 1, False)
        return result, records

    async def test_codex_planner_receives_only_navigation_and_hosted_reader(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "a.py").write_text("def answer():\n    return 42\n")
            calls = []

            class Provider:
                async def tools(self, messages, specs, **kwargs):
                    calls.append((specs, kwargs))
                    if specs:
                        result = await kwargs["direct_tool_handlers"]["planner_reader"]({
                            "objective": "Find answer", "files": [{"path": "a.py",
                            "start_line": 1, "end_line": 2,
                            "question": "What does it return?"}]})
                        self_outer.assertIn("a.py:2", result)
                        yield TextEvent("planner done")
                    else:
                        yield TextEvent("a.py:2 returns 42")
                    yield DoneEvent("endTurn", {"input_tokens": 10})

            self_outer = self
            result, records = await self._attempt(root, Provider(), "codex-sol-high")
            self.assertEqual(result, (None, "planner done"))
            names = {spec.name for spec in calls[0][0]}
            self.assertEqual(names, {"kb_search_graph", "planner_reader",
                                     "planner_declarations"})
            self.assertIsNone(calls[0][1]["subagents"])
            self.assertEqual(calls[1][0], [])
            self.assertEqual(records[0]["tool_metrics"]["native_tool_metrics"]
                             ["hosted_reader"]["calls"], 1)

    async def test_cat_request_has_no_shell_and_goes_through_reader(self):
        # The user asks the planner to `cat` a file through exec_command. The planner
        # surface has no shell or source tool, the hook policy denies the attempt,
        # and the reader answers instead.
        with tempfile.TemporaryDirectory() as root:
            Path(root, "a.py").write_text("TOKEN = 'secret'\n")
            calls = []

            class Provider:
                async def tools(self, messages, specs, **kwargs):
                    calls.append((specs, kwargs))
                    if not specs:
                        yield TextEvent("a.py:1 defines TOKEN")
                    else:
                        handlers = kwargs["direct_tool_handlers"]
                        self_outer.assertNotIn("exec_command", handlers)
                        self_outer.assertNotIn("read_file", handlers)
                        policy = policy_from_run(SimpleNamespace(
                            native=kwargs["native"], metadata=kwargs["run_metadata"]), [])
                        denial = decide({"tool_name": "exec_command",
                                         "tool_input": {"cmd": "cat a.py"}}, policy)
                        self_outer.assertEqual(
                            denial["hookSpecificOutput"]["permissionDecision"], "deny")
                        brief = await handlers["planner_reader"]({
                            "objective": "What does a.py define?", "files": [{
                                "path": "a.py", "start_line": 1, "end_line": 1,
                                "question": "Which names are defined?"}]})
                        self_outer.assertNotIn("secret", brief)
                        yield TextEvent("planner done")
                    yield DoneEvent("endTurn", {"input_tokens": 5})

            self_outer = self
            result, records = await self._attempt(root, Provider(), "codex-sol-high")
            self.assertEqual(result, (None, "planner done"))
            reader = records[0]["tool_metrics"]["native_tool_metrics"]["hosted_reader"]
            self.assertEqual(reader["calls"], 1)
            self.assertEqual(reader["usage"]["input_tokens"], 5)

    async def test_missing_codex_reader_model_is_unavailable_before_execution(self):
        config = load_routing_config()
        config = replace(config, subagents=replace(config.subagents, reader_models={}))
        with tempfile.TemporaryDirectory() as root:
            class Provider:
                async def tools(self, messages, specs, **kwargs):
                    raise AssertionError("the planner model must not start")
                    yield  # Keep this an async generator.

            result, _ = await self._attempt(root, Provider(), "codex-sol-high", config)
        # "unavailable" moves routing to the next provider, such as a Claude planner.
        self.assertEqual(result[0], "unavailable")

    async def test_claude_planner_keeps_its_read_tools(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []

            class Provider:
                async def tools(self, messages, specs, **kwargs):
                    calls.append((specs, kwargs))
                    yield TextEvent("planner done")
                    yield DoneEvent("endTurn", {})

            result, _ = await self._attempt(root, Provider(), "claude-sonnet")
            self.assertEqual(result, (None, "planner done"))
            specs, kwargs = calls[0]
            names = {spec.name for spec in specs}
            self.assertNotIn("planner_reader", names)
            self.assertTrue({"read_file", "kb_get_file_outline"} <= names)
            self.assertFalse(kwargs["run_metadata"]["hosted_reader"])
            self.assertTrue(kwargs["run_metadata"]["enforce_scope"])

    @staticmethod
    async def _handler(_arguments):
        return "navigation only"

    @staticmethod
    async def _progress(_line):
        return None


class CodexArgumentsTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_and_child_disable_shell_after_extra_args(self):
        @contextlib.asynccontextmanager
        async def endpoint(_run):
            yield "http://127.0.0.1:7777/mcp/test"

        class Provider(CodexProvider):
            def __init__(self):
                self.args = []

            async def _exec(self, args, prompt, **kwargs):
                self.args.append(args)
                yield {"type": "turn.completed", "usage": {"input_tokens": 1}}

        provider = Provider()
        with mock.patch("providers.codex.mcp_http.serve_run", endpoint), \
             mock.patch.dict("providers.codex.os.environ",
                             {"CODEX_ARGS": "-c features.shell_tool=true"}):
            for planner in (True, False):
                run = Run(registry=ToolRegistry(allowed_for(ROLE_OPTIMIZER)))
                if planner:
                    run.registry.register([ToolSpec("planner_reader", "Reader",
                                                     {"type": "object", "properties": {}})])
                    run.native = NativeRun(workspace="/tmp", read_only=True)
                run.metadata = {"hosted_reader": True, "planner": planner,
                                "enforce_scope": True}
                result = await provider._run_strategy(run, "prompt", None,
                                                      concrete_model=True)
                self.assertEqual(result, "endTurn")
                args = provider.args[-1]
                self.assertIn("features.shell_tool=false", args)
                self.assertGreater(args.index("features.shell_tool=false"),
                                   args.index("features.shell_tool=true"))
                self.assertIn("mcp_servers={}", args)
                self.assertIn("features.apps=false", args)
                self.assertNotIn("--dangerously-bypass-hook-trust", args)
                self.assertFalse(any(arg.startswith("hooks.PreToolUse=") for arg in args))
                if planner:
                    self.assertFalse(run.native.agent_can_write)
                else:
                    self.assertNotIn("planner_reader", " ".join(args))

    async def test_missing_reader_stops_before_codex(self):
        run = Run(registry=ToolRegistry(allowed_for(ROLE_OPTIMIZER)))
        run.native = NativeRun(workspace="/tmp", read_only=True)
        run.metadata = {"planner": True, "hosted_reader": True,
                        "enforce_scope": True}
        result = await CodexProvider()._run_strategy(run, "prompt", None)
        self.assertEqual(result, "failed")
        self.assertIn("no hosted reader", (await run.next_event()).message)

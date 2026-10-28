"""Provider-neutral scope decisions and the command-hook adapter."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from providers.scope_guard import ScopePolicy, decide, policy_from_run
from providers.claude import _add_usage
from providers.subagents import Tracker
from providers.codex import CodexProvider


class ScopeDecisionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.workspace = Path(self.root.name)
        (self.workspace / "own.py").write_text("line\n" * 250)
        (self.workspace / "other.py").write_text("secret\n")
        self.policy = ScopePolicy("coder", str(self.workspace), ("own.py",), True)

    def event(self, name, **arguments):
        return {"tool_name": name, "tool_input": arguments}

    def verdict(self, event, policy=None):
        result = decide(event, policy or self.policy)
        return result["hookSpecificOutput"]["permissionDecision"] if result else None

    def test_own_ranged_read_is_allowed_and_whole_long_read_is_denied(self):
        self.assertIsNone(self.verdict(self.event("Read", file_path="own.py", offset=10,
                                                  limit=20)))
        self.assertEqual(self.verdict(self.event("Read", file_path="own.py")), "deny")
        self.assertEqual(self.verdict(self.event("Read", file_path="own.py", offset=1,
                                                limit=251)), "deny")

    def test_planned_edit_range_is_the_direct_read_boundary(self):
        policy = ScopePolicy("coder", str(self.workspace), ("own.py",), True,
                             {"own.py": [[10, 30]]})
        self.assertIsNone(self.verdict(self.event("Read", file_path="own.py", offset=10,
                                                  limit=21), policy))
        self.assertEqual(self.verdict(self.event("Read", file_path="own.py", offset=1,
                                                limit=20), policy), "deny")

    def test_non_own_outside_and_symlink_reads_are_denied(self):
        outside = self.workspace.parent / "outside.py"
        (self.workspace / "escape.py").symlink_to(outside)
        for path in ("other.py", str(outside), "escape.py"):
            with self.subTest(path=path):
                self.assertEqual(self.verdict(self.event("Read", file_path=path)), "deny")

    def test_shell_search_and_unrestricted_mcp_are_denied(self):
        for name in ("Bash", "Grep", "mcp__jetbrains__kb_get_code_snippet"):
            self.assertEqual(self.verdict(self.event(name, command="cat other.py")), "deny")

    def test_writes_are_scoped_and_reviewer_cannot_write(self):
        self.assertIsNone(self.verdict(self.event("Edit", file_path="own.py")))
        self.assertEqual(self.verdict(self.event("Write", file_path="other.py")), "deny")
        reviewer = ScopePolicy("reviewer", str(self.workspace), ("own.py",), True)
        self.assertEqual(self.verdict(self.event("Edit", file_path="own.py"), reviewer),
                         "deny")

    def test_subagent_is_exempt_and_spawn_is_foreground(self):
        child = {**self.event("Read", file_path="other.py"), "agent_id": "child"}
        self.assertIsNone(self.verdict(child))
        decision = decide(self.event("Agent", run_in_background=True), self.policy)
        self.assertEqual(decision["hookSpecificOutput"]["updatedInput"]
                         ["run_in_background"], False)
        self.assertIsNone(self.verdict(self.event("collaborationspawn_agent",
                                            agent_type="reader")))

    def test_agent_calls_run_in_foreground_and_only_offered_types(self):
        policy = ScopePolicy("coder", str(self.workspace), ("own.py",), True,
                             subagent_types=("reader", "checks"))
        decision = decide(self.event("Agent", subagent_type="reader", prompt="Read a.py"),
                          policy)
        self.assertEqual(decision["hookSpecificOutput"]["updatedInput"]
                         ["run_in_background"], False)
        self.assertIsNone(decide(self.event("Agent", subagent_type="checks",
                                            run_in_background=False), policy))
        self.assertEqual(self.verdict(self.event("Agent", subagent_type="general-purpose"),
                                      policy), "deny")

    def test_second_reader_request_needs_a_scope_exception(self):
        policy = ScopePolicy("coder", str(self.workspace), ("own.py",), True,
                             subagent_types=("reader", "checks"))
        state = {}
        first = decide(self.event("Agent", subagent_type="reader", prompt="Read a.py"),
                       policy, state)
        self.assertEqual(first["hookSpecificOutput"]["permissionDecision"], "allow")
        again = decide(self.event("Agent", subagent_type="reader", prompt="Read b.py"),
                       policy, state)
        self.assertEqual(again["hookSpecificOutput"]["permissionDecision"], "deny")
        excepted = decide(self.event("Agent", subagent_type="reader", prompt=(
            "SCOPE EXCEPTION: the planned range moved\nRead b.py")), policy, state)
        self.assertEqual(excepted["hookSpecificOutput"]["permissionDecision"], "allow")

    def test_parent_shell_is_denied_and_checks_child_shell_is_allowed(self):
        self.assertEqual(self.verdict(self.event("Bash", command="uv run pytest")), "deny")
        child = {**self.event("Bash", command="uv run pytest"), "agent_id": "checks-1"}
        self.assertIsNone(self.verdict(child))

    def test_kb_navigation_is_allowed_and_source_tools_are_not(self):
        policy = ScopePolicy("coder", str(self.workspace), ("own.py",), True,
                             navigation_tools=("kb_get_file_outline", "kb_search_graph"))
        self.assertIsNone(self.verdict(self.event(
            "mcp__codepartner__kb_get_file_outline", file_path="other.py"), policy))
        self.assertEqual(self.verdict(self.event(
            "mcp__codepartner__kb_get_code_snippet", qualified_name="x"), policy), "deny")

    def test_no_reader_means_no_scope_decisions(self):
        policy = ScopePolicy("coder", str(self.workspace), ("own.py",), False)
        self.assertIsNone(self.verdict(self.event("Read", file_path="other.py"), policy))

    def test_no_scope_metadata_does_not_activate_policy(self):
        run = SimpleNamespace(native=SimpleNamespace(workspace=str(self.workspace)),
                              metadata={"task_execution": True, "enforce_scope": True})
        self.assertIsNone(policy_from_run(run, [SimpleNamespace(name="reader")]))

    def test_command_adapter_logs_denial(self):
        policy_file = self.workspace / "scope_policy.json"
        policy_file.write_text(json.dumps(asdict(self.policy)))
        result = subprocess.run(
            [sys.executable, "-m", "providers.scope_guard", str(policy_file)],
            input=json.dumps(self.event("Read", file_path="other.py")),
            text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)["hookSpecificOutput"]
                         ["permissionDecision"], "deny")
        log = (self.workspace / "scope_denials.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(log[0])["tool"], "Read")


class AccountingTests(unittest.TestCase):
    def test_claude_result_segments_add_nested_usage_and_iterations(self):
        total = None
        for part in ({"input_tokens": 110000, "cache": {"read": 3}, "iterations": [1]},
                     {"input_tokens": 120000, "cache": {"read": 5}, "iterations": [2]},
                     {"input_tokens": 121000, "cache": {"read": 7}, "iterations": [3]},
                     {"input_tokens": 154000, "cache": {"read": 11}, "iterations": [4]}):
            total = _add_usage(total, part)
        self.assertEqual(total, {"input_tokens": 505000, "cache": {"read": 26},
                                 "iterations": [1, 2, 3, 4]})

    def test_tracker_counts_foreground_and_background_spawns(self):
        metrics = {}
        tracker = Tracker(metrics, offered=True)
        tracker.spawned("reader", "src/a.py", description="Read a")
        tracker.spawned("checks", "`pytest`", background=True,
                        description="Run tests")
        data = metrics["subagents"]
        self.assertEqual(data["foreground_calls"], 1)
        self.assertEqual(data["background_calls"], 1)
        self.assertEqual(data["descriptions"], ["Read a", "Run tests"])


class CodexBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_parent_fails_before_cli_when_reader_cannot_be_isolated(self):
        native = SimpleNamespace(workspace="/workspace", metrics={})
        run = SimpleNamespace(native=native, subagents=object(),
                              metadata={"task_execution": True, "enforce_scope": True,
                                        "edit_files": ["own.py"]},
                              put=mock.AsyncMock())
        with mock.patch("providers.codex.subagents.resolve", return_value=[
            SimpleNamespace(name="reader")]), \
             mock.patch.object(CodexProvider, "_exec", side_effect=AssertionError(
                 "CLI must not start")):
            terminal = await CodexProvider()._run_strategy(run, "task", None)
        self.assertEqual(terminal, "failed")
        self.assertIn("cannot separate parent and reader", run.put.await_args.args[0].message)


if __name__ == "__main__":
    unittest.main()

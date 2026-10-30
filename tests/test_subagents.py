"""Native subagents: source delegation, Claude/Codex wiring and telemetry."""

import os
import tempfile
import tomllib
import unittest
from types import SimpleNamespace
from unittest import mock

from claude_agent_sdk import (AssistantMessage, ResultMessage, TextBlock, ToolResultBlock,
                              ToolUseBlock, UserMessage)

from agent import workspace
from agent.prompt import PlannerPromptContext, PromptFactory, PromptRole
from agent.prompt.coder import EXPLORATION_RULES, CoderPromptBuilder, _planned
from agent.routing.config import load_routing_config
from agent.routing.task_execution import TaskExecutionService
from mcp_bridge.models import Run
from mcp_bridge.registry import ToolSpec
from providers import subagents
from providers.codex import CodexProvider, record_collab_item, role_args, role_config
from tests.native_mode import native_pinned
from tests.test_routing import task

ON = {subagents.ENV: "true"}
KB_SPECS = [ToolSpec("kb_search_graph", "KB", {"type": "object"}),
            ToolSpec("kb_get_code_snippet", "KB", {"type": "object"})]
MODELS = {"claude": ("haiku", None), "codex": ("gpt-cheap", "low")}
SCOPE = ("data_loader/factory.py", "tests/test_factory.py")


def native(read_only=False):
    return workspace.NativeRun(workspace="/work", read_only=read_only)


def request(purpose="task", roles=(subagents.READER, subagents.CHECKS), scope=SCOPE,
            read_only=()):
    with mock.patch.dict(os.environ, ON):
        return subagents.request(purpose, roles=roles, scope=scope, models=MODELS,
                                 max_turns=7, read_only=read_only)


def turn_prompt(brief, discussion, current, message, **options) -> str:
    """The planner's turn prompt, built the way the planning service builds it."""
    return PromptFactory.create(PromptRole.PLANNER).with_context(PlannerPromptContext(
        brief, discussion, current, message, **options)).build().text


class DefinitionTests(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {subagents.ENV: ""}):
            self.assertIsNone(subagents.request("task"))
        self.assertEqual(subagents.exploration_rules(None), "")
        self.assertEqual(subagents.resolve(None, native()), [])

    def test_read_only_runs_never_get_checks(self):
        req = request()
        self.assertEqual([c.name for c in subagents.resolve(req, native())],
                         ["reader", "checks"])
        self.assertEqual([c.name for c in subagents.resolve(req, native(read_only=True))],
                         ["reader"])
        self.assertEqual(subagents.resolve(req, None), [])

    def test_subagents_execute_what_they_are_given(self):
        (reader, checks) = subagents.resolve(request(), native())
        for step in ("exact files, line ranges and questions", "in a single response",
                     "Read only what is listed", "path:line", "Never search",
                     "Missing context:"):
            self.assertIn(step, reader.prompt)
        self.assertIn("Never transcribe whole files or long ranges, even when asked",
                      reader.prompt)
        self.assertIn("parent should Read the file itself", reader.prompt)
        # No KB procedure and no scope: the parent resolved everything already.
        self.assertNotIn("Codebase Memory", reader.prompt)
        self.assertFalse(any(path in reader.prompt for path in SCOPE))
        self.assertIn("Never look for tests or commands", checks.prompt)
        self.assertIn("Missing context: no command", checks.prompt)
        self.assertNotIn("locate its tests", checks.prompt)
        self.assertIn("does not search", reader.description)
        (planner,) = subagents.resolve(request("planner", (subagents.READER,), ()),
                                       native(read_only=True))
        self.assertIn("Work Package planner", planner.prompt)

    def test_rules_per_role(self):
        task_rules = subagents.exploration_rules(
            request(scope=(*SCOPE, "trainer/factory.py"),
                    read_only=("trainer/factory.py", "evaluation/vendor.py")))
        for text in ("DELEGATION RULES", "direct reads are limited to planned edit ranges",
                     "a concrete question the plan leaves open", "SCOPE EXCEPTION:",
                     "Files to change (planner-assigned):\n- data_loader/factory.py\n"
                     "- tests/test_factory.py\nRead-only context (reader only):\n"
                     "- trainer/factory.py\n- evaluation/vendor.py",
                     "Do not use shell commands", "checks subagent", "exact commands"):
            self.assertIn(text, task_rules)
        self.assertNotIn("NO DIRECT SOURCE READS", task_rules)
        self.assertNotIn("Read source yourself", task_rules)
        self.assertNotIn("with a shell, read them", task_rules)
        self.assertNotIn("Justification", task_rules)
        planner = subagents.exploration_rules(request("planner", (subagents.READER,), ()))
        self.assertIn("explore as widely as the Work Package needs", planner)
        self.assertNotIn("planner-assigned files", planner)
        self.assertNotIn("checks", planner)
        # The reviewer owns distinct scope wording for its journaled files.
        reviewer = subagents.exploration_rules(
            request("reviewer", scope=(*SCOPE, "trainer/factory.py"),
                    read_only=("trainer/factory.py", "evaluation/vendor.py")))
        self.assertNotEqual(reviewer, task_rules)
        self.assertIn("journaled-file regions", reviewer)

    def test_request_carries_models_and_turns(self):
        req = request()
        self.assertEqual(req.model_for("codex"), ("gpt-cheap", "low"))
        self.assertEqual(req.model_for("other"), (None, None))
        self.assertEqual(req.turns(), 7)
        with mock.patch.dict(os.environ, {subagents.TURNS_ENV: "3"}):
            self.assertEqual(req.turns(), 3)

    def test_config_names_existing_cheap_models(self):
        config = load_routing_config()
        self.assertEqual(config.subagent_models(),
                         {"claude": ("haiku", None), "codex": ("gpt-5.6-luna", "low")})


class ClassifierTests(unittest.TestCase):
    def test_exploration_exact_reads_and_other(self):
        classify = subagents.classify_command
        self.assertEqual(classify("rg -n Loader src")[0], "exploration")
        self.assertEqual(classify("cd src && grep -rn Loader .")[0], "exploration")
        self.assertEqual(classify("find . -name '*.py'")[0], "exploration")
        self.assertEqual(classify("ls -R pkg")[0], "exploration")
        self.assertEqual(classify("/bin/bash -lc 'git grep Loader'")[0], "exploration")
        self.assertEqual(classify("sed -n '10,40p' a.py; sed -n '5,25p' pkg/b.py"),
                         ("exact_read", [("a.py", True), ("pkg/b.py", True)]))
        self.assertEqual(classify("cat pkg/a.py"), ("exact_read", [("pkg/a.py", False)]))
        self.assertEqual(classify("cat pkg/a.py | head -20"),
                         ("exact_read", [("pkg/a.py", True)]))
        self.assertEqual(classify("uv run pytest -q | tail -40"), ("other", []))
        self.assertEqual(classify("ls -la"), ("other", []))


class TrackerTests(unittest.TestCase):
    def test_reads_rereads_batches_and_briefs(self):
        metrics = {"tool_calls": 0, "tools": {}, "files": []}
        tracker = subagents.Tracker(metrics, request(), offered=True, workspace="/work")
        tracker.parent_reads([("pkg/a.py", 1, 20), ("/work/pkg/b.py", None, None)])
        tracker.parent_reads([("pkg/a.py", 1, 20)])  # the same range again
        tracker.brief("Defined at `pkg/a.py:3-9`, used in ./pkg/b.py:12.\n"
                      "Missing context: pkg/c.py:4", "reader")
        tracker.parent_read("pkg/b.py", ranged=False)
        tracker.parent_command("rg -n Loader")
        tracker.parent_command("sed -n '1,9p' pkg/a.py")
        reads = metrics["exploration"]
        self.assertEqual(reads["mode"], "reader")
        self.assertEqual((reads["direct_reads"], reads["ranged_reads"],
                          reads["whole_file_reads"]), (5, 3, 2))
        self.assertEqual((reads["batched_read_turns"], reads["reads_in_batches"]), (1, 2))
        self.assertEqual(reads["exact_rereads"], 1)
        self.assertEqual((reads["bash_exploration"], reads["bash_exact_reads"]), (1, 1))
        self.assertEqual((reads["brief_file_reads"], reads["brief_full_rereads"]), (2, 1))
        self.assertEqual(metrics["subagents"]["missing_context_reported"], 1)
        self.assertEqual(metrics["subagents"]["empty_briefs"], 0)

    def test_underspecified_requests_and_empty_briefs_are_counted(self):
        metrics = {}
        tracker = subagents.Tracker(metrics, request(), offered=True)
        tracker.spawned("reader", "Check data_loader/factory.py:1-40 and tests/test_factory.py")
        tracker.spawned("reader", "How are the loaders registered?")
        tracker.spawned("checks", "Run `uv run pytest -q tests/test_factory.py`")
        tracker.spawned("checks", "Run the factory tests")
        tracker.brief("I'll investigate the repository. agentId: a1", "reader")
        tracker.brief("1 passed", "checks")  # a short checks result is not a brief
        data = metrics["subagents"]
        self.assertEqual((data["reader_calls"], data["checks_calls"],
                          data["requests_without_paths"], data["empty_briefs"]), (2, 2, 2, 1))

    def test_host_and_workspace_paths_count_as_one_file(self):
        metrics = {}
        run = workspace.NativeRun(workspace="/workspace/proj", host_path="/home/u/Proj")
        tracker = subagents.Tracker(metrics, request(), offered=True, native=run)
        tracker.parent_reads([("pkg/a.py", None, None), ("/workspace/proj/pkg/a.py", None, None),
                              ("/home/u/Proj/pkg/a.py", None, None)])
        self.assertEqual(metrics["exploration"]["exact_rereads"], 2)

    def test_scope_and_edit_discipline_counters(self):
        metrics = {}
        tracker = subagents.Tracker(
            metrics, request(read_only=("trainer/factory.py",)), offered=True,
            workspace="/work")
        tracker.spawned("reader", "Check data_loader/factory.py:1-40, trainer/factory.py "
                                  "and pkg/decoders/base.py")
        tracker.spawned("checks", "Run `uv run pytest pkg/other.py`")
        tracker.parent_read("data_loader/factory.py", ranged=False)
        tracker.parent_read("/work/tests/test_factory.py", ranged=True)
        tracker.parent_edit("/work/data_loader/factory.py")
        tracker.parent_edit("tests/test_factory.py")
        tracker.parent_edit("data_loader/factory.py")
        tracker.parent_read("data_loader/factory.py", ranged=False)
        self.assertEqual(metrics["subagents"]["out_of_scope_paths"], 1)
        self.assertEqual(metrics["exploration"]["whole_reads_of_edited"], 1)
        self.assertEqual(metrics["exploration"]["post_edit_rereads"], 1)

    def test_absolute_paths_are_in_scope_and_symbols_are_not_paths(self):
        metrics = {}
        tracker = subagents.Tracker(metrics, request(), offered=True, workspace="/work")
        tracker.spawned("reader", "Read /work/data_loader/factory.py:1-40 and explain "
                                  "Model.remi_predict, e.g. its return value.")
        data = metrics["subagents"]
        self.assertEqual((data["objectives_paths"], data["out_of_scope_paths"]), (1, 0))

    def test_async_launch_is_no_brief_and_absolute_citations_count(self):
        metrics = {}
        tracker = subagents.Tracker(metrics, request(), offered=True, workspace="/work")
        tracker.brief("Async agent launched successfully. agentId: a1", "reader")
        data = metrics["subagents"]
        self.assertEqual((data["async_launches"], data["brief_bytes"], data["empty_briefs"]),
                         (1, 0, 0))
        tracker.brief("Defined at /work/data_loader/factory.py:12-20.", "reader")
        self.assertEqual(data["cited_files"], 1)

    def test_read_turns_count_responses_with_parent_reads(self):
        metrics = {}
        tracker = subagents.Tracker(metrics, request(), offered=True, workspace="/work")
        tracker.parent_reads([("a.py", 1, 10), ("b.py", 1, 10)])
        tracker.parent_reads([])
        tracker.parent_reads([("c.py", 1, 10)])
        self.assertEqual(metrics["exploration"]["read_turns"], 2)

    def test_off_runs_track_reads_without_subagent_block(self):
        metrics = {}
        tracker = subagents.Tracker(metrics, None, offered=False)
        tracker.parent_command("grep -rn x .")
        self.assertEqual(metrics["exploration"]["mode"], "off")
        self.assertEqual(metrics["exploration"]["bash_exploration"], 1)
        self.assertNotIn("subagents", metrics)

    def test_cost_attribution_is_labelled(self):
        usage = {"claude-sonnet-5": {"costUSD": 1.0, "inputTokens": 10, "outputTokens": 5},
                 "claude-haiku-5": {"costUSD": 0.1, "inputTokens": 20, "outputTokens": 2}}
        split = subagents.cost_attribution(usage, "haiku", None)
        self.assertEqual(split["method"], "estimated_by_model")
        self.assertEqual(split["tokens_method"], "estimated_by_model")
        self.assertAlmostEqual(split["subagents"]["cost_usd"], 0.1)
        self.assertEqual(split["parent"]["tokens"], 15)
        self.assertFalse(split["ambiguous"])
        exact = subagents.cost_attribution(usage, "haiku", {"input_tokens": 20})
        self.assertEqual(exact["tokens_method"], "exact")
        shared = subagents.cost_attribution({"claude-haiku-5": {"costUSD": 1}}, "haiku", None)
        self.assertTrue(shared["ambiguous"])


class PromptTests(unittest.TestCase):
    def test_flag_off_coder_prompt_is_unchanged(self):
        plain = TaskExecutionService._execution_messages(
            task(), [{"role": "user", "content": "x"}])[-1]["content"]
        self.assertIn(EXPLORATION_RULES, plain)
        self.assertNotIn("DELEGATION RULES", plain)
        self.assertIn("tools to read, search and edit files", plain)

    def test_coder_prompt_replaces_exploration_part_when_enabled(self):
        rules = subagents.exploration_rules(request())
        prompt = TaskExecutionService._execution_messages(
            task(), [{"role": "user", "content": "x"}],
            subagent_request=request())[-1]["content"]
        self.assertIn(rules, prompt)
        self.assertNotIn(EXPLORATION_RULES, prompt)
        self.assertNotIn("Never read the whole repository", prompt)
        self.assertIn("more context comes through the reader", prompt)
        self.assertIn("tools to locate paths and edit files", prompt)
        self.assertNotIn("tools to read known ranges", prompt)

    def test_planner_prompt_adds_rules_only_when_native(self):
        planner_request = request("planner", (subagents.READER,), ())
        native_prompt = turn_prompt("brief", [], None, "go", native=True,
                                    subagent_request=planner_request)
        self.assertEqual(native_prompt.count("DELEGATION RULES"), 1)
        self.assertEqual(native_prompt.count("## Read pipeline"), 1)
        self.assertIn("NO DIRECT SOURCE READS", native_prompt)
        self.assertIn("the evidence gap instead of reading source yourself", native_prompt)
        self.assertNotIn("make only the minimal bounded parent reads needed and state", native_prompt)
        self.assertNotIn("DELEGATION RULES", turn_prompt("brief", [], None, "go",
                                                         native=False,
                                                         subagent_request=planner_request))
        self.assertNotIn("DELEGATION RULES", turn_prompt("brief", [], None, "go",
                                                         native=True))


class ClaudeTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, messages=(), read_only=False, enabled=True, metadata=None):
        from providers import claude
        seen = {}

        async def fake_query(prompt, options):
            seen["options"] = options
            for message in messages:
                yield message

        events = []

        async def put(event):
            events.append(event)

        commands = []

        async def on_command(command, output, code):
            commands.append(command)

        run = Run()
        run.put = put
        run.metadata.update(metadata or {})
        run.native = workspace.NativeRun(workspace="/work", read_only=read_only,
                                         on_command=on_command)
        run.subagents = request() if enabled else None
        with mock.patch.object(claude, "_query_messages", fake_query):
            await claude.ClaudeProvider()._run_strategy(run, "p", None, KB_SPECS)
        return seen["options"], run, events, commands

    async def test_parent_explores_and_each_subagent_gets_one_tool(self):
        options, *_ = await self._run()
        for kept in ("Agent", "Read", "Grep", "Glob", "Edit", "Write", "Bash"):
            self.assertIn(kept, options.tools)
        reader = options.agents["reader"]
        self.assertEqual((reader.model, reader.maxTurns, reader.background), ("haiku", 7, False))
        self.assertEqual(reader.tools, ["Read"])
        self.assertEqual(options.agents["checks"].tools, ["Bash"])
        # Every subagent tool is allowed, so no subagent call is ever permission-denied.
        for agent in options.agents.values():
            self.assertTrue(set(agent.tools) <= set(options.allowed_tools))
        self.assertTrue(options.strict_mcp_config)
        self.assertEqual(options.permission_mode, "acceptEdits")
        read_only, *_ = await self._run(read_only=True)
        self.assertEqual(sorted(read_only.agents), ["reader"])
        self.assertEqual(sorted(read_only.tools), ["Agent", "Glob", "Grep", "Read"])
        self.assertEqual(read_only.agents["reader"].tools, ["Read"])
        self.assertEqual(read_only.permission_mode, "default")

    async def test_scoped_coder_gets_bash_for_checks_and_kb_navigation_only(self):
        options, *_ = await self._run(metadata={
            "task_execution": True, "enforce_scope": True, "edit_files": ["a.py"]})
        self.assertIn("Bash", options.tools)
        self.assertEqual(options.agents["checks"].tools, ["Bash"])
        self.assertIn("mcp__jetbrains__kb_search_graph", options.allowed_tools)
        self.assertNotIn("mcp__jetbrains__kb_get_code_snippet", options.allowed_tools)
        self.assertIn("PreToolUse", options.hooks)
        self.assertEqual(options.env["CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS"], "1")

    async def test_unscoped_run_with_subagents_still_normalizes_agent_calls(self):
        options, *_ = await self._run()
        hook = options.hooks["PreToolUse"][0].hooks[0]
        result = await hook({"tool_name": "Agent", "tool_input": {
            "subagent_type": "reader", "prompt": "Read a.py"}}, None, None)
        self.assertEqual(result["hookSpecificOutput"]["updatedInput"]["run_in_background"],
                         False)
        denied = await hook({"tool_name": "Agent", "tool_input": {
            "subagent_type": "general-purpose", "prompt": "Run tests"}}, None, None)
        self.assertEqual(denied["hookSpecificOutput"]["permissionDecision"], "deny")

    async def test_flag_off_keeps_today_s_tools(self):
        off, run, *_ = await self._run(enabled=False)
        self.assertEqual(off.tools, ["Read", "Grep", "Glob", "Edit", "MultiEdit", "Write",
                                     "Bash"])
        self.assertIsNone(getattr(off, "agents", None))
        self.assertEqual(run.metadata["exploration_mode"], "off")

    async def test_stream_telemetry(self):
        messages = [
            AssistantMessage([ToolUseBlock("agent-1", "Agent", {
                "subagent_type": "reader", "prompt": "Check data_loader/factory.py"})], "opus"),
            AssistantMessage([TextBlock("child chatter"),
                              ToolUseBlock("c1", "mcp__jetbrains__kb_search_graph", {}),
                              ToolUseBlock("c2", "Bash", {"command": "rg -n Loader"})],
                             "haiku", parent_tool_use_id="agent-1"),
            UserMessage([ToolResultBlock("c2", "x", False)], parent_tool_use_id="agent-1"),
            UserMessage([ToolResultBlock("agent-1", [{"type": "text",
                                                      "text": "See pkg/a.py:3-9."}])],
                        tool_use_result={"usage": {"input_tokens": 40, "output_tokens": 4}}),
            AssistantMessage([ToolUseBlock("p2", "Read", {"file_path": "pkg/a.py"}),
                              ToolUseBlock("p3", "Read", {"file_path": "pkg/b.py",
                                                          "offset": 3, "limit": 9})], "opus"),
            AssistantMessage([ToolUseBlock("p4", "Bash", {"command": "grep -rn x ."})], "opus"),
            ResultMessage("success", 1, 1, False, 1, "s", total_cost_usd=1.1, usage={},
                          model_usage={"claude-opus-5": {"costUSD": 1.0, "inputTokens": 9},
                                       "claude-haiku-5": {"costUSD": 0.1, "inputTokens": 40}}),
        ]
        _, run, events, commands = await self._run(messages)
        self.assertNotIn("child chatter", [getattr(event, "text", "") for event in events])
        metrics = run.native.metrics
        self.assertEqual(metrics["tools"], {"Agent": 1, "Read": 2, "Bash": 1})
        data = metrics["subagents"]
        self.assertEqual((data["reader_calls"], data["child_tool_calls"],
                          data["child_kb_calls"], data["child_bash_exploration"]),
                         (1, 2, 1, 1))
        self.assertEqual(data["child_usage"], {"input_tokens": 40, "output_tokens": 4})
        reads = metrics["exploration"]
        self.assertEqual((reads["direct_reads"], reads["batched_read_turns"],
                          reads["brief_full_rereads"], reads["bash_exploration"]), (2, 1, 1, 1))
        split = metrics["cost_attribution"]
        self.assertEqual((split["method"], split["tokens_method"]),
                         ("estimated_by_model", "exact"))
        self.assertAlmostEqual(split["subagents"]["cost_usd"], 0.1)
        self.assertEqual(run.metadata["exploration_mode"], "reader")
        # The subagent's command became execution evidence once its result arrived.
        self.assertEqual(commands, ["rg -n Loader"])


class CodexTests(unittest.TestCase):
    def setUp(self):
        self.request = request()
        self.children = subagents.resolve(self.request, native())

    def test_role_files_use_the_configured_reader_model(self):
        config = tomllib.loads(role_config(self.children[0], self.request))
        self.assertEqual((config["model"], config["model_reasoning_effort"]),
                         ("gpt-cheap", "low"))
        self.assertIn("Never search for files", config["developer_instructions"])
        self.assertIn("Missing context:", config["developer_instructions"])

    def test_role_flags_precede_the_escape_hatch_and_sandbox(self):
        with tempfile.TemporaryDirectory() as directory:
            flags = role_args(self.children, directory, self.request)
            self.assertTrue(os.path.exists(os.path.join(directory, "reader.toml")))
            with mock.patch.dict(os.environ, {"CODEX_ARGS": "-c x=1"}):
                args = CodexProvider()._base_args(native=native(), subagent_args=flags)
        self.assertIn("agents.max_depth=1", args)
        self.assertLess(args.index("agents.max_depth=1"), args.index("x=1"))
        self.assertEqual(args[-1], "--dangerously-bypass-approvals-and-sandbox")
        self.assertEqual(role_args([], "/unused"), [])

    def test_collab_items_and_shell_commands(self):
        metrics = {}
        tracker, briefed = subagents.Tracker(metrics, self.request, offered=True), set()
        record_collab_item(tracker, {"type": "collab_tool_call", "tool": "spawn_agent",
                                     "prompt": "Inspect trainer/loop.py",
                                     "agents_states": {"t1": {"status": "pending_init"}}},
                           briefed)
        done = {"type": "collab_tool_call", "tool": "wait",
                "agents_states": {"t1": {"status": "completed", "message": "calc.py:1"}}}
        record_collab_item(tracker, done, briefed)
        record_collab_item(tracker, done, briefed)
        tracker.parent_command("/bin/bash -lc 'rg -n Loader'")
        data = metrics["subagents"]
        self.assertEqual((data["calls"], data["cited_files"], data["requests_without_paths"]),
                         (1, 1, 0))
        self.assertEqual(metrics["exploration"]["bash_exploration"], 1)


class ProgressLineTests(unittest.TestCase):
    def test_native_tool_lines(self):
        from mcp_bridge.activity import describe_native
        self.assertEqual(describe_native("Read", {"file_path": "/work/pkg/a.py", "offset": 10,
                                                  "limit": 31}, "/work"),
                         "Reading `pkg/a.py` (lines 10–40)")
        self.assertEqual(describe_native("Read", {"file_path": "a.py"}), "Reading `a.py`")
        self.assertEqual(describe_native("Edit", {"file_path": "a.py"}), "Editing `a.py`")
        self.assertEqual(describe_native("Bash", {"command": "uv run pytest -q"}),
                         "Running `uv run pytest -q`")
        self.assertEqual(describe_native("Grep", {"pattern": "Loader"}),
                         "Searching for `Loader`")
        self.assertIsNone(describe_native("mcp__jetbrains__kb_search_graph", {}))

    def test_subagent_lines(self):
        line = subagents.describe_spawn(
            "reader", "How are loaders registered?\nCheck data_loader/factory.py, "
                      "trainer/loop.py")
        self.assertEqual(line, "🧭 Reader: How are loaders registered? — "
                               "`data_loader/factory.py`, `trainer/loop.py`")
        self.assertEqual(subagents.describe_spawn("checks", "Run `uv run pytest -q` now"),
                         "🧪 Checks: `uv run pytest -q`")
        self.assertEqual(subagents.describe_brief("reader", "a.py:3 and b/c.py:4-9 ok"),
                         "📋 Reader brief: 2 files cited (~4 words)")
        self.assertIn("missing context reported", subagents.describe_brief(
            "reader", "a.py:1\nMissing context: d.py:2"))
        self.assertEqual(subagents.describe_brief("reader", "I'll investigate the repo."),
                         "⚠️ Reader returned no brief (likely its turn limit)")
        self.assertEqual(subagents.describe_brief("checks", "1 passed"), "🧪 Checks done")

    def test_codex_item_lines(self):
        from providers.codex import progress_line
        self.assertEqual(progress_line({"type": "command_execution",
                                        "command": "/bin/bash -lc \"sed -n '1,9p' a.py\""}),
                         "🔎 Reading `a.py`")
        self.assertEqual(progress_line({"type": "command_execution", "command": "rg -n x"}),
                         "🔎 Running `rg -n x`")
        self.assertEqual(progress_line({"type": "collab_tool_call", "tool": "spawn_agent",
                                        "prompt": "Map the loaders in pkg/a.py"}),
                         "🧭 Reader: Map the loaders in pkg/a.py — `pkg/a.py`")
        self.assertIsNone(progress_line({"type": "collab_tool_call", "tool": "wait"}))


class ProgressStreamTests(unittest.IsolatedAsyncioTestCase):
    MESSAGES = [
        AssistantMessage([ToolUseBlock("agent-1", "Agent", {
            "subagent_type": "reader", "prompt": "Map loaders in data_loader/factory.py"})],
            "opus"),
        AssistantMessage([ToolUseBlock("c1", "Read", {"file_path": "hidden.py"})], "haiku",
                         parent_tool_use_id="agent-1"),
        UserMessage([ToolResultBlock("agent-1", "See data_loader/factory.py:3-9.")]),
        AssistantMessage([ToolUseBlock("p1", "Read", {"file_path": "/work/a.py",
                                                      "offset": 3, "limit": 7})], "opus"),
    ]

    async def _lines(self, metadata):
        from mcp_bridge.models import ProgressEvent
        from providers import claude

        async def fake_query(prompt, options):
            for message in self.MESSAGES:
                yield message

        events = []

        async def put(event):
            events.append(event)

        run = Run()
        run.put = put
        run.metadata.update(metadata)
        run.native = workspace.NativeRun(workspace="/work")
        run.subagents = request()
        with mock.patch.object(claude, "_query_messages", fake_query):
            await claude.ClaudeProvider()._run_strategy(run, "p", None, KB_SPECS)
        return [event.text for event in events if isinstance(event, ProgressEvent)]

    async def test_coder_run_streams_parent_steps_only(self):
        lines = await self._lines({"task_execution": True})
        self.assertEqual(lines, [
            "- 🧭 Reader: Map loaders in data_loader/factory.py — `data_loader/factory.py`\n",
            "- 📋 Reader brief: 1 file cited (~2 words)\n",
            "- 🔎 Reading `a.py` (lines 3–9)\n"])
        self.assertFalse(any("hidden.py" in line for line in lines))

    async def test_runs_without_a_progress_consumer_stay_quiet(self):
        self.assertEqual(await self._lines({}), [])

    async def test_planner_forwards_native_progress(self):
        from mcp_bridge.models import DoneEvent, ProgressEvent, TextEvent
        from agent.planning.service import WPPlannerService
        from agent.routing.persistence import RoutingPersistence
        from tests.test_turn_routing import _specs
        from tests.test_wp_planner import PlanArtifacts, plan_payload
        import json

        class Provider:
            def tools(self, conversation, specs, **kwargs):
                async def events():
                    yield ProgressEvent("- 🧭 Reader: map loaders\n")
                    yield TextEvent(json.dumps(plan_payload()))
                    yield DoneEvent("endTurn")
                return events()

        lines = []

        async def capture(line):
            lines.append(line)

        with tempfile.TemporaryDirectory() as directory, \
                native_pinned():
            planner = WPPlannerService(
                PlanArtifacts(), RoutingPersistence(os.path.join(directory, "r.sqlite3")),
                config=load_routing_config(), provider_lookup=lambda _name: Provider())
            await planner.start("conv", "WP4", "go", _specs(), {}, progress=capture)
        self.assertIn("- 🧭 Reader: map loaders\n", lines)


class TaskComparisonTests(unittest.TestCase):
    def test_per_task_totals_by_mode(self):
        from telemetry.aggregation import TelemetryAggregation
        from telemetry.storage import TelemetryStorage
        with tempfile.TemporaryDirectory() as directory:
            storage = TelemetryStorage(os.path.join(directory, "t.sqlite3"))
            claude = {"implementation_job_id": "J", "task_id": "T1", "provider": "claude",
                      "model": "s", "cost_usd": 1.5}
            storage.record_aci_attempt("r1", claude, {
                "execution_mode": "native", "exploration_mode": "reader",
                "exploration": {"direct_reads": 2, "bash_exploration": 1},
                "subagents": {"reader_calls": 3}},
                {"input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 5})
            codex = {**claude, "provider": "codex", "cost_usd": None}
            storage.record_aci_attempt("r2", codex, {
                "execution_mode": "native", "exploration_mode": "reader"},
                {"input_tokens": 50, "cached_input_tokens": 40, "output_tokens": 1})
            storage.record_aci_attempt("r3", {**claude, "task_id": "T2"}, {
                "execution_mode": "native", "exploration_mode": "off"}, {"input_tokens": 1})
            report = TelemetryAggregation(storage).task_comparison(["J"])
        row = report["J"]["T1"]["reader"]
        self.assertEqual(row["attempts"], 2)
        self.assertEqual(row["cache_read_tokens"], 140)
        self.assertEqual(row["uncached_input_tokens"], 20)
        self.assertIsNone(row["cost_usd"])  # Codex reported no cost: no partial total
        self.assertEqual((row["direct_reads"], row["reader_calls"]), (2, 3))
        self.assertEqual(report["J"]["T2"]["off"]["cost_usd"], 1.5)



class CoderRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_claude_executor_forces_5m_cache_with_the_gateway(self):
        from agent.routing.executors.claude import ClaudeCLIExecutor
        from agent.routing.models import ExecutionContext

        class Provider:
            def __init__(self):
                self.env = None

            async def tools(self, messages, specs, **kwargs):
                self.env = kwargs["execution_env"]
                return
                yield

        decision = SimpleNamespace(decision_id="d1")
        model = SimpleNamespace(cli_model="sonnet", name="claude-sonnet")
        for ttl, gateway, expected in (
                ("5m", True, {"ANTHROPIC_BASE_URL": "http://gw",
                              "FORCE_PROMPT_CACHING_5M": "1"}),
                ("1h", False, {})):
            provider = Provider()
            context = ExecutionContext(messages=[], specs=[], decision=decision,
                                       gateway_url="http://gw", use_gateway=gateway,
                                       prompt_cache_ttl=ttl)
            async for _ in ClaudeCLIExecutor(provider).execute(task(), model, context):
                pass
            self.assertEqual(provider.env, expected)

    def test_coder_prompt_gets_inventory_and_plan_precedence(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "do"}]
        out = CoderPromptBuilder().with_workspace(messages, "/workspace", "src/: a.py")
        self.assertEqual(out[0], messages[0])
        self.assertEqual(out[1]["content"], "do\nWorking directory: /workspace; use paths under it."
                                        "\nProject files (for locating paths only; "
                                            "not a reading list):\nsrc/: a.py")
        self.assertIs(CoderPromptBuilder().with_workspace(messages, "", ""), messages)
        self.assertIn("follow the plan: it reflects the current repository",
                      _planned("ctx"))

    def test_execution_cache_ttl_is_configurable(self):
        from agent.routing.config import load_routing_config
        self.assertEqual(load_routing_config().execution_cache_ttl, "5m")

if __name__ == "__main__":
    unittest.main()

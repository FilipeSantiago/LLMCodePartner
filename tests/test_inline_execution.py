"""Inline execution: the plan's source in the prompt, tool-free replies, CodePartner edits."""

import asyncio
import json
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from agent.planning.models import EditRange, ReferenceRange, TaskContext
from agent.planning.snippets import (EDIT, NEW, REFERENCE, changed_ranges,
                                     earlier_task_references, materialize)
from agent.prompt import InlineCoderPromptBuilder, InlineCoderPromptContext
from agent.prompt.reviewer import ReviewerPromptBuilder, ReviewerPromptContext
from agent.routing import patch_apply
from agent.routing.config import InlineConfig, RoutingConfigError, _inline_config
from agent.routing.implementation import ImplementationService, handoff_blocker
from agent.routing.inline_execution import InlineCoder, InlineOutcome, InlineRequest
from agent.routing.models import ExecutionDecision, ModelRef
from agent.routing.persistence import RoutingPersistence
from agent.spec.models import Task
from agent.workspace import RegisteredProject
from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent
from tests import temp_db


def setUpModule():
    temp_db.start()


def tearDownModule():
    temp_db.stop()


def lines(count: int, prefix: str = "line") -> str:
    return "".join(f"{prefix} {number}\n" for number in range(1, count + 1))


def context(*edits, references=(), depends_on=()):
    return TaskContext("WP1-T2", "Change it", edits=list(edits),
                       references=list(references), depends_on=list(depends_on))


class SnippetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, path: str, text: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def test_large_file_edit_range_is_widened_and_numbered(self):
        self.write("big.py", lines(1000))
        snippets = materialize(context(EditRange("big.py", "fix", 500, 502)), self.root,
                               margin=3)
        [region] = snippets.regions
        self.assertEqual((region.kind, region.start, region.end), (EDIT, 497, 505))
        self.assertIn("\n497| line 497", snippets.text)
        self.assertIn("\n505| line 505", snippets.text)
        self.assertNotIn("line 506", snippets.text)

    def test_small_file_is_sent_whole_and_ranges_merge(self):
        self.write("small.py", lines(40))
        snippets = materialize(context(
            EditRange("small.py", "fix", 5, 6),
            references=[ReferenceRange("small.py", "context", 30, 32)]), self.root)
        [region] = snippets.regions
        self.assertEqual((region.kind, region.start, region.end), (EDIT, 1, 40))

    def test_overlapping_regions_of_a_large_file_merge_with_edit_winning(self):
        self.write("big.py", lines(1000))
        snippets = materialize(context(
            EditRange("big.py", "fix", 100, 110),
            references=[ReferenceRange("big.py", "caller", 105, 130),
                        ReferenceRange("big.py", "far", 900, 905)]), self.root, margin=0)
        kinds = [(region.kind, region.start, region.end) for region in snippets.regions]
        self.assertEqual(kinds, [(EDIT, 100, 130), (REFERENCE, 900, 905)])

    def test_new_and_missing_files_are_named_not_read(self):
        snippets = materialize(context(
            EditRange("pkg/new.py", "create it"),
            references=[ReferenceRange("gone.py", "was here")]), self.root)
        self.assertEqual([region.kind for region in snippets.regions], [NEW])
        self.assertIn("pkg/new.py (NEW FILE", snippets.text)
        self.assertEqual(snippets.missing, ("gone.py",))

    def test_budget_reports_whether_the_source_fits(self):
        self.write("small.py", lines(40))
        task = context(EditRange("small.py", "fix", 1, 2))
        self.assertTrue(materialize(task, self.root, budget=10_000).fits)
        self.assertFalse(materialize(task, self.root, budget=50).fits)

    def test_changed_ranges_grow_to_the_enclosing_definition(self):
        before = "import os\n\n\ndef first():\n    return 1\n\n\ndef second():\n    return 2\n"
        after = before.replace("return 2", "return 3")
        self.assertEqual(changed_ranges(before, after, "mod.py"), [(8, 9)])
        self.assertEqual(changed_ranges(before, after, "notes.txt"), [(9, 9)])

    def test_earlier_task_changes_become_references(self):
        self.write("lib.py", "def helper():\n    return 'new'\n")
        self.write("created.py", "VALUE = 1\n")
        references = earlier_task_references("WP1-T2", {
            "WP1-T1": {
                "lib.py": {"before_exists": True,
                           "before_content": "def helper():\n    return 'old'\n"},
                "created.py": {"before_exists": False, "before_content": None},
            }}, self.root)
        self.assertIn(ReferenceRange("lib.py", "changed by WP1-T1", 1, 2), references)
        self.assertIn(ReferenceRange("created.py", "created by WP1-T1"), references)


class PatchApplyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "app.py").write_text("a = 1\nb = 2\nb = 2\n")

    def tearDown(self):
        self.directory.cleanup()

    def reply(self, **payload) -> patch_apply.PatchResponse:
        return patch_apply.parse_response("Here it is:\n" + json.dumps(payload))

    def test_parses_the_last_reply_object(self):
        response = self.reply(edits=[{"file": "app.py", "old_text": "a = 1",
                                      "new_text": "a = 10"}], summary="done")
        self.assertEqual(response.edits[0].new_text, "a = 10")
        self.assertEqual(response.summary, "done")

    def test_malformed_replies_are_format_errors(self):
        with self.assertRaises(patch_apply.PatchFormatError):
            patch_apply.parse_response("no json here")
        with self.assertRaises(patch_apply.PatchFormatError):
            self.reply(edits=[{"file": "app.py", "old_text": 1}])

    def test_unique_edits_resolve_in_order(self):
        contents, errors = patch_apply.resolve(self.reply(edits=[
            {"file": "app.py", "old_text": "a = 1", "new_text": "a = 10"},
            {"file": "app.py", "old_text": "a = 10\nb", "new_text": "a = 11\nb"},
        ]), self.root, frozenset({"app.py"}))
        self.assertEqual(errors, [])
        self.assertEqual(contents["app.py"], "a = 11\nb = 2\nb = 2\n")

    def test_ambiguous_missing_and_out_of_scope_edits_are_rejected(self):
        contents, errors = patch_apply.resolve(self.reply(edits=[
            {"file": "app.py", "old_text": "b = 2", "new_text": "b = 3"},
            {"file": "app.py", "old_text": "zzz", "new_text": "y"},
            {"file": "other.py", "old_text": "x", "new_text": "y"},
        ]), self.root, frozenset({"app.py"}))
        self.assertEqual(len(errors), 3)
        self.assertIn("occurs 2 times", errors[0])
        self.assertIn("occurs 0 times", errors[1])
        self.assertIn("not one of this task's edit files", errors[2])
        self.assertEqual((self.root / "app.py").read_text(), "a = 1\nb = 2\nb = 2\n")

    def test_copied_line_numbers_are_stripped(self):
        contents, errors = patch_apply.resolve(self.reply(edits=[
            {"file": "app.py", "old_text": "1| a = 1\n2| b = 2\n3| b = 2\n",
             "new_text": "1| a = 1\n2| b = 2\n3| b = 4\n"}]), self.root, frozenset({"app.py"}))
        self.assertEqual(errors, [])
        self.assertEqual(contents["app.py"], "a = 1\nb = 2\nb = 4\n")

    def test_new_files_must_be_planned_and_absent(self):
        contents, errors = patch_apply.resolve(self.reply(new_files=[
            {"path": "pkg/new.py", "content": "X = 1\n"},
            {"path": "app.py", "content": "replaced"}]), self.root,
            frozenset({"pkg/new.py", "app.py"}))
        self.assertEqual(len(errors), 1)
        self.assertIn("already exists", errors[0])
        patch_apply.write(self.root, {"pkg/new.py": contents["pkg/new.py"]})
        self.assertEqual((self.root / "pkg/new.py").read_text(), "X = 1\n")


MODEL = ModelRef("claude-sonnet", "claude", "claude-cli", "sonnet", "disabled", None, None, 2)
FALLBACK = ModelRef("codex-sol-medium", "codex", "codex-cli", "gpt", "disabled", None, None, 2)


def decision(fallbacks=()):
    return ExecutionDecision("d1", "p", "c", "WP1-T2", "medium", ("claude-sonnet",),
                             "claude-sonnet", "claude", "claude-cli", fallbacks, None,
                             "test", datetime.now(UTC))


class FakeExecutor:
    """Replays scripted replies; records each call's conversation and tool surface."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def execute(self, task, model, context):
        self.calls.append((model.name, [dict(message) for message in context.messages],
                           context.specs, context.native, context.tool_free))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            yield ErrorEvent(str(reply))
            return
        yield TextEvent(reply if isinstance(reply, str) else json.dumps(reply))
        yield DoneEvent("endTurn", {"input_tokens": 10, "output_tokens": 5})


class FakePersistence:
    def __init__(self):
        self.attempts = []

    async def record_running_model(self, *args):
        pass

    async def record_attempt(self, decision, model, provider, executor, gateway, status,
                             *args):
        self.attempts.append((model, status))

    async def finalize(self, *args):
        pass


def task():
    return Task("WP1-T2", "Change it", "Make b three", "medium", "test", "", "write",
                ["b is 3"])


class InlineCoderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "app.py").write_text("a = 1\nb = 2\n")
        self.project = RegisteredProject("demo", str(self.root))

    def tearDown(self):
        self.directory.cleanup()

    def request(self, checks=(), fix_rounds=2, edits=None):
        task_context = context(*(edits or [EditRange("app.py", "set b", 2, 2)]))
        return InlineRequest(self.project, task_context, materialize(task_context, self.root),
                             checks, fix_rounds)

    def run_inline(self, replies, request, fallbacks=()):
        executor = FakeExecutor(replies)
        persistence = FakePersistence()
        models = {"claude-sonnet": MODEL, "codex-sol-medium": FALLBACK}
        config = SimpleNamespace(models=models, agentgateway_url="", execution_cache_ttl="5m")
        coder = InlineCoder(config, persistence,
                            SimpleNamespace(get=lambda name: executor),
                            SimpleNamespace(record=lambda *args: None))
        outcome = InlineOutcome()

        async def collect():
            return [event async for event in coder.run(
                task(), decision(fallbacks), request, [{"role": "user", "content": "go"}],
                {}, (), outcome)]

        return asyncio.run(collect()), executor, persistence, outcome

    @staticmethod
    def edit(old, new, **extra):
        return {"edits": [{"file": "app.py", "old_text": old, "new_text": new}],
                "summary": "set b", **extra}

    def test_one_reply_finishes_the_task_with_a_valid_handoff(self):
        events, executor, _, outcome = self.run_inline(
            [self.edit("b = 2", "b = 3")], self.request(checks=("true",)))
        self.assertEqual((self.root / "app.py").read_text(), "a = 1\nb = 3\n")
        self.assertEqual(len(executor.calls), 1)
        _, _, specs, native, tool_free = executor.calls[0]
        self.assertEqual((specs, native, tool_free), ([], None, True))
        self.assertIsNone(outcome.fallback)
        self.assertEqual(outcome.stats["llm_calls"], 1)
        report = next(event.text for event in events if isinstance(event, TextEvent))
        self.assertIsNone(handoff_blocker(report, "WP1-T2"))
        handoff = json.loads(report[report.index("{"):])
        self.assertEqual(handoff["changed_files"],
                         [{"path": "app.py", "change": "modified", "purpose": "inline edit"}])
        self.assertIsInstance(events[-1], DoneEvent)
        self.assertEqual(events[-1].usage, {"input_tokens": 10, "output_tokens": 5})

    def test_a_rejected_reply_gets_one_corrective_call(self):
        events, executor, _, outcome = self.run_inline(
            [self.edit("b = 9", "b = 3"), self.edit("b = 2", "b = 3")], self.request())
        self.assertEqual((self.root / "app.py").read_text(), "a = 1\nb = 3\n")
        self.assertEqual(outcome.stats["patch_apply_failures"], 1)
        self.assertEqual(outcome.stats["fix_rounds"], 1)
        feedback = executor.calls[1][1][-1]["content"]
        self.assertIn("occurs 0 times", feedback)
        self.assertIn("2| b = 2", feedback)

    def test_failing_checks_return_with_the_updated_source(self):
        check = "grep -q 'b = 4' app.py"
        events, executor, _, outcome = self.run_inline(
            [self.edit("b = 2", "b = 3"), self.edit("b = 3", "b = 4")],
            self.request(checks=(check,)))
        self.assertEqual((self.root / "app.py").read_text(), "a = 1\nb = 4\n")
        feedback = executor.calls[1][1][-1]["content"]
        self.assertIn(check, feedback)
        self.assertIn("2| b = 3", feedback)
        self.assertFalse(outcome.stats["checks_failed_final"])

    def test_checks_still_failing_after_the_last_round_are_reported(self):
        events, executor, _, outcome = self.run_inline(
            [self.edit("b = 2", "b = 3")], self.request(checks=("false",), fix_rounds=0))
        self.assertTrue(outcome.stats["checks_failed_final"])
        report = next(event.text for event in events if isinstance(event, TextEvent))
        self.assertIn("still fail", report)

    def test_requested_context_is_supplied_once_then_falls_back(self):
        (self.root / "lib.py").write_text("def helper(x):\n    return x\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        need = {"edits": [], "need_context": [{"symbol": "helper", "why": "signature"}]}
        events, executor, _, outcome = self.run_inline([need, need], self.request())
        self.assertIn("def helper(x):", executor.calls[1][1][-1]["content"])
        self.assertEqual(outcome.stats["need_context_requests"], 2)
        self.assertIsNotNone(outcome.fallback)
        self.assertFalse(any(isinstance(event, TextEvent) for event in events))

    def test_unusable_replies_fall_back_to_an_agentic_session(self):
        events, _, _, outcome = self.run_inline(["not json", "still not"],
                                                self.request(fix_rounds=1))
        self.assertIn("unusable", outcome.fallback)
        self.assertEqual((self.root / "app.py").read_text(), "a = 1\nb = 2\n")

    def test_a_failed_first_call_moves_to_the_fallback_model(self):
        events, executor, persistence, outcome = self.run_inline(
            [RuntimeError("rate limit"), self.edit("b = 2", "b = 3")], self.request(),
            fallbacks=("codex-sol-medium",))
        self.assertEqual(outcome.stats["llm_calls"], 2)
        self.assertEqual([call[0] for call in executor.calls],
                         ["claude-sonnet", "codex-sol-medium"])
        self.assertEqual(persistence.attempts, [("claude-sonnet", "failed"),
                                                ("codex-sol-medium", "completed")])


class JournaledInlineTests(unittest.TestCase):
    def test_inline_edits_and_checks_are_journaled_for_review(self):
        from agent.routing.models import ImplementationJob, ImplementationJobTask
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            (root / "app.py").write_text("a = 1\nb = 2\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            store = RoutingPersistence(Path(directory) / "routing.sqlite3")
            job = ImplementationJob.create(["WP1"], ["WP1-T2"], "")

            async def scenario():
                await store.create_implementation_job(
                    job, [ImplementationJobTask(job.job_id, 0, "c", "WP1-T2")])
                await store.record_decision(decision())
                task_context = context(EditRange("app.py", "set b", 2, 2))
                request = InlineRequest(RegisteredProject("demo", str(root)), task_context,
                                        materialize(task_context, root), ("true",))
                config = SimpleNamespace(models={"claude-sonnet": MODEL},
                                         agentgateway_url="", execution_cache_ttl="5m")
                executor = FakeExecutor([{"edits": [{"file": "app.py", "old_text": "b = 2",
                                                     "new_text": "b = 3"}]}])
                coder = InlineCoder(config, store, SimpleNamespace(get=lambda name: executor),
                                    SimpleNamespace(record=lambda *args: None))
                async for _ in coder.run(task(), decision(), request, [], {
                        "implementation_job_id": job.job_id,
                        "implementation_position": 0}, (), InlineOutcome()):
                    pass
                return (await store.rollback_preimages(job.job_id),
                        await store.verification_evidence(job.job_id, 0))

            preimages, evidence = asyncio.run(scenario())
        self.assertEqual(preimages["app.py"]["before_content"], "a = 1\nb = 2\n")
        self.assertEqual([item["arguments"]["command"] for item in evidence], ["true"])


class InlineRequestTests(unittest.IsolatedAsyncioTestCase):
    """What the implementation service puts in a dependent task's inline prompt."""

    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = RoutingPersistence(self.root / "routing.sqlite3")
        (self.root / "src").mkdir()
        # Task 1 created helpers.py and changed one function of app.py.
        (self.root / "src/helpers.py").write_text("def slugify(text):\n    return text\n")
        # Large enough that only the changed function is referenced, not the whole file.
        self.app_before = ("def keep():\n" + "    x = 0\n" * 400 + "    return x\n\n\n"
                           "def route():\n    return 1\n")
        (self.root / "src/app.py").write_text(self.app_before.replace("return 1", "return 2"))
        await self.store.record_rollback_file("job", 0, "src/helpers.py", False, None, None)
        await self.store.record_rollback_file("job", 0, "src/app.py", True, None,
                                              self.app_before)
        (self.root / "src/views.py").write_text("def view():\n    return None\n")
        self.service = ImplementationService.__new__(ImplementationService)
        self.service.persistence = self.store
        self.events = [{"task_id": "WP1-T1", "position": 0, "kind": "changed_files",
                        "payload": ["src/helpers.py", "src/app.py"]}]

    async def asyncTearDown(self):
        self.directory.cleanup()

    def plan(self, **overrides):
        values = {"edits": [EditRange("src/views.py", "use slugify", 1, 2)],
                  "depends_on": ["WP1-T1"], **overrides}
        return SimpleNamespace(tasks={"WP1-T2": TaskContext("WP1-T2", "o", **values)})

    async def request(self, mode="inline", plan=None, chars=120_000):
        self.service.config = SimpleNamespace(
            inline=InlineConfig(mode, context_chars=chars),
            planner=SimpleNamespace(edit_margin_lines=10))
        from unittest import mock
        with mock.patch("agent.routing.implementation.native_project",
                        return_value=RegisteredProject("demo", str(self.root))):
            return await self.service._inline_request(plan or self.plan(), task(), "job",
                                                      self.events)

    async def test_dependent_task_prompt_holds_the_earlier_tasks_code(self):
        request, reason = await self.request()
        self.assertEqual(reason, "")
        text = request.source.text
        self.assertIn("src/views.py (EDIT", text)
        self.assertIn("src/helpers.py (REFERENCE, whole file) — created by WP1-T1", text)
        self.assertIn("def slugify(text):", text)
        self.assertIn("— changed by WP1-T1", text)
        self.assertIn("405| def route():", text)
        self.assertIn("406|     return 2", text)
        self.assertNotIn("def keep", text)

    async def test_unrelated_earlier_tasks_are_left_out(self):
        request, _ = await self.request(plan=self.plan(depends_on=[]))
        self.assertNotIn("def slugify", request.source.text)

    async def test_agentic_mode_auto_without_lines_and_budget_run_agentic(self):
        self.assertEqual(await self.request(mode="agentic"), (None, ""))
        request, reason = await self.request(mode="auto", plan=self.plan(
            edits=[EditRange("src/views.py", "rewrite it")]))
        self.assertIsNone(request)
        self.assertIn("no lines", reason)
        request, reason = await self.request(chars=20)
        self.assertIsNone(request)
        self.assertIn("exceeds inline_context_chars", reason)


class PromptAndConfigTests(unittest.TestCase):
    def test_inline_prompt_carries_source_rules_and_checks(self):
        text = InlineCoderPromptBuilder().with_context(InlineCoderPromptContext(
            task(), "### app.py (EDIT, whole file)\n1| a = 1", ("app.py",),
            checks=("uv run pytest tests/test_app.py",))).build().text
        self.assertIn("1| a = 1", text)
        self.assertIn("Files you may change: app.py", text)
        self.assertIn("`uv run pytest tests/test_app.py`", text)
        self.assertIn('"need_context"', text)
        self.assertNotIn("Read", text.split("## Reply")[0].split("## Current source")[1])

    def test_tool_free_reviewer_prompt_has_no_read_instructions(self):
        review = ReviewerPromptBuilder().with_context(ReviewerPromptContext(
            task(), "change", native=True, tool_free=True,
            reference_source="### lib.py (REFERENCE, whole file)\n1| x")).build().text
        self.assertIn("You have no tools", review)
        self.assertIn("1| x", review)
        self.assertNotIn("Read pipeline", review)
        agentic = ReviewerPromptBuilder().with_context(ReviewerPromptContext(
            task(), "change", native=True)).build().text
        self.assertIn("Read pipeline", agentic)

    def test_coder_mode_config(self):
        self.assertEqual(_inline_config({}), InlineConfig())
        self.assertEqual(_inline_config({"coder_mode": "auto", "inline_fix_rounds": 0}).mode,
                         "auto")
        for bad in ({"coder_mode": "fast"}, {"inline_fix_rounds": -1},
                    {"inline_context_chars": 0}):
            with self.assertRaises(RoutingConfigError):
                _inline_config(bad)

    def test_expected_tests_become_check_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n")
            (root / "tests").mkdir()
            (root / "tests/test_a.py").write_text("")
            task_context = TaskContext("T", "o", expected_tests=[
                "tests/test_a.py: covers the new branch", "tests/test_b.py::test_new",
                "uv run ruff check app", "check the docs by hand"],
                edits=[EditRange("tests/test_b.py", "add test")])
            service = ImplementationService.__new__(ImplementationService)
            checks = asyncio.run(service._inline_checks(
                task_context, RegisteredProject("x", directory)))
        self.assertEqual(checks, ("uv run ruff check app",
                                  "uv run pytest tests/test_a.py tests/test_b.py::test_new"))


class TaskExecutionRoutingTests(unittest.TestCase):
    """`TaskExecutionService.execute`: inline first, the agentic session only on fallback."""

    def run_execute(self, fallback: str | None):
        from unittest import mock
        from agent.routing.config import load_routing_config
        from agent.routing.task_execution import TaskExecutionService
        from mcp_bridge.models import ProgressEvent

        async def route(*args, **kwargs):
            return decision()

        async def record_decision(_decision):
            pass

        service = TaskExecutionService(
            config=load_routing_config(),
            router=SimpleNamespace(route=route, health=None),
            persistence=SimpleNamespace(record_decision=record_decision),
            executors=SimpleNamespace())
        prompts = []

        async def inline_run(coder, task_, decision_, request, messages, *args):
            prompts.append(messages[-1]["content"])
            args[-1].fallback = fallback
            if fallback is None:
                yield TextEvent("inline report")

        async def agentic_run(*args, **kwargs):
            yield ProgressEvent("agentic session\n")

        root = tempfile.mkdtemp()
        Path(root, "app.py").write_text("b = 2\n")
        task_context = context(EditRange("app.py", "set b", 1, 1))
        request = InlineRequest(RegisteredProject("demo", root), task_context,
                                materialize(task_context, root))

        async def collect():
            return [event async for event in service.execute(
                task(), "c", [{"role": "user", "content": "x"}], [], inline=request)]

        with mock.patch("agent.routing.task_execution.InlineCoder.run", inline_run), \
                mock.patch.object(service, "_run", agentic_run):
            return asyncio.run(collect()), prompts

    def test_finished_inline_runs_no_agentic_session(self):
        events, prompts = self.run_execute(None)
        self.assertEqual([event.text for event in events], ["inline report"])
        self.assertIn("1| b = 2", prompts[0])

    def test_inline_fallback_continues_as_an_agentic_session(self):
        events, _ = self.run_execute("the plan was not enough")
        texts = [event.text for event in events]
        self.assertIn("the plan was not enough", texts[0])
        self.assertEqual(texts[-1], "agentic session\n")


class FindingOwnerTests(unittest.IsolatedAsyncioTestCase):
    """A work-package review's findings go to the task that wrote the code they name."""

    async def asyncSetUp(self):
        from agent.routing.models import ImplementationJobTask
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = RoutingPersistence(self.root / "routing.sqlite3")
        base = "def run_experiment(config):\n    return config\n"
        after_t2 = base + "\n\ndef evaluate_generated_dataset(dataset):\n    return {}\n"
        after_t3 = after_t2 + ("\n\ndef select_and_render_piano_roll(dataset, seed):\n"
                               "    return dataset[seed]\n")
        (self.root / "experiment_pipeline.py").write_text(after_t3)
        (self.root / "tests").mkdir()
        (self.root / "tests/test_generation.py").write_text("def test_remi():\n    pass\n")
        # The journal: each task's preimage of what it changed.
        for position, path, before in ((1, "experiment_pipeline.py", base),
                                       (2, "experiment_pipeline.py", after_t2),
                                       (3, "tests/test_generation.py", None)):
            await self.store.record_rollback_file("job", position, path, before is not None,
                                                  "hash", before)
        self.rows = [ImplementationJobTask("job", position, "c", f"WP2-T{position + 1}")
                     for position in range(4)]
        self.service = ImplementationService.__new__(ImplementationService)
        self.service.persistence = self.store

        async def no_plan(_job_id):
            return None

        self.service._approved_plan = no_plan

    async def asyncTearDown(self):
        self.directory.cleanup()

    @staticmethod
    def finding(problem, *paths):
        from agent.routing.review import ReviewFinding
        return ReviewFinding("implementation", problem, "fix it", ("fixed",), paths)

    async def owners(self, *findings):
        from unittest import mock
        with mock.patch("agent.routing.implementation.native_project",
                        return_value=RegisteredProject("demo", str(self.root))):
            owners = await self.service._finding_owners("job", self.rows, findings,
                                                        self.rows[-1])
        return [owners[finding].task_id for finding in findings]

    async def test_each_finding_goes_to_the_task_that_wrote_its_code(self):
        self.assertEqual(await self.owners(
            self.finding("select_and_render_piano_roll ignores non-evaluable pieces",
                         "experiment_pipeline.py"),
            self.finding("evaluate_generated_dataset drops distribution metrics",
                         "experiment_pipeline.py"),
            self.finding("only remi is covered", "tests/test_generation.py"),
        ), ["WP2-T3", "WP2-T2", "WP2-T4"])

    async def test_unknown_paths_fall_back_to_the_reviewed_task(self):
        self.assertEqual(await self.owners(self.finding("missing docs", "README.md")),
                         ["WP2-T4"])

    async def test_a_file_no_task_changed_goes_to_its_planned_owner(self):
        async def plan(_job_id):
            return SimpleNamespace(tasks={"WP2-T1": TaskContext(
                "WP2-T1", "o", edits=[EditRange("generation/factory.py", "create it")])})

        self.service._approved_plan = plan
        self.assertEqual(await self.owners(
            self.finding("factory missing", "generation/factory.py")), ["WP2-T1"])


class ReviewEvidenceTests(unittest.TestCase):
    def test_package_manifests_merge_first_before_and_last_after(self):
        from agent.routing.implementation import merged_manifest
        merged = merged_manifest([
            [{"path": "a.py", "change": "created", "before_exists": False,
              "after_exists": True, "operations": ["file_create"]}],
            [{"path": "a.py", "change": "modified", "before_exists": True,
              "after_exists": True, "operations": ["file_replace"]},
             {"path": "b.py", "change": "modified", "before_exists": True,
              "after_exists": True, "operations": ["file_replace"]}],
        ])
        self.assertEqual(merged[0]["change"], "created")
        self.assertEqual(merged[0]["operations"], ["file_create", "file_replace"])
        self.assertEqual([entry["path"] for entry in merged], ["a.py", "b.py"])

    def test_lockfiles_are_summarized_not_shown(self):
        from agent.routing.review import _lockfile_view
        view = _lockfile_view("a\nb\nc\n", {"before_exists": True, "before_content": "a\n"})
        self.assertIn("2 lines added or removed", view)
        self.assertNotIn("b\nc", view)

    def test_identifiers_ignore_plain_words(self):
        from agent.routing.implementation import _identifiers
        self.assertEqual(_identifiers("The select_and_render_piano_roll test of "
                                      "PianoRollVideoArtifact is missing a check"),
                         {"select_and_render_piano_roll", "PianoRollVideoArtifact"})


class ToolFreeSurfaceTests(unittest.TestCase):
    def test_executors_give_a_tool_free_call_no_coder_tools(self):
        from agent.routing.executors.claude import ClaudeCLIExecutor
        from agent.routing.executors.codex import CodexCLIExecutor
        from agent.routing.models import ExecutionContext
        from mcp_bridge.registry import ROLE_CODER, ROLE_OPTIMIZER

        class Provider:
            async def tools(self, messages, specs, **kwargs):
                self.specs, self.kwargs = specs, kwargs
                return
                yield

        for executor_class in (ClaudeCLIExecutor, CodexCLIExecutor):
            for tool_free, role in ((True, ROLE_OPTIMIZER), (False, ROLE_CODER)):
                provider = Provider()
                context = ExecutionContext(messages=[], specs=[], decision=decision(),
                                           tool_free=tool_free)

                async def drain():
                    async for _ in executor_class(provider).execute(task(), MODEL, context):
                        pass

                asyncio.run(drain())
                self.assertEqual(provider.kwargs["role"], role)
                self.assertEqual(provider.kwargs["run_metadata"].get("tool_free", False),
                                 tool_free)

    def test_task_comparison_reports_inline_attempts(self):
        from telemetry.aggregation import TelemetryAggregation
        from telemetry.storage import TelemetryStorage
        with tempfile.TemporaryDirectory() as directory:
            storage = TelemetryStorage(Path(directory) / "t.sqlite3")
            metadata = {"implementation_job_id": "J", "task_id": "T1", "provider": "claude",
                        "model": "s", "cost_usd": 0.5}
            storage.record_aci_attempt("i1", metadata, {
                "role": "coder", "execution_mode": "inline", "llm_calls": 2, "fix_rounds": 1,
                "fallback_to_agentic": False}, {"input_tokens": 10, "output_tokens": 5})
            storage.record_aci_attempt("n1", metadata, {
                "execution_mode": "native", "exploration_mode": "off", "tool_calls": 9},
                {"input_tokens": 30})
            report = TelemetryAggregation(storage).task_comparison(["J"])
        inline = report["J"]["T1"]["inline"]
        self.assertEqual((inline["llm_calls"], inline["fix_rounds"], inline["cost_usd"]),
                         (2, 1, 0.5))
        self.assertEqual(report["J"]["T1"]["off"]["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()

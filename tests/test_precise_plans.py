"""Precise plans: exact edit and reference ranges, scoped coders and reviewers, diff
snapshots, bulk KB outlines, the planning-start KB refresh and the `explain` view."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.planning import knowledge_base
from agent.planning.models import EditRange, TaskContext, WPPlanDraft
from agent.planning.projection import project, validate_draft
from agent.planning.render import (affected_files, coder_context, render_plan, task_files,
                                   task_read_only_files, widen)
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import _change_view, _range_view, _planned_ranges
from agent.routing import rollback
from agent.workflow.controller import WorkflowController
from providers import subagents
from tests.test_wp_planner import plan_payload

TASKS = ("WP1-T1", "WP1-T2")


def payload(**task_overrides):
    value = plan_payload(task_ids=TASKS)
    value["tasks"]["WP1-T1"].update({
        "edits": [{"file": "src/a.py", "start_line": 40, "end_line": 58,
                   "symbol": "a.load", "change": "Accept a name"},
                  {"file": "src/new.py", "change": "create the factory"}],
        "references": [{"file": "src/registry.py", "start_line": 3, "end_line": 9,
                        "why": "Registry API"},
                       {"file": "src/a.py", "why": "already edited, not a reference"}],
        **task_overrides})
    return value


def approved(value=None):
    draft = WPPlanDraft.from_dict(value or payload(), "WP1", 1)
    return project(draft, plan_id="plan_x_r1", session_id="x", change_id="CHG1",
                   wp_fingerprint="f", wp_complexity="medium", planner_model="claude-sonnet")


class EditRangeTests(unittest.TestCase):
    def test_edits_and_references_parse_and_round_trip(self):
        task = WPPlanDraft.from_dict(payload(), "WP1", 1).tasks["WP1-T1"]
        self.assertEqual(task.edits[0], EditRange("src/a.py", "Accept a name", 40, 58, "a.load"))
        self.assertIsNone(task.edits[1].start_line)
        self.assertEqual(task.edit_files(), ["src/a.py", "src/new.py"])
        self.assertEqual(task.reference_files(), ["src/registry.py"])
        restored = TaskContext.from_dict(json.loads(json.dumps(
            WPPlanDraft.from_dict(payload(), "WP1", 1).to_dict()))["tasks"]["WP1-T1"])
        self.assertEqual(restored.edits, task.edits)
        self.assertEqual(restored.references, task.references)

    def test_invalid_ranges_are_dropped(self):
        task = TaskContext.from_dict({"edits": [
            {"file": "../x.py", "start_line": 1, "end_line": 2, "change": "c"},
            {"file": "a.py", "start_line": 5, "end_line": 2, "change": "c"},
            {"file": "a.py", "start_line": 0, "end_line": 2, "change": "c"},
            {"file": "a.py", "start_line": 1, "change": "half a range"},
            {"file": "a.py", "start_line": 1, "end_line": 2},
            {"file": "a.py", "start_line": 1, "end_line": 2, "change": "kept"}]}, "T")
        self.assertEqual([edit.change for edit in task.edits], ["kept"])

    def test_plans_without_ranges_still_load(self):
        task = TaskContext.from_dict({"objective": "o", "relevant_files": ["src/x.py"]}, "T")
        self.assertEqual((task.edits, task.references), ([], []))

    def test_writing_tasks_need_edits_outside_do_not_modify(self):
        value = payload()
        value["tasks"]["WP1-T2"]["edits"] = []
        draft = WPPlanDraft.from_dict(value, "WP1", 1)
        self.assertEqual(validate_draft(draft, list(TASKS)), [])
        gaps = validate_draft(draft, list(TASKS), ["WP1-T2"])
        self.assertTrue(any("tasks.WP1-T2.edits is empty" in gap for gap in gaps))
        value = payload()
        value["shared"]["do_not_modify"] = ["src/a.py"]
        gaps = validate_draft(WPPlanDraft.from_dict(value, "WP1", 1), list(TASKS), ["WP1-T1"])
        self.assertTrue(any("do-not-modify paths: src/a.py" in gap for gap in gaps))


class ContextTests(unittest.TestCase):
    def test_margin_never_goes_before_line_one(self):
        self.assertEqual(widen(40, 58, 10), (30, 68))
        self.assertEqual(widen(3, 4, 10), (1, 14))

    def test_coder_scope_is_the_edits_and_the_references(self):
        plan = approved()
        self.assertEqual(task_files(plan, "WP1-T1"), ["src/a.py", "src/new.py"])
        self.assertEqual(task_read_only_files(plan, "WP1-T1"),
                         ["src/registry.py", "src/legacy/"])

    def test_coder_context_lists_exact_ranges_and_drops_leads(self):
        context = coder_context(approved(), "WP1-T1", margin=5)
        self.assertIn("- src/a.py:35-63 a.load (planned lines 40-58) — Accept a name", context)
        self.assertIn("- src/new.py (new file) — create the factory", context)
        self.assertIn("REFERENCES (read-only context):\n- src/registry.py:1-14 — Registry API",
                      context)
        self.assertIn("[F3, assumption] Config loading is global", context)
        for dropped in ("Discovery leads", "Symbols:", "Relevant files:"):
            self.assertNotIn(dropped, context)


class UserPlanTests(unittest.TestCase):
    def render(self, detail):
        value = payload()
        value.pop("plan_markdown")  # planners no longer write prose; it is rendered
        return render_plan(WPPlanDraft.from_dict(value, "WP1", 1), wp_complexity="medium",
                           planner_label="Claude Sonnet", gaps=[], token=None, detail=detail)

    def test_default_view_is_tasks_and_affected_files(self):
        text = self.render(False)
        self.assertIn("- **WP1-T1** — Do WP1-T1", text)
        self.assertIn("**Affected files**\n- `src/a.py` (modified)\n- `src/new.py` (new)", text)
        self.assertIn("3. Reply `explain`", text)
        for detail in ("Approach:", "**Decisions**", "What I inspected", "planned lines"):
            self.assertNotIn(detail, text)

    def test_explain_view_adds_approach_and_exact_lines(self):
        text = self.render(True)
        for detail in ("Approach: Extend the registry", "**Decisions**", "What I inspected",
                       "- src/a.py:40-58 a.load — Accept a name",
                       "References:\n- src/registry.py:3-9 — Registry API"):
            self.assertIn(detail, text)
        self.assertNotIn("Reply `explain`", text)

    def test_affected_files_of_older_plans_are_their_relevant_files(self):
        tasks = {"T": TaskContext("T", "o", relevant_files=["src/x.py"])}
        self.assertEqual(affected_files(tasks), ["`src/x.py`"])

    def test_explain_is_routed_while_planning(self):
        snapshot = SimpleNamespace(context=SimpleNamespace(
            planning_session_id="s", planning_wp_id="WP1"))
        route = WorkflowController.planning_route
        for text in ("explain", "Explain the plan", "details", "show details."):
            self.assertEqual(route(None, text, snapshot), "explain")
        self.assertEqual(route(None, "explain why WP1 needs a cache", snapshot), "discuss")


class DiffSnapshotTests(unittest.TestCase):
    before = "".join(f"line {number}\n" for number in range(1, 41))

    def test_modified_file_is_a_diff_with_untouched_planned_lines(self):
        after = self.before.replace("line 5\n", "line 5 changed\n")
        ranges = (EditRange("a.py", "c", 5, 5), EditRange("a.py", "c", 30, 31))
        kind, text = _change_view("a.py", after, {"before_exists": True,
                                                  "before_content": self.before}, ranges, 2)
        self.assertEqual(kind, "diff")
        self.assertIn("-line 5\n+line 5 changed\n", text)
        self.assertNotIn("line 20", text)
        self.assertIn("@@ planned lines 28-33, unchanged @@\n   28| line 28\n", text)

    def test_new_and_unknown_files_stay_whole(self):
        self.assertEqual(_change_view("n.py", "x\n", {"before_exists": False}, (), 2),
                         ("new", "x\n"))
        self.assertEqual(_change_view("n.py", "x\n", {"before_exists": True,
                                                      "before_content": None}, (), 2),
                         ("file", "x\n"))
        self.assertEqual(_change_view("n.py", "x\n", None, (), 2), ("file", "x\n"))

    def test_unchanged_planned_file_shows_only_its_ranges(self):
        spans = _planned_ranges((EditRange("b.py", "c", 10, 11),), "b.py", 1, 40)
        text = _range_view(self.before, spans)
        self.assertEqual(text.splitlines()[0], "@@ planned lines 9-12 @@")
        self.assertEqual(len(text.splitlines()), 5)


class PreimageTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_touch_preimage_per_path_and_position(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RoutingPersistence(Path(directory) / "routing.sqlite3")
            await store.record_rollback_file("j", 0, "a.py", True, None, "v0")
            await store.record_rollback_file("j", 0, "a.py", True, None, "ignored")
            await store.record_rollback_file("j", 1, "a.py", True, None, "v1")
            await store.record_rollback_file("j", 1, "new.py", False)
            self.assertEqual(await store.rollback_preimages("j"), {
                "a.py": {"before_exists": True, "before_content": "v0"},
                "new.py": {"before_exists": False, "before_content": None}})
            self.assertEqual((await store.rollback_preimages("j", [1]))["a.py"]
                             ["before_content"], "v1")

    async def test_native_baseline_keeps_planned_files_and_reads_clean_blobs(self):
        with tempfile.TemporaryDirectory() as root:
            for args in (("init", "-q"), ("config", "user.email", "t@t"),
                         ("config", "user.name", "t")):
                rollback._git(root, *args)
            Path(root, "planned.py").write_text("planned\n")
            Path(root, "clean.py").write_text("clean\n")
            rollback._git(root, "add", ".")
            rollback._git(root, "commit", "-q", "-m", "init")
            Path(root, "dirty.py").write_text("dirty\n")
            baseline = rollback._capture_baseline(root)
            baseline["preimages"] = rollback._read_preimages(root, ("planned.py", "gone.py"))
            self.assertEqual(rollback._baseline_preimage(baseline, "planned.py"), "planned\n")
            self.assertEqual(rollback._baseline_preimage(baseline, "clean.py"), "clean\n")
            # An untracked file's blob was never written to git: no preimage, whole file.
            self.assertIsNone(rollback._baseline_preimage(baseline, "dirty.py"))


class ScopeTelemetryTests(unittest.TestCase):
    def test_out_of_scope_reads_and_exceptions_are_counted(self):
        metrics = {}
        metadata = {"scope_files": ["src/a.py", "src/b.py"]}
        subagents.note_scope(metadata, metrics, ["src/a.py", "./src/b.py", "src/c.py",
                                                 "src/c.py"])
        subagents.note_scope(metadata, metrics, text="SCOPE EXCEPTION: where is X defined?")
        self.assertEqual(metrics, {"out_of_scope_reads": ["src/c.py"], "scope_exceptions": 1})

    def test_runs_without_a_scope_are_not_judged(self):
        metrics = {}
        subagents.note_scope({}, metrics, ["anything.py"])
        self.assertEqual(metrics, {})


class _Result:
    def __init__(self, text, error=False):
        self.content = [SimpleNamespace(type="text", text=text)]
        self.isError = error
        self.structuredContent = None


class KnowledgeBaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_outlines_of_several_files_come_from_one_query(self):
        calls = []

        class Session:
            async def call_tool(self, name, arguments):
                calls.append(name)
                return _Result(json.dumps({"rows": [
                    ["a.py", '["File"]', "a.py", "0", "0"],
                    ["a.py", '["Function"]', "load", "4", "9"],
                    ["b.py", '["Class"]', "B", "1", "30"]]}))

        sections = await knowledge_base.CodebaseMemoryClient._bulk_outlines(
            Session(), "proj", ["a.py", "b.py"], None)
        self.assertEqual(calls, ["query_graph"])
        self.assertEqual(sections, ["File outline a.py: load 4-9", "File outline b.py: B 1-30"])

    async def test_failed_bulk_query_falls_back_to_one_outline_per_file(self):
        calls = []

        class Session:
            async def call_tool(self, name, arguments):
                calls.append(name)
                return (_Result("boom", error=True) if name == "query_graph"
                        else _Result('{"outline": []}'))

        sections = await knowledge_base.CodebaseMemoryClient._bulk_outlines(
            Session(), "proj", ["a.py", "b.py"], None)
        self.assertEqual(calls, ["query_graph", "get_file_outline", "get_file_outline"])
        self.assertEqual(len(sections), 2)

    async def test_planning_start_always_reindexes_and_never_blocks(self):
        config = {"enabled": True, "provider": "codebase-memory", "project_id": "kb"}
        env = {"KB_ENABLED": "true"}
        with mock.patch.dict("os.environ", env), \
                mock.patch.object(knowledge_base, "_host_reindex",
                                  mock.AsyncMock()) as reindex:
            message = await knowledge_base.catch_up_index(
                None, "proj", config, object(), "/repo", force=True)
            reindex.assert_awaited_once()
            self.assertEqual(message, "KB index refreshed before planning.")
            reindex.side_effect = knowledge_base.KnowledgeBaseError("no terminal")
            message = await knowledge_base.catch_up_index(
                None, "proj", config, object(), "/repo", force=True)
            self.assertIn("KB refresh failed (no terminal)", message)


if __name__ == "__main__":
    unittest.main()

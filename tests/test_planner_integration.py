"""The approved plan reaching OpenSpec, the reviewer, and older databases."""

import contextlib
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.planning.knowledge_base import KB_READING_GUIDANCE, KB_SELECTIVE_GUIDANCE
from agent.planning.models import ApprovedWPPlan, WPPlanDraft
from agent.planning.projection import project, validate_draft
from agent.planning.prompts import SYSTEM
from agent.planning.render import coder_context, reviewer_context
from agent.routing.implementation import ImplementationService
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import ReviewDecision, ReviewService
from agent.routing.task_execution import TaskExecutionService
from agent.spec.models import OpenSpecChange, render_tasks
from agent.spec.openspec_adapter import CliOpenSpecAdapter, OpenSpecError
from mcp_bridge.models import DoneEvent, TextEvent
from mcp_bridge.registry import ToolSpec
from tests.test_routing import _Artifacts, task
from tests.test_wp_planner import plan_payload
from tests import ide_mode


def setUpModule():
    ide_mode.pin()


def tearDownModule():
    ide_mode.unpin()


WP1_TASKS = ("WP1-T1", "WP1-T2")


def _raw_package(**extra):
    return {
        "title": "Loader", "kind": "technical-outcome", "objective": "One factory",
        "acceptance_criteria": ["Works"], "requirement_refs": ["cap#Req"],
        "tasks": [{"title": "t", "description": "d", "acceptance_criteria": ["done"],
                   "complexity": "simple", "reasoning": "r", "context": "",
                   "preferred_capability": "write"}],
        **extra,
    }


class WorkPackageComplexitySpecTests(unittest.TestCase):
    def test_complexity_is_parsed_validated_and_optional(self):
        parse = CliOpenSpecAdapter._parse_work_packages
        self.assertEqual(parse([_raw_package(complexity="high")])[0].complexity, "high")
        self.assertIsNone(parse([_raw_package()])[0].complexity)
        with self.assertRaisesRegex(OpenSpecError, "invalid complexity"):
            parse([_raw_package(complexity="simple")])

    def test_complexity_round_trips_and_renders(self):
        package = CliOpenSpecAdapter._parse_work_packages([_raw_package(complexity="complex")])[0]
        package = replace(package, id="WP1")
        change = OpenSpecChange.from_dict({
            "id": "CHG1", "number": 1, "slug": "chg", "title": "Change", "summary": "s",
            "proposal": {"why": "w", "what_changes": ["c"], "impact": ["i"]},
            "design": {"context": "", "goals": ["g"], "non_goals": [], "decisions": [],
                       "risks": []},
            "capabilities": [], "work_packages": [],
        })
        change = replace(change, work_packages=[package])
        restored = OpenSpecChange.from_dict(json.loads(json.dumps(change.to_dict())))
        self.assertEqual(restored.work_packages[0].complexity, "complex")
        self.assertIn("**Complexity:** complex", render_tasks(restored))
        legacy = change.to_dict()
        legacy["work_packages"][0].pop("complexity")
        self.assertIsNone(OpenSpecChange.from_dict(legacy).work_packages[0].complexity)


class DiscoveryHandoffTests(unittest.TestCase):
    def test_plan_projects_compact_discovery_to_the_right_agents(self):
        payload = plan_payload(task_ids=WP1_TASKS)
        payload["discovery"] = [
            {"file": "src/registry.py", "reason": "Defines the loader registry",
             "relevant_symbols": [{"name": "Registry.register", "start_line": 12,
                                   "end_line": 31}], "task_ids": ["WP1-T1"]},
            {"file": "src/trainer.py", "reason": "Calls the loader factory",
             "relevant_symbols": [{"name": "train"}], "task_ids": []},
        ]
        draft = WPPlanDraft.from_dict(payload, "WP1", 1)
        self.assertEqual(validate_draft(draft, list(WP1_TASKS)), [])
        plan = project(draft, plan_id="plan_x_r1", session_id="x", change_id="CHG1",
                       wp_fingerprint="f", wp_complexity="medium",
                       planner_model="claude-sonnet")
        restored = ApprovedWPPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        first = coder_context(restored, "WP1-T1")
        second = coder_context(restored, "WP1-T2")
        review = reviewer_context(restored, [], ["WP1-T1"])
        self.assertIn("Registry.register lines 12–31", first)
        self.assertNotIn("Registry.register lines 12–31", second)
        self.assertIn("src/trainer.py", second)
        self.assertIn("Registry.register lines 12–31", review)
        self.assertIn("navigation only", review)

    def test_coder_context_carries_earlier_task_results(self):
        draft = WPPlanDraft.from_dict(plan_payload(task_ids=WP1_TASKS), "WP1", 1)
        plan = project(draft, plan_id="plan_x_r1", session_id="x", change_id="CHG1",
                       wp_fingerprint="f", wp_complexity="medium",
                       planner_model="claude-sonnet")
        events = [
            {"position": 0, "task_id": "WP1-T1", "kind": "decisions", "summary": "first try"},
            {"position": 0, "task_id": "WP1-T1", "kind": "tests", "summary": "1 failed"},
            {"position": 0, "task_id": "WP1-T1", "kind": "decisions",
             "summary": "Inventory: Registry.register is reusable"},
            {"position": 0, "task_id": "WP1-T1", "kind": "changed_files",
             "summary": "src/registry.py"},
        ]
        context = coder_context(plan, "WP1-T2", events)
        self.assertIn("EARLIER TASK RESULTS", context)
        self.assertIn("Result: Inventory: Registry.register is reusable", context)
        self.assertIn("Changed files: src/registry.py", context)
        self.assertNotIn("first try", context)
        self.assertNotIn("1 failed", context)
        self.assertNotIn("EARLIER TASK RESULTS", coder_context(plan, "WP1-T1", events))
        self.assertNotIn("EARLIER TASK RESULTS", coder_context(plan, "WP1-T2"))

    def test_coder_context_is_scoped_to_the_task(self):
        payload = plan_payload(task_ids=WP1_TASKS)
        payload["shared"].update({
            "requirements": ["WP-wide requirement only the reviewer needs"],
            "architecture": ["Loaders register through the registry"],
            "constraints": ["Keep the trainer API stable"]})
        payload["tasks"]["WP1-T2"].update({"depends_on": ["WP1-T1"],
                                           "expected_tests": ["tests/test_t2.py"]})
        draft = WPPlanDraft.from_dict(payload, "WP1", 1)
        plan = project(draft, plan_id="plan_x_r1", session_id="x", change_id="CHG1",
                       wp_fingerprint="f", wp_complexity="medium",
                       planner_model="claude-sonnet")
        events = [{"position": 0, "task_id": "WP1-T1", "kind": "changed_files",
                   "summary": "src/registry.py"}]
        context = coder_context(plan, "WP1-T2", events, "Do WP1-T2")
        # Other tasks' findings, files and WP-wide requirements stay out.
        self.assertNotIn("A registry already exists", context)
        self.assertNotIn("src/wp1-t1.py", context)
        self.assertNotIn("WP-wide requirement", context)
        self.assertNotIn("WP SHARED CONTEXT", context)
        # Findings for every task, architecture, constraints and dependencies stay in.
        self.assertIn("The trainer API is public", context)
        self.assertIn("Config loading is global", context)
        self.assertIn("Loaders register through the registry", context)
        self.assertIn("Keep the trainer API stable", context)
        self.assertIn("src/legacy/", context)
        self.assertIn("src/wp1-t2.py", context)
        self.assertIn("WP1-T1 (dependency):", context)
        self.assertIn("VALIDATION", context)
        self.assertIn("tests/test_t2.py", context)
        # The objective equal to the OpenSpec description is not repeated.
        self.assertNotIn("Objective: Do WP1-T2", context)
        self.assertIn("Objective: Do WP1-T2", coder_context(plan, "WP1-T2", events))

    def test_execution_prompts_share_rules_and_handoff(self):
        from agent.routing.task_execution import EXPLORATION_RULES
        for native in (False, True):
            prompt = TaskExecutionService._execution_messages(
                task(), [], "", native=native)[0]["content"]
            self.assertIn(EXPLORATION_RULES, prompt)
            self.assertIn('"execution_dispute":null', prompt)
            self.assertIn("Never read the whole repository", prompt)
            self.assertIn("never limit necessary work", prompt)
        contract = SimpleNamespace(revision=2, goal=task().description,
                                   primary_files=("src/a.py",))
        prompt = TaskExecutionService._execution_messages(
            task(), [], "", contract)[0]["content"]
        self.assertEqual(prompt.count(task().description),
                         TaskExecutionService._execution_messages(
                             task(), [], "")[0]["content"].count(task().description))
        self.assertIn("TaskContract revision 2. Expected files: src/a.py", prompt)

    def test_missing_or_stale_discovery_remains_optional(self):
        legacy = WPPlanDraft.from_dict(plan_payload(task_ids=WP1_TASKS), "WP1", 1)
        self.assertEqual(legacy.discovery, [])
        payload = plan_payload(task_ids=WP1_TASKS)
        payload["discovery"] = [
            {"file": "src/old.py", "reason": "Indexed lead", "relevant_symbols": [
                {"name": "old", "start_line": 9, "end_line": 11}], "task_ids": []},
            {"file": "../unsafe.py", "reason": "Invalid path"},
        ]
        draft = WPPlanDraft.from_dict(payload, "WP1", 1)
        self.assertEqual(len(draft.discovery), 1)
        self.assertEqual(validate_draft(draft, list(WP1_TASKS)), [])

    def test_shared_guidance_reaches_planner_and_coder(self):
        self.assertIn(KB_READING_GUIDANCE, SYSTEM)
        self.assertIn("Before edits or review verdicts, inspect exact current IDE source", KB_READING_GUIDANCE)
        # The default selective policy keeps KB-first guidance for planners only.
        prompt = TaskExecutionService._execution_messages(task(), [], "")[0]["content"]
        self.assertIn(KB_SELECTIVE_GUIDANCE, prompt)
        self.assertNotIn(KB_READING_GUIDANCE, prompt)
        self.assertNotIn("KB search and outline seeds", prompt)
        prompt = TaskExecutionService._execution_messages(
            task(), [], "", kb_policy="kb_first")[0]["content"]
        self.assertIn(KB_READING_GUIDANCE, prompt)
        self.assertIn("KB search and outline seeds", prompt)


class _CapturingReviewer:
    def __init__(self):
        self.tasks = []

    async def review(self, change, review_task, report, coder_model, specs, handlers):
        self.tasks.append(review_task)
        return ReviewDecision("done", "Reviewed with plan context", "codex-luna-low")


class _HandoffExecution:
    async def execute(self, queued_task, change_id, messages, specs, **kwargs):
        yield TextEvent(json.dumps({
            "schema_version": 1, "task_id": queued_task.id,
            "summary": f"Implemented {queued_task.id}",
            "changed_files": [{"path": f"src/{queued_task.id.lower()}.py"},
                              {"path": "src/unplanned.py"}],
            "criteria": [], "remaining_gaps": ["docs not updated"],
            "execution_dispute": None}))
        yield DoneEvent("endTurn")


class PlannedJobReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_reviewer_starts_from_the_plan_and_execution_record(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-haiku"

            persistence.actual_model_for = actual_model
            draft = WPPlanDraft.from_dict(plan_payload(task_ids=WP1_TASKS), "WP1", 1)
            plan = project(draft, plan_id="plan_x_r1", session_id="x", change_id="CHG1",
                           wp_fingerprint="f", wp_complexity="medium",
                           planner_model="claude-sonnet")
            await persistence.save_approved_plan(plan.to_dict())
            reviewer = _CapturingReviewer()
            service = ImplementationService(_Artifacts(), _HandoffExecution(), persistence,
                                            reviewer)
            job = await service.create_job(["WP1"])
            with contextlib.closing(sqlite3.connect(persistence.path)) as db:
                db.execute("UPDATE implementation_jobs SET plan_id=? WHERE job_id=?",
                           (plan.plan_id, job.job_id))
                db.commit()
            [event async for event in service.start(job, [], [])]

            reviewed = reviewer.tasks[-1]
            context = reviewed._planner_review_context
            self.assertIn("Approved plan:", context)
            self.assertIn("Loaders are created by name", context)
            self.assertIn("WP1-T1 changed_files: src/wp1-t1.py, src/unplanned.py", context)
            self.assertIn("WP1-T2 deviations: changed files not named in the plan: "
                          "src/unplanned.py", context)
            self.assertIn("WP1-T1 unresolved: docs not updated", context)
            self.assertIn("src/wp1-t2.py", reviewed._planner_relevant_files)
            # Planner context is never folded into task text, where paths become required.
            self.assertNotIn("Approved plan:", reviewed.context)


class ReviewerPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_planned_files_are_optional_hints_in_the_prompt(self):
        class Provider:
            async def tools(self, messages, specs, **kwargs):
                self.prompt = messages[0]["content"]
                yield TextEvent(json.dumps({
                    "verdict": "done", "rationale": "Inspected the factory",
                    "evidence": ["The factory exists"], "inspected_paths": ["src/factory.py"],
                    "findings": []}))
                yield DoneEvent("endTurn")

        async def read_file(arguments):
            if arguments["file_path"] == "src/factory.py":
                return "class Factory: ..."
            return "Error: file not found"

        queued = task()
        object.__setattr__(queued, "_planner_review_context", "Objective: one factory")
        object.__setattr__(queued, "_planner_relevant_files", ("src/not_created_yet.py",))
        provider = Provider()
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            decision = await ReviewService().review(
                SimpleNamespace(title="Factory", work_packages=[]), queued,
                "Changed `src/factory.py`.", "claude-haiku",
                [ToolSpec("read_file", "Read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}}})], {"read_file": read_file})
        self.assertEqual(decision.verdict, "done")
        self.assertIn("Approved WP plan context", provider.prompt)
        self.assertIn("Objective: one factory", provider.prompt)
        # The configured kb_first policy gives IDE-path reviewers the KB-first guidance.
        self.assertIn(KB_READING_GUIDANCE, provider.prompt)
        self.assertNotIn(KB_SELECTIVE_GUIDANCE, provider.prompt)
        self.assertIn("not correctness evidence", provider.prompt)

    async def test_planner_only_file_is_not_eagerly_read_when_change_is_known(self):
        class Provider:
            async def tools(self, messages, specs, **kwargs):
                self.prompt = messages[0]["content"]
                yield TextEvent(json.dumps({
                    "verdict": "done", "rationale": "Factory inspected",
                    "evidence": ["Factory exists"], "inspected_paths": ["src/factory.py"],
                    "findings": []}))
                yield DoneEvent("endTurn")

        reads = []

        async def read_file(arguments):
            reads.append(arguments["file_path"])
            return "L1: class Factory: pass"

        queued = task()
        object.__setattr__(queued, "_planner_relevant_files", ("src/old_dependency.py",))
        provider = Provider()
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            decision = await ReviewService().review(
                SimpleNamespace(title="Factory", work_packages=[]), queued,
                "Changed `src/factory.py`.", "claude-haiku",
                [ToolSpec("read_file", "Read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}}})], {"read_file": read_file})
        self.assertEqual(decision.verdict, "done")
        self.assertEqual(reads, ["src/factory.py"])
        self.assertIn("src/old_dependency.py", provider.prompt)


class PlanningMigrationTests(unittest.TestCase):
    def test_old_database_gains_plan_columns_and_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.sqlite3"
            with contextlib.closing(sqlite3.connect(path)) as db:
                db.execute("""CREATE TABLE implementation_jobs (
                  job_id TEXT PRIMARY KEY, selectors_json TEXT NOT NULL,
                  status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                  error TEXT)""")
                db.execute("INSERT INTO implementation_jobs VALUES ('j', '[]', 'completed', 0, 0, NULL)")
                db.commit()
            RoutingPersistence(path)
            with contextlib.closing(sqlite3.connect(path)) as db:
                columns = {row[1] for row in db.execute("PRAGMA table_info(implementation_jobs)")}
                tables = {row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                legacy = db.execute("SELECT plan_id FROM implementation_jobs").fetchone()
            self.assertIn("plan_id", columns)
            self.assertIsNone(legacy[0])
            self.assertTrue({"wp_planning_sessions", "wp_planning_turns", "wp_approved_plans",
                             "wp_reviewer_context_events"} <= tables)


if __name__ == "__main__":
    unittest.main()

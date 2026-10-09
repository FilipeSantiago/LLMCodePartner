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

from agent.planning.models import WPPlanDraft
from agent.planning.projection import project
from agent.routing.implementation import ImplementationService
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import ReviewDecision, ReviewService
from agent.spec.models import OpenSpecChange, render_tasks
from agent.spec.openspec_adapter import CliOpenSpecAdapter, OpenSpecError
from mcp_bridge.models import DoneEvent, TextEvent
from mcp_bridge.registry import ToolSpec
from tests.test_routing import _Artifacts, task
from tests.test_wp_planner import plan_payload

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

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

from agent.planning.models import ApprovedWPPlan, WPPlanDraft
from agent.planning.projection import project, validate_draft
from agent.prompt.coder import CoderPromptBuilder
from agent.planning.render import coder_context, reviewer_context
from agent.routing.implementation import ImplementationService, _compact_context_handoff
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import ReviewDecision
from agent.routing.task_execution import TaskExecutionService
from agent.spec.models import OpenSpecChange, render_tasks
from agent.spec.openspec_adapter import CliOpenSpecAdapter, OpenSpecError
from mcp_bridge.models import DoneEvent, TextEvent
from tests.test_routing import _Artifacts, task
from tests.test_wp_planner import plan_payload
from tests import native_mode

KB_FIRST_RULE = CoderPromptBuilder().read_pipeline()
KB_SELECTIVE_GUIDANCE = KB_FIRST_RULE


def setUpModule():
    native_mode.pin()


def tearDownModule():
    native_mode.unpin()


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
    def test_structured_handoff_is_bounded_and_rejects_unsafe_paths(self):
        compact = _compact_context_handoff({
            "files": ["../outside.py", "/absolute.py", "src/api.py"],
            "symbols": [f"src/api.py::symbol_{i}" for i in range(20)],
            "decisions": ["Keep API stable"],
        }, ["src/changed.py"])
        self.assertEqual(compact["files"], ["src/changed.py", "src/api.py"])
        self.assertEqual(len(compact["symbols"]), 10)

    def test_unrelated_prior_task_is_not_prompted(self):
        draft = WPPlanDraft.from_dict(plan_payload(task_ids=WP1_TASKS), "WP1", 1)
        plan = project(draft, plan_id="plan_x_r1", session_id="x", change_id="CHG1",
                       wp_fingerprint="f", wp_complexity="medium",
                       planner_model="claude-sonnet")
        context = coder_context(plan, "WP1-T2", [{
            "task_id": "WP1-T1", "kind": "task_handoff", "summary": "Earlier result",
            "payload": {"files": ["src/unrelated.py"], "symbols": ["Unrelated"]},
        }])
        self.assertNotIn("EARLIER TASK RESULTS", context)


    def test_plan_projects_compact_discovery_to_the_right_agents(self):
        # Plans from before edit ranges hand discovery leads to coders and reviewers.
        payload = plan_payload(task_ids=WP1_TASKS)
        for context in payload["tasks"].values():
            context.pop("edits")
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
        payload = plan_payload(task_ids=WP1_TASKS)
        payload["tasks"]["WP1-T2"]["depends_on"] = ["WP1-T1"]
        draft = WPPlanDraft.from_dict(payload, "WP1", 1)
        plan = project(draft, plan_id="plan_x_r1", session_id="x", change_id="CHG1",
                       wp_fingerprint="f", wp_complexity="medium",
                       planner_model="claude-sonnet")
        events = [
            {"position": 0, "task_id": "WP1-T1", "kind": "decisions", "summary": "first try"},
            {"position": 0, "task_id": "WP1-T1", "kind": "tests", "summary": "1 passed"},
            {"position": 0, "task_id": "WP1-T1", "kind": "decisions",
             "summary": "Inventory: Registry.register is reusable"},
            {"position": 0, "task_id": "WP1-T1", "kind": "changed_files",
             "summary": "src/registry.py"},
            {"position": 0, "task_id": "WP1-T1", "kind": "task_handoff",
             "payload": {"files": ["src/registry.py"],
                         "symbols": ["src/registry.py::Registry.register"],
                         "verification": ["registry test passed"]}},
        ]
        context = coder_context(plan, "WP1-T2", events)
        self.assertIn("EARLIER TASK RESULTS", context)
        self.assertIn("Result: Inventory: Registry.register is reusable", context)
        self.assertIn("Interfaces: src/registry.py::Registry.register", context)
        self.assertIn("Verification: registry test passed", context)
        self.assertIn("Changed files: src/registry.py", context)
        self.assertNotIn("first try", context)
        self.assertIn("Captured verification: 1 passed", context)
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

class PersistedContextHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_unplanned_job_uses_saved_report_from_same_package(self):
        service = ImplementationService.__new__(ImplementationService)
        service._package_for_task = mock.AsyncMock(
            return_value=(None, SimpleNamespace(id="WP1")))
        service._package_rows = mock.AsyncMock(return_value=[
            SimpleNamespace(job_id="job", position=0, task_id="WP1-T1",
                            status="completed")])
        report = json.dumps({"summary": "Registry updated", "changed_files": [
            {"path": "src/registry.py"}], "context_handoff": {
                "symbols": ["src/registry.py::Registry.register"]}})
        service.persistence = SimpleNamespace(coder_report_for=mock.AsyncMock(
            return_value=report))
        later = replace(task(), id="WP1-T2", depends_on=["WP1-T1"])
        context = await service._unplanned_prior_context(
            SimpleNamespace(job_id="job", position=1, task_id=later.id), later,
            SimpleNamespace(primary_files=()))
        self.assertIn("src/registry.py::Registry.register", context)
        prompt = TaskExecutionService._execution_messages(
            later, [], handoff_context=context)[0]["content"]
        self.assertIn("src/registry.py::Registry.register", prompt)
        self.assertNotIn("A planner already investigated", prompt)

    async def test_completed_task_handoff_reaches_dependent_task(self):
        payload = plan_payload(task_ids=WP1_TASKS)
        payload["tasks"]["WP1-T2"]["depends_on"] = ["WP1-T1"]
        plan = project(WPPlanDraft.from_dict(payload, "WP1", 1),
                       plan_id="plan_x_r1", session_id="x", change_id="CHG1",
                       wp_fingerprint="f", wp_complexity="medium",
                       planner_model="claude-sonnet")
        events = []

        class Persistence:
            async def verification_evidence(self, job_id, position):
                return []

            async def record_reviewer_context_event(self, plan_id, job_id, position,
                                                    task_id, kind, summary, value):
                events.append({"position": position, "task_id": task_id, "kind": kind,
                               "summary": summary, "payload": value})

        service = ImplementationService.__new__(ImplementationService)
        service.persistence = Persistence()
        report = json.dumps({"schema_version": 1, "task_id": "WP1-T1",
                             "summary": "Registry updated", "changed_files": [
                                 {"path": "src/registry.py", "change": "modified"}],
                             "context_handoff": {
                                 "symbols": ["src/registry.py::Registry.register"],
                                 "decisions": ["Keep the trainer API stable"]}})
        await service._record_plan_events(
            plan, SimpleNamespace(job_id="job", position=0), "WP1-T1", report,
            None, 1, False)
        context = coder_context(plan, "WP1-T2", events)
        self.assertIn("src/registry.py::Registry.register", context)
        self.assertIn("Keep the trainer API stable", context)


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


if __name__ == "__main__":
    unittest.main()

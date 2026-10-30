"""Dependency-free unit tests for OpenSpec task routing and audit persistence."""

import asyncio
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.routing.config import RoutingConfig, RoutingConfigError, load_routing_config
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.implementation import ImplementationService, handoff_blocker
from agent.routing.models import ModelRef
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import (ReviewDecision, ReviewError, ReviewFinding,
                                  ReviewFormatError, ReviewService, ReviewUnavailableError,
                                  _artifact_candidates,
                                  _candidate_paths)
from agent.routing.rollback import RollbackJournal, journal_handlers
from agent.routing.semantic_router import SemanticRouterError
from agent.routing.service import ModelRoutingService
from agent.routing.task_execution import TaskExecutionService
from agent.spec.command import (
    SpecCommandError,
    extract_implementation_selectors,
    extract_implementation_request,
    extract_run_task_id,
    is_implementation_command,
    is_run_command,
)
from agent.spec.models import Task
from mcp_bridge.models import DoneEvent, ErrorEvent, MutationEvent, Run, TextEvent
from mcp_bridge.registry import ToolSpec
from tests import native_mode


def setUpModule():
    native_mode.pin()


def tearDownModule():
    native_mode.unpin()



def task(complexity="medium"):
    return Task(
        id="WP1-T1", title="Add routing", description="Add model routing.",
        complexity=complexity, reasoning="Crosses execution boundaries.",
        context="The project already has OpenSpec tasks.", preferred_capability="write",
    )


class Router:
    def __init__(self, model=None, error=None):
        self.model, self.error = model, error
        self.calls = []

    async def select(self, text, complexity):
        self.calls.append((text, complexity))
        if self.error:
            raise self.error
        from agent.routing.models import RoutingRecommendation
        return RoutingRecommendation(self.model, "chosen for test")


class RoutingConfigTests(unittest.TestCase):
    def test_config_has_candidate_pool_for_every_openspec_complexity(self):
        config = load_routing_config()
        for complexity in ("simple", "medium", "complex"):
            candidates = config.candidates_for(complexity)
            self.assertTrue(candidates)
            self.assertTrue(all(candidate in config.models for candidate in candidates))

    def test_unknown_complexity_is_rejected(self):
        with self.assertRaises(RoutingConfigError):
            load_routing_config().candidates_for("huge")


class RoutingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_recommendation_must_belong_to_tier(self):
        config = load_routing_config()
        service = ModelRoutingService(config, Router(config.candidates_for("medium")[0]))
        decision = await service.route(task(), "CHG1", "/work/project")
        self.assertEqual(decision.recommended_model, config.candidates_for("medium")[0])
        self.assertEqual(decision.recommended_provider, "claude")
        self.assertEqual(decision.executor, "claude-cli")

    async def test_foreign_router_selection_uses_configured_fallback(self):
        config = load_routing_config()
        service = ModelRoutingService(config, Router(config.candidates_for("complex")[0]))
        decision = await service.route(task(), "CHG1", "/work/project")
        self.assertEqual(decision.recommended_model, config.candidates_for("medium")[0])
        self.assertEqual(decision.routing_source, "deterministic-fallback")

    async def test_router_connection_failure_is_deterministic(self):
        config = load_routing_config()
        service = ModelRoutingService(config, Router(error=SemanticRouterError("offline")))
        decision = await service.route(task("simple"), "CHG1", "/work/project")
        self.assertEqual(decision.recommended_model, config.candidates_for("simple")[0])
        self.assertIn("offline", decision.routing_reason)


class ExecutorDispatchTests(unittest.TestCase):
    def test_registry_uses_provider_specific_executors(self):
        registry = ExecutorRegistry()
        self.assertEqual(registry.get("claude-cli").name, "claude-cli")
        self.assertEqual(registry.get("codex-cli").name, "codex-cli")


class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_unfinished_job_lookup_is_scoped_to_wp_and_project(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RoutingPersistence(Path(directory) / "routing.sqlite3")
            with store._database() as db:
                for job_id, project_id, status, task_id in (
                    ("other-wp", "project-a", "running", "WP10-T1"),
                    ("other-project", "project-b", "running", "WP1-T1"),
                    ("finished", "project-a", "completed", "WP1-T1"),
                    ("active", "project-a", "awaiting_review_resolution", "WP1-T2"),
                ):
                    db.execute("""INSERT INTO implementation_jobs
                      (job_id, selectors_json, instruction, conversation_id, project_id,
                       status, created_at, updated_at)
                      VALUES (?, '[]', '', 'conversation-a', ?, ?, 1, 1)""",
                      (job_id, project_id, status))
                    db.execute("""INSERT INTO implementation_job_tasks
                      (job_id, position, change_id, task_id, status)
                      VALUES (?, 0, 'CHG1', ?, 'queued')""", (job_id, task_id))
            self.assertEqual(
                await store.unfinished_job_for_wp("project-a", "conversation-a", "WP1"),
                "active")
            self.assertIsNone(
                await store.unfinished_job_for_wp("project-a", "conversation-a", "WP2"))
            with store._database() as db:
                db.execute("UPDATE implementation_jobs SET status='stopped' WHERE job_id='active'")
            self.assertIsNone(
                await store.unfinished_job_for_wp("project-a", "conversation-a", "WP1"))

    async def test_verification_evidence_is_scoped_to_current_revision_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RoutingPersistence(Path(directory) / "routing.sqlite3")
            with store._database() as db:
                db.execute("""INSERT INTO implementation_jobs
                  (job_id, selectors_json, instruction, status, created_at, updated_at)
                  VALUES ('job', '[]', '', 'running', 1, 1)""")
                db.execute("""INSERT INTO implementation_job_tasks
                  (job_id, position, change_id, task_id, status)
                  VALUES ('job', 0, 'CHG', 'WP1-T1', 'running')""")
            first = await store.reserve_evidence(
                "job", 0, "verification", "execute_tool", {"command": "pytest"}, True)
            await store.complete_evidence(first, "/tmp/first", "hash-0", "1 passed")
            self.assertEqual(len(await store.verification_evidence("job", 0)), 1)
            self.assertTrue(await store.consume_revision("job", 0))
            self.assertEqual(await store.verification_evidence("job", 0), [])
            second = await store.reserve_evidence(
                "job", 0, "verification", "execute_tool", {"command": "pytest"}, True)
            await store.complete_evidence(second, "/tmp/second", "hash-1", "2 passed")
            current = await store.verification_evidence("job", 0)
            self.assertEqual([item["evidence_id"] for item in current], [second])

    async def test_records_recommendation_and_actual_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.sqlite3"
            config = load_routing_config()
            recommendation = config.candidates_for("medium")[0]
            actual = config.candidates_for("medium")[1]
            decision = await ModelRoutingService(config, Router(recommendation)).route(
                task(), "CHG1", "/work/project"
            )
            store = RoutingPersistence(path)
            await store.record_decision(decision)
            await store.record_attempt(
                decision, actual, config.models[actual].provider, config.models[actual].executor, False,
                "completed", 12,
            )
            await store.finalize(
                decision, actual, config.models[actual].provider, False, "completed", 12,
                "executor-fallback",
            )
            db = sqlite3.connect(path)
            try:
                row = db.execute("""SELECT recommended_model, actual_model,
                    actual_provider, fallback_reason FROM routing_decisions""").fetchone()
            finally:
                db.close()
            self.assertEqual(row, (recommendation, actual, config.models[actual].provider, "executor-fallback"))


class _FailingExecutor:
    name = "claude-cli"

    async def execute(self, task, model, context):
        yield ErrorEvent("gateway unavailable")
        yield DoneEvent("failed")


class _SuccessfulExecutor:
    name = "codex-cli"

    async def execute(self, task, model, context):
        yield TextEvent("completed by fallback")
        yield DoneEvent("endTurn")


class ExecutionFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_mutation_stops_in_run_fallback(self):
        class MutatingExecutor:
            async def execute(self, _task, _model, _context):
                yield MutationEvent("apply_patch", {"path": "src/a.py"})
                yield ErrorEvent("connection reset")
                yield DoneEvent("failed")

        class BackupExecutor:
            def __init__(self):
                self.calls = 0

            async def execute(self, _task, _model, _context):
                self.calls += 1
                yield DoneEvent("endTurn")

        with tempfile.TemporaryDirectory() as directory:
            config = load_routing_config()
            backup = BackupExecutor()
            service = TaskExecutionService(
                config=config,
                router=ModelRoutingService(config, Router("claude-haiku")),
                persistence=RoutingPersistence(Path(directory) / "routing.sqlite3"),
                executors=ExecutorRegistry({
                    "claude-cli": MutatingExecutor(), "codex-cli": backup,
                }),
            )
            events = [event async for event in service.execute(
                task("simple"), "CHG1", [], [])]
            self.assertTrue(any(isinstance(event, ErrorEvent) for event in events))
            self.assertEqual(backup.calls, 0)


class RunCommandTests(unittest.TestCase):
    def test_accepts_exact_canonical_task_id(self):
        self.assertTrue(is_run_command("/run WP2-T3"))
        self.assertEqual(extract_run_task_id("/run WP2-T3"), "WP2-T3")

    def test_rejects_multiple_or_non_task_targets(self):
        with self.assertRaises(SpecCommandError):
            extract_run_task_id("/run WP2-T3 WP2-T4")
        with self.assertRaises(SpecCommandError):
            extract_run_task_id("/run CHG1")


class ImplementationCommandTests(unittest.TestCase):
    def test_accepts_task_and_work_package_selectors_with_execute_alias(self):
        self.assertTrue(is_implementation_command("/execute WP1 WP2-T3"))
        self.assertEqual(
            extract_implementation_selectors("/implement us1 wp2-task3 WP1"),
            ["WP1", "WP2-T3"],
        )

    def test_rejects_empty_or_invalid_implementation_selectors(self):
        with self.assertRaises(SpecCommandError):
            extract_implementation_selectors("/implement")
        with self.assertRaises(SpecCommandError):
            extract_implementation_selectors("/implement CHG1")

    def test_accepts_instruction_after_colon(self):
        selectors, instruction = extract_implementation_request(
            "/implement WP1-T1 WP2: add MLflow tracking and tests"
        )
        self.assertEqual(selectors, ["WP1-T1", "WP2"])
        self.assertEqual(instruction, "add MLflow tracking and tests")

    def test_rejects_unseparated_instruction_after_selector(self):
        with self.assertRaisesRegex(SpecCommandError, "put execution guidance after ':'"):
            extract_implementation_request("/implement WP1-T1 Don't use globals")

    def test_rejects_empty_instruction_after_colon(self):
        with self.assertRaisesRegex(SpecCommandError, "instruction after ':'"):
            extract_implementation_request("/implement WP1-T1:")


class _Artifacts:
    def __init__(self):
        self.tasks = {
            "WP1-T1": task("simple"),
            "WP1-T2": replace(task("medium"), id="WP1-T2", title="Second task"),
        }
        self.change = type("Change", (), {"id": "CHG1"})()
        self.completed = []
        self.reviews = []

    async def tasks_for_selectors(self, selectors):
        result = []
        for selector in selectors:
            ids = ["WP1-T1", "WP1-T2"] if selector == "WP1" else [selector]
            for task_id in ids:
                if task_id in self.tasks and all(existing.id != task_id for _, existing in result):
                    result.append((self.change, self.tasks[task_id]))
        return result

    async def task_by_id(self, task_id):
        return self.change, self.tasks[task_id]

    async def mark_task_completed(self, task_id, review):
        self.completed.append(task_id)
        self.reviews.append(review)


class _TaskExecution:
    def __init__(self):
        self.started = []
        self.instructions = []

    async def execute(self, queued_task, change_id, messages, specs, **kwargs):
        self.started.append((queued_task.id, change_id))
        self.instructions.append(kwargs.get("execution_instruction"))
        yield TextEvent("Task result with supporting evidence.")
        yield DoneEvent("endTurn")

    async def resume(self, run, messages, specs):
        yield DoneEvent("endTurn")


class ImplementationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_later_task_gap_does_not_pause_work_package(self):
        class SiblingGapExecution(_TaskExecution):
            strict_handoff = True

            async def execute(self, queued_task, change_id, messages, specs, **kwargs):
                self.started.append((queued_task.id, change_id))
                yield TextEvent(json.dumps({
                    "schema_version": 1, "task_id": queued_task.id,
                    "summary": "Inspected the existing loader seams",
                    "changed_files": [],
                    "criteria": [{"criterion": "Inspect source", "status": "satisfied"}],
                    "remaining_gaps": (["T2 still needs to implement its adapter"]
                                       if queued_task.id == "WP1-T1" else []),
                }))
                yield DoneEvent("endTurn")

        with tempfile.TemporaryDirectory() as directory:
            execution = SiblingGapExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            service = ImplementationService(
                _Artifacts(), execution, persistence, _Reviewer(["done"]))
            job = await service.create_job(["WP1"], conversation_id="conv")
            [event async for event in service.start(job, [], [])]
            self.assertEqual(execution.started[:2],
                             [("WP1-T1", "CHG1"), ("WP1-T2", "CHG1")])
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "implemented")

    async def test_later_task_gap_after_tool_resume_reaches_reviewer(self):
        class ResumedExecution(_TaskExecution):
            strict_handoff = True

            async def resume(self, run, messages, specs):
                yield TextEvent(json.dumps({
                    "schema_version": 1, "task_id": "WP1-T1",
                    "summary": "Inspected the existing loader seams",
                    "changed_files": [],
                    "criteria": [{"criterion": "Inspect source", "status": "satisfied"}],
                    "remaining_gaps": ["T2 still needs to implement its adapter"],
                }))
                yield DoneEvent("endTurn")

        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "codex-luna-low"

            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["done"])
            service = ImplementationService(
                _Artifacts(), ResumedExecution(), persistence, reviewer)
            job = await service.create_job(["WP1-T1"], conversation_id="conv")
            await persistence.update_implementation_task(job.job_id, 0, "waiting")
            run = Run(metadata={
                "implementation_job_id": job.job_id,
                "implementation_position": 0,
                "implementation_task_id": "WP1-T1",
            })
            [event async for event in service.resume(run, [], [])]
            self.assertEqual(len(reviewer.reports), 1)
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_user_decision")

    async def test_satisfied_criterion_with_current_task_gap_reaches_reviewer(self):
        report = json.dumps({
            "schema_version": 1, "task_id": "WP1-T1",
            "summary": "Inspected some sources", "changed_files": [],
            "criteria": [{"criterion": "Inspect all sources", "status": "satisfied"}],
            "remaining_gaps": ["One required source was not inspected"],
        })
        self.assertIsNone(handoff_blocker(report, "WP1-T1"))
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "codex-luna-low"

            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["needs_work"] * 4)

            class GapExecution(_TaskExecution):
                strict_handoff = True

                async def execute(self, queued_task, change_id, messages, specs, **kwargs):
                    self.started.append((queued_task.id, change_id))
                    yield TextEvent(report)
                    yield DoneEvent("endTurn")

            service = ImplementationService(
                _Artifacts(), GapExecution(), persistence, reviewer)
            job = await service.create_job(["WP1-T1"], conversation_id="conv")
            [event async for event in service.start(job, [], [])]
            self.assertEqual(len(reviewer.reports), 4)
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_user_decision")

    async def test_blocked_criterion_without_gap_still_pauses(self):
        report = json.dumps({
            "schema_version": 1, "task_id": "WP1-T1",
            "summary": "Inspection could not finish", "changed_files": [],
            "criteria": [{"criterion": "Inspect source", "status": "blocked"}],
            "remaining_gaps": [],
        })
        self.assertIn("Coder handoff blocked", handoff_blocker(report, "WP1-T1"))

    async def test_existing_paused_sibling_gap_can_use_review_anyway(self):
        report = json.dumps({
            "schema_version": 1, "task_id": "WP1-T1",
            "summary": "Inspected existing loader seams", "changed_files": [],
            "criteria": [{"criterion": "Inspect source", "status": "satisfied"}],
            "remaining_gaps": ["T2 still needs to implement its adapter"],
        })
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            execution = _TaskExecution()
            service = ImplementationService(
                _Artifacts(), execution, persistence, _Reviewer(["done"]))
            job = await service.create_job(["WP1"], conversation_id="conv")
            await persistence.save_coder_report(job.job_id, 0, report)
            await persistence.update_implementation_task(
                job.job_id, 0, "awaiting_coder_recovery",
                error="Coder handoff blocked: acceptance criteria or remaining gaps are unresolved")
            events = [event async for event in service.review_blocked_handoff(
                job.job_id, 0, [], [], "", {})]
            self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))
            self.assertEqual(execution.started[0], ("WP1-T2", "CHG1"))
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "implemented")

    async def test_malformed_coder_handoff_switches_with_saved_failure_context(self):
        class MalformedThenValid(_TaskExecution):
            strict_handoff = True

            async def execute(self, queued_task, change_id, messages, specs, **kwargs):
                self.started.append((queued_task.id, change_id))
                self.instructions.append(kwargs.get("execution_instruction"))
                if len(self.started) == 1:
                    yield TextEvent("The first coder stopped with an invalid handoff.")
                else:
                    yield TextEvent(json.dumps({
                        "schema_version": 1, "task_id": queued_task.id,
                        "summary": "Finished after checking the saved task state",
                        "changed_files": [],
                        "criteria": [{"criterion": "Task finished", "status": "satisfied"}],
                        "remaining_gaps": [],
                    }))
                yield DoneEvent("endTurn")

        with tempfile.TemporaryDirectory() as directory:
            execution = MalformedThenValid()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-haiku" if len(execution.started) == 1 else "codex-luna-low"

            persistence.actual_model_for = actual_model
            service = ImplementationService(
                _Artifacts(), execution, persistence, _Reviewer(["done"]))
            job = await service.create_job(["WP1-T1"], conversation_id="conv")
            events = [event async for event in service.start(job, [], [])]
            self.assertEqual(len(execution.started), 2)
            self.assertEqual(execution.instructions[1].count("Coder handoff invalid"), 1)
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_user_decision")
            self.assertTrue(any("continuing WP1-T1" in event.text
                                for event in events if isinstance(event, TextEvent)))

    async def test_blocked_coder_handoff_pauses_before_next_wp_task(self):
        class BlockedExecution(_TaskExecution):
            strict_handoff = True

            async def execute(self, queued_task, change_id, messages, specs, **kwargs):
                self.started.append((queued_task.id, change_id))
                yield TextEvent(json.dumps({
                    "schema_version": 1, "task_id": queued_task.id,
                    "summary": "IDE reader needed approval; no edits made",
                    "changed_files": [],
                    "criteria": [{"criterion": "Inspect source", "status": "blocked"}],
                    "remaining_gaps": ["Read source through IDE MCP"],
                }))
                yield DoneEvent("endTurn")

        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = BlockedExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            reviewer = _Reviewer(["done"])
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1"], conversation_id="blocked-conversation")
            events = [event async for event in service.start(job, [], [])]
            self.assertEqual(execution.started, [("WP1-T1", "CHG1")])
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_coder_recovery")
            self.assertEqual((await persistence.implementation_task(job.job_id, 1)).status,
                             "queued")
            text = "".join(event.text for event in events if isinstance(event, TextEvent))
            self.assertIn("Coder reported unresolved gaps", text)
            self.assertIn("Gap: Read source through IDE MCP", text)
            self.assertIn("`review anyway`", text)

            reviewed = [event async for event in service.review_blocked_handoff(
                job.job_id, 0, [], [], "", {})]
            self.assertFalse(any(isinstance(event, ErrorEvent) for event in reviewed))
            self.assertEqual(execution.started, [("WP1-T1", "CHG1"), ("WP1-T2", "CHG1")])
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "implemented")
            again = [event async for event in service.review_blocked_handoff(
                job.job_id, 0, [], [], "", {})]
            self.assertTrue(any(isinstance(event, ErrorEvent) for event in again))

    async def test_coder_prose_is_saved_but_not_echoed_to_chat(self):
        class ChattyExecution(_TaskExecution):
            async def execute(self, queued_task, change_id, messages, specs, **kwargs):
                self.started.append((queued_task.id, change_id))
                yield TextEvent("I'll help you implement this. Now let me read the files.\n")
                yield TextEvent(json.dumps({
                    "schema_version": 1, "task_id": queued_task.id,
                    "summary": "Added the factory. It registers every loader by name.",
                    "changed_files": [], "remaining_gaps": [],
                    "criteria": [{"criterion": "Task finished", "status": "satisfied"}],
                }))
                yield DoneEvent("endTurn")

        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            service = ImplementationService(
                _Artifacts(), ChattyExecution(), persistence, _Reviewer(["done"]))
            job = await service.create_job(["WP1-T1"], conversation_id="conv")
            events = [event async for event in service.start(job, [], [])]
            text = "".join(event.text for event in events if isinstance(event, TextEvent))
            self.assertNotIn("I'll help you", text)
            self.assertIn("Added the factory.", text)
            self.assertNotIn("registers every loader", text)
            self.assertIn("I'll help you", await persistence.coder_report_for(job.job_id, 0))

    async def test_review_anyway_is_refused_for_an_interrupted_coder(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            service = ImplementationService(
                _Artifacts(), _TaskExecution(), persistence, _Reviewer(["done"]))
            job = await service.create_job(["WP1-T1"], conversation_id="conv")
            await persistence.update_implementation_task(
                job.job_id, 0, "awaiting_coder_recovery", error="provider timed out")
            events = [event async for event in service.review_blocked_handoff(
                job.job_id, 0, [], [], "", {})]
            self.assertTrue(any(isinstance(event, ErrorEvent) for event in events))
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_coder_recovery")

    def test_revision_prompt_requires_direct_artifact_edits_and_real_verification(self):
        finding = ReviewFinding(
            "implementation", "Counts conflict", "Correct the named reports",
            ("Counts agree",), ("REPORT.md",))
        prompt = ImplementationService._revision_context("WP1-T1", (finding,))
        self.assertIn("Edit every reviewer-named artifact_path directly", prompt)
        self.assertIn("Do not create substitute reports", prompt)
        self.assertIn("Never claim an unexecuted check passed", prompt)

    async def test_work_package_runs_all_tasks_before_one_review(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["done"])
            refreshes = []

            async def refresh(job_id, wp_id):
                refreshes.append((job_id, wp_id))
                return "KB refresh succeeded for WP1."

            service = ImplementationService(artifacts, execution, persistence, reviewer,
                                            kb_refresh=refresh)
            job = await service.create_job(["WP1", "WP1-T2"], "Add traceability")
            events = [event async for event in service.start(
                job, [{"role": "user", "content": "/implement WP1"}], []
            )]
            self.assertEqual(execution.started, [("WP1-T1", "CHG1"), ("WP1-T2", "CHG1")])
            self.assertEqual(execution.instructions, ["Add traceability"] * 2)
            self.assertEqual(artifacts.completed, ["WP1-T1", "WP1-T2"])
            self.assertEqual(len(reviewer.reports), 1)
            self.assertEqual(refreshes, [(job.job_id, "WP1")])
            self.assertIn("Task WP1-T1 coder handoff", reviewer.reports[0])
            self.assertIn("Task WP1-T2 coder handoff", reviewer.reports[0])
            self.assertTrue(any("WP1 — PASS" in event.text
                                for event in events if isinstance(event, TextEvent)))
            db = sqlite3.connect(persistence.path)
            try:
                row = db.execute(
                    "SELECT status, instruction FROM implementation_jobs WHERE job_id=?", (job.job_id,)
                ).fetchone()
            finally:
                db.close()
            self.assertEqual(row, ("completed", "Add traceability"))

    async def test_work_package_review_failure_is_unverified_not_coder_failure(self):
        class UnavailableReviewer:
            def __init__(self):
                self.calls = 0

            async def review(self, *args):
                self.calls += 1
                raise ReviewError("required service is unavailable", "codex-sol-medium")

        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-haiku"

            persistence.actual_model_for = actual_model
            reviewer = UnavailableReviewer()
            refreshes = []

            async def refresh(job_id, wp_id):
                refreshes.append((job_id, wp_id))

            service = ImplementationService(artifacts, execution, persistence, reviewer,
                                            kb_refresh=refresh)
            job = await service.create_job(["WP1"])
            events = [event async for event in service.start(job, [], [])]
            self.assertEqual(execution.started,
                             [("WP1-T1", "CHG1"), ("WP1-T2", "CHG1")])
            self.assertEqual(reviewer.calls, 1)
            self.assertEqual(refreshes, [])
            self.assertTrue(any("WP1 — UNVERIFIED" in event.text
                                for event in events if isinstance(event, TextEvent)))
            self.assertEqual(
                (await persistence.implementation_task(job.job_id, 1)).status,
                "review_pending",
            )

    async def test_research_task_uses_review_resolution_interaction(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            artifacts.tasks["WP1-T1"] = replace(artifacts.tasks["WP1-T1"],
                                                 preferred_capability="research")
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["ask_user"])
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"], conversation_id="conv-test")
            events = [event async for event in service.start(job, [], [])]
            self.assertEqual(artifacts.completed, [])
            self.assertTrue(any("Do you consider" in event.text
                                for event in events if isinstance(event, TextEvent)))
            decision_index = next(i for i, event in enumerate(events)
                                  if isinstance(event, TextEvent) and "**Why**" in event.text)
            question_index = next(i for i, event in enumerate(events)
                                  if isinstance(event, TextEvent) and "Reviewer question:" in event.text)
            self.assertLess(decision_index, question_index)
            self.assertEqual((await persistence.pending_resolution("conv-test"))[0], job.job_id)
            events = [event async for event in service.accept_anyway(
                job.job_id, 0, [], [], "", {})]
            self.assertEqual(artifacts.completed, ["WP1-T1"])
            self.assertEqual(artifacts.reviews[0].reason, "Reviewed task result")
            self.assertEqual(artifacts.reviews[0].evidence, ["Inspected task result"])
            self.assertTrue(any("completed" in event.text for event in events if isinstance(event, TextEvent)))

    async def test_submit_again_recodes_and_starts_fresh_review_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            artifacts.tasks["WP1-T1"] = replace(artifacts.tasks["WP1-T1"],
                                                 preferred_capability="research")
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["ask_user", "ask_user"])
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"], conversation_id="conv-test")
            [event async for event in service.start(job, [], [])]
            self.assertEqual(len(execution.started), 1)
            self.assertEqual((await persistence.pending_resolution("conv-test"))[0], job.job_id)
            events = [event async for event in service.retry_coder(
                job.job_id, 0, [], [], "", {}, "add coverage")]
            self.assertEqual(len(execution.started), 2)
            self.assertEqual(await persistence.review_count(job.job_id, 0), 2)
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_user_decision")
            self.assertEqual(await persistence.revision_state(job.job_id, 0), (0, 3))
            self.assertEqual(reviewer.reports,
                             ["Task result with supporting evidence."] * 2)
            self.assertTrue(any("**Why**" in event.text
                                for event in events if isinstance(event, TextEvent)))

    async def test_unrelated_write_does_not_override_review_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["needs_work"] * 4)
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"])
            [event async for event in service.start(job, [], [])]
            self.assertEqual(artifacts.completed, [])
            self.assertEqual(len(execution.started), 4)
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_user_decision")
            latest = await persistence.latest_review(job.job_id, 0)
            self.assertEqual(latest["details"]["gaps"], ["Requested outcome missing"])
            self.assertEqual(latest["details"]["feedback"], "Requested outcome missing")
            self.assertEqual(latest["details"]["findings_schema_version"], 1)
            self.assertEqual(latest["details"]["findings"][0]["problem"],
                             "Requested outcome missing")
            self.assertTrue(latest["details"]["findings"][0]["approval_criteria"])

    async def test_revision_records_final_models_and_chat_explanation(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["needs_work", "done"])
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"])
            events = [event async for event in service.start(job, [], [])]
            self.assertEqual(len(execution.started), 2)
            self.assertEqual(len(artifacts.reviews), 0)
            self.assertEqual(await persistence.review_count(job.job_id, 0), 2)
            self.assertTrue(any("**Implementation findings**" in event.text
                                and "Requested outcome missing" in event.text
                                for event in events if isinstance(event, TextEvent)))
            self.assertTrue(any("approval_criteria" in instruction
                                and "suggested_task" in instruction
                                and "advisory" in instruction
                                for instruction in execution.instructions if instruction))
            [event async for event in service.accept_anyway(
                job.job_id, 0, [], [], "", {})]
            self.assertEqual(len(artifacts.reviews), 1)

    async def test_chat_and_completion_use_models_that_actually_ran(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "codex-sol-high"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["done"], reviewer_models=["claude-opus"])
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"])
            events = [event async for event in service.start(job, [], [])]
            [event async for event in service.accept_anyway(
                job.job_id, 0, [], [], "", {})]
            self.assertEqual(artifacts.reviews[0].coder.label,
                             "Codex GPT-6.1 Sol (high effort)")
            self.assertEqual(artifacts.reviews[0].reviewer.label, "Claude Opus")
            self.assertTrue(any("reviewer: Claude Opus" in event.text
                                for event in events if isinstance(event, TextEvent)))

    async def test_submit_again_and_accept_use_latest_review(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["ask_user", "ask_user"],
                                 reviewer_models=["codex-luna-low", "claude-haiku"])
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"])
            [event async for event in service.start(job, [], [])]
            events = [event async for event in service.retry_coder(
                job.job_id, 0, [], [], "", {}, "add coverage")]
            self.assertEqual(artifacts.completed, [])
            self.assertEqual(len(execution.started), 2)
            self.assertTrue(any("reviewer: Claude Haiku" in event.text
                                for event in events if isinstance(event, TextEvent)))
            [event async for event in service.accept_anyway(
                job.job_id, 0, [], [], "", {})]
            self.assertEqual(artifacts.reviews[0].reviewer.label, "Claude Haiku")

    async def test_review_error_identifies_coder_without_inventing_reviewer(self):
        class FailingReviewer:
            async def review(self, *args):
                raise ReviewError("provider unavailable")

        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            service = ImplementationService(artifacts, execution, persistence,
                                            FailingReviewer())
            job = await service.create_job(["WP1-T1"])
            events = [event async for event in service.start(job, [], [])]
            self.assertEqual(artifacts.completed, [])
            self.assertTrue(any("coder: Claude Haiku" in event.text
                                and "reviewer: not run" in event.text
                                for event in events if isinstance(event, TextEvent)))

    async def test_review_error_identifies_reviewer_that_ran_but_could_not_verify(self):
        class IncompleteReviewer:
            async def review(self, *args):
                raise ReviewError("reviewer returned no concrete evidence",
                                  "codex-sol-medium")

        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-sonnet"

            persistence.actual_model_for = actual_model
            service = ImplementationService(artifacts, execution, persistence,
                                            IncompleteReviewer())
            job = await service.create_job(["WP1-T1"])
            events = [event async for event in service.start(job, [], [])]
            self.assertEqual(artifacts.completed, [])
            self.assertTrue(any(
                "reviewer: Codex GPT-6.1 Sol (medium effort) "
                "(ran, verification incomplete)" in event.text
                for event in events if isinstance(event, TextEvent)))

    async def test_incomplete_mutation_journal_warns_the_reviewer(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-haiku"

            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["done"])
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"])
            await persistence.record_resolution_action(
                job.job_id, 0, "rollback_snapshot_failed", {
                    "tool": "edit_notebook",
                    "path": "models/train.ipynb",
                    "error": "raw read unavailable",
                })
            events = [event async for event in service.start(job, [], [])]
            # Not blocking any more: the reviewer runs and is told about the gap.
            self.assertEqual(len(reviewer.reports), 1)
            self.assertIn("CodePartner journal warning", reviewer.reports[0])
            self.assertIn("models/train.ipynb", reviewer.reports[0])
            self.assertNotEqual(
                (await persistence.implementation_task(job.job_id, 0)).status,
                "recovery_required")
            self.assertTrue(any("not fully recorded in the change journal" in event.text
                                for event in events if isinstance(event, TextEvent)))

    async def test_failed_current_review_cannot_accept_stale_needs_work_verdict(self):
        finding = ReviewFinding(
            "implementation", "First attempt has a defect", "Correct the behavior",
            ("The corrected behavior is visible in src/trainer.py",),
            ("src/trainer.py",),
        )

        class FailsAfterRevision:
            def __init__(self):
                self.calls = 0

            async def review(self, *args):
                self.calls += 1
                if self.calls == 1:
                    return ReviewDecision(
                        "needs_work", "First attempt needs work", "codex-luna-low",
                        feedback="Correct the behavior", findings=(finding,))
                raise ReviewError("current revised attempt could not be verified")

        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-haiku"

            persistence.actual_model_for = actual_model
            service = ImplementationService(
                artifacts, execution, persistence, FailsAfterRevision())
            job = await service.create_job(["WP1-T1"], conversation_id="conv-test")
            [event async for event in service.start(job, [], [])]
            self.assertEqual(len(execution.started), 2)
            self.assertEqual(
                (await persistence.implementation_task(job.job_id, 0)).status,
                "review_pending")
            self.assertIsNone(await persistence.pending_resolution("conv-test"))
            events = [event async for event in service.accept_anyway(
                job.job_id, 0, [], [], "", {})]
            self.assertEqual(artifacts.completed, [])
            self.assertTrue(any(isinstance(event, ErrorEvent) for event in events))

    async def test_verification_findings_do_not_rerun_the_coder(self):
        finding = ReviewFinding(
            "verification", "Integration tests could not be read",
            "Restore IDE MCP read access and retry review",
            ("The reviewer can inspect tests/test_integration.py",),
            ("tests/test_integration.py",),
        )

        class IncompleteReviewer:
            async def review(self, *args):
                raise ReviewError("Required evidence was unavailable",
                                  "codex-sol-medium", (finding,))

        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-sonnet"

            persistence.actual_model_for = actual_model
            service = ImplementationService(artifacts, execution, persistence,
                                            IncompleteReviewer())
            job = await service.create_job(["WP1-T1"])
            events = [event async for event in service.start(job, [], [])]
            self.assertEqual(len(execution.started), 1)
            self.assertEqual(await persistence.review_count(job.job_id, 0), 1)
            self.assertEqual(await persistence.revision_count(job.job_id, 0), 0)
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "review_pending")
            latest = await persistence.latest_review(job.job_id, 0)
            self.assertEqual(latest["verdict"], "review_pending")
            self.assertEqual(latest["details"]["run_status"], "ran_incomplete")
            self.assertEqual(latest["details"]["findings"][0]["kind"], "verification")
            self.assertTrue(any("**Verification incomplete**" in event.text
                                and "**Integration tests could not be read**" in event.text
                                for event in events if isinstance(event, TextEvent)))

    async def test_coder_owned_verification_finding_starts_revision(self):
        finding = ReviewFinding(
            "verification", "No captured pytest execution result is available",
            "Run the required test module through the mediated runner",
            ("A successful captured test run is linked to this attempt",),
            ("tests/test_integration.py",), "coder", "missing_execution",
        )

        class MissingThenVerified:
            def __init__(self):
                self.calls = 0

            async def review(self, *args):
                self.calls += 1
                if self.calls == 1:
                    return ReviewDecision(
                        "needs_work", "Execution evidence is missing", "codex-sol-medium",
                        feedback=finding.problem, findings=(finding,))
                return ReviewDecision(
                    "done", "Execution is now captured", "codex-sol-medium",
                    evidence=("Captured run passed",))

        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-sonnet"

            persistence.actual_model_for = actual_model
            service = ImplementationService(
                artifacts, execution, persistence, MissingThenVerified())
            job = await service.create_job(["WP1-T1"])
            [event async for event in service.start(job, [], [])]
            self.assertEqual(len(execution.started), 2)
            self.assertIn("No captured pytest execution result", execution.instructions[1])
            self.assertIn('"owner": "coder"', execution.instructions[1])

    async def test_unchanged_evidence_service_failure_cannot_retry_review(self):
        finding = ReviewFinding(
            "verification", "The required source snapshot is truncated",
            "Complete the paginated source read",
            ("The complete source is available",),
            ("tests/test_large.py",), "evidence_service", "incomplete_snapshot",
        )

        class IncompleteEvidence:
            def __init__(self):
                self.calls = 0

            async def review(self, *args):
                self.calls += 1
                raise ReviewError(
                    "Required evidence remains incomplete", "codex-sol-medium", (finding,))

        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-sonnet"

            persistence.actual_model_for = actual_model
            reviewer = IncompleteEvidence()
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"])
            [event async for event in service.start(job, [], [])]
            events = [event async for event in service.retry_review(
                job.job_id, 0, [], [], "", {})]
            self.assertEqual(reviewer.calls, 1)
            self.assertTrue(any(
                isinstance(event, ErrorEvent) and "evidence is unchanged" in event.message
                for event in events))

    async def test_three_revisions_then_submit_again_starts_fresh_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-haiku"

            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["needs_work"] * 8)
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"], conversation_id="conv-test")
            first_events = [event async for event in service.start(job, [], [])]
            self.assertEqual(len(execution.started), 4)
            self.assertEqual(await persistence.revision_state(job.job_id, 0), (3, 3))
            self.assertTrue(any("submit to coder again" in event.text
                                and "`accept`" in event.text
                                and "`undo`" not in event.text
                                for event in first_events if isinstance(event, TextEvent)))
            self.assertEqual((await persistence.pending_resolution("conv-test"))[0],
                             job.job_id)

            retry_events = [event async for event in service.retry_coder(
                job.job_id, 0, [], [], "", {}, "Prefer immutable configuration")]
            self.assertEqual(len(execution.started), 8)
            self.assertEqual(await persistence.revision_state(job.job_id, 0), (3, 3))
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_user_decision")
            self.assertTrue(any("fresh 3-revision review cycle" in event.text
                                for event in retry_events if isinstance(event, TextEvent)))
            self.assertTrue(any("Prefer immutable configuration" in value
                                for value in execution.instructions if value))

    async def test_accept_anyway_preserves_needs_work_as_user_override(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-haiku"

            persistence.actual_model_for = actual_model
            service = ImplementationService(
                artifacts, execution, persistence, _Reviewer(["needs_work"] * 4))
            job = await service.create_job(["WP1-T1"], conversation_id="conv-test")
            [event async for event in service.start(job, [], [])]
            events = [event async for event in service.accept_anyway(
                job.job_id, 0, [], [], "", {}, "Accepted for the prototype")]
            self.assertEqual(artifacts.completed, ["WP1-T1"])
            self.assertIn("User override", artifacts.reviews[0].reason)
            self.assertIn("needs_work", artifacts.reviews[0].reason)
            self.assertTrue(any("reviewer verdict remains needs_work" in event.text
                                for event in events if isinstance(event, TextEvent)))
            db = sqlite3.connect(persistence.path)
            try:
                action = db.execute("""SELECT action FROM implementation_resolution_actions
                  WHERE job_id=? ORDER BY id DESC LIMIT 1""", (job.job_id,)).fetchone()
            finally:
                db.close()
            self.assertEqual(action, ("accept_override",))
            duplicate = [event async for event in service.accept_anyway(
                job.job_id, 0, [], [], "", {}, "duplicate")]
            self.assertEqual(artifacts.completed, ["WP1-T1"])
            self.assertTrue(any(isinstance(event, ErrorEvent) for event in duplicate))

    def test_findings_render_as_parseable_json(self):
        finding = ReviewFinding(
            "implementation", "Factory ignores config", "Honor supplied configuration",
            ("Configured creation succeeds",), ("trainer/factory.py",),
        )
        lines = ImplementationService._findings_json((finding,))
        self.assertEqual(lines[0], "```json")
        self.assertEqual(lines[-1], "```")
        payload = json.loads(lines[1])
        self.assertEqual(payload[0]["problem"], "Factory ignores config")
        self.assertEqual(payload[0]["approval_criteria"],
                         ["Configured creation succeeds"])


class RollbackJournalTests(unittest.IsolatedAsyncioTestCase):
    async def test_communication_only_artifact_is_blocked_unless_task_declares_it(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            files = {}
            calls = []

            async def read_file(arguments):
                path = arguments["file_path"]
                return f"File {path} doesn't exist or can't be opened"

            async def create_file(arguments):
                calls.append(arguments["pathInProject"])
                files[arguments["pathInProject"]] = arguments["text"]
                return "created"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("create_new_file", "create", {"type": "object", "properties": {
                    "pathInProject": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["pathInProject", "text"]}),
            ]
            handlers = {"read_file": read_file, "create_new_file": create_file}
            blocked = journal_handlers(
                persistence, "job", 0, specs, handlers)["create_new_file"]
            result = await blocked({
                "pathInProject": "WP2_FINAL_SUMMARY.md", "text": "status complete"})
            self.assertIn("communication artifacts", result)
            self.assertEqual(calls, [])

            allowed = journal_handlers(
                persistence, "other-job", 0, specs, handlers,
                frozenset({"WP2_FINAL_SUMMARY.md"}))["create_new_file"]
            self.assertEqual(await allowed({
                "pathInProject": "WP2_FINAL_SUMMARY.md", "text": "required doc"}),
                "created")
            self.assertEqual(calls, ["WP2_FINAL_SUMMARY.md"])

    async def test_ad_hoc_validation_scripts_and_notebooks_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            calls = []

            async def read_file(arguments):
                return f"File {arguments['file_path']} doesn't exist or can't be opened"

            async def create_file(arguments):
                calls.append(arguments["pathInProject"])
                return "created"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("create_new_file", "create", {"type": "object", "properties": {
                    "pathInProject": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["pathInProject", "text"]}),
            ]
            wrapped = journal_handlers(
                persistence, "job", 0, specs,
                {"read_file": read_file, "create_new_file": create_file})
            paths = [
                "wp2_t4_final_validation.py",
                "verify_wp2_t4_comprehensive.py",
                "test_wp2_t4_execution.py",
                "tmp_pytest_runner.ipynb",
            ]
            for path in paths:
                result = await wrapped["create_new_file"]({
                    "pathInProject": path, "text": "import pytest\npytest.main()"})
                self.assertIn("evidence workspace", result)
            generic = await wrapped["create_new_file"]({
                "pathInProject": "helper.py",
                "text": "import subprocess\nsubprocess.run(['pytest'])",
            })
            self.assertIn("evidence workspace", generic)
            self.assertEqual(calls, [])

    async def test_explicit_task_artifact_can_use_validation_like_name(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def read_file(arguments):
                return f"File {arguments['file_path']} doesn't exist or can't be opened"

            async def create_file(_arguments):
                return "created"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("create_new_file", "create", {"type": "object", "properties": {
                    "pathInProject": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["pathInProject", "text"]}),
            ]
            wrapped = journal_handlers(
                persistence, "job", 0, specs,
                {"read_file": read_file, "create_new_file": create_file},
                frozenset({"src/input_validation.py"}))
            result = await wrapped["create_new_file"]({
                "pathInProject": "src/input_validation.py", "text": "def validate(): pass"})
            self.assertEqual(result, "created")

    async def test_task_created_helper_is_quarantined_on_later_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            await persistence.record_rollback_file(
                "job", 0, "verify_feature.py", False, "old-hash")
            calls = []

            async def read_file(_arguments):
                return "old helper"

            async def replace_file(arguments):
                calls.append(arguments)
                return "replaced"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("replace_file_text_by_path", "replace", {
                    "type": "object", "properties": {
                        "file_path": {"type": "string"}, "text": {"type": "string"}}}),
            ]
            wrapped = journal_handlers(
                persistence, "job", 0, specs,
                {"read_file": read_file, "replace_file_text_by_path": replace_file})
            result = await wrapped["replace_file_text_by_path"]({
                "file_path": "verify_feature.py", "text": "import pytest"})
            self.assertIn("quarantined", result)
            self.assertEqual(calls, [])

    async def test_execution_dispute_is_replayed_into_evidence_store(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def execute_tool(arguments):
                self.assertIn("pytest", arguments["command"])
                return "collected 1 item\n1 passed\nexit_code: 0"

            spec = ToolSpec("execute_tool", "dispatch", {
                "type": "object", "properties": {"command": {"type": "string"}},
                "required": ["command"],
            })
            handlers = {"execute_tool": execute_tool}
            journal = RollbackJournal(persistence, "job", 0, [spec], handlers)
            report = json.dumps({
                "schema_version": 1,
                "execution_dispute": {
                    "claim": "The target tests pass",
                    "source_paths": ["tests/test_feature.py"],
                    "working_directory": ".",
                    "steps": [{
                        "tool": "execute_tool",
                        "arguments": {"command": "execute_terminal_command pytest tests/test_feature.py"},
                    }],
                },
            })
            service = object.__new__(ImplementationService)
            with mock.patch.dict(
                    "os.environ", {"CODEPARTNER_EVIDENCE_DIR": str(Path(directory) / "evidence")}):
                errors = await service._capture_execution_dispute(
                    journal, report, handlers, [])
            self.assertEqual(errors, [])
            evidence = await persistence.verification_evidence("job", 0)
            self.assertEqual(len(evidence), 1)
            self.assertTrue(evidence[0]["success"])
            self.assertEqual(evidence[0]["metadata"]["pytest"]["totals"]["passed"], 1)

    def test_handoff_evidence_ids_must_resolve_to_captured_records(self):
        report = json.dumps({
            "schema_version": 1,
            "criteria": [{"evidence_ids": ["syntax_valid", 7]}],
        })
        errors = ImplementationService._handoff_evidence_errors(
            report, [{"evidence_id": 7}])
        self.assertEqual(len(errors), 1)
        self.assertIn("syntax_valid", errors[0])

    async def test_execution_dispute_cannot_replay_generic_shell_commands(self):
        service = object.__new__(ImplementationService)
        handler = mock.AsyncMock(return_value="deleted")
        report = json.dumps({
            "schema_version": 1,
            "execution_dispute": {
                "claim": "cleanup proves the test passes",
                "source_paths": ["tests/test_feature.py"],
                "working_directory": ".",
                "steps": [{
                    "tool": "execute_tool",
                    "arguments": {"command": "execute_terminal_command rm -rf build"},
                }],
            },
        })
        errors = await service._capture_execution_dispute(
            mock.AsyncMock(), report, {"execute_tool": handler}, [])
        self.assertEqual(len(errors), 1)
        handler.assert_not_awaited()

    async def test_verification_output_is_stored_outside_project_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def execute_tool(_arguments):
                return "collected 2 items\n2 passed\nexit_code: 0"

            spec = ToolSpec("execute_tool", "dispatch", {
                "type": "object", "properties": {"command": {"type": "string"}},
                "required": ["command"],
            })
            with mock.patch.dict(
                    "os.environ", {"CODEPARTNER_EVIDENCE_DIR": str(Path(directory) / "evidence")}):
                wrapped = journal_handlers(
                    persistence, "job", 0, [spec], {"execute_tool": execute_tool})
                result = await wrapped["execute_tool"]({
                    "command": "execute_run_configuration --name tests"})
            self.assertIn("2 passed", result)
            evidence = await persistence.verification_evidence("job", 0)
            self.assertEqual(len(evidence), 1)
            self.assertTrue(evidence[0]["success"])
            self.assertEqual(evidence[0]["metadata"]["pytest"]["collected"], 2)
            self.assertEqual(evidence[0]["metadata"]["pytest"]["totals"]["passed"], 2)
            self.assertTrue(Path(evidence[0]["payload_path"]).is_file())
            self.assertEqual(await persistence.mutations("job", 0), [])

    async def test_notebook_mutation_is_journaled(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            files = {"models/train.ipynb": "before notebook"}

            async def read_file(arguments):
                return files[arguments["file_path"]]

            async def create_file(arguments):
                files[arguments["pathInProject"]] = arguments["text"]
                return "created"

            async def edit_notebook(arguments):
                files[arguments["file_path"]] = "after notebook"
                return "edited"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("create_new_file", "create", {"type": "object", "properties": {
                    "pathInProject": {"type": "string"}, "text": {"type": "string"},
                    "overwrite": {"type": "boolean"}}}),
                ToolSpec("edit_notebook", "edit", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "new_source": {"type": "string"}}}),
            ]
            handlers = {"read_file": read_file, "create_new_file": create_file,
                        "edit_notebook": edit_notebook}
            wrapped = journal_handlers(persistence, "job", 0, specs, handlers)
            await wrapped["edit_notebook"]({
                "file_path": "models/train.ipynb", "new_source": "changed",
            })
            journal = RollbackJournal(persistence, "job", 0, specs, handlers)
            self.assertEqual(await journal.changed_paths(), ["models/train.ipynb"])
            self.assertEqual([entry["change"] for entry in await journal.review_manifest()],
                             ["modified"])

    async def test_unreadable_preimage_does_not_block_the_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            files = {"src/tracked.py": "before", "models/train.ipynb": "before notebook"}

            async def read_file(arguments):
                path = arguments["file_path"]
                if path.endswith(".ipynb"):
                    raise RuntimeError("notebook raw read unavailable")
                return files[path]

            async def create_file(arguments):
                files[arguments["pathInProject"]] = arguments["text"]
                return "created"

            async def replace_file(arguments):
                files[arguments["file_path"]] = arguments["text"]
                return "replaced"

            async def edit_notebook(arguments):
                files[arguments["file_path"]] = "after notebook"
                return "edited"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("create_new_file", "create", {"type": "object", "properties": {
                    "pathInProject": {"type": "string"}, "text": {"type": "string"}}}),
                ToolSpec("replace_file_text_by_path", "replace", {
                    "type": "object", "properties": {
                        "file_path": {"type": "string"}, "text": {"type": "string"}}}),
                ToolSpec("edit_notebook", "edit", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "new_source": {"type": "string"}}}),
            ]
            handlers = {"read_file": read_file, "create_new_file": create_file,
                        "replace_file_text_by_path": replace_file,
                        "edit_notebook": edit_notebook}
            wrapped = journal_handlers(persistence, "job", 0, specs, handlers)
            await wrapped["replace_file_text_by_path"](
                {"file_path": "src/tracked.py", "text": "after"})
            result = await wrapped["edit_notebook"](
                {"file_path": "models/train.ipynb", "new_source": "changed"})
            # The write runs; its unreadable postimage is an integrity warning for review.
            self.assertNotIn("blocked before execution", str(result))
            self.assertEqual(files["src/tracked.py"], "after")
            self.assertEqual(files["models/train.ipynb"], "after notebook")
            journal = RollbackJournal(persistence, "job", 0, specs, handlers)
            self.assertIn("src/tracked.py", await journal.changed_paths())
            self.assertTrue(any(issue["action"] == "rollback_postimage_failed"
                                for issue in await journal.integrity_issues()))

    async def test_apply_patch_input_tracks_multifile_move_and_unified_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            files = {
                "src/old.py": "old", "src/a.py": "a", "src/b.py": "b",
            }

            async def read_file(arguments):
                path = arguments["file_path"]
                return files.get(path, f"File {path} doesn't exist or can't be opened")

            async def apply_patch(arguments):
                files["src/new.py"] = files.pop("src/old.py")
                files["src/a.py"] = "changed a"
                files["src/b.py"] = "changed b"
                return "Done!"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("apply_patch", "patch", {"type": "object", "properties": {
                    "input": {"type": "string"}}}),
            ]
            handlers = {"read_file": read_file, "apply_patch": apply_patch}
            patch = """*** Update File: src/old.py
*** Move to: src/new.py
--- a/src/a.py
+++ b/src/a.py
--- a/src/b.py
+++ b/src/b.py
"""
            result = await journal_handlers(
                persistence, "job", 0, specs, handlers)["apply_patch"]({"input": patch})
            self.assertEqual(result, "Done!")
            self.assertCountEqual(
                await RollbackJournal(persistence, "job", 0, specs, handlers).changed_paths(),
                ["src/old.py", "src/new.py", "src/a.py", "src/b.py"])

    async def test_unknown_mutation_shape_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            called = False

            async def mutate(_arguments):
                nonlocal called
                called = True
                return "changed"

            spec = ToolSpec("replace_file_text_by_path", "replace", {
                "type": "object", "properties": {"opaque": {"type": "string"}}})
            result = await journal_handlers(
                persistence, "job", 0, [spec],
                {"replace_file_text_by_path": mutate})["replace_file_text_by_path"](
                    {"opaque": "value"})
            self.assertFalse(called)
            self.assertTrue(result.startswith("Error: mutation blocked before execution"))
            mutation = (await persistence.mutations("job", 0))[0]
            self.assertEqual(mutation["stage"], "planned")
            self.assertEqual(mutation["failure_stage"], "target_resolution")
            self.assertEqual(mutation["targets"], [])
            self.assertEqual(
                await RollbackJournal(
                    persistence, "job", 0, [spec], {}).integrity_issues(), [])

    async def test_postimage_failure_marks_recovery_and_blocks_later_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            files = {"src/a.py": "before"}
            reads = 0
            mutations = 0

            async def read_file(arguments):
                nonlocal reads
                reads += 1
                if reads > 1:
                    return "Error: IDE read failed"
                return files[arguments["file_path"]]

            async def replace_file(arguments):
                nonlocal mutations
                mutations += 1
                files[arguments["file_path"]] = arguments["text"]
                return "replaced"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("replace_file_text_by_path", "replace", {
                    "type": "object", "properties": {
                        "file_path": {"type": "string"}, "text": {"type": "string"}}}),
            ]
            handlers = {"read_file": read_file,
                        "replace_file_text_by_path": replace_file}
            wrapped = journal_handlers(persistence, "job", 0, specs, handlers)
            first = await wrapped["replace_file_text_by_path"](
                {"file_path": "src/a.py", "text": "after"})
            second = await wrapped["replace_file_text_by_path"](
                {"file_path": "src/a.py", "text": "again"})
            self.assertIn("post-image could not be verified", first)
            self.assertIn("requires journal recovery", second)
            self.assertEqual(mutations, 1)
            mutation = (await persistence.mutations("job", 0))[0]
            self.assertEqual(mutation["stage"], "applied")
            self.assertEqual(mutation["failure_stage"], "postimage")
            self.assertEqual(
                (await RollbackJournal(
                    persistence, "job", 0, specs, handlers).integrity_issues())[0]["action"],
                "rollback_postimage_failed")

    async def test_path_arrays_are_snapshotted_before_reformat(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            files = {"src/a.py": "a", "src/b.py": "b"}

            async def read_file(arguments):
                return files[arguments["file_path"]]

            async def reformat(arguments):
                for path in arguments["files"]:
                    files[path] += " formatted"
                return "formatted"

            specs = [
                ToolSpec("read_file", "read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "limit": {"type": "integer"}}}),
                ToolSpec("reformat_file", "format", {"type": "object", "properties": {
                    "files": {"type": "array", "items": {"type": "string"}}}}),
            ]
            handlers = {"read_file": read_file, "reformat_file": reformat}
            await journal_handlers(persistence, "job", 0, specs, handlers)["reformat_file"](
                {"files": ["src/a.py", "src/b.py"]})
            self.assertCountEqual(
                await RollbackJournal(persistence, "job", 0, specs, handlers).changed_paths(),
                ["src/a.py", "src/b.py"])

    async def test_review_manifest_describes_rename_and_deletion_without_preimages(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            await persistence.record_rollback_file("job", 0, "old.py", True, "__missing__")
            await persistence.record_rollback_file("job", 0, "new.py", False, "new-hash")
            mutation = await persistence.plan_mutation(
                "job", 0, "rename_refactoring", "rename", ["old.py", "new.py"])
            await persistence.advance_mutation(mutation, "postimage_verified")
            journal = RollbackJournal(persistence, "job", 0, [], {})
            manifest = await journal.review_manifest()
            self.assertEqual({item["path"]: item["change"] for item in manifest},
                             {"old.py": "deleted", "new.py": "created"})
            self.assertEqual({item["path"]: item["rename_peer"] for item in manifest},
                             {"old.py": "new.py", "new.py": "old.py"})
            self.assertNotIn("private old", json.dumps(manifest))
            self.assertCountEqual(await journal.changed_paths(), ["old.py", "new.py"])


class _RecordingReviewer:
    """Fake reviewer provider: records its prompt, optionally reads one file, returns a verdict."""

    def __init__(self, read_path=None, verdict=None):
        self.prompt = None
        self.read_path = read_path
        self.verdict = verdict

    async def tools(self, messages, specs, **kwargs):
        self.prompt = messages[0]["content"]
        if self.read_path:
            await kwargs["direct_tool_handlers"]["read_file"]({"file_path": self.read_path})
        if self.verdict:
            yield TextEvent(json.dumps(self.verdict))
        yield DoneEvent("endTurn")


class _Reviewer:
    def __init__(self, verdicts, reviewer_models=None):
        self.verdicts = list(verdicts)
        self.reviewer_models = list(reviewer_models or ["codex-luna-low"] * len(verdicts))
        self.reports = []

    async def review(self, change, queued_task, report, coder_model, specs, handlers):
        self.reports.append(report)
        verdict = self.verdicts.pop(0)
        return ReviewDecision(verdict, "Reviewed task result", self.reviewer_models.pop(0),
                              f"Do you consider {queued_task.id} complete?" if verdict == "ask_user" else None,
                              "Requested outcome missing" if verdict == "needs_work" else None,
                              ("Inspected task result",),
                              ("Requested outcome missing",) if verdict == "needs_work" else (),
                              (),
                              ((ReviewFinding(
                                  "implementation", "Requested outcome missing",
                                  "Implement the missing outcome",
                                  ("The requested outcome is observable in the implementation",),
                                  ("report.md",),
                              ),) if verdict == "needs_work" else ()))


class _SelectingReviewer(ReviewService):
    def __init__(self, config, unavailable=False, model_unavailable=False):
        super().__init__(config)
        self.selected = []
        self.unavailable = unavailable
        self.model_unavailable = model_unavailable

    async def _run_review(self, change, queued_task, report, reviewer_name, specs, handlers):
        self.selected.append(reviewer_name)
        if self.unavailable and len(self.selected) == 1:
            raise ReviewUnavailableError("authentication required")
        if self.model_unavailable and len(self.selected) == 1:
            raise ReviewUnavailableError("model not available")
        return ReviewDecision("done", "Task outcome reviewed", reviewer_name)


class ReviewerSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # These assertions cover the legacy preloaded-snapshot review mode.
        config = load_routing_config()
        selective = replace(config, discovery=replace(config.discovery, policy="selective"))
        patch = mock.patch("agent.routing.review.load_routing_config", return_value=selective)
        patch.start()
        self.addCleanup(patch.stop)

    def test_review_paths_are_relative_and_exclude_mcp_configuration(self):
        paths = _candidate_paths("Created `src/trainer.py` and `tests/test_trainer.py`; "
                                 "not ../private.py, /outside.py, or "
                                 "`.codepartner/ide_mcp.json`.")
        self.assertEqual(paths, ["src/trainer.py", "tests/test_trainer.py"])

    def test_structured_coder_handoff_is_a_discovery_source(self):
        report = json.dumps({
            "schema_version": 1,
            "task_id": "WP1-T1",
            "summary": "Implemented routing",
            "changed_files": [{
                "path": "src/routing/service.py",
                "change": "modified",
                "purpose": "Implement routing",
            }],
            "criteria": [],
            "remaining_gaps": [],
        })
        candidates, unsupported = _artifact_candidates(report)
        self.assertEqual(unsupported, [])
        self.assertEqual(candidates[0].path, "src/routing/service.py")
        self.assertIn("coder_manifest", candidates[0].sources)

    def test_review_decision_parser_accepts_final_json_after_commentary(self):
        value = ReviewService._parse_decision(
            'I inspected the result.\n```json\n'
            '{"verdict":"done","rationale":"The task outcome is complete."}\n```'
        )
        self.assertEqual(value["verdict"], "done")

    async def test_cross_provider_pair_uses_actual_coder_model(self):
        reviewer = _SelectingReviewer(load_routing_config())
        await reviewer.review(None, task(), "result", "claude-haiku", [], {})
        await reviewer.review(None, task(), "result", "codex-sol-high", [], {})
        self.assertEqual(reviewer.selected, ["codex-luna-low", "claude-opus"])

    async def test_single_provider_uses_same_provider(self):
        model = ModelRef("claude-sonnet", "claude", "claude-cli", "sonnet")
        config = RoutingConfig({model.name: model}, {"medium": (model.name,)},
                               "http://router", "http://gateway", reviewers={})
        reviewer = _SelectingReviewer(config)
        await reviewer.review(None, task(), "result", model.name, [], {})
        self.assertEqual(reviewer.selected, [model.name])

    async def test_unavailable_opposite_provider_uses_same_provider(self):
        reviewer = _SelectingReviewer(load_routing_config(), unavailable=True)
        await reviewer.review(None, task(), "result", "claude-haiku", [], {})
        self.assertEqual(reviewer.selected, ["codex-luna-low", "claude-haiku"])

    async def test_codex_coder_uses_codex_if_claude_is_unavailable(self):
        reviewer = _SelectingReviewer(load_routing_config(), unavailable=True)
        decision = await reviewer.review(None, task(), "result", "codex-sol-high", [], {})
        self.assertEqual(reviewer.selected, ["claude-opus", "codex-sol-high"])
        self.assertEqual(decision.reviewer_model, "codex-sol-high")
        self.assertIn("same-provider fallback", decision.routing_reason)

    async def test_unavailable_model_tries_other_opposite_model_in_same_tier(self):
        reviewer = _SelectingReviewer(load_routing_config(), model_unavailable=True)
        await reviewer.review(None, task(), "result", "claude-sonnet", [], {})
        self.assertEqual(reviewer.selected, ["codex-sol-medium", "codex-luna-medium"])

    async def test_all_opposite_models_unavailable_then_same_provider(self):
        class UnavailableReviewer(_SelectingReviewer):
            async def _run_review(self, change, queued_task, report, reviewer_name, specs, handlers):
                self.selected.append(reviewer_name)
                if reviewer_name.startswith("codex"):
                    raise ReviewUnavailableError("model not available")
                return ReviewDecision("done", "Reviewed", reviewer_name)

        reviewer = UnavailableReviewer(load_routing_config())
        decision = await reviewer.review(None, task(), "result", "claude-sonnet", [], {})
        self.assertEqual(reviewer.selected,
                         ["codex-sol-medium", "codex-luna-medium", "claude-sonnet"])
        self.assertEqual(decision.reviewer_model, "claude-sonnet")

    async def test_substantive_review_error_does_not_switch_provider(self):
        class EvidenceFailure(_SelectingReviewer):
            async def _run_review(self, change, queued_task, report, reviewer_name, specs, handlers):
                self.selected.append(reviewer_name)
                raise ReviewError("required project evidence is unreadable")

        reviewer = EvidenceFailure(load_routing_config())
        with self.assertRaisesRegex(ReviewError, "evidence is unreadable"):
            await reviewer.review(None, task(), "result", "codex-sol-high", [], {})
        self.assertEqual(reviewer.selected, ["claude-opus"])

    async def test_no_available_reviewer_still_reports_failure(self):
        class NoProvider(_SelectingReviewer):
            async def _run_review(self, change, queued_task, report, reviewer_name, specs, handlers):
                self.selected.append(reviewer_name)
                raise ReviewUnavailableError("authentication required", reviewer_name)

        reviewer = NoProvider(load_routing_config())
        with self.assertRaises(ReviewUnavailableError):
            await reviewer.review(None, task(), "result", "codex-sol-high", [], {})
        self.assertEqual(reviewer.selected, ["claude-opus", "codex-sol-high"])

    async def test_reviewer_preferred_gateway_failure_retries_same_model_direct(self):
        class Provider:
            def __init__(self):
                self.calls = []

            async def tools(self, messages, specs, **kwargs):
                self.calls.append(kwargs)
                if kwargs.get("execution_env"):
                    yield ErrorEvent("HTTP 503 service unavailable")
                    yield DoneEvent("endTurn")
                else:
                    await kwargs["direct_tool_handlers"]["read_file"]({"file_path": "report.md"})
                    yield TextEvent(json.dumps({
                        "verdict": "done", "rationale": "Inspected the report",
                        "evidence": ["report.md confirms the outcome"],
                        "inspected_paths": ["report.md"], "findings": [],
                    }))
                    yield DoneEvent("endTurn")

        async def read_file(_arguments):
            return "The implemented report."

        provider = Provider()
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            decision = await ReviewService().review(
                SimpleNamespace(title="Review", work_packages=[]), task(),
                "Implemented `report.md`.", "codex-sol-medium",
                [ToolSpec("read_file", "Read", {"type": "object", "properties": {
                    "file_path": {"type": "string"}}})], {"read_file": read_file})
        self.assertEqual(decision.reviewer_model, "claude-sonnet")
        self.assertFalse(decision.gateway_used)
        self.assertEqual(provider.calls[0]["execution_env"],
                         {"ANTHROPIC_BASE_URL": load_routing_config().agentgateway_url})
        self.assertNotIn("execution_env", provider.calls[1])

    async def test_reviewer_required_and_disabled_gateway_modes(self):
        class Provider:
            def __init__(self):
                self.calls = []

            async def tools(self, messages, specs, **kwargs):
                self.calls.append(kwargs)
                await kwargs["direct_tool_handlers"]["read_file"]({"file_path": "report.md"})
                yield TextEvent(json.dumps({
                    "verdict": "done", "rationale": "Inspected the report",
                    "evidence": ["report.md confirms the outcome"],
                    "inspected_paths": ["report.md"], "findings": [],
                }))
                yield DoneEvent("endTurn")

        async def read_file(_arguments):
            return "The implemented report."

        config = load_routing_config()
        for mode in ("required", "disabled"):
            with self.subTest(mode=mode):
                models = {**config.models, "claude-sonnet": replace(
                    config.models["claude-sonnet"], gateway_mode=mode)}
                provider = Provider()
                with mock.patch("agent.routing.review.providers.get", return_value=provider):
                    decision = await ReviewService(replace(config, models=models)).review(
                        SimpleNamespace(title="Review", work_packages=[]), task(),
                        "Implemented `report.md`.", "codex-sol-medium",
                        [ToolSpec("read_file", "Read", {"type": "object", "properties": {
                            "file_path": {"type": "string"}}})], {"read_file": read_file})
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual("execution_env" in provider.calls[0], mode == "required")
                self.assertEqual(decision.gateway_used, mode == "required")

    async def test_reviewer_runs_with_read_only_role_and_structured_verdict(self):
        class FakeProvider:
            async def tools(self, messages, specs, **kwargs):
                self.role = kwargs["role"]
                self.reviewer_mode = kwargs["reviewer_mode"]
                self.reasoning_effort = kwargs["reasoning_effort"]
                self.run_metadata = kwargs["run_metadata"]
                await kwargs["direct_tool_handlers"]["read_file"]({"file_path": "report.md"})
                yield TextEvent('{"verdict":"done","rationale":"Outcome meets the task",'
                                '"evidence":["Inspected report.md"],'
                                '"inspected_paths":["report.md"],"findings":[],'
                                '"question":""}')
                yield DoneEvent("endTurn")

        provider = FakeProvider()
        change = SimpleNamespace(title="Routing change", work_packages=[])
        async def read_file(_arguments):
            return "The report explains the routing change."
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            decision = await ReviewService().review(
                change, task(), "Implemented `report.md`.", "claude-haiku",
                [ToolSpec("read_file", "Read a file", {"type": "object",
                 "properties": {"file_path": {"type": "string"}}})],
                {"read_file": read_file})
        self.assertEqual(provider.role, "optimizer")
        self.assertTrue(provider.reviewer_mode)
        self.assertEqual(provider.reasoning_effort, "medium")
        self.assertEqual(provider.run_metadata,
                         {"reviewer": True, "reviewer_model": "codex-luna-low",
                      "enforce_scope": True, "edit_files": [],
                          "gateway_used": False})
        self.assertEqual(decision.verdict, "done")
        self.assertEqual(decision.reviewer_model, "codex-luna-low")
        self.assertEqual(decision.evidence, ("Inspected report.md",))
        self.assertEqual(decision.inspected_paths, ("report.md",))

    async def test_reviewer_discovers_artifact_when_coder_supplies_no_path(self):
        class FakeProvider:
            async def tools(self, messages, specs, **kwargs):
                self.prompt = messages[0]["content"]
                found = await kwargs["direct_tool_handlers"]["search_file"]({
                    "q": "**/*factory*.py", "limit": 20,
                })
                self.asserted_search_result = found
                await kwargs["direct_tool_handlers"]["read_file"]({
                    "file_path": "trainer/factory.py",
                })
                yield TextEvent(json.dumps({
                    "verdict": "done",
                    "rationale": "Independent project discovery found the implementation",
                    "evidence": ["trainer/factory.py defines TrainerFactory"],
                    "inspected_paths": ["trainer/factory.py"],
                    "findings": [],
                }))
                yield DoneEvent("endTurn")

        async def search_file(_arguments):
            return {"items": [{"filePath": "trainer/factory.py"}]}

        async def read_file(arguments):
            self.assertEqual(arguments["file_path"], "trainer/factory.py")
            return "class TrainerFactory: pass"

        provider = FakeProvider()
        specs = [
            ToolSpec("read_file", "Read", {"type": "object", "properties": {
                "file_path": {"type": "string"}}}),
            ToolSpec("search_file", "Search", {"type": "object", "properties": {
                "q": {"type": "string"}, "limit": {"type": "integer"}}}),
        ]
        change = SimpleNamespace(title="Trainer factory", work_packages=[])
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            decision = await ReviewService().review(
                change, task(), "Implementation complete.", "claude-haiku", specs,
                {"read_file": read_file, "search_file": search_file})
        self.assertIn("initial artifact snapshot is empty", provider.prompt)
        self.assertEqual(decision.inspected_paths, ("trainer/factory.py",))

    async def test_missing_supplemental_basename_goes_to_reviewer(self):
        async def read_file(arguments):
            if arguments["file_path"] == "tests/test_other.py":
                return "def test_other(): pass"
            raise RuntimeError(f"File {arguments['file_path']} doesn't exist")

        async def search_file(_arguments):
            return {"items": []}

        provider = _RecordingReviewer("tests/test_other.py", {
            "verdict": "needs_work", "rationale": "The claimed test file was never created",
            "evidence": ["`test_missing.py` does not exist anywhere in the project"],
            "inspected_paths": ["tests/test_other.py"],
            "findings": [{"kind": "implementation", "problem": "Claimed test file is missing",
                          "suggested_task": "Create the claimed test file",
                          "approval_criteria": ["The claimed test file exists"],
                          "artifact_paths": ["test_missing.py"], "owner": "coder",
                          "reason": "implementation_defect"}]})
        specs = [ToolSpec("read_file", "Read", {"type": "object", "properties": {
                "file_path": {"type": "string"}}}),
                 ToolSpec("search_file", "Search", {"type": "object", "properties": {
                "q": {"type": "string"}, "limit": {"type": "integer"}}})]
        change = SimpleNamespace(title="Contract", work_packages=[])
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            decision = await ReviewService().review(
                change, task(), "Added `test_missing.py`.", "claude-haiku",
                specs, {"read_file": read_file, "search_file": search_file})
        self.assertEqual(decision.verdict, "needs_work")
        self.assertEqual(decision.unresolved_references[0]["status"], "not_found")
        self.assertIn("test_missing.py", provider.prompt)

    async def test_journal_path_is_never_remapped_by_project_search(self):
        search_file = mock.AsyncMock()

        async def read_file(_arguments):
            raise RuntimeError("File tests/test_contract.py doesn't exist")

        queued_task = replace(task(), context="\n".join((
            "Authoritative task mutation journal (all paths are required review evidence):",
            "Required mutation artifact: tests/test_contract.py",
        )))
        specs = [
            ToolSpec("read_file", "Read", {"type": "object", "properties": {
                "file_path": {"type": "string"}}}),
            ToolSpec("search_file", "Search", {"type": "object", "properties": {
                "q": {"type": "string"}, "limit": {"type": "integer"}}}),
        ]
        change = SimpleNamespace(title="Contract", work_packages=[])
        with self.assertRaisesRegex(ReviewError, "tests/test_contract.py"):
            await ReviewService().review(
                change, queued_task, "Changed `tests/test_contract.py`.",
                "claude-haiku", specs,
                {"read_file": read_file, "search_file": search_file})
        search_file.assert_not_awaited()

    async def test_cited_line_ranges_are_stripped_from_finding_paths(self):
        class FakeProvider:
            async def tools(self, messages, specs, **kwargs):
                yield TextEvent(json.dumps({
                    "verdict": "needs_work",
                    "rationale": "The factory omits the recurrent trainer",
                    "evidence": ["trainer_factory.py contains no recurrent mapping"],
                    "inspected_paths": ["src/trainer_factory.py:1-3"],
                    "findings": [{
                        "kind": "implementation",
                        "problem": "The recurrent trainer is not registered",
                        "suggested_task": "Add a recurrent trainer mapping",
                        "approval_criteria": ["The recurrent model type resolves to its trainer"],
                        "artifact_paths": ["src/trainer_factory.py:22-25",
                                           "src/trainer_factory.py#L1"],
                    }],
                    "question": "",
                }))
                yield DoneEvent("endTurn")

        native_mode.write("src/trainer_factory.py", "class TrainerFactory: pass\n")
        change = SimpleNamespace(title="Trainer factory", work_packages=[])
        with mock.patch("agent.routing.review.providers.get", return_value=FakeProvider()):
            decision = await ReviewService().review(
                change, task(), "Changed `src/trainer_factory.py`.", "claude-haiku", [], {})
        self.assertEqual(decision.findings[0].artifact_paths, ("src/trainer_factory.py",))

    async def test_protected_root_claim_is_never_read(self):
        read_file = mock.AsyncMock(return_value="secret")
        report = json.dumps({"schema_version": 1, "changed_files": [
            {"path": "/config/.env", "change": "modified"}]})
        with self.assertRaisesRegex(ReviewError, "unsupported project artifact path"):
            await ReviewService().review(
                SimpleNamespace(title="Secrets", work_packages=[]), task(),
                report, "claude-haiku",
                [ToolSpec("read_file", "Read a file", {"type": "object",
                 "properties": {"file_path": {"type": "string"}}})],
                {"read_file": read_file})
        read_file.assert_not_called()

    async def test_journaled_deletion_is_verified_as_absent(self):
        provider = _RecordingReviewer(verdict={
            "verdict": "done", "rationale": "Obsolete file is absent",
            "evidence": ["IDE confirmed the deleted file is absent"],
            "inspected_paths": ["obsolete.py"], "findings": [],
        })
        manifest = [{"path": "obsolete.py", "change": "deleted",
                     "before_exists": True, "after_exists": False,
                     "operations": ["delete"], "postimage_verified": True}]
        queued = replace(task(), context="\n".join((
            "Required mutation artifact: obsolete.py",
        )))
        object.__setattr__(queued, "_review_journal_manifest", manifest)
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            decision = await ReviewService().review(
                SimpleNamespace(title="Cleanup", work_packages=[]), queued,
                "Deleted `obsolete.py`.", "claude-haiku",
                [ToolSpec("read_file", "Read a file", {"type": "object",
                 "properties": {"file_path": {"type": "string"}}})],
                {"read_file": mock.AsyncMock(
                    return_value=json.dumps({"content": [{"type": "text",
                        "text": "File obsolete.py doesn't exist or can't be opened"}]}))})
        self.assertEqual(decision.verdict, "done")
        self.assertIn('"status": "verified_absent"', provider.prompt)

    async def test_reviewer_rejects_unexplained_verdict(self):
        class FakeProvider:
            async def tools(self, messages, specs, **kwargs):
                await kwargs["direct_tool_handlers"]["read_file"]({"file_path": "report.md"})
                yield TextEvent('{"verdict":"done","rationale":"Looks good",'
                                '"evidence":[],"gaps":[]}')
                yield DoneEvent("endTurn")

        change = SimpleNamespace(title="Routing change", work_packages=[])
        async def read_file(_arguments):
            return "Report content"
        with mock.patch("agent.routing.review.providers.get", return_value=FakeProvider()):
            with self.assertRaisesRegex(ReviewError, "evidence") as raised:
                await ReviewService().review(
                    change, task(), "Implemented `report.md`.", "claude-haiku",
                    [ToolSpec("read_file", "Read a file", {"type": "object",
                     "properties": {"file_path": {"type": "string"}}})],
                    {"read_file": read_file})
        self.assertEqual(raised.exception.reviewer_model, "codex-luna-low")

    async def test_reviewer_without_ide_inspection_fails_with_explanation(self):
        async def read_file(_arguments):
            return "Error: file not found"
        provider = _RecordingReviewer()
        change = SimpleNamespace(title="Review", work_packages=[])
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            with self.assertRaisesRegex(
                    ReviewError, r"did not successfully inspect[\s\S]*Unresolved references given "
                                 r"to the reviewer:\n- `report\.md` from [^\n]*: not_found "):
                await ReviewService().review(
                    change, task(), "Implemented `report.md`.", "claude-haiku",
                    [ToolSpec("read_file", "Read a file", {"type": "object", "properties": {"file_path": {"type": "string"}}})], {"read_file": read_file})
        self.assertIn("Unresolved references", provider.prompt)

class ReviewPersistenceMigrationTests(unittest.TestCase):
    def test_existing_review_table_gains_details_without_losing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.sqlite3"
            db = sqlite3.connect(path)
            try:
                db.execute("""CREATE TABLE implementation_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    position INTEGER NOT NULL, coder_model TEXT NOT NULL,
                    reviewer_model TEXT NOT NULL, verdict TEXT NOT NULL,
                    rationale TEXT NOT NULL, question TEXT, created_at REAL NOT NULL
                )""")
                db.execute("""INSERT INTO implementation_reviews
                    (job_id, position, coder_model, reviewer_model, verdict, rationale,
                     question, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                           ("old-job", 0, "claude-haiku", "codex-luna-low",
                            "done", "Legacy decision", None, 1.0))
                db.commit()
            finally:
                db.close()
            RoutingPersistence(path)
            db = sqlite3.connect(path)
            try:
                columns = {row[1] for row in db.execute(
                    "PRAGMA table_info(implementation_reviews)")}
                row = db.execute("SELECT rationale FROM implementation_reviews").fetchone()
            finally:
                db.close()
            self.assertIn("details_json", columns)
            self.assertEqual(row, ("Legacy decision",))

    def test_existing_exhausted_job_becomes_actionable_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.sqlite3"
            db = sqlite3.connect(path)
            try:
                db.executescript("""
                    CREATE TABLE implementation_jobs (
                      job_id TEXT PRIMARY KEY, selectors_json TEXT NOT NULL,
                      instruction TEXT NOT NULL DEFAULT '', conversation_id TEXT,
                      status TEXT NOT NULL, created_at REAL NOT NULL,
                      updated_at REAL NOT NULL, error TEXT
                    );
                    CREATE TABLE implementation_job_tasks (
                      job_id TEXT NOT NULL, position INTEGER NOT NULL,
                      change_id TEXT NOT NULL, task_id TEXT NOT NULL,
                      status TEXT NOT NULL, decision_id TEXT, error TEXT,
                      coder_report TEXT, review_question TEXT, review_started_at REAL,
                      PRIMARY KEY(job_id, position)
                    );
                    CREATE TABLE implementation_reviews (
                      id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                      position INTEGER NOT NULL, coder_model TEXT NOT NULL,
                      reviewer_model TEXT NOT NULL, verdict TEXT NOT NULL,
                      rationale TEXT NOT NULL, question TEXT, details_json TEXT,
                      created_at REAL NOT NULL
                    );
                """)
                db.execute("""INSERT INTO implementation_jobs VALUES
                  ('legacy', '[]', '', 'conv', 'review_pending', 1, 2, NULL)""")
                db.execute("""INSERT INTO implementation_job_tasks VALUES
                  ('legacy', 0, 'CHG1', 'WP2-T2', 'review_pending', NULL, NULL,
                   'report', NULL, NULL)""")
                db.executemany("""INSERT INTO implementation_reviews
                  (job_id, position, coder_model, reviewer_model, verdict, rationale,
                   question, details_json, created_at)
                  VALUES ('legacy', 0, 'claude-sonnet', 'codex-sol-medium',
                          'needs_work', 'gap', NULL, '{}', ?)""",
                               [(1,), (2,), (3,)])
                db.commit()
            finally:
                db.close()
            persistence = RoutingPersistence(path)
            self.assertEqual(asyncio.run(persistence.revision_state("legacy", 0)), (2, 3))
            pending = asyncio.run(persistence.pending_resolution("conv"))
            self.assertEqual(pending[:3], ("legacy", 0, "WP2-T2"))

    def test_active_legacy_job_counts_every_launched_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.sqlite3"
            db = sqlite3.connect(path)
            try:
                db.executescript("""
                    CREATE TABLE implementation_jobs (
                      job_id TEXT PRIMARY KEY, selectors_json TEXT NOT NULL,
                      instruction TEXT NOT NULL DEFAULT '', conversation_id TEXT,
                      status TEXT NOT NULL, created_at REAL NOT NULL,
                      updated_at REAL NOT NULL, error TEXT
                    );
                    CREATE TABLE implementation_job_tasks (
                      job_id TEXT NOT NULL, position INTEGER NOT NULL,
                      change_id TEXT NOT NULL, task_id TEXT NOT NULL,
                      status TEXT NOT NULL, decision_id TEXT, error TEXT,
                      coder_report TEXT, review_question TEXT, review_started_at REAL,
                      PRIMARY KEY(job_id, position)
                    );
                    CREATE TABLE implementation_reviews (
                      id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                      position INTEGER NOT NULL, coder_model TEXT NOT NULL,
                      reviewer_model TEXT NOT NULL, verdict TEXT NOT NULL,
                      rationale TEXT NOT NULL, question TEXT, details_json TEXT,
                      created_at REAL NOT NULL
                    );
                """)
                db.execute("""INSERT INTO implementation_jobs VALUES
                  ('active', '[]', '', 'conv', 'running', 1, 2, NULL)""")
                db.execute("""INSERT INTO implementation_job_tasks VALUES
                  ('active', 0, 'CHG1', 'WP2-T2', 'running', NULL, NULL,
                   'report', NULL, NULL)""")
                db.executemany("""INSERT INTO implementation_reviews
                  (job_id, position, coder_model, reviewer_model, verdict, rationale,
                   question, details_json, created_at)
                  VALUES ('active', 0, 'claude-sonnet', 'codex-sol-medium',
                          'needs_work', 'gap', NULL, '{}', ?)""", [(1,), (2,)])
                db.commit()
            finally:
                db.close()
            persistence = RoutingPersistence(path)
            self.assertEqual(
                asyncio.run(persistence.revision_state("active", 0)), (2, 3))


async def _value(value):
    return value


_PARSE_FAILURE = ("Claude Code returned an error result: "
                  "The model's tool call could not be parsed (retry also failed).")


class _InterruptedTaskExecution(_TaskExecution):
    """A configured number of runs fail mid-task; later runs succeed."""

    def __init__(self, failure=_PARSE_FAILURE, failures=1):
        super().__init__()
        self.failure = failure
        self.failures = failures
        self.excluded = []

    async def execute(self, queued_task, change_id, messages, specs, **kwargs):
        self.excluded.append(kwargs.get("exclude_models", frozenset()))
        if len(self.excluded) <= self.failures:
            self.started.append((queued_task.id, change_id))
            self.instructions.append(kwargs.get("execution_instruction"))
            yield TextEvent("Wrote part of the tests.")
            yield ErrorEvent(self.failure)
            yield DoneEvent("endTurn")
            return
        async for event in super().execute(queued_task, change_id, messages, specs, **kwargs):
            yield event


class CoderRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def _interrupted_job(self, directory, execution, verdicts=("done",)):
        persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

        async def actual_model(_decision_id):
            return ("codex-luna-low" if len(execution.excluded) > 1
                    else "claude-haiku")

        persistence.actual_model_for = actual_model
        artifacts = _Artifacts()
        service = ImplementationService(artifacts, execution, persistence, _Reviewer(list(verdicts)))
        job = await service.create_job(["WP1-T1"], conversation_id="conv-recovery")
        events = [event async for event in service.start(job, [], [])]
        return service, persistence, artifacts, job, events

    async def test_parse_failure_waits_for_user_instead_of_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            service, persistence, _, job, events = await self._interrupted_job(
                directory, _InterruptedTaskExecution(failures=2))
            text = "\n".join(event.text for event in events if isinstance(event, TextEvent))
            self.assertIn("Coding interrupted", text)
            self.assertIn("could not be parsed", text)
            self.assertIn("try coder again", text)
            self.assertIn(f"job {job.job_id[:8]}", text)
            item = await persistence.implementation_task(job.job_id, 0)
            self.assertEqual(item.status, "awaiting_coder_recovery")
            self.assertIsNotNone(await persistence.pending_resolution("conv-recovery"))

    async def test_try_again_excludes_the_failed_model_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = _InterruptedTaskExecution(failures=2)
            service, persistence, artifacts, job, _ = await self._interrupted_job(
                directory, execution)
            events = [event async for event in service.retry_coder(
                job.job_id, 0, [], [], "", {}, "")]
            self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))
            self.assertEqual(execution.excluded[-1], frozenset({"codex-luna-low"}))
            self.assertIn("interrupted before finishing", execution.instructions[-1])
            self.assertTrue(any("Approved" in event.text
                                for event in events if isinstance(event, TextEvent)))
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_user_decision")
            duplicate = [event async for event in service.retry_coder(
                job.job_id, 0, [], [], "", {}, "")]
            self.assertTrue(any(isinstance(event, ErrorEvent) for event in duplicate))

    async def test_exhausted_operational_failures_remain_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            _, persistence, _, job, events = await self._interrupted_job(
                directory, _InterruptedTaskExecution(
                    "all simple routing candidates failed: boom", failures=2))
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_coder_recovery")
            self.assertTrue(any("try coder again" in event.text
                                for event in events if isinstance(event, TextEvent)))

    async def test_failed_tool_result_resumes_with_another_coder(self):
        class FailedResume(_TaskExecution):
            def __init__(self):
                super().__init__()
                self.resume_calls = 0

            async def resume(self, run, messages, specs):
                self.resume_calls += 1
                yield TextEvent("Earlier coder output.")
                yield ErrorEvent("connection reset after IDE result")
                yield DoneEvent("failed")

        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            execution = FailedResume()

            async def actual_model(_decision_id):
                return "codex-luna-low" if execution.started else "claude-haiku"

            persistence.actual_model_for = actual_model
            service = ImplementationService(
                _Artifacts(), execution, persistence, _Reviewer(["done"]))
            job = await service.create_job(["WP1-T1"], conversation_id="conv")
            await persistence.update_implementation_task(job.job_id, 0, "waiting")
            run = Run(metadata={
                "implementation_job_id": job.job_id,
                "implementation_position": 0,
                "implementation_task_id": "WP1-T1",
                "implementation_instruction": "Finish the feature",
            })
            events = [event async for event in service.resume(run, [], [])]
            self.assertEqual(execution.resume_calls, 1)
            self.assertEqual(execution.started, [("WP1-T1", "CHG1")])
            self.assertIn("Inspect the current project", execution.instructions[-1])
            self.assertTrue(any("continuing WP1-T1" in event.text
                                for event in events if isinstance(event, TextEvent)))
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "awaiting_user_decision")

    async def test_routing_excludes_models_and_rejects_exhausted_tier(self):
        config = load_routing_config()
        candidates = config.candidates_for("medium")
        service = ModelRoutingService(config, Router(candidates[0]))
        decision = await service.route(task(), "CHG1", "/work/project",
                                       exclude=frozenset({candidates[0]}))
        if len(candidates) > 1:
            self.assertNotEqual(decision.recommended_model, candidates[0])
            self.assertNotIn(candidates[0], decision.candidates)
        with self.assertRaises(RoutingConfigError):
            await service.route(task(), "CHG1", "/work/project",
                                exclude=frozenset(candidates))


class ClaudeProviderErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_error_result_text_replaces_sdk_success_message(self):
        from claude_agent_sdk import ResultMessage
        from providers import claude

        async def fake_query(prompt, options):
            yield ResultMessage(
                subtype="success", duration_ms=1, duration_api_ms=1, is_error=True,
                num_turns=3, session_id="s",
                result="The model's tool call could not be parsed (retry also failed).")
            raise Exception("Claude Code returned an error result: success")

        with mock.patch.object(claude, "query", fake_query):
            with self.assertRaisesRegex(claude.ClaudeResultError, "could not be parsed"):
                async for _ in claude._query_messages("p", None):
                    pass

    async def test_unrelated_sdk_errors_are_not_rewritten(self):
        from providers import claude

        async def fake_query(prompt, options):
            raise RuntimeError("transport closed")
            yield

        with mock.patch.object(claude, "query", fake_query):
            with self.assertRaisesRegex(RuntimeError, "transport closed"):
                async for _ in claude._query_messages("p", None):
                    pass

    async def test_cli_receives_tool_timeout_in_milliseconds(self):
        from providers import claude
        from mcp_bridge.bridge import TOOL_TIMEOUT
        seen = {}

        async def fake_query(prompt, options):
            seen["env"] = options.env
            return
            yield

        with mock.patch.object(claude, "query", fake_query):
            await claude.ClaudeProvider()._run_strategy(
                SimpleNamespace(), "p", None, [], execution_env={"EXTRA": "1"})
        self.assertEqual(seen["env"]["MCP_TOOL_TIMEOUT"], str(TOOL_TIMEOUT * 1000))
        self.assertEqual(seen["env"]["EXTRA"], "1")


class UnresolvedReferenceTests(unittest.IsolatedAsyncioTestCase):
    specs = [
        ToolSpec("read_file", "Read", {"type": "object", "properties": {
            "file_path": {"type": "string"}}}),
        ToolSpec("search_file", "Search", {"type": "object", "properties": {
            "q": {"type": "string"}, "limit": {"type": "integer"}}}),
    ]

    @staticmethod
    def handlers(files):
        async def read_file(arguments):
            if arguments["file_path"] not in files:
                raise RuntimeError(f"File {arguments['file_path']} doesn't exist")
            return files[arguments["file_path"]]

        async def search_file(arguments):
            name = arguments["q"].rsplit("/", 1)[-1]
            return {"items": [{"filePath": path} for path in files
                              if path.rsplit("/", 1)[-1] == name]}
        return {"read_file": read_file, "search_file": search_file}

    async def test_unreadable_journaled_file_still_stops_with_named_path(self):
        class FakeProvider:
            def tools(self, *args, **kwargs):
                raise AssertionError("journal evidence failures must not reach the reviewer")

        queued_task = replace(task(), context="Required mutation artifact: src/__init__.py")
        with mock.patch("agent.routing.review.providers.get", return_value=FakeProvider()):
            with self.assertRaisesRegex(ReviewError, r"change journal.*`src/__init__.py`"):
                await ReviewService().review(
                    SimpleNamespace(title="Journal", work_packages=[]), queued_task,
                    "Changed the package init.", "claude-haiku", self.specs,
                    self.handlers({}))


class RequiredTestRunTests(unittest.IsolatedAsyncioTestCase):
    specs = [ToolSpec("read_file", "Read", {"type": "object", "properties": {
        "file_path": {"type": "string"}}})]
    passing = "collected 3 items\n\n3 passed in 0.42s\nexit code 0"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = mock.patch.dict("os.environ", {"CODEPARTNER_EVIDENCE_DIR": self.directory.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.persistence = RoutingPersistence(Path(self.directory.name) / "routing.sqlite3")
        self.calls = []

    def journal(self, project_files=("uv.lock",)):
        async def execute_tool(arguments):
            self.calls.append(arguments["command"])
            return self.passing

        async def read_file(arguments):
            if arguments["file_path"] in project_files:
                return "1: content"
            raise RuntimeError(f"File {arguments['file_path']} doesn't exist")

        handlers = {"execute_tool": execute_tool, "read_file": read_file}
        return RollbackJournal(self.persistence, "job", 0, self.specs, handlers), handlers

    async def test_malformed_dispatcher_calls_are_rewritten_and_audited(self):
        journal, _ = self.journal()
        wrapped = journal.wrapped_handlers()
        for command in ("pytest tests/test_a.py -v",
                        "execute_terminal_command pytest tests/test_a.py -v"):
            await wrapped["execute_tool"]({"command": command})
        self.assertTrue(all(call.startswith(
            "execute_terminal_command --command 'pytest tests/test_a.py -v'")
            for call in self.calls))
        evidence = await self.persistence.verification_evidence("job", 0)
        self.assertEqual([item["metadata"]["normalized_from"] for item in evidence],
                         ["pytest tests/test_a.py -v",
                          "execute_terminal_command pytest tests/test_a.py -v"])
        self.assertTrue(evidence[0]["success"])

    async def test_unsafe_command_is_passed_through_unchanged(self):
        journal, _ = self.journal()
        await journal.wrapped_handlers()["execute_tool"]({"command": "pytest x && rm -rf y"})
        self.assertEqual(self.calls, ["pytest x && rm -rf y"])

    def test_blocker_message_shows_the_command_and_next_step(self):
        never_started = ImplementationService._test_blocker_lines([{
            "arguments": {"command": "execute_terminal_command pytest tests/test_a.py"},
            "metadata": {"pytest": {"totals": {}}},
            "result_excerpt": "Invalid argument format: 'pytest'"}])
        self.assertIn("never started pytest", never_started[0])
        self.assertIn("Invalid argument format", never_started[1])
        self.assertIn("retry review", never_started[2])
        failed = ImplementationService._test_blocker_lines([{
            "arguments": {"command": "uv run pytest"},
            "metadata": {"pytest": {"totals": {"passed": 2, "failed": 1}}},
            "result_excerpt": ""}])
        self.assertIn("2 passed, 1 failed", failed[0])

    def test_unrelated_review_failure_does_not_mention_unresolved_references(self):
        from agent.routing.review import _unresolved_summary
        unresolved = [{"path": "__init__.py", "sources": ["coder_report"], "status": "ambiguous",
                       "matches": ["a/__init__.py", "b/__init__.py"], "detail": ""}]
        unrelated = ReviewError("required unit-test validation is unverified", findings=(
            ReviewFinding("verification", "Tests not run", "Run them", ("Tests pass",),
                          ("tests/test_a.py",)),))
        self.assertEqual(_unresolved_summary(unresolved, unrelated), "")
        inspection = _unresolved_summary(
            unresolved, ReviewError("reviewer did not successfully inspect project content"))
        self.assertIn("\n- `__init__.py` from coder_report: ambiguous", inspection)


class ReviewerFormatRepairTests(unittest.IsolatedAsyncioTestCase):
    specs = [ToolSpec("read_file", "Read", {"type": "object", "properties": {
        "file_path": {"type": "string"}}})]

    class Provider:
        def __init__(self, replies):
            self.replies = list(replies)
            self.conversations = []

        async def tools(self, messages, specs, **kwargs):
            self.conversations.append(messages)
            await kwargs["direct_tool_handlers"]["read_file"]({"file_path": "data/factory.py"})
            yield TextEvent(self.replies.pop(0))
            yield DoneEvent("endTurn")

    @staticmethod
    async def read_file(_arguments):
        return "class Factory: pass"

    @staticmethod
    def decision(**overrides):
        value = {"verdict": "done", "rationale": "Factory and tests delivered",
                 "evidence": ["25 tests passed in the captured run"],
                 "inspected_paths": ["data/factory.py"], "findings": [],
                 "validation_results": []}
        value.update(overrides)
        return json.dumps(value)

    def wp_task(self):
        return replace(task(), id="WP1", context="\n".join((
            "BEGIN WORK PACKAGE REVIEW CONTRACT",
            'Required validation: ["All DataLoaderFactory unit tests pass."]',
            "END WORK PACKAGE REVIEW CONTRACT")))

    async def review(self, provider, queued_task=None):
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            return await ReviewService().review(
                SimpleNamespace(title="Loaders", work_packages=[]), queued_task or task(),
                "Changed `data/factory.py`.", "claude-haiku", self.specs,
                {"read_file": self.read_file})

    async def test_reply_without_json_gets_one_correction_request(self):
        provider = self.Provider(["Looks good to me!", self.decision()])
        decision = await self.review(provider)
        self.assertEqual(decision.verdict, "done")
        self.assertEqual(len(provider.conversations), 2)
        self.assertIn("CodePartner rejected your JSON decision: reviewer returned no valid "
                      "JSON decision", provider.conversations[1][-1]["content"])
        self.assertEqual(provider.conversations[1][-2],
                         {"role": "assistant", "content": "Looks good to me!"})

    async def test_status_synonyms_and_paraphrased_expectations_are_accepted(self):
        provider = self.Provider([self.decision(validation_results=[{
            "expectation": "all DataLoaderFactory unit tests pass",
            "category": "Required", "status": "passed", "evidence": "25 passed"}])])
        decision = await self.review(provider, self.wp_task())
        self.assertEqual(decision.verdict, "done")
        self.assertEqual(len(provider.conversations), 1)
        self.assertEqual(decision.validation_results[0]["status"], "satisfied")
        self.assertEqual(decision.validation_results[0]["evidence"], ["25 passed"])

    async def test_still_malformed_reply_names_the_exact_field(self):
        bad = self.decision(validation_results=[{
            "expectation": "All DataLoaderFactory unit tests pass.",
            "category": "required", "status": "green", "evidence": []}])
        provider = self.Provider([bad, bad, bad, bad])
        with self.assertRaises(ReviewFormatError) as raised:
            await self.review(provider, self.wp_task())
        message = str(raised.exception)
        self.assertIn("validation_results[0] status='green'", message)
        self.assertIn("still malformed after one correction request", message)
        self.assertEqual(len(provider.conversations), 4)

    async def test_malformed_primary_reviewer_uses_other_provider(self):
        bad = self.decision(validation_results=[{
            "expectation": "All DataLoaderFactory unit tests pass.",
            "category": "required", "status": "green", "evidence": []}])
        provider = self.Provider([bad, bad, self.decision(validation_results=[{
            "expectation": "All DataLoaderFactory unit tests pass.",
            "category": "required", "status": "satisfied", "evidence": ["25 passed"]}])])
        decision = await self.review(provider, self.wp_task())
        self.assertEqual(decision.reviewer_model, "claude-haiku")
        self.assertEqual(len(provider.conversations), 3)
        self.assertEqual(
            [(entry["model"], entry["status"]) for entry in decision.routing_attempts],
            [("codex-luna-low", "failed"), ("claude-haiku", "completed")])
        self.assertIn("malformed", decision.routing_attempts[0]["error"])


if __name__ == "__main__":
    unittest.main()

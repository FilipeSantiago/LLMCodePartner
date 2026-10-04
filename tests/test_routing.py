"""Dependency-free unit tests for OpenSpec task routing and audit persistence."""

import asyncio
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.routing.config import RoutingConfig, RoutingConfigError, load_routing_config
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.implementation import ImplementationService
from agent.routing.models import ModelRef
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import ReviewDecision, ReviewError, ReviewService
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
from agent.spec.models import Task, OpenSpecChange
from mcp_bridge.models import DoneEvent, ErrorEvent, Run, TextEvent
from mcp_bridge.registry import ToolSpec


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
    def test_execution_replaces_run_transport_command_with_task_content(self):
        messages = TaskExecutionService._execution_messages(
            task(), [{"role": "system", "content": "system"}, {"role": "user", "content": "/run WP1-T1"}],
            "Keep the tracking URI injectable",
        )
        self.assertEqual(messages[0]["content"], "system")
        self.assertIn("Implement OpenSpec task WP1-T1", messages[-1]["content"])
        self.assertIn("Keep the tracking URI injectable", messages[-1]["content"])
        self.assertNotIn("/run", messages[-1]["content"])

    async def test_pre_output_executor_failure_tries_same_tier_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_routing_config()
            router = ModelRoutingService(config, Router(config.candidates_for("simple")[0]))
            registry = ExecutorRegistry({
                "claude-cli": _FailingExecutor(),
                "codex-cli": _SuccessfulExecutor(),
            })
            service = TaskExecutionService(
                config=config, router=router,
                persistence=RoutingPersistence(Path(directory) / "routing.sqlite3"),
                executors=registry,
            )
            events = [event async for event in service.execute(
                task("simple"), "CHG1", [{"role": "user", "content": "/run WP1-T1"}], []
            )]
            self.assertEqual([event.text for event in events if isinstance(event, TextEvent)],
                             ["completed by fallback"])


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

    async def mark_task_completed(self, task_id):
        self.completed.append(task_id)


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
    async def test_review_can_complete_without_a_mutation_event(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["done", "done"])
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1", "WP1-T2"], "Add traceability")
            events = [event async for event in service.start(
                job, [{"role": "user", "content": "/implement WP1"}], []
            )]
            self.assertEqual(execution.started, [("WP1-T1", "CHG1"), ("WP1-T2", "CHG1")])
            self.assertEqual(execution.instructions, ["Add traceability"] * 2)
            self.assertEqual(artifacts.completed, ["WP1-T1", "WP1-T2"])
            self.assertEqual(reviewer.reports, ["Task result with supporting evidence."] * 2)
            self.assertTrue(any("completed after review" in event.text
                                for event in events if isinstance(event, TextEvent)))
            db = sqlite3.connect(persistence.path)
            try:
                row = db.execute(
                    "SELECT status, instruction FROM implementation_jobs WHERE job_id=?", (job.job_id,)
                ).fetchone()
            finally:
                db.close()
            self.assertEqual(row, ("completed", "Add traceability"))

    async def test_research_task_waits_for_user_confirmation(self):
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
            self.assertEqual((await persistence.pending_confirmation("conv-test"))[0], job.job_id)
            events = [event async for event in service.confirm(job.job_id, 0, True, "yes", [], [])]
            self.assertEqual(artifacts.completed, ["WP1-T1"])
            self.assertTrue(any("completed" in event.text for event in events if isinstance(event, TextEvent)))

    async def test_unrelated_write_does_not_override_review_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _Artifacts()
            execution = _TaskExecution()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            async def actual_model(_decision_id):
                return "claude-haiku"
            persistence.actual_model_for = actual_model
            reviewer = _Reviewer(["needs_work"] * 3)
            service = ImplementationService(artifacts, execution, persistence, reviewer)
            job = await service.create_job(["WP1-T1"])
            [event async for event in service.start(job, [], [])]
            self.assertEqual(artifacts.completed, [])
            self.assertEqual(len(execution.started), 3)
            self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                             "review_pending")


class _Reviewer:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.reports = []

    async def review(self, change, queued_task, report, coder_model, specs, handlers):
        self.reports.append(report)
        verdict = self.verdicts.pop(0)
        return ReviewDecision(verdict, "Reviewed task result", "codex-luna-low",
                              f"Do you consider {queued_task.id} complete?" if verdict == "ask_user" else None,
                              "Requested outcome missing" if verdict == "needs_work" else None)


class _SelectingReviewer(ReviewService):
    def __init__(self, config, unavailable=False, model_unavailable=False):
        super().__init__(config)
        self.selected = []
        self.unavailable = unavailable
        self.model_unavailable = model_unavailable

    async def _run_review(self, change, queued_task, report, reviewer_name, specs, handlers):
        self.selected.append(reviewer_name)
        if self.unavailable and len(self.selected) == 1:
            raise ReviewError("authentication required")
        if self.model_unavailable and len(self.selected) == 1:
            raise ReviewError("model not available")
        return ReviewDecision("done", "Task outcome reviewed", reviewer_name)


class ReviewerSelectionTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_unavailable_model_tries_other_opposite_model_in_same_tier(self):
        reviewer = _SelectingReviewer(load_routing_config(), model_unavailable=True)
        await reviewer.review(None, task(), "result", "claude-sonnet", [], {})
        self.assertEqual(reviewer.selected, ["codex-sol-medium", "codex-luna-medium"])

    async def test_reviewer_runs_with_read_only_role_and_structured_verdict(self):
        class FakeProvider:
            async def tools(self, messages, specs, **kwargs):
                self.role = kwargs["role"]
                yield TextEvent('{"verdict":"done","rationale":"Outcome meets the task",'
                                '"feedback":"","question":""}')
                yield DoneEvent("endTurn")

        provider = FakeProvider()
        change = SimpleNamespace(title="Routing change", work_packages=[])
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            decision = await ReviewService().review(
                change, task(), "Implemented routing", "claude-haiku", [], {})
        self.assertEqual(provider.role, "optimizer")
        self.assertEqual(decision.verdict, "done")
        self.assertEqual(decision.reviewer_model, "codex-luna-low")


async def _value(value):
    return value


if __name__ == "__main__":
    unittest.main()

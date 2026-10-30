"""Context-aware workflow interpretation and the universal confirmation boundary."""

import json
import re
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.routing.implementation import ImplementationService
from agent.routing.persistence import RoutingPersistence
from agent.spec.models import Task
from agent.task_contracts.persistence import TaskContractPersistence
from agent.workflow.controller import ConversationPhase, WorkflowController, parse_decision
from agent.workflow.interpreter import WorkflowInterpreter
from agent.workflow.states import WorkflowAction as A, action_allowed
from api import dialog_controller
from mcp_bridge.models import DoneEvent, TextEvent
from mcp_bridge.registry import ToolSpec
from model.chat import ChatCompletionRequest, ChatMessage

ENV_REPLY = "Ignore the .env. I'll fill it after the implementation."
ENV_GUIDANCE = "Do not inspect or modify .env. Continue using .env.example."


def _task(task_id: str) -> Task:
    return Task(
        id=task_id, title=f"Configure {task_id}",
        description="Document required settings in .env.example.",
        complexity="simple", reasoning="Localized.",
        context="Settings are documented in .env.example.",
        preferred_capability="write", acceptance_criteria=["Settings are documented."],
    )


def _specs():
    return [
        ToolSpec("read_file", "read", {"type": "object", "properties": {
            "file_path": {"type": "string"}}, "required": ["file_path"]}),
        ToolSpec("apply_patch", "patch", {"type": "object", "properties": {
            "input": {"type": "string"}}, "required": ["input"]}),
    ]


class _Artifacts:
    def __init__(self):
        self.change = SimpleNamespace(id="CHG1")
        self.tasks = {"WP4-T1": _task("WP4-T1"), "WP5-T1": _task("WP5-T1")}

    async def tasks_for_selectors(self, selectors, include_completed=False):
        found = []
        for selector in selectors:
            for task_id in {"WP4": ["WP4-T1"], "WP5": ["WP5-T1"]}.get(selector, [selector]):
                found.append((self.change, self.tasks[task_id]))
        return found

    async def task_by_id(self, task_id):
        return self.change, self.tasks[task_id]


class _Execution:
    """A Coder stand-in that records every contract it is asked to execute."""

    def __init__(self, persistence):
        self.persistence = persistence
        self.contracts = []

    async def execute(self, task, change_id, messages, specs, contract=None,
                      execution_guard=None, **kwargs):
        self.contracts.append(contract)
        yield TextEvent("no structured handoff")
        yield DoneEvent("endTurn")


class _Provider:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    async def emit_json(self, messages, schema, model=None):
        self.calls.append((messages, schema))
        if self.error:
            raise self.error
        return self.value


def _retry_contract_reply(guidance=ENV_GUIDANCE, **overrides):
    return {"action": "RETRY_CONTRACT", "target_task": "WP4-T1", "guidance": guidance,
            "scope_paths": [], "recovery_mode": "none", "approved": False,
            "confidence": "high", "ambiguities": [], **overrides}


class _Fixture:
    def __init__(self, directory: str, provider: _Provider | None = None):
        self.persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
        self.contracts = TaskContractPersistence(self.persistence.path)
        self.artifacts = _Artifacts()
        self.execution = _Execution(self.persistence)
        self.service = ImplementationService(
            self.artifacts, self.execution, self.persistence,
            reviewer=SimpleNamespace(), contract_persistence=self.contracts)
        self.provider = provider or _Provider(error=AssertionError("model not expected"))
        self.workflow = WorkflowController(
            self.persistence, self.contracts,
            interpreter=WorkflowInterpreter(self.provider, "test-model"))
        # Direct-flow fixture: WP planning is covered in test_wp_planner.
        self.workflow._planning_available = False

    async def paused(self, selector="WP4", conversation="conv",
                     status="awaiting_contract_resolution",
                     error="path outside contract scope: .env"):
        job = await self.service.create_job([selector], conversation_id=conversation)
        task = self.artifacts.tasks[job.task_ids[0]]
        contract = self.service.contractor.initial_contract(task, _specs())
        await self.contracts.save(job.job_id, 0, contract)
        if status == "awaiting_contract_resolution":
            await self.service._record_contract_escalation(
                job.job_id, 0, contract, error, "execution_policy")
        await self.persistence.update_implementation_task(job.job_id, 0, status, error=error)
        return job, contract

    async def needs_work_review(self, job_id):
        await self.persistence.record_review(
            job_id, 0, "claude-sonnet", "codex-sol-medium", "needs_work", "gap", None,
            {"findings": [{"kind": "implementation", "problem": "missing coverage",
                           "approval_criteria": ["covered"], "artifact_paths": []}]})

    def rows(self, sql, *args):
        with sqlite3.connect(self.persistence.path) as db:
            return db.execute(sql, args).fetchall()


class WorkflowInterpretationTests(unittest.IsolatedAsyncioTestCase):
    async def test_natural_language_maps_to_retry_contract_from_bounded_context(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            job, _ = await fixture.paused()
            context = await fixture.workflow.context("conv")
            intent = await fixture.workflow.interpret(ENV_REPLY, context)

            self.assertEqual(intent.action, A.RETRY_CONTRACT)
            self.assertEqual(intent.target_task, "WP4-T1")
            self.assertEqual(intent.target_job, job.job_id)
            self.assertEqual(intent.guidance, ENV_GUIDANCE)
            self.assertEqual(intent.interpretation_source, "model")
            messages, schema = fixture.provider.calls[0]
            prompt = messages[1]["content"]
            self.assertIn("State: awaiting_contract_resolution", prompt)
            self.assertIn("Files allowed: .env.example", prompt)
            self.assertIn("Files requested: .env", prompt)
            self.assertIn("Review completed: false", prompt)
            self.assertEqual(len(messages), 2)  # no conversation history is sent
            self.assertEqual(set(schema["properties"]["action"]["enum"]), {
                "RETRY_CONTRACT", "EXPAND_CONTRACT_SCOPE", "STOP_JOB", "ANSWER",
                "ASK_CLARIFICATION"})

    async def test_proposal_preview_runs_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            job, _ = await fixture.paused()
            intent = await fixture.workflow.interpret(
                ENV_REPLY, await fixture.workflow.context("conv"))
            proposal = await fixture.workflow.propose(intent)
            text = proposal.render()

            self.assertIn("Action: Retry WP4-T1 with a revised TaskContract", text)
            self.assertIn("Keep `.env` outside the contract", text)
            self.assertIn("Files in the contract: `.env.example`", text)
            self.assertIn("Nothing has executed yet.", text)
            self.assertIn(f"confirm {proposal.confirmation_token}", text)
            self.assertEqual(fixture.execution.contracts, [])
            self.assertEqual(await fixture.contracts.count(job.job_id, 0), 1)
            self.assertEqual((await fixture.persistence.implementation_task(
                job.job_id, 0)).status, "awaiting_contract_resolution")

    async def test_low_confidence_becomes_clarification_not_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply(
                confidence="low", ambiguities=["Unclear whether to retry"])))
            await fixture.paused()
            intent = await fixture.workflow.interpret(
                "hmm", await fixture.workflow.context("conv"))
            self.assertEqual(intent.action, A.ASK_CLARIFICATION)
            self.assertIsNone(await fixture.workflow.propose(intent))
            self.assertIn("Unclear whether to retry", intent.clarification)
            self.assertEqual(await fixture.persistence.pending_workflow_proposals("conv"), [])

    async def test_review_failure_still_routes_to_review_specific_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            job, _ = await fixture.paused(status="awaiting_review_resolution", error="gap")
            await fixture.needs_work_review(job.job_id)
            context = await fixture.workflow.context("conv")
            intent = await fixture.workflow.interpret(
                "submit to coder again: add coverage", context)
            self.assertEqual(intent.action, A.RETRY_CODER_AFTER_REVIEW)
            self.assertNotIn(A.RETRY_CONTRACT, context.paused[0].actions)

            proposal = await fixture.workflow.propose(intent)
            row = await fixture.workflow.find("conv", proposal.confirmation_token)
            result = await fixture.workflow.claim(row)
            service = mock.Mock()

            async def retry_coder(*args):
                yield TextEvent("retrying after review")
            service.retry_coder = mock.Mock(side_effect=retry_coder)
            service.retry_contract = mock.Mock()
            events = [event async for event in fixture.workflow.execute(
                result.proposal, service, [], _specs(), {})]
            service.retry_coder.assert_called_once()
            self.assertEqual(service.retry_coder.call_args.args[6], "add coverage")
            service.retry_contract.assert_not_called()
            self.assertEqual(events[0].text, "retrying after review\n")

    async def test_review_anyway_sends_a_blocked_handoff_to_the_reviewer(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            job, _ = await fixture.paused(
                status="awaiting_coder_recovery",
                error="Coder handoff blocked: acceptance criteria or remaining gaps "
                      "are unresolved")
            intent = await fixture.workflow.interpret(
                "review anyway", await fixture.workflow.context("conv"))
            self.assertEqual(intent.action, A.RESOLVE_CODER_RECOVERY)
            self.assertEqual(intent.parameters["recovery_mode"], "review_handoff")
            self.assertEqual(fixture.provider.calls, [])

            proposal = await fixture.workflow.propose(intent)
            row = await fixture.workflow.find("conv", proposal.confirmation_token)
            result = await fixture.workflow.claim(row)
            service = mock.Mock()

            async def review_blocked_handoff(*args):
                yield TextEvent("reviewing")
            service.review_blocked_handoff = mock.Mock(side_effect=review_blocked_handoff)
            service.retry_coder = mock.Mock()
            events = [event async for event in fixture.workflow.execute(
                result.proposal, service, [], _specs(), {})]
            service.review_blocked_handoff.assert_called_once()
            service.retry_coder.assert_not_called()
            self.assertEqual(events[0].text, "reviewing\n")

    def test_review_handoff_mode_is_only_valid_during_coder_recovery(self):
        self.assertTrue(action_allowed(
            "awaiting_coder_recovery", "RESOLVE_CODER_RECOVERY", "review_handoff"))
        self.assertFalse(action_allowed(
            "recovery_required", "RESOLVE_CODER_RECOVERY", "review_handoff"))

    async def test_same_phrase_means_contract_retry_in_contract_state(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            await fixture.paused()
            intent = await fixture.workflow.interpret(
                "submit to coder again: use only .env.example",
                await fixture.workflow.context("conv"))
            self.assertEqual(intent.action, A.RETRY_CONTRACT)
            self.assertEqual(intent.guidance, "use only .env.example")
            self.assertEqual(fixture.provider.calls, [])

    async def test_multiple_paused_workflows_require_clarification(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            await fixture.paused("WP4")
            other, _ = await fixture.paused(
                "WP5", status="awaiting_review_resolution", error="gap")
            await fixture.needs_work_review(other.job_id)
            context = await fixture.workflow.context("conv")
            self.assertEqual(len(context.paused), 2)

            for message in ("try again: use the example file", "please just redo it"):
                intent = await fixture.workflow.interpret(message, context)
                self.assertEqual(intent.action, A.ASK_CLARIFICATION, message)
                self.assertIn("`WP4-T1`", intent.clarification)
                self.assertIn("`WP5-T1`", intent.clarification)
            self.assertEqual(fixture.provider.calls, [])

            named = await fixture.workflow.interpret(
                "WP4-T1: try again: use the example file", context)
            self.assertEqual(named.action, A.RETRY_CONTRACT)
            self.assertEqual(named.target_task, "WP4-T1")

    async def test_invalid_action_for_state_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            await fixture.paused()
            intent = await fixture.workflow.interpret(
                "accept anyway", await fixture.workflow.context("conv"))
            self.assertEqual(intent.action, A.ASK_CLARIFICATION)
            self.assertIn("not a valid action", intent.clarification)

    async def test_user_supplied_answer_becomes_contract_guidance(self):
        fact = "The user confirms there is no MLflow in this project."
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply(guidance=fact)))
            await fixture.paused(error="path outside contract scope: main.py")
            intent = await fixture.workflow.interpret(
                "I'll answer this task for you, there is no MLFLOW in this project",
                await fixture.workflow.context("conv"))
            self.assertEqual(intent.action, A.RETRY_CONTRACT)
            self.assertEqual(intent.guidance, fact)
            system = fixture.provider.calls[0][0][0]["content"]
            self.assertIn("supplies facts or the answer itself", system)

    async def test_failed_model_call_is_diagnosable(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(error=TimeoutError()))
            await fixture.paused()
            intent = await fixture.workflow.interpret(
                "there is no MLFLOW in this project", await fixture.workflow.context("conv"))
            self.assertEqual(intent.action, A.ASK_CLARIFICATION)
            self.assertIn("retry contract: <guidance>", intent.clarification)
            self.assertEqual(fixture.rows(
                "SELECT detail FROM telemetry_workflow_events WHERE stage='interpreted'"),
                [("TimeoutError",)])

    def test_explicit_decision_tokens(self):
        self.assertEqual(parse_decision("confirm 7K4MAB").verb, "confirm")
        self.assertEqual(parse_decision("reject 7k4mab").token, "7K4MAB")
        change = parse_decision("change 7K4MAB: also add tests")
        self.assertEqual((change.verb, change.correction), ("change", "also add tests"))
        self.assertIsNone(parse_decision("change 7K4MAB"))
        self.assertIsNone(parse_decision("confirm"))


class WorkflowConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def _proposal(self, fixture):
        intent = await fixture.workflow.interpret(
            ENV_REPLY, await fixture.workflow.context("conv"))
        proposal = await fixture.workflow.propose(intent)
        return await fixture.workflow.find("conv", proposal.confirmation_token)

    async def test_confirmation_creates_revision_two_without_review(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            job, first = await fixture.paused()
            row = await self._proposal(fixture)
            result = await fixture.workflow.claim(row)
            self.assertTrue(result.claimed)
            [event async for event in fixture.workflow.execute(
                result.proposal, fixture.service, [], _specs(), {})]

            self.assertEqual(len(fixture.execution.contracts), 1)
            revised = fixture.execution.contracts[0]
            self.assertEqual(revised.revision, 2)
            self.assertEqual(revised.parent_contract_id, first.contract_id)
            self.assertEqual(revised.primary_files, (".env.example",))
            fact = revised.known_context[-1]
            self.assertEqual(fact.source, "user.contract_resolution")
            self.assertIn(ENV_GUIDANCE, fact.fact)
            self.assertIn(".env", fact.attributes["excluded_files"])
            self.assertEqual(fact.attributes["scope_decision"], "preserve")
            self.assertEqual(await fixture.persistence.review_count(job.job_id, 0), 0)
            actions = fixture.rows(
                "SELECT action, details_json FROM implementation_resolution_actions "
                "WHERE job_id=?", job.job_id)
            self.assertEqual(actions[0][0], "retry_contract")
            self.assertEqual(json.loads(actions[0][1])["scope_decision"], "preserve")

    async def test_expand_scope_adds_confirmed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            await fixture.paused()
            intent = await fixture.workflow.interpret(
                "expand scope: config/settings.py", await fixture.workflow.context("conv"))
            self.assertEqual(intent.action, A.EXPAND_CONTRACT_SCOPE)
            proposal = await fixture.workflow.propose(intent)
            row = await fixture.workflow.find("conv", proposal.confirmation_token)
            result = await fixture.workflow.claim(row)
            [event async for event in fixture.workflow.execute(
                result.proposal, fixture.service, [], _specs(), {})]
            revised = fixture.execution.contracts[0]
            self.assertEqual(revised.primary_files, (".env.example", "config/settings.py"))
            self.assertEqual(revised.known_context[-1].attributes["scope_decision"], "expand")

    async def test_rejection_executes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            job, _ = await fixture.paused()
            row = await self._proposal(fixture)
            self.assertIn("rejected", await fixture.workflow.reject(row))
            row = await fixture.workflow.find("conv", row["confirmation_token"])
            result = await fixture.workflow.claim(row)
            self.assertFalse(result.claimed)
            self.assertEqual(fixture.execution.contracts, [])
            self.assertEqual(await fixture.contracts.count(job.job_id, 0), 1)

    async def test_conversation_phase_is_derived_from_durable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            self.assertEqual((await fixture.workflow.snapshot("conv")).phase,
                             ConversationPhase.READY)
            await fixture.service.create_job(["WP5"], conversation_id="running-conv")
            self.assertEqual((await fixture.workflow.snapshot("running-conv")).phase,
                             ConversationPhase.RUNNING)
            await fixture.paused()
            self.assertEqual((await fixture.workflow.snapshot("conv")).phase,
                             ConversationPhase.PAUSED)
            intent = await fixture.workflow.interpret(
                "retry contract: use .env.example",
                await fixture.workflow.context("conv"))
            proposal = await fixture.workflow.propose(intent)
            self.assertEqual((await fixture.workflow.snapshot("conv")).phase,
                             ConversationPhase.PROPOSED)
            await fixture.workflow.reject(
                await fixture.workflow.find("conv", proposal.confirmation_token))
            self.assertEqual((await fixture.workflow.snapshot("conv")).phase,
                             ConversationPhase.PAUSED)

    async def test_repeated_confirmation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            await fixture.paused()
            row = await self._proposal(fixture)
            first = await fixture.workflow.claim(row)
            second = await fixture.workflow.claim(row)  # stale in-memory copy
            third = await fixture.workflow.claim(
                await fixture.workflow.find("conv", row["confirmation_token"]))
            self.assertTrue(first.claimed)
            self.assertFalse(second.claimed)
            self.assertFalse(third.claimed)
            self.assertIn("already confirmed", third.message)
            with self.assertRaises(RuntimeError):
                [event async for event in fixture.workflow.execute(
                    row, fixture.service, [], _specs(), {})]

    async def test_stale_contract_revision_executes_nothing_and_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            job, first = await fixture.paused()
            row = await self._proposal(fixture)
            newer = fixture.service.contractor.revise_after_escalation(
                first, "something else changed", _specs())
            await fixture.contracts.save(job.job_id, 0, newer)

            result = await fixture.workflow.claim(row)
            self.assertEqual(result.status, "stale")
            self.assertIsNotNone(result.refreshed)
            self.assertIn("refreshed proposal", result.render())
            self.assertIn("TaskContract revision 3", result.render())
            self.assertNotEqual(result.refreshed.confirmation_token, row["confirmation_token"])
            self.assertEqual(fixture.execution.contracts, [])

    async def test_changed_state_executes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            job, _ = await fixture.paused()
            row = await self._proposal(fixture)
            await fixture.persistence.update_implementation_task(job.job_id, 0, "running")
            result = await fixture.workflow.claim(row)
            self.assertEqual(result.status, "stale")
            self.assertIsNone(result.refreshed)
            self.assertIn("no longer applies", result.message)
            self.assertEqual(fixture.execution.contracts, [])

    async def test_expired_proposal_executes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            await fixture.paused()
            row = await self._proposal(fixture)
            with sqlite3.connect(fixture.persistence.path) as db:
                db.execute("UPDATE workflow_action_proposals SET expires_at=?",
                           (time.time() - 1,))
            expired = await fixture.workflow.find("conv", row["confirmation_token"])
            self.assertEqual(expired["status"], "expired")
            result = await fixture.workflow.claim(expired)
            self.assertEqual(result.status, "expired")
            self.assertIsNotNone(result.refreshed)
            self.assertEqual(fixture.execution.contracts, [])

    async def test_new_message_supersedes_and_change_amends(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            await fixture.paused()
            first = await self._proposal(fixture)
            second = await self._proposal(fixture)
            stale = await fixture.workflow.find("conv", first["confirmation_token"])
            self.assertEqual(stale["status"], "superseded")
            self.assertFalse((await fixture.workflow.claim(stale)).claimed)

            changed = await fixture.workflow.change(second, "also add tests")
            self.assertIn("also add tests", changed.intent.guidance)
            self.assertIn(ENV_GUIDANCE, changed.intent.guidance)
            old = await fixture.workflow.find("conv", second["confirmation_token"])
            self.assertEqual(old["status"], "superseded")
            stages = [row[0] for row in fixture.rows(
                "SELECT stage FROM telemetry_workflow_events WHERE proposal_id=?",
                second["proposal_id"])]
            self.assertIn("corrected", stages)
            self.assertEqual(fixture.execution.contracts, [])

    async def test_telemetry_is_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            await fixture.paused()
            row = await self._proposal(fixture)
            result = await fixture.workflow.claim(row)
            [event async for event in fixture.workflow.execute(
                result.proposal, fixture.service, [], _specs(), {})]
            rows = fixture.rows(
                "SELECT stage, action, source_state, interpretation_source, confidence, "
                "ambiguity_count, confirmation_latency_ms, transition_result, "
                "message_fingerprint FROM telemetry_workflow_events ORDER BY id")
            stages = [item[0] for item in rows]
            self.assertEqual(stages[:4], ["interpreted", "proposed", "confirmed", "executed"])
            interpreted = rows[0]
            self.assertEqual(interpreted[1:6], (
                "RETRY_CONTRACT", "awaiting_contract_resolution", "model", "high", 0))
            self.assertIsNotNone(rows[2][6])
            self.assertTrue(rows[3][7])
            self.assertEqual(len(interpreted[8]), 64)
            dump = json.dumps(fixture.rows("SELECT * FROM telemetry_workflow_events"))
            self.assertNotIn("fill it after", dump)


class PausedStateMigrationTests(unittest.TestCase):
    def test_legacy_paused_jobs_are_reclassified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.sqlite3"
            RoutingPersistence(path)
            TaskContractPersistence(path)
            rows = {
                "contract": ("awaiting_user_decision", "path outside contract scope: .env"),
                "review": ("awaiting_user_decision", "gap"),
                "late_escalation": ("awaiting_user_decision", "scope"),
                "exhausted": ("awaiting_user_decision", "review failures"),
                "coder": ("awaiting_user_decision", "tool call could not be parsed"),
                "unknown": ("awaiting_user_decision", None),
                "question": ("awaiting_confirmation", None),
                "evidence": ("review_pending", "no evidence"),
            }
            with sqlite3.connect(path) as db:
                for job_id, (status, error) in rows.items():
                    db.execute("""INSERT INTO implementation_jobs
                      (job_id, selectors_json, instruction, conversation_id, status,
                       created_at, updated_at) VALUES (?, '[]', '', 'conv', ?, 1, 1)""",
                               (job_id, status))
                    db.execute("""INSERT INTO implementation_job_tasks
                      (job_id, position, change_id, task_id, status, error)
                      VALUES (?, 0, 'CHG1', 'WP1-T1', ?, ?)""", (job_id, status, error))

                def escalation(job_id, created_at, limit="execution_policy"):
                    db.execute("""INSERT INTO task_contract_escalations
                      (job_id, position, task_id, contract_id, reason, escalation_json,
                       created_at) VALUES (?, 0, 'WP1-T1', 'c', 'r', ?, ?)""",
                               (job_id, json.dumps({"exhausted_limit": limit}), created_at))

                def review(job_id, created_at, verdict="needs_work"):
                    db.execute("""INSERT INTO implementation_reviews
                      (job_id, position, coder_model, reviewer_model, verdict, rationale,
                       created_at) VALUES (?, 0, 'c', 'r', ?, 'x', ?)""",
                               (job_id, verdict, created_at))

                escalation("contract", 5)
                review("review", 5)
                review("late_escalation", 5)
                escalation("late_escalation", 9)
                review("exhausted", 5)
                escalation("exhausted", 9, "review_failures")
                review("evidence", 5, "review_pending")

            # Reopen in the reverse order to prove the migration is order-independent.
            TaskContractPersistence(path)
            persistence = RoutingPersistence(path)
            with sqlite3.connect(path) as db:
                tasks = dict(db.execute(
                    "SELECT job_id, status FROM implementation_job_tasks").fetchall())
                jobs = dict(db.execute(
                    "SELECT job_id, status FROM implementation_jobs").fetchall())
                verdict = db.execute("""SELECT verdict FROM implementation_reviews
                  WHERE job_id='evidence'""").fetchone()[0]
            self.assertEqual(tasks, {
                "contract": "awaiting_contract_resolution",
                "review": "awaiting_review_resolution",
                "late_escalation": "awaiting_contract_resolution",
                "exhausted": "awaiting_review_resolution",
                "coder": "awaiting_coder_recovery",
                "unknown": "awaiting_manual_resolution",
                "question": "awaiting_human_confirmation",
                "evidence": "awaiting_evidence_resolution",
            })
            self.assertEqual(jobs["contract"], "awaiting_contract_resolution")
            self.assertEqual(jobs["unknown"], "awaiting_manual_resolution")
            self.assertEqual(verdict, "review_pending")

            # The previously stuck job is now recoverable through the workflow.
            import asyncio
            context = asyncio.run(WorkflowController(persistence).context("conv"))
            stuck = next(task for task in context.paused if task.job_id == "contract")
            self.assertIn(A.RETRY_CONTRACT, stuck.actions)
            manual = next(task for task in context.paused if task.job_id == "unknown")
            self.assertEqual(manual.actions, (A.STOP_JOB,))


def _body_text(chunks) -> str:
    raw = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks)
    text = []
    for line in raw.decode().splitlines():
        if line.startswith("data: {"):
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            text.append(delta.get("content") or "")
    return "".join(text)


class TranscriptTests(unittest.IsolatedAsyncioTestCase):
    """Drive the HTTP controller exactly as the IDE chat does."""

    def _patches(self, fixture):
        session = SimpleNamespace(
            client=object(), tools=SimpleNamespace(specs=_specs(), handlers={}))
        task_execution = SimpleNamespace(persistence=fixture.persistence,
                                         execute=fixture.execution.execute)
        return (
            mock.patch.object(dialog_controller, "_workflow", return_value=fixture.workflow),
            mock.patch.object(dialog_controller, "get_session", return_value=session),
            mock.patch.object(dialog_controller, "task_execution", task_execution),
            mock.patch.object(dialog_controller, "_artifact_store",
                              return_value=fixture.artifacts),
            mock.patch.object(dialog_controller.workspace, "native_project",
                              return_value=object()),
            mock.patch.object(ImplementationService, "__init__", _service_init(fixture)),
        )

    async def _say(self, content, conversation="conv", model="test"):
        response = await dialog_controller.chat_completions(ChatCompletionRequest(
            model=model, stream=True, conversation_id=conversation,
            messages=[ChatMessage(role="user", content=content)]))
        return _body_text([chunk async for chunk in response.body_iterator])

    async def test_env_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory, _Provider(_retry_contract_reply()))
            job, first = await fixture.paused()
            patches = self._patches(fixture)
            for patch in patches:
                patch.start()
            try:
                preview = await self._say(ENV_REPLY)
                self.assertIn("I understood your request as:", preview)
                self.assertIn("Retry WP4-T1 with a revised TaskContract", preview)
                self.assertIn("Keep `.env` outside the contract", preview)
                self.assertIn("Nothing has executed yet.", preview)
                self.assertEqual(fixture.execution.contracts, [])
                token = re.search(r"confirm ([A-Z0-9]{6})", preview).group(1)

                executed = await self._say(f"confirm {token}")
                self.assertIn("TaskContract revision 2", executed)
                self.assertNotIn("no completed review to submit again", executed)
                self.assertEqual([c.revision for c in fixture.execution.contracts], [2])
                self.assertEqual(fixture.execution.contracts[0].parent_contract_id,
                                 first.contract_id)
                self.assertEqual(await fixture.persistence.review_count(job.job_id, 0), 0)
                self.assertEqual(len(fixture.rows("SELECT * FROM implementation_jobs")), 1)

                repeated = await self._say(f"confirm {token}")
                self.assertIn("already confirmed", repeated)
                self.assertEqual(len(fixture.execution.contracts), 1)
            finally:
                for patch in patches:
                    patch.stop()

    async def test_explicit_implement_command_requires_planner(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            patches = self._patches(fixture)
            for patch in patches:
                patch.start()
            try:
                preview = await self._say("/implement WP4 WP5", conversation="fresh")
                self.assertIn("Work Package planning is unavailable", preview)
                self.assertEqual(fixture.rows("SELECT * FROM implementation_jobs"), [])
                self.assertEqual(fixture.execution.contracts, [])
            finally:
                for patch in patches:
                    patch.stop()

    async def test_free_form_correction_replaces_pending_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            await fixture.paused()
            patches = self._patches(fixture)
            for patch in patches:
                patch.start()
            try:
                first = await self._say("retry contract: use .env.example")
                first_token = re.search(r"confirm ([A-Z0-9]{6})", first).group(1)
                revised = await self._say("No, retry contract: avoid .env and add tests")
                self.assertIn("I understood your request as:", revised)
                second_token = re.search(r"confirm ([A-Z0-9]{6})", revised).group(1)
                self.assertNotEqual(first_token, second_token)
                old = await fixture.workflow.find("conv", first_token)
                self.assertEqual(old["status"], "superseded")
                self.assertEqual(fixture.execution.contracts, [])
            finally:
                for patch in patches:
                    patch.stop()

    async def test_correction_can_switch_to_a_different_work_package(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            await fixture.paused()
            patches = self._patches(fixture)
            for patch in patches:
                patch.start()
            try:
                first = await self._say("retry contract: use .env.example")
                token = re.search(r"confirm ([A-Z0-9]{6})", first).group(1)
                classification = {"action": "START_IMPLEMENTATION",
                                  "selectors": ("WP5",), "instruction": "",
                                  "confidence": "high"}
                with mock.patch("agent.workflow.controller.ConversationIntentInterpreter") as parser:
                    parser.return_value.interpret = mock.AsyncMock(
                        return_value=classification)
                    revised = await self._say("No, start WP5 instead",
                                              model="codepartner-implement")
                self.assertIn("Work Package planning is unavailable", revised)
                self.assertEqual((await fixture.workflow.find("conv", token))["status"],
                                 "pending")
                self.assertEqual(len(fixture.rows("SELECT * FROM implementation_jobs")), 1)
            finally:
                for patch in patches:
                    patch.stop()

    async def test_mode_routed_commands_are_not_hijacked_by_a_paused_job(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            await fixture.paused()
            patches = self._patches(fixture)
            for patch in patches:
                patch.start()
            try:
                reply = await self._say("WP4: are tests missing?", model="codepartner-review")
                self.assertIn("Action: Run a workflow command", reply)
                self.assertIn("`/review WP4: are tests missing?`", reply)
                self.assertEqual(fixture.provider.calls, [])
            finally:
                for patch in patches:
                    patch.stop()

    async def test_unknown_token_executes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(directory)
            patches = self._patches(fixture)
            for patch in patches:
                patch.start()
            try:
                reply = await self._say("confirm ZZZZZZ")
                self.assertIn("No workflow proposal matches", reply)
            finally:
                for patch in patches:
                    patch.stop()


def _service_init(fixture):
    original = ImplementationService.__init__

    def init(self, artifacts, task_execution=None, persistence=None, *args, **kwargs):
        original(self, artifacts, fixture.execution, fixture.persistence,
                 reviewer=SimpleNamespace(), contract_persistence=fixture.contracts)
    return init


if __name__ == "__main__":
    unittest.main()

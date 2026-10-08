"""Turn detection, two-way interpretation, and switching between flows.

A new request may be understood immediately, but it must never silently displace
work already in progress, and a fresh user message must never be mistaken for a
tool continuation.
"""

import asyncio
import contextlib
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
from agent.spec.implementation_intent import selectors_in_text
from agent.spec.models import Task
from agent.task_contracts.persistence import TaskContractPersistence
from agent.workflow import switching
from agent.workflow.controller import WorkflowController
from agent.workflow.interpreter import ConversationIntentInterpreter, WorkflowInterpreter
from agent.workflow.reconciler import Classification, Route, reconcile
from api import dialog_controller
from conversation import openai_request as oreq
from mcp_bridge import bridge
from mcp_bridge import server as bridge_server
from mcp_bridge.models import DoneEvent, Run, TextEvent
from mcp_bridge.registry import ToolSpec
from model.chat import ChatCompletionRequest, ChatMessage
from providers.codex import CodexProvider

PROJECT = "proj_a"


def _task(task_id: str) -> Task:
    return Task(
        id=task_id, title=f"Build {task_id}", description="Implement the feature.",
        complexity="simple", reasoning="Localized.", context="Small change.",
        preferred_capability="write", acceptance_criteria=["It works."],
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
        self.tasks = {f"WP{n}-T1": _task(f"WP{n}-T1") for n in (4, 5, 6, 7)}

    async def tasks_for_selectors(self, selectors, include_completed=False):
        return [(self.change, self.tasks[selector if "-" in selector else f"{selector}-T1"])
                for selector in selectors]

    async def task_by_id(self, task_id):
        return self.change, self.tasks[task_id]


class _Execution:
    """A Coder stand-in that records which task ran and what it saw when it started."""

    def __init__(self):
        self.started = []
        self.observe = None

    async def execute(self, task, change_id, messages, specs, **kwargs):
        seen = await self.observe() if self.observe else None
        self.started.append((task.id, seen))
        yield TextEvent("no structured handoff")
        yield DoneEvent("endTurn")


class _NoModel:
    async def emit_json(self, messages, schema, model=None):
        raise AssertionError("the contextual model was not expected")


async def _free_reading(self, message, mode_hint="", context_summary=""):
    """Deterministic stand-in for the context-free classifier."""
    assert context_summary == "", "the context-free reading must not see workflow state"
    seen = set(selectors_in_text(message))
    value = ConversationIntentInterpreter._fallback(message, mode_hint, seen)
    return {"action": value["action"], "selectors": tuple(sorted(seen)),
            "instruction": "", "confidence": "high"}


def _body_text(chunks) -> str:
    raw = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks)
    text = []
    for line in raw.decode().splitlines():
        if line.startswith("data: {"):
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            text.append(delta.get("content") or "")
    return "".join(text)


class _World:
    """One routing database shared by several chats and IDE projects."""

    def __init__(self, directory: str):
        self.persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
        self.contracts = TaskContractPersistence(self.persistence.path)
        self.artifacts = _Artifacts()
        self.execution = _Execution()
        self.service = ImplementationService(
            self.artifacts, self.execution, self.persistence,
            reviewer=SimpleNamespace(), contract_persistence=self.contracts)
        self.interpreter = WorkflowInterpreter(_NoModel(), "test-model")
        self.sessions = {}

    def workflow(self, session=None) -> WorkflowController:
        return WorkflowController(self.persistence, self.contracts,
                                  interpreter=self.interpreter,
                                  project_id=getattr(session, "project_id", None))

    def join(self, conversation: str, project: str | None = PROJECT):
        self.sessions[conversation] = SimpleNamespace(
            client=object(), tools=SimpleNamespace(specs=_specs(), handlers={}),
            project_id=project)

    def patches(self):
        world = self

        def service_init(service, artifacts, task_execution=None, persistence=None,
                         *args, **kwargs):
            original_init(service, world.artifacts, world.execution, world.persistence,
                          reviewer=SimpleNamespace(), contract_persistence=world.contracts)

        original_init = ImplementationService.__init__
        return (
            mock.patch.object(dialog_controller, "_workflow", side_effect=self.workflow),
            mock.patch.object(dialog_controller, "get_session",
                              side_effect=lambda conversation: self.sessions.get(conversation)),
            mock.patch.object(dialog_controller, "task_execution", SimpleNamespace(
                persistence=self.persistence, execute=self.execution.execute)),
            mock.patch.object(dialog_controller, "_artifact_store",
                              return_value=self.artifacts),
            mock.patch.object(ImplementationService, "__init__", service_init),
            mock.patch.object(ConversationIntentInterpreter, "interpret", _free_reading),
            mock.patch.object(dialog_controller.providers, "active",
                              return_value=SimpleNamespace(tools=self._no_coder)),
        )

    @staticmethod
    def _no_coder(*_args, **_kwargs):
        raise AssertionError("the raw coding agent must not run for a fresh user turn")

    async def say(self, content, conversation="conv", history=(), stream=True,
                  model="test", conversation_id=True):
        messages = [*history, ChatMessage(role="user", content=content)]
        response = await dialog_controller.chat_completions(ChatCompletionRequest(
            model=model, stream=stream, messages=messages,
            conversation_id=conversation if conversation_id else None))
        if not stream:
            return response.choices[0].message.content
        return _body_text([chunk async for chunk in response.body_iterator])

    async def job(self, selector, conversation, status="running", project=PROJECT):
        job = await self.service.create_job([selector], conversation_id=conversation,
                                            project_id=project)
        await self.persistence.update_implementation_task(job.job_id, 0, status)
        return job

    def live_run(self, test: unittest.TestCase, job_id: str) -> Run:
        """Pretend a provider run is executing the job in this process."""
        run = Run()
        run.metadata["implementation_job_id"] = job_id
        run.task = asyncio.get_running_loop().create_task(asyncio.Event().wait())
        bridge_server._live_runs[run.run_id] = run

        def cleanup():
            bridge_server._live_runs.pop(run.run_id, None)
            run.task.cancel()
        test.addCleanup(cleanup)
        return run

    def rows(self, sql, *args):
        with contextlib.closing(sqlite3.connect(self.persistence.path)) as db:
            return db.execute(sql, args).fetchall()

    def job_status(self, job_id):
        return self.rows("SELECT status FROM implementation_jobs WHERE job_id=?", job_id)[0][0]


class _WorldTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.world = _World(directory.name)
        self.world.join("conv")
        for patch in self.world.patches():
            patch.start()
            self.addCleanup(patch.stop)


# --- turn detection --------------------------------------------------------------------

class TurnDetectionTests(_WorldTest):
    async def test_fresh_message_after_bootstrap_history_is_not_a_continuation(self):
        """The logged failure: setup results in the history hid every later user turn."""
        dialog_controller.remember_bootstrap_call("boot-1", "conv")
        history = (
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="", tool_calls=[{
                "id": "boot-1", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"file_path": ".codepartner/ide_mcp.json"})}}]),
            ChatMessage(role="tool", tool_call_id="boot-1", content="{}"),
            ChatMessage(role="assistant", content="IDE MCP session configured."),
        )
        answer = await self.world.say("implement WP5 and WP6", history=history,
                                      conversation_id=False)
        self.assertIn("Action: Start implementation", answer)
        self.assertIn("Nothing has executed yet.", answer)
        self.assertEqual(self.world.execution.started, [])

    async def test_old_coder_tool_results_do_not_resume_anything(self):
        history = (
            ChatMessage(role="user", content="fix it"),
            ChatMessage(role="assistant", content="", tool_calls=[{
                "id": "call_old", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"}}]),
            ChatMessage(role="tool", tool_call_id="call_old", content="text"),
            ChatMessage(role="assistant", content="Done."),
        )
        answer = await self.world.say("implement WP5", history=history)
        self.assertIn("Task WP5-T1", answer)

    async def test_bootstrap_result_replays_the_original_message_exactly_once(self):
        run = Run()
        run.metadata.update({"mcp_bootstrap": True, "conversation_id": "conv"})
        future = bridge.register("boot-2", run, "read_file",
                                 {"file_path": ".codepartner/ide_mcp.json"})

        async def bootstrap():
            await future
        run.task = asyncio.get_running_loop().create_task(bootstrap())
        self.addCleanup(bridge.discard, run)
        messages = [
            ChatMessage(role="user", content="implement WP5"),
            ChatMessage(role="assistant", content="", tool_calls=[{
                "id": "boot-2", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"file_path": ".codepartner/ide_mcp.json"})}}]),
            ChatMessage(role="tool", tool_call_id="boot-2", content="{}"),
        ]
        request = ChatCompletionRequest(model="test", stream=True, messages=messages)
        first = _body_text([chunk async for chunk in (
            await dialog_controller.chat_completions(request)).body_iterator])
        self.assertIn("Task WP5-T1", first)
        again = _body_text([chunk async for chunk in (
            await dialog_controller.chat_completions(request)).body_iterator])
        self.assertIn("already processed", again)
        self.assertEqual(len(self.world.rows("SELECT * FROM workflow_action_proposals")), 1)

    async def test_result_for_a_finished_run_is_reported_not_rerouted(self):
        response = await dialog_controller.chat_completions(ChatCompletionRequest(
            model="test", stream=True, conversation_id="conv", messages=[
                ChatMessage(role="user", content="go"),
                ChatMessage(role="assistant", content="", tool_calls=[{
                    "id": "call_gone", "type": "function",
                    "function": {"name": "apply_patch", "arguments": "{}"}}]),
                ChatMessage(role="tool", tool_call_id="call_gone", content="ok"),
            ]))
        body = _body_text([chunk async for chunk in response.body_iterator])
        self.assertIn("no longer active", body)

    async def test_fresh_turn_without_identity_never_reaches_the_coder(self):
        response = await dialog_controller.chat_completions(ChatCompletionRequest(
            model="test", stream=True, tool_choice="none",
            tools=[{"type": "function", "function": {
                "name": "apply_patch", "parameters": {"type": "object"}}}],
            messages=[ChatMessage(role="user", content="rewrite the parser")]))
        body = _body_text([chunk async for chunk in response.body_iterator])
        self.assertIn("will not run the coding agent directly", body)

    def test_only_trailing_tool_results_count(self):
        request = ChatCompletionRequest(model="test", messages=[
            ChatMessage(role="tool", tool_call_id="old", content="x"),
            ChatMessage(role="user", content="new"),
        ])
        self.assertEqual(oreq.trailing_tool_results(request), [])
        self.assertEqual(len(oreq.tool_results(request)), 1)
        self.assertEqual(CodexProvider()._trailing_tool_results([
            {"role": "tool", "tool_call_id": "old"}, {"role": "user", "content": "new"},
        ]), [])


# --- interpretation ------------------------------------------------------------------

class InterpretationTests(_WorldTest):
    async def test_new_selectors_start_a_new_flow_while_another_task_is_paused(self):
        await self.world.job("WP4", "conv", status="awaiting_contract_resolution")
        answer = await self.world.say("implement WP5 and WP6")
        self.assertIn("Action: Start implementation", answer)
        self.assertIn("Task WP5-T1", answer)
        self.assertIn("Task WP6-T1", answer)
        self.assertIn("WP4 is paused", answer)
        self.assertIn("It will stay paused", answer)
        self.assertNotIn("No paused task matches", answer)

    async def test_bare_yes_answers_the_single_reviewer_question(self):
        await self.world.job("WP4", "conv", status="awaiting_human_confirmation")
        answer = await self.world.say("yes")
        self.assertIn("Answer the Reviewer's question for WP4-T1", answer)
        self.assertNotIn("No workflow proposal matches", answer)

    async def test_bare_yes_with_two_waiting_items_executes_nothing(self):
        await self.world.job("WP4", "conv", status="awaiting_human_confirmation")
        await self.world.say("/implement WP5")
        answer = await self.world.say("yes")
        self.assertIn("More than one thing is waiting", answer)
        self.assertIn("The Reviewer's question for `WP4-T1`", answer)
        self.assertEqual(self.world.execution.started, [])
        self.assertEqual(len(self.world.rows(
            "SELECT * FROM workflow_action_proposals WHERE status='pending'")), 1)

    async def test_bare_yes_with_nothing_waiting_is_explicit(self):
        answer = await self.world.say("yes")
        self.assertIn("Nothing in this conversation is waiting", answer)

    async def test_three_chats_stay_independent(self):
        self.world.join("chat-a", "proj_a")
        self.world.join("chat-b", "proj_b")
        self.world.join("chat-c", "proj_c")
        preview = await self.world.say("/implement WP5", conversation="chat-a")
        token = re.search(r"confirm ([A-Z0-9]{6})", preview).group(1)
        await self.world.job("WP4", "chat-b", status="awaiting_human_confirmation",
                             project="proj_b")

        self.assertIn("Nothing in this conversation is waiting",
                      await self.world.say("yes", conversation="chat-c"))
        self.assertIn("No workflow proposal matches",
                      await self.world.say(f"confirm {token}", conversation="chat-c"))
        self.assertIn("Answer the Reviewer's question",
                      await self.world.say("yes", conversation="chat-b"))
        executed = await self.world.say("yes", conversation="chat-a")
        self.assertIn("queued 1 task(s)", executed)
        self.assertEqual([task for task, _ in self.world.execution.started], ["WP5-T1"])

    async def test_correction_replaces_the_pending_proposal(self):
        first = await self.world.say("implement WP5")
        old = re.search(r"confirm ([A-Z0-9]{6})", first).group(1)
        second = await self.world.say("implement WP6")
        self.assertIn("Task WP6-T1", second)
        self.assertIn("superseded", await self.world.say(f"confirm {old}"))
        self.assertEqual(self.world.execution.started, [])

    async def test_turns_are_logged_without_the_prompt(self):
        await self.world.say("implement WP5 and WP6")
        rows = self.world.rows("SELECT * FROM telemetry_turn_events")
        self.assertTrue(rows)
        self.assertNotIn("implement WP5", json.dumps(rows))
        route = self.world.rows("SELECT conversation_id, project_id, turn_type, route, "
                                "free_json FROM telemetry_turn_events")[-1]
        self.assertEqual(route[:4], ("conv", PROJECT, "user", "new_flow"))
        self.assertEqual(json.loads(route[4])["action"], "START_IMPLEMENTATION")


class ReconcilerTests(unittest.TestCase):
    def test_context_free_new_work_without_waiting_work(self):
        self.assertEqual(reconcile(Classification("START_IMPLEMENTATION", ("WP5",)), None).route,
                         Route.NEW_FLOW)
        self.assertEqual(reconcile(Classification("ANSWER"), None).route, Route.QUESTION)

    def test_explicit_reply_is_a_continuation(self):
        contextual = Classification("RETRY_REVIEW", ("WP4-T1",), "high", (), "deterministic")
        self.assertEqual(reconcile(None, contextual).route, Route.CONTINUATION)

    def test_competing_readings_are_unclear_not_guessed(self):
        result = reconcile(Classification("START_IMPLEMENTATION", ("WP4",), "high"),
                           Classification("RETRY_CONTRACT", ("WP4-T1",), "medium"))
        self.assertEqual(result.route, Route.UNCLEAR)
        self.assertEqual(result.alternatives, ("START_IMPLEMENTATION", "RETRY_CONTRACT"))

    def test_guidance_for_a_paused_task_stays_a_continuation(self):
        result = reconcile(Classification("CODING", (), "high"),
                           Classification("RETRY_CONTRACT", ("WP4-T1",), "medium"))
        self.assertEqual(result.route, Route.CONTINUATION)

    def test_model_clarification_yields_to_clear_new_work(self):
        result = reconcile(Classification("SPEC_DRAFT", (), "high"),
                           Classification("ASK_CLARIFICATION", ("WP4-T1",), "low"))
        self.assertEqual(result.route, Route.NEW_FLOW)


# --- switching -------------------------------------------------------------------------

class SwitchTests(_WorldTest):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.wp4 = await self.world.job("WP4", "other-chat")
        self.run = self.world.live_run(self, self.wp4.job_id)

    async def _propose_switch(self):
        preview = await self.world.say("implement WP5 and WP6")
        self.assertIn("CodePartner is currently working on WP4 (started from another chat).",
                      preview)
        self.assertIn("Would you like to stop WP4 and start implementing WP5 and WP6?",
                      preview)
        self.assertIn("WP4's existing file changes will remain; they will not be rolled back.",
                      preview)
        return re.search(r"confirm ([A-Z0-9]{6})", preview).group(1)

    def _switch_state(self):
        return self.world.rows("SELECT state FROM workflow_switches")[0][0]

    async def test_proposal_changes_nothing_until_answered(self):
        await self._propose_switch()
        self.assertEqual(self.world.job_status(self.wp4.job_id), "running")
        self.assertFalse(self.run.task.done())
        self.assertEqual(len(self.world.rows("SELECT * FROM implementation_jobs")), 1)

    async def test_confirmed_switch_stops_first_then_starts(self):
        await self._propose_switch()
        wp4 = self.wp4.job_id
        self.world.execution.observe = lambda: self.world.persistence.implementation_job_status(wp4)
        executed = await self.world.say("yes")
        self.assertIn("Stopped WP4. Its file changes were kept.", executed)
        self.assertIn("queued 2 task(s)", executed)
        self.assertTrue(self.run.task.cancelled())
        self.assertEqual(self.world.execution.started[0], ("WP5-T1", "stopped"))
        self.assertEqual(self.world.job_status(wp4), "stopped")
        self.assertEqual(self.world.rows(
            "SELECT status FROM implementation_job_tasks WHERE job_id=?", wp4), [("stopped",)])
        self.assertEqual(self._switch_state(), "started")

    async def test_rejection_leaves_the_running_job_alone(self):
        await self._propose_switch()
        self.assertIn("rejected", await self.world.say("no"))
        self.assertFalse(self.run.task.done())
        self.assertEqual(self.world.job_status(self.wp4.job_id), "running")
        self.assertEqual(self.world.execution.started, [])

    async def test_duplicate_confirmation_switches_once(self):
        token = await self._propose_switch()
        await self.world.say(f"confirm {token}")
        repeated = await self.world.say(f"confirm {token}")
        self.assertIn("already confirmed", repeated)
        self.assertEqual(len(self.world.execution.started), 1)
        self.assertEqual(len(self.world.rows("SELECT * FROM workflow_switches")), 1)

    async def test_job_finishing_during_the_decision_refreshes_the_proposal(self):
        await self._propose_switch()
        await self.world.persistence.finish_implementation_job(self.wp4.job_id)
        self.run.task.cancel()
        bridge_server._live_runs.pop(self.run.run_id, None)
        answer = await self.world.say("yes")
        self.assertIn("The workflow changed after this proposal was created.", answer)
        self.assertIn("refreshed proposal", answer)
        self.assertNotIn("currently working on WP4", answer)
        self.assertEqual(self.world.execution.started, [])

    async def test_failed_cancellation_starts_nothing(self):
        pending = asyncio.get_running_loop().create_future()
        self.addCleanup(pending.cancel)
        self.run.pending["call_write"] = pending
        self.run.tool_names["call_write"] = "apply_patch"
        await self._propose_switch()
        with mock.patch.object(switching, "STOP_WAIT_SECONDS", 0.1):
            answer = await self.world.say("yes")
        self.assertIn("could not be stopped safely", answer)
        self.assertIn("Nothing new was started", answer)
        self.assertFalse(self.run.task.done())
        self.assertNotIn("stop_requested", self.run.metadata)
        self.assertEqual(self.world.job_status(self.wp4.job_id), "running")
        self.assertEqual(self.world.execution.started, [])
        self.assertEqual(self._switch_state(), "stop_failed")
        self.assertEqual(self.world.rows(
            "SELECT status FROM implementation_jobs WHERE job_id != ?", self.wp4.job_id),
            [("cancelled",)])

    async def test_new_tool_calls_are_refused_while_stopping(self):
        self.run.metadata["stop_requested"] = True
        result = await bridge.call_tool(self.run, "read_file", {"file_path": "x"})
        self.assertIn("being stopped", result)

    async def test_stopped_job_does_not_advance_when_its_request_finishes(self):
        await self.world.persistence.finalize_stopped_job(self.wp4.job_id, "stopped")
        events = [event async for event in self.world.service._advance(
            self.wp4.job_id, [], _specs())]
        self.assertIn("was stopped", events[0].text)
        await self.world.persistence.update_implementation_task(self.wp4.job_id, 0, "waiting")
        self.assertEqual(self.world.job_status(self.wp4.job_id), "stopped")

    async def test_crash_mid_switch_is_recovered_idempotently(self):
        token = await self._propose_switch()
        workflow = self.world.workflow(self.world.sessions["conv"])
        claimed = await workflow.claim(await workflow.find("conv", token),
                                       self.world.artifacts)
        self.assertTrue(claimed.claimed)
        # The process died while stopping: nothing executes WP4 any more.
        await self.world.persistence.mark_job_stopping(self.wp4.job_id)
        self.run.task.cancel()
        bridge_server._live_runs.pop(self.run.run_id, None)
        with contextlib.closing(sqlite3.connect(self.world.persistence.path)) as db, db:
            db.execute("UPDATE workflow_switches SET state='stopping', updated_at=?",
                       (time.time() - 3600,))
        await workflow.snapshot("conv")
        await workflow.snapshot("conv")
        self.assertEqual(self._switch_state(), "interrupted")
        self.assertEqual(self.world.job_status(self.wp4.job_id), "stopped")
        self.assertEqual(self.world.job_status(claimed.job.job_id), "cancelled")
        notice = await self.world.say("implement WP7")
        self.assertIn("was interrupted", notice)
        self.assertIn("Task WP7-T1", notice)
        self.assertNotIn("was interrupted", await self.world.say("implement WP7"))
        self.assertEqual(self.world.execution.started, [])

    async def test_paused_job_is_kept_paused_not_stopped(self):
        await self.world.persistence.update_implementation_task(
            self.wp4.job_id, 0, "awaiting_review_resolution")
        self.run.task.cancel()
        bridge_server._live_runs.pop(self.run.run_id, None)
        preview = await self.world.say("implement WP5")
        self.assertNotIn("currently working on", preview)
        self.assertIn("WP4 is paused (the Reviewer finished and a decision is needed)", preview)
        executed = await self.world.say("yes")
        self.assertIn("queued 1 task(s)", executed)
        self.assertEqual(self.world.job_status(self.wp4.job_id), "awaiting_review_resolution")

    async def test_orphaned_job_is_described_accurately(self):
        self.run.task.cancel()
        bridge_server._live_runs.pop(self.run.run_id, None)
        preview = await self.world.say("implement WP5")
        self.assertIn("is recorded as running, but nothing is executing it", preview)
        self.assertIn("mark WP4 as stopped", preview)

    async def test_other_projects_do_not_conflict(self):
        self.world.join("elsewhere", "proj_b")
        preview = await self.world.say("implement WP5", conversation="elsewhere")
        self.assertIn("I understood your request as:", preview)
        self.assertNotIn("WP4", preview)

    async def test_read_only_review_does_not_displace_work(self):
        preview = await self.world.say("/review WP4")
        self.assertNotIn("Would you like to stop", preview)


if __name__ == "__main__":
    unittest.main()

"""HTTP controller for the OpenAI-compatible chat endpoint.

Thin: it routes (tool turn / plain chat) and translates between OpenAI wire types
and the neutral inputs the active provider consumes. Each provider exposes one
uniform `tools(messages, specs)` surface, so the controller never branches on
which provider is active — every stateful/stateless detail is private to it.
OpenAI request parsing lives in `conversation/openai_request.py`; the engine
(driven from inside the Claude provider) lives in `mcp_bridge/`.
"""
import json
import logging
import os
import re
import time
from contextvars import ContextVar
from collections import deque

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

import providers
from agent.spec import (
    ArtifactSelectionError,
    ArtifactStoreError,
    CliOpenSpecAdapter,
    OpenSpecArtifactStore,
    SpecCommandError,
    SpecCommandHandler,
    extract_help_response,
    extract_project_command,
    extract_object_id,
    extract_implementation_request,
    extract_run_task_id,
    extract_reset_wp_id,
    extract_scoped_request,
    is_help_command,
    is_project_command,
    is_apply_command,
    is_discard_command,
    is_explain_command,
    is_implementation_command,
    is_review_command,
    is_run_command,
    is_rework_command,
    is_reset_command,
    is_refine_command,
    is_spec_command,
    is_update_command,
)
from agent.spec.modes import request_shape
from agent.spec.refinement import RefinementService
from agent.spec.bridge_client import BridgeMcpFileClient
from agent import workspace
from agent.spec.filesystem_client import FilesystemFileClient
from agent.spec.ide_mcp_sessions import (
    bind_workspace, claim_bootstrap_replay, configure_session, conversation_for_request,
    conversation_marker, get_session,
    is_bootstrap_call, native_bootstrap_config, new_conversation_id, remember_bootstrap_call,
    remember_bootstrap_from_history, restore_session, strip_bootstrap_exchange,
    strip_conversation_marker,
)
from agent.routing import ImplementationService, TaskExecutionService
from agent.workflow.controller import (WorkflowController, parse_decision,
                                       BARE_CONFIRM, BARE_REJECT, PASSTHROUGH_ACTIONS,
                                       PLAN_APPROVAL, PLAN_DISCUSS, PlanStart)
from agent.planning.models import WPPlanDraft
from agent.planning.render import render_plan
from agent.routing.review import model_identity
from agent.planning.service import PlanningError, WPPlannerService, with_progress
from agent.planning.knowledge_base import (KnowledgeBaseError, configured_execution_tools,
                                           catch_up_index, configured_planner_tools,
                                           planning_context, refresh_accepted_wp)
from conversation.presentation import progress_line
from agent.routing.config import RoutingConfigError
from agent.workflow.states import WorkflowAction as WorkflowAction
from agent.workflow.review_retry import (explicit_operational_retry_request,
                                         explicit_retry_request,
                                         menu_choices, offered_choice)
from agent.workflow.switching import mark_switch_started
from conversation import openai_request as oreq
from conversation.non_streaming_responder import NonStreamingResponder
from conversation.streaming_responder import StreamingResponder
from mcp_bridge.registry import (
    OP_FILE_CREATE, OP_FILE_READ, OP_FILE_SEARCH, ToolRegistry,
    ROLE_CODER, ROLE_OPTIMIZER, classify_tool,
)
from mcp_bridge import bridge
from mcp_bridge import server as bridge_server
from mcp_bridge.models import DoneEvent, ErrorEvent, Event, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec
from model.chat import ChatCompletionRequest, ChatCompletionResponse
from logger.diagnostic import debug, new_trace_id
from telemetry import tracing

log = logging.getLogger("codepartner.api")

_IMPLEMENTATION_RESOLUTION = re.compile(
    r"\s*(submit to coder again|try coder again|accept anyway|accept)"
    r"(?:\s*:\s*(.+))?\s*", re.IGNORECASE | re.DOTALL)
def _implementation_resolution(content: str | None) -> tuple[str, str] | None:
    match = _IMPLEMENTATION_RESOLUTION.fullmatch(content or "")
    if match is None:
        return None
    command = match.group(1).lower()
    action = {"submit to coder again": "submit_again",
              "try coder again": "submit_again",
              "accept anyway": "accept",
              "accept": "accept"}[command]
    return action, (match.group(2) or "").strip()


router = APIRouter()
artifact_store = OpenSpecArtifactStore()
spec_handler = SpecCommandHandler(CliOpenSpecAdapter(), artifact_store)
task_execution = TaskExecutionService()
_workflow_replay: ContextVar[bool] = ContextVar("workflow_replay", default=False)


def _workflow(session=None) -> WorkflowController:
    return WorkflowController(task_execution.persistence,
                              project_id=getattr(session, "project_id", None))


def _kb_refresher(session):
    if session is None or not getattr(session, "knowledge_base", None):
        return None

    async def refresh(job_id: str, wp_id: str):
        return await refresh_accepted_wp(
            task_execution.persistence, job_id, wp_id, session.knowledge_base,
            session.client, session.project_path)

    return refresh


def _resume_service(run, session, active_specs):
    """The service that resumes an implementation run after an IDE tool round-trip.

    Reviewer runs carry the job id but not the service; the fresh service must still
    refresh the KB when the resumed review accepts the Work Package.
    """
    return run.metadata.get("implementation_service") or ImplementationService(
        _artifact_store() if session else artifact_store,
        task_execution, task_execution.persistence, kb_refresh=_kb_refresher(session))


async def _explained_plan(workflow, conversation_id: str) -> str:
    """The detailed rendering of the plan under discussion, from its stored draft."""
    session = await workflow.persistence.active_planning(conversation_id)
    if session is None or session.get("draft") is None:
        return "There is no plan to explain yet."
    config = getattr(task_execution, "config", None)
    try:
        label = model_identity(config, session["planner_model"]).label
    except Exception:
        label = session["planner_model"]
    return render_plan(
        WPPlanDraft.from_dict(session["draft"], session["wp_id"], session["revision"]),
        wp_complexity=session["wp_complexity"], planner_label=label,
        gaps=session.get("gaps") or [], token=None,
        queued_wp_ids=session.get("queued_wp_ids") or (), detail=True,
        margin=getattr(getattr(config, "planner", None), "edit_margin_lines", 10))


def _kb_catch_up(session, force: bool = False):
    """Reindex when accepted edits are newer than the index; `force` always reindexes
    (planning starts from a fresh index)."""
    if session is None or not getattr(session, "knowledge_base", None):
        return None

    async def catch_up(project_id: str | None, progress=None):
        return await catch_up_index(
            task_execution.persistence, project_id, session.knowledge_base,
            getattr(session, "client", None), getattr(session, "project_path", None),
            progress=progress, force=force)

    return catch_up


async def _conversation_workflow_turn(request, messages, conversation_id, session,
                                      active_specs, direct_handlers, created):
    """One proposal/confirmation boundary for user-authored workflow turns.

    Legacy slash handlers below are execution adapters during migration; this
    boundary alone decides whether a user turn may reach them.
    """
    if (_workflow_replay.get() or not conversation_id or not messages
            or messages[-1].get("role") != "user"):
        return None
    text = (messages[-1].get("content") or "").strip()
    if not text or is_help_command(text) or is_project_command(text):
        return None
    workflow = _workflow(session)
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    if text.isdigit():
        prior = messages[-2] if len(messages) > 1 else None
        choice = (offered_choice(text, messages[:-1])
                  if prior and prior.get("role") == "assistant"
                  else await workflow.persistence.offered_choice(
                      conversation_id, int(text)))
        if choice is None:
            return responder.reply(
                "I cannot link that number to a recent option. Nothing was executed. "
                "Please describe the action you want.", request.model, created)
        text = choice
    decision = parse_decision(text)
    snapshot = await workflow.snapshot(conversation_id)
    proposal = None
    forced_verb = None
    planning = (workflow.planning_route(text, snapshot)
                if decision is None and request.model != "codepartner-review"
                and not is_reset_command(text) else None)
    if planning == "abandon":
        closed = await WPPlannerService(
            None, workflow.persistence,
            config=getattr(task_execution, "config", None)).abandon(conversation_id)
        workflow.record_turn(conversation_id, "user", "planning", text, outcome="abandoned")
        return responder.reply(
            f"Planning for {closed['wp_id'] if closed else 'the work package'} closed. "
            "Nothing was implemented.", request.model, created)
    if planning == "explain":
        workflow.record_turn(conversation_id, "user", "planning", text, outcome="explain")
        return responder.reply(await _explained_plan(workflow, conversation_id),
                               request.model, created)
    if planning == "discuss":
        if PLAN_DISCUSS.fullmatch(text):
            return responder.reply(
                f"What would you like to change in the {snapshot.context.planning_wp_id} plan? "
                "You can adjust scope, assumptions, the implementation strategy, or task "
                "boundaries, or ask me to investigate something further.",
                request.model, created)
        workflow.record_turn(conversation_id, "user", "planning", text, outcome="discuss")
        return _plan_turn(request, workflow, session, active_specs, direct_handlers,
                          conversation_id, created, text, snapshot.context.planning_wp_id,
                          discuss=True)
    if planning == "approve" or (snapshot.context.planning_session_id
                                 and PLAN_APPROVAL.fullmatch(text) and decision is None):
        proposal = _plan_approval(snapshot)
        if proposal is None:
            return responder.reply(
                "This plan revision cannot be approved yet; it is incomplete or still being "
                "revised. Tell me what to change, or ask me to investigate further. "
                "Nothing was executed.", request.model, created)
        forced_verb = "confirm"
    if (not snapshot.context.paused and decision is None
            and explicit_operational_retry_request(text, messages[:-1])
            and not text.startswith("/")):
        if explicit_retry_request(text, messages[:-1]):
            return responder.reply(
                "I understood that you want to retry a review, but no paused review was "
                "found in this conversation. Nothing was executed.",
                request.model, created)
        return responder.reply(
            "No paused coder or reviewer job is available to retry here. "
            "Nothing was executed.", request.model, created)
    bare = (forced_verb is None and decision is None
            and text.lower() in BARE_CONFIRM | BARE_REJECT)
    if bare:
        # Protocol, not interpretation: a bare yes/no binds to exactly one waiting item.
        proposal, problem = workflow.resolve_bare_reply(text, snapshot)
        if problem:
            workflow.record_turn(conversation_id, "user", "protocol", text, outcome="unbound")
            return responder.reply(problem, request.model, created)
    if decision or proposal is not None:
        if decision:
            proposal = await workflow.find(conversation_id, decision.token)
            verb = decision.verb
        else:
            verb = forced_verb or ("confirm" if text.lower() in BARE_CONFIRM else "reject")
        workflow.record_turn(conversation_id, "user", "protocol", text,
                             proposal_id=proposal["proposal_id"] if proposal else None,
                             outcome=verb if proposal else "unmatched")
        if proposal is None:
            return responder.reply(
                "No workflow proposal matches that reply in this conversation. "
                "Nothing was executed.", request.model, created)
        if verb == "reject":
            return responder.reply(await workflow.reject(proposal), request.model, created)
        if verb == "change":
            artifacts = (_artifact_store()
                         if session is not None else None)
            changed = await workflow.change(proposal, decision.correction, artifacts)
            return responder.reply(changed if isinstance(changed, str) else changed.render(),
                                   request.model, created)
        if not request.stream or session is None:
            return responder.reply(
                "Confirmation requires a streaming IDE MCP session. Nothing was executed; "
                "the proposal remains pending.", request.model, created)
        proposed_action = WorkflowAction(proposal["action"])
        proposed_command = proposal["payload"]["parameters"].get("command", "")
        if request.model == "codepartner-review" and not (
                proposed_action == WorkflowAction.RETRY_REVIEW or
                (proposed_action == WorkflowAction.EXECUTE_COMMAND
                 and is_review_command(proposed_command))):
            return responder.reply(
                "Review mode is read-only; this proposal was not confirmed. "
                "Switch modes to continue.", request.model, created)
        artifacts = _artifact_store()
        claimed = await workflow.claim(proposal, artifacts)
        if not claimed.claimed:
            return responder.reply(claimed.render(), request.model, created)
        action = WorkflowAction(claimed.proposal["action"])
        if action in PASSTHROUGH_ACTIONS:
            switched = await workflow.run_switch(claimed.proposal)
            if switched is not None and not switched.ok:
                await workflow.transition_finished(claimed.proposal,
                                                   "switch: " + switched.message)
                return responder.reply(switched.message, request.model, created)
            if switched is not None:
                await mark_switch_started(workflow.persistence,
                                          "switch_" + claimed.proposal["proposal_id"])
            command = claimed.proposal["payload"]["parameters"]["command"]
            if command.startswith("/__workflow_coding__ "):
                coding_messages = [*messages[:-1], {
                    "role": "user", "content": command[len("/__workflow_coding__ "):],
                }]
                return _native_tools(False, coding_messages, active_specs, ROLE_CODER,
                                     direct_handlers, request, created,
                                     switched.message if switched is not None else None)
            for marker, operation in (("/__workflow_spec_draft__ ", "draft"),
                                      ("/__workflow_spec_update__ ", "update")):
                if command.startswith(marker):
                    run = _start_refinement_command(
                        active_specs, new_trace_id(), session.client, conversation_id,
                        operation, instruction=command[len(marker):],
                        kb_config=session.knowledge_base)
                    return StreamingResponder().drain_events(
                        _drain_run(run), request.model, created)
            replay = request.model_copy(update={"messages": [
                *request.messages[:-1],
                request.messages[-1].model_copy(update={"content": command}),
            ]})
            guard = _workflow_replay.set(True)
            try:
                return await chat_completions(replay)
            finally:
                _workflow_replay.reset(guard)
        service = ImplementationService(artifacts, task_execution,
                                        task_execution.persistence,
                                        kb_refresh=_kb_refresher(session),
                                        kb_catch_up=_kb_catch_up(session))
        events = workflow.execute(claimed.proposal, service, messages, active_specs,
                                  direct_handlers, claimed.job)
        return StreamingResponder().drain_events(events, request.model, created)

    routed = text
    if routed.startswith("/"):
        if routed.lower() == "/retry-review":
            if not snapshot.context.paused:
                return responder.reply("No paused review is available. Nothing was executed.",
                                       request.model, created)
            intent = await workflow.interpret(routed, snapshot.context)
            if intent.action == WorkflowAction.ASK_CLARIFICATION:
                return responder.reply(intent.clarification, request.model, created)
            proposal = await workflow.propose(intent)
            return responder.reply(proposal.render(), request.model, created)
        known_command = any(check(routed) for check in (
            is_implementation_command, is_run_command, is_review_command,
            is_rework_command, is_refine_command, is_apply_command,
            is_discard_command, is_spec_command, is_explain_command,
            is_update_command, is_reset_command))
        if not known_command:
            return responder.reply(
                "That is not a registered CodePartner command. Nothing was executed.",
                request.model, created)
        if request.model == "codepartner-review" and not (
                is_review_command(routed) or is_help_command(routed)):
            return responder.reply(
                "Review mode is read-only. Switch modes to propose a mutating action. "
                "Nothing was executed.", request.model, created)
        if is_implementation_command(routed) or is_run_command(routed):
            try:
                if is_run_command(routed):
                    selectors, instruction = (extract_run_task_id(routed),), ""
                else:
                    selectors, instruction = extract_implementation_request(routed)
                if session is None:
                    return responder.reply(
                        "I can propose implementation once the IDE MCP session is active. "
                        "Nothing was executed.", request.model, created)
                plan = await workflow.plan_start_for_selectors(
                    _artifact_store(), selectors,
                    instruction or text)
                workflow.record_turn(conversation_id, "user", "command", text,
                                     outcome="plan_start" if isinstance(plan, PlanStart)
                                     else "reply")
                if isinstance(plan, PlanStart):
                    return _plan_turn(request, workflow, session, active_specs,
                                      direct_handlers, conversation_id, created, text,
                                      plan.wp_ids[0], queued_wp_ids=plan.wp_ids[1:])
                return responder.reply(plan, request.model, created)
            except (SpecCommandError, ArtifactSelectionError, ArtifactStoreError) as exc:
                return responder.reply(f"Nothing was executed. {exc}", request.model, created)
        else:
            action = (WorkflowAction.APPLY_SPEC if is_apply_command(routed) else
                      WorkflowAction.DISCARD_SPEC if is_discard_command(routed) else
                      WorkflowAction.EXECUTE_COMMAND)
            active_draft = (await task_execution.persistence.active_refinement(conversation_id)
                            if action in {WorkflowAction.APPLY_SPEC,
                                          WorkflowAction.DISCARD_SPEC} else None)
            if action in {WorkflowAction.APPLY_SPEC, WorkflowAction.DISCARD_SPEC} and not active_draft:
                return responder.reply(
                    "There is no active OpenSpec draft to apply or discard. "
                    "Nothing was executed.", request.model, created)
            proposal = await workflow.propose(
                workflow.command_intent(
                    conversation_id, routed, action,
                    active_draft["session_id"] if active_draft else None))
        workflow.record_turn(conversation_id, "user", "command", text,
                             proposal_id=proposal.proposal_id, outcome="proposal")
        return responder.reply(proposal.render(), request.model, created)

    mode_hint = ("implement" if request.model == "codepartner-implement" else
                 "plan" if request.model == PLAN_MODEL else
                 "review" if request.model == "codepartner-review" else
                 "spec" if request.model == "codepartner-spec" else "")
    artifacts = (_artifact_store()
                 if session is not None else None)
    result = await workflow.route_turn(conversation_id, text, mode_hint, artifacts,
                                       snapshot, messages[:-1])
    if result.kind == "plan_start":
        return _plan_turn(request, workflow, session, active_specs, direct_handlers,
                          conversation_id, created, text, result.wp_id,
                          queued_wp_ids=result.queued_wp_ids)
    if result.kind == "answer":
        if request.stream and active_specs:
            return _native_tools(True, messages, active_specs, ROLE_OPTIMIZER,
                                 direct_handlers, request, created)
        prompt, system = oreq.build_prompt(request)
        return await responder.build_completion(prompt, system, request.model, created)
    if (result.kind == "proposal" and result.proposal is not None
            and result.proposal.action in {
                WorkflowAction.RETRY_REVIEW, WorkflowAction.RESOLVE_CODER_RECOVERY}
            and result.proposal.intent.parameters.get("recovery_mode", "retry_coder")
                == "retry_coder"
            and explicit_operational_retry_request(text, messages[:-1])
            and not result.proposal.intent.ambiguities
            and not result.proposal.intent.parameters.get("displace")):
        if not request.stream:
            return responder.reply(result.render(), request.model, created)
        if session is None:
            return responder.reply(
                "Retry needs a streaming IDE MCP session to continue the saved job. "
                "Nothing was executed.", request.model, created)
        if (request.model == "codepartner-review"
                and result.proposal.action != WorkflowAction.RETRY_REVIEW):
            return responder.reply(
                "Coding retry needs implement mode. Nothing was executed.",
                request.model, created)
        proposal = await workflow.find(conversation_id, result.proposal.confirmation_token)
        claimed = await workflow.claim(proposal, artifacts) if proposal else None
        if claimed is None or not claimed.claimed:
            return responder.reply(
                claimed.render() if claimed else "The retry is no longer available.",
                request.model, created)
        service = ImplementationService(
            artifacts, task_execution, task_execution.persistence,
            kb_refresh=_kb_refresher(session), kb_catch_up=_kb_catch_up(session))
        events = workflow.execute(claimed.proposal, service, messages, active_specs,
                                  direct_handlers, claimed.job)
        return StreamingResponder().drain_events(events, request.model, created)
    return responder.reply(result.render(), request.model, created)


PLAN_MODEL = "codepartner-plan"


def _plan_approval(snapshot) -> dict | None:
    """The pending approval bound to the active planning session, if it is approvable."""
    session_id = snapshot.context.planning_session_id
    return next((row for row in snapshot.pending
                 if row["action"] == WorkflowAction.APPROVE_WP_PLAN.value
                 and (row["payload"].get("parameters") or {}).get("planning_session_id")
                 == session_id), None)


def _plan_turn(request, workflow, session, active_specs, direct_handlers,
               conversation_id: str, created: int, text: str, wp_id: str,
               discuss: bool = False, queued_wp_ids: tuple[str, ...] = ()):
    """Run one planner turn (start or discuss) and offer Approve / Discuss more."""
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    if not request.stream or session is None:
        return responder.reply(
            "Work Package planning needs a streaming IDE MCP session. Nothing was executed.",
            request.model, created)
    artifacts = _artifact_store()
    planner = WPPlannerService(artifacts, workflow.persistence,
                               config=getattr(task_execution, "config", None))

    async def events():
        try:
            _, _, package = await artifacts.object_by_id(wp_id)
            kb_request = "\n".join((
                package.objective,
                *(f"{task.title}: {task.description}" for task in package.tasks
                  if task.status != "cancelled"),
                text,
            ))
            kb_context = ""
            # A new plan starts from a fresh index; revisions reuse it.
            catch_up = _kb_catch_up(session, force=not discuss)
            if catch_up is not None:
                async for kind, value in with_progress(lambda progress: catch_up(
                        getattr(session, "project_id", None),
                        lambda line: progress(progress_line("tool", line)))):
                    if kind == "progress":
                        yield TextEvent(value)
                    elif value:
                        yield TextEvent(progress_line("tool", value))
            async for kind, value in with_progress(lambda progress: planning_context(
                    getattr(session, "knowledge_base", None), "implementation",
                    kb_request, lambda line: progress(progress_line("tool", line)))):
                if kind == "progress":
                    yield TextEvent(value)
                else:
                    kb_context = value
            if kb_context and not any(spec.name.startswith("kb_") for spec in active_specs):
                try:
                    kb_specs, kb_handlers, kb_status = await configured_planner_tools(
                        getattr(session, "knowledge_base", None))
                except KnowledgeBaseError as exc:
                    log.warning("kb_wp_tools_unavailable error=%s", exc)
                    kb_specs, kb_handlers, kb_status = [], {}, ""
            else:
                kb_specs, kb_handlers, kb_status = [], {}, ""
        except (KnowledgeBaseError, ArtifactSelectionError, ArtifactStoreError) as exc:
            yield ErrorEvent(f"Nothing was executed. {exc}")
            yield DoneEvent("endTurn")
            return
        planner_specs = [*active_specs, *kb_specs]
        planner_handlers = {**direct_handlers, **kb_handlers}

        def call(progress):
            if discuss:
                return planner.discuss(conversation_id, text, planner_specs, planner_handlers,
                                       progress=progress, kb_status=kb_context or kb_status)
            return planner.start(conversation_id, wp_id, text, planner_specs, planner_handlers,
                                 project_id=session.project_id, queued_wp_ids=queued_wp_ids,
                                 progress=progress, kb_status=kb_context or kb_status)

        turn = None
        try:
            async for kind, value in with_progress(call):
                if kind == "progress":
                    yield TextEvent(value)
                else:
                    turn = value
            proposal = (None if turn.gaps else
                        await workflow.propose_plan_approval(conversation_id, turn, artifacts))
        except (PlanningError, ArtifactSelectionError, ArtifactStoreError,
                RoutingConfigError) as exc:
            yield ErrorEvent(f"Nothing was executed. {exc}")
            yield DoneEvent("endTurn")
            return
        reply = render_plan(turn.draft, wp_complexity=turn.wp_complexity,
                            planner_label=turn.planner_label, gaps=turn.gaps,
                            token=proposal.confirmation_token if proposal else None,
                            queued_wp_ids=turn.queued_wp_ids)
        if proposal is not None and proposal.intent.parameters.get("displace"):
            reply += "\n\n" + proposal.render()
        await workflow.persistence.record_offered_choices(conversation_id, menu_choices(reply))
        yield TextEvent(progress_line("separator"))
        yield TextEvent(reply + "\n")
        yield DoneEvent("endTurn")

    return StreamingResponder().drain_events(events(), request.model, created)


async def _prefixed(text: str, events):
    yield TextEvent(text + "\n\n")
    async for event in events:
        yield event


async def _replay_original_turn(request: ChatCompletionRequest, bootstrap_call_id: str,
                                conversation_id: str, created: int):
    """Process the user message that triggered IDE MCP setup, exactly once.

    The client answers the setup file read with a tool result, so the request that
    arrives afterwards ends in that result rather than in the user's message.
    """
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    if not claim_bootstrap_replay(bootstrap_call_id):
        return responder.reply("IDE MCP session configured. That request was already "
                               "processed; nothing new was executed.", request.model, created)
    index = next((position for position, message in enumerate(request.messages)
                  if any(call.get("id") == bootstrap_call_id
                         for call in message.tool_calls or [])), None)
    original = request.messages[:index] if index is not None else []
    _workflow().record_turn(conversation_id, "bootstrap_replay",
                            "replay" if original and original[-1].role == "user" else "none")
    if not original or original[-1].role != "user":
        return responder.reply("IDE MCP session configured.", request.model, created)
    return await chat_completions(request.model_copy(update={
        "messages": original, "conversation_id": conversation_id}))


# Artifact commands state what they need; the registry resolves compatible
# aliases/adapters from the current IDE advertisement.  No tool wire name is a
# policy here.
_ARTIFACT_OPERATIONS = frozenset({OP_FILE_READ, OP_FILE_SEARCH, OP_FILE_CREATE})
_RECENT_REQUESTS: deque[dict] = deque(maxlen=64)


class _NativeUnavailable(Exception):
    """Native execution is required for this conversation but cannot run."""


def _chat_native(read_only: bool):
    """Native settings for a chat/workflow model turn of the active provider.

    Answers and `/review` are read-only sessions. Ad-hoc coding turns write without
    a change journal; journaled writes belong to implementation jobs."""
    plan = workspace.plan_native(getattr(providers.active(), "NAME", None))
    if not plan.native:
        raise _NativeUnavailable(f"Native execution is unavailable: {plan.reason}. "
                                 "Nothing was executed.")
    if read_only:
        return workspace.read_only_session(plan.project)
    env, notes = workspace.toolchain(plan.project)
    return workspace.NativeRun(workspace=plan.project.workspace, env=env, notes=notes)


def _native_tools(read_only: bool, messages, specs, role, handlers, request, created,
                  prefix: str | None = None):
    """Stream one provider tool turn, natively when the conversation's project allows."""
    try:
        native = _chat_native(read_only)
    except _NativeUnavailable as exc:
        return StreamingResponder().reply(str(exc), request.model, created)
    events = providers.active().tools(messages, specs, role=role,
                                      direct_tool_handlers=handlers, native=native)
    if prefix is not None:
        events = _prefixed(prefix, events)
    return StreamingResponder().drain_events(events, request.model, created)


def _with_marker(response: StreamingResponse, conversation_id: str) -> StreamingResponse:
    """Append the conversation marker to the reply, just before its finish chunk, so
    the IDE resends the conversation id with the history (no bootstrap call)."""
    marker = conversation_marker(conversation_id)
    body = response.body_iterator

    async def stream():
        added = False
        async for chunk in body:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            if not added and text.startswith("data: {"):
                try:
                    payload = json.loads(text[len("data: "):])
                except json.JSONDecodeError:
                    payload = None
                choices = (payload or {}).get("choices") or []
                if choices and choices[0].get("finish_reason"):
                    added = True
                    yield "data: " + json.dumps({**payload, "choices": [{
                        "index": 0, "delta": {"content": marker},
                        "finish_reason": None}]}) + "\n\n"
            yield chunk
    return StreamingResponse(stream(), media_type=response.media_type,
                             headers=dict(response.headers))


def _artifact_store():
    """OpenSpec artifacts of the conversation's native project, read from disk; without
    one, the store refuses target-project access."""
    project = workspace.native_project()
    if project is not None:
        return OpenSpecArtifactStore(FilesystemFileClient(project.workspace))
    return OpenSpecArtifactStore()


def _advertisement_audit(request: ChatCompletionRequest, trace_id: str) -> dict:
    """Correlate request shapes without retaining message or tool-result text."""
    audit = oreq.safe_request_structure(request)
    now = time.time()
    same_session = [item for item in reversed(_RECENT_REQUESTS)
                    if item["model"] == request.model
                    and item["user_fingerprint"] == audit["user_fingerprint"]]
    previous = same_session[0] if same_session else None
    audit.update({
        "trace_id": trace_id,
        "timestamp": now,
        "model": request.model,
        "previous_trace_id": previous["trace_id"] if previous else None,
        "previous_origin": previous["origin"] if previous else None,
        "previous_tool_surface_fingerprint": previous["tool_surface_fingerprint"] if previous else None,
        "seconds_since_previous": round(now - previous["timestamp"], 3) if previous else None,
        "correlation_scope": "model_and_user_fingerprint" if audit["user_fingerprint"] else "model_only",
    })
    _RECENT_REQUESTS.append(audit)
    return audit


def _bridge_required_response(command: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(f"{command} needs this conversation bound to a registered native project "
                "(see /project)."),
    )


def _tool_calls_permitted(request: ChatCompletionRequest) -> bool:
    choice = request.tool_choice
    if choice == "none":
        return False
    return not (isinstance(choice, dict) and choice.get("type") == "none")


async def _project_reply(session, conversation_id: str | None, content: str) -> str:
    """`/project`: show, list or bind the registered workspace of this conversation."""
    try:
        action, project_id = extract_project_command(content)
        projects = workspace.registry().projects()
    except (SpecCommandError, workspace.WorkspaceError) as exc:
        return f"{exc}. Nothing was changed."

    def status(item) -> str:
        try:
            workspace.validate_workspace(item)
            return "mounted"
        except workspace.WorkspaceError as exc:
            return str(exc)

    if action == "list":
        if not projects:
            return ("No projects are registered. See docs/native-execution.md to register "
                    "and mount one.")
        return "\n".join(["Registered projects:", *(
            f"- `{item.id}` → {item.workspace}: {status(item)}"
            for item in projects.values())])
    if action == "use":
        if project_id not in projects:
            return (f"Project {project_id!r} is not registered; use `/project list`. "
                    "Nothing was changed.")
        state = status(projects[project_id])
        if state != "mounted":
            return f"Cannot bind {project_id!r}: {state}. Nothing was changed."
        if session is None or conversation_id is None:
            return "Binding a project needs an active IDE MCP session. Nothing was changed."
        host_path = projects[project_id].host_path
        ide_path = getattr(session, "project_path", None)
        if host_path and ide_path and os.path.normpath(ide_path) != host_path:
            # OpenSpec artifacts are read through this IDE project; code must live there too.
            return (f"{project_id!r} is mounted from {host_path}, but this IDE conversation "
                    f"belongs to {ide_path}. Open that project in the IDE to work in it. "
                    "Nothing was changed.")
        if project_id != session.workspace_id and await task_execution.persistence.project_jobs(
                session.project_id, conversation_id):
            return ("An implementation job of this conversation is still unfinished; finish "
                    "or stop it before switching projects. Nothing was changed.")
        bind_workspace(conversation_id, project_id)
        return (f"This conversation now executes natively in `{project_id}` "
                f"({projects[project_id].workspace}).")
    current = getattr(session, "workspace_id", None)
    if current is None:
        return ("This conversation is not bound to a registered project, so tasks cannot "
                "run. Use `/project list` and `/project use <id>`.")
    item = projects.get(current)
    if item is None:
        return f"Bound project {current!r} is no longer registered; tasks cannot run."
    return f"Project `{current}` → {item.workspace}: {status(item)}."


def _requires_mcp(request: ChatCompletionRequest, last: dict) -> bool:
    if last.get("role") == "user":
        content = (last.get("content") or "").lstrip().lower()
        if content.startswith(("/spec", "/explain", "/refine", "/apply", "/discard", "/update", "/run",
                               "/implement", "/execute", "/review", "/rework", "/project")):
            return True
    return bool(request.tools) and _tool_calls_permitted(request)


def _start_mcp_bootstrap(specs: list[ToolSpec], conversation_id: str, trace_id: str):
    """Read the per-project MCP config through the current IDE bridge once."""
    async def strategy(run, _prompt, _system, _specs):
        try:
            config_text = await BridgeMcpFileClient(run).read_file(".codepartner/ide_mcp.json")
            await configure_session(conversation_id, config_text)
        except Exception as exc:
            run.metadata["bootstrap_error"] = str(exc)
            raise
        await run.put(TextEvent("IDE MCP session configured."))
        return "endTurn"

    run = bridge_server.start_run(
        "", None, specs, strategy, allowed_operations=frozenset({OP_FILE_READ})
    )
    run.metadata.update({
        "trace_id": trace_id,
        "mcp_bootstrap": True,
        "conversation_id": conversation_id,
    })
    return run


async def _bootstrap_events(run, conversation_id: str):
    async for event in _drain_run(run):
        if isinstance(event, ToolCallEvent):
            remember_bootstrap_call(event.tool_call_id, conversation_id)
        yield event


async def _drain_run(run) -> "AsyncIterator[Event]":
    while True:
        event = await run.next_event()
        yield event
        if isinstance(event, DoneEvent):
            return


async def _resume_child_to_parent(child, parent, messages: list[dict], specs: list[ToolSpec]):
    """Resume a provider child, then continue its request-scoped bridge owner.

    The child owns the source-tool call.  The parent remains alive to perform
    subsequent catalog operations through the very same advertised tools.
    """
    async for event in task_execution.resume(child, messages, specs):
        if not isinstance(event, DoneEvent):
            yield event
    async for event in _drain_run(parent):
        yield event


def _start_artifact_command(messages: list[dict], specs: list[ToolSpec], trace_id: str,
                            client, project_id: str | None = None,
                            conversation_id: str | None = None,
                            kb_config: dict | None = None):
    """Run OpenSpec persistence through the conversation's configured MCP server."""
    async def strategy(run, _prompt, _system, _specs):
        store = _artifact_store()
        content = messages[-1].get("content") or ""
        if is_reset_command(content):
            wp_id = extract_reset_wp_id(content)
            blocking_job = await task_execution.persistence.unfinished_job_for_wp(
                project_id, conversation_id, wp_id)
            if blocking_job:
                raise ArtifactSelectionError(
                    f"{wp_id} has unfinished implementation job {blocking_job}; "
                    "finish or stop that job before resetting it")
        kb_context = ""
        if is_spec_command(content) or is_update_command(content):
            async def kb_progress(line: str):
                await run.put(TextEvent(progress_line("tool", line)))

            kb_context = await planning_context(
                kb_config, "openspec", content, progress=kb_progress)
        handler = SpecCommandHandler(CliOpenSpecAdapter(), store, kb_context)
        result = await handler.handle(messages)
        await run.put(TextEvent(result.content))
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    # ##DELETE AFTER CORRECTION##
    run.metadata["trace_id"] = trace_id
    run.metadata["artifact_command"] = True
    debug(log, "api.artifact_command_started", trace_id=trace_id, run_id=run.run_id,
          advertised_tool_names=[spec.name for spec in specs])
    return run


def _start_refinement_command(specs: list[ToolSpec], trace_id: str, client,
                              conversation_id: str, action: str,
                              target_id: str = "", instruction: str = "",
                              kb_config: dict | None = None):
    async def strategy(run, _prompt, _system, _specs):
        async def kb_progress(line: str):
            await run.put(TextEvent(progress_line("tool", line)))

        kb_context = (await planning_context(kb_config, "openspec", instruction,
                                             progress=kb_progress)
                      if action in {"draft", "update"} else "")
        service = RefinementService(
            _artifact_store(), CliOpenSpecAdapter(), task_execution.persistence,
            kb_context)
        if action == "start":
            content = await service.start(conversation_id, target_id)
        elif action == "apply":
            content = await service.apply(conversation_id)
        elif action == "discard":
            content = await service.discard(conversation_id)
        elif action == "draft":
            content = await service.start_draft(conversation_id, instruction)
        else:
            content = await service.update(conversation_id, instruction)
        await run.put(TextEvent(content))
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    run.metadata.update({"trace_id": trace_id, "artifact_command": True,
                         "artifact_command_kind": f"refinement_{action}"})
    return run


def _start_rework_command(specs: list[ToolSpec], selectors: list[str],
                          feedback: str, trace_id: str, client,
                          kb_config: dict | None = None):
    """Create corrective work using the conversation's configured MCP server."""
    async def strategy(run, _prompt, _system, _specs):
        store = _artifact_store()
        targets = await store.tasks_for_selectors(selectors, include_completed=True)
        kb_request = "\n".join((feedback, *(task.description for _, task in targets)))
        async def kb_progress(line: str):
            await run.put(TextEvent(progress_line("tool", line)))

        kb_context = await planning_context(kb_config, "openspec", kb_request,
                                            progress=kb_progress)
        handler = SpecCommandHandler(CliOpenSpecAdapter(), store, kb_context)
        result = await handler.rework(targets, feedback + ("\n\n" + kb_context if kb_context else ""))
        await run.put(TextEvent(result.content))
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    run.metadata["trace_id"] = trace_id
    run.metadata["artifact_command"] = True
    run.metadata["artifact_command_kind"] = "rework"
    debug(log, "api.artifact_command_started", trace_id=trace_id, run_id=run.run_id,
          command="rework", selector_count=len(selectors),
          advertised_tool_names=[spec.name for spec in specs])
    return run


async def _consume_implementation_events(run, service, events, messages, specs):
    last_child = None
    async for event in events:
        await run.put(event)
        if isinstance(event, DoneEvent):
            return
        if isinstance(event, ToolCallEvent):
            child = bridge.run_for(event.tool_call_id)
            if child is None:
                await run.put(TextEvent("Implementation stopped: pending IDE tool call was lost."))
                return
            child.metadata["parent_command_run"] = run
            await child.task
            last_child = child
    if last_child is not None:
        await _consume_implementation_events(
            run, service, service.finish_child(last_child, messages, specs),
            messages, specs)


def _start_implementation_confirmation(messages: list[dict], specs: list[ToolSpec],
                                       trace_id: str, client, direct_tool_handlers: dict,
                                       job_id: str, position: int, approved: bool,
                                       feedback: str, instruction: str):
    async def strategy(run, _prompt, _system, _specs):
        service = ImplementationService(_artifact_store(), task_execution)
        await _consume_implementation_events(
            run, service, service.confirm(job_id, position, approved, feedback,
                                          messages, specs, direct_tool_handlers, instruction),
            messages, specs)
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    run.metadata.update({"trace_id": trace_id, "bridge_command": "confirm_implementation"})
    return run


def _start_implementation_review_retry(messages: list[dict], specs: list[ToolSpec],
                                       trace_id: str, client, direct_tool_handlers: dict,
                                       job_id: str, position: int, instruction: str):
    async def strategy(run, _prompt, _system, _specs):
        service = ImplementationService(_artifact_store(), task_execution)
        await _consume_implementation_events(
            run, service, service.retry_review(job_id, position, messages, specs,
                                               instruction, direct_tool_handlers),
            messages, specs)
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    run.metadata.update({"trace_id": trace_id, "bridge_command": "retry_implementation_review"})
    return run


def _start_implementation_resolution(messages: list[dict], specs: list[ToolSpec],
                                     trace_id: str, client, direct_tool_handlers: dict,
                                     job_id: str, position: int, instruction: str,
                                     action: str, detail: str = ""):
    async def strategy(run, _prompt, _system, _specs):
        service = ImplementationService(_artifact_store(), task_execution)
        if action == "submit_again":
            events = service.retry_coder(
                job_id, position, messages, specs, instruction,
                direct_tool_handlers, detail)
        else:
            events = service.accept_anyway(
                job_id, position, messages, specs, instruction,
                direct_tool_handlers, detail)
        await _consume_implementation_events(run, service, events, messages, specs)
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    run.metadata.update({"trace_id": trace_id,
                         "bridge_command": f"implementation_{action}"})
    return run


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    created = int(time.time())

    # Only tool results that END the history make this a continuation. Results
    # from earlier turns stay in the resent history of every later request.
    results = oreq.trailing_tool_results(request)
    conversation_id = conversation_for_request(request)
    tracing.set_session(conversation_id)
    resumed_run = bridge.run_for(results[-1].tool_call_id) if results else None
    if (resumed_run is not None and resumed_run.metadata.get("mcp_bootstrap")
            and results):
        bridge_server.resume(results[-1].tool_call_id, results[-1].content or "")
        if resumed_run.task is not None:
            await resumed_run.task
        conversation_id = resumed_run.metadata.get("conversation_id") or conversation_id
        if conversation_id:
            remember_bootstrap_from_history(request, conversation_id)
        if get_session(conversation_id) is None:
            responder = StreamingResponder() if request.stream else NonStreamingResponder()
            detail = resumed_run.metadata.get("bootstrap_error") or "IDE MCP setup did not complete."
            return responder.reply(f"IDE MCP setup failed: {detail}", request.model, created)
        return await _replay_original_turn(request, results[-1].tool_call_id,
                                           conversation_id, created)
    if results and resumed_run is None and all(
            bridge.run_for(result.tool_call_id) is None for result in results):
        # Nothing is waiting for these results: a repeated setup delivery, or a
        # run that finished, timed out, or was stopped. Never start a new agent.
        if conversation_id and is_bootstrap_call(results[-1].tool_call_id):
            return await _replay_original_turn(request, results[-1].tool_call_id,
                                               conversation_id, created)
        _workflow().record_turn(conversation_id, "tool_continuation", "orphaned_result")
        responder = StreamingResponder() if request.stream else NonStreamingResponder()
        return responder.reply(
            "That tool result belongs to a CodePartner run that is no longer active "
            "(it finished, timed out, or was stopped). Nothing was executed.",
            request.model, created)

    # A known conversation is rebuilt silently after a restart or idle expiry.
    session = get_session(conversation_id) or await restore_session(conversation_id)
    # Native execution in this request (and runs it starts) uses the conversation's
    # registered workspace, never a path from the request.
    workspace.activate(getattr(session, "workspace_id", None))
    trace_id = (resumed_run.metadata.get("trace_id") if resumed_run else None) or new_trace_id()
    audit = _advertisement_audit(request, trace_id)
    debug(log, "api.request_advertisement", **audit,
          resumed_run_id=resumed_run.run_id if resumed_run else None)
    messages = strip_conversation_marker(strip_bootstrap_exchange(oreq.to_messages(request)))
    last = messages[-1] if messages else {}
    advertised = oreq.tool_specs(request)
    debug(log, "api.request_parsed", trace_id=trace_id,
          message_count=len(messages), advertised_tool_names=[spec.name for spec in advertised])
    if request.tools is not None:
        decisions = [classify_tool(spec) for spec in advertised]
        log.info("tool_advertisement count=%d names=%s accepted=%s rejected=%s",
                 len(advertised), [spec.name for spec in advertised],
                 [d.name for d in decisions if d.accepted],
                 [(d.name, d.reason) for d in decisions if not d.accepted])

    # Modes are interpretation hints. They no longer rewrite user messages.
    requires_mcp = (_requires_mcp(request, last)
                    or (request.model in {"codepartner-implement", "codepartner-review",
                                          "codepartner-spec", PLAN_MODEL}
                        and last.get("role") == "user" and bool(last.get("content"))))
    is_help = last.get("role") == "user" and is_help_command(last.get("content"))
    needs_mcp = session is None and not is_help and requires_mcp
    if needs_mcp and not request.stream:
        return NonStreamingResponder().reply(
            "An active streaming IDE MCP session is needed to interpret and confirm "
            "workflow actions. Nothing was executed.", request.model, created)
    native_config = native_bootstrap_config() if needs_mcp and not results else None
    if native_config is not None:
        # The mounted project's own ide_mcp.json: configure without an IDE tool call.
        new_id = conversation_id is None
        conversation_id = conversation_id or new_conversation_id()
        try:
            await configure_session(conversation_id, native_config)
        except Exception as exc:
            return StreamingResponder().reply(f"IDE MCP setup failed: {exc}",
                                              request.model, created)
        log.info("ide_mcp_session_configured_from_disk conversation_id=%s", conversation_id)
        response = await chat_completions(request.model_copy(
            update={"conversation_id": conversation_id}))
        if new_id and isinstance(response, StreamingResponse):
            response = _with_marker(response, conversation_id)
        response.headers["X-CodePartner-Conversation-ID"] = conversation_id
        return response
    if needs_mcp:
        if not _tool_calls_permitted(request):
            raise HTTPException(
                status_code=409,
                detail="IDE MCP setup requires tool calls to be enabled for the bootstrap file read.",
            )
        if conversation_id is None:
            conversation_id = new_conversation_id()
        reader = ToolRegistry(allowed_operations=frozenset({OP_FILE_READ}))
        reader.register(advertised)
        try:
            reader.prepare_call(OP_FILE_READ, {"file_path": ".codepartner/ide_mcp.json"})
        except (LookupError, ValueError):
            raise _bridge_required_response("MCP setup")
        bootstrap = _start_mcp_bootstrap(advertised, conversation_id, trace_id)
        remember_bootstrap_from_history(request, conversation_id)
        response = StreamingResponder().drain_events(
            _bootstrap_events(bootstrap, conversation_id), request.model, created
        )
        response.headers["X-CodePartner-Conversation-ID"] = conversation_id
        return response

    active_specs = session.tools.specs if session is not None else advertised
    direct_handlers = session.tools.handlers if session is not None else {}
    if session is not None:
        try:
            kb_specs, kb_handlers = await configured_execution_tools(
                getattr(session, "knowledge_base", None))
        except KnowledgeBaseError as exc:
            responder = StreamingResponder() if request.stream else NonStreamingResponder()
            return responder.reply(f"Codebase Memory configuration error: {exc}",
                                   request.model, created)
        active_specs = [*active_specs, *kb_specs]
        direct_handlers = {**direct_handlers, **kb_handlers}
    # Proposal construction is read-only and may use a non-streaming request.
    # Confirmed execution and legacy command adapters validate streaming below.

    if request.stream and results:
        result = results[-1]
        run = bridge.run_for(result.tool_call_id)
        debug(log, "api.tool_result_resume", trace_id=trace_id, run_id=run.run_id if run else None,
              tool_call_id=result.tool_call_id, tool_result_length=len(result.content or ""))
        if run is not None and (run.metadata.get("artifact_command") or run.metadata.get("bridge_command")):
            bridge_server.resume(result.tool_call_id, result.content or "")
            return StreamingResponder().drain_events(_drain_run(run), request.model, created)

        if run is not None and run.metadata.get("parent_command_run"):
            parent = run.metadata["parent_command_run"]
            return StreamingResponder().drain_events(
                _resume_child_to_parent(run, parent, messages, advertised), request.model, created
            )

    if not results:
        try:
            workflow_response = await _conversation_workflow_turn(
                request, messages, conversation_id, session, active_specs,
                direct_handlers, created)
        except HTTPException:
            raise
        except Exception:
            # Answer the IDE in-band: a 500 surfaces as an opaque client stack trace.
            log.exception("workflow_turn_failed trace_id=%s conversation_id=%s",
                          trace_id, conversation_id)
            responder = StreamingResponder() if request.stream else NonStreamingResponder()
            return responder.reply(
                f"CodePartner hit an internal error while handling this message "
                f"(trace {trace_id}). See the server log.", request.model, created)
        if workflow_response is not None:
            return workflow_response

    if last.get("role") == "user" and is_project_command(last.get("content")):
        responder = StreamingResponder() if request.stream else NonStreamingResponder()
        return responder.reply(await _project_reply(session, conversation_id,
                                                    last.get("content") or ""),
                               request.model, created)

    if is_help:
        responder = StreamingResponder() if request.stream else NonStreamingResponder()
        try:
            response = extract_help_response(last.get("content") or "")
        except SpecCommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return responder.reply(response, request.model, created)

    is_rework = last.get("role") == "user" and is_rework_command(last.get("content"))
    if is_rework:
        if not request.stream:
            raise HTTPException(status_code=400, detail="/rework requires a streaming request")
        try:
            selectors, feedback = extract_scoped_request(last.get("content") or "", "rework", True)
        except SpecCommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if workspace.native_project() is None:
            raise _bridge_required_response("/rework")
        run = _start_rework_command(active_specs, selectors, feedback, trace_id,
                                    session.client, session.knowledge_base)
        return StreamingResponder().drain_events(_drain_run(run), request.model, created)

    is_review = last.get("role") == "user" and is_review_command(last.get("content"))
    if is_review:
        if not request.stream:
            raise HTTPException(status_code=400, detail="/review requires a streaming request")
        try:
            selectors, question = extract_scoped_request(last.get("content") or "", "review")
            targets = await _artifact_store().tasks_for_selectors(
                selectors, include_completed=True
            )
        except SpecCommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ArtifactSelectionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ArtifactStoreError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        task_context = [
            {"change_id": change.id, "change_title": change.title, "task": vars(task)}
            for change, task in targets
        ]
        review_messages = [
            *[message for message in messages[:-1] if message.get("role") == "system"],
            {"role": "system", "content": (
                "You are a read-only implementation reviewer. Inspect the project only with "
                + ("your read-only file tools; the current directory is the project root"
                   if workspace.native_project() is not None
                   else "the configured IDE MCP read tools")
                + ". Never write, edit, create, format, or refactor files. "
                "Report evidence, gaps against the selected OpenSpec tasks, and suggested follow-up work."
            )},
            {"role": "user", "content": (
                f"Review these selected OpenSpec tasks and their implementation:\n{task_context}\n\n"
                f"Review focus: {question or 'Assess conformance, tests, and likely gaps.'}"
            )},
        ]
        return _native_tools(True, review_messages, active_specs, ROLE_OPTIMIZER,
                             direct_handlers, request, created)
    content = last.get("content") or ""
    if last.get("role") == "user" and is_refine_command(content):
        if not request.stream or session is None or conversation_id is None:
            raise HTTPException(status_code=409, detail="/refine requires an active IDE MCP session")
        target_id = extract_object_id(content, "refine")
        return StreamingResponder().drain_events(
            _drain_run(_start_refinement_command(
                active_specs, trace_id, session.client, conversation_id, "start", target_id)),
            request.model, created,
        )

    if last.get("role") == "user" and is_apply_command(content):
        if not request.stream or session is None or conversation_id is None:
            raise HTTPException(status_code=409, detail="/apply requires an active IDE MCP session")
        return StreamingResponder().drain_events(
            _drain_run(_start_refinement_command(
                active_specs, trace_id, session.client, conversation_id, "apply")),
            request.model, created,
        )

    if last.get("role") == "user" and is_discard_command(content):
        if not request.stream or session is None or conversation_id is None:
            raise HTTPException(status_code=409, detail="/discard requires an active IDE MCP session")
        return StreamingResponder().drain_events(
            _drain_run(_start_refinement_command(
                active_specs, trace_id, session.client, conversation_id, "discard")),
            request.model, created,
        )

    is_planning = last.get("role") == "user" and (
            is_spec_command(last.get("content"))
            or is_explain_command(last.get("content"))
            or is_update_command(last.get("content"))
            or is_reset_command(last.get("content"))
    )
    if is_planning:
        if not request.stream:
            raise HTTPException(status_code=400, detail="planning commands require a streaming request")
        if is_spec_command(last.get("content")):
            log.info("spec_command_request %s", request_shape(last.get("content") or ""))
        return StreamingResponder().drain_events(
            _drain_run(_start_artifact_command(
                messages, active_specs, trace_id, session.client,
                session.project_id, conversation_id, session.knowledge_base)),
            request.model, created,
        )

    is_implementation = last.get("role") == "user" and is_implementation_command(last.get("content"))
    if is_implementation:
        responder = StreamingResponder() if request.stream else NonStreamingResponder()
        return responder.reply(
            "New implementation requires an approved Work Package plan. "
            "Start with `/implement WP1` in a conversation with an active IDE session; "
            "nothing was executed.", request.model, created)

    is_task_run = last.get("role") == "user" and is_run_command(last.get("content"))
    if is_task_run:
        responder = StreamingResponder() if request.stream else NonStreamingResponder()
        task_id = extract_run_task_id(last.get("content") or "")
        return responder.reply(
            f"Start with `/implement {task_id.split('-T', 1)[0]}` so the Work Package "
            f"is planned before {task_id}; nothing was executed.", request.model, created)

    # An implementation tool round-trip resumes its persisted sequential job.
    if request.stream and results:
        run = bridge.run_for(results[-1].tool_call_id)
        if run is not None and run.metadata.get("implementation_job_id"):
            service = _resume_service(run, session, active_specs)
            events = service.resume(run, messages, active_specs)
            return StreamingResponder().drain_events(events, request.model, created)

    # A /run tool round-trip is resumed by the executor that started it, rather
    # than by LLM_PROVIDER/CODER_PROVIDER. This keeps a recorded Codex decision
    # from silently continuing under Claude (or the reverse).
    if request.stream and results:
        run = bridge.run_for(results[-1].tool_call_id)
        if run is not None and run.metadata.get("task_execution"):
            events = task_execution.resume(run, messages, active_specs)
            return StreamingResponder().drain_events(events, request.model, created)

    provider = providers.active()

    # TOOL TURN: hand the full resent history to the active provider's uniform
    # `tools()` and drain its neutral events. Every stateful/stateless detail —
    # start vs. resume, background run vs. one-shot — is private to the provider,
    # so this controller (and the engine) stay provider-agnostic. This is a coding
    # agent talking straight to the IDE, so it runs as ROLE_CODER — stated explicitly rather than inheriting the un-roled ceiling.
    if request.stream and (active_specs or results):
        if not results and last.get("role") == "user" and not _workflow_replay.get():
            # A fresh user turn reaches the coding agent only through a confirmed
            # proposal. Without conversation identity there is no proposal boundary.
            return StreamingResponder().reply(
                "CodePartner could not establish this conversation's identity or IDE MCP "
                "session, so it will not run the coding agent directly. Nothing was "
                "executed. Enable tool calls and send the request again.",
                request.model, created)
        if results:
            # Resuming a paused run: the provider continues the SAME run, native or not.
            events = provider.tools(messages, active_specs, role=ROLE_CODER,
                                    direct_tool_handlers=direct_handlers)
            return StreamingResponder().drain_events(events, request.model, created)
        return _native_tools(False, messages, active_specs, ROLE_CODER, direct_handlers,
                             request, created)

    # PLAIN CHAT: no tools advertised and no tool results → straight completion.
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    prompt, system = oreq.build_prompt(request)
    try:
        return await responder.build_completion(prompt, system, request.model, created)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{provider.NAME} call failed: {exc}") from exc

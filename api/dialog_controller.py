"""HTTP controller for the OpenAI-compatible chat endpoint.

Thin: it routes (tool turn / plain chat) and translates between OpenAI wire types
and the neutral inputs the active provider consumes. Each provider exposes one
uniform `tools(messages, specs)` surface, so the controller never branches on
which provider is active — every stateful/stateless detail is private to it.
OpenAI request parsing lives in `conversation/openai_request.py`; the engine
(driven from inside the Claude provider) lives in `mcp_bridge/`.
"""
import logging
import re
import time
from collections import deque
from uuid import uuid4

from fastapi import APIRouter, HTTPException

import providers
from agent import pipeline
from agent.spec import (
    ArtifactSelectionError,
    ArtifactStoreError,
    CliOpenSpecAdapter,
    OpenSpecArtifactStore,
    OpenSpecError,
    SpecCommandError,
    SpecCommandHandler,
    extract_help_response,
    extract_implementation_request,
    extract_run_task_id,
    extract_scoped_request,
    is_help_command,
    is_implementation_command,
    is_review_command,
    is_run_command,
    is_rework_command,
    is_spec_command,
    is_update_command,
)
from agent.spec.bridge_client import BridgeMcpFileClient, DirectMcpFileClient
from agent.spec.ide_mcp_sessions import (
    configure_session, conversation_for_request, get_session, new_conversation_id,
    remember_bootstrap_call, remember_bootstrap_from_history, strip_bootstrap_exchange,
)
from agent.routing import ImplementationService, TaskExecutionService
from conversation import openai_request as oreq
from conversation.non_streaming_responder import NonStreamingResponder
from conversation.streaming_responder import StreamingResponder
from mcp_bridge.registry import (
    DOMAIN_SOURCE, OP_FILE_CREATE, OP_FILE_READ, OP_FILE_SEARCH, ToolRegistry,
    ROLE_CODER, ROLE_OPTIMIZER, classify_tool, has_existing_source_editor,
)
from mcp_bridge import bridge
from mcp_bridge import server as bridge_server
from mcp_bridge.models import DoneEvent, Event, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec
from model.chat import ChatCompletionRequest, ChatCompletionResponse
from logger.diagnostic import debug, new_trace_id

log = logging.getLogger("codepartner.api")

router = APIRouter()
artifact_store = OpenSpecArtifactStore()
spec_handler = SpecCommandHandler(CliOpenSpecAdapter(), artifact_store)
task_execution = TaskExecutionService()
implementation = ImplementationService(artifact_store, task_execution)

# Artifact commands state what they need; the registry resolves compatible
# aliases/adapters from the current IDE advertisement.  No tool wire name is a
# policy here.
_ARTIFACT_OPERATIONS = frozenset({OP_FILE_READ, OP_FILE_SEARCH, OP_FILE_CREATE})
_RECENT_REQUESTS: deque[dict] = deque(maxlen=64)


def _artifact_store(client, specs):
    return OpenSpecArtifactStore(DirectMcpFileClient(client, specs))


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
        detail=(f"{command} requires the request-scoped IDE MCP bridge for target-project artifacts. "
                "Run it from the intended IDE project with compatible file tools advertised."),
    )


def _tool_calls_permitted(request: ChatCompletionRequest) -> bool:
    choice = request.tool_choice
    if choice == "none":
        return False
    return not (isinstance(choice, dict) and choice.get("type") == "none")


def _artifact_tools_compatible(specs: list[ToolSpec]) -> bool:
    """Check that the configured server can read, search, and persist artifacts."""
    registry = ToolRegistry(allowed_operations=_ARTIFACT_OPERATIONS)
    registry.register(specs)
    try:
        registry.prepare_call(OP_FILE_READ, {"file_path": ".codepartner/spec-index.json"})
        registry.prepare_call(OP_FILE_SEARCH, {"q": ".codepartner/spec-index.json", "limit": 2})
        registry.prepare_call(OP_FILE_CREATE, {
            "pathInProject": ".codepartner/.bridge-check",
            "text": "",
            "overwrite": True,
        })
    except (LookupError, ValueError):
        return False
    return True


async def _blocked_capability_response(request: ChatCompletionRequest, operation: str,
                                       specs: list[ToolSpec], created: int):
    """Return a normal OpenAI reply for a well-formed but unexecutable command."""
    decisions = [classify_tool(spec) for spec in specs]
    names = ", ".join(spec.name for spec in specs) or "none"
    supported = ", ".join(decision.name for decision in decisions if decision.accepted) or "none"
    reason = ("No supported source-write or refactor tool was discovered from this chat's IDE MCP server. "
              f"Discovered tools: {names}. Supported definitions: {supported}. "
              "No changes were applied. Expose a supported write/refactor MCP tool in the IDE server.")
    operation_id = str(uuid4())
    await task_execution.persistence.record_blocked_operation(operation_id, operation, reason)
    log.info("capability_blocked operation=%s operation_id=%s advertised_count=%d supported_count=%d",
             operation, operation_id, len(specs), len([d for d in decisions if d.accepted]))
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    return responder.reply(reason, request.model, created)


def _requires_mcp(request: ChatCompletionRequest, last: dict) -> bool:
    if last.get("role") == "user":
        content = (last.get("content") or "").lstrip().lower()
        if content.startswith(("/spec", "/update", "/run", "/implement", "/execute",
                               "/review", "/rework")):
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
                            client):
    """Run OpenSpec persistence through the conversation's configured MCP server."""
    async def strategy(run, _prompt, _system, _specs):
        store = _artifact_store(client, specs)
        handler = SpecCommandHandler(CliOpenSpecAdapter(), store)
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


def _start_rework_command(specs: list[ToolSpec], selectors: list[str],
                          feedback: str, trace_id: str, client):
    """Create corrective work using the conversation's configured MCP server."""
    async def strategy(run, _prompt, _system, _specs):
        store = _artifact_store(client, specs)
        handler = SpecCommandHandler(CliOpenSpecAdapter(), store)
        targets = await store.tasks_for_selectors(selectors, include_completed=True)
        result = await handler.rework(targets, feedback)
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


def _start_run_command(messages: list[dict], specs: list[ToolSpec], task_id: str,
                       trace_id: str, client, direct_tool_handlers: dict):
    """Load and execute a task through the conversation's configured MCP server."""
    async def strategy(run, _prompt, _system, _specs):
        store = _artifact_store(client, specs)
        change, task = await store.task_by_id(task_id)
        async for event in task_execution.execute(
                task, change.id, messages, specs,
                direct_tool_handlers=direct_tool_handlers):
            await run.put(event)
            if isinstance(event, DoneEvent):
                return "endTurn"
            # A child provider Run now owns the pending tool call and will be
            # resumed by its call ID.  This catalog-loading Run can end safely.
            if getattr(event, "tool_call_id", None):
                child = bridge.run_for(event.tool_call_id)
                if child is not None:
                    child.metadata["parent_command_run"] = run
                    await child.task
                return "endTurn"
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    # ##DELETE AFTER CORRECTION##
    run.metadata["trace_id"] = trace_id
    run.metadata["bridge_command"] = "run"
    debug(log, "api.run_command_started", trace_id=trace_id, run_id=run.run_id,
          task_id=task_id, advertised_tool_names=[spec.name for spec in specs])
    return run


def _start_implementation_command(messages: list[dict], specs: list[ToolSpec],
                                   selectors: list[str], instruction: str, trace_id: str,
                                   client, direct_tool_handlers: dict,
                                   conversation_id: str | None):
    """Keep task selection and source execution on the configured MCP server."""
    async def strategy(run, _prompt, _system, _specs):
        store = _artifact_store(client, specs)
        service = ImplementationService(store, task_execution)
        job = await service.create_job(selectors, instruction, conversation_id)
        await _consume_implementation_events(
            run, service, service.start(job, messages, specs, direct_tool_handlers),
            messages, specs)
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    # ##DELETE AFTER CORRECTION##
    run.metadata["trace_id"] = trace_id
    run.metadata["bridge_command"] = "implement"
    debug(log, "api.implementation_command_started", trace_id=trace_id, run_id=run.run_id,
          selector_count=len(selectors), has_instruction=bool(instruction),
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
        service = ImplementationService(_artifact_store(client, specs), task_execution)
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
        service = ImplementationService(_artifact_store(client, specs), task_execution)
        await _consume_implementation_events(
            run, service, service.retry_review(job_id, position, messages, specs,
                                               instruction, direct_tool_handlers),
            messages, specs)
        return "endTurn"

    run = bridge_server.start_run("", None, specs, strategy,
                                  allowed_operations=_ARTIFACT_OPERATIONS)
    run.metadata.update({"trace_id": trace_id, "bridge_command": "retry_implementation_review"})
    return run


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    created = int(time.time())

    results = oreq.tool_results(request)
    conversation_id = conversation_for_request(request)
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
        return await chat_completions(request)

    session = get_session(conversation_id)
    trace_id = (resumed_run.metadata.get("trace_id") if resumed_run else None) or new_trace_id()
    audit = _advertisement_audit(request, trace_id)
    debug(log, "api.request_advertisement", **audit,
          resumed_run_id=resumed_run.run_id if resumed_run else None)
    messages = strip_bootstrap_exchange(oreq.to_messages(request))
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

    is_help = last.get("role") == "user" and is_help_command(last.get("content"))
    needs_mcp = session is None and not is_help and _requires_mcp(request, last)
    if needs_mcp and not request.stream:
        raise HTTPException(status_code=400, detail="IDE MCP setup requires a streaming request")
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
    if session is not None and not request.stream and not is_help and _requires_mcp(request, last):
        raise HTTPException(status_code=400, detail="IDE MCP operations require a streaming request")

    if request.stream and oreq.tool_results(request):
        result = oreq.tool_results(request)[-1]
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

    pending_confirmation = await task_execution.persistence.pending_confirmation(conversation_id)
    if pending_confirmation is None and last.get("role") == "user":
        for prior in reversed(messages[:-1]):
            if prior.get("role") != "assistant":
                continue
            match = re.search(r"\bjob ([0-9a-f-]{36})\b", prior.get("content") or "")
            if match:
                pending_confirmation = await task_execution.persistence.pending_confirmation_for_job(
                    match.group(1))
                if pending_confirmation:
                    break
    if pending_confirmation and last.get("role") == "user":
        reply = (last.get("content") or "").strip()
        normalized = reply.lower()
        approved = normalized in {"yes", "y", "approve", "confirm"}
        rejected = normalized in {"no", "n"} or normalized.startswith("no:")
        if approved or rejected:
            if not request.stream:
                raise HTTPException(status_code=400, detail="Implementation confirmation requires a streaming request")
            if session is None:
                raise HTTPException(status_code=409, detail="IDE MCP session required to resume implementation")
            job_id, position, _, _, instruction = pending_confirmation
            run = _start_implementation_confirmation(
                messages, active_specs, trace_id, session.client, direct_handlers,
                job_id, position, approved, reply, instruction)
            return StreamingResponder().drain_events(_drain_run(run), request.model, created)

    if last.get("role") == "user" and (last.get("content") or "").strip().lower() in {
            "retry review", "/retry-review"}:
        pending_review = await task_execution.persistence.pending_review(conversation_id)
        if pending_review is not None:
            if not request.stream or session is None:
                raise HTTPException(status_code=409, detail="IDE MCP session required to retry review")
            job_id, position, _, instruction = pending_review
            run = _start_implementation_review_retry(
                messages, active_specs, trace_id, session.client, direct_handlers,
                job_id, position, instruction)
            return StreamingResponder().drain_events(_drain_run(run), request.model, created)

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
        if not _artifact_tools_compatible(active_specs):
            raise _bridge_required_response("/rework")
        run = _start_rework_command(active_specs, selectors, feedback, trace_id, session.client)
        return StreamingResponder().drain_events(_drain_run(run), request.model, created)

    is_review = last.get("role") == "user" and is_review_command(last.get("content"))
    if is_review:
        if not request.stream:
            raise HTTPException(status_code=400, detail="/review requires a streaming request")
        try:
            selectors, question = extract_scoped_request(last.get("content") or "", "review")
            targets = await _artifact_store(session.client, active_specs).tasks_for_selectors(
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
                "the configured IDE MCP read tools. Never write, edit, create, format, or refactor files. "
                "Report evidence, gaps against the selected OpenSpec tasks, and suggested follow-up work."
            )},
            {"role": "user", "content": (
                f"Review these selected OpenSpec tasks and their implementation:\n{task_context}\n\n"
                f"Review focus: {question or 'Assess conformance, tests, and likely gaps.'}"
            )},
        ]
        events = providers.active().tools(
            review_messages, active_specs, role=ROLE_OPTIMIZER,
            direct_tool_handlers=direct_handlers,
        )
        return StreamingResponder().drain_events(events, request.model, created)
    is_planning = last.get("role") == "user" and (
            is_spec_command(last.get("content"))
            or is_update_command(last.get("content"))
    )
    if is_planning:
        if not request.stream:
            raise HTTPException(status_code=400, detail="/spec and /update require a streaming request")
        return StreamingResponder().drain_events(
            _drain_run(_start_artifact_command(
                messages, active_specs, trace_id, session.client)),
            request.model, created,
        )

    is_implementation = last.get("role") == "user" and is_implementation_command(last.get("content"))
    if is_implementation:
        if not request.stream:
            raise HTTPException(status_code=400, detail="/implement requires a streaming request")
        try:
            if not _tool_calls_permitted(request):
                return await _blocked_capability_response(request, "implement", active_specs, created)
            if not _artifact_tools_compatible(active_specs):
                raise _bridge_required_response("/implement")
            content = last.get("content") or ""
            selectors, instruction = extract_implementation_request(content)
            log.info("implementation_command selector_count=%d has_instruction=%s",
                     len(selectors), bool(instruction))
            return StreamingResponder().drain_events(
                _drain_run(_start_implementation_command(
                    messages, active_specs, selectors, instruction, trace_id,
                    session.client, direct_handlers, conversation_id)),
                request.model, created,
            )
        except SpecCommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ArtifactSelectionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ArtifactStoreError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    is_task_run = last.get("role") == "user" and is_run_command(last.get("content"))
    if is_task_run:
        if not request.stream:
            raise HTTPException(status_code=400, detail="/run requires a streaming request")
        try:
            if not _tool_calls_permitted(request) or not has_existing_source_editor(active_specs):
                return await _blocked_capability_response(request, "run", active_specs, created)
            task_id = extract_run_task_id(last.get("content") or "")
            return StreamingResponder().drain_events(
                _drain_run(_start_run_command(
                    messages, active_specs, task_id, trace_id, session.client, direct_handlers)),
                request.model, created,
            )
        except SpecCommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ArtifactSelectionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ArtifactStoreError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # An implementation tool round-trip resumes its persisted sequential job.
    if request.stream and oreq.tool_results(request):
        run = bridge.run_for(oreq.tool_results(request)[-1].tool_call_id)
        if run is not None and run.metadata.get("implementation_job_id"):
            service = run.metadata.get("implementation_service") or implementation
            events = service.resume(run, messages, active_specs)
            return StreamingResponder().drain_events(events, request.model, created)

    # A /run tool round-trip is resumed by the executor that started it, rather
    # than by LLM_PROVIDER/CODER_PROVIDER. This keeps a recorded Codex decision
    # from silently continuing under Claude (or the reverse).
    if request.stream and oreq.tool_results(request):
        run = bridge.run_for(oreq.tool_results(request)[-1].tool_call_id)
        if run is not None and run.metadata.get("task_execution"):
            events = task_execution.resume(run, messages, active_specs)
            return StreamingResponder().drain_events(events, request.model, created)

    provider = providers.active()

    # OPTIMIZER PIPELINE (opt-in): when enabled, tool-advertising streams are driven
    # by the workflow (Optimizer → human accept → Coder) instead of a single provider.
    # The pipeline yields the same neutral events, so drain_events serializes it too.
    if pipeline.is_enabled() and request.stream and active_specs:
        return StreamingResponder().drain_events(
            pipeline.run(request, active_specs, direct_handlers), request.model, created
        )

    # TOOL TURN: hand the full resent history to the active provider's uniform
    # `tools()` and drain its neutral events. Every stateful/stateless detail —
    # start vs. resume, background run vs. one-shot — is private to the provider,
    # so this controller (and the engine) stay provider-agnostic. With the pipeline
    # off this is a coding agent talking straight to the IDE, so it runs as
    # ROLE_CODER — stated explicitly rather than inheriting the un-roled ceiling.
    if request.stream and (active_specs or oreq.tool_results(request)):
        events = provider.tools(
            messages, active_specs, role=ROLE_CODER,
            direct_tool_handlers=direct_handlers,
        )
        return StreamingResponder().drain_events(events, request.model, created)

    # PLAIN CHAT: no tools advertised and no tool results → straight completion.
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    prompt, system = oreq.build_prompt(request)
    try:
        return await responder.build_completion(prompt, system, request.model, created)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{provider.NAME} call failed: {exc}") from exc

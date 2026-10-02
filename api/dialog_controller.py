"""HTTP controller for the OpenAI-compatible chat endpoint.

Thin: it routes (tool turn / plain chat) and translates between OpenAI wire types
and the neutral inputs the active provider consumes. Each provider exposes one
uniform `tools(messages, specs)` surface, so the controller never branches on
which provider is active — every stateful/stateless detail is private to it.
OpenAI request parsing lives in `conversation/openai_request.py`; the engine
(driven from inside the Claude provider) lives in `mcp_bridge/`.
"""
import logging
import time

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
    extract_implementation_selectors,
    extract_run_task_id,
    is_implementation_command,
    is_run_command,
    is_spec_command,
    is_update_command,
)
from agent.routing import ImplementationService, TaskExecutionService
from conversation import openai_request as oreq
from conversation.non_streaming_responder import NonStreamingResponder
from conversation.streaming_responder import StreamingResponder
from mcp_bridge.registry import ROLE_CODER
from mcp_bridge import bridge
from model.chat import ChatCompletionRequest, ChatCompletionResponse

log = logging.getLogger("mcp_bridge")

router = APIRouter()
artifact_store = OpenSpecArtifactStore()
spec_handler = SpecCommandHandler(CliOpenSpecAdapter(), artifact_store)
task_execution = TaskExecutionService()
implementation = ImplementationService(artifact_store, task_execution)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    created = int(time.time())

    messages = oreq.to_messages(request)
    last = messages[-1] if messages else {}
    if request.tools:
        log.info("tools received from JetBrains: %s", oreq.advertised_names(request))
    is_planning = last.get("role") == "user" and (
            is_spec_command(last.get("content"))
            or is_update_command(last.get("content"))
    )
    if is_planning:
        responder = StreamingResponder() if request.stream else NonStreamingResponder()
        try:
            result = await spec_handler.handle(messages)
        except SpecCommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ArtifactStoreError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except OpenSpecError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return responder.reply(result.content, request.model, created)

    is_implementation = last.get("role") == "user" and is_implementation_command(last.get("content"))
    if is_implementation:
        if not request.stream:
            raise HTTPException(status_code=400, detail="/implement requires a streaming request")
        try:
            selectors = extract_implementation_selectors(last.get("content") or "")
            job = await implementation.create_job(selectors)
        except SpecCommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ArtifactSelectionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ArtifactStoreError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        events = implementation.start(job, messages, oreq.tool_specs(request))
        return StreamingResponder().drain_events(events, request.model, created)

    is_task_run = last.get("role") == "user" and is_run_command(last.get("content"))
    if is_task_run:
        if not request.stream:
            raise HTTPException(status_code=400, detail="/run requires a streaming request")
        try:
            task_id = extract_run_task_id(last.get("content") or "")
            change, task = await artifact_store.task_by_id(task_id)
        except SpecCommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ArtifactSelectionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ArtifactStoreError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        events = task_execution.execute(task, change.id, messages, oreq.tool_specs(request))
        return StreamingResponder().drain_events(events, request.model, created)

    # An implementation tool round-trip resumes its persisted sequential job.
    if request.stream and oreq.tool_results(request):
        run = bridge.run_for(oreq.tool_results(request)[-1].tool_call_id)
        if run is not None and run.metadata.get("implementation_job_id"):
            events = implementation.resume(run, messages, oreq.tool_specs(request))
            return StreamingResponder().drain_events(events, request.model, created)

    # A /run tool round-trip is resumed by the executor that started it, rather
    # than by LLM_PROVIDER/CODER_PROVIDER. This keeps a recorded Codex decision
    # from silently continuing under Claude (or the reverse).
    if request.stream and oreq.tool_results(request):
        run = bridge.run_for(oreq.tool_results(request)[-1].tool_call_id)
        if run is not None and run.metadata.get("task_execution"):
            events = task_execution.resume(run, messages, oreq.tool_specs(request))
            return StreamingResponder().drain_events(events, request.model, created)

    provider = providers.active()

    # OPTIMIZER PIPELINE (opt-in): when enabled, tool-advertising streams are driven
    # by the workflow (Optimizer → human accept → Coder) instead of a single provider.
    # The pipeline yields the same neutral events, so drain_events serializes it too.
    if pipeline.is_enabled() and request.stream and oreq.tool_specs(request):
        return StreamingResponder().drain_events(pipeline.run(request), request.model, created)

    # TOOL TURN: hand the full resent history to the active provider's uniform
    # `tools()` and drain its neutral events. Every stateful/stateless detail —
    # start vs. resume, background run vs. one-shot — is private to the provider,
    # so this controller (and the engine) stay provider-agnostic. With the pipeline
    # off this is a coding agent talking straight to the IDE, so it runs as
    # ROLE_CODER — stated explicitly rather than inheriting the un-roled ceiling.
    if request.stream and (request.tools or oreq.tool_results(request)):
        events = provider.tools(messages, oreq.tool_specs(request), role=ROLE_CODER)
        return StreamingResponder().drain_events(events, request.model, created)

    # PLAIN CHAT: no tools advertised and no tool results → straight completion.
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    prompt, system = oreq.build_prompt(request)
    try:
        return await responder.build_completion(prompt, system, request.model, created)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{provider.NAME} call failed: {exc}") from exc

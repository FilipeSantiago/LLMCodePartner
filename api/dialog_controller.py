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
from conversation import openai_request as oreq
from conversation.non_streaming_responder import NonStreamingResponder
from conversation.streaming_responder import StreamingResponder
from model.chat import ChatCompletionRequest, ChatCompletionResponse

log = logging.getLogger("mcp_bridge")

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    created = int(time.time())

    provider = providers.active()

    if request.tools:
        log.info("tools received from JetBrains: %s", oreq.advertised_names(request))

    # OPTIMIZER PIPELINE (opt-in): when enabled, tool-advertising streams are driven
    # by the workflow (Optimizer → human accept → Coder) instead of a single provider.
    # The pipeline yields the same neutral events, so drain_events serializes it too.
    if pipeline.is_enabled() and request.stream and oreq.tool_specs(request):
        return StreamingResponder().drain_events(pipeline.run(request), request.model, created)

    # TOOL TURN: hand the full resent history to the active provider's uniform
    # `tools()` and drain its neutral events. Every stateful/stateless detail —
    # start vs. resume, background run vs. one-shot — is private to the provider,
    # so this controller (and the engine) stay provider-agnostic.
    if request.stream and (request.tools or oreq.tool_results(request)):
        messages = oreq.to_messages(request)
        events = provider.tools(messages, oreq.tool_specs(request))
        return StreamingResponder().drain_events(events, request.model, created)

    # PLAIN CHAT: no tools advertised and no tool results → straight completion.
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    prompt, system = oreq.build_prompt(request)
    try:
        return await responder.build_completion(prompt, system, request.model, created)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{provider.NAME} call failed: {exc}") from exc

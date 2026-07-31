"""HTTP controller for the OpenAI-compatible chat endpoint.

Thin: it routes (resume / start-bridge / plain chat) and acts as the composition
root — it injects the active provider's tool strategy (`providers.active().run`)
into the provider-agnostic MCP engine. OpenAI request parsing lives in
`conversation/openai_request.py`; the engine lives in `mcp_bridge/`.
"""
import logging
import time

from fastapi import APIRouter, HTTPException

import providers
from conversation import openai_request as oreq
from conversation.non_streaming_responder import NonStreamingResponder
from conversation.streaming_responder import StreamingResponder
from mcp_bridge import server as engine
from model.chat import ChatCompletionRequest, ChatCompletionResponse

log = logging.getLogger("mcp_bridge")

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    created = int(time.time())

    provider = providers.active()

    if request.tools:
        log.info("tools received from JetBrains: %s", oreq.advertised_names(request))

    # STATELESS tool calling (Ollama): re-run the model from the full resent
    # history each request. No background run / futures / resume — the client
    # (JetBrains) drives the loop by resending the conversation with tool results.
    if provider.NAME == "ollama" and request.stream and request.tools:
        messages = oreq.to_messages(request)
        events = provider.tool_chat(messages, oreq.all_tool_specs(request))
        return StreamingResponder().tool_chat_response(events, request.model, created)

    # RESUME: the client returned one or more tool results. Resolve each pending
    # future by tool_call_id and continue streaming the same execution.
    results = oreq.tool_results(request)
    if results:
        run = None
        for m in results:
            run = engine.resume(m.tool_call_id, m.content or "") or run
        if run is None:
            return StreamingResponder().reply(
                "No pending tool call matched this result (it may have timed out).",
                request.model, created,
            )
        return StreamingResponder().drain(run, request.model, created)

    # START BRIDGE: streaming request advertising supported tools. The active
    # provider's tool strategy is injected here so the engine stays agnostic.
    if request.stream and oreq.supported_names(request):
        prompt, system = oreq.build_prompt(request)
        run = engine.start_run(prompt, system, oreq.tool_specs(request), strategy=providers.active().run)
        return StreamingResponder().drain(run, request.model, created)

    # PLAIN CHAT: no supported tools → tool calling disabled, unchanged behavior.
    responder = StreamingResponder() if request.stream else NonStreamingResponder()
    prompt, system = oreq.build_prompt(request)
    try:
        return await responder.build_completion(prompt, system, request.model, created)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Claude call failed: {exc}") from exc

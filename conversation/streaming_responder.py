"""Engine → OpenAI outbound adapter (streaming).

`drain_events()` is the provider-events → OpenAI SSE side of the bridge; it
consumes a provider's neutral tool-turn events and never touches the MCP engine's
internals beyond them.
"""
import json
import logging

from starlette.responses import StreamingResponse

import providers
from conversation.chat_responder import ChatResponder
from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent, ToolCallEvent
from logger.diagnostic import debug

log = logging.getLogger("codepartner.openai.sse")


class StreamingResponder(ChatResponder):
    """Streaming delivery: returns a `text/event-stream` of `chat.completion.chunk`
    Server-Sent Events, emitted incrementally, terminated by `data: [DONE]`."""

    def __init__(self):
        super().__init__()

    async def build_completion(self, prompt: str, system: str | None, model: str,
                                created: int) -> StreamingResponse:
        async def event_stream():
            first = True
            terminal = "endTurn"
            try:
                async for text, term in providers.active().stream(prompt, system):
                    if term is not None:
                        terminal = term
                        continue
                    if not text:
                        continue
                    delta = {"role": "assistant", "content": text} if first else {"content": text}
                    first = False
                    yield self._chunk(delta, model, created, None)
            except Exception as exc:
                msg = f"[error: {exc}]"
                delta = {"role": "assistant", "content": msg} if first else {"content": msg}
                yield self._chunk(delta, model, created, None)

            yield self._chunk({}, model, created, self._finish_reason(terminal))
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def drain_events(self, events, model: str, created: int) -> StreamingResponse:
        """Translate a provider's neutral tool-turn events into OpenAI SSE.

        `events` is an async iterator of neutral event dataclasses, produced
        identically by every provider's `tools()` — stateful (Claude, draining a
        bridged Run) or stateless (Ollama, one-shot). A `ToolCallEvent` ends the
        stream with finish_reason "tool_calls"; a `DoneEvent` ends it normally.
        """
        async def event_stream():
            first = True
            terminal = "endTurn"
            usage = None
            try:
                async for event in events:
                    if isinstance(event, TextEvent):
                        if not event.text:
                            continue
                        delta = ({"role": "assistant", "content": event.text}
                                 if first else {"content": event.text})
                        first = False
                        yield self._chunk(delta, model, created, None)
                    elif isinstance(event, ErrorEvent):
                        msg = f"[error: {event.message}]"
                        delta = ({"role": "assistant", "content": msg}
                                 if first else {"content": msg})
                        first = False
                        yield self._chunk(delta, model, created, None)
                    elif isinstance(event, ToolCallEvent):
                        # ##DELETE AFTER CORRECTION## Exact OpenAI SSE tool-call wire payload.
                        chunk = self._tool_call_chunk(event, model, created, first)
                        debug(log, "sse.tool_call_emitted", model=model,
                              tool_call_id=event.tool_call_id, tool_name=event.name,
                              arguments=event.arguments, chunk=chunk)
                        yield chunk
                        terminal_chunk = self._chunk({}, model, created, "tool_calls")
                        debug(log, "sse.tool_call_terminated", tool_call_id=event.tool_call_id,
                              chunk=terminal_chunk)
                        yield terminal_chunk
                        yield "data: [DONE]\n\n"
                        return
                    elif isinstance(event, DoneEvent):
                        terminal = event.terminal
                        usage = event.usage
                        break
            except Exception as exc:
                msg = f"[error: {exc}]"
                delta = {"role": "assistant", "content": msg} if first else {"content": msg}
                yield self._chunk(delta, model, created, None)

            yield self._chunk({}, model, created, self._finish_reason(terminal))
            if usage:
                yield self._usage_chunk(usage, model, created)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def _tool_call_chunk(self, event: ToolCallEvent, model: str, created: int, first: bool) -> str:
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": event.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": event.name,
                        "arguments": json.dumps(event.arguments),
                    },
                }
            ]
        }
        if first:
            delta["role"] = "assistant"
        return self._chunk(delta, model, created, None)

    def _usage_chunk(self, usage: dict, model: str, created: int) -> str:
        """A data-only terminal chunk carrying token usage (OpenAI's `include_usage`
        shape: `choices: []` plus a `usage` object). Best-effort normalization across
        providers — Anthropic (`input_tokens`/`output_tokens`) and Ollama
        (`inputTokens`/`outputTokens`) — with the raw provider usage kept under `raw`."""
        prompt = usage.get("input_tokens") or usage.get("inputTokens")
        completion = usage.get("output_tokens") or usage.get("outputTokens")
        total = (prompt or 0) + (completion or 0) if (prompt or completion) else None
        payload = {
            "id": "chatcmpl-dumb-router",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "raw": usage,
            },
        }
        return f"data: {json.dumps(payload)}\n\n"

    def reply(self, text: str, model: str, created: int) -> StreamingResponse:
        async def event_stream():
            yield self._chunk({"role": "assistant", "content": text}, model, created, None)
            yield self._chunk({}, model, created, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

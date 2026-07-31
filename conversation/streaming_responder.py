"""Engine → OpenAI outbound adapter (streaming).

`drain()` is the engine-events → OpenAI SSE side of the bridge; it consumes a
`Run`'s neutral events and never touches the MCP engine's internals beyond them.
"""
import json

from starlette.responses import StreamingResponse

import providers
from conversation.chat_responder import ChatResponder
from mcp_bridge.models import DoneEvent, ErrorEvent, Run, TextEvent, ToolCallEvent


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

    def reply(self, text: str, model: str, created: int) -> StreamingResponse:
        async def event_stream():
            yield self._chunk({"role": "assistant", "content": text}, model, created, None)
            yield self._chunk({}, model, created, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def drain(self, run: Run, model: str, created: int) -> StreamingResponse:
        """Translate a bridged Run's neutral events into OpenAI SSE.

        Used for both request A (fresh run — stops at the tool_call) and request B
        (resumed run — streams the final answer). A `ToolCallEvent` ends the stream
        with finish_reason "tool_calls"; a `DoneEvent` ends it normally.
        """
        async def event_stream():
            first = True
            terminal = "endTurn"
            while True:
                event = await run.next_event()
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
                    yield self._tool_call_chunk(event, model, created, first)
                    yield self._chunk({}, model, created, "tool_calls")
                    yield "data: [DONE]\n\n"
                    return
                elif isinstance(event, DoneEvent):
                    terminal = event.terminal
                    break
            yield self._chunk({}, model, created, self._finish_reason(terminal))
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def tool_chat_response(self, events, model: str, created: int) -> StreamingResponse:
        """Stateless tool turn → OpenAI SSE. `events` is an async iterator yielding
        ("text", chunk) or a single ("tool_calls", [{id,name,arguments}...]).
        Tool calls end the turn with finish_reason "tool_calls"; text ends "stop".
        """
        async def event_stream():
            first = True
            try:
                async for kind, payload in events:
                    if kind == "text":
                        if not payload:
                            continue
                        delta = ({"role": "assistant", "content": payload}
                                 if first else {"content": payload})
                        first = False
                        yield self._chunk(delta, model, created, None)
                    elif kind == "tool_calls":
                        delta = {
                            "tool_calls": [
                                {"index": i, "id": c["id"], "type": "function",
                                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                for i, c in enumerate(payload)
                            ]
                        }
                        if first:
                            delta["role"] = "assistant"
                        yield self._chunk(delta, model, created, None)
                        yield self._chunk({}, model, created, "tool_calls")
                        yield "data: [DONE]\n\n"
                        return
            except Exception as exc:
                msg = f"[error: {exc}]"
                delta = {"role": "assistant", "content": msg} if first else {"content": msg}
                yield self._chunk(delta, model, created, None)

            yield self._chunk({}, model, created, "stop")
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

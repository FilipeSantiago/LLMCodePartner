import abc
import json
from abc import ABC

from starlette.responses import StreamingResponse

from model.chat import ChatCompletionResponse


class ChatResponder(ABC):
    """Renders a chat-completion request into an HTTP response.

    Subclasses differ only in *delivery mode* (the OpenAI `stream` flag), not in
    what they compute:
      - NonStreamingResponder → one complete `chat.completion` JSON object.
      - StreamingResponder    → a stream of `chat.completion.chunk` SSE events,
                                 terminated by `data: [DONE]`.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    async def build_completion(
        self, prompt: str, system: str | None, model: str, created: int
    ) -> ChatCompletionResponse | StreamingResponse:
        pass

    @abc.abstractmethod
    def reply(self, text: str, model: str, created: int) -> ChatCompletionResponse | StreamingResponse:
        pass

    def _finish_reason(self, terminal: str | None) -> str:
        if terminal == "maxTurnsReached":
            return "length"
        return "stop"

    def _chunk(self, delta: dict, model: str, created: int, finish_reason: str | None) -> str:
        payload = {
            "id": "chatcmpl-dumb-router",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

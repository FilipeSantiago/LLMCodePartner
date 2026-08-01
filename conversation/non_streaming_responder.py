import providers
from conversation.chat_responder import ChatResponder
from model.chat import ChatCompletionResponse, ChatCompletionChoice, ChatMessage


class NonStreamingResponder(ChatResponder):
    """Non-streaming delivery: returns one complete `chat.completion` JSON object,
    all at once, after the active provider finishes."""

    def __init__(self):
        super().__init__()

    async def build_completion(self,
            prompt: str, system: str | None, model: str, created: int
    ) -> ChatCompletionResponse:
        text, terminal = await providers.active().complete(prompt, system)
        return ChatCompletionResponse(
            id="chatcmpl-dumb-router",
            created=created,
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=self._finish_reason(terminal),
                )
            ],
        )

    def reply(self, text: str, model: str, created: int) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-dumb-router",
            created=created,
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
        )

from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    # Tool-calling fields (present on assistant tool_calls and role:"tool" messages).
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    # Preserved as received from JetBrains; the supported tool is exposed to Claude via MCP.
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    # A client may supply an opaque request identifier.  Code Partner uses it only
    # for diagnostic correlation; it is never treated as a project selector.
    user: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]

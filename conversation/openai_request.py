"""OpenAI → engine inbound adapter.

The single place that understands the OpenAI request shape. It turns a
`ChatCompletionRequest` into the neutral inputs the MCP engine consumes
(`prompt`, `system`, `ToolSpec`s) and extracts tool results. The engine never
sees these OpenAI/pydantic types.
"""
from model.chat import ChatCompletionRequest, ChatMessage
from mcp_bridge.registry import ToolSpec

_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def _function(tool: dict) -> dict:
    # OpenAI shape is {"type":"function","function":{...}}; tolerate a flat dict.
    return tool.get("function") or tool


def build_prompt(request: ChatCompletionRequest) -> tuple[str, str | None]:
    """OpenAI messages → (prompt, system) for the LLM.

    Control/tool-plumbing messages are excluded: assistant `tool_calls` turns and
    `role:"tool"` results aren't prose (the LLM gets tool results via the resumed
    MCP handler, not the prompt).
    """
    system = "\n\n".join(
        m.content for m in request.messages if m.role == "system" and m.content
    ) or None
    convo = [
        m
        for m in request.messages
        if m.role in ("user", "assistant") and m.content and not m.tool_calls
    ]
    if len(convo) == 1:
        prompt = convo[0].content
    else:
        prompt = "\n\n".join(f"{m.role.capitalize()}: {m.content}" for m in convo)
    return prompt, system


def to_messages(request: ChatCompletionRequest) -> list[dict]:
    """Full conversation as a neutral, ordered message list (role-preserving).

    Stateless providers need the whole history — including assistant `tool_calls`
    turns and `role:"tool"` results — because the client (JetBrains) resends it
    every turn. Unlike `build_prompt`, nothing is dropped or flattened.
    """
    out: list[dict] = []
    for m in request.messages:
        msg: dict = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name
        out.append(msg)
    return out


def tool_specs(request: ChatCompletionRequest) -> list[ToolSpec]:
    """Every advertised tool → neutral ToolSpec (engine filters to SUPPORTED)."""
    specs = []
    for t in request.tools or []:
        fn = _function(t)
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters")
        specs.append(ToolSpec(
            name=name,
            description=fn.get("description") or name,
            schema=schema if isinstance(schema, dict) else _EMPTY_SCHEMA,
        ))
    return specs


def advertised_names(request: ChatCompletionRequest) -> list[str]:
    return [n for n in (_function(t).get("name") for t in request.tools or []) if n]


def tool_results(request: ChatCompletionRequest) -> list[ChatMessage]:
    """`role:"tool"` messages carrying a tool_call_id (client tool results)."""
    return [m for m in request.messages if m.role == "tool" and m.tool_call_id]

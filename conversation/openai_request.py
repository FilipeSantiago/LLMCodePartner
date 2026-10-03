"""OpenAI → engine inbound adapter.

The single place that understands the OpenAI request shape. It turns a
`ChatCompletionRequest` into the neutral inputs the MCP engine consumes
(`prompt`, `system`, `ToolSpec`s) and extracts tool results. The engine never
sees these OpenAI/pydantic types.
"""
import logging
from hashlib import sha256

from logger.diagnostic import debug
from model.chat import ChatCompletionRequest, ChatMessage
from mcp_bridge.registry import ToolSpec

log = logging.getLogger("codepartner.openai.request")


def safe_request_structure(request: ChatCompletionRequest) -> dict:
    """Return correlation metadata without retaining request/tool-result text."""
    roles = [message.role for message in request.messages]
    last = request.messages[-1] if request.messages else None
    command = "none"
    if last and last.role == "user" and last.content:
        token = last.content.lstrip().split(maxsplit=1)[0].lower()
        command = token if token in {
            "/spec", "/update", "/run", "/implement", "/execute", "/review", "/rework",
        } else "ordinary"
    has_results = any(message.role == "tool" and message.tool_call_id for message in request.messages)
    has_calls = any(message.tool_calls for message in request.messages)
    if has_results:
        origin = "tool_result_resume"
    elif command != "none" and command != "ordinary":
        origin = "primary_user_command"
    elif last and last.role == "user":
        origin = "primary_user_message"
    else:
        origin = "ide_helper_or_continuation"
    names = sorted(advertised_names(request))
    surface = sha256("\0".join(names).encode()).hexdigest()[:16]
    user_key = sha256((request.user or "").encode()).hexdigest()[:16] if request.user else None
    return {
        "origin": origin,
        "command": command,
        "message_roles": roles,
        "message_content_lengths": [len(message.content or "") for message in request.messages],
        "message_count": len(request.messages),
        "has_tool_calls": has_calls,
        "has_tool_results": has_results,
        "tool_count": len(names),
        "tool_names": names,
        "tool_surface_fingerprint": surface,
        "user_fingerprint": user_key,
    }


def _function(tool: dict) -> dict:
    # OpenAI shape is {"type":"function","function":{...}}; tolerate a flat dict.
    nested = tool.get("function")
    if isinstance(nested, dict):
        return nested
    return tool if isinstance(tool, dict) else {}


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
        # OpenAI calls this `parameters`; MCP-derived clients occasionally retain
        # either spelling of `inputSchema`.  Preserve the advertised wire name and
        # Preserve malformed/missing schemas.  The registry needs to distinguish
        # an advertised-but-incompatible tool from a valid empty object schema.
        schema = (fn.get("parameters") or fn.get("input_schema") or
                  fn.get("inputSchema"))
        specs.append(ToolSpec(
            name=name,
            description=fn.get("description") or name,
            schema=schema,
        ))
        debug(log, "openai.tool_parsed", tool_name=name,
              schema_type=type(schema).__name__, has_description=bool(fn.get("description")))
    debug(log, "openai.tool_specs_complete", tool_count=len(specs),
          tool_names=[spec.name for spec in specs])
    return specs


def advertised_names(request: ChatCompletionRequest) -> list[str]:
    return [n for n in (_function(t).get("name") for t in request.tools or []) if n]


def tool_results(request: ChatCompletionRequest) -> list[ChatMessage]:
    """`role:"tool"` messages carrying a tool_call_id (client tool results)."""
    return [m for m in request.messages if m.role == "tool" and m.tool_call_id]

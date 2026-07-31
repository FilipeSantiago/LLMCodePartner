"""Ollama provider (local Ollama server via HTTP).

Two capabilities:
  - plain chat: `stream` / `complete` over `/api/chat`.
  - stateless tool calling: `tool_chat(messages, specs)` runs ONE `/api/chat`
    turn over the full conversation and yields either streamed text or the tool
    calls the model wants. No background run / futures — the client (JetBrains)
    resends the whole history each turn, so each request is independent.
"""
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx

from mcp_bridge.registry import ToolSpec

log = logging.getLogger("mcp_bridge")

NAME = "ollama"

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")


def _terminal(done_reason: str | None) -> str:
    return "maxTurnsReached" if done_reason == "length" else "endTurn"


async def _chat(payload: dict[str, Any]):
    """Yield decoded NDJSON objects from a streaming /api/chat call."""
    # No read timeout: local generation (14B) can be slow.
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        async with client.stream("POST", f"{HOST}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield json.loads(line)


async def _chat_once(payload: dict[str, Any]) -> str:
    """Non-streaming /api/chat call → the assistant message content."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        resp = await client.post(f"{HOST}/api/chat", json=payload)
        resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content", "")


# --- plain chat (no tools) ---------------------------------------------------

def _messages(prompt: str, system: str | None) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


async def stream(prompt: str, system: str | None = None) -> AsyncIterator[tuple[str, str | None]]:
    payload = {"model": MODEL, "messages": _messages(prompt, system), "stream": True}
    async for obj in _chat(payload):
        chunk = (obj.get("message") or {}).get("content", "")
        if chunk:
            yield chunk, None
        if obj.get("done"):
            yield "", _terminal(obj.get("done_reason"))
            return


async def complete(prompt: str, system: str | None = None) -> tuple[str, str]:
    text, terminal = "", "endTurn"
    async for chunk, term in stream(prompt, system):
        if term is not None:
            terminal = term
        else:
            text += chunk
    return text, terminal


# --- stateless tool calling --------------------------------------------------

def _to_ollama_messages(messages: list[dict]) -> list[dict[str, Any]]:
    """Map the neutral OpenAI history (from openai_request.to_messages) to Ollama's
    /api/chat message shape. Tool results carry `tool_name` (Ollama's key), resolved
    from the assistant tool_calls that produced them."""
    id_to_name: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            tid = tc.get("id")
            name = (tc.get("function") or {}).get("name")
            if tid and name:
                id_to_name[tid] = name

    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            name = m.get("name") or id_to_name.get(m.get("tool_call_id"), "")
            out.append({"role": "tool", "content": m.get("content") or "", "tool_name": name})
        elif role == "assistant":
            am: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                calls.append({"function": {"name": fn.get("name"), "arguments": args or {}}})
            if calls:
                am["tool_calls"] = calls
            out.append(am)
        else:  # system / user
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def _tool_system(specs: list[ToolSpec]) -> str:
    lines = ["You decide the next step. Available tools:"]
    for s in specs:
        props = (s.schema or {}).get("properties") or {}
        params = ", ".join(props.keys()) or "(no args)"
        lines.append(f"- {s.name}({params}): {s.description}")
    lines.append(
        'If a tool is needed to fulfill the request, set "tool" to its name and fill '
        '"arguments" with the required parameters. If you can answer without tools, '
        'set "tool" to "none".'
    )
    return "\n".join(lines)


def _decision_schema(specs: list[ToolSpec]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": [s.name for s in specs] + ["none"]},
            "arguments": {"type": "object"},
        },
        "required": ["tool", "arguments"],
    }


async def tool_chat(messages: list[dict], specs: list[ToolSpec]) -> AsyncIterator[tuple[str, Any]]:
    """One stateless turn, in two steps:

    1. A constrained (JSON-schema) decision: which tool to call, or "none". Grammar
       constraint means the model can't emit templates/comments/junk.
    2. If "none", a free-text streamed answer (the tool results are already in the
       history), yielded as ("text", chunk).
    """
    hist = _to_ollama_messages(messages)
    valid = {s.name for s in specs}

    raw = await _chat_once({
        "model": MODEL,
        "messages": [{"role": "system", "content": _tool_system(specs)}, *hist],
        "format": _decision_schema(specs),
        "stream": False,
        "options": {"temperature": 0},
    })
    try:
        decision = json.loads(raw)
    except Exception:
        decision = {}
    tool = decision.get("tool")

    if tool in valid:
        args = decision.get("arguments") or {}
        wire = [{"id": "call_" + uuid4().hex[:24], "name": tool, "arguments": args}]
        log.info("ollama tool_calls: %s", [w["name"] for w in wire])
        yield "tool_calls", wire
        return

    # No tool needed → stream a natural-language answer (no tools, no schema).
    async for obj in _chat({"model": MODEL, "messages": hist, "stream": True}):
        chunk = (obj.get("message") or {}).get("content", "")
        if chunk:
            yield "text", chunk

"""Ollama provider (local Ollama server via HTTP).

`OllamaProvider` implements the `Provider` surface:
  - plain chat: `stream` (and inherited `complete`) over `/api/chat`.
  - stateless tool calling: `tools(messages, specs)` runs ONE `/api/chat` turn
    over the full conversation and yields neutral events — either streamed text or
    the tool call the model wants. No background run / futures — the client
    (JetBrains) resends the whole history each turn, so each request is independent.
"""
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx

from mcp_bridge.models import DoneEvent, ErrorEvent, Event, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec, allowed_for
from providers.base import Provider

log = logging.getLogger("codepartner.providers.ollama")

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")


class OllamaProvider(Provider):
    NAME = "ollama"

    async def stream(self, prompt: str, system: str | None = None
                     ) -> AsyncIterator[tuple[str, str | None]]:
        payload = {"model": MODEL, "messages": self._messages(prompt, system), "stream": True}
        async for obj in self._chat(payload):
            chunk = (obj.get("message") or {}).get("content", "")
            if chunk:
                yield chunk, None
            if obj.get("done"):
                yield "", self._terminal(obj.get("done_reason"))
                return

    async def tools(self, messages: list[dict], specs: list[ToolSpec],
                    model: str | None = None, role: str | None = None,
                    run_metadata: dict | None = None, **kwargs) -> AsyncIterator[Event]:
        """One stateless tool turn → neutral events. `model` is accepted for the uniform
        provider surface but ignored here — the Ollama model is fixed via OLLAMA_MODEL.
        `role` IS honored: there is no per-run `ToolRegistry` to enforce it (this
        provider keeps no Run), so the specs are narrowed to the role's allowlist here.

        Stateless: JetBrains resends the whole history (including prior tool results)
        each request, so there is no run/future/resume — every call is independent.
        Two steps:

        1. A constrained (JSON-schema) decision: which tool to call, or "none". Grammar
           constraint means the model can't emit templates/comments/junk. A chosen tool
           is yielded as a single `ToolCallEvent` and the turn ends there.
        2. If "none", a free-text streamed answer (the tool results are already in the
           history), yielded as `TextEvent`s and terminated by a `DoneEvent`.
        """
        # Narrowing here propagates into everything `decide_tool` does with the specs:
        # what the model is told exists (`_tool_system`), the grammar `enum` it must
        # pick from (`_decision_schema`), and the post-hoc name re-check.
        allowed = allowed_for(role)
        specs = [s for s in specs if s.name in allowed]

        direct_handlers = kwargs.get("direct_tool_handlers") or {}
        working = list(messages)
        for _ in range(8):
            # Step 1 is shared with the optimizer's context-gathering turn.
            chosen = await self.decide_tool(working, specs)
            if not chosen:
                break
            name, args = chosen
            handler = direct_handlers.get(name)
            if handler is None:
                yield ToolCallEvent("call_" + uuid4().hex[:24], name, args)
                return
            try:
                result = await handler(args)
            except Exception as exc:
                yield ErrorEvent(f"IDE MCP tool {name!r} failed: {exc}")
                yield DoneEvent("endTurn")
                return
            call_id = "call_" + uuid4().hex[:24]
            working.extend((
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": call_id, "type": "function", "function": {
                        "name": name, "arguments": json.dumps(args),
                    },
                }]},
                {"role": "tool", "tool_call_id": call_id, "name": name,
                 "content": result},
            ))
        hist = self._to_ollama_messages(working)

        # No tool needed → stream a natural-language answer (no tools, no schema).
        terminal = "endTurn"
        usage: dict | None = None
        async for obj in self._chat({"model": MODEL, "messages": hist, "stream": True}):
            chunk = (obj.get("message") or {}).get("content", "")
            if chunk:
                yield TextEvent(chunk)
            if obj.get("done"):
                terminal = self._terminal(obj.get("done_reason"))
                usage = self._usage(obj)
        yield DoneEvent(terminal, usage=usage, model_usage=({usage["model"]: usage} if usage else None))

    async def decide_tool(self, messages: list[dict], specs: list[ToolSpec],
                          model: str | None = None) -> tuple[str, dict[str, Any]] | None:
        """One grammar-constrained `/api/chat` turn answering a single question: which tool
        should be called next, or none at all. Returns `(name, arguments)` for a chosen tool,
        or None when the model wants no tool (or emitted something unusable).

        The model can only pick from `specs`: `_decision_schema` puts the names in a JSON
        Schema `enum`, and the answer is re-checked against `specs` afterwards — so a
        hallucinated name yields None rather than an unknown tool call. Callers that narrow
        `specs` by role therefore get that narrowing enforced here too.

        `model` defaults to the module-level `MODEL`. `tools()` deliberately passes nothing
        (its contract is that the Ollama model is fixed via OLLAMA_MODEL); the optimizer
        passes its cheapest ladder rung.
        """
        if not specs:
            return None
        raw = await self._chat_once({
            "model": model or MODEL,
            "messages": [
                {"role": "system", "content": self._tool_system(specs)},
                *self._to_ollama_messages(messages),
            ],
            "format": self._decision_schema(specs),
            "stream": False,
            "options": {"temperature": 0},
        })
        try:
            decision = json.loads(raw)
        except Exception:
            return None
        name = decision.get("tool")
        if name not in {s.name for s in specs}:
            return None
        log.info("ollama tool_calls: %s", [name])
        return name, decision.get("arguments") or {}

    async def emit_json(self, messages: list[dict], schema: dict[str, Any],
                        model: str | None = None) -> dict:
        """One grammar-constrained, non-streaming `/api/chat` turn: the model is forced
        to return JSON matching `schema` (Ollama's `format=`, the same mechanism the
        tool decision uses). Returns the parsed object, or `{}` if the model produced
        unparseable output. Used by agents that need a structured artifact rather than
        prose (e.g. the prompt optimizer's enhanced-prompt object)."""
        raw = await self._chat_once({
            "model": model or MODEL,
            "messages": self._to_ollama_messages(messages),
            "format": schema,
            "stream": False,
            "options": {"temperature": 0},
        })
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # --- internals -----------------------------------------------------------

    def _terminal(self, done_reason: str | None) -> str:
        return "maxTurnsReached" if done_reason == "length" else "endTurn"

    def _usage(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Extract token/model usage from a final /api/chat object (the one with
        `done: true`). Ollama reports `prompt_eval_count` (input) and `eval_count`
        (output). Logs a one-line summary for per-stage attribution."""
        u = {
            "model": obj.get("model") or MODEL,
            "inputTokens": obj.get("prompt_eval_count"),
            "outputTokens": obj.get("eval_count"),
        }
        log.info(
            "ollama usage: model=%s input=%s output=%s",
            u["model"], u["inputTokens"], u["outputTokens"],
        )
        return u

    async def _chat(self, payload: dict[str, Any]):
        """Yield decoded NDJSON objects from a streaming /api/chat call."""
        # No read timeout: local generation (14B) can be slow.
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream("POST", f"{HOST}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield json.loads(line)

    async def _chat_once(self, payload: dict[str, Any]) -> str:
        """Non-streaming /api/chat call → the assistant message content. Logs token usage
        so every non-streaming Ollama call (the tool decision AND the optimizer's
        emit_json rewrite) reports its per-stage spend."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            resp = await client.post(f"{HOST}/api/chat", json=payload)
            resp.raise_for_status()
            obj = resp.json()
        self._usage(obj)
        return (obj.get("message") or {}).get("content", "")

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _to_ollama_messages(self, messages: list[dict]) -> list[dict[str, Any]]:
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

    def _tool_system(self, specs: list[ToolSpec]) -> str:
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

    def _decision_schema(self, specs: list[ToolSpec]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [s.name for s in specs] + ["none"]},
                "arguments": {"type": "object"},
            },
            "required": ["tool", "arguments"],
        }

"""Local Ollama server via HTTP: the cheap structured-output model behind the
conversation and workflow interpreters (`emit_json`). It is not a chat/tool provider.
"""
import json
import logging
import os
from typing import Any

import httpx


log = logging.getLogger("codepartner.providers.ollama")

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")


class OllamaProvider:
    async def emit_json(self, messages: list[dict], schema: dict[str, Any],
                        model: str | None = None) -> dict:
        """One grammar-constrained, non-streaming `/api/chat` turn: the model is forced
        to return JSON matching `schema` (Ollama's `format=`). Returns the parsed object, or `{}` if the model produced
        unparseable output."""
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

    async def _chat_once(self, payload: dict[str, Any]) -> str:
        """Non-streaming /api/chat call → the assistant message content. Logs token usage
        so every call reports its spend."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            resp = await client.post(f"{HOST}/api/chat", json=payload)
            resp.raise_for_status()
            obj = resp.json()
        self._usage(obj)
        return (obj.get("message") or {}).get("content", "")

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

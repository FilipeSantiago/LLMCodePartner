"""PromptOptimizer agent — owns the *enhance* role of the pipeline (backed by Ollama).

It does NOT answer the user's request, and it does NOT list or guess files (a local
model hallucinates paths). It rewrites the request into a single, clear, objective,
self-contained prompt for the Coder — preserving the user's original goal. The output
is forced through `ENHANCED_SCHEMA` (Ollama `format=`) so a flaky model can only fill
the `prompt` field — no free-form prose or hallucinated tool calls. The Coder (Claude)
discovers the real files itself via its own tools.
"""
import os
from collections.abc import AsyncIterator

from mcp_bridge.models import DoneEvent, Event, TextEvent
from mcp_bridge.registry import ToolSpec
from providers.ollama import OllamaProvider


def _optimizer_model() -> str | None:
    """The optimizer may run on a sturdier instruction-follower than the code model.
    Read at call time; defaults to the provider's OLLAMA_MODEL when unset."""
    return os.getenv("OPTIMIZER_MODEL") or None


OPTIMIZER_SYSTEM = (
    "You are a prompt optimizer for a downstream coding agent. Do NOT answer or "
    "implement the user's request, and do NOT list files or guess file paths — the "
    "coding agent finds those itself. Rewrite the user's request into a single, clear, "
    "objective, self-contained prompt for the coding agent: preserve the original goal "
    "exactly (never turn a change/improve/fix request into a question or an "
    "explanation), make implicit requirements explicit, and keep it concise. If the "
    "user replies with feedback on a previous prompt, treat it as refinement and revise "
    "accordingly. Return only the rewritten prompt."
)

ENHANCED_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
}


def _render(structured: dict) -> str:
    """Structured result → the rewritten prompt. Degrades gracefully (never raises)."""
    if not isinstance(structured, dict):
        return ""
    return (structured.get("prompt") or "").strip()


class PromptOptimizer:
    """Constructed per request with the neutral message history (the conversation is the
    optimizer's context)."""

    def __init__(self, messages: list[dict]):
        self._messages = messages

    async def run(self, specs: list[ToolSpec]) -> AsyncIterator[Event]:
        # specs unused: the optimizer only rewrites the prompt; it makes no tool calls.
        augmented = [{"role": "system", "content": OPTIMIZER_SYSTEM}, *self._messages]
        structured = await OllamaProvider().emit_json(augmented, ENHANCED_SCHEMA, _optimizer_model())
        yield TextEvent(_render(structured))
        yield DoneEvent("endTurn")

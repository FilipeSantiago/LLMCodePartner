"""The abstract LLM provider contract.

Every provider is a `Provider` subclass exposing the SAME surface: `NAME`, plain
chat `stream`/`complete`, and a uniform tool turn `tools(messages, specs)`. The
ABC defines those signatures once, so adding a new provider is "subclass and
implement the abstract methods" — a missing/renamed method or an unset `NAME`
fails immediately (at instantiation / class definition) instead of at some call
site. How a provider fulfils `tools` (stateful background run vs. stateless
one-shot) is entirely private to the subclass.
"""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar

from mcp_bridge.models import DoneEvent, Event, Run, ToolCallEvent
from mcp_bridge.registry import ToolSpec


class Provider(ABC):
    """The uniform LLM provider surface. Subclasses implement `stream` and
    `tools`; `complete` is derived. `NAME` is required (enforced below)."""

    NAME: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "NAME", ""):
            raise TypeError(f"{cls.__name__} must set a non-empty NAME")

    @abstractmethod
    def stream(self, prompt: str, system: str | None = None
               ) -> AsyncIterator[tuple[str, str | None]]:
        """Plain chat, no tools. Yields (text_chunk, None) then ("", terminal)."""
        ...

    async def complete(self, prompt: str, system: str | None = None) -> tuple[str, str]:
        """Concrete default: fold `stream` into (full_text, terminal). Providers
        need not override — this is exactly ollama's original `complete`."""
        text, terminal = "", "endTurn"
        async for chunk, term in self.stream(prompt, system):
            if term is not None:
                terminal = term
            else:
                text += chunk
        return text, terminal

    @abstractmethod
    def tools(self, messages: list[dict], specs: list[ToolSpec],
              model: str | None = None, role: str | None = None,
              run_metadata: dict | None = None, **kwargs) -> AsyncIterator[Event]:
        """One tool turn → neutral events (TextEvent/ToolCallEvent/ErrorEvent/
        DoneEvent). Stateful vs. stateless is entirely private to the subclass.
        `model` optionally selects the backend model/tier for this turn (None =
        provider default); providers that can't honor it ignore it. `role` selects
        the bridge's per-role tool allowlist (`registry.allowed_for`) — None = the
        full ceiling; every provider MUST honor it, since it is a permission bound."""
        ...

    # --- shared helpers for run-backed providers -----------------------------
    #
    # Concrete and provider-agnostic: every provider that drives a background Run
    # (Claude, Codex) needs exactly this history-shaping and queue-draining logic.
    # A stateless one-shot provider (Ollama) simply never calls them.

    def _flatten(self, messages: list[dict]) -> tuple[str, str | None]:
        """Neutral history → (prompt, system) for a fresh turn. Tool-plumbing turns
        (assistant `tool_calls`, `role:"tool"` results) are dropped — the model gets
        tool results through its own resumed tool call, not the prompt."""
        system = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system" and m.get("content")
        ) or None
        convo = [
            m for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content") and not m.get("tool_calls")
        ]
        if len(convo) == 1:
            prompt = convo[0]["content"]
        else:
            prompt = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in convo)
        return prompt, system

    def _trailing_tool_results(self, messages: list[dict]) -> list[dict]:
        """`role:"tool"` messages carrying a tool_call_id (client tool results)."""
        return [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id")]

    async def _drain_queue(self, run: Run) -> AsyncIterator[Event]:
        """Yield a Run's neutral events until it stops for this request: a ToolCallEvent
        (background task now blocked on the future — resumed by a later request) or a
        DoneEvent (turn finished)."""
        while True:
            event = await run.next_event()
            yield event
            if isinstance(event, (ToolCallEvent, DoneEvent)):
                return

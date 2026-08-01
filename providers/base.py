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

from mcp_bridge.models import Event
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
    def tools(self, messages: list[dict], specs: list[ToolSpec]) -> AsyncIterator[Event]:
        """One tool turn → neutral events (TextEvent/ToolCallEvent/ErrorEvent/
        DoneEvent). Stateful vs. stateless is entirely private to the subclass."""
        ...

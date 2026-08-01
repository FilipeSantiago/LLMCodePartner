"""LLM providers + selection.

Every provider subclasses `providers.base.Provider`, so they share the SAME public
surface by construction: `NAME`, plain-chat `stream`/`complete`, and a uniform
`tools(messages, specs)` that yields neutral events
(`TextEvent`/`ToolCallEvent`/`ErrorEvent`/`DoneEvent`). How a provider fulfils
`tools` is private — Claude drives a stateful background Run with start/resume;
Ollama runs one stateless shot — but the surface is identical, so callers never
branch on which provider is active. `active()` picks the one named by the
`LLM_PROVIDER` env var (default "claude").
"""
import os

from providers.base import Provider
from providers.claude import ClaudeProvider
from providers.ollama import OllamaProvider

_PROVIDERS: dict[str, Provider] = {"claude": ClaudeProvider(), "ollama": OllamaProvider()}


def active() -> Provider:
    return _PROVIDERS.get(os.getenv("LLM_PROVIDER", "claude").lower(), _PROVIDERS["claude"])

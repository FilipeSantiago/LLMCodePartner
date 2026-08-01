"""Coder agent — owns the *coding* role of the pipeline (backed by Claude).

Thin, single-responsibility wrapper: given the neutral message history the
pipeline hands it, prepend the coder system prompt and delegate to
`ClaudeProvider.tools`, whose internal start-vs-resume logic transparently handles
both the first turn (fresh accepted prompt) and later tool-call resumes. Yields the
same neutral `Event`s every provider emits, so the pipeline can drain it uniformly.
"""
from collections.abc import AsyncIterator

from mcp_bridge.models import Event
from mcp_bridge.registry import ToolSpec
from providers.claude import ClaudeProvider

CODER_SYSTEM = (
    "You are a coding agent working inside the user's IDE project. You receive an "
    "objective, self-contained request that already lists the files relevant to the "
    "task. Use the available JetBrains tools to read those files (and any others you "
    "need) before answering, then implement the request precisely and concisely. Do "
    "not ask the user to clarify what the enhanced prompt already specifies."
)


class Coder:
    """Constructed per request with the neutral message list the pipeline routes to
    the coding stage (either a single accepted-prompt user message on the first turn,
    or the full history on a tool resume)."""

    def __init__(self, messages: list[dict]):
        self._messages = messages

    async def run(self, specs: list[ToolSpec]) -> AsyncIterator[Event]:
        augmented = [{"role": "system", "content": CODER_SYSTEM}, *self._messages]
        async for event in ClaudeProvider().tools(augmented, specs):
            yield event

"""Coder agent — owns the *coding* role of the pipeline.

Thin, single-responsibility wrapper: given the neutral message history the
pipeline hands it, prepend the coder system prompt and delegate to the coding
provider's `tools`, whose internal start-vs-resume logic transparently handles
both the first turn (fresh accepted prompt) and later tool-call resumes. Yields the
same neutral `Event`s every provider emits, so the pipeline can drain it uniformly.

Which provider does the coding is its own choice (`CODER_PROVIDER`), separate from
`LLM_PROVIDER`: the Optimizer runs locally while the Coder is Claude or Codex.
"""
import os
from collections.abc import AsyncIterator

import providers
from mcp_bridge.models import Event
from mcp_bridge.registry import ROLE_CODER, ToolSpec

CODER_SYSTEM = (
    "You are a coding agent working inside the user's IDE project. You receive an "
    "objective, self-contained request that already lists the files relevant to the "
    "task. Use the available JetBrains read tools to read those files (and any others "
    "you need) before making changes. Then ACTUALLY APPLY the change by calling the "
    "available write tools — create files with the create-file tool and edit existing "
    "files with the replace/edit-file tool (or the rename-refactoring tool for symbol "
    "renames). Make minimal, targeted edits; do not rewrite whole files when a small "
    "replacement suffices. Do not merely describe or paste the code as text and expect "
    "the user to apply it — persist the change with the tools. When a write tool is not "
    "available, fall back to showing the code. After applying the edits, give a short "
    "summary of what you changed. Do not ask the user to clarify what the enhanced "
    "prompt already specifies."
)


class Coder:
    """Constructed per request with the neutral message list the pipeline routes to
    the coding stage (either a single accepted-prompt user message on the first turn,
    or the full history on a tool resume)."""

    def __init__(self, messages: list[dict], model: str | None = None):
        self._messages = messages
        self._model = model

    async def run(self, specs: list[ToolSpec],
                  direct_tool_handlers: dict | None = None) -> AsyncIterator[Event]:
        augmented = [{"role": "system", "content": CODER_SYSTEM}, *self._messages]
        # Read at call time (not import), like `pipeline.is_enabled` — `.env` may load
        # after this module is imported.
        provider = providers.get(os.getenv("CODER_PROVIDER", "claude"))
        async for event in provider.tools(augmented, specs, model=self._model,
                                          role=ROLE_CODER,
                                          direct_tool_handlers=direct_tool_handlers):
            yield event

"""Conversation-flow pipeline — Optimizer → human accept → Coder.

This module owns ONLY the workflow: derive the current stage from the resent
message history, apply the acceptance gate, and delegate the model work to the
`PromptOptimizer` / `Coder` agents. It holds no model logic and imports no provider.

Because the backend is stateless (JetBrains re-sends the full conversation each
turn, assistant messages included), stage is a pure function of that history:

  - Optimize: no enhanced-prompt proposal anywhere yet (a fresh user prompt, or the
    Optimizer still gathering context via tool round-trips). Run the Optimizer; on a
    text turn wrap it as the acceptance proposal (sentinel + footer), on a tool turn
    pass it through so JetBrains executes the tool.
  - Accept: the LAST assistant message is a proposal (starts with the sentinel). The
    user's reply is the acceptance — `accept` sends the proposed prompt to the Coder,
    anything else is treated as the edited prompt.
  - Code: a proposal exists earlier in history but isn't the last assistant message
    (the Coder already started). Hand the full history to the Coder so Claude's
    start-vs-resume continues the run.
"""
import os
from collections.abc import AsyncIterator

from agent.coder import Coder
from agent.prompt_optimizer import PromptOptimizer
from conversation import openai_request as oreq
from mcp_bridge.models import DoneEvent, Event, TextEvent, ToolCallEvent
from model.chat import ChatCompletionRequest

def is_enabled() -> bool:
    """Whether the optimizer pipeline is active. Read at call time (not import) so it
    survives `load_dotenv()` running after this module is imported — mirroring
    `providers.active()` reading `LLM_PROVIDER` per call. The env var name matches
    the `.env` key (`OPTIMIZER_ENABLED`)."""
    enabled = os.getenv("OPTIMIZER_ENABLED", "false").lower() == "true"

    if enabled:
        print("RUNNING PIPELINE")

    return os.getenv("OPTIMIZER_ENABLED", "false").lower() == "true"

# Machine marker prefixed to the Optimizer's enhanced-prompt assistant message. It
# rides along in the history JetBrains re-sends, so the next request can detect the
# acceptance stage. The footer tells the human how to proceed.
ENHANCED_PROMPT_MARKER = "⟦enhanced-prompt⟧"
_FOOTER_SEP = "\n\n---\n"
_FOOTER = _FOOTER_SEP + "Reply `accept` to send this to the Coder, or reply with feedback to keep planning."


def _is_proposal(msg: dict | None) -> bool:
    return bool(msg) and (msg.get("content") or "").lstrip().startswith(ENHANCED_PROMPT_MARKER)


def _last_assistant_idx(history: list[dict]) -> int:
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "assistant":
            return i
    return -1


def _latest_proposal_idx(history: list[dict]) -> int:
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "assistant" and _is_proposal(history[i]):
            return i
    return -1


def _accepted(history: list[dict], proposal_idx: int) -> bool:
    """True if the user replied `accept` to the latest proposal (case-insensitive)."""
    return any(
        m.get("role") == "user" and (m.get("content") or "").strip().lower() == "accept"
        for m in history[proposal_idx + 1:]
    )


def _extract_prompt(content: str) -> str:
    """Recover the enhanced prompt from a proposal assistant message (strip the
    leading sentinel and the trailing footer)."""
    body = content.strip()
    if body.startswith(ENHANCED_PROMPT_MARKER):
        body = body[len(ENHANCED_PROMPT_MARKER):].lstrip("\n")
    idx = body.rfind(_FOOTER_SEP)
    if idx != -1:
        body = body[:idx]
    return body.strip()


async def _optimize(history: list[dict], specs) -> AsyncIterator[Event]:
    """Run the Optimizer and, if this turn produces the enhanced prompt (a text
    turn), wrap it as the acceptance proposal. Tool turns pass through untouched."""
    started = False
    text_turn = False
    async for ev in PromptOptimizer(history).run(specs):
        if not started:
            started = True
            if isinstance(ev, ToolCallEvent):
                yield ev
                continue
            text_turn = True
            yield TextEvent(ENHANCED_PROMPT_MARKER + "\n")
        if text_turn and isinstance(ev, DoneEvent):
            yield TextEvent(_FOOTER)
        yield ev


async def run(request: ChatCompletionRequest) -> AsyncIterator[Event]:
    """The stage router — one async generator of neutral events per request. Stage is a
    pure function of the resent history (see module docstring). Scoped to one
    planning→code cycle; `accept` is the only path to the Coder."""
    history = oreq.to_messages(request)
    specs = oreq.tool_specs(request)

    last = history[-1] if history else {}
    prop_idx = _latest_proposal_idx(history)

    # A tool result came back — resume whoever is running.
    if last.get("role") == "tool":
        if prop_idx != -1 and _accepted(history, prop_idx):
            async for ev in Coder(history).run(specs):   # Code resume
                yield ev
        else:
            async for ev in _optimize(history, specs):    # Optimizer still gathering
                yield ev
        return

    # The user is replying to the freshest proposal.
    if prop_idx != -1 and prop_idx == _last_assistant_idx(history):
        if (last.get("content") or "").strip().lower() == "accept":
            enhanced = _extract_prompt(history[prop_idx]["content"])
            async for ev in Coder([{"role": "user", "content": enhanced}]).run(specs):  # Code start
                yield ev
        else:
            async for ev in _optimize(history, specs):    # Refine on feedback
                yield ev
        return

    # Fresh prompt (no proposal yet) — start optimizing.
    async for ev in _optimize(history, specs):
        yield ev

"""PromptOptimizer agent — owns the *enhance* role of the pipeline (backed by Ollama).

It does NOT answer or implement the user's request. It rewrites the request into a
single, clear, objective, self-contained prompt for the Coder — preserving the user's
original goal. The output is forced through `ENHANCED_SCHEMA` (Ollama `format=`) so a
flaky model can only fill the declared fields — no free-form prose.

Two capabilities beyond a single rewrite call:

  - **Read-only context** — before rewriting, the optimizer may spend a bounded number of
    round trips reading the project through the IDE's tools, so the rewrite can name files
    it actually looked at instead of guessing. It holds `ROLE_OPTIMIZER`, which is READ
    only (`mcp_bridge/registry.py`), so it cannot edit anything. Because the backend is
    stateless there is no internal loop: each invocation either emits ONE tool call and
    returns (the pipeline ships it to JetBrains and calls us again with the result), or
    decides it has seen enough and produces the rewrite.

  - **Weak→strong escalation** — the rewrite runs on a ladder of local models, weakest
    first. A result that is low-confidence or structurally degenerate escalates to the
    next rung. The ladder is a single model unless `OPTIMIZER_MODEL_LADDER` says otherwise,
    so the default behaviour is exactly one call.
"""
import logging
import os
from collections.abc import AsyncIterator
from uuid import uuid4

from mcp_bridge.models import DoneEvent, Event, TextEvent, ToolCallEvent
from mcp_bridge.registry import ROLE_OPTIMIZER, ToolSpec, allowed_for
from providers.ollama import OllamaProvider

log = logging.getLogger("codepartner.agent.optimizer")


def _optimizer_model() -> str | None:
    """The optimizer may run on a sturdier instruction-follower than the code model.
    Read at call time; defaults to the provider's OLLAMA_MODEL when unset."""
    return os.getenv("OPTIMIZER_MODEL") or None


def _ladder() -> list[str | None]:
    """The escalation ladder, WEAKEST FIRST, from `OPTIMIZER_MODEL_LADDER` (comma-separated).

    Unset (the default) yields a single rung of `None`, which `emit_json` resolves to
    OPTIMIZER_MODEL/OLLAMA_MODEL — i.e. exactly the pre-escalation behaviour: one call, one
    model. Read at call time so it survives `load_dotenv()` running after import, matching
    `_optimizer_model()` and `pipeline.is_enabled()`.
    """
    rungs = [m.strip() for m in os.getenv("OPTIMIZER_MODEL_LADDER", "").split(",") if m.strip()]
    return list(rungs) if rungs else [_optimizer_model()]


def _min_confidence() -> str:
    """The lowest self-reported confidence accepted without escalating."""
    value = os.getenv("OPTIMIZER_MIN_CONFIDENCE", "medium").strip().lower()
    return value if value in CONFIDENCE_CHOICES else "medium"


def _may_read_context() -> bool:
    """Kill switch for the read-only context stage. Default on. An empty value counts as
    unset (`FOO=` in a .env means "I didn't configure this", not "disable the feature")."""
    return (os.getenv("OPTIMIZER_READ_CONTEXT") or "true").lower() == "true"


def _max_context_calls() -> int:
    """How many read round trips the optimizer may spend before it must rewrite."""
    try:
        return int(os.getenv("OPTIMIZER_MAX_CONTEXT_CALLS") or "3")
    except ValueError:
        return 3


# The coder is Claude; these map to `ClaudeAgentOptions.model` (see providers/claude.py).
MODEL_CHOICES = ["haiku", "sonnet", "opus"]

# Ordered weakest → strongest, so `index()` compares them.
CONFIDENCE_CHOICES = ["low", "medium", "high"]

# Delimiter between the rewritten prompt and the suggested-model-config block in the
# proposal. The pipeline splits on this to recover the clean prompt (and parse the model)
# on accept, so it must stay in sync with agent/pipeline.py.
MODEL_SUGGESTION_SEP = "\n\n--- suggested model config ---\n"

# Prefix of the line reporting which local model produced the rewrite. It must NOT start
# with "model:" — `pipeline._extract_model` returns the FIRST block line beginning with
# that prefix, so naming this one "Model:" would shadow the coder tier above it and
# silently fall back to the CLI default.
ANALYZED_BY_LABEL = "Analyzed by:"

# System prompt for the context-gathering turn. Deliberately separate from
# OPTIMIZER_SYSTEM: this turn answers only "which file should I look at next, if any",
# and must not be tempted into rewriting or editing.
CONTEXT_SYSTEM = (
    "You are gathering read-only context for a prompt you are about to rewrite for a "
    "coding agent. Decide whether reading one more file (or listing the project tree, or "
    "searching the sources) would let you write a more precise prompt. If the request is "
    "already concrete enough, choose no tool. Never propose or make edits — you only read. "
    "Do not answer the user's request."
)

OPTIMIZER_SYSTEM = (
    "You are a prompt optimizer for a downstream coding agent. Do NOT answer or "
    "implement the user's request. Never invent or guess a file path: mention a path "
    "only if it appears in the context you were given by the read tools; otherwise "
    "describe the target in words and let the coding agent locate it. Rewrite the user's "
    "request into a single, clear, objective, self-contained prompt for the coding agent: "
    "preserve the original goal exactly (never turn a change/improve/fix request into a "
    "question or an explanation), make implicit requirements explicit, and keep it "
    "concise. If the user replies with feedback on a previous prompt, treat it as "
    "refinement and revise accordingly.\n\n"
    "Also judge the task's complexity from the request text alone and pick the coder "
    "model tier:\n"
    "- simple: a single-file, localized edit; clear and unambiguous -> haiku\n"
    "- medium: a few files or moderate logic -> sonnet\n"
    "- complex: multi-file, cross-cutting, algorithmic, or ambiguous -> opus\n"
    "Fill 'prompt' with the rewritten prompt, 'complexity' with simple/medium/complex, "
    "'model' with the matching tier, 'reasoning' with a one-line justification, and "
    "'confidence' with how sure you are that the rewritten prompt is complete and "
    "unambiguous (low/medium/high)."
)

ENHANCED_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "complexity": {"type": "string", "enum": ["simple", "medium", "complex"]},
        "model": {"type": "string", "enum": MODEL_CHOICES},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": CONFIDENCE_CHOICES},
    },
    "required": ["prompt", "complexity", "model", "confidence"],
}


def _degenerate(structured: dict, original: str = "") -> str | None:
    """Why this result is unusable regardless of the confidence it claims, or None if it
    looks fine. This is the trigger that earns its keep: a small model's self-reported
    confidence is weak signal, but an empty prompt is an empty prompt."""
    if not isinstance(structured, dict):
        return "not an object"
    prompt = (structured.get("prompt") or "").strip()
    if not prompt:
        return "empty prompt"
    if (structured.get("model") or "").strip() not in MODEL_CHOICES:
        # `_render` would silently drop the suggestion block for this.
        return "model tier outside MODEL_CHOICES"
    if original and len(prompt) < len(original.strip()) // 2:
        # A rewrite is allowed to be terser, but half the original usually means the model
        # dropped requirements rather than tightened them.
        return "rewrite much shorter than the request"
    return None


def _sufficient(structured: dict, original: str = "") -> tuple[bool, str]:
    """Whether to accept this rung's result. Returns (accept, reason_if_rejected)."""
    reason = _degenerate(structured, original)
    if reason:
        return False, reason
    claimed = (structured.get("confidence") or "").strip().lower()
    if claimed not in CONFIDENCE_CHOICES:
        return False, "no usable confidence"
    if CONFIDENCE_CHOICES.index(claimed) < CONFIDENCE_CHOICES.index(_min_confidence()):
        return False, f"confidence {claimed} below {_min_confidence()}"
    return True, ""


def _render(structured: dict, analyzed_by: str | None = None, escalations: int = 0) -> str:
    """Structured result → the rewritten prompt followed by a suggested-model-config
    block. Degrades gracefully (never raises): a missing/invalid model tier drops the
    suggestion block and returns just the prompt.

    `analyzed_by`/`escalations` report which local model produced this and how many rungs
    it took. That line lives inside the suggestion block, which `pipeline._extract_prompt`
    strips before the Coder — the Coder never sees it.
    """
    if not isinstance(structured, dict):
        return ""
    prompt = (structured.get("prompt") or "").strip()
    model = (structured.get("model") or "").strip()
    if model not in MODEL_CHOICES:
        return prompt
    complexity = (structured.get("complexity") or "").strip() or "unknown"
    reasoning = (structured.get("reasoning") or "").strip()
    block = f"Model: {model}   (complexity: {complexity})"
    if reasoning:
        block += f"\nWhy: {reasoning}"
    if analyzed_by:
        confidence = (structured.get("confidence") or "").strip() or "unknown"
        detail = f"confidence: {confidence}"
        if escalations:
            detail += f", escalated {escalations}×"
        block += f"\n{ANALYZED_BY_LABEL} {analyzed_by}   ({detail})"
    return prompt + MODEL_SUGGESTION_SEP + block


class PromptOptimizer:
    """Constructed per request with the neutral message history (the conversation is the
    optimizer's context)."""

    def __init__(self, messages: list[dict]):
        self._messages = messages

    async def run(self, specs: list[ToolSpec]) -> AsyncIterator[Event]:
        # ROLE_OPTIMIZER is READ-only, so this agent physically cannot be handed a write
        # tool even if the IDE advertises one.
        specs = [s for s in specs if s.name in allowed_for(ROLE_OPTIMIZER)]

        if self._may_gather(specs):
            # Rung 0 (the cheapest model) decides *what to read*; picking a file is a much
            # easier call than the rewrite, so escalation is reserved for the latter.
            chosen = await OllamaProvider().decide_tool(
                [{"role": "system", "content": CONTEXT_SYSTEM}, *self._messages],
                specs, _ladder()[0],
            )
            if chosen:
                name, args = chosen
                log.info("optimizer gathering context: tool=%s args=%s", name, args)
                # One call per turn: the pipeline ships this to JetBrains and re-enters
                # `run` with the result appended to the history.
                yield ToolCallEvent("call_" + uuid4().hex[:24], name, args)
                return

        structured, analyzed_by, escalations = await self._rewrite()
        yield TextEvent(_render(structured, analyzed_by, escalations))
        yield DoneEvent("endTurn")

    # --- internals -----------------------------------------------------------

    def _may_gather(self, specs: list[ToolSpec]) -> bool:
        """Whether to spend another read round trip. The budget is derived from the resent
        history rather than held in memory, because the backend is stateless — and it is
        what guarantees the gather loop terminates. Only the Optimizer stage runs before
        `accept`, so no Coder tool results can inflate this count."""
        if not specs or not _may_read_context():
            return False
        spent = sum(1 for m in self._messages if m.get("role") == "tool")
        return spent < _max_context_calls()

    def _original_request(self) -> str:
        """The first user message — the yardstick for the too-short degeneracy check."""
        for m in self._messages:
            if m.get("role") == "user" and (m.get("content") or "").strip():
                return m["content"]
        return ""

    async def _rewrite(self) -> tuple[dict, str | None, int]:
        """Run the rewrite up the ladder, weakest first, stopping at the first sufficient
        result. Returns (structured, analyzing_model, escalation_count). If every rung is
        rejected, the strongest rung's result is used anyway — a flawed prompt the user can
        edit beats no proposal at all."""
        augmented = [{"role": "system", "content": OPTIMIZER_SYSTEM}, *self._messages]
        original = self._original_request()
        rungs = _ladder()

        structured: dict = {}
        model: str | None = None
        for index, model in enumerate(rungs):
            structured = await OllamaProvider().emit_json(augmented, ENHANCED_SCHEMA, model)
            accept, reason = _sufficient(structured, original)
            if accept:
                return structured, model or _optimizer_model(), index
            if index < len(rungs) - 1:
                log.info(
                    "optimizer escalating: rung=%d model=%s reason=%s",
                    index, model or _optimizer_model(), reason,
                )
            else:
                log.warning(
                    "optimizer ladder exhausted: model=%s reason=%s (using it anyway)",
                    model or _optimizer_model(), reason,
                )
        return structured, model or _optimizer_model(), len(rungs) - 1

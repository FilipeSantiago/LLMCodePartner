"""Conversation modes selected through the IDE chat model picker.

JetBrains chat sends the selected model id with every request, so each virtual
model below is a sticky, visible mode.  Plain messages are interpreted under the
mode's verb; messages starting with ``/`` keep their one-shot command meaning.
Modes are stateless: a review scope is recovered from the resent history.
"""

import re
from dataclasses import dataclass

DEFAULT_MODEL = "dumb-router"


@dataclass(frozen=True)
class Mode:
    model_id: str
    verb: str | None
    summary: str


MODES = (
    Mode(DEFAULT_MODEL, None, "Default chat: plain messages go to the coding agent."),
    Mode("codepartner-spec", "spec",
         "Plan an OpenSpec change as a draft; keep chatting to revise it, "
         "`/apply` saves it, `/discard` drops it."),
    Mode("codepartner-review", "review",
         "Read-only review: `WP1: <question>`; later questions reuse the last named scope."),
    Mode("codepartner-implement", "implement",
         "Implement selected work: `WP1-T1: <guidance>` runs `/implement` for that scope."),
)

_REVIEW_HINT = ("Review mode needs a scope first, for example `WP1: are tests missing?` "
                "or `WP1-T2`. Later questions reuse it.")
_IMPLEMENT_HINT = ("Implement mode needs an explicit scope in each message, for example "
                   "`WP1-T1: add tests` or `WP2`. Nothing was executed.")

_TOKEN = r"(?:EPIC|WP|US)\d+(?:-T(?:ASK)?\d+)?"
_LEADING_SCOPE = re.compile(
    rf"\s*({_TOKEN}(?:[\s,]+{_TOKEN})*)(?![\w-])\s*(?::\s*)?(.*)", re.IGNORECASE | re.DOTALL)
_REVIEW_PREFIX = re.compile(r"^\s*/review\s+", re.IGNORECASE)


@dataclass(frozen=True)
class ModeRoute:
    """How a plain message is handled under the selected mode (exactly one is set)."""
    content: str | None = None        # equivalent slash command
    draft_request: str | None = None  # spec mode: create or revise a draft change
    reply: str | None = None          # usage hint; nothing is executed


def mode_for_model(model: str | None) -> Mode | None:
    """Return the verb mode for a model id, or None for default/unknown models."""
    mode = next((item for item in MODES if item.model_id == model), None)
    return mode if mode is not None and mode.verb else None


def split_scope(content: str) -> tuple[str, str] | None:
    """Split leading selectors (``WP1 WP2-T1: rest``) from the remaining text."""
    match = _LEADING_SCOPE.fullmatch(content)
    if match is None:
        return None
    return " ".join(re.split(r"[\s,]+", match.group(1).strip())), match.group(2).strip()


def _previous_scope(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        prefix = _REVIEW_PREFIX.match(content)
        if content.lstrip().startswith("/") and prefix is None:
            continue
        scope = split_scope(content[prefix.end():] if prefix else content)
        if scope is not None:
            return scope[0]
    return None


_STRUCTURE_LINE = re.compile(r"^\s*(?:<[/\w!?]|#{1,6}\s|```|---|\[[\w ./-]+\]\s*$)")


def request_shape(content: str) -> dict:
    """Describe a request's layout for logs, to reveal context appended by the IDE."""
    lines = content.splitlines()
    markers = [(number, line.strip()[:80]) for number, line in enumerate(lines, 1)
               if _STRUCTURE_LINE.match(line)]
    return {"chars": len(content), "lines": len(lines),
            "head": content[:200], "markers": markers[:20]}


def route_message(model: str | None, messages: list[dict]) -> ModeRoute | None:
    """Map the last plain user message to the selected mode's action."""
    mode = mode_for_model(model)
    if mode is None or not messages:
        return None
    last = messages[-1]
    content = (last.get("content") or "").strip()
    if last.get("role") != "user" or not content or content.startswith("/"):
        return None
    if mode.verb == "spec":
        return ModeRoute(draft_request=content)
    scope = split_scope(content)
    if mode.verb == "implement":
        # The /implement parser reports guidance that is missing its ':'.
        return ModeRoute(content=f"/implement {content}") if scope else ModeRoute(reply=_IMPLEMENT_HINT)
    if scope is not None:
        selectors, question = scope
    else:
        selectors, question = _previous_scope(messages[:-1]), content
        if selectors is None:
            return ModeRoute(reply=_REVIEW_HINT)
    return ModeRoute(content=f"/review {selectors}: {question}" if question else f"/review {selectors}")

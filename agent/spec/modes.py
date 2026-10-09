"""Conversation mode labels; the workflow interpreter treats them as hints."""

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
         "Specification-focused conversation; actions are proposed for confirmation."),
    Mode("codepartner-review", "review",
         "Read-only review conversation; name a task when several could match."),
    Mode("codepartner-implement", "implement",
         "Implementation-focused conversation; free-form requests are interpreted and confirmed."),
    Mode("codepartner-plan", "plan",
         "Plan, then implement: name a work package (\"Let's do WP4\") and a strong planner "
         "investigates it and proposes a plan to approve or refine before any coding."),
)

_TOKEN = r"(?:EPIC|WP|US)\d+(?:-T(?:ASK)?\d+)?"
_LEADING_SCOPE = re.compile(
    rf"\s*({_TOKEN}(?:[\s,]+{_TOKEN})*)(?![\w-])\s*(?::\s*)?(.*)", re.IGNORECASE | re.DOTALL)


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


_STRUCTURE_LINE = re.compile(r"^\s*(?:<[/\w!?]|#{1,6}\s|```|---|\[[\w ./-]+\]\s*$)")


def request_shape(content: str) -> dict:
    """Describe a request's layout for logs, to reveal context appended by the IDE."""
    lines = content.splitlines()
    markers = [(number, line.strip()[:80]) for number, line in enumerate(lines, 1)
               if _STRUCTURE_LINE.match(line)]
    return {"chars": len(content), "lines": len(lines),
            "head": content[:200], "markers": markers[:20]}


def route_message(model: str | None, messages: list[dict]) -> ModeRoute | None:
    """Legacy API: modes no longer rewrite or reject free-form user messages."""
    return None

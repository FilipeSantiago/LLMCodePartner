"""Compatibility helpers for interpreting a preceding numbered choice."""

import re

_RETRY_REVIEW = re.compile(
    r"\b(?:retry|rerun|re-run|run again)\b.{0,100}\breview\b|"
    r"\breview\b.{0,60}\b(?:again|retry)\b", re.IGNORECASE | re.DOTALL)
_NEGATED_RETRY = re.compile(
    r"\b(?:do not|don't|dont|never|no)\s+(?:want to\s+)?"
    r"(?:retry|rerun|re-run|run again)\b", re.IGNORECASE)
_NUMBER = re.compile(r"^[1-9]\d*$")
_MENU_LINE = re.compile(r"^\s*(\d+)\.\s+(.+)$")


def menu_choices(content: str) -> dict[int, str]:
    """Extract a short numbered choice menu from one assistant observation."""
    found = {}
    for line in content.splitlines():
        match = _MENU_LINE.match(line)
        if match and len(found) < 10:
            label = match.group(2)
            command = re.search(r"`([^`]+)`", label)
            if label.lstrip().lower().startswith("reply ") and command:
                label = re.sub(r":\s*<[^>]+>.*$", "", command.group(1)).strip()
            found[int(match.group(1))] = label.replace("**", "").replace("`", "").strip()
    return found if len(found) >= 2 else {}


def offered_choice(message: str, history: list[dict]) -> str | None:
    """Resolve a number only against the immediately preceding assistant menu."""
    if not _NUMBER.fullmatch(message.strip()) or not history:
        return None
    previous = history[-1]
    if previous.get("role") != "assistant":
        return None
    return menu_choices(str(previous.get("content") or "")).get(int(message.strip()))


def explicit_retry_request(message: str, history: list[dict]) -> bool:
    """Recognize a review retry even when no paused task can be found."""
    interpreted = offered_choice(message, history) or message.strip()
    return bool(_RETRY_REVIEW.search(interpreted) and not _NEGATED_RETRY.search(interpreted))

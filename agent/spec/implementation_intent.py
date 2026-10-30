"""OpenSpec selectors in free text, and the cheap intent model's settings.

`ImplementationIntent` is the scope a confirmed START_IMPLEMENTATION proposal carries.
"""

import asyncio
import os
import re
from dataclasses import dataclass, field

from providers.ollama import MODEL as DEFAULT_OLLAMA_MODEL


_SELECTOR = re.compile(
    r"^(?:(EPIC)\s*(\d+)|(WP|US)\s*(\d+)(?:\s*-\s*T(?:ASK)?\s*(\d+))?)$",
    re.IGNORECASE,
)
SELECTOR_IN_TEXT = re.compile(
    r"\b(?:EPIC\s*\d+|(?:WP|US)\s*\d+(?:\s*-\s*T(?:ASK)?\s*\d+)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImplementationIntent:
    selectors: tuple[str, ...]
    instruction: str
    confidence: str
    ambiguities: tuple[str, ...]
    model: str
    fallback_used: bool = False
    latency_ms: int | None = None
    usage: dict = field(default_factory=dict)


def intent_model() -> str:
    return os.getenv("IMPLEMENT_INTENT_MODEL") or DEFAULT_OLLAMA_MODEL


def intent_timeout() -> float:
    return float(os.getenv("IMPLEMENT_INTENT_TIMEOUT_SECONDS", "15"))


async def emit_json_with_usage(provider, messages: list[dict], schema: dict,
                               model: str) -> tuple[dict, dict]:
    """Call the cheap model, returning token usage when the provider reports it."""
    if hasattr(provider, "emit_json_usage"):
        return await asyncio.wait_for(
            provider.emit_json_usage(messages, schema, model), timeout=intent_timeout())
    value = await asyncio.wait_for(
        provider.emit_json(messages, schema, model), timeout=intent_timeout())
    return value, {}


def canonical_selector(value: str) -> str | None:
    match = _SELECTOR.fullmatch(value.strip())
    if match is None:
        return None
    if match.group(1):
        return f"EPIC{match.group(2)}"
    package = f"WP{match.group(4)}"
    return f"{package}-T{match.group(5)}" if match.group(5) else package


def selectors_in_text(text: str) -> tuple[str, ...]:
    found = (canonical_selector(match.group(0)) for match in SELECTOR_IN_TEXT.finditer(text))
    return tuple(dict.fromkeys(item for item in found if item))

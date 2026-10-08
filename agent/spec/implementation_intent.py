"""Cheap-model extraction of OpenSpec scope from a plain implement-mode request.

This module only identifies *what* to implement. The workflow controller turns
the result into a START_IMPLEMENTATION proposal that the user must confirm.
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass, field

from providers.ollama import MODEL as DEFAULT_OLLAMA_MODEL, OllamaProvider


INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "selectors": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "instruction": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "ambiguities": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    },
    "required": ["selectors", "instruction", "confidence", "ambiguities"],
    "additionalProperties": False,
}

_SELECTOR = re.compile(
    r"^(?:(EPIC)\s*(\d+)|(WP|US)\s*(\d+)(?:\s*-\s*T(?:ASK)?\s*(\d+))?)$",
    re.IGNORECASE,
)
SELECTOR_IN_TEXT = re.compile(
    r"\b(?:EPIC\s*\d+|(?:WP|US)\s*\d+(?:\s*-\s*T(?:ASK)?\s*\d+)?)\b",
    re.IGNORECASE,
)
_FALLBACK_FILLER = re.compile(
    r"(?:implement|execute|please|and|,|\.|\s|from\s+(?:code|open)spec)+",
    re.IGNORECASE,
)
_SOURCE_ONLY = re.compile(r"(?:from\s+)?(?:code|open)spec", re.IGNORECASE)


class ImplementationIntentError(ValueError):
    """The request could not safely become a bounded implementation proposal."""


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


class ImplementationIntentService:
    """Extracts selectors and explicit guidance from one implement-mode message."""

    def __init__(self, provider: OllamaProvider | None = None):
        self.provider = provider or OllamaProvider()
        self.model = intent_model()

    async def interpret(self, request: str) -> ImplementationIntent:
        if not request.strip() or len(request) > 4000:
            raise ImplementationIntentError(
                "implementation intent must contain between 1 and 4000 characters")
        started = time.monotonic()
        try:
            value, usage = await emit_json_with_usage(self.provider, [
                {"role": "system", "content": (
                    "Extract the bounded OpenSpec implementation intent from one user message. "
                    "Selectors must be canonical EPICn, WPn, or WPn-Tn identifiers. Treat "
                    "'from CodeSpec' and 'from OpenSpec' as source qualifiers, never as an "
                    "execution instruction. Put only explicit implementation guidance in "
                    "instruction. Do not invent selectors. Return only the required JSON.")},
                {"role": "user", "content": request},
            ], INTENT_SCHEMA, self.model)
            intent = self._validated(value, self.model)
        except Exception:
            intent = self._fallback(request, self.model)
            usage = {}
        latency = int((time.monotonic() - started) * 1000)
        return ImplementationIntent(
            intent.selectors, intent.instruction, intent.confidence, intent.ambiguities,
            intent.model, intent.fallback_used, latency, usage)

    @staticmethod
    def _validated(value: dict, model: str) -> ImplementationIntent:
        selectors = value.get("selectors")
        instruction = value.get("instruction")
        confidence = value.get("confidence")
        ambiguities = value.get("ambiguities")
        if (not isinstance(selectors, list) or not selectors or len(selectors) > 20
                or not isinstance(instruction, str)
                or len(instruction) > 2000
                or confidence not in {"high", "medium", "low"}
                or not isinstance(ambiguities, list) or len(ambiguities) > 10
                or not all(isinstance(item, str) for item in ambiguities)):
            raise ImplementationIntentError("the intent model returned an invalid result")
        canonical = [canonical_selector(item) for item in selectors if isinstance(item, str)]
        if len(canonical) != len(selectors) or any(item is None for item in canonical):
            raise ImplementationIntentError("the intent model returned an invalid selector")
        instruction = instruction.strip()
        if _SOURCE_ONLY.fullmatch(instruction):
            instruction = ""
        return ImplementationIntent(
            tuple(dict.fromkeys(canonical)), instruction, confidence,
            tuple(item.strip() for item in ambiguities if item.strip()), model)

    @staticmethod
    def _fallback(request: str, model: str) -> ImplementationIntent:
        remainder = SELECTOR_IN_TEXT.sub(" ", request)
        selectors = selectors_in_text(request)
        if not selectors or _FALLBACK_FILLER.fullmatch(remainder) is None:
            raise ImplementationIntentError(
                "I could not safely identify an OpenSpec scope. Name selectors such as "
                "WP4 and WP5, or use `/implement WP4 WP5` for the explicit path.")
        return ImplementationIntent(selectors, "", "medium", (), model, True)

"""Round-trip tests for the optimizer proposal ↔ pipeline extraction.

Dependency-free (stdlib `unittest`, no pytest): run with

    .venv/bin/python -m unittest tests.test_optimizer_proposal -v

Covers that `_render` emits the two-section proposal (optimized prompt + suggested
model config) and that the pipeline's `_extract_prompt` / `_extract_model` recover
exactly the clean prompt and the suggested tier back out of a wrapped proposal.
"""
import unittest

from agent.prompt_optimizer import MODEL_CHOICES, _render
from agent.pipeline import (
    ENHANCED_PROMPT_MARKER,
    _FOOTER,
    _extract_model,
    _extract_prompt,
)


def _as_proposal(rendered: str) -> str:
    """Wrap a rendered body exactly as the pipeline does before JetBrains resends it."""
    return ENHANCED_PROMPT_MARKER + "\n" + rendered + _FOOTER


class OptimizerProposalRoundTrip(unittest.TestCase):
    def test_prompt_and_model_round_trip(self):
        structured = {
            "prompt": "Add pagination to the users endpoint, service, and UI list.",
            "complexity": "complex",
            "model": "opus",
            "reasoning": "multi-file change with non-trivial logic",
        }
        proposal = _as_proposal(_render(structured))
        self.assertEqual(_extract_prompt(proposal), structured["prompt"])
        self.assertEqual(_extract_model(proposal), "opus")

    def test_all_tiers_recovered(self):
        for tier in MODEL_CHOICES:
            proposal = _as_proposal(
                _render({"prompt": "do a thing", "complexity": "simple", "model": tier})
            )
            self.assertEqual(_extract_prompt(proposal), "do a thing")
            self.assertEqual(_extract_model(proposal), tier)

    def test_invalid_model_drops_suggestion(self):
        # An out-of-enum model degrades gracefully: no suggestion block, no model.
        rendered = _render({"prompt": "rename x to y", "model": "bogus"})
        self.assertEqual(rendered, "rename x to y")
        proposal = _as_proposal(rendered)
        self.assertEqual(_extract_prompt(proposal), "rename x to y")
        self.assertIsNone(_extract_model(proposal))

    def test_missing_model_field(self):
        rendered = _render({"prompt": "just the prompt"})
        self.assertEqual(rendered, "just the prompt")
        self.assertIsNone(_extract_model(_as_proposal(rendered)))


if __name__ == "__main__":
    unittest.main()

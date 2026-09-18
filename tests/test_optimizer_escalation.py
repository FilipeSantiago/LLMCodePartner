"""Tests for the optimizer's model ladder, escalation triggers, and context budget.

Dependency-free (stdlib `unittest`, no pytest, no running Ollama): run with

    .venv/bin/python -m unittest tests.test_optimizer_escalation -v

`OllamaProvider` is replaced with a fake that returns scripted results per rung, so
these exercise the escalation policy itself rather than any model behaviour.
"""
import asyncio
import os
import unittest
from unittest import mock

from agent import prompt_optimizer as po
from agent.pipeline import ENHANCED_PROMPT_MARKER, _FOOTER, _extract_model, _extract_prompt
from agent.prompt_optimizer import (
    ANALYZED_BY_LABEL,
    PromptOptimizer,
    _ladder,
    _render,
    _sufficient,
)
from mcp_bridge.models import TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec

GOOD = {
    "prompt": "Add pagination to the users endpoint and its service layer.",
    "complexity": "medium",
    "model": "sonnet",
    "reasoning": "a couple of files",
    "confidence": "high",
}


def _result(**overrides) -> dict:
    return {**GOOD, **overrides}


class FakeProvider:
    """Stands in for OllamaProvider: hands back one scripted dict per emit_json call.

    State is class-level and `__init__` is a no-op, because the code under test builds a
    fresh `OllamaProvider()` for every call — an instance-level script would be reset on
    each rung and the ladder would look one call deep. Set it up with `script()`.
    """

    @classmethod
    def script(cls, results=None, tool=None):
        cls.results = list(results or [])
        cls.tool = tool
        cls.models_used = []
        cls.decide_models = []
        cls.last_specs = []

    async def emit_json(self, messages, schema, model=None):
        FakeProvider.models_used.append(model)
        return FakeProvider.results.pop(0) if FakeProvider.results else {}

    async def decide_tool(self, messages, specs, model=None):
        FakeProvider.decide_models.append(model)
        FakeProvider.last_specs = list(specs)
        return FakeProvider.tool


def _env(**kv):
    """Patch exactly these optimizer vars, clearing any inherited from the shell."""
    base = {
        "OPTIMIZER_MODEL": "",
        "OPTIMIZER_MODEL_LADDER": "",
        "OPTIMIZER_MIN_CONFIDENCE": "",
        "OPTIMIZER_READ_CONTEXT": "",
        "OPTIMIZER_MAX_CONTEXT_CALLS": "",
    }
    return mock.patch.dict(os.environ, {**base, **kv})


def _drain(optimizer, specs=()):
    async def go():
        return [ev async for ev in optimizer.run(list(specs))]

    return asyncio.run(go())


class Ladder(unittest.TestCase):
    def test_unset_is_a_single_rung(self):
        with _env():
            self.assertEqual(_ladder(), [None])

    def test_falls_back_to_optimizer_model(self):
        with _env(OPTIMIZER_MODEL="qwen2.5:14b-instruct-q5_K_M"):
            self.assertEqual(_ladder(), ["qwen2.5:14b-instruct-q5_K_M"])

    def test_parses_weakest_first_and_preserves_order(self):
        with _env(OPTIMIZER_MODEL_LADDER=" small:7b , mid:14b ,, big:32b "):
            self.assertEqual(_ladder(), ["small:7b", "mid:14b", "big:32b"])


class Sufficient(unittest.TestCase):
    def test_high_confidence_accepted(self):
        with _env():
            self.assertTrue(_sufficient(_result())[0])

    def test_low_confidence_rejected(self):
        with _env():
            accept, reason = _sufficient(_result(confidence="low"))
            self.assertFalse(accept)
            self.assertIn("low", reason)

    def test_threshold_is_configurable(self):
        with _env(OPTIMIZER_MIN_CONFIDENCE="low"):
            self.assertTrue(_sufficient(_result(confidence="low"))[0])

    def test_empty_prompt_rejected_despite_high_confidence(self):
        # The degeneracy trigger — the one that doesn't trust the model's self-report.
        with _env():
            accept, reason = _sufficient(_result(prompt="   ", confidence="high"))
            self.assertFalse(accept)
            self.assertEqual(reason, "empty prompt")

    def test_invalid_tier_rejected_despite_high_confidence(self):
        # `_render` would silently drop the suggestion block for this result.
        with _env():
            accept, reason = _sufficient(_result(model="gpt-4", confidence="high"))
            self.assertFalse(accept)
            self.assertIn("MODEL_CHOICES", reason)

    def test_much_shorter_rewrite_rejected(self):
        with _env():
            original = "x" * 200
            accept, reason = _sufficient(_result(prompt="do it", confidence="high"), original)
            self.assertFalse(accept)
            self.assertIn("shorter", reason)

    def test_missing_confidence_rejected(self):
        with _env():
            result = _result()
            del result["confidence"]
            self.assertFalse(_sufficient(result)[0])


class Escalation(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(po, "OllamaProvider", FakeProvider)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, results, **env):
        FakeProvider.script(results)
        with _env(**env):
            return _drain(PromptOptimizer([{"role": "user", "content": "add pagination"}]))

    def test_confident_first_rung_does_not_escalate(self):
        events = self._run([_result()], OPTIMIZER_MODEL_LADDER="small:7b,big:32b")
        self.assertEqual(FakeProvider.models_used, ["small:7b"])
        self.assertIn(f"{ANALYZED_BY_LABEL} small:7b", events[0].text)
        self.assertNotIn("escalated", events[0].text)

    def test_low_confidence_climbs_the_ladder(self):
        events = self._run(
            [_result(confidence="low"), _result(confidence="low"), _result(confidence="high")],
            OPTIMIZER_MODEL_LADDER="small:7b,mid:14b,big:32b",
        )
        self.assertEqual(FakeProvider.models_used, ["small:7b", "mid:14b", "big:32b"])
        self.assertIn(f"{ANALYZED_BY_LABEL} big:32b", events[0].text)
        self.assertIn("escalated 2×", events[0].text)

    def test_degenerate_result_climbs_even_when_confident(self):
        events = self._run(
            [_result(prompt="", confidence="high"), _result()],
            OPTIMIZER_MODEL_LADDER="small:7b,big:32b",
        )
        self.assertEqual(FakeProvider.models_used, ["small:7b", "big:32b"])
        self.assertIn(f"{ANALYZED_BY_LABEL} big:32b", events[0].text)

    def test_exhausted_ladder_uses_the_strongest_result(self):
        events = self._run(
            [_result(confidence="low"), _result(confidence="low")],
            OPTIMIZER_MODEL_LADDER="small:7b,big:32b",
        )
        self.assertEqual(FakeProvider.models_used, ["small:7b", "big:32b"])
        # Still a usable proposal rather than nothing.
        self.assertIn("Add pagination", events[0].text)

    def test_unset_ladder_is_one_call(self):
        events = self._run([_result()])
        self.assertEqual(FakeProvider.models_used, [None])
        self.assertIsInstance(events[0], TextEvent)


class ContextGathering(unittest.TestCase):
    SPECS = [ToolSpec("get_file_text_by_path", "read a file", {"type": "object"})]

    def setUp(self):
        patcher = mock.patch.object(po, "OllamaProvider", FakeProvider)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_chosen_tool_becomes_a_tool_call_and_ends_the_turn(self):
        FakeProvider.script([_result()], tool=("get_file_text_by_path", {"pathInProject": "a.py"}))
        with _env():
            events = _drain(
                PromptOptimizer([{"role": "user", "content": "fix the bug"}]), self.SPECS
            )
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], ToolCallEvent)
        self.assertEqual(events[0].name, "get_file_text_by_path")
        # No rewrite happened on a gather turn.
        self.assertEqual(FakeProvider.models_used, [])

    def test_no_tool_falls_through_to_the_rewrite(self):
        FakeProvider.script([_result()], tool=None)
        with _env():
            events = _drain(
                PromptOptimizer([{"role": "user", "content": "fix the bug"}]), self.SPECS
            )
        self.assertIsInstance(events[0], TextEvent)

    def test_write_tools_are_never_offered(self):
        FakeProvider.script([_result()], tool=None)
        specs = [*self.SPECS, ToolSpec("create_new_file", "write", {"type": "object"})]
        with _env():
            _drain(PromptOptimizer([{"role": "user", "content": "fix"}]), specs)
        offered = {s.name for s in FakeProvider.last_specs}
        self.assertNotIn("create_new_file", offered)
        self.assertIn("get_file_text_by_path", offered)

    def test_budget_stops_gathering(self):
        history = [{"role": "user", "content": "fix"}] + [
            {"role": "tool", "content": "...", "tool_call_id": f"call_{i}"} for i in range(3)
        ]
        FakeProvider.script([_result()], tool=("get_file_text_by_path", {}))
        with _env(OPTIMIZER_MAX_CONTEXT_CALLS="3"):
            events = _drain(PromptOptimizer(history), self.SPECS)
        # Budget spent → rewrite instead of another read, so the loop terminates.
        self.assertIsInstance(events[0], TextEvent)

    def test_kill_switch_disables_gathering(self):
        FakeProvider.script([_result()], tool=("get_file_text_by_path", {}))
        with _env(OPTIMIZER_READ_CONTEXT="false"):
            events = _drain(
                PromptOptimizer([{"role": "user", "content": "fix"}]), self.SPECS
            )
        self.assertIsInstance(events[0], TextEvent)


class ProposalRoundTrip(unittest.TestCase):
    """The regression that would silently poison coder prompts."""

    def _proposal(self, rendered: str) -> str:
        return ENHANCED_PROMPT_MARKER + "\n" + rendered + _FOOTER

    def test_analyzed_by_is_stripped_before_the_coder(self):
        rendered = _render(_result(), analyzed_by="qwen2.5:32b-instruct", escalations=2)
        proposal = self._proposal(rendered)
        self.assertIn(ANALYZED_BY_LABEL, proposal)
        self.assertEqual(_extract_prompt(proposal), GOOD["prompt"])
        self.assertNotIn(ANALYZED_BY_LABEL, _extract_prompt(proposal))

    def test_analyzed_by_does_not_shadow_the_coder_tier(self):
        # `_extract_model` returns the FIRST line starting with "model:"; a label like
        # "Model: qwen2.5:32b" here would shadow the tier and silently pick the default.
        rendered = _render(_result(), analyzed_by="qwen2.5:32b-instruct", escalations=1)
        self.assertFalse(ANALYZED_BY_LABEL.lower().startswith("model:"))
        self.assertEqual(_extract_model(self._proposal(rendered)), "sonnet")

    def test_render_without_escalation_metadata_is_unchanged(self):
        # Back-compat: the pre-escalation call shape still produces the old block.
        rendered = _render(_result())
        self.assertNotIn(ANALYZED_BY_LABEL, rendered)
        self.assertEqual(_extract_model(self._proposal(rendered)), "sonnet")
        self.assertEqual(_extract_prompt(self._proposal(rendered)), GOOD["prompt"])


if __name__ == "__main__":
    unittest.main()

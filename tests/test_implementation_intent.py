"""Implement-mode scope extraction is bounded and conservative."""

import unittest

from agent.spec.implementation_intent import (
    ImplementationIntentError, ImplementationIntentService, selectors_in_text,
)


class _Provider:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    async def emit_json(self, messages, schema, model=None):
        self.calls.append((messages, schema, model))
        if self.error:
            raise self.error
        return self.value


class ImplementationIntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_selectors_are_canonicalized(self):
        provider = _Provider({
            "selectors": ["wp4", "WP 5"], "instruction": "from codespec",
            "confidence": "high", "ambiguities": [],
        })
        intent = await ImplementationIntentService(provider).interpret(
            "implement WP4 and WP5 from codespec")
        self.assertEqual(intent.selectors, ("WP4", "WP5"))
        self.assertEqual(intent.instruction, "")
        self.assertFalse(intent.fallback_used)
        self.assertEqual(len(provider.calls), 1)

    async def test_cheap_model_failure_has_conservative_selector_fallback(self):
        service = ImplementationIntentService(_Provider(error=RuntimeError("offline")))
        intent = await service.interpret("implement WP4 and WP5 from codespec")
        self.assertEqual(intent.selectors, ("WP4", "WP5"))
        self.assertTrue(intent.fallback_used)
        with self.assertRaises(ImplementationIntentError):
            await service.interpret("make it faster")

    def test_selectors_in_text(self):
        self.assertEqual(selectors_in_text("retry wp4-t1 and WP 5"), ("WP4-T1", "WP5"))


if __name__ == "__main__":
    unittest.main()

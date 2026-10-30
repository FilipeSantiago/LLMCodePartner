"""OpenSpec selectors are recognized in free text."""

import unittest

from agent.spec.implementation_intent import selectors_in_text


class SelectorTests(unittest.TestCase):
    def test_selectors_in_text(self):
        self.assertEqual(selectors_in_text("retry wp4-t1 and WP 5"), ("WP4-T1", "WP5"))


if __name__ == "__main__":
    unittest.main()

"""Every agent prompt matches its pinned text in `tests/fixtures/prompts`.

A failure here means a prompt changed. If the change is intended, regenerate the
fixtures with `python scripts/capture_prompt_goldens.py` and review the diff."""

import asyncio
import difflib
import unittest

from tests import prompt_scenarios


class PromptGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rendered = asyncio.run(prompt_scenarios.render_all())

    def test_every_fixture_has_a_scenario(self):
        fixtures = {path.stem for path in prompt_scenarios.FIXTURES.glob("*.txt")}
        self.assertEqual(fixtures, set(self.rendered))

    def test_prompts_match_fixtures(self):
        for name, text in sorted(self.rendered.items()):
            with self.subTest(prompt=name):
                expected = (prompt_scenarios.FIXTURES / f"{name}.txt").read_text()
                if text != expected:
                    diff = "".join(difflib.unified_diff(
                        expected.splitlines(keepends=True), text.splitlines(keepends=True),
                        "fixture", "rendered", n=1))
                    self.fail(f"{name} prompt changed:\n{diff[:4000]}")


if __name__ == "__main__":
    unittest.main()

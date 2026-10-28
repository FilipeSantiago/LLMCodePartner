"""Regenerate the prompt fixtures pinned by tests/test_prompt_golden.py.

Run after an intended prompt change, then review the fixture diff:

    .venv/bin/python scripts/capture_prompt_goldens.py && git diff tests/fixtures/prompts
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import prompt_scenarios  # noqa: E402


def main() -> None:
    rendered = asyncio.run(prompt_scenarios.render_all())
    prompt_scenarios.FIXTURES.mkdir(parents=True, exist_ok=True)
    for stale in prompt_scenarios.FIXTURES.glob("*.txt"):
        if stale.stem not in rendered:
            stale.unlink()
    for name, text in sorted(rendered.items()):
        (prompt_scenarios.FIXTURES / f"{name}.txt").write_text(text)
    print(f"wrote {len(rendered)} prompt fixtures to {prompt_scenarios.FIXTURES}")


if __name__ == "__main__":
    main()

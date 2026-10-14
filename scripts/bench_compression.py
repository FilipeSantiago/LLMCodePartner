"""Offline replay benchmark for coder tool-result compression.

Replays recorded tool outputs through `ContextCompressionService` with the real
context-compress bridge and reports, per sample and mode, bytes and estimated
tokens before/after, whether the guards accepted the result, and whether
repeated runs were byte-identical. Estimated tokens are bytes/4, not provider
counts; actual savings come from the online A/B (docs/context-compression.md).

    uv run python scripts/bench_compression.py                # generated corpus
    uv run python scripts/bench_compression.py --corpus DIR   # your recorded outputs
    uv run python scripts/bench_compression.py --json out.json

A sample's category comes from its name: `search*` → search, `list*` →
listing, `cmd*`/`test*`/`log*` → command, anything else → other.
Requires `npm ci --ignore-scripts` in infra/compress-bridge (local backend),
or `--backend sidecar --socket PATH` for a running sidecar.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_bridge.compression import (  # noqa: E402
    CompressionConfig, ContextCompressionService, estimate_tokens, make_backend)

_FAILING_TEST = '''
import unittest
def helper(d):
    return d["missing"]
class TestThing(unittest.TestCase):
    def test_keyerror(self):
        helper({"a": 1})
    def test_assert_diff(self):
        self.assertEqual({"x": [1, 2, 3], "y": "abc"}, {"x": [1, 2, 4], "y": "abd"})
    def test_many(self):
        for n in range(30):
            with self.subTest(n=n):
                print(f"captured stdout line for case {n}")
                self.assertNotEqual(n % 7, 3, f"bad n={n}")
'''


def category_of(name: str) -> str:
    if name.startswith("search"):
        return "search"
    if name.startswith("list"):
        return "listing"
    if name.startswith(("cmd", "test", "log")):
        return "command"
    return "other"


def _run(*command: str, cwd: Path = ROOT) -> str:
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return done.stdout + done.stderr


def generate_corpus(target: Path) -> None:
    """Representative outputs captured from this repository."""
    (target / "other_git_log.txt").write_text(_run("git", "log", "-n", "60", "--stat"))
    (target / "search_defs.txt").write_text(
        _run("git", "grep", "-n", "def ", "--", "agent", "mcp_bridge", "telemetry"))
    (target / "list_tree.txt").write_text(_run("git", "ls-files"))
    with tempfile.TemporaryDirectory() as directory:
        test = Path(directory) / "test_bench_fail.py"
        test.write_text(_FAILING_TEST)
        (target / "test_unittest_fail.txt").write_text(
            _run(sys.executable, "-I", "-m", "unittest", "-v", str(test), cwd=Path(directory)))
    (target / "other_structured.json").write_text(json.dumps({"results": [
        {"file": f"agent/x{i}.py", "line": i, "symbol": f"Foo{i}.bar", "id": f"call_{i:024x}"}
        for i in range(400)]}, indent=2))
    heartbeat = [f"2026-10-13T10:{i % 60:02d}:{i % 60:02d} INFO worker[{i % 4}] heartbeat ok"
                 for i in range(3000)]
    (target / "log_heartbeat.txt").write_text("\n".join(heartbeat))
    (target / "log_with_errors.txt").write_text("\n".join(
        heartbeat[:1000] + ['Traceback (most recent call last):',
                            '  File "agent/routing/service.py", line 88, in run',
                            "ValueError: bad state id=job_42"] + heartbeat[1000:]))
    colored = "".join(f"\x1b[32mPASSED\x1b[0m tests/test_m.py::test_{i}\n" for i in range(300))
    (target / "cmd_colored.txt").write_text(colored)


async def bench(corpus: Path, config: CompressionConfig, repeats: int) -> list[dict]:
    rows = []
    backend = make_backend(config)
    try:
        for path in sorted(p for p in corpus.iterdir() if p.is_file()):
            text = path.read_text(errors="replace")
            category = category_of(path.name)
            for mode in ("conservative", "balanced"):
                outputs = set()
                result = None
                for _ in range(repeats):
                    # A fresh service per repeat bypasses the memo: this measures
                    # the upstream pipeline's own determinism.
                    service = ContextCompressionService(replace(config, mode=mode), backend)
                    result = await service.compress(category, text)
                    outputs.add(result.text)
                rows.append({
                    "sample": path.name, "category": category, "mode": mode,
                    "effective_mode": result.mode, "applied": result.applied,
                    "reason": result.reason, "bytes_in": result.bytes_in,
                    "bytes_out": result.bytes_out, "est_tokens_in": estimate_tokens(text),
                    "est_tokens_out": result.est_tokens_out,
                    "deterministic": len(outputs) == 1, "version": result.version,
                })
    finally:
        if hasattr(backend, "_close"):
            await backend._close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--backend", choices=("local", "sidecar"), default="local")
    parser.add_argument("--socket", default=CompressionConfig.socket)
    parser.add_argument("--min-bytes", type=int, default=CompressionConfig.min_bytes)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, help="also write rows as JSON")
    args = parser.parse_args()
    config = CompressionConfig(enabled=True, backend=args.backend, socket=args.socket,
                               min_bytes=args.min_bytes, timeout_ms=30_000)
    with tempfile.TemporaryDirectory() as generated:
        corpus = args.corpus
        if corpus is None:
            corpus = Path(generated)
            generate_corpus(corpus)
        rows = asyncio.run(bench(corpus, config, args.repeats))
    print(f"{'sample':26} {'cat':8} {'mode':12} {'eff':12} {'bytes in→out':>17} "
          f"{'est tok in→out':>16} {'save':>5} det  result")
    for row in rows:
        saved = 1 - row["bytes_out"] / row["bytes_in"] if row["bytes_in"] else 0
        print(f"{row['sample'][:26]:26} {row['category']:8} {row['mode']:12} "
              f"{str(row['effective_mode']):12} {row['bytes_in']:>8}→{row['bytes_out']:<8} "
              f"{row['est_tokens_in']:>7}→{row['est_tokens_out']:<8} {saved:>5.0%} "
              f"{'yes' if row['deterministic'] else 'NO ':4} {row['reason']}")
    for mode in ("conservative", "balanced"):
        selected = [row for row in rows if row["mode"] == mode]
        before = sum(row["est_tokens_in"] for row in selected)
        after = sum(row["est_tokens_out"] for row in selected)
        print(f"\n{mode}: estimated tokens {before}→{after} "
              f"({1 - after / before if before else 0:.1%} saved), "
              f"{sum(row['applied'] for row in selected)}/{len(selected)} applied, "
              f"all deterministic={all(row['deterministic'] for row in selected)}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

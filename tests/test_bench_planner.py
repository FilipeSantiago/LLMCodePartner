"""Planner benchmark reduction: usage shapes, estimates and quality fields."""

import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "bench_planner", Path(__file__).resolve().parents[1] / "scripts" / "bench_planner.py")
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def _discovery(est=100):
    return {"retrieval_levels": {"L1": 2, "L2": 1, "L3": 1}, "l3_direct": 1,
            "l3_escalated": 0, "unexplained_reads": 1, "est_added_tokens": est,
            "evidence_strength": {"strong": 1, "moderate": 0, "weak": 1, "none": 0},
            "kb_fallback": False}


def _turn(usage, kind="plan", revision=1, est=100):
    return {"revision": revision, "kind": kind, "usage": usage, "latency_ms": 10,
            "tool_metrics": {"tool_calls": 3, "file_reads": 2, "repeated_source_reads": 1,
                             "discovery": _discovery(est)}}


SESSION = {"revision": 1, "gaps": [], "draft": {
    "shared": {"relevant_files": ["src/a.py"]},
    "tasks": {"T1": {"relevant_files": ["src/b.py", "src/extra.py"]}},
    "reviewer": {"relevant_files": ["src/a.py"]},
    "discovery": [{"file": "tests/test_a.py", "reason": "tests"}]}}


class UsageShapeTests(unittest.TestCase):
    def test_anthropic_cache_tokens_are_added_once(self):
        usage = {"input_tokens": 100, "cache_read_input_tokens": 900,
                 "cache_creation_input_tokens": 50, "output_tokens": 20}
        run = bench.run_metrics([_turn(usage), _turn(usage, kind="repair")], None, None)
        self.assertEqual((run["uncached_input_tokens"], run["cache_read_tokens"],
                          run["cache_write_tokens"], run["total_input_tokens"]),
                         (200, 1800, 100, 2100))
        self.assertEqual(run["repair_turns"], 1)

    def test_codex_cached_tokens_are_not_counted_twice(self):
        usage = {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 20}
        run = bench.run_metrics([_turn(usage)], None, None)
        self.assertEqual((run["uncached_input_tokens"], run["cache_read_tokens"],
                          run["cache_write_tokens"], run["total_input_tokens"]),
                         (200, 800, 0, 1000))

    def test_missing_usage_is_not_reported_as_zero(self):
        run = bench.run_metrics([_turn({"input_tokens": 5}), _turn(None)], None, None)
        self.assertIsNone(run["total_input_tokens"])
        summary = bench.summarize([run, bench.run_metrics([_turn({"input_tokens": 7})],
                                                          None, None)])
        self.assertEqual(summary["total_input_tokens"]["n"], 1)
        self.assertEqual(summary["total_input_tokens"]["missing"], 1)

    def test_metrics_older_rows_never_recorded_are_not_zero(self):
        turn = _turn({"input_tokens": 1})
        turn["tool_metrics"] = {"tool_calls": 4, "file_reads": 2}
        run = bench.run_metrics([turn], None, None)
        self.assertEqual(run["tool_calls"], 4)
        self.assertIsNone(run["whole_file_reads"])
        self.assertNotIn("L3", run)

    def test_estimate_stays_separate_from_provider_usage(self):
        run = bench.run_metrics([_turn({"input_tokens": 5000}, est=123)], None, None)
        self.assertEqual(run["est_added_tokens"], 123)
        self.assertEqual(run["total_input_tokens"], 5000)

    def test_gate_outcomes_and_redundant_reads_are_separate(self):
        turn = _turn({"input_tokens": 5000})
        turn["tool_metrics"]["discovery"].update({
            "search_gate": {"kb_sufficient": 1, "kb_insufficient": 0},
            "outline_gate": {"refined_to_range": 1, "escalated_to_full": 0},
            "kb_fallback_or_escalation_count": 0,
            "redundant_source_reads": 2,
        })
        run = bench.run_metrics([turn], None, None)
        self.assertEqual(run["refined_to_range"], 1)
        self.assertEqual(run["kb_fallback_or_escalation_count"], 0)
        self.assertEqual(run["redundant_source_reads"], 2)


class QualityTests(unittest.TestCase):
    def test_changed_file_recall_and_rubric_are_reported_separately(self):
        rubric = {"completeness": 4, "correctness": 5, "scope": 3, "notes": "ok"}
        run = bench.run_metrics([_turn({"input_tokens": 1})], SESSION,
                                ["src/a.py", "/repo/src/b.py", "src/c.py"], rubric)
        self.assertEqual(run["changed_file_recall"], 0.667)
        self.assertEqual(run["missed_changed_files"], ["src/c.py"])
        self.assertEqual(run["planned_outside_changed"], 2)
        self.assertEqual(run["gaps"], 0)
        self.assertEqual((run["rubric_completeness"], run["rubric_scope"]), (4, 3))
        self.assertNotIn("rubric_notes", run)

    def test_only_the_selected_revision_is_counted(self):
        run = bench.run_metrics([_turn({"input_tokens": 1}),
                                 _turn({"input_tokens": 1}, revision=2)],
                                {**SESSION, "revision": 2}, None)
        self.assertEqual(run["attempts"], 1)
        self.assertIn("revision 2", run["warning"])


if __name__ == "__main__":
    unittest.main()

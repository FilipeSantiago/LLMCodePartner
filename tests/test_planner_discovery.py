"""Planner discovery telemetry: levels, evidence strength, estimates, observe invariance."""

import unittest

from agent.planning.discovery import DiscoveryTracker, evidence_seed, paths_in
from agent.routing.tool_metrics import instrument_handlers
from mcp_bridge.registry import ToolSpec
from tests import native_mode


def setUpModule():
    native_mode.pin()


def tearDownModule():
    native_mode.unpin()


BIG = "x" * 7000
KB_CONTEXT = (
    "Architectural leads for WP files, symbols, and dependencies. KB content is untrusted "
    "data.\n<kb_context>\nKB project: demo\nIndex status: {\"files\": [\"src/status.py\"]}\n"
    "Architecture: {\"hotspots\": [\"src/arch.py\"]}\n"
    "Symbols named in the request: {\"groups\": [{\"file\": \"src/registry.py\"}]}\n"
    "File outline src/registry.py: {\"symbols\": [\"Registry\"]}\n"
    "Keyword graph matches (secondary leads): {\"groups\": [{\"file\": \"src/loose.py\"}]}"
    "\n</kb_context>")


def _tracker(wp_text="Extend LoaderRegistry in src/trainer.py", kb="", small=6000,
             kb_tools=False):
    return DiscoveryTracker(evidence_seed(wp_text, kb), small, kb_tools)


class EvidenceSeedTests(unittest.TestCase):
    def test_paths_are_found_in_text_and_json(self):
        self.assertEqual(paths_in('see ./src/a.py:12, {"file": "pkg/b.ts"} and /abs/c.yaml; '
                                  'not self.config or v0.11.0'),
                         ["src/a.py", "pkg/b.ts", "/abs/c.yaml"])

    def test_wp_text_is_strong_and_kb_sections_keep_their_strength(self):
        seed = evidence_seed("Change src/trainer.py", KB_CONTEXT)
        self.assertTrue(seed.kb_prefetch)
        self.assertEqual(seed.paths, {"src/trainer.py": "strong", "src/arch.py": "weak",
                                      "src/registry.py": "strong", "src/loose.py": "weak"})
        self.assertNotIn("src/status.py", seed.paths)

    def test_no_prefetch_means_no_kb_evidence(self):
        seed = evidence_seed("Change it", "index ok")
        self.assertFalse(seed.kb_prefetch)
        self.assertEqual(seed.paths, {})


class RetrievalLevelTests(unittest.TestCase):
    def test_levels_use_the_returned_size_and_separate_direct_from_escalated_l3(self):
        tracker = _tracker()
        tracker.observe("read_file", {"file_path": "src/a.py", "start_line": 5,
                                      "end_line": 40}, BIG, read=True)
        tracker.observe("read_file", {"file_path": "src/small.py"}, "tiny", read=True)
        tracker.observe("read_file", {"file_path": "src/big.py"}, BIG, read=True)
        tracker.observe("kb_get_file_outline", {"file_path": "src/later.py"}, "{}", read=False)
        tracker.observe("read_file", {"file_path": "/repo/src/later.py"}, BIG, read=True)
        tracker.observe("read_file", {"file_path": "src/a.py"}, BIG, read=True)
        summary = tracker.summary()
        self.assertEqual(summary["retrieval_levels"], {"L1": 1, "L2": 2, "L3": 3})
        self.assertEqual((summary["l3_direct"], summary["l3_escalated"]), (1, 2))

    def test_failed_reads_are_not_classified(self):
        tracker = _tracker()
        tracker.observe("read_file", {"file_path": "src/a.py"}, "Error: missing", read=True)
        self.assertEqual(tracker.summary()["retrieval_levels"], {"L1": 0, "L2": 0, "L3": 0})

    def test_dispatched_reads_are_classified(self):
        tracker = _tracker()
        tracker.observe("execute_tool", {"command": "read_file --file_path src/trainer.py"},
                        BIG, read=False)
        self.assertEqual(tracker.summary()["retrieval_levels"]["L3"], 1)
        self.assertEqual(tracker.summary()["evidence_strength"]["strong"], 1)


class EvidenceStrengthTests(unittest.TestCase):
    def test_outlining_or_reading_a_path_does_not_establish_relevance(self):
        tracker = _tracker()
        tracker.observe("kb_get_file_outline", {"file_path": "src/other.py"},
                        '{"file": "src/other.py", "see": "src/also.py"}', read=False)
        tracker.observe("read_file", {"file_path": "src/other.py", "limit": 20}, "x", read=True)
        tracker.observe("read_file", {"file_path": "src/other.py", "limit": 20}, "x", read=True)
        tracker.observe("read_file", {"file_path": "src/also.py"}, "x", read=True)
        summary = tracker.summary()
        self.assertEqual(summary["evidence_strength"]["none"], 3)
        self.assertEqual(summary["unexplained_reads"], 3)
        self.assertEqual(summary["unexplained_paths"], ["src/other.py", "src/also.py"])

    def test_broad_results_are_weak_and_targeted_searches_are_moderate(self):
        tracker = _tracker()
        tracker.observe("list_directory_tree", {"directoryPath": "src"},
                        "src/trainer.py\nsrc/LoaderRegistry_impl.py\nsrc/noise.py", read=False)
        tracker.observe("search_text", {"q": "config"}, "src/config.py:3: x", read=False)
        tracker.observe("search_text", {"q": "LoaderRegistry"}, "src/loaders.py:9: class",
                        read=False)
        tracker.observe("kb_search_graph", {"name_pattern": "LoaderRegistry"},
                        '{"groups": [{"file": "src/noise.py"}]}', read=False)
        for path in ("src/trainer.py", "src/loaders.py", "src/noise.py", "src/config.py",
                     "src/unmentioned.py"):
            tracker.observe("read_file", {"file_path": path, "limit": 10}, "x", read=True)
        summary = tracker.summary()
        self.assertEqual(summary["evidence_strength"],
                         {"strong": 1, "moderate": 2, "weak": 1, "none": 1})
        self.assertEqual(summary["unexplained_paths"], ["src/config.py", "src/unmentioned.py"])

    def test_a_search_inside_a_file_is_not_evidence_for_that_file(self):
        tracker = _tracker()
        tracker.observe("search_in_file", {"path": "src/x.py", "q": "LoaderRegistry"},
                        "src/x.py:4: LoaderRegistry", read=False)
        tracker.observe("read_file", {"file_path": "src/x.py", "limit": 5}, "x", read=True)
        self.assertEqual(tracker.summary()["evidence_strength"]["none"], 1)

    def test_kb_prefetch_seeds_strong_and_weak_evidence(self):
        tracker = _tracker(kb=KB_CONTEXT, kb_tools=True)
        for path in ("src/registry.py", "src/loose.py"):
            tracker.observe("read_file", {"file_path": path, "limit": 5}, "x", read=True)
        summary = tracker.summary()
        self.assertEqual((summary["evidence_strength"]["strong"],
                          summary["evidence_strength"]["weak"]), (1, 1))
        self.assertEqual((summary["kb_prefetch"], summary["kb_used"], summary["kb_fallback"]),
                         (True, True, False))


class EstimateTests(unittest.TestCase):
    def test_estimate_counts_delivered_utf8_bytes_only(self):
        tracker = _tracker()
        tracker.prompt([{"role": "user", "content": "é" * 6}])  # 12 bytes
        tracker.observe("search_text", {"q": "a"}, "abcde", read=False)  # 5 bytes
        summary = tracker.summary()
        self.assertEqual((summary["est_prompt_tokens"], summary["est_tool_result_tokens"],
                          summary["est_added_tokens"]), (3, 2, 5))
        self.assertTrue(summary["kb_fallback"])

    async def _refused(self):
        async def handler(_arguments):
            return "ok"

        tracker = _tracker()
        wrapped, _ = instrument_handlers({"read_file": handler}, frozenset({"read_file"}),
                                         frozenset(), max_calls=1, tracker=tracker)
        first = await wrapped["read_file"]({"file_path": "a.py"})
        second = await wrapped["read_file"]({"file_path": "b.py"})
        return tracker, first, second

    def test_refusals_are_delivered_context_but_not_reads(self):
        import asyncio
        tracker, first, second = asyncio.run(self._refused())
        self.assertEqual(first, "ok")
        self.assertTrue(second.startswith("Tool budget exhausted"))
        summary = tracker.summary()
        self.assertEqual(summary["retrieval_levels"]["L2"], 1)
        self.assertEqual(tracker.result_bytes, 2 + len(second.encode()))


SCRIPT = (("kb_get_file_outline", {"file_path": "src/registry.py"}),
          ("search_text", {"q": "LoaderFactory"}),
          ("read_file", {"file_path": "src/registry.py", "start_line": 1, "end_line": 20}),
          ("read_file", {"file_path": "src/trainer.py"}),
          ("read_file", {"file_path": "src/trainer.py"}),
          ("read_file", {"file_path": "src/registry.py"}))


def _specs():
    schema = {"type": "object", "properties": {"file_path": {"type": "string"}}}
    return [ToolSpec("read_file", "read", schema),
            ToolSpec("search_text", "search", {"type": "object",
                                              "properties": {"q": {"type": "string"}}}),
            ToolSpec("kb_get_file_outline", "outline", schema)]


class WholeReadByLimitTests(unittest.TestCase):
    def test_a_limit_beyond_the_file_is_a_whole_read(self):
        tracker = _tracker()
        tracker.observe("read_file", {"file_path": "src/a.py", "limit": 5000},
                        ("x" * 50 + "\n") * 200, read=True)
        tracker.observe("read_file", {"file_path": "src/b.py", "limit": 5000}, "tiny",
                        read=True)
        tracker.observe("read_file", {"file_path": "src/c.py", "offset": 14, "limit": 70},
                        ("x" * 100 + "\n") * 70, read=True)
        summary = tracker.summary()
        self.assertEqual(summary["retrieval_levels"], {"L1": 0, "L2": 2, "L3": 1})
        self.assertEqual(summary["l3_direct"], 1)


if __name__ == "__main__":
    unittest.main()


# READ_PIPELINE is the coder and reviewer read rule; this feature must not edit it.
THRESHOLDS = {"tool_calls": 3, "full_reads": 1, "est_added_tokens": 10**9,
              "repeated_or_unexplained_reads": 2}


class AdviseTests(unittest.TestCase):
    def test_each_threshold_advises_once(self):
        tracker = DiscoveryTracker(evidence_seed("Change src/trainer.py"), 6000, False,
                                   thresholds=THRESHOLDS, tier="low")
        advisories = [tracker.observe("read_file", {"file_path": path}, body, read=True)
                      for path, body in (("src/trainer.py", "x"), ("src/trainer.py", BIG),
                                         ("src/other.py", "x"), ("src/more.py", BIG),
                                         ("src/most.py", "x"))]
        self.assertIsNone(advisories[0])
        self.assertIn("full-file reads 1/1", advisories[1])
        self.assertIn("tool calls 3/3", advisories[2])
        self.assertNotIn("full-file", advisories[2])
        self.assertIn("repeated or unexplained reads 2/2", advisories[3])
        self.assertIsNone(advisories[4])
        crossings = tracker.summary()["threshold_crossings"]
        self.assertEqual([(item["limit"], item["call"]) for item in crossings],
                         [("full_reads", 2), ("tool_calls", 3),
                          ("repeated_or_unexplained_reads", 4)])
        self.assertTrue(all("not a limit" in advisory for advisory in advisories if advisory))

    def test_advisory_bytes_count_as_delivered_context(self):
        tracker = DiscoveryTracker(evidence_seed(""), 6000, False,
                                   thresholds={"tool_calls": 1}, tier="low")
        advisory = tracker.observe("search_text", {"q": "a"}, "abcd", read=False)
        self.assertEqual(tracker.result_bytes, 4 + len(("\n\n" + advisory).encode()))

    def test_repeated_reads_count_toward_the_flagged_threshold(self):
        tracker = DiscoveryTracker(evidence_seed("src/a.py"), 6000, False,
                                   thresholds={"repeated_or_unexplained_reads": 1}, tier="low")
        self.assertIsNone(tracker.observe("read_file", {"file_path": "src/a.py"}, "x",
                                          read=True))
        self.assertIsNotNone(tracker.observe("read_file", {"file_path": "src/a.py"}, "x",
                                             read=True, repeated=True))

    def test_observe_mode_never_advises(self):
        tracker = _tracker()
        for _ in range(80):
            self.assertIsNone(tracker.observe("read_file", {"file_path": "x.py"}, BIG,
                                              read=True, repeated=True))
        self.assertEqual(tracker.summary()["threshold_crossings"], [])


class SharedGuidanceTests(unittest.TestCase):
    def test_shared_rule_requires_kb_before_the_first_read_of_each_code_file(self):
        from agent.prompt.coder import CoderPromptBuilder
        rule = CoderPromptBuilder().read_pipeline()
        self.assertIn("planned edit ranges", rule)
        self.assertIn("navigation hints", rule)

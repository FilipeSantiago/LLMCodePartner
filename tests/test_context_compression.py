"""Compression of coder tool results: guards, fail-open, provenance, retrieval."""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mcp_bridge.agent_tools import AgentToolInterface
from mcp_bridge.compression import (
    CompressionConfig, CompressorUnavailable, ContextCompressionService, LocalBackend)
from mcp_bridge.models import Run
from mcp_bridge.registry import ToolSpec
from telemetry.aggregation import TelemetryAggregation, normalize_usage
from telemetry.storage import TelemetryStorage

BRIDGE_DIR = Path(__file__).resolve().parent.parent / "infra" / "compress-bridge"


def _minify(text):
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except ValueError:
        return text


class FakeBackend:
    name = "fake"

    def __init__(self, transform=_minify, version="1.0", error=None):
        self.transform, self._version, self.error = transform, version, error
        self.calls = []

    async def version(self):
        if self.error:
            raise CompressorUnavailable(self.error)
        return self._version

    async def compress(self, mode, text):
        self.calls.append((mode, text))
        if self.error:
            raise CompressorUnavailable(self.error)
        return self.transform(text)


def _config(**overrides):
    return CompressionConfig(**{"enabled": True, "backend": "noop", "mode": "balanced",
                                "min_bytes": 200, **overrides})


def _big_json(count=60):
    return json.dumps({"items": [{"file": f"src/m{i}.py", "line": i, "id": f"call_{i:024x}"}
                                 for i in range(count)]}, indent=2)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_below_threshold_is_untouched_and_backend_not_called(self):
        backend = FakeBackend()
        service = ContextCompressionService(_config(min_bytes=10_000), backend)
        result = await service.compress("other", _big_json(5))
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "below_threshold")
        self.assertEqual(backend.calls, [])

    async def test_ineligible_category_and_failed_tool_are_untouched(self):
        service = ContextCompressionService(_config(), FakeBackend())
        self.assertEqual((await service.compress("read", _big_json())).reason, "not_eligible")
        self.assertEqual((await service.compress("other", _big_json(), failed=True)).reason,
                         "tool_failed")

    async def test_json_minify_is_accepted_only_when_parse_equal(self):
        text = _big_json()
        accepted = await ContextCompressionService(_config(), FakeBackend()).compress("other", text)
        self.assertTrue(accepted.applied)
        self.assertEqual(json.loads(accepted.text), json.loads(text))
        self.assertLess(accepted.bytes_out, accepted.bytes_in)
        changed = await ContextCompressionService(
            _config(), FakeBackend(lambda t: _minify(t).replace("m1.py", "m9.py"))
        ).compress("other", text)
        self.assertFalse(changed.applied)
        self.assertEqual(changed.reason, "json_changed")
        self.assertEqual(changed.text, text)

    async def test_error_output_is_forced_conservative(self):
        backend = FakeBackend(lambda t: t.replace("noise\n", ""))
        text = "noise\n" * 100 + "Traceback (most recent call last):\nValueError: bad\n"
        await ContextCompressionService(_config(mode="balanced"), backend).compress("command", text)
        self.assertEqual(backend.calls[0][0], "conservative")

    async def test_lost_identifier_falls_back(self):
        text = "\n".join(f"src/pkg/mod{i}.py:{i}: match" for i in range(40))
        drop_last = FakeBackend(lambda t: "\n".join(t.splitlines()[:-1]))
        result = await ContextCompressionService(_config(), drop_last).compress("search", text)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "identifier_lost")
        self.assertEqual(result.text, text)

    async def test_pytest_node_ids_and_frames_are_identifiers(self):
        text = ("FAILED tests/test_x.py::TestA::test_b[3] - AssertionError: n\n" * 3
                + '  File "agent/x.py", line 12, in run\n' + "pad\n" * 80)
        collapsed = FakeBackend(lambda t: "FAILED (3 tests) - AssertionError: n\n")
        result = await ContextCompressionService(_config(), collapsed).compress("command", text)
        self.assertEqual(result.reason, "identifier_lost")

    async def test_lost_error_line_falls_back(self):
        text = "info ok\n" * 80 + "ERROR worker 3 crashed\n"
        result = await ContextCompressionService(
            _config(), FakeBackend(lambda t: "info ok (x80)\n")).compress("other", text)
        self.assertEqual(result.reason, "error_line_lost")

    async def test_insufficient_saving_keeps_original(self):
        text = "plain line\n" * 60
        result = await ContextCompressionService(
            _config(), FakeBackend(lambda t: t[:-1])).compress("other", text)
        self.assertEqual(result.reason, "insufficient_saving")

    async def test_backend_failure_fails_open(self):
        text = _big_json()
        result = await ContextCompressionService(
            _config(), FakeBackend(error="socket missing")).compress("other", text)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "backend_error")
        self.assertEqual(result.text, text)

    async def test_upstream_hint_is_rewritten_to_authorized_retrieval(self):
        text = "line a\n" * 120
        backend = FakeBackend(lambda t: "line a\n[+119 lines omitted — use execute(intent:…) "
                                        "or search() for the full output]\n")
        result = await ContextCompressionService(_config(), backend).compress("other", text)
        self.assertTrue(result.applied)
        self.assertNotIn("execute(", result.text)
        self.assertIn("codepartner_get_observation", result.text)

    async def test_memo_key_covers_version_mode_and_config(self):
        text = _big_json()
        backend = FakeBackend()
        service = ContextCompressionService(_config(), backend)
        first = await service.compress("other", text)
        second = await service.compress("other", text)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(first.text, second.text)
        backend._version = "2.0"
        await service.compress("other", text)
        self.assertEqual(len(backend.calls), 2)
        service.config = _config(mode="conservative")
        await service.compress("other", text)
        self.assertEqual(len(backend.calls), 3)
        service.config = _config(mode="conservative", min_saving_ratio=0.5)
        await service.compress("other", text)
        self.assertEqual(len(backend.calls), 4)

    def test_error_detection_ignores_identifiers_that_contain_error(self):
        from mcp_bridge.compression import is_error_output
        for line in ("agent/x.py:12: class ErrorEvent:", "def f(position: int, error: str):",
                     "raise ValueError(\"x\")", "exit_code=0"):
            self.assertFalse(is_error_output(line), line)
        for line in ("ValueError: bad", "E   AssertionError: assert 1 == 2",
                     "FAILED tests/t.py::a - x", "src/main.rs:3:5: error[E0308]: mismatched",
                     "Traceback (most recent call last):", "exit_code=2", "npm ERR! code 1"):
            self.assertTrue(is_error_output(line), line)

    def test_auto_and_aggressive_modes_disable_compression(self):
        for mode in ("auto", "aggressive"):
            with mock.patch.dict(os.environ, {"CODEPARTNER_COMPRESS_ENABLED": "true",
                                              "CODEPARTNER_COMPRESS_MODE": mode}):
                config = CompressionConfig.from_env()
            self.assertFalse(config.enabled, mode)

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(CompressionConfig.from_env().enabled)


def _spec(name, properties=None):
    return ToolSpec(name, name, {"type": "object", "properties": properties or {}})


class InterfaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.payload = _big_json()
        self.backend_impl = FakeBackend()

    def _interface(self, compression=True):
        run = Run()
        run.registry.register([
            _spec("read_file", {"file_path": {}, "offset": {}, "limit": {}}),
            _spec("search_text", {"q": {}, "paths": {}, "limit": {}}),
            _spec("apply_patch", {"input": {}}),
        ])
        service = ContextCompressionService(_config(), self.backend_impl) if compression else None
        return AgentToolInterface(run, compression=service)

    async def ide(self, name, arguments):
        if name == "read_file":
            return "\n".join(f"L{i}: line {i}" for i in range(arguments["offset"], 21))
        if name == "search_text":
            return json.dumps({"items": [{"filePath": "src/app.py", "startLine": i}
                                         for i in range(1, 300)], "more": False})
        if name == "apply_patch":
            return "Done!"
        return self.payload

    async def test_ineligible_results_are_byte_identical_to_disabled(self):
        off = self._interface(compression=False)
        on = self._interface()
        on.compression.config = _config(categories=frozenset())
        calls = (("get_symbol_info", {"path": "src/app.py"}), ("search_text", {"q": "item"}),
                 ("read_file", {"file_path": "src/app.py", "offset": 1, "limit": 20}),
                 ("codepartner_get_observation", {"observation_id": "index"}))
        for name, args in calls:
            self.assertEqual(await off.execute(name, args, self.ide),
                             await on.execute(name, args, self.ide), name)
        self.assertEqual(self.backend_impl.calls, [])
        self.assertEqual(off.summary()["compression_mode"], "off")

    async def test_malformed_backend_response_fails_open(self):
        class Broken(FakeBackend):
            async def compress(self, mode, text):
                raise ValueError("not json")
        self.backend_impl = Broken()
        interface = self._interface()
        result = await interface.execute("get_symbol_info", {"path": "src/app.py"}, self.ide)
        self.assertIn(self.payload, result)
        self.assertEqual(interface.metrics["compress_fallbacks"], 1)

    async def test_persisted_text_excludes_file_reads(self):
        interface = self._interface()
        await interface.execute("read_file", {"file_path": "src/app.py", "offset": 1,
                                              "limit": 20}, self.ide)
        await interface.execute("get_symbol_info", {"path": "src/app.py"}, self.ide)
        categories = {record["observation_id"]: record["category"]
                      for record in interface.provenance}
        self.assertEqual({categories[key] for key in interface.persistable_keys()}, {"other"})

    async def test_compressed_result_has_marker_and_original_is_retrievable(self):
        interface = self._interface()
        result = await interface.execute("get_symbol_info", {"path": "src/app.py"}, self.ide)
        self.assertIn("compressed=balanced", result)
        key = result.split("observation_id='", 1)[1].split("'", 1)[0]
        self.assertTrue(result.startswith("TOOL COMPLETE\ntool=get_symbol_info\ntruncated=false\n"))
        reload = interface._reload({"observation_id": key})
        self.assertIn(self.payload, reload)
        self.assertEqual(interface.metrics["compress_applied"], 1)
        self.assertGreater(interface.metrics["est_tokens_saved"], 0)
        record = interface.provenance[-1]
        self.assertTrue(record["applied"])
        self.assertEqual(record["observation_id"], key)
        self.assertEqual(record["version"], "1.0")
        self.assertIn("src/app.py", record["resources"])

    async def test_duplicate_content_points_at_first_observation(self):
        interface = self._interface()
        first = await interface.execute("get_symbol_info", {"path": "src/app.py"}, self.ide)
        second = await interface.execute("get_symbol_info", {"path": "src/app.py"}, self.ide)
        key = first.split("observation_id='", 1)[1].split("'", 1)[0]
        self.assertIn("IDENTICAL RESULT", second)
        self.assertIn(key, second)
        self.assertEqual(interface.provenance[-1]["duplicate_of"], key)
        self.assertEqual(interface.provenance[-1]["reason"], "duplicate")
        self.assertFalse(interface.provenance[-1]["applied"])
        self.assertEqual(interface.metrics["dedup_hits"], 1)

    async def test_edit_marks_observations_stale_and_stops_dedup(self):
        interface = self._interface()
        searched = await interface.execute("search_text", {"q": "item"}, self.ide)
        read = await interface.execute("read_file", {"file_path": "src/app.py", "offset": 1,
                                                     "limit": 20}, self.ide)
        search_key = searched.rsplit("observation_id='", 1)[1].split("'", 1)[0]
        read_key = read.rsplit("observation_id='", 1)[1].split("'", 1)[0]
        await interface.execute("apply_patch", {"input": (
            "*** Begin Patch\n*** Update File: src/app.py\n@@\n-x\n+y\n*** End Patch")}, self.ide)
        self.assertIn("STALE:", interface._reload({"observation_id": search_key}))
        self.assertIn("STALE:", interface._reload({"observation_id": read_key}))
        again = await interface.execute("search_text", {"q": "item"}, self.ide)
        self.assertNotIn("IDENTICAL RESULT", again)

    async def test_search_is_scoped_to_the_run(self):
        interface = self._interface()
        result = await interface.execute("get_symbol_info", {"path": "src/app.py"}, self.ide)
        key = result.split("observation_id='", 1)[1].split("'", 1)[0]
        found = interface._reload({"observation_id": "search", "query": "m42.py"})
        self.assertIn(key, found)
        other = self._interface()
        self.assertIn("No observation in this run matches",
                      other._reload({"observation_id": "search", "query": "m42.py"}))
        self.assertIn("OBSERVATION UNAVAILABLE", other._reload({"observation_id": key}))
        self.assertIn("query must be", interface._reload({"observation_id": "search"}))

    async def test_backend_failure_delivers_uncompressed_result(self):
        self.backend_impl = FakeBackend(error="down")
        interface = self._interface()
        expected = await AgentToolInterface(interface.run).execute(
            "get_symbol_info", {"path": "src/app.py"}, self.ide)
        result = await interface.execute("get_symbol_info", {"path": "src/app.py"}, self.ide)
        self.assertEqual(result, expected)
        self.assertEqual(interface.metrics["compress_fallbacks"], 1)


class StorageAndMeasurementTests(unittest.IsolatedAsyncioTestCase):
    async def test_provenance_is_stored_without_text_unless_persist(self):
        interface = AgentToolInterface(Run(), compression=ContextCompressionService(
            _config(), FakeBackend()))

        async def ide(name, arguments):
            return _big_json()
        await interface.execute("get_symbol_info", {"path": "src/app.py"}, ide)
        with tempfile.TemporaryDirectory() as directory:
            storage = TelemetryStorage(Path(directory) / "t.sqlite3")
            storage.record_context_observations("run-1", {"task_id": "WP1-T1"},
                                                interface.provenance)
            storage.record_context_observations("run-2", {"task_id": "WP1-T1"},
                                                interface.provenance, interface.observations)
            with sqlite3.connect(storage.path) as db:
                rows = db.execute("SELECT run_id, applied, compressor_version, resources_json "
                                  "FROM context_observations ORDER BY run_id").fetchall()
                texts = db.execute("SELECT run_id FROM context_observation_text").fetchall()
        self.assertEqual([row[0] for row in rows], ["run-1", "run-2"])
        self.assertEqual(rows[0][1:3], (1, "1.0"))
        self.assertIn("src/app.py", json.loads(rows[0][3]))
        self.assertEqual(texts, [("run-2",)])

    def test_usage_normalization_for_claude_and_codex(self):
        claude = normalize_usage({"input_tokens": 100, "cache_read_input_tokens": 900,
                                  "cache_creation_input_tokens": 50, "output_tokens": 20})
        codex = normalize_usage({"input_tokens": 1000, "cached_input_tokens": 900,
                                 "output_tokens": 20})
        self.assertEqual(claude["uncached_input_tokens"], 100)
        self.assertEqual(claude["total_input_tokens"], 1050)
        self.assertEqual(codex["uncached_input_tokens"], 100)
        self.assertEqual(codex["cache_read_tokens"], 900)

    def test_comparison_separates_estimated_and_actual(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = TelemetryStorage(Path(directory) / "t.sqlite3")
            storage.record_aci_attempt("a", {"task_id": "T"}, {
                "observation_mode": "compact", "compression_mode": "off"},
                {"input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 10})
            storage.record_aci_attempt("b", {"task_id": "T"}, {
                "observation_mode": "compact", "compression_mode": "balanced",
                "est_tokens_saved": 300, "compress_applied": 2},
                {"input_tokens": 700, "cached_input_tokens": 400, "output_tokens": 10})
            comparison = TelemetryAggregation(storage).aci_comparison(["T"])["T"]
        self.assertEqual(set(comparison), {"compact", "compact+compress:balanced"})
        on = comparison["compact+compress:balanced"]
        self.assertEqual(on["estimated"]["est_tokens_saved"], 300)
        self.assertEqual(on["actual"]["uncached_input_tokens"], 300)
        self.assertEqual(comparison["compact"]["actual"]["uncached_input_tokens"], 600)
        self.assertAlmostEqual(on["cache_read_ratio"], 400 / 700, places=3)


@unittest.skipUnless((BRIDGE_DIR / "node_modules" / "context-compress").is_dir(),
                     "run `npm ci --ignore-scripts` in infra/compress-bridge")
class RealBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = LocalBackend(str(BRIDGE_DIR), timeout_ms=10_000)

    async def asyncTearDown(self):
        await self.backend._close()

    async def test_bridge_is_deterministic_and_rejects_unsafe_requests(self):
        self.assertEqual(await self.backend.version(), "2026.8.3")
        text = "\x1b[31mred\x1b[0m line\n" * 400
        first = await self.backend.compress("conservative", text)
        second = await self.backend.compress("conservative", text)
        self.assertEqual(first, second)
        self.assertNotIn("\x1b[", first)
        for payload in ({"op": "compress", "mode": "auto", "text": "x"},
                        {"op": "compress", "mode": "balanced", "cmd": "id", "text": "x"},
                        {"op": "execute", "code": "id"}):
            with self.assertRaises(CompressorUnavailable):
                await self.backend._request(payload)

    async def test_real_pipeline_through_service_keeps_json_values(self):
        service = ContextCompressionService(_config(backend="local"), self.backend)
        text = _big_json(400)
        result = await service.compress("other", text)
        self.assertTrue(result.applied, result.reason)
        self.assertEqual(json.loads(result.text), json.loads(text))


if __name__ == "__main__":
    unittest.main()

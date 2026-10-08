"""Behavioral checks for the coder-facing interface at the IDE boundary."""

import json
import tempfile
import unittest
from pathlib import Path

from mcp_bridge import bridge
from mcp_bridge.agent_tools import AgentToolInterface, _check_command, _command_result, _read_result
from mcp_bridge.edit_validation import validate_postimages
from mcp_bridge.models import Run
from mcp_bridge.registry import ToolSpec


def _spec(name, properties=None):
    return ToolSpec(name, name, {"type": "object", "properties": properties or {}})


class AgentToolInterfaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.run = Run()
        self.run.registry.register([
            _spec("read_file", {"file_path": {}, "offset": {}, "limit": {}}),
            _spec("search_text", {"q": {}, "paths": {}, "limit": {}}),
            _spec("apply_patch", {"input": {}}),
        ])
        self.interface = AgentToolInterface(self.run)
        self.calls = []

    async def backend(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "read_file":
            offset = arguments["offset"]
            limit = arguments["limit"]
            # PyCharm returns the first and last half of an overlarge file.
            first = range(offset, offset + limit // 2)
            tail = range(337, 438)
            return ("\n".join(f"L{i}: line {i}" for i in first)
                    + "\n…280 lines truncated…\n"
                    + "\n".join(f"L{i}: line {i}" for i in tail))
        if name == "search_text":
            return json.dumps({"items": [
                {"filePath": "src/app.py", "startLine": 12},
                {"filePath": ".env", "startLine": 2},
            ], "more": False})
        return "Done!"

    async def test_read_returns_contiguous_window_with_real_total(self):
        result = await self.interface.execute(
            "codepartner_read_file", {"path": "src/app.py", "start_line": 140,
                                      "end_line": 220}, self.backend)
        self.assertIn("lines 140–220 of 437", result)
        self.assertIn("Remaining: 221–437", result)
        self.assertIn("  220 | line 220", result)
        self.assertNotIn("line 431", result)
        self.assertEqual(self.calls[0][1]["limit"], 164)

    def test_missing_file_is_reported_as_failure_not_source(self):
        result, lines, body = _read_result(
            "File src/missing.py doesn't exist or can't be opened",
            "src/missing.py", 1, 100)
        self.assertIn("READ FAILED", result)
        self.assertEqual(lines, 0)
        self.assertEqual(body, "")

    def test_terminal_check_rejects_single_pipe_and_newline(self):
        self.assertIsNotNone(_check_command("pytest | tee results.txt"))
        self.assertIsNotNone(_check_command("pytest\nenv"))

    async def test_duplicate_read_is_verified_again_before_short_reply(self):
        arguments = {"path": "src/app.py", "start_line": 1, "end_line": 10}
        await self.interface.execute("codepartner_read_file", arguments, self.backend)
        result = await self.interface.execute("codepartner_read_file", arguments, self.backend)
        self.assertIn("DUPLICATE READ", result)
        self.assertEqual(len(self.calls), 2)

    async def test_large_ide_read_reaches_model_as_small_window(self):
        self.run.tool_interface = self.interface
        self.run.direct_tool_handlers = {"read_file": lambda arguments: self.backend("read_file", arguments)}
        result = await bridge.call_tool(
            self.run, "codepartner_read_file", {"path": "src/app.py"})
        self.assertIn("lines 1–100 of 437", result)
        self.assertGreater(self.interface.metrics["backend_output_bytes"],
                           self.interface.metrics["model_output_bytes"])
        self.assertEqual(self.interface.metrics["lines_returned"], 100)

    async def test_protected_paths_never_reach_backend(self):
        for path in (".env", "config/.env.local", ".git/config", "../outside.py",
                     "/tmp/file.py", ".codepartner/ide_mcp.json"):
            result = await self.interface.execute(
                "codepartner_read_file", {"path": path}, self.backend)
            self.assertIn("ACCESS DENIED", result, path)
        self.assertEqual(self.calls, [])

    async def test_env_example_is_an_ordinary_documentation_file(self):
        result = await self.interface.execute(
            "codepartner_read_file", {"path": ".env.example", "end_line": 2}, self.backend)
        self.assertIn("READ COMPLETE", result)

    async def test_search_filters_secret_hits_and_expansion_cannot_restore_them(self):
        result = await self.interface.execute("search_text", {"q": "TOKEN"}, self.backend)
        self.assertIn("src/app.py:12", result)
        self.assertNotIn(".env:2", result)
        self.assertIn("Full authorized output", result)
        expanded = self.interface._reload({"observation_id": "obs-1"})
        self.assertIn("src/app.py", expanded)
        self.assertNotIn(".env", expanded)

    async def test_file_search_passes_exact_path_filter(self):
        result = await self.interface.execute(
            "codepartner_search_in_file", {"path": "src/app.py", "query": "TOKEN"},
            self.backend)
        self.assertIn("src/app.py:12", result)
        self.assertEqual(self.calls[0][1]["paths"], ["src/app.py"])

    async def test_patch_header_for_secret_is_blocked(self):
        result = await self.interface.execute(
            "apply_patch", {"input": "*** Begin Patch\n*** Update File: .env\n@@\n-x\n+y\n*** End Patch"},
            self.backend)
        self.assertIn("ACCESS DENIED", result)
        self.assertEqual(self.calls, [])

    async def test_dispatcher_terminal_rejects_dangerous_command(self):
        result = await self.interface.execute(
            "execute_tool", {"command": "execute_terminal_command --command 'rm -rf src'"},
            self.backend)
        self.assertIn("ACCESS DENIED", result)
        self.assertEqual(self.calls, [])

    async def test_dispatcher_cannot_invoke_unknown_tool(self):
        result = await self.interface.execute(
            "execute_tool", {"command": "secret_dump --all true"}, self.backend)
        self.assertIn("ACCESS DENIED", result)
        self.assertEqual(self.calls, [])

    async def test_read_only_mode_blocks_edit_before_dispatch(self):
        self.run.metadata["read_only"] = True
        result = await self.interface.execute(
            "apply_patch", {"input": "*** Begin Patch\n*** Update File: src/app.py\n@@\n-x\n+y\n*** End Patch"},
            self.backend)
        self.assertIn("ACCESS DENIED", result)
        self.assertEqual(self.calls, [])

    async def test_baseline_mode_keeps_guard_but_measures_legacy_read(self):
        baseline = AgentToolInterface(self.run, compact=False)
        async def backend(name, arguments):
            self.calls.append((name, arguments))
            return "L1: line 1\nL2: line 2"
        denied = await baseline.execute("read_file", {"file_path": ".env"}, backend)
        self.assertIn("ACCESS DENIED", denied)
        result = await baseline.execute("read_file", {"file_path": "src/app.py"}, backend)
        self.assertIn("L1: line 1", result)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(baseline.summary()["observation_mode"], "baseline")

    async def test_mutation_response_reports_validation(self):
        from mcp_bridge.edit_validation import ValidatedResult
        async def backend(_name, _arguments):
            return ValidatedResult("Done!", "Validation:\nsrc/app.py: Python syntax valid")
        result = await self.interface.execute(
            "apply_patch", {"input": "*** Begin Patch\n*** Update File: src/app.py\n@@\n-x\n+y\n*** End Patch"},
            backend)
        self.assertIn("EDIT APPLIED", result)
        self.assertIn("Python syntax valid", result)

    async def test_bridge_uses_same_interface_for_direct_handler(self):
        async def handler(arguments):
            return await self.backend("read_file", arguments)
        self.run.tool_interface = self.interface
        self.run.direct_tool_handlers = {"read_file": handler}
        result = await bridge.call_tool(
            self.run, "codepartner_read_file",
            {"path": "src/app.py", "start_line": 1, "end_line": 10})
        self.assertIn("lines 1–10 of 437", result)
        self.assertEqual(self.interface.metrics["backend_calls"], 1)

    async def test_fallback_edit_validates_complete_postimage(self):
        async def backend(name, _arguments):
            if name == "read_file":
                return "L1: def broken(:\nL2:     pass"
            return "Done!"
        result = await self.interface.execute(
            "apply_patch", {"input": "*** Begin Patch\n*** Update File: src/app.py\n@@\n-x\n+y\n*** End Patch"},
            backend)
        self.assertIn("EDIT APPLIED", result)
        self.assertIn("VALIDATION FAILED at line 1", result)


class ObservationParsingTests(unittest.TestCase):
    def test_missing_middle_line_is_an_error(self):
        result, _, _ = _read_result("L1: first\n…3 lines truncated…\nL5: last", "a.py", 1, 5)
        self.assertIn("READ ERROR", result)

    def test_syntax_failure_identifies_line(self):
        result = validate_postimages({"src/a.py": "def broken(:\n    pass\n"})
        self.assertIn("VALIDATION FAILED at line 1", result)

    def test_empty_command_has_explicit_unknown_exit_code(self):
        result = _command_result("")
        self.assertIn("COMMAND COMPLETE", result)
        self.assertIn("exit_code=unknown", result)

    def test_metrics_are_persisted_without_source_text(self):
        from telemetry.aggregation import TelemetryAggregation
        from telemetry.storage import TelemetryStorage
        with tempfile.TemporaryDirectory() as directory:
            storage = TelemetryStorage(Path(directory) / "metrics.sqlite3")
            storage.record_aci_attempt(
                "run-1", {"provider": "claude", "model": "sonnet", "task_id": "WP1-T1"},
                {"agent_calls": 2, "lines_returned": 100, "observation_mode": "compact"},
                {"input_tokens": 1200})
            with storage._database() as db:
                row = db.execute("SELECT metrics_json, usage_json FROM telemetry_aci_attempts").fetchone()
            self.assertEqual(json.loads(row[0])["lines_returned"], 100)
            self.assertEqual(json.loads(row[1])["input_tokens"], 1200)
            comparison = TelemetryAggregation(storage).aci_comparison(["WP1-T1"])
            self.assertEqual(comparison["WP1-T1"]["compact"]["input_tokens"], 1200)


if __name__ == "__main__":
    unittest.main()

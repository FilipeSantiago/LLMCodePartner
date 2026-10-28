"""Behavioral checks for the coder-facing interface at the IDE boundary."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.routing.persistence import RoutingPersistence
from agent.routing.rollback import journal_handlers
from mcp_bridge import bridge
from mcp_bridge.agent_tools import AgentToolInterface, _check_command, _command_result, _read_result
from mcp_bridge.edit_validation import validate_postimages
from mcp_bridge.models import Run
from mcp_bridge.registry import ToolSpec
from telemetry.storage import TelemetryStorage


def _spec(name, properties=None):
    return ToolSpec(name, name, {"type": "object", "properties": properties or {}})


class TelemetryMigrationTests(unittest.TestCase):
    def test_existing_aci_table_accepts_new_read_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.sqlite3"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE telemetry_aci_attempts ("
                           "run_id TEXT PRIMARY KEY, metrics_json TEXT NOT NULL)")
            storage = TelemetryStorage(path)
            storage.record_aci_attempt("review_1", {"task_id": "WP1-T1"},
                                       {"kb_outline_calls": 1}, {"input_tokens": 7})
            with sqlite3.connect(path) as db:
                row = db.execute("SELECT task_id, metrics_json, usage_json "
                                 "FROM telemetry_aci_attempts WHERE run_id='review_1'").fetchone()
            self.assertEqual(row[0], "WP1-T1")
            self.assertEqual(json.loads(row[1])["kb_outline_calls"], 1)
            self.assertEqual(json.loads(row[2])["input_tokens"], 7)


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
            remaining = list(range(offset, 438))
            if len(remaining) <= limit:
                return "\n".join(f"L{i}: line {i}" for i in remaining)
            half = limit // 2
            first, tail = remaining[:half], remaining[-half:]
            return ("\n".join(f"L{i}: line {i}" for i in first)
                    + f"\n…{len(remaining) - 2 * half} lines truncated…\n"
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

    async def test_duplicate_read_uses_cache_without_backend_request(self):
        arguments = {"path": "src/app.py", "start_line": 1, "end_line": 10}
        await self.interface.execute("codepartner_read_file", arguments, self.backend)
        result = await self.interface.execute("codepartner_read_file", arguments, self.backend)
        self.assertIn("READ COMPLETE", result)
        self.assertEqual(len(self.calls), 1)

    async def test_large_repeated_read_uses_retrievable_reference(self):
        args = {"path": "src/app.py", "full_file": True}
        first = await self.interface.execute("codepartner_read_file", args, self.backend)
        second = await self.interface.execute("codepartner_read_file", args, self.backend)
        self.assertIn("IDENTICAL READ RESULT", second)
        self.assertLess(len(second), len(first))
        key = next(iter(self.interface.read_observations.values()))
        original = await self.interface.execute("codepartner_get_observation",
                                                {"observation_id": key}, self.backend)
        self.assertIn("src/app.py — lines 1–437 of 437", original)
        self.assertIn("  437 | line 437", original)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.interface.metrics["read_reference_hits"], 1)
        await self.interface.execute("codepartner_read_file", {**args, "refresh": True},
                                     self.backend)
        self.assertIn("STALE", await self.interface.execute(
            "codepartner_get_observation", {"observation_id": key}, self.backend))
        self.assertEqual(len(self.calls), 2)

    async def test_read_metrics_count_delivered_ranges_and_repeats(self):
        self.run.tool_interface = self.interface
        self.run.direct_tool_handlers = {
            "read_file": lambda arguments: self.backend("read_file", arguments)}
        arguments = {"path": "src/app.py", "start_line": 140, "end_line": 145}
        await bridge.call_tool(self.run, "codepartner_read_file", arguments)
        await bridge.call_tool(self.run, "codepartner_read_file", arguments)
        summary = self.interface.summary()
        self.assertEqual(summary["ranged_source_reads"], 2)
        self.assertEqual(summary["whole_file_reads"], 0)
        self.assertEqual(summary["repeated_source_reads"], 1)
        self.assertGreater(summary["source_response_bytes"], 0)

    async def test_path_only_read_returns_bounded_preview(self):
        self.run.tool_interface = self.interface
        self.run.direct_tool_handlers = {"read_file": lambda arguments: self.backend("read_file", arguments)}
        result = await bridge.call_tool(
            self.run, "codepartner_read_file", {"path": "src/app.py"})
        self.assertIn("lines 1–200 of 437", result)
        self.assertIn("truncated=true", result)
        self.assertIn("Remaining: 201–437", result)
        self.assertNotIn("  437 | line 437", result)
        self.assertEqual(self.interface.metrics["lines_returned"], 200)
        summary = self.interface.summary()
        self.assertEqual(summary["bounded_preview_reads"], 1)
        self.assertGreater(summary["bounded_preview_bytes"], 0)
        self.assertEqual(summary["first_source_read_bytes"], summary["bounded_preview_bytes"])

    async def test_explicit_full_read_after_preview(self):
        self.run.tool_interface = self.interface
        self.run.direct_tool_handlers = {"read_file": lambda arguments: self.backend("read_file", arguments)}
        await bridge.call_tool(self.run, "codepartner_read_file", {"path": "src/app.py"})
        result = await bridge.call_tool(self.run, "codepartner_read_file",
                                        {"path": "src/app.py", "full_file": True})
        self.assertIn("lines 1–437 of 437", result)
        self.assertIn("  437 | line 437", result)
        summary = self.interface.summary()
        self.assertEqual(summary["explicit_full_file_reads"], 1)
        self.assertEqual(summary["preview_followup_reads"], 1)
        self.assertEqual(len(self.calls), 2)

    async def test_complete_short_path_only_read(self):
        async def backend(name, arguments):
            return "\n".join(f"L{i}: line {i}" for i in range(1, 40))
        self.run.tool_interface = self.interface
        self.run.direct_tool_handlers = {"read_file": lambda arguments: backend("read_file", arguments)}
        result = await bridge.call_tool(self.run, "codepartner_read_file", {"path": "short.py"})
        self.assertIn("lines 1–39 of 39", result)
        self.assertIn("truncated=false", result)
        self.assertEqual(self.interface.summary()["path_only_complete_reads"], 1)

    async def test_preview_does_not_invent_total_or_hide_retrieval(self):
        async def backend(name, arguments):
            self.calls.append((name, arguments))
            start, limit = arguments["offset"], arguments["limit"]
            return "\n".join(f"L{i}: line {i}" for i in range(start, start + limit))
        first = await self.interface.execute(
            "codepartner_read_file", {"path": "big.py"}, backend)
        self.assertIn("lines 1–200 of unknown", first)
        self.assertIn("Remaining: 201 onward", first)
        self.assertNotIn("  201 | line 201", first)
        self.assertIn("Original IDE output for this read", first)
        observation = self.interface._reload({"observation_id": "obs-1"})
        self.assertIn("L201: line 201", observation)
        self.assertNotIn("L403: line 403", observation)
        self.assertEqual(self.interface.summary()["preview_observation_expansions"], 1)
        self.assertGreater(self.interface.summary()["preview_observation_bytes"], 0)
        next_lines = await self.interface.execute(
            "codepartner_read_file", {"path": "big.py", "start_line": 201,
                                      "end_line": 220}, backend)
        self.assertIn("lines 201–220 of unknown", next_lines)
        self.assertIn("  220 | line 220", next_lines)
        self.assertEqual(len(self.calls), 2)

    async def test_preview_refresh_marks_prior_observation_stale(self):
        args = {"path": "src/app.py"}
        await self.interface.execute("codepartner_read_file", args, self.backend)
        key = "obs-1"
        await self.interface.execute("codepartner_read_file", {**args, "refresh": True},
                                     self.backend)
        self.assertIn("STALE", self.interface._reload({"observation_id": key}))
        self.assertEqual(len(self.calls), 2)

    async def test_noncompact_read_file_keeps_raw_behavior(self):
        self.interface.compact = False
        result = await self.interface.execute(
            "read_file", {"file_path": "src/app.py", "offset": 1, "limit": 500},
            self.backend)
        self.assertIn("L437: line 437", result)
        self.assertNotIn("READ COMPLETE", result)

    async def test_full_file_cannot_be_combined_with_range(self):
        result = await self.interface.execute(
            "codepartner_read_file", {"path": "src/app.py", "full_file": True,
                                      "start_line": 2}, self.backend)
        self.assertIn("READ ERROR", result)
        self.assertEqual(self.calls, [])

    async def test_refresh_forces_new_read(self):
        args = {"path": "src/app.py"}
        await self.interface.execute("codepartner_read_file", args, self.backend)
        await self.interface.execute("codepartner_read_file", {**args, "refresh": True}, self.backend)
        self.assertEqual(len(self.calls), 2)

    async def test_full_read_cache_serves_range(self):
        await self.interface.execute("codepartner_read_file", {
            "path": "src/app.py", "full_file": True}, self.backend)
        result = await self.interface.execute("codepartner_read_file", {
            "path": "src/app.py", "start_line": 200, "end_line": 201}, self.backend)
        self.assertIn("lines 200–201 of 437", result)
        self.assertEqual(len(self.calls), 1)

    async def test_large_file_is_assembled_without_repeating_ide_request(self):
        async def backend(name, arguments):
            self.calls.append((name, arguments))
            offset, limit = arguments["offset"], arguments["limit"]
            return "\n".join(f"L{i}: line {i}" for i in range(offset, min(offset + limit, 6201)))
        result = await self.interface.execute("codepartner_read_file", {
            "path": "big.txt", "full_file": True}, backend)
        self.assertIn("lines 1–6200 of 6200", result)
        self.assertIn(" 6200 | line 6200", result)
        self.assertEqual([call[1]["offset"] for call in self.calls], [1, 5001])

    async def test_head_tail_ide_pages_are_filled_without_repeating_requests(self):
        async def backend(name, arguments):
            self.calls.append((name, arguments))
            offset, limit = arguments["offset"], arguments["limit"]
            remaining = list(range(offset, 8001))
            if len(remaining) <= limit:
                return "\n".join(f"L{i}: line {i}" for i in remaining)
            half = limit // 2
            return ("\n".join(f"L{i}: line {i}" for i in remaining[:half])
                    + f"\n…{len(remaining) - 2 * half} lines truncated…\n"
                    + "\n".join(f"L{i}: line {i}" for i in remaining[-half:]))
        result = await self.interface.execute("codepartner_read_file", {
            "path": "big.txt", "full_file": True}, backend)
        self.assertIn("lines 1–8000 of 8000", result)
        self.assertIn(" 8000 | line 8000", result)
        self.assertEqual([call[1]["offset"] for call in self.calls], [1, 2501, 5001])

    async def test_denied_second_page_is_not_reported_complete(self):
        async def backend(name, arguments):
            if arguments["offset"] > 1:
                return "Error: request denied"
            return "\n".join(f"L{i}: line {i}" for i in range(1, 5001))
        result = await self.interface.execute("codepartner_read_file", {
            "path": "big.txt", "full_file": True}, backend)
        self.assertIn("READ FAILED", result)
        self.assertNotIn("READ COMPLETE", result)

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

    async def test_search_has_no_implicit_match_limit(self):
        async def backend(name, arguments):
            self.calls.append((name, arguments))
            return json.dumps({"items": [
                {"filePath": "src/app.py", "startLine": line} for line in range(1, 41)
            ], "more": False})
        result = await self.interface.execute(
            "codepartner_search_in_file", {"path": "src/app.py", "query": "item"}, backend)
        self.assertIn("40 matches returned", result)
        self.assertIn("src/app.py:40", result)
        self.assertNotIn("limit", self.calls[0][1])

    def test_observations_have_no_count_or_byte_retention_limit(self):
        payload = "x" * 2_100_000
        first = self.interface._remember("summary", payload, category="other")
        for number in range(129):
            self.interface._remember("summary", f"raw-{number}", category="other")
        self.assertIn("obs-1", first)
        expanded = self.interface._reload({"observation_id": "obs-1"})
        self.assertIn(payload, expanded)
        self.assertEqual(len(self.interface.observations), 130)

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

    async def test_large_existing_file_rewrite_requires_identical_repeat(self):
        self.run.registry.register([_spec("create_new_file", {
            "pathInProject": {}, "text": {}, "overwrite": {}})])
        args = {"pathInProject": "src/large.py", "text": "replacement", "overwrite": True}
        async def backend(name, arguments):
            self.calls.append((name, arguments))
            if name == "read_file":
                return "\n".join(f"L{i}: line {i}" for i in range(1, 62))
            return "Done!"
        first = await self.interface.execute("create_new_file", args, backend)
        self.assertIn("EDIT NOT APPLIED", first)
        self.assertIn("61 lines", first)
        self.assertNotIn("create_new_file", [name for name, _ in self.calls])
        second = await self.interface.execute("create_new_file", args, backend)
        self.assertIn("EDIT APPLIED", second)
        self.assertIn("create_new_file", [name for name, _ in self.calls])
        self.assertEqual(self.interface.metrics["rewrite_blocked"], 1)
        self.assertEqual(self.interface.metrics["rewrite_overridden"], 1)

    async def test_small_and_new_files_can_be_created(self):
        self.run.registry.register([_spec("create_new_file_with_text", {
            "pathInProject": {}, "text": {}, "overwrite": {}})])
        async def backend(name, arguments):
            self.calls.append((name, arguments))
            if name == "read_file":
                if arguments["file_path"] == "src/new.py":
                    return "File src/new.py doesn't exist or can't be opened"
                return "\n".join(f"L{i}: line {i}" for i in range(1, 61))
            return "Done!"
        for path in ("src/small.py", "src/new.py"):
            result = await self.interface.execute("create_new_file_with_text", {
                "pathInProject": path, "text": "replacement", "overwrite": True}, backend)
            self.assertIn("EDIT APPLIED", result)
        self.assertEqual(self.interface.metrics["rewrite_blocked"], 0)

    async def test_override_still_records_journaled_write(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            specs = [
                _spec("read_file", {"file_path": {}, "offset": {}, "limit": {}}),
                _spec("create_new_file", {"pathInProject": {}, "text": {}, "overwrite": {}}),
            ]
            run = Run()
            run.registry.register(specs)
            content = "\n".join(f"L{i}: # line {i}" for i in range(1, 62))
            async def read(_arguments):
                return content
            async def create(_arguments):
                return "Done!"
            run.direct_tool_handlers = journal_handlers(
                persistence, "job", 0, specs,
                {"read_file": read, "create_new_file": create})
            run.tool_interface = AgentToolInterface(run)
            args = {"pathInProject": "src/large.py", "text": "x = 1", "overwrite": True}
            self.assertIn("EDIT NOT APPLIED", await bridge.call_tool(
                run, "create_new_file", args))
            self.assertIn("EDIT APPLIED", await bridge.call_tool(
                run, "create_new_file", args))
            mutations = await persistence.mutations("job", 0)
            self.assertEqual(len(mutations), 1)
            self.assertEqual(mutations[0]["operation"], "create")

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


class KnowledgeBasePathTests(unittest.IsolatedAsyncioTestCase):
    """KB results cite the host checkout; a native agent must get workspace paths."""

    async def test_kb_paths_are_translated_both_ways(self):
        from agent.workspace import NativeRun
        run = Run()
        run.registry.register([_spec("kb_get_file_outline", {"file_path": {}}),
                               _spec("get_file_problems", {"filePath": {}})])
        run.native = NativeRun(workspace="/workspace/demo", host_path="/home/u/Demo")
        seen = {}

        def handler(name):
            async def call(arguments):
                seen[name] = arguments
                return f"{arguments}: defined in /home/u/Demo/pkg/a.py:3"
            return call

        run.direct_tool_handlers = {name: handler(name)
                                    for name in ("kb_get_file_outline", "get_file_problems")}
        result = await bridge.call_tool(run, "kb_get_file_outline",
                                        {"file_path": "/workspace/demo/pkg/a.py"})
        self.assertEqual(seen["kb_get_file_outline"], {"file_path": "/home/u/Demo/pkg/a.py"})
        self.assertIn("defined in /workspace/demo/pkg/a.py:3", result)
        self.assertNotIn("/home/u/Demo", result)
        # Other tools (the IDE's own MCP) work on host paths and stay untouched.
        other = await bridge.call_tool(run, "get_file_problems",
                                       {"filePath": "/home/u/Demo/pkg/a.py"})
        self.assertIn("defined in /home/u/Demo/pkg/a.py:3", other)


if __name__ == "__main__":
    unittest.main()

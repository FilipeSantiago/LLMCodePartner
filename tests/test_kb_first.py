"""Authorization and intent regressions for the shared KB-first policy."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.kb_first import KbFirstPolicy
from agent.routing.persistence import RoutingPersistence
from mcp_bridge import bridge
from mcp_bridge.agent_tools import AgentToolInterface
from mcp_bridge.models import Run
from mcp_bridge.registry import ToolSpec


class _Registry:
    def __init__(self, names):
        self._names = names

    def names(self):
        return tuple(self._names)


class KbFirstTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.run = SimpleNamespace(registry=_Registry({
            "kb_search_graph", "search_text", "kb_get_file_outline", "read_file"}))
        self.policy = KbFirstPolicy(role="planner")

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        if name == "kb_search_graph":
            return json.dumps({"groups": [{"file": "src/loader.py", "rows": [["Loader"]]}]})
        if name == "kb_get_file_outline":
            return json.dumps({"cols": ["symbol", "label", "lines"],
                               "rows": [["Loader", "Class", "1-200"]],
                               "total": 1})
        return "src/loader.py:10: Loader"

    async def test_graph_gate_is_explicit_and_repeat_bypasses(self):
        args = {"q": "Loader"}
        first = await self.policy.call("search_text", args, self.dispatch, run=self.run)
        self.assertIn("TEXT SEARCH NOT EXECUTED", first)
        self.assertEqual([name for name, _ in self.calls], ["kb_search_graph"])
        second = await self.policy.call("search_text", args, self.dispatch, run=self.run)
        self.assertEqual(second, "src/loader.py:10: Loader")
        self.assertEqual(self.calls[-1][0], "search_text")
        self.assertEqual(self.policy.summary()["search_gate"]["kb_insufficient"], 1)

    async def test_usage_and_literal_searches_are_not_intercepted(self):
        for query in ("Loader(", "import Loader", '"Loader"', "Loader config"):
            await self.policy.call("search_text", {"q": query}, self.dispatch, run=self.run)
        self.assertTrue(all(name == "search_text" for name, _ in self.calls))

    async def test_no_graph_match_falls_back_in_same_call(self):
        async def no_match(name, args):
            self.calls.append((name, args))
            return '{"groups":[]}' if name == "kb_search_graph" else "actual search"
        result = await self.policy.call("search_text", {"q": "Loader"}, no_match, run=self.run)
        self.assertEqual(result, "actual search")
        self.assertEqual([name for name, _ in self.calls], ["kb_search_graph", "search_text"])

    async def test_outline_is_a_hint_and_repeat_reads(self):
        first = await self.policy.call("read_file", {"file_path": "src/loader.py"},
                                       self.dispatch, run=self.run)
        self.assertIn("Read not performed", first)
        self.assertIn("last index", first)
        self.assertEqual(self.calls[0][0], "kb_get_file_outline")
        await self.policy.call("read_file", {"file_path": "src/loader.py"},
                               self.dispatch, run=self.run)
        self.assertEqual(self.calls[-1][0], "read_file")

    async def test_enforced_read_uses_one_kb_lookup_then_bounded_ranges(self):
        policy = KbFirstPolicy(role="planner", enforce_reads=True)
        first = await policy.call("read_file", {"file_path": "src/loader.py"},
                                  self.dispatch, run=self.run)
        self.assertIn("READ NOT PERFORMED", first)
        self.assertEqual([name for name, _ in self.calls], ["kb_get_file_outline"])
        blocked = await policy.call("read_file", {"file_path": "src/loader.py"},
                                    self.dispatch, run=self.run)
        self.assertIn("Specify an explicit start and end line", blocked)
        self.assertEqual(len(self.calls), 1)
        blocked = await policy.call("read_file", {"file_path": "src/loader.py",
                                                  "start_line": 1, "end_line": 121},
                                    self.dispatch, run=self.run)
        self.assertIn("smaller relevant range", blocked)
        result = await policy.call("read_file", {"file_path": "src/loader.py",
                                                 "start_line": 10, "end_line": 20},
                                   self.dispatch, run=self.run)
        self.assertIn("Loader", result)
        self.assertEqual([name for name, _ in self.calls],
                         ["kb_get_file_outline", "read_file"])
        self.assertEqual(policy.summary()["read_enforcement"]["kb_outlines"], 1)

    async def test_enforced_fallback_requires_file_search(self):
        policy = KbFirstPolicy(role="coder", enforce_reads=True)
        async def no_kb(name, args):
            self.calls.append((name, args))
            return '{"rows":[]}' if name == "kb_get_file_outline" else "source"
        args = {"file_path": "docs/notes.md", "start_line": 1, "end_line": 5}
        self.assertIn("Search within this file", await policy.call(
            "read_file", args, no_kb, run=self.run))
        self.assertIn("Search within docs/notes.md first", await policy.call(
            "read_file", args, no_kb, run=self.run))
        await policy.call("codepartner_search_in_file", {"path": "docs/notes.md", "query": "summary"},
                          no_kb, run=self.run)
        self.assertEqual(await policy.call("read_file", args, no_kb, run=self.run),
                         "source")
        self.assertEqual([name for name, _ in self.calls].count("kb_get_file_outline"), 1)

    async def test_failed_or_unscoped_search_does_not_unlock_fallback(self):
        policy = KbFirstPolicy(role="coder", enforce_reads=True)
        async def dispatch(name, args):
            return '{"rows":[]}' if name == "kb_get_file_outline" else "SEARCH UNAVAILABLE"
        args = {"file_path": "README.md", "start_line": 1, "end_line": 5}
        await policy.call("read_file", args, dispatch, run=self.run)
        await policy.call("codepartner_search_in_file", {"path": "README.md", "query": "usage"},
                          dispatch, run=self.run)
        await policy.call("search_text", {"q": "usage"}, dispatch, run=self.run)
        self.assertIn("Search within README.md first", await policy.call(
            "read_file", args, dispatch, run=self.run))

    async def test_unbounded_transport_is_refused(self):
        policy = KbFirstPolicy(role="planner", enforce_reads=True)
        result = await policy.call("get_file_text_by_path", {
            "path": "src/loader.py", "offset": 1, "limit": 5}, self.dispatch, run=self.run)
        self.assertIn("cannot guarantee a bounded response", result)
        self.assertEqual(self.calls, [])

    async def test_edited_file_is_reread_without_a_stale_outline(self):
        policy = KbFirstPolicy(role="coder", gates=False, enforce_reads=True)
        await policy.call("read_file", {"file_path": "src/loader.py"},
                          self.dispatch, run=self.run)
        await policy.call("replace_specific_text", {
            "path": "src/loader.py", "old_text": "before", "new_text": "after"},
            self.dispatch, run=self.run)
        result = await policy.call("read_file", {"file_path": "src/loader.py"},
                                   self.dispatch, run=self.run)
        self.assertEqual(result, "src/loader.py:10: Loader")
        self.assertEqual([name for name, _ in self.calls].count("kb_get_file_outline"), 1)

    async def test_plan_file_reads_current_source_once_then_ranges(self):
        policy = KbFirstPolicy(role="coder", enforce_reads=True,
                               direct_paths=["src/loader.py"])
        first = await policy.call("read_file", {"file_path": "src/loader.py"},
                                  self.dispatch, run=self.run)
        self.assertEqual(first, "src/loader.py:10: Loader")
        again = await policy.call("read_file", {"file_path": "src/loader.py"},
                                  self.dispatch, run=self.run)
        self.assertIn("Specify an explicit start and end line", again)
        self.assertEqual([name for name, _ in self.calls], ["read_file"])
        self.assertEqual(policy.summary()["read_enforcement"]["direct_plan"], 1)

    async def test_planner_still_outlines_plan_files(self):
        policy = KbFirstPolicy(role="planner", enforce_reads=True,
                               direct_paths=["src/loader.py"])
        result = await policy.call("read_file", {"file_path": "src/loader.py"},
                                   self.dispatch, run=self.run)
        self.assertIn("READ NOT PERFORMED", result)
        self.assertEqual(self.calls[0][0], "kb_get_file_outline")

    async def test_file_changed_since_index_reads_directly(self):
        self.run.registry = _Registry({"kb_get_file_outline", "kb_index_status", "read_file"})
        async def recent(since):
            self.assertEqual(since, 100.0)
            return [{"targets": ["src/loader.py"]}]
        policy = KbFirstPolicy(role="coder", enforce_reads=True, recent_mutations=recent)
        async def dispatch(name, args):
            self.calls.append((name, args))
            if name == "kb_index_status":
                return json.dumps({"root_path": "/repo", "indexed_at": 100.0})
            return await self.dispatch(name, args)
        result = await policy.call("read_file", {"file_path": "src/loader.py"},
                                   dispatch, run=self.run)
        self.assertEqual(result, "src/loader.py:10: Loader")
        self.assertNotIn("kb_get_file_outline", [name for name, _ in self.calls])
        self.assertEqual(policy.summary()["read_enforcement"]["direct_changed"], 1)

    async def test_bridge_parses_generated_outline_behind_interface(self):
        run = Run()
        run.registry.register([
            ToolSpec("kb_get_file_outline", "outline", {"type": "object"}),
            ToolSpec("read_file", "read", {"type": "object", "properties": {
                "file_path": {}, "offset": {}, "limit": {}}}),
        ])
        calls = []
        async def outline(_arguments):
            calls.append("outline")
            return json.dumps({"cols": ["symbol", "label", "lines"],
                               "rows": [["Loader", "Class", "1-200"]], "total": 1})
        async def read(_arguments):
            calls.append("read")
            return "L10: class Loader:\nL11:     pass"
        run.direct_tool_handlers = {"kb_get_file_outline": outline, "read_file": read}
        run.tool_interface = AgentToolInterface(run)
        run.kb_first = KbFirstPolicy(role="coder", enforce_reads=True)
        first = await bridge.call_tool(run, "codepartner_read_file", {"path": "src/loader.py"})
        self.assertIn("READ NOT PERFORMED. KB outline", first)
        self.assertEqual(calls, ["outline"])
        result = await bridge.call_tool(run, "codepartner_read_file", {
            "path": "src/loader.py", "start_line": 10, "end_line": 11})
        self.assertIn("READ COMPLETE", result)
        self.assertEqual(calls, ["outline", "read"])
        self.assertEqual(run.kb_first.read_enforcement["kb_outlines"], 1)
        self.assertEqual(run.kb_first.fallback_reasons["kb_error"], 0)
        model_outline = await bridge.call_tool(run, "kb_get_file_outline", {
            "file_path": "src/loader.py", "format": "json"})
        self.assertIn("TOOL COMPLETE\ntool=kb_get_file_outline", model_outline)

    async def test_excluded_read_skips_outline_and_uses_interface_denial(self):
        run = Run()
        run.registry.register([
            ToolSpec("kb_get_file_outline", "outline", {"type": "object"}),
            ToolSpec("read_file", "read", {"type": "object", "properties": {
                "file_path": {}, "offset": {}, "limit": {}}}),
        ])
        calls = []
        async def outline(_arguments):
            calls.append("outline")
            return '{"rows":[]}'
        async def read(_arguments):
            calls.append("read")
            return "L1: secret"
        run.direct_tool_handlers = {"kb_get_file_outline": outline, "read_file": read}
        run.metadata["excluded_files"] = ("private/secret.py",)
        run.tool_interface = AgentToolInterface(run)
        run.kb_first = KbFirstPolicy(role="coder", enforce_reads=True,
                                     excluded_files=("private/secret.py",))
        result = await bridge.call_tool(run, "codepartner_read_file", {
            "path": "private/secret.py"})
        self.assertIn("ACCESS DENIED", result)
        self.assertEqual(calls, [])
        self.assertEqual(run.kb_first.read_enforcement["kb_outlines"], 0)
        self.assertEqual(run.kb_first.read_enforcement["fallbacks"], 0)

    async def test_bridge_uses_index_time_for_recent_coder_changes(self):
        run = Run()
        run.registry.register([
            ToolSpec("kb_index_status", "status", {"type": "object"}),
            ToolSpec("kb_get_file_outline", "outline", {"type": "object"}),
            ToolSpec("read_file", "read", {"type": "object", "properties": {
                "file_path": {}, "offset": {}, "limit": {}}}),
        ])
        calls = []
        async def status(_arguments):
            calls.append("status")
            return json.dumps({"indexed_at": 100.0})
        async def outline(_arguments):
            calls.append("outline")
            return '{"rows":[]}'
        async def read(_arguments):
            calls.append("read")
            return "L1: current source"
        async def recent(since):
            self.assertEqual(since, 100.0)
            return [{"targets": ["src/changed.py"]}]
        run.direct_tool_handlers = {"kb_index_status": status,
                                    "kb_get_file_outline": outline, "read_file": read}
        run.tool_interface = AgentToolInterface(run)
        run.kb_first = KbFirstPolicy(role="coder", enforce_reads=True,
                                     recent_mutations=recent)
        result = await bridge.call_tool(run, "codepartner_read_file", {
            "path": "src/changed.py"})
        self.assertIn("READ COMPLETE", result)
        self.assertEqual(calls, ["status", "read"])
        self.assertEqual(run.kb_first.read_enforcement["direct_changed"], 1)

    async def test_bridge_snippet_uses_raw_kb_location(self):
        from mcp_bridge.agent_tools import tool_specs
        run = Run()
        run.registry.register([
            ToolSpec("kb_get_code_snippet", "snippet", {"type": "object"}),
            ToolSpec("kb_index_status", "status", {"type": "object"}),
            ToolSpec("read_file", "read", {"type": "object", "properties": {
                "file_path": {}, "offset": {}, "limit": {}}}),
            *tool_specs()[:1],
        ])
        async def snippet(_arguments):
            return json.dumps({"path": "/repo/pkg/loader.py",
                               "qualified_name": "repo.pkg.loader.Loader.load",
                               "start_line": 1, "end_line": 3, "label": "Method"})
        async def status(_arguments):
            return json.dumps({"root_path": "/repo"})
        async def read(_arguments):
            return "L1: class Loader:\nL2:     def load(self):\nL3:         return 1"
        run.direct_tool_handlers = {"kb_get_code_snippet": snippet,
                                    "kb_index_status": status, "read_file": read}
        run.tool_interface = AgentToolInterface(run)
        run.kb_first = KbFirstPolicy(role="coder")
        result = await bridge.call_tool(run, "kb_get_code_snippet", {
            "symbol": "Loader.load"})
        self.assertIn("Verified current IDE source", result)
        self.assertEqual(run.kb_first.snippet_outcomes["verified"], 1)

    async def test_empty_outline_of_code_file_reads_source_with_stale_note(self):
        policy = KbFirstPolicy(role="coder", enforce_reads=True)
        async def no_rows(name, args):
            self.calls.append((name, args))
            return '{"rows":[]}' if name == "kb_get_file_outline" else "source"
        result = await policy.call("read_file", {"file_path": "trainer/contracts.py"},
                                   no_rows, run=self.run)
        self.assertIn("index may be stale", result)
        self.assertTrue(result.endswith("source"))
        self.assertEqual([name for name, _ in self.calls],
                         ["kb_get_file_outline", "read_file"])
        self.assertEqual(policy.summary()["read_enforcement"]["direct_stale_index"], 1)

    async def test_terminal_command_does_not_reoutline_located_files(self):
        self.run.registry = _Registry({"kb_get_file_outline", "read_file",
                                       "execute_terminal_command"})
        policy = KbFirstPolicy(role="coder", gates=False, enforce_reads=True)
        await policy.call("read_file", {"file_path": "src/loader.py"},
                          self.dispatch, run=self.run)
        await policy.call("execute_terminal_command", {"command": "pytest -q"},
                          self.dispatch, run=self.run)
        await policy.call("read_file", {"file_path": "src/loader.py"},
                          self.dispatch, run=self.run)
        self.assertEqual([name for name, _ in self.calls].count("kb_get_file_outline"), 1)

    def test_dispatcher_blocks_unmediated_source_and_shell(self):
        from mcp_bridge.bridge import _controlled_tool_refusal
        self.assertIn("unrestricted file content", _controlled_tool_refusal(
            "execute_tool", {"command": "get_git_diff_all"}))
        self.assertIn("unrestricted file content", _controlled_tool_refusal(
            "read_notebook", {"file_path": "analysis.ipynb"}))
        self.assertIn("arbitrary terminal commands", _controlled_tool_refusal(
            "execute_tool", {"command": "execute_terminal_command --command 'cat src/app.py'"}))
        self.assertIsNone(_controlled_tool_refusal(
            "execute_tool", {"command": "execute_terminal_command --command 'pytest -q'"}))

    async def test_verify_path_skips_outline(self):
        self.policy.verify_paths.add("src/loader.py")
        await self.policy.call("read_file", {"file_path": "src/loader.py"},
                               self.dispatch, run=self.run)
        self.assertEqual(self.calls[0][0], "read_file")

    async def test_unadvertised_kb_never_generated(self):
        self.run.registry = _Registry({"search_text", "read_file"})
        await self.policy.call("search_text", {"q": "Loader"},
                               self.dispatch, run=self.run)
        self.assertEqual([name for name, _ in self.calls], ["search_text"])

    async def test_snippet_reads_current_source_and_checks_scope(self):
        self.run.registry = _Registry({"kb_get_code_snippet", "kb_index_status", "read_file"})
        source = "class Loader:\n    def load(self):\n        return 1"
        async def dispatch(name, args):
            self.calls.append((name, args))
            if name == "kb_get_code_snippet":
                return json.dumps({"path": "/repo/pkg/loader.py",
                                   "qualified_name": "repo.pkg.loader.Loader.load",
                                   "start_line": 1, "end_line": 3, "label": "Method"})
            if name == "kb_index_status":
                return json.dumps({"root_path": "/repo"})
            return source
        result = await self.policy.call(
            "kb_get_code_snippet", {"symbol": "Loader.load"}, dispatch, run=self.run)
        self.assertIn("Verified current IDE source", result)
        self.assertEqual(self.calls[-1][0], "read_file")
        self.assertEqual(self.policy.summary()["snippet"]["verified"], 1)

    async def test_snippet_path_outside_root_is_not_read(self):
        self.run.registry = _Registry({"kb_get_code_snippet", "kb_index_status", "read_file"})
        async def dispatch(name, args):
            self.calls.append((name, args))
            if name == "kb_get_code_snippet":
                return json.dumps({"path": "/private/secret.py",
                                   "qualified_name": "secret.Secret",
                                   "start_line": 1, "end_line": 2})
            return json.dumps({"root_path": "/repo"})
        result = await self.policy.call(
            "kb_get_code_snippet", {"symbol": "Secret"}, dispatch, run=self.run)
        self.assertIn("unresolved", result)
        self.assertNotIn("read_file", [name for name, _ in self.calls])

    async def test_stale_snippet_relocates_only_to_matching_class(self):
        self.run.registry = _Registry({"kb_get_code_snippet", "kb_index_status",
                                       "read_file", "search_in_files_by_text"})
        async def dispatch(name, args):
            self.calls.append((name, args))
            if name == "kb_get_code_snippet":
                return json.dumps({"path": "/repo/pkg/loader.py",
                                   "qualified_name": "repo.pkg.loader.Loader.load",
                                   "start_line": 1, "end_line": 3, "label": "Method"})
            if name == "kb_index_status":
                return '{"root_path":"/repo"}'
            if name == "search_in_files_by_text":
                return "pkg/loader.py:50: def load"
            if args.get("offset") == 1:
                return "class Wrong:\n    def load(self):\n        pass"
            return "x\n" * 39 + "class Loader:\n    def load(self):\n        return 1"
        result = await self.policy.call(
            "kb_get_code_snippet", {"symbol": "Loader.load"}, dispatch, run=self.run)
        self.assertIn("Verified relocated", result)
        self.assertEqual(self.policy.summary()["snippet"]["relocated"], 1)

    async def test_generated_call_uses_same_registry_authorization(self):
        class RefusingRegistry:
            def names(self):
                return ("kb_search_graph", "search_text")
            def prepare_advertised_call(self, name, args):
                if name == "kb_search_graph":
                    raise LookupError("refused by request registry")
                return name, args
        called = []
        async def search(args):
            called.append(args)
            return "direct search"
        run = Run(registry=RefusingRegistry(),
                  direct_tool_handlers={"search_text": search},
                  kb_first=KbFirstPolicy(role="planner"))
        result = await bridge.call_tool(run, "search_text", {"q": "Loader"})
        self.assertEqual(result, "direct search")
        self.assertEqual(called, [{"q": "Loader"}])
        self.assertEqual(run.kb_first.summary()["search_gate"]["authorization_refused"], 1)

    async def test_kb_assisted_edit_requires_current_region_and_allows_next_edit(self):
        self.run.registry = _Registry({"kb_search_graph", "read_file",
                                       "replace_specific_text"})
        source = "class Loader:\n    value = 'old'"
        async def dispatch(name, args):
            nonlocal source
            self.calls.append((name, args))
            if name == "kb_search_graph":
                return '{"groups":[{"file":"src/loader.py","rows":[["Loader"]]}]}'
            if name == "read_file":
                return source
            source = source.replace(args["old_text"], args["new_text"])
            return "Edit applied"
        await self.policy.call("kb_search_graph", {"name_pattern": "Loader"},
                               dispatch, run=self.run)
        edit = {"path": "src/loader.py", "old_text": "'old'", "new_text": "'new'"}
        blocked = await self.policy.call("replace_specific_text", edit, dispatch,
                                         run=self.run)
        self.assertIn("Read the text you are replacing", blocked)
        self.assertEqual(len(self.calls), 1)
        await self.policy.call("read_file", {"file_path": "src/loader.py",
                                            "offset": 1, "limit": 2},
                               dispatch, run=self.run)
        self.assertEqual(await self.policy.call("replace_specific_text", edit,
                                                 dispatch, run=self.run), "Edit applied")
        second = {"path": "src/loader.py", "old_text": "'new'", "new_text": "'newer'"}
        self.assertEqual(await self.policy.call("replace_specific_text", second,
                                                 dispatch, run=self.run), "Edit applied")
        self.assertEqual(self.policy.summary()["edit_verification_blocks"], 1)

    async def test_search_gate_cap_and_doc_intent(self):
        self.policy.search_gate_max = 1
        await self.policy.call("search_text", {"q": "Loader"}, self.dispatch, run=self.run)
        await self.policy.call("search_text", {"q": "Other"}, self.dispatch, run=self.run)
        await self.policy.call("search_text", {"q": "Doc", "paths": ["docs/readme.md"]},
                               self.dispatch, run=self.run)
        self.assertEqual([name for name, _ in self.calls],
                         ["kb_search_graph", "search_text", "search_text"])

    async def test_excluded_path_is_filtered_from_kb_results(self):
        self.policy.excluded_files.add("private/secret.py")
        async def dispatch(name, args):
            return json.dumps({"groups": [
                {"file": "private/secret.py", "rows": [["Secret"]]},
                {"file": "src/public.py", "rows": [["Public"]]}]})
        result = await self.policy.call(
            "kb_search_graph", {"name_pattern": "Secret|Public"}, dispatch, run=self.run)
        self.assertNotIn("private/secret.py", result)
        self.assertIn("src/public.py", result)

    async def test_recent_mutations_are_project_scoped_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            with persistence._database() as db:
                db.execute("""INSERT INTO implementation_jobs
                    (job_id, selectors_json, status, created_at, updated_at, project_id)
                    VALUES ('j1', '[]', 'completed', 1, 1, 'demo')""")
                db.execute("""INSERT INTO implementation_mutations
                    (job_id, position, tool_name, operation, targets_json, stage,
                     created_at, updated_at)
                    VALUES ('j1', 1, 'apply_patch', 'patch', '["src/a.py"]',
                            'postimage_verified', 2, 3)""")
            self.assertEqual(
                [row["targets"] for row in await persistence.recent_mutations("demo", 2)],
                [["src/a.py"]])
            self.assertEqual(await persistence.recent_mutations("other", 2), [])

    async def test_second_edit_after_verified_patch_uses_authored_new_text(self):
        self.run.registry = _Registry({"apply_patch", "replace_specific_text",
                                       "read_file"})
        self.policy.hinted_paths.add("src/loader.py")
        self.policy._record_read({"file_path": "src/loader.py", "offset": 1,
                                  "limit": 2},
                                 "class Loader:\n    value = 'old'")
        patch = ("*** Begin Patch\n*** Update File: src/loader.py\n"
                 "@@ -1,2 +1,2 @@\n class Loader:\n"
                 "-    value = 'old'\n+    value = 'new'\n*** End Patch")
        async def dispatch(name, args):
            self.calls.append((name, args))
            return "Edit applied"
        self.assertEqual(await self.policy.call(
            "apply_patch", {"input": patch}, dispatch, run=self.run), "Edit applied")
        self.assertEqual(await self.policy.call(
            "replace_specific_text", {"path": "src/loader.py",
                                      "old_text": "'new'", "new_text": "'newer'"},
            dispatch, run=self.run), "Edit applied")

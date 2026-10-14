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
from mcp_bridge.models import Run


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

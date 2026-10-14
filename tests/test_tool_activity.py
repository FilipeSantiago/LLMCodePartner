"""One readable progress line per IDE tool call."""

import unittest

from mcp_bridge.activity import describe_call
from mcp_bridge.read_metrics import SourceReadStats


class DescribeCallTests(unittest.TestCase):
    def test_reads_name_the_file_and_line_range(self):
        self.assertEqual(describe_call("read_file", {"file_path": "src/app.py"}),
                         "Reading `src/app.py`")
        self.assertEqual(describe_call("mcp__pycharm__read_file", {
            "file_path": "src/app.py", "offset": 10, "limit": 5}),
            "Reading `src/app.py` (lines 10–14)")

    def test_writes_and_edits_are_distinguished(self):
        self.assertEqual(describe_call("create_new_file", {"pathInProject": "a/new.py"}),
                         "Writing `a/new.py`")
        self.assertEqual(describe_call("replace_text_in_file", {"pathInProject": "a/old.py"}),
                         "Editing `a/old.py`")
        self.assertEqual(describe_call("delete_file", {"path": "a/gone.py"}),
                         "Deleting `a/gone.py`")

    def test_patches_list_each_file_by_kind(self):
        patch = ("*** Begin Patch\n*** Add File: a/new.py\n+x\n"
                 "*** Update File: a/old.py\n@@\n-x\n+y\n"
                 "*** Update File: a/other.py\n@@\n-x\n+y\n*** End Patch")
        self.assertEqual(describe_call("apply_patch", {"input": patch}),
                         "Writing `a/new.py`; Editing `a/old.py`, `a/other.py`")
        diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        self.assertEqual(describe_call("apply_patch", {"patch": diff}), "Editing `x.py`")
        self.assertEqual(describe_call("apply_patch", {}), "Editing files (patch)")

    def test_dispatched_writes_use_the_inner_tool(self):
        self.assertEqual(describe_call("execute_tool", {
            "command": "create_new_file --pathInProject a/new.py --text hi"}),
            "Writing `a/new.py`")

    def test_searches_listings_and_unknown_tools(self):
        self.assertEqual(describe_call("search_text", {"q": "Registry"}),
                         "Searching for `Registry`")
        self.assertEqual(describe_call("search_symbol", {"q": "Registry"}),
                         "Searching for `Registry`")
        self.assertEqual(describe_call("list_directory_tree", {"directoryPath": "src"}),
                         "Listing `src`")
        self.assertEqual(describe_call("get_file_problems", {"filePath": "a.py"}),
                         "`get_file_problems` on `a.py`")
        self.assertEqual(describe_call("create_new_file", {}), "Writing a file")

    def test_codebase_memory_labels_are_shared_across_roles(self):
        self.assertEqual(describe_call("kb_get_file_outline", {"file_path": "src/app.py"}),
                         "Reading indexed outline of `src/app.py`")
        self.assertEqual(describe_call("kb_search_graph", {}),
                         "Searching Codebase Memory graph")
        self.assertEqual(describe_call("kb_trace_path", {}),
                         "Tracing Codebase Memory dependencies")

    def test_dispatched_ranged_reads_are_measured(self):
        stats = SourceReadStats()
        args = {"command": "read_file --file_path src/app.py --offset 10 --limit 5"}
        stats.observe("execute_tool", args, "L10: x", read=False)
        stats.observe("execute_tool", args, "L10: x", read=False)
        self.assertEqual(stats.summary()["ranged_source_reads"], 2)
        self.assertEqual(stats.summary()["repeated_source_reads"], 1)


if __name__ == "__main__":
    unittest.main()

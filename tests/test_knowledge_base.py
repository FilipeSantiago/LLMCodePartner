"""KB project selection is resolved before a planner sees the graph tools."""

import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from agent.planning.knowledge_base import (CodebaseMemoryClient, KnowledgeBaseError,
                                           configured_execution_tools, planning_context,
                                           catch_up_index, refresh_accepted_wp)
from agent.routing.tool_metrics import read_only_handlers
from agent.spec.command import SpecResult
from agent.task_contracts.contractor import TaskContractor
from agent.spec.models import Task
from api import dialog_controller
from mcp_bridge.registry import ROLE_OPTIMIZER, ToolRegistry, allowed_for


class _Session:
    def __init__(self):
        self.calls = []

    async def list_tools(self):
        names = ("list_projects", "index_status", "get_architecture", "search_graph",
                 "trace_path", "get_file_outline", "get_code_snippet")
        return SimpleNamespace(tools=[SimpleNamespace(
            name=name, description=name,
            inputSchema={"type": "object", "properties": {
                "project": {"type": "string"}, "aspects": {"type": "array"}},
                "required": ["project"]}) for name in names])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "list_projects":
            payload = {"projects": [{"name": "other"}] if arguments["offset"] == 0
                       else [{"name": "target"}],
                       "has_more": arguments["offset"] == 0}
        elif name == "index_status":
            payload = {"project": arguments["project"], "status": "ready"}
        else:
            payload = {"project": arguments["project"]}
        return SimpleNamespace(isError=False, structuredContent=payload,
                               content=[SimpleNamespace(type="text", text=json.dumps(payload))])


class KnowledgeBaseTest(unittest.IsolatedAsyncioTestCase):
    async def test_execution_tools_are_project_bound_and_optional(self):
        fake = _Session()

        @asynccontextmanager
        async def session(_self):
            yield fake

        config = {"enabled": True, "provider": "codebase-memory",
                  "project_id": "target"}
        with patch.dict("os.environ", {"KB_ENABLED": "true"}), \
                patch.object(CodebaseMemoryClient, "session", session):
            specs, handlers = await configured_execution_tools(config)
            self.assertEqual(len(specs), 6)
            reviewer = ToolRegistry(allowed_for(ROLE_OPTIMIZER))
            reviewer.register(specs)
            self.assertIn("kb_search_graph", reviewer.names())
            coder_task = Task(
                id="WP1-T1", title="Implement", description="Build the feature",
                complexity="simple", reasoning="Small", context="",
                preferred_capability="write")
            contract = TaskContractor().initial_contract(coder_task, specs)
            self.assertIn("kb_search_graph", contract.allowed_tools)
            self.assertIn("kb_inspect", contract.allowed_operations)
            await handlers["kb_search_graph"]({"project": "other", "query": "Loader"})
        self.assertEqual(fake.calls[-1][1]["project"], "target")
        with patch.dict("os.environ", {"KB_ENABLED": "false"}):
            self.assertEqual(await configured_execution_tools(config), ([], {}))
        with patch.dict("os.environ", {"KB_ENABLED": "true"}), \
                patch.object(CodebaseMemoryClient, "planner_tools",
                             side_effect=KnowledgeBaseError("bridge unavailable")):
            self.assertEqual(await configured_execution_tools(config), ([], {}))
        with patch.dict("os.environ", {"KB_ENABLED": "true"}), \
                patch.object(CodebaseMemoryClient, "planner_tools",
                             side_effect=KnowledgeBaseError(
                                 "project absent from list_projects")):
            with self.assertRaisesRegex(KnowledgeBaseError, "absent from list_projects"):
                await configured_execution_tools(config)

    async def test_openspec_progress_precedes_generation(self):
        events = []
        captured = {}

        class Run:
            metadata = {}
            run_id = "test"

            async def put(self, event):
                events.append(event.text)

        run = Run()

        def start_run(_prompt, _system, _specs, strategy, **_kwargs):
            captured["strategy"] = strategy
            return run

        async def kb_context(config, stage, request, progress=None):
            self.assertEqual(stage, "openspec")
            self.assertEqual(config["project_id"], "target")
            await progress("Reading indexed architecture")
            return "Architecture context"

        class Handler:
            def __init__(self, _adapter, _store, context):
                self.assert_context = context

            async def handle(self, _messages):
                captured["context"] = self.assert_context
                return SpecResult("CHG1", "OpenSpec generated", "openspec/changes/CHG1")

        with patch.object(dialog_controller.bridge_server, "start_run", start_run), \
                patch.object(dialog_controller, "_artifact_store", return_value=object()), \
                patch.object(dialog_controller, "planning_context", kb_context), \
                patch.object(dialog_controller, "SpecCommandHandler", Handler):
            dialog_controller._start_artifact_command(
                [{"role": "user", "content": "/spec Build a scheduler"}], [], "trace",
                object(), kb_config={"enabled": True, "project_id": "target"})
            await captured["strategy"](run, "", "", [])
        self.assertEqual(captured["context"], "Architecture context")
        self.assertIn("Reading indexed architecture", events[0])
        self.assertEqual(events[-1], "OpenSpec generated")

    async def test_unavailable_kb_preserves_planning_fallback(self):
        progress = []

        async def report(message):
            progress.append(message)

        with patch.dict("os.environ", {"KB_ENABLED": "false"}):
            context = await planning_context({"enabled": True,
                                              "provider": "codebase-memory",
                                              "project_id": "target"},
                                             "openspec", "Build a scheduler", report)
        self.assertEqual(context, "")
        self.assertEqual(progress, ["Codebase Memory unavailable; continuing without it"])

    async def test_no_kb_configuration_emits_no_progress(self):
        progress = []

        async def report(message):
            progress.append(message)

        self.assertEqual(await planning_context(None, "openspec", "Build", report), "")
        self.assertEqual(progress, [])

    async def test_project_is_validated_and_bound_to_every_query(self):
        fake = _Session()

        @asynccontextmanager
        async def session(_self):
            yield fake

        with patch.object(CodebaseMemoryClient, "session", session):
            specs, handlers, status = await CodebaseMemoryClient().planner_tools("target")
            self.assertIn('"status": "ready"', status)
            self.assertEqual(len(specs), 6)
            self.assertNotIn("project", specs[0].schema["properties"])
            readable, _, _ = read_only_handlers(specs, handlers)
            self.assertEqual(set(readable), set(handlers))
            await handlers["kb_get_architecture"]({"project": "other", "aspects": []})
        self.assertEqual(fake.calls[-1][1]["project"], "target")
        self.assertEqual([call[1]["offset"] for call in fake.calls
                          if call[0] == "list_projects"], [0, 1])

    async def test_unknown_project_fails_before_graph_call(self):
        fake = _Session()

        @asynccontextmanager
        async def session(_self):
            yield fake

        with patch.object(CodebaseMemoryClient, "session", session):
            with self.assertRaisesRegex(KnowledgeBaseError, "absent from list_projects"):
                await CodebaseMemoryClient().planner_tools("missing")
        self.assertFalse(any(name == "index_status" for name, _ in fake.calls))

    async def test_stage_contexts_are_bounded_and_use_different_search_limits(self):
        fake = _Session()
        original = fake.call_tool

        async def call(name, arguments):
            if name == "list_projects":
                fake.calls.append((name, arguments))
                payload = {"projects": [{"name": "target", "root_path": "/repo"}]}
                return SimpleNamespace(isError=False, structuredContent=payload, content=[])
            return await original(name, arguments)

        fake.call_tool = call

        @asynccontextmanager
        async def session(_self):
            yield fake

        with patch.object(CodebaseMemoryClient, "session", session):
            client = CodebaseMemoryClient()
            progress = []

            async def report(message):
                progress.append(message)

            high = await client.context(
                "target", "openspec", "Add scheduler dependency tracking", report)
            wp = await client.context("target", "implementation", "Add scheduler dependency tracking")
        self.assertEqual(progress, [
            "Checking Codebase Memory project and index status",
            "Reading indexed architecture",
            "Searching indexed components and dependencies (1/1)",
            "KB context ready; checking the current code",
        ])
        self.assertIn("OpenSpec scope", high)
        self.assertIn("WP files", wp)
        self.assertLessEqual(len(high), 12000)
        self.assertLessEqual(len(wp), 24000)
        searches = [args for name, args in fake.calls if name == "search_graph"]
        self.assertEqual([args["limit"] for args in searches], [8, 12])
        self.assertTrue(all(args["project"] == "target" for args in searches))

    async def test_symbols_named_in_the_request_are_searched_by_name_first(self):
        fake = _Session()
        original = fake.call_tool

        async def call(name, arguments):
            if name == "list_projects":
                fake.calls.append((name, arguments))
                payload = {"projects": [{"name": "target", "root_path": "/repo"}]}
                return SimpleNamespace(isError=False, structuredContent=payload, content=[])
            if name == "search_graph" and "name_pattern" in arguments:
                fake.calls.append((name, arguments))
                payload = {"groups": [{"file": f"pkg/mod{index % 8}.py", "rows": []}
                                      for index in range(10)]}
                return SimpleNamespace(isError=False, structuredContent=payload, content=[])
            return await original(name, arguments)

        fake.call_tool = call

        @asynccontextmanager
        async def session(_self):
            yield fake

        with patch.object(CodebaseMemoryClient, "session", session):
            progress = []

            async def report(message):
                progress.append(message)

            wp = await CodebaseMemoryClient().context(
                "target", "implementation",
                "Implement evaluate_generated_dataset using MGDataset with the WP3 metrics",
                report)
        searches = [args for name, args in fake.calls if name == "search_graph"]
        self.assertEqual(searches[0]["name_pattern"], "evaluate_generated_dataset|MGDataset")
        self.assertNotIn("query", searches[0])
        self.assertNotIn("using", searches[1]["query"].split())
        self.assertIn("Searching indexed components and dependencies (1/2)", progress)
        self.assertIn("Symbols named in the request: ", wp)
        outlines = [args["file_path"] for name, args in fake.calls
                    if name == "get_file_outline"]
        self.assertEqual(outlines, [f"pkg/mod{index}.py" for index in range(6)])
        self.assertIn("Reading outlines of 6 files named in the WP", progress)
        self.assertIn("File outline pkg/mod0.py: ", wp)

    async def test_refresh_is_idempotent_and_nonblocking(self):
        class Persistence:
            def __init__(self):
                self.claimed = False
                self.finished = []

            async def claim_kb_refresh(self, *_args):
                if self.claimed:
                    return False
                self.claimed = True
                return True

            async def finish_kb_refresh(self, *args):
                self.finished.append(args)

        class IDE:
            calls = []

            async def list_tools(self):
                return [SimpleNamespace(name="execute_terminal_command")]

            async def call_tool(self, name, args):
                self.calls.append((name, args))
                return "CBM_EXIT_CODE=0"

        async def metadata(_self, _project):
            return {"name": "target", "root_path": "/repo"}, {"state": "ready"}

        store, ide = Persistence(), IDE()
        with patch.object(CodebaseMemoryClient, "project_metadata", metadata), \
                patch.dict("os.environ", {"KB_ENABLED": "true", "KB_BINARY_PATH": "/bin/cbm",
                                          "KB_DATA_PATH": "/cache"}):
            first = await refresh_accepted_wp(store, "job", "WP1", {
                "enabled": True, "provider": "codebase-memory", "project_id": "target"},
                ide, "/repo")
            again = await refresh_accepted_wp(store, "job", "WP1", {
                "enabled": True, "provider": "codebase-memory", "project_id": "target"},
                ide, "/repo")
        self.assertIn("succeeded", first)
        self.assertIsNone(again)
        self.assertEqual(len(ide.calls), 1)
        self.assertEqual(store.finished[0][2], "succeeded")

    async def test_refresh_rejects_mismatched_project_root_without_host_command(self):
        class Store:
            result = None

            async def claim_kb_refresh(self, *_args):
                return True

            async def finish_kb_refresh(self, *_args):
                self.result = _args

        class IDE:
            async def list_tools(self):
                raise AssertionError("host terminal must not be queried")

        async def metadata(_self, _project):
            return {"name": "target", "root_path": "/different/repo"}, {}

        store = Store()
        with patch.object(CodebaseMemoryClient, "project_metadata", metadata), \
                patch.dict("os.environ", {"KB_ENABLED": "true", "KB_BINARY_PATH": "/bin/cbm",
                                          "KB_DATA_PATH": "/cache"}):
            message = await refresh_accepted_wp(store, "job", "WP1", {
                "enabled": True, "provider": "codebase-memory", "project_id": "target"},
                IDE(), "/repo")
        self.assertIn("accepted review remains valid", message)
        self.assertEqual(store.result[2], "failed")

    async def _catch_up(self, mutations, exit_text="CBM_EXIT_CODE=0"):
        class Store:
            since = None

            async def recent_mutations(self, project_id, since):
                self.project_id, self.since = project_id, since
                return mutations

        class IDE:
            calls = []

            async def list_tools(self):
                return [SimpleNamespace(name="execute_terminal_command")]

            async def call_tool(self, name, args):
                self.calls.append((name, args))
                return exit_text

        async def metadata(_self, _project):
            return ({"name": "target", "root_path": "/repo"},
                    {"indexed_at": "2026-10-19T13:58:43Z"})

        store, ide, progress = Store(), IDE(), []

        async def note(line):
            progress.append(line)

        with patch.object(CodebaseMemoryClient, "project_metadata", metadata), \
                patch.dict("os.environ", {"KB_ENABLED": "true", "KB_BINARY_PATH": "/bin/cbm",
                                          "KB_DATA_PATH": "/cache"}):
            message = await catch_up_index(store, "proj_1", {
                "enabled": True, "provider": "codebase-memory", "project_id": "target"},
                ide, "/repo", progress=note)
        return message, store, ide, progress

    async def test_catch_up_reindexes_when_edits_are_newer_than_index(self):
        message, store, ide, progress = await self._catch_up(
            [{"targets": ["trainer/contracts.py", "models/rnn/model.py"]},
             {"targets": ["trainer/contracts.py"]}])
        self.assertEqual(store.since, 1792418323.0)
        self.assertEqual(len(ide.calls), 1)
        self.assertIn("2 files changed", message)
        self.assertEqual(progress, ["KB index is behind 2 changed files; refreshing"])

    async def test_catch_up_is_a_noop_when_index_is_current(self):
        message, _, ide, progress = await self._catch_up([])
        self.assertIsNone(message)
        self.assertEqual(ide.calls, [])
        self.assertEqual(progress, [])

    async def test_catch_up_failure_never_raises(self):
        message, _, ide, _ = await self._catch_up(
            [{"targets": ["trainer/contracts.py"]}], exit_text="CBM_EXIT_CODE=1")
        self.assertIsNone(message)
        self.assertEqual(len(ide.calls), 1)

    def test_resumed_review_without_service_still_refreshes_kb(self):
        session = SimpleNamespace(client=object(), project_path="/repo", knowledge_base={
            "enabled": True, "provider": "codebase-memory", "project_id": "target"})
        run = SimpleNamespace(metadata={"implementation_job_id": "job"})
        with patch.object(dialog_controller, "_artifact_store", return_value=object()):
            service = dialog_controller._resume_service(run, session, [])
        self.assertIsNotNone(service.kb_refresh)
        attached = object()
        run.metadata["implementation_service"] = attached
        self.assertIs(dialog_controller._resume_service(run, session, []), attached)

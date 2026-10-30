"""Native filesystem execution: project registry, native plans, provider flags,
git-baseline rollback and the selective KB policy.

    .venv/bin/python -m unittest tests.test_workspace -v
"""

import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent import workspace
from agent.routing.config import load_routing_config
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.persistence import RoutingPersistence
from agent.routing.rollback import RollbackJournal
from agent.routing.service import ModelRoutingService
from agent.routing.task_execution import TaskExecutionService
from agent.spec.command import extract_project_command
from mcp_bridge.models import (DoneEvent, ErrorEvent, MutationEvent, ProgressEvent, Run,
                               TextEvent)
from mcp_bridge.registry import ToolSpec, native_bridged_specs
from providers.codex import CodexProvider
from tests.test_routing import Router, task


@contextmanager
def _env(**values):
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update({key: value for key, value in values.items() if value is not None})
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    *args], check=True, capture_output=True)


def _repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("print('v1')\n")
    (root / "src" / "keep.py").write_text("keep = 1\n")
    (root / "staged.py").write_text("base\n")
    (root / "run.sh").write_text("echo hi\n")
    (root / ".gitignore").write_text("ignored/\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


class _Fixture(unittest.TestCase):
    """A workspace root with a registry; mounts are simulated with `ismount`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(os.path.realpath(self.tmp.name))
        self.ws_root = self.root / "workspace"
        self.ws_root.mkdir()
        self.config = self.root / "projects.yaml"
        self.mounted: set[str] = set()
        self.patches = [
            _env(CODEPARTNER_WORKSPACE_ROOT=str(self.ws_root),
                 CODEPARTNER_PROJECTS_CONFIG=str(self.config),
                 CODEPARTNER_NATIVE_PROVIDERS=None),
            mock.patch("agent.workspace.os.path.ismount",
                       side_effect=lambda path: str(path) in self.mounted),
        ]
        for patch in self.patches:
            patch.__enter__()
        workspace.activate(None)

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.__exit__(None, None, None)
        workspace.activate(None)
        self.tmp.cleanup()

    def register(self, text: str) -> None:
        self.config.write_text(text)
        # Force the mtime/size stamp to change even within one timestamp tick.
        stamp = self.config.stat().st_mtime_ns + 1_000_000
        os.utime(self.config, ns=(stamp, stamp))

    def project(self, project_id="demo", git=True, mount=True) -> Path:
        path = self.ws_root / project_id
        if git:
            _repo(path)
        else:
            path.mkdir()
        if mount:
            self.mounted.add(str(path))
        self.register(f"projects:\n  {project_id}:\n    workspace: {path}\n"
                      f"    host_path: /home/u/{project_id}\n")
        return path


class RegistryTests(_Fixture):
    def test_valid_registry_resolves_and_reloads(self):
        path = self.project("music-generation")
        registry = workspace.registry()
        self.assertEqual(registry.resolve("music-generation").workspace, str(path))
        self.register(self.config.read_text()
                      + f"  volley-vision:\n    workspace: {self.ws_root}/volley-vision\n")
        self.assertIn("volley-vision", registry.projects())

    def test_rejects_unregistered_outside_root_unmounted_and_symlinks(self):
        self.project("demo", mount=False)
        registry = workspace.registry()
        with self.assertRaisesRegex(workspace.WorkspaceError, "not registered"):
            registry.resolve("other")
        with self.assertRaisesRegex(workspace.WorkspaceError, "not a bind mount"):
            registry.resolve("demo")
        self.register("projects:\n  missing:\n    workspace: "
                      f"{self.ws_root}/missing\n")
        with self.assertRaisesRegex(workspace.WorkspaceError, "recreate the container"):
            registry.resolve("missing")
        self.register("projects:\n  outside:\n    workspace: /etc\n")
        with self.assertRaisesRegex(workspace.WorkspaceError, "must be inside"):
            registry.resolve("outside")
        target = self.root / "elsewhere"
        target.mkdir()
        (self.ws_root / "link").symlink_to(target)
        self.mounted.add(str(self.ws_root / "link"))
        self.register(f"projects:\n  link:\n    workspace: {self.ws_root}/link\n")
        with self.assertRaisesRegex(workspace.WorkspaceError, "symlink"):
            registry.resolve("link")

    def test_schema_validation(self):
        for bad in ("projects:\n  Bad Id:\n    workspace: /workspace/x\n",
                    "projects:\n  x:\n    workspace: relative\n",
                    "projects:\n  x:\n    workspace: /workspace/x\n    execution: fast\n"):
            self.register(bad)
            with self.assertRaises(workspace.WorkspaceError):
                workspace.registry().projects()

    def test_match_prefers_project_id_then_host_path(self):
        self.project("demo")
        registry = workspace.registry()
        self.assertEqual(registry.match("demo", "/elsewhere"), "demo")
        self.assertEqual(registry.match("unknown", "/home/u/demo/"), "demo")
        self.assertIsNone(registry.match("unknown", "/home/u/other"))

    def test_project_command_parsing(self):
        self.assertEqual(extract_project_command("/project"), ("show", None))
        self.assertEqual(extract_project_command("/project use demo"), ("use", "demo"))
        with self.assertRaises(ValueError):
            extract_project_command("/project use ../etc")


class NativePlanTests(_Fixture):
    def test_native_when_registered_mounted_and_git(self):
        self.project("demo")
        self.assertTrue(workspace.plan_native("claude", project_id="demo").native)

    def test_unavailable_plans_give_reasons(self):
        unbound = workspace.plan_native("codex", project_id=None)
        self.assertFalse(unbound.native)
        self.assertIn("register_project.py", unbound.reason)
        self.project("demo", git=False)
        plan = workspace.plan_native("claude", project_id="demo")
        self.assertFalse(plan.native)
        self.assertIn("git", plan.reason)

    def test_unmounted_project_reports_the_mount(self):
        self.project("demo", mount=False)
        plan = workspace.plan_native("claude", project_id="demo")
        self.assertFalse(plan.native)
        self.assertIn("bind mount", plan.reason)

    def test_excluded_files_stay_native_and_provider_allowlist_applies(self):
        self.project("demo")
        self.assertTrue(workspace.plan_native("claude", ("secret.py",), project_id="demo").native)
        with _env(CODEPARTNER_NATIVE_PROVIDERS="claude"):
            self.assertIn("CODEPARTNER_NATIVE_PROVIDERS",
                          workspace.plan_native("codex", project_id="demo").reason)

    def test_toolchain_uses_container_uv_environment(self):
        path = self.project("demo")
        (path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (path / ".venv").mkdir()
        env, notes = workspace.toolchain(workspace.registry().get("demo"))
        self.assertTrue(env["UV_PROJECT_ENVIRONMENT"].endswith("codepartner-venvs/demo"))
        self.assertTrue(any("host" in note for note in notes))


class InventoryTests(unittest.TestCase):
    def test_inventory_groups_by_directory_and_truncates(self):
        with tempfile.TemporaryDirectory() as root:
            subprocess.run(["git", "init", "-q", root], check=True)
            for path in ("main.py", "src/a.py", "src/b.py", "tests/test_a.py", ".gitignore"):
                Path(root, path).parent.mkdir(parents=True, exist_ok=True)
                Path(root, path).write_text("ignored.py\n" if path == ".gitignore" else "")
            Path(root, "ignored.py").write_text("")
            inventory = workspace.project_inventory(root)
            self.assertEqual(inventory.splitlines(), [
                "./: .gitignore main.py", "src/: a.py b.py", "tests/: test_a.py"])
            short = workspace.project_inventory(root, limit_chars=40)
            self.assertEqual(short.splitlines(), [
                "./: .gitignore main.py", "src/: a.py b.py",
                "(+1 more files in 1 directories)"])
        self.assertEqual(workspace.project_inventory("/nonexistent/project"), "")


class _Executor:
    def __init__(self):
        self.contexts = []

    async def execute(self, _task, _model, context):
        self.contexts.append(context)
        yield TextEvent("done")
        yield DoneEvent("endTurn")


class TaskExecutionNativeTests(_Fixture):
    async def _execute(self, owner=None, contract=None):
        executor = _Executor()
        config = load_routing_config()
        config = replace(config, discovery=replace(config.discovery, policy="selective"))
        service = TaskExecutionService(
            config=config, router=ModelRoutingService(config, Router("claude-haiku")),
            persistence=RoutingPersistence(self.root / "routing.sqlite3"),
            executors=ExecutorRegistry({"claude-cli": executor, "codex-cli": executor}))
        events = [event async for event in service.execute(
            task("simple"), "CHG1", [], [], contract=contract,
            owner_metadata=owner if owner is not None else {
                "implementation_job_id": "job", "implementation_position": 0})]
        return executor, events

    def test_native_attempt_gets_workspace_and_native_prompt(self):
        import asyncio
        path = self.project("demo")
        workspace.activate("demo")
        executor, _ = asyncio.run(self._execute())
        context = executor.contexts[0]
        self.assertEqual(context.native.workspace, str(path))
        self.assertEqual(context.owner_metadata["execution_mode"], "native")
        prompt = context.messages[-1]["content"]
        self.assertIn("project's working directory", prompt)
        self.assertNotIn("codepartner_read_file", prompt)

    def test_kb_first_runs_natively_with_the_prompt_rule(self):
        from agent.prompt.coder import CoderPromptBuilder
        KB_FIRST_RULE = CoderPromptBuilder().read_pipeline()
        import asyncio
        self.project("demo")
        workspace.activate("demo")
        executor = _Executor()
        config = load_routing_config()
        service = TaskExecutionService(
            config=config, router=ModelRoutingService(config, Router("claude-haiku")),
            persistence=RoutingPersistence(self.root / "routing.sqlite3"),
            executors=ExecutorRegistry({"claude-cli": executor, "codex-cli": executor}))
        async def collect():
            return [event async for event in service.execute(
                task("simple"), "CHG1", [], [], owner_metadata={
                    "implementation_job_id": "job", "implementation_position": 0})]
        asyncio.run(collect())
        context = executor.contexts[0]
        self.assertIsNotNone(context.native)
        self.assertEqual(context.owner_metadata["execution_mode"], "native")
        self.assertIn(KB_FIRST_RULE, context.messages[-1]["content"])

    def test_default_without_project_is_an_error(self):
        import asyncio
        executor, events = asyncio.run(self._execute())
        self.assertEqual(executor.contexts, [])
        errors = [event.message for event in events if isinstance(event, ErrorEvent)]
        self.assertTrue(errors and "register_project.py" in errors[0])

    def test_excluded_files_run_natively(self):
        import asyncio
        self.project("demo")
        workspace.activate("demo")
        contract = SimpleNamespace(revision=1, goal="g", primary_files=(),
                                   excluded_files=("secret.py",))
        executor, _ = asyncio.run(self._execute(contract=contract))
        self.assertIsNotNone(executor.contexts[0].native)
        self.assertFalse(executor.contexts[0].native.read_only)

    def test_attempt_without_job_is_read_only(self):
        import asyncio
        self.project("demo")
        workspace.activate("demo")
        executor, _ = asyncio.run(self._execute(owner={}))
        self.assertTrue(executor.contexts[0].native.read_only)

    def test_explicit_native_unavailable_is_an_error(self):
        import asyncio
        self.project("demo", git=False)
        workspace.activate("demo")
        executor, events = asyncio.run(self._execute())
        self.assertEqual(executor.contexts, [])
        errors = [event.message for event in events if isinstance(event, ErrorEvent)]
        self.assertTrue(errors and "Native execution was requested" in errors[0])


class ProviderFlagTests(unittest.TestCase):
    def test_codex_native_args(self):
        native = workspace.NativeRun(workspace="/workspace/demo")
        args = CodexProvider()._base_args("gpt-x", concrete_model=True, native=native)
        self.assertEqual(args[args.index("--cd") + 1], "/workspace/demo")
        # bubblewrap cannot start in the container: native Codex runs unsandboxed.
        self.assertEqual(args[-1], "--dangerously-bypass-approvals-and-sandbox")
        self.assertNotIn("-s", args)
        self.assertNotIn("features.shell_tool=false", args)
        read_only = CodexProvider()._base_args(
            native=workspace.NativeRun(workspace="/workspace/demo", read_only=True))
        self.assertEqual(read_only[-1], "--dangerously-bypass-approvals-and-sandbox")

    def test_codex_ide_args_unchanged(self):
        with _env(CODEX_CWD=None):
            args = CodexProvider()._base_args("gpt-x", concrete_model=True)
        self.assertEqual(args[-4:], ["-c", "features.shell_tool=false", "-s", "read-only"])
        self.assertNotIn("--cd", args)

    def test_claude_native_options(self):
        import asyncio
        from providers import claude
        captured = {}

        async def fake_query(prompt, options):
            captured["options"] = options
            if False:
                yield None

        run = Run()
        run.native = workspace.NativeRun(workspace="/workspace/demo", env={"A": "1"})
        with mock.patch.object(claude, "_query_messages", fake_query):
            asyncio.run(claude.ClaudeProvider()._run_strategy(
                run, "p", None, [ToolSpec("get_file_problems", "d", {"type": "object"})]))
        options = captured["options"]
        self.assertEqual(options.cwd, "/workspace/demo")
        self.assertIn("Bash", options.tools)
        self.assertIn("Edit", options.allowed_tools)
        self.assertIn("mcp__jetbrains__get_file_problems", options.allowed_tools)
        self.assertIn("WebFetch", options.disallowed_tools)
        self.assertEqual(options.env["A"], "1")
        self.assertTrue(options.strict_mcp_config)  # no claude.ai connectors or user MCP
        run.native = workspace.NativeRun(workspace="/workspace/demo", read_only=True)
        with mock.patch.object(claude, "_query_messages", fake_query):
            asyncio.run(claude.ClaudeProvider()._run_strategy(run, "p", None, []))
        self.assertEqual(sorted(captured["options"].tools), ["Glob", "Grep", "Read"])

    def test_native_overlap_filter_keeps_ide_specific_and_kb_tools(self):
        specs = [ToolSpec(name, "d", {"type": "object"}) for name in (
            "read_file", "mcp__pycharm__search_text", "apply_patch", "list_directory_tree",
            "get_file_problems", "rename_refactoring", "execute_tool", "kb_search_graph")]
        self.assertEqual([spec.name for spec in native_bridged_specs(specs)],
                         ["get_file_problems", "rename_refactoring", "execute_tool",
                          "kb_search_graph"])


class PathTranslationTests(unittest.TestCase):
    """Codebase Memory indexes the host checkout; native agents see the workspace."""

    def setUp(self):
        self.native = workspace.NativeRun(workspace="/workspace/demo",
                                          host_path="/home/u/Demo")

    def test_host_paths_become_workspace_paths_on_a_boundary_only(self):
        text = ('{"file": "/home/u/Demo/pkg/a.py", "root": "/home/u/Demo", '
                '"other": "/home/u/Demo2/b.py"} at /home/u/Demo/c.py:12')
        self.assertEqual(workspace.to_workspace(text, self.native),
                         '{"file": "/workspace/demo/pkg/a.py", "root": "/workspace/demo", '
                         '"other": "/home/u/Demo2/b.py"} at /workspace/demo/c.py:12')

    def test_arguments_go_back_to_host_paths(self):
        arguments = {"file_path": "/workspace/demo/pkg/a.py", "path": "pkg/a.py",
                     "paths": ["/workspace/demo/x.py", "/workspace/demo2/y.py"], "limit": 5}
        self.assertEqual(workspace.to_host(arguments, self.native),
                         {"file_path": "/home/u/Demo/pkg/a.py", "path": "pkg/a.py",
                          "paths": ["/home/u/Demo/x.py", "/workspace/demo2/y.py"], "limit": 5})

    def test_no_translation_without_a_host_path(self):
        plain = workspace.NativeRun(workspace="/workspace/demo")
        self.assertEqual(workspace.to_workspace("/home/u/Demo/a.py", plain), "/home/u/Demo/a.py")
        self.assertEqual(workspace.to_host({"p": "/workspace/demo/a.py"}, plain),
                         {"p": "/workspace/demo/a.py"})
        self.assertEqual(workspace.to_workspace("/home/u/Demo/a.py", None), "/home/u/Demo/a.py")

    def test_relative_collapses_every_path_form(self):
        for path in ("pkg/a.py", "./pkg/a.py", "/workspace/demo/pkg/a.py",
                     "/home/u/Demo/pkg/a.py"):
            self.assertEqual(self.native.relative(path), "pkg/a.py")
        self.assertIsNone(self.native.relative("/etc/passwd"))
        self.assertIsNone(self.native.relative("/workspace/demo"))
        self.native.record_write("/home/u/Demo/pkg/b.py")
        self.assertEqual(self.native.written, {"pkg/b.py"})

    def test_sessions_carry_the_registered_host_path(self):
        project = workspace.RegisteredProject("demo", "/workspace/demo", "/home/u/Demo")
        with mock.patch.object(workspace, "toolchain", return_value=({}, ())):
            self.assertEqual(workspace.read_only_session(project).host_path, "/home/u/Demo")


class GitBaselineRollbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.repo = _repo(base / "repo")
        self.persistence = RoutingPersistence(base / "routing.sqlite3")
        self.journal = RollbackJournal(self.persistence, "job", 0, [], {})

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def _status(self) -> str:
        return subprocess.run(["git", "-C", str(self.repo), "status", "--porcelain=v1",
                               "--untracked-files=all"],
                              capture_output=True, text=True, check=True).stdout

    def _index(self) -> str:
        return subprocess.run(["git", "-C", str(self.repo), "ls-files", "-s"],
                              capture_output=True, text=True, check=True).stdout

    async def test_moved_head_requires_recovery(self):
        await self.journal.record_git_baseline(str(self.repo))
        (self.repo / "src" / "app.py").write_text("committed\n")
        _git(self.repo, "commit", "-qam", "agent commit")
        await self.journal.record_native_changes()
        issues = await self.journal.integrity_issues()
        self.assertTrue(any("HEAD moved" in issue.get("error", "") for issue in issues))

    async def test_engine_journals_native_run_and_emits_mutations(self):
        from mcp_bridge import server as engine
        await self.journal.record_git_baseline(str(self.repo))

        async def strategy(run, prompt, system, specs):
            (self.repo / "src" / "app.py").write_text("by agent\n")
            return "endTurn"

        native = workspace.NativeRun(
            workspace=str(self.repo),
            start=lambda: self.journal.record_git_baseline(str(self.repo)),
            finish=self.journal.record_native_changes)
        run = engine.start_run("p", None, [], strategy)
        run.native = native
        events = []
        while True:
            event = await run.next_event()
            events.append(event)
            if isinstance(event, DoneEvent):
                break
        self.assertIn(MutationEvent("native_cli", {"path": "src/app.py"}), events)
        self.assertEqual(await self.journal.changed_paths(), ["src/app.py"])


    async def test_excluded_change_fails_and_is_journaled(self):
        await self.journal.record_git_baseline(str(self.repo))
        (self.repo / "src" / "app.py").write_text("allowed\n")
        (self.repo / "staged.py").write_text("forbidden\n")
        with self.assertRaises(workspace.NativeSessionError) as caught:
            await self.journal.record_native_changes(("staged.py",))
        self.assertIn("excludes: staged.py", str(caught.exception))
        self.assertEqual((self.repo / "staged.py").read_text(), "forbidden\n")  # not reverted
        self.assertEqual(sorted(await self.journal.changed_paths()), ["src/app.py", "staged.py"])

    async def test_baseline_writes_no_git_objects(self):
        def objects():
            return sorted(str(path) for path in (self.repo / ".git" / "objects").rglob("*"))
        before = objects()
        (self.repo / "untracked.txt").write_text("new content\n")
        await self.journal.record_git_baseline(str(self.repo))
        (self.repo / "src" / "app.py").write_text("changed\n")
        self.assertEqual(await self.journal.record_native_changes(), ["src/app.py"])
        self.assertEqual(objects(), before)


async def _drain(run):
    events = []
    while True:
        event = await run.next_event()
        events.append(event)
        if isinstance(event, DoneEvent):
            return events


class ReadOnlySessionTests(unittest.IsolatedAsyncioTestCase):
    """Read-only native runs: detect-and-restore through the engine lifecycle."""

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = _repo(Path(self.tmp.name) / "repo")
        self.project = workspace.RegisteredProject("demo", str(self.repo))

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def _start(self, strategy, can_write=True):
        from mcp_bridge import server as engine
        native = workspace.read_only_session(self.project)
        native.agent_can_write = can_write
        run = engine.start_run("p", None, [], strategy)
        run.native = native
        return run, native

    def _agent_writes(self, native, path="src/app.py", text="by agent\n"):
        (self.repo / path).write_text(text)
        native.record_write(str(self.repo / path))

    def _assert_failed_and_reported(self, events):
        errors = [e.message for e in events if isinstance(e, ErrorEvent)]
        self.assertTrue(any("the agent wrote src/app.py" in e and "Nothing was reverted" in e
                            for e in errors), errors)
        self.assertEqual(events[-1].terminal, "failed")
        self.assertEqual((self.repo / "src" / "app.py").read_text(), "by agent\n")

    async def test_clean_run_succeeds(self):
        run, _ = self._start(lambda *a: asyncio.sleep(0, "endTurn"))
        events = await _drain(run)
        self.assertEqual(events[-1].terminal, "endTurn")
        self.assertFalse([e for e in events if isinstance(e, ErrorEvent)])

    async def test_agent_write_fails_the_run(self):
        async def strategy(run, *_):
            self._agent_writes(run.native)
            return "endTurn"
        run, _ = self._start(strategy)
        self._assert_failed_and_reported(await _drain(run))

    async def test_agent_write_is_reported_when_the_agent_fails(self):
        async def strategy(run, *_):
            self._agent_writes(run.native)
            raise RuntimeError("model crashed")
        run, _ = self._start(strategy)
        self._assert_failed_and_reported(await _drain(run))

    async def test_check_runs_on_timeout_and_cancel(self):
        for label in ("timeout", "cancel"):
            started = asyncio.Event()

            async def strategy(run, *_):
                self._agent_writes(run.native)
                started.set()
                await asyncio.sleep(3600)
            run, native = self._start(strategy)
            await started.wait()
            if label == "timeout":
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(run.task), 0.05)
            run.task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await run.task
            self.assertEqual((self.repo / "src" / "app.py").read_text(), "by agent\n", label)
            self.assertIsNone(native.finish)  # the check ran despite the cancellation

    async def test_unattributable_change_is_reported_not_reverted(self):
        async def strategy(run, *_):
            (self.repo / "src" / "keep.py").write_text("user edit in the IDE\n")
            return "endTurn"
        run, _ = self._start(strategy, can_write=False)  # read-only Claude: no write tools
        events = await _drain(run)
        self.assertEqual(events[-1].terminal, "endTurn")
        warnings = [e.text for e in events if isinstance(e, ProgressEvent)]
        self.assertTrue(any("possibly by you or another run" in w and "src/keep.py" in w
                            for w in warnings))
        self.assertEqual((self.repo / "src" / "keep.py").read_text(), "user edit in the IDE\n")

    async def test_shell_capable_agent_fails_on_unattributed_change(self):
        async def strategy(run, *_):
            (self.repo / "src" / "keep.py").write_text("echo > file\n")
            return "endTurn"
        run, _ = self._start(strategy, can_write=True)  # Codex: its shell could have written
        events = await _drain(run)
        self.assertEqual(events[-1].terminal, "failed")
        self.assertEqual((self.repo / "src" / "keep.py").read_text(), "echo > file\n")


if __name__ == "__main__":
    unittest.main()


class NativePlannerTests(_Fixture):
    def test_kb_first_planner_runs_read_only_native(self):
        from agent.planning.service import WPPlannerService
        from tests.test_turn_routing import _specs
        from tests.test_wp_planner import PlanArtifacts, PlannerProvider, plan_payload
        self.project("demo")
        workspace.activate("demo")
        provider = PlannerProvider(plan_payload())
        planner = WPPlannerService(
            PlanArtifacts(), RoutingPersistence(self.root / "routing.sqlite3"),
            config=load_routing_config(), provider_lookup=lambda _name: provider)
        asyncio.run(planner.start("conv", "WP4", "Let's do WP4", _specs(), {}))
        call = provider.calls[0]
        self.assertTrue(call["native"].read_only)
        self.assertNotIn("kb_first", call)

    def test_bound_selective_planner_runs_read_only_native(self):
        from agent.prompt.planner import SYSTEM as NATIVE_SYSTEM
        from agent.planning.service import WPPlannerService
        from tests.test_turn_routing import _specs
        from tests.test_wp_planner import PlanArtifacts, PlannerProvider, plan_payload
        path = self.project("demo")
        workspace.activate("demo")
        provider = PlannerProvider(plan_payload())
        persistence = RoutingPersistence(self.root / "routing.sqlite3")
        config = load_routing_config()
        config = replace(config, discovery=replace(config.discovery, policy="selective"))
        planner = WPPlannerService(PlanArtifacts(), persistence, config=config,
                                   provider_lookup=lambda _name: provider)
        turn = asyncio.run(planner.start("conv", "WP4", "Let's do WP4", _specs(), {}))
        self.assertEqual(turn.gaps, [])
        call = provider.calls[0]
        self.assertTrue(call["native"].read_only)
        self.assertEqual(call["native"].workspace, str(path))
        prompt = call["conversation"][0]["content"]
        self.assertTrue(prompt.startswith(NATIVE_SYSTEM))
        self.assertNotIn("IDE", NATIVE_SYSTEM)

    def test_unbound_planner_fails_with_registration_hint(self):
        from agent.planning.service import PlanningError, WPPlannerService
        from tests.test_turn_routing import _specs
        from tests.test_wp_planner import PlanArtifacts, PlannerProvider, plan_payload
        provider = PlannerProvider(plan_payload())
        config = load_routing_config()
        config = replace(config, discovery=replace(config.discovery, policy="selective"))
        planner = WPPlannerService(PlanArtifacts(),
                                   RoutingPersistence(self.root / "routing.sqlite3"),
                                   config=config,
                                   provider_lookup=lambda _name: provider)
        with self.assertRaises(PlanningError) as caught:
            asyncio.run(planner.start("conv", "WP4", "Let's do WP4", _specs(), {}))
        self.assertIn("register_project.py", str(caught.exception))
        self.assertEqual(provider.calls, [])


class NativeReviewerTests(_Fixture):
    """The review service reads the mounted project; the model pass is read-only native."""

    def _review(self, provider, manifest, report="Changed `src/app.py`.", tool_free=False):
        from agent.routing.review import ReviewService
        from tests.test_routing import task as make_task
        queued = replace(make_task(), context="")
        object.__setattr__(queued, "_review_journal_manifest", manifest)
        if tool_free:
            object.__setattr__(queued, "_review_tool_free", True)
            object.__setattr__(queued, "_review_reference_source", "### src/ref.py\n1| x")
            object.__setattr__(queued, "_review_reference_paths", frozenset({"src/ref.py"}))
        with mock.patch("agent.routing.review.providers.get", return_value=provider):
            config = load_routing_config()
            config = replace(config, discovery=replace(config.discovery, policy="selective"))
            return asyncio.run(ReviewService(config=config).review(
                SimpleNamespace(title="Change", work_packages=[]), queued, report,
                "claude-haiku", [], {}))

    def _provider(self, verdict, model_reads=()):
        class Provider:
            calls = []

            async def tools(self, messages, specs, **kwargs):
                Provider.calls.append({"prompt": messages[0]["content"], **kwargs})
                native = kwargs["native"]
                for path in model_reads:  # what the provider's record_native_tool keeps
                    native.metrics["files"].append(os.path.join(native.workspace, path))
                yield TextEvent(json.dumps(verdict))
                yield DoneEvent("endTurn")
        return Provider()

    def test_snapshot_is_read_from_disk_and_model_runs_read_only(self):
        path = self.project("demo")
        workspace.activate("demo")
        (path / "src" / "app.py").write_text("print('v2')\n")
        provider = self._provider({
            "verdict": "done", "rationale": "app prints v2", "evidence": ["src/app.py prints v2"],
            "inspected_paths": ["src/app.py", "src/keep.py", "never/read.py"], "findings": []},
            model_reads=["src/keep.py"])
        decision = self._review(provider, [{
            "path": "src/app.py", "change": "modified", "before_exists": True,
            "after_exists": True, "operations": ["replace"], "postimage_verified": True}])
        self.assertEqual(decision.verdict, "done")
        call = provider.calls[0]
        self.assertTrue(call["native"].read_only)
        self.assertIn("print('v2')", call["prompt"])
        self.assertIn("Filesystem-verified artifact snapshot", call["prompt"])
        self.assertNotIn("request-scoped IDE MCP", call["prompt"])

    def test_inline_review_is_tool_free_only_with_the_whole_evidence(self):
        path = self.project("demo")
        workspace.activate("demo")
        (path / "src" / "app.py").write_text("print('v2')\n")
        manifest = [{"path": "src/app.py", "change": "modified", "before_exists": True,
                     "after_exists": True, "operations": ["replace"],
                     "postimage_verified": True}]
        verdict = {"verdict": "done", "rationale": "ok", "evidence": ["app prints v2"],
                   "inspected_paths": ["src/app.py"], "findings": []}
        provider = self._provider(verdict)
        self._review(provider, manifest, tool_free=True)
        call = provider.calls[0]
        self.assertIsNone(call["native"])
        self.assertIn("You have no tools", call["prompt"])
        self.assertIn("1| x", call["prompt"])
        # Reference source is evidence the reviewer may cite in a finding.
        provider = self._provider({
            "verdict": "needs_work", "rationale": "app ignores ref", "evidence": ["ref.py x"],
            "inspected_paths": ["src/app.py", "src/ref.py"],
            "findings": [{"kind": "implementation", "problem": "app does not use ref.x",
                          "suggested_task": "use it", "approval_criteria": ["uses ref.x"],
                          "artifact_paths": ["src/ref.py"], "owner": "coder",
                          "reason": "implementation_defect"}]})
        decision = self._review(provider, manifest, tool_free=True)
        self.assertEqual(decision.verdict, "needs_work")
        self.assertEqual(decision.findings[0].artifact_paths, ("src/ref.py",))
        # A file the coder names but did not journal is cut to the snapshot budget:
        # the reviewer then keeps its read-only tools.
        (path / "src" / "big.py").write_text("x = 1\n" * 5000)
        provider = self._provider(verdict)
        self._review(provider, manifest, report="Changed `src/app.py` and `src/big.py`.",
                     tool_free=True)
        call = provider.calls[0]
        self.assertTrue(call["native"].read_only)
        self.assertNotIn("You have no tools", call["prompt"])

    def test_lockfiles_are_summarized_in_the_snapshot(self):
        path = self.project("demo")
        workspace.activate("demo")
        (path / "uv.lock").write_text("pkg = 1\n" * 3000)
        provider = self._provider({"verdict": "done", "rationale": "ok", "evidence": ["x"],
                                   "inspected_paths": ["uv.lock"], "findings": []})
        self._review(provider, [{"path": "uv.lock", "change": "created",
                                 "before_exists": False, "after_exists": True,
                                 "operations": ["create"], "postimage_verified": True}],
                     report="Updated `uv.lock`.")
        prompt = provider.calls[0]["prompt"]
        self.assertIn("dependency lockfile, 3000 lines; content omitted", prompt)
        self.assertNotIn("pkg = 1", prompt)

    def test_absolute_workspace_paths_in_the_report_are_project_files(self):
        path = self.project("demo")
        workspace.activate("demo")
        (path / "src" / "app.py").write_text("print('v2')\n")
        provider = self._provider({
            "verdict": "done", "rationale": "ok", "evidence": ["src/app.py prints v2"],
            "inspected_paths": ["src/app.py"], "findings": []})
        decision = self._review(provider, [{
            "path": "src/app.py", "change": "modified", "before_exists": True,
            "after_exists": True, "operations": ["replace"], "postimage_verified": True}],
            report=f"Updated `{path}/src/app.py`.")
        self.assertEqual(decision.verdict, "done")

    def test_journaled_deletion_is_verified_absent_on_disk(self):
        path = self.project("demo")
        workspace.activate("demo")
        (path / "staged.py").unlink()
        provider = self._provider({
            "verdict": "done", "rationale": "removed", "evidence": ["staged.py is absent"],
            "inspected_paths": ["staged.py"], "findings": []})
        decision = self._review(provider, [{
            "path": "staged.py", "change": "deleted", "before_exists": True,
            "after_exists": False, "operations": ["delete"], "postimage_verified": True}],
            report="Deleted `staged.py`.")
        self.assertEqual(decision.verdict, "done")
        self.assertIn('"status": "verified_absent"', provider.calls[0]["prompt"])

    def test_symlink_outside_the_project_is_not_read(self):
        from agent.routing.review import _disk_read
        path = self.project("demo")
        outside = self.root / "secret.txt"
        outside.write_text("secret")
        (path / "link.txt").symlink_to(outside)
        content, error, missing = _disk_read(str(path), "link.txt")
        self.assertIsNone(content)
        self.assertIn("outside the project", error)


class NativeEvidenceTests(_Fixture):
    def test_claude_bash_exit_code_matches_observed_cli_format(self):
        from claude_agent_sdk import ToolResultBlock
        from providers.claude import bash_exit_code
        # Formats observed from claude_agent_sdk 0.2.128 / CLI 2.1.287 in the container.
        self.assertEqual(bash_exit_code(ToolResultBlock("t", "Ran 1 test\n\nOK", False)), 0)
        self.assertEqual(bash_exit_code(ToolResultBlock("t", "Exit code 3\nboom", True)), 3)
        self.assertIsNone(bash_exit_code(ToolResultBlock("t", "Command timed out", True)))

    def test_native_test_commands_become_execution_evidence(self):
        from agent.routing.evidence import has_passing_test_run
        self.project("demo")
        persistence = RoutingPersistence(self.root / "routing.sqlite3")
        service = TaskExecutionService(
            config=load_routing_config(), router=None, persistence=persistence,
            executors=ExecutorRegistry({}))
        native = service._native_run(workspace.registry().get("demo"), {
            "implementation_job_id": "job", "implementation_position": 0}, task("simple"))

        async def run():
            await native.command_finished("ls -la", "files", 0)  # not a verification call
            await native.command_finished("uv run pytest -q", "1 failed, 2 passed", 1)
            failing = await persistence.verification_evidence("job", 0)
            await native.command_finished("uv run pytest -q", "3 passed in 0.1s", 0)
            return failing, await persistence.verification_evidence("job", 0)

        with _env(CODEPARTNER_EVIDENCE_DIR=str(self.root / "evidence")):
            failing, evidence = asyncio.run(run())
        self.assertEqual(len(failing), 1)
        self.assertFalse(has_passing_test_run(failing))
        self.assertEqual(evidence[-1]["tool"], workspace.NATIVE_SHELL_TOOL)
        self.assertTrue(has_passing_test_run(evidence))

    def test_codex_command_items_reach_the_evidence_hook(self):
        seen = []

        async def on_command(command, output, code):
            seen.append((command, output, code))
        native = workspace.NativeRun(workspace="/w", on_command=on_command)
        run = Run(registry=None)
        run.native = native
        item = {"type": "command_execution", "command": "pytest -q",
                "aggregated_output": "2 passed", "exit_code": 0, "status": "completed"}

        async def drive():
            from providers.codex import record_native_item
            record_native_item(run, item)
            await native.command_finished(item["command"], item["aggregated_output"],
                                          item["exit_code"])
        asyncio.run(drive())
        self.assertEqual(seen, [("pytest -q", "2 passed", 0)])
        self.assertEqual(native.metrics["tools"], {"shell": 1})

    def test_required_tests_run_natively_without_ide_dispatcher(self):
        path = self.project("demo")
        (path / "test_ok.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n")
        output, code = asyncio.run(workspace.run_native_command(
            workspace.registry().get("demo"), "python3 -m unittest -q test_ok 2>&1; echo done"))
        self.assertEqual(code, 0)
        self.assertIn("OK", output)


class FilesystemArtifactTests(_Fixture):
    def test_artifacts_are_read_and_written_atomically_inside_the_project(self):
        from agent.spec.filesystem_client import FilesystemFileClient
        from agent.spec.jetbrains_mcp import JetBrainsMcpError
        path = self.project("demo")
        client = FilesystemFileClient(str(path))

        async def run():
            self.assertFalse(await client.file_exists(".codepartner/spec-index.json"))
            await client.create_file(".codepartner/spec-index.json", '{"version": 2}')
            self.assertTrue(await client.file_exists(".codepartner/spec-index.json"))
            self.assertEqual(await client.read_file(".codepartner/spec-index.json"),
                             '{"version": 2}')
            with self.assertRaises(JetBrainsMcpError):
                await client.create_file(".codepartner/spec-index.json", "x", overwrite=False)
            with self.assertRaisesRegex(JetBrainsMcpError, "doesn't exist"):
                await client.read_file("openspec/changes/missing.md")
            for bad in ("src/app.py", "openspec/../src/app.py", "openspec/../openspecx/a.md"):
                with self.assertRaises(JetBrainsMcpError):
                    await client.create_file(bad, "x")
        asyncio.run(run())
        self.assertEqual(sorted(p.name for p in (path / ".codepartner").iterdir()),
                         ["spec-index.json"])  # no temp file left behind
        self.assertEqual((path / "src" / "app.py").read_text(), "print('v1')\n")

    def test_symlinked_artifact_root_cannot_escape(self):
        from agent.spec.filesystem_client import FilesystemFileClient
        from agent.spec.jetbrains_mcp import JetBrainsMcpError
        path = self.project("demo")
        outside = self.root / "elsewhere"
        outside.mkdir()
        (path / "openspec").symlink_to(outside)
        with self.assertRaisesRegex(JetBrainsMcpError, "outside"):
            asyncio.run(FilesystemFileClient(str(path)).create_file("openspec/a.md", "x"))

    def test_bound_conversation_uses_the_filesystem_store_and_native_chat(self):
        from agent.spec.filesystem_client import FilesystemFileClient
        from api import dialog_controller
        self.project("demo")
        workspace.activate("demo")
        store = dialog_controller._artifact_store()
        self.assertIsInstance(store._client, FilesystemFileClient)
        self.assertTrue(dialog_controller._chat_native(True).read_only)
        writer = dialog_controller._chat_native(False)
        self.assertFalse(writer.read_only)
        self.assertIsNone(writer.finish)
        workspace.activate(None)
        with self.assertRaises(dialog_controller._NativeUnavailable):
            dialog_controller._chat_native(True)

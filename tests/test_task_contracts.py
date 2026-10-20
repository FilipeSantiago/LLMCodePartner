"""TaskContract policy, persistence, and telemetry regression tests."""

import json
import sqlite3
import shlex
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from agent.routing.implementation import ImplementationService
from agent.routing.persistence import RoutingPersistence
from agent.spec.models import Task
from agent.task_contracts.contractor import TaskContractor
from agent.task_contracts.guard import ContractExecutionGuard
from agent.task_contracts.models import CoderOutcome
from agent.task_contracts.persistence import TaskContractPersistence
from agent.task_contracts.validation import CoderResultError, parse_coder_result
from mcp_bridge.models import DoneEvent, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec
from telemetry.aggregation import TelemetryAggregation
from telemetry.models import TelemetryContext
from telemetry.recorder import TelemetryRecorder
from telemetry.storage import TelemetryStorage


def _task(complexity="simple"):
    return Task(
        id="WP1-T1", title="Update router", description="Update agent/router.py.",
        complexity=complexity, reasoning="Targeted change",
        context="The Router class lives in agent/router.py.",
        preferred_capability="write", acceptance_criteria=["Router returns the selected model"],
    )


def _specs():
    return [
        ToolSpec("read_file", "read", {"type": "object", "properties": {
            "file_path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["file_path"]}),
        ToolSpec("apply_patch", "patch", {"type": "object", "properties": {
            "input": {"type": "string"}}, "required": ["input"]}),
        ToolSpec("execute_tool", "dispatcher", {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}),
    ]


class TaskContractTests(unittest.TestCase):
    def test_initial_contract_is_bounded_and_uses_stable_criterion_ids(self):
        contract = TaskContractor().initial_contract(_task(), _specs())
        self.assertEqual(contract.primary_files, ("agent/router.py",))
        self.assertEqual(contract.acceptance_criteria[0].id, "AC1")
        self.assertEqual(contract.execution_limits.repeated_calls, 2)
        self.assertIn("read_file", contract.allowed_tools)

    def test_write_task_without_named_files_can_still_write(self):
        target = replace(_task(), description="Locate the router implementation.",
                         context="The project contains a router.")
        contract = TaskContractor().initial_contract(target, _specs())
        self.assertEqual(contract.primary_files, ())
        self.assertIn("apply_patch", contract.allowed_tools)

    def test_coder_result_must_reference_exact_contract(self):
        contract = TaskContractor().initial_contract(_task(), _specs())
        report = json.dumps({
            "schema_version": 2, "task_id": contract.task_id,
            "contract_id": contract.contract_id,
            "contract_revision": contract.revision, "outcome": "SUCCESS",
            "summary": "updated", "changed_files": [], "criteria": [],
            "verified": [], "discoveries": [], "missing_context": [],
            "requested_files": [], "remaining_gaps": [],
            "meaningful_progress": True,
        })
        result = parse_coder_result(report, contract)
        self.assertEqual(result.outcome, CoderOutcome.SUCCESS)
        with self.assertRaises(CoderResultError):
            parse_coder_result(report.replace(contract.contract_id, "wrong"), contract)

    def test_plain_text_criteria_and_paths_are_normalized_not_rejected(self):
        contract = TaskContractor().initial_contract(_task(), _specs())
        report = "Inventory done.\n```json\n" + json.dumps({
            "schema_version": 2, "task_id": contract.task_id,
            "contract_id": contract.contract_id,
            "contract_revision": contract.revision, "outcome": "SUCCESS",
            "summary": "No MLflow integration found.",
            "changed_files": ["docs/inventory.md"],
            "criteria": ["AC1: Inventory completed - no MLflow variables found",
                         {"id": "AC2", "status": "satisfied", "evidence": "read"}],
            "verified": [], "discoveries": [], "missing_context": [],
            "requested_files": [], "remaining_gaps": [],
            "meaningful_progress": True,
        }) + "\n```"
        result = parse_coder_result(report, contract)
        self.assertEqual(result.changed_files, ({"path": "docs/inventory.md"},))
        self.assertEqual(result.criteria[0], {
            "id": "AC1", "status": "unknown",
            "criterion": "AC1: Inventory completed - no MLflow variables found"})
        self.assertEqual(result.criteria[1]["status"], "satisfied")

    def test_handoff_integrity_failures_are_still_rejected(self):
        contract = TaskContractor().initial_contract(_task(), _specs())
        base = {
            "schema_version": 2, "task_id": contract.task_id,
            "contract_id": contract.contract_id,
            "contract_revision": contract.revision, "outcome": "SUCCESS",
            "summary": "done", "changed_files": [], "criteria": [],
        }
        for broken in ({"outcome": "DONE"}, {"criteria": "AC1 met"},
                       {"criteria": [1]}, {"contract_revision": 99}, {"summary": ""}):
            with self.subTest(broken=broken):
                with self.assertRaises(CoderResultError):
                    parse_coder_result(json.dumps({**base, **broken}), contract)

    def test_prompt_spells_out_handoff_element_shapes(self):
        prompt = TaskContractor.prompt(TaskContractor().initial_contract(_task(), _specs()))
        self.assertIn('"criteria":[{"id":"AC1","status":"satisfied|unsatisfied|unknown"', prompt)
        self.assertIn('"changed_files":[{"path":"project/relative/path"', prompt)

    def test_user_confirmed_escalation_creates_child_contract(self):
        first = TaskContractor().initial_contract(_task(), _specs())
        second = TaskContractor().revise_after_escalation(
            first, "Do not inspect .env; use .env.example.", _specs())
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.parent_contract_id, first.contract_id)
        self.assertEqual(second.primary_files, first.primary_files)
        self.assertEqual(second.known_context[-1].source, "user.contract_resolution")


class ContractPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_contract_with_old_limits_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = TaskContractPersistence(Path(directory) / "routing.sqlite3")
            contract = TaskContractor().initial_contract(_task(), _specs())
            value = contract.as_dict()
            value["execution_limits"] = {
                "total_tool_calls": 12, "file_reads": 4, "searches": 2, "test_runs": 2,
                "max_output_chars": 8192, "max_view_lines": 100, "repeated_calls": 2,
                "calls_without_progress": 6}
            with sqlite3.connect(persistence.path) as db:
                db.execute("""INSERT INTO task_contracts
                  (contract_id, parent_contract_id, job_id, position, task_id, revision,
                   contract_json, planning_reason, created_at)
                  VALUES (?, NULL, 'job', 0, ?, 1, ?, 'legacy', 0)""",
                           (contract.contract_id, contract.task_id, json.dumps(value)))
            loaded = await persistence.latest("job", 0)
            self.assertEqual(loaded.contract_id, contract.contract_id)
            self.assertEqual(loaded.execution_limits.repeated_calls, 2)

    async def test_contract_round_trip_preserves_extensible_known_context(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskContractPersistence(Path(directory) / "routing.sqlite3")
            contract = TaskContractor().initial_contract(_task(), _specs(), "Keep API stable")
            await store.save("job", 0, contract)
            restored = await store.latest("job", 0)
            self.assertEqual(restored, contract)
            self.assertEqual(restored.known_context[-1].source, "implementation.guidance")


class TaskerLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_pre_review_escalation_retries_without_a_review(self):
        class Artifacts:
            async def tasks_for_selectors(self, _selectors):
                return [(SimpleNamespace(id="CHG1"), _task())]

            async def task_by_id(self, _task_id):
                return SimpleNamespace(id="CHG1"), _task()

        class Execution:
            def __init__(self):
                self.contracts = []

            async def execute(self, task, change_id, messages, specs,
                              contract=None, execution_guard=None, **kwargs):
                self.contracts.append(contract)
                yield ToolCallEvent("call-retry", "read_file", {"file_path": "agent/router.py"})

        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            execution = Execution()
            service = ImplementationService(
                Artifacts(), execution, persistence, reviewer=SimpleNamespace())
            job = await service.create_job(["WP1-T1"], conversation_id="conversation")
            first = service.contractor.initial_contract(_task(), _specs())
            await service.contract_persistence.save(job.job_id, 0, first)
            await persistence.update_implementation_task(
                job.job_id, 0, "awaiting_contract_resolution",
                error="path outside contract scope: .env")

            events = [event async for event in service.retry_contract(
                job.job_id, 0, [], _specs(), "", {},
                "Ignore .env and use only the contracted file.")]

            self.assertEqual(execution.contracts[0].revision, 2)
            self.assertEqual(execution.contracts[0].parent_contract_id, first.contract_id)
            self.assertEqual(await persistence.review_count(job.job_id, 0), 0)
            self.assertTrue(any("TaskContract revision 2" in event.text
                                for event in events if isinstance(event, TextEvent)))

    async def test_repeated_context_blockers_end_in_structured_escalation(self):
        class Artifacts:
            async def tasks_for_selectors(self, _selectors):
                return [(SimpleNamespace(id="CHG1"), _task())]

            async def task_by_id(self, _task_id):
                return SimpleNamespace(id="CHG1"), _task()

        class Execution:
            def __init__(self):
                self.contracts = []

            async def execute(self, task, change_id, messages, specs,
                              contract=None, execution_guard=None, **kwargs):
                self.contracts.append(contract)
                yield TextEvent(json.dumps({
                    "schema_version": 2, "task_id": task.id,
                    "contract_id": contract.contract_id,
                    "contract_revision": contract.revision,
                    "outcome": "BLOCKED_CONTEXT", "summary": "Need another file",
                    "changed_files": [], "criteria": [], "verified": [],
                    "discoveries": ["Router delegates elsewhere"],
                    "missing_context": ["Delegated implementation"],
                    "requested_files": ["agent/delegate.py"], "remaining_gaps": [],
                    "meaningful_progress": False,
                }))
                yield DoneEvent("endTurn")

        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            execution = Execution()
            service = ImplementationService(
                Artifacts(), execution, persistence, reviewer=SimpleNamespace())
            job = await service.create_job(["WP1-T1"])
            events = [event async for event in service.start(job, [], _specs())]
            self.assertEqual(len(execution.contracts), 4)
            self.assertEqual([item.revision for item in execution.contracts], [1, 2, 3, 4])
            self.assertTrue(any("escalated after 4 bounded executions" in event.text
                                for event in events if isinstance(event, TextEvent)))
            row = await persistence.implementation_task(job.job_id, 0)
            self.assertEqual(row.status, "awaiting_contract_resolution")
            with persistence._database() as db:
                count = db.execute(
                    "SELECT COUNT(*) FROM task_contract_escalations WHERE job_id=?",
                    (job.job_id,)).fetchone()[0]
            self.assertEqual(count, 1)


class ContractGuardTests(unittest.TestCase):
    def test_guard_enforces_scope_without_reshaping_calls_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = TelemetryStorage(Path(directory) / "routing.sqlite3")
            recorder = TelemetryRecorder(storage)
            contract = TaskContractor().initial_contract(_task(), _specs())
            guard = ContractExecutionGuard(
                contract, TelemetryContext(task_id=contract.task_id,
                    contract_id=contract.contract_id, attempt=1), _specs(), recorder)
            arguments = {"file_path": "agent/router.py", "limit": 5000}
            self.assertIsNone(guard.authorize("call-1", "read_file", arguments))
            self.assertEqual(arguments["limit"], 5000)
            self.assertEqual(guard.complete("call-1", "x" * 10_000), "x" * 10_000)
            self.assertIsNone(guard.authorize(
                "call-2", "read_file", {"file_path": "unrelated/module.py", "limit": 10}))
            self.assertEqual(guard.unexpected_reads, ["unrelated/module.py"])
            self.assertIn("secrets file", guard.authorize(
                "call-3", "read_file", {"file_path": ".env"}))

    def test_guardrails_replace_the_file_wall(self):
        specs = _write_specs()
        contract = replace(TaskContractor().initial_contract(_task(), specs),
                           excluded_files=("local/settings.py",))
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), specs)
        self.assertIsNone(guard.authorize("create", "create_new_file", {
            "pathInProject": "mlflow/config.py", "text": "x"}))
        self.assertIsNone(guard.authorize("example", "read_file", {"file_path": ".env.example"}))
        self.assertEqual(guard.unexpected_writes, ["mlflow/config.py"])
        for path, reason in (("openspec/x.md", "protected path"),
                             (".codepartner/a.json", "protected path"),
                             (".env", "protected path"),
                             (".git/config", "could not be identified")):
            fresh = ContractExecutionGuard(
                contract, TelemetryContext(task_id=contract.task_id), specs)
            self.assertIn(reason, fresh.authorize("w", "create_new_file", {
                "pathInProject": path, "text": "x"}), path)
        for path, reason in ((".env", "secrets file"),
                             ("local/settings.py", "excluded by user decision")):
            fresh = ContractExecutionGuard(
                contract, TelemetryContext(task_id=contract.task_id), specs)
            self.assertIn(reason, fresh.authorize("r", "read_file", {"file_path": path}))

    def test_search_results_never_show_secrets_files(self):
        specs = [*_write_specs(), *_discovery_specs()[len(_specs()):]]
        contract = TaskContractor().initial_contract(_config_task(), specs)
        self.assertIn("search_in_files_by_text", contract.allowed_tools)
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), specs)
        self.assertIsNone(guard.authorize("s", "search_in_files_by_text", {"q": "MLFLOW"}))
        text = guard.complete("s", "config/.env:2: MLFLOW_TRACKING_URI=secret\n"
                                   ".env.example:4: MLFLOW_TRACKING_URI=")
        self.assertNotIn("URI=secret", text)
        self.assertIn(".env.example:4", text)

    def test_guard_has_no_call_budget(self):
        specs = [*_specs(), ToolSpec("search_file", "search", {
            "type": "object", "properties": {"q": {"type": "string"}},
            "required": ["q"]})]
        contract = TaskContractor().initial_contract(_task(), specs)
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), specs)
        for index in range(10):
            self.assertIsNone(guard.authorize(
                f"read-{index}", "read_file",
                {"file_path": "agent/router.py", "offset": index + 1}))
            guard.complete(f"read-{index}", "source")
            self.assertIsNone(guard.authorize(
                f"search-{index}", "search_file", {"q": f"term{index}"}))
            guard.complete(f"search-{index}", "[]")
        self.assertIsNone(guard.escalation_reason)

    def test_simple_contract_cannot_use_dispatcher_as_shell_escape(self):
        contract = TaskContractor().initial_contract(_task(), _specs())
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), _specs())
        denial = guard.authorize("call-shell", "execute_tool", {
            "command": "execute_terminal_command --command 'rm -rf build'"})
        self.assertIn("only focused pytest checks", denial)

    def test_simple_contract_allows_a_focused_test(self):
        contract = TaskContractor().initial_contract(_task(), _specs())
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), _specs())
        self.assertIsNone(guard.authorize("call-test", "execute_tool", {
            "command": "execute_terminal_command --command 'uv run pytest tests/test_router.py -q'"}))

    def test_dispatcher_patch_cannot_hide_a_protected_target(self):
        contract = TaskContractor().initial_contract(_task(), _specs())
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), _specs())
        patch = "*** Begin Patch\n*** Update File: other/file.py\n@@\n-x\n+y\n*** End Patch"
        self.assertIsNone(guard.authorize("call-patch", "execute_tool", {
            "command": "apply_patch --input " + shlex.quote(patch)}))
        self.assertEqual(guard.unexpected_writes, ["other/file.py"])
        protected = patch.replace("other/file.py", "openspec/tasks.md")
        self.assertIn("protected path", guard.authorize("call-protected", "execute_tool", {
            "command": "apply_patch --input " + shlex.quote(protected)}))

    def test_identical_calls_are_stopped_deterministically(self):
        contract = TaskContractor().initial_contract(_task(), _specs())
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), _specs())
        args = {"file_path": "agent/router.py", "limit": 10}
        self.assertIsNone(guard.authorize("repeat-1", "read_file", dict(args)))
        guard.complete("repeat-1", "one")
        self.assertIsNone(guard.authorize("repeat-2", "read_file", dict(args)))
        guard.complete("repeat-2", "two")
        self.assertIn("repeated identical", guard.authorize(
            "repeat-3", "read_file", dict(args)))

    def test_telemetry_contains_measurements_but_not_tool_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.sqlite3"
            storage = TelemetryStorage(path)
            contract = TaskContractor().initial_contract(_task(), _specs())
            guard = ContractExecutionGuard(
                contract, TelemetryContext(task_id=contract.task_id,
                    contract_id=contract.contract_id, attempt=1),
                _specs(), TelemetryRecorder(storage))
            args = {"file_path": "agent/router.py", "limit": 20}
            self.assertIsNone(guard.authorize("call", "read_file", args))
            guard.complete("call", "source text that must not be retained")
            guard.finish("SUCCESS", {"input_tokens": 10, "output_tokens": 2})
            summary = TelemetryAggregation(storage).task_summary(contract.task_id)
            self.assertEqual(summary["total_tool_calls"], 1)
            db = sqlite3.connect(path)
            try:
                columns = {row[1] for row in db.execute(
                    "PRAGMA table_info(telemetry_tool_activity)")}
                self.assertNotIn("arguments_json", columns)
                self.assertNotIn("output", columns)
            finally:
                db.close()



def _inventory_task(capability="read"):
    """WP4-T1 as defined in the MusicGeneration OpenSpec change."""
    return Task(
        id="WP4-T1",
        title="Inspect existing .env, .env.example, Docker config, and any current MLflow integration",
        description=("Identify existing environment variable names and conventions used for "
                     "MLflow tracking URI, experiment name, and any credentials, as well as any "
                     "existing MLflow client usage in the codebase."),
        complexity="simple", reasoning="Discover conventions rather than invent them.",
        context="MLflow runs externally in a separate container; configuration must come from existing .env.",
        preferred_capability=capability,
        acceptance_criteria=["A short written inventory lists existing MLflow-related environment "
                             "variable names and any existing MLflow integration code found, or "
                             "explicitly states none exist."],
    )


def _discovery_specs():
    return [
        *_specs(),
        ToolSpec("search_in_files_by_text", "content search", {"type": "object", "properties": {
            "q": {"type": "string"}, "paths": {"type": "array"}}, "required": ["q"]}),
        ToolSpec("list_directory_tree", "tree", {"type": "object", "properties": {
            "directoryPath": {"type": "string"}}, "required": ["directoryPath"]}),
    ]


class DiscoveryContractTests(unittest.IsolatedAsyncioTestCase):
    def test_read_capability_gets_repository_wide_read_only_contract(self):
        contract = TaskContractor().initial_contract(_inventory_task(), _discovery_specs())
        self.assertTrue(contract.read_only)
        self.assertEqual(contract.primary_files, ())
        self.assertIn("search_in_files_by_text", contract.allowed_tools)
        self.assertIn("list_directory_tree", contract.allowed_tools)
        self.assertNotIn("apply_patch", contract.allowed_tools)
        self.assertNotIn("execute_tool", contract.allowed_tools)
        self.assertTrue(any(fact.source == "tasker.discovery" and ".env.example" in fact.fact
                            for fact in contract.known_context))
        prompt = TaskContractor.prompt(contract)
        self.assertIn("read-only investigation", prompt)
        self.assertIn("search file contents across the repository", prompt)

    def test_write_capability_keeps_expected_files_and_can_search(self):
        contract = TaskContractor().initial_contract(
            _inventory_task("write"), _discovery_specs())
        self.assertFalse(contract.read_only)
        self.assertEqual(contract.primary_files, (".env.example",))
        self.assertIn("search_in_files_by_text", contract.allowed_tools)

    def test_discovery_guard_reads_anywhere_but_never_writes(self):
        contract = TaskContractor().initial_contract(_inventory_task(), _discovery_specs())
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), _discovery_specs())
        self.assertIsNone(guard.authorize("read", "read_file", {"file_path": "main.py"}))
        guard.complete("read", "import os")
        self.assertIsNone(guard.authorize(
            "search", "search_in_files_by_text", {"q": "mlflow"}))
        guard.complete("search", '{"items":[]}')
        self.assertIn("outside the contract", guard.authorize("patch", "apply_patch", {
            "input": "*** Begin Patch\n*** Add File: notes.md\n+x\n*** End Patch"}))

    def test_exclusions_are_enforced_and_filtered_from_search_results(self):
        contract = replace(
            TaskContractor().initial_contract(_inventory_task(), _discovery_specs()),
            excluded_files=(".env",))
        guard = ContractExecutionGuard(
            contract, TelemetryContext(task_id=contract.task_id), _discovery_specs())
        self.assertIsNone(guard.authorize(
            "search", "search_in_files_by_text", {"q": "MLFLOW"}))
        filtered = json.loads(guard.complete("search", json.dumps({"items": [
            {"filePath": ".env", "lineText": "MLFLOW_TRACKING_URI=secret"},
            {"filePath": "docker/compose.yaml", "lineText": "MLFLOW_TRACKING_URI"},
        ]})).split("\n")[0])
        self.assertEqual([item["filePath"] for item in filtered["items"]],
                         ["docker/compose.yaml"])
        self.assertIsNone(guard.authorize("search-2", "search_in_files_by_text",
                                          {"q": "tracking"}))
        text = guard.complete("search-2", ".env:3: MLFLOW_TRACKING_URI=secret\n"
                                          "main.py:9: tracking = None")
        self.assertNotIn("URI=secret", text)
        self.assertIn("main.py:9", text)
        self.assertIn("excluded by user decision", guard.authorize(
            "env", "read_file", {"file_path": ".env"}))

    def test_confirmed_retry_keeps_discovery_read_only_and_accumulates_exclusions(self):
        contractor = TaskContractor()
        first = contractor.initial_contract(_inventory_task(), _discovery_specs())
        second = contractor.revise_after_escalation(
            first, "Ignore the .env. I'll fill it after the implementation.",
            _discovery_specs(), add_files=("main.py",), requested_files=("main.py",))
        third = contractor.revise_after_escalation(
            second, "There is no MLflow in this project.", _discovery_specs())
        for contract in (second, third):
            self.assertTrue(contract.read_only)
            self.assertEqual(contract.primary_files, ())
            self.assertEqual(contract.excluded_files, (".env",))
            self.assertIn("search_in_files_by_text", contract.allowed_tools)
            self.assertNotIn("apply_patch", contract.allowed_tools)
        self.assertEqual(second.known_context[-1].attributes["excluded_files"], [".env"])
        self.assertIn("There is no MLflow", third.known_context[-1].fact)

    def test_coder_revision_keeps_discovery_read_only(self):
        contractor = TaskContractor()
        contract = contractor.initial_contract(_inventory_task(), _discovery_specs())
        result = parse_coder_result(json.dumps({
            "schema_version": 2, "task_id": contract.task_id,
            "contract_id": contract.contract_id, "contract_revision": contract.revision,
            "outcome": "BLOCKED_CONTEXT", "summary": "need more",
            "requested_files": ["Dockerfile"],
        }), contract)
        revised = contractor.revise_after_coder(contract, result, _discovery_specs())
        self.assertTrue(revised.read_only)
        self.assertEqual(revised.primary_files, ())
        self.assertNotIn("apply_patch", revised.allowed_tools)

    async def test_contract_without_new_fields_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = TaskContractPersistence(Path(directory) / "routing.sqlite3")
            contract = TaskContractor().initial_contract(_task(), _specs())
            value = contract.as_dict()
            value.pop("excluded_files")
            value.pop("read_only")
            with sqlite3.connect(persistence.path) as db:
                db.execute("""INSERT INTO task_contracts
                  (contract_id, parent_contract_id, job_id, position, task_id, revision,
                   contract_json, planning_reason, created_at)
                  VALUES (?, NULL, 'job', 0, ?, 1, ?, 'legacy', 0)""",
                           (contract.contract_id, contract.task_id, json.dumps(value)))
            loaded = await persistence.latest("job", 0)
            self.assertEqual(loaded.excluded_files, ())
            self.assertFalse(loaded.read_only)



def _config_task():
    """WP4-T2 as defined in the MusicGeneration OpenSpec change: no files named."""
    return Task(
        id="WP4-T2", title="Implement MLflowConfig.from_env()",
        description=("Create a configuration object that loads tracking URI and other required "
                     "MLflow settings from the discovered environment variable names."),
        complexity="simple", reasoning="Centralize env access.", context="",
        preferred_capability="write",
        acceptance_criteria=["MLflowConfig.from_env() raises a clear error when a required "
                             "variable is missing."],
    )


def _write_specs():
    return [*_specs(), ToolSpec("create_new_file", "create", {
        "type": "object", "properties": {
            "pathInProject": {"type": "string"}, "text": {"type": "string"}},
        "required": ["pathInProject", "text"]})]


def _rename_spec():
    return ToolSpec("rename_refactoring", "rename", {"type": "object", "properties": {
        "pathInProject": {"type": "string"}, "symbolName": {"type": "string"},
        "newName": {"type": "string"}},
        "required": ["pathInProject", "symbolName", "newName"]})


def _handoff(contract, outcome="BLOCKED_CONTEXT", **values):
    return json.dumps({
        "schema_version": 2, "task_id": contract.task_id,
        "contract_id": contract.contract_id, "contract_revision": contract.revision,
        "outcome": outcome, "summary": "needs a new module", **values})


class ScopeNegotiationTests(unittest.IsolatedAsyncioTestCase):
    def test_tasker_grants_rule_compliant_request(self):
        contractor = TaskContractor()
        first = contractor.initial_contract(_config_task(), _write_specs())
        self.assertEqual(first.primary_files, ())
        self.assertIn("create_new_file", first.allowed_tools)
        self.assertIn("Expected files (guidance, not a limit)", TaskContractor.prompt(first))
        result = parse_coder_result(_handoff(first, scope_requests=[
            {"path": "mlflow/config.py", "operation": "create", "reason": "new module"}]),
            first)
        decision = contractor.grant_scope(first, result)
        self.assertEqual(decision.granted, ("mlflow/config.py",))
        second = contractor.revise_after_coder(first, result, _write_specs(), decision)
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.primary_files, ("mlflow/config.py",))
        self.assertIn("create_new_file", second.allowed_tools)
        self.assertEqual(second.planning_reason, "tasker granted coder scope request")
        self.assertEqual(second.known_context[-1].source, "tasker.scope_grant")

    def test_tasker_denies_requests_that_break_rules(self):
        contractor = TaskContractor()
        contract = replace(contractor.initial_contract(_config_task(), _write_specs()),
                           excluded_files=("secrets/local.py",))
        requests = [{"path": path, "operation": "create"} for path in (
            ".env", "config/.env.local", "openspec/x.md", "../outside.py",
            "secrets/local.py", "All Python source files")]
        requests += [{"path": f"pkg/module{index}.py", "operation": "create"}
                     for index in range(6)]
        decision = contractor.grant_scope(
            contract, parse_coder_result(_handoff(contract, scope_requests=requests),
                                         contract))
        reasons = dict(decision.denied)
        self.assertIn("secrets", reasons[".env"])
        self.assertIn("secrets", reasons["config/.env.local"])
        self.assertIn("protected", reasons["openspec/x.md"])
        self.assertIn("outside the project", reasons["../outside.py"])
        self.assertIn("excluded", reasons["secrets/local.py"])
        self.assertIn("not a project-relative", reasons["All Python source files"])
        self.assertEqual(len(decision.granted), 5)
        self.assertIn("more than 5", reasons["pkg/module5.py"])

    def test_discovery_contract_never_gains_writes(self):
        contractor = TaskContractor()
        contract = contractor.initial_contract(_inventory_task(), _discovery_specs())
        result = parse_coder_result(_handoff(contract, scope_requests=[
            {"path": "notes/inventory.md", "operation": "create"}]), contract)
        self.assertEqual(contractor.grant_scope(contract, result).granted, ())
        revised = contractor.revise_after_coder(contract, result, _discovery_specs())
        self.assertEqual(revised.primary_files, ())
        self.assertNotIn("apply_patch", revised.allowed_tools)

    def test_free_text_requested_files_are_reduced_to_paths(self):
        contract = TaskContractor().initial_contract(_config_task(), _write_specs())
        result = parse_coder_result(_handoff(contract, requested_files=[
            "mlflow/__init__.py (new module)", "All Python source files"]), contract)
        self.assertEqual(result.requested_files,
                         ("mlflow/__init__.py", "All Python source files"))

    def test_scope_requests_must_be_structured(self):
        contract = TaskContractor().initial_contract(_config_task(), _write_specs())
        with self.assertRaises(CoderResultError):
            parse_coder_result(_handoff(contract, scope_requests=[
                {"path": "mlflow/config.py", "operation": "delete"}]), contract)

    async def _run(self, outcomes):
        class Artifacts:
            async def tasks_for_selectors(self, _selectors):
                return [(SimpleNamespace(id="CHG1"), _config_task())]

            async def task_by_id(self, _task_id):
                return SimpleNamespace(id="CHG1"), _config_task()

        class Execution:
            def __init__(self):
                self.contracts = []

            async def execute(self, task, change_id, messages, specs,
                              contract=None, execution_guard=None, **kwargs):
                self.contracts.append(contract)
                outcome, values, denied_call = outcomes[len(self.contracts) - 1]
                if denied_call:
                    execution_guard.authorize("deny", "rename_refactoring", {
                        "pathInProject": "main.py", "symbolName": "a", "newName": "b"})
                yield TextEvent(_handoff(contract, outcome, **values))
                yield DoneEvent("endTurn")

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        persistence = RoutingPersistence(Path(directory.name) / "routing.sqlite3")
        execution = Execution()
        service = ImplementationService(
            Artifacts(), execution, persistence, reviewer=SimpleNamespace())
        job = await service.create_job(["WP4-T2"])
        specs = [*_write_specs(), _rename_spec()]
        events = [event async for event in service.start(job, [], specs)]
        text = "\n".join(event.text for event in events if isinstance(event, TextEvent))
        return service, persistence, job, execution, text

    async def test_denied_tool_is_negotiated_without_user_escalation(self):
        request = {"scope_requests": [
            {"path": "mlflow/config.py", "operation": "create", "reason": "new module"}]}
        stop = {"scope_requests": []}
        service, persistence, job, execution, text = await self._run([
            ("SCOPE_TOO_LARGE", request, True),
            ("BLOCKED_CONTEXT", stop, False),
            ("BLOCKED_CONTEXT", stop, False),
            ("BLOCKED_CONTEXT", stop, False),
        ])
        self.assertNotIn("rename_refactoring", execution.contracts[0].allowed_tools)
        second = execution.contracts[1]
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.complexity, "medium")
        self.assertIn("rename_refactoring", second.allowed_tools)
        self.assertEqual(second.primary_files, ("mlflow/config.py",))
        self.assertIn("granted mlflow/config.py", text)
        self.assertNotIn("is outside the contract.", text)
        with persistence._database() as db:
            action = db.execute("""SELECT details_json FROM implementation_resolution_actions
              WHERE job_id=? AND action='tasker_scope_grant'""", (job.job_id,)).fetchone()
        self.assertEqual(json.loads(action[0])["granted"], ["mlflow/config.py"])

    async def test_success_claim_after_denial_still_escalates(self):
        _, persistence, job, execution, text = await self._run([("SUCCESS", {}, True)])
        self.assertEqual(len(execution.contracts), 1)
        self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                         "awaiting_contract_resolution")
        self.assertIn("outside the contract", text)

    async def test_fully_denied_request_escalates_with_reasons(self):
        _, persistence, job, execution, text = await self._run([(
            "BLOCKED_CONTEXT", {"scope_requests": [
                {"path": "openspec/notes.md", "operation": "create"}]}, True)])
        self.assertEqual(len(execution.contracts), 1)
        self.assertEqual((await persistence.implementation_task(job.job_id, 0)).status,
                         "awaiting_contract_resolution")
        self.assertIn("denied openspec/notes.md (protected path", text)
        self.assertIn("expand scope", text)



class DenialParsingTests(unittest.TestCase):
    def test_requested_paths_reads_current_and_legacy_denials(self):
        from agent.task_contracts.contractor import requested_paths
        for reason in ("secrets file: .env", "protected path: .env",
                       "path outside contract scope: .env",
                       "path excluded by user decision: .env"):
            self.assertEqual(requested_paths(reason), (".env",), reason)
        self.assertEqual(requested_paths("repeated identical tool-call limit reached"), ())


class ScopeDriftTests(unittest.IsolatedAsyncioTestCase):
    async def test_unexpected_journaled_writes_are_audited_for_review(self):
        class Artifacts:
            async def tasks_for_selectors(self, _selectors):
                return [(SimpleNamespace(id="CHG1"), _task())]

        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            service = ImplementationService(
                Artifacts(), SimpleNamespace(persistence=persistence), persistence,
                reviewer=SimpleNamespace())
            job = await service.create_job(["WP1-T1"])
            contract = service.contractor.initial_contract(_task(), _specs())
            await service.contract_persistence.save(job.job_id, 0, contract)
            for path in ("agent/router.py", "agent/helpers.py"):
                await persistence.record_rollback_file(
                    job.job_id, 0, path, True, expected_hash="hash")
            rows = await persistence.implementation_job_tasks(job.job_id)

            drift = await service._scope_drift(job.job_id, rows, _specs(), {})

            self.assertEqual(drift, {"WP1-T1": ["agent/helpers.py"]})
            with persistence._database() as db:
                details = json.loads(db.execute("""SELECT details_json
                  FROM implementation_resolution_actions
                  WHERE job_id=? AND action='scope_drift'""", (job.job_id,)).fetchone()[0])
            self.assertEqual(details["expected_files"], ["agent/router.py"])
            self.assertEqual(details["unexpected_files"], ["agent/helpers.py"])

    def test_attempt_summary_records_unexpected_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = TelemetryStorage(Path(directory) / "routing.sqlite3")
            contract = TaskContractor().initial_contract(_task(), _write_specs())
            guard = ContractExecutionGuard(
                contract, TelemetryContext(task_id=contract.task_id, attempt=1),
                _write_specs(), TelemetryRecorder(storage))
            self.assertIsNone(guard.authorize("c", "create_new_file", {
                "pathInProject": "agent/helpers.py", "text": "x"}))
            guard.complete("c", "created")
            guard.finish("SUCCESS")
            with sqlite3.connect(storage.path) as db:
                extra = json.loads(db.execute(
                    "SELECT extra_json FROM telemetry_attempt_summaries").fetchone()[0])
            self.assertEqual(extra["unexpected_writes"], ["agent/helpers.py"])


if __name__ == "__main__":
    unittest.main()

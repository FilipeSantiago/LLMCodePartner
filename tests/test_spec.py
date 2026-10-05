"""Tests for additive OpenSpec generation and scoped task refinement."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException

from agent.spec.artifact_store import (
    CATALOG_PATH,
    DASHBOARD_PATH,
    ArtifactStoreError,
    BridgeRequiredMcpClient,
    ArtifactLookupIndeterminate,
    OpenSpecArtifactStore,
)
from agent.spec.bridge_client import BridgeMcpFileClient
from agent.spec.command import (
    SpecCommandError,
    SpecCommandHandler,
    extract_object_id,
    extract_help_response,
    extract_scoped_request,
    extract_spec_request,
    extract_update_request,
    is_spec_command,
    is_explain_command,
    is_update_command,
)
from agent.spec.jetbrains_mcp import JetBrainsMcpClient, JetBrainsMcpError
from agent.spec.models import (
    Capability,
    CompletionReview,
    Decision,
    Design,
    ModelIdentity,
    OpenSpecChange,
    Proposal,
    Requirement,
    Scenario,
    Task,
    ValidationContract,
    WorkPackage,
)
from agent.spec.openspec_adapter import (
    CliOpenSpecAdapter, MalformedJsonError, OpenSpecAdapter, OpenSpecError,
)
from agent.routing.persistence import RoutingPersistence
from agent.spec.command import help_text, is_discard_command
from agent.spec.modes import DEFAULT_MODEL, request_shape, route_message, split_scope
from agent.spec.refinement import RefinementError, RefinementService
from api import dialog_controller
from api.model_controller import list_models
from model.chat import ChatCompletionRequest, ChatMessage


def draft(title: str = "Add authentication") -> OpenSpecChange:
    return OpenSpecChange(
        id="",
        number=0,
        slug="",
        title=title,
        summary="Add secure authentication.",
        proposal=Proposal(
            why="Users need a secure way to authenticate.",
            what_changes=["Add password authentication.", "Create authenticated sessions."],
            impact=["Authentication API", "Session storage"],
        ),
        design=Design(
            context="The application currently has no authentication boundary.",
            goals=["Authenticate users securely."],
            non_goals=["External identity providers."],
            decisions=[Decision("Signed sessions", "They support stateless validation.")],
            risks=["Credential enumeration must be prevented."],
        ),
        capabilities=[
            Capability(
                path="authentication",
                purpose=(
                    "Provide secure credential validation and authenticated sessions "
                    "for application users."
                ),
                requirements=[
                    Requirement(
                        name="Password authentication",
                        statement="The system SHALL authenticate valid user credentials.",
                        scenarios=[
                            Scenario(
                                name="Valid credentials",
                                when="a user submits valid credentials",
                                then="an authenticated session is created",
                            )
                        ],
                    )
                ],
            )
        ],
        work_packages=[
            WorkPackage(
                id="",
                title="Credential validation",
                kind="user-story",
                objective="Users can prove their identity with credentials.",
                acceptance_criteria=["Valid credentials authenticate the user."],
                requirement_refs=["authentication#Password authentication"],
                tasks=[
                    Task(
                        id="",
                        title="Create authentication service",
                        description="Add credential validation interfaces.",
                        complexity="medium",
                        reasoning="The service crosses application boundaries.",
                        context="No authentication service exists.",
                        preferred_capability="write",
                    ),
                    Task(
                        id="",
                        title="Test invalid credentials",
                        description="Cover rejected credential combinations.",
                        complexity="simple",
                        reasoning="Focused behavior tests are sufficient.",
                        context="",
                        preferred_capability="test",
                    ),
                ],
            ),
            WorkPackage(
                id="",
                title="Authenticated sessions",
                kind="technical-outcome",
                objective="Successful authentication creates a durable session.",
                acceptance_criteria=["A valid session can be verified."],
                requirement_refs=["authentication#Password authentication"],
                tasks=[
                    Task(
                        id="",
                        title="Create session tokens",
                        description="Issue signed session tokens.",
                        complexity="medium",
                        reasoning="Token lifecycle and signing require coordination.",
                        context="",
                        preferred_capability="write",
                    )
                ],
            ),
        ],
    )


class FakeAdapter(OpenSpecAdapter):
    def __init__(self):
        self.generated = []
        self.refined = []

    async def generate_change(self, request: str) -> OpenSpecChange:
        self.generated.append(request)
        return draft(request.title())

    async def refine_tasks(
            self, change: OpenSpecChange, task_ids: list[str], instruction: str
    ) -> dict[str, Task]:
        self.refined.append((change.id, task_ids, instruction))
        selected = {
            task.id: task
            for package in change.work_packages
            for task in package.tasks
            if task.id in task_ids
        }
        return {
            task_id: replace(task, title=f"Refined {task.title}")
            for task_id, task in selected.items()
        }


class FakeMcpClient:
    def __init__(self):
        self.files: dict[str, str] = {}
        self.writes: list[tuple[str, str, bool]] = []

    async def create_file(self, path: str, content: str, overwrite: bool = True):
        if path in self.files and not overwrite:
            raise JetBrainsMcpError("file exists")
        self.files[path] = content
        self.writes.append((path, content, overwrite))

    async def read_file(self, path: str) -> str:
        if path not in self.files:
            raise JetBrainsMcpError("file not found")
        return self.files[path]

    async def file_exists(self, path: str) -> bool:
        return path in self.files


class FailingMcpClient(FakeMcpClient):
    async def create_file(self, path: str, content: str, overwrite: bool = True):
        raise JetBrainsMcpError("PyCharm is closed")


class IndeterminateLookupMcpClient(FakeMcpClient):
    async def file_exists(self, path: str) -> bool:
        raise ArtifactLookupIndeterminate("deterministic function was already called with this args")


class AsyncContext:
    def __init__(self, value=None, error: Exception | None = None):
        self.value = value
        self.error = error

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class CommandParsing(unittest.TestCase):
    def test_detects_spec_and_update_without_prefix_collisions(self):
        self.assertTrue(is_spec_command("/spec add authentication"))
        self.assertTrue(is_update_command("/UPDATE WP1-T1: simplify it"))
        self.assertFalse(is_spec_command("/speculative behavior"))
        self.assertFalse(is_update_command("please update WP1-T1"))

    def test_extracts_spec_request(self):
        self.assertEqual(
            extract_spec_request('/spec "implement authentication"'),
            "implement authentication",
        )

    def test_extracts_canonical_and_forgiving_update_targets(self):
        self.assertEqual(
            extract_update_request(
                "/update about US1-TASK1 and wp1-t2, use event callbacks"
            ),
            (["WP1-T1", "WP1-T2"], "use event callbacks"),
        )
        self.assertEqual(
            extract_update_request("/update WP3-T4 WP3-T5: split validation"),
            (["WP3-T4", "WP3-T5"], "split validation"),
        )

    def test_rejects_update_without_targets_or_instruction(self):
        with self.assertRaises(SpecCommandError):
            extract_update_request("/update make it better")
        with self.assertRaises(SpecCommandError):
            extract_update_request("/update WP1-T1")

    def test_help_and_manual_are_registry_backed(self):
        self.assertIn("/implement", extract_help_response("/help"))
        self.assertIn("sequential", extract_help_response("/man implement"))
        with self.assertRaises(SpecCommandError):
            extract_help_response("/man unknown")

    def test_review_and_rework_parse_work_packages_and_feedback(self):
        self.assertEqual(
            extract_scoped_request("/review US1 WP2-TASK3: verify errors", "review"),
            (["WP1", "WP2-T3"], "verify errors"),
        )
        self.assertEqual(
            extract_scoped_request("/rework WP1-T1: add encryption", "rework", True),
            (["WP1-T1"], "add encryption"),
        )
        with self.assertRaises(SpecCommandError):
            extract_scoped_request("/rework WP1-T1", "rework", True)


class ArtifactSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_created_change_has_explicit_epic_and_validation_contract(self):
        client = FakeMcpClient()
        store = OpenSpecArtifactStore(client)
        planned = draft()
        planned = replace(
            planned,
            work_packages=[replace(
                planned.work_packages[0],
                validation=ValidationContract(
                    required=["Exercise the authentication path."],
                    preferred=["Run relevant existing tests."],
                    optional=["Run broader regression tests."],
                    not_required=["Run the entire repository suite."],
                ),
            )],
        )
        created = await store.create_change(planned, "authentication")
        self.assertEqual(created.epic.id, "EPIC1")
        self.assertEqual(created.epic.work_package_ids, ["WP1"])
        self.assertEqual(
            created.work_packages[0].validation.required,
            ["Exercise the authentication path."],
        )

    async def test_explain_is_read_only(self):
        client = FakeMcpClient()
        store = OpenSpecArtifactStore(client)
        await store.create_change(draft(), "authentication")
        writes_before = list(client.writes)
        handler = SpecCommandHandler(FakeAdapter(), store)
        result = await handler.handle([{"role": "user", "content": "/explain WP1"}])
        self.assertIn("WP1 — Credential validation", result.content)
        self.assertIn("Review boundary", result.content)
        self.assertEqual(client.writes, writes_before)

    def test_object_command_parsing(self):
        self.assertTrue(is_explain_command("/explain EPIC1"))
        self.assertEqual(extract_object_id("/explain epic 3", "explain"), "EPIC3")
        self.assertEqual(extract_object_id("/refine WP2-T4", "refine"), "WP2-T4")

    async def test_reconciliation_preserves_ids_and_history(self):
        store = OpenSpecArtifactStore(FakeMcpClient())
        canonical = await store.create_change(
            replace(draft(), work_packages=[draft().work_packages[0]]), "authentication")
        package = canonical.work_packages[0]
        first = replace(package.tasks[0], completed=True, status="completed")
        canonical = replace(
            canonical,
            work_packages=[replace(package, tasks=[first, package.tasks[1]])],
        )
        changed = replace(first, description="Use a provider-neutral credential boundary.")
        new_task = replace(first, id="", title="Wire credential boundary", completed=False,
                           status="pending", completion_review=None)
        proposed = replace(
            canonical,
            work_packages=[replace(package, tasks=[changed, new_task])],
        )
        reconciled = RefinementService._reconcile(canonical, proposed)
        tasks = {task.id: task for task in reconciled.work_packages[0].tasks}
        self.assertEqual(tasks["WP1-T1"].status, "needs_reexecution")
        self.assertEqual(tasks["WP1-T2"].status, "cancelled")
        self.assertIn("WP1-T3", tasks)
        self.assertEqual(tasks["WP1-T3"].status, "pending")

    async def test_work_package_selection_skips_completed_tasks_and_completion_rewrites_artifacts(self):
        client = FakeMcpClient()
        store = OpenSpecArtifactStore(client)
        change = await store.create_change(draft(), "authentication")
        selected = await store.tasks_for_selectors(["WP1", "WP1-T2"])
        self.assertEqual([task.id for _, task in selected], ["WP1-T1", "WP1-T2"])

        review = CompletionReview(
            coder=ModelIdentity("claude-opus", "claude", "opus", None, "Claude Opus"),
            reviewer=ModelIdentity("codex-sol-high", "codex", "gpt-6.1-sol", "high",
                                   "Codex GPT-6.1 Sol (high effort)"),
            reason="The implementation meets the requested interface.",
            evidence=["Reviewed the implementation and tests."],
        )
        await store.mark_task_completed("WP1-T1", review)
        selected_after_completion = await store.tasks_for_selectors(["WP1"])
        self.assertEqual([task.id for _, task in selected_after_completion], ["WP1-T2"])
        self.assertIn("- [x] WP1-T1", client.files[f"{change.path}/tasks.md"])
        self.assertIn("Coded by: Claude Opus", client.files[f"{change.path}/tasks.md"])
        self.assertIn("Reviewed by: Codex GPT-6.1 Sol (high effort)",
                      client.files[f"{change.path}/tasks.md"])
        self.assertIn("Review decision: Approved — The implementation meets the requested interface.",
                      client.files[f"{change.path}/tasks.md"])
        self.assertIn("Coded by Claude Opus; reviewed by Codex GPT-6.1 Sol (high effort)",
                      client.files[DASHBOARD_PATH])
        catalog = json.loads(client.files[CATALOG_PATH])
        self.assertEqual(catalog["changes"][0]["work_packages"][0]["tasks"][0]
                         ["completion_review"]["reviewer"]["route_name"], "codex-sol-high")
        selected_change = await store.change_for_tasks(["WP1-T1"])
        await store.replace_tasks(selected_change, {})
        self.assertIn("Coded by: Claude Opus", client.files[f"{change.path}/tasks.md"])
        original_task = selected_change.work_packages[0].tasks[0]
        refinement = json.dumps([{
            "id": original_task.id,
            "title": "Refined title",
            "description": original_task.description,
            "acceptance_criteria": original_task.acceptance_criteria or [original_task.description],
            "complexity": original_task.complexity,
            "reasoning": original_task.reasoning,
            "context": original_task.context,
            "preferred_capability": original_task.preferred_capability,
        }])
        refined = CliOpenSpecAdapter._parse_refined_tasks(
            refinement, {original_task.id: original_task})
        self.assertEqual(refined[original_task.id].completion_review, review)
        adapter = CliOpenSpecAdapter()
        with mock.patch.object(adapter, "_complete", new=mock.AsyncMock(return_value=refinement)):
            updated_tasks = await adapter.refine_tasks(
                selected_change, [original_task.id], "Refine the title")
        self.assertEqual(updated_tasks[original_task.id].completion_review, review)


class JetBrainsClient(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _transport(result):
        http_client = object()
        read_stream = object()
        write_stream = object()
        session = mock.AsyncMock()
        session.call_tool.return_value = result
        patches = (
            mock.patch(
                "agent.spec.jetbrains_mcp.httpx.AsyncClient",
                return_value=AsyncContext(http_client),
            ),
            mock.patch(
                "agent.spec.jetbrains_mcp.streamable_http_client",
                return_value=AsyncContext((read_stream, write_stream, lambda: None)),
            ),
            mock.patch(
                "agent.spec.jetbrains_mcp.ClientSession",
                return_value=AsyncContext(session),
            ),
        )
        return http_client, session, patches

    async def test_initializes_session_sends_header_and_controls_overwrite(self):
        result = SimpleNamespace(content=[], isError=False, structuredContent=None)
        http_client, session, patches = self._transport(result)
        client = JetBrainsMcpClient(
            "http://127.0.0.1:64462/stream", "/workspace/project", timeout_seconds=5
        )
        with patches[0] as httpx_type, patches[1] as stream_type, patches[2]:
            await client.create_file("openspec/change.md", "content", overwrite=False)
        session.initialize.assert_awaited_once()
        session.call_tool.assert_awaited_once_with(
            "create_new_file",
            {
                "pathInProject": "openspec/change.md",
                "text": "content",
                "overwrite": False,
            },
        )
        self.assertEqual(
            httpx_type.call_args.kwargs["headers"],
            {"IJ_MCP_SERVER_PROJECT_PATH": "/workspace/project"},
        )
        self.assertIs(stream_type.call_args.kwargs["http_client"], http_client)

    async def test_file_exists_uses_structured_search_results(self):
        result = SimpleNamespace(
            content=[],
            isError=False,
            structuredContent={"items": [{"filePath": CATALOG_PATH}]},
        )
        _, session, patches = self._transport(result)
        client = JetBrainsMcpClient("http://mcp/stream", "/workspace/project")
        with patches[0], patches[1], patches[2]:
            self.assertTrue(await client.file_exists(CATALOG_PATH))
        session.call_tool.assert_awaited_once_with(
            "search_file", {"q": CATALOG_PATH, "limit": 2}
        )

    async def test_configuration_is_validated_lazily(self):
        with mock.patch.dict(
            os.environ,
            {"JETBRAINS_MCP_URL": "", "JETBRAINS_MCP_PROJECT_PATH": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(JetBrainsMcpError, "JETBRAINS_MCP_URL"):
                await JetBrainsMcpClient().file_exists(CATALOG_PATH)


class ArtifactPersistence(unittest.IsolatedAsyncioTestCase):
    async def test_spec_writes_every_openspec_artifact_and_dashboard(self):
        client = FakeMcpClient()
        change = await OpenSpecArtifactStore(client).create_change(
            draft(), "add-authentication"
        )
        self.assertEqual(change.id, "CHG1")
        self.assertEqual(
            [package.id for package in change.work_packages], ["WP1", "WP2"]
        )
        self.assertEqual(
            [task.id for task in change.work_packages[0].tasks],
            ["WP1-T1", "WP1-T2"],
        )
        expected = {
            f"{change.path}/.openspec.yaml",
            f"{change.path}/proposal.md",
            f"{change.path}/design.md",
            f"{change.path}/tasks.md",
            f"{change.path}/specs/authentication/spec.md",
            CATALOG_PATH,
            DASHBOARD_PATH,
        }
        self.assertTrue(expected.issubset(client.files))
        self.assertIn("WP1-T1", client.files[f"{change.path}/tasks.md"])
        self.assertIn("WP2-T1", client.files[DASHBOARD_PATH])

    async def test_later_specs_are_additive_and_continue_global_package_ids(self):
        client = FakeMcpClient()
        store = OpenSpecArtifactStore(client)
        first = await store.create_change(draft(), "authentication")
        second = await store.create_change(draft("Add recovery"), "recovery")
        self.assertEqual(first.id, "CHG1")
        self.assertEqual(second.id, "CHG2")
        self.assertEqual(
            [package.id for package in second.work_packages], ["WP3", "WP4"]
        )
        self.assertIn(f"{first.path}/proposal.md", client.files)
        self.assertIn(f"{second.path}/proposal.md", client.files)
        catalog = json.loads(client.files[CATALOG_PATH])
        self.assertEqual(len(catalog["changes"]), 2)

    async def test_update_changes_only_selected_tasks(self):
        client = FakeMcpClient()
        store = OpenSpecArtifactStore(client)
        original = await store.create_change(draft(), "authentication")
        selected = await store.change_for_tasks(["WP1-T1", "WP1-T2"])
        replacements = {
            "WP1-T1": replace(selected.work_packages[0].tasks[0], title="New service"),
            "WP1-T2": replace(selected.work_packages[0].tasks[1], title="New tests"),
        }
        updated = await store.replace_tasks(selected, replacements)
        self.assertEqual(updated.work_packages[0].tasks[0].title, "New service")
        self.assertEqual(
            updated.work_packages[1].tasks[0],
            original.work_packages[1].tasks[0],
        )
        self.assertEqual(
            client.files[f"{updated.path}/proposal.md"],
            client.files[f"{original.path}/proposal.md"],
        )

    async def test_unknown_or_cross_change_update_is_rejected(self):
        client = FakeMcpClient()
        store = OpenSpecArtifactStore(client)
        await store.create_change(draft(), "authentication")
        await store.create_change(draft("Add recovery"), "recovery")
        with self.assertRaisesRegex(ArtifactStoreError, "unknown task IDs"):
            await store.change_for_tasks(["WP99-T1"])
        with self.assertRaisesRegex(ArtifactStoreError, "only one change"):
            await store.change_for_tasks(["WP1-T1", "WP3-T1"])

    async def test_connection_error_is_focused(self):
        with self.assertRaisesRegex(ArtifactStoreError, "PyCharm is closed"):
            await OpenSpecArtifactStore(FailingMcpClient()).create_change(
                draft(), "authentication"
            )

    async def test_read_response_decode_error_does_not_claim_the_index_is_invalid(self):
        client = FakeMcpClient()
        client.files[CATALOG_PATH] = "not a transport response"
        with self.assertRaisesRegex(ArtifactStoreError, "could not parse the read response"):
            await OpenSpecArtifactStore(client)._load_catalog()

    async def test_indeterminate_lookup_attempts_the_bridge_reader(self):
        client = IndeterminateLookupMcpClient()
        client.files[CATALOG_PATH] = json.dumps({
            "version": 1, "next_change": 2, "next_work_package": 2, "changes": [],
        })
        catalog = await OpenSpecArtifactStore(client)._load_catalog(required=True)
        self.assertEqual(catalog["next_change"], 2)


class BridgeResponseExtraction(unittest.TestCase):
    def test_unwraps_openai_text_blocks_then_numbered_lines(self):
        raw = json.dumps({"content": [{"type": "text", "text": 'L1: {"version": 1}'}]})
        text, wrapper = BridgeMcpFileClient._text_from_tool_response(raw)
        self.assertEqual(wrapper, "content_blocks")
        self.assertEqual(text, 'L1: {"version": 1}')

    def test_leaves_a_raw_json_catalog_untouched(self):
        raw = '{"version": 1, "changes": []}'
        text, wrapper = BridgeMcpFileClient._text_from_tool_response(raw)
        self.assertEqual((text, wrapper), (raw, "raw_json"))

    def test_strips_numbered_body_after_client_status_preamble(self):
        text, numbered, preamble = BridgeMcpFileClient._strip_numbered_lines(
            "Read result:\nL1: {\nL2:   \"version\": 1\nL3: }"
        )
        self.assertEqual(text, '{\n  "version": 1\n}')
        self.assertTrue(numbered)
        self.assertEqual(preamble, 1)

    def test_does_not_treat_mixed_plain_text_as_numbered_file_content(self):
        raw = "Read result:\nL1: {\nnot a numbered record"
        self.assertEqual(
            BridgeMcpFileClient._strip_numbered_lines(raw), (raw, False, 0)
        )


class AdapterValidation(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def payload() -> dict[str, Any]:
        return {
            "title": "Add authentication",
            "summary": "Add secure user authentication.",
            "proposal": {
                "why": "Users need authenticated access.",
                "what_changes": ["Add credential validation."],
                "impact": ["Authentication API"],
            },
            "design": {
                "context": "Authentication is not currently available.",
                "goals": ["Authenticate users."],
                "non_goals": [],
                "decisions": [{
                    "title": "Signed sessions",
                    "rationale": "They permit stateless verification.",
                }],
                "risks": [],
            },
            "capabilities": [{
                "path": "authentication",
                "purpose": (
                    "Provide secure credential validation and authenticated sessions "
                    "for application users."
                ),
                "requirements": [{
                    "name": "Password authentication",
                    "statement": "The system SHALL authenticate valid credentials.",
                    "scenarios": [{
                        "name": "Valid login",
                        "when": "valid credentials are submitted",
                        "then": "a session is created",
                    }],
                }],
            }],
            "work_packages": [{
                "title": "Credential validation",
                "kind": "user-story",
                "objective": "Users can authenticate.",
                "acceptance_criteria": ["Valid users receive sessions."],
                "requirement_refs": ["authentication#Password authentication"],
                "tasks": [{
                    "title": "Add validation",
                    "description": "Validate submitted credentials.",
                    "acceptance_criteria": [
                        "The existing validation path and its dependencies are identified."
                    ],
                    "complexity": "simple",
                    "reasoning": "The behavior is focused.",
                    "context": "",
                    "preferred_capability": "write",
                }],
            }],
        }

    @staticmethod
    def adapter(response):
        provider = mock.AsyncMock()
        provider.complete.return_value = (response, "endTurn")
        adapter = CliOpenSpecAdapter(lambda: provider)
        adapter._load_templates = mock.AsyncMock(return_value={
            "proposal": "# Proposal",
            "specs": "# Spec Delta",
            "design": "# Design",
            "tasks": "# Tasks",
        })
        return adapter, provider

    async def test_generates_grouped_change_without_model_ids(self):
        adapter, provider = self.adapter(json.dumps(self.payload()))
        result = await adapter.generate_change("add authentication")
        self.assertEqual(result.title, "Add authentication")
        self.assertEqual(result.work_packages[0].id, "")
        self.assertEqual(result.work_packages[0].tasks[0].id, "")
        provider.complete.assert_awaited_once()

    async def test_rejects_unknown_requirement_reference(self):
        payload = self.payload()
        payload["work_packages"][0]["requirement_refs"] = ["missing#Requirement"]
        adapter, _ = self.adapter(json.dumps(payload))
        with self.assertRaisesRegex(OpenSpecError, "unknown requirement"):
            await adapter.generate_change("add authentication")

    async def test_refinement_requires_exact_selected_ids(self):
        assigned = replace(
            draft(),
            id="CHG1",
            work_packages=[
                replace(
                    draft().work_packages[0],
                    id="WP1",
                    tasks=[
                        replace(
                            draft().work_packages[0].tasks[0],
                            id="WP1-T1",
                            completed=True,
                        ),
                        replace(draft().work_packages[0].tasks[1], id="WP1-T2"),
                    ],
                )
            ],
        )
        response = [{
            "id": "WP1-T1",
            "title": "Refined validation",
            "description": "Use the revised validation flow.",
            "acceptance_criteria": ["The revised validation flow is implemented."],
            "complexity": "medium",
            "reasoning": "The revision crosses boundaries.",
            "context": "",
            "preferred_capability": "write",
        }]
        adapter, _ = self.adapter(json.dumps(response))
        result = await adapter.refine_tasks(assigned, ["WP1-T1"], "revise validation")
        self.assertEqual(list(result), ["WP1-T1"])
        self.assertEqual(result["WP1-T1"].title, "Refined validation")
        self.assertEqual(
            result["WP1-T1"].acceptance_criteria,
            ["The revised validation flow is implemented."],
        )
        self.assertTrue(result["WP1-T1"].completed)


class HandlerAndController(unittest.IsolatedAsyncioTestCase):
    def test_implementation_review_actions_and_legacy_aliases_route_consistently(self):
        cases = {
            "accept": ("accept", ""),
            "accept anyway: prototype": ("accept", "prototype"),
            "undo": ("undo", ""),
            "undo code": ("undo", ""),
            "submit to coder again: add coverage": ("submit_again", "add coverage"),
            "try coder again: preserve API": ("submit_again", "preserve API"),
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(
                    dialog_controller._implementation_resolution(command), expected)
        self.assertIsNone(dialog_controller._implementation_resolution("accept this plan"))

    def test_recovery_command_requires_an_explicit_nonempty_prompt(self):
        self.assertEqual(
            dialog_controller._implementation_recovery(
                "establish new baseline: repair named originals and run tests"),
            "repair named originals and run tests")
        self.assertIsNone(
            dialog_controller._implementation_recovery("establish new baseline:"))
        self.assertIsNone(dialog_controller._implementation_recovery("retry review"))
        self.assertEqual(
            dialog_controller._implementation_recovery_action("inspect recovery"),
            ("inspect_recovery", ""))
        self.assertEqual(
            dialog_controller._implementation_recovery_action(
                "stop and repair manually"),
            ("stop_recovery", ""))
        self.assertEqual(
            dialog_controller._implementation_recovery_action(
                "establish new baseline: repair originals"),
            ("establish_baseline", "repair originals"))

    async def test_default_artifact_store_refuses_unbridged_target_project_access(self):
        store = OpenSpecArtifactStore()
        self.assertIsInstance(store._client, BridgeRequiredMcpClient)
        with self.assertRaises(ArtifactStoreError) as raised:
            await store._load_catalog()
        self.assertIn("request-scoped IDE MCP bridge", str(raised.exception))

    async def test_missing_catalog_is_bridge_context_error_for_task_selection(self):
        store = OpenSpecArtifactStore(FakeMcpClient())
        with self.assertRaises(ArtifactStoreError) as raised:
            await store.tasks_for_selectors(["WP1-T2"])
        self.assertIn("not found through the current IDE MCP bridge", str(raised.exception))

    async def test_handler_creates_then_updates_selected_tasks(self):
        adapter = FakeAdapter()
        client = FakeMcpClient()
        handler = SpecCommandHandler(adapter, OpenSpecArtifactStore(client))
        created = await handler.handle([
            {"role": "user", "content": "/spec add authentication"}
        ])
        self.assertEqual(created.change_id, "CHG1")
        self.assertIn("WP1-T1", created.content)

        updated = await handler.handle([{
            "role": "user",
            "content": "/update WP1-T1 WP1-T2: make validation explicit",
        }])
        self.assertEqual(updated.updated_ids, ("WP1-T1", "WP1-T2"))
        self.assertIn("Refined Create authentication service", updated.content)
        self.assertEqual(
            adapter.refined[0],
            ("CHG1", ["WP1-T1", "WP1-T2"], "make validation explicit"),
        )

    async def test_rework_creates_new_change_with_source_task_traceability(self):
        adapter = FakeAdapter()
        store = OpenSpecArtifactStore(FakeMcpClient())
        handler = SpecCommandHandler(adapter, store)
        created = await handler.handle([{"role": "user", "content": "/spec add authentication"}])
        source = await store.task_by_id("WP1-T1")
        follow_up = await handler.rework([source], "Store credentials securely")
        self.assertEqual(follow_up.change_id, "CHG2")
        self.assertIn("corrective follow-up CHG2", follow_up.content)
        catalog = await store._load_catalog()
        self.assertEqual(catalog["changes"][1]["follow_up_for"], ["WP1-T1"])

    async def test_controller_requires_streaming_for_bridge_backed_planning(self):
        with self.assertRaises(HTTPException) as raised:
            await dialog_controller.chat_completions(
                ChatCompletionRequest(
                    model="test",
                    messages=[ChatMessage(role="user", content="/spec add authentication")],
                )
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("requires a streaming request", raised.exception.detail)

    async def test_unknown_update_returns_400(self):
        old_handler = dialog_controller.spec_handler
        dialog_controller.spec_handler = SpecCommandHandler(
            FakeAdapter(), OpenSpecArtifactStore(FakeMcpClient())
        )
        try:
            with self.assertRaises(HTTPException) as raised:
                await dialog_controller.chat_completions(
                    ChatCompletionRequest(
                        model="test",
                        messages=[
                            ChatMessage(
                                role="user",
                                content="/update WP99-T1: change it",
                            )
                        ],
                    )
                )
        finally:
            dialog_controller.spec_handler = old_handler
        self.assertEqual(raised.exception.status_code, 400)

    async def test_implementation_bootstraps_ide_mcp_before_task_execution(self):
        response = await dialog_controller.chat_completions(
            ChatCompletionRequest(
                model="test", stream=True,
                messages=[ChatMessage(role="user", content="/implement WP1-T1")],
                tools=[{
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {"type": "object"}},
                }],
            )
        )
        chunks = [chunk async for chunk in response.body_iterator]
        body = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks).decode()
        self.assertIn(".codepartner/ide_mcp.json", body)
        self.assertIn('"finish_reason": "tool_calls"', body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    async def test_implementation_respects_tool_choice_none(self):
        with self.assertRaises(HTTPException) as raised:
            await dialog_controller.chat_completions(
                ChatCompletionRequest(
                    model="test", stream=True, tool_choice="none",
                    messages=[ChatMessage(role="user", content="/implement WP1-T1")],
                    tools=[{"type": "function", "function": {
                        "name": "apply_patch", "parameters": {"type": "object"},
                    }}],
                )
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("requires tool calls", raised.exception.detail)

    async def test_normal_request_keeps_existing_provider_flow(self):
        provider = mock.AsyncMock()
        provider.NAME = "fake"
        provider.complete.return_value = ("normal response", "endTurn")
        with mock.patch("api.dialog_controller.providers.active", return_value=provider), mock.patch(
            "conversation.non_streaming_responder.providers.active",
            return_value=provider,
        ):
            response = await dialog_controller.chat_completions(
                ChatCompletionRequest(
                    model="test",
                    messages=[ChatMessage(role="user", content="hello")],
                )
            )
        self.assertEqual(response.choices[0].message.content, "normal response")


class ModeAdapter(FakeAdapter):
    async def refine_change(self, change: OpenSpecChange, target_id: str,
                            instruction: str) -> OpenSpecChange:
        self.refined.append((target_id, instruction))
        return replace(change, title=f"{change.title} ({instruction})")


class ModeRouting(unittest.TestCase):
    def route(self, model, *contents):
        roles = ["user", "assistant"]
        messages = [{"role": roles[index % 2], "content": content}
                    for index, content in enumerate(contents)]
        return route_message(model, messages)

    def test_default_and_unknown_models_have_no_mode(self):
        self.assertIsNone(self.route(DEFAULT_MODEL, "WP1: hello"))
        self.assertIsNone(self.route("test", "WP1: hello"))

    def test_slash_commands_bypass_the_mode(self):
        self.assertIsNone(self.route("codepartner-review", "/explain EPIC1"))
        self.assertIsNone(self.route("codepartner-spec", "/apply"))

    def test_review_mode_uses_and_reuses_scope(self):
        self.assertEqual(self.route("codepartner-review", "wp1, WP2-T1: are tests missing?").content,
                         "/review wp1 WP2-T1: are tests missing?")
        self.assertEqual(self.route("codepartner-review", "WP3 check naming").content,
                         "/review WP3: check naming")
        self.assertEqual(self.route("codepartner-review", "WP1-T2").content, "/review WP1-T2")
        sticky = self.route("codepartner-review", "WP1: first", "answer", "what about errors?")
        self.assertEqual(sticky.content, "/review WP1: what about errors?")
        sticky = self.route("codepartner-review", "/review WP4", "answer", "and tests?")
        self.assertEqual(sticky.content, "/review WP4: and tests?")
        self.assertIn("needs a scope", self.route("codepartner-review", "anything wrong?").reply)

    def test_review_parses_rewritten_command(self):
        command = self.route("codepartner-review", "wp1 WP2-T1: why?").content
        self.assertEqual(extract_scoped_request(command, "review"), (["WP1", "WP2-T1"], "why?"))

    def test_implement_mode_requires_scope_in_each_message(self):
        self.assertEqual(self.route("codepartner-implement", "WP1-T1: add tests").content,
                         "/implement WP1-T1: add tests")
        self.assertIn("Nothing was executed",
                      self.route("codepartner-implement", "WP1", "done", "now add logging").reply)
        self.assertIsNone(split_scope("WP1x: nope"))

    def test_spec_mode_drafts_plain_messages(self):
        self.assertEqual(self.route("codepartner-spec", "add authentication").draft_request,
                         "add authentication")

    def test_help_and_model_list_advertise_modes(self):
        self.assertIn("codepartner-review", help_text())
        ids = [model.id for model in list_models().data]
        self.assertEqual(ids[0], DEFAULT_MODEL)
        self.assertIn("codepartner-spec", ids)
        self.assertTrue(is_discard_command(" /discard "))


class SpecModeDrafts(unittest.IsolatedAsyncioTestCase):
    async def test_draft_is_saved_only_on_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeMcpClient()
            adapter = ModeAdapter()
            service = RefinementService(
                OpenSpecArtifactStore(client), adapter,
                RoutingPersistence(Path(directory) / "routing.sqlite3"))
            created = await service.start_draft("conv", "add authentication")
            self.assertIn("not saved", created)
            self.assertIn("WP1-T1", created)
            self.assertEqual(client.files, {})
            updated = await service.update("conv", "use tokens")
            self.assertIn("Add Authentication (use tokens)", updated)
            self.assertEqual(adapter.refined[0][1], "use tokens")
            self.assertEqual(client.files, {})
            applied = await service.apply("conv")
            self.assertIn("Created CHG1", applied)
            self.assertNotEqual(client.files, {})
            with self.assertRaises(RefinementError):
                await service.apply("conv")

    async def test_discard_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeMcpClient()
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")
            service = RefinementService(OpenSpecArtifactStore(client), ModeAdapter(), persistence)
            await service.start_draft("conv", "add authentication")
            self.assertIn("new change", await service.discard("conv"))
            self.assertIsNone(await persistence.active_refinement("conv"))
            self.assertEqual(client.files, {})

    async def test_implement_mode_without_scope_replies_with_hint(self):
        response = await dialog_controller.chat_completions(ChatCompletionRequest(
            model="codepartner-implement",
            messages=[ChatMessage(role="user", content="make it faster")],
        ))
        self.assertIn("needs an explicit scope", response.choices[0].message.content)


class JsonReplyTolerance(unittest.IsolatedAsyncioTestCase):
    def test_fenced_and_prefixed_objects_parse(self):
        value = {"title": "x"}
        for reply in ('```json\n{"title": "x"}\n```', 'Here is the change:\n{"title": "x"}\nDone.',
                      '  {"title": "x"}  '):
            self.assertEqual(CliOpenSpecAdapter._json_object(reply, "change"), value)
        self.assertEqual(CliOpenSpecAdapter._extract_json('```\n[1, 2]\n```', "["), "[1, 2]")

    def test_empty_or_prose_reply_is_malformed(self):
        for reply in ("", "I could not produce a plan."):
            with self.assertRaises(MalformedJsonError):
                CliOpenSpecAdapter._json_object(reply, "change")

    async def test_malformed_reply_is_retried_once_with_json_only_reminder(self):
        adapter = CliOpenSpecAdapter()
        complete = mock.AsyncMock(side_effect=["Sure! Planning now.", '{"title": "x"}'])
        with mock.patch.object(adapter, "_complete", new=complete):
            result = await adapter._complete_json(
                "req", "system", "change generation",
                lambda text: adapter._json_object(text, "change"))
        self.assertEqual(result, {"title": "x"})
        self.assertEqual(complete.await_count, 2)
        self.assertIn("not bare JSON", complete.await_args_list[1].args[1])

    async def test_second_malformed_reply_reports_what_was_received(self):
        adapter = CliOpenSpecAdapter()
        complete = mock.AsyncMock(return_value="No JSON here")
        with mock.patch.object(adapter, "_complete", new=complete):
            with self.assertRaisesRegex(MalformedJsonError, "after a retry .*No JSON here"):
                await adapter._complete_json(
                    "req", "system", "change generation",
                    lambda text: adapter._json_object(text, "change"))
        self.assertEqual(complete.await_count, 2)

    async def test_schema_errors_are_not_retried(self):
        adapter = CliOpenSpecAdapter()
        complete = mock.AsyncMock(return_value='{"unexpected": true}')
        with mock.patch.object(adapter, "_complete", new=complete):
            with self.assertRaises(OpenSpecError) as raised:
                await adapter._complete_json("req", "system", "change generation",
                                             adapter._parse_change)
        self.assertNotIsInstance(raised.exception, MalformedJsonError)
        self.assertEqual(complete.await_count, 1)

    def test_request_shape_reports_structure_lines(self):
        shape = request_shape("add password reset\n\n<attachments>\nfile.py\n</attachments>")
        self.assertEqual(shape["lines"], 5)
        self.assertEqual([number for number, _ in shape["markers"]], [3, 5])


if __name__ == "__main__":
    unittest.main()

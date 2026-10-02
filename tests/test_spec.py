"""Tests for additive OpenSpec generation and scoped task refinement."""

import json
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException

from agent.spec.artifact_store import (
    CATALOG_PATH,
    DASHBOARD_PATH,
    ArtifactStoreError,
    OpenSpecArtifactStore,
)
from agent.spec.command import (
    SpecCommandError,
    SpecCommandHandler,
    extract_spec_request,
    extract_update_request,
    is_spec_command,
    is_update_command,
)
from agent.spec.jetbrains_mcp import JetBrainsMcpClient, JetBrainsMcpError
from agent.spec.models import (
    Capability,
    Decision,
    Design,
    OpenSpecChange,
    Proposal,
    Requirement,
    Scenario,
    Task,
    WorkPackage,
)
from agent.spec.openspec_adapter import CliOpenSpecAdapter, OpenSpecAdapter, OpenSpecError
from api import dialog_controller
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


class ArtifactSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_work_package_selection_skips_completed_tasks_and_completion_rewrites_artifacts(self):
        client = FakeMcpClient()
        store = OpenSpecArtifactStore(client)
        change = await store.create_change(draft(), "authentication")
        selected = await store.tasks_for_selectors(["WP1", "WP1-T2"])
        self.assertEqual([task.id for _, task in selected], ["WP1-T1", "WP1-T2"])

        await store.mark_task_completed("WP1-T1")
        selected_after_completion = await store.tasks_for_selectors(["WP1"])
        self.assertEqual([task.id for _, task in selected_after_completion], ["WP1-T2"])
        self.assertIn("- [x] WP1-T1", client.files[f"{change.path}/tasks.md"])


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
            "http://127.0.0.1:64462/stream", "/workspace/project", 5
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
            "complexity": "medium",
            "reasoning": "The revision crosses boundaries.",
            "context": "",
            "preferred_capability": "write",
        }]
        adapter, _ = self.adapter(json.dumps(response))
        result = await adapter.refine_tasks(assigned, ["WP1-T1"], "revise validation")
        self.assertEqual(list(result), ["WP1-T1"])
        self.assertEqual(result["WP1-T1"].title, "Refined validation")
        self.assertTrue(result["WP1-T1"].completed)


class HandlerAndController(unittest.IsolatedAsyncioTestCase):
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

    async def test_controller_returns_planning_response_without_tool_round_trip(self):
        old_handler = dialog_controller.spec_handler
        client = FakeMcpClient()
        dialog_controller.spec_handler = SpecCommandHandler(
            FakeAdapter(), OpenSpecArtifactStore(client)
        )
        try:
            response = await dialog_controller.chat_completions(
                ChatCompletionRequest(
                    model="test",
                    messages=[ChatMessage(role="user", content="/spec add authentication")],
                )
            )
        finally:
            dialog_controller.spec_handler = old_handler
        self.assertIn("Created CHG1", response.choices[0].message.content)
        self.assertFalse(response.choices[0].message.tool_calls)

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


if __name__ == "__main__":
    unittest.main()

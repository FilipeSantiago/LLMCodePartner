"""Persist complete OpenSpec changes and the CodePartner work-package catalog."""

import abc
import json
from dataclasses import replace
from typing import Protocol

from agent.spec.jetbrains_mcp import JetBrainsMcpClient, JetBrainsMcpError
from agent.spec.models import (
    OpenSpecChange,
    Task,
    render_capability,
    render_dashboard,
    render_design,
    render_metadata,
    render_proposal,
    render_tasks,
)

CATALOG_PATH = ".codepartner/spec-index.json"
DASHBOARD_PATH = ".codepartner/tasks.md"


class ArtifactStoreError(RuntimeError):
    """OpenSpec artifacts could not be read or persisted through JetBrains MCP."""


class ArtifactSelectionError(ArtifactStoreError):
    """An update selected unknown or incompatible task IDs."""


class McpFileClient(Protocol):
    async def create_file(
            self, path: str, content: str, overwrite: bool = True
    ) -> None: ...

    async def read_file(self, path: str) -> str: ...

    async def file_exists(self, path: str) -> bool: ...


class ArtifactStore(abc.ABC):
    @abc.abstractmethod
    async def create_change(self, draft: OpenSpecChange, slug: str) -> OpenSpecChange:
        ...

    @abc.abstractmethod
    async def change_for_tasks(self, task_ids: list[str]) -> OpenSpecChange:
        ...

    @abc.abstractmethod
    async def replace_tasks(
            self, change: OpenSpecChange, replacements: dict[str, Task]
    ) -> OpenSpecChange:
        ...

    @abc.abstractmethod
    async def task_by_id(self, task_id: str) -> tuple[OpenSpecChange, Task]:
        ...

    @abc.abstractmethod
    async def tasks_for_selectors(self, selectors: list[str]) -> list[tuple[OpenSpecChange, Task]]:
        ...

    @abc.abstractmethod
    async def mark_task_completed(self, task_id: str) -> None:
        ...


class OpenSpecArtifactStore(ArtifactStore):
    """MCP-backed multi-file store with globally stable work-package IDs."""

    def __init__(self, client: McpFileClient | None = None):
        self._client = client or JetBrainsMcpClient()

    async def create_change(self, draft: OpenSpecChange, slug: str) -> OpenSpecChange:
        catalog = await self._load_catalog()
        change_number = catalog["next_change"]
        next_package = catalog["next_work_package"]
        packages = []
        for package in draft.work_packages:
            package_id = f"WP{next_package}"
            tasks = [
                replace(task, id=f"{package_id}-T{task_number}")
                for task_number, task in enumerate(package.tasks, start=1)
            ]
            packages.append(replace(package, id=package_id, tasks=tasks))
            next_package += 1

        change = replace(
            draft,
            id=f"CHG{change_number}",
            number=change_number,
            slug=f"change-{change_number:04d}-{slug}",
            work_packages=packages,
        )
        changes = [
            OpenSpecChange.from_dict(value) for value in catalog.get("changes", [])
        ]
        changes.append(change)
        updated_catalog = {
            "version": 1,
            "next_change": change_number + 1,
            "next_work_package": next_package,
            "changes": [item.to_dict() for item in changes],
        }
        try:
            await self._write_change(change)
            await self._client.create_file(
                DASHBOARD_PATH, render_dashboard(changes), overwrite=True
            )
            await self._client.create_file(
                CATALOG_PATH,
                json.dumps(updated_catalog, indent=2, ensure_ascii=False) + "\n",
                overwrite=True,
            )
        except JetBrainsMcpError as exc:
            raise ArtifactStoreError(
                f"could not persist {change.id} through JetBrains MCP: {exc}"
            ) from exc
        return change

    async def change_for_tasks(self, task_ids: list[str]) -> OpenSpecChange:
        catalog = await self._load_catalog()
        changes = [
            OpenSpecChange.from_dict(value) for value in catalog.get("changes", [])
        ]
        owners = {
            change.id: change
            for change in changes
            if any(
                task.id in task_ids
                for package in change.work_packages
                for task in package.tasks
            )
        }
        found = {
            task.id
            for change in owners.values()
            for package in change.work_packages
            for task in package.tasks
            if task.id in task_ids
        }
        missing = set(task_ids) - found
        if missing:
            raise ArtifactSelectionError(
                "unknown task IDs: " + ", ".join(sorted(missing))
            )
        if len(owners) != 1:
            raise ArtifactSelectionError("one /update may target tasks from only one change")
        return next(iter(owners.values()))

    async def task_by_id(self, task_id: str) -> tuple[OpenSpecChange, Task]:
        """Load one task with its owning OpenSpec change without mutating artifacts."""
        catalog = await self._load_catalog()
        for value in catalog.get("changes", []):
            change = OpenSpecChange.from_dict(value)
            for package in change.work_packages:
                for task in package.tasks:
                    if task.id == task_id:
                        return change, task
        raise ArtifactSelectionError(f"unknown task ID: {task_id}")

    async def tasks_for_selectors(self, selectors: list[str]) -> list[tuple[OpenSpecChange, Task]]:
        """Expand task and work-package selectors in user order, without repeats."""
        catalog = await self._load_catalog()
        changes = [OpenSpecChange.from_dict(value) for value in catalog.get("changes", [])]
        packages = {
            package.id: (change, package)
            for change in changes for package in change.work_packages
        }
        tasks = {
            task.id: (change, task)
            for change in changes for package in change.work_packages for task in package.tasks
        }
        selected: list[tuple[OpenSpecChange, Task]] = []
        selected_ids: set[str] = set()
        for selector in selectors:
            if selector in tasks:
                candidates = [tasks[selector]]
            elif selector in packages:
                change, package = packages[selector]
                candidates = [(change, task) for task in package.tasks]
            else:
                raise ArtifactSelectionError(f"unknown task or work package ID: {selector}")
            for change, task in candidates:
                if task.completed or task.id in selected_ids:
                    continue
                selected.append((change, task))
                selected_ids.add(task.id)
        if not selected:
            raise ArtifactSelectionError("the selected tasks are already completed")
        return selected

    async def mark_task_completed(self, task_id: str) -> None:
        catalog = await self._load_catalog()
        changes = [OpenSpecChange.from_dict(value) for value in catalog.get("changes", [])]
        owner: OpenSpecChange | None = None
        updated_changes: list[OpenSpecChange] = []
        found = False
        for change in changes:
            packages = []
            for package in change.work_packages:
                tasks = []
                for task in package.tasks:
                    if task.id == task_id:
                        task = replace(task, completed=True)
                        found = True
                    tasks.append(task)
                packages.append(replace(package, tasks=tasks))
            updated = replace(change, work_packages=packages)
            if found and owner is None:
                owner = updated
            updated_changes.append(updated)
        if not found or owner is None:
            raise ArtifactSelectionError(f"unknown task ID: {task_id}")
        catalog["changes"] = [change.to_dict() for change in updated_changes]
        try:
            await self._client.create_file(
                f"{owner.path}/tasks.md", render_tasks(owner), overwrite=True
            )
            await self._client.create_file(
                DASHBOARD_PATH, render_dashboard(updated_changes), overwrite=True
            )
            await self._client.create_file(
                CATALOG_PATH,
                json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                overwrite=True,
            )
        except JetBrainsMcpError as exc:
            raise ArtifactStoreError(
                f"could not mark {task_id} complete through JetBrains MCP: {exc}"
            ) from exc

    async def replace_tasks(
            self, change: OpenSpecChange, replacements: dict[str, Task]
    ) -> OpenSpecChange:
        catalog = await self._load_catalog()
        updated_packages = []
        for package in change.work_packages:
            updated_packages.append(replace(
                package,
                tasks=[replacements.get(task.id, task) for task in package.tasks],
            ))
        updated = replace(change, work_packages=updated_packages)
        changes = [
            updated if value["id"] == change.id else OpenSpecChange.from_dict(value)
            for value in catalog.get("changes", [])
        ]
        catalog["changes"] = [item.to_dict() for item in changes]
        try:
            await self._client.create_file(
                f"{updated.path}/tasks.md", render_tasks(updated), overwrite=True
            )
            await self._client.create_file(
                DASHBOARD_PATH, render_dashboard(changes), overwrite=True
            )
            await self._client.create_file(
                CATALOG_PATH,
                json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                overwrite=True,
            )
        except JetBrainsMcpError as exc:
            raise ArtifactStoreError(
                f"could not update {change.id} through JetBrains MCP: {exc}"
            ) from exc
        return updated

    async def _load_catalog(self) -> dict:
        try:
            if not await self._client.file_exists(CATALOG_PATH):
                return {
                    "version": 1,
                    "next_change": 1,
                    "next_work_package": 1,
                    "changes": [],
                }
            payload = json.loads(await self._client.read_file(CATALOG_PATH))
            if (
                    not isinstance(payload, dict)
                    or payload.get("version") != 1
                    or not isinstance(payload.get("changes"), list)
            ):
                raise ArtifactStoreError(f"{CATALOG_PATH} has an unsupported format")
            return payload
        except json.JSONDecodeError as exc:
            raise ArtifactStoreError(f"{CATALOG_PATH} is not valid JSON") from exc
        except JetBrainsMcpError as exc:
            raise ArtifactStoreError(f"could not load {CATALOG_PATH}: {exc}") from exc

    async def _write_change(self, change: OpenSpecChange) -> None:
        files = {
            f"{change.path}/.openspec.yaml": render_metadata(),
            f"{change.path}/proposal.md": render_proposal(change),
            f"{change.path}/design.md": render_design(change),
            f"{change.path}/tasks.md": render_tasks(change),
        }
        for capability in change.capabilities:
            files[f"{change.path}/specs/{capability.path}/spec.md"] = render_capability(
                capability
            )
        for path, content in files.items():
            await self._client.create_file(path, content, overwrite=True)

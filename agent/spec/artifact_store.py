"""Persist complete OpenSpec changes and the CodePartner work-package catalog."""

import abc
import json
import logging
import re
from dataclasses import replace
from typing import Any, Protocol

from agent.spec.jetbrains_mcp import JetBrainsMcpError
from agent.spec.models import (
    CompletionReview,
    Epic,
    OpenSpecChange,
    Task,
    render_capability,
    render_dashboard,
    render_design,
    render_metadata,
    render_proposal,
    render_tasks,
)
from logger.diagnostic import debug

log = logging.getLogger("codepartner.spec.artifact")

CATALOG_PATH = ".codepartner/spec-index.json"
DASHBOARD_PATH = ".codepartner/tasks.md"


class ArtifactStoreError(RuntimeError):
    """OpenSpec artifacts could not be read or persisted through JetBrains MCP."""


class ArtifactSelectionError(ArtifactStoreError):
    """An update selected unknown or incompatible task IDs."""


class ArtifactLookupIndeterminate(RuntimeError):
    """The IDE declined a repeat lookup without asserting the file is absent."""


class BridgeRequiredMcpClient:
    """Fail closed when a target-project request escaped its request bridge.

    This is intentionally not a filesystem or standalone-MCP fallback.  Target
    project artifacts are valid only in the IDE request that advertised the
    compatible tools, represented by ``BridgeMcpFileClient``.
    """
    _message = (
        "target-project artifacts require the request-scoped IDE MCP bridge; "
        "start this command from the intended IDE project with file read/search tools advertised"
    )

    async def create_file(self, path: str, content: str, overwrite: bool = True) -> None:
        raise JetBrainsMcpError(self._message)

    async def read_file(self, path: str) -> str:
        raise JetBrainsMcpError(self._message)

    async def file_exists(self, path: str) -> bool:
        raise JetBrainsMcpError(self._message)


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
    async def tasks_for_selectors(self, selectors: list[str],
                                  include_completed: bool = False) -> list[tuple[OpenSpecChange, Task]]:
        ...

    @abc.abstractmethod
    async def mark_task_completed(self, task_id: str,
                                  review: CompletionReview | None = None) -> None:
        ...

    async def object_by_id(self, object_id: str) -> tuple[str, OpenSpecChange, Any]:
        raise NotImplementedError

    async def update_change(self, change: OpenSpecChange) -> OpenSpecChange:
        raise NotImplementedError

    async def package_for_task(self, task_id: str):
        raise NotImplementedError

    async def record_work_package_review(self, package_id: str, review,
                                         completed: bool) -> OpenSpecChange:
        raise NotImplementedError


class OpenSpecArtifactStore(ArtifactStore):
    """MCP-backed multi-file store with globally stable work-package IDs."""

    def __init__(self, client: McpFileClient | None = None):
        # Do not infer a target project from server CWD or a process-level MCP
        # configuration.  Only a request-scoped bridge may access target files.
        self._client = client or BridgeRequiredMcpClient()

    async def create_change(self, draft: OpenSpecChange, slug: str) -> OpenSpecChange:
        catalog = await self._load_catalog()
        change_number = catalog["next_change"]
        next_package = catalog["next_work_package"]
        packages = []
        draft_package_ids = {
            f"WP{index}": f"WP{next_package + index - 1}"
            for index in range(1, len(draft.work_packages) + 1)
        }
        def task_dependency(value: str, current_package: str) -> str:
            normalized = value.upper()
            if re_match := re.fullmatch(r"WP(\d+)-T(\d+)", normalized):
                mapped = draft_package_ids.get(f"WP{re_match.group(1)}")
                return f"{mapped}-T{re_match.group(2)}" if mapped else normalized
            if normalized.startswith("T"):
                return f"{current_package}-{normalized}"
            return normalized
        for package in draft.work_packages:
            package_id = f"WP{next_package}"
            tasks = [
                replace(
                    task,
                    id=f"{package_id}-T{task_number}",
                    depends_on=[task_dependency(dependency, package_id)
                                for dependency in task.depends_on],
                )
                for task_number, task in enumerate(package.tasks, start=1)
            ]
            packages.append(replace(
                package,
                id=package_id,
                tasks=tasks,
                depends_on=[draft_package_ids.get(item.upper(), item.upper())
                            for item in package.depends_on],
            ))
            next_package += 1

        change = replace(
            draft,
            id=f"CHG{change_number}",
            number=change_number,
            slug=f"change-{change_number:04d}-{slug}",
            work_packages=packages,
            epic=replace(
                draft.epic or Epic("", draft.title, draft.proposal.why),
                id=f"EPIC{change_number}",
                work_package_ids=[package.id for package in packages],
            ),
        )
        changes = [
            OpenSpecChange.from_dict(value) for value in catalog.get("changes", [])
        ]
        changes.append(change)
        updated_catalog = {
            "version": 2,
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
        catalog = await self._load_catalog(required=True)
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
        catalog = await self._load_catalog(required=True)
        for value in catalog.get("changes", []):
            change = OpenSpecChange.from_dict(value)
            for package in change.work_packages:
                for task in package.tasks:
                    if task.id == task_id:
                        return change, task
        raise ArtifactSelectionError(f"unknown task ID: {task_id}")

    async def tasks_for_selectors(self, selectors: list[str],
                                  include_completed: bool = False) -> list[tuple[OpenSpecChange, Task]]:
        """Expand task and work-package selectors in user order, without repeats."""
        catalog = await self._load_catalog(required=True)
        changes = [OpenSpecChange.from_dict(value) for value in catalog.get("changes", [])]
        packages = {
            package.id: (change, package)
            for change in changes for package in change.work_packages
        }
        tasks = {
            task.id: (change, task)
            for change in changes for package in change.work_packages for task in package.tasks
        }
        epics = {
            change.epic.id: change for change in changes if change.epic is not None
        }
        selected: list[tuple[OpenSpecChange, Task]] = []
        selected_ids: set[str] = set()
        for selector in selectors:
            if selector in tasks:
                candidates = [tasks[selector]]
            elif selector in packages:
                change, package = packages[selector]
                candidates = [(change, task) for task in package.tasks]
            elif selector in epics:
                change = epics[selector]
                candidates = [
                    (change, task) for package in change.work_packages
                    for task in package.tasks
                ]
            else:
                raise ArtifactSelectionError(f"unknown task or work package ID: {selector}")
            for change, task in candidates:
                if ((task.completed or task.status in {"completed", "cancelled"})
                        and not include_completed) or task.id in selected_ids:
                    continue
                selected.append((change, task))
                selected_ids.add(task.id)
        if not selected:
            raise ArtifactSelectionError("the selected work is already completed or cancelled")
        selected = self._dependency_order(selected)
        return selected

    @staticmethod
    def _dependency_order(
            selected: list[tuple[OpenSpecChange, Task]]) -> list[tuple[OpenSpecChange, Task]]:
        by_id = {task.id: (change, task) for change, task in selected}
        ordered: list[tuple[OpenSpecChange, Task]] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ArtifactSelectionError(f"cyclic task dependency involving {task_id}")
            visiting.add(task_id)
            change, task = by_id[task_id]
            for dependency in task.depends_on:
                if dependency in by_id:
                    visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)
            ordered.append((change, task))

        for _, task in selected:
            visit(task.id)
        return ordered

    async def mark_task_completed(self, task_id: str,
                                  review: CompletionReview | None = None) -> None:
        catalog = await self._load_catalog(required=True)
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
                        task = replace(task, completed=True, status="completed",
                                       completion_review=review)
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
        catalog = await self._load_catalog(required=True)
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

    async def object_by_id(self, object_id: str) -> tuple[str, OpenSpecChange, Any]:
        target = object_id.strip().upper()
        catalog = await self._load_catalog(required=True)
        for value in catalog.get("changes", []):
            change = OpenSpecChange.from_dict(value)
            if change.epic and change.epic.id.upper() == target:
                return "epic", change, change.epic
            for package in change.work_packages:
                if package.id.upper() == target:
                    return "work_package", change, package
                for task in package.tasks:
                    if task.id.upper() == target:
                        return "task", change, task
        raise ArtifactSelectionError(f"unknown OpenSpec object ID: {object_id}")

    async def update_change(self, change: OpenSpecChange) -> OpenSpecChange:
        catalog = await self._load_catalog(required=True)
        changes = [OpenSpecChange.from_dict(value) for value in catalog.get("changes", [])]
        if not any(item.id == change.id for item in changes):
            raise ArtifactSelectionError(f"unknown OpenSpec change: {change.id}")
        changes = [change if item.id == change.id else item for item in changes]
        catalog["version"] = 2
        catalog["changes"] = [item.to_dict() for item in changes]
        try:
            await self._write_change(change)
            await self._client.create_file(
                DASHBOARD_PATH, render_dashboard(changes), overwrite=True)
            await self._client.create_file(
                CATALOG_PATH, json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                overwrite=True)
        except JetBrainsMcpError as exc:
            raise ArtifactStoreError(
                f"could not update {change.id} through JetBrains MCP: {exc}"
            ) from exc
        return change

    async def package_for_task(self, task_id: str):
        catalog = await self._load_catalog(required=True)
        for value in catalog.get("changes", []):
            change = OpenSpecChange.from_dict(value)
            for package in change.work_packages:
                if any(task.id == task_id for task in package.tasks):
                    return change, package
        raise ArtifactSelectionError(f"unknown task ID: {task_id}")

    async def record_work_package_review(self, package_id: str, review,
                                         completed: bool) -> OpenSpecChange:
        kind, change, target = await self.object_by_id(package_id)
        if kind != "work_package":
            raise ArtifactSelectionError(f"{package_id} is not a work package")
        packages = []
        for package in change.work_packages:
            if package.id != package_id:
                packages.append(package)
                continue
            tasks = [replace(task, completed=True, status="completed")
                     for task in package.tasks] if completed else package.tasks
            packages.append(replace(
                package, tasks=tasks, review=review,
                status="completed" if completed else (
                    "blocked" if review.outcome == "BLOCK" else "unverified"),
            ))
        return await self.update_change(replace(change, work_packages=packages))

    async def _load_catalog(self, required: bool = False) -> dict:
        try:
            lookup_indeterminate = False
            try:
                exists = await self._client.file_exists(CATALOG_PATH)
            except ArtifactLookupIndeterminate as exc:
                # PyCharm's deterministic-call guard is not an absence result.
                # Let the request-scoped reader establish whether the catalog is
                # present, rather than converting the guard into a false negative.
                lookup_indeterminate = True
                exists = True
                debug(log, "artifact.catalog_lookup_indeterminate", path=CATALOG_PATH,
                      reason=str(exc))
            if not exists:
                if required:
                    raise ArtifactStoreError(
                        f"{CATALOG_PATH} was not found through the current IDE MCP bridge; "
                        "the request may be attached to a different target project"
                    )
                return {
                    "version": 2,
                    "next_change": 1,
                    "next_work_package": 1,
                    "changes": [],
                }
            try:
                raw_catalog = await self._client.read_file(CATALOG_PATH)
            except JetBrainsMcpError as exc:
                text = str(exc).lower()
                if lookup_indeterminate and ("doesn't exist" in text or "not found" in text):
                    if required:
                        raise ArtifactStoreError(
                            f"{CATALOG_PATH} was not found when the current IDE MCP bridge attempted to read it"
                        ) from exc
                    return {
                        "version": 2,
                        "next_change": 1,
                        "next_work_package": 1,
                        "changes": [],
                    }
                raise
            # ##DELETE AFTER CORRECTION## Preserve the exact JSON parser input.
            debug(log, "artifact.catalog_json_input", path=CATALOG_PATH, raw_content=raw_catalog)
            payload = json.loads(raw_catalog)
            # ##DELETE AFTER CORRECTION## Preserve decoded data before schema validation.
            debug(log, "artifact.catalog_json_decoded", path=CATALOG_PATH, catalog=payload)
            if (
                    not isinstance(payload, dict)
                    or payload.get("version") not in {1, 2}
                    or not isinstance(payload.get("changes"), list)
            ):
                # ##DELETE AFTER CORRECTION## Keep schema failures distinct from JSON parsing.
                debug(log, "artifact.catalog_schema_rejected", path=CATALOG_PATH, catalog=payload)
                raise ArtifactStoreError(f"{CATALOG_PATH} has an unsupported format")
            return payload
        except json.JSONDecodeError as exc:
            # ##DELETE AFTER CORRECTION## JSON parser coordinates distinguish wrapper from schema failures.
            debug(log, "artifact.catalog_json_decode_failed", path=CATALOG_PATH,
                  error=exc.msg, position=exc.pos, line=exc.lineno, column=exc.colno)
            # This can be an unrecognised tool-response envelope.  Do not claim
            # the stored index itself is malformed when only the transport text
            # failed to decode.
            raise ArtifactStoreError(
                f"could not parse the read response for {CATALOG_PATH} as JSON"
            ) from exc
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

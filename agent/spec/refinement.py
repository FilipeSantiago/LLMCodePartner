"""Conversation-scoped OpenSpec refinement drafts and identity-preserving apply."""

import hashlib
import json
import re
from dataclasses import replace
from uuid import uuid4

from agent.routing.persistence import RoutingPersistence
from agent.spec.artifact_store import ArtifactSelectionError, ArtifactStore
from agent.spec.command import create_change_id
from agent.spec.models import OpenSpecChange, Task, WorkPackage, render_response, render_tasks
from agent.spec.openspec_adapter import OpenSpecAdapter


class RefinementError(RuntimeError):
    pass


# A spec-mode draft is a refinement session with no canonical object behind it.
NEW_CHANGE_KIND = "new_change"
NEW_CHANGE_TARGET = "DRAFT"


def _fingerprint(change: OpenSpecChange) -> str:
    value = json.dumps(change.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def _task_plan(task: Task) -> tuple:
    return (
        task.title, task.description, tuple(task.acceptance_criteria), task.complexity,
        task.reasoning, task.context, task.preferred_capability, tuple(task.depends_on),
    )


class RefinementService:
    def __init__(self, artifacts: ArtifactStore, adapter: OpenSpecAdapter,
                 persistence: RoutingPersistence):
        self.artifacts = artifacts
        self.adapter = adapter
        self.persistence = persistence

    async def start(self, conversation_id: str, target_id: str) -> str:
        kind, change, target = await self.artifacts.object_by_id(target_id)
        session_id = str(uuid4())
        await self.persistence.create_refinement(
            session_id, conversation_id, target_id, kind, change.id, change.to_dict())
        return "\n".join((
            f"Refinement draft started for {target_id} ({kind.replace('_', ' ')}).",
            "Canonical OpenSpec has not changed.",
            f"Current scope: {getattr(target, 'title', target_id)}.",
            "Discuss the desired changes, then use `/apply` to commit the draft.",
        ))

    async def start_draft(self, conversation_id: str, request: str) -> str:
        """Generate a new change as a conversation draft; nothing is written yet."""
        draft = self._provisional_ids(await self.adapter.generate_change(request))
        await self.persistence.create_refinement(
            str(uuid4()), conversation_id, NEW_CHANGE_TARGET, NEW_CHANGE_KIND, "",
            draft.to_dict())
        return self._draft_text("Draft created", draft)

    async def update(self, conversation_id: str, instruction: str) -> str:
        session = await self.persistence.active_refinement(conversation_id)
        if session is None:
            raise RefinementError("no active refinement; start one with /refine <object>")
        current = OpenSpecChange.from_dict(session["draft"])
        if session["target_kind"] == NEW_CHANGE_KIND:
            refined = self._provisional_ids(await self.adapter.refine_change(
                current, "the whole draft change (all work packages and tasks)", instruction))
            await self.persistence.update_refinement(session["session_id"], refined.to_dict())
            return self._draft_text("Draft updated", refined)
        refined = await self.adapter.refine_change(
            current, session["target_id"], instruction)
        self._check_scope(current, refined, session["target_kind"], session["target_id"])
        await self.persistence.update_refinement(session["session_id"], refined.to_dict())
        return self._summary(current, refined, session["target_id"])

    async def apply(self, conversation_id: str) -> str:
        session = await self.persistence.active_refinement(conversation_id)
        if session is None:
            raise RefinementError("no active refinement to apply")
        if session["target_kind"] == NEW_CHANGE_KIND:
            draft = OpenSpecChange.from_dict(session["draft"])
            change = await self.artifacts.create_change(draft, create_change_id(draft.title))
            await self.persistence.finish_refinement(session["session_id"], "applied")
            return render_response(change)
        _, canonical, _ = await self.artifacts.object_by_id(session["target_id"])
        if _fingerprint(canonical) != session["base_fingerprint"]:
            raise RefinementError(
                "canonical OpenSpec changed since refinement began; start /refine again to rebase")
        draft = OpenSpecChange.from_dict(session["draft"])
        reconciled = self._reconcile(canonical, draft)
        await self.artifacts.update_change(reconciled)
        await self.persistence.finish_refinement(session["session_id"], "applied")
        return self._summary(canonical, reconciled, session["target_id"], applied=True)

    async def discard(self, conversation_id: str) -> str:
        session = await self.persistence.active_refinement(conversation_id)
        if session is None:
            raise RefinementError("no active draft to discard")
        await self.persistence.finish_refinement(session["session_id"], "discarded")
        target = ("the new change" if session["target_kind"] == NEW_CHANGE_KIND
                  else session["target_id"])
        return f"Discarded the draft for {target}. Canonical OpenSpec has not changed."

    @staticmethod
    def _provisional_ids(draft: OpenSpecChange) -> OpenSpecChange:
        """Label draft work positionally; create_change renumbers it the same way on save."""
        return replace(draft, work_packages=[
            replace(package, id=f"WP{number}", tasks=[
                replace(task, id=f"WP{number}-T{task_number}")
                for task_number, task in enumerate(package.tasks, start=1)
            ])
            for number, package in enumerate(draft.work_packages, start=1)
        ])

    @staticmethod
    def _draft_text(state: str, draft: OpenSpecChange) -> str:
        return "\n".join([
            f"{state} — not saved. Work-package and task IDs are provisional; "
            "final IDs are assigned on `/apply`.",
            "", draft.summary, "", render_tasks(draft), "",
            "Keep chatting to revise it, `/apply` to save it, or `/discard` to drop it.",
        ])

    @staticmethod
    def _check_scope(before: OpenSpecChange, after: OpenSpecChange,
                     kind: str, target_id: str) -> None:
        if kind == "epic":
            return
        before_packages = {package.id: package for package in before.work_packages}
        after_packages = {package.id: package for package in after.work_packages}
        package_id = target_id.split("-T", 1)[0] if kind == "task" else target_id
        for item_id, package in before_packages.items():
            if item_id != package_id and after_packages.get(item_id) != package:
                raise RefinementError(
                    f"draft changed {item_id}, outside refinement target {target_id}")
        if kind == "task":
            before_package = before_packages.get(package_id)
            after_package = after_packages.get(package_id)
            if before_package is None or after_package is None:
                raise RefinementError("refinement removed the target task's work package")
            before_tasks = {task.id: task for task in before_package.tasks}
            after_tasks = {task.id: task for task in after_package.tasks}
            for task_id, task in before_tasks.items():
                if task_id != target_id and after_tasks.get(task_id) != task:
                    raise RefinementError(
                        f"draft changed {task_id}, outside refinement target {target_id}")

    @classmethod
    def _reconcile(cls, canonical: OpenSpecChange,
                   draft: OpenSpecChange) -> OpenSpecChange:
        old_packages = {package.id: package for package in canonical.work_packages}
        reconciled_packages = []
        for package in draft.work_packages:
            old_package = old_packages.get(package.id)
            if old_package is None:
                reconciled_packages.append(package)
                continue
            old_tasks = {task.id: task for task in old_package.tasks}
            supplied_ids = {task.id for task in package.tasks if task.id}
            suffixes = [int(match.group(1)) for task in old_package.tasks
                        if (match := re.fullmatch(re.escape(package.id) + r"-T(\d+)", task.id))]
            next_number = max(suffixes, default=0) + 1
            tasks = []
            for task in package.tasks:
                if not task.id:
                    task = replace(
                        task, id=f"{package.id}-T{next_number}", status="pending",
                        completed=False, planning_revision=package.planning_revision + 1,
                        reconciliation=[*task.reconciliation, {
                            "disposition": "new", "reason": "added by refinement",
                        }],
                    )
                    next_number += 1
                    tasks.append(task)
                    continue
                previous = old_tasks.get(task.id)
                if previous is None:
                    tasks.append(task)
                    continue
                if _task_plan(previous) == _task_plan(task):
                    tasks.append(replace(
                        task, completed=previous.completed, status=previous.status,
                        completion_review=previous.completion_review,
                        planning_revision=previous.planning_revision,
                        reconciliation=previous.reconciliation,
                    ))
                    continue
                disposition = "needs_reexecution" if previous.completed else "modified"
                tasks.append(replace(
                    task, completed=False,
                    status="needs_reexecution" if previous.completed else "pending",
                    completion_review=previous.completion_review,
                    planning_revision=previous.planning_revision + 1,
                    reconciliation=[*previous.reconciliation, {
                        "disposition": disposition,
                        "reason": "task plan affected by applied refinement",
                    }],
                ))
            for task in old_package.tasks:
                if task.id not in supplied_ids:
                    tasks.append(replace(
                        task, status="cancelled", completed=False,
                        planning_revision=task.planning_revision + 1,
                        reconciliation=[*task.reconciliation, {
                            "disposition": "cancelled",
                            "reason": "made obsolete by applied refinement",
                        }],
                    ))
            changed = package != old_package
            reconciled_packages.append(replace(
                package, tasks=tasks,
                planning_revision=old_package.planning_revision + (1 if changed else 0),
                review=None if changed else old_package.review,
                status="planned" if changed else old_package.status,
            ))
        return replace(draft, work_packages=reconciled_packages)

    @staticmethod
    def _summary(before: OpenSpecChange, after: OpenSpecChange,
                 target_id: str, applied: bool = False) -> str:
        old = {task.id: task for package in before.work_packages for task in package.tasks}
        new = {task.id: task for package in after.work_packages for task in package.tasks}
        changes = []
        for task_id in sorted(set(old) | set(new)):
            if task_id not in old:
                changes.append(f"- {task_id or '(new task)'}: new")
            elif task_id not in new or new[task_id].status == "cancelled":
                changes.append(f"- {task_id}: cancelled")
            elif _task_plan(old[task_id]) != _task_plan(new[task_id]):
                suffix = "needs re-execution" if old[task_id].completed else "modified"
                changes.append(f"- {task_id}: {suffix}")
            else:
                changes.append(f"- {task_id}: unchanged")
        state = "Applied to canonical OpenSpec" if applied else "Updated refinement draft"
        return "\n".join([
            f"{state} for {target_id}.",
            "Canonical OpenSpec has not changed." if not applied else "Task identities and history were reconciled.",
            "", "Proposed task impact:", *(changes or ["- No task changes."]),
            "", "Use `/apply` to commit this draft." if not applied else "Refinement session closed.",
        ])

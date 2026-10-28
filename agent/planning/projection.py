"""Validate a planner draft and freeze it into an approved plan (no model call)."""

import hashlib
import json
import time
from dataclasses import replace

from agent.planning.models import (
    ApprovedWPPlan, PlannerFinding, ReviewerContext, TaskContext, WPPlanDraft,
    WPSharedContext,
)


def validate_draft(draft: WPPlanDraft, task_ids: list[str],
                   write_task_ids: list[str] | None = None) -> list[str]:
    """Return the gaps that keep this draft from being approvable (empty when complete).

    `write_task_ids` are the tasks that change files; each needs exact edit ranges."""
    gaps = []
    if not draft.summary:
        gaps.append("summary is empty")
    if not draft.shared.objective:
        gaps.append("shared.objective is empty")
    expected = [task_id.upper() for task_id in task_ids]
    writing = {task_id.upper() for task_id in write_task_ids or ()}
    for task_id in expected:
        context = draft.tasks.get(task_id)
        if context is None:
            gaps.append(f"tasks.{task_id} is missing")
            continue
        if not context.objective:
            gaps.append(f"tasks.{task_id}.objective is empty")
        if not context.approach:
            gaps.append(f"tasks.{task_id}.approach is empty")
        if task_id in writing and not context.edits:
            gaps.append(f"tasks.{task_id}.edits is empty: name the exact file lines to change "
                        "(null lines for a new file)")
        protected = set(context.do_not_modify) | set(draft.shared.do_not_modify)
        blocked = sorted({path for path in context.edit_files() if path in protected})
        if blocked:
            gaps.append(f"tasks.{task_id}.edits target do-not-modify paths: "
                        + ", ".join(blocked))
    unknown = sorted(set(draft.tasks) - set(expected))
    if unknown:
        gaps.append(f"tasks names unknown task ids: {', '.join(unknown)}")
    if not draft.reviewer.expected_behavior:
        gaps.append("reviewer.expected_behavior is empty")
    if not draft.reviewer.validation_expectations:
        gaps.append("reviewer.validation_expectations is empty")
    ids = [finding.id for finding in draft.findings]
    if any(not finding_id for finding_id in ids):
        gaps.append("every finding needs an id")
    duplicates = sorted({finding_id for finding_id in ids if finding_id and ids.count(finding_id) > 1})
    if duplicates:
        gaps.append(f"duplicate finding ids: {', '.join(duplicates)}")
    known = set(ids)
    references = [*draft.shared.finding_ids,
                  *(finding_id for context in draft.tasks.values()
                    for finding_id in context.finding_ids)]
    dangling = sorted(set(references) - known)
    if dangling:
        gaps.append(f"finding_ids reference unknown findings: {', '.join(dangling)}")
    stray = sorted({task_id for finding in draft.findings for task_id in finding.task_ids
                    if task_id not in expected})
    if stray:
        gaps.append(f"findings reference unknown task ids: {', '.join(stray)}")
    unknown_discovery = sorted({task_id for entry in draft.discovery
                                for task_id in entry.task_ids if task_id not in expected})
    if unknown_discovery:
        gaps.append("discovery references unknown task ids: " + ", ".join(unknown_discovery))
    return gaps


def _task_finding_ids(draft: WPPlanDraft) -> dict[str, list[str]]:
    assigned: dict[str, list[str]] = {task_id: list(context.finding_ids)
                                      for task_id, context in draft.tasks.items()}
    for finding in draft.findings:
        for task_id in finding.task_ids:
            if task_id in assigned and finding.id not in assigned[task_id]:
                assigned[task_id].append(finding.id)
    return assigned


def project(draft: WPPlanDraft, *, plan_id: str, session_id: str, change_id: str,
            wp_fingerprint: str, wp_complexity: str, planner_model: str,
            provenance: dict | None = None) -> ApprovedWPPlan:
    """Freeze a validated draft: give each consumer exactly the findings it needs."""
    by_id = {finding.id: finding for finding in draft.findings}
    order = {finding.id: index for index, finding in enumerate(draft.findings)}

    def resolve(ids) -> list[PlannerFinding]:
        return [by_id[finding_id] for finding_id in sorted(set(ids), key=order.get)
                if finding_id in by_id]

    per_task = _task_finding_ids(draft)
    tasks: dict[str, TaskContext] = {
        task_id: replace(context, finding_ids=sorted(set(per_task[task_id]), key=order.get),
                         findings=resolve(per_task[task_id]))
        for task_id, context in draft.tasks.items()
    }
    usage: dict[str, int] = {}
    for ids in per_task.values():
        for finding_id in set(ids):
            usage[finding_id] = usage.get(finding_id, 0) + 1
    unassigned = [finding.id for finding in draft.findings
                  if not usage.get(finding.id)]
    shared_ids = [*draft.shared.finding_ids,
                  *(finding_id for finding_id, count in usage.items() if count > 1),
                  *unassigned]
    shared: WPSharedContext = replace(
        draft.shared, finding_ids=sorted(set(shared_ids), key=order.get),
        findings=resolve(shared_ids))
    reviewer_ids = [finding.id for finding in draft.findings
                    if finding.kind in {"risk", "assumption", "constraint", "architecture"}]
    reviewer: ReviewerContext = replace(
        draft.reviewer, approved_plan=draft.plan_markdown, findings=resolve(reviewer_ids))
    plan = ApprovedWPPlan(
        plan_id=plan_id, session_id=session_id, change_id=change_id, wp_id=draft.wp_id,
        wp_fingerprint=wp_fingerprint, wp_complexity=wp_complexity, revision=draft.revision,
        planner_model=planner_model, summary=draft.summary, plan_markdown=draft.plan_markdown,
        shared=shared, tasks=tasks, reviewer=reviewer, findings=list(draft.findings),
        discovery=list(draft.discovery),
        decisions=list(draft.decisions), provenance=dict(provenance or {}),
    )
    content = json.dumps(plan.to_dict(), sort_keys=True, ensure_ascii=False)
    return replace(plan, approved_at=time.time(),
                   fingerprint=hashlib.sha256(content.encode()).hexdigest())

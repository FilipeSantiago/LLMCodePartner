"""Text renderings of planning state for the user, the coder, and the reviewer."""

from agent.planning.models import (
    ApprovedWPPlan, PlannerFinding, ReviewerContext, TaskContext, WPPlanDraft,
)

_MAX_INSPECTED = 10
APPROVE_REPLY = "approve"
DISCUSS_REPLY = "discuss"


def render_plan(draft: WPPlanDraft, *, wp_complexity: str, planner_label: str,
                gaps: list[str], token: str | None,
                queued_wp_ids: list[str] | tuple[str, ...] = ()) -> str:
    lines = [f"## Plan for {draft.wp_id} (revision {draft.revision})",
             f"_Planner: {planner_label}; WP complexity: {wp_complexity}. "
             "Nothing has been changed in the project._", "",
             draft.summary, "", draft.plan_markdown]
    if draft.decisions:
        lines += ["", "**Decisions**", *(f"- {item}" for item in draft.decisions)]
    if draft.open_questions:
        lines += ["", "**Open questions**", *(f"- {item}" for item in draft.open_questions)]
    inspected = [finding for finding in draft.findings
                 if finding.validated and finding.evidence][:_MAX_INSPECTED]
    if inspected:
        lines += ["", "**What I inspected and why**",
                  *(f"- {_evidence_paths(finding)} — {finding.statement}"
                    for finding in inspected)]
    if queued_wp_ids:
        lines += ["", f"After {draft.wp_id} is implemented and reviewed, I will plan "
                  + ", ".join(queued_wp_ids) + " next, against the updated code."]
    lines += ["", "**Per-task context prepared for the coders:** "
              + ", ".join(f"{task_id} ({len(context.relevant_files)} files, "
                          f"{len(context.finding_ids)} findings)"
                          for task_id, context in sorted(draft.tasks.items()))]
    if gaps:
        lines += ["", "The structured plan is incomplete, so it cannot be approved yet:",
                  *(f"- {gap}" for gap in gaps), "",
                  "Tell me how to proceed, or ask me to investigate further."]
        return "\n".join(lines)
    lines += ["", "**What next?**",
              f"1. Reply `{APPROVE_REPLY}` to approve this plan and start implementing "
              f"{draft.wp_id}.",
              f"2. Reply `{DISCUSS_REPLY}` (or just tell me what to change) to keep refining "
              "the plan."]
    if token:
        lines += ["", f"_(`confirm {token}` also approves this revision.)_"]
    return "\n".join(lines)


def _evidence_paths(finding: PlannerFinding) -> str:
    return ", ".join(dict.fromkeys(f"`{item['path']}`" for item in finding.evidence))


def _list(title: str, items: list[str]) -> list[str]:
    return [f"{title}:", *(f"- {item}" for item in items)] if items else []


def _findings(findings: list[PlannerFinding]) -> list[str]:
    lines = []
    for finding in findings:
        where = ", ".join(item["path"] + (f" ({item['symbol']})" if item.get("symbol") else "")
                          for item in finding.evidence)
        status = "verified" if finding.validated else "assumption"
        lines.append(f"- [{finding.id}, {finding.kind}, {status}] {finding.statement}"
                     + (f" — {where}" if where else ""))
    return ["Planner findings:", *lines] if lines else []


def coder_context(plan: ApprovedWPPlan, task_id: str) -> str:
    """Approved WP context plus this task's context, for the coder prompt."""
    shared = plan.shared
    task: TaskContext | None = plan.tasks.get(task_id)
    lines = ["BEGIN APPROVED WP PLAN CONTEXT",
             f"WP SHARED CONTEXT ({plan.wp_id}, plan {plan.plan_id})",
             f"Objective: {shared.objective}",
             *_list("Requirements", shared.requirements),
             *_list("Architecture", shared.architecture),
             *_list("Relevant files", shared.relevant_files),
             *_list("Symbols", shared.symbols),
             *_list("Constraints", shared.constraints),
             *_list("Planning decisions", shared.decisions),
             *_list("Validated assumptions", shared.validated_assumptions),
             *_list("Do not modify", shared.do_not_modify),
             *_list("WP test strategy", shared.test_strategy),
             *_findings(shared.findings)]
    if task is not None:
        lines += ["", f"TASK CONTEXT ({task.task_id})",
                  f"Objective: {task.objective}",
                  f"Expected approach: {task.approach}",
                  *_list("Requirements", task.requirements),
                  *_list("Relevant files", task.relevant_files),
                  *_list("Symbols", task.symbols),
                  *_list("Existing implementation", task.existing_implementation),
                  *_list("Depends on", task.depends_on),
                  *_list("Constraints", task.constraints),
                  *_list("Do not modify", task.do_not_modify),
                  *_list("Expected tests / validation", task.expected_tests),
                  *_findings(task.findings)]
    lines.append("END APPROVED WP PLAN CONTEXT")
    return "\n".join(lines)


def reviewer_context(plan: ApprovedWPPlan, events: list[dict],
                     task_ids: list[str] | None = None) -> str:
    """Frozen reviewer context plus what happened during execution."""
    reviewer: ReviewerContext = plan.reviewer
    lines = [f"Plan {plan.plan_id} for {plan.wp_id} (revision {plan.revision}, "
             f"planner {plan.planner_model})",
             f"Objective: {reviewer.objective}",
             "Approved plan:", reviewer.approved_plan,
             *_list("Architecture", reviewer.architecture),
             *_list("Expected behavior", reviewer.expected_behavior),
             *_list("Task boundaries", reviewer.task_boundaries),
             *_list("Dependencies", reviewer.dependencies),
             *_list("Relevant files", reviewer.relevant_files),
             *_list("Risks", reviewer.risks),
             *_list("Assumptions", reviewer.assumptions),
             *_list("Validation expectations", reviewer.validation_expectations),
             *_findings(reviewer.findings)]
    selected = [event for event in events
                if task_ids is None or event.get("task_id") in task_ids]
    if selected:
        lines.append("Execution record (from CodePartner, per task):")
        for event in selected:
            lines.append(f"- {event['task_id']} {event['kind']}: {event['summary']}")
    return "\n".join(lines)

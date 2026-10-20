"""Text renderings of planning state for the user, the coder, and the reviewer."""

from agent.planning.models import (
    ApprovedWPPlan, DiscoveryFile, PlannerFinding, ReviewerContext, TaskContext, WPPlanDraft,
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


def _discovery(entries: list[DiscoveryFile]) -> list[str]:
    if not entries:
        return []
    lines = ["Discovery leads (navigation only; verify current source and line ranges):"]
    for entry in entries:
        symbols = ", ".join(
            symbol.name + (f" lines {symbol.start_line}–{symbol.end_line}"
                           if symbol.start_line is not None and symbol.end_line is not None
                           else "") for symbol in entry.relevant_symbols)
        lines.append(f"- {entry.file}: {entry.reason}" + (f"; {symbols}" if symbols else ""))
    return lines


_PRIOR_KINDS = ("decisions", "changed_files", "deviations", "unresolved")
_PRIOR_LABELS = {"decisions": "Result", "changed_files": "Changed files",
                 "deviations": "Deviations from plan", "unresolved": "Unresolved"}


def _prior_results(events: list[dict], task_id: str,
                   depends_on: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Latest handoff facts per earlier task, so later coders build on them."""
    latest: dict[tuple[str, str], dict] = {}
    order: list[str] = []
    for event in events:
        prior = event.get("task_id")
        if not prior or prior == task_id or event.get("kind") not in _PRIOR_KINDS:
            continue
        if prior not in order:
            order.append(prior)
        latest[(prior, event["kind"])] = event
    lines = []
    for prior in order:
        lines.append(f"{prior}" + (" (dependency)" if prior in depends_on else "") + ":")
        lines += [f"- {_PRIOR_LABELS[kind]}: {latest[(prior, kind)]['summary']}"
                  for kind in _PRIOR_KINDS if (prior, kind) in latest]
    if not lines:
        return []
    return ["", "EARLIER TASK RESULTS (this job; already done, build on them rather than "
            "re-deriving them)", *lines]


def _unique(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(item for group in groups for item in group))


def _task_findings(plan: ApprovedWPPlan, task: TaskContext | None,
                   task_id: str) -> list[PlannerFinding]:
    """Shared and task findings that concern this task (unscoped ones apply to all)."""
    seen: dict[str, PlannerFinding] = {}
    for finding in [*plan.shared.findings, *(task.findings if task else [])]:
        if finding.id not in seen and (not finding.task_ids or task_id in finding.task_ids):
            seen[finding.id] = finding
    return list(seen.values())


def coder_context(plan: ApprovedWPPlan, task_id: str,
                  prior_events: list[dict] | tuple[dict, ...] = (),
                  description: str = "") -> str:
    """Task-scoped execution context: only the parts of the approved WP plan that bear
    on `task_id`, plus earlier tasks' results. Other tasks' requirements, files and
    findings stay out of the prompt; the coder can still search for more."""
    shared = plan.shared
    task: TaskContext | None = plan.tasks.get(task_id)
    files = (task.relevant_files if task and task.relevant_files else shared.relevant_files)
    symbols = (task.symbols if task and task.symbols else shared.symbols)
    lines = ["BEGIN APPROVED WP PLAN CONTEXT",
             f"RELEVANT CONTEXT ({task_id}; {plan.wp_id}, plan {plan.plan_id})",
             f"Work package objective: {shared.objective}",
             *_list("Architecture", shared.architecture),
             *_list("Planning decisions", shared.decisions),
             *_list("Existing implementation", task.existing_implementation if task else []),
             *_list("Relevant files", files),
             *_list("Symbols", symbols),
             *_list("Depends on", task.depends_on if task else []),
             *_findings(_task_findings(plan, task, task_id)),
             *_discovery([entry for entry in plan.discovery
                          if not entry.task_ids or task_id in entry.task_ids]),
             *_prior_results(list(prior_events), task_id, task.depends_on if task else ())]
    if task is not None:
        objective = (task.objective if task.objective.strip() != description.strip() else "")
        changes = [*([f"Objective: {objective}"] if objective else []),
                   *([f"Expected approach: {task.approach}"] if task.approach else []),
                   *_list("Requirements", task.requirements)]
        if changes:
            lines += ["", "REQUIRED CHANGES", *changes]
    constraints = _unique(shared.constraints, task.constraints if task else [])
    protected = _unique(shared.do_not_modify, task.do_not_modify if task else [])
    if constraints or protected:
        lines += ["", "CONSTRAINTS", *_list("Constraints", constraints),
                  *_list("Do not modify", protected)]
    validation = (task.expected_tests if task and task.expected_tests
                  else shared.test_strategy)
    if validation:
        lines += ["", "VALIDATION", *_list("Expected tests / validation", validation)]
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
             *_findings(reviewer.findings),
             *_discovery([entry for entry in plan.discovery
                          if task_ids is None or not entry.task_ids
                          or any(task_id in task_ids for task_id in entry.task_ids)])]
    selected = [event for event in events
                if task_ids is None or event.get("task_id") in task_ids]
    if selected:
        lines.append("Execution record (from CodePartner, per task):")
        for event in selected:
            lines.append(f"- {event['task_id']} {event['kind']}: {event['summary']}")
    return "\n".join(lines)

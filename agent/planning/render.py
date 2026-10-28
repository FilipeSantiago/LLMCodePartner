"""Text renderings of planning state for the user, the coder, and the reviewer."""

from agent.planning.models import (
    ApprovedWPPlan, DiscoveryFile, EditRange, PlannerFinding, ReferenceRange, ReviewerContext,
    TaskContext, WPPlanDraft,
)

_MAX_INSPECTED = 10
APPROVE_REPLY = "approve"
DISCUSS_REPLY = "discuss"
EXPLAIN_REPLY = "explain"
# Lines added above and below a planner edit range when none is configured.
DEFAULT_MARGIN = 10


def widen(start: int, end: int, margin: int) -> tuple[int, int]:
    """A planner range with `margin` lines above and below, never before line 1."""
    return max(1, start - margin), end + margin


def edit_line(edit: EditRange, margin: int = DEFAULT_MARGIN) -> str:
    if edit.start_line is None or edit.end_line is None:
        return f"- {edit.file} (new file) — {edit.change}"
    start, end = widen(edit.start_line, edit.end_line, margin)
    symbol = f" {edit.symbol}" if edit.symbol else ""
    planned = f" (planned lines {edit.start_line}-{edit.end_line})" if margin else ""
    return f"- {edit.file}:{start}-{end}{symbol}{planned} — {edit.change}"


def reference_line(ref: ReferenceRange, margin: int = DEFAULT_MARGIN) -> str:
    where = ref.file
    if ref.start_line is not None and ref.end_line is not None:
        start, end = widen(ref.start_line, ref.end_line, margin)
        where += f":{start}-{end}"
    return f"- {where}" + (f" — {ref.why}" if ref.why else "")


def affected_files(tasks: dict[str, TaskContext]) -> list[str]:
    """Every file the plan changes, marked new or modified; older plans list their files."""
    files: dict[str, str] = {}
    for _, task in sorted(tasks.items()):
        for edit in task.edits:
            new = edit.start_line is None
            files[edit.file] = "new" if new and files.get(edit.file, "new") == "new" else "modified"
        if not task.edits:
            for path in task.relevant_files:
                files.setdefault(path, "")
    return [f"`{path}`" + (f" ({kind})" if kind else "") for path, kind in files.items()]


def render_plan(draft: WPPlanDraft, *, wp_complexity: str, planner_label: str,
                gaps: list[str], token: str | None,
                queued_wp_ids: list[str] | tuple[str, ...] = (), detail: bool = False,
                margin: int = DEFAULT_MARGIN) -> str:
    """The plan for the user: tasks and affected files by default; `detail` adds the
    approach, decisions, inspected evidence and every edit and reference range."""
    lines = [f"## Plan for {draft.wp_id} (revision {draft.revision})"
             + (" — details" if detail else ""),
             f"_Planner: {planner_label}; WP complexity: {wp_complexity}. "
             "Nothing has been changed in the project._", "", draft.summary, ""]
    if detail:
        lines.append(draft.plan_markdown)
        for task_id, task in sorted(draft.tasks.items()):
            if task.edits or task.references:
                lines += ["", f"**{task_id} lines**",
                          *(edit_line(edit, 0) for edit in task.edits),
                          *(["References:", *(reference_line(ref, 0) for ref in task.references)]
                            if task.references else [])]
    else:
        lines += [f"- **{task_id}** — {task.objective}"
                  for task_id, task in sorted(draft.tasks.items())]
        files = affected_files(draft.tasks)
        if files:
            lines += ["", "**Affected files**", *(f"- {item}" for item in files)]
    if detail and draft.decisions:
        lines += ["", "**Decisions**", *(f"- {item}" for item in draft.decisions)]
    if draft.open_questions:
        lines += ["", "**Open questions**", *(f"- {item}" for item in draft.open_questions)]
    inspected = [finding for finding in draft.findings
                 if finding.validated and finding.evidence][:_MAX_INSPECTED] if detail else []
    if inspected:
        lines += ["", "**What I inspected and why**",
                  *(f"- {_evidence_paths(finding)} — {finding.statement}"
                    for finding in inspected)]
    if queued_wp_ids:
        lines += ["", f"After {draft.wp_id} is implemented and reviewed, I will plan "
                  + ", ".join(queued_wp_ids) + " next, against the updated code."]
    if detail:
        lines += ["", "**Per-task context prepared for the coders:** "
                  + ", ".join(f"{task_id} ({len(context.edits) or len(context.relevant_files)}"
                              f" {'edit ranges' if context.edits else 'files'}, "
                              f"{len(context.references)} references, "
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
    if not detail:
        lines.append(f"3. Reply `{EXPLAIN_REPLY}` to see the approach, decisions and the exact "
                     "lines each task changes.")
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


_PRIOR_KINDS = ("decisions", "changed_files", "deviations", "unresolved", "tests",
                "task_handoff")
_PRIOR_LABELS = {"decisions": "Result", "changed_files": "Changed files",
                 "deviations": "Deviations from plan", "unresolved": "Unresolved",
                 "tests": "Captured verification"}


def _prior_results(events: list[dict], task_id: str,
                   depends_on: list[str] | tuple[str, ...] = (),
                   relevant_files: list[str] | tuple[str, ...] = ()) -> list[str]:
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
        handoff = latest.get((prior, "task_handoff"), {}).get("payload") or {}
        changed = latest.get((prior, "changed_files"), {}).get("payload") or []
        paths = {path for path in changed if isinstance(path, str)} if isinstance(
            changed, list) else set()
        if isinstance(handoff, dict):
            paths.update(path for path in (handoff.get("files") or [])
                         if isinstance(path, str))
        if prior not in depends_on and not paths.intersection(relevant_files):
            continue
        detail = [f"{prior}" + (" (dependency)" if prior in depends_on else "") + ":"]
        detail += [f"- {_PRIOR_LABELS[kind]}: {latest[(prior, kind)]['summary'][:350]}"
                   for kind in _PRIOR_KINDS if kind != "task_handoff" and (prior, kind) in latest]
        if isinstance(handoff, dict):
            for key, label in (("files", "File leads"), ("symbols", "Interfaces"),
                               ("decisions", "Decisions"), ("dependencies", "Dependencies"),
                               ("verification", "Verification"),
                               ("constraints", "Constraints")):
                values = handoff.get(key)
                if isinstance(values, list) and values:
                    detail.append(f"- {label}: " + "; ".join(str(value) for value in values)[:350])
        kept = []
        for line in detail:
            if sum(len(value) + 1 for value in kept) + len(line) > 1800:
                break
            kept.append(line)
        block = "\n".join(kept)
        if sum(len(line) for line in lines) + len(block) > 6000:
            break
        lines.append(block)
    if not lines:
        return []
    return ["", "EARLIER TASK RESULTS (this job; already done, build on them rather than "
            "re-deriving them)", *lines]


def prior_task_context(events: list[dict], task_id: str,
                       depends_on: list[str] | tuple[str, ...],
                       relevant_files: list[str] | tuple[str, ...]) -> str:
    """Bounded, task-relevant navigation context independent of an approved plan."""
    return "\n".join(_prior_results(events, task_id, depends_on, relevant_files))


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


def task_files(plan: ApprovedWPPlan, task_id: str) -> list[str]:
    """The files a task changes: its edit ranges' files; plans without edit ranges fall
    back to the task's relevant files, then the WP's shared files."""
    task = plan.tasks.get(task_id)
    if task and task.edits:
        return task.edit_files()
    return list(task.relevant_files if task and task.relevant_files
                else plan.shared.relevant_files)


def task_read_only_files(plan: ApprovedWPPlan, task_id: str) -> list[str]:
    """Paths a task reads but must not modify: its reference files plus its own and the
    WP's do-not-modify lists."""
    task = plan.tasks.get(task_id)
    return list(dict.fromkeys([*(task.reference_files() if task else []),
                               *(task.do_not_modify if task else []),
                               *plan.shared.do_not_modify]))


def _scoped_coder_context(plan: ApprovedWPPlan, task: TaskContext,
                          prior_events: list[dict], description: str, margin: int) -> str:
    """The plan's exact edit and reference ranges, with the objective, approach and
    checks; the ranges replace the architecture, symbol and discovery blocks."""
    shared = plan.shared
    objective = task.objective if task.objective.strip() != description.strip() else ""
    lines = ["BEGIN APPROVED WP PLAN CONTEXT",
             f"RELEVANT CONTEXT ({task.task_id}; {plan.wp_id}, plan {plan.plan_id})",
             *([f"Objective: {objective}"] if objective else []),
             *([f"Approach: {task.approach}"] if task.approach else []),
             *_list("Requirements", task.requirements),
             *_list("Architecture", shared.architecture),
             f"EDIT (your files; planned lines widened by {margin}):",
             *(edit_line(edit, margin) for edit in task.edits)]
    if task.references:
        lines += ["REFERENCES (read-only context):",
                  *(reference_line(ref, margin) for ref in task.references)]
    findings = _task_findings(plan, task, task.task_id)
    if findings:
        lines += ["Planner facts (assumptions are yours to verify):",
                  *(f"- [{finding.id}" + ("" if finding.validated else ", assumption")
                    + f"] {finding.statement}" for finding in findings)]
    lines += [*_list("Depends on", task.depends_on),
              *_prior_results(list(prior_events), task.task_id, task.depends_on,
                              task_files(plan, task.task_id))]
    constraints = _unique(shared.constraints, task.constraints)
    protected = _unique(shared.do_not_modify, task.do_not_modify)
    if constraints or protected:
        lines += ["", "CONSTRAINTS", *_list("Constraints", constraints),
                  *_list("Do not modify", protected)]
    validation = task.expected_tests or shared.test_strategy
    if validation:
        lines += ["", "VALIDATION", *_list("Expected tests / validation", validation)]
    lines.append("END APPROVED WP PLAN CONTEXT")
    return "\n".join(lines)


def coder_context(plan: ApprovedWPPlan, task_id: str,
                  prior_events: list[dict] | tuple[dict, ...] = (),
                  description: str = "", margin: int = DEFAULT_MARGIN) -> str:
    """Task-scoped execution context: only the parts of the approved WP plan that bear
    on `task_id`, plus earlier tasks' results. Other tasks' requirements, files and
    findings stay out of the prompt."""
    shared = plan.shared
    task: TaskContext | None = plan.tasks.get(task_id)
    if task is not None and task.edits:
        return _scoped_coder_context(plan, task, list(prior_events), description, margin)
    files = task_files(plan, task_id)
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
             *_prior_results(list(prior_events), task_id,
                             task.depends_on if task else (), files)]
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


def _scoped_reviewer_context(plan: ApprovedWPPlan, task_ids: list[str] | None,
                             margin: int) -> list[str]:
    """The reviewed tasks' planned edit and reference ranges."""
    lines = []
    for task_id, task in sorted(plan.tasks.items()):
        if task_ids is not None and task_id not in task_ids or not task.edits:
            continue
        lines += [f"{task_id} planned edits (lines widened by {margin}):",
                  *(edit_line(edit, margin) for edit in task.edits)]
        if task.references:
            lines += [f"{task_id} references:",
                      *(reference_line(ref, margin) for ref in task.references)]
    return lines


def reviewer_context(plan: ApprovedWPPlan, events: list[dict],
                     task_ids: list[str] | None = None, margin: int = DEFAULT_MARGIN) -> str:
    """Frozen reviewer context plus what happened during execution."""
    reviewer: ReviewerContext = plan.reviewer
    ranges = _scoped_reviewer_context(plan, task_ids, margin)
    if ranges:
        lines = [f"Plan {plan.plan_id} for {plan.wp_id} (revision {plan.revision}, "
                 f"planner {plan.planner_model})",
                 f"Objective: {reviewer.objective}",
                 "Approved plan:", reviewer.approved_plan,
                 *_list("Expected behavior", reviewer.expected_behavior),
                 *_list("Task boundaries", reviewer.task_boundaries),
                 *_list("Risks", reviewer.risks),
                 *_list("Validation expectations", reviewer.validation_expectations),
                 *ranges]
        selected = [event for event in events
                    if task_ids is None or event.get("task_id") in task_ids]
        if selected:
            lines.append("Execution record (from CodePartner, per task):")
            lines += [f"- {event['task_id']} {event['kind']}: {event['summary']}"
                      for event in selected]
        return "\n".join(lines)
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

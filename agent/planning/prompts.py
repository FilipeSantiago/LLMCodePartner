"""Planner prompts. The planner returns its plan and structured state together."""

import json
from dataclasses import asdict

from agent.planning.knowledge_base import KB_READING_GUIDANCE
from agent.spec.models import OpenSpecChange, WorkPackage
from agent.task_contracts.policy import DISCOVERY_CAPABILITIES

_SHAPE = {
    "summary": "Two or three sentences: what this WP will deliver and how",
    "plan_markdown": "The human-readable plan for the whole WP, in Markdown",
    "decisions": ["Decision made during planning, with its reason"],
    "open_questions": ["Question the user should answer before approval (may be empty)"],
    "findings": [{
        "id": "F1",
        "kind": "file|symbol|architecture|constraint|test|existing_impl|risk|assumption",
        "statement": "One verified fact or explicit assumption",
        "evidence": [{"path": "project/relative/path.py", "symbol": "optional", "lines": "optional"}],
        "validated": True,
        "task_ids": ["WP4-T1"],
    }],
    "discovery": [{
        "file": "project/relative/path.py",
        "relevant_symbols": [{"name": "module.function", "start_line": 10,
                              "end_line": 35}],
        "reason": "Short explanation of task relevance",
        "task_ids": ["WP4-T1"],
    }],
    "shared": {
        "objective": "WP objective in implementation terms",
        "requirements": ["OpenSpec requirement relevant to several tasks"],
        "architecture": ["Architectural relationship coders must respect"],
        "relevant_files": ["project/relative/path.py"],
        "symbols": ["module.Class.method"],
        "constraints": ["Constraint common to several tasks"],
        "decisions": ["Planning decision common to several tasks"],
        "validated_assumptions": ["Assumption you verified in the repository"],
        "do_not_modify": ["Path or area that must not be modified"],
        "test_strategy": ["How the WP as a whole is validated"],
        "finding_ids": ["F1"],
    },
    "tasks": {
        "<task id>": {
            "objective": "What this task must achieve",
            "approach": "Expected implementation approach",
            "requirements": ["Task-specific OpenSpec requirement"],
            "relevant_files": ["project/relative/path.py"],
            "symbols": ["module.function"],
            "existing_implementation": ["What already exists and must be reused or extended"],
            "depends_on": ["Other task ids whose output this task needs"],
            "constraints": ["Task-specific constraint"],
            "do_not_modify": ["Path or area this task must leave unchanged"],
            "expected_tests": ["Test to add or run, with its path"],
            "finding_ids": ["F1"],
        },
    },
    "reviewer": {
        "objective": "What the reviewer must confirm was delivered",
        "architecture": ["Architectural finding the reviewer should check against"],
        "expected_behavior": ["Observable behavior after the WP"],
        "task_boundaries": ["WP4-T1 owns X; WP4-T2 owns Y"],
        "dependencies": ["Ordering or integration dependency between tasks"],
        "relevant_files": ["project/relative/path.py"],
        "risks": ["Risk to watch for"],
        "assumptions": ["Assumption the reviewer should verify"],
        "validation_expectations": ["Evidence the reviewer should expect"],
    },
}

SYSTEM = (
    "You are the Work Package planner. A strong model plans an entire OpenSpec Work Package "
    "once, so that smaller coding models can implement each task without rediscovering the "
    "repository, and the reviewer can start from your findings.\n\n"
    "Investigate the repository with the read-only IDE tools you are given: the WP objective, "
    "current implementation, relevant architecture, dependencies, tests, constraints, and "
    "functionality that already exists. Prefer targeted reads over broad listings. You never "
    "edit files and never start implementation.\n\n"
    "Return ONLY one JSON object matching the shape below, as your final message. It holds "
    "both the human-readable plan (plan_markdown) and the structured execution context; both "
    "must describe the same plan. Rules:\n"
    "- `tasks` has exactly one entry for every OpenSpec task id listed, keyed by that id.\n"
    "- Put information needed by several tasks in `shared`; each task entry holds only its "
    "relevant subset plus task-specific findings.\n"
    "- Findings are facts you verified (validated=true, with file evidence) or explicit "
    "assumptions (validated=false). Give them stable ids F1, F2, ...; reference them from "
    "`finding_ids` and list the tasks they matter to in `task_ids`.\n"
    "- Paths are project-relative. Name real files and symbols you inspected; mark files that "
    "do not exist yet as planned in the approach text.\n"
    "- Put compact file and symbol leads in `discovery` when you identify relevant code. "
    "Use line ranges from an outline only as hints; do not copy source into the plan. "
    "An empty task_ids list makes a lead shared. KB-only leads are not validated findings; "
    "validate implementation claims against current IDE source.\n"
    "- Do not invent requirements beyond the OpenSpec WP and the user's instructions.\n"
    "- Tasks whose preferred_capability is one of "
    f"{', '.join(sorted(DISCOVERY_CAPABILITIES))} run read-only: the coder cannot create "
    "or edit files, and its findings are delivered in its report. Never plan file writes "
    "for them; any file the WP needs belongs to an implementation task.\n"
    "- When revising, update the current state in place: keep existing finding ids and task "
    "ids stable, and change only what the user's message or new evidence requires.\n\n"
    f"\n{KB_READING_GUIDANCE}\n\nJSON shape:\n{json.dumps(_SHAPE, indent=2)}"
)


def wp_brief(change: OpenSpecChange, package: WorkPackage, complexity: str) -> str:
    referenced = set(package.requirement_refs)
    requirements = [{
        "ref": f"{capability.path}#{requirement.name}",
        "statement": requirement.statement,
        "scenarios": [asdict(scenario) for scenario in requirement.scenarios],
    } for capability in getattr(change, "capabilities", ())
        for requirement in capability.requirements
        if f"{capability.path}#{requirement.name}" in referenced]
    tasks = [{
        "id": task.id, "title": task.title, "description": task.description,
        "complexity": task.complexity, "definition_of_done": task.acceptance_criteria,
        "depends_on": task.depends_on, "context": task.context,
        "preferred_capability": task.preferred_capability, "status": task.status,
    } for task in package.tasks if task.status != "cancelled"]
    return json.dumps({
        "change": {"id": change.id, "title": change.title, "summary": change.summary,
                   "design_context": change.design.context,
                   "design_decisions": [asdict(item) for item in change.design.decisions]},
        "work_package": {
            "id": package.id, "title": package.title, "kind": package.kind,
            "objective": package.objective, "complexity": complexity,
            "acceptance_criteria": package.acceptance_criteria,
            "requirement_refs": package.requirement_refs,
            "requirements": requirements,
            "validation": asdict(package.validation), "depends_on": package.depends_on,
            "tasks": tasks,
        },
    }, ensure_ascii=False, indent=2)


NARRATION = (
    "Progress notes: before each tool call, write one short line (at most 15 words) "
    "saying what you are checking and why, for example \"Checking how loaders are "
    "registered to reuse the registry\". Write nothing else besides these lines and "
    "the final JSON object.")


# Planner-only (planner.discovery.mode: advise); the shared KB_READING_GUIDANCE used by
# coders and reviewers is unchanged.
DISCOVERY_GUIDANCE = (
    "Discovery: retrieve progressively and stop once the plan is actionable.\n"
    "1. Outline: use the prefetched context, Codebase Memory graph tools when available, "
    "searches and file outlines to find the modules, files, symbols and dependencies the WP "
    "touches.\n"
    "2. Targeted inspection: read only the line ranges of the functions or classes you need.\n"
    "3. Full file: read a whole file only when it is small or a narrower read cannot settle "
    "a specific question that matters to the plan.\n"
    "Before each read, know which question it answers; skip files that neither the WP, the "
    "prefetched context nor a search result connects to the work, and do not re-read regions "
    "you already have.\n"
    "Sufficiency checklist: stop investigating and return the plan when, for every task, you "
    "can name the components and files to change or reuse, their dependencies, the task's "
    "implementation responsibilities, and how the result is verified (tests and acceptance "
    "criteria). Correctness comes first: keep investigating while any of these is uncertain. "
    "A tool result may end with a short planning advisory; it is guidance, not a limit.")


# Planner-specific sufficiency checklist next to the shared KB-first guidance.
KB_FIRST_GUIDANCE = (
    "Planner checklist: for every task, identify the components and current files, "
    "their dependencies, the implementation responsibility and the verification "
    "criteria. Treat indexed locations as discovery leads; use current IDE source "
    "for any implementation detail the plan relies on. Stop once those questions "
    "are answered, unless a specific gap remains.")


def turn_prompt(brief: str, discussion: list[dict], current: dict | None, message: str,
                narrate: bool = False, kb_status: str = "",
                discovery_guidance: bool = False, kb_tools: bool = False) -> str:
    parts = [SYSTEM, *(["", NARRATION] if narrate else []),
             *(["", DISCOVERY_GUIDANCE] if discovery_guidance else []),
             "", "OpenSpec Work Package:", brief]
    if kb_tools:
        parts += ["", KB_FIRST_GUIDANCE]
    if kb_status:
        parts += ["", "Codebase Memory context (prefetched for this WP):", kb_status]
    if discussion:
        parts += ["", "Planning discussion so far (oldest first):",
                  *(f"- [revision {item['revision']}] user: {item['text']}" for item in discussion)]
    if current is not None:
        parts += ["", "Current plan state to revise in place:",
                  json.dumps(current, ensure_ascii=False)]
        parts += ["", f"The user's new message: {message}",
                  "Investigate further if the message needs it, then return the complete "
                  "revised JSON object."]
    else:
        parts += ["", f"The user's request: {message}",
                  "Investigate the repository, then return the complete JSON object."]
    return "\n".join(parts)


def repair_prompt(gaps: list[str]) -> str:
    return ("Your plan JSON is incomplete or invalid:\n"
            + "\n".join(f"- {gap}" for gap in gaps)
            + "\nReturn the complete corrected JSON object only, with no other text.")

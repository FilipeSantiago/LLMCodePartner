"""Planner prompts. The planner returns its plan and structured state together."""

import json
from dataclasses import asdict

from agent.spec.models import OpenSpecChange, WorkPackage

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
    "- Do not invent requirements beyond the OpenSpec WP and the user's instructions.\n"
    "- When revising, update the current state in place: keep existing finding ids and task "
    "ids stable, and change only what the user's message or new evidence requires.\n\n"
    f"JSON shape:\n{json.dumps(_SHAPE, indent=2)}"
)


def wp_brief(change: OpenSpecChange, package: WorkPackage, complexity: str) -> str:
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
            "validation": asdict(package.validation), "depends_on": package.depends_on,
            "tasks": tasks,
        },
    }, ensure_ascii=False, indent=2)


NARRATION = (
    "Progress notes: before each tool call, write one short line (at most 15 words) "
    "saying what you are checking and why, for example \"Checking how loaders are "
    "registered to reuse the registry\". Write nothing else besides these lines and "
    "the final JSON object.")


def turn_prompt(brief: str, discussion: list[dict], current: dict | None, message: str,
                narrate: bool = False) -> str:
    parts = [SYSTEM, *(["", NARRATION] if narrate else []), "", "OpenSpec Work Package:", brief]
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

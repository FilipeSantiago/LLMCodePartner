"""Planner prompts. The planner returns its plan and structured state together."""

import json
from dataclasses import asdict

from agent.planning.knowledge_base import KB_NATIVE_GUIDANCE, KB_READING_GUIDANCE
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

_RULES = """You are the Work Package planner.

Your responsibility is to investigate and plan an entire OpenSpec Work Package once, producing an actionable plan that smaller coding models can execute with minimal additional discovery.

Optimize for implementation readiness, accurate evidence, efficient KB usage, and minimal repository reads — not exhaustive repository documentation.

## Planning Strategy

Follow a **KB-first, source-second** investigation strategy.

1. Start from the Work Package objective, task descriptions, acceptance criteria, and dependencies.
2. Identify the architectural questions and implementation decisions required for each task.
3. Use the Knowledge Base (KB) to understand the system architecture, component relationships, dependencies, and existing functionality.
4. Use KB results to identify the specific files, classes, functions, and integration points relevant to each task.
5. Read only the source code necessary to verify implementation-critical details that the KB cannot establish authoritatively.
6. Reuse findings across tasks instead of repeating investigations.
7. Stop investigating when every task has sufficient information for implementation and validation.

Do not reconstruct the repository architecture through broad file exploration when the KB already provides that information.

## Knowledge Base — Primary Discovery Mechanism

Use the KB as the primary source for architectural discovery.

Prefer KB tools for:

- Understanding the overall system architecture.
- Identifying components and their responsibilities.
- Discovering dependencies between modules.
- Tracing execution paths and data flows.
- Finding existing implementations and reusable abstractions.
- Identifying integration points.
- Understanding the impact of proposed changes.
- Locating files, symbols, and relevant tests.

### KB Investigation Rules

- Begin with task-specific architectural questions, not broad repository exploration.
- Prefer targeted KB queries over exhaustive graph traversal.
- Reuse previous KB results when they answer the current question.
- Avoid repeatedly querying the same components or relationships.
- Follow dependencies only when they affect an implementation decision.
- Use the KB to narrow the set of files requiring direct inspection.
- Do not retrieve large amounts of unrelated architectural context.
- Avoid generating repository-wide summaries unless explicitly required.

KB findings may establish architectural context for planning, but implementation-specific claims must be verified against current source before being marked as validated.

If the KB is unavailable, outdated, or insufficient, fall back to targeted source discovery and record the limitation.

## Source Code — Targeted Verification

Direct IDE file reads are a secondary mechanism.

Use them to inspect relevant source identified by the KB or OpenSpec, rather than to discover the entire architecture.

Read source when necessary to verify:

- Exact function or class signatures.
- Existing implementation behavior.
- Configuration fields and defaults.
- Error handling and edge cases.
- Compatibility constraints.
- Test contracts and expected behavior.
- Implementation details that materially affect the planned changes.
- Conflicting or potentially outdated KB findings.

### Source Reading Rules

- Read only files relevant to the current implementation question.
- Prefer specific functions, classes, and line ranges over complete files.
- Avoid reading entire files when the required information is available in a smaller region.
- Avoid reading the same unchanged file or source range repeatedly.
- Do not perform broad directory scans or repository-wide searches by default.
- Use targeted source searches only when the KB cannot locate the required implementation.
- Do not inspect every implementation of an abstraction when its interface and the relevant concrete implementation are sufficient.
- Avoid unrelated tests, documentation, generated files, and vendored code.
- Expand source inspection only when necessary to resolve uncertainty, contradictions, or missing information.

**Source code is authoritative for implementation details. The KB is the primary mechanism for architectural understanding and navigation.**

Do not automatically read every file returned by the KB.

## Findings and Evidence

Record findings only when they materially affect:

- An implementation approach.
- An interface or dependency.
- An architectural constraint.
- A known limitation or risk.
- A validation requirement.

Every finding must be either:

**Verified:** Confirmed against current source code, with file evidence and `validated=true`.

**Assumption:** Not confirmed against current source, with `validated=false`, including architectural information obtained only from KB results.

Use stable finding IDs F1, F2, ... and associate findings with relevant tasks through `task_ids`.

Do not duplicate findings across shared and task-specific context.

Prefer concise, actionable findings over extensive explanations of existing code.

A KB discovery lead is not automatically a verified implementation finding.

## Task Planning

Create exactly one task entry for each OpenSpec task ID.

For each task, identify:

- Objective and expected outcome.
- Relevant existing implementation.
- Expected files and symbols.
- Concrete implementation approach.
- Dependencies and constraints.
- Acceptance criteria and validation strategy.
- Findings necessary to execute the task.

Use the KB to identify relevant architecture and dependencies before selecting files for source verification.

Place information needed by multiple tasks in `shared`. Each task entry must contain only its relevant subset and task-specific findings.

Shared context must not become a repository-wide knowledge dump.

Do not invent requirements beyond OpenSpec or the user's instructions.

Tasks whose `preferred_capability` is one of the configured discovery capabilities (@CAPABILITIES@) are read-only. Never plan file modifications for these tasks; their findings belong in their reports. Never plan file writes for them; any file the WP needs belongs to an implementation task.

## Investigation Stopping Condition

Stop repository investigation when every task has:

1. A feasible implementation approach.
2. Identified interfaces and integration points.
3. Known dependencies and constraints.
4. An appropriate validation strategy.
5. No unresolved question that would materially change the approach.

Architecture and dependency understanding obtained from the KB is sufficient when exact source implementation details are not required for the decision.

Do not read additional files solely to increase the number of verified findings.

If uncertainty remains, record it as an assumption and identify what the executor should verify.

Do not continue investigating solely to produce more comprehensive documentation.

## Output and Revisions

Return ONLY one JSON object matching the provided schema.

The JSON contains both `plan_markdown` and structured execution context. They must describe the same plan without unnecessary repetition.

Paths must be project-relative. Reference inspected files and symbols accurately. Mark nonexistent files as planned in the approach.

Use compact discovery entries as navigation leads. Line ranges obtained from KB outlines are hints, not verified source evidence.

When revising, update the current plan in place. Preserve existing task IDs and finding IDs, changing only what new requirements or evidence justify.

You never edit files or start implementation."""

_SHAPE_TEXT = f"JSON shape:\n{json.dumps(_SHAPE, indent=2)}"

SYSTEM = (_RULES.replace("@CAPABILITIES@", ", ".join(sorted(DISCOVERY_CAPABILITIES)))
          + f"\n\n{KB_READING_GUIDANCE}\n\n{_SHAPE_TEXT}")


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


# A native planner reads the mounted project with its own file tools. The KB-first gates
# apply to IDE-path planners only, so its KB guidance names the tools without them.
NATIVE_KB_GUIDANCE = KB_NATIVE_GUIDANCE

NATIVE_SYSTEM = SYSTEM.replace(KB_READING_GUIDANCE, NATIVE_KB_GUIDANCE).replace(
    "Direct IDE file reads",
    "Direct file reads (your own read-only file tools; the current directory is the "
    "project root)")


def turn_prompt(brief: str, discussion: list[dict], current: dict | None, message: str,
                narrate: bool = False, kb_status: str = "",
                discovery_guidance: bool = False, kb_tools: bool = False,
                native: bool = False) -> str:
    parts = [NATIVE_SYSTEM if native else SYSTEM, *(["", NARRATION] if narrate else []),
             *(["", DISCOVERY_GUIDANCE] if discovery_guidance else []),
             "", "OpenSpec Work Package:", brief]
    if kb_tools and not native:
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

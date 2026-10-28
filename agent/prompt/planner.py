"""The Work Package planner's prompt. The planner returns its plan and structured state
together."""

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from agent.prompt.base import ParentPromptBuilder, PromptRole
from agent.task_contracts.policy import DISCOVERY_CAPABILITIES

if TYPE_CHECKING:
    from agent.spec.models import OpenSpecChange, WorkPackage
    from providers.subagents import Request

_SHAPE = {
    "summary": "Two or three sentences: what this WP will deliver and how",
    "decisions": ["Decision made during planning, with its reason"],
    "open_questions": ["Question the user should answer before approval (may be empty)"],
    "findings": [{
        "id": "F1",
        "kind": "file|symbol|architecture|constraint|test|existing_impl|risk|assumption",
        "statement": "One verified fact or explicit assumption, at most 25 words",
        "evidence": [{"path": "project/relative/path.py", "symbol": "optional", "lines": "optional"}],
        "validated": True,
        "task_ids": ["WP4-T1"],
    }],
    "shared": {
        "objective": "WP objective in implementation terms, at most 25 words",
        "requirements": ["OpenSpec requirement relevant to several tasks"],
        "architecture": ["Architectural relationship coders must respect"],
        "constraints": ["Constraint common to several tasks"],
        "decisions": ["Planning decision common to several tasks"],
        "do_not_modify": ["Path or area that must not be modified"],
        "test_strategy": ["How the WP as a whole is validated"],
        "finding_ids": ["F1"],
    },
    "tasks": {
        "<task id>": {
            "objective": "What this task must achieve, at most 25 words",
            "approach": "Expected implementation approach, at most 40 words",
            "edits": [{"file": "project/relative/path.py", "start_line": 40, "end_line": 58,
                       "symbol": "module.function", "change": "What changes here, at most 15 words"}],
            "references": [{"file": "project/relative/other.py", "start_line": 10,
                            "end_line": 30, "why": "What the coder needs from it, at most 10 words"}],
            "requirements": ["Task-specific OpenSpec requirement"],
            "depends_on": ["Other task ids whose output this task needs"],
            "constraints": ["Task-specific constraint"],
            "do_not_modify": ["Path or area this task must leave unchanged"],
            "expected_tests": ["Test to add or run, with its path"],
            "finding_ids": ["F1"],
        },
    },
    "reviewer": {
        "objective": "What the reviewer must confirm was delivered",
        "expected_behavior": ["Observable behavior after the WP"],
        "task_boundaries": ["WP4-T1 owns X; WP4-T2 owns Y"],
        "dependencies": ["Ordering or integration dependency between tasks"],
        "risks": ["Risk to watch for"],
        "assumptions": ["Assumption the reviewer should verify"],
        "validation_expectations": ["Evidence the reviewer should expect"],
    },
}

_RULES = """You are the Work Package planner.

Your responsibility is to investigate and plan an entire OpenSpec Work Package once, producing an actionable plan that smaller coding models can execute without further discovery: for every task, the exact lines to change and the exact context to read.

Optimize for implementation readiness and accurate evidence, with minimal repository context.

@READ_PIPELINE@

@BULK_RULE@

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

Do not duplicate findings across shared and task-specific context, and do not restate in a finding what an edit or reference range already says.

## Task Planning

Create exactly one task entry for each OpenSpec task ID.

For each task, give:

- `edits`: every place the coder changes, as the file and the exact current `start_line`-`end_line` of the code to change (a function, a block, an insertion point), with the symbol and a short `change`. A new file has null lines and a `change` starting with "create". A new Python package directory also gets its `__init__.py` as a new-file edit when the sibling packages have one. The coder may only edit these files.
- `references`: the exact files and line ranges the coder must read to make the change (interfaces it calls, patterns to follow, the tests to extend), each with a short `why`. A small file may omit lines. Nothing else is read.
- Objective, approach, dependencies, constraints and the validation.

Edit and reference ranges must match current source: take them from KB declaration results, confirmed by the reader's brief or your bounded reads. CodePartner widens every range by a few lines of margin itself; give the exact lines.

Place information needed by multiple tasks in `shared`. Each task entry must contain only its relevant subset and task-specific findings.

Write for a coder that already gets the ranges: be objective and short, respect the word limits in the schema, and never restate file contents, line details or explanations the ranges already carry. Shared context must not become a repository-wide knowledge dump.

Do not invent requirements beyond OpenSpec or the user's instructions.

Tasks whose `preferred_capability` is one of the configured discovery capabilities (@CAPABILITIES@) are read-only. Never plan file modifications for these tasks; give them `references` only, and their findings belong in their reports. Never plan file writes for them; any file the WP needs belongs to an implementation task.

## Investigation Stopping Condition

Stop repository investigation when every task has:

1. Exact edit ranges and the references they need.
2. Identified interfaces and integration points.
3. Known dependencies and constraints.
4. An appropriate validation strategy.
5. No unresolved question that would materially change the approach.

If uncertainty remains, record it as an assumption and identify what the executor should verify.

Do not continue investigating solely to produce more comprehensive documentation.

## Output and Revisions

Return ONLY one compact JSON object matching the provided schema: no indentation, no code fence, no text around it.

The human-readable plan is rendered from the structured fields; do not write a separate prose version of the plan. State each fact once, in the field where it belongs.

Paths must be project-relative. Reference inspected files and symbols accurately.

When revising, update the current plan in place. Preserve existing task IDs and finding IDs, changing only what new requirements or evidence justify.

You never edit files or start implementation."""

_SHAPE_TEXT = f"JSON shape:\n{json.dumps(_SHAPE, indent=2)}"



NARRATION = (
    "Progress notes: before each tool call, write one short line (at most 15 words) "
    "saying what you are checking and why, for example \"Checking how loaders are "
    "registered to reuse the registry\". Write nothing else besides these lines and "
    "the final JSON object.")


# Planner-only sufficiency advisory; the read pipeline above is the sole read instruction.
DISCOVERY_GUIDANCE = (
    "Sufficiency checklist: return the plan when, for every task, you "
    "can name the components and files to change or reuse, their dependencies, the task's "
    "implementation responsibilities, and how the result is verified (tests and acceptance "
    "criteria). Correctness comes first: keep investigating while any of these is uncertain. "
    "A tool result may end with a short planning advisory; it is guidance, not a limit.")


@dataclass(frozen=True)
class PlannerPromptContext:
    brief: str
    discussion: list[dict]
    current: dict | None
    message: str
    narrate: bool = False
    kb_status: str = ""
    discovery_guidance: bool = False
    native: bool = False
    # The native planner's reader subagent; its delegation rules join the prompt.
    subagent_request: "Request | None" = None
    file_inventory: str = ""
    workspace: str = ""
    kb_tools: frozenset[str] = frozenset()
    hosted_reader: bool = False


class PlannerPromptBuilder(ParentPromptBuilder[PlannerPromptContext]):
    role = PromptRole.PLANNER
    # Empty parts are blank-line spacers in this prompt.
    drop_empty = False

    def sections(self, context: PlannerPromptContext) -> list[str]:
        rules = ("SOURCE INSPECTION: Use `planner_reader` with one objective, explicit "
                 "project-relative paths, line ranges, and a question for each range. "
                 "Wait for its path:line brief. You have no direct shell or source-content "
                 "tool; never try `exec_command` or a content search to read files."
                 if context.hosted_reader else
                 self.delegation_rules(context.subagent_request) if context.native else "")
        parts = [self.system(), *(["", NARRATION] if context.narrate else []),
                 *(["", DISCOVERY_GUIDANCE] if context.discovery_guidance else []),
                 # Native planners with a reader subagent: who explores and when to read.
                 *(["", rules] if rules else []),
                 "", "OpenSpec Work Package:", context.brief]
        if context.workspace:
            parts += ["", f"Working directory: {context.workspace}; use paths under it."]
        if context.file_inventory:
            parts += ["", "Project files (current; one directory per line):",
                      context.file_inventory]
        if context.kb_status:
            parts += ["", "Codebase Memory context (prefetched for this WP):", context.kb_status]
        if context.discussion:
            parts += ["", "Planning discussion so far (oldest first):",
                      *(f"- [revision {item['revision']}] user: {item['text']}"
                        for item in context.discussion)]
        if context.current is not None:
            parts += ["", "Current plan state to revise in place:",
                      json.dumps(context.current, ensure_ascii=False)]
            parts += ["", f"The user's new message: {context.message}",
                      "Investigate further if the message needs it, then return the complete "
                      "revised JSON object."]
        else:
            parts += ["", f"The user's request: {context.message}",
                      "Investigate the repository, then return the complete JSON object."]
        return parts

    def repair(self, problem: list[str]) -> str:
        return ("Your plan JSON is incomplete or invalid:\n"
                + "\n".join(f"- {gap}" for gap in problem)
                + "\nReturn the complete corrected JSON object only, with no other text.")

    def system(self) -> str:
        """The planner's rules and JSON shape."""
        return (_RULES.replace("@CAPABILITIES@", ", ".join(sorted(DISCOVERY_CAPABILITIES)))
                .replace("@READ_PIPELINE@", self.read_pipeline())
                .replace("@BULK_RULE@", self.bulk_rule())
                + f"\n\n{_SHAPE_TEXT}")

    def read_pipeline(self) -> str:
        # The planner explores the whole Work Package, KB first and in bulk.
        tools = self._context.kb_tools if self._context is not None else frozenset()
        declaration_call = (
            "one `planner_declarations` call with several project-relative file paths"
            if "planner_declarations" in tools else
            "one `kb_query_graph` for declarations and line ranges across several files"
            if "kb_query_graph" in tools else
            "one `kb_search_graph` with a `file_pattern` alternation such as "
            "`a\\.py|b\\.py` and `fields` including `file_path`, `name`, "
            "`start_line`, and `end_line`")
        return """## Read pipeline — **KB-first, bulk, source-second**

Start from the prefetched Codebase Memory (KB) context; it already holds the architecture overview, so do not fetch it again. Query the KB in bulk for all tasks at once: one `kb_search_graph` with a `name_pattern` alternation (`A|B|C`) or a `semantic_query` array for every symbol of a step, and @DECLARATIONS@. Never query file by file or symbol by symbol. Use those leads and the OpenSpec request to identify the minimum current source ranges needed. The project file list given with the Work Package is current: do not Glob or search for paths it already answers. """.replace("@DECLARATIONS@", declaration_call) + self.own_reads_rule() + """ The reader only reads what you list; it never searches and has no KB, so give it the exact project-relative paths and line ranges plus the specific questions, all of one objective in one request, and plan from its concise `path:line` brief. KB results are navigation hints, not source evidence; the reader's brief of current source is. Never make placeholder or no-op tool calls (an empty or "noop" `Agent` request, reading `/dev/null`). If an offered reader is unavailable or fails, record the evidence gap instead of reading source yourself. Stop investigating when every task has exact edit and reference ranges and a validation strategy."""

    def bulk_rule(self) -> str:
        return ("ALWAYS BULK: send independent KB lookups and bounded reads together "
                "and make one reader request per objective. Do not repeat context already "
                "in the plan or a brief.")

    def own_reads_rule(self) -> str:
        if self._context is not None and self._context.hosted_reader:
            return ("Send every source-content question to `planner_reader` in one "
                    "batched request per objective; use graph tools only for navigation.")
        return ("When a `reader` subagent is offered, delegate every source-content read to "
                "it and keep your own searches path-only; when none is offered, make only "
                "minimal bounded reads, batched in one response.")

    def delegation_rules(self, request: "Request | None") -> str:
        """The planner explores the whole Work Package: no assigned files, every source
        read goes to the reader."""
        if request is None:
            return ""
        parts = [
            "DELEGATION RULES: You locate the code yourself (Codebase Memory first, in bulk, "
            "then path-only Glob/Grep or shell searches) and delegate source inspection to the "
            "`reader` subagent; its brief is that call's result. A reader only reads what "
            "you list: it never searches and has no Codebase Memory, so each request must give "
            "the full context: the project-relative paths with line ranges (or whole files when "
            "needed) and the questions to answer. Put the related files and questions of one "
            "objective in a single foreground request, and send independent requests together rather than "
            "one after another. The brief returns findings with path:line references; work "
            "from it. When a brief lists \"Missing context:\", locate what it names and send "
            "one follow-up request. You may explore as widely as the Work Package needs.",
            "NO DIRECT SOURCE READS: Do not bypass the `reader` subagent by using command "
            "lines, scripts or your own file tools to display project-file contents "
            "(including source, tests and configuration), even for known line "
            "ranges. This includes `rg`/`grep` searches that print matching lines, "
            "`cat`/`sed`/`head`/`tail`, and commands that print source diffs. Shell discovery "
            "may return paths or non-source metadata only (for example `rg --files`, `rg -l`, "
            "or `git diff --name-only`). Send every source-inspection question and its known "
            "paths and ranges to the `reader`; its brief is that call's result. If the reader is unavailable "
            "or fails, report the evidence gap rather than reading source yourself.",
        ]
        if PromptRole.CHECKS in request.roles:
            parts.append(self.checks_rule())
        return "\n".join(parts)

    def scope_exception_rule(self) -> str:
        return ("For a question outside the initial plan, state `SCOPE EXCEPTION: "
                "<question>` before locating paths and asking the reader.")

    def scope_block(self, scope: tuple[str, ...], read_only: tuple[str, ...] = ()) -> str:
        return "\n".join(["Work Package paths:", *(f"- {path}" for path in scope),
                          *(f"- {path}" for path in read_only)])

    def checks_rule(self) -> str:
        return ("Run test suites and other long-output commands through the checks "
                "subagent, giving exact commands; its result is that call's result.")

    @staticmethod
    def brief(change: "OpenSpecChange", package: "WorkPackage", complexity: str) -> str:
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


# The planner's system prompt, for callers that need it without building a turn.
SYSTEM = PlannerPromptBuilder().system()

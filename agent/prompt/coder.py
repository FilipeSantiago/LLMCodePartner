"""The coding agent's task prompt, for IDE MCP and native execution."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.prompt.base import ParentPromptBuilder, PromptRole

if TYPE_CHECKING:
    from agent.spec.models import Task
    from providers.subagents import Request

# Applies to every executor and model. The plan's files and ranges are the coder's
# context; going beyond them needs a stated reason. Shared by the IDE and native coder
# prompts; with native subagents the scope part gives way to `delegation_rules`.
_SCOPE_RULES = (
    "Stay within the files and line ranges named above: they are your context, already "
    "investigated. Never read the whole repository or explore the codebase. Go beyond them "
    "only for a concrete question they leave open (a missing signature, a failing test, a "
    "contradiction with current source), after writing one line "
    "`SCOPE EXCEPTION: <that question>`. Reuse what earlier tool calls returned: do not "
    "re-read an unchanged file region, repeat a search that already answered the question, "
    "fetch the same information with several tools, or re-query context already given here.")

_WORK_RULES = (
    "Treat verified planner findings "
    "and earlier task results as navigation and do not repeat their investigation; when "
    "current source contradicts them, current source wins. Settle a minimal approach "
    "before editing, make small focused changes, reuse existing utilities and interfaces, "
    "and avoid unrelated refactoring, speculative improvements and new abstractions the "
    "task does not need. Validate with the targeted tests first, broaden only when needed "
    "or required, do not rerun unchanged tests without a reason, and run the required "
    "lint, format and type checks")

EXPLORATION_RULES = f"EXECUTION RULES: {_SCOPE_RULES} {_WORK_RULES}."

_DELEGATED_EXECUTION_RULES = (
    f"EXECUTION RULES: {_WORK_RULES}; more context comes through the reader under the "
    "rules above.")

_HANDOFF_SCHEMA = (
    "Finish with exactly one JSON object using this handoff schema: "
    '{"schema_version":1,"task_id":"<task id>","summary":"<work done>",'
    '"changed_files":[{"path":"project/relative/path","change":'
    '"created|modified|deleted|renamed|moved","purpose":"<why>"}],'
    '"criteria":[{"criterion":"<requirement>","status":'
    '"satisfied|blocked","paths":["project/relative/path"],'
    '"evidence_ids":[]}],"remaining_gaps":[],"execution_dispute":null,'
    '"context_handoff":{"files":[],"symbols":[],"decisions":[],"dependencies":[],'
    '"verification":[],"constraints":[]}}. '
    "The optional context_handoff is for the next task: include only useful project-relative "
    "file paths and short symbol names (as path::name) or facts, with no source excerpts or conversation "
    "history. It is navigation context; the next coder must verify current source. "
    "Use remaining_gaps only for unfinished work required by this task's own "
    "acceptance criteria. ")


def _task_contract(task: "Task", execution_instruction: str, contract) -> list[str]:
    """The task's own contract, each fact stated once."""
    goal = getattr(contract, "goal", "") if contract is not None else ""
    return [
        f"Implement OpenSpec task {task.id}: {task.title}",
        f"Description: {task.description}",
        f"Acceptance criteria: {task.acceptance_criteria or [task.description]}",
        f"Context: {task.context}" if task.context else "",
        f"Preferred capability: {task.preferred_capability}",
        (f"Additional execution instruction: {execution_instruction}"
         if execution_instruction else ""),
        (f"TaskContract revision {contract.revision}"
         + (f": {goal}" if goal and goal.strip() != task.description.strip() else "")
         + f". Expected files: {', '.join(contract.primary_files) or 'none named'}. "
         "These are guidance, not a file allowlist."
         if contract is not None else ""),
    ]


def _planned(plan_context: str, ide: bool) -> str:
    if not plan_context:
        return ""
    return ("A planner already investigated this work package; the context above is "
            "scoped to this task" + (" and its line ranges are leads" if ide else "")
            + ". Report any deviation from it in the summary, or in remaining_gaps if it "
            "leaves this task incomplete" + (" (use execution_dispute when it concerns "
                                            "execution)" if ide else "")
            + ". Respect the do-not-modify areas. Where the OpenSpec description or context "
            "disagrees with the approved plan, follow the plan: it reflects the current "
            "repository.")


@dataclass(frozen=True)
class CoderPromptContext:
    task: "Task"
    execution_instruction: str = ""
    contract: object = None
    plan_context: str = ""
    handoff_context: str = ""
    native: bool = False
    kb_policy: str = "selective"
    # Native runs with subagents: the coder locates context and the reader reads it.
    subagent_request: "Request | None" = None
    aci_enabled: bool = True


class CoderPromptBuilder(ParentPromptBuilder[CoderPromptContext]):
    role = PromptRole.CODER

    def sections(self, context: CoderPromptContext) -> list[str]:
        if context.native:
            return self._native_sections(context)
        return [
            *_task_contract(context.task, context.execution_instruction, context.contract),
            context.plan_context,
            context.handoff_context,
            _planned(context.plan_context, ide=True),
            EXPLORATION_RULES,
            self.own_reads_rule(),
            self.bulk_rule(),
            ("" if context.plan_context else
             "Use the IDE MCP tools as needed to complete the task's actual objective."),
            self.read_pipeline(),
            ("Use the approved plan's files and symbols as KB search and outline seeds. "
             "Use current IDE reads for the exact signatures, edit regions and tests; "
             "indexed line numbers may have shifted. Broaden discovery only for a concrete "
             "question the plan does not settle." if context.kb_policy == "kb_first" else ""),
            ("Use codepartner_read_file with explicit line ranges for targeted source "
             "reads. A path-only call returns up to 200 lines; continue with ranges or "
             "request full_file=true when the whole file is genuinely needed. "
             "Repeated reads use a run-local cache; pass "
             "refresh=true when you need a fresh IDE read. Use codepartner_search_in_file "
             "to locate text within one file. Tool results include observation IDs that "
             "codepartner_get_observation can expand when diagnostics require more detail."
             if context.aci_enabled
             else "Use the advertised IDE file tools to inspect relevant source."),
            "Edit existing files with apply_patch; use create_new_file only for files "
            "that do not exist yet. Keep each tool call's arguments small; very large tool-call "
            "payloads can fail to parse and stop the task.",
            "If only the universal `execute_tool` is available, it runs IDE tools, not shell "
            "commands: run tests as `execute_terminal_command --command \"uv run pytest "
            "tests/test_x.py -v\"` (named `--param value` arguments only).",
            "If this task requires Python verification, inspect the project metadata and "
            "configured IDE interpreter or run configurations before declaring the checks "
            "unavailable. Prefer the project-native environment (for example `uv run pytest` "
            "or `.venv/bin/python -m pytest` when those are actually present) over a bare "
            "system `python` or `pytest`. Do not add test execution to a research or inspection "
            "task unless its own description or acceptance criteria requires it.",
            _HANDOFF_SCHEMA +
            "Mention planned work assigned to later tasks in the summary; "
            "it does not make this task incomplete. "
            "List every created or changed file. Write every path relative to the repository "
            "root and include its directories (for example, tests/test_trainer.py, never only "
            "test_trainer.py). This manifest helps discovery but does not replace the mutation "
            "journal or captured execution evidence. "
            "Evidence IDs must be numeric IDs actually issued by CodePartner; descriptive labels "
            "are claims, not evidence. If you disagree with a reviewer statement that code or "
            "tests do not compile, collect, or run, replace execution_dispute:null with an object "
            "containing claim, source_paths, working_directory, and steps. Each step must name an "
            "advertised tool and its exact arguments so CodePartner can replay it for the reviewer. "
            "Do not create a project-local runner to support a dispute. "
            "Do not claim a test ran unless it did. Do not create summary, completion, "
            "evidence, verification, or test-result files solely to communicate with the "
            "reviewer; return those details in your final response instead.",
        ]

    def _native_sections(self, context: CoderPromptContext) -> list[str]:
        """The CLI's own file and shell tools in the project's working directory; IDE
        tools only for IDE-specific work."""
        rules = self.delegation_rules(context.subagent_request)
        return [
            *_task_contract(context.task, context.execution_instruction, context.contract),
            context.plan_context,
            context.handoff_context,
            _planned(context.plan_context, ide=False),
            *((rules, _DELEGATED_EXECUTION_RULES) if rules
              else (EXPLORATION_RULES,)),
            self.own_reads_rule(),
            self.bulk_rule(),
            "You are working directly in the project's working directory. Use your own file "
            + ("tools to locate paths and edit files" if rules
               else "tools to read, search and edit files")
            + ", and your shell to run the project's own "
            "commands (for example `uv run pytest tests/test_x.py`). Current files are "
            "authoritative. Name files in your report by their project-relative paths.",
            self.read_pipeline(),
            "Bridged IDE tools, when offered, are for IDE-specific work such as inspections, "
            "symbol lookups and refactorings, not for ordinary reads and edits.",
            "Do not commit, switch branches, stash or rewrite git history; CodePartner "
            "journals your file changes and the reviewer inspects the work tree. Only change "
            "files this task needs.",
            "Do not claim a test ran unless it did, and say so when a command could not run "
            "in this environment.",
            _HANDOFF_SCHEMA +
            "List every created or changed file, with paths relative to "
            "the repository root. Evidence IDs must be numeric IDs actually issued by "
            "CodePartner; leave them empty otherwise. Do not create summary, completion, "
            "evidence, verification, or test-result files solely to communicate with the "
            "reviewer; return those details in your final response instead.",
        ]

    def with_workspace(self, messages: list[dict], workspace: str,
                       inventory: str) -> list[dict]:
        """Give the native coder its actual working directory and path inventory."""
        if not (workspace or inventory) or not messages or not isinstance(messages[-1].get("content"), str):
            return messages
        last = messages[-1]
        return [*messages[:-1], {**last, "content": last["content"] + (
            f"\nWorking directory: {workspace}; use paths under it."
            + ("\nProject files (for locating paths only; not a reading list):\n"
               + inventory if inventory else ""))}]

    def bulk_rule(self) -> str:
        return ("ALWAYS BULK: send independent KB lookups and permitted ranged reads "
                "together, with one reader request per objective. Do not re-read content "
                "already in the plan or a brief.")

    def read_pipeline(self) -> str:
        # Every extra message re-reads the whole context, so each step is one response.
        return ("## Work order — one response per step\n"
                "1. Locate: in one response, send every KB navigation call you need "
                "(file outlines, graph search, call paths) for the plan's reference and "
                "context files, to get their symbols and line ranges. Read no source yet. "
                "Skip this step when the plan's ranges already answer it.\n"
                "2. Context: only if the plan, its references and the KB results leave a "
                "question open, send one reader request with every remaining path, line "
                "range and question (without a reader, read those bounded ranges yourself "
                "in one response). The reader's brief is the result of that call.\n"
                "3. Edit files: then read all your planned edit ranges yourself in one "
                "response, as parallel ranged reads, not one file per message.\n"
                "4. Edit, then validate: run every test, lint and type command in one "
                "checks request (or one shell response without checks).\n"
                "Never end your turn to wait for a subagent or a tool: results come back in "
                "the same turn. KB results are navigation hints, not current-source "
                "evidence.")

    def delegation_rules(self, request: "Request | None") -> str:
        if request is None:
            return ""
        return "\n".join([
            "DELEGATION RULES: Your direct reads are limited to planned edit ranges. "
            "Ask the reader for references and any other content in one request, with "
            "exact paths, ranges and questions; its path:line brief is that call's result. "
            "A second reader request is refused unless its prompt states "
            "`SCOPE EXCEPTION: <question>`. Use only the `reader` and `checks` subagents.",
            self.scope_block(request.scope, request.read_only),
            self.scope_exception_rule(),
            "Do not use shell commands, content searches or KB source tools to print "
            "project file contents. Path discovery may return paths only.",
            *([self.checks_rule()] if PromptRole.CHECKS in request.roles else []),
        ])

    def scope_exception_rule(self) -> str:
        return ("OUT OF SCOPE: For a concrete question the plan leaves open, first "
                "write `SCOPE EXCEPTION: <question>`, then locate paths only and ask "
                "the reader. Never read or edit those files yourself.")

    def scope_block(self, scope: tuple[str, ...], read_only: tuple[str, ...] = ()) -> str:
        protected = set(read_only)
        change = [path for path in scope if path not in protected]
        context = list(dict.fromkeys((*read_only, *(path for path in scope if path in protected))))
        return "\n".join(["Files to change (planner-assigned):",
                          *(f"- {path}" for path in change),
                          "Read-only context (reader only):",
                          *(f"- {path}" for path in context)])

    def checks_rule(self) -> str:
        return ("Run test suites and other long-output commands through the checks "
                "subagent, giving exact commands, all of them in one request. Its result is "
                "that call's result.")

    def own_reads_rule(self) -> str:
        return ("Read the ranges you edit yourself (`Read` with offset/limit, or one ranged "
                "`sed -n` print when your CLI has no Read tool), all of them in one response "
                "after the context step, because editing needs the exact current text; "
                "never ask a reader to "
                "transcribe them. Do not reread a file after editing it: trust the Edit/Write "
                "results and the checks output, and look at a changed region again only to "
                "fix a failure.")

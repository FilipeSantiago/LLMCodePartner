"""Tasker service that creates and revises bounded coder contracts."""

import re
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from agent.spec.models import Task
from agent.task_contracts.models import (
    AcceptanceCriterion, CoderOutcome, CoderResult, ContextFact,
    ReviewerResult, TaskContract,
)
from agent.task_contracts.policy import (
    DISCOVERY_CAPABILITIES, DISCOVERY_OPERATIONS, MAX_EXECUTION_ATTEMPTS,
    MAX_GRANTED_FILES_PER_REVISION, VERIFICATION_CAPABILITIES, path_excluded,
    policy_for, protected_path,
)
from mcp_bridge.registry import OP_SHELL_EXECUTE, ToolSpec, classify_tool

_PATH = re.compile(r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,12}(?![\w/-])")
_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]{2,})`")
_VERB = r"(?:read|touch|modify|inspect|use|change|edit|open)"
_EXCLUSION = re.compile(
    rf"\b(?:ignore|exclude|skip|leave out|(?:do not|don't|never)\s+{_VERB}"
    rf"(?:\s*(?:,|or|and)\s*{_VERB})*)\s+(?:the\s+|any\s+)?(?:file\s+)?`?"
    r"([\w./-]*[./][\w./-]*\w)", re.IGNORECASE)
# TaskContractPersistence counts execution attempts from the latest revision with
# this reason, so a confirmed retry starts a fresh attempt budget.
USER_RESOLUTION_REASON = "user-confirmed contract escalation resolution"


_OUTSIDE_SCOPE = re.compile(
    r"(?:path outside contract scope|protected path|secrets file"
    r"|path excluded by user decision):\s*(.+)", re.IGNORECASE)


def requested_paths(reason: str) -> tuple[str, ...]:
    """Paths an execution guard refused (older releases: outside contract scope)."""
    match = _OUTSIDE_SCOPE.search(reason or "")
    if match is None:
        return ()
    return tuple(dict.fromkeys(
        item.strip().rstrip(".") for item in match.group(1).split(",") if item.strip()))


def excluded_paths(guidance: str) -> tuple[str, ...]:
    """Paths the user explicitly told Tasker to keep out of scope."""
    return tuple(dict.fromkeys(match.group(1) for match in _EXCLUSION.finditer(guidance)))


SCOPE_GRANT_REASON = "tasker granted coder scope request"


@dataclass(frozen=True)
class ScopeDecision:
    """Tasker's rule-based answer to the Coder's requests for more scope."""

    granted: tuple[str, ...] = ()
    denied: tuple[tuple[str, str], ...] = ()  # (path, reason)

    @property
    def considered(self) -> bool:
        return bool(self.granted or self.denied)

    def describe(self) -> str:
        parts = []
        if self.granted:
            parts.append("granted " + ", ".join(self.granted))
        if self.denied:
            parts.append("denied " + "; ".join(
                f"{path} ({reason})" for path, reason in self.denied))
        return "; ".join(parts) or "no scope was requested"


class TaskContractor:
    """Owns scope decisions; Coder only executes the returned contract."""

    def grant_scope(self, contract: TaskContract, result: CoderResult) -> ScopeDecision:
        """Decide the Coder's scope requests by deterministic rules."""
        requested = list(dict.fromkeys((
            *(item["path"] for item in result.scope_requests),
            *result.requested_files,
        )))
        granted: list[str] = []
        denied: list[tuple[str, str]] = []
        for path in requested:
            if path in contract.primary_files:
                continue
            reason = self._scope_denial(contract, path)
            if reason is None and len(granted) >= MAX_GRANTED_FILES_PER_REVISION:
                reason = (f"more than {MAX_GRANTED_FILES_PER_REVISION} new files "
                          "requested in one revision")
            if reason is None:
                granted.append(path)
            else:
                denied.append((path, reason))
        return ScopeDecision(tuple(granted), tuple(denied))

    def _scope_denial(self, contract: TaskContract, path: str) -> str | None:
        if contract.read_only:
            return "this is a read-only investigation contract"
        if (not path or any(char.isspace() for char in path)
                or ("/" not in path and "." not in path)):
            return "not a project-relative file path"
        if not self._safe_path(path):
            return "outside the project"
        if path_excluded(path, contract.excluded_files):
            return "excluded by user decision"
        if protected_path(path):
            return "protected path (tracking, VCS, or secrets)"
        return None

    def initial_contract(self, task: Task, specs: list[ToolSpec],
                         instruction: str = "", task_context=None) -> TaskContract:
        """`task_context` is an approved planner `TaskContext` for this task, if any."""
        policy = policy_for(task.complexity)
        text = "\n".join((task.title, task.description, task.context, instruction))
        planned_paths = tuple(task_context.relevant_files) if task_context else ()
        paths = tuple(dict.fromkeys(
            path for path in (*_PATH.findall(text), *planned_paths) if self._safe_path(path)))
        symbols = tuple(dict.fromkeys((*_SYMBOL.findall(text),
                                       *(task_context.symbols if task_context else ()))))
        criteria = task.acceptance_criteria or [task.description]
        capability = (task.preferred_capability or "").strip().lower()
        read_only = capability in DISCOVERY_CAPABILITIES
        if read_only:
            # Investigating presence or absence needs the whole repository;
            # named files are starting points, never boundaries or write targets.
            operations = (*DISCOVERY_OPERATIONS, *(
                (OP_SHELL_EXECUTE,) if capability in VERIFICATION_CAPABILITIES else ()))
        else:
            # Named files are expected scope only; the guard does not limit writes
            # to them, so a task that names no file can still create one.
            operations = policy.operations
        allowed = self._allowed_tools(specs, operations)
        context = tuple(filter(None, (
            ContextFact(task.context, "openspec.task.context", paths)
            if task.context.strip() else None,
            ContextFact(instruction, "implementation.guidance")
            if instruction.strip() else None,
            ContextFact(
                "Starting points named by the task: " + ", ".join(paths)
                + ". They are not boundaries; search and read the repository as needed.",
                "tasker.discovery", paths)
            if read_only and paths else None,
            *self._planner_facts(task_context),
        )))
        return TaskContract.create(
            task_id=task.id, planning_revision=task.planning_revision,
            goal=f"{task.title}: {task.description}", complexity=task.complexity,
            acceptance_criteria=tuple(
                AcceptanceCriterion(f"AC{index}", criterion)
                for index, criterion in enumerate(criteria, 1)),
            primary_files=() if read_only else paths, primary_symbols=symbols,
            reference_files=(), known_context=context, allowed_tools=allowed,
            allowed_operations=operations, execution_limits=policy.limits,
            escalation_conditions=(
                "A required file or symbol is outside the contract scope",
                "The task requires broader repository exploration",
                "The assigned complexity or execution budget is unsuitable",
                "A repeated identical tool call limit is reached",
            ),
            read_only=read_only,
        )

    @staticmethod
    def _planner_facts(task_context) -> tuple[ContextFact, ...]:
        if task_context is None:
            return ()
        facts = []
        if task_context.approach:
            facts.append(ContextFact(task_context.approach, "planner.approach",
                                     tuple(task_context.relevant_files)))
        facts.extend(ContextFact(item, "planner.existing_implementation")
                     for item in task_context.existing_implementation)
        facts.extend(ContextFact(item, "planner.constraint") for item in task_context.constraints)
        if task_context.do_not_modify:
            # Advisory, not an exclusion: excluded files cannot even be read.
            facts.append(ContextFact(
                "Do not modify: " + ", ".join(task_context.do_not_modify), "planner.do_not_modify",
                attributes={"do_not_modify": list(task_context.do_not_modify)}))
        facts.extend(ContextFact(finding.statement, f"planner.finding.{finding.id}",
                                 tuple(item["path"] for item in finding.evidence))
                     for finding in task_context.findings)
        return tuple(facts)

    def revise_after_coder(self, contract: TaskContract, result: CoderResult,
                           specs: list[ToolSpec],
                           decision: ScopeDecision | None = None,
                           guard_denial: str = "") -> TaskContract | None:
        if result.outcome == CoderOutcome.SUCCESS:
            return None
        decision = decision or self.grant_scope(contract, result)
        # Only files that pass the grant rules become write scope; a discovery
        # contract already reads everywhere and never gains writes.
        files = () if contract.read_only else tuple(dict.fromkeys(
            (*contract.primary_files, *decision.granted)))
        complexity = contract.complexity
        if result.outcome in {CoderOutcome.SCOPE_TOO_LARGE, CoderOutcome.WRONG_COMPLEXITY}:
            complexity = {"simple": "medium", "medium": "complex"}.get(complexity, "complex")
        policy = policy_for(complexity)
        operations = (contract.allowed_operations if contract.read_only
                      else policy.operations)
        facts = list(contract.known_context)
        facts.extend(ContextFact(item, "coder.discovery") for item in result.discoveries)
        facts.extend(ContextFact(item, "coder.missing_context") for item in result.missing_context)
        if decision.considered or guard_denial:
            facts.append(ContextFact(
                "Tasker scope decision: " + decision.describe()
                + (f". The previous attempt was stopped: {guard_denial}." if guard_denial else "."),
                "tasker.scope_grant", decision.granted, {
                    "granted": list(decision.granted),
                    "denied": [{"path": path, "reason": reason}
                               for path, reason in decision.denied],
                    "guard_denial": guard_denial or None,
                }))
        return replace(
            contract, contract_id=TaskContract.create(
                task_id=contract.task_id, planning_revision=contract.planning_revision,
                goal=contract.goal, complexity=complexity,
                acceptance_criteria=contract.acceptance_criteria,
                primary_files=files, primary_symbols=contract.primary_symbols,
                reference_files=contract.reference_files, known_context=tuple(facts),
                allowed_tools=(), allowed_operations=(), execution_limits=policy.limits,
                escalation_conditions=contract.escalation_conditions,
            ).contract_id,
            parent_contract_id=contract.contract_id, revision=contract.revision + 1,
            complexity=complexity, primary_files=files, known_context=tuple(facts),
            allowed_tools=self._allowed_tools(specs, operations),
            allowed_operations=operations, execution_limits=policy.limits,
            planning_reason=(SCOPE_GRANT_REASON if decision.granted
                             else f"coder reported {result.outcome.value}"),
        )

    def revise_after_review(self, contract: TaskContract, result: ReviewerResult, findings,
                            specs: list[ToolSpec]) -> TaskContract:
        paths = () if contract.read_only else tuple(dict.fromkeys((
            *contract.primary_files,
            *(path for finding in findings for path in finding.artifact_paths),
        )))
        facts = (*contract.known_context, ContextFact(
            result.explanation, "reviewer.result", result.paths,
            {"outcome": result.outcome.value,
             "category": result.category,
             "criterion_ids": list(result.violated_criteria)},
        ), *(
            ContextFact(
                f"Criterion failure: {finding.problem}. Required outcome: "
                + "; ".join(finding.approval_criteria),
                "reviewer.failure", tuple(finding.artifact_paths),
                {"category": finding.kind, "owner": finding.owner,
                 "criterion_ids": list(getattr(finding, "criterion_ids", ()))},
            ) for finding in findings
        ))
        return replace(
            contract, contract_id=TaskContract.create(
                task_id=contract.task_id, planning_revision=contract.planning_revision,
                goal=contract.goal, complexity=contract.complexity,
                acceptance_criteria=contract.acceptance_criteria,
                primary_files=paths, primary_symbols=contract.primary_symbols,
                reference_files=contract.reference_files, known_context=facts,
                allowed_tools=contract.allowed_tools,
                allowed_operations=contract.allowed_operations,
                execution_limits=contract.execution_limits,
                escalation_conditions=contract.escalation_conditions,
            ).contract_id,
            parent_contract_id=contract.contract_id, revision=contract.revision + 1,
            primary_files=paths, known_context=facts,
            planning_reason="review failure evidence",
        )

    def revise_after_escalation(self, contract: TaskContract, guidance: str,
                                specs: list[ToolSpec], add_files: tuple[str, ...] = (),
                                requested_files: tuple[str, ...] = ()) -> TaskContract:
        """Create a user-authorized retry contract without pretending review occurred.

        Tasker decides the scope: files the user explicitly excludes are removed
        (narrow), confirmed `add_files` are added (expand), and otherwise the
        previous scope is preserved. Files the failed attempt requested stay
        outside the contract unless the user confirmed adding them. A discovery
        contract stays read-only and repository-wide; only exclusions change it.
        Exclusions accumulate across revisions and are enforced by the guard.
        """
        excluded = tuple(dict.fromkeys(
            path for path in excluded_paths(guidance) if self._safe_path(path)))
        excluded_files = tuple(dict.fromkeys((*contract.excluded_files, *excluded)))
        added = () if contract.read_only else tuple(dict.fromkeys(
            path for path in add_files
            if self._safe_path(path) and path not in excluded_files
            and path not in contract.primary_files))
        files = tuple(dict.fromkeys((
            *(path for path in contract.primary_files if path not in excluded), *added)))
        narrowed = (len(files) < len(contract.primary_files)
                    or (contract.read_only and bool(excluded)))
        decision = "expand" if added else "narrow" if narrowed else "preserve"
        kept_outside = tuple(dict.fromkeys(
            path for path in (*excluded_files,
                              *(() if contract.read_only else requested_files))
            if path not in files))
        fact = guidance.strip()
        if kept_outside:
            fact += (" Outside this contract by user decision: "
                     + ", ".join(kept_outside) + ". Do not inspect or modify them.")
        policy = policy_for(contract.complexity)
        operations = (contract.allowed_operations if contract.read_only
                      else policy.operations)
        facts = (*contract.known_context, ContextFact(
            fact, "user.contract_resolution", files, {
                "scope_decision": decision,
                "added_files": list(added),
                "excluded_files": list(kept_outside),
            }))
        fresh = TaskContract.create(
            task_id=contract.task_id,
            planning_revision=contract.planning_revision,
            goal=contract.goal,
            complexity=contract.complexity,
            acceptance_criteria=contract.acceptance_criteria,
            primary_files=files,
            primary_symbols=contract.primary_symbols,
            reference_files=contract.reference_files,
            known_context=facts,
            allowed_tools=(), allowed_operations=(),
            execution_limits=policy.limits,
            escalation_conditions=contract.escalation_conditions,
        )
        return replace(
            fresh, parent_contract_id=contract.contract_id,
            revision=contract.revision + 1,
            allowed_tools=self._allowed_tools(specs, operations),
            allowed_operations=operations,
            planning_reason=USER_RESOLUTION_REASON,
            excluded_files=excluded_files, read_only=contract.read_only,
        )

    @staticmethod
    def may_attempt(attempt_count: int) -> bool:
        return attempt_count < MAX_EXECUTION_ATTEMPTS

    @staticmethod
    def _allowed_tools(specs: list[ToolSpec], operations: tuple[str, ...]) -> tuple[str, ...]:
        requested = set(operations)
        # The generic dispatcher can reach any IDE tool; the guard still checks each
        # dispatched call, but read-only contracts without shell access omit it.
        dispatcher_allowed = bool(requested - set(DISCOVERY_OPERATIONS))
        names = []
        for spec in specs:
            decision = classify_tool(spec)
            if decision.accepted and decision.executable and (
                    decision.operations.intersection(requested)
                    or (dispatcher_allowed and not decision.operations
                        and spec.name.endswith("execute_tool"))):
                names.append(spec.name)
        return tuple(names)

    @staticmethod
    def _safe_path(path: str) -> bool:
        parsed = PurePosixPath(path)
        return not parsed.is_absolute() and ".." not in parsed.parts

    @staticmethod
    def _discovery_note(contract: TaskContract) -> str:
        if not contract.read_only:
            return ""
        return """
This is a read-only investigation. You may read and search any file in the
repository except the excluded files; you may not create or change files. To
establish whether something exists, search file contents across the repository
(not only file names). Report SUCCESS only when the searches and reads you actually
ran are listed in verified as evidence; deliver the findings in summary and
discoveries. If you could not run the searches the answer depends on, report
BLOCKED_CONTEXT.
"""

    @staticmethod
    def _scope_note(contract: TaskContract) -> str:
        if contract.read_only:
            return ""
        return """
Expected files are Tasker's estimate, not a limit. Prefer them, but read and
change other project files when the task genuinely needs it, following the
project's existing conventions; give each changed file a purpose in changed_files.
Files changed outside the expected files are shown to the Reviewer, who judges
whether they were needed. Never read or change secrets files (.env and its
variants other than .env.example) or excluded files, and never change
.codepartner/, openspec/, or .git/. Do not work around a denied tool call with
another tool; stop and report it instead.
"""

    @staticmethod
    def prompt(contract: TaskContract) -> str:
        criteria = "\n".join(
            f"- {item.id}: {item.text}" for item in contract.acceptance_criteria)
        context = "\n".join(
            f"- {item.fact} (source: {item.source})" for item in contract.known_context)
        return f"""Execute TaskContract {contract.contract_id} revision {contract.revision}.
Task: {contract.task_id}
Goal: {contract.goal}
Complexity: {contract.complexity}
Acceptance criteria:
{criteria}
Expected files (guidance, not a limit): {list(contract.primary_files)}
Primary symbols: {list(contract.primary_symbols)}
Reference files: {list(contract.reference_files)}
Known context:
{context or '- none'}
Allowed tools: {list(contract.allowed_tools)}
Excluded files (never read, search for, or modify): {list(contract.excluded_files)}
The run stops if you call a tool outside the allowed list, touch a protected or
excluded path, or repeat an identical tool call more than
{contract.execution_limits.repeated_calls} times.
{TaskContractor._discovery_note(contract)}{TaskContractor._scope_note(contract)}
Execute only this task; do not replan it or add unrequested features. If required
context is missing or access you need is denied, stop and report BLOCKED_CONTEXT or
SCOPE_TOO_LARGE with the missing items in requested_files, scope_requests, and
remaining_gaps; do not report SUCCESS for work you could not inspect. End with
exactly one JSON object:
{{"schema_version":2,"task_id":"{contract.task_id}",
"contract_id":"{contract.contract_id}","contract_revision":{contract.revision},
"outcome":"SUCCESS|BLOCKED_CONTEXT|SCOPE_TOO_LARGE|WRONG_COMPLEXITY|IMPLEMENTATION_FAILURE",
"summary":"concise result",
"changed_files":[{{"path":"project/relative/path","change":"created|modified|deleted|renamed|moved","purpose":"why"}}],
"criteria":[{{"id":"AC1","status":"satisfied|unsatisfied|unknown","evidence":"what you checked"}}],
"verified":[],
"discoveries":[],"missing_context":[],"requested_files":[],"remaining_gaps":[],
"scope_requests":[{{"path":"project/relative/path","operation":"create|modify","reason":"why"}}],
"meaningful_progress":false}}
Use [] for changed_files and scope_requests when they do not apply. Report every acceptance criterion
in criteria. List every changed path relative to the project root. Cite only verification that
actually ran. SUCCESS means this contract appears satisfied; Tasker and Reviewer
make the completion decision."""

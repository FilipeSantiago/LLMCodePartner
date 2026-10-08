"""Versioned contracts exchanged by Tasker, Coder, and Reviewer."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class CoderOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    BLOCKED_CONTEXT = "BLOCKED_CONTEXT"
    SCOPE_TOO_LARGE = "SCOPE_TOO_LARGE"
    WRONG_COMPLEXITY = "WRONG_COMPLEXITY"
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"


class ReviewerOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class AcceptanceCriterion:
    id: str
    text: str


@dataclass(frozen=True)
class ContextFact:
    fact: str
    source: str
    paths: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionLimits:
    """Enforced execution limits.

    Call-count budgets were removed until provenance data can justify them;
    telemetry still measures every call. Only the identical-call loop guard remains.
    """

    repeated_calls: int


@dataclass(frozen=True)
class TaskContract:
    schema_version: int
    contract_id: str
    revision: int
    task_id: str
    planning_revision: int
    goal: str
    complexity: str
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    primary_files: tuple[str, ...]
    primary_symbols: tuple[str, ...]
    reference_files: tuple[str, ...]
    known_context: tuple[ContextFact, ...]
    allowed_tools: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    execution_limits: ExecutionLimits
    escalation_conditions: tuple[str, ...]
    parent_contract_id: str | None = None
    planning_reason: str = "initial OpenSpec contract"
    # Paths the user excluded; enforced even when reads are repository-wide.
    excluded_files: tuple[str, ...] = ()
    # A discovery contract: repository-wide reads and search, no writes.
    read_only: bool = False

    @classmethod
    def create(cls, **values) -> "TaskContract":
        return cls(schema_version=1, contract_id="contract_" + uuid4().hex,
                   revision=1, **values)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CoderResult:
    schema_version: int
    task_id: str
    contract_id: str
    contract_revision: int
    outcome: CoderOutcome
    summary: str
    changed_files: tuple[dict[str, Any], ...] = ()
    criteria: tuple[dict[str, Any], ...] = ()
    verified: tuple[str, ...] = ()
    discoveries: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()
    requested_files: tuple[str, ...] = ()
    remaining_gaps: tuple[str, ...] = ()
    meaningful_progress: bool = False
    # Structured requests for files the contract does not grant:
    # {"path": "project/relative", "operation": "create|modify", "reason": "..."}.
    scope_requests: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ReviewerResult:
    outcome: ReviewerOutcome
    violated_criteria: tuple[str, ...] = ()
    category: str | None = None
    explanation: str = ""
    evidence: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionEscalation:
    task_id: str
    contract_id: str
    reason: str
    exhausted_limit: str | None
    counters: dict[str, int]
    unresolved_criteria: tuple[str, ...]
    evidence: tuple[str, ...] = ()

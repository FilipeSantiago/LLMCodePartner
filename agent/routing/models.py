"""Typed contracts for model selection and task execution."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from agent.spec.models import Task
from mcp_bridge.models import Event
from mcp_bridge.registry import ToolSpec


@dataclass(frozen=True)
class ModelRef:
    name: str
    provider: str
    executor: str
    cli_model: str | None
    gateway_mode: str = "preferred"
    reasoning_effort: str | None = None
    review_reasoning_effort: str | None = None
    # Role-independent strength rank used by per-role capability floors.
    capability: int | None = None


@dataclass(frozen=True)
class RoutingRecommendation:
    model: str
    reason: str | None = None


@dataclass(frozen=True)
class ExecutionDecision:
    decision_id: str
    project_id: str
    change_id: str
    task_id: str
    complexity: str
    candidates: tuple[str, ...]
    recommended_model: str
    recommended_provider: str
    executor: str
    fallback_models: tuple[str, ...]
    routing_reason: str | None
    routing_source: str
    routed_at: datetime


@dataclass(frozen=True)
class ExecutionContext:
    messages: list[dict]
    specs: list[ToolSpec]
    decision: ExecutionDecision
    gateway_url: str | None = None
    use_gateway: bool = False
    started_at: float = 0.0
    direct_tool_handlers: dict[str, Any] = field(default_factory=dict)
    # Owner identity (implementation job/position) stamped on the run at creation.
    owner_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    latency_ms: int
    gateway_used: bool
    usage: dict | None = None
    error: str | None = None


@dataclass(frozen=True)
class ImplementationJob:
    """A durable, sequential request to implement one or more OpenSpec tasks."""

    job_id: str
    selectors: tuple[str, ...]
    task_ids: tuple[str, ...]
    instruction: str = ""
    status: str = "queued"
    # The approved WP plan whose context this job's coders and reviewer receive.
    plan_id: str | None = None

    @classmethod
    def create(cls, selectors: list[str], task_ids: list[str],
               instruction: str = "", plan_id: str | None = None) -> "ImplementationJob":
        return cls(str(uuid4()), tuple(selectors), tuple(task_ids), instruction=instruction,
                   plan_id=plan_id)


# Job-level statuses. Active jobs may still change project files; a halted job
# must never advance, even if a cancelled run's request finishes later.
ACTIVE_JOB_STATUSES = frozenset({
    "queued", "running", "waiting", "reviewing", "implemented",
    "stopping", "pending_switch",
})
HALTED_JOB_STATUSES = frozenset({"stopping", "stopped", "cancelled"})
TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "stopped", "undone", "cancelled"})


@dataclass(frozen=True)
class ImplementationJobTask:
    job_id: str
    position: int
    change_id: str
    task_id: str
    status: str = "queued"
    decision_id: str | None = None
    error: str | None = None


class CLIExecutor(Protocol):
    """Provider-neutral task executor.

    Execution is an async event stream because a JetBrains tool call deliberately
    pauses the CLI and resumes on a later HTTP request.
    """

    name: str

    def execute(
        self, task: Task, model: ModelRef, context: ExecutionContext
    ) -> AsyncIterator[Event]: ...

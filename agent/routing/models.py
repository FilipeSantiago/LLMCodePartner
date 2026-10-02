"""Typed contracts for model selection and task execution."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
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
    status: str = "queued"

    @classmethod
    def create(cls, selectors: list[str], task_ids: list[str]) -> "ImplementationJob":
        return cls(str(uuid4()), tuple(selectors), tuple(task_ids))


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

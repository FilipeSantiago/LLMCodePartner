"""Data transferred into the telemetry store.

Telemetry deliberately contains measurements and identifiers, never prompts,
source text, tool arguments, or complete tool output.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TelemetryContext:
    job_id: str | None = None
    task_id: str | None = None
    contract_id: str | None = None
    contract_revision: int | None = None
    attempt: int | None = None
    role: str = "coder"


@dataclass(frozen=True)
class ToolActivity:
    context: TelemetryContext
    call_id: str
    tool_name: str
    category: str
    signature: str
    paths: tuple[str, ...] = ()
    repeated: bool = False
    calls_before_first_edit: int = 0


@dataclass(frozen=True)
class AttemptSummary:
    context: TelemetryContext
    outcome: str
    tool_calls: int
    calls_by_category: dict[str, int]
    unique_files_read: tuple[str, ...]
    output_chars: int
    repeated_calls: int
    calls_before_first_edit: int
    meaningful_progress: int
    usage: dict[str, Any] | None = None
    escalation_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowEvent:
    """Metadata-only observation of one workflow-proposal lifecycle step.

    The raw user message is never stored; `message_fingerprint` is its SHA-256.
    """

    stage: str
    proposal_id: str | None = None
    action: str | None = None
    source_state: str | None = None
    interpretation_source: str | None = None
    model: str | None = None
    confidence: str | None = None
    ambiguity_count: int = 0
    selector_count: int = 0
    guidance_present: bool = False
    latency_ms: int | None = None
    confirmation_latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    transition_result: str | None = None
    message_fingerprint: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class TurnEvent:
    """Metadata-only record of how one request was routed.

    `turn_type` is `user`, `tool_continuation`, or `bootstrap_replay`. The two
    classifications are summaries (action, confidence, counts); prompts, guidance,
    and file contents are never recorded.
    """

    conversation_id: str | None
    project_id: str | None
    turn_type: str
    route: str
    free_classification: dict[str, Any] | None = None
    contextual_classification: dict[str, Any] | None = None
    proposal_id: str | None = None
    outcome: str | None = None
    message_fingerprint: str | None = None

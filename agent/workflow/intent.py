"""Workflow intents, proposals, and the bounded context they are interpreted in."""

from dataclasses import dataclass, field, replace
from typing import Any

from agent.workflow.states import (
    ACTION_DESCRIPTIONS, RECOVERY_MODES, STATE_DESCRIPTIONS, WorkflowAction,
    WorkflowState, valid_actions,
)

_MAX_CAUSE_CHARS = 300
_MAX_CONTEXT_TASKS = 5


@dataclass(frozen=True)
class PausedTask:
    """Durable facts about one task awaiting a user decision."""

    job_id: str
    position: int
    task_id: str
    state: str
    cause: str = ""
    instruction: str = ""
    contract_id: str | None = None
    contract_revision: int | None = None
    files_allowed: tuple[str, ...] = ()
    files_requested: tuple[str, ...] = ()
    review_completed: bool = False
    review_question: str | None = None

    @property
    def actions(self) -> tuple[WorkflowAction, ...]:
        return valid_actions(self.state)

    def render(self) -> str:
        try:
            meaning = STATE_DESCRIPTIONS[WorkflowState(self.state)]
        except ValueError:
            meaning = "unknown"
        lines = [
            f"Current job: {self.job_id}",
            f"Current task: {self.task_id}",
            f"State: {self.state} ({meaning})",
            f"Cause: {(self.cause or 'not recorded')[:_MAX_CAUSE_CHARS]}",
        ]
        if self.contract_revision is not None:
            lines.extend((
                f"Contract revision: {self.contract_revision}",
                f"Files allowed: {', '.join(self.files_allowed) or 'none'}",
            ))
        if self.files_requested:
            lines.append(f"Files requested: {', '.join(self.files_requested)}")
        lines.append(f"Review completed: {str(self.review_completed).lower()}")
        if self.review_question:
            lines.append(f"Reviewer question: {self.review_question[:_MAX_CAUSE_CHARS]}")
        lines.append("Valid actions:")
        for action in self.actions:
            modes = RECOVERY_MODES.get(WorkflowState(self.state), ())
            suffix = (f" (recovery_mode: {', '.join(modes)})"
                      if action == WorkflowAction.RESOLVE_CODER_RECOVERY else "")
            lines.append(f"- {action.value}: {ACTION_DESCRIPTIONS[action]}{suffix}")
        return "\n".join(lines)


@dataclass(frozen=True)
class WorkflowContext:
    conversation_id: str | None
    paused: tuple[PausedTask, ...] = ()
    refinement_session_id: str | None = None
    pending_action: str = ""
    planning_session_id: str | None = None
    planning_wp_id: str | None = None

    def render(self) -> str:
        shown = self.paused[:_MAX_CONTEXT_TASKS]
        blocks = [task.render() for task in shown]
        if len(self.paused) > len(shown):
            blocks.append(f"({len(self.paused) - len(shown)} more paused task(s) not shown)")
        if self.pending_action:
            blocks.append(f"Pending proposal: {self.pending_action[:300]}")
        return "\n\n".join(blocks) or "No paused workflow."

    def task(self, job_id: str | None, position: int | None) -> PausedTask | None:
        return next((task for task in self.paused
                     if task.job_id == job_id and task.position == position), None)


@dataclass(frozen=True)
class WorkflowIntent:
    """What the user asked the workflow to do, before anything executes."""

    conversation_id: str
    message_fingerprint: str
    action: WorkflowAction
    workflow_context: str = ""
    source_state: str | None = None
    target_job: str | None = None
    target_position: int | None = None
    target_task: str | None = None
    selectors: tuple[str, ...] = ()
    guidance: str = ""
    confidence: str = "high"
    ambiguities: tuple[str, ...] = ()
    expected_effects: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_revision: int | None = None
    interpretation_source: str = "deterministic"
    model: str | None = None
    latency_ms: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    clarification: str = ""
    interpretation_error: str = ""

    def with_effects(self, effects: tuple[str, ...]) -> "WorkflowIntent":
        return replace(self, expected_effects=effects)

    def payload(self) -> dict[str, Any]:
        """The normalized action data persisted with a proposal (never the raw message)."""
        return {
            "target_task": self.target_task,
            "selectors": list(self.selectors),
            "guidance": self.guidance,
            "expected_effects": list(self.expected_effects),
            "parameters": self.parameters,
            "refinement_session_id": self.parameters.get("refinement_session_id"),
        }

    @classmethod
    def from_proposal(cls, row: dict) -> "WorkflowIntent":
        payload = row["payload"]
        return cls(
            conversation_id=row["conversation_id"],
            message_fingerprint=row["request_fingerprint"],
            action=WorkflowAction(row["action"]),
            source_state=row.get("expected_status"),
            target_job=row.get("target_job_id"),
            target_position=row.get("target_position"),
            target_task=row.get("target_task_id") or payload.get("target_task"),
            selectors=tuple(payload.get("selectors", ())),
            guidance=payload.get("guidance", ""),
            confidence=row["confidence"],
            ambiguities=tuple(row.get("ambiguities", ())),
            expected_effects=tuple(payload.get("expected_effects", ())),
            parameters=dict(payload.get("parameters", {})),
            expected_revision=row.get("expected_revision"),
            interpretation_source=row.get("interpretation_source") or "deterministic",
            model=row.get("interpreter_model"),
        )


ACTION_TITLES: dict[WorkflowAction, str] = {
    WorkflowAction.START_IMPLEMENTATION: "Start implementation",
    WorkflowAction.CODING: "Ask the coding agent to change the project",
    WorkflowAction.RETRY_CONTRACT: "Retry {task} with a revised TaskContract",
    WorkflowAction.EXPAND_CONTRACT_SCOPE: "Retry {task} with an expanded TaskContract",
    WorkflowAction.RETRY_CODER_AFTER_REVIEW: "Send {task} back to the Coder after review",
    WorkflowAction.RETRY_REVIEW: "Run the Reviewer again for {task}",
    WorkflowAction.ACCEPT_REVIEW: "Accept {task}",
    WorkflowAction.RESOLVE_CODER_RECOVERY: "Recover the interrupted Coder run for {task}",
    WorkflowAction.ANSWER_REVIEWER_QUESTION: "Answer the Reviewer's question for {task}",
    WorkflowAction.APPLY_SPEC: "Apply the active OpenSpec draft",
    WorkflowAction.DISCARD_SPEC: "Discard the active OpenSpec draft",
    WorkflowAction.EXECUTE_COMMAND: "Run a workflow command",
    WorkflowAction.STOP_JOB: "Stop the job at {task}",
    WorkflowAction.APPROVE_WP_PLAN: "Approve the plan and implement {task}",
}


@dataclass(frozen=True)
class WorkflowActionProposal:
    proposal_id: str
    confirmation_token: str
    intent: WorkflowIntent
    refreshed: bool = False

    @property
    def action(self) -> WorkflowAction:
        return self.intent.action

    def render(self) -> str:
        intent = self.intent
        title = ACTION_TITLES[intent.action].format(task=intent.target_task or "the task")
        lines = []
        if self.refreshed:
            lines.extend(("The workflow changed or the previous proposal expired, "
                          "so here is a refreshed proposal.", ""))
        # Imported here: switching depends on routing persistence, which this
        # lightweight value module must not import at load time.
        from agent.workflow.switching import paused_note, switch_preamble
        displaced = intent.parameters.get("displace") or []
        if displaced:
            lines.extend((switch_preamble(displaced, intent.conversation_id,
                                          intent.parameters.get("start_phrase", "continue")),
                          "", "If you confirm:", "",
                          f"- Stop {', '.join(item['label'] for item in displaced)} "
                          "and wait until it has stopped safely",
                          f"- Then: {title}"))
        else:
            lines.extend(("I understood your request as:", "", f"- Action: {title}"))
        lines.extend(f"- {effect}" for effect in intent.expected_effects)
        paused = intent.parameters.get("paused_jobs") or []
        if paused:
            lines.extend(("", paused_note(paused)))
        assumptions = []
        if intent.target_task:
            assumptions.append(
                f"The target is `{intent.target_task}` in job `{intent.target_job}`"
                + (f", currently `{intent.source_state}`" if intent.source_state else ""))
        if intent.selectors:
            assumptions.append("The selected OpenSpec scope is "
                               + ", ".join(f"`{item}`" for item in intent.selectors))
        if assumptions:
            lines.extend(("", "Assumptions:",
                          *[f"- {item}" for item in assumptions]))
        if intent.ambiguities:
            lines.extend(("", "Ambiguities:", *[f"- {item}" for item in intent.ambiguities]))
        token = self.confirmation_token
        lines.extend((
            "", "Nothing has executed yet.", "",
            f"Reply `confirm {token}` or `reject {token}`. "
            f"To adjust it, reply `change {token}: <correction>`.",
        ))
        return "\n".join(lines)

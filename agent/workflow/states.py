"""Paused workflow states and the actions each state permits.

The table is the single source of truth for which user actions may follow a
paused implementation task. The interpreter may only choose from it, and
confirmation re-checks it before anything executes.
"""

from enum import StrEnum


class WorkflowState(StrEnum):
    CONTRACT_RESOLUTION = "awaiting_contract_resolution"
    REVIEW_RESOLUTION = "awaiting_review_resolution"
    CODER_RECOVERY = "awaiting_coder_recovery"
    JOURNAL_RECOVERY = "recovery_required"
    HUMAN_CONFIRMATION = "awaiting_human_confirmation"
    EVIDENCE_RESOLUTION = "awaiting_evidence_resolution"
    MANUAL_RESOLUTION = "awaiting_manual_resolution"


class WorkflowAction(StrEnum):
    START_IMPLEMENTATION = "START_IMPLEMENTATION"
    CODING = "CODING"
    RETRY_CONTRACT = "RETRY_CONTRACT"
    EXPAND_CONTRACT_SCOPE = "EXPAND_CONTRACT_SCOPE"
    RETRY_CODER_AFTER_REVIEW = "RETRY_CODER_AFTER_REVIEW"
    RETRY_REVIEW = "RETRY_REVIEW"
    ACCEPT_REVIEW = "ACCEPT_REVIEW"
    UNDO_IMPLEMENTATION = "UNDO_IMPLEMENTATION"
    RESOLVE_CODER_RECOVERY = "RESOLVE_CODER_RECOVERY"
    ANSWER_REVIEWER_QUESTION = "ANSWER_REVIEWER_QUESTION"
    APPLY_SPEC = "APPLY_SPEC"
    DISCARD_SPEC = "DISCARD_SPEC"
    EXECUTE_COMMAND = "EXECUTE_COMMAND"
    STOP_JOB = "STOP_JOB"
    # Approve one revision of an interactive WP plan and start implementing it.
    APPROVE_WP_PLAN = "APPROVE_WP_PLAN"
    ANSWER = "ANSWER"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


A = WorkflowAction
S = WorkflowState

ACTIONS_BY_STATE: dict[WorkflowState, tuple[WorkflowAction, ...]] = {
    S.CONTRACT_RESOLUTION: (A.RETRY_CONTRACT, A.EXPAND_CONTRACT_SCOPE, A.STOP_JOB),
    S.REVIEW_RESOLUTION: (A.RETRY_CODER_AFTER_REVIEW, A.ACCEPT_REVIEW,
                          A.UNDO_IMPLEMENTATION, A.STOP_JOB),
    S.CODER_RECOVERY: (A.RESOLVE_CODER_RECOVERY, A.UNDO_IMPLEMENTATION, A.STOP_JOB),
    S.JOURNAL_RECOVERY: (A.RESOLVE_CODER_RECOVERY, A.STOP_JOB),
    S.HUMAN_CONFIRMATION: (A.ANSWER_REVIEWER_QUESTION, A.RETRY_REVIEW, A.STOP_JOB),
    S.EVIDENCE_RESOLUTION: (A.RETRY_REVIEW, A.STOP_JOB),
    S.MANUAL_RESOLUTION: (A.STOP_JOB,),
}

# RESOLVE_CODER_RECOVERY covers two different recovery states. `review_handoff`
# sends a blocked (complete but gap-reporting) coder handoff to the Reviewer.
RECOVERY_MODES: dict[WorkflowState, tuple[str, ...]] = {
    S.CODER_RECOVERY: ("retry_coder", "review_handoff"),
    S.JOURNAL_RECOVERY: ("inspect", "establish_baseline"),
}

PAUSED_STATES = frozenset(state.value for state in ACTIONS_BY_STATE)

# Actions that only make sense with user-supplied guidance.
GUIDANCE_REQUIRED = frozenset({A.RETRY_CONTRACT, A.RETRY_CODER_AFTER_REVIEW})

ACTION_DESCRIPTIONS: dict[WorkflowAction, str] = {
    A.START_IMPLEMENTATION: "start implementing the selected OpenSpec tasks",
    A.CODING: "ask the coding agent to perform an ad-hoc project change",
    A.RETRY_CONTRACT: "retry the task with a revised TaskContract, keeping its file scope",
    A.EXPAND_CONTRACT_SCOPE: "retry the task with additional files added to its TaskContract",
    A.RETRY_CODER_AFTER_REVIEW: "send the reviewed task back to the Coder with guidance",
    A.RETRY_REVIEW: "run the Reviewer again",
    A.ACCEPT_REVIEW: "accept the implementation (overriding a failed review if needed)",
    A.UNDO_IMPLEMENTATION: "restore this task's changes and stop the job",
    A.RESOLVE_CODER_RECOVERY: "recover an interrupted or blocked Coder run",
    A.ANSWER_REVIEWER_QUESTION: "answer the Reviewer's question (yes completes the task)",
    A.APPLY_SPEC: "apply the active OpenSpec draft",
    A.DISCARD_SPEC: "discard the active OpenSpec draft",
    A.EXECUTE_COMMAND: "run the workflow command",
    A.STOP_JOB: "stop the job and keep the current files for manual inspection",
    A.APPROVE_WP_PLAN: "approve the current WP plan revision and start implementing it",
    A.ASK_CLARIFICATION: "ask which action is intended",
}

STATE_DESCRIPTIONS: dict[WorkflowState, str] = {
    S.CONTRACT_RESOLUTION: "stopped before review: the TaskContract was violated or exhausted",
    S.REVIEW_RESOLUTION: "the Reviewer finished and a decision is needed",
    S.CODER_RECOVERY: "the Coder was interrupted before finishing",
    S.JOURNAL_RECOVERY: "the mutation journal is incomplete and needs recovery",
    S.HUMAN_CONFIRMATION: "the Reviewer asked a question",
    S.EVIDENCE_RESOLUTION: "the Reviewer could not obtain enough evidence",
    S.MANUAL_RESOLUTION: "paused for an unknown reason; manual resolution needed",
}


def valid_actions(state: str) -> tuple[WorkflowAction, ...]:
    try:
        return ACTIONS_BY_STATE[WorkflowState(state)]
    except ValueError:
        return ()


def action_allowed(state: str, action: str, recovery_mode: str = "") -> bool:
    try:
        workflow_state = WorkflowState(state)
        workflow_action = WorkflowAction(action)
    except ValueError:
        return False
    if workflow_action not in ACTIONS_BY_STATE.get(workflow_state, ()):
        return False
    if workflow_action == A.RESOLVE_CODER_RECOVERY:
        return recovery_mode in RECOVERY_MODES.get(workflow_state, ())
    return True

"""Decide what a fresh user turn is: a new flow, a continuation, a question, or unclear.

Two bounded classifications feed this module. The *context-free* one asks what
the message requests on its own ("implement WP5 and WP6" is new implementation
work). The *contextual* one asks whether it answers something already waiting:
a paused task, a Reviewer question, a pending proposal. Neither may execute
anything. `reconcile` only picks a route; it never forces an ambiguous reply
onto a guessed target.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from agent.workflow.intent import WorkflowIntent
from agent.workflow.states import WorkflowAction as A

NEW_FLOW_ACTIONS = frozenset({"START_IMPLEMENTATION", "CODING", "REVIEW", "SPEC_DRAFT"})
# New flows that compete with a continuation rather than refine it. A free-form
# coding instruction while a task is paused is usually guidance for that task.
_COMPETING_FLOWS = frozenset({"START_IMPLEMENTATION", "SPEC_DRAFT", "REVIEW"})


class Route(StrEnum):
    NEW_FLOW = "new_flow"
    CONTINUATION = "continuation"
    QUESTION = "question"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class Classification:
    """A typed, metadata-only view of one interpretation (never the raw message)."""

    action: str
    target: tuple[str, ...] = ()
    confidence: str = "medium"
    uncertainties: tuple[str, ...] = ()
    source: str = "model"

    @classmethod
    def from_fresh(cls, value: dict) -> "Classification":
        action = value.get("action", "ASK_CLARIFICATION")
        uncertainties = (value["reason"],) if value.get("reason") else ()
        return cls(action, tuple(value.get("selectors", ())),
                   value.get("confidence", "low" if action == "ASK_CLARIFICATION" else "medium"),
                   uncertainties, value.get("source", "model"))

    @classmethod
    def from_intent(cls, intent: WorkflowIntent) -> "Classification":
        target = (intent.target_task,) if intent.target_task else ()
        uncertainties = intent.ambiguities or (
            (intent.clarification.splitlines()[0],) if intent.clarification else ())
        return cls(intent.action.value, target, intent.confidence, uncertainties,
                   intent.interpretation_source)

    def summary(self) -> dict:
        return {"action": self.action, "target_count": len(self.target),
                "confidence": self.confidence, "uncertainty_count": len(self.uncertainties),
                "source": self.source}


@dataclass(frozen=True)
class Reconciliation:
    route: Route
    reason: str
    free: Classification | None = None
    contextual: Classification | None = None
    alternatives: tuple[str, ...] = field(default_factory=tuple)


def needs_free_classification(contextual: WorkflowIntent | None) -> bool:
    """Skip the second model call when the contextual reading is already certain.

    Explicit protocol phrases ("retry review", "accept", "stop job") and replies
    bound to a named task are deterministic; only model readings are cross-checked.
    """
    return contextual is None or contextual.interpretation_source != "deterministic"


def reconcile(free: Classification | None,
              contextual: Classification | None) -> Reconciliation:
    if contextual is None:
        if free is None:
            return Reconciliation(Route.UNCLEAR, "nothing was classified")
        if free.action in NEW_FLOW_ACTIONS:
            return Reconciliation(Route.NEW_FLOW, "the message requests new work", free)
        if free.action == "ANSWER":
            return Reconciliation(Route.QUESTION, "an ordinary question", free)
        return Reconciliation(Route.UNCLEAR, "the request is unclear", free)

    action = contextual.action
    if action not in {A.ANSWER.value, A.ASK_CLARIFICATION.value}:
        if contextual.source == "deterministic" or free is None:
            return Reconciliation(Route.CONTINUATION, "an explicit reply to waiting work",
                                  free, contextual)
        if (free.action in _COMPETING_FLOWS and free.confidence in {"high", "medium"}
                and contextual.confidence != "high"):
            return Reconciliation(
                Route.UNCLEAR, "the message could start new work or continue paused work",
                free, contextual, (free.action, action))
        return Reconciliation(Route.CONTINUATION, "a reply to waiting work", free, contextual)

    if free is not None and free.action in NEW_FLOW_ACTIONS and free.confidence != "low":
        return Reconciliation(Route.NEW_FLOW, "the message requests new work", free, contextual)
    if action == A.ANSWER.value:
        return Reconciliation(Route.QUESTION, "a question about waiting work", free, contextual)
    return Reconciliation(Route.UNCLEAR, "the reply does not match waiting work",
                          free, contextual)

"""Two-phase workflow control: interpret → propose → confirm/reject → execute.

Before confirmation this module only reads workflow metadata, interprets the
message, and saves a pending proposal. It never invokes the Coder, Reviewer,
tests, or mutation tools until a proposal has been atomically claimed and
revalidated.
"""

import hashlib
import re
import secrets
import sqlite3
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from enum import StrEnum
from uuid import uuid4

from agent.planning.models import plan_id_for
from agent.routing.models import ImplementationJob, ImplementationJobTask
from agent.routing.persistence import RoutingPersistence
from agent.spec.implementation_intent import ImplementationIntent, selectors_in_text
from agent.task_contracts.contractor import requested_paths
from agent.task_contracts.persistence import TaskContractPersistence
from agent.workflow import switching
from agent.workflow.intent import (
    ACTION_TITLES, PausedTask, WorkflowActionProposal, WorkflowContext, WorkflowIntent,
)
from agent.workflow.interpreter import (
    ConversationIntentInterpreter, WorkflowInterpreter,
    clarification_text, expected_effects,
)
from agent.workflow.reconciler import (
    Classification, Reconciliation, Route, needs_free_classification, reconcile,
)
from agent.workflow.review_retry import explicit_retry_request, menu_choices
from agent.workflow.states import WorkflowAction as A, WorkflowState as S, action_allowed
from mcp_bridge.models import ErrorEvent, Event, TextEvent
from conversation.presentation import ensure_newline
from telemetry.models import TurnEvent, WorkflowEvent
from telemetry.recorder import TelemetryRecorder
from telemetry.storage import TelemetryStorage

PROPOSAL_TTL_SECONDS = 30 * 60
# ImplementationIntent.model for scope that came from a command, not a model call.
COMMAND_SOURCE = "command"
_TOKEN_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_TOKEN_LENGTH = 6
_DECISION = re.compile(
    r"\s*(confirm|reject|change)\s+([A-Za-z0-9]{4,8})(?:\s*:\s*(.+))?\s*",
    re.IGNORECASE | re.DOTALL)
BARE_CONFIRM = frozenset({"yes", "y", "confirm", "approve"})
BARE_REJECT = frozenset({"no", "n", "reject", "cancel"})
# Natural approvals of the plan under discussion (only while a WP is being planned).
PLAN_APPROVAL = re.compile(
    r"\s*(?:approved?|approve and implement|approve the plan|lgtm|looks good"
    r"(?:[,;]?\s*(?:go ahead|let'?s (?:implement|do) it|implement it|ship it))?"
    r"|go ahead|ship it|let'?s (?:implement|do|build) it)[.!]*\s*",
    re.IGNORECASE)
PLAN_DISCUSS = re.compile(r"\s*discuss(?: more)?[.!]*\s*", re.IGNORECASE)
PLAN_ABANDON = re.compile(
    r"\s*(?:stop|abandon|drop|cancel|never ?mind)(?: (?:the )?(?:plan|planning))?[.!]*\s*",
    re.IGNORECASE)
_WP_ONLY = re.compile(r"WP\d+")
_SPEC_MARKERS = {
    "/__workflow_spec_draft__": "Draft a new OpenSpec change",
    "/__workflow_spec_update__": "Revise the active OpenSpec draft",
}
# Actions the HTTP layer executes by replaying the confirmed command.
PASSTHROUGH_ACTIONS = frozenset({A.CODING, A.EXECUTE_COMMAND,
                                 A.APPLY_SPEC, A.DISCARD_SPEC})
# Actions that may change project files. Only one such flow may be active per
# project; proposing one while another job is active asks to switch instead.
MUTATING_ACTIONS = frozenset({
    A.START_IMPLEMENTATION, A.APPROVE_WP_PLAN, A.CODING, A.RETRY_CONTRACT, A.EXPAND_CONTRACT_SCOPE,
    A.RETRY_CODER_AFTER_REVIEW, A.ACCEPT_REVIEW,
    A.RESOLVE_CODER_RECOVERY, A.ANSWER_REVIEWER_QUESTION,
})


def _preview(text: str, limit: int = 200) -> str:
    """One-line excerpt for proposal text; the full request is kept in the payload."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def fingerprint(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _start_phrase(intent: WorkflowIntent) -> str:
    """How the switch question names the new work ("start implementing WP5 and WP6")."""
    if intent.action in {A.START_IMPLEMENTATION, A.APPROVE_WP_PLAN}:
        return f"start implementing {switching.join_names(intent.selectors) or 'the selected tasks'}"
    if intent.action == A.CODING:
        return "start the requested coding change"
    title = ACTION_TITLES[intent.action].format(task=intent.target_task or "the task")
    return title[0].lower() + title[1:]


@dataclass(frozen=True)
class Decision:
    verb: str
    token: str
    correction: str = ""


def parse_decision(message: str | None) -> Decision | None:
    match = _DECISION.fullmatch(message or "")
    if match is None:
        return None
    verb = match.group(1).lower()
    correction = (match.group(3) or "").strip()
    if (verb == "change") != bool(correction):
        return None
    return Decision(verb, match.group(2).upper(), correction)


@dataclass(frozen=True)
class ClaimResult:
    """Outcome of a confirmation attempt. Only `claimed` may execute."""

    status: str
    proposal: dict
    message: str = ""
    job: ImplementationJob | None = None
    refreshed: WorkflowActionProposal | None = None

    @property
    def claimed(self) -> bool:
        return self.status == "claimed"

    def render(self) -> str:
        if self.refreshed is not None:
            return f"{self.message}\n\n{self.refreshed.render()}"
        return self.message


class ConversationPhase(StrEnum):
    READY = "ready"
    PROPOSED = "proposed"
    PAUSED = "paused"
    RUNNING = "running"
    PLANNING = "planning"


@dataclass(frozen=True)
class ConversationSnapshot:
    """Authoritative workflow facts for interpreting one user turn."""

    phase: ConversationPhase
    context: WorkflowContext
    pending: tuple[dict, ...]
    running_jobs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnResult:
    """How a fresh user turn was routed. Only `answer` reaches a model reply."""

    kind: str  # "reply", "proposal", "answer", or "plan_start"
    text: str = ""
    proposal: WorkflowActionProposal | None = None
    wp_id: str = ""
    queued_wp_ids: tuple[str, ...] = ()

    def render(self) -> str:
        return self.proposal.render() if self.proposal is not None else self.text


@dataclass(frozen=True)
class PlanStart:
    """A WP implementation request starts interactive planning of its first WP.

    The other WPs are planned one at a time, each after the previous one completes,
    so every plan sees the code its predecessors actually produced.
    """

    wp_ids: tuple[str, ...]
    instruction: str


class ConversationWorkflow:
    def __init__(self, persistence: RoutingPersistence,
                 contract_persistence: TaskContractPersistence | None = None,
                 telemetry: TelemetryRecorder | None = None,
                 interpreter: WorkflowInterpreter | None = None,
                 project_id: str | None = None):
        self.persistence = persistence
        path = getattr(persistence, "path", None)
        self.contracts = contract_persistence or TaskContractPersistence(path)
        self.telemetry = telemetry or TelemetryRecorder(TelemetryStorage(path))
        self._interpreter = interpreter
        self.project_id = project_id
        self._planning_available: bool | None = None

    @property
    def planning_available(self) -> bool:
        """WP implementation is planned first whenever a planner is configured."""
        if self._planning_available is None:
            try:
                from agent.routing.config import load_routing_config
                self._planning_available = load_routing_config().planner is not None
            except Exception:
                self._planning_available = False
        return self._planning_available

    async def plan_start_for_selectors(self, artifacts, selectors,
                                       instruction: str) -> PlanStart | str:
        if not self.planning_available:
            return "Work Package planning is unavailable; implementation cannot start."
        wp_ids = []
        for selector in selectors:
            if _WP_ONLY.fullmatch(selector):
                kind, _, _ = await artifacts.object_by_id(selector)
                if kind != "work_package":
                    return f"{selector} is not a Work Package."
                wp_ids.append(selector)
            elif re.fullmatch(r"WP\d+-T\d+", selector):
                parent = selector.split("-T", 1)[0]
                await artifacts.task_by_id(selector)
                return (f"Start with `{parent}` so the Work Package can be planned before "
                        f"implementing {selector}.")
            elif re.fullmatch(r"EPIC\d+", selector):
                kind, change, _ = await artifacts.object_by_id(selector)
                if kind != "epic":
                    return f"{selector} is not an Epic."
                wp_ids.extend(package.id for package in change.work_packages
                              if any(task.status not in {"completed", "cancelled"}
                                     and not task.completed for task in package.tasks))
            else:
                return f"Unknown implementation selector: {selector}."
        wp_ids = list(dict.fromkeys(wp_ids))
        if not wp_ids:
            return "The selected work has no unfinished Work Packages."
        return PlanStart(tuple(wp_ids), instruction)

    @property
    def interpreter(self) -> WorkflowInterpreter:
        if self._interpreter is None:
            self._interpreter = WorkflowInterpreter()
        return self._interpreter

    async def interpret_fresh(self, conversation_id: str, message: str,
                              mode_hint: str, artifacts=None,
                              snapshot: ConversationSnapshot | None = None):
        """Return a proposal, a clarification, or None for an ordinary answer."""
        classification = await self.classify_fresh(message, mode_hint)
        return await self.propose_fresh(conversation_id, message, mode_hint,
                                        classification, artifacts)

    async def classify_fresh(self, message: str, mode_hint: str) -> dict:
        """The context-free reading: what the message requests on its own.

        It deliberately sees no workflow state, so a paused task cannot make a new
        request look like a reply; `reconcile` compares it with the contextual one.
        """
        return await ConversationIntentInterpreter().interpret(message, mode_hint, "")

    async def propose_fresh(self, conversation_id: str, message: str, mode_hint: str,
                            classification: dict, artifacts=None):
        """Turn a context-free classification into a proposal or a reply."""
        action = classification["action"]
        if mode_hint == "review" and action in {"START_IMPLEMENTATION", "SPEC_DRAFT"}:
            return ("Review mode is read-only. Switch modes to propose implementation "
                    "or specification changes. Nothing was executed.")
        if action == "ANSWER":
            return None
        if action == "ASK_CLARIFICATION":
            return f"Nothing was executed. {classification['reason']}"
        selectors = classification["selectors"]
        instruction = classification["instruction"]
        if action == "START_IMPLEMENTATION":
            if artifacts is None:
                return "An IDE MCP session is needed to identify the tasks. Nothing was executed."
            return await self.plan_start_for_selectors(
                artifacts, selectors, instruction or message.strip())
        if action == "CODING":
            if mode_hint == "review":
                return "Review mode is read-only. Nothing was executed."
            return await self.propose(WorkflowIntent(
                conversation_id=conversation_id,
                message_fingerprint=fingerprint(message), action=A.CODING,
                expected_effects=(f"Ask the coding agent to: {_preview(message)}",
                                  "Project files may be changed"),
                parameters={"command": "/__workflow_coding__ " + message.strip()},
                confidence=classification["confidence"],
                interpretation_source="model"))
        if action == "REVIEW":
            command = "/review " + " ".join(selectors)
            if instruction:
                command += ": " + instruction
            return await self.propose(self.command_intent(conversation_id, command))
        if action == "SPEC_DRAFT":
            active = await self.persistence.active_refinement(conversation_id)
            operation = "update" if active else "draft"
            # The spec is generated from the user's full text, never the classifier's
            # paraphrase of it.
            return await self.propose(self.command_intent(
                conversation_id, f"/__workflow_spec_{operation}__ " + message.strip(),
                refinement_session_id=active["session_id"] if active else None))
        return "Nothing was executed. I could not identify an available action."

    # --- turn routing --------------------------------------------------------

    async def route_turn(self, conversation_id: str, text: str, mode_hint: str,
                         artifacts, snapshot: ConversationSnapshot,
                         history: list[dict]) -> TurnResult:
        """Interpret a fresh free-form turn twice and reconcile the readings.

        Slash commands and confirmation replies are protocol and never reach here.
        """
        context = snapshot.context
        named = selectors_in_text(text)
        paused_scopes = {scope for task in context.paused
                         for scope in (task.task_id, task.task_id.split("-", 1)[0])}
        # A message naming work outside the paused tasks is not a reply to them.
        about_paused = bool(context.paused) and not any(
            selector not in paused_scopes for selector in named)
        if (about_paused and mode_hint == "review" and named
                and not explicit_retry_request(text, history)):
            about_paused = False
        if not context.paused and explicit_retry_request(text, history):
            return await self._finish_turn(conversation_id, text, TurnResult("reply", (
                "I understood that you want to retry a review, but no paused review was "
                "found in this conversation. Nothing was executed.")),
                Reconciliation(Route.UNCLEAR, "no paused review"))

        contextual = await self.interpret(text, context) if about_paused else None
        free_value = (await self.classify_fresh(text, mode_hint)
                      if needs_free_classification(contextual) else None)
        decision = reconcile(
            Classification.from_fresh(free_value) if free_value else None,
            Classification.from_intent(contextual) if contextual else None)

        if decision.route == Route.CONTINUATION:
            if mode_hint == "review" and contextual.action != A.RETRY_REVIEW:
                result = TurnResult("reply", (
                    "Review mode is read-only. Switch modes to propose this workflow action. "
                    "Nothing was executed."))
            else:
                result = TurnResult("proposal", proposal=await self.propose(contextual))
        elif decision.route == Route.NEW_FLOW:
            outcome = await self.propose_fresh(conversation_id, text, mode_hint,
                                               free_value, artifacts)
            result = (TurnResult("answer") if outcome is None else
                      TurnResult("reply", outcome) if isinstance(outcome, str) else
                      TurnResult("plan_start", outcome.instruction, wp_id=outcome.wp_ids[0],
                                 queued_wp_ids=outcome.wp_ids[1:])
                      if isinstance(outcome, PlanStart) else
                      TurnResult("proposal", proposal=outcome))
        elif decision.route == Route.QUESTION:
            result = TurnResult("answer")
        elif decision.alternatives:
            task = context.task(contextual.target_job, contextual.target_position)
            result = TurnResult("reply", (
                "Nothing was executed. I am not sure whether you want to start new work "
                f"or reply to the paused task `{task.task_id if task else contextual.target_task}`.\n\n"
                f"- To start new work, name it explicitly, for example `/implement <selector>`.\n"
                f"- To continue `{contextual.target_task}`, start your reply with "
                f"`{contextual.target_task}:`."))
        elif contextual is not None and contextual.clarification:
            result = TurnResult("reply", contextual.clarification)
        elif free_value is not None and free_value.get("action") == "ASK_CLARIFICATION":
            result = TurnResult("reply", f"Nothing was executed. {free_value['reason']}")
        else:
            result = TurnResult("reply", "Nothing was executed. I could not identify "
                                         "an available action.")
        return await self._finish_turn(conversation_id, text, result, decision)

    async def _finish_turn(self, conversation_id: str, text: str, result: TurnResult,
                           decision: Reconciliation) -> TurnResult:
        if result.kind != "answer":
            notices = await self.persistence.take_switch_notices(conversation_id)
            if notices:
                notice = switching.interruption_notice(notices)
                result = TurnResult("reply", f"{notice}\n\n{result.render()}")
        self.record_turn(conversation_id, "user", decision.route.value, text,
                         free=decision.free, contextual=decision.contextual,
                         proposal_id=result.proposal.proposal_id if result.proposal else None,
                         outcome=result.kind)
        return result

    def planning_route(self, text: str, snapshot: ConversationSnapshot) -> str | None:
        """While a WP is being planned: "approve", "discuss", "abandon", or None.

        None means the turn is not about the current plan (it names another WP),
        so ordinary routing applies and may start planning that WP instead.
        """
        context = snapshot.context
        if context.planning_session_id is None:
            return None
        if PLAN_APPROVAL.fullmatch(text) or text.strip().lower() in BARE_CONFIRM:
            return "approve"
        if text.strip().lower() in BARE_REJECT:
            return None  # an ordinary rejection of the pending approval closes planning
        if PLAN_ABANDON.fullmatch(text):
            return "abandon"
        named = selectors_in_text(text)
        if any(selector.split("-", 1)[0] != context.planning_wp_id for selector in named):
            return None
        return "discuss"

    async def propose_plan_approval(self, conversation_id: str, turn, artifacts
                                    ) -> WorkflowActionProposal:
        """Bind "Approve and implement" to exactly this plan revision."""
        targets = await artifacts.tasks_for_selectors([turn.wp_id])
        task_ids = [task.id for _, task in targets]
        return await self.propose(WorkflowIntent(
            conversation_id=conversation_id,
            message_fingerprint=fingerprint(f"{turn.session_id}:{turn.revision}"),
            action=A.APPROVE_WP_PLAN, selectors=(turn.wp_id,), target_task=turn.wp_id,
            expected_effects=(
                f"Freeze revision {turn.revision} of the {turn.wp_id} plan as execution context",
                *(f"Task {task.id}: {task.title}" for _, task in targets),
                "Create one implementation job and run the tasks in order",
            ),
            parameters={"task_ids": task_ids, "planning_session_id": turn.session_id,
                        "planning_revision": turn.revision}))

    def resolve_bare_reply(self, text: str, snapshot: ConversationSnapshot
                           ) -> tuple[dict | None, str | None]:
        """Bind a bare yes/no to exactly one waiting item, or explain why not.

        Returns (proposal, None) to confirm or reject that proposal, (None, None) when
        the reply answers the single Reviewer question, or (None, message) otherwise.
        """
        pending = snapshot.pending
        questions = [task for task in snapshot.context.paused
                     if task.state == S.HUMAN_CONFIRMATION.value]
        if len(pending) == 1 and not questions:
            return pending[0], None
        if not pending and len(questions) == 1:
            return None, None
        if not pending and not questions:
            return None, ("Nothing in this conversation is waiting for a yes or no. "
                          "Nothing was executed.")
        lines = ["Nothing was executed. More than one thing is waiting for an answer:"]
        lines.extend(f"- Proposal `{row['confirmation_token']}`: {row['action']}"
                     + (f" for {row['target_task_id']}" if row.get("target_task_id") else "")
                     + f" (reply `confirm {row['confirmation_token']}` or "
                       f"`reject {row['confirmation_token']}`)" for row in pending)
        lines.extend(f"- The Reviewer's question for `{task.task_id}` (reply "
                     f"`{task.task_id}: yes` or `{task.task_id}: no: <feedback>`)"
                     for task in questions)
        return None, "\n".join(lines)

    def record_turn(self, conversation_id: str | None, turn_type: str, route: str,
                    text: str = "", free: Classification | None = None,
                    contextual: Classification | None = None,
                    proposal_id: str | None = None, outcome: str | None = None) -> None:
        """Log how a turn was routed: identities and summaries, never the prompt."""
        self.telemetry.turn_event(TurnEvent(
            conversation_id=conversation_id, project_id=self.project_id,
            turn_type=turn_type, route=route,
            free_classification=free.summary() if free else None,
            contextual_classification=contextual.summary() if contextual else None,
            proposal_id=proposal_id, outcome=outcome,
            message_fingerprint=fingerprint(text) if text else None))

    # --- context -----------------------------------------------------------

    async def context(self, conversation_id: str | None,
                      referenced_job_ids: tuple[str, ...] = ()) -> WorkflowContext:
        rows = await self.persistence.paused_tasks(conversation_id, referenced_job_ids)
        paused = []
        for row in rows:
            contract = await self.contracts.latest(row["job_id"], row["position"])
            escalation = await self.contracts.latest_escalation(row["job_id"], row["position"])
            cause = row["error"] or ""
            if row["status"] == S.CONTRACT_RESOLUTION.value and escalation:
                cause = escalation.get("reason") or cause
            if row["latest_verdict"] and row["status"] != S.CONTRACT_RESOLUTION.value:
                cause = "; ".join(filter(None, (
                    f"Reviewer verdict: {row['latest_verdict']}", cause)))
            paused.append(PausedTask(
                job_id=row["job_id"], position=row["position"], task_id=row["task_id"],
                state=row["status"], cause=cause, instruction=row["instruction"] or "",
                contract_id=contract.contract_id if contract else None,
                contract_revision=contract.revision if contract else None,
                files_allowed=contract.primary_files if contract else (),
                files_requested=requested_paths(cause),
                review_completed=row["latest_verdict"] in {"done", "needs_work", "ask_user"},
                review_question=row["review_question"],
            ))
        refinement = await self.persistence.active_refinement(conversation_id)
        planning = await self.persistence.active_planning(conversation_id)
        return WorkflowContext(conversation_id, tuple(paused),
                               refinement["session_id"] if refinement else None,
                               planning_session_id=planning["session_id"] if planning else None,
                               planning_wp_id=planning["wp_id"] if planning else None)

    async def snapshot(self, conversation_id: str | None) -> ConversationSnapshot:
        await switching.recover_switches(self.persistence)
        context = await self.context(conversation_id)
        pending = tuple(await self.pending(conversation_id))
        if pending:
            latest = pending[0]
            context = replace(context, pending_action=(
                f"{latest['action']} for {latest.get('target_task_id') or 'the conversation'}; "
                f"effects: {', '.join(latest['payload'].get('expected_effects', ()))[:200]}"))
        running_jobs = await self.persistence.running_jobs(conversation_id)
        phase = (ConversationPhase.PLANNING if context.planning_session_id else
                 ConversationPhase.PROPOSED if pending else
                 ConversationPhase.PAUSED if context.paused else
                 ConversationPhase.RUNNING if running_jobs else ConversationPhase.READY)
        return ConversationSnapshot(phase, context, pending, running_jobs)

    # --- interpretation and proposals ----------------------------------------

    async def interpret(self, message: str, context: WorkflowContext) -> WorkflowIntent:
        intent = await self.interpreter.interpret(message, context, fingerprint(message))
        self._event("interpreted", intent)
        return intent

    async def propose(self, intent: WorkflowIntent,
                      corrected_from: dict | None = None,
                      refreshed: bool = False) -> WorkflowActionProposal | None:
        """Persist a pending proposal. Clarifications are never executable."""
        if intent.action in {A.ASK_CLARIFICATION, A.ANSWER}:
            self._event("clarification", intent)
            return None
        intent = await self._with_conflicts(intent)
        for _ in range(5):
            proposal = WorkflowActionProposal(
                str(uuid4()),
                "".join(secrets.choice(_TOKEN_ALPHABET) for _ in range(_TOKEN_LENGTH)),
                intent, refreshed)
            try:
                superseded = await self.persistence.create_workflow_proposal({
                    "proposal_id": proposal.proposal_id,
                    "confirmation_token": proposal.confirmation_token,
                    "conversation_id": intent.conversation_id,
                    "action": intent.action.value,
                    "target_job_id": intent.target_job,
                    "target_position": intent.target_position,
                    "target_task_id": intent.target_task,
                    "expected_status": intent.source_state,
                    "expected_revision": intent.expected_revision,
                    "payload": intent.payload(),
                    "request_fingerprint": intent.message_fingerprint,
                    "confidence": intent.confidence,
                    "ambiguities": list(intent.ambiguities),
                    "interpreter_model": intent.model,
                    "interpretation_source": intent.interpretation_source,
                    "corrected_from": corrected_from["proposal_id"] if corrected_from else None,
                    "expires_at": time.time() + PROPOSAL_TTL_SECONDS,
                })
                break
            except sqlite3.IntegrityError as exc:
                # Only a token collision is worth retrying; any other constraint
                # failure (for example a schema mismatch) must surface as itself.
                if "confirmation_token" not in str(exc):
                    raise
                continue
        else:
            raise RuntimeError("could not allocate a unique confirmation token")
        for row in superseded:
            stage = ("corrected" if corrected_from
                     and row["proposal_id"] == corrected_from["proposal_id"] else "superseded")
            self._event(stage, WorkflowIntent.from_proposal(row), row["proposal_id"])
        self._event("proposed", intent, proposal.proposal_id)
        return proposal

    async def _with_conflicts(self, intent: WorkflowIntent) -> WorkflowIntent:
        """Attach the project's other active jobs (to stop) and paused jobs (to keep).

        Read-only actions never displace anything. The claim re-checks the same
        set atomically, so a job that finished or started meanwhile makes the
        proposal stale instead of silently stopping or overlapping work.
        """
        if (intent.action not in MUTATING_ACTIONS
                or intent.parameters.get("recovery_mode") == "inspect"):
            return intent
        activity = await switching.project_activity(
            self.persistence, self.project_id, intent.conversation_id)
        others = [item for item in activity if item.job_id != intent.target_job]
        return replace(intent, parameters={
            **intent.parameters,
            "conflict_checked": True,
            "project_id": self.project_id,
            "displace": [item.payload() for item in others if item.active],
            "paused_jobs": [item.payload() for item in others if not item.active],
            "start_phrase": _start_phrase(intent),
        })

    def command_intent(self, conversation_id: str, command: str, action: A = A.EXECUTE_COMMAND,
                       refinement_session_id: str | None = None) -> WorkflowIntent:
        """A deterministic proposal for an explicit slash command."""
        label = command.strip().split(maxsplit=1)[0] if command.strip() else command
        effects = {
            A.APPLY_SPEC: ("Write the active draft into OpenSpec",),
            A.DISCARD_SPEC: ("Discard the active draft; OpenSpec is unchanged",),
        }.get(action, (f"Run `{label}`", f"Command: `{command.strip()}`"))
        spec = _SPEC_MARKERS.get(label)
        if spec is not None:
            request = command.strip()[len(label):].strip()
            effects = (f"{spec} from your request ({len(request):,} characters)",
                       f"Starts: \"{_preview(request)}\"")
        elif label.lower() == "/reset":
            effects = (
                f"Reset the selected OpenSpec Work Package and active tasks to pending: `{command.strip()}`",
                "Clear their OpenSpec completion and review records; keep project files",
            )
        parameters = {"command": command}
        if refinement_session_id:
            parameters["refinement_session_id"] = refinement_session_id
        return WorkflowIntent(
            conversation_id=conversation_id, message_fingerprint=fingerprint(command),
            action=action, expected_effects=effects, parameters=parameters)

    async def propose_start(self, conversation_id: str, message_fingerprint: str,
                            intent: ImplementationIntent, artifacts,
                            corrected_from: dict | None = None,
                            refreshed: bool = False) -> WorkflowActionProposal:
        """Resolve selectors to tasks and preview them; nothing is created yet."""
        targets = await artifacts.tasks_for_selectors(list(intent.selectors))
        task_ids = [task.id for _, task in targets]
        effects = (
            f"Selectors: {', '.join(intent.selectors)}",
            *(f"Task {task.id}: {task.title}" for _, task in targets),
            f"Execution instruction: {intent.instruction or '(none)'}",
            "Create one implementation job and run the tasks in order",
        )
        workflow_intent = WorkflowIntent(
            conversation_id=conversation_id, message_fingerprint=message_fingerprint,
            action=A.START_IMPLEMENTATION, selectors=intent.selectors,
            guidance=intent.instruction, confidence=intent.confidence,
            ambiguities=intent.ambiguities, expected_effects=effects,
            parameters={"task_ids": task_ids},
            interpretation_source="deterministic" if intent.fallback_used else "model",
            model=None if intent.model == COMMAND_SOURCE else intent.model,
            latency_ms=intent.latency_ms, usage=intent.usage)
        if intent.model != COMMAND_SOURCE:
            self._event("interpreted", workflow_intent)
        return await self.propose(workflow_intent, corrected_from, refreshed)

    # --- confirmation boundary ---------------------------------------------

    async def find(self, conversation_id: str | None, token: str) -> dict | None:
        await self._expire(conversation_id)
        return await self.persistence.workflow_proposal_by_token(conversation_id, token)

    async def pending(self, conversation_id: str | None) -> list[dict]:
        await self._expire(conversation_id)
        return await self.persistence.pending_workflow_proposals(conversation_id)

    async def reject(self, proposal: dict) -> str:
        token = proposal["confirmation_token"]
        if proposal["status"] == "pending" and await self.persistence.resolve_workflow_proposal(
                proposal["proposal_id"], "rejected"):
            self._event("rejected", WorkflowIntent.from_proposal(proposal),
                        proposal["proposal_id"], proposal=proposal)
            session_id = (proposal["payload"].get("parameters") or {}).get("planning_session_id")
            if proposal["action"] == A.APPROVE_WP_PLAN.value and session_id:
                await self.persistence.finish_planning(session_id, "abandoned")
                return ("Planning closed without implementing anything. "
                        "Ask again whenever you want to plan this work package.")
            return f"Proposal {token} rejected. Nothing was executed."
        return f"Proposal {token} is {proposal['status']}; nothing was executed."

    async def change(self, proposal: dict, correction: str,
                     artifacts=None) -> WorkflowActionProposal | str:
        """Amend a pending proposal's guidance and replace it with a new proposal."""
        if proposal["status"] != "pending":
            return (f"Proposal {proposal['confirmation_token']} is {proposal['status']} "
                    "and cannot be changed. Send your request again.")
        intent = WorkflowIntent.from_proposal(proposal)
        guidance = "\n".join(filter(None, (intent.guidance, correction.strip())))
        if intent.action == A.START_IMPLEMENTATION:
            if artifacts is None:
                return "Changing an implementation proposal requires the IDE MCP session."
            return await self.propose_start(
                intent.conversation_id, fingerprint(correction), ImplementationIntent(
                    intent.selectors, guidance, intent.confidence, intent.ambiguities,
                    COMMAND_SOURCE), artifacts, corrected_from=proposal)
        if intent.action in PASSTHROUGH_ACTIONS or intent.action == A.STOP_JOB:
            return ("This proposal has no guidance to change. Reject it and send a "
                    "new request instead.")
        context = await self.context(intent.conversation_id, (intent.target_job,))
        task = context.task(intent.target_job, intent.target_position)
        if task is None or task.state != intent.source_state:
            return "The workflow changed after this proposal was created. Send a new request."
        amended = replace(intent, guidance=guidance,
                          message_fingerprint=fingerprint(correction))
        amended = amended.with_effects(expected_effects(amended, task))
        return await self.propose(amended, corrected_from=proposal)

    async def claim(self, proposal: dict, artifacts=None) -> ClaimResult:
        """Atomically claim and revalidate a proposal before any execution."""
        token = proposal["confirmation_token"]
        if proposal["status"] == "confirmed":
            return ClaimResult("confirmed", proposal, (
                f"Proposal {token} was already confirmed. Nothing new was executed."))
        if proposal["status"] in {"rejected", "superseded", "corrected"}:
            return ClaimResult(proposal["status"], proposal, (
                f"Proposal {token} is {proposal['status']}; nothing was executed."))
        intent = WorkflowIntent.from_proposal(proposal)
        if intent.action == A.START_IMPLEMENTATION:
            if proposal["status"] == "pending":
                await self.persistence.resolve_workflow_proposal(
                    proposal["proposal_id"], "stale")
            return ClaimResult("stale", proposal, (
                "New implementation requires an approved Work Package plan. "
                "Start with its Work Package selector; nothing was executed."))
        job = tasks = None
        if (intent.action in {A.START_IMPLEMENTATION, A.APPROVE_WP_PLAN}
                and proposal["status"] == "pending"):
            if artifacts is None:
                return ClaimResult("unavailable", proposal,
                                   "Confirming an implementation requires the IDE MCP session.")
            targets = await artifacts.tasks_for_selectors(list(intent.selectors))
            current_ids = [task.id for _, task in targets]
            if current_ids != intent.parameters.get("task_ids"):
                await self.persistence.resolve_workflow_proposal(proposal["proposal_id"], "stale")
                return await self._refresh("stale", proposal, intent, artifacts)
            plan_id = (plan_id_for(intent.parameters["planning_session_id"],
                                   intent.parameters["planning_revision"])
                       if intent.action == A.APPROVE_WP_PLAN else None)
            job = ImplementationJob.create(list(intent.selectors), current_ids, intent.guidance,
                                           plan_id=plan_id)
            tasks = [ImplementationJobTask(job.job_id, position, change.id, task.id)
                     for position, (change, task) in enumerate(targets)]
        status = (proposal["status"] if proposal["status"] == "expired"
                  else await self.persistence.claim_workflow_proposal(
                      proposal["proposal_id"], job, tasks))
        if status == "claimed":
            self._event("confirmed", intent, proposal["proposal_id"], proposal=proposal)
            return ClaimResult("claimed", {**proposal, "status": "confirmed"}, job=job)
        if status in {"expired", "stale"}:
            return await self._refresh(status, proposal, intent, artifacts)
        if status in {"confirmed", "rejected", "superseded"}:
            # Another request resolved it between lookup and claim.
            return await self.claim({**proposal, "status": status}, artifacts)
        return ClaimResult(status, proposal, (
            f"Proposal {token} could not be confirmed ({status}); nothing was executed."))

    async def _refresh(self, status: str, proposal: dict, intent: WorkflowIntent,
                       artifacts) -> ClaimResult:
        """Execute nothing; offer a refreshed proposal when the action is still valid."""
        self._event(status, intent, proposal["proposal_id"], proposal=proposal)
        reason = ("This proposal expired." if status == "expired"
                  else "The workflow changed after this proposal was created.")
        message = f"Nothing was executed. {reason}"
        if intent.action == A.APPROVE_WP_PLAN:
            return ClaimResult(status, proposal, (
                f"{message} That approval referred to an earlier revision of the plan, or "
                "planning has ended. Approve the latest plan revision instead."))
        if intent.action == A.START_IMPLEMENTATION:
            if artifacts is None:
                return ClaimResult(status, proposal, message)
            refreshed = await self.propose_start(
                intent.conversation_id, intent.message_fingerprint,
                ImplementationIntent(intent.selectors, intent.guidance, intent.confidence,
                                     intent.ambiguities, COMMAND_SOURCE),
                artifacts, refreshed=True)
            return ClaimResult(status, proposal, message, refreshed=refreshed)
        if intent.action in PASSTHROUGH_ACTIONS:
            context = await self.context(intent.conversation_id)
            session = intent.parameters.get("refinement_session_id")
            if session and context.refinement_session_id != session:
                return ClaimResult(status, proposal,
                                   f"{message} The draft it referred to is no longer active.")
            refreshed = await self.propose(replace(intent, parameters={
                **intent.parameters,
                "refinement_session_id": context.refinement_session_id if session else None,
            }), refreshed=True)
            return ClaimResult(status, proposal, message, refreshed=refreshed)
        context = await self.context(intent.conversation_id, (intent.target_job,))
        task = context.task(intent.target_job, intent.target_position)
        mode = intent.parameters.get("recovery_mode", "")
        if task is None or not action_allowed(task.state, intent.action, mode):
            return ClaimResult(status, proposal, clarification_text(
                context, f"{reason} The proposed action no longer applies.", context.paused))
        rebound = replace(intent, source_state=task.state,
                          expected_revision=task.contract_revision)
        refreshed = await self.propose(
            rebound.with_effects(expected_effects(rebound, task)), refreshed=True)
        return ClaimResult(status, proposal, message, refreshed=refreshed)

    # --- execution ---------------------------------------------------------

    async def execute(self, proposal: dict, service, messages: list[dict], specs: list,
                      handlers: dict, job: ImplementationJob | None = None
                      ) -> AsyncIterator[Event]:
        """Run a claimed proposal. Callers must have received `ClaimResult.claimed`."""
        if proposal.get("status") != "confirmed":
            raise RuntimeError("only a claimed workflow proposal may execute")
        intent = WorkflowIntent.from_proposal(proposal)
        switched = await self.run_switch(proposal)
        if switched is not None:
            if not switched.ok:
                await self.transition_finished(proposal, "switch: " + switched.message)
                yield TextEvent(switched.message)
                return
            yield TextEvent(switched.message)
        job_id, position = intent.target_job, intent.target_position
        instruction = await self._job_instruction(job_id)
        action, guidance = intent.action, intent.guidance
        if action == A.START_IMPLEMENTATION:
            events = service.start(job, messages, specs, handlers)
        elif action == A.APPROVE_WP_PLAN:
            events = self._approve_and_start(intent, service, job, messages, specs, handlers)
        elif action in {A.RETRY_CONTRACT, A.EXPAND_CONTRACT_SCOPE}:
            events = service.retry_contract(
                job_id, position, messages, specs, instruction, handlers, guidance,
                tuple(intent.parameters.get("scope_paths", ())))
        elif action == A.RETRY_CODER_AFTER_REVIEW:
            events = service.retry_coder(
                job_id, position, messages, specs, instruction, handlers, guidance)
        elif action == A.ACCEPT_REVIEW:
            events = service.accept_anyway(
                job_id, position, messages, specs, instruction, handlers, guidance)
        elif action == A.RETRY_REVIEW:
            events = service.retry_review(
                job_id, position, messages, specs, instruction, handlers)
        elif action == A.ANSWER_REVIEWER_QUESTION:
            events = service.confirm(
                job_id, position, bool(intent.parameters.get("approved")), guidance,
                messages, specs, handlers, instruction)
        elif action == A.RESOLVE_CODER_RECOVERY:
            mode = intent.parameters.get("recovery_mode")
            if mode == "review_handoff":
                events = service.review_blocked_handoff(
                    job_id, position, messages, specs, instruction, handlers, guidance)
            else:
                events = service.retry_coder(
                    job_id, position, messages, specs, instruction, handlers, guidance)
        elif action == A.STOP_JOB:
            events = service.stop_job(job_id, position, intent.source_state)
        else:
            raise RuntimeError(f"{action} is executed by replaying its command")
        if switched is not None:
            await switching.mark_switch_started(
                self.persistence, "switch_" + proposal["proposal_id"])
        failure = None
        try:
            async for event in events:
                if isinstance(event, ErrorEvent) and failure is None:
                    failure = event.message
                if isinstance(event, TextEvent):
                    choices = menu_choices(event.text)
                    if choices:
                        await self.persistence.record_offered_choices(
                            intent.conversation_id, choices)
                    # Streamed messages are concatenated as-is; never let two run together.
                    event = TextEvent(ensure_newline(event.text))
                yield event
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await self.transition_finished(proposal, failure)
        if failure is None:
            async for event in self._plan_next(intent.conversation_id,
                                               job.job_id if job else job_id,
                                               service, specs, handlers):
                yield event

    async def _approve_and_start(self, intent: WorkflowIntent, service, job: ImplementationJob,
                                 messages: list[dict], specs: list, handlers: dict
                                 ) -> AsyncIterator[Event]:
        """Freeze the approved plan (no model call), then run the existing job pipeline."""
        from agent.planning.service import PlanningError, WPPlannerService
        planner = WPPlannerService(service.artifacts, self.persistence,
                                   config=getattr(service, "config", None))
        try:
            plan = await planner.approve(intent.parameters["planning_session_id"],
                                         intent.parameters["planning_revision"])
        except PlanningError as exc:
            await self.persistence.finish_implementation_job(job.job_id, "failed", str(exc))
            yield ErrorEvent(f"Nothing was implemented: {exc}")
            return
        yield TextEvent(f"Plan {plan.plan_id} approved and frozen for {plan.wp_id}. "
                        "Starting implementation with the planned context.\n\n")
        async for event in service.start(job, messages, specs, handlers):
            yield event

    async def _plan_next(self, conversation_id: str, job_id: str | None, service,
                         specs: list, handlers: dict) -> AsyncIterator[Event]:
        """When a planned WP's job completes, plan the next WP from the same request."""
        if not job_id or await self.persistence.implementation_job_status(job_id) != "completed":
            return
        plan = await self.persistence.approved_plan(await self.persistence.job_plan_id(job_id))
        session = (await self.persistence.planning_session(plan["session_id"])
                   if plan else None)
        queued = list(session.get("queued_wp_ids") or ()) if session else []
        if not queued:
            return
        from agent.planning.render import render_plan
        from agent.planning.service import PlanningError, WPPlannerService, with_progress
        from conversation.presentation import progress_line
        planner = WPPlannerService(service.artifacts, self.persistence,
                                   config=getattr(service, "config", None))

        def call(progress):
            return planner.start(
                conversation_id, queued[0], session.get("request") or f"Implement {queued[0]}",
                specs, handlers, project_id=self.project_id, queued_wp_ids=tuple(queued[1:]),
                progress=progress,
                heading=f"{plan['wp_id']} is complete — investigating {queued[0]}")

        turn = None
        try:
            async for kind, value in with_progress(call):
                if kind == "progress":
                    yield TextEvent(value)
                else:
                    turn = value
            approval = (None if turn.gaps else
                        await self.propose_plan_approval(conversation_id, turn,
                                                         service.artifacts))
        except PlanningError as exc:
            yield ErrorEvent(f"Planning {queued[0]} failed: {exc}. Ask again to plan it.")
            return
        text = render_plan(turn.draft, wp_complexity=turn.wp_complexity,
                           planner_label=turn.planner_label, gaps=turn.gaps,
                           token=approval.confirmation_token if approval else None,
                           queued_wp_ids=turn.queued_wp_ids)
        await self.persistence.record_offered_choices(conversation_id, menu_choices(text))
        yield TextEvent(progress_line("separator"))
        yield TextEvent(text + "\n")

    async def run_switch(self, proposal: dict) -> switching.SwitchOutcome | None:
        """Stop the jobs a confirmed proposal displaces. None when nothing is displaced."""
        switch = await self.persistence.switch_for_proposal(proposal["proposal_id"])
        if switch is None:
            return None
        parameters = proposal["payload"].get("parameters") or {}
        outcome = await switching.run_switch(
            self.persistence, switch, parameters.get("start_phrase") or "start the new work")
        self._event("switched" if outcome.ok else "switch_failed",
                    WorkflowIntent.from_proposal(proposal), proposal["proposal_id"],
                    transition_result=outcome.message[:200])
        return outcome

    async def transition_finished(self, proposal: dict, failure: str | None) -> None:
        result = f"failed: {failure}" if failure else "started"
        await self.persistence.record_workflow_transition(proposal["proposal_id"], result)
        self._event("executed", WorkflowIntent.from_proposal(proposal),
                    proposal["proposal_id"], transition_result=result[:200])

    async def _job_instruction(self, job_id: str | None) -> str:
        return await self.persistence.implementation_job_instruction(job_id) if job_id else ""

    async def _expire(self, conversation_id: str | None) -> None:
        for row in await self.persistence.expire_workflow_proposals(conversation_id):
            self._event("expired", WorkflowIntent.from_proposal(row), row["proposal_id"],
                        proposal=row)

    # --- telemetry ---------------------------------------------------------

    def _event(self, stage: str, intent: WorkflowIntent, proposal_id: str | None = None,
               proposal: dict | None = None, transition_result: str | None = None) -> None:
        latency = None
        if proposal is not None and stage in {"confirmed", "rejected"}:
            latency = int((time.time() - proposal["created_at"]) * 1000)
        self.telemetry.workflow_event(WorkflowEvent(
            stage=stage, proposal_id=proposal_id, action=intent.action.value,
            source_state=intent.source_state,
            interpretation_source=intent.interpretation_source, model=intent.model,
            confidence=intent.confidence, ambiguity_count=len(intent.ambiguities),
            selector_count=len(intent.selectors), guidance_present=bool(intent.guidance),
            latency_ms=intent.latency_ms if stage == "interpreted" else None,
            confirmation_latency_ms=latency,
            input_tokens=intent.usage.get("inputTokens") if stage == "interpreted" else None,
            output_tokens=intent.usage.get("outputTokens") if stage == "interpreted" else None,
            transition_result=transition_result,
            message_fingerprint=intent.message_fingerprint or None,
            detail=intent.interpretation_error or None,
        ))


# Public compatibility name used by existing integrations and test fixtures.
WorkflowController = ConversationWorkflow

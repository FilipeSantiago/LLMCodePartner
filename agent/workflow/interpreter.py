"""Interpret one user message relative to the paused workflow state.

The interpreter only classifies the requested workflow action. It never
replans: Tasker revises contracts, the Coder executes them, and the workflow
controller enforces confirmation and valid transitions.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from agent.spec.implementation_intent import (
    canonical_selector, emit_json_with_usage, intent_model, selectors_in_text,
)
from agent.task_contracts.contractor import excluded_paths
from agent.workflow.intent import PausedTask, WorkflowContext, WorkflowIntent
from agent.workflow.states import (
    GUIDANCE_REQUIRED, RECOVERY_MODES, WorkflowAction as A, WorkflowState as S,
    action_allowed,
)

_MAX_MESSAGE_CHARS = 4000
_MAX_GUIDANCE_CHARS = 2000


@dataclass(frozen=True)
class _Phrase:
    kind: str
    guidance: str = ""
    paths: tuple[str, ...] = ()


def _group(match: re.Match, name: str) -> str:
    return (match.groupdict().get(name) or "").strip()


_PHRASES: tuple[tuple[re.Pattern, Callable[[re.Match], _Phrase]], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), build) for pattern, build in (
        (r"retry (?:the )?contract(?:\s*:\s*(?P<g>.+))?",
         lambda m: _Phrase("retry_contract", _group(m, "g"))),
        (r"expand (?:the )?(?:contract )?scope(?:\s+to)?\s*:?\s*(?P<p>[^:]+?)"
         r"(?:\s*:\s*(?P<g>.+))?",
         lambda m: _Phrase("expand", _group(m, "g"), tuple(
             item for item in re.split(r"[,\s]+", _group(m, "p"))
             if item and item.lower() != "and"))),
        (r"(?:submit to coder again|try coder again|try again|retry)(?:\s*:\s*(?P<g>.+))?",
         lambda m: _Phrase("retry", _group(m, "g"))),
        (r"accept(?: anyway)?(?:\s*:\s*(?P<g>.+))?",
         lambda m: _Phrase("accept", _group(m, "g"))),
        (r"undo(?: code)?", lambda m: _Phrase("undo")),
        (r"/?retry[- ]review", lambda m: _Phrase("retry_review")),
        (r"inspect recovery", lambda m: _Phrase("inspect")),
        (r"establish new baseline\s*:\s*(?P<g>.+)",
         lambda m: _Phrase("baseline", _group(m, "g"))),
        (r"stop(?: the)?(?: job)?|stop and repair manually|cancel (?:the )?job",
         lambda m: _Phrase("stop")),
        (r"yes|y|approve", lambda m: _Phrase("yes")),
        (r"(?:no|n)(?:\s*:\s*(?P<g>.+))?", lambda m: _Phrase("no", _group(m, "g"))),
    ))
_TASK_PREFIX = re.compile(r"\s*((?:WP|US)\d+-T\d+)\s*[:,]?\s+(.+)", re.IGNORECASE | re.DOTALL)
_NATURAL_REVIEW_RETRY = re.compile(
    r"\b(?:retry|rerun|re-run|run again)\b.{0,100}\breview\b|"
    r"\breview\b.{0,60}\b(?:again|retry)\b", re.IGNORECASE | re.DOTALL)
_NEGATED_RETRY = re.compile(r"\b(?:do not|don't|dont|never|no)\s+(?:want to\s+)?"
                            r"(?:retry|rerun|re-run|run again)\b", re.IGNORECASE)
_JOB_REFERENCE = re.compile(
    r"\bjob\s+([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\b", re.IGNORECASE)


def _phrase_action(phrase: _Phrase, state: str) -> tuple[A, dict] | None:
    """Map a recognized phrase to the action it means in one workflow state."""
    kind = phrase.kind
    if kind == "retry":
        return {S.CONTRACT_RESOLUTION.value: (A.RETRY_CONTRACT, {}),
                S.REVIEW_RESOLUTION.value: (A.RETRY_CODER_AFTER_REVIEW, {}),
                S.CODER_RECOVERY.value: (A.RESOLVE_CODER_RECOVERY,
                                         {"recovery_mode": "retry_coder"}),
                }.get(state)
    simple = {
        "retry_contract": (A.RETRY_CONTRACT, {}),
        "expand": (A.EXPAND_CONTRACT_SCOPE, {"scope_paths": list(phrase.paths)}),
        "accept": (A.ACCEPT_REVIEW, {}),
        "undo": (A.UNDO_IMPLEMENTATION, {}),
        "retry_review": (A.RETRY_REVIEW, {}),
        "inspect": (A.RESOLVE_CODER_RECOVERY, {"recovery_mode": "inspect"}),
        "baseline": (A.RESOLVE_CODER_RECOVERY, {"recovery_mode": "establish_baseline"}),
        "stop": (A.STOP_JOB, {}),
        "yes": (A.ANSWER_REVIEWER_QUESTION, {"approved": True}),
        "no": (A.ANSWER_REVIEWER_QUESTION, {"approved": False}),
    }
    action, parameters = simple[kind]
    if not action_allowed(state, action, parameters.get("recovery_mode", "")):
        return None
    return action, parameters


_HINTS = {
    A.RETRY_CONTRACT: "`retry contract: <guidance>`",
    A.EXPAND_CONTRACT_SCOPE: "`expand scope: <paths>`",
    A.RETRY_CODER_AFTER_REVIEW: "`submit to coder again: <guidance>`",
    A.ACCEPT_REVIEW: "`accept` or `accept anyway: <reason>`",
    A.UNDO_IMPLEMENTATION: "`undo`",
    A.RETRY_REVIEW: "`retry review`",
    A.ANSWER_REVIEWER_QUESTION: "`yes` or `no: <feedback>`",
    A.STOP_JOB: "`stop job`",
}


def _hint(action: A, state: str) -> str:
    if action == A.RESOLVE_CODER_RECOVERY:
        return {S.CODER_RECOVERY.value: "`try coder again[: <guidance>]`",
                S.JOURNAL_RECOVERY.value:
                    "`inspect recovery` or `establish new baseline: <guidance>`"}.get(state, "")
    return _HINTS.get(action, "")


def clarification_text(context: WorkflowContext, reason: str,
                       tasks: tuple[PausedTask, ...] | None = None) -> str:
    tasks = context.paused if tasks is None else tasks
    lines = [f"Nothing was executed. {reason}"]
    if tasks:
        lines.extend(("", f"I found {len(tasks)} paused task(s):"))
        for task in tasks:
            cause = (task.cause or task.state)[:160]
            lines.append(f"- `{task.task_id}` (job {task.job_id[:8]}…): {cause}")
            hints = [hint for action in task.actions if (hint := _hint(action, task.state))]
            lines.append(f"  You can reply: {', '.join(hints)}")
        if len(tasks) > 1:
            lines.extend(("", "Name the task your reply is about, for example "
                              f"`{tasks[0].task_id}: <your instruction>`."))
    return "\n".join(lines)


def expected_effects(intent: WorkflowIntent, task: PausedTask | None) -> tuple[str, ...]:
    """Describe deterministically what confirmation would do."""
    action = intent.action
    name = intent.target_task or "the task"
    resume = (f"Resume job `{task.job_id[:8]}…`",) if task else ()
    guidance = (f"Coder guidance: {intent.guidance}",) if intent.guidance else ()
    if action in {A.RETRY_CONTRACT, A.EXPAND_CONTRACT_SCOPE} and task:
        excluded = set(excluded_paths(intent.guidance))
        added = tuple(path for path in intent.parameters.get("scope_paths", ())
                      if path not in excluded)
        files = tuple(dict.fromkeys(
            (*(path for path in task.files_allowed if path not in excluded), *added)))
        outside = tuple(dict.fromkeys(
            path for path in (*excluded, *task.files_requested) if path not in files))
        revision = (task.contract_revision or 0) + 1
        return (
            f"Tasker creates TaskContract revision {revision} for {name}",
            *((f"Add to the contract: {', '.join(f'`{p}`' for p in added)}",) if added else ()),
            f"Files in the contract: {', '.join(f'`{p}`' for p in files) or 'none'}",
            *((f"Keep {', '.join(f'`{p}`' for p in outside)} outside the contract",)
              if outside else ()),
            *guidance, *resume,
            "No completed review is required before the Coder runs",
        )
    if action == A.RETRY_CODER_AFTER_REVIEW:
        return (f"Start a fresh Coder/Reviewer cycle for {name}", *guidance, *resume)
    if action == A.ACCEPT_REVIEW:
        override = task is not None and task.cause and "needs_work" in task.cause
        return (f"Mark {name} complete" + (" as a user override" if override else ""),
                *((f"Reason: {intent.guidance}",) if intent.guidance else ()),
                "Continue with the job's next task")
    if action == A.UNDO_IMPLEMENTATION:
        return (f"Restore the files {name} changed", "Stop the job")
    if action == A.RESOLVE_CODER_RECOVERY:
        mode = intent.parameters.get("recovery_mode")
        if mode == "inspect":
            return ("Show the mutation journal and integrity issues", "Nothing is changed")
        if mode == "establish_baseline":
            return ("Accept the current files as the new undo baseline",
                    "Prior uncertain mutations can no longer be undone automatically",
                    *guidance, *resume)
        return ("Continue on a different model, keeping files written so far",
                *guidance, *resume)
    if action == A.ANSWER_REVIEWER_QUESTION:
        if intent.parameters.get("approved"):
            return (f"Mark {name} complete", "Continue with the job's next task")
        return (f"Send {name} back to the Coder with your feedback", *guidance, *resume)
    if action == A.RETRY_REVIEW:
        return (f"Run the Reviewer again for {name}",
                "Inspect the mutation journal before other project evidence",
                "Do not discard a path claim solely because the user asked to ignore it",
                "No Coder run or file change is proposed")
    if action == A.STOP_JOB:
        return (f"Stop the job at {name}", "Nothing is undone; files stay for inspection")
    return ()


class WorkflowInterpreter:
    """Bounded, context-aware classification of one user message."""

    def __init__(self, provider=None, model: str | None = None):
        if provider is None:
            from providers.ollama import OllamaProvider
            provider = OllamaProvider()
        self.provider = provider
        self.model = model or intent_model()

    async def interpret(self, message: str, context: WorkflowContext,
                        fingerprint: str) -> WorkflowIntent:
        base = WorkflowIntent(
            conversation_id=context.conversation_id or "",
            message_fingerprint=fingerprint, action=A.ASK_CLARIFICATION,
            workflow_context=context.render())
        text = message.strip()
        if not text or len(text) > _MAX_MESSAGE_CHARS:
            return replace(base, clarification=clarification_text(
                context, "Replies must contain between 1 and 4000 characters."))
        if not context.paused:
            return replace(base, clarification=clarification_text(
                context, "No paused workflow is waiting for a decision."))

        prefixed = _TASK_PREFIX.fullmatch(text)
        body = prefixed.group(2).strip() if prefixed else text
        body = re.sub(r"^\s*(?:no,?\s+|actually,?\s+)", "", body,
                      count=1, flags=re.IGNORECASE)
        mentioned = {item.upper() for item in selectors_in_text(text)}
        named = tuple(task for task in context.paused if any(
            task.task_id.upper() == selector or
            task.task_id.upper().startswith(selector + "-")
            for selector in mentioned))
        job_reference = _JOB_REFERENCE.search(text)
        if job_reference:
            named = tuple(task for task in (named or context.paused)
                          if task.job_id.lower() == job_reference.group(1).lower())
        if mentioned and not named:
            return replace(base, clarification=clarification_text(
                context, "No paused task matches the task or work package you named."))
        if job_reference and not named:
            return replace(base, clarification=clarification_text(
                context, "No paused task matches the job you named."))

        if _NEGATED_RETRY.search(text):
            return replace(base, clarification=clarification_text(
                context, "I understood that you do not want to retry review."))
        if (text.endswith("?") and re.match(r"\s*(?:what|why|how|which|where|when|is|are|can)\b",
                                           text, re.IGNORECASE)):
            return replace(base, action=A.ANSWER, confidence="high",
                           interpretation_source="deterministic")

        phrase = next((build(match) for pattern, build in _PHRASES
                       if (match := pattern.fullmatch(body))), None)
        if phrase is None and _NATURAL_REVIEW_RETRY.search(body):
            phrase = _Phrase("retry_review")
        if phrase is not None:
            return self._deterministic(base, context, phrase, named)
        return await self._model(base, context, text, named)

    def _deterministic(self, base: WorkflowIntent, context: WorkflowContext,
                       phrase: _Phrase, named: tuple[PausedTask, ...]) -> WorkflowIntent:
        pool = named or context.paused
        candidates = [(task, mapped) for task in pool
                      if (mapped := _phrase_action(phrase, task.state)) is not None]
        if not candidates:
            return replace(base, clarification=clarification_text(
                context, "That reply is not a valid action for the paused task(s).", pool))
        if len(candidates) > 1:
            return replace(base, clarification=clarification_text(
                context, "More than one paused task could match that reply.",
                tuple(task for task, _ in candidates)))
        task, (action, parameters) = candidates[0]
        return self._finish(replace(
            base, action=action, guidance=phrase.guidance[:_MAX_GUIDANCE_CHARS],
            parameters=parameters, confidence="high",
            interpretation_source="deterministic"), context, task)

    async def _model(self, base: WorkflowIntent, context: WorkflowContext,
                     message: str, named: tuple[PausedTask, ...]) -> WorkflowIntent:
        # With several paused tasks the user must name the target; never guess.
        if len(context.paused) > 1 and len(named) != 1:
            return replace(base, interpretation_source="deterministic",
                           clarification=clarification_text(
                               context, "Several tasks are waiting for a decision."))
        task = named[0] if named else context.paused[0]
        allowed = [action.value for action in task.actions]
        modes = list(RECOVERY_MODES.get(S(task.state), ()))
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": [*allowed, A.ANSWER.value,
                                                      A.ASK_CLARIFICATION.value]},
                "target_task": {"type": "string", "enum": [task.task_id]},
                "guidance": {"type": "string"},
                "scope_paths": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "recovery_mode": {"type": "string", "enum": ["none", *modes]},
                "approved": {"type": "boolean"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "ambiguities": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            },
            "required": ["action", "target_task", "guidance", "scope_paths",
                         "recovery_mode", "approved", "confidence", "ambiguities"],
            "additionalProperties": False,
        }
        started = time.monotonic()
        model_base = replace(base, interpretation_source="model", model=self.model)
        try:
            value, usage = await emit_json_with_usage(self.provider, [
                {"role": "system", "content": (
                    "Classify one user reply for a paused coding workflow. The reply refers to "
                    "the workflow state below, not to a new implementation request. Choose "
                    "exactly one valid action, ANSWER for a question, or ASK_CLARIFICATION "
                    "when the reply does not clearly request one. Write guidance as a short imperative instruction "
                    "for the Coder that preserves the user's constraints. Use "
                    "EXPAND_CONTRACT_SCOPE only when the user explicitly wants more files in "
                    "the contract, and list them in scope_paths. When the user supplies facts "
                    "or the answer itself (for example 'there is no X in this project'), "
                    "choose RETRY_CONTRACT when it is valid and put the user's fact in guidance "
                    "as a statement the Coder can rely on. Use approved only for "
                    "ANSWER_REVIEWER_QUESTION. Never invent tasks, files, actions, or claim "
                    "that a review happened. Do not plan the implementation. Return only JSON.")},
                {"role": "user", "content": (
                    f"Workflow state:\n{context.render()}\n\nUser reply: {message}")},
            ], schema, self.model)
        except Exception as exc:
            return replace(model_base, clarification=clarification_text(
                context, "I could not interpret that reply automatically.", (task,)),
                latency_ms=int((time.monotonic() - started) * 1000),
                interpretation_error=type(exc).__name__)
        latency = int((time.monotonic() - started) * 1000)
        model_base = replace(model_base, latency_ms=latency, usage=usage or {})
        action = value.get("action")
        guidance = value.get("guidance")
        confidence = value.get("confidence")
        ambiguities = value.get("ambiguities")
        scope_paths = value.get("scope_paths") or []
        mode = value.get("recovery_mode") or "none"
        if (action not in {*allowed, A.ANSWER.value, A.ASK_CLARIFICATION.value}
                or value.get("target_task") not in {task.task_id, None}
                or not isinstance(guidance, str) or len(guidance) > _MAX_GUIDANCE_CHARS
                or confidence not in {"high", "medium", "low"}
                or not isinstance(ambiguities, list) or len(ambiguities) > 10
                or not all(isinstance(item, str) for item in ambiguities)
                or not isinstance(scope_paths, list)
                or not all(isinstance(item, str) for item in scope_paths)):
            return replace(model_base, clarification=clarification_text(
                context, "I could not safely determine the next workflow action.", (task,)))
        ambiguities = tuple(item.strip() for item in ambiguities if item.strip())
        if action == A.ASK_CLARIFICATION.value or confidence == "low":
            detail = "; ".join(ambiguities) or "I am not confident which action you want."
            return replace(model_base, confidence=confidence, ambiguities=ambiguities,
                           clarification=clarification_text(context, detail, (task,)))
        if action == A.ANSWER.value:
            return replace(model_base, action=A.ANSWER, confidence=confidence,
                           ambiguities=ambiguities)
        parameters: dict = {}
        if action == A.EXPAND_CONTRACT_SCOPE.value:
            parameters["scope_paths"] = list(dict.fromkeys(
                path.strip() for path in scope_paths if path.strip()))
        if action == A.RESOLVE_CODER_RECOVERY.value:
            parameters["recovery_mode"] = mode
        if action == A.ANSWER_REVIEWER_QUESTION.value:
            parameters["approved"] = bool(value.get("approved"))
        return self._finish(replace(
            model_base, action=A(action), guidance=guidance.strip(),
            confidence=confidence, ambiguities=ambiguities,
            parameters=parameters), context, task)

    @staticmethod
    def _finish(intent: WorkflowIntent, context: WorkflowContext,
                task: PausedTask) -> WorkflowIntent:
        """Bind the intent to its task and reject incomplete actions."""
        intent = replace(
            intent, source_state=task.state, target_job=task.job_id,
            target_position=task.position, target_task=task.task_id,
            expected_revision=task.contract_revision)
        mode = intent.parameters.get("recovery_mode", "")
        problem = None
        if not action_allowed(task.state, intent.action, mode):
            problem = "That action is not valid for this task's current state."
        elif intent.action in GUIDANCE_REQUIRED and not intent.guidance:
            problem = (f"Say what should change, for example "
                       f"{_hint(intent.action, task.state)}.")
        elif mode == "establish_baseline" and not intent.guidance:
            problem = "Establishing a new baseline needs recovery guidance."
        elif (intent.action == A.EXPAND_CONTRACT_SCOPE
              and not intent.parameters.get("scope_paths")):
            problem = "Name the files to add, for example `expand scope: <paths>`."
        if problem:
            return replace(intent, action=A.ASK_CLARIFICATION,
                           clarification=clarification_text(context, problem, (task,)))
        return intent.with_effects(expected_effects(intent, task))


class ConversationIntentInterpreter:
    """Classify a fresh free-form turn without granting execution authority.

    The model chooses an intent; CodePartner resolves selectors and builds the
    actual command. A malformed, unavailable, or speculative result is never
    executable.
    """

    _SCHEMA = {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": [
                "START_IMPLEMENTATION", "CODING", "REVIEW", "SPEC_DRAFT", "ANSWER",
                "ASK_CLARIFICATION"]},
            "selectors": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            "instruction": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "ambiguities": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        },
        "required": ["action", "selectors", "instruction", "confidence", "ambiguities"],
        "additionalProperties": False,
    }

    def __init__(self, provider=None, model: str | None = None):
        if provider is None:
            from providers.ollama import OllamaProvider
            provider = OllamaProvider()
        self.provider = provider
        self.model = model or intent_model()

    async def interpret(self, message: str, mode_hint: str = "",
                        context_summary: str = "") -> dict:
        if not message.strip() or len(message) > _MAX_MESSAGE_CHARS:
            return {"action": "ASK_CLARIFICATION", "reason": "Please send a shorter request."}
        seen = set(selectors_in_text(message))
        if (mode_hint == "review" and seen
                and not re.search(r"\b(?:implement|code|edit|write|build|fix)\b",
                                  message, re.IGNORECASE)):
            question = re.sub(r"^\s*(?:EPIC|WP|US)\d+(?:-T\d+)?\s*:\s*",
                              "", message.strip(), count=1, flags=re.IGNORECASE)
            return {"action": "REVIEW", "selectors": tuple(sorted(seen)),
                    "instruction": question, "confidence": "medium"}
        try:
            value, _ = await emit_json_with_usage(self.provider, [
                {"role": "system", "content": (
                    "Classify one new CodePartner conversation turn. The mode is a hint, "
                    "not an action gate. Choose ANSWER for an ordinary question; "
                    "START_IMPLEMENTATION for an OpenSpec-scoped coding task; CODING for "
                    "a coding task without an OpenSpec selector; REVIEW for read-only review; "
                    "SPEC_DRAFT for planning or changing an OpenSpec draft; otherwise "
                    "ASK_CLARIFICATION. Do not invent selectors. Return only JSON.")},
                {"role": "user", "content": (
                    f"Mode hint: {mode_hint or 'default'}\n"
                    f"Conversation state: {context_summary[:1200] or 'ready'}\n"
                    f"Selectors explicitly named: {', '.join(sorted(seen)) or 'none'}\n"
                    f"User message: {message}")},
            ], self._SCHEMA, self.model)
        except Exception:
            value = self._fallback(message, mode_hint, seen)
        action = value.get("action") if isinstance(value, dict) else None
        selectors = value.get("selectors") if isinstance(value, dict) else None
        instruction = value.get("instruction") if isinstance(value, dict) else None
        confidence = value.get("confidence") if isinstance(value, dict) else None
        if (action not in {"START_IMPLEMENTATION", "CODING", "REVIEW", "SPEC_DRAFT", "ANSWER",
                           "ASK_CLARIFICATION"}
                or not isinstance(selectors, list)
                or not all(isinstance(item, str) for item in selectors)
                or not isinstance(instruction, str)
                or confidence not in {"high", "medium", "low"}):
            return {"action": "ASK_CLARIFICATION", "reason": "I could not safely interpret that request."}
        normalized = tuple(dict.fromkeys(canonical_selector(item) for item in selectors))
        if (None in normalized or any(item not in seen for item in normalized)
                or confidence == "low"):
            return {"action": "ASK_CLARIFICATION", "reason": (
                "I am not confident about the intended action or task. "
                "Please name the task and what you want done.")}
        if action in {"START_IMPLEMENTATION", "REVIEW"} and not normalized:
            return {"action": "ASK_CLARIFICATION", "reason": (
                "Which OpenSpec task or work package do you mean?")}
        if action == "ASK_CLARIFICATION":
            ambiguities = value.get("ambiguities", [])
            detail = "; ".join(item for item in ambiguities if isinstance(item, str))
            return {"action": action, "reason": detail or (
                "Please clarify what you want CodePartner to do next.")}
        return {"action": action, "selectors": normalized,
                "instruction": instruction.strip()[:_MAX_GUIDANCE_CHARS],
                "confidence": confidence}

    @staticmethod
    def _fallback(message: str, mode_hint: str, selectors: set[str]) -> dict:
        text = message.lower()
        action = "ANSWER"
        if selectors and re.search(r"\b(?:start|implement|build|code|add|fix)\b", text):
            action = "START_IMPLEMENTATION"
        elif not selectors and re.search(r"\b(?:implement|build|code|add|fix)\b", text):
            action = "CODING" if mode_hint != "implement" else "ASK_CLARIFICATION"
        elif selectors and re.search(r"\b(?:review|audit|inspect)\b", text):
            action = "REVIEW"
        elif mode_hint == "implement" and selectors and not message.strip().endswith("?"):
            action = "START_IMPLEMENTATION"
        elif mode_hint == "spec" and not message.strip().endswith("?"):
            action = "SPEC_DRAFT"
        return {"action": action, "selectors": list(selectors),
                "instruction": message.strip(), "confidence": "medium", "ambiguities": []}

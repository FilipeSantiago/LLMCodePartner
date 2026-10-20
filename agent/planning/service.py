"""Interactive Work Package planning: start, discuss, approve (freeze), abandon.

Every planner turn returns the human-readable plan and the structured context
together, so approval only validates and freezes the current state.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from uuid import uuid4

import providers
from agent.kb_first import (DiscoveryTracker, EvidenceSeed, KbFirstPolicy,
                            evidence_seed, enforced_reads, native_gate_metadata)
from agent.planning.complexity import effective_wp_complexity
from agent.planning.models import ApprovedWPPlan, WPPlanDraft, plan_id_for
from agent.planning.projection import project, validate_draft
from agent.planning.prompts import repair_prompt, turn_prompt, wp_brief
from agent.planning.routing import PlannerRoutingService
from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.gateway import gateway_attempts, gateway_kwargs
from agent.routing.review import model_identity
from agent.routing.tool_metrics import instrument_handlers, read_only_handlers
from agent.spec.models import WorkPackage
from agent.workspace import plan_native, read_only_session
from conversation.presentation import progress_line
from mcp_bridge.activity import describe_call
from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent, ToolCallEvent
from mcp_bridge.registry import ROLE_OPTIMIZER, ToolSpec

log = logging.getLogger("codepartner.planning")

Progress = Callable[[str], Awaitable[None]]
_NARRATION_MAX_CHARS = 200
_SEARCH_KEYS = ("q", "query", "pattern", "searchText", "text", "nameKeyword", "regex")


def _failure_kind(message: str, use_gateway: bool = False) -> str:
    """Classify only operational failures as candidates for planner failover."""
    value = message.lower()
    if any(marker in value for marker in (
            "session limit", "usage limit", "quota exceeded", "quota exhausted",
            "insufficient_quota", "rate limit", "rate_limit", "too many requests")):
        return "quota"
    # A CLI login preflight is account-wide even when this attempt was configured
    # to use the gateway; no request reached a route worth retrying directly.
    if any(marker in value for marker in (
            "not authenticated", "login required", "please log in")):
        return "unavailable"
    if use_gateway and any(marker in value for marker in (
            "http 401", "status 401", "http 403", "status 403")):
        return "gateway_auth"
    if any(marker in value for marker in (
            "authentication failed", "authentication error", "not authenticated",
            "unauthorized", "invalid api key", "invalid_api_key", "permission denied",
            "login required", "please log in", "model not available",
            "model is unavailable", "model_not_found", "executable not found",
            "http 401", "status 401", "http 403", "status 403")):
        return "unavailable"
    if any(marker in value for marker in (
            "http 429", "status 429", "http 500", "status 500",
            "http 502", "status 502", "http 503",
            "status 503", "http 504", "status 504", "bad gateway",
            "service unavailable", "gateway timeout", "connection refused",
            "connection reset", "connection error", "timed out", "timeout",
            "planner stopped without a terminal result")):
        return "transport"
    return "other"


def _failure_label(kind: str) -> str:
    """Brief chat-facing reason; raw provider diagnostics remain in turn records."""
    return {
        "quota": "session or usage limit",
        "gateway_auth": "gateway authorization denied",
        "unavailable": "provider unavailable or authentication denied",
        "transport": "connection failed",
    }[kind]


async def _silent(_text: str) -> None:
    return None


def describe_tool(name: str, arguments: dict, reads: frozenset[str],
                  searches: frozenset[str]) -> str:
    """What a read-only tool call inspects, in a few words (no model tokens involved)."""
    if name.rsplit("__", 1)[-1].startswith("kb_"):
        return describe_call(name, arguments)
    if name in reads:
        path = arguments.get("file_path") or arguments.get("path") or "a file"
        offset, limit = arguments.get("offset"), arguments.get("limit")
        span = (f" (lines {offset}–{offset + limit - 1})"
                if isinstance(offset, int) and isinstance(limit, int) else "")
        return f"Reading `{path}`{span}"
    if name in searches:
        query = next((arguments[key] for key in _SEARCH_KEYS if arguments.get(key)), None)
        return f"Searching for `{query}`" if query else f"Searching (`{name}`)"
    return describe_call(name, arguments)


def narration_lines(text: str) -> list[str]:
    """Short prose lines from planner text; JSON (the final answer) is never echoed."""
    if "{" in text and "plan_markdown" in text:
        return []
    lines = []
    for line in text.splitlines():
        line = line.strip().strip("*_").strip()
        if (line and len(line) <= _NARRATION_MAX_CHARS
                and not line.startswith(("{", "}", "[", "]", "```", '"'))):
            lines.append(line)
    return lines


async def with_progress(call: Callable[[Progress], Awaitable["PlanningTurn"]]
                        ) -> AsyncIterator[tuple[str, object]]:
    """Run a planner call, yielding ("progress", text) live, then ("result", turn)."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def progress(text: str) -> None:
        queue.put_nowait(text)

    task = asyncio.create_task(call(progress))
    try:
        while not task.done() or not queue.empty():
            if not queue.empty():
                yield "progress", queue.get_nowait()
                continue
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield "progress", getter.result()
            else:
                getter.cancel()
        yield "result", task.result()
    finally:
        if not task.done():
            task.cancel()


class PlanningError(Exception):
    pass


@dataclass(frozen=True)
class PlanningTurn:
    session_id: str
    wp_id: str
    revision: int
    draft: WPPlanDraft
    gaps: list[str]
    wp_complexity: str
    planner_model: str
    planner_label: str
    task_ids: list[str]
    # WPs from the same request, planned one at a time after this one completes.
    queued_wp_ids: list[str] = field(default_factory=list)


def wp_fingerprint(package: WorkPackage) -> str:
    """Identity of the planned WP content; review outcomes do not change the plan basis."""
    value = asdict(package)
    value.pop("review", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()


def active_task_ids(package: WorkPackage) -> list[str]:
    return [task.id for task in package.tasks if task.status != "cancelled"]


def _kb_offered(specs: list[ToolSpec]) -> bool:
    return any(spec.name.rsplit("__", 1)[-1].startswith("kb_") for spec in specs)


def _plan_json(text: str) -> dict | None:
    """The last JSON object in the reply that looks like a plan."""
    decoder = json.JSONDecoder()
    found = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "plan_markdown" in value:
            found = value
    return found


class WPPlannerService:
    def __init__(self, artifacts, persistence, config: RoutingConfig | None = None,
                 router: PlannerRoutingService | None = None, provider_lookup=None):
        self.artifacts = artifacts
        self.persistence = persistence
        self.config = config or load_routing_config()
        self._router = router
        self.provider_lookup = provider_lookup or providers.get

    @property
    def router(self) -> PlannerRoutingService:
        if self._router is None:
            self._router = PlannerRoutingService(self.config)
        return self._router

    async def _package(self, wp_id: str):
        kind, change, package = await self.artifacts.object_by_id(wp_id)
        if kind != "work_package":
            raise PlanningError(f"{wp_id} is not a work package")
        return change, package

    async def start(self, conversation_id: str, wp_id: str, message: str,
                    specs: list[ToolSpec], handlers: dict,
                    project_id: str | None = None,
                    queued_wp_ids: tuple[str, ...] = (),
                    progress: Progress | None = None,
                    heading: str | None = None,
                    kb_status: str = "") -> PlanningTurn:
        change, package = await self._package(wp_id)
        if not active_task_ids(package):
            raise PlanningError(f"{package.id} has no tasks to plan")
        complexity, source = effective_wp_complexity(package)
        decision = self.router.route(complexity)
        session_id = "wpplan_" + uuid4().hex
        await self.persistence.create_planning({
            "session_id": session_id, "conversation_id": conversation_id,
            "project_id": project_id, "change_id": change.id, "wp_id": package.id,
            "wp_fingerprint": wp_fingerprint(package), "wp_complexity": complexity,
            "complexity_source": source, "planner_model": decision.model,
            "planner_effort": decision.reasoning_effort,
            "queued_wp_ids": list(queued_wp_ids), "request": message,
        })
        log.info("wp_planning_started session=%s wp=%s complexity=%s source=%s model=%s",
                 session_id, package.id, complexity, source, decision.model)
        session = await self.persistence.planning_session(session_id)
        progress = progress or _silent
        await progress(progress_line("header", (heading or f"Investigating {package.id}")
                                     + f" (planner: {self._label(decision.model)})"))
        return await self._turn(session, change, package, message, specs, handlers, "plan",
                                progress, kb_status)

    async def discuss(self, conversation_id: str, message: str, specs: list[ToolSpec],
                      handlers: dict, progress: Progress | None = None,
                      kb_status: str = "") -> PlanningTurn:
        session = await self.persistence.active_planning(conversation_id)
        if session is None:
            raise PlanningError("no Work Package is being planned in this conversation")
        change, package = await self._package(session["wp_id"])
        progress = progress or _silent
        await progress(progress_line(
            "header", f"Revising the {package.id} plan "
                      f"(planner: {self._label(session['planner_model'])})"))
        return await self._turn(session, change, package, message, specs, handlers, "discuss",
                                progress, kb_status)

    async def abandon(self, conversation_id: str) -> dict | None:
        session = await self.persistence.active_planning(conversation_id)
        if session is not None:
            await self.persistence.finish_planning(session["session_id"], "abandoned")
        return session

    async def approve(self, session_id: str, revision: int) -> ApprovedWPPlan:
        """Freeze one validated revision. No model is called."""
        session = await self.persistence.planning_session(session_id)
        if session is None or session["status"] != "active":
            raise PlanningError("that planning session is no longer active")
        if session["revision"] != revision or session["draft"] is None:
            raise PlanningError("the plan was revised after this approval was offered")
        change, package = await self._package(session["wp_id"])
        if wp_fingerprint(package) != session["wp_fingerprint"]:
            raise PlanningError(f"{package.id} changed in OpenSpec after planning began; "
                                "discuss the plan again before approving")
        draft = WPPlanDraft.from_dict(session["draft"], session["wp_id"], revision)
        gaps = validate_draft(draft, active_task_ids(package))
        if gaps:
            raise PlanningError("the plan is incomplete: " + "; ".join(gaps))
        plan = project(
            draft, plan_id=plan_id_for(session_id, revision), session_id=session_id,
            change_id=change.id, wp_fingerprint=session["wp_fingerprint"],
            wp_complexity=session["wp_complexity"], planner_model=session["planner_model"],
            provenance={
                "source": "wp_planner", "session_id": session_id, "revision": revision,
                "planner_model": session["planner_model"],
                "planner_effort": session["planner_effort"],
                "wp_complexity": session["wp_complexity"],
                "complexity_source": session["complexity_source"],
                "conversation_id": session["conversation_id"],
            })
        await self.persistence.save_approved_plan(plan.to_dict())
        await self.persistence.finish_planning(session_id, "approved")
        log.info("wp_plan_approved session=%s plan=%s wp=%s revision=%s",
                 session_id, plan.plan_id, plan.wp_id, revision)
        return plan

    async def _turn(self, session: dict, change, package: WorkPackage, message: str,
                    specs: list[ToolSpec], handlers: dict, kind: str,
                    progress: Progress = _silent, kb_status: str = "") -> PlanningTurn:
        revision = session["revision"] + 1
        task_ids = active_task_ids(package)
        brief = wp_brief(change, package, session["wp_complexity"])
        prompt = turn_prompt(brief, session["discussion"], session["draft"], message,
                             narrate=self.config.planner.narrate, kb_status=kb_status,
                             discovery_guidance=(
                                 self.config.planner.discovery.mode_for("planner") == "advise"),
                             kb_tools=(_kb_offered(specs)
                                       and self.config.planner.discovery.enabled),
                             native=plan_native(self.config.models[
                                 session["planner_model"]].provider,
                                 controlled_reads=enforced_reads(self.config.discovery)).native)
        mode = self.config.planner.discovery.mode_for("planner")
        evidence = evidence_seed(brief + "\n" + message, kb_status) if mode != "off" else None
        conversation = [{"role": "user", "content": prompt}]
        reply = await self._call(session, revision, kind, message, conversation, specs, handlers,
                                 progress, evidence)
        draft, gaps = self._parse(reply, session["wp_id"], revision, task_ids)
        if draft is None or gaps:
            problems = gaps or ["the reply contained no plan JSON object with plan_markdown"]
            conversation += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": repair_prompt(problems)}]
            await progress(progress_line(
                "note", "Completing the structured plan: " + "; ".join(problems)[:160]))
            repaired = await self._call(session, revision, "repair", None, conversation,
                                        specs, handlers, progress, evidence)
            repaired_draft, repaired_gaps = self._parse(
                repaired, session["wp_id"], revision, task_ids)
            if repaired_draft is not None:
                draft, gaps = repaired_draft, repaired_gaps
        if draft is None:
            raise PlanningError("the planner did not return a plan; nothing changed. "
                                "Try again or rephrase your request.")
        discussion = [*session["discussion"], {"revision": revision, "text": message}]
        await self.persistence.update_planning(
            session["session_id"], revision, draft.to_dict(), discussion, gaps)
        model = session["planner_model"]
        return PlanningTurn(
            session_id=session["session_id"], wp_id=session["wp_id"], revision=revision,
            draft=draft, gaps=gaps, wp_complexity=session["wp_complexity"],
            planner_model=model, planner_label=self._label(model), task_ids=task_ids,
            queued_wp_ids=list(session.get("queued_wp_ids") or ()))

    def _label(self, name: str) -> str:
        try:
            return model_identity(self.config, name).label
        except Exception:
            return name

    @staticmethod
    def _parse(reply: str, wp_id: str, revision: int, task_ids: list[str]
               ) -> tuple[WPPlanDraft | None, list[str]]:
        payload = _plan_json(reply)
        if payload is None:
            return None, []
        draft = WPPlanDraft.from_dict(payload, wp_id, revision)
        return draft, validate_draft(draft, task_ids)

    async def _call(self, session: dict, revision: int, kind: str, message: str | None,
                    conversation: list[dict], specs: list[ToolSpec], handlers: dict,
                    progress: Progress = _silent,
                    evidence: EvidenceSeed | None = None) -> str:
        narrate = self.config.planner.narrate
        readable, reads, searches = read_only_handlers(specs, handlers)
        current = session["planner_model"]
        candidates = (current, *(name for name in self.config.planner.candidates_for(
            session["wp_complexity"]) if name != current))
        failures: list[str] = []
        unavailable_providers: set[str] = set()
        for index, name in enumerate(candidates):
            model = self.config.models[name]
            if model.provider in unavailable_providers:
                continue
            effort = (self.config.planner.reasoning_effort.get(session["wp_complexity"])
                      or model.reasoning_effort)
            for attempt, use_gateway in enumerate(gateway_attempts(model.gateway_mode), 1):
                failure_kind = await self._attempt_planner(
                    session, revision, kind, message, conversation, specs, progress,
                    readable, reads, searches, name, model, effort, narrate,
                    attempt, use_gateway, evidence)
                if failure_kind[0] is None:
                    if name != current:
                        await self.persistence.update_planning_model(
                            session["session_id"], current, name, effort)
                        session["planner_model"] = name
                        session["planner_effort"] = effort
                    return failure_kind[1]
                classification, failure = failure_kind
                if classification == "other":
                    raise PlanningError(f"the planner failed: {failure}")
                route = "gateway" if use_gateway else "direct"
                failures.append(f"{self._label(name)} ({route}): "
                                f"{_failure_label(classification)}")
                if classification in {"quota", "unavailable"}:
                    unavailable_providers.add(model.provider)
                    break
                if use_gateway and model.gateway_mode == "preferred":
                    reason = ("authorization failed" if classification == "gateway_auth"
                              else "attempt failed")
                    await progress(progress_line(
                        "note", f"Planner gateway {reason}; retrying directly"))
            next_name = next((candidate for candidate in candidates[index + 1:]
                              if self.config.models[candidate].provider
                              not in unavailable_providers), None)
            if next_name:
                await progress(progress_line(
                    "note", f"{self._label(name)} planner unavailable; "
                            f"switching to {self._label(next_name)}"))
        raise PlanningError("all planner candidates failed: " + "; ".join(failures))

    async def _attempt_planner(self, session: dict, revision: int, kind: str,
                               message: str | None, conversation: list[dict],
                               specs: list[ToolSpec], progress: Progress,
                               readable: dict, reads: frozenset[str],
                               searches: frozenset[str], name: str, model,
                               effort: str | None, narrate: bool, attempt: int,
                               use_gateway: bool, evidence: EvidenceSeed | None = None
                               ) -> tuple[str | None, str]:
        reported: set[str] = set()
        notes = {"calls": "Investigation budget reached; concluding with what I found",
                 "reads": "File-read budget reached; concluding from what was found"}

        async def on_call(tool: str, arguments: dict, refused: str | None) -> None:
            if not refused:
                await progress(progress_line("tool", describe_tool(tool, arguments,
                                                                   reads, searches)))
            elif refused not in reported:
                reported.add(refused)
                await progress(progress_line("note", notes[refused]))

        plan = plan_native(model.provider,
                           controlled_reads=enforced_reads(self.config.discovery))
        if plan.mode == "native" and not plan.native:
            raise PlanningError(f"native execution is unavailable: {plan.reason}")
        # Read-only native run: its own file tools, the KB on demand, no KB-first gates.
        native = read_only_session(plan.project) if plan.native else None
        discovery = self.config.planner.discovery
        planner_mode = discovery.mode_for("planner")
        # In observe the tracker changes nothing; in advise it may append advisories.
        tracker = (DiscoveryTracker(
            evidence, discovery.small_file_bytes, _kb_offered(specs),
            thresholds=(discovery.thresholds_for("planner", session["wp_complexity"])
                        if planner_mode == "advise" else None),
            tier=session["wp_complexity"])
                   if evidence is not None and planner_mode != "off" and native is None
                   else None)
        if tracker is not None:
            tracker.prompt(conversation)
        # KB-first whenever Codebase Memory is offered, whatever the discovery mode.
        wrapped, metrics = instrument_handlers(
            readable, reads, searches, self.config.planner.max_tool_calls, on_call,
            max_reads=self.config.planner.max_file_reads, tracker=tracker)
        policy = (KbFirstPolicy(
            role="planner", tool_names=[spec.name for spec in specs],
            seed=evidence, tracker=tracker,
            outline_gate_lines=discovery.outline_gate_lines,
            search_gate_max=getattr(discovery, "search_gate_max", 5),
            mode=discovery.mode_for("planner"), tier=session["wp_complexity"],
            small_file_bytes=discovery.small_file_bytes,
            enforce_reads=enforced_reads(discovery))
            if getattr(discovery, "enabled", True)
            and native is None else None)
        kwargs = {"model": model.cli_model, "role": ROLE_OPTIMIZER,
                  "direct_tool_handlers": wrapped,
                  "kb_first": policy, "native": native,
                  "run_metadata": {"planner": True, "planner_model": name,
                                   # Planners are KB-first under every policy.
                                   **native_gate_metadata(discovery, always=True),
                                   "planning_session_id": session["session_id"],
                                   "wp_id": session["wp_id"],
                                   "gateway_used": use_gateway}}
        if model.provider == "codex":
            kwargs["concrete_model"] = True
            kwargs["reasoning_effort"] = effort
            kwargs["reviewer_mode"] = True
        kwargs.update(gateway_kwargs(model.provider, self.config.agentgateway_url,
                                     use_gateway))
        parts: list[str] = []
        errors: list[str] = []
        usage = None
        completed = False
        hard_failure = False
        cancelled: asyncio.CancelledError | None = None
        started = time.monotonic()
        try:
            async for event in self.provider_lookup(model.provider).tools(
                    conversation, specs, **kwargs):
                if isinstance(event, TextEvent):
                    parts.append(event.text)
                    if narrate and _failure_kind(event.text) not in {"quota", "unavailable"}:
                        for line in narration_lines(event.text):
                            await progress(progress_line("narration", line))
                elif isinstance(event, ErrorEvent):
                    errors.append(event.message)
                elif isinstance(event, ToolCallEvent):
                    hard_failure = True
                    raise PlanningError("the planner requested an IDE tool that is not "
                                        "available to it")
                elif isinstance(event, DoneEvent):
                    completed = True
                    usage = event.usage
        except asyncio.CancelledError as exc:
            # A stopped run still records what it consumed, then stops.
            cancelled = exc
            errors.append("cancelled")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        reply = "".join(parts)
        failure = "; ".join(errors) or (
            None if completed else "planner stopped without a terminal result")
        if ((failure is None or failure == "planner stopped without a terminal result")
                and len(reply) < 1000 and _plan_json(reply) is None
                and _failure_kind(reply) in {"quota", "unavailable"}):
            failure = reply.strip()
        classification = ("cancelled" if cancelled else
                          _failure_kind(failure, use_gateway) if failure else None)
        await self.persistence.record_planning_turn({
            "session_id": session["session_id"], "revision": revision, "kind": kind,
            "user_message": message, "output": reply, "model": name, "usage": usage,
            "cost_usd": (usage or {}).get("total_cost_usd"),
            "tool_metrics": {**metrics.summary(), "narrate": narrate,
                             **({"execution_mode": "native",
                                 "native_tool_metrics": native.metrics} if native else {}),
                             "gateway_used": use_gateway, "gateway_attempt": attempt,
                             "failure_kind": classification,
                             **({"cancelled": True} if cancelled else {}),
                             **({"outline_gate": policy.outline.summary()} if policy else {}),
                             **({"discovery": policy.summary()} if policy else
                                {"discovery": tracker.summary()} if tracker else {})},
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": failure,
        })
        if cancelled is not None:
            raise cancelled
        self.router.health.record(name, model.executor, use_gateway, failure is None)
        if failure is None:
            return None, reply
        if hard_failure:
            raise PlanningError(failure)
        return classification, failure

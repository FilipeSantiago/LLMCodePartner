"""Read-only, outcome-based review of an implemented OpenSpec task."""

import json
import logging
from dataclasses import dataclass

import providers
from agent.routing.config import RoutingConfig, load_routing_config
from agent.spec.models import OpenSpecChange, Task
from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent, ToolCallEvent
from mcp_bridge.registry import ROLE_OPTIMIZER, ToolSpec

log = logging.getLogger("codepartner.review")


@dataclass(frozen=True)
class ReviewDecision:
    verdict: str
    rationale: str
    reviewer_model: str
    question: str | None = None
    feedback: str | None = None


class ReviewError(RuntimeError):
    pass


class ReviewService:
    def __init__(self, config: RoutingConfig | None = None):
        self.config = config or load_routing_config()

    def _same_provider_model(self, coder_model: str) -> str:
        coder = self.config.models[coder_model]
        tier = next((name for name, models in self.config.tiers.items()
                     if coder_model in models), None)
        choices = [name for name in self.config.tiers.get(tier, ())
                   if self.config.models[name].provider == coder.provider and name != coder_model]
        return choices[0] if choices else coder_model

    async def review(self, change: OpenSpecChange, task: Task, report: str,
                     coder_model: str, specs: list[ToolSpec],
                     direct_tool_handlers: dict | None = None) -> ReviewDecision:
        if coder_model not in self.config.models:
            raise ReviewError(f"unknown actual coder model {coder_model!r}")
        preferred = (self.config.reviewers or {}).get(coder_model)
        same_provider = self._same_provider_model(coder_model)
        if preferred is None:
            log.info("review_provider_mode mode=same_provider reason=no_opposite_pair coder_model=%s",
                     coder_model)
            return await self._run_review(change, task, report, same_provider,
                                          specs, direct_tool_handlers)
        coder = self.config.models[coder_model]
        tier = next((name for name, names in self.config.tiers.items()
                     if coder_model in names), None)
        alternatives = [name for name in self.config.tiers.get(tier, ())
                        if name != preferred and self.config.models[name].provider != coder.provider]
        for reviewer_name in (preferred, *alternatives):
            try:
                return await self._run_review(change, task, report, reviewer_name,
                                              specs, direct_tool_handlers)
            except ReviewError as exc:
                if self._provider_unavailable(str(exc)):
                    log.warning("review_provider_mode mode=same_provider reason=opposite_unavailable "
                                "coder_model=%s reviewer_model=%s", coder_model, reviewer_name)
                    return await self._run_review(change, task, report, same_provider,
                                                  specs, direct_tool_handlers)
                if not self._model_unavailable(str(exc)):
                    raise
        raise ReviewError("no comparable opposite-provider reviewer model is available")

    @staticmethod
    def _provider_unavailable(message: str) -> bool:
        lower = message.lower()
        return any(phrase in lower for phrase in (
            "not authenticated", "authentication required", "please log in",
            "login required", "not logged in", "unauthorized", "invalid credentials",
            "api key missing", "not installed", "no such file or directory",
            "command not found", "executable not found",
        ))

    @staticmethod
    def _model_unavailable(message: str) -> bool:
        lower = message.lower()
        return "model" in lower and any(phrase in lower for phrase in (
            "not found", "not available", "unavailable", "not supported", "access denied"
        ))

    @staticmethod
    def _parse_decision(response: str) -> dict:
        decoder = json.JSONDecoder()
        candidates = []
        for index, char in enumerate(response):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("verdict") in {
                    "done", "needs_work", "ask_user"}:
                candidates.append(value)
        if not candidates:
            raise ReviewError("reviewer returned no valid JSON decision")
        return candidates[-1]

    async def _run_review(self, change: OpenSpecChange, task: Task, report: str,
                          reviewer_name: str, specs: list[ToolSpec],
                          direct_tool_handlers: dict | None) -> ReviewDecision:
        model = self.config.models[reviewer_name]
        package = next((package for package in change.work_packages
                        if any(item.id == task.id for item in package.tasks)), None)
        criteria = package.acceptance_criteria if package else []
        messages = [{"role": "user", "content": (
            "Review the completed work independently against its purpose. You have read-only "
            "IDE tools to inspect the result when useful. Do not edit files. Tool calls, file "
            "counts, and test commands are not completion criteria. Judge the quality and "
            "completeness of the delivered result. For research or verification work whose "
            "completion requires human judgment, choose ask_user, even if a report file exists. "
            "Return only one JSON object with keys verdict (done, needs_work, or ask_user), "
            "rationale, feedback, and question. Give concrete gaps in feedback when needs_work; "
            "give one specific question when ask_user.\n\n"
            f"Change: {change.title}\nTask: {task.id} {task.title}\n"
            f"Description: {task.description}\nContext: {task.context}\n"
            f"Preferred capability: {task.preferred_capability}\n"
            f"Work package objective: {package.objective if package else ''}\n"
            f"Work package acceptance criteria: {json.dumps(criteria)}\n\n"
            f"Coder's result (a claim to assess):\n{report or '(no result supplied)'}"
        )}]
        provider = providers.get(model.provider)
        kwargs = {"model": model.cli_model, "role": ROLE_OPTIMIZER,
                  "direct_tool_handlers": direct_tool_handlers or {}}
        if model.provider == "codex":
            kwargs["concrete_model"] = True
            kwargs["reasoning_effort"] = model.reasoning_effort
        elif model.gateway_mode == "required":
            kwargs["execution_env"] = {"ANTHROPIC_BASE_URL": self.config.agentgateway_url}
        if model.provider == "codex" and model.gateway_mode == "required":
            kwargs["gateway_url"] = self.config.agentgateway_url
        parts: list[str] = []
        errors: list[str] = []
        completed = False
        try:
            async for event in provider.tools(
                    messages, specs if direct_tool_handlers else [], **kwargs):
                if isinstance(event, TextEvent):
                    parts.append(event.text)
                elif isinstance(event, ErrorEvent):
                    errors.append(event.message)
                elif isinstance(event, ToolCallEvent):
                    raise ReviewError("reviewer requested an unavailable IDE tool round trip")
                elif isinstance(event, DoneEvent):
                    completed = True
        except ReviewError:
            raise
        except Exception as exc:
            raise ReviewError(f"{type(exc).__name__}: {exc}") from exc
        if errors or not completed:
            raise ReviewError("; ".join(errors) or "reviewer stopped without a terminal result")
        value = self._parse_decision("".join(parts))
        if not isinstance(value, dict) or value.get("verdict") not in {
                "done", "needs_work", "ask_user"} or not isinstance(value.get("rationale"), str):
            raise ReviewError("reviewer returned an invalid verdict")
        verdict = value["verdict"]
        # Research outcomes remain the user's decision even when the reviewer is satisfied.
        if task.preferred_capability.lower() == "research" and verdict == "done":
            verdict = "ask_user"
        question = value.get("question")
        if verdict == "ask_user" and not isinstance(question, str):
            question = f"Do you consider {task.id} complete based on these findings?"
        feedback = value.get("feedback")
        if verdict == "needs_work" and not isinstance(feedback, str):
            feedback = value["rationale"]
        log.info("review_decision task_id=%s reviewer_model=%s verdict=%s",
                 task.id, reviewer_name, verdict)
        return ReviewDecision(verdict, value["rationale"], reviewer_name,
                              question if verdict == "ask_user" else None,
                              feedback if verdict == "needs_work" else None)

"""Complexity-constrained model selection."""

from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

from agent.routing.config import RoutingConfig
from agent.routing.health import ExecutorHealth
from agent.routing.models import ExecutionDecision
from agent.routing.semantic_router import SemanticRouterClient, SemanticRouterError
from agent.spec.models import Task


class ModelRoutingService:
    def __init__(self, config: RoutingConfig, client: SemanticRouterClient | None = None,
                 health: ExecutorHealth | None = None):
        self.config = config
        self.client = client or SemanticRouterClient(
            config.semantic_router_url, config.router_timeout_seconds
        )
        self.health = health or ExecutorHealth(config.failure_threshold)

    async def route(self, task: Task, change_id: str, project_path: str) -> ExecutionDecision:
        candidates = self.config.candidates_for(task.complexity)
        eligible = tuple(candidate for candidate in candidates if self.health.available(
            candidate, self.config.models[candidate].executor, True
        )) or candidates
        source, reason = "semantic-router", None
        try:
            recommendation = await self.client.select(self._router_text(task), task.complexity)
            if recommendation.model not in candidates:
                raise SemanticRouterError(
                    f"router selected {recommendation.model!r}, outside {list(candidates)!r}"
                )
            selected = recommendation.model
            reason = recommendation.reason
        except SemanticRouterError as exc:
            selected = eligible[0]
            source = "deterministic-fallback"
            reason = str(exc)
        model = self.config.models[selected]
        fallbacks = tuple(candidate for candidate in candidates if candidate != selected)
        project_id = sha256(project_path.encode()).hexdigest()[:16]
        return ExecutionDecision(
            decision_id="route_" + uuid4().hex,
            project_id=project_id,
            change_id=change_id,
            task_id=task.id,
            complexity=task.complexity,
            candidates=candidates,
            recommended_model=selected,
            recommended_provider=model.provider,
            executor=model.executor,
            fallback_models=fallbacks,
            routing_reason=reason,
            routing_source=source,
            routed_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _router_text(task: Task) -> str:
        return "\n".join(filter(None, [
            f"Task: {task.title}",
            f"Description: {task.description}",
            f"Complexity: {task.complexity}",
            f"Context: {task.context}",
            f"Preferred capability: {task.preferred_capability}",
        ]))

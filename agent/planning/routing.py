"""Planner model selection: WP complexity -> planner tier, independent of coder routing."""

from dataclasses import dataclass

from agent.routing.config import RoutingConfig, RoutingConfigError
from agent.routing.gateway import gateway_attempts
from agent.routing.health import ExecutorHealth


@dataclass(frozen=True)
class PlannerDecision:
    model: str
    provider: str
    cli_model: str | None
    reasoning_effort: str | None
    wp_complexity: str
    candidates: tuple[str, ...]
    fallbacks: tuple[str, ...]
    reason: str


class PlannerRoutingService:
    def __init__(self, config: RoutingConfig, health: ExecutorHealth | None = None):
        if config.planner is None:
            raise RoutingConfigError("model routing config has no planner section")
        self.config = config
        self.health = health or ExecutorHealth(config.failure_threshold)

    def route(self, wp_complexity: str, exclude: frozenset[str] = frozenset()
              ) -> PlannerDecision:
        """Deterministic first healthy candidate; no semantic router for planning."""
        planner = self.config.planner
        candidates = planner.candidates_for(wp_complexity)
        remaining = tuple(name for name in candidates if name not in exclude) or candidates
        eligible = tuple(name for name in remaining if any(
            self.health.available(name, self.config.models[name].executor, use_gateway)
            for use_gateway in gateway_attempts(self.config.models[name].gateway_mode)
        )) or remaining
        selected = eligible[0]
        model = self.config.models[selected]
        effort = planner.reasoning_effort.get(wp_complexity) or model.reasoning_effort
        return PlannerDecision(
            model=selected, provider=model.provider, cli_model=model.cli_model,
            reasoning_effort=effort, wp_complexity=wp_complexity, candidates=candidates,
            fallbacks=tuple(name for name in remaining if name != selected),
            reason=f"planner tier {wp_complexity!r}: first healthy candidate",
        )

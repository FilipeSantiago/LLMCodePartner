"""Validated, environment-aware model-routing configuration."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent.routing.models import ModelRef

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model-routing.yaml"
_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


class RoutingConfigError(ValueError):
    pass


ROLES = ("coder", "reviewer", "planner")
PLANNER_TIERS = ("low", "medium", "high", "complex")
_EFFORTS = {"low", "medium", "high", "xhigh"}


def _expand(value: object) -> object:
    if isinstance(value, str):
        return _ENV.sub(lambda match: os.getenv(match.group(1), match.group(2) or ""), value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


@dataclass(frozen=True)
class PlannerConfig:
    """WP planner routing: WP complexity -> candidate models, independent of coder tiers."""
    tiers: dict[str, tuple[str, ...]]
    reasoning_effort: dict[str, str] = field(default_factory=dict)
    max_tool_calls: int = 60
    # The planner writes one short "what and why" line before each tool call.
    narrate: bool = True

    def candidates_for(self, complexity: str) -> tuple[str, ...]:
        try:
            return self.tiers[complexity]
        except KeyError as exc:
            raise RoutingConfigError(f"no planner tier for WP complexity {complexity!r}") from exc


@dataclass(frozen=True)
class RoutingConfig:
    models: dict[str, ModelRef]
    tiers: dict[str, tuple[str, ...]]
    semantic_router_url: str
    agentgateway_url: str
    reviewers: dict[str, str] | None = None
    router_timeout_seconds: float = 10.0
    failure_threshold: int = 2
    role_floors: dict[str, int] = field(default_factory=dict)
    planner: PlannerConfig | None = None

    def candidates_for(self, complexity: str) -> tuple[str, ...]:
        try:
            return self.tiers[complexity]
        except KeyError as exc:
            raise RoutingConfigError(f"no routing tier for complexity {complexity!r}") from exc


def load_routing_config(path: Path | str | None = None) -> RoutingConfig:
    config_path = Path(path or os.getenv("MODEL_ROUTING_CONFIG", DEFAULT_CONFIG_PATH))
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except OSError as exc:
        raise RoutingConfigError(f"could not read routing config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RoutingConfigError(f"invalid routing YAML: {exc}") from exc
    raw = _expand(raw)
    if not isinstance(raw, dict):
        raise RoutingConfigError("routing config must be a mapping")

    models: dict[str, ModelRef] = {}
    for name, value in (raw.get("models") or {}).items():
        if not isinstance(value, dict):
            raise RoutingConfigError(f"model {name!r} must be a mapping")
        provider = str(value.get("provider") or "").lower()
        executor = str(value.get("executor") or "")
        cli_model = str(value.get("cli_model") or "").strip() or None
        gateway_mode = str(value.get("gateway_mode") or "preferred").lower()
        reasoning_effort = value.get("reasoning_effort")
        review_reasoning_effort = value.get("review_reasoning_effort")
        if provider not in {"claude", "codex"}:
            raise RoutingConfigError(f"model {name!r} has unsupported provider")
        if executor not in {"claude-cli", "codex-cli"}:
            raise RoutingConfigError(f"model {name!r} has unsupported executor")
        if (provider == "claude") != (executor == "claude-cli"):
            raise RoutingConfigError(f"model {name!r} provider and executor disagree")
        if gateway_mode not in {"preferred", "required", "disabled"}:
            raise RoutingConfigError(f"model {name!r} has invalid gateway_mode")
        if reasoning_effort is not None and reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise RoutingConfigError(f"model {name!r} has invalid reasoning_effort")
        if (review_reasoning_effort is not None
                and review_reasoning_effort not in {"low", "medium", "high", "xhigh"}):
            raise RoutingConfigError(
                f"model {name!r} has invalid review_reasoning_effort")
        capability = value.get("capability")
        if capability is not None and (isinstance(capability, bool) or not isinstance(capability, int)
                                       or capability < 1):
            raise RoutingConfigError(f"model {name!r} has invalid capability")
        models[name] = ModelRef(name, provider, executor, cli_model, gateway_mode,
                                reasoning_effort, review_reasoning_effort, capability)

    tiers: dict[str, tuple[str, ...]] = {}
    raw_tiers = ((raw.get("routing") or {}).get("tiers") or {})
    for complexity in ("simple", "medium", "complex"):
        candidates = ((raw_tiers.get(complexity) or {}).get("candidates") or [])
        if not isinstance(candidates, list) or not candidates:
            raise RoutingConfigError(f"routing tier {complexity!r} needs candidates")
        if len(candidates) != len(set(candidates)):
            raise RoutingConfigError(f"routing tier {complexity!r} repeats a candidate")
        unknown = [candidate for candidate in candidates if candidate not in models]
        if unknown:
            raise RoutingConfigError(f"routing tier {complexity!r} names unknown models: {unknown}")
        tiers[complexity] = tuple(candidates)

    runtime = raw.get("runtime") or {}
    reviewers = raw.get("reviewers") or {}
    if not isinstance(reviewers, dict):
        raise RoutingConfigError("reviewers must be a mapping")
    for coder_name, reviewer_name in reviewers.items():
        if coder_name not in models or reviewer_name not in models:
            raise RoutingConfigError(f"unknown reviewer pairing: {coder_name} -> {reviewer_name}")
        coder_tiers = {tier for tier, names in tiers.items() if coder_name in names}
        reviewer_tiers = {tier for tier, names in tiers.items() if reviewer_name in names}
        if not coder_tiers.intersection(reviewer_tiers):
            raise RoutingConfigError(f"reviewer pairing crosses complexity tiers: {coder_name} -> {reviewer_name}")
        if models[coder_name].provider == models[reviewer_name].provider:
            raise RoutingConfigError(f"reviewer pairing must cross providers: {coder_name} -> {reviewer_name}")
    if {model.provider for model in models.values()} == {"claude", "codex"}:
        routed_models = {name for candidates in tiers.values() for name in candidates}
        missing_pairs = sorted(routed_models.difference(reviewers))
        if missing_pairs:
            raise RoutingConfigError(f"missing cross-provider reviewer pairs: {missing_pairs}")
    role_floors = _role_floors(raw.get("roles"))
    _check_floor(models, "coder", role_floors, {name for names in tiers.values() for name in names})
    _check_floor(models, "reviewer", role_floors, set(reviewers.values()))
    planner = _planner_config(raw.get("planner"), models, tiers, role_floors)
    return RoutingConfig(
        models=models,
        tiers=tiers,
        role_floors=role_floors,
        planner=planner,
        reviewers=dict(reviewers),
        semantic_router_url=str(os.getenv("SEMANTIC_ROUTER_URL") or runtime.get("semantic_router_url") or "http://semantic-router:8080").rstrip("/"),
        agentgateway_url=str(os.getenv("AGENTGATEWAY_URL") or runtime.get("agentgateway_url") or "http://agentgateway:4000").rstrip("/"),
        router_timeout_seconds=float(runtime.get("router_timeout_seconds", 10)),
        failure_threshold=int((raw.get("health") or {}).get("failure_threshold", 2)),
    )


def _role_floors(raw: object) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RoutingConfigError("roles must be a mapping")
    floors = {}
    for role, value in raw.items():
        if role not in ROLES:
            raise RoutingConfigError(f"unknown routing role {role!r}")
        floor = (value or {}).get("min_capability") if isinstance(value, dict) else None
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 1:
            raise RoutingConfigError(f"role {role!r} needs an integer min_capability >= 1")
        floors[role] = floor
    return floors


def _check_floor(models: dict[str, ModelRef], role: str, floors: dict[str, int],
                 names: set[str]) -> None:
    floor = floors.get(role)
    if floor is None:
        return
    for name in sorted(names):
        capability = models[name].capability
        if capability is None:
            raise RoutingConfigError(f"{role} model {name!r} needs a capability for the {role} floor")
        if capability < floor:
            raise RoutingConfigError(
                f"{role} model {name!r} (capability {capability}) is below the {role} floor {floor}")


def _planner_config(raw: object, models: dict[str, ModelRef], coder_tiers: dict[str, tuple[str, ...]],
                    floors: dict[str, int]) -> PlannerConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RoutingConfigError("planner must be a mapping")
    raw_tiers = raw.get("tiers") or {}
    tiers: dict[str, tuple[str, ...]] = {}
    for complexity in PLANNER_TIERS:
        candidates = raw_tiers.get(complexity) or []
        if not isinstance(candidates, list) or not candidates:
            raise RoutingConfigError(f"planner tier {complexity!r} needs candidates")
        if len(candidates) != len(set(candidates)):
            raise RoutingConfigError(f"planner tier {complexity!r} repeats a candidate")
        unknown = [candidate for candidate in candidates if candidate not in models]
        if unknown:
            raise RoutingConfigError(f"planner tier {complexity!r} names unknown models: {unknown}")
        tiers[complexity] = tuple(candidates)
    planner_models = {name for names in tiers.values() for name in names}
    _check_floor(models, "planner", floors, planner_models)
    # Complexity is role-relative: even a low planner must out-rank every simple coder.
    simple_coders = [models[name].capability for name in coder_tiers["simple"]]
    low_planners = [models[name].capability for name in tiers["low"]]
    if None in simple_coders or None in low_planners:
        raise RoutingConfigError("planner low tier and coder simple tier models need capabilities")
    if min(low_planners) <= max(simple_coders):
        raise RoutingConfigError(
            "planner low tier must be strictly more capable than every simple coder")
    efforts = raw.get("reasoning_effort") or {}
    if not isinstance(efforts, dict):
        raise RoutingConfigError("planner reasoning_effort must be a mapping")
    for complexity, effort in efforts.items():
        if complexity not in PLANNER_TIERS or effort not in _EFFORTS:
            raise RoutingConfigError(f"invalid planner reasoning_effort {complexity!r}: {effort!r}")
    max_tool_calls = raw.get("max_tool_calls", 60)
    if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 1:
        raise RoutingConfigError("planner max_tool_calls must be a positive integer")
    narrate = raw.get("narrate", True)
    if not isinstance(narrate, bool):
        raise RoutingConfigError("planner narrate must be true or false")
    return PlannerConfig(tiers, dict(efforts), max_tool_calls, narrate)

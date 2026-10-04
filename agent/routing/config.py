"""Validated, environment-aware model-routing configuration."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent.routing.models import ModelRef

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model-routing.yaml"
_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


class RoutingConfigError(ValueError):
    pass


def _expand(value: object) -> object:
    if isinstance(value, str):
        return _ENV.sub(lambda match: os.getenv(match.group(1), match.group(2) or ""), value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


@dataclass(frozen=True)
class RoutingConfig:
    models: dict[str, ModelRef]
    tiers: dict[str, tuple[str, ...]]
    semantic_router_url: str
    agentgateway_url: str
    reviewers: dict[str, str] | None = None
    router_timeout_seconds: float = 10.0
    failure_threshold: int = 2

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
        models[name] = ModelRef(name, provider, executor, cli_model, gateway_mode,
                                reasoning_effort)

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
    return RoutingConfig(
        models=models,
        tiers=tiers,
        reviewers=dict(reviewers),
        semantic_router_url=str(os.getenv("SEMANTIC_ROUTER_URL") or runtime.get("semantic_router_url") or "http://semantic-router:8080").rstrip("/"),
        agentgateway_url=str(os.getenv("AGENTGATEWAY_URL") or runtime.get("agentgateway_url") or "http://agentgateway:4000").rstrip("/"),
        router_timeout_seconds=float(runtime.get("router_timeout_seconds", 10)),
        failure_threshold=int((raw.get("health") or {}).get("failure_threshold", 2)),
    )

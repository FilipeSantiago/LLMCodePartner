"""Small client for vLLM Semantic Router's decision-only preview API."""

from typing import Any

import httpx

from agent.routing.models import RoutingRecommendation


class SemanticRouterError(RuntimeError):
    pass


class SemanticRouterClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def select(self, task_text: str, complexity: str) -> RoutingRecommendation:
        """Preview the tier entrypoint; no inference backend is called."""
        payload = {
            "model": f"codepartner/{complexity}",
            "messages": [{"role": "user", "content": task_text}],
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/v1/routing/preview", json=payload
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SemanticRouterError(f"semantic router preview failed: {exc}") from exc
        model = self._find(data, "selected_model", "selectedModel", "recommended_model", "model")
        if not isinstance(model, str) or not model:
            raise SemanticRouterError("semantic router preview returned no selected model")
        reason = self._find(data, "reason", "decision", "matched_decision")
        return RoutingRecommendation(model=model, reason=reason if isinstance(reason, str) else None)

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=min(self._timeout, 3)) as client:
                response = await client.get(f"{self._base_url}/health")
                return response.is_success
        except httpx.HTTPError:
            return False

    @classmethod
    def _find(cls, value: Any, *keys: str) -> Any:
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    return value[key]
            for nested in value.values():
                found = cls._find(nested, *keys)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = cls._find(nested, *keys)
                if found is not None:
                    return found
        return None

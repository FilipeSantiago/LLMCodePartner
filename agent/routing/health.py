"""In-memory, conservative executor health tracking."""

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class HealthKey:
    model: str
    executor: str
    gateway_used: bool


class ExecutorHealth:
    def __init__(self, failure_threshold: int = 2):
        self._threshold = failure_threshold
        self._recent: dict[HealthKey, deque[bool]] = defaultdict(lambda: deque(maxlen=failure_threshold))

    def available(self, model: str, executor: str, gateway_used: bool) -> bool:
        results = self._recent[HealthKey(model, executor, gateway_used)]
        return not (len(results) == self._threshold and not any(results))

    def record(self, model: str, executor: str, gateway_used: bool, success: bool) -> None:
        self._recent[HealthKey(model, executor, gateway_used)].append(success)

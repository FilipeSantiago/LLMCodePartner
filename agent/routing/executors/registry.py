"""Executor lookup; provider selection is driven by model registry data."""

from agent.routing.executors.claude import ClaudeCLIExecutor
from agent.routing.executors.codex import CodexCLIExecutor


class ExecutorRegistry:
    def __init__(self, executors: dict | None = None):
        self._executors = executors or {
            "claude-cli": ClaudeCLIExecutor(),
            "codex-cli": CodexCLIExecutor(),
        }

    def get(self, name: str):
        try:
            return self._executors[name]
        except KeyError as exc:
            raise ValueError(f"no CLI executor registered for {name!r}") from exc

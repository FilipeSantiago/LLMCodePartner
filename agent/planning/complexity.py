"""Effective Work Package complexity for planner routing."""

from agent.spec.models import WP_COMPLEXITIES, WorkPackage

# Catalogs written before WPs carried a complexity fall back to their hardest task.
_FROM_TASK = {"simple": "low", "medium": "medium", "complex": "high"}


def effective_wp_complexity(package: WorkPackage) -> tuple[str, str]:
    """Return (level, source), where source is "declared" or "derived"."""
    if package.complexity in WP_COMPLEXITIES:
        return package.complexity, "declared"
    ranks = {"low": 0, "medium": 1, "high": 2}
    derived = max((_FROM_TASK.get(task.complexity, "medium") for task in package.tasks
                   if task.status != "cancelled"), key=ranks.get, default="medium")
    return derived, "derived"

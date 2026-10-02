"""OpenSpec task routing and provider-specific execution."""

from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.implementation import ImplementationService
from agent.routing.service import ModelRoutingService
from agent.routing.task_execution import TaskExecutionService

__all__ = [
    "ImplementationService",
    "ModelRoutingService",
    "RoutingConfig",
    "TaskExecutionService",
    "load_routing_config",
]

"""Builder lookup by role."""

from agent.prompt.base import PromptBuilder, PromptRole
from agent.prompt.coder import CoderPromptBuilder
from agent.prompt.planner import PlannerPromptBuilder
from agent.prompt.reviewer import ReviewerPromptBuilder
from agent.prompt.subagents import ChecksPromptBuilder, ReaderPromptBuilder

# A subagent request's `purpose` names the parent run kind that delegates.
_PURPOSE_ROLES = {"planner": PromptRole.PLANNER, "task": PromptRole.CODER,
                  "reviewer": PromptRole.REVIEWER}


class PromptFactory:
    _builders: dict[PromptRole, type[PromptBuilder]] = {
        PromptRole.PLANNER: PlannerPromptBuilder,
        PromptRole.CODER: CoderPromptBuilder,
        PromptRole.REVIEWER: ReviewerPromptBuilder,
        PromptRole.READER: ReaderPromptBuilder,
        PromptRole.CHECKS: ChecksPromptBuilder,
    }

    @classmethod
    def create(cls, role: PromptRole | str) -> PromptBuilder:
        """A new builder for `role`; builders are never shared between prompts."""
        try:
            return cls._builders[PromptRole(role)]()
        except (KeyError, ValueError) as exc:
            raise ValueError(f"no prompt builder registered for {role!r}") from exc

    @classmethod
    def for_purpose(cls, purpose: str) -> PromptBuilder:
        """The builder of the parent role behind a subagent request's purpose."""
        try:
            return cls.create(_PURPOSE_ROLES[purpose])
        except KeyError as exc:
            raise ValueError(f"no prompt builder for subagent purpose {purpose!r}") from exc

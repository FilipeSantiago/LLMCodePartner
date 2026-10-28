"""Agent prompts: one builder per role, created by `PromptFactory`.

    PromptFactory.create(PromptRole.CODER).with_context(CoderPromptContext(...)).build()
"""

from agent.prompt.base import Prompt, PromptBuilder, PromptRole
from agent.prompt.coder import (CoderPromptBuilder, CoderPromptContext,
                                InlineCoderPromptBuilder, InlineCoderPromptContext)
from agent.prompt.factory import PromptFactory
from agent.prompt.planner import PlannerPromptBuilder, PlannerPromptContext
from agent.prompt.reviewer import ReviewerPromptBuilder, ReviewerPromptContext
from agent.prompt.subagents import (ChecksPromptBuilder, ChecksPromptContext,
                                    ReaderPromptBuilder, ReaderPromptContext)

__all__ = [
    "ChecksPromptBuilder", "ChecksPromptContext", "CoderPromptBuilder", "CoderPromptContext",
    "InlineCoderPromptBuilder", "InlineCoderPromptContext",
    "PlannerPromptBuilder", "PlannerPromptContext", "Prompt", "PromptBuilder",
    "PromptFactory", "PromptRole", "ReaderPromptBuilder", "ReaderPromptContext",
    "ReviewerPromptBuilder", "ReviewerPromptContext",
]

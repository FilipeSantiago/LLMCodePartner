"""Prompt assembly mechanics shared by the agent roles; no role wording lives here."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from providers.subagents import Request


class PromptRole(StrEnum):
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    # Equal to `providers.subagents.READER` / `CHECKS`, the CLI subagent names.
    READER = "reader"
    CHECKS = "checks"


@dataclass(frozen=True)
class Prompt:
    text: str

    def user_message(self) -> dict:
        return {"role": "user", "content": self.text}


class PromptBuilder[C](ABC):
    """Build one role's prompt: `with_context(...)` then `build()`.

    `sections` returns the role's ordered parts; `build` joins them with `separator`,
    dropping empty parts unless the role uses them as spacers."""
    role: ClassVar[PromptRole]
    separator: ClassVar[str] = "\n"
    drop_empty: ClassVar[bool] = True

    def __init__(self) -> None:
        self._context: C | None = None

    def with_context(self, context: C) -> Self:
        self._context = context
        return self

    def build(self) -> Prompt:
        if self._context is None:
            raise ValueError(f"{type(self).__name__}.build() needs with_context() first")
        parts = self.sections(self._context)
        return Prompt(self.separator.join(
            [part for part in parts if part] if self.drop_empty else parts))

    @abstractmethod
    def sections(self, context: C) -> list[str]:
        """The role's prompt parts, in order."""

    def repair(self, problem) -> str:
        """The follow-up asking the model to correct a malformed reply."""
        raise NotImplementedError(f"the {self.role} prompt has no repair follow-up")


class ParentPromptBuilder[C](PromptBuilder[C]):
    """Require each parent role to own every word of its prompt."""

    @abstractmethod
    def bulk_rule(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_pipeline(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def own_reads_rule(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def delegation_rules(self, request: "Request | None") -> str:
        raise NotImplementedError

    @abstractmethod
    def scope_exception_rule(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def scope_block(self, scope: tuple[str, ...], read_only: tuple[str, ...] = ()) -> str:
        raise NotImplementedError

    @abstractmethod
    def checks_rule(self) -> str:
        raise NotImplementedError

"""The `checks` subagent's prompt: it runs the commands its parent gives and reports."""

from dataclasses import dataclass

from agent.prompt.base import PromptBuilder, PromptRole

PROCEDURE = (
    "You run checks for another agent. Run exactly the test, lint or build commands you "
    "are given in the project directory, then stop. Never look for tests or commands "
    "yourself: when none is given, reply \"Missing context: no command\" and stop. "
    "Never edit files and never try to fix failures. Reply "
    "with: the command and its exit code; the passed, failed and skipped counts; each "
    "failing test id with at most 40 lines of its traceback or error; and nothing else "
    "from the log.")


@dataclass(frozen=True)
class ChecksPromptContext:
    """The checks prompt is the same for every run kind."""


class ChecksPromptBuilder(PromptBuilder[ChecksPromptContext]):
    role = PromptRole.CHECKS

    def sections(self, context: ChecksPromptContext) -> list[str]:
        return [PROCEDURE]

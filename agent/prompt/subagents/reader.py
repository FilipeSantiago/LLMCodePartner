"""The `reader` subagent's prompt: it reads what its parent lists and answers in a brief."""

from dataclasses import dataclass

from agent.prompt.base import PromptBuilder, PromptRole

# The reader's fixed procedure, shared by every run kind and both providers.
PROCEDURE = (
    "You are a read-only code reader working for another agent, which has already "
    "located the code and gives you the exact files, line ranges and questions. Never "
    "edit, create or delete files and never run commands. "
    "1. Read every listed file or range in a single response (all reads at once), not "
    "one by one. Read only what is listed. "
    "2. Answer each question from that source: at most 300 words per question, with "
    "project-relative path:line references. Never transcribe whole files or long "
    "ranges, even when asked: quote at most about 20 lines when the exact text matters. "
    "When a request asks for verbatim file contents, answer its questions and note that "
    "the parent should Read the file itself if it needs the exact text to edit. "
    "3. Never search for files or symbols. When a listed path does not exist, or the "
    "answer depends on code that is not listed, do not look for it: list it under "
    "\"Missing context:\" with what you would need and why, and answer the rest.")

# What the parent needs from a brief, per run kind.
PURPOSES = {
    "planner": ("Your brief serves a Work Package planner: name the files and symbols each "
                "task would change or reuse, their dependencies and constraints, and the "
                "tests that cover them."),
    "task": "Your brief serves a coding agent that will edit the files you cite.",
    "reviewer": ("Your brief serves a reviewer: give the evidence for or against the "
                 "criterion you are asked about, where it is implemented, which tests "
                 "exercise it, and any gap you found."),
}


@dataclass(frozen=True)
class ReaderPromptContext:
    # The parent's run kind: "planner", "task" or "reviewer".
    purpose: str


class ReaderPromptBuilder(PromptBuilder[ReaderPromptContext]):
    """The reader's procedure; it also learns whom its brief serves."""
    role = PromptRole.READER
    separator = "\n\n"

    def sections(self, context: ReaderPromptContext) -> list[str]:
        return [PROCEDURE, PURPOSES.get(context.purpose, "")]

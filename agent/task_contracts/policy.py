"""Deterministic execution profiles selected by task complexity."""

from dataclasses import dataclass
from pathlib import PurePosixPath

from agent.task_contracts.models import ExecutionLimits
from mcp_bridge.registry import (
    OP_FILE_CREATE, OP_FILE_PATCH, OP_FILE_READ, OP_FILE_REPLACE,
    OP_FILE_SEARCH, OP_SHELL_EXECUTE, OP_SYMBOL_RENAME,
)


@dataclass(frozen=True)
class ComplexityPolicy:
    limits: ExecutionLimits
    operations: tuple[str, ...]


# Content search and project inspection let the Coder find what the deterministic
# Tasker could not predict; secrets and excluded files are filtered by the guard.
OP_FILE_TEXT_SEARCH, OP_PROJECT_INSPECT = "file_text_search", "project_inspect"
_COMMON = (OP_FILE_READ, OP_FILE_SEARCH, OP_FILE_TEXT_SEARCH, OP_PROJECT_INSPECT,
           OP_FILE_CREATE, OP_FILE_REPLACE, OP_FILE_PATCH)
POLICIES = {
    "simple": ComplexityPolicy(
        ExecutionLimits(repeated_calls=2),
        (*_COMMON, OP_SHELL_EXECUTE)),
    "medium": ComplexityPolicy(
        ExecutionLimits(repeated_calls=3),
        (*_COMMON, OP_SYMBOL_RENAME, OP_SHELL_EXECUTE)),
    "complex": ComplexityPolicy(
        ExecutionLimits(repeated_calls=4),
        (*_COMMON, OP_SYMBOL_RENAME, OP_SHELL_EXECUTE)),
}

# Read-only discovery contracts for tasks that investigate rather than edit.
DISCOVERY_OPERATIONS = (OP_FILE_READ, OP_FILE_SEARCH, OP_FILE_TEXT_SEARCH,
                        OP_PROJECT_INSPECT)
# OpenSpec preferred capabilities that mean "investigate, do not edit".
DISCOVERY_CAPABILITIES = frozenset({
    "read", "research", "inspect", "inspection", "analysis", "review", "verification"})
# Verification may still run a focused pytest check through the guard.
VERIFICATION_CAPABILITIES = frozenset({"verification", "test"})

# Tasker scope negotiation: Coder requests are granted only within these rules.
PROTECTED_PREFIXES = (".codepartner/", "openspec/", ".git/")
MAX_GRANTED_FILES_PER_REVISION = 5

MAX_EXECUTION_ATTEMPTS = 4
MAX_REVIEW_FAILURES = 2


def policy_for(complexity: str) -> ComplexityPolicy:
    return POLICIES.get(complexity, POLICIES["medium"])


def path_excluded(path: str, excluded_files: tuple[str, ...]) -> bool:
    """Whether a project path is, or lies under the name of, a user-excluded file."""
    candidate = str(PurePosixPath(path))
    return any(candidate == excluded or candidate.endswith("/" + excluded)
               for excluded in excluded_files)


def secret_path(path: str) -> bool:
    """Environment files that may hold credentials; `.env.example` is documentation."""
    name = PurePosixPath(path).name
    return name.startswith(".env") and name != ".env.example"


def protected_path(path: str) -> bool:
    """Paths the Coder never writes: tracking artifacts, VCS data, and secrets files."""
    candidate = str(PurePosixPath(path))
    return candidate.startswith(PROTECTED_PREFIXES) or secret_path(candidate)

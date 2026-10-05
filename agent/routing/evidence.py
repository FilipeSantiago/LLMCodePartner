"""Task-scoped verification evidence stored outside the target repository."""

import hashlib
import json
import os
import re
import shlex
import tempfile
import time
from pathlib import Path

from mcp_bridge.adapters import prepare_dispatcher_terminal


_VERIFICATION_COMMAND = re.compile(
    r"(?:^|\s)(?:pytest|py\.test|unittest|test|execute_run_configuration|"
    r"run_configuration|execute_terminal_command)(?:\s|$)",
    re.IGNORECASE,
)
_REPRODUCIBLE_CHECK = re.compile(
    r"(?:pytest|py\.test|unittest|py_compile|compileall|"
    r"execute_run_configuration|run_configuration)", re.IGNORECASE)
_SHELL_CONTROL = re.compile(r"(?:;|&&|\|\||`|\$\(|[<>])")


_SHELL_TEST = re.compile(
    r"^(?:uv\s+run\s+|\S*python3?\s+-m\s+)?(?:pytest|py\.test)(?:\s|$)", re.IGNORECASE)
TERMINAL_TOOL = "execute_terminal_command"
# Stay under CodePartner's own MCP tool timeout (120 s) so the IDE reports first.
TERMINAL_TIMEOUT_MS = 110_000


def shell_test_command(command: object) -> str | None:
    """Return the shell test command inside any form a coder sends to ``execute_tool``.

    Accepts a bare ``pytest ...``, ``execute_terminal_command pytest ...`` and the
    well-formed ``execute_terminal_command --command '...'``.  Commands with shell
    control characters are never extracted.
    """
    if not isinstance(command, str) or _SHELL_CONTROL.search(command):
        return None
    text = command.strip()
    if text.startswith(TERMINAL_TOOL):
        rest = text[len(TERMINAL_TOOL):].strip()
        if rest.startswith("--"):
            try:
                tokens = shlex.split(rest)
            except ValueError:
                return None
            text = next((tokens[index + 1] for index, token in enumerate(tokens[:-1])
                         if token == "--command"), "")
        else:
            text = rest
    return text if _SHELL_TEST.match(text) else None


def normalize_dispatcher_call(arguments: dict) -> tuple[dict, str | None]:
    """Rewrite a malformed ``execute_tool`` test call into the dispatcher syntax.

    Returns the arguments to send and the original command when it was rewritten.
    """
    command = arguments.get("command") if isinstance(arguments, dict) else None
    shell = shell_test_command(command)
    if (shell is None or command.strip().startswith(TERMINAL_TOOL + " --")
            or not is_reproducible_verification_call({"command": shell})):
        return arguments, None
    return {**arguments, **prepare_dispatcher_terminal(shell, TERMINAL_TIMEOUT_MS)}, command


def has_passing_test_run(evidence: list[dict]) -> bool:
    """True when captured evidence includes a test run that passed tests and failed none."""
    for item in evidence:
        totals = ((item.get("metadata") or {}).get("pytest") or {}).get("totals") or {}
        if (item.get("success") and totals.get("passed", 0) > 0
                and not totals.get("failed") and not totals.get("error")):
            return True
    return False


def is_verification_call(arguments: dict) -> bool:
    command = arguments.get("command") if isinstance(arguments, dict) else None
    return isinstance(command, str) and bool(_VERIFICATION_COMMAND.search(command))


def is_reproducible_verification_call(arguments: dict) -> bool:
    """Accept replayable test/compile calls, never a generic shell command."""
    command = arguments.get("command") if isinstance(arguments, dict) else None
    return (isinstance(command, str)
            and not _SHELL_CONTROL.search(command)
            and bool(_REPRODUCIBLE_CHECK.search(command)))


def verification_succeeded(result: object) -> bool:
    if isinstance(result, dict):
        for key in ("exit_code", "exitCode", "returncode"):
            value = result.get(key)
            if isinstance(value, int):
                return value == 0
        if result.get("isError") or result.get("error"):
            return False
    text = result if isinstance(result, str) else json.dumps(result)
    lower = text.lstrip().lower()
    if lower.startswith("error"):
        return False
    exit_match = re.search(r"(?:exit[_ ]?code|returncode)\D{0,8}(-?\d+)", text, re.I)
    if exit_match:
        return int(exit_match.group(1)) == 0
    return not bool(re.search(r"(?:^|\s)(?:failed|errors?)\s*[:=]\s*[1-9]\d*", text, re.I))


def pytest_inventory(text: str) -> dict:
    collected_match = re.search(r"collected\s+(\d+)\s+items?", text, re.I)
    totals = {}
    for count, label in re.findall(
            r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?|deselected)",
            text, re.I):
        totals[label.lower().rstrip("s") if label.lower() == "errors" else label.lower()] = int(count)
    node_ids = []
    for line in text.splitlines():
        candidate = line.strip().split(" ", 1)[0]
        if "::" in candidate and candidate not in node_ids:
            node_ids.append(candidate)
    return {
        "collected": int(collected_match.group(1)) if collected_match else None,
        "totals": totals,
        "node_ids": node_ids,
    }


def evidence_root() -> Path:
    configured = os.getenv("CODEPARTNER_EVIDENCE_DIR")
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "codepartner-evidence"


def store_execution_payload(job_id: str, position: int, evidence_id: int,
                            tool_name: str, arguments: dict, result: object) -> dict:
    """Persist raw execution output outside the repository and return its manifest."""
    directory = evidence_root() / job_id / str(position)
    directory.mkdir(parents=True, exist_ok=True)
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = directory / f"execution-{evidence_id}.json"
    payload = {
        "evidence_id": evidence_id,
        "tool": tool_name,
        "arguments": arguments,
        "result": text,
        "sha256": digest,
        "captured_at": time.time(),
        "pytest": pytest_inventory(text),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path), "sha256": digest, "result_excerpt": text[:4000],
            "metadata": {"pytest": payload["pytest"]}}

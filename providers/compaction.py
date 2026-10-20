"""Settings for the CLIs' own automatic context compaction.

CodePartner does not summarize or compress agent history itself. It only passes the
native knobs per invocation (never through the user's global CLI config):
  - Claude Code: the `autoCompactWindow` setting via `--settings`, plus a `PreCompact`
    command hook whose stdout becomes the native summary's custom instructions.
  - Codex CLI: `-c model_auto_compact_token_limit=N`.
Both CLIs cap the threshold at the model's context window (docs/native-execution.md).
"""

import json
import logging
import os
import shlex

log = logging.getLogger("codepartner.providers.compaction")

ENV = "CODEPARTNER_AUTO_COMPACT_TOKENS"
DEFAULT_TOKENS = 250_000
# Claude Code accepts autoCompactWindow values in this range.
MIN_TOKENS, MAX_TOKENS = 100_000, 1_000_000


def auto_compact_tokens() -> int | None:
    """The requested compaction threshold, or None to leave the CLI default."""
    raw = os.getenv(ENV, "").strip().lower()
    if not raw:
        return DEFAULT_TOKENS
    if raw in {"0", "off", "false", "none"}:
        return None
    try:
        value = int(raw.replace("_", ""))
    except ValueError:
        value = -1
    if not MIN_TOKENS <= value <= MAX_TOKENS:
        log.warning("%s=%r is outside %d..%d; using %d", ENV, raw, MIN_TOKENS,
                    MAX_TOKENS, DEFAULT_TOKENS)
        return DEFAULT_TOKENS
    return value


def compaction_brief(task, contract=None) -> str:
    """What a native compaction summary must keep for this task."""
    criteria = task.acceptance_criteria or [task.description]
    if isinstance(criteria, str):
        criteria = [criteria]
    goal = getattr(contract, "goal", "") or task.description
    excluded = tuple(getattr(contract, "excluded_files", ()) or ())
    return " ".join(filter(None, [
        f"You are implementing OpenSpec task {task.id}: {task.title}. Goal: {goal}.",
        "Acceptance criteria: " + "; ".join(str(item) for item in criteria) + ".",
        f"Never modify: {', '.join(excluded)}." if excluded else "",
        "In the summary keep: this objective and its acceptance criteria; the "
        "constraints and do-not-modify areas from the original task prompt; the "
        "implementation decisions made; every file modified so far; the outstanding "
        "steps; the important validation results and any unresolved errors; and the "
        "required final JSON handoff. The CodePartner task contract and its execution "
        "records remain authoritative after compaction.",
    ]))


def claude_settings(brief: str | None = None) -> str | None:
    """`--settings` JSON for one Claude Code run, or None when nothing is set."""
    tokens = auto_compact_tokens()
    settings: dict = {}
    if tokens is not None:
        settings["autoCompactWindow"] = tokens
    if brief:
        settings["hooks"] = {"PreCompact": [{"hooks": [
            {"type": "command", "command": "printf '%s' " + shlex.quote(brief)}]}]}
    return json.dumps(settings) if settings else None


def codex_args() -> list[str]:
    tokens = auto_compact_tokens()
    return ["-c", f"model_auto_compact_token_limit={tokens}"] if tokens is not None else []


def requested(mechanism: str) -> dict:
    """Run metadata: what was requested (observed compactions are recorded separately)."""
    tokens = auto_compact_tokens()
    return {"requested_auto_compact_tokens": tokens,
            "mechanism": mechanism if tokens is not None else "cli default"}

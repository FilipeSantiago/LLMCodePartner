"""Native subagents and delegation rules for native parent agents.

A native run keeps every tool result in the CLI-owned transcript and re-reads it on
every later turn. Reading source is therefore delegated to a `reader` subagent on a
cheap model whose transcript stays outside the parent's context, and long test runs
to a `checks` subagent.

A subagent is an executor, not an investigator: the parent locates the code itself
(Codebase Memory, Glob/Grep) and hands over the full context, the exact paths and
ranges for a reader or the exact commands for checks. The subagent uses one basic
tool (Read or Bash), never searches and never consults the KB; when its request lacks
something it reports it under "Missing context:" instead of looking for it.

CodePartner does not run subagents itself. It defines them per invocation for the
CLI's own mechanism (never through the user's global CLI config):
  - Claude Code: `ClaudeAgentOptions.agents`, started through the built-in `Agent` tool;
    each role gets only its one tool.
  - Codex CLI: `[agents.<role>]` roles passed with `-c`, started with `spawn_agent`.
    Its shell stays, so the rules are prompt-only there.

Nothing here intercepts or refuses a call: shell searches and reads are classified for
telemetry only. Off unless CODEPARTNER_READER_SUBAGENT=true (docs/native-execution.md).
"""

import os
import re
import shlex
from dataclasses import dataclass, field

from agent.paths import paths_in

ENV = "CODEPARTNER_READER_SUBAGENT"
TURNS_ENV = "CODEPARTNER_SUBAGENT_MAX_TURNS"
DEFAULT_TURNS = 5

READER, CHECKS = "reader", "checks"
# Claude's cheap tier when the routing config names no reader model.
DEFAULT_CLAUDE_MODEL = "haiku"

DESCRIPTIONS = {
    READER: ("Reads the exact files and line ranges you list and answers your questions "
             "about them in a compact brief with path:line references. It does not "
             "search: give it every path, range and question it needs."),
    CHECKS: ("Runs the exact test, lint or build commands you give it and returns only "
             "the result summary and the failures, not the full log. It does not look "
             "for tests: give it the commands."),
}

# project/relative/path.ext:12 or :12-30, as briefs cite them.
_CITATION = re.compile(r"([\w./-]+\.\w+):(\d+)(?:-(\d+))?")
# A path-like token: has a directory separator or a file extension.
_PATHISH = re.compile(r"^[\w./-]*[\w-]\.\w+$|^[\w.-]+/[\w./-]+$")
# How the Claude CLI answers an Agent call it started in the background.
ASYNC_LAUNCH = "Async agent launched"
_MISSING = re.compile(r"missing context\s*:", re.I)
# A reply shorter than this with no path:line citation is no brief at all.
_EMPTY_BRIEF_CHARS = 400


@dataclass(frozen=True)
class Request:
    """The subagents a run asks for. Built by the call site, resolved by the provider.

    `models` maps a provider name to the reader's (cli model, reasoning effort): the
    request is built before routing settles the provider, and a fallback attempt may
    change it."""
    roles: tuple[str, ...] = (READER, CHECKS)
    purpose: str = "task"
    scope: tuple[str, ...] = ()
    # Planner-assigned files this task must not modify: read-only context.
    read_only: tuple[str, ...] = ()
    models: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    max_turns: int | None = None

    def model_for(self, provider: str) -> tuple[str | None, str | None]:
        return self.models.get(provider, (None, None))

    def turns(self) -> int:
        override = os.getenv(TURNS_ENV, "").strip()
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        return self.max_turns or DEFAULT_TURNS


@dataclass(frozen=True)
class Subagent:
    name: str
    description: str
    prompt: str


def enabled() -> bool:
    return os.getenv(ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def max_turns() -> int:
    return Request().turns()


def request(purpose: str, roles: tuple[str, ...] = (READER, CHECKS), scope=(),
            models: dict | None = None, max_turns: int | None = None,
            read_only=()) -> Request | None:
    """The run's subagent request, or None when the feature is off. The reader follows
    its own KB-first procedure, so no parent guidance text is passed in."""
    if not enabled():
        return None
    return Request(roles=tuple(roles), purpose=purpose,
                   scope=tuple(dict.fromkeys(str(path) for path in scope if path)),
                   read_only=tuple(dict.fromkeys(str(path) for path in read_only if path)),
                   models=dict(models or {}), max_turns=max_turns)


def resolve(req: Request | None, native) -> list[Subagent]:
    """The subagents a native run gets. Read-only runs never get `checks`: running
    commands is not theirs to do."""
    if req is None or native is None:
        return []
    roles = [role for role in req.roles if role in DESCRIPTIONS
             and not (role == CHECKS and native.read_only)]
    return [Subagent(role, DESCRIPTIONS[role], _prompt(role, req)) for role in roles]


def _prompt(role: str, req: Request) -> str:
    """The role's procedure; a reader also learns whom its brief serves. The scope is
    the parent's to respect: a subagent reads or runs only what it is given."""
    # Imported here: agent.prompt reaches back into `providers` through agent.spec.
    from agent.prompt import ChecksPromptContext, PromptFactory, ReaderPromptContext
    context = ReaderPromptContext(req.purpose) if role == READER else ChecksPromptContext()
    return PromptFactory.create(role).with_context(context).build().text


def exploration_rules(req: Request | None) -> str:
    """The parent's source delegation rules, owned by the parent role's prompt builder;
    empty when no subagents were asked for."""
    if req is None:
        return ""
    from agent.prompt import PromptFactory
    return PromptFactory.for_purpose(req.purpose).delegation_rules(req)


# --- progress lines (shown in the chat; never part of a prompt or report) -----------

def _request_paths(text: str) -> list[str]:
    return list(dict.fromkeys(token.strip("`'\",:()") for token in text.split()
                              if _PATHISH.match(token.strip("`'\",:()"))))


def describe_spawn(role: str | None, prompt: object, command: object = None) -> str:
    """"Reader: <objective> — `a`, `b`" for a subagent request, from its own text."""
    text = prompt if isinstance(prompt, str) else ""
    if role == CHECKS:
        quoted = re.search(r"`([^`]+)`", text)
        shown = command or (quoted.group(1) if quoted else next(
            (line for line in text.splitlines() if line.strip()), ""))
        shown = " ".join(str(shown).split())
        return f"🧪 Checks: `{shown[:70]}{'…' if len(shown) > 70 else ''}`"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    objective = lines[0] if lines else "investigation"
    objective = objective[:80] + ("…" if len(objective) > 80 else "")
    paths = _request_paths(text)
    files = ", ".join(f"`{path}`" for path in paths[:6])
    if len(paths) > 6:
        files += f" +{len(paths) - 6} more"
    return f"🧭 Reader: {objective}" + (f" — {files}" if files else "")


def describe_brief(role: str | None, text: str) -> str:
    """"Reader brief: N files cited (~W words)" for a subagent's reply."""
    if role == CHECKS:
        return "🧪 Checks done"
    cited = {_normal(match.group(1)) for match in _CITATION.finditer(text)}
    words = len(text.split())
    line = f"📋 Reader brief: {len(cited)} file{'s' if len(cited) != 1 else ''} cited (~{words} words)"
    if is_empty_brief(text):
        return "⚠️ Reader returned no brief (likely its turn limit)"
    if _MISSING.search(text):
        line += " · missing context reported"
    return line


def is_empty_brief(text: str) -> bool:
    """A reader reply that is no brief: short and without any path:line citation."""
    return len(text.strip()) < _EMPTY_BRIEF_CHARS and not _CITATION.search(text)


# --- shell classification (telemetry only) -------------------------------------------

_EXPLORE_COMMANDS = frozenset({"rg", "grep", "egrep", "fgrep", "ag", "ack", "find", "fd",
                               "fdfind", "tree", "locate"})
_READ_COMMANDS = frozenset({"cat", "sed", "nl", "head", "tail", "less", "more", "bat"})
_SEGMENTS = re.compile(r"\|\||&&|[;|\n]")


def classify_command(command: object) -> tuple[str, list[tuple[str, bool]]]:
    """("exploration" | "exact_read" | "other", [(path, ranged), ...]) for one shell
    command. Observation only: nothing is refused on the basis of this."""
    if not isinstance(command, str) or not command.strip():
        return "other", []
    command = _unwrap_shell(command)
    segments = [segment.strip() for segment in _SEGMENTS.split(command)]
    piped_range = bool(re.search(r"\|\s*(head|tail|sed\s+-n)\b", command))
    kind, reads = "other", []
    for segment in segments:
        try:
            words = shlex.split(segment)
        except ValueError:
            words = segment.split()
        while words and ("=" in words[0] and not words[0].startswith("-")):
            words = words[1:]  # VAR=value prefixes
        if not words:
            continue
        name = os.path.basename(words[0])
        if (name in _EXPLORE_COMMANDS
                or words[:2] in (["git", "grep"], ["git", "ls-files"])
                or (name == "ls" and any(w.startswith("-") and "R" in w for w in words))):
            kind = "exploration"
            continue
        if name not in _READ_COMMANDS:
            continue
        paths = [w for w in words[1:] if not w.startswith("-") and _PATHISH.match(w)
                 and not re.fullmatch(r"\d+(,\d+)?p", w)]
        if not paths:
            continue
        ranged = (name in {"head", "tail"} or piped_range
                  or (name == "sed" and "-n" in words))
        reads += [(path, ranged) for path in paths]
    if kind != "exploration" and reads:
        kind = "exact_read"
    return kind, reads


SCOPE_EXCEPTION = "SCOPE EXCEPTION:"


def note_scope(metadata: dict, metrics: dict, paths=(), text: str = "") -> None:
    """Count a parent's source reads outside the plan's files (`scope_files` in the run
    metadata) and the scope exceptions it stated. Telemetry only: nothing is refused."""
    if text and SCOPE_EXCEPTION in text:
        metrics["scope_exceptions"] = metrics.get("scope_exceptions", 0) + text.count(
            SCOPE_EXCEPTION)
    scope = metadata.get("scope_files")
    if not scope:
        return
    allowed = {_normal(path) for path in scope}
    outside = metrics.setdefault("out_of_scope_reads", [])
    for path in paths:
        if isinstance(path, str) and path and _normal(path) not in allowed \
                and path not in outside:
            outside.append(path)


def _unwrap_shell(command: str) -> str:
    """The script of `bash -lc '...'` (how Codex reports its shell commands)."""
    try:
        words = shlex.split(command)
    except ValueError:
        return command
    if (len(words) >= 3 and os.path.basename(words[0]) in {"bash", "sh", "zsh", "dash"}
            and words[1] in {"-c", "-lc", "-cl"}):
        return words[2]
    return command


# --- telemetry ----------------------------------------------------------------------

def _counters() -> dict:
    return {"mode": "off", "direct_reads": 0, "ranged_reads": 0, "whole_file_reads": 0,
            "read_turns": 0, "batched_read_turns": 0, "reads_in_batches": 0,
            "exact_rereads": 0,
            "bash_exploration": 0, "bash_exact_reads": 0,
            "brief_file_reads": 0, "brief_full_rereads": 0,
            "whole_reads_of_edited": 0, "post_edit_rereads": 0}


def _subagent_counters() -> dict:
    return {"calls": 0, "roles": {}, "reader_calls": 0, "checks_calls": 0,
            "background_calls": 0, "foreground_calls": 0,
            "background_requests": 0, "descriptions": [],
            "objectives_paths": 0, "out_of_scope_paths": 0, "requests_without_paths": 0,
            "missing_context_reported": 0, "empty_briefs": 0, "async_launches": 0,
            "child_tool_calls": 0, "child_kb_calls": 0,
            "child_tools": {}, "child_bash_exploration": 0, "brief_bytes": 0,
            "cited_files": 0, "child_usage": None}


class Tracker:
    """Exploration telemetry for one native run, in `native_tool_metrics`.

    `exploration` is recorded for every native run (flag on or off) so A/B comparisons
    see the parent's direct reads, rereads and shell searches either way; `subagents`
    only when subagents were offered. `brief_file_reads` counts the parent's own reads
    of files a brief already cited and `brief_full_rereads` those without a range: a
    high share means the briefs are too thin to replace reading. `whole_reads_of_edited`
    counts files the parent read whole and later edited (a range would have done);
    `post_edit_rereads` its reads of files it had already edited; `out_of_scope_paths`
    reader-request paths outside the planner-assigned files."""

    def __init__(self, metrics: dict, req: Request | None = None,
                 offered: bool = False, workspace: str | None = None, native=None):
        self.workspace = workspace or getattr(native, "workspace", None)
        self.native = native
        self.reads = metrics.setdefault("exploration", _counters())
        self.reads["mode"] = "reader" if offered else "off"
        self.data = metrics.setdefault("subagents", _subagent_counters()) if offered else None
        self.req = req
        self._cited: set[str] = set()
        self._seen_reads: set[tuple] = set()
        self._read_whole: set[str] = set()
        self._edited: set[str] = set()

    def _sub(self) -> dict:
        if self.data is None:
            self.data = _subagent_counters()
        return self.data

    def spawned(self, role: str | None, prompt: object = "", *,
                background: bool = False, requested_background: bool = False,
                description: str | None = None) -> None:
        data = self._sub()
        data["calls"] += 1
        if requested_background:
            data["background_requests"] = data.get("background_requests", 0) + 1
        key = "background_calls" if background else "foreground_calls"
        data[key] = data.get(key, 0) + 1
        if description:
            data.setdefault("descriptions", []).append(description)
        role = role or "unknown"
        data["roles"][role] = data["roles"].get(role, 0) + 1
        if role == READER:
            data["reader_calls"] += 1
        elif role == CHECKS:
            data["checks_calls"] += 1
        text = prompt if isinstance(prompt, str) else ""
        paths = {self._relative(path) for path in paths_in(text)}
        data["objectives_paths"] += len(paths)
        planned = {_normal(path) for path in (*self.req.scope, *self.req.read_only)} \
            if self.req is not None else set()
        if role == READER and planned:
            data["out_of_scope_paths"] += len(paths - planned)
        # The parent must hand over the full context: paths for a reader, a quoted
        # command for checks.
        if (not re.search(r"`[^`]+`", text)) if role == CHECKS else not paths:
            data["requests_without_paths"] += 1

    def child_tool(self, name: str, arguments: dict | None = None) -> None:
        data = self._sub()
        data["child_tool_calls"] += 1
        data["child_tools"][name] = data["child_tools"].get(name, 0) + 1
        if name.rsplit("__", 1)[-1].startswith("kb_"):
            data["child_kb_calls"] += 1
        if name == "Bash" and classify_command((arguments or {}).get("command"))[0] == \
                "exploration":
            data["child_bash_exploration"] += 1

    def child_usage(self, usage: object) -> None:
        """Usage the CLI reported for one subagent invocation (exact attribution)."""
        if not isinstance(usage, dict):
            return
        data = self._sub()
        total = data["child_usage"] or {}
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
        data["child_usage"] = total

    def brief(self, text: str, role: str | None = None) -> None:
        """A subagent's reply; `role` is None when the CLI does not report it."""
        data = self._sub()
        if text.lstrip().startswith(ASYNC_LAUNCH):
            # The CLI ran the subagent in the background: this is no brief, and the real
            # one arrives later as a notification that ends the parent's turn.
            data["async_launches"] = data.get("async_launches", 0) + 1
            return
        data["brief_bytes"] += len(text.encode())
        if _MISSING.search(text):
            data["missing_context_reported"] += 1
        if role != CHECKS and is_empty_brief(text):
            data["empty_briefs"] += 1
        for match in _CITATION.finditer(text):
            self._cited.add(self._relative(match.group(1)))
        data["cited_files"] = len(self._cited)

    def parent_reads(self, reads: list[tuple[object, object, object]]) -> None:
        """The parent's direct reads issued in one response: (path, offset, limit)."""
        counted = [read for read in reads if isinstance(read[0], str) and read[0]]
        if counted:
            self.reads["read_turns"] = self.reads.get("read_turns", 0) + 1
        if len(counted) >= 2:
            self.reads["batched_read_turns"] += 1
            self.reads["reads_in_batches"] += len(counted)
        for path, offset, limit in counted:
            self.parent_read(path, bool(offset or limit), key=(offset, limit))

    def parent_read(self, path: object, ranged: bool, workspace: str | None = None,
                    key: tuple = ()) -> None:
        if not isinstance(path, str) or not path:
            return
        path = self._relative(path, workspace)
        if path in self._edited:
            self.reads["post_edit_rereads"] += 1
        elif not ranged:
            self._read_whole.add(path)
        self.reads["direct_reads"] += 1
        self.reads["ranged_reads" if ranged else "whole_file_reads"] += 1
        signature = (path, ranged, *key)
        if signature in self._seen_reads:
            self.reads["exact_rereads"] += 1
        self._seen_reads.add(signature)
        if path in self._cited:
            self.reads["brief_file_reads"] += 1
            if not ranged:
                self.reads["brief_full_rereads"] += 1

    def parent_edit(self, path: object) -> None:
        """The parent's own Edit/Write of a project file."""
        if not isinstance(path, str) or not path:
            return
        path = self._relative(path)
        if path not in self._edited and path in self._read_whole:
            self.reads["whole_reads_of_edited"] += 1
        self._edited.add(path)

    def _relative(self, path: str, workspace: str | None = None) -> str:
        relative = self.native.relative(path) if self.native is not None else None
        if relative is not None:
            return relative
        workspace = workspace or self.workspace
        if os.path.isabs(path) and workspace:
            path = os.path.relpath(path, workspace)
        return _normal(path)

    def parent_command(self, command: object) -> None:
        kind, reads = classify_command(command)
        if kind == "exploration":
            self.reads["bash_exploration"] += 1
        elif kind == "exact_read":
            self.reads["bash_exact_reads"] += 1
            for path, ranged in reads:
                self.parent_read(path, ranged, key=(str(command),))


def cost_attribution(model_usage: object, reader_model: str | None,
                     child_usage: dict | None) -> dict | None:
    """Parent/subagent split of one Claude run. Tokens are exact when the CLI reported
    per-invocation usage; the cost split is always estimated from per-model usage, and
    flagged ambiguous when the reader shares the parent's model."""
    if not isinstance(model_usage, dict) or not model_usage:
        return None
    hint = (reader_model or "").lower()
    def is_reader(name: str) -> bool:
        return bool(hint) and hint in name.lower()
    parent = {"cost_usd": 0.0, "tokens": 0}
    subagents = {"cost_usd": 0.0, "tokens": 0}
    for name, usage in model_usage.items():
        if not isinstance(usage, dict):
            continue
        side = subagents if is_reader(name) else parent
        side["cost_usd"] += float(usage.get("costUSD") or 0)
        side["tokens"] += sum(int(usage.get(key) or 0) for key in (
            "inputTokens", "outputTokens", "cacheReadInputTokens",
            "cacheCreationInputTokens"))
    readers = [name for name in model_usage if is_reader(name)]
    return {"parent": parent, "subagents": subagents,
            "method": "estimated_by_model",
            "tokens_method": "exact" if child_usage else "estimated_by_model",
            "subagent_tokens_reported": child_usage,
            "ambiguous": len(readers) == len(model_usage) and bool(readers)}


def _normal(path: str) -> str:
    return os.path.normpath(path.removeprefix("./"))

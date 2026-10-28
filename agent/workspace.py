"""Registered project workspaces mounted into the container, and native execution plans.

A project is executed natively only from a registry entry. Paths supplied by a
model or a request are never accepted as execution roots.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

log = logging.getLogger("codepartner.workspace")

MODES = ("auto", "native", "ide")
# The filesystem is the main execution path; `auto` (IDE fallback) and `ide` are opt-ins.
DEFAULT_MODE = "native"
# Task capabilities that never write (matches the executors' `read_only` metadata).
READ_ONLY_CAPABILITIES = frozenset({
    "read", "research", "inspect", "inspection", "analysis", "review", "verification"})
REGISTER_HINT = "register it with scripts/register_project.py (docs/native-execution.md)"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "projects.yaml"
_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_.")


class WorkspaceError(ValueError):
    """A registry entry is missing, invalid or not usable for native execution."""


@dataclass(frozen=True)
class RegisteredProject:
    id: str
    workspace: str
    host_path: str | None = None
    execution: str = DEFAULT_MODE


def _workspace_root() -> Path:
    return Path(os.getenv("CODEPARTNER_WORKSPACE_ROOT", "/workspace"))


def _parse(raw: Any) -> dict[str, RegisteredProject]:
    # Reuse the routing config's ${VAR:-default} expansion.
    from agent.routing.config import _expand
    raw = _expand(raw or {})
    if not isinstance(raw, dict):
        raise WorkspaceError("projects config must be a mapping")
    entries = raw.get("projects") or {}
    if not isinstance(entries, dict):
        raise WorkspaceError("projects must be a mapping of project id to settings")
    projects = {}
    for project_id, value in entries.items():
        if not isinstance(project_id, str) or not project_id or set(project_id) - _ID_CHARS:
            raise WorkspaceError(
                f"project id {project_id!r} must use lowercase letters, digits, '-', '_' or '.'")
        if not isinstance(value, dict):
            raise WorkspaceError(f"project {project_id!r} must be a mapping")
        workspace = value.get("workspace")
        if not isinstance(workspace, str) or not workspace.startswith("/"):
            raise WorkspaceError(f"project {project_id!r} needs an absolute workspace path")
        host_path = value.get("host_path")
        if host_path is not None and (not isinstance(host_path, str)
                                      or not host_path.startswith("/")):
            raise WorkspaceError(f"project {project_id!r} host_path must be absolute")
        execution = str(value.get("execution") or DEFAULT_MODE).lower()
        if execution not in MODES:
            raise WorkspaceError(
                f"project {project_id!r} execution must be one of {', '.join(MODES)}")
        projects[project_id] = RegisteredProject(
            project_id, os.path.normpath(workspace),
            os.path.normpath(host_path) if host_path else None, execution)
    return projects


class ProjectRegistry:
    """`config/projects.yaml`, re-read whenever the file changes."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("CODEPARTNER_PROJECTS_CONFIG", DEFAULT_CONFIG_PATH))
        self._stamp: tuple[int, int] | None = None
        self._projects: dict[str, RegisteredProject] = {}

    def projects(self) -> dict[str, RegisteredProject]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._stamp, self._projects = None, {}
            return {}
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp != self._stamp:
            try:
                raw = yaml.safe_load(self.path.read_text())
            except (OSError, yaml.YAMLError) as exc:
                raise WorkspaceError(f"could not read {self.path}: {exc}") from exc
            self._projects = _parse(raw)
            self._stamp = stamp
        return dict(self._projects)

    def get(self, project_id: str) -> RegisteredProject:
        project = self.projects().get(project_id)
        if project is None:
            raise WorkspaceError(f"project {project_id!r} is not registered in {self.path.name}")
        return project

    def resolve(self, project_id: str) -> RegisteredProject:
        """The registered project, validated as a usable mounted workspace."""
        project = self.get(project_id)
        validate_workspace(project)
        return project

    def match(self, project_name: str | None, project_path: str | None) -> str | None:
        """The registry id for an IDE project: `.codepartner/config.yaml` id first,
        then the IDE project path against `host_path`."""
        projects = self.projects()
        if project_name and project_name in projects:
            return project_name
        if project_path:
            path = os.path.normpath(project_path)
            return next((item.id for item in projects.values() if item.host_path == path), None)
        return None


def validate_workspace(project: RegisteredProject) -> Path:
    root = _workspace_root()
    path = Path(project.workspace)
    if path == root or root not in path.parents:
        raise WorkspaceError(
            f"project {project.id!r} workspace {path} must be inside {root}")
    if not path.is_dir():
        raise WorkspaceError(
            f"project {project.id!r} is registered but {path} is not mounted; add its bind "
            "mount and recreate the container (docker compose up -d codepartner)")
    if os.path.realpath(path) != str(path):
        raise WorkspaceError(f"project {project.id!r} workspace {path} must not be a symlink")
    if not os.path.ismount(path):
        raise WorkspaceError(
            f"project {project.id!r} workspace {path} exists but is not a bind mount; add its "
            "mount and recreate the container (docker compose up -d codepartner)")
    return path


_registry: ProjectRegistry | None = None


def registry() -> ProjectRegistry:
    global _registry
    if _registry is None or _registry.path != Path(
            os.getenv("CODEPARTNER_PROJECTS_CONFIG", DEFAULT_CONFIG_PATH)):
        _registry = ProjectRegistry()
    return _registry


# The registry id of the project the current request's conversation is bound to.
# Background runs inherit it from the request that started them.
_active: ContextVar[str | None] = ContextVar("codepartner_workspace", default=None)


def activate(project_id: str | None) -> None:
    _active.set(project_id)


def active_project_id() -> str | None:
    return _active.get()


# --- native execution ------------------------------------------------------


class NativeSessionError(RuntimeError):
    """The native run must be reported as failed (e.g. a read-only run wrote files)."""


@dataclass
class NativeRun:
    """Per-run native execution settings, attached to `Run.native`.

    `start` runs before the CLI starts; a failure there aborts the run before any
    write. `finish` runs after the CLI exits, even on failure or cancellation, and
    returns the project paths the run changed; it raises `NativeSessionError` when
    the run must fail. `warnings` are reported to the user without failing the run.

    `written` holds the project-relative paths the agent itself reported writing
    (Claude Edit/Write, Codex file_change). `agent_can_write` says whether the
    agent had any way to write; a read-only Claude run has only Read/Grep/Glob.
    """
    workspace: str
    read_only: bool = False
    # The project's root on the host, where Codebase Memory indexed it; KB paths are
    # translated between it and `workspace` (see `to_workspace`/`to_host`).
    host_path: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    start: Callable[[], Awaitable[None]] | None = None
    finish: Callable[[], Awaitable[list[str]]] | None = None
    written: set[str] = field(default_factory=set)
    agent_can_write: bool | None = None
    warnings: list[str] = field(default_factory=list)
    # The run's `native_tool_metrics` (tool counts, best-effort file paths).
    metrics: dict = field(default_factory=lambda: {"tool_calls": 0, "tools": {}, "files": []})
    # Called with (command, output, exit code or None) for every completed shell
    # command, so test runs become execution evidence.
    on_command: Callable[[str, str, int | None], Awaitable[None]] | None = None

    async def command_finished(self, command: object, output: object,
                               exit_code: int | None) -> None:
        if self.on_command is None or not isinstance(command, str) or not command:
            return
        try:
            await self.on_command(command, output if isinstance(output, str) else "",
                                  exit_code)
        except Exception:
            log.exception("could not record native command evidence: %s", command[:200])

    def relative(self, path: object) -> str | None:
        """The project-relative form of a path the agent named (relative, workspace or
        host absolute), or None when it is empty or outside the project."""
        if not isinstance(path, str) or not path:
            return None
        path = to_workspace(path, self)
        if os.path.isabs(path):
            path = os.path.relpath(os.path.normpath(path), self.workspace)
        path = os.path.normpath(path)
        return None if path == "." or path.startswith("..") else path

    def record_write(self, path: object) -> None:
        path = self.relative(path)
        if path is not None:
            self.written.add(path)


def _swap_root(text: str, old: str | None, new: str | None) -> str:
    """Replace the root `old` with `new` wherever it starts a path: followed by `/` or
    ending the path, so a sibling such as `<old>2/` is left alone."""
    if not old or not new or old == new or old not in text:
        return text
    old, new = old.rstrip("/"), new.rstrip("/")
    return re.sub(re.escape(old) + r"(?=/|$|[\s\"'`,:;)\]}])", lambda _: new, text)


def to_workspace(text: str, native: NativeRun | None) -> str:
    """Translate host paths (as Codebase Memory reports them) to the run's workspace."""
    if native is None:
        return text
    return _swap_root(text, native.host_path, native.workspace)


def to_host(value: Any, native: NativeRun | None) -> Any:
    """Translate workspace paths in tool arguments (recursively) to host paths."""
    if native is None or not native.host_path:
        return value
    if isinstance(value, str):
        return _swap_root(value, native.workspace, native.host_path)
    if isinstance(value, dict):
        return {key: to_host(item, native) for key, item in value.items()}
    if isinstance(value, list):
        return [to_host(item, native) for item in value]
    return value


# Evidence tool name for commands the native CLIs (or Code Partner) ran in a workspace.
NATIVE_SHELL_TOOL = "native_shell"


def command_result(output: str, exit_code: int | None) -> str:
    """Execution evidence text: the exit status first, then the raw output."""
    status = f"exit_code: {exit_code}" if exit_code is not None else "exit_code: unknown"
    return f"{status}\n{output}"


async def run_native_command(project: RegisteredProject, command: str,
                             timeout: float = 900) -> tuple[str, int | None]:
    """Run a shell command in a registered workspace (Code Partner's own test runs)."""
    env, _ = toolchain(project)
    proc = await asyncio.create_subprocess_shell(
        command, cwd=project.workspace, env={**os.environ, **env},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"command timed out after {timeout:.0f}s", None
    return output.decode("utf-8", errors="replace"), proc.returncode


def read_only_session(project: RegisteredProject) -> NativeRun:
    """A native run that must not change the project (planner, reviewer, answers).

    This is detection, not a sandbox, and nothing is restored. A git baseline taken
    before the run (no objects are written) is compared afterwards. The baseline
    cannot tell the agent's writes from IDE edits or other runs on the same project,
    so the run fails only when the agent reported writing, or could have written
    (Codex's shell); other changes are reported as a warning.
    """
    from agent.routing import rollback
    env, notes = toolchain(project)
    native = NativeRun(workspace=project.workspace, read_only=True, host_path=project.host_path,
                       env=env, notes=notes)
    state: dict = {}

    async def start() -> None:
        state["baseline"] = await asyncio.to_thread(rollback._capture_baseline, project.workspace)

    async def finish() -> list[str]:
        baseline = state.pop("baseline", None)
        if baseline is None:
            return []
        changes, moved = await asyncio.to_thread(rollback._detect_changes, baseline)
        if not changes and not moved:
            return []
        own = [change["path"] for change in changes if change["path"] in native.written]
        other = [change["path"] for change in changes if change["path"] not in native.written]
        lines = []
        if own:
            lines.append("the agent wrote " + ", ".join(own))
        if other:
            lines.append("changed during the run (possibly by you or another run): "
                         + ", ".join(other))
        if moved:
            lines.append(moved)
        report = (f"Read-only run in project {project.id}: " + "; ".join(lines)
                  + ". Nothing was reverted.")
        if own or (other and native.agent_can_write):
            raise NativeSessionError(report)
        native.warnings.append(report)
        return []

    native.start, native.finish = start, finish
    return native


@dataclass(frozen=True)
class NativePlan:
    mode: str
    project: RegisteredProject | None = None
    reason: str | None = None

    @property
    def native(self) -> bool:
        return self.project is not None


def _native_providers() -> frozenset[str]:
    # Codex runs without its own sandbox here (bubblewrap cannot start in the
    # container); see docs/native-execution.md for what that does and does not isolate.
    value = os.getenv("CODEPARTNER_NATIVE_PROVIDERS", "claude,codex")
    return frozenset(item.strip().lower() for item in value.split(",") if item.strip())


def is_git_worktree(path: str | Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return (result.returncode == 0
            and os.path.realpath(result.stdout.strip()) == os.path.realpath(path))


def disk_files(root: str) -> list[str]:
    """Tracked and untracked, non-ignored project files of a native workspace."""
    result = subprocess.run(["git", "-C", root, "ls-files", "-z", "-c", "-o",
                             "--exclude-standard"], capture_output=True, timeout=60)
    return [os.fsdecode(item) for item in result.stdout.split(b"\0") if item]


def project_inventory(root: str, limit_chars: int = 8000) -> str:
    """The project's file list, one directory per line ("dir/: a.py b.py"), so a
    planner can see what exists without probing paths. Empty when git fails."""
    try:
        files = sorted(disk_files(root))
    except (OSError, subprocess.SubprocessError):
        return ""
    groups: dict[str, list[str]] = {}
    for path in files:
        folder, _, name = path.rpartition("/")
        groups.setdefault(folder, []).append(name)
    lines: list[str] = []
    used = 0
    for index, (folder, names) in enumerate(groups.items()):
        line = f"{folder or '.'}/: " + " ".join(names)
        if used + len(line) + 1 > limit_chars:
            rest = sum(len(items) for items in list(groups.values())[index:])
            lines.append(f"(+{rest} more files in {len(groups) - index} directories)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def plan_native(provider: str | None, excluded_files=(),
                project_id: str | None = None) -> NativePlan:
    """Decide whether one attempt runs natively. Excluded files do not prevent it:
    changes to them are detected and fail the attempt afterwards.

    `auto` falls back to the IDE path with a reason; an explicit `native` mode
    returns the reason so the caller can fail instead of silently falling back.
    """
    project_id = project_id if project_id is not None else active_project_id()
    override = os.getenv("CODEPARTNER_EXECUTION_MODE", "").strip().lower()
    if override and override not in MODES:
        return NativePlan("native", reason=f"CODEPARTNER_EXECUTION_MODE={override!r} is invalid")
    if project_id is None:
        if (override or DEFAULT_MODE) == "native":
            return NativePlan("native", reason="this conversation is not bound to a registered "
                                               f"project (see /project); {REGISTER_HINT}")
        return NativePlan("ide", reason="no registered project")
    try:
        project = registry().get(project_id)
    except WorkspaceError as exc:
        return NativePlan(override or DEFAULT_MODE, reason=str(exc))
    mode = override or project.execution
    if mode == "ide":
        return NativePlan("ide", reason="execution mode is ide")
    if provider is not None and provider.lower() not in _native_providers():
        return NativePlan(mode, reason=f"provider {provider!r} is not in "
                                       "CODEPARTNER_NATIVE_PROVIDERS")
    try:
        validate_workspace(project)
    except WorkspaceError as exc:
        return NativePlan(mode, reason=str(exc))
    if not is_git_worktree(project.workspace):
        return NativePlan(mode, reason=f"{project.workspace} is not the root of a git work "
                                       "tree; native change detection needs git")
    return NativePlan(mode, project)


def native_project() -> RegisteredProject | None:
    """The conversation's project when Code Partner itself works on its files
    (artifacts, its own test runs); None on the IDE path."""
    return plan_native(None).project


_MARKERS = (
    (("pyproject.toml", "setup.py", "requirements.txt"), "python3"),
    (("uv.lock",), "uv"),
    (("package.json",), "node"),
    (("pnpm-lock.yaml",), "pnpm"),
    (("yarn.lock",), "yarn"),
    (("go.mod",), "go"),
    (("Cargo.toml",), "cargo"),
    (("pom.xml",), "mvn"),
    (("build.gradle", "build.gradle.kts"), "java"),
    (("Makefile",), "make"),
)


def toolchain(project: RegisteredProject) -> tuple[dict[str, str], tuple[str, ...]]:
    """Container environment overrides and notes about missing project tools.

    Bind mounts expose host files, not host interpreters: a host `.venv` is not
    used, and uv projects get a container-local environment instead.
    """
    root = Path(project.workspace)
    env: dict[str, str] = {}
    missing = []
    for markers, tool in _MARKERS:
        if any((root / marker).exists() for marker in markers) and shutil.which(tool) is None:
            missing.append(tool)
    if (root / "uv.lock").exists() or (root / "pyproject.toml").exists():
        home = Path(os.getenv("HOME", "/home/codepartner"))
        env["UV_PROJECT_ENVIRONMENT"] = str(home / ".cache" / "codepartner-venvs" / project.id)
    notes = []
    if missing:
        notes.append("Missing in the Code Partner container: " + ", ".join(missing)
                     + ". Commands needing them will fail; do not claim they ran.")
    if (root / ".venv").exists():
        notes.append("The project's .venv was created on the host and is not usable here; "
                     "use `uv run ...`, which builds a separate container environment.")
    return env, tuple(notes)

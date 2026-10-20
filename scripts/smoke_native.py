"""Smoke-test native Claude Code / Codex execution in a mounted project (run in the container).

    docker compose exec codepartner python scripts/smoke_native.py --project music-generation
    docker compose exec codepartner python scripts/smoke_native.py --project music-generation --provider codex
    docker compose exec codepartner python scripts/smoke_native.py --project music-generation --e2e

Default: the real CLI must, with its own tools, read a tracked file, create a uniquely
named file, and run `git status` on it. The script verifies all three on disk and in
the output, then removes only the file it asked the CLI to create.

`--e2e` drives the real services natively, end to end: OpenSpec catalog read from disk
(read-only), the WP planner (read-only session), a journaled coder attempt that writes a
unique test file and runs pytest (execution evidence), the reviewer (filesystem snapshot,
read-only session), then removes the test file. The Work Package is synthetic and in memory, and the
routing database is a temporary file, so the project's OpenSpec state is never changed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import providers  # noqa: E402
from agent import workspace  # noqa: E402
from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent  # noqa: E402
from mcp_bridge.registry import ROLE_CODER  # noqa: E402


def _probe_file(root: Path) -> tuple[str, str]:
    """A small tracked text file and its first non-empty line."""
    tracked = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True,
                             text=True, check=True).stdout.splitlines()
    for path in tracked:
        candidate = root / path
        try:
            if not candidate.is_file() or candidate.stat().st_size > 64_000:
                continue
            line = next((item.strip() for item in candidate.read_text().splitlines()
                         if item.strip()), None)
        except (OSError, UnicodeDecodeError):
            continue
        if line and "`" not in line and len(line) < 120:
            return path, line
    raise SystemExit("no small tracked text file to read in this project")


async def _run(provider_name: str, native: workspace.NativeRun, prompt: str,
               model: str | None) -> tuple[str, list[str]]:
    provider = providers.get(provider_name)
    text, errors = [], []
    async for event in provider.tools([{"role": "user", "content": prompt}], [],
                                      model=model, role=ROLE_CODER, native=native):
        if isinstance(event, TextEvent):
            text.append(event.text)
        elif isinstance(event, ErrorEvent):
            errors.append(event.message)
        elif isinstance(event, DoneEvent):
            break
    return "\n".join(text), errors


def _status(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1",
                           "--untracked-files=all"], capture_output=True, text=True,
                          check=True).stdout


async def _e2e(project, provider_name: str, planner_model: str, coder_model: str) -> int:
    import dataclasses
    import json
    import tempfile
    from types import SimpleNamespace

    from agent.planning.service import WPPlannerService
    from agent.routing.config import load_routing_config
    from agent.routing.evidence import has_passing_test_run
    from agent.routing.executors.registry import ExecutorRegistry
    from agent.routing.persistence import RoutingPersistence
    from agent.routing.review import ReviewService
    from agent.routing.rollback import RollbackJournal
    from agent.routing.task_execution import TaskExecutionService
    from agent.spec.artifact_store import OpenSpecArtifactStore
    from agent.spec.filesystem_client import FilesystemFileClient
    from agent.spec.models import Task, WorkPackage

    root = Path(project.workspace)
    workspace.activate(project.id)
    before = _status(root)
    checks: dict[str, bool] = {}
    tmp = tempfile.TemporaryDirectory(prefix="codepartner-e2e-")
    os.environ["CODEPARTNER_EVIDENCE_DIR"] = str(Path(tmp.name) / "evidence")
    persistence = RoutingPersistence(Path(tmp.name) / "routing.sqlite3")
    config = load_routing_config()
    config = dataclasses.replace(config, reviewers={}, planner=dataclasses.replace(
        config.planner, narrate=False,
        tiers={tier: (planner_model,) for tier in config.planner.tiers}))
    for name in (planner_model, coder_model):
        if config.models[name].provider != provider_name:
            raise SystemExit(f"{name} is not a {provider_name} model")

    # 1. OpenSpec artifacts, read from disk only.
    store = OpenSpecArtifactStore(FilesystemFileClient(project.workspace))
    catalog = await store._load_catalog()
    checks["openspec read from disk"] = isinstance(catalog.get("changes"), list)
    print(f"openspec: {len(catalog.get('changes', []))} change(s) in .codepartner/spec-index.json")

    stamp = uuid.uuid4().hex[:8]
    module = f"codepartner_e2e_{stamp}"
    test_file = f"test_{module}.py"
    task = Task(id="WP990-T1", title="Add a smoke test",
                description=f"Create `{test_file}` in the project root with one pytest test "
                            "function that asserts 1 + 1 == 2, and nothing else.",
                complexity="simple", reasoning="smoke", context="",
                acceptance_criteria=[f"{test_file} exists", "the test passes"],
                preferred_capability="implementation")
    package = WorkPackage(id="WP990", title="Smoke", kind="technical-outcome",
                          objective="Prove native planning, coding and review",
                          acceptance_criteria=["One passing smoke test"], requirement_refs=[],
                          tasks=[task], complexity="low")
    change = SimpleNamespace(id="CHG990", title="Native smoke", summary="Smoke test",
                             design=SimpleNamespace(context="", decisions=[]),
                             work_packages=[package], epic=None, capabilities=())

    class Artifacts:
        async def object_by_id(self, object_id):
            return "work_package", change, package

    # 2. WP planner: read-only native session.
    print(f"planner: {planner_model} …")
    planner = WPPlannerService(Artifacts(), persistence, config=config)
    turn = await planner.start("e2e", "WP990", "Plan WP990. Keep it minimal.", [], {})
    checks["planner produced a plan"] = turn.draft is not None and not turn.gaps
    session = await persistence.active_planning("e2e")
    metrics = (await persistence.planning_turns(session["session_id"]))[-1]["tool_metrics"]
    checks["planner ran natively"] = metrics.get("execution_mode") == "native"
    print(f"planner native_tool_metrics: {metrics.get('native_tool_metrics')}")
    checks["planner left the project unchanged"] = _status(root) == before

    # 3. Coder: journaled native attempt; pytest becomes execution evidence.
    print(f"coder: {coder_model} …")
    service = TaskExecutionService(config=config, router=None, persistence=persistence,
                                   executors=ExecutorRegistry({}))
    native = service._native_run(project, {"implementation_job_id": "e2e",
                                           "implementation_position": 0}, task)
    prompt = (f"You are working in the current directory, which is the project root. "
              f"{task.description} Write it at the relative path `{test_file}` (not /tmp). "
              f"Then run exactly `uv run --no-project --with pytest pytest -q {test_file}` "
              "and report its result, naming files by project-relative path. Do not commit.")
    output, errors = await _run(provider_name, native, prompt,
                                config.models[coder_model].cli_model)
    journal = RollbackJournal(persistence, "e2e", 0, [], {})
    changed = await journal.changed_paths()
    checks["coder wrote the file natively (journaled)"] = changed == [test_file]
    evidence = await persistence.verification_evidence("e2e", 0)
    checks["pytest captured as passing evidence"] = has_passing_test_run(evidence)
    for error in errors:
        print(f"coder ERROR {error}")

    # 4. Reviewer: filesystem snapshot + read-only native model pass.
    print("reviewer …")
    object.__setattr__(task, "_review_journal_manifest", await journal.review_manifest())
    try:
        decision = await ReviewService(config).review(
            change, task, output or "(no report)", coder_model, [], {}, job_id="e2e")
        print(f"review verdict: {decision.verdict} by {decision.reviewer_model}: "
              f"{decision.rationale[:200]}")
        checks["reviewer reached a verdict"] = decision.verdict in {"done", "needs_work"}
    except Exception as exc:
        print(f"review ERROR {type(exc).__name__}: {exc}")
        checks["reviewer reached a verdict"] = False
    checks["review left the project unchanged"] = (root / test_file).exists() and \
        {line[3:] for line in _status(root).splitlines()} - \
        {line[3:] for line in before.splitlines()} <= {test_file}

    # 5. Remove only the file this run asked for.
    if (root / test_file).exists():
        (root / test_file).unlink()
    checks["project back to its pre-run state"] = _status(root) == before

    for check, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {check}")
    if not all(checks.values()):
        print("--- coder output ---\n" + output)
    tmp.cleanup()
    return 0 if all(checks.values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True, help="registry id from config/projects.yaml")
    parser.add_argument("--provider", choices=("claude", "codex"), default="claude")
    parser.add_argument("--model", default=None, help="CLI model (default: the CLI's default)")
    parser.add_argument("--e2e", action="store_true",
                        help="plan, implement and review through the real services")
    parser.add_argument("--planner-model", default=None,
                        help="routing model for --e2e planning (default per provider)")
    parser.add_argument("--coder-model", default=None,
                        help="routing model for --e2e coding (default per provider)")
    args = parser.parse_args(argv)

    try:
        project = workspace.registry().resolve(args.project)
    except workspace.WorkspaceError as exc:
        print(f"FAIL registry/mount: {exc}")
        return 1
    root = Path(project.workspace)
    if not workspace.is_git_worktree(root):
        print(f"FAIL {root} is not the root of a git work tree")
        return 1
    if args.e2e:
        defaults = {"claude": ("claude-sonnet", "claude-haiku"),
                    "codex": ("codex-sol-medium", "codex-luna-low")}[args.provider]
        return asyncio.run(_e2e(project, args.provider, args.planner_model or defaults[0],
                                args.coder_model or defaults[1]))
    probe, first_line = _probe_file(root)
    name = f"codepartner-smoke-{uuid.uuid4().hex[:12]}.txt"
    target = root / name
    if target.exists():
        print(f"FAIL {name} already exists; refusing to run")
        return 1
    token = uuid.uuid4().hex
    env, notes = workspace.toolchain(project)
    native = workspace.NativeRun(workspace=str(root), env=env, notes=notes)
    prompt = (
        "This is a Code Partner smoke test. Use your own tools in the current directory, "
        "do exactly these three steps, and change nothing else:\n"
        f"1. Read the file `{probe}` and print its first non-empty line as "
        "`FIRST_LINE: <line>`.\n"
        f"2. Create the file `{name}` containing exactly `{token}`.\n"
        f"3. Run the shell command `git status --porcelain -- {name}` and print its output "
        "as `GIT_STATUS: <output>`.")
    created = False
    try:
        output, errors = asyncio.run(_run(args.provider, native, prompt, args.model))
        created = target.exists()
        checks = {
            "read": first_line in output,
            "write": created and target.read_text().strip() == token,
            "command": f"?? {name}" in output,
        }
        for check, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'} native {check}")
        for error in errors:
            print(f"ERROR {error}")
        if not all(checks.values()):
            print("--- CLI output ---\n" + output)
        return 0 if all(checks.values()) and not errors else 1
    finally:
        # Only the uniquely named file this run asked for, and only if it now exists.
        if target.exists():
            created = True
            target.unlink()
        print(f"cleanup: {'removed ' + name if created else 'nothing to remove'}")


if __name__ == "__main__":
    sys.exit(main())

"""Smoke-test Code Partner -> OpenTelemetry Collector -> Langfuse with a real execution.

    # in a one-off container built from this tree (the running service is untouched)
    docker compose run --rm --no-deps -e OTEL_ENABLED=true \
        -e LANGFUSE_BASE_URL -e LANGFUSE_PUBLIC_KEY -e LANGFUSE_SECRET_KEY \
        codepartner python scripts/smoke_observability.py --mode native --project music-generation
    ... python scripts/smoke_observability.py --mode mcp --kb-project <codebase-memory project>

Modes are separate because native CLI tools are not individually instrumented:

- `native`: a real, read-only native CLI run (it reads the first line of a tracked file
  with its own tools). Requires execution -> llm.execution -> llm.turn,
  with native tool counts on the LLM span. No MCP spans are expected.
- `mcp`: an orchestration Run that routes a model (real planner routing) and starts a
  provider Run whose only tools are the read-only Codebase Memory MCP tools (through the
  kb-bridge sidecar). Requires execution -> model.routing and
  llm.execution -> llm.turn -> mcp.tool_call (at least one).

Both modes then read the trace back from Langfuse with the v4 Observations API
(`GET /api/public/v2/observations`, bounded by fromStartTime/toStartTime, filtered by
traceId) and check the nesting via parentObservationId. Langfuse keys come from the
environment or observability/.env; Code Partner itself never needs them.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from telemetry import tracing  # noqa: E402

EXECUTION, ROUTING, LLM, TURN, TOOL = (
    tracing.EXECUTION_SPAN, tracing.ROUTING_SPAN, tracing.LLM_SPAN,
    tracing.TURN_SPAN, tracing.TOOL_SPAN)


def _langfuse_settings() -> tuple[str, str, str]:
    values = dict(os.environ)
    env_file = ROOT / "observability" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values.setdefault(key.strip(), value.strip().strip('"'))
    try:
        return (values.get("LANGFUSE_BASE_URL") or "http://localhost:3000",
                values["LANGFUSE_PUBLIC_KEY"], values["LANGFUSE_SECRET_KEY"])
    except KeyError:
        raise SystemExit("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set "
                         "(environment or observability/.env)")


def _observations(trace_id: str, started: datetime, timeout: float,
                  expected: set[str]) -> list[dict]:
    """All observations of one trace, via the v4 Observations API (cursor paged).
    Polls until every `expected` kind has arrived: ingestion is asynchronous."""
    base, public, secret = _langfuse_settings()
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    deadline = time.monotonic() + timeout
    while True:
        rows, cursor = [], None
        while True:
            query = {"traceId": trace_id, "limit": "100",
                     "fromStartTime": (started - timedelta(minutes=5)).isoformat(),
                     "toStartTime": (datetime.now(timezone.utc)
                                     + timedelta(minutes=5)).isoformat(),
                     "fields": "core,basic,metadata"}
            if cursor:
                query["cursor"] = cursor
            request = urllib.request.Request(
                f"{base}/api/public/v2/observations?{urllib.parse.urlencode(query)}",
                headers={"Authorization": f"Basic {auth}"})
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    page = json.load(response)
            except urllib.error.HTTPError as exc:
                raise SystemExit(f"Langfuse API error {exc.code}: {exc.read()[:300]!r}")
            rows += page.get("data", [])
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor:
                break
        if expected <= {_kind(row) for row in rows} or time.monotonic() > deadline:
            return rows
        time.sleep(3)


def _kind(row: dict) -> str:
    # Langfuse names TOOL observations after gen_ai.tool.name, so match tools by type.
    return TOOL if row.get("type") == "TOOL" else row.get("name") or ""


async def _native(project_id: str, provider_name: str, model: str | None) -> tuple[object, bool]:
    import providers
    from agent import workspace
    from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent
    from mcp_bridge.registry import ROLE_CODER
    from scripts.smoke_native import _probe_file

    project = workspace.registry().resolve(project_id)
    workspace.activate(project.id)
    root = Path(project.workspace)
    probe, first_line = _probe_file(root)
    env, notes = workspace.toolchain(project)
    native = workspace.NativeRun(workspace=str(root), env=env, notes=notes, read_only=True)
    prompt = ("This is a Code Partner observability smoke test. Using your own tools, read "
              f"the file `{probe}` and print its first non-empty line as `FIRST_LINE: <line>`. "
              "Do nothing else.")
    text, errors, captured = [], [], {}
    from mcp_bridge import server as engine
    original = engine.start_run

    def capture(*args, **kwargs):
        captured["run"] = original(*args, **kwargs)
        return captured["run"]

    engine.start_run = capture
    try:
        async for event in providers.get(provider_name).tools(
                [{"role": "user", "content": prompt}], [], model=model, role=ROLE_CODER,
                native=native):
            if isinstance(event, TextEvent):
                text.append(event.text)
            elif isinstance(event, ErrorEvent):
                errors.append(event.message)
            elif isinstance(event, DoneEvent):
                break
    finally:
        engine.start_run = original
    ok = first_line in "\n".join(text) and not errors
    print(f"{'PASS' if ok else 'FAIL'} native execution read {probe}")
    for error in errors:
        print(f"ERROR {error}")
    return captured["run"], ok


async def _mcp(kb_project: str, provider_name: str | None) -> tuple[object, bool]:
    import providers
    from agent.planning.knowledge_base import CodebaseMemoryClient
    from agent.planning.routing import PlannerRoutingService
    from agent.routing.config import load_routing_config
    from mcp_bridge import server as engine
    from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent
    from mcp_bridge.registry import ROLE_OPTIMIZER

    specs, handlers, _ = await CodebaseMemoryClient().planner_tools(kb_project)
    outcome: dict = {"text": [], "errors": []}

    async def orchestrate(run, _prompt, _system, _specs):
        # Real planner routing inside the execution, then the provider Run it chose.
        config = load_routing_config()
        decision = PlannerRoutingService(config).route("low")
        provider = provider_name or decision.provider
        model = decision.cli_model if provider == decision.provider else None
        prompt = ("This is a Code Partner observability smoke test. Call the kb_get_architecture "
                  "tool exactly once, then reply with one short sentence naming how many "
                  "packages or modules it reports. Use no other tools.")
        async for event in providers.get(provider).tools(
                [{"role": "user", "content": prompt}], specs, model=model,
                role=ROLE_OPTIMIZER, direct_tool_handlers=handlers):
            if isinstance(event, TextEvent):
                outcome["text"].append(event.text)
            elif isinstance(event, ErrorEvent):
                outcome["errors"].append(event.message)
            elif isinstance(event, DoneEvent):
                break
        return "endTurn"

    run = engine.start_run("", None, [], orchestrate)
    run.metadata["bridge_command"] = "observability_smoke"
    while not isinstance(await run.next_event(), DoneEvent):
        pass
    ok = bool(outcome["text"]) and not outcome["errors"]
    print(f"{'PASS' if ok else 'FAIL'} MCP execution answered")
    for error in outcome["errors"]:
        print(f"ERROR {error}")
    return run, ok


def _check_trace(mode: str, run, rows: list[dict]) -> dict[str, bool]:
    by_id = {row["id"]: row for row in rows}

    def parent_name(row):
        parent = by_id.get(row.get("parentObservationId") or "")
        return _kind(parent) if parent else None

    roots = [row for row in rows if row.get("name") == EXECUTION and not row.get("parentObservationId")]
    llm = [row for row in rows if row.get("name") == LLM]
    turns = [row for row in rows if row.get("name") == TURN]
    tools = [row for row in rows if _kind(row) == TOOL]
    root_meta = (roots[0].get("metadata") or {}) if roots else {}
    checks = {
        "trace has one root codepartner.execution": len(roots) == 1,
        "root carries the Code Partner execution id": run.run_id in json.dumps(root_meta),
        "llm.execution is a child of the root": bool(llm) and all(
            parent_name(row) == EXECUTION for row in llm),
        "llm.turn is a child of llm.execution": bool(turns) and all(
            parent_name(row) == LLM for row in turns),
    }
    if mode == "mcp":
        routing = [row for row in rows if row.get("name") == ROUTING]
        checks["model.routing is a child of the root"] = bool(routing) and all(
            parent_name(row) == EXECUTION for row in routing)
        checks["mcp.tool_call spans are children of llm.turn"] = bool(tools) and all(
            parent_name(row) == TURN for row in tools)
    else:
        checks["no MCP spans are required for native-only tools"] = True
    print("langfuse observations: " + ", ".join(
        f"{_kind(row)}({row.get('name')})<-{parent_name(row)}" for row in rows))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("native", "mcp"), required=True)
    parser.add_argument("--project", help="native: registry id from config/projects.yaml")
    parser.add_argument("--kb-project", help="mcp: Codebase Memory project name")
    parser.add_argument("--provider", choices=("claude", "codex"), default=None)
    parser.add_argument("--model", default=None, help="native: CLI model (default: CLI default)")
    parser.add_argument("--wait", type=float, default=90, help="seconds to wait for Langfuse")
    args = parser.parse_args(argv)
    if not tracing.configure_tracing():
        print("FAIL tracing is not enabled (set OTEL_ENABLED=true)")
        return 1
    started = datetime.now(timezone.utc)
    # Stands in for the conversation id a chat request carries (the Langfuse session).
    session = "conv_smoke" + os.urandom(6).hex()
    tracing.set_session(session)
    try:
        if args.mode == "native":
            if not args.project:
                parser.error("--mode native needs --project")
            run, executed = asyncio.run(_native(args.project, args.provider or "claude",
                                                args.model))
        else:
            if not args.kb_project:
                parser.error("--mode mcp needs --kb-project")
            run, executed = asyncio.run(_mcp(args.kb_project, args.provider))
    finally:
        flushed = tracing.force_flush(10_000)
    trace_id = run.metadata.get("otel_trace_id")
    print(f"execution id: {run.run_id}\notel trace id: {trace_id}\nsession: {session}\n"
          f"flushed: {flushed}")
    expected = {EXECUTION, LLM, TURN} | ({ROUTING, TOOL} if args.mode == "mcp" else set())
    rows = _observations(trace_id, started, args.wait, expected)
    checks = {"execution succeeded": executed, "trace reached Langfuse": bool(rows),
              **(_check_trace(args.mode, run, rows) if rows else {}),
              "trace is in the Langfuse session": bool(rows) and all(
                  row.get("sessionId") == session for row in rows)}
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    base = _langfuse_settings()[0]
    print(f"open: {base}/project/codepartner/traces/{trace_id}")
    tracing.shutdown_tracing()
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

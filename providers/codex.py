"""Codex provider (local `codex` CLI, driven non-interactively).

`CodexProvider` implements the `Provider` surface:
  - plain chat: `stream` (and inherited `complete`) — `codex exec` with no tools.
  - tool calling: `tools(messages, specs)` runs `codex exec` as a background Run and
    hands it the bridged JetBrains tools over a private MCP endpoint
    (`mcp_bridge.mcp_http`); the start-vs-resume decision is internal, exactly as in
    `ClaudeProvider`.

The difference from Claude is only transport. Claude hosts its bridged tools in
process; `codex` is a child process, so the same tools are published over HTTP for
the lifetime of the run. Everything downstream — role allowlist, tool timeout, the
`ToolCallEvent` round-trip through the IDE — is the shared bridge machinery.

Codex runs with `-s read-only`, so its own shell/apply_patch cannot write: the only
way for it to change the project is the IDE-backed write tools.

This is the only module that shells out to `codex`.
"""
import asyncio
import contextlib
import json
import logging
import os
import shlex
import tempfile
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path

from mcp_bridge import mcp_http
from mcp_bridge import server as engine
from mcp_bridge.models import ErrorEvent, Event, ProgressEvent, Run, TextEvent
from mcp_bridge.registry import ToolSpec, allowed_for, native_bridged_specs
from providers import scope_guard, subagents
from providers.base import Provider
from telemetry import tracing

log = logging.getLogger("codepartner.providers.codex")

BIN = os.getenv("CODEX_BIN", "codex")

# The neutral tiers the Optimizer emits (`agent.prompt_optimizer.MODEL_CHOICES`) are
# Claude-shaped names; each provider maps them to its own models. Unset = whatever
# `CODEX_MODEL` says, and failing that the CLI's own default.
_TIER_ENV = {
    "haiku": "CODEX_MODEL_SIMPLE",
    "sonnet": "CODEX_MODEL_MEDIUM",
    "opus": "CODEX_MODEL_COMPLEX",
}


# Codex reports subagent spawns and waits as this item type; the child's own tool
# calls never appear in the parent's stream.
COLLAB_ITEM = "collab_tool_call"


def parse_event(obj: dict) -> Event | None:
    """One `codex exec --json` line → a neutral event, or None if it carries nothing
    the conversation needs (thread/turn lifecycle, reasoning, tool-call bookkeeping —
    the bridge already emits our own ToolCallEvents). Pure: no I/O, no state."""
    kind = obj.get("type")
    if kind == "item.completed":
        item = obj.get("item") or {}
        if item.get("type") == "agent_message":
            return TextEvent(item.get("text") or "") if item.get("text") else None
        if item.get("type") == "error":
            return ErrorEvent(item.get("message") or "codex reported an error")
        return None
    if kind == "error":
        return ErrorEvent(obj.get("message") or "codex reported an error")
    if kind == "turn.failed":
        return ErrorEvent((obj.get("error") or {}).get("message") or "codex turn failed")
    return None


def usage_of(obj: dict) -> dict | None:
    """Token usage from a terminal `turn.completed` line, if present. The
    `input_tokens`/`output_tokens` spelling is already normalized downstream by
    `conversation.streaming_responder`."""
    if obj.get("type") == "turn.completed":
        usage = obj.get("usage")
        return usage if isinstance(usage, dict) else None
    return None


def record_native_item(run: Run, item: dict) -> None:
    """Count a completed native codex item (shell command, file change, MCP call)."""
    kind = item.get("type")
    if kind == "agent_message" and run.native is not None:
        subagents.note_scope(run.metadata, run.metadata.setdefault(
            "native_tool_metrics", run.native.metrics), text=item.get("text") or "")
        return
    names = {"command_execution": "shell", "file_change": "file_change",
             "mcp_tool_call": str(item.get("tool") or "mcp")}
    if kind not in names:
        return
    metrics = run.metadata.setdefault("native_tool_metrics", run.native.metrics if run.native
                                      else {"tool_calls": 0, "tools": {}, "files": []})
    metrics["tool_calls"] += 1
    metrics["tools"][names[kind]] = metrics["tools"].get(names[kind], 0) + 1
    if kind == "mcp_tool_call" and names[kind].startswith("kb_"):
        metrics["kb_calls"] = metrics.get("kb_calls", 0) + 1
    if kind == "command_execution":
        reads = [read for read, _ in subagents.classify_command(item.get("command"))[1]]
        subagents.note_scope(run.metadata, metrics, [
            (run.native.relative(read) or read) if run.native is not None else read
            for read in reads])
    for change in item.get("changes") or ():
        path = change.get("path") if isinstance(change, dict) else None
        if isinstance(path, str) and run.native is not None:
            path = run.native.relative(path) or path
        if isinstance(path, str) and path not in metrics["files"]:
            metrics["files"].append(path)
        if run.native is not None:
            run.native.record_write(path)


def record_collab_item(tracker: subagents.Tracker, item: dict, briefed: set[str]
                       ) -> list[str]:
    """Count a completed subagent spawn, and each child's final reply once. Returns the
    briefs newly received, for the progress lines."""
    if item.get("type") != COLLAB_ITEM:
        return []
    if item.get("tool") == "spawn_agent":
        # The item carries the request text but not the role name.
        tracker.spawned(item.get("agent_type"), item.get("prompt"),
                        background=bool(item.get("run_in_background") or item.get("async")),
                        description=item.get("description"))
    received = []
    for thread, state in (item.get("agents_states") or {}).items():
        message = state.get("message") if isinstance(state, dict) else None
        if (isinstance(message, str) and message and thread not in briefed
                and state.get("status") == "completed"):
            briefed.add(thread)
            tracker.brief(message)
            received.append(message)
    return received


def progress_line(item: dict) -> str | None:
    """The chat line for a native codex item as it starts: a shell command (reads shown
    as reads) or a subagent request; None for anything else."""
    kind = item.get("type")
    if kind == "command_execution":
        command = item.get("command")
        verdict, reads = subagents.classify_command(command)
        if verdict == "exact_read" and reads:
            return "🔎 Reading " + ", ".join(f"`{path}`" for path, _ in reads[:4])
        text = " ".join(str(command or "").split())
        return f"🔎 Running `{text[:60]}{'…' if len(text) > 60 else ''}`"
    if kind == COLLAB_ITEM and item.get("tool") == "spawn_agent":
        return subagents.describe_spawn(item.get("agent_type") or subagents.READER,
                                        item.get("prompt"))
    return None


def role_config(child: subagents.Subagent,
                request: subagents.Request | None = None) -> str:
    """One subagent's Codex role file on the configured reader model. JSON strings are
    valid TOML basic strings."""
    model, effort = (request.model_for(CodexProvider.NAME) if request is not None
                     else (None, None))
    lines = [f"model_reasoning_effort = {json.dumps(effort or 'low')}",
             f"developer_instructions = {json.dumps(child.prompt)}"]
    model = model or model_for("haiku")
    if model:  # unset: the role inherits the parent's model
        lines.insert(0, f"model = {json.dumps(model)}")
    return "\n".join(lines) + "\n"


def role_args(children: list[subagents.Subagent], directory: str,
              request: subagents.Request | None = None) -> list[str]:
    """`-c` flags defining this run's subagent roles; their files go in `directory`."""
    args: list[str] = []
    for child in children:
        path = os.path.join(directory, f"{child.name}.toml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(role_config(child, request))
        args += ["-c", f"agents.{child.name}.description={json.dumps(child.description)}",
                 "-c", f"agents.{child.name}.config_file={json.dumps(path)}"]
    return args + ["-c", "agents.max_depth=1"] if args else args


def model_for(model: str | None) -> str | None:
    """Neutral tier → codex model id. An unrecognized tier is dropped rather than
    forwarded, so a bad routing decision costs a default, not a CLI error."""
    default = os.getenv("CODEX_MODEL") or None
    if not model:
        return default
    env = _TIER_ENV.get(model.lower())
    if env is None:
        return default
    return os.getenv(env) or default


class CodexProvider(Provider):
    NAME = "codex"

    async def stream(self, prompt: str, system: str | None = None
                     ) -> AsyncIterator[tuple[str, str | None]]:
        """Plain chat: no tools at all — `mcp_servers={}` drops anything configured in
        the user's `~/.codex/config.toml`, and `-s read-only` keeps the sandbox shut."""
        args = self._base_args() + ["-c", "mcp_servers={}"]
        async for obj in self._exec(args, self._join(prompt, system)):
            event = parse_event(obj)
            if isinstance(event, TextEvent):
                yield event.text, None
            elif isinstance(event, ErrorEvent):
                yield f"[error: {event.message}]", None
        yield "", "endTurn"

    async def tools(self, messages: list[dict], specs: list[ToolSpec],
                    model: str | None = None, role: str | None = None,
                    run_metadata: dict | None = None,
                     gateway_url: str | None = None, concrete_model: bool = False,
                     reasoning_effort: str | None = None,
                     direct_tool_handlers: dict | None = None,
                     reviewer_mode: bool = False,
                     preapprove_ide_tools: bool = False,
                     native=None, subagents=None,
                    **kwargs) -> AsyncIterator[Event]:
        """Stateful like Claude's, and for the same reason: the `codex` process stays
        alive across HTTP requests, blocked inside its MCP tool call, while JetBrains
        executes the tool. A history carrying tool results resolves the pending futures
        and continues the SAME process; otherwise a fresh run starts. `model` is a
        neutral tier (see `model_for`); `role` selects the tool allowlist."""
        results = self._trailing_tool_results(messages)
        if results:
            run: Run | None = None
            for m in results:
                run = engine.resume(m["tool_call_id"], m.get("content") or "") or run
            if run is None:
                yield TextEvent("No pending tool call matched this result (it may have timed out).")
                return
        else:
            prompt, system = self._flatten(messages)

            async def strategy(run: Run, prompt: str, system: str | None,
                                specs: list[ToolSpec]) -> str:
                return await self._run_strategy(run, prompt, system, model, gateway_url,
                                                concrete_model, reasoning_effort, reviewer_mode,
                                                preapprove_ide_tools)

            if native is not None:
                specs = native_bridged_specs(specs)
            run = engine.start_run(prompt, system, specs, strategy=strategy,
                                   allowed=allowed_for(role),
                                   direct_tool_handlers=direct_tool_handlers,
                                   trace_kind="llm", trace_provider=self.NAME,
                                   trace_model=model if concrete_model else model_for(model),
                                   trace_reasoning_effort=reasoning_effort)
            run.metadata.update(run_metadata or {})
            run.native = native
            run.subagents = subagents

        async for event in self._drain_queue(run):
            yield event

    # --- internals -----------------------------------------------------------

    async def _run_strategy(self, run: Run, prompt: str, system: str | None,
                            model: str | None = None, gateway_url: str | None = None,
                            concrete_model: bool = False,
                            reasoning_effort: str | None = None,
                            reviewer_mode: bool = False,
                            preapprove_ide_tools: bool = False) -> str:
        """The engine `Strategy`: publish this Run's bridged tools, run `codex exec`
        against them, and push its output onto the Run as neutral events. Returns the
        terminal reason. The MCP endpoint is opened and closed with the process, so a
        finished (or cancelled) run leaves nothing callable behind."""
        native = run.native
        hosted_planner = bool(run.metadata.get("planner") and
                              run.metadata.get("hosted_reader"))
        hosted_reader_child = bool(run.metadata.get("hosted_reader") and not
                                   run.metadata.get("planner"))
        if native is not None:
            native.agent_can_write = not hosted_planner
        request = getattr(run, "subagents", None)
        children = subagents.resolve(request, native)
        policy = scope_guard.policy_from_run(run, children)
        if hosted_planner and "planner_reader" not in run.registry.names():
            await run.put(ErrorEvent(
                "Scoped Codex planner has no hosted reader; stopped before model execution."))
            return "failed"
        if policy is not None and policy.active:
            # This CLI currently inherits the parent's disabled shell surface into
            # reader children and exposes no child identity to PreToolUse. A hook
            # cannot make that an access boundary, so do not start the model.
            await run.put(ErrorEvent(
                "Scoped Codex native execution cannot separate parent and reader file "
                "access with this CLI; the run was stopped before model execution."))
            return "failed"
        tracker = (subagents.Tracker(native.metrics, request, offered=bool(children),
                                     native=native)
                   if native is not None else None)
        if native is not None:
            run.metadata["exploration_mode"] = "reader" if children else "off"
        briefed: set[str] = set()
        shows_progress = any(run.metadata.get(kind)
                             for kind in ("task_execution", "planner", "reviewer"))
        async with contextlib.AsyncExitStack() as stack:
            # Role files live only as long as the process that reads them.
            roles = (stack.enter_context(tempfile.TemporaryDirectory(
                prefix="codepartner-agents-")) if children or policy is not None else "")
            url = await stack.enter_async_context(mcp_http.serve_run(run))
            args = self._base_args(model, gateway_url, concrete_model, reasoning_effort,
                                   reviewer_mode, native,
                                   role_args(children, roles, request))
            tool_free = bool(run.metadata.get("tool_free"))
            if hosted_planner or hosted_reader_child or tool_free:
                args += ["-c", "mcp_servers={}", "-c", "features.apps=false",
                         "-c", "features.plugins=false", "-c", "features.browser_use=false",
                         "-c", "features.computer_use=false"]
            args += self._mcp_args(url, run.registry.names(), reviewer_mode,
                                   preapprove_ide_tools)
            if hosted_planner or hosted_reader_child or tool_free:
                # Added after CODEX_ARGS so a local override cannot restore the
                # parent shell; the separate reader process gets its own config.
                args += ["-c", "features.shell_tool=false"]
            if hosted_planner:
                run.metadata["exploration_mode"] = "reader"
            item_types: Counter[str] = Counter()
            reviewer_tool_activity: list[dict[str, str | bool]] = []
            turn_open = False
            async for obj in self._exec(args, self._join(prompt, system),
                                        env=native.env if native is not None else None,
                                        report_exit=native is not None):
                if obj.get("type") == "turn.started":
                    tracing.start_turn(run, model if concrete_model else model_for(model))
                    turn_open = True
                if (native is not None and shows_progress
                        and obj.get("type") == "item.started"):
                    line = progress_line(obj.get("item") or {})
                    if line:
                        await run.put(ProgressEvent(f"- {line}\n"))
                if native is not None and obj.get("type") == "item.completed":
                    item = obj.get("item") or {}
                    record_native_item(run, item)
                    if tracker is not None:
                        for brief in record_collab_item(tracker, item, briefed):
                            if shows_progress:
                                await run.put(ProgressEvent(
                                    f"- {subagents.describe_brief(None, brief)}\n"))
                    if item.get("type") == "command_execution":
                        if tracker is not None:
                            tracker.parent_command(item.get("command"))
                        code = item.get("exit_code")
                        await native.command_finished(
                            item.get("command"), item.get("aggregated_output"),
                            code if isinstance(code, int) else None)
                if reviewer_mode and obj.get("type") in {"item.started", "item.completed"}:
                    item = obj.get("item") or {}
                    if isinstance(item, dict):
                        item_type = str(item.get("type") or "unknown")
                        item_types[item_type] += 1
                        if item_type == "mcp_tool_call":
                            reviewer_tool_activity.append({
                                "server": str(item.get("server") or "unknown"),
                                "tool": str(item.get("tool") or item.get("name") or "unknown"),
                                "status": str(item.get("status") or "unknown"),
                                "has_error": bool(item.get("error")),
                            })
                event = parse_event(obj)
                if event is not None:
                    await run.put(event)
                usage = usage_of(obj)
                if usage is not None:
                    if not turn_open:
                        tracing.start_turn(run, model if concrete_model else model_for(model))
                    tracing.record_turn_usage(run, usage)
                    tracing.finish_turn(run)
                    turn_open = False
                    if run.usage is None:
                        run.usage = dict(usage)
                    else:
                        for key, value in usage.items():
                            if isinstance(value, int) and not isinstance(value, bool):
                                run.usage[key] = run.usage.get(key, 0) + value
                    log.info("codex usage: role=%s model=%s usage=%s",
                             "reviewer" if reviewer_mode else "coder",
                             model_for(model) or "default", usage)
            if reviewer_mode:
                log.info("codex_reviewer_activity model=%s item_types=%s mcp_tools=%s",
                         model or "default", dict(item_types), reviewer_tool_activity)
            if policy is not None:
                denial_file = Path(roles) / "scope_denials.jsonl"
                if denial_file.exists():
                    for line in denial_file.read_text().splitlines():
                        try:
                            tool = json.loads(line).get("tool") or "unknown"
                        except json.JSONDecodeError:
                            tool = "unknown"
                        counts = native.metrics.setdefault("scope_denials", {})
                        counts[tool] = counts.get(tool, 0) + 1
        if children and native is not None:
            # The parent's stream carries no per-child usage: only the parent's totals.
            native.metrics["cost_attribution"] = {
                "parent": dict(run.usage or {}), "subagents": None,
                "method": "parent_only", "tokens_method": "parent_only"}
        return "endTurn"

    @staticmethod
    def _mcp_args(url: str, tool_names: list[str], reviewer_mode: bool,
                  preapprove_ide_tools: bool = False) -> list[str]:
        server = mcp_http.SERVER_NAME
        args = [
            "-c", f'mcp_servers.{server}.url="{url}"',
            "-c", f"mcp_servers.{server}.enabled_tools={json.dumps(tool_names)}",
        ]
        if reviewer_mode or preapprove_ide_tools:
            # Only the private, run-scoped, role-filtered IDE registry is approved.
            # The bridge still validates every call and journals project writes.
            args += [
                "-c", f'mcp_servers.{server}.default_tools_approval_mode="approve"',
            ]
        return args

    def _base_args(self, model: str | None = None, gateway_url: str | None = None,
                    concrete_model: bool = False,
                    reasoning_effort: str | None = None,
                    reviewer_mode: bool = False, native=None,
                    subagent_args: list[str] | None = None) -> list[str]:
        """`codex exec` flags shared by both surfaces. Without `native`, `read-only`
        with no shell is the locked tool surface: codex may reason, but every read and
        write goes through the IDE tools. With `native`, codex works in the registered
        workspace with its shell and without its own sandbox."""
        args = [BIN, "exec", "--json", "--skip-git-repo-check", "--ephemeral",
                "-c", "tools.web_search=false"]
        cwd = native.workspace if native is not None else os.getenv("CODEX_CWD")
        if cwd:
            args += ["--cd", cwd]
        resolved = model if concrete_model else model_for(model)
        if resolved:
            args += ["-m", resolved]
        if reasoning_effort:
            args += ["-c", f'model_reasoning_effort="{reasoning_effort}"']
        if gateway_url:
            base_url = gateway_url.rstrip("/") + "/v1"
            args += [
                "-c", 'model_provider="agentgateway"',
                "-c", 'model_providers.agentgateway.name="OpenAI via agentgateway"',
                "-c", f'model_providers.agentgateway.base_url="{base_url}"',
                "-c", 'model_providers.agentgateway.wire_api="responses"',
                # Keep the existing ChatGPT/OpenAI login; no gateway key or
                # persistent config mutation is required for this attempt.
                "-c", "model_providers.agentgateway.requires_openai_auth=true",
            ]
        # This run's subagent roles (`providers.subagents`); native runs only.
        args += subagent_args or []
        # Escape hatch for CLI-level choices we don't model (`--oss`, `--profile`,
        # extra `-c` overrides).
        args += shlex.split(os.getenv("CODEX_ARGS", ""))
        # The sandbox goes LAST on purpose: a repeated flag wins on the right, so no
        # escape-hatch value can loosen it.
        if native is not None:
            # Codex's bubblewrap sandbox cannot start in this container, so native runs
            # skip it. The container and its mounts are the only boundary, and every
            # mounted project is reachable; read-only runs are restored afterwards,
            # not prevented from writing (docs/native-execution.md).
            return args + ["--ignore-user-config",
                           "--dangerously-bypass-approvals-and-sandbox"]
        # Every agent read must pass through the request-scoped MCP boundary.
        # A read-only Codex shell could still open .env or an out-of-project file.
        args += ["--ignore-user-config", "-c", "features.shell_tool=false"]
        return args + ["-s", "read-only"]

    def _join(self, prompt: str, system: str | None) -> str:
        """`codex exec` takes no system prompt — prepend it to the instructions."""
        return f"{system}\n\n{prompt}" if system else prompt

    async def _exec(self, args: list[str], prompt: str, env: dict[str, str] | None = None,
                    report_exit: bool = False) -> AsyncIterator[dict]:
        """Run `codex exec`, feed the prompt on stdin, yield one dict per JSONL line.

        Unparseable lines are skipped rather than fatal: the CLI mixes the odd
        non-JSON diagnostic into its stream, and losing a line should not lose the run.
        """
        proc = await asyncio.create_subprocess_exec(
            *args, "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=8 * 1024 * 1024,  # agent messages can be far larger than the 64KiB default
            env={**os.environ, **env} if env else None,
        )
        log.info("codex exec: %s", " ".join(args[1:]))
        # Drain stderr concurrently — an undrained pipe fills up and stalls the child.
        errors: list[str] = []
        stderr_task = asyncio.create_task(self._drain_stderr(proc, errors))
        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt.encode())
                await proc.stdin.drain()
                proc.stdin.close()
            while True:
                try:
                    line = await proc.stdout.readline()
                except ValueError:  # line beyond the buffer limit
                    log.warning("codex line dropped: exceeded the stdout buffer limit")
                    continue
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
            await proc.wait()
            if proc.returncode:
                log.warning("codex exited rc=%s stderr=%s",
                            proc.returncode, " | ".join(errors[-10:]))
                if report_exit:
                    yield {"type": "error", "message": (
                        f"codex exited with status {proc.returncode}: "
                        + (" | ".join(errors[-3:]) or "no diagnostics"))}
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            stderr_task.cancel()

    async def _drain_stderr(self, proc, errors: list[str]) -> None:
        """Keep the child's stderr pipe moving; hold on to the tail for diagnostics."""
        while proc.stderr is not None:
            try:
                line = await proc.stderr.readline()
            except ValueError:
                continue
            if not line:
                return
            errors.append(line.decode(errors="replace").strip())
            del errors[:-20]

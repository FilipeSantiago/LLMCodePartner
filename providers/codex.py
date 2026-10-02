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
import json
import logging
import os
import shlex
from collections.abc import AsyncIterator

from mcp_bridge import mcp_http
from mcp_bridge import server as engine
from mcp_bridge.models import ErrorEvent, Event, Run, TextEvent
from mcp_bridge.registry import ToolSpec, allowed_for
from providers.base import Provider

log = logging.getLogger("mcp_bridge")

BIN = os.getenv("CODEX_BIN", "codex")

# The neutral tiers the Optimizer emits (`agent.prompt_optimizer.MODEL_CHOICES`) are
# Claude-shaped names; each provider maps them to its own models. Unset = whatever
# `CODEX_MODEL` says, and failing that the CLI's own default.
_TIER_ENV = {
    "haiku": "CODEX_MODEL_SIMPLE",
    "sonnet": "CODEX_MODEL_MEDIUM",
    "opus": "CODEX_MODEL_COMPLEX",
}


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
                return await self._run_strategy(run, prompt, system, model, gateway_url, concrete_model)

            run = engine.start_run(prompt, system, specs, strategy=strategy,
                                   allowed=allowed_for(role))
            run.metadata.update(run_metadata or {})

        async for event in self._drain_queue(run):
            yield event

    # --- internals -----------------------------------------------------------

    async def _run_strategy(self, run: Run, prompt: str, system: str | None,
                            model: str | None = None, gateway_url: str | None = None,
                            concrete_model: bool = False) -> str:
        """The engine `Strategy`: publish this Run's bridged tools, run `codex exec`
        against them, and push its output onto the Run as neutral events. Returns the
        terminal reason. The MCP endpoint is opened and closed with the process, so a
        finished (or cancelled) run leaves nothing callable behind."""
        async with mcp_http.serve_run(run) as url:
            args = self._base_args(model, gateway_url, concrete_model) + [
                "-c", f'mcp_servers.{mcp_http.SERVER_NAME}.url="{url}"',
            ]
            async for obj in self._exec(args, self._join(prompt, system)):
                event = parse_event(obj)
                if event is not None:
                    await run.put(event)
                usage = usage_of(obj)
                if usage is not None:
                    run.usage = usage
                    log.info("codex coder usage: model=%s usage=%s",
                             model_for(model) or "default", usage)
        return "endTurn"

    def _base_args(self, model: str | None = None, gateway_url: str | None = None,
                   concrete_model: bool = False) -> list[str]:
        """`codex exec` flags shared by both surfaces. `read-only` is the locked tool
        surface: codex may reason, but every write has to go through the IDE tools."""
        args = [BIN, "exec", "--json", "--skip-git-repo-check", "--ephemeral",
                "-c", "tools.web_search=false"]
        cwd = os.getenv("CODEX_CWD")
        if cwd:
            args += ["--cd", cwd]
        resolved = model if concrete_model else model_for(model)
        if resolved:
            args += ["-m", resolved]
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
        # Escape hatch for CLI-level choices we don't model (`--oss`, `--profile`,
        # extra `-c` overrides).
        args += shlex.split(os.getenv("CODEX_ARGS", ""))
        # The sandbox goes LAST on purpose: a repeated flag wins on the right, so no
        # escape-hatch value can loosen it.
        return args + ["-s", "read-only"]

    def _join(self, prompt: str, system: str | None) -> str:
        """`codex exec` takes no system prompt — prepend it to the instructions."""
        return f"{system}\n\n{prompt}" if system else prompt

    async def _exec(self, args: list[str], prompt: str) -> AsyncIterator[dict]:
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

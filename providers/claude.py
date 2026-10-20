"""Claude provider (local `claude` CLI via the Agent SDK).

`ClaudeProvider` implements the `Provider` surface:
  - plain chat: `stream` (and inherited `complete`) — no tools, no filesystem.
  - tool calling: `tools(messages, specs)` exposes the JetBrains tools as an
    in-process MCP server (each handler bridges out via `bridge.call_tool`) and
    drives a background Run; the stateful start-vs-resume decision is internal.

This is the only module that imports `claude_agent_sdk`.
"""
import logging
import os
import re
from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from mcp_bridge import bridge
from mcp_bridge.bridge import TOOL_TIMEOUT
from mcp_bridge import server as engine
from mcp_bridge.models import Event, Run, TextEvent
from mcp_bridge.registry import ROLE_CODER, ToolSpec, allowed_for, native_bridged_specs
from providers import compaction
from providers.kb_gate import NativeKbGate
from providers.base import Provider

log = logging.getLogger("codepartner.providers.claude")

SERVER_NAME = "jetbrains"
# Built-in Claude Code tools for native execution in a registered workspace.
NATIVE_TOOLS = ("Read", "Grep", "Glob", "Edit", "MultiEdit", "Write", "Bash")
NATIVE_READ_TOOLS = ("Read", "Grep", "Glob")
NATIVE_WRITE_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})
NATIVE_DISALLOWED = ("WebFetch", "WebSearch")


def record_native_tool(run: Run, name: str, arguments: dict | None) -> None:
    """Count a native tool call and the project file it targets (run telemetry)."""
    metrics = run.metadata.setdefault("native_tool_metrics", run.native.metrics if run.native
                                      else {"tool_calls": 0, "tools": {}, "files": []})
    metrics["tool_calls"] += 1
    metrics["tools"][name] = metrics["tools"].get(name, 0) + 1
    if name.rsplit("__", 1)[-1].startswith("kb_"):
        metrics["kb_calls"] = metrics.get("kb_calls", 0) + 1
    path = (arguments or {}).get("file_path") or (arguments or {}).get("path")
    if isinstance(path, str) and path not in metrics["files"]:
        metrics["files"].append(path)
    if name in NATIVE_WRITE_TOOLS and run.native is not None:
        run.native.record_write((arguments or {}).get("file_path"))


def record_compaction(run: Run, data: dict) -> None:
    """Record a compaction the CLI actually performed (not one merely configured)."""
    meta = data.get("compact_metadata") or data
    entry = {"trigger": meta.get("trigger"), "pre_tokens": meta.get("pre_tokens")}
    run.metadata.setdefault("compactions", []).append(entry)
    log.info("claude native compaction: %s", entry)


def bash_exit_code(result: ToolResultBlock) -> int | None:
    """Exit status of a native Bash result. The CLI reports success as
    `is_error=False` with no code, and failure as content starting `Exit code N`;
    an error without a code (interrupted, timed out) is unknown."""
    if not result.is_error:
        return 0
    text = _block_text(result.content)
    match = re.match(r"\s*(?:Error:\s*)?Exit code (-?\d+)", text)
    return int(match.group(1)) if match else None


def _block_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content
                         if isinstance(item, dict))
    return ""


class ClaudeResultError(RuntimeError):
    """The CLI finished with an error result; carries its real explanation."""


def _result_error_text(message: ResultMessage) -> str | None:
    """Explain an error result. The SDK's own exception only names the subtype,
    which is "success" when the final turn was an API or tool-call-parse error."""
    if not message.is_error:
        return None
    text = "; ".join(message.errors or []) or (message.result or "").strip() or message.subtype
    if message.api_error_status:
        text = f"{text} (API status {message.api_error_status})"
    log.warning("claude error result: subtype=%s api_status=%s text=%r",
                message.subtype, message.api_error_status, text[:500])
    return text


async def _query_messages(prompt: str, options: ClaudeAgentOptions):
    """`query()` that re-raises the SDK's trailing exception with the reported reason."""
    reported: str | None = None
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                reported = _result_error_text(message)
            yield message
    except Exception as exc:
        if reported is not None:
            raise ClaudeResultError(f"Claude Code returned an error result: {reported}") from exc
        raise


class ClaudeProvider(Provider):
    NAME = "claude"

    async def stream(self, prompt: str, system: str | None = None
                     ) -> AsyncIterator[tuple[str, str | None]]:
        async for message in _query_messages(prompt, self._plain_options(system)):
            if isinstance(message, AssistantMessage):
                for b in message.content:
                    if isinstance(b, TextBlock):
                        yield b.text, None
            elif isinstance(message, ResultMessage):
                yield "", message.terminal_reason

    async def tools(self, messages: list[dict], specs: list[ToolSpec],
                    model: str | None = None, role: str | None = None,
                    run_metadata: dict | None = None,
                    execution_env: dict[str, str] | None = None,
                    direct_tool_handlers: dict | None = None,
                    kb_first=None, native=None, **kwargs) -> AsyncIterator[Event]:
        """Stateful, but the start-vs-resume decision is private: a history carrying
        tool results resolves the pending futures and continues the SAME background
        run; otherwise a fresh run is started. Either way, yields the same neutral
        events every provider's `tools()` emits. `model` selects the Claude tier
        ("haiku"/"sonnet"/"opus"/full id) for a fresh run; None = CLI default. `role`
        selects the tool allowlist for a fresh run; a resume inherits the role already
        baked into the Run."""
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
                return await self._run_strategy(run, prompt, system, specs, model, execution_env)

            if native is not None:
                # Native file tools replace their IDE equivalents; IDE-specific
                # and KB tools stay bridged.
                specs = native_bridged_specs(specs)
            run = engine.start_run(prompt, system, specs, strategy=strategy,
                                   allowed=allowed_for(role),
                                   direct_tool_handlers=direct_tool_handlers,
                                   agent_tool_interface=role == ROLE_CODER and native is None)
            run.metadata.update(run_metadata or {})
            run.kb_first = kb_first
            run.native = native

        async for event in self._drain_queue(run):
            yield event

    # --- internals -----------------------------------------------------------

    def _plain_options(self, system: str | None) -> ClaudeAgentOptions:
        # `tools=[]` removes built-in Read/Bash/etc. — no backend filesystem access.
        return ClaudeAgentOptions(system_prompt=system, tools=[], allowed_tools=[], mcp_servers={})

    @staticmethod
    def _kb_gate(run: Run, specs: list[ToolSpec], native) -> NativeKbGate | None:
        """KB-first nudge for a native run that asked for it and was offered the KB."""
        metadata = getattr(run, "metadata", None) or {}
        if native is None or not metadata.get("kb_gate"):
            return None
        if not any(spec.name == "kb_search_graph" for spec in specs):
            return None
        return NativeKbGate(f"mcp__{SERVER_NAME}__kb_search_graph",
                            int(metadata.get("kb_gate_max") or 5))

    @staticmethod
    def _gate_hook(gate: NativeKbGate, workspace: str | None):
        async def hook(input_data: dict, _tool_use_id, _context) -> dict:
            name = str(input_data.get("tool_name") or "")
            gate.observe(name)
            reason = gate.decide(name, input_data.get("tool_input") or {}, workspace)
            if reason is None:
                return {}
            log.info("native kb gate advised: tool=%s", name)
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                           "permissionDecision": "deny",
                                           "permissionDecisionReason": reason}}

        return hook

    def _make_tool(self, run: Run, spec: ToolSpec):
        @tool(spec.name, spec.description, spec.schema)
        async def _handler(args: dict[str, Any], _run: Run = run, _name: str = spec.name):
            try:
                text = await bridge.call_tool(_run, _name, args)
            except TimeoutError:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: tool '{_name}' timed out after {bridge.TOOL_TIMEOUT}s",
                    }],
                    "is_error": True,
                }
            return {"content": [{"type": "text", "text": text}]}

        return _handler

    async def _run_strategy(self, run: Run, prompt: str, system: str | None,
                            specs: list[ToolSpec], model: str | None = None,
                            execution_env: dict[str, str] | None = None) -> str:
        """The engine `Strategy`: drive the model in-process, pushing TextEvents onto
        the Run and (via `_make_tool` → `bridge.call_tool`) parking a future per tool
        call. Returns the terminal reason. Runs as the Run's background task. `model`
        selects the Claude tier (None = CLI default). Captures token/cost usage from the
        terminal `ResultMessage` onto the Run for the engine's DoneEvent."""
        server = create_sdk_mcp_server(
            name=SERVER_NAME, version="1.0.0",
            tools=[self._make_tool(run, s) for s in specs],
        )
        allowed = [f"mcp__{SERVER_NAME}__{s.name}" for s in specs]
        native = getattr(run, "native", None)
        builtin: list[str] = []
        native_options: dict[str, Any] = {}
        if native is not None:
            builtin = list(NATIVE_READ_TOOLS if native.read_only else NATIVE_TOOLS)
            # Read/Grep/Glob cannot write, so a read-only run's changes are not its own.
            native.agent_can_write = not native.read_only
            native_options = {"cwd": native.workspace,
                              "disallowed_tools": list(NATIVE_DISALLOWED),
                              "permission_mode": "default" if native.read_only
                              else "acceptEdits"}
        gate = self._kb_gate(run, specs, native)
        if gate is not None:
            native_options["hooks"] = {"PreToolUse": [HookMatcher(
                matcher=None, hooks=[self._gate_hook(gate, native.workspace)])]}
        options = ClaudeAgentOptions(
            system_prompt=system, tools=builtin, allowed_tools=[*builtin, *allowed],
            mcp_servers={SERVER_NAME: server}, model=model,
            # The Agent SDK may treat ``env`` as the complete subprocess
            # environment. Preserve the authenticated CLI session while adding
            # a gateway override.
            # The CLI reads MCP_TOOL_TIMEOUT in milliseconds while CodePartner's
            # setting is in seconds; without this the CLI abandons IDE tools after ~0.1s.
            env={**os.environ, "MCP_TOOL_TIMEOUT": str(TOOL_TIMEOUT * 1000),
                 **(native.env if native is not None else {}),
                 **(execution_env or {})},
            # Native auto-compaction threshold and preservation brief for this run only.
            settings=compaction.claude_settings(
                (getattr(run, "metadata", None) or {}).get("compaction_brief")),
            **native_options,
        )
        if isinstance(getattr(run, "metadata", None), dict):
            run.metadata["context_compaction"] = compaction.requested("claude autoCompactWindow")
        terminal = "endTurn"
        commands: dict[str, str] = {}  # pending native Bash tool_use id -> command
        async for message in _query_messages(prompt, options):
            if isinstance(message, AssistantMessage):
                for b in message.content:
                    if isinstance(b, TextBlock) and b.text:
                        await run.put(TextEvent(b.text))
                    elif isinstance(b, ToolUseBlock) and native is not None:
                        record_native_tool(run, b.name, b.input)
                        if b.name == "Bash":
                            commands[b.id] = (b.input or {}).get("command")
            elif (isinstance(message, UserMessage) and native is not None
                  and isinstance(message.content, list)):
                for b in message.content:
                    if isinstance(b, ToolResultBlock) and b.tool_use_id in commands:
                        await native.command_finished(
                            commands.pop(b.tool_use_id), _block_text(b.content),
                            bash_exit_code(b))
            elif isinstance(message, SystemMessage) and message.subtype == "compact_boundary":
                record_compaction(run, message.data)
            elif isinstance(message, ResultMessage):
                terminal = message.terminal_reason
                run.usage = message.usage
                run.model_usage = message.model_usage
                run.metadata["cost_usd"] = message.total_cost_usd
                if gate is not None:
                    run.metadata.setdefault(
                        "native_tool_metrics", native.metrics)["kb_gate"] = gate.summary()
                log.info(
                    "claude coder usage: model=%s cost_usd=%s model_usage=%s",
                    model or "default", message.total_cost_usd, message.model_usage,
                )
        return terminal

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
    AgentDefinition,
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
from mcp_bridge.activity import describe_native
from mcp_bridge.models import Event, ProgressEvent, Run, TextEvent
from mcp_bridge.registry import (KB_NAVIGATION_TOOLS, ROLE_CODER, ToolSpec, allowed_for,
                                   native_bridged_specs)
from providers import compaction, scope_guard, subagents
from providers.base import Provider
from telemetry import tracing

log = logging.getLogger("codepartner.providers.claude")

SERVER_NAME = "jetbrains"
# Built-in Claude Code tools for native execution in a registered workspace.
NATIVE_TOOLS = ("Read", "Grep", "Glob", "Edit", "MultiEdit", "Write", "Bash")
NATIVE_READ_TOOLS = ("Read", "Grep", "Glob")
SCOPED_CODER_TOOLS = ("Read", "Edit", "MultiEdit", "Write", "Agent")
SCOPED_REVIEWER_TOOLS = ("Read", "Agent")
NATIVE_WRITE_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})
NATIVE_DISALLOWED = ("WebFetch", "WebSearch")
# A subagent executes a fully specified request with one basic tool: the parent locates
# code (KB, Glob/Grep) and hands over paths or commands. No subagent gets KB tools.
SUBAGENT_TOOLS = {subagents.READER: ("Read",), subagents.CHECKS: ("Bash",)}
# The built-in tool that starts a subagent defined in `ClaudeAgentOptions.agents`.
AGENT_TOOL = "Agent"


def _add_usage(total: dict | None, part: dict | None) -> dict:
    """Add independently reported CLI result segments without losing nested tokens."""
    combined = dict(total or {})
    for key, value in (part or {}).items():
        previous = combined.get(key)
        if isinstance(value, bool):
            combined[key] = value
        elif isinstance(value, (int, float)):
            combined[key] = (previous if isinstance(previous, (int, float))
                             and not isinstance(previous, bool) else 0) + value
        elif isinstance(value, dict):
            combined[key] = _add_usage(previous if isinstance(previous, dict) else None,
                                       value)
        elif isinstance(value, list):
            combined[key] = [*(previous if isinstance(previous, list) else []), *value]
        else:
            combined[key] = value
    return combined


def record_native_tool(run: Run, name: str, arguments: dict | None) -> None:
    """Count a native tool call and the project file it targets (run telemetry)."""
    metrics = run.metadata.setdefault("native_tool_metrics", run.native.metrics if run.native
                                      else {"tool_calls": 0, "tools": {}, "files": []})
    metrics["tool_calls"] += 1
    metrics["tools"][name] = metrics["tools"].get(name, 0) + 1
    if name.rsplit("__", 1)[-1].startswith("kb_"):
        metrics["kb_calls"] = metrics.get("kb_calls", 0) + 1
    path = (arguments or {}).get("file_path") or (arguments or {}).get("path")
    if isinstance(path, str) and run.native is not None:
        path = run.native.relative(path) or path
    if isinstance(path, str) and path not in metrics["files"]:
        metrics["files"].append(path)
    if name in NATIVE_WRITE_TOOLS and run.native is not None:
        run.native.record_write((arguments or {}).get("file_path"))
    # Source reads: Read, content searches, and shell prints of project files.
    reads = ([path] if name == "Read" else
             [path or "."] if name == "Grep"
             and (arguments or {}).get("output_mode") == "content" else
             [run.native.relative(read) or read if run.native is not None else read
              for read, _ in subagents.classify_command((arguments or {}).get("command"))[1]]
             if name == "Bash" else [])
    subagents.note_scope(run.metadata, metrics, reads)


# Run kinds whose progress lines reach the chat (coder, planner, reviewer).
_PROGRESS_KINDS = ("task_execution", "planner", "reviewer")


def _shows_progress(run: Run) -> bool:
    metadata = getattr(run, "metadata", None) or {}
    return any(metadata.get(kind) for kind in _PROGRESS_KINDS)


async def _progress(run: Run, name: str, arguments: dict | None,
                    workspace: str | None) -> None:
    """One chat line for a parent's native tool call; subagent calls never get one."""
    if name == AGENT_TOOL:
        arguments = arguments or {}
        line = subagents.describe_spawn(arguments.get("subagent_type"),
                                        arguments.get("prompt"))
    else:
        line = describe_native(name, arguments, workspace)
        if line is None:
            return
        line = ("✏️ " if name in NATIVE_WRITE_TOOLS else "🔎 ") + line
    await run.put(ProgressEvent(f"- {line}\n"))


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
                    native=None, subagents=None,
                    reasoning_effort: str | None = None,
                    **kwargs) -> AsyncIterator[Event]:
        """Stateful, but the start-vs-resume decision is private: a history carrying
        tool results resolves the pending futures and continues the SAME background
        run; otherwise a fresh run is started. Either way, yields the same neutral
        events every provider's `tools()` emits. `model` selects the Claude tier
        ("haiku"/"sonnet"/"opus"/full id) for a fresh run; None = CLI default. `role`
        selects the tool allowlist for a fresh run; a resume inherits the role already
        baked into the Run. `reasoning_effort` sets the CLI's effort for a fresh run;
        None keeps the CLI default."""
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
                return await self._run_strategy(run, prompt, system, specs, model, execution_env,
                                                reasoning_effort)

            if native is not None:
                # Native file tools replace their IDE equivalents; IDE-specific
                # and KB tools stay bridged.
                specs = native_bridged_specs(specs)
            run = engine.start_run(prompt, system, specs, strategy=strategy,
                                   allowed=allowed_for(role),
                                   direct_tool_handlers=direct_tool_handlers,
                                   agent_tool_interface=role == ROLE_CODER and native is None,
                                   trace_kind="llm", trace_provider=self.NAME,
                                   trace_model=model)
            run.metadata.update(run_metadata or {})
            run.native = native
            run.subagents = subagents

        async for event in self._drain_queue(run):
            yield event

    # --- internals -----------------------------------------------------------

    def _plain_options(self, system: str | None) -> ClaudeAgentOptions:
        # `tools=[]` removes built-in Read/Bash/etc. — no backend filesystem access.
        return ClaudeAgentOptions(system_prompt=system, tools=[], allowed_tools=[], mcp_servers={})

    @staticmethod
    def _agent_definitions(children: list[subagents.Subagent],
                           request: subagents.Request) -> dict[str, AgentDefinition]:
        """Claude subagents for this run on the configured reader model, each with its
        one tool (`SUBAGENT_TOOLS`). `background=False`: the brief is the Agent tool's
        result, so the parent never polls for it or redoes the work."""
        model, effort = request.model_for(ClaudeProvider.NAME)
        return {child.name: AgentDefinition(
                    description=child.description, prompt=child.prompt,
                    tools=list(SUBAGENT_TOOLS[child.name]),
                    model=model or subagents.DEFAULT_CLAUDE_MODEL,
                    maxTurns=request.turns(), background=False,
                    **({"effort": effort} if effort else {}))
                for child in children}

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
                            execution_env: dict[str, str] | None = None,
                            reasoning_effort: str | None = None) -> str:
        """The engine `Strategy`: drive the model in-process, pushing TextEvents onto
        the Run and (via `_make_tool` → `bridge.call_tool`) parking a future per tool
        call. Returns the terminal reason. Runs as the Run's background task. `model`
        selects the Claude tier (None = CLI default). Captures token/cost usage from the
        terminal `ResultMessage` onto the Run for the engine's DoneEvent."""
        native = getattr(run, "native", None)
        request = getattr(run, "subagents", None)
        children = subagents.resolve(request, native)
        policy = scope_guard.policy_from_run(run, children)
        if policy is None and children and native is not None:
            # Agent calls are normalized on every run that offers subagents.
            policy = scope_guard.ScopePolicy(
                "other", native.workspace, subagent_types=scope_guard.subagent_types(children))
        scoped = policy is not None and policy.active
        # No unrestricted MCP file or KB content tools on a scoped parent, only KB
        # navigation (paths, symbols, line ranges). Its reader subagent has the CLI's
        # separate, read-only tool definition.
        parent_specs = ([spec for spec in specs if spec.name in KB_NAVIGATION_TOOLS]
                        if scoped else specs)
        server = create_sdk_mcp_server(
            name=SERVER_NAME, version="1.0.0",
            tools=[self._make_tool(run, s) for s in parent_specs],
        )
        allowed = [f"mcp__{SERVER_NAME}__{s.name}" for s in parent_specs]
        builtin: list[str] = []
        native_options: dict[str, Any] = {}
        if native is not None:
            builtin = list((SCOPED_REVIEWER_TOOLS if policy.role == "reviewer"
                            else SCOPED_CODER_TOOLS) if scoped else
                           (NATIVE_READ_TOOLS if native.read_only else NATIVE_TOOLS))
            # Read/Grep/Glob cannot write, so a read-only run's changes are not its own.
            native.agent_can_write = not native.read_only
            native_options = {"cwd": native.workspace,
                              "disallowed_tools": list(NATIVE_DISALLOWED),
                              "permission_mode": "default" if native.read_only
                              else "acceptEdits"}
            if children:
                if AGENT_TOOL not in builtin:
                    builtin.append(AGENT_TOOL)
                # A subagent can only use tools that exist in the session; the scope hook
                # keeps the parent itself off Bash.
                if (any(child.name == subagents.CHECKS for child in children)
                        and "Bash" not in builtin):
                    builtin.append("Bash")
                native_options["agents"] = self._agent_definitions(children, request)
        if policy is not None:
            hook_state: dict = {}

            async def scope_hook(event: dict, _tool_use_id: str | None,
                                 _context: object) -> dict:
                decision = scope_guard.decide(event, policy, hook_state)
                if native is not None:
                    scope_guard.record_denial(native.metrics, event, decision)
                return decision or {}

            native_options["hooks"] = {"PreToolUse": [HookMatcher(
                matcher=".*", hooks=[scope_hook])]}
        options = ClaudeAgentOptions(
            system_prompt=system, tools=builtin,
            allowed_tools=[*builtin, *allowed],
            # Only Code Partner's server: never the account's claude.ai connectors or
            # any user/project MCP config the CLI would otherwise load.
            mcp_servers={SERVER_NAME: server}, strict_mcp_config=True, model=model,
            # The Agent SDK may treat ``env`` as the complete subprocess
            # environment. Preserve the authenticated CLI session while adding
            # a gateway override.
            # The CLI reads MCP_TOOL_TIMEOUT in milliseconds while CodePartner's
            # setting is in seconds; without this the CLI abandons IDE tools after ~0.1s.
            env={**os.environ, "MCP_TOOL_TIMEOUT": str(TOOL_TIMEOUT * 1000),
                 **(native.env if native is not None else {}),
                 # Only CodePartner's reader/checks subagents, never the built-in ones.
                 **({"CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1"} if children else {}),
                 **(execution_env or {})},
            # Native auto-compaction threshold and preservation brief for this run only.
            settings=compaction.claude_settings(
                (getattr(run, "metadata", None) or {}).get("compaction_brief")),
            **({"effort": reasoning_effort} if reasoning_effort else {}),
            **native_options,
        )
        if isinstance(getattr(run, "metadata", None), dict):
            run.metadata["context_compaction"] = compaction.requested("claude autoCompactWindow")
        terminal = "endTurn"
        commands: dict[str, str] = {}  # pending native Bash tool_use id -> command
        spawned: dict[str, str | None] = {}  # Agent tool_use id -> role; its result is a brief
        shows_progress = _shows_progress(run)
        tracker = (subagents.Tracker(native.metrics, request, offered=bool(children),
                                     native=native)
                   if native is not None else None)
        if native is not None and isinstance(getattr(run, "metadata", None), dict):
            run.metadata["exploration_mode"] = "reader" if children else "off"
        assistant_seen = False
        tracing.start_turn(run, model)
        async for message in _query_messages(prompt, options):
            if (isinstance(message, AssistantMessage) and message.parent_tool_use_id
                    and children and tracker is not None):
                # A subagent's turn: its text and tool calls stay out of this run's
                # output and turn telemetry (its cost is in `model_usage`). Its shell
                # commands still become execution evidence.
                for b in message.content:
                    if isinstance(b, ToolUseBlock):
                        tracker.child_tool(b.name, b.input)
                        if b.name == "Bash":
                            commands[b.id] = (b.input or {}).get("command")
                continue
            if isinstance(message, AssistantMessage):
                if assistant_seen:
                    tracing.start_turn(run, message.model)
                assistant_seen = True
                tracing.record_turn_usage(run, message.usage, message.model)
                if tracker is not None:
                    tracker.parent_reads([
                        ((b.input or {}).get("file_path"), (b.input or {}).get("offset"),
                         (b.input or {}).get("limit"))
                        for b in message.content
                        if isinstance(b, ToolUseBlock) and b.name == "Read"])
                for b in message.content:
                    if isinstance(b, TextBlock) and b.text:
                        if native is not None:
                            subagents.note_scope(run.metadata, run.metadata.setdefault(
                                "native_tool_metrics", native.metrics), text=b.text)
                        await run.put(TextEvent(b.text))
                    elif isinstance(b, ToolUseBlock) and native is not None:
                        record_native_tool(run, b.name, b.input)
                        if shows_progress:
                            await _progress(run, b.name, b.input, native.workspace)
                        if b.name == "Bash":
                            commands[b.id] = (b.input or {}).get("command")
                            if tracker is not None:
                                tracker.parent_command(commands[b.id])
                        elif tracker is not None and b.name in ("Edit", "Write"):
                            tracker.parent_edit((b.input or {}).get("file_path"))
                        elif tracker is not None and b.name == AGENT_TOOL:
                            spawned[b.id] = (b.input or {}).get("subagent_type")
                            tracker.spawned((b.input or {}).get("subagent_type"),
                                            (b.input or {}).get("prompt"),
                                            background=bool((b.input or {}).get("run_in_background"))
                                            and policy is None,
                                            requested_background=bool((b.input or {}).get(
                                                "run_in_background")),
                                            description=(b.input or {}).get("description"))
            elif (isinstance(message, UserMessage) and native is not None
                  and isinstance(message.content, list)):
                for b in message.content:
                    if isinstance(b, ToolResultBlock) and b.tool_use_id in commands:
                        await native.command_finished(
                            commands.pop(b.tool_use_id), _block_text(b.content),
                            bash_exit_code(b))
                    elif (isinstance(b, ToolResultBlock) and tracker is not None
                          and b.tool_use_id in spawned):
                        role = spawned.pop(b.tool_use_id)
                        tracker.brief(_block_text(b.content), role)
                        if shows_progress:
                            await run.put(ProgressEvent(
                                f"- {subagents.describe_brief(role, _block_text(b.content))}\n"))
                        result = message.tool_use_result
                        if isinstance(result, dict):
                            tracker.child_usage(result.get("usage"))
            elif isinstance(message, SystemMessage) and message.subtype == "compact_boundary":
                record_compaction(run, message.data)
            elif isinstance(message, ResultMessage):
                terminal = message.terminal_reason
                run.usage = _add_usage(run.usage, message.usage)
                run.usage["total_cost_usd"] = message.total_cost_usd
                run.model_usage = message.model_usage
                run.metadata["cost_usd"] = message.total_cost_usd
                if native is not None:
                    native.metrics["result_segments"] = native.metrics.get("result_segments", 0) + 1
                if children and tracker is not None:
                    native.metrics["cost_attribution"] = subagents.cost_attribution(
                        message.model_usage,
                        request.model_for(self.NAME)[0] or subagents.DEFAULT_CLAUDE_MODEL,
                        tracker.data and tracker.data.get("child_usage"))
                log.info(
                    "claude coder usage: model=%s cost_usd=%s model_usage=%s",
                    model or "default", message.total_cost_usd, message.model_usage,
                )
        return terminal

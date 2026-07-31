# LLMCodePartner

A minimal FastAPI backend that exposes an **OpenAI-compatible** API
(`POST /v1/chat/completions`, `GET /v1/models`) and bridges it to a **local
Claude/Codex** via the Claude Agent SDK (uses the installed `claude` CLI's
login — no `ANTHROPIC_API_KEY` required).

Run it:

```bash
uv run main.py          # serves on http://0.0.0.0:7777
```

- Raw request logging: `.env` toggle `RAW_LOG_ENABLED=true`.
- Tool-call timeout: `.env` `MCP_TOOL_TIMEOUT=120` (seconds).
- LLM provider: `.env` `LLM_PROVIDER=claude|ollama` (see Providers).

## Providers

The LLM is pluggable via `LLM_PROVIDER` (`providers/`). **Both** providers can do
plain chat *and* drive the JetBrains tools through the same bridge — the only
difference is how a tool call is intercepted:

| Provider | Chat | Tool interception |
|---|---|---|
| `claude` | local `claude` CLI (Agent SDK) | in-process **MCP server**; the SDK calls a Python handler mid-generation |
| `ollama` | local Ollama `/api/chat` (`OLLAMA_HOST`/`OLLAMA_MODEL`) | **native tool calling**: a chat loop that sends `tools`, reads `message.tool_calls`, and feeds `role:"tool"` results back |

Both strategies converge on `mcp_bridge.bridge.call_tool` (mint id → emit a
`ToolCallEvent` → await the JetBrains result), so the events/SSE/resume layer and
JetBrains itself are identical regardless of provider. The engine (`mcp_bridge/`)
is provider-neutral — it imports no LLM SDK; the model-specific run strategy lives
in each `providers/<name>.py` and is injected by the controller.

## How Claude reaches the project (MCP bridge over JetBrains tools)

The backend has **no filesystem access** and never receives a project path — no
`cwd`, no absolute root. Claude/Codex reads the **active JetBrains project only
through JetBrains' own tools**, advertised by the client in the request `tools`
array. Supported tools (registered per request):

- `list_directory_tree`
- `get_file_text_by_path`
- `search_in_files_by_text`

### Final flow

```
JetBrains Chat
  → OpenAI-compatible backend        (POST /v1/chat/completions, tools: [...])
    → Claude/Codex                   (tools exposed via an in-process MCP server)
      → MCP tool request             (Claude calls e.g. get_file_text_by_path)
        → OpenAI-compatible tool call (streamed back: delta.tool_calls + finish_reason:"tool_calls")
          → JetBrains executes it against the ACTIVE project
            → tool result returns     (next request: {role:"tool", tool_call_id, content})
              → Claude/Codex resumes  (same execution continues; answer streams out)
```

Because the tool runs in the IDE, a call is bounced out to JetBrains and its
result arrives on the *next* HTTP request. The in-flight Claude execution is held
open in the meantime by an `asyncio.Future` keyed by `tool_call_id`.

### Notes

- **Per-request isolation:** each execution has its own `Run` with its own
  `ToolRegistry` (`mcp_bridge/registry.py`) — no global tool registry that could
  mix concurrent requests. Pending futures are keyed by unique `tool_call_id`.
- **Same tool name + args:** the MCP tool mirrors the name, description and JSON
  schema JetBrains advertised, so the emitted `tool_call` matches what the IDE
  expects.
- **Multiple sequential tool calls:** each tool call ends the current SSE
  response with `finish_reason:"tool_calls"`; the next request resumes the same
  run and surfaces the next call (or the final answer) — N round trips, one run.
- **Timeout / cleanup:** if JetBrains doesn't return a result within
  `MCP_TOOL_TIMEOUT`, the call fails with a clear timeout tool result; pending
  futures are cancelled and cleaned up on completion, cancellation, or timeout.
- **No supported tools → plain chat:** requests without a supported JetBrains
  tool behave as normal OpenAI chat (streaming or non-streaming); tool calling is
  disabled.
- **State:** in process memory only — no database, no auth, no IDE plugin.
- **Logs (`mcp_bridge`):** tools received · mcp tool registration · tool call
  emitted · tool result received · tool timeout · cleanup.

All MCP/bridge code lives in the `mcp_bridge/` package
(`registry.py`, `models.py`, `bridge.py`, `server.py`). It is named `mcp_bridge`
rather than `mcp` on purpose: a top-level `mcp/` package would shadow the
installed `mcp` library the Claude Agent SDK imports.

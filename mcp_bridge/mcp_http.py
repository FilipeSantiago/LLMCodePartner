"""Per-run MCP endpoint over streamable HTTP (for out-of-process providers).

Claude hosts its bridged tools *in process* (the Agent SDK accepts an in-process
MCP server), so its handlers can call `bridge.call_tool` directly. A provider that
runs as a separate process — the `codex` CLI — can only reach tools over a real MCP
transport. This module is that transport, and nothing more: it exposes ONE Run's
already-registered tools at a private URL, and each call lands in the same
`bridge.call_tool` every provider uses.

The guardrails are therefore unchanged, not re-implemented:
  - the tool list is `run.registry.tools()` — the role-filtered set `start_run`
    already accepted, never the raw specs the IDE advertised;
  - a call parks on a bridge future and surfaces a `ToolCallEvent`, so the IDE (and
    the human) still execute it, bounded by `bridge.TOOL_TIMEOUT`;
  - the endpoint lives exactly as long as the Run's background task, and its token
    is unguessable, so a finished run leaves nothing callable behind.
"""
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from urllib.parse import urlparse
from uuid import uuid4

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import PlainTextResponse

from mcp_bridge import bridge
from mcp_bridge.models import Run

log = logging.getLogger("mcp_bridge")

SERVER_NAME = "jetbrains"

# Where an out-of-process provider should reach this app. Read at import time,
# like `bridge.TOOL_TIMEOUT` — `main.py` loads `.env` before importing us.
PUBLIC_BASE = os.getenv("MCP_PUBLIC_BASE", "http://127.0.0.1:7777").rstrip("/")

# Live endpoints: token -> session manager. Same routing-table shape `bridge` uses
# for pending calls; entries exist only while their Run is running.
_endpoints: dict[str, StreamableHTTPSessionManager] = {}


def _security() -> TransportSecuritySettings:
    """Keep the SDK's DNS-rebinding protection on, scoped to loopback and whatever
    host we hand out. The endpoint is meant for a child process on this machine."""
    host = urlparse(PUBLIC_BASE).netloc
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[h for h in (host, "127.0.0.1:*", "localhost:*") if h],
        allowed_origins=[h for h in (PUBLIC_BASE, "http://127.0.0.1:*", "http://localhost:*") if h],
    )


def _build_server(run: Run) -> Server:
    """An MCP server whose whole surface is this Run's bridged tools."""
    server: Server = Server(SERVER_NAME, version="1.0.0")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        # run.registry is the enforcement point: only what this run's role allows.
        log.info("mcp http tool list served: %s", run.registry.names())
        return [
            types.Tool(name=s.name, description=s.description, inputSchema=s.schema)
            for s in run.registry.tools()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> types.CallToolResult:
        if name not in run.registry.names():
            return _error(f"Error: tool '{name}' is not available to this agent")
        try:
            text = await bridge.call_tool(run, name, arguments or {})
        except TimeoutError:
            return _error(f"Error: tool '{name}' timed out after {bridge.TOOL_TIMEOUT}s")
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    return server


def _error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], isError=True)


@contextlib.asynccontextmanager
async def serve_run(run: Run) -> AsyncIterator[str]:
    """Publish this Run's tools at a private URL for the body of the context.

    Entered from inside the Run's background task, so the endpoint's lifetime is the
    run's: when the strategy returns, raises, or is cancelled, the token is dropped
    and the session manager is torn down.
    """
    token = uuid4().hex
    manager = StreamableHTTPSessionManager(
        app=_build_server(run), stateless=True, security_settings=_security(),
    )
    async with manager.run():
        _endpoints[token] = manager
        log.info("mcp http endpoint open token=%s tools=%s", token[:8], run.registry.names())
        try:
            yield f"{PUBLIC_BASE}/mcp/{token}"
        finally:
            _endpoints.pop(token, None)
            log.info("mcp http endpoint closed token=%s", token[:8])


def _token_of(scope) -> str:
    """The `<token>` of `/mcp/<token>`. A mount hands us the full path plus the mount
    prefix in `root_path` (per ASGI), so strip that before reading the first segment."""
    path = scope.get("path", "")
    root = scope.get("root_path", "")
    if root and path.startswith(root):
        path = path[len(root):]
    return path.strip("/").split("/")[0]


async def asgi_app(scope, receive, send) -> None:
    """ASGI entrypoint mounted at `/mcp` — routes `/mcp/<token>` to its Run."""
    if scope["type"] != "http":
        await PlainTextResponse("not found", status_code=404)(scope, receive, send)
        return
    token = _token_of(scope)
    manager = _endpoints.get(token)
    if manager is None:
        log.warning("mcp http request for unknown/expired endpoint token=%s", token[:8])
        await PlainTextResponse("unknown mcp endpoint", status_code=404)(scope, receive, send)
        return
    await manager.handle_request(scope, receive, send)

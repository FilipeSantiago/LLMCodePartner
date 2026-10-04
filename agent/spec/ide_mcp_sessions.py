"""Conversation-scoped configuration and tool discovery for the IDE MCP server."""

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import urlparse
from uuid import uuid4

from agent.routing.jetbrains_tools import DirectIdeTools
from agent.spec.jetbrains_mcp import JetBrainsMcpClient, JetBrainsMcpError

SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 256
log = logging.getLogger("codepartner.ide_mcp")


@dataclass(frozen=True)
class IdeMcpSession:
    conversation_id: str
    client: JetBrainsMcpClient
    tools: DirectIdeTools
    expires_at: float


_sessions: OrderedDict[str, IdeMcpSession] = OrderedDict()
_bootstrap_calls: dict[str, str] = {}


def new_conversation_id() -> str:
    return "conv_" + uuid4().hex


def conversation_for_request(request) -> str | None:
    if request.conversation_id:
        return request.conversation_id
    for message in request.messages:
        for call in message.tool_calls or []:
            call_id = call.get("id")
            if call_id in _bootstrap_calls:
                return _bootstrap_calls[call_id]
        if message.tool_call_id in _bootstrap_calls:
            return _bootstrap_calls[message.tool_call_id]
    return None


def remember_bootstrap_call(tool_call_id: str, conversation_id: str) -> None:
    _bootstrap_calls[tool_call_id] = conversation_id


def remember_bootstrap_from_history(request, conversation_id: str) -> None:
    for message in request.messages:
        for call in message.tool_calls or []:
            call_id = call.get("id")
            function = call.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                args = {}
            if function.get("name") in {"read_file", "get_file_text_by_path"} and (
                    args.get("file_path") == ".codepartner/ide_mcp.json"
                    or args.get("path") == ".codepartner/ide_mcp.json"):
                if call_id:
                    remember_bootstrap_call(call_id, conversation_id)


def strip_bootstrap_exchange(messages: list[dict]) -> list[dict]:
    """Remove config-read transport messages before handing chat history to a model."""
    known_ids = set(_bootstrap_calls)
    cleaned = []
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") in known_ids:
            continue
        calls = message.get("tool_calls")
        if calls:
            remaining = [call for call in calls if call.get("id") not in known_ids]
            message = dict(message)
            if remaining:
                message["tool_calls"] = remaining
            else:
                message.pop("tool_calls", None)
        cleaned.append(message)
    return cleaned


def get_session(conversation_id: str | None) -> IdeMcpSession | None:
    if not conversation_id:
        return None
    now = time.monotonic()
    for key, value in list(_sessions.items()):
        if value.expires_at <= now:
            _sessions.pop(key, None)
            for call_id, owner in list(_bootstrap_calls.items()):
                if owner == key:
                    _bootstrap_calls.pop(call_id, None)
    session = _sessions.get(conversation_id)
    if session is None:
        return None
    refreshed = IdeMcpSession(
        session.conversation_id, session.client, session.tools, now + SESSION_TTL_SECONDS
    )
    _sessions[conversation_id] = refreshed
    _sessions.move_to_end(conversation_id)
    return refreshed


async def configure_session(conversation_id: str, raw_config: str) -> IdeMcpSession:
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise JetBrainsMcpError(".codepartner/ide_mcp.json must contain valid JSON") from exc
    if not isinstance(config, dict) or config.get("type") != "streamable-http":
        raise JetBrainsMcpError("ide_mcp.json type must be 'streamable-http'")
    url = config.get("url")
    parsed = urlparse(url if isinstance(url, str) else "")
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.hostname.lower() not in {"127.0.0.1", "localhost", "::1"}):
        raise JetBrainsMcpError("ide_mcp.json URL must use HTTP(S) on the local machine")
    if parsed.username or parsed.password or parsed.fragment:
        raise JetBrainsMcpError("ide_mcp.json URL must not contain credentials or a fragment")
    headers = config.get("headers")
    if not isinstance(headers, dict) or set(headers) != {"IJ_MCP_SERVER_PROJECT_PATH"}:
        raise JetBrainsMcpError(
            "ide_mcp.json headers must contain only IJ_MCP_SERVER_PROJECT_PATH"
        )
    project_path = headers.get("IJ_MCP_SERVER_PROJECT_PATH")
    if (not isinstance(project_path, str) or not project_path.startswith("/")
            or any(ord(char) < 32 for char in project_path)):
        raise JetBrainsMcpError("IJ_MCP_SERVER_PROJECT_PATH must be an absolute path")

    client = JetBrainsMcpClient(url=url, headers={"IJ_MCP_SERVER_PROJECT_PATH": project_path})
    tools = await DirectIdeTools.discover(client)
    session = IdeMcpSession(
        conversation_id, client, tools, time.monotonic() + SESSION_TTL_SECONDS
    )
    _sessions[conversation_id] = session
    _sessions.move_to_end(conversation_id)
    while len(_sessions) > MAX_SESSIONS:
        evicted_id, _ = _sessions.popitem(last=False)
        for call_id, owner in list(_bootstrap_calls.items()):
            if owner == evicted_id:
                _bootstrap_calls.pop(call_id, None)
    log.info("ide_mcp_session_configured conversation_id=%s discovered_tool_count=%d",
             conversation_id, len(tools.specs))
    return session

"""Durable IDE-session identity: survives container restarts and idle expiry.

Stores, in the routing SQLite database, what an in-memory `IdeMcpSession` is rebuilt
from: the conversation's raw `.codepartner/ide_mcp.json`, its `/project use` binding,
and the bootstrap tool-call ids that identify IDE-bootstrapped conversations.
Rebuilding is a server-side MCP discovery; it never sends the IDE a tool call.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ide_conversations (
    conversation_id TEXT PRIMARY KEY,
    raw_config TEXT NOT NULL,
    workspace_id TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ide_bootstrap_calls (
    tool_call_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL
);
"""


def _path() -> Path:
    return Path(os.getenv("ROUTING_DB_PATH")
                or Path.home() / ".codepartner" / "routing.sqlite3")


def _connect() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.executescript(_SCHEMA)
    return connection


def save_config(conversation_id: str, raw_config: str) -> None:
    with _connect() as db:
        db.execute(
            "INSERT INTO ide_conversations (conversation_id, raw_config, updated_at) "
            "VALUES (?, ?, ?) ON CONFLICT(conversation_id) DO UPDATE SET "
            "raw_config = excluded.raw_config, updated_at = excluded.updated_at",
            (conversation_id, raw_config, time.time()))


def save_binding(conversation_id: str, workspace_id: str) -> None:
    with _connect() as db:
        db.execute("UPDATE ide_conversations SET workspace_id = ?, updated_at = ? "
                   "WHERE conversation_id = ?", (workspace_id, time.time(), conversation_id))


def load(conversation_id: str) -> tuple[str, str | None] | None:
    """(raw ide_mcp.json, workspace binding) for a known conversation."""
    if not _path().exists():
        return None
    with _connect() as db:
        row = db.execute("SELECT raw_config, workspace_id FROM ide_conversations "
                         "WHERE conversation_id = ?", (conversation_id,)).fetchone()
    return (row[0], row[1]) if row else None


def save_bootstrap_call(tool_call_id: str, conversation_id: str) -> None:
    with _connect() as db:
        db.execute("INSERT OR IGNORE INTO ide_bootstrap_calls VALUES (?, ?)",
                   (tool_call_id, conversation_id))


def bootstrap_conversation(tool_call_ids: list[str]) -> dict[str, str]:
    """Known bootstrap call id -> conversation id, for the ids in a request history."""
    if not tool_call_ids or not _path().exists():
        return {}
    with _connect() as db:
        rows = db.execute(
            "SELECT tool_call_id, conversation_id FROM ide_bootstrap_calls WHERE tool_call_id "
            f"IN ({','.join('?' * len(tool_call_ids))})", tool_call_ids).fetchall()
    return dict(rows)

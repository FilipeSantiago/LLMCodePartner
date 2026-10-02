"""Small client for the JetBrains IDE's streamable HTTP MCP server."""

import json
import os
import re
from datetime import timedelta

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class JetBrainsMcpError(RuntimeError):
    """The JetBrains MCP server could not complete a task-store operation."""


class JetBrainsMcpClient:
    def __init__(
            self,
            url: str | None = None,
            project_path: str | None = None,
            timeout_seconds: float | None = None,
    ):
        self._url = url or os.getenv("JETBRAINS_MCP_URL", "")
        self._project_path = project_path or os.getenv("JETBRAINS_MCP_PROJECT_PATH", "")
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("MCP_TOOL_TIMEOUT", "120"))
        )

    async def create_file(self, path: str, content: str, overwrite: bool = True) -> None:
        await self._call("create_new_file", {
            "pathInProject": path,
            "text": content,
            "overwrite": overwrite,
        })

    async def read_file(self, path: str) -> str:
        text = await self._call("read_file", {"file_path": path, "limit": 5000})
        if re.search(r"^…\d+ lines truncated…$", text, flags=re.MULTILINE):
            raise JetBrainsMcpError(f"JetBrains truncated {path} while reading it")

        lines = text.splitlines()
        if lines and all(re.match(r"^L\d+: ?", line) for line in lines):
            return "\n".join(re.sub(r"^L\d+: ?", "", line, count=1) for line in lines)
        return text

    async def file_exists(self, path: str) -> bool:
        result = await self._call_result("search_file", {"q": path, "limit": 2})
        structured = getattr(result, "structuredContent", None) or {}
        items = structured.get("items") if isinstance(structured, dict) else None
        if isinstance(items, list):
            return any(item.get("filePath") == path for item in items if isinstance(item, dict))

        text = self._result_text(result)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return path in text
        return any(
            isinstance(item, dict) and item.get("filePath") == path
            for item in payload.get("items", [])
        ) if isinstance(payload, dict) else False

    async def _call(self, name: str, arguments: dict) -> str:
        result = await self._call_result(name, arguments)
        return self._result_text(result)

    async def _call_result(self, name: str, arguments: dict):
        if not self._url:
            raise JetBrainsMcpError("JETBRAINS_MCP_URL is not configured")
        if not self._project_path:
            raise JetBrainsMcpError("JETBRAINS_MCP_PROJECT_PATH is not configured")

        headers = {"IJ_MCP_SERVER_PROJECT_PATH": self._project_path}
        timeout = httpx.Timeout(self._timeout_seconds, read=self._timeout_seconds)
        try:
            async with httpx.AsyncClient(headers=headers, timeout=timeout) as http_client:
                async with streamable_http_client(
                        self._url, http_client=http_client
                ) as (read_stream, write_stream, _):
                    async with ClientSession(
                            read_stream,
                            write_stream,
                            read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
                    ) as session:
                        await session.initialize()
                        result = await session.call_tool(name, arguments)
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise JetBrainsMcpError(
                f"JetBrains MCP {name} call failed at {self._url}: {detail}"
            ) from exc

        text = self._result_text(result)
        if result.isError:
            raise JetBrainsMcpError(f"JetBrains MCP {name} call failed: {text or 'unknown error'}")
        return result

    @staticmethod
    def _result_text(result) -> str:
        return "\n".join(
            block.text for block in result.content
            if getattr(block, "type", None) == "text" and hasattr(block, "text")
        )

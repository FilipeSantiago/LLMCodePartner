"""Bridge-backed project file access for OpenSpec operations."""

import json
from itertools import count
import logging
import re

from mcp_bridge import bridge
from mcp_bridge.models import Run
from mcp_bridge.registry import DOMAIN_SOURCE, OP_FILE_CREATE, OP_FILE_READ, OP_FILE_SEARCH
from logger.diagnostic import debug

from agent.spec.jetbrains_mcp import JetBrainsMcpError
from agent.spec.artifact_store import ArtifactLookupIndeterminate
from agent.spec.jetbrains_mcp import JetBrainsMcpClient

log = logging.getLogger("codepartner.spec.bridge")
_NUMBERED_LINE = re.compile(r"^L\d+: ?")
_TRUNCATED = re.compile(r"(?:^|\n)…\d+ lines truncated…(?:\n|$)")
_MISSING_FILE_RESPONSE = re.compile(
    r"^File .+ doesn't exist or can't be opened$", re.IGNORECASE
)
_DUPLICATE_DETERMINISTIC_CALL = re.compile(
    r"deterministic function was already called with this args", re.IGNORECASE
)
_read_limit_sequence = count()


class BridgeMcpFileClient:
    """Use the calling IDE's advertised tools; never connect to an IDE directly."""

    def __init__(self, run: Run):
        self._run = run

    async def create_file(self, path: str, content: str, overwrite: bool = True) -> None:
        self._artifact_path(path)
        await self._call(OP_FILE_CREATE, {
            "pathInProject": path,
            "text": content,
            "overwrite": overwrite,
        })

    async def read_file(self, path: str) -> str:
        self._artifact_path(path)
        # PyCharm can reject an identical deterministic MCP call, including a
        # second read in the same run. Advance a valid limit on every attempt;
        # if an IDE cache has seen that argument already, retry with the next.
        response = ""
        for _ in range(8):
            limit = 4000 + next(_read_limit_sequence) % 1001
            response = await self._call(
                OP_FILE_READ, {"file_path": path, "limit": limit}
            )
            if not _DUPLICATE_DETERMINISTIC_CALL.search(response):
                break
        else:
            raise JetBrainsMcpError(
                f"calling IDE rejected repeated deterministic reads of {path}"
            )
        # ##DELETE AFTER CORRECTION## Complete raw IDE read response.
        if not self._run.metadata.get("mcp_bootstrap"):
            debug(log, "artifact.read_response_raw", trace_id=self._run.metadata.get("trace_id"),
                  run_id=self._run.run_id, path=path, raw_response=response)
        text, wrapper = self._text_from_tool_response(response)
        if _TRUNCATED.search(text):
            raise JetBrainsMcpError(f"calling IDE truncated {path} read response")
        text, numbered, preamble_lines = self._strip_numbered_lines(text)
        # ##DELETE AFTER CORRECTION## Exact content passed to catalog JSON parsing.
        if not self._run.metadata.get("mcp_bootstrap"):
            debug(log, "artifact.read_response_normalized", trace_id=self._run.metadata.get("trace_id"),
                  run_id=self._run.run_id, path=path, wrapper=wrapper, numbered=numbered,
                  preamble_lines=preamble_lines, normalized_text=text)
        log.info(
            "artifact_read_response path=%s chars=%d wrapper=%s numbered_lines=%s "
            "preamble_lines=%d truncated=%s",
            path, len(response), wrapper, numbered, preamble_lines, False,
        )
        return text

    async def file_exists(self, path: str) -> bool:
        self._artifact_path(path)
        try:
            text = await self._call(OP_FILE_SEARCH, {"q": path, "limit": 2})
        except JetBrainsMcpError as exc:
            if _DUPLICATE_DETERMINISTIC_CALL.search(str(exc)):
                raise ArtifactLookupIndeterminate(str(exc)) from exc
            raise
        if _DUPLICATE_DETERMINISTIC_CALL.search(text):
            raise ArtifactLookupIndeterminate(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return path in text
        return any(item.get("filePath") == path for item in payload.get("items", [])
                   if isinstance(item, dict)) if isinstance(payload, dict) else False

    async def _call(self, operation: str, arguments: dict) -> str:
        try:
            name, arguments = self._run.registry.prepare_call(operation, arguments, DOMAIN_SOURCE)
        except (LookupError, ValueError) as exc:
            # ##DELETE AFTER CORRECTION## Record the complete advertised registry on bridge failure.
            debug(log, "artifact.tool_unavailable", trace_id=self._run.metadata.get("trace_id"),
                  run_id=self._run.run_id, operation=operation,
                  registered_tools=self._run.registry.names())
            raise JetBrainsMcpError(
                f"calling IDE did not advertise a compatible OpenSpec {operation!r} tool: {exc}"
            ) from exc
        result = await bridge.call_tool(self._run, name, arguments)
        # ##DELETE AFTER CORRECTION## Complete artifact tool arguments/result pair.
        if not self._run.metadata.get("mcp_bootstrap"):
            debug(log, "artifact.tool_completed", trace_id=self._run.metadata.get("trace_id"),
                  run_id=self._run.run_id, operation=operation, tool_name=name,
                  arguments=arguments, result=result)
        if (result.lstrip().lower().startswith("error") or
                _MISSING_FILE_RESPONSE.fullmatch(result.strip())):
            # ##DELETE AFTER CORRECTION## Preserve the exact IDE error response.
            if not self._run.metadata.get("mcp_bootstrap"):
                debug(log, "artifact.tool_error_result", trace_id=self._run.metadata.get("trace_id"),
                      run_id=self._run.run_id, operation=operation, tool_name=name, result=result)
            raise JetBrainsMcpError(result)
        return result

    @staticmethod
    def _artifact_path(path: str) -> None:
        if not (path.startswith("openspec/") or path.startswith(".codepartner/")):
            raise JetBrainsMcpError(f"OpenSpec bridge refuses non-artifact path {path!r}")


    @staticmethod
    def _text_from_tool_response(response: str) -> tuple[str, str]:
        """Unwrap the standard OpenAI/MCP text-block response shapes only.

        A raw file is often valid JSON itself, so arbitrary JSON objects remain
        untouched.  Wrapper detection is structural and never interprets file data.
        """
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            return response, "plain"

        if isinstance(value, dict) and isinstance(value.get("content"), list):
            blocks = value["content"]
            texts = [block.get("text") for block in blocks
                     if isinstance(block, dict) and block.get("type", "text") == "text"
                     and isinstance(block.get("text"), str)]
            if texts and len(texts) == len(blocks):
                return "\n".join(texts), "content_blocks"
            raise JetBrainsMcpError("calling IDE returned a non-text read tool response wrapper")
        if isinstance(value, list):
            texts = [block.get("text") for block in value
                     if isinstance(block, dict) and block.get("type", "text") == "text"
                     and isinstance(block.get("text"), str)]
            if texts and len(texts) == len(value):
                return "\n".join(texts), "text_blocks"
        if isinstance(value, dict) and value.get("type") == "text" and isinstance(value.get("text"), str):
            return value["text"], "text_block"
        return response, "raw_json"

    @staticmethod
    def _strip_numbered_lines(text: str) -> tuple[str, bool, int]:
        """Extract the documented numbered `read_file` body from an optional preamble.

        PyCharm's read_file contract returns one ``L<n>:`` record per source line.
        Some OpenAI-compatible client paths add a textual status line before those
        records.  We accept that envelope only when *every* remaining line is a
        numbered record; arbitrary mixed text is never treated as a file body.
        """
        lines = text.splitlines()
        first_numbered = next(
            (index for index, line in enumerate(lines) if _NUMBERED_LINE.match(line)),
            None,
        )
        if first_numbered is None:
            return text, False, 0
        body = lines[first_numbered:]
        if not body or not all(_NUMBERED_LINE.match(line) for line in body):
            return text, False, 0
        return (
            "\n".join(_NUMBERED_LINE.sub("", line, count=1) for line in body),
            True,
            first_numbered,
        )


class DirectMcpFileClient:
    """Resolve artifact operations against the conversation's standalone MCP tools."""

    def __init__(self, client: JetBrainsMcpClient, specs):
        from mcp_bridge.registry import ToolRegistry
        self._client = client
        self._registry = ToolRegistry()
        self._registry.register(list(specs))

    async def create_file(self, path: str, content: str, overwrite: bool = True) -> None:
        BridgeMcpFileClient._artifact_path(path)
        await self._call(OP_FILE_CREATE, {
            "pathInProject": path, "text": content, "overwrite": overwrite,
        })

    async def read_file(self, path: str) -> str:
        BridgeMcpFileClient._artifact_path(path)
        result = ""
        for _ in range(8):
            limit = 4000 + next(_read_limit_sequence) % 1001
            try:
                result = await self._call(OP_FILE_READ, {"file_path": path, "limit": limit})
            except JetBrainsMcpError as exc:
                if _DUPLICATE_DETERMINISTIC_CALL.search(str(exc)):
                    continue
                raise
            if not _DUPLICATE_DETERMINISTIC_CALL.search(result):
                break
        else:
            raise JetBrainsMcpError(f"configured IDE MCP server rejected repeated reads of {path}")
        text, _wrapper = BridgeMcpFileClient._text_from_tool_response(result)
        if _TRUNCATED.search(text):
            raise JetBrainsMcpError(f"PyCharm truncated {path} while reading it")
        text, _numbered, _preamble = BridgeMcpFileClient._strip_numbered_lines(text)
        return text

    async def file_exists(self, path: str) -> bool:
        BridgeMcpFileClient._artifact_path(path)
        result = ""
        for limit in (2, 3, 4, 5, 6, 7, 8, 9):
            try:
                result = await self._call(OP_FILE_SEARCH, {"q": path, "limit": limit})
            except JetBrainsMcpError as exc:
                if _DUPLICATE_DETERMINISTIC_CALL.search(str(exc)):
                    continue
                raise
            if not _DUPLICATE_DETERMINISTIC_CALL.search(result):
                break
        else:
            raise ArtifactLookupIndeterminate(
                "configured IDE MCP server rejected repeated deterministic file searches"
            )
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return path in result
        return any(item.get("filePath") == path for item in payload.get("items", [])
                   if isinstance(item, dict)) if isinstance(payload, dict) else False

    async def _call(self, operation: str, arguments: dict) -> str:
        try:
            name, prepared = self._registry.prepare_call(operation, arguments, DOMAIN_SOURCE)
        except (LookupError, ValueError) as exc:
            raise JetBrainsMcpError(
                f"configured IDE MCP server has no compatible {operation!r} tool: {exc}"
            ) from exc
        result = await self._client.call_tool(name, prepared)
        if result.lstrip().lower().startswith("error"):
            raise JetBrainsMcpError(result)
        return result

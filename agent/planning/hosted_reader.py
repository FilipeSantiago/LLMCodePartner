"""A bounded source reader for a Codex planner without parent shell access."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path, PurePosixPath

from agent.prompt import PromptFactory, PromptRole, ReaderPromptContext
from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent
from mcp_bridge.registry import ROLE_OPTIMIZER, ToolSpec

MAX_FILES = 12
MAX_LINES_PER_FILE = 200
MAX_TOTAL_LINES = 1200
MAX_SOURCE_CHARS = 80000
MAX_BRIEF_CHARS = 6000

READER_SPEC = ToolSpec(
    "planner_reader",
    "Ask the read-only planner reader one objective about exact project files and line "
    "ranges. It returns a concise path:line brief, never raw file contents.",
    {"type": "object", "properties": {
        "objective": {"type": "string"},
        "files": {"type": "array", "minItems": 1, "maxItems": MAX_FILES,
                  "items": {"type": "object", "properties": {
                      "path": {"type": "string"},
                      "start_line": {"type": "integer"},
                      "end_line": {"type": "integer"},
                      "question": {"type": "string"}},
                      "required": ["path", "start_line", "end_line", "question"]}},
    }, "required": ["objective", "files"]},
)

DECLARATIONS_SPEC = ToolSpec(
    "planner_declarations",
    "Return only symbol names, file paths, and declaration line ranges for several "
    "project-relative files in one Codebase Memory query.",
    {"type": "object", "properties": {
        "files": {"type": "array", "minItems": 1, "maxItems": 30,
                  "items": {"type": "string"}}}, "required": ["files"]},
)


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 300:
        raise ValueError("each file needs a project-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("reader paths must stay under the project workspace")
    return str(path)


def _source_batch(workspace: str, arguments: dict) -> str:
    objective = arguments.get("objective")
    files = arguments.get("files")
    if not isinstance(objective, str) or not objective.strip() or len(objective) > 1000:
        raise ValueError("reader objective must be a concise question")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError(f"reader needs 1 to {MAX_FILES} file ranges")
    root = Path(workspace).resolve()
    items = []
    total_lines = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("reader file ranges must be objects")
        relative = _relative_path(item.get("path"))
        start, end, question = (item.get("start_line"), item.get("end_line"),
                                item.get("question"))
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or start < 1 or end < start or end - start + 1 > MAX_LINES_PER_FILE):
            raise ValueError(f"reader range for {relative} must contain 1–200 lines")
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            raise ValueError(f"reader needs a concise question for {relative}")
        total_lines += end - start + 1
        if total_lines > MAX_TOTAL_LINES:
            raise ValueError("reader request has too many total lines")
        target = (root / relative).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"reader path is not a project file: {relative}")
        if target.stat().st_size > 2_000_000:
            raise ValueError(f"reader file is too large: {relative}")
        with target.open(encoding="utf-8", errors="replace") as handle:
            selected = [f"{relative}:{number} {line.rstrip()}" for number, line in
                        enumerate(handle, 1) if start <= number <= end]
        if not selected:
            raise ValueError(f"reader range starts after end of file: {relative}")
        items.append(f"Question: {question}\n" + "\n".join(selected))
    source = "\n\n".join(items)
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError("reader request exceeds the source size limit")
    return f"Objective: {objective}\n\n" + source


def declarations_query(arguments: dict) -> dict:
    files = arguments.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 30:
        raise ValueError("planner_declarations needs 1 to 30 paths")
    paths = [_relative_path(path) for path in files]
    values = ", ".join(json.dumps(path) for path in paths)
    return {"query": f"MATCH (n) WHERE n.file_path IN [{values}] "
            "AND n.start_line IS NOT NULL RETURN n.file_path, n.name, "
            "n.start_line, n.end_line ORDER BY n.file_path, n.start_line",
            "format": "json", "max_rows": 400}


async def read_brief(workspace: str, arguments: dict, provider, model: str,
                     effort: str | None, metrics: dict) -> str:
    """Read exact ranges in the service and summarize in a separate tool-free run."""
    data = metrics.setdefault("hosted_reader", {"calls": 0, "usage": {}})
    data["calls"] += 1
    try:
        source = await asyncio.to_thread(_source_batch, workspace, arguments)
    except (OSError, ValueError) as exc:
        data["rejected"] = data.get("rejected", 0) + 1
        return f"Error: reader request rejected: {exc}"
    instructions = PromptFactory.create(PromptRole.READER).with_context(
        ReaderPromptContext("planner")).build().text
    reply: list[str] = []
    usage = None
    try:
        async for event in provider.tools(
                [{"role": "user", "content": instructions + "\n\n" + source}], [],
                model=model, role=ROLE_OPTIMIZER, concrete_model=True,
                reasoning_effort=effort, run_metadata={"hosted_reader": True}):
            if isinstance(event, TextEvent):
                reply.append(event.text)
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(event.message)
            elif isinstance(event, DoneEvent):
                usage = event.usage
                if event.terminal in {"failed", "error"}:
                    raise RuntimeError("reader model failed")
    except Exception:
        data["failures"] = data.get("failures", 0) + 1
        return "Error: hosted reader failed; source was not returned to the planner."
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            data["usage"][key] = data["usage"].get(key, 0) + value
    brief = "".join(reply).strip()[:MAX_BRIEF_CHARS]
    if not brief or not re.search(r"[^\s:]+:\d+", brief):
        data["failures"] = data.get("failures", 0) + 1
        return "Error: reader brief lacked a path:line citation; ask again with exact ranges."
    return brief

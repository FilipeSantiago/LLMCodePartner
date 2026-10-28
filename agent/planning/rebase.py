"""Fitting a planned task to the files as they are when it runs.

The planner's ranges are absolute line numbers of the source it saw. Once an earlier task
inserts or removes lines above a range, the range points at other code; mapping it through
the diff between that source and the current file keeps it on the planned code. A planned
new package also needs its package marker, which plans tend to leave out.
"""

from __future__ import annotations

import difflib
from dataclasses import replace
from pathlib import Path, PurePosixPath

from agent.planning.models import EditRange, ReferenceRange, TaskContext


class _LineMap:
    """Line numbers of `before` mapped onto `after` (both 1-based)."""

    def __init__(self, before: str, after: str):
        self._before = len(before.splitlines())
        self._after = len(after.splitlines())
        self._opcodes = difflib.SequenceMatcher(
            None, before.splitlines(), after.splitlines(), autojunk=False).get_opcodes()

    def start(self, line: int) -> int:
        return self._map(line, end=False)

    def end(self, line: int) -> int:
        return self._map(line, end=True)

    def _map(self, line: int, end: bool) -> int:
        index = line - 1
        for tag, i1, i2, j1, j2 in self._opcodes:
            if i1 <= index < i2:
                if tag == "equal":
                    return j1 + (index - i1) + 1
                # A changed or removed line maps to the bounds of what replaced it.
                return max(j2, j1 + 1) if end else j1 + 1
        # Past the old end of file: keep the distance from the end.
        return max(1, line - self._before + self._after)


def _shifted(item, lines: _LineMap):
    if item.start_line is None or item.end_line is None:
        return item
    start, end = lines.start(item.start_line), lines.end(item.end_line)
    return replace(item, start_line=start, end_line=max(start, end))


def rebase_task(task: TaskContext, before: dict[str, str],
                current: dict[str, str]) -> TaskContext:
    """`task` with its edit and reference ranges moved from `before` (each file's content
    the plan was made against) to `current`. Files missing from either, unchanged files
    and new-file edits keep their ranges."""
    maps = {path: _LineMap(before[path], current[path]) for path in before
            if path in current and before[path] != current[path]}
    if not maps:
        return task
    edits: list[EditRange] = [_shifted(edit, maps[edit.file]) if edit.file in maps else edit
                              for edit in task.edits]
    references: list[ReferenceRange] = [
        _shifted(ref, maps[ref.file]) if ref.file in maps else ref for ref in task.references]
    return replace(task, edits=edits, references=references)


def with_package_markers(task: TaskContext, planned: set[str], root: Path) -> TaskContext:
    """`task` plus an `__init__.py` edit for each new Python package it creates, when no
    task plans one and a sibling package has one (the project uses regular packages)."""
    markers: list[str] = []
    for edit in task.edits:
        if edit.start_line is not None or not edit.file.endswith(".py"):
            continue
        directory = PurePosixPath(edit.file).parent
        marker = str(directory / "__init__.py")
        if (str(directory) == "." or marker in planned or marker in markers
                or (root / marker).exists()):
            continue
        parent = root / directory.parent
        siblings = ([path for path in parent.iterdir()
                     if path.is_dir() and path.name != directory.name]
                    if parent.is_dir() else [])
        if any((path / "__init__.py").is_file() for path in siblings):
            markers.append(marker)
    if not markers:
        return task
    return replace(task, edits=[*task.edits, *(EditRange(
        marker, "create the package marker of this new package") for marker in markers)])

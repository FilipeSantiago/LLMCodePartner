"""Current source of a planned task's ranges, for prompts that carry their own context.

An inline coder or reviewer gets no file tools, so everything it may need is read here:
the edit ranges (widened by the planner margin), the references, and the code earlier
tasks of the job changed that this task depends on. Each region is numbered with its
current line numbers. Line numbers come from ranges already fitted to the current files
(`agent.planning.rebase`), and a small file is sent whole so a range edge cannot cut off
what the model needs.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agent.planning.models import ReferenceRange, TaskContext

EDIT, REFERENCE, NEW = "EDIT", "REFERENCE", "NEW"
# Files up to this many lines are sent whole.
WHOLE_FILE_LINES = 300
# A whole-file reference of a larger file is cut to its first lines.
REFERENCE_FILE_LINES = 400
DEFAULT_BUDGET_CHARS = 120_000


@dataclass(frozen=True)
class Region:
    path: str
    kind: str  # EDIT, REFERENCE or NEW (a file the task creates)
    start: int = 0  # 1-based, inclusive; 0 for a NEW file
    end: int = 0
    why: str = ""


@dataclass(frozen=True)
class Snippets:
    text: str
    regions: tuple[Region, ...]
    # Referenced files that do not exist (named in the prompt, not read).
    missing: tuple[str, ...]
    budget: int

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def fits(self) -> bool:
        return self.chars <= self.budget


def _safe(path: str) -> bool:
    pure = PurePosixPath(path)
    return bool(path) and not pure.is_absolute() and ".." not in pure.parts


def _read(root: Path, path: str) -> str | None:
    if not _safe(path):
        return None
    try:
        return (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _merged(regions: list[Region]) -> list[Region]:
    """Overlapping or adjacent regions of one file as one; EDIT wins over REFERENCE."""
    result: list[Region] = []
    for region in sorted(regions, key=lambda item: (item.start, item.end)):
        last = result[-1] if result else None
        if last is not None and region.start <= last.end + 1:
            kind = EDIT if EDIT in (last.kind, region.kind) else REFERENCE
            why = "; ".join(dict.fromkeys(filter(None, (last.why, region.why))))
            result[-1] = Region(last.path, kind, last.start, max(last.end, region.end), why)
        else:
            result.append(region)
    return result


def _regions(task: TaskContext, root: Path, margin: int,
             extra: tuple[ReferenceRange, ...]) -> tuple[list[Region], list[str],
                                                          dict[str, str]]:
    wanted: dict[str, list[Region]] = {}
    new: list[Region] = []
    missing: list[str] = []
    contents: dict[str, str] = {}

    def content(path: str) -> str | None:
        if path not in contents:
            text = _read(root, path)
            if text is None:
                return None
            contents[path] = text
        return contents[path]

    for edit in task.edits:
        text = content(edit.file)
        if text is None:
            new.append(Region(edit.file, NEW, why=edit.change))
            continue
        total = max(1, len(text.splitlines()))
        if edit.start_line is None or edit.end_line is None:
            start, end = 1, total
        else:
            start = max(1, edit.start_line - margin)
            end = min(total, edit.end_line + margin)
        why = edit.change + (f" ({edit.symbol})" if edit.symbol else "")
        wanted.setdefault(edit.file, []).append(Region(edit.file, EDIT, start, end, why))
    for ref in (*task.references, *extra):
        text = content(ref.file)
        if text is None:
            if ref.file not in missing:
                missing.append(ref.file)
            continue
        total = max(1, len(text.splitlines()))
        if ref.start_line is None or ref.end_line is None:
            start, end = 1, min(total, REFERENCE_FILE_LINES)
        else:
            start, end = max(1, ref.start_line), min(total, ref.end_line)
        wanted.setdefault(ref.file, []).append(Region(ref.file, REFERENCE, start, end, ref.why))
    regions: list[Region] = []
    for path, items in wanted.items():
        total = len(contents[path].splitlines())
        if total <= WHOLE_FILE_LINES:
            kind = EDIT if any(item.kind == EDIT for item in items) else REFERENCE
            why = "; ".join(dict.fromkeys(item.why for item in items if item.why))
            items = [Region(path, kind, 1, max(1, total), why)]
        regions.extend(_merged(items))
    return [*regions, *new], missing, contents


def _render(region: Region, contents: dict[str, str]) -> str:
    if region.kind == NEW:
        return f"### {region.path} (NEW FILE — create it) — {region.why}\n"
    lines = contents[region.path].splitlines()
    total = len(lines)
    span = "whole file" if region.start == 1 and region.end >= total else (
        f"lines {region.start}-{region.end} of {total}")
    width = len(str(region.end))
    body = "\n".join(f"{number:>{width}}| {lines[number - 1]}"
                     for number in range(region.start, min(region.end, total) + 1))
    return (f"### {region.path} ({region.kind}, {span})"
            + (f" — {region.why}" if region.why else "") + f"\n{body}\n")


def materialize(task: TaskContext, root: Path | str, *, margin: int = 10,
                budget: int = DEFAULT_BUDGET_CHARS,
                extra: tuple[ReferenceRange, ...] = ()) -> Snippets:
    """The current source of `task`'s edit and reference ranges, plus `extra` references
    (code earlier tasks changed). EDIT regions come first so they survive a reader's
    attention best; missing references are only named."""
    regions, missing, contents = _regions(task, Path(root), margin, extra)
    ordered = sorted(regions, key=lambda item: ({EDIT: 0, NEW: 1, REFERENCE: 2}[item.kind],
                                                item.path, item.start))
    text = "\n".join(_render(region, contents) for region in ordered)
    if missing:
        text += ("\nReferenced files that do not exist: " + ", ".join(missing) + "\n")
    return Snippets(text, tuple(ordered), tuple(missing), budget)


def changed_ranges(before: str, after: str, path: str = "") -> list[tuple[int, int]]:
    """1-based line ranges of `after` that differ from `before`. In a Python file each
    range grows to the top-level definition around it, so a changed method arrives with
    its signature."""
    matcher = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines(),
                                      autojunk=False)
    ranges = [(j1 + 1, max(j1 + 1, j2)) for tag, _i1, _i2, j1, j2 in matcher.get_opcodes()
              if tag != "equal" and j2 > j1]
    if not ranges or not path.endswith(".py"):
        return ranges
    try:
        tree = ast.parse(after)
    except (SyntaxError, ValueError):
        return ranges
    spans = [(node.lineno - len(getattr(node, "decorator_list", ())),
              node.end_lineno or node.lineno) for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    grown = []
    for start, end in ranges:
        for low, high in spans:
            if low <= end and start <= high:
                start, end = min(start, low), max(end, high)
        grown.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(grown):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def earlier_task_references(task_id: str, changes: dict[str, dict[str, dict]],
                            root: Path | str) -> tuple[ReferenceRange, ...]:
    """References to what earlier tasks changed, read from the current files.

    `changes` maps an earlier task id to its preimages ({path: {"before_exists",
    "before_content"}}), as the job's rollback journal holds them. A file the earlier task
    created is referenced whole; a modified one by its changed ranges."""
    root = Path(root)
    references: list[ReferenceRange] = []
    for earlier, preimages in changes.items():
        if earlier == task_id:
            continue
        why = f"changed by {earlier}"
        for path, preimage in preimages.items():
            current = _read(root, path)
            if current is None:
                continue
            before = preimage.get("before_content")
            if not preimage.get("before_exists") or before is None:
                references.append(ReferenceRange(
                    path, why if preimage.get("before_exists") else f"created by {earlier}"))
                continue
            references.extend(ReferenceRange(path, why, start, end)
                              for start, end in changed_ranges(before, current, path))
    return tuple(references)

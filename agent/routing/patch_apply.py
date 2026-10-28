"""An inline coder's reply: parsed, checked against the task's files, then written.

The inline coder has no tools; it answers with one JSON object listing exact text
replacements and new files. Every change is resolved in memory first, so a reply with one
bad edit writes nothing, and each problem is reported precisely enough for one corrective
call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# A line copied with the prompt's numbering ("  12| code").
_NUMBERED = re.compile(r"^\s*\d+\| ?")


class PatchFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Edit:
    file: str
    old_text: str
    new_text: str


@dataclass(frozen=True)
class NewFile:
    path: str
    content: str


@dataclass(frozen=True)
class PatchResponse:
    edits: tuple[Edit, ...] = ()
    new_files: tuple[NewFile, ...] = ()
    # [{"file" or "symbol": ..., "why": ...}]: context the plan did not give.
    need_context: tuple[dict, ...] = ()
    summary: str = ""
    criteria: tuple[dict, ...] = ()
    remaining_gaps: tuple[str, ...] = ()
    context_handoff: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.edits and not self.new_files


def _objects(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _strings(value) -> tuple[str, ...]:
    return tuple(item for item in value if isinstance(item, str)) \
        if isinstance(value, list) else ()


def parse_response(text: str) -> PatchResponse:
    """The last JSON object in `text` that has an `edits` or `new_files` key."""
    found = None
    for value in _objects(text):
        if "edits" in value or "new_files" in value or "need_context" in value:
            found = value
    if found is None:
        raise PatchFormatError("no JSON object with `edits`, `new_files` or `need_context`")
    edits, new_files = [], []
    for number, item in enumerate(found.get("edits") or [], 1):
        if (not isinstance(item, dict) or not isinstance(item.get("file"), str)
                or not isinstance(item.get("old_text"), str)
                or not isinstance(item.get("new_text"), str)):
            raise PatchFormatError(
                f"edit {number} needs string `file`, `old_text` and `new_text`")
        edits.append(Edit(item["file"].strip(), item["old_text"], item["new_text"]))
    for number, item in enumerate(found.get("new_files") or [], 1):
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                or not isinstance(item.get("content"), str)):
            raise PatchFormatError(f"new file {number} needs string `path` and `content`")
        new_files.append(NewFile(item["path"].strip(), item["content"]))
    need = tuple(item for item in found.get("need_context") or [] if isinstance(item, dict))
    criteria = tuple(item for item in found.get("criteria") or [] if isinstance(item, dict))
    handoff = found.get("context_handoff")
    return PatchResponse(tuple(edits), tuple(new_files), need,
                         str(found.get("summary") or "").strip(), criteria,
                         _strings(found.get("remaining_gaps")),
                         handoff if isinstance(handoff, dict) else {})


def _safe(path: str) -> bool:
    pure = PurePosixPath(path)
    return bool(path) and not pure.is_absolute() and ".." not in pure.parts


def _unnumbered(text: str) -> str | None:
    lines = text.splitlines(keepends=True)
    if lines and all(_NUMBERED.match(line) for line in lines if line.strip()):
        return "".join(_NUMBERED.sub("", line, count=1) for line in lines)
    return None


def resolve(response: PatchResponse, root: Path | str,
            allowed: frozenset[str]) -> tuple[dict[str, str], list[str]]:
    """The files' new contents ({path: content}) and every problem found. Nothing is
    written; with any problem the caller writes nothing either.

    `allowed` holds the task's edit files (a planned new file included). Edits apply in
    order, each to the result of the ones before it in the same file."""
    root = Path(root)
    contents: dict[str, str] = {}
    errors: list[str] = []
    for item in response.new_files:
        if not _safe(item.path) or item.path not in allowed:
            errors.append(f"new file {item.path}: not one of this task's edit files "
                          f"({', '.join(sorted(allowed))})")
        elif (root / item.path).exists() or item.path in contents:
            errors.append(f"new file {item.path}: it already exists; change it with `edits`")
        else:
            contents[item.path] = item.content
    for number, edit in enumerate(response.edits, 1):
        label = f"edit {number} ({edit.file})"
        if not _safe(edit.file) or edit.file not in allowed:
            errors.append(f"{label}: not one of this task's edit files "
                          f"({', '.join(sorted(allowed))})")
            continue
        if edit.file not in contents:
            try:
                contents[edit.file] = (root / edit.file).read_text(encoding="utf-8")
            except FileNotFoundError:
                errors.append(f"{label}: the file does not exist; create it in `new_files`")
                continue
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"{label}: unreadable ({type(exc).__name__})")
                continue
        current = contents[edit.file]
        if not edit.old_text:
            errors.append(f"{label}: `old_text` is empty; quote the exact text to replace")
            continue
        old, new = edit.old_text, edit.new_text
        count = current.count(old)
        if count == 0 and (plain := _unnumbered(old)) is not None:
            # The model copied the prompt's line numbers; its new text likely has them too.
            old, new = plain, _unnumbered(new) or new
            count = current.count(old)
        if count != 1:
            errors.append(f"{label}: `old_text` occurs {count} times; it must match the "
                          "current file exactly once (copy it verbatim, without line "
                          "numbers, and include enough surrounding lines to be unique)")
            continue
        contents[edit.file] = current.replace(old, new, 1)
    return contents, errors


def write(root: Path | str, contents: dict[str, str]) -> list[str]:
    """Write resolved contents; returns the paths written."""
    root = Path(root)
    for path, text in contents.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return list(contents)

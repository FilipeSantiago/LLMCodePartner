"""Metadata-only measurements of source responses delivered to an agent."""

from dataclasses import dataclass, field
import shlex


def dispatched(name: str, arguments: dict, read: bool) -> tuple[str, dict, bool]:
    """The bare tool name and arguments, unwrapping an `execute_tool` dispatch."""
    bare = name.rsplit("__", 1)[-1]
    if bare == "execute_tool" and isinstance(arguments.get("command"), str):
        try:
            tokens = shlex.split(arguments["command"])
        except ValueError:
            tokens = []
        if tokens:
            bare = tokens[0]
            arguments = {key[2:]: value for key, value in zip(tokens[1::2], tokens[2::2])
                         if key.startswith("--")}
            read = read or bare in {"read_file", "get_file_text_by_path"}
    return bare, arguments, read


def read_region(arguments: dict) -> tuple[str | None, int | None, int | None, bool]:
    """(path, start, end, ranged) of a source read's arguments."""
    path = arguments.get("file_path") or arguments.get("path")
    if not isinstance(path, str) or not path:
        return None, None, None, False
    start = arguments.get("start_line", arguments.get("offset"))
    end = arguments.get("end_line")
    limit = arguments.get("limit")
    if isinstance(start, str) and start.isdecimal():
        start = int(start)
    if isinstance(end, str) and end.isdecimal():
        end = int(end)
    if isinstance(limit, str) and limit.isdecimal():
        limit = int(limit)
    if start is None and isinstance(limit, int):
        start = 1
    if end is None and isinstance(start, int) and isinstance(limit, int):
        end = start + limit - 1
    ranged = end is not None or isinstance(start, int) and start > 1
    return path, start, end, ranged


def failed_read(result) -> bool:
    return not isinstance(result, str) or result.startswith(("Error:", "READ FAILED"))


@dataclass
class SourceReadStats:
    kb_outline_calls: int = 0
    whole_file_reads: int = 0
    ranged_source_reads: int = 0
    source_response_bytes: int = 0
    repeated_source_reads: int = 0
    regions: set[tuple[str, int | None, int | None]] = field(default_factory=set)

    def observe(self, name: str, arguments: dict, result: str, *, read: bool) -> None:
        bare, arguments, read = dispatched(name, arguments, read)
        if bare == "kb_get_file_outline":
            self.kb_outline_calls += 1
        if not read or failed_read(result):
            return
        path, start, end, ranged = read_region(arguments)
        if path is None:
            return
        self.ranged_source_reads += int(ranged)
        self.whole_file_reads += int(not ranged)
        region = (path, start if ranged and isinstance(start, int) else None,
                  end if ranged and isinstance(end, int) else None)
        if region in self.regions:
            self.repeated_source_reads += 1
        self.regions.add(region)
        self.source_response_bytes += len(result.encode("utf-8"))

    def summary(self) -> dict:
        return {"kb_outline_calls": self.kb_outline_calls,
                "whole_file_reads": self.whole_file_reads,
                "ranged_source_reads": self.ranged_source_reads,
                "source_response_bytes": self.source_response_bytes,
                "repeated_source_reads": self.repeated_source_reads}

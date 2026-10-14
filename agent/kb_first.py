"""Run-local KB-first navigation. Generated calls use the normal authorized dispatch."""

from __future__ import annotations

import contextvars
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from agent.planning.knowledge_base import _identifiers
from mcp_bridge.adapters import mutation_targets
from mcp_bridge.read_metrics import dispatched, failed_read, read_region
from mcp_bridge.registry import is_mutating_tool

STRENGTHS = ("strong", "moderate", "weak", "none")
_RANK = {name: rank for rank, name in enumerate(STRENGTHS)}
_EXTENSIONS = ("py|pyi|ipynb|js|jsx|mjs|ts|tsx|java|kt|kts|go|rs|rb|c|h|cc|cpp|hpp|cs|swift|"
               "scala|sql|sh|md|rst|txt|yaml|yml|json|toml|cfg|ini|xml|html|css|gradle|properties")
_PATH = re.compile(rf"(?<![\w.-])(/?(?:[\w.-]+/)*[\w-][\w.-]*\.(?:{_EXTENSIONS}))(?![\w/-])")
_PATH_KEYS = ("file_path", "filePath", "pathInProject", "path", "directoryPath")
# Calls whose results list files broadly, whatever their arguments.
_BROAD = frozenset(("kb_get_architecture", "kb_index_status", "list_directory_tree",
                    "list_files_in_folder", "get_all_open_file_paths", "get_project_modules",
                    "get_project_dependencies"))
# Calls that inspect a path without saying anything about other paths.
_TARGET_ONLY = frozenset(("kb_get_file_outline",))
# KB prefetch section labels (knowledge_base.CodebaseMemoryClient.context) and the
# strength of the paths they contain; outlines there were fetched for named symbols.
_KB_SECTIONS = (("Symbols named in the request: ", "strong"),
                ("File outline ", "strong"),
                ("Keyword graph matches (secondary leads): ", "weak"),
                ("Architecture: ", "weak"),
                ("Index status: ", None))
_UNEXPLAINED_SAMPLE = 20
_LIMIT_LABELS = {"tool_calls": "tool calls", "discovery_calls": "discovery calls",
                 "full_reads": "full-file reads",
                 "est_added_tokens": "estimated added context tokens",
                 "repeated_or_unexplained_reads": "repeated or unexplained reads"}


def _norm(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def paths_in(text: str) -> list[str]:
    """Path-like tokens with a source or config extension."""
    return list(dict.fromkeys(_norm(match.group(1)) for match in _PATH.finditer(text or "")))


def _same(a: str, b: str) -> bool:
    """Equal, or one is a directory-qualified suffix of the other (relative vs absolute)."""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return "/" in short and long.endswith("/" + short.lstrip("/"))


def _target(arguments: dict) -> str | None:
    value = next((arguments[key] for key in _PATH_KEYS
                  if isinstance(arguments.get(key), str) and arguments[key]), None)
    return _norm(value) if value else None


@dataclass
class EvidenceSeed:
    """Evidence known before the planner runs: the WP text and the KB prefetch."""
    paths: dict[str, str] = field(default_factory=dict)
    identifiers: tuple[str, ...] = ()
    kb_prefetch: bool = False


def evidence_seed(wp_text: str, kb_context: str = "") -> EvidenceSeed:
    seed = EvidenceSeed(identifiers=tuple(_identifiers(wp_text, limit=40)),
                        kb_prefetch="<kb_context>" in (kb_context or ""))
    for path in paths_in(wp_text):
        seed.paths[path] = "strong"
    if seed.kb_prefetch:
        labels = "|".join(re.escape(label) for label, _ in _KB_SECTIONS)
        strengths = dict(_KB_SECTIONS)
        parts = re.split(f"({labels})", kb_context)
        for label, body in zip(parts[1::2], parts[2::2]):
            strength = strengths[label]
            if strength is None:
                continue
            for path in paths_in(body):
                if _RANK[strength] < _RANK[seed.paths.get(path, "none")]:
                    seed.paths[path] = strength
    return seed


class DiscoveryTracker:
    def __init__(self, seed: EvidenceSeed, small_file_bytes: int, kb_tools_offered: bool,
                 thresholds: dict[str, int] | None = None, tier: str = ""):
        self.seed = seed
        self.thresholds = thresholds
        self.tier = tier
        self.calls = 0
        self.flagged_reads = 0
        self.crossings: list[dict] = []
        self.small_file_bytes = small_file_bytes
        self.kb_tools_offered = kb_tools_offered
        self.evidence: dict[str, str] = dict(seed.paths)
        self.touched: set[str] = set()
        self.levels = {"L1": 0, "L2": 0, "L3": 0}
        self.l3_direct = 0
        self.l3_escalated = 0
        self.kb_calls = 0
        self.reads_by_strength = dict.fromkeys(STRENGTHS, 0)
        self.unexplained: list[str] = []
        self.prompt_bytes = 0
        self.result_bytes = 0

    def prompt(self, conversation: list[dict]) -> None:
        self.prompt_bytes += sum(len(message["content"].encode("utf-8"))
                                 for message in conversation
                                 if isinstance(message.get("content"), str))

    def delivered(self, result) -> None:
        if isinstance(result, str):
            self.result_bytes += len(result.encode("utf-8"))

    def strength(self, path: str) -> str:
        return min((strength for known, strength in self.evidence.items()
                    if _same(path, known)), key=_RANK.get, default="none")

    def observe(self, name: str, arguments: dict, result, *, read: bool,
                repeated: bool = False) -> str | None:
        """Classify one executed call; return an advisory to append, in advise mode only.

        The advisory's bytes are counted as delivered context here.
        """
        self.delivered(result)
        self.calls += 1
        self._classify(name, arguments, result, read, repeated)
        advisory = self._advisory()
        if advisory:
            self.delivered("\n\n" + advisory)
        return advisory

    def _classify(self, name: str, arguments: dict, result, read: bool,
                  repeated: bool) -> None:
        bare, arguments, read = dispatched(name, arguments, read)
        if read:
            self._read(arguments, result, repeated)
            return
        self.levels["L1"] += 1
        self.kb_calls += int(bare.startswith("kb_"))
        target = _target(arguments)
        if target:
            self.touched.add(target)
        if bare in _TARGET_ONLY or not isinstance(result, str):
            return
        strength = "weak" if bare in _BROAD or not self._targeted(arguments) else "moderate"
        for path in paths_in(result):
            if target and _same(path, target):
                continue
            if _RANK[strength] < _RANK[self.evidence.get(path, "none")]:
                self.evidence[path] = strength

    def _targeted(self, arguments: dict) -> bool:
        """A search or graph call aimed at a WP identifier or a strongly evidenced path."""
        values = [value for value in arguments.values() if isinstance(value, str)]
        strong = [path for path, strength in self.evidence.items() if strength == "strong"]
        return any(identifier in value for value in values
                   for identifier in self.seed.identifiers) or any(
            _same(_norm(value), path) for value in values for path in strong)

    def _read(self, arguments: dict, result, repeated: bool) -> None:
        path, start, end, ranged = read_region(arguments)
        if path is None or failed_read(result):
            return
        path = _norm(path)
        if (ranged and arguments.get("end_line") is None and (start or 1) <= 1
                and end is not None and result.count("\n") + 1 < end):
            # A `limit` larger than the file returned all of it: a whole read.
            ranged = False
        small = len(result.encode("utf-8")) < self.small_file_bytes
        if ranged or small:
            self.levels["L2"] += 1
            self.touched.add(path)
        else:
            self.levels["L3"] += 1
            if any(_same(path, known) for known in self.touched):
                self.l3_escalated += 1
            else:
                self.l3_direct += 1
        strength = self.strength(path)
        self.reads_by_strength[strength] += 1
        unexplained = strength in {"weak", "none"}
        self.flagged_reads += int(repeated or unexplained)
        if unexplained and path not in self.unexplained:
            self.unexplained.append(path)

    def _advisory(self) -> str | None:
        """One line naming each soft threshold reached for the first time."""
        if not self.thresholds:
            return None
        values = {"tool_calls": self.calls, "discovery_calls": self.calls,
                  "full_reads": self.levels["L3"],
                  "est_added_tokens": self._estimate(),
                  "repeated_or_unexplained_reads": self.flagged_reads}
        crossed = {item["limit"] for item in self.crossings}
        new = [(limit, values[limit], threshold)
               for limit, threshold in self.thresholds.items()
               if limit not in crossed and values.get(limit, 0) >= threshold]
        if not new:
            return None
        self.crossings += [{"limit": limit, "value": value, "threshold": threshold,
                            "call": self.calls} for limit, value, threshold in new]
        reached = ", ".join(f"{_LIMIT_LABELS[limit]} {value}/{threshold}"
                            for limit, value, threshold in new)
        return (f"[Planning advisory, not a limit] The {self.tier} WP guideline was reached "
                f"({reached}). Check the sufficiency checklist: if components, dependencies, "
                "task responsibilities and verification are established, return the plan "
                "now; otherwise investigate only a specific open question, preferring "
                "outlines and ranged reads.")

    def _estimate(self) -> int:
        return math.ceil(self.prompt_bytes / 4) + math.ceil(self.result_bytes / 4)

    def summary(self) -> dict:
        prompt_tokens = math.ceil(self.prompt_bytes / 4)
        result_tokens = math.ceil(self.result_bytes / 4)
        return {
            "retrieval_levels": dict(self.levels),
            "l3_direct": self.l3_direct, "l3_escalated": self.l3_escalated,
            "evidence_strength": dict(self.reads_by_strength),
            "unexplained_reads": (self.reads_by_strength["weak"]
                                  + self.reads_by_strength["none"]),
            "unexplained_paths": self.unexplained[:_UNEXPLAINED_SAMPLE],
            "kb_prefetch": self.seed.kb_prefetch,
            "kb_tools_offered": self.kb_tools_offered,
            "kb_used": self.seed.kb_prefetch or self.kb_calls > 0,
            "kb_fallback": not (self.seed.kb_prefetch or self.kb_tools_offered),
            # Estimates of newly added context from delivered UTF-8 bytes (≈4 per
            # token); provider-reported cumulative usage is recorded separately.
            "est_prompt_tokens": prompt_tokens,
            "est_tool_result_tokens": result_tokens,
            "est_added_tokens": prompt_tokens + result_tokens,
            "threshold_crossings": list(self.crossings),
        }


_CODE_EXTENSIONS = frozenset((
    "py", "pyi", "js", "jsx", "mjs", "ts", "tsx", "java", "kt", "kts", "go", "rs", "rb", "c",
    "h", "cc", "cpp", "hpp", "cs", "swift", "scala", "sh", "sql"))
_OUTLINE_ROWS = 60
GATE_NOTE = ("Read not performed. Outline from the last index: line ranges are hints "
             "and may have shifted; read current source via the IDE with ranges, "
             "or repeat this read for the whole file.")


def _int(value) -> int | None:
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class OutlineGate:
    """KB-first step for every agent: outline a large code file before a broad read.

    A broad read (from the start, no end line, no limit or one above `lines`) of a code
    file not yet outlined, range-read or gated is answered with its Codebase Memory
    outline. Repeating the read, non-code files, files the index lacks or that are
    smaller than `lines`, and outline failures pass straight through to the IDE.
    """

    REASONS = ("repeat", "after_outline", "small", "not_indexed", "non_code", "error")

    def __init__(self, outline_handler, lines: int):
        self.outline = outline_handler
        self.lines = lines
        self.located: list[str] = []  # outlined or range-read in this run
        self.gated: list[str] = []
        self.served = 0
        self.passthrough = dict.fromkeys(self.REASONS, 0)

    def _broad(self, arguments: dict) -> bool:
        start = _int(arguments.get("start_line", arguments.get("offset")))
        limit = _int(arguments.get("limit"))
        return (arguments.get("end_line") is None and (start is None or start <= 1)
                and (limit is None or limit > self.lines))

    def note(self, name: str, arguments: dict, read: bool) -> None:
        """Record an executed call that located a path (outline or narrow read)."""
        bare, arguments, read = dispatched(name, arguments, read)
        if bare == "kb_get_file_outline" or read and not self._broad(arguments):
            path = _target(arguments)
            if path:
                self.located.append(path)

    def _pass(self, reason: str) -> None:
        self.passthrough[reason] += 1

    async def check(self, arguments: dict) -> tuple[str, str] | None:
        """(path, outline text) to return instead of the read, or None to read."""
        path = _target(arguments)
        if not path or not self._broad(arguments):
            return None
        name = path.rsplit("/", 1)[-1]
        if "." not in name or name.rsplit(".", 1)[-1].lower() not in _CODE_EXTENSIONS:
            self._pass("non_code")
            return None
        if any(_same(path, known) for known in self.gated):
            self._pass("repeat")
            return None
        if any(_same(path, known) for known in self.located):
            self._pass("after_outline")
            return None
        self.gated.append(path)
        try:
            raw = await self.outline({"file_path": path, "limit": _OUTLINE_ROWS,
                                      "format": "json"})
            payload = json.loads(raw) if isinstance(raw, str) else raw
            cols = payload["cols"]
            rows = payload.get("rows") or []
        except Exception:
            self._pass("error")
            return None
        if not rows:
            self._pass("not_indexed")
            return None
        index = cols.index("lines") if "lines" in cols else None
        ends = [int(match.group(1)) for row in rows if index is not None
                for match in [re.search(r"(\d+)\s*$", str(row[index]))] if match]
        last = max(ends, default=0)
        if last < self.lines:
            self._pass("small")
            return None
        self.served += 1
        label = cols.index("label") if "label" in cols else None
        lines = [f"- {row[0]}" + (f" ({row[label]})" if label is not None else "")
                 + (f" lines {row[index]}" if index is not None else "") for row in rows]
        total = payload.get("total") or len(rows)
        more = (f"\n... {total - len(rows)} more symbols; call kb_get_file_outline with an "
                "offset for the rest." if payload.get("has_more") else "")
        return path, (f"Codebase Memory outline of `{path}` ({total} symbols, indexed through "
                      f"line {last}):\n" + "\n".join(lines) + more + "\n" + GATE_NOTE)

    def summary(self) -> dict:
        return {"served": self.served, "passthrough": dict(self.passthrough)}

_GENERATED = contextvars.ContextVar("kb_first_generated", default=False)
_SEARCHES = {"search_text", "search_regex", "search_in_files_by_text", "find_text"}
_READS = {"read_file", "get_file_text_by_path", "codepartner_read_file"}
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def _payload(raw):
    if not isinstance(raw, str) or raw.startswith("Error:"):
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _rows(payload):
    if isinstance(payload, dict):
        return (payload.get("rows") or payload.get("results") or payload.get("matches")
                or payload.get("groups") or payload.get("nodes") or [])
    return payload if isinstance(payload, list) else []


def _error(result):
    return not isinstance(result, str) or result.startswith(("Error:", "READ ERROR", "READ FAILED", "READ UNAVAILABLE", "ACCESS DENIED", "Tool budget exhausted", "File-read budget exhausted"))


def _source_body(result):
    lines = result.splitlines()
    if result.startswith("READ COMPLETE"):
        lines = lines[3:]
        if lines and lines[0].startswith("Remaining: "):
            lines = lines[1:]
    return "\n".join(re.sub(r"^L\d+: ?", "", line) for line in lines)


@dataclass
class VerifiedRegion:
    start: int
    end: int
    text: str
    whole: bool = False


class KbFirstPolicy:
    """One policy per agent run; never invokes a handler directly."""

    def __init__(self, *, role: str, tool_names=(), seed: EvidenceSeed | None = None,
                 tracker: DiscoveryTracker | None = None, outline_gate_lines: int = 120,
                 search_gate_max: int = 5, verify_paths=(), excluded_files=(),
                 root_path: str | None = None, indexed_at: str | None = None,
                 snippet_max_lines: int = 300, recent_mutations=None,
                 mode: str = "advise", tier: str = "",
                 advisory_thresholds: dict[str, int] | None = None,
                 small_file_bytes: int = 6000):
        self.role = role
        self.tool_names = set(tool_names)
        self.tracker = tracker
        self.seed = seed or EvidenceSeed()
        self.outline = OutlineGate(None, outline_gate_lines)
        self.search_gate_max = search_gate_max
        self.verify_paths = set(verify_paths)
        self.excluded_files = set(excluded_files)
        self.root_path = root_path
        self.indexed_at = indexed_at
        self.recent_mutations = recent_mutations
        self.recent_paths: set[str] = set()
        self._index_checked = False
        self.snippet_max_lines = snippet_max_lines
        self.mode = mode
        self.tier = tier
        self.advisory_thresholds = advisory_thresholds or {}
        self.small_file_bytes = small_file_bytes
        self.discovery_calls = 0
        self.full_reads = 0
        self.threshold_crossings: list[dict] = []
        self.gated_searches: set[str] = set()
        self.graph_searched: set[str] = set()
        self.gated_identifiers: set[str] = set()
        self.search_outcomes = Counter()
        self.outline_outcomes = Counter()
        self.outline_followed: set[str] = set()
        self.snippet_outcomes = Counter()
        self.fallback_reasons = Counter()
        self.verified: dict[str, list[VerifiedRegion]] = defaultdict(list)
        self.snippet_regions: set[tuple[str, int, int]] = set()
        self.hinted_paths: set[str] = set()
        self.mutated_paths: set[str] = set()
        self.redundant_source_reads = 0
        self.redundant_source_bytes = 0
        self.reads_by_strength = Counter()
        self.unexplained_reads = 0
        self.flagged_reads = 0
        self.source_response_bytes = 0
        self.result_bytes = 0
        self.edit_verification_blocks = 0
        self.edits_verified_by_snippet = 0
        self.generated_calls = 0
        self.generated_tools = Counter()
        self.levels = {"L1": 0, "L2": 0, "L3": 0}
        self.l3_direct = 0
        self.l3_escalated = 0

    @staticmethod
    def _name(run, bare):
        return next((name for name in run.registry.names()
                     if name == bare or name.endswith("__" + bare)), None)

    def fork(self):
        """Fresh counters and verified regions for a new provider attempt."""
        return KbFirstPolicy(
            role=self.role, tool_names=self.tool_names, seed=self.seed,
            tracker=self.tracker, outline_gate_lines=self.outline.lines,
            search_gate_max=self.search_gate_max, verify_paths=self.verify_paths,
            excluded_files=self.excluded_files, root_path=self.root_path,
            indexed_at=self.indexed_at, snippet_max_lines=self.snippet_max_lines,
            recent_mutations=self.recent_mutations, mode=self.mode, tier=self.tier,
            advisory_thresholds=self.advisory_thresholds,
            small_file_bytes=self.small_file_bytes)

    def _location(self, payload):
        if not isinstance(payload, dict):
            return None
        candidates = [payload, *[value for value in _rows(payload) if isinstance(value, dict)]]
        item = next((value for value in candidates if any(
            value.get(key) for key in ("file_path", "path", "file", "absolute_path"))), None)
        if item is None:
            return None
        raw_path = next(str(item[key]) for key in ("file_path", "path", "file", "absolute_path")
                        if item.get(key))
        root = self.root_path or payload.get("root_path")
        if not root:
            return None
        root = os.path.realpath(root)
        absolute = os.path.realpath(raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path))
        if os.path.commonpath((root, absolute)) != root:
            return None
        path = os.path.relpath(absolute, root).replace(os.sep, "/")
        if self._excluded(path):
            return None
        qualified = next((str(item[key]) for key in ("qualified_name", "qualifiedName", "symbol", "name")
                          if item.get(key)), "")
        line_range = item.get("lines") or item.get("line_range")
        start = item.get("start_line") or item.get("startLine") or item.get("line")
        end = item.get("end_line") or item.get("endLine")
        if isinstance(line_range, str):
            digits = re.findall(r"\d+", line_range)
            if len(digits) >= 2:
                start, end = int(digits[0]), int(digits[1])
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            return None
        if start < 1 or end < start:
            return None
        return path, qualified, start, end, item

    def _mark_snippet(self, path, body):
        for known, entries in self.verified.items():
            if _same(path, known):
                for region in entries:
                    if region.text == body:
                        self.snippet_regions.add((known, region.start, region.end))

    def _identity(self, path, qualified, source, label, declaration_index=None):
        """Require module path plus the full Python enclosing-scope chain."""
        if not path.endswith((".py", ".pyi")) or not qualified:
            return False
        module = path.rsplit(".", 1)[0].replace("/", ".")
        parts = qualified.split(".")
        module_parts = module.split(".")
        for offset in range(len(parts) - len(module_parts)):
            if parts[offset:offset + len(module_parts)] == module_parts:
                scopes = parts[offset + len(module_parts):]
                break
        else:
            return False
        if not scopes:
            return False
        lines = source.splitlines()
        leaf = scopes[-1]
        kind = "class" if label.lower() == "class" else r"(?:async\s+def|def)"
        definition = re.compile(r"^(\s*)" + kind + r"\s+" + re.escape(leaf) + r"\b")
        found = next(((i, match) for i, line in enumerate(lines)
                      if (match := definition.match(line))
                      and (declaration_index is None or i == declaration_index)), None)
        if found is None:
            return False
        index, match = found
        indent = len(match.group(1).expandtabs(8))
        for scope in reversed(scopes[:-1]):
            candidates = [(i, line) for i, line in enumerate(lines[:index])
                          if len(line) - len(line.lstrip()) < indent
                          and re.match(r"^\s*(?:class|async\s+def|def)\s+", line)]
            if not candidates:
                return False
            index, header = candidates[-1]
            if not re.match(r"^\s*(?:class|async\s+def|def)\s+" + re.escape(scope) + r"\b", header):
                return False
            indent = len(header) - len(header.lstrip())
        return True

    async def _verify_symbol(self, path, qualified, start, end, source, label,
                             read_tool, dispatch, run, candidate_line=None):
        body = _source_body(source)
        leaf = qualified.rsplit(".", 1)[-1]
        definition = re.compile(
            r"^\s*(?:class|async\s+def|def)\s+" + re.escape(leaf) + r"\b")
        declaration = next((index for index, line in enumerate(body.splitlines())
                            if definition.match(line)
                            and (candidate_line is None or start + index == candidate_line)), None)
        if declaration is None:
            return False, "none", body
        symbol_tool = self._name(run, "get_symbol_info")
        if symbol_tool:
            column = body.splitlines()[declaration].find(leaf) + 1
            result = await self._dispatch(symbol_tool, {
                "filePath": path, "line": start + declaration, "column": column}, dispatch)
            if _error(result):
                return False, "jetbrains", body
            info = _payload(result)
            symbol_info = info.get("symbolInfo") if isinstance(info, dict) else None
            reported = (info.get("qualifiedName") or info.get("qualified_name")
                        if isinstance(info, dict) else None)
            if not reported and isinstance(symbol_info, dict):
                reported = (symbol_info.get("qualifiedName")
                            or symbol_info.get("qualified_name"))
            if (reported and "." in reported
                    and (reported == qualified or qualified.endswith("." + reported))):
                return True, "jetbrains", body
            # JetBrains may return only name/declarationText, not a qualified
            # name. That is insufficient to prove the enclosing scope.
            if reported and "." in reported:
                return False, "jetbrains", body
        if self._identity(path, qualified, body, label, declaration):
            return True, "structural", body
        if start > 1:
            above_start = max(1, start - 40)
            above_args = ({"path": path, "start_line": above_start,
                           "end_line": start - 1, "refresh": True}
                          if self.role == "coder" else
                          {"file_path": path, "offset": above_start,
                           "limit": start - above_start})
            above = await self._dispatch(read_tool, above_args, dispatch)
            if not _error(above):
                self._record_read(above_args, above)
                combined = _source_body(above) + "\n" + body
                if self._identity(path, qualified, combined, label,
                                  declaration + len(_source_body(above).splitlines()) + 1):
                    return True, "structural", body
        return False, "structural", body

    async def _relocate(self, path, qualified, label, read_tool, dispatch, run):
        leaf = qualified.rsplit(".", 1)[-1]
        query = f"def {leaf}" if label.lower() != "class" else f"class {leaf}"
        search_tool = self._name(run, ("codepartner_search_in_file" if self.role == "coder"
                                      else "search_in_files_by_text"))
        if not search_tool:
            self.snippet_outcomes["not_found"] += 1
            repo_tool = self._name(run, "search_text") or self._name(run, "search_regex")
            matches = (await self._dispatch(repo_tool, {"q": query}, dispatch)
                       if repo_tool else "")
            return (f"Symbol not found at KB location; KB hint stale. "
                    f"Search current IDE source for {query}.\n{matches}")
        search_args = ({"path": path, "query": query, "limit": 15}
                       if self.role == "coder" else {"q": query, "paths": [path]})
        found = await self._dispatch(search_tool, search_args, dispatch)
        if _error(found):
            return found
        candidate_lines = {int(value) for value in re.findall(
            re.escape(path) + r":(\d+)", found)}
        search_payload = _payload(found)
        if isinstance(search_payload, dict):
            for item in (search_payload.get("items") or
                         search_payload.get("matches") or []):
                if not isinstance(item, dict):
                    continue
                item_path = item.get("filePath") or item.get("file_path") or item.get("path")
                line = item.get("startLine") or item.get("line")
                if isinstance(item_path, str) and _same(path, item_path):
                    try:
                        candidate_lines.add(int(line))
                    except (TypeError, ValueError):
                        pass
        candidates = sorted(candidate_lines)[:15]
        if not candidates:
            self.snippet_outcomes["not_found"] += 1
            repo_tool = next((self._name(run, item) for item in ("search_text", "search_regex")
                              if self._name(run, item)), None)
            matches = (await self._dispatch(repo_tool, {"q": query}, dispatch)
                       if repo_tool else "")
            return f"Symbol not found at KB location; KB hint stale.\n{matches}"
        verified = []
        for candidate in candidates:
            low, high = max(1, candidate - 40), candidate + min(self.snippet_max_lines - 40, 80)
            read_args = ({"path": path, "start_line": low, "end_line": high, "refresh": True}
                         if self.role == "coder" else
                         {"file_path": path, "offset": low, "limit": high - low + 1})
            source = await self._dispatch(read_tool, read_args, dispatch)
            if _error(source):
                continue
            self._record_read(read_args, source)
            valid, method, body = await self._verify_symbol(
                path, qualified, low, high, source, label, read_tool, dispatch, run,
                candidate_line=candidate)
            if valid:
                verified.append((candidate, method, body))
        if len(verified) == 1:
            self.snippet_outcomes["relocated"] += 1
            line, method, body = verified[0]
            self._mark_snippet(path, body)
            return (f"Verified relocated current IDE source ({method}): {qualified}, "
                    f"{path}:{line}. KB hint stale.\n{body}")
        if len(verified) > 1:
            self.snippet_outcomes["ambiguous"] += 1
            return f"Ambiguous current definitions for {qualified} in {path}: {', '.join(str(item[0]) for item in verified)}. Read the right one."
        self.snippet_outcomes["not_found"] += 1
        return f"Symbol not found at KB location; KB hint stale. Candidate lines: {candidates}"

    async def _snippet(self, arguments, dispatch, run):
        raw = await dispatch("kb_get_code_snippet", arguments)
        if _error(raw):
            return raw
        payload = _payload(raw)
        if payload is None:
            self.snippet_outcomes["unresolved"] += 1
            return "Codebase Memory snippet unresolved: no parseable symbol location."
        await self._ensure_index(dispatch, run)
        location = self._location(payload)
        if location is None:
            self.snippet_outcomes["unresolved"] += 1
            return "Codebase Memory snippet unresolved: path outside indexed root or invalid range."
        path, qualified, start, end, item = location
        self.hinted_paths.add(path)
        if end - start + 1 > self.snippet_max_lines:
            self.snippet_outcomes["too_large"] += 1
            members = item.get("members") or payload.get("members") or []
            listing = "\n".join(str(member) for member in members[:30])
            return (f"{qualified} at {path}:{start}–{end} is too large; inspect its "
                    f"members and read the needed current IDE ranges.\n{listing}")
        read_tool = self._name(run, "codepartner_read_file" if self.role == "coder" else "read_file")
        if not read_tool:
            self.snippet_outcomes["unresolved"] += 1
            return "Codebase Memory location found, but no authorized IDE source reader is advertised."
        read_args = ({"path": path, "start_line": start, "end_line": end, "refresh": True}
                     if self.role == "coder" else
                     {"file_path": path, "offset": start, "limit": end - start + 1})
        source = await self._dispatch(read_tool, read_args, dispatch)
        if _error(source):
            return source
        self._record_read(read_args, source)
        label = str(item.get("label") or item.get("kind") or "Function")
        valid, method, body = await self._verify_symbol(
            path, qualified, start, end, source, label, read_tool, dispatch, run)
        if not valid:
            self.fallback_reasons["content_mismatch"] += 1
            return await self._relocate(path, qualified, label, read_tool, dispatch, run)
        self.snippet_outcomes["verified"] += 1
        self.snippet_outcomes[method] += 1
        self._mark_snippet(path, body)
        return (f"Verified current IDE source ({method}): {qualified}, {path}:{start}–{end}. "
                f"KB callers: {item.get('callers', [])}; callees: {item.get('callees', [])}.\n{body}")

    async def _dispatch(self, name, args, dispatch):
        token = _GENERATED.set(True)
        try:
            self.generated_calls += 1
            self.generated_tools[name] += 1
            if name.rsplit("__", 1)[-1] not in _READS:
                self.levels["L1"] += 1
            return await dispatch(name, args)
        finally:
            _GENERATED.reset(token)

    async def _ensure_index(self, dispatch, run):
        if self._index_checked:
            return
        self._index_checked = True
        index_tool = self._name(run, "kb_index_status")
        if index_tool:
            raw = await self._dispatch(index_tool, {"format": "json"}, dispatch)
            payload = _payload(raw)
            if isinstance(payload, dict):
                self.root_path = self.root_path or payload.get("root_path")
                self.indexed_at = self.indexed_at or payload.get("indexed_at")
            elif _error(raw):
                self.fallback_reasons["kb_unavailable"] += 1
        if self.recent_mutations and self.indexed_at:
            try:
                since = (float(self.indexed_at) if isinstance(self.indexed_at, (int, float))
                         else datetime.fromisoformat(str(self.indexed_at).replace("Z", "+00:00")).timestamp())
                for record in await self.recent_mutations(since):
                    for path in record.get("targets", ()):
                        if isinstance(path, str):
                            self.recent_paths.add(path)
            except (TypeError, ValueError, OverflowError):
                self.fallback_reasons["stale_signal"] += 1

    def _excluded(self, path):
        return any(_same(path, excluded) or path.startswith(excluded.rstrip("/") + "/")
                   for excluded in self.excluded_files)

    def _filter_kb(self, result):
        if not self.excluded_files or not isinstance(result, str):
            return result
        payload = _payload(result)
        if payload is None:
            return "\n".join(line for line in result.splitlines()
                             if not any(excluded in line for excluded in self.excluded_files))

        def clean(item):
            if isinstance(item, list):
                return [value for value in (clean(value) for value in item) if value is not None]
            if isinstance(item, dict):
                if any(self._excluded(str(item.get(key))) for key in
                       ("path", "file", "file_path", "filePath", "pathInProject")
                       if item.get(key)):
                    return None
                return {key: clean(value) for key, value in item.items()}
            if isinstance(item, str) and any(
                    excluded in item for excluded in self.excluded_files):
                return None
            return item

        return json.dumps(clean(payload), ensure_ascii=False)

    def _search_query(self, arguments):
        query = next((arguments.get(key) for key in ("q", "query", "text", "searchText", "regex", "pattern")
                      if isinstance(arguments.get(key), str)), None)
        if not query:
            return None, "literal"
        raw = re.sub(r"^(?:\\b|\^)+", "", query.strip())
        raw = re.sub(r"(?:\\b)+$", "", raw)
        names = [part.strip() for part in raw.split("|")]
        if not names or not all(_IDENTIFIER.fullmatch(part) for part in names):
            if re.search(r"\b(?:import|from)\b", query):
                return None, "intent_import"
            if re.search(r"(?:\w\s*\(|\.\w|=\s*\w)", query):
                return None, "intent_usage"
            return None, "literal"
        for key in ("paths", "path", "file_path", "fileMask", "file_mask", "directory"):
            value = str(arguments.get(key) or "").lower()
            if value and re.search(r"(?:test|docs?|\.md|config|\.ya?ml|\.json)", value):
                return None, "intent_tests_docs"
            if value and key in {"paths", "path", "file_path", "directory"}:
                return None, "intent_tests_docs"
        return names, None

    def _match_paths(self, result):
        for path in paths_in(result):
            if not self._excluded(path):
                self.hinted_paths.add(path)

    def _record_read(self, arguments, result):
        if _error(result):
            return
        path, start, end, ranged = read_region(arguments)
        if not path:
            return
        start = start or 1
        body = _source_body(result)
        lines = body.splitlines()
        end = min(end or start + len(lines) - 1, start + len(lines) - 1)
        if end < start:
            return
        previous = self.verified[path]
        redundant = any(region.start <= start and region.end >= end for region in previous)
        if redundant:
            self.redundant_source_reads += 1
            self.redundant_source_bytes += len(result.encode())
        strength = next((value for known, value in self.seed.paths.items()
                         if _same(path, known)), "weak" if any(
                             _same(path, hint) for hint in self.hinted_paths) else "none")
        self.reads_by_strength[strength] += 1
        unexplained = strength in {"weak", "none"}
        self.unexplained_reads += int(unexplained)
        self.flagged_reads += int(redundant or unexplained)
        self.verified[path] = [region for region in previous
                               if region.end < start or region.start > end]
        self.snippet_regions = {item for item in self.snippet_regions
                                if not (_same(item[0], path) and item[1] <= end
                                        and item[2] >= start)}
        whole = (not ranged and "truncated=true" not in result)
        self.verified[path].append(VerifiedRegion(start, end, body, whole))
        if len(self.verified[path]) > 32:
            self.verified[path] = self.verified[path][-32:]
        self.source_response_bytes += len(result.encode())
        if ranged or len(result.encode()) < self.small_file_bytes:
            self.levels["L2"] += 1
        else:
            self.levels["L3"] += 1
            if previous or any(_same(path, hint) for hint in self.hinted_paths):
                self.l3_escalated += 1
            else:
                self.l3_direct += 1
        if whole and len(result.encode()) >= self.small_file_bytes:
            self.full_reads += 1
        if (any(_same(path, gated) for gated in self.outline.gated)
                and not any(_same(path, counted) for counted in self.outline_followed)):
            self.outline_followed.add(path)
            if arguments.get("end_line") is not None or arguments.get("start_line", arguments.get("offset", 1)) != 1:
                self.outline_outcomes["refined_to_range"] += 1
            else:
                self.outline_outcomes["escalated_to_full"] += 1

    def _edit_covered(self, arguments, path):
        patch = (arguments.get("patch") or arguments.get("input")
                 or arguments.get("patch_text") or arguments.get("diff"))
        if isinstance(patch, str) and (
                f"*** Add File: {path}" in patch or
                re.search(r"(?m)^--- /dev/null\n\+\+\+ (?:b/)?" + re.escape(path) + r"$", patch)):
            return True, ""
        regions = [value for known, entries in self.verified.items() if _same(path, known)
                   for value in entries]
        old = next((arguments.get(key) for key in (
                    "old_text", "oldText", "old_string", "oldString",
                    "textToReplace", "oldContent", "searchText")
                    if isinstance(arguments.get(key), str)), None)
        if old:
            return any(old in region.text for region in regions), "the text you are replacing"
        if isinstance(patch, str) and "@@" in patch:
            hunks = re.findall(r"@@\s*-(\d+)(?:,(\d+))?", patch)
            for line, count in hunks:
                a, b = int(line), int(line) + max(int(count or 1), 1) - 1
                if not any(region.start <= a and region.end >= b for region in regions):
                    return False, f"lines {a}–{b}"
            return bool(hunks), "the patch's old lines"
        if any(key in arguments for key in ("content", "text", "new_text", "newText")):
            return any(region.whole for region in regions), "the whole file"
        return any(region.whole for region in regions), "the affected edit region"

    @staticmethod
    def _authored_patch_text(patch, previous):
        """Apply one verified patch hunk to an in-memory IDE-read region."""
        for block in re.split(r"(?m)^@@[^\n]*\n", patch)[1:]:
            old_lines, new_lines = [], []
            for line in block.splitlines():
                if line.startswith(("*** ", "diff --git ", "--- ", "+++ ")):
                    break
                if line.startswith((" ", "-")):
                    old_lines.append(line[1:])
                if line.startswith((" ", "+")):
                    new_lines.append(line[1:])
            old = "\n".join(old_lines)
            new = "\n".join(new_lines)
            if old and old in previous:
                previous = previous.replace(old, new, 1)
            elif old_lines:
                return None
        return previous

    async def call(self, name: str, arguments: dict, dispatch, *, run):
        bare, _, _ = dispatched(name, arguments, False)
        result = await self._call_inner(name, arguments, dispatch, run=run)
        if (bare.startswith("kb_") or bare in _SEARCHES | _READS
                or bare in {"codepartner_search_in_file", "search_file",
                            "list_directory_tree", "get_symbol_info"}):
            self.discovery_calls += 1
            if bare not in _READS and bare != "kb_get_code_snippet":
                self.levels["L1"] += 1
        if self.mode != "advise" or not self.advisory_thresholds or not isinstance(result, str):
            if isinstance(result, str):
                self.result_bytes += len(result.encode())
            return result
        values = {
            "discovery_calls": self.discovery_calls + self.generated_calls,
            "full_reads": self.full_reads,
            "retrieval_levels": (self.tracker.summary().get("retrieval_levels")
                                 if self.tracker else dict(self.levels)),
            "l3_direct": (self.tracker.summary().get("l3_direct")
                          if self.tracker else self.l3_direct),
            "l3_escalated": (self.tracker.summary().get("l3_escalated")
                             if self.tracker else self.l3_escalated),
            "est_added_tokens": (self.result_bytes + len(result.encode()) + 3) // 4,
            "repeated_or_unexplained_reads": self.flagged_reads,
        }
        crossed = {item["limit"] for item in self.threshold_crossings}
        new = [(key, values.get(key, 0), threshold)
               for key, threshold in self.advisory_thresholds.items()
               if key not in crossed and values.get(key, 0) >= threshold]
        if not new:
            self.result_bytes += len(result.encode())
            return result
        self.threshold_crossings += [{"limit": key, "value": value,
                                      "threshold": threshold}
                                     for key, value, threshold in new]
        labels = ", ".join(f"{key} {value}/{threshold}"
                           for key, value, threshold in new)
        advised = (result + f"\n\n[KB-first advisory, not a limit] {self.role}/{self.tier}: "
                   f"{labels}. Check whether the current IDE evidence is sufficient; "
                   "otherwise investigate the specific open question.")
        self.result_bytes += len(advised.encode())
        return advised

    async def _call_inner(self, name: str, arguments: dict, dispatch, *, run):
        if _GENERATED.get():
            return await dispatch(name, arguments)
        bare, actual, read = dispatched(name, arguments, name.rsplit("__", 1)[-1] in _READS)
        if bare == "kb_get_code_snippet":
            return await self._snippet(arguments, dispatch, run)
        if bare == "kb_search_graph" and isinstance(actual.get("name_pattern"), str):
            self.graph_searched.update(actual["name_pattern"].split("|"))
        if is_mutating_tool(bare):
            targets = (mutation_targets("patch", actual) if bare == "apply_patch"
                       else mutation_targets("replace", actual))
            path = _target(actual) or (targets[0] if len(targets) == 1 else None)
            paths = targets or ((path,) if path else ())
            prior = [(known, region) for known, entries in self.verified.items()
                     if any(_same(target, known) for target in paths) for region in entries]
            for target in paths:
                if bare in {"create_new_file", "create_new_file_with_text"}:
                    continue
                if any(_same(target, hint) for hint in self.hinted_paths):
                    covered, location = self._edit_covered(actual, target)
                    if not covered:
                        self.edit_verification_blocks += 1
                        return (f"Read {location} of {target} through the IDE first; "
                                "the KB location is only a hint.")
                    old = next((actual.get(key) for key in
                                ("old_text", "oldText", "old_string", "oldString")
                                if isinstance(actual.get(key), str)), None)
                    if old and any(
                        _same(target, known) and old in region.text
                        and (known, region.start, region.end) in self.snippet_regions
                        for known, entries in self.verified.items() for region in entries):
                        self.edits_verified_by_snippet += 1
            result = await dispatch(name, arguments)
            if paths and not _error(result):
                self.mutated_paths.update(paths)
                for target in paths:
                    self.verified.pop(target, None)
                old = next((actual.get(key) for key in
                            ("old_text", "oldText", "old_string", "oldString",
                             "textToReplace", "oldContent", "searchText")
                            if isinstance(actual.get(key), str)), None)
                new = next((actual.get(key) for key in
                            ("new_text", "newText", "new_string", "newString",
                             "replacementText", "newContent")
                            if isinstance(actual.get(key), str)), None)
                if path and old is not None and new is not None:
                    for known, region in prior:
                        if old in region.text:
                            self.verified[path].append(VerifiedRegion(
                                region.start, region.end,
                                region.text.replace(old, new, 1), region.whole))
                elif path and bare == "apply_patch" and len(paths) == 1:
                    patch = next((actual.get(key) for key in
                                  ("input", "patch", "patch_text", "text")
                                  if isinstance(actual.get(key), str)), None)
                    if patch:
                        for known, region in prior:
                            updated = self._authored_patch_text(patch, region.text)
                            if updated is not None and updated != region.text:
                                self.verified[path].append(VerifiedRegion(
                                    region.start, region.end, updated, region.whole))
            return result
        graph_tool = self._name(run, "kb_search_graph")
        if bare in _SEARCHES and graph_tool:
            names, reason = self._search_query(actual)
            signature = json.dumps([name, arguments], sort_keys=True, default=str)
            if names and signature in self.gated_searches:
                reason = "repeat"
                self.search_outcomes["kb_insufficient"] += 1
            elif names and any(item in self.graph_searched | self.gated_identifiers for item in names):
                reason = "already_graph_searched"
            elif names and len(self.gated_searches) >= self.search_gate_max:
                reason = "cap_reached"
            if names and reason is None:
                query = "|".join(names)
                result = await self._dispatch(graph_tool, {
                    "name_pattern": query, "limit": 15, "max_output_tokens": 1600}, dispatch)
                if _error(result):
                    reason = "authorization_refused" if result.startswith("Error: invalid MCP tool call") else "kb_error"
                elif not _rows(_payload(result)):
                    reason = "no_graph_match"
                else:
                    self.gated_searches.add(signature)
                    self.gated_identifiers.update(names)
                    filtered = self._filter_kb(result)
                    if not _rows(_payload(filtered)):
                        self.gated_searches.discard(signature)
                        self.gated_identifiers.difference_update(names)
                        reason = "no_graph_match"
                    else:
                        self._match_paths(filtered)
                        return (f"TEXT SEARCH NOT EXECUTED. Showing Codebase Memory definitions for {query} "
                                "(navigation hints from the last index, not text matches). To run the "
                                "original text search (usages, strings, comments, non-code), repeat the "
                                f"same call; it will execute unchanged.\n{filtered}")
            if reason:
                self.search_outcomes[reason] += 1
                if reason in {"kb_error", "no_graph_match"}:
                    self.fallback_reasons[reason] += 1
        outline_tool = self._name(run, "kb_get_file_outline")
        if read and outline_tool:
            path = _target(actual)
            await self._ensure_index(dispatch, run)
            if path and any(_same(path, item) for item in self.recent_paths):
                self.fallback_reasons["stale_signal"] += 1
            if path and not self._excluded(path) and not any(_same(path, item) for item in
                    self.verify_paths | self.mutated_paths | self.recent_paths):
                async def outline(args):
                    return await self._dispatch(outline_tool, args, dispatch)
                self.outline.outline = outline
                before = dict(self.outline.passthrough)
                gated = await self.outline.check(actual)
                for reason, fallback in (("not_indexed", "not_indexed"),
                                         ("error", "kb_error")):
                    if self.outline.passthrough[reason] > before[reason]:
                        self.fallback_reasons[fallback] += 1
                if gated is not None:
                    self.hinted_paths.add(gated[0])
                    self.outline_outcomes["served"] += 1
                    return gated[1]
        result = await dispatch(name, arguments)
        if read:
            self._record_read(actual, result)
            self.outline.note(name, arguments, True)
        elif bare.startswith("kb_"):
            result = self._filter_kb(result)
            self._match_paths(result)
            self.outline.note(name, arguments, False)
        return result

    def summary(self):
        self.search_outcomes["kb_sufficient"] = max(
            0, len(self.gated_searches) - self.search_outcomes["kb_insufficient"])
        outline_sufficient = max(0, self.outline_outcomes["served"]
                                 - self.outline_outcomes["refined_to_range"]
                                 - self.outline_outcomes["escalated_to_full"])
        return {
            **(self.tracker.summary() if self.tracker else {}),
            "search_gate": dict(self.search_outcomes),
            "outline_gate": {**self.outline.summary(), **self.outline_outcomes,
                             "outline_sufficient": outline_sufficient},
            "snippet": dict(self.snippet_outcomes),
            "kb_fallback_or_escalation_count": (self.search_outcomes["kb_insufficient"]
                + self.outline_outcomes["escalated_to_full"] + self.snippet_outcomes["not_found"]
                + self.snippet_outcomes["ambiguous"]),
            "redundant_source_reads": self.redundant_source_reads,
            "redundant_source_bytes": self.redundant_source_bytes,
            "source_response_bytes": self.source_response_bytes,
            "est_added_tokens": (self.tracker.summary().get("est_added_tokens", 0)
                                 if self.tracker else (self.result_bytes + 3) // 4),
            "result_bytes": self.result_bytes,
            "edit_verification_blocks": self.edit_verification_blocks,
            "edits_verified_by_snippet": self.edits_verified_by_snippet,
            "fallback_reasons": dict(self.fallback_reasons),
            "generated_calls": self.generated_calls,
            "kb_first_generated": dict(self.generated_tools),
            "discovery_calls": self.discovery_calls + self.generated_calls,
            "full_reads": self.full_reads,
            "evidence_strength": (self.tracker.summary().get("evidence_strength")
                                  if self.tracker else dict(self.reads_by_strength)),
            "unexplained_reads": (self.tracker.summary().get("unexplained_reads")
                                  if self.tracker else self.unexplained_reads),
            "threshold_crossings": (self.tracker.summary().get("threshold_crossings")
                                    if self.tracker else list(self.threshold_crossings)),
        }

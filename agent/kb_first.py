"""Planner discovery telemetry: evidence seeds and the advise-mode discovery tracker.

KB-first is a prompt rule (`agent.prompt.PromptBuilder.read_pipeline`); nothing here
intercepts or refuses a tool call."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from agent.planning.knowledge_base import _identifiers
from agent.paths import norm as _norm, paths_in
from mcp_bridge.read_metrics import dispatched, failed_read, read_region

STRENGTHS = ("strong", "moderate", "weak", "none")
_RANK = {name: rank for rank, name in enumerate(STRENGTHS)}
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


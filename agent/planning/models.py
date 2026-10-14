"""Structured Work Package planning artifacts.

The planner maintains a `WPPlanDraft` (human-readable plan plus structured
context) on every turn. Approval freezes one revision into an `ApprovedWPPlan`
without another model call. Every object is versioned and serializable so a
later knowledge base or provenance layer can persist and index it unchanged.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import PurePosixPath

SCHEMA_VERSION = 1
FINDING_KINDS = ("file", "symbol", "architecture", "constraint", "test",
                 "existing_impl", "risk", "assumption")


def plan_id_for(session_id: str, revision: int) -> str:
    """Approval of one session revision always yields the same plan id."""
    return f"plan_{session_id}_r{revision}"


def _strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).split()) for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()]


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _known(cls, value: dict) -> dict:
    names = {item.name for item in fields(cls)}
    return {key: item for key, item in value.items() if key in names}


@dataclass(frozen=True)
class DiscoverySymbol:
    name: str
    start_line: int | None = None
    end_line: int | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "DiscoverySymbol | None":
        name = _text(value.get("name"))
        start, end = value.get("start_line"), value.get("end_line")
        if (not name or start is not None and (type(start) is not int or start < 1)
                or end is not None and (type(end) is not int or end < 1)
                or start is not None and end is not None and end < start):
            return None
        return cls(name, start, end)


@dataclass(frozen=True)
class DiscoveryFile:
    file: str
    reason: str
    relevant_symbols: list[DiscoverySymbol] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict) -> "DiscoveryFile | None":
        path, reason = _text(value.get("file")), _text(value.get("reason"))
        parts = path.split("/")
        if (not path or not reason or PurePosixPath(path).is_absolute()
                or any(part in {"", ".", "..", ".git"} for part in parts)
                or "\\" in path):
            return None
        raw_symbols = value.get("relevant_symbols") or []
        if not isinstance(raw_symbols, list):
            return None
        symbols = [symbol for item in raw_symbols if isinstance(item, dict)
                   if (symbol := DiscoverySymbol.from_dict(item)) is not None]
        return cls(path, reason, symbols,
                   [item.upper() for item in _strings(value.get("task_ids"))])


def _discovery(value) -> list[DiscoveryFile]:
    return [entry for item in value if isinstance(item, dict)
            if (entry := DiscoveryFile.from_dict(item)) is not None] if isinstance(value, list) else []


@dataclass(frozen=True)
class PlannerFinding:
    id: str
    kind: str
    statement: str
    evidence: list[dict] = field(default_factory=list)  # [{path, symbol?, lines?}]
    validated: bool = False
    task_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict) -> "PlannerFinding":
        evidence = []
        for item in value.get("evidence") or []:
            if isinstance(item, str) and item.strip():
                item = {"path": item.strip()}
            if isinstance(item, dict) and _text(item.get("path")):
                evidence.append({key: item[key] for key in ("path", "symbol", "lines")
                                 if item.get(key) not in (None, "")})
        kind = _text(value.get("kind")).lower()
        return cls(
            id=_text(value.get("id")), kind=kind if kind in FINDING_KINDS else "architecture",
            statement=_text(value.get("statement")), evidence=evidence,
            validated=bool(value.get("validated")),
            task_ids=[item.upper() for item in _strings(value.get("task_ids"))],
        )


@dataclass(frozen=True)
class WPSharedContext:
    wp_id: str
    objective: str
    requirements: list[str] = field(default_factory=list)
    architecture: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    validated_assumptions: list[str] = field(default_factory=list)
    do_not_modify: list[str] = field(default_factory=list)
    test_strategy: list[str] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)
    # Resolved at approval; drafts reference findings by id only.
    findings: list[PlannerFinding] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict, wp_id: str = "") -> "WPSharedContext":
        return cls(**{
            **{name: _strings(value.get(name)) for name in (
                "requirements", "architecture", "relevant_files", "symbols", "constraints",
                "decisions", "validated_assumptions", "do_not_modify", "test_strategy",
                "finding_ids")},
            "wp_id": _text(value.get("wp_id")) or wp_id,
            "objective": _text(value.get("objective")),
            "findings": [PlannerFinding.from_dict(item) for item in value.get("findings") or []
                         if isinstance(item, dict)],
        })


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    objective: str
    approach: str = ""
    requirements: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    existing_implementation: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    do_not_modify: list[str] = field(default_factory=list)
    expected_tests: list[str] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)
    findings: list[PlannerFinding] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict, task_id: str = "") -> "TaskContext":
        return cls(**{
            **{name: _strings(value.get(name)) for name in (
                "requirements", "relevant_files", "symbols", "existing_implementation",
                "depends_on", "constraints", "do_not_modify", "expected_tests", "finding_ids")},
            "task_id": (_text(value.get("task_id")) or task_id).upper(),
            "objective": _text(value.get("objective")),
            "approach": _text(value.get("approach")),
            "findings": [PlannerFinding.from_dict(item) for item in value.get("findings") or []
                         if isinstance(item, dict)],
        })


@dataclass(frozen=True)
class ReviewerContext:
    wp_id: str
    objective: str
    approved_plan: str = ""
    architecture: list[str] = field(default_factory=list)
    expected_behavior: list[str] = field(default_factory=list)
    task_boundaries: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    validation_expectations: list[str] = field(default_factory=list)
    findings: list[PlannerFinding] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict, wp_id: str = "") -> "ReviewerContext":
        return cls(**{
            **{name: _strings(value.get(name)) for name in (
                "architecture", "expected_behavior", "task_boundaries", "dependencies",
                "relevant_files", "risks", "assumptions", "validation_expectations")},
            "wp_id": _text(value.get("wp_id")) or wp_id,
            "objective": _text(value.get("objective")),
            "approved_plan": _text(value.get("approved_plan")),
            "findings": [PlannerFinding.from_dict(item) for item in value.get("findings") or []
                         if isinstance(item, dict)],
        })


@dataclass(frozen=True)
class WPPlanDraft:
    """The live planning state the planner rewrites on every turn."""

    wp_id: str
    revision: int
    summary: str
    plan_markdown: str
    shared: WPSharedContext
    tasks: dict[str, TaskContext]
    reviewer: ReviewerContext
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    findings: list[PlannerFinding] = field(default_factory=list)
    discovery: list[DiscoveryFile] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict, wp_id: str = "", revision: int | None = None
                  ) -> "WPPlanDraft":
        """Build from stored state or from planner JSON (tolerating loose shapes)."""
        wp_id = _text(value.get("wp_id")) or wp_id
        raw_tasks = value.get("tasks") or {}
        if isinstance(raw_tasks, list):
            raw_tasks = {_text(item.get("task_id")).upper(): item
                         for item in raw_tasks if isinstance(item, dict)}
        tasks = {}
        for key, item in raw_tasks.items():
            if isinstance(item, dict):
                context = TaskContext.from_dict(item, str(key))
                tasks[context.task_id] = context
        return cls(
            wp_id=wp_id,
            revision=revision if revision is not None else int(value.get("revision") or 1),
            summary=_text(value.get("summary")),
            plan_markdown=_text(value.get("plan_markdown")),
            shared=WPSharedContext.from_dict(value.get("shared") or {}, wp_id),
            tasks=tasks,
            reviewer=ReviewerContext.from_dict(value.get("reviewer") or {}, wp_id),
            decisions=_strings(value.get("decisions")),
            open_questions=_strings(value.get("open_questions")),
            findings=[PlannerFinding.from_dict(item) for item in value.get("findings") or []
                      if isinstance(item, dict)],
            discovery=_discovery(value.get("discovery")),
        )


@dataclass(frozen=True)
class ApprovedWPPlan:
    """A frozen, approved plan: the execution context for coders and the reviewer."""

    plan_id: str
    session_id: str
    change_id: str
    wp_id: str
    wp_fingerprint: str
    wp_complexity: str
    revision: int
    planner_model: str
    summary: str
    plan_markdown: str
    shared: WPSharedContext
    tasks: dict[str, TaskContext]
    reviewer: ReviewerContext
    findings: list[PlannerFinding]
    discovery: list[DiscoveryFile] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    approved_at: float = 0.0
    fingerprint: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ApprovedWPPlan":
        wp_id = value["wp_id"]
        return cls(**{
            **_known(cls, value),
            "shared": WPSharedContext.from_dict(value.get("shared") or {}, wp_id),
            "tasks": {key: TaskContext.from_dict(item, key)
                      for key, item in (value.get("tasks") or {}).items()},
            "reviewer": ReviewerContext.from_dict(value.get("reviewer") or {}, wp_id),
            "findings": [PlannerFinding.from_dict(item) for item in value.get("findings") or []],
            "discovery": _discovery(value.get("discovery")),
            "decisions": list(value.get("decisions") or []),
            "provenance": dict(value.get("provenance") or {}),
        })

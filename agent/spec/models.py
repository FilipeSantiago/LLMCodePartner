"""Structured OpenSpec change model and deterministic Markdown renderers."""

from dataclasses import asdict, dataclass, field
from datetime import date

# Whole-package planning difficulty. Independent of task complexity
# (simple/medium/complex), which routes coders; this one routes planners.
WP_COMPLEXITIES = ("low", "medium", "high", "complex")


@dataclass(frozen=True)
class ModelIdentity:
    route_name: str
    provider: str
    configured_model: str
    reasoning_effort: str | None
    label: str


@dataclass(frozen=True)
class CompletionReview:
    coder: ModelIdentity
    reviewer: ModelIdentity
    reason: str
    evidence: list[str]

    @classmethod
    def from_dict(cls, value: dict) -> "CompletionReview":
        return cls(
            coder=ModelIdentity(**value["coder"]),
            reviewer=ModelIdentity(**value["reviewer"]),
            reason=value["reason"],
            evidence=list(value.get("evidence", [])),
        )


@dataclass(frozen=True)
class ValidationContract:
    required: list[str] = field(default_factory=list)
    preferred: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    not_required: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict | None) -> "ValidationContract":
        value = value or {}
        return cls(
            required=list(value.get("required", [])),
            preferred=list(value.get("preferred", [])),
            optional=list(value.get("optional", [])),
            not_required=list(value.get("not_required", [])),
        )


@dataclass(frozen=True)
class WorkPackageReview:
    outcome: str
    reason: str
    evidence: list[str] = field(default_factory=list)
    failure_type: str | None = None
    reviewer: ModelIdentity | None = None
    validation_results: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict | None) -> "WorkPackageReview | None":
        if value is None:
            return None
        reviewer = value.get("reviewer")
        return cls(
            outcome=value["outcome"],
            reason=value["reason"],
            evidence=list(value.get("evidence", [])),
            failure_type=value.get("failure_type"),
            reviewer=ModelIdentity(**reviewer) if reviewer else None,
            validation_results=list(value.get("validation_results", [])),
        )


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    description: str
    complexity: str
    reasoning: str
    context: str
    preferred_capability: str
    acceptance_criteria: list[str] = field(default_factory=list)
    # Legacy execution hint; review determines completion from the task outcome.
    required_operations: list[str] = field(default_factory=lambda: ["file_replace", "file_patch"])
    completed: bool = False
    completion_review: CompletionReview | None = None
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    planning_revision: int = 1
    reconciliation: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class WorkPackage:
    id: str
    title: str
    kind: str
    objective: str
    acceptance_criteria: list[str]
    requirement_refs: list[str]
    tasks: list[Task]
    validation: ValidationContract = field(default_factory=ValidationContract)
    depends_on: list[str] = field(default_factory=list)
    status: str = "planned"
    planning_revision: int = 1
    review: WorkPackageReview | None = None
    complexity: str | None = None


@dataclass(frozen=True)
class Epic:
    id: str
    title: str
    objective: str
    work_package_ids: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    status: str = "planned"


@dataclass(frozen=True)
class Scenario:
    name: str
    when: str
    then: str


@dataclass(frozen=True)
class Requirement:
    name: str
    statement: str
    scenarios: list[Scenario]


@dataclass(frozen=True)
class Capability:
    path: str
    purpose: str
    requirements: list[Requirement]


@dataclass(frozen=True)
class Decision:
    title: str
    rationale: str


@dataclass(frozen=True)
class Proposal:
    why: str
    what_changes: list[str]
    impact: list[str]


@dataclass(frozen=True)
class Design:
    context: str
    goals: list[str]
    non_goals: list[str]
    decisions: list[Decision]
    risks: list[str]


@dataclass(frozen=True)
class OpenSpecChange:
    id: str
    number: int
    slug: str
    title: str
    summary: str
    proposal: Proposal
    design: Design
    capabilities: list[Capability]
    work_packages: list[WorkPackage]
    follow_up_for: list[str] = field(default_factory=list)
    epic: Epic | None = None

    @property
    def path(self) -> str:
        return f"openspec/changes/{self.slug}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "OpenSpecChange":
        epic_value = value.get("epic")
        packages = value.get("work_packages", [])
        if epic_value is None:
            epic_value = {
                "id": value.get("epic_id") or f"EPIC{value.get('number', 0)}",
                "title": value.get("title", "Legacy OpenSpec change"),
                "objective": value.get("proposal", {}).get("why") or value.get("summary", ""),
                "work_package_ids": [package.get("id", "") for package in packages],
                "acceptance_criteria": [],
                "status": "planned",
            }
        return cls(
            id=value["id"],
            number=value["number"],
            slug=value["slug"],
            title=value["title"],
            summary=value["summary"],
            proposal=Proposal(**value["proposal"]),
            design=Design(
                **{
                    **value["design"],
                    "decisions": [Decision(**item) for item in value["design"]["decisions"]],
                }
            ),
            capabilities=[
                Capability(
                    path=capability["path"],
                    purpose=capability["purpose"],
                    requirements=[
                        Requirement(
                            name=requirement["name"],
                            statement=requirement["statement"],
                            scenarios=[
                                Scenario(**scenario) for scenario in requirement["scenarios"]
                            ],
                        )
                        for requirement in capability["requirements"]
                    ],
                )
                for capability in value["capabilities"]
            ],
            work_packages=[
                WorkPackage(
                    **{
                        **package,
                        "validation": ValidationContract.from_dict(package.get("validation")),
                        "review": WorkPackageReview.from_dict(package.get("review")),
                        "tasks": [Task(**{
                            **task,
                            "status": task.get(
                                "status", "completed" if task.get("completed") else "pending"
                            ),
                            "completion_review": _completion_review_from_dict(
                                task.get("completion_review")
                            ),
                        }) for task in package["tasks"]],
                    }
                )
                for package in packages
            ],
            follow_up_for=value.get("follow_up_for", []),
            epic=Epic(**epic_value),
        )


def _completion_review_from_dict(value: dict | None) -> CompletionReview | None:
    if value is None:
        return None
    return CompletionReview.from_dict(value)


def _one_line(value: str) -> str:
    return " ".join(value.split())


def render_metadata() -> str:
    return f"schema: spec-driven\ncreated: {date.today().isoformat()}\n"


def render_proposal(change: OpenSpecChange) -> str:
    new_capabilities = "\n".join(
        f"- `{capability.path}`: {capability.purpose}"
        for capability in change.capabilities
    )
    changes = "\n".join(f"- {item}" for item in change.proposal.what_changes)
    impact = "\n".join(f"- {item}" for item in change.proposal.impact)
    packages = "\n".join(
        f"- **{package.id} — {package.title}** ({package.kind}): {package.objective}"
        for package in change.work_packages
    )
    follow_up = (
        "## Follow-up For\n\n" + "\n".join(f"- `{task_id}`" for task_id in change.follow_up_for) + "\n\n"
        if change.follow_up_for else ""
    )
    epic = change.epic or Epic(
        f"EPIC{change.number}", change.title, change.proposal.why,
        [package.id for package in change.work_packages],
    )
    return (
        f"# Proposal: {change.title}\n\n"
        f"## Epic\n\n**{epic.id} — {epic.title}**\n\n{epic.objective}\n\n"
        f"## Why\n\n{change.proposal.why}\n\n"
        f"## What Changes\n\n{changes}\n\n"
        f"## Work Packages\n\n{packages}\n\n"
        f"{follow_up}"
        f"## Capabilities\n\n### New Capabilities\n\n{new_capabilities}\n\n"
        "### Modified Capabilities\n\n"
        "_None._\n\n"
        f"## Impact\n\n{impact}\n"
    )


def render_capability(capability: Capability) -> str:
    lines = ["# Spec Delta", "", "## Purpose", "", capability.purpose, "",
             "## ADDED Requirements", ""]
    for requirement in capability.requirements:
        lines.extend([
            f"### Requirement: {requirement.name}",
            requirement.statement,
            "",
        ])
        for scenario in requirement.scenarios:
            lines.extend([
                f"#### Scenario: {scenario.name}",
                f"- **WHEN** {scenario.when}",
                f"- **THEN** {scenario.then}",
                "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def render_design(change: OpenSpecChange) -> str:
    goals = "\n".join(f"- {item}" for item in change.design.goals)
    non_goals = (
        "\n".join(f"- {item}" for item in change.design.non_goals) or "- None."
    )
    decisions = "\n\n".join(
        f"### {item.title}\n\n{item.rationale}" for item in change.design.decisions
    )
    risks = "\n".join(f"- {item}" for item in change.design.risks) or "- None identified."
    packages = "\n\n".join(
        "\n".join([
            f"### {package.id} — {package.title}",
            "",
            package.objective,
            "",
            "**Acceptance criteria:**",
            *[f"- {criterion}" for criterion in package.acceptance_criteria],
            "",
            "**Requirements:**",
            *[f"- `{reference}`" for reference in package.requirement_refs],
            "",
            "**Validation contract:**",
            *[f"- Required: {item}" for item in package.validation.required],
            *[f"- Preferred: {item}" for item in package.validation.preferred],
            *[f"- Optional: {item}" for item in package.validation.optional],
            *[f"- Explicitly not required: {item}" for item in package.validation.not_required],
        ])
        for package in change.work_packages
    )
    return (
        f"# Design: {change.title}\n\n"
        f"## Context\n\n{change.design.context}\n\n"
        f"## Goals / Non-Goals\n\n**Goals:**\n{goals}\n\n"
        f"**Non-Goals:**\n{non_goals}\n\n"
        f"## Work Packages\n\n{packages}\n\n"
        f"## Decisions\n\n{decisions}\n\n"
        f"## Risks / Trade-offs\n\n{risks}\n"
    )


def render_tasks(change: OpenSpecChange) -> str:
    lines = [f"# Tasks: {change.title}", ""]
    for group_number, package in enumerate(change.work_packages, start=1):
        lines.extend([
            f"## {group_number}. {package.id} — {package.title}",
            "",
            f"**Objective:** {package.objective}",
            "",
        ])
        if package.complexity:
            lines.extend([f"**Complexity:** {package.complexity}", ""])
        for task in package.tasks:
            marker = "x" if task.completed or task.status == "completed" else " "
            lines.extend([
                f"- [{marker}] {task.id} {task.title}",
                f"  - Description: {task.description}",
                f"  - Status: {task.status}",
                f"  - Complexity: {task.complexity}",
                f"  - Reasoning: {task.reasoning}",
            ])
            if task.depends_on:
                lines.append(f"  - Depends on: {', '.join(task.depends_on)}")
            if task.context:
                lines.append(f"  - Context: {task.context}")
            if task.acceptance_criteria:
                lines.append("  - Definition of done:")
                lines.extend(f"    - {criterion}" for criterion in task.acceptance_criteria)
            lines.extend([
                f"  - Preferred capability: {task.preferred_capability}",
            ])
            if task.completed and task.completion_review:
                review = task.completion_review
                lines.extend([
                    f"  - Coded by: {review.coder.label}",
                    f"  - Reviewed by: {review.reviewer.label}",
                    f"  - Review decision: Approved — {_one_line(review.reason)}",
                ])
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_dashboard(changes: list[OpenSpecChange]) -> str:
    lines = ["# Active Work Packages", ""]
    for change in changes:
        lines.extend([
            f"## {change.id} — {change.title}",
            "",
            f"OpenSpec change: `{change.path}`",
            "",
        ])
        for package in change.work_packages:
            lines.extend([
                f"### {package.id} — {package.title}", "",
                f"Status: {package.status}", "",
            ])
            for task in package.tasks:
                marker = "x" if task.completed or task.status == "completed" else " "
                lines.append(f"- [{marker}] {task.id} {task.title}")
                if task.completed and task.completion_review:
                    review = task.completion_review
                    lines.append(
                        f"  - Coded by {review.coder.label}; reviewed by {review.reviewer.label}. "
                        f"Approved: {_one_line(review.reason)}"
                    )
            if package.review:
                lines.append(
                    f"- WP review: {package.review.outcome} — {_one_line(package.review.reason)}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_response(change: OpenSpecChange, updated_ids: list[str] | None = None) -> str:
    prefix = (
        f"Updated {', '.join(updated_ids)} in {change.id}.\n\n"
        if updated_ids
        else f"Created {change.id}: {change.title}\n\n"
    )
    return prefix + render_tasks(change)

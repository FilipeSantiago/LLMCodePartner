"""Structured OpenSpec change model and deterministic Markdown renderers."""

from dataclasses import asdict, dataclass, field
from datetime import date


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    description: str
    complexity: str
    reasoning: str
    context: str
    preferred_capability: str
    # Legacy execution hint; review determines completion from the task outcome.
    required_operations: list[str] = field(default_factory=lambda: ["file_replace", "file_patch"])
    completed: bool = False


@dataclass(frozen=True)
class WorkPackage:
    id: str
    title: str
    kind: str
    objective: str
    acceptance_criteria: list[str]
    requirement_refs: list[str]
    tasks: list[Task]


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

    @property
    def path(self) -> str:
        return f"openspec/changes/{self.slug}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "OpenSpecChange":
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
                        "tasks": [Task(**task) for task in package["tasks"]],
                    }
                )
                for package in value["work_packages"]
            ],
            follow_up_for=value.get("follow_up_for", []),
        )


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
    return (
        f"# Proposal: {change.title}\n\n"
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
        for task in package.tasks:
            marker = "x" if task.completed else " "
            lines.extend([
                f"- [{marker}] {task.id} {task.title}",
                f"  - Description: {task.description}",
                f"  - Complexity: {task.complexity}",
                f"  - Reasoning: {task.reasoning}",
            ])
            if task.context:
                lines.append(f"  - Context: {task.context}")
            lines.extend([
                f"  - Preferred capability: {task.preferred_capability}",
                "",
            ])
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
            lines.extend([f"### {package.id} — {package.title}", ""])
            for task in package.tasks:
                marker = "x" if task.completed else " "
                lines.append(f"- [{marker}] {task.id} {task.title}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_response(change: OpenSpecChange, updated_ids: list[str] | None = None) -> str:
    prefix = (
        f"Updated {', '.join(updated_ids)} in {change.id}.\n\n"
        if updated_ids
        else f"Created {change.id}: {change.title}\n\n"
    )
    return prefix + render_tasks(change)

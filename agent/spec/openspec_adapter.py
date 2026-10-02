"""Generate and refine complete, structured OpenSpec changes."""

import abc
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Callable

import providers
from agent.spec.models import (
    Capability,
    Decision,
    Design,
    OpenSpecChange,
    Proposal,
    Requirement,
    Scenario,
    Task,
    WorkPackage,
)
from providers.base import Provider


class OpenSpecError(RuntimeError):
    """OpenSpec could not produce a valid change."""


class OpenSpecAdapter(abc.ABC):
    @abc.abstractmethod
    async def generate_change(self, request: str) -> OpenSpecChange:
        """Generate every planning artifact for one new OpenSpec change."""
        ...

    @abc.abstractmethod
    async def refine_tasks(
            self, change: OpenSpecChange, task_ids: list[str], instruction: str
    ) -> dict[str, Task]:
        """Return replacements for exactly the selected tasks."""
        ...


class CliOpenSpecAdapter(OpenSpecAdapter):
    """Use OpenSpec templates and the active provider to create planning artifacts."""

    def __init__(self, provider_factory: Callable[[], Provider] = providers.active):
        self._provider_factory = provider_factory

    async def generate_change(self, request: str) -> OpenSpecChange:
        templates = await self._load_templates()
        shape = {
            "title": "Concise change title",
            "summary": "One-sentence change summary",
            "proposal": {
                "why": "Motivation",
                "what_changes": ["Concrete change"],
                "impact": ["Affected area"],
            },
            "design": {
                "context": "Current state and constraints",
                "goals": ["Goal"],
                "non_goals": ["Explicitly excluded scope"],
                "decisions": [{"title": "Decision", "rationale": "Reason and alternatives"}],
                "risks": ["Risk or trade-off"],
            },
            "capabilities": [{
                "path": "kebab-case/capability",
                "purpose": "At least fifty characters explaining this capability.",
                "requirements": [{
                    "name": "Requirement name",
                    "statement": "The system SHALL ...",
                    "scenarios": [{
                        "name": "Scenario name",
                        "when": "a condition occurs",
                        "then": "the expected behavior occurs",
                    }],
                }],
            }],
            "work_packages": [{
                "title": "Purpose-oriented package title",
                "kind": "user-story or technical-outcome or another accurate kind",
                "objective": "One cohesive outcome",
                "acceptance_criteria": ["Observable completion criterion"],
                "requirement_refs": ["kebab-case/capability#Requirement name"],
                "tasks": [{
                    "title": "Implementation task",
                    "description": "Concrete work",
                    "complexity": "simple",
                    "reasoning": "Why this task and complexity are appropriate",
                    "context": "Relevant context, or empty string",
                    "preferred_capability": "write",
                }],
            }],
        }
        system = (
            "Create a complete OpenSpec spec-driven change from the user's request. "
            "Return only one JSON object matching the exact shape below; do not add IDs, "
            "Markdown fences, commentary, or implementation. Decompose the request into "
            "one or more cohesive work packages. A work package owns tasks with one shared "
            "purpose and measurable acceptance criteria. Use kind=user-story only for a "
            "genuine user-value slice; use technical-outcome, migration, refactoring, "
            "research, or operational when more accurate. Every task belongs to exactly "
            "one work package. Every requirement has at least one scenario. Capability "
            "paths are safe kebab-case paths. Requirement references use "
            "'<capability-path>#<requirement-name>' and must resolve. Complexity is exactly "
            "simple, medium, or complex. Required strings and lists are non-empty, except "
            "task context, design non_goals, and design risks may be empty.\n\n"
            f"JSON shape:\n{json.dumps(shape, indent=2)}\n\n"
            "Follow the semantics of these installed OpenSpec templates:\n"
            + "\n\n".join(
                f"<{name}-template>\n{content}\n</{name}-template>"
                for name, content in templates.items()
            )
        )
        text = await self._complete(request, system, "change generation")
        return self._parse_change(text)

    async def refine_tasks(
            self, change: OpenSpecChange, task_ids: list[str], instruction: str
    ) -> dict[str, Task]:
        selected = {
            task.id: task
            for package in change.work_packages
            for task in package.tasks
            if task.id in task_ids
        }
        system = (
            "Refine exactly the selected implementation tasks according to the user's "
            "instruction. The complete owning OpenSpec change is supplied for context, "
            "but proposal, requirements, design, work packages, unselected tasks, task IDs, "
            "and completion state are immutable. Return only a JSON array. Return exactly "
            "one object per selected ID with exactly these fields: id, title, description, "
            "complexity, reasoning, context, preferred_capability. Complexity is simple, "
            "medium, or complex; context alone may be empty.\n\n"
            f"Selected tasks:\n{json.dumps({key: vars(value) for key, value in selected.items()}, indent=2)}\n\n"
            f"Complete change:\n{json.dumps(change.to_dict(), indent=2)}"
        )
        text = await self._complete(instruction, system, "task refinement")
        return self._parse_refined_tasks(text, selected)

    async def _complete(self, prompt: str, system: str, operation: str) -> str:
        try:
            text, _ = await self._provider_factory().complete(prompt, system)
            return text
        except Exception as exc:
            raise OpenSpecError(f"{operation} failed: {exc}") from exc

    @staticmethod
    async def _load_templates() -> dict[str, str]:
        binary = os.getenv("OPENSPEC_BIN", "openspec")
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                "templates",
                "--schema",
                "spec-driven",
                "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise OpenSpecError(f"OpenSpec CLI not found: {binary}") from exc

        stdout, stderr = await proc.communicate()
        if proc.returncode:
            detail = stderr.decode(errors="replace").strip()
            raise OpenSpecError(f"OpenSpec template lookup failed: {detail or proc.returncode}")
        try:
            payload = json.loads(stdout)
            templates = {}
            for name in ("proposal", "specs", "design", "tasks"):
                entry = payload[name]
                path = entry["path"] if isinstance(entry, dict) else entry
                templates[name] = Path(path).read_text(encoding="utf-8")
            return templates
        except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
            raise OpenSpecError("OpenSpec returned unusable spec-driven templates") from exc

    @classmethod
    def _parse_change(cls, content: str) -> OpenSpecChange:
        payload = cls._json_object(content, "generated change")
        cls._exact(payload, {
            "title", "summary", "proposal", "design", "capabilities", "work_packages"
        }, "change")
        proposal_value = cls._mapping(payload["proposal"], "proposal")
        cls._exact(proposal_value, {"why", "what_changes", "impact"}, "proposal")
        design_value = cls._mapping(payload["design"], "design")
        cls._exact(
            design_value,
            {"context", "goals", "non_goals", "decisions", "risks"},
            "design",
        )
        proposal = Proposal(
            why=cls._text(proposal_value["why"], "proposal.why"),
            what_changes=cls._strings(proposal_value["what_changes"], "proposal.what_changes"),
            impact=cls._strings(proposal_value["impact"], "proposal.impact"),
        )
        decisions = []
        for index, raw in enumerate(cls._list(design_value["decisions"], "design.decisions"), 1):
            item = cls._mapping(raw, f"decision {index}")
            cls._exact(item, {"title", "rationale"}, f"decision {index}")
            decisions.append(Decision(
                cls._text(item["title"], f"decision {index}.title"),
                cls._text(item["rationale"], f"decision {index}.rationale"),
            ))
        if not decisions:
            raise OpenSpecError("design.decisions must not be empty")
        design = Design(
            context=cls._text(design_value["context"], "design.context"),
            goals=cls._strings(design_value["goals"], "design.goals"),
            non_goals=cls._strings(design_value["non_goals"], "design.non_goals", empty=True),
            decisions=decisions,
            risks=cls._strings(design_value["risks"], "design.risks", empty=True),
        )
        capabilities = cls._parse_capabilities(payload["capabilities"])
        packages = cls._parse_work_packages(payload["work_packages"])
        valid_refs = {
            f"{capability.path}#{requirement.name}"
            for capability in capabilities
            for requirement in capability.requirements
        }
        for package in packages:
            unknown = set(package.requirement_refs) - valid_refs
            if unknown:
                raise OpenSpecError(
                    f"work package {package.title} has unknown requirement references: "
                    + ", ".join(sorted(unknown))
                )
        return OpenSpecChange(
            id="",
            number=0,
            slug="",
            title=cls._text(payload["title"], "title"),
            summary=cls._text(payload["summary"], "summary"),
            proposal=proposal,
            design=design,
            capabilities=capabilities,
            work_packages=packages,
        )

    @classmethod
    def _parse_capabilities(cls, raw: object) -> list[Capability]:
        capabilities = []
        for index, raw_capability in enumerate(cls._list(raw, "capabilities"), 1):
            value = cls._mapping(raw_capability, f"capability {index}")
            cls._exact(value, {"path", "purpose", "requirements"}, f"capability {index}")
            path = cls._text(value["path"], f"capability {index}.path")
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*(?:/[a-z0-9]+(?:-[a-z0-9]+)*)*", path):
                raise OpenSpecError(f"capability {index} has an unsafe path")
            purpose = cls._text(value["purpose"], f"capability {index}.purpose")
            if len(purpose) < 50:
                raise OpenSpecError(f"capability {index} purpose must be at least 50 characters")
            requirements = []
            for req_index, raw_requirement in enumerate(
                    cls._list(value["requirements"], f"capability {index}.requirements"), 1
            ):
                requirement = cls._mapping(
                    raw_requirement, f"capability {index} requirement {req_index}"
                )
                cls._exact(
                    requirement,
                    {"name", "statement", "scenarios"},
                    f"capability {index} requirement {req_index}",
                )
                scenarios = []
                for scenario_index, raw_scenario in enumerate(
                        cls._list(requirement["scenarios"], "requirement.scenarios"), 1
                ):
                    scenario = cls._mapping(raw_scenario, f"scenario {scenario_index}")
                    cls._exact(scenario, {"name", "when", "then"}, f"scenario {scenario_index}")
                    scenarios.append(Scenario(
                        cls._text(scenario["name"], "scenario.name"),
                        cls._text(scenario["when"], "scenario.when"),
                        cls._text(scenario["then"], "scenario.then"),
                    ))
                requirements.append(Requirement(
                    cls._text(requirement["name"], "requirement.name"),
                    cls._text(requirement["statement"], "requirement.statement"),
                    scenarios,
                ))
            capabilities.append(Capability(path, purpose, requirements))
        if not capabilities:
            raise OpenSpecError("capabilities must not be empty")
        return capabilities

    @classmethod
    def _parse_work_packages(cls, raw: object) -> list[WorkPackage]:
        packages = []
        for index, raw_package in enumerate(cls._list(raw, "work_packages"), 1):
            value = cls._mapping(raw_package, f"work package {index}")
            cls._exact(value, {
                "title", "kind", "objective", "acceptance_criteria",
                "requirement_refs", "tasks",
            }, f"work package {index}")
            tasks = [
                cls._parse_task(task, f"work package {index} task {task_index}")
                for task_index, task in enumerate(cls._list(value["tasks"], "tasks"), 1)
            ]
            packages.append(WorkPackage(
                id="",
                title=cls._text(value["title"], f"work package {index}.title"),
                kind=cls._text(value["kind"], f"work package {index}.kind"),
                objective=cls._text(value["objective"], f"work package {index}.objective"),
                acceptance_criteria=cls._strings(
                    value["acceptance_criteria"], f"work package {index}.acceptance_criteria"
                ),
                requirement_refs=cls._strings(
                    value["requirement_refs"], f"work package {index}.requirement_refs"
                ),
                tasks=tasks,
            ))
        if not packages:
            raise OpenSpecError("work_packages must not be empty")
        return packages

    @classmethod
    def _parse_task(cls, raw: object, label: str, task_id: str = "") -> Task:
        value = cls._mapping(raw, label)
        cls._exact(value, {
            "title", "description", "complexity", "reasoning", "context",
            "preferred_capability",
        }, label)
        complexity = cls._text(value["complexity"], f"{label}.complexity")
        if complexity not in {"simple", "medium", "complex"}:
            raise OpenSpecError(f"{label} has invalid complexity")
        context = value["context"]
        if not isinstance(context, str):
            raise OpenSpecError(f"{label}.context must be a string")
        return Task(
            id=task_id,
            title=cls._text(value["title"], f"{label}.title"),
            description=cls._text(value["description"], f"{label}.description"),
            complexity=complexity,
            reasoning=cls._text(value["reasoning"], f"{label}.reasoning"),
            context=" ".join(context.split()),
            preferred_capability=cls._text(
                value["preferred_capability"], f"{label}.preferred_capability"
            ),
        )

    @classmethod
    def _parse_refined_tasks(
            cls, content: str, selected: dict[str, Task]
    ) -> dict[str, Task]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OpenSpecError("refined tasks are not valid JSON") from exc
        if not isinstance(payload, list):
            raise OpenSpecError("refined tasks must be a JSON array")
        replacements = {}
        for index, raw in enumerate(payload, 1):
            value = cls._mapping(raw, f"refined task {index}")
            cls._exact(value, {
                "id", "title", "description", "complexity", "reasoning", "context",
                "preferred_capability",
            }, f"refined task {index}")
            task_id = cls._text(value["id"], f"refined task {index}.id").upper()
            task_value = {key: item for key, item in value.items() if key != "id"}
            parsed = cls._parse_task(task_value, f"refined task {task_id}", task_id)
            replacements[task_id] = Task(**{**vars(parsed), "completed": selected.get(
                task_id, Task("", "", "", "simple", "", "", "")
            ).completed})
        if set(replacements) != set(selected):
            raise OpenSpecError("refinement must return exactly the selected task IDs")
        return replacements

    @staticmethod
    def _json_object(content: str, label: str) -> dict:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OpenSpecError(f"{label} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise OpenSpecError(f"{label} must be a JSON object")
        return payload

    @staticmethod
    def _mapping(value: object, label: str) -> dict:
        if not isinstance(value, dict):
            raise OpenSpecError(f"{label} must be an object")
        return value

    @staticmethod
    def _list(value: object, label: str) -> list:
        if not isinstance(value, list) or not value:
            raise OpenSpecError(f"{label} must be a non-empty array")
        return value

    @classmethod
    def _strings(cls, value: object, label: str, empty: bool = False) -> list[str]:
        if not isinstance(value, list) or (not value and not empty):
            raise OpenSpecError(f"{label} must be {'an array' if empty else 'a non-empty array'}")
        return [cls._text(item, f"{label} item") for item in value]

    @staticmethod
    def _text(value: object, label: str) -> str:
        if not isinstance(value, str) or not " ".join(value.split()):
            raise OpenSpecError(f"{label} must be a non-empty string")
        return " ".join(value.split())

    @staticmethod
    def _exact(value: dict, fields: set[str], label: str) -> None:
        if set(value) != fields:
            raise OpenSpecError(f"{label} must contain exactly: {', '.join(sorted(fields))}")

"""Generate and refine complete, structured OpenSpec changes."""

import abc
import asyncio
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import providers
from agent.spec.models import (
    Capability,
    Decision,
    Design,
    Epic,
    OpenSpecChange,
    Proposal,
    Requirement,
    Scenario,
    Task,
    ValidationContract,
    WorkPackage,
)
from providers.base import Provider

log = logging.getLogger("codepartner.spec.openspec")

_FENCE = re.compile(r"\A\s*```[\w-]*[ \t]*\n(.*?)\n?```\s*\Z", re.DOTALL)
_JSON_ONLY_RETRY = (
    "\n\nYour previous reply was rejected because it was not bare JSON. Reply with the "
    "JSON value only: no Markdown fences, no commentary before or after it."
)


class OpenSpecError(RuntimeError):
    """OpenSpec could not produce a valid change."""


class MalformedJsonError(OpenSpecError):
    """The model reply contained no parseable JSON value (retryable)."""


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

    async def refine_change(self, change: OpenSpecChange, target_id: str,
                            instruction: str) -> OpenSpecChange:
        raise NotImplementedError


class CliOpenSpecAdapter(OpenSpecAdapter):
    """Use OpenSpec templates and the active provider to create planning artifacts."""

    def __init__(self, provider_factory: Callable[[], Provider] = providers.active):
        self._provider_factory = provider_factory

    async def generate_change(self, request: str) -> OpenSpecChange:
        templates = await self._load_templates()
        shape = {
            "title": "Concise change title",
            "summary": "One-sentence change summary",
            "epic": {
                "title": "Product capability or architectural initiative",
                "objective": "Why this group of features exists",
                "acceptance_criteria": ["Optional cross-WP integration outcome"],
            },
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
                "validation": {
                    "required": ["Minimum evidence required for review"],
                    "preferred": ["Useful validation when available"],
                    "optional": ["Broader confidence-building validation"],
                    "not_required": ["Validation explicitly outside this WP contract"],
                },
                "depends_on": [],
                "requirement_refs": ["kebab-case/capability#Requirement name"],
                "tasks": [{
                    "title": "Implementation task",
                    "description": "Concrete work",
                    "definition_of_done": ["Concrete bounded completion condition"],
                    "depends_on": [],
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
            "one Epic and one or more cohesive work packages. The Epic explains why the work "
            "exists. Each work package is a complete demonstrable feature and owns observable "
            "acceptance criteria plus an explicit required/preferred/optional/not-required "
            "validation contract. Tasks are bounded execution instructions with definitions of "
            "done and dependencies; do not turn them into miniature feature contracts or assign later "
            "implementation or testing outcomes to an earlier inspection task. Use "
            "kind=user-story only for a "
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
        return await self._complete_json(request, system, "change generation", self._parse_change)

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
            "acceptance_criteria, complexity, reasoning, context, preferred_capability. "
            "Acceptance criteria must remain scoped to that task. Complexity is simple, "
            "medium, or complex; context alone may be empty.\n\n"
            f"Selected tasks:\n{json.dumps({key: asdict(value) for key, value in selected.items()}, indent=2)}\n\n"
            f"Complete change:\n{json.dumps(change.to_dict(), indent=2)}"
        )
        return await self._complete_json(
            instruction, system, "task refinement",
            lambda text: self._parse_refined_tasks(text, selected))

    async def refine_change(self, change: OpenSpecChange, target_id: str,
                            instruction: str) -> OpenSpecChange:
        system = (
            "Refine the selected canonical OpenSpec planning object in a draft only. "
            "Return one JSON object containing the complete OpenSpec change in exactly the "
            "same serialized schema supplied below. Preserve the change id, number, slug, "
            "Epic id, existing work-package ids, existing task ids, completion records, and "
            "execution history. Do not create a versioned duplicate. You may add tasks with "
            "an empty id; CodePartner will allocate stable ids when the draft is applied. "
            "Tasks removed from the design must remain in the returned WP with status=cancelled "
            "so history is preserved. Acceptance criteria belong primarily to WPs; task "
            "acceptance_criteria is its bounded definition of done. Modify only the selected "
            f"object {target_id} and its descendants.\n\n"
            f"Current draft:\n{json.dumps(change.to_dict(), ensure_ascii=False, indent=2)}"
        )
        payload = await self._complete_json(
            instruction, system, "object refinement",
            lambda text: self._json_object(text, "refined change"))
        refined = OpenSpecChange.from_dict(payload)
        if (refined.id, refined.number, refined.slug) != (change.id, change.number, change.slug):
            raise OpenSpecError("refinement changed immutable OpenSpec change identity")
        if change.epic and (not refined.epic or refined.epic.id != change.epic.id):
            raise OpenSpecError("refinement changed immutable Epic identity")
        return refined

    async def _complete(self, prompt: str, system: str, operation: str) -> str:
        try:
            text, terminal = await self._provider_factory().complete(prompt, system)
        except Exception as exc:
            raise OpenSpecError(f"{operation} failed: {exc}") from exc
        log.info("openspec_completion operation=%s prompt_chars=%d reply_chars=%d terminal=%s",
                 operation, len(prompt), len(text), terminal)
        return text

    async def _complete_json(self, prompt: str, system: str, operation: str, parse):
        """Complete and parse, re-asking once when the reply holds no JSON value.

        Schema errors are not retried: they mean the model produced JSON that
        disagrees with the contract, which a "JSON only" reminder does not fix.
        """
        text = await self._complete(prompt, system, operation)
        try:
            return parse(text)
        except MalformedJsonError:
            self._log_malformed(operation, 1, text)
        text = await self._complete(prompt, system + _JSON_ONLY_RETRY, operation)
        try:
            return parse(text)
        except MalformedJsonError as exc:
            self._log_malformed(operation, 2, text)
            head = " ".join(text[:80].split())
            raise MalformedJsonError(
                f"{exc} after a retry ({len(text)} chars, starting {head!r})") from exc

    @staticmethod
    def _log_malformed(operation: str, attempt: int, text: str) -> None:
        log.warning("openspec_malformed_json operation=%s attempt=%d reply_chars=%d head=%r tail=%r",
                    operation, attempt, len(text), text[:300], text[-300:])

    @staticmethod
    def _extract_json(content: str, opener: str) -> str:
        """Unwrap a Markdown fence or surrounding prose around one JSON value."""
        closer = {"{": "}", "[": "]"}[opener]
        text = content.strip()
        fenced = _FENCE.match(text)
        if fenced:
            text = fenced.group(1).strip()
        if not text.startswith(opener):
            start, end = text.find(opener), text.rfind(closer)
            if start != -1 and end > start:
                text = text[start:end + 1]
        return text

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
        allowed = {
            "title", "summary", "proposal", "design", "capabilities", "work_packages", "epic"
        }
        if set(payload).difference(allowed) or set(payload).difference({"epic"}) != allowed.difference({"epic"}):
            raise OpenSpecError(
                "change must contain title, summary, proposal, design, capabilities, "
                "work_packages, and optional epic"
            )
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
        epic_value = payload.get("epic")
        epic = None
        if epic_value is not None:
            epic_value = cls._mapping(epic_value, "epic")
            cls._exact(epic_value, {"title", "objective", "acceptance_criteria"}, "epic")
            epic = Epic(
                id="",
                title=cls._text(epic_value["title"], "epic.title"),
                objective=cls._text(epic_value["objective"], "epic.objective"),
                acceptance_criteria=cls._strings(
                    epic_value["acceptance_criteria"], "epic.acceptance_criteria", empty=True
                ),
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
            epic=epic,
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
            legacy = {"title", "kind", "objective", "acceptance_criteria", "requirement_refs", "tasks"}
            modern = legacy | {"validation", "depends_on"}
            if frozenset(value) not in {frozenset(legacy), frozenset(modern)}:
                raise OpenSpecError(
                    f"work package {index} must contain the legacy fields or add validation and depends_on"
                )
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
                validation=ValidationContract.from_dict(value.get("validation")),
                depends_on=cls._strings(
                    value.get("depends_on", []), f"work package {index}.depends_on", empty=True
                ),
            ))
        if not packages:
            raise OpenSpecError("work_packages must not be empty")
        return packages

    @classmethod
    def _parse_task(cls, raw: object, label: str, task_id: str = "") -> Task:
        value = cls._mapping(raw, label)
        legacy = {"title", "description", "acceptance_criteria", "complexity", "reasoning", "context", "preferred_capability"}
        modern = {"title", "description", "definition_of_done", "depends_on", "complexity", "reasoning", "context", "preferred_capability"}
        if frozenset(value) not in {frozenset(legacy), frozenset(modern)}:
            raise OpenSpecError(f"{label} must contain either legacy task fields or definition_of_done and depends_on")
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
            acceptance_criteria=cls._strings(
                value.get("definition_of_done", value.get("acceptance_criteria")),
                f"{label}.definition_of_done"
            ),
            depends_on=cls._strings(value.get("depends_on", []), f"{label}.depends_on", empty=True),
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
            payload = json.loads(cls._extract_json(content, "["))
        except json.JSONDecodeError as exc:
            raise MalformedJsonError("refined tasks are not valid JSON") from exc
        if not isinstance(payload, list):
            raise OpenSpecError("refined tasks must be a JSON array")
        replacements = {}
        for index, raw in enumerate(payload, 1):
            value = cls._mapping(raw, f"refined task {index}")
            cls._exact(value, {
                "id", "title", "description", "acceptance_criteria", "complexity", "reasoning", "context",
                "preferred_capability",
            }, f"refined task {index}")
            task_id = cls._text(value["id"], f"refined task {index}.id").upper()
            task_value = {key: item for key, item in value.items() if key != "id"}
            parsed = cls._parse_task(task_value, f"refined task {task_id}", task_id)
            previous = selected.get(task_id)
            replacements[task_id] = Task(**{
                **vars(parsed),
                "completed": previous.completed if previous else False,
                "completion_review": previous.completion_review if previous else None,
            })
        if set(replacements) != set(selected):
            raise OpenSpecError("refinement must return exactly the selected task IDs")
        return replacements

    @classmethod
    def _json_object(cls, content: str, label: str) -> dict:
        try:
            payload = json.loads(cls._extract_json(content, "{"))
        except json.JSONDecodeError as exc:
            raise MalformedJsonError(f"{label} is not valid JSON") from exc
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

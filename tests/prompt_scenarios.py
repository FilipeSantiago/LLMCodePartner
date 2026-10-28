"""Prompt scenarios pinned by `tests/test_prompt_golden.py`.

Each scenario renders one agent prompt; the fixtures in `tests/fixtures/prompts` hold the
expected text. Regenerate them after an intended prompt change with
`python scripts/capture_prompt_goldens.py`.
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.spec.models import Task
from mcp_bridge.models import DoneEvent, TextEvent
from mcp_bridge.registry import ToolSpec
from providers import subagents

FIXTURES = Path(__file__).parent / "fixtures" / "prompts"
_ROOT = "<PROJECT_ROOT>"


def task() -> Task:
    return Task(
        id="WP1-T1", title="Add factory", description="Add the loader factory.",
        complexity="medium", reasoning="Crosses modules.",
        context="The project already has loaders.", preferred_capability="write",
        acceptance_criteria=["Factory returns a loader", "Unknown names raise"],
    )


def _request(purpose, roles=(subagents.READER,), scope=(), read_only=()):
    return subagents.Request(roles=tuple(roles), purpose=purpose, scope=tuple(scope),
                             read_only=tuple(read_only))


_CONTRACT = SimpleNamespace(goal="Build the loader factory", revision=2,
                            primary_files=("src/factory.py", "tests/test_factory.py"),
                            excluded_files=())
_PLAN = "Approved plan: reuse src/registry.py::REGISTRY.\nLine ranges: src/factory.py:1-40"
_HANDOFF = "Earlier task handoff: src/registry.py::register"

PLANNER = {
    "planner_ide_new": dict(brief="BRIEF", discussion=[], current=None,
                            message="implement WP1"),
    "planner_native_full": dict(
        brief="BRIEF", discussion=[], current=None, message="implement WP1", narrate=True,
        kb_status="KB: indexed", discovery_guidance=True, native=True,
        request=_request("planner"), file_inventory="src/\ntests/"),
    "planner_native_checks_revision": dict(
        brief="BRIEF", discussion=[{"revision": 1, "text": "implement WP1"}],
        current={"summary": "Old plan", "tasks": {}}, message="split T1",
        native=True, request=_request("planner", (subagents.READER, subagents.CHECKS))),
    "planner_ide_ignores_request": dict(
        brief="BRIEF", discussion=[], current=None, message="implement WP1",
        request=_request("planner")),
    "planner_codex_hosted_reader": dict(
        brief="BRIEF", discussion=[], current=None, message="implement WP1",
        native=True, hosted_reader=True, workspace="/workspace/project",
        kb_tools=frozenset({"kb_search_graph", "planner_declarations"}),
        file_inventory="src/\ntests/"),
}
PLANNER_REPAIR = ["tasks is missing WP1-T2", "summary is empty"]

CODER = {
    "coder_ide_plain": dict(),
    "coder_ide_planned_kb_first": dict(
        execution_instruction="Keep the API stable", contract=_CONTRACT,
        plan_context=_PLAN, handoff_context=_HANDOFF, kb_policy="kb_first"),
    "coder_ide_no_aci": dict(plan_context=_PLAN, aci_enabled=False),
    "coder_native_plain": dict(native=True),
    "coder_native_delegated": dict(
        native=True, contract=_CONTRACT, plan_context=_PLAN, handoff_context=_HANDOFF,
        request=_request("task", (subagents.READER, subagents.CHECKS),
                         ("src/factory.py", "src/registry.py"), ("src/registry.py",))),
    "coder_native_delegated_no_scope": dict(
        native=True, plan_context=_PLAN, request=_request("task")),
}
CODER_MESSAGES = [{"role": "system", "content": "SYSTEM PREFIX"},
                  {"role": "assistant", "content": "dropped"},
                  {"role": "user", "content": "transport command"}]
INVENTORY = "src/\ntests/"

SUBAGENTS = {
    "reader_planner": (subagents.READER, "planner"),
    "reader_task": (subagents.READER, "task"),
    "reader_reviewer": (subagents.READER, "reviewer"),
    "checks_task": (subagents.CHECKS, "task"),
}
DELEGATION = {
    "delegation_planner": _request("planner"),
    "delegation_planner_checks": _request("planner", (subagents.READER, subagents.CHECKS)),
    "delegation_task_scoped": _request(
        "task", (subagents.READER, subagents.CHECKS), ("src/a.py", "src/b.py"), ("src/b.py",)),
    "delegation_task_unscoped": _request("task"),
    "delegation_reviewer": _request("reviewer", scope=("src/a.py",)),
    "delegation_reviewer_checks": _request(
        "reviewer", (subagents.READER, subagents.CHECKS), read_only=("src/c.py",)),
}

REVIEWER = ("reviewer_ide", "reviewer_native_reader", "reviewer_native_plain",
            "reviewer_repair", "reviewer_native_diff")

# The diff scenario: a modified file, a new file, and a planned file left unchanged.
_BEFORE = "".join(f"line {number}\n" for number in range(1, 41))
_AFTER = _BEFORE.replace("line 20\n", "line 20 changed\n")
_REGISTRY = "".join(f"registry {number}\n" for number in range(1, 31))


# --- renderers through the current code ---------------------------------------------

def render_planner(case: dict) -> str:
    from agent.prompt import PlannerPromptContext, PromptFactory, PromptRole
    builder = PromptFactory.create(PromptRole.PLANNER)
    return builder.with_context(PlannerPromptContext(
        brief=case["brief"], discussion=case["discussion"], current=case["current"],
        message=case["message"], narrate=case.get("narrate", False),
        kb_status=case.get("kb_status", ""),
        discovery_guidance=case.get("discovery_guidance", False),
        native=case.get("native", False), subagent_request=case.get("request"),
        file_inventory=case.get("file_inventory", ""),
        workspace=case.get("workspace", ""),
        kb_tools=case.get("kb_tools", frozenset()),
        hosted_reader=case.get("hosted_reader", False))).build().text


def render_planner_repair() -> str:
    from agent.prompt import PromptFactory, PromptRole
    return PromptFactory.create(PromptRole.PLANNER).repair(PLANNER_REPAIR)


def render_coder(case: dict) -> str:
    from agent.routing.task_execution import TaskExecutionService
    messages = TaskExecutionService._execution_messages(
        task(), CODER_MESSAGES, case.get("execution_instruction", ""),
        case.get("contract"), case.get("plan_context", ""), case.get("handoff_context", ""),
        native=case.get("native", False), kb_policy=case.get("kb_policy", "selective"),
        subagent_request=case.get("request"), aci_enabled=case.get("aci_enabled", True))
    return json.dumps(messages, indent=2, ensure_ascii=False)


def render_inventory() -> str:
    from agent.prompt import PromptFactory, PromptRole
    builder = PromptFactory.create(PromptRole.CODER)
    return json.dumps(builder.with_workspace(CODER_MESSAGES, "/workspace/project", INVENTORY), indent=2)


def render_subagent(role: str, purpose: str) -> str:
    from agent.prompt import ChecksPromptContext, PromptFactory, ReaderPromptContext
    context = (ReaderPromptContext(purpose) if role == subagents.READER
               else ChecksPromptContext())
    return PromptFactory.create(role).with_context(context).build().text


def render_delegation(request) -> str:
    return subagents.exploration_rules(request)


async def render_reviewer(variant: str) -> str:
    """The reviewer's prompt through `ReviewService.review`, with a fake provider."""
    from agent.routing.review import ReviewService
    calls: list[list[dict]] = []

    class Provider:
        async def tools(self, messages, specs, **kwargs):
            calls.append(list(messages))
            if variant == "reviewer_repair" and len(calls) == 1:
                yield TextEvent("not json")
            else:
                yield TextEvent(json.dumps({
                    "verdict": "done", "rationale": "Inspected the factory",
                    "evidence": ["The factory exists"], "inspected_paths": ["src/factory.py"],
                    "findings": []}))
            yield DoneEvent("endTurn")

    async def read_file(arguments):
        if arguments["file_path"] == "src/factory.py":
            return "class Factory: ..."
        return "Error: file not found"

    queued = task()
    object.__setattr__(queued, "_planner_review_context", "Objective: one factory")
    object.__setattr__(queued, "_planner_relevant_files", ("src/not_created_yet.py",))
    change = SimpleNamespace(title="Factory", work_packages=[])
    specs = [ToolSpec("read_file", "Read", {"type": "object", "properties": {
        "file_path": {"type": "string"}}})]
    if variant == "reviewer_native_diff":
        from agent.planning.models import EditRange
        object.__setattr__(queued, "_review_journal_manifest", [
            {"path": "src/factory.py", "change": "modified", "after_exists": True},
            {"path": "src/new.py", "change": "created", "after_exists": True}])
        object.__setattr__(queued, "_review_preimages", {
            "src/factory.py": {"before_exists": True, "before_content": _BEFORE},
            "src/new.py": {"before_exists": False, "before_content": None}})
        object.__setattr__(queued, "_planner_edit_ranges", (
            EditRange("src/factory.py", "change line 20", 18, 21),
            EditRange("src/factory.py", "change line 35", 35, 36),
            EditRange("src/registry.py", "register the factory", 12, 14)))
    native = variant.startswith("reviewer_native")
    env = {"CODEPARTNER_EXECUTION_MODE": "native" if native else "ide",
           "CODEPARTNER_READER_SUBAGENT": "true" if variant == "reviewer_native_reader"
           else "false"}
    with tempfile.TemporaryDirectory() as root, \
            mock.patch.dict(os.environ, env), \
            mock.patch("agent.routing.review.providers.get", return_value=Provider()):
        patches = []
        if native:
            Path(root, "src").mkdir()
            Path(root, "src", "factory.py").write_text(
                _AFTER if variant == "reviewer_native_diff" else "class Factory: ...\n")
            if variant == "reviewer_native_diff":
                Path(root, "src", "new.py").write_text("def new():\n    return 1\n")
                Path(root, "src", "registry.py").write_text(_REGISTRY)
            project = SimpleNamespace(workspace=root, project_id="demo", host_path=root)
            plan = SimpleNamespace(mode="native", native=True, project=project, reason="")
            patches = [mock.patch("agent.routing.review.plan_native", return_value=plan),
                       mock.patch("agent.routing.review.read_only_session",
                                  return_value=SimpleNamespace(read_only=True))]
        for patch in patches:
            patch.start()
        try:
            try:
                await ReviewService().review(change, queued, "Changed `src/factory.py`.",
                                             "claude-haiku", specs, {"read_file": read_file})
            except Exception as exc:  # the prompt is what is pinned, not the verdict
                if not calls:
                    raise RuntimeError(f"{variant}: reviewer never called: {exc}") from exc
        finally:
            for patch in patches:
                patch.stop()
        conversation = calls[-1]
        text = json.dumps(conversation, indent=2, ensure_ascii=False)
        return text.replace(root, _ROOT)


async def render_all() -> dict[str, str]:
    rendered = {name: render_planner(case) for name, case in PLANNER.items()}
    rendered["planner_repair"] = render_planner_repair()
    rendered.update({name: render_coder(case) for name, case in CODER.items()})
    rendered["coder_inventory"] = render_inventory()
    rendered.update({name: render_subagent(*args) for name, args in SUBAGENTS.items()})
    rendered.update({name: render_delegation(req) for name, req in DELEGATION.items()})
    for variant in REVIEWER:
        rendered[variant] = await render_reviewer(variant)
    return rendered

"""Interactive WP planning: planner turns, approval freeze, and conversation flow."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.planning import service as planning_service
from agent.planning.models import ApprovedWPPlan, WPPlanDraft, plan_id_for
from agent.prompt import PlannerPromptBuilder
from agent.planning.projection import project, validate_draft
from agent.planning.render import coder_context
from agent.planning.service import PlanningError, WPPlannerService, describe_tool
from agent.routing.config import load_routing_config
from agent.routing.implementation import ImplementationService
from agent.routing.persistence import RoutingPersistence
from agent.routing.task_execution import TaskExecutionService
from agent.spec.models import Capability, Requirement, Scenario, Task, WorkPackage
from agent.task_contracts.contractor import TaskContractor
from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent
from model.chat import ChatMessage
from tests.test_turn_routing import _Artifacts, _Execution, _WorldTest, _specs
from tests import native_mode


def setUpModule():
    native_mode.pin()


def tearDownModule():
    native_mode.unpin()


TASKS = ("WP4-T1", "WP4-T2")


class KbToolFeedbackTests(unittest.TestCase):
    def test_kb_tool_calls_have_readable_progress(self):
        self.assertEqual(describe_tool("kb_search_graph", {"query": "private terms"},
                                       frozenset(), frozenset()),
                         "Searching Codebase Memory graph")
        self.assertEqual(describe_tool("kb_get_file_outline", {
            "file_path": "src/registry.py"}, frozenset(), frozenset()),
            "Reading indexed outline of `src/registry.py`")

    def test_planner_brief_contains_referenced_requirement_details(self):
        package = _package()
        package = replace(package, requirement_refs=["loaders#Create loader"])
        change = _change(package)
        change.capabilities = [Capability(
            path="loaders", purpose="Create loaders by name for callers in the project.",
            requirements=[Requirement(
                name="Create loader", statement="The system SHALL create a loader by name.",
                scenarios=[Scenario(name="Known loader", when="a name is registered",
                                    then="the matching loader is returned")])])]
        brief = json.loads(PlannerPromptBuilder.brief(change, package, "high"))
        self.assertEqual(brief["work_package"]["requirements"][0]["statement"],
                         "The system SHALL create a loader by name.")
        self.assertEqual(brief["work_package"]["requirements"][0]["scenarios"][0]["then"],
                         "the matching loader is returned")


def _task(task_id):
    return Task(id=task_id, title=f"Build {task_id}", description="Implement the feature.",
                complexity="simple", reasoning="Localized.", context="Small change.",
                preferred_capability="write", acceptance_criteria=["It works."])


def _package(tasks=TASKS, complexity="high"):
    return WorkPackage(id="WP4", title="Loader", kind="technical-outcome",
                       objective="Load data through one factory", acceptance_criteria=["Works"],
                       requirement_refs=[], tasks=[_task(task_id) for task_id in tasks],
                       complexity=complexity)


def _change(package):
    return SimpleNamespace(id="CHG1", title="Data", summary="Unify loading",
                           design=SimpleNamespace(context="", decisions=[]),
                           work_packages=[package], epic=None)


def plan_payload(task_ids=TASKS, summary="Add a loader factory.", **overrides):
    payload = {
        "summary": summary,
        "plan_markdown": "1. Add `LoaderFactory`.\n2. Wire it into the trainer.",
        "decisions": ["Reuse the existing registry"],
        "open_questions": [],
        "findings": [
            {"id": "F1", "kind": "existing_impl", "statement": "A registry already exists",
             "evidence": [{"path": "src/registry.py", "symbol": "Registry"}],
             "validated": True, "task_ids": [task_ids[0]]},
            {"id": "F2", "kind": "constraint", "statement": "The trainer API is public",
             "evidence": ["src/trainer.py"], "validated": True, "task_ids": list(task_ids)},
            {"id": "F3", "kind": "risk", "statement": "Config loading is global",
             "evidence": [], "validated": False, "task_ids": []},
        ],
        "shared": {"objective": "One factory for every loader",
                   "relevant_files": ["src/registry.py"], "do_not_modify": ["src/legacy/"],
                   "finding_ids": []},
        "tasks": {task_id: {"objective": f"Do {task_id}", "approach": "Extend the registry",
                            "relevant_files": [f"src/{task_id.lower()}.py"],
                            "symbols": ["Registry.register"], "finding_ids": [],
                            "edits": [{"file": f"src/{task_id.lower()}.py", "start_line": 1,
                                       "end_line": 5, "change": f"Implement {task_id}"}]}
                  for task_id in task_ids},
        "reviewer": {"objective": "Factory delivered",
                     "expected_behavior": ["Loaders are created by name"],
                     "validation_expectations": ["Factory unit tests pass"],
                     "relevant_files": ["src/registry.py"]},
    }
    payload.update(overrides)
    return payload


class PlannerProvider:
    """Stand-in planner model: optionally reads a file, then replies with plan JSON."""

    def __init__(self, *replies, read=None, notes=None):
        self.replies = list(replies)
        self.calls = []
        self.read = read
        self.notes = notes or {}

    def tools(self, conversation, specs, **kwargs):
        self.calls.append({"conversation": copy.deepcopy(conversation), **kwargs})
        reply = self.replies.pop(0)

        async def events():
            handlers = kwargs.get("direct_tool_handlers") or {}
            for path in self.read or ():
                if path in self.notes:
                    yield TextEvent(self.notes[path])
                await handlers["read_file"]({"file_path": path})
            yield TextEvent(reply if isinstance(reply, str) else json.dumps(reply))
            yield DoneEvent("endTurn", usage={"input_tokens": 1000, "output_tokens": 200})
        return events()


class PlanArtifacts(_Artifacts):
    def __init__(self, package=None):
        super().__init__()
        self.package = package or _package()
        self.change = _change(self.package)
        self.tasks.update({task.id: task for task in self.package.tasks})

    async def object_by_id(self, object_id):
        if object_id.upper() == self.package.id:
            return "work_package", self.change, self.package
        raise AssertionError(f"unexpected object {object_id}")

    async def tasks_for_selectors(self, selectors, include_completed=False):
        if list(selectors) == [self.package.id]:
            return [(self.change, task) for task in self.package.tasks]
        return await super().tasks_for_selectors(selectors, include_completed)


async def _read(arguments):
    return f"contents of {arguments['file_path']}"


class PlannerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.persistence = RoutingPersistence(Path(directory.name) / "routing.sqlite3")
        self.artifacts = PlanArtifacts()
        self.config = load_routing_config()

    def planner(self, provider):
        return WPPlannerService(self.artifacts, self.persistence, config=self.config,
                                provider_lookup=lambda _name: provider)

    def planner_by_provider(self, responses):
        calls = []

        class Provider:
            def __init__(self, provider):
                self.provider = provider

            async def tools(self, conversation, specs, **kwargs):
                calls.append((self.provider, copy.deepcopy(conversation), kwargs))
                events = responses[self.provider].pop(0)
                for event in events:
                    yield event

        planner = WPPlannerService(
            self.artifacts, self.persistence, config=self.config,
            provider_lookup=lambda provider: Provider(provider))
        return planner, calls

    async def test_session_quota_skips_direct_and_switches_to_codex(self):
        quota = "You've hit your session limit · resets 11pm (UTC) (API status 429)"
        planner, calls = self.planner_by_provider({
            "claude": [[TextEvent(quota), ErrorEvent(quota),
                        DoneEvent("endTurn", usage={"total_cost_usd": 0.01})]],
            "codex": [[TextEvent(json.dumps(plan_payload())),
                       DoneEvent("endTurn", usage={"total_cost_usd": 0.02})],
                      [TextEvent(json.dumps(plan_payload(summary="Revised"))),
                       DoneEvent("endTurn")]],
        })
        progress = []

        async def capture(line):
            progress.append(line)

        turn = await planner.start("conv", "WP4", "go", _specs(), {}, progress=capture)
        self.assertEqual(turn.planner_model, "codex-sol-high")
        self.assertEqual([name for name, _, _ in calls], ["claude", "codex"])
        self.assertIn("switching to", " ".join(progress))
        self.assertFalse(any("session limit" in line for line in progress))
        session = await self.persistence.planning_session(turn.session_id)
        self.assertEqual(session["planner_model"], "codex-sol-high")
        self.assertEqual(session["planner_effort"], "high")
        attempts = await self.persistence.planning_turns(turn.session_id)
        self.assertEqual([row["model"] for row in attempts],
                         ["claude-opus", "codex-sol-high"])
        self.assertEqual([row["cost_usd"] for row in attempts], [0.01, 0.02])
        self.assertEqual(attempts[0]["tool_metrics"]["failure_kind"], "quota")
        revised = await planner.discuss("conv", "revise", _specs(), {})
        self.assertEqual(revised.planner_model, "codex-sol-high")
        self.assertEqual([name for name, _, _ in calls], ["claude", "codex", "codex"])
        approved = await planner.approve(revised.session_id, revised.revision)
        self.assertEqual(approved.planner_model, "codex-sol-high")
        self.assertEqual(approved.provenance["planner_effort"], "high")

    async def test_gateway_transport_then_direct_quota_switches_provider(self):
        planner, calls = self.planner_by_provider({
            "claude": [[ErrorEvent("HTTP 502 bad gateway"), DoneEvent("endTurn")],
                       [ErrorEvent("You've hit your session limit (API status 429)"),
                        DoneEvent("endTurn")]],
            "codex": [[TextEvent(json.dumps(plan_payload())), DoneEvent("endTurn")]],
        })
        turn = await planner.start("conv", "WP4", "go", _specs(), {})
        self.assertEqual(turn.planner_model, "codex-sol-high")
        self.assertEqual([name for name, _, _ in calls], ["claude", "claude", "codex"])
        self.assertNotIn("ANTHROPIC_BASE_URL", calls[1][2].get("execution_env", {}))

    async def test_gateway_401_retries_codex_direct_after_claude_quota(self):
        # Exercise the opt-in gateway mode even though the bundled
        # subscription-only Codex aliases are now direct-only.
        self.config = replace(self.config, models={**self.config.models,
            "codex-sol-high": replace(self.config.models["codex-sol-high"],
                                       gateway_mode="preferred")})
        gateway_error = ("Reconnecting... 1/5 (unexpected status 401 Unauthorized: "
                         "Missing scopes: api.responses.write, "
                         "url: http://agentgateway:4000/v1/responses, "
                         "request id: req_private)")
        planner, calls = self.planner_by_provider({
            "claude": [[ErrorEvent("You've hit your session limit (API status 429)"),
                        DoneEvent("endTurn", usage={"total_cost_usd": 0.01})]],
            "codex": [[ErrorEvent(gateway_error),
                       DoneEvent("endTurn", usage={"total_cost_usd": 0.02})],
                      [TextEvent(json.dumps(plan_payload())),
                       DoneEvent("endTurn", usage={"total_cost_usd": 0.03})]],
        })
        progress = []

        async def capture(line):
            progress.append(line)

        turn = await planner.start("conv", "WP4", "go", _specs(), {}, progress=capture)
        self.assertEqual(turn.planner_model, "codex-sol-high")
        self.assertEqual([name for name, _, _ in calls], ["claude", "codex", "codex"])
        self.assertEqual(calls[1][2]["gateway_url"], self.config.agentgateway_url)
        self.assertNotIn("gateway_url", calls[2][2])
        self.assertEqual(calls[2][2]["reasoning_effort"], "high")
        self.assertIn("gateway authorization failed; retrying directly", " ".join(progress))
        self.assertNotIn("req_private", " ".join(progress))
        attempts = await self.persistence.planning_turns(turn.session_id)
        self.assertEqual([row["tool_metrics"]["gateway_used"] for row in attempts],
                         [True, True, False])
        self.assertEqual(attempts[1]["tool_metrics"]["failure_kind"], "gateway_auth")
        self.assertEqual(attempts[1]["error"], gateway_error)
        self.assertEqual([row["cost_usd"] for row in attempts], [0.01, 0.02, 0.03])
        approved = await planner.approve(turn.session_id, turn.revision)
        self.assertEqual(approved.planner_model, "codex-sol-high")

    async def test_gateway_and_direct_401_keep_diagnostics_but_shorten_error(self):
        self.config = replace(self.config, models={**self.config.models,
            "codex-sol-high": replace(self.config.models["codex-sol-high"],
                                       gateway_mode="preferred")})
        gateway_error = ("unexpected status 401 Unauthorized: "
                         "Missing scopes: api.responses.write, request id: req_private")
        planner, calls = self.planner_by_provider({
            "claude": [[ErrorEvent("session limit"), DoneEvent("endTurn")]],
            "codex": [[ErrorEvent(gateway_error), DoneEvent("endTurn")],
                      [ErrorEvent("HTTP 401 Unauthorized: direct credential invalid"),
                       DoneEvent("endTurn")]],
        })
        with self.assertRaises(PlanningError) as caught:
            await planner.start("conv", "WP4", "go", _specs(), {})
        self.assertEqual([name for name, _, _ in calls], ["claude", "codex", "codex"])
        self.assertIn("gateway authorization denied", str(caught.exception))
        self.assertIn("provider unavailable or authentication denied", str(caught.exception))
        self.assertNotIn("req_private", str(caught.exception))
        self.assertNotIn("api.responses.write", str(caught.exception))
        session = await self.persistence.active_planning("conv")
        self.assertIsNone(session["draft"])
        attempts = await self.persistence.planning_turns(session["session_id"])
        self.assertEqual(attempts[1]["error"], gateway_error)
        self.assertEqual(attempts[2]["tool_metrics"]["failure_kind"], "unavailable")

    async def test_required_gateway_401_never_retries_direct(self):
        models = {**self.config.models, "codex-sol-high": replace(
            self.config.models["codex-sol-high"], gateway_mode="required")}
        self.config = replace(self.config, models=models)
        planner, calls = self.planner_by_provider({
            "claude": [[ErrorEvent("session limit"), DoneEvent("endTurn")]],
            "codex": [[ErrorEvent("HTTP 401 Unauthorized"), DoneEvent("endTurn")]],
        })
        with self.assertRaisesRegex(PlanningError, "all planner candidates failed"):
            await planner.start("conv", "WP4", "go", _specs(), {})
        self.assertEqual([name for name, _, _ in calls], ["claude", "codex"])

    async def test_direct_auth_failure_skips_other_aliases_on_same_account(self):
        models = {**self.config.models, "codex-sol-high": replace(
            self.config.models["codex-sol-high"], gateway_mode="disabled")}
        tiers = {**self.config.planner.tiers,
                 "high": ("codex-sol-high", "codex-sol-medium", "claude-opus")}
        self.config = replace(self.config, models=models,
                              planner=replace(self.config.planner, tiers=tiers))
        planner, calls = self.planner_by_provider({
            "codex": [[ErrorEvent("HTTP 401 Unauthorized"), DoneEvent("endTurn")]],
            "claude": [[TextEvent(json.dumps(plan_payload())), DoneEvent("endTurn")]],
        })
        turn = await planner.start("conv", "WP4", "go", _specs(), {})
        self.assertEqual([name for name, _, _ in calls], ["codex", "claude"])
        self.assertEqual(turn.planner_model, "claude-opus")

    async def test_cli_preflight_quota_text_switches_without_narrating_error(self):
        planner, calls = self.planner_by_provider({
            "claude": [[TextEvent("You've hit your session limit · resets 11pm (UTC)"),
                        DoneEvent("endTurn")]],
            "codex": [[TextEvent(json.dumps(plan_payload())), DoneEvent("endTurn")]],
        })
        progress = []

        async def capture(line):
            progress.append(line)

        turn = await planner.start("conv", "WP4", "go", _specs(), {}, progress=capture)
        self.assertEqual(turn.planner_model, "codex-sol-high")
        self.assertEqual([name for name, _, _ in calls], ["claude", "codex"])
        self.assertFalse(any("session limit" in line for line in progress))

    async def test_both_claude_routes_fail_then_codex_repairs(self):
        planner, calls = self.planner_by_provider({
            "claude": [[ErrorEvent("HTTP 502 bad gateway"), DoneEvent("endTurn")],
                       [ErrorEvent("connection refused"), DoneEvent("endTurn")]],
            "codex": [[TextEvent("No JSON yet"), DoneEvent("endTurn")],
                      [TextEvent(json.dumps(plan_payload())), DoneEvent("endTurn")]],
        })
        turn = await planner.start("conv", "WP4", "go", _specs(), {})
        self.assertEqual(turn.planner_model, "codex-sol-high")
        self.assertEqual([name for name, _, _ in calls],
                         ["claude", "claude", "codex", "codex"])
        self.assertEqual(calls[3][2]["reasoning_effort"], "high")
        self.assertEqual([row["kind"] for row in
                          await self.persistence.planning_turns(turn.session_id)],
                         ["plan", "plan", "plan", "repair"])

    async def test_nonoperational_error_does_not_switch_model(self):
        planner, calls = self.planner_by_provider({
            "claude": [[ErrorEvent("invalid plan request"), DoneEvent("endTurn")]],
            "codex": [],
        })
        with self.assertRaisesRegex(PlanningError, "invalid plan request"):
            await planner.start("conv", "WP4", "go", _specs(), {})
        self.assertEqual([name for name, _, _ in calls], ["claude"])

    async def test_all_planner_candidates_fail_without_approval(self):
        planner, calls = self.planner_by_provider({
            "claude": [[ErrorEvent("session limit"), DoneEvent("endTurn")]],
            "codex": [[ErrorEvent("authentication failed"), DoneEvent("endTurn")]],
        })
        with self.assertRaisesRegex(PlanningError, "all planner candidates failed"):
            await planner.start("conv", "WP4", "go", _specs(), {})
        self.assertEqual([name for name, _, _ in calls], ["claude", "codex"])
        session = await self.persistence.active_planning("conv")
        self.assertIsNone(session["draft"])
        self.assertEqual(session["planner_model"], "claude-opus")

    async def test_preferred_planner_gateway_failure_retries_direct_and_records_both(self):
        class Provider:
            def __init__(self):
                self.calls = []

            async def tools(self, conversation, specs, **kwargs):
                self.calls.append(kwargs)
                if "ANTHROPIC_BASE_URL" in kwargs.get("execution_env", {}):
                    yield ErrorEvent("HTTP 502 bad gateway")
                    yield DoneEvent("endTurn", usage={"input_tokens": 10,
                                                       "total_cost_usd": 0.01})
                else:
                    yield TextEvent(json.dumps(plan_payload()))
                    yield DoneEvent("endTurn", usage={"input_tokens": 20,
                                                       "total_cost_usd": 0.02})

        provider = Provider()
        turn = await self.planner(provider).start("conv", "WP4", "go", _specs(), {})
        self.assertEqual(turn.planner_model, "claude-opus")
        self.assertEqual(provider.calls[0]["execution_env"],
                         {"ANTHROPIC_BASE_URL": self.config.agentgateway_url,
                          "FORCE_PROMPT_CACHING_5M": "1"})
        self.assertEqual(provider.calls[1]["execution_env"], {"FORCE_PROMPT_CACHING_5M": "1"})
        attempts = await self.persistence.planning_turns(turn.session_id)
        self.assertEqual([row["tool_metrics"]["gateway_used"] for row in attempts],
                         [True, False])
        self.assertEqual([row["cost_usd"] for row in attempts], [0.01, 0.02])

    async def test_codex_planner_uses_direct_auth_and_keeps_reasoning_effort(self):
        self.config = replace(self.config, planner=replace(
            self.config.planner, tiers={**self.config.planner.tiers,
                                        "high": ("codex-sol-high", "claude-opus")}))
        provider = PlannerProvider(plan_payload())
        turn = await self.planner(provider).start("conv", "WP4", "go", _specs(), {})
        self.assertEqual(turn.planner_model, "codex-sol-high")
        self.assertNotIn("gateway_url", provider.calls[0])
        self.assertEqual(provider.calls[0]["reasoning_effort"], "high")
        self.assertTrue(provider.calls[0]["reviewer_mode"])

    async def test_claude_planner_uses_5m_cache_and_configured_effort(self):
        self.config = replace(self.config, planner=replace(
            self.config.planner, reasoning_effort={"high": "medium"}))
        provider = PlannerProvider(plan_payload())
        await self.planner(provider).start("conv", "WP4", "go", _specs(), {})
        self.assertEqual(provider.calls[0]["execution_env"]["FORCE_PROMPT_CACHING_5M"], "1")
        self.assertEqual(provider.calls[0]["reasoning_effort"], "medium")

    async def test_claude_planner_keeps_cli_cache_ttl_and_effort_when_configured(self):
        self.config = replace(self.config, planner=replace(
            self.config.planner, reasoning_effort={}, prompt_cache_ttl="1h"))
        provider = PlannerProvider(plan_payload())
        await self.planner(provider).start("conv", "WP4", "go", _specs(), {})
        self.assertNotIn("FORCE_PROMPT_CACHING_5M", provider.calls[0].get("execution_env", {}))
        self.assertNotIn("reasoning_effort", provider.calls[0])

    def test_plan_without_markdown_is_rendered_from_structure(self):
        payload = plan_payload()
        del payload["plan_markdown"]
        reply = "Done.\n```json\n" + json.dumps(payload, indent=2) + "\n```"
        for text in (reply, json.dumps(payload, separators=(",", ":"))):
            draft = WPPlanDraft.from_dict(planning_service._plan_json(text), "WP4", 1)
            self.assertIn("**Objective:** One factory for every loader", draft.plan_markdown)
            self.assertIn("- **WP4-T1** — Do WP4-T1", draft.plan_markdown)
            self.assertIn("  - Approach: Extend the registry", draft.plan_markdown)
            self.assertEqual(validate_draft(draft, list(TASKS)), [])
        stored = WPPlanDraft.from_dict(plan_payload(), "WP4", 1)
        self.assertTrue(stored.plan_markdown.startswith("1. Add `LoaderFactory`."))

    def test_prompt_lists_project_files_and_asks_for_compact_json(self):
        from agent.prompt.planner import SYSTEM
        from tests.test_subagents import turn_prompt
        prompt = turn_prompt("brief", [], None, "go", native=True,
                             file_inventory="src/: a.py b.py")
        self.assertIn("Project files (current; one directory per line):\nsrc/: a.py b.py",
                      prompt)
        self.assertNotIn("Project files", turn_prompt("brief", [], None, "go"))
        self.assertIn("compact JSON object", SYSTEM)
        self.assertIn("no-op tool calls", SYSTEM)
        self.assertNotIn("plan_markdown", SYSTEM)

    async def test_planner_required_and_disabled_modes_do_not_retry(self):
        for mode in ("required", "disabled"):
            with self.subTest(mode=mode):
                models = {**self.config.models, "claude-opus": replace(
                    self.config.models["claude-opus"], gateway_mode=mode)}
                self.config = replace(self.config, models=models)
                provider = PlannerProvider(plan_payload())
                await self.planner(provider).start(f"conv-{mode}", "WP4", "go", _specs(), {})
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(
                    "ANTHROPIC_BASE_URL" in provider.calls[0].get("execution_env", {}),
                    mode == "required")

    async def test_discuss_revises_the_same_session_with_current_state(self):
        provider = PlannerProvider(plan_payload(), plan_payload(summary="Smaller scope."))
        planner = self.planner(provider)
        first = await planner.start("conv", "WP4", "Let's do WP4", _specs(), {})
        second = await planner.discuss("conv", "Skip the trainer wiring", _specs(), {})
        self.assertEqual(second.session_id, first.session_id)
        self.assertEqual(second.revision, 2)
        prompt = provider.calls[1]["conversation"][0]["content"]
        self.assertIn("Current plan state to revise in place", prompt)
        self.assertIn("Skip the trainer wiring", prompt)
        self.assertIn("Extend the registry", prompt)

    async def test_approve_freezes_without_calling_the_model(self):
        provider = PlannerProvider(plan_payload())
        planner = self.planner(provider)
        turn = await planner.start("conv", "WP4", "Let's do WP4", _specs(), {})
        plan = await planner.approve(turn.session_id, turn.revision)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(plan.plan_id, plan_id_for(turn.session_id, 1))
        self.assertEqual([item.id for item in plan.tasks["WP4-T1"].findings], ["F1", "F2"])
        self.assertEqual([item.id for item in plan.tasks["WP4-T2"].findings], ["F2"])
        self.assertEqual([item.id for item in plan.shared.findings], ["F2", "F3"])
        self.assertIn("LoaderFactory", plan.reviewer.approved_plan)
        self.assertEqual({item.id for item in plan.reviewer.findings}, {"F2", "F3"})
        stored = await self.persistence.approved_plan(plan.plan_id)
        self.assertEqual(ApprovedWPPlan.from_dict(stored), plan)
        self.assertIsNone(await self.persistence.active_planning("conv"))

    async def test_incomplete_state_is_repaired_once_then_withheld(self):
        incomplete = plan_payload(task_ids=("WP4-T1",))
        provider = PlannerProvider(incomplete, incomplete)
        planner = self.planner(provider)
        turn = await planner.start("conv", "WP4", "Let's do WP4", _specs(), {})
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("tasks.WP4-T2 is missing", provider.calls[1]["conversation"][-1]["content"])
        self.assertIn("tasks.WP4-T2 is missing", turn.gaps)
        with self.assertRaisesRegex(PlanningError, "incomplete"):
            await planner.approve(turn.session_id, turn.revision)

    async def test_repair_recovers_a_reply_without_json(self):
        provider = PlannerProvider("I looked around; here is my plan.", plan_payload())
        turn = await self.planner(provider).start("conv", "WP4", "go", _specs(), {})
        self.assertEqual(turn.gaps, [])

    async def test_stale_revision_and_changed_wp_block_approval(self):
        provider = PlannerProvider(plan_payload(), plan_payload())
        planner = self.planner(provider)
        await planner.start("conv", "WP4", "go", _specs(), {})
        turn = await planner.discuss("conv", "tweak", _specs(), {})
        with self.assertRaisesRegex(PlanningError, "revised"):
            await planner.approve(turn.session_id, 1)
        self.artifacts.package = replace(self.artifacts.package, objective="Something else")
        with self.assertRaisesRegex(PlanningError, "changed in OpenSpec"):
            await planner.approve(turn.session_id, 2)

    async def test_kb_lookups_are_free_and_file_reads_have_their_own_budget(self):
        from agent.routing.tool_metrics import (BUDGET_EXHAUSTED, READ_BUDGET_EXHAUSTED,
                                                instrument_handlers)

        async def handler(_arguments):
            return "ok"

        seen = []

        async def on_call(name, _arguments, refused):
            seen.append((name, refused))

        wrapped, metrics = instrument_handlers(
            {"read_file": handler, "search_text": handler, "kb_get_file_outline": handler},
            frozenset({"read_file"}), frozenset({"search_text"}), max_calls=3,
            on_call=on_call, max_reads=1)
        self.assertEqual(await wrapped["read_file"]({"file_path": "a.py"}), "ok")
        self.assertEqual(await wrapped["read_file"]({"file_path": "b.py"}),
                         READ_BUDGET_EXHAUSTED)
        for _ in range(4):
            self.assertEqual(await wrapped["kb_get_file_outline"]({"file_path": "c.py"}), "ok")
        self.assertEqual(await wrapped["search_text"]({"q": "x"}), "ok")
        self.assertEqual(await wrapped["search_text"]({"q": "y"}), "ok")
        self.assertEqual(await wrapped["search_text"]({"q": "z"}), BUDGET_EXHAUSTED)
        summary = metrics.summary()
        self.assertEqual((summary["tool_calls"], summary["kb_calls"], summary["file_reads"],
                          summary["refused_reads"], summary["refused_over_budget"],
                          summary["other_calls"]), (3, 4, 1, 1, 1, 0))
        self.assertEqual(summary["kb_outline_calls"], 4)
        self.assertEqual(summary["whole_file_reads"], 1)
        self.assertEqual(summary["ranged_source_reads"], 0)
        self.assertEqual(summary["source_response_bytes"], 2)
        self.assertEqual([refused for _, refused in seen if refused], ["reads", "calls"])

    async def test_new_session_supersedes_and_abandon_closes(self):
        provider = PlannerProvider(plan_payload(), plan_payload())
        planner = self.planner(provider)
        first = await planner.start("conv", "WP4", "go", _specs(), {})
        await planner.start("conv", "WP4", "again", _specs(), {})
        self.assertEqual((await self.persistence.planning_session(first.session_id))["status"],
                         "superseded")
        closed = await planner.abandon("conv")
        self.assertEqual(closed["wp_id"], "WP4")
        self.assertIsNone(await self.persistence.active_planning("conv"))


class ArtifactTests(unittest.TestCase):
    def test_gateway_auth_classification_respects_route_and_cli_login(self):
        self.assertEqual(planning_service._failure_kind("status 403 Forbidden", True),
                         "gateway_auth")
        self.assertEqual(planning_service._failure_kind("status 403 Forbidden", False),
                         "unavailable")
        self.assertEqual(planning_service._failure_kind(
            "Please log in (status 401)", True), "unavailable")

    def draft(self, **overrides):
        return WPPlanDraft.from_dict(plan_payload(**overrides), "WP4", 1)

    def test_planner_prompt_keeps_discovery_tasks_read_only(self):
        from agent.prompt.planner import SYSTEM
        self.assertIn("inspection", SYSTEM)
        self.assertIn("Never plan file writes for them", SYSTEM)

    def test_planner_prompt_is_kb_first_in_both_modes(self):
        from agent.prompt.coder import CoderPromptBuilder
        KB_FIRST_RULE = CoderPromptBuilder().read_pipeline()
        KB_SELECTIVE_GUIDANCE = KB_FIRST_RULE
        from agent.prompt.planner import SYSTEM
        NATIVE_SYSTEM = SYSTEM  # one planner system prompt for both execution modes
        from agent.task_contracts.policy import DISCOVERY_CAPABILITIES
        for prompt in (SYSTEM, NATIVE_SYSTEM):
            self.assertTrue(prompt.startswith("You are the Work Package planner."))
            self.assertIn("**KB-first, bulk, source-second**", prompt)
            self.assertEqual(prompt.count("## Read pipeline"), 1)
            self.assertIn("## Investigation Stopping Condition", prompt)
            self.assertTrue(all(name in prompt for name in DISCOVERY_CAPABILITIES))
            self.assertIn('JSON shape:\n{\n  "summary"', prompt)
            self.assertNotIn(KB_SELECTIVE_GUIDANCE, prompt)
        self.assertNotIn(KB_FIRST_RULE, SYSTEM)
        self.assertNotIn(KB_FIRST_RULE, NATIVE_SYSTEM)
        self.assertNotIn("approved plan context", NATIVE_SYSTEM)

    def test_validation_reports_dangling_ids_and_unknown_tasks(self):
        payload = plan_payload()
        payload["tasks"]["WP4-T1"]["finding_ids"] = ["F9"]
        payload["tasks"]["WP9-T1"] = {"objective": "x", "approach": "y"}
        gaps = validate_draft(WPPlanDraft.from_dict(payload, "WP4", 1), list(TASKS))
        self.assertIn("finding_ids reference unknown findings: F9", gaps)
        self.assertIn("tasks names unknown task ids: WP9-T1", gaps)

    def test_draft_and_plan_round_trip(self):
        draft = self.draft()
        self.assertEqual(WPPlanDraft.from_dict(draft.to_dict()), draft)
        plan = project(draft, plan_id="p", session_id="s", change_id="CHG1",
                       wp_fingerprint="f", wp_complexity="high", planner_model="claude-opus")
        self.assertEqual(ApprovedWPPlan.from_dict(json.loads(json.dumps(plan.to_dict()))), plan)
        self.assertEqual(len(plan.fingerprint), 64)

    def test_coder_prompt_and_contract_use_the_task_context(self):
        plan = project(self.draft(), plan_id="p", session_id="s", change_id="CHG1",
                       wp_fingerprint="f", wp_complexity="high", planner_model="claude-opus")
        context = coder_context(plan, "WP4-T1")
        self.assertIn("RELEVANT CONTEXT (WP4-T1;", context)
        self.assertIn("Approach: Extend the registry", context)
        self.assertIn("EDIT (your files; planned lines widened by 10):\n"
                      "- src/wp4-t1.py:1-15 (planned lines 1-5) — Implement WP4-T1", context)
        self.assertIn("src/legacy/", context)
        self.assertIn("src/wp4-t1.py", context)
        self.assertNotIn("src/wp4-t2.py", context)
        messages = TaskExecutionService._execution_messages(
            _task("WP4-T1"), [{"role": "user", "content": "x"}], plan_context=context)
        prompt = messages[-1]["content"]
        self.assertIn("A planner already investigated this work package", prompt)
        self.assertIn("RELEVANT CONTEXT (WP4-T1;", prompt)
        self.assertNotIn("WP SHARED CONTEXT", prompt)
        contract = TaskContractor().initial_contract(
            _task("WP4-T1"), _specs(), task_context=plan.tasks["WP4-T1"])
        self.assertIn("src/wp4-t1.py", contract.primary_files)
        self.assertIn("Registry.register", contract.primary_symbols)
        self.assertEqual(contract.excluded_files, ())
        sources = {fact.source for fact in contract.known_context}
        self.assertIn("planner.approach", sources)
        self.assertIn("planner.finding.F1", sources)


class _RecordingExecution(_Execution):
    def __init__(self):
        super().__init__()
        self.kwargs = []

    async def execute(self, task, change_id, messages, specs, **kwargs):
        self.kwargs.append(kwargs)
        async for event in super().execute(task, change_id, messages, specs, **kwargs):
            yield event


class PlanModeConversationTests(_WorldTest):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.world.planning = True
        self.world.artifacts.__class__ = PlanArtifacts
        PlanArtifacts.__init__(self.world.artifacts)
        self.world.execution.__class__ = _RecordingExecution
        self.world.execution.kwargs = []
        self.provider = PlannerProvider(plan_payload(), plan_payload(summary="Revised plan."))
        patch = mock.patch.object(planning_service.providers, "get",
                                  return_value=self.provider)
        patch.start()
        self.addCleanup(patch.stop)

    async def say(self, text, history=()):
        return await self.world.say(text, history=history, model="codepartner-plan")

    async def test_all_candidate_errors_are_concise_in_chat(self):
        gateway_error = ("Reconnecting... 1/5 (unexpected status 401 Unauthorized: "
                         "Missing scopes: api.responses.write, "
                         "url: http://agentgateway:4000/v1/responses, "
                         "request id: req_private)")
        responses = {
            "claude": [[ErrorEvent("You've hit your session limit (API status 429)"),
                        DoneEvent("endTurn")]],
            "codex": [[ErrorEvent(gateway_error), DoneEvent("endTurn")]],
        }

        class Provider:
            def __init__(self, name):
                self.name = name

            async def tools(self, conversation, specs, **kwargs):
                for event in responses[self.name].pop(0):
                    yield event

        with mock.patch.object(planning_service.providers, "get",
                               side_effect=lambda name: Provider(name)):
            answer = await self.say("Let's do WP4")
        self.assertIn("Nothing was executed", answer)
        self.assertIn("provider unavailable or authentication denied", answer)
        self.assertNotIn("Reconnecting", answer)
        self.assertNotIn("api.responses.write", answer)
        self.assertNotIn("req_private", answer)
        self.assertNotIn("1. Reply `approve`", answer)

    async def test_lets_do_wp_plans_and_offers_two_actions_without_coding(self):
        answer = await self.say("Let's do WP4")
        self.assertIn("## Plan for WP4 (revision 1)", answer)
        self.assertIn("1. Reply `approve`", answer)
        self.assertIn("2. Reply `discuss`", answer)
        self.assertEqual(self.world.execution.started, [])
        self.assertEqual(len(self.provider.calls), 1)

    async def test_free_text_discusses_and_approval_starts_the_planned_job(self):
        await self.say("Let's do WP4")
        revised = await self.say("Please keep the trainer untouched")
        self.assertIn("revision 2", revised)
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(self.world.execution.started, [])
        started = await self.say("approve")
        self.assertIn("approved and frozen", started)
        self.assertEqual([task_id for task_id, _ in self.world.execution.started],
                         list(TASKS))
        self.assertIn("RELEVANT CONTEXT (WP4-T1;",
                      self.world.execution.kwargs[0]["plan_context"])
        plan_id = self.world.rows("SELECT plan_id FROM implementation_jobs")[0][0]
        self.assertTrue(plan_id.endswith("_r2"))

    async def test_numbered_choice_two_asks_what_to_change(self):
        plan = await self.say("Let's do WP4")
        history = (ChatMessage(role="user", content="Let's do WP4"),
                   ChatMessage(role="assistant", content=plan))
        answer = await self.say("2", history=history)
        self.assertIn("What would you like to change in the WP4 plan?", answer)
        self.assertEqual(len(self.provider.calls), 1)

    async def test_stale_approval_token_executes_nothing(self):
        first = await self.say("Let's do WP4")
        token = first.rsplit("`confirm ", 1)[1].split("`", 1)[0]
        await self.say("Change the approach")
        answer = await self.say(f"confirm {token}")
        self.assertIn("nothing was executed", answer.lower())
        self.assertEqual(self.world.execution.started, [])

    async def test_bare_yes_approves_the_current_revision(self):
        await self.say("Let's do WP4")
        answer = await self.say("yes")
        self.assertIn("approved and frozen", answer)
        self.assertEqual(len(self.provider.calls), 1)

    async def test_bare_no_rejects_and_closes_planning(self):
        await self.say("Let's do WP4")
        answer = await self.say("no")
        self.assertIn("Planning closed", answer)
        self.assertIsNone(await self.world.persistence.active_planning("conv"))
        self.assertEqual(self.world.execution.started, [])

    async def test_progress_streams_tools_and_reasons_before_the_plan(self):
        self.provider.read = ["src/registry.py", "src/trainer.py"]
        self.provider.notes = {"src/registry.py": "Checking how loaders are registered",
                               "src/trainer.py": "Confirming the trainer API stays public"}
        self.world.sessions["conv"].tools.handlers = {"read_file": _read}
        answer = await self.say("Let's do WP4")
        header = answer.index("**Investigating WP4 (planner: Claude Opus)**\n\n")
        reason = answer.index("\n💭 Checking how loaders are registered\n")
        tool = answer.index("- 🔎 Reading `src/registry.py`\n")
        rule = answer.index("\n---\n\n## Plan for WP4")
        self.assertLess(header, reason)
        self.assertLess(reason, tool)
        self.assertLess(tool, answer.index("💭 Confirming the trainer API"))
        self.assertLess(answer.index("- 🔎 Reading `src/trainer.py`"), rule)
        self.assertNotIn('"plan_markdown"', answer)
        self.assertNotRegex(answer, r"\S💭|`🔎|\)💭")
        self.assertIn("**What I inspected and why**", answer)
        self.assertIn("- `src/registry.py` — A registry already exists", answer)
        prompt = self.provider.calls[0]["conversation"][0]["content"]
        self.assertIn("Progress notes: before each tool call", prompt)

    async def test_cancel_abandons_planning(self):
        await self.say("Let's do WP4")
        answer = await self.say("never mind")
        self.assertIn("Planning for WP4 closed", answer)
        self.assertIsNone(await self.world.persistence.active_planning("conv"))

    async def test_task_selector_requires_parent_wp_plan(self):
        answer = await self.say("implement WP4-T1")
        self.assertIn("Start with `WP4`", answer)
        self.assertEqual(self.provider.calls, [])

    async def test_implement_command_starts_wp_planning(self):
        answer = await self.say("/implement WP4")
        self.assertIn("## Plan for WP4 (revision 1)", answer)
        self.assertEqual(self.world.execution.started, [])

    async def test_plan_approval_reports_an_active_job_conflict(self):
        job = await self.world.job("WP4", "other-chat")
        answer = await self.say("/implement WP4")
        self.assertIn("## Plan for WP4 (revision 1)", answer)
        self.assertIn("WP4 (started from another chat) is recorded as running", answer)
        self.assertEqual(self.world.job_status(job.job_id), "running")
        self.assertEqual(self.world.execution.started, [])
        approved = await self.say("yes")
        self.assertIn("Stopped WP4", approved)
        self.assertEqual(self.world.job_status(job.job_id), "stopped")
        self.assertTrue(self.world.execution.started)

    async def test_run_command_directs_to_parent_wp(self):
        answer = await self.say("/run WP4-T1")
        self.assertIn("Start with `WP4`", answer)
        self.assertEqual(self.world.execution.started, [])

    async def test_every_mode_plans_a_wp_before_coding(self):
        answer = await self.world.say("implement WP4", model="codepartner-implement")
        self.assertIn("## Plan for WP4 (revision 1)", answer)
        self.assertEqual(self.world.execution.started, [])

    async def test_several_wps_are_planned_one_at_a_time(self):
        self.world.artifacts.tasks["WP5-T1"] = _task("WP5-T1")
        wp5 = WorkPackage(id="WP5", title="Next", kind="technical-outcome",
                          objective="Use the loader", acceptance_criteria=["Works"],
                          requirement_refs=[], tasks=[_task("WP5-T1")], complexity="low")
        artifacts = self.world.artifacts
        original_object, original_tasks = artifacts.object_by_id, artifacts.tasks_for_selectors

        async def object_by_id(object_id):
            if object_id == "WP5":
                return "work_package", _change(wp5), wp5
            return await original_object(object_id)

        async def tasks_for_selectors(selectors, include_completed=False):
            if list(selectors) == ["WP5"]:
                return [(artifacts.change, wp5.tasks[0])]
            return await original_tasks(selectors, include_completed)

        artifacts.object_by_id, artifacts.tasks_for_selectors = object_by_id, tasks_for_selectors
        self.provider.replies[1] = plan_payload(task_ids=("WP5-T1",), summary="WP5 plan.")
        first = await self.say("Let's do WP4 and WP5")
        self.assertIn("## Plan for WP4", first)
        self.assertIn("I will plan WP5 next", first)

        persistence = self.world.persistence

        async def approved_review(service, job_id, *args, **kwargs):
            await persistence.finish_implementation_job(job_id, "completed")
            yield TextEvent("WP4 approved by the reviewer.")

        with mock.patch.object(ImplementationService, "_review_and_advance", approved_review):
            started = await self.say("approve")
        self.assertEqual([task_id for task_id, _ in self.world.execution.started], list(TASKS))
        self.assertIn("**WP4 is complete — investigating WP5 (planner: ", started)
        self.assertIn("## Plan for WP5 (revision 1)", started)
        session = await self.world.persistence.active_planning("conv")
        self.assertEqual((session["wp_id"], session["queued_wp_ids"]), ("WP5", []))


if __name__ == "__main__":
    unittest.main()

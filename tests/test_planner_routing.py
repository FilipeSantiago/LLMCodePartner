"""Role-aware planner routing with an optional planner capability floor."""

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from agent.planning.complexity import effective_wp_complexity
from agent.planning.routing import PlannerRoutingService
from agent.routing.config import DEFAULT_CONFIG_PATH, RoutingConfigError, load_routing_config
from agent.routing.gateway import gateway_attempts, gateway_kwargs
from agent.spec.models import Task, WorkPackage

BASE = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())


def _load(mutate=None):
    raw = copy.deepcopy(BASE)
    if mutate:
        mutate(raw)
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "routing.yaml"
    path.write_text(yaml.safe_dump(raw))
    try:
        return load_routing_config(path)
    finally:
        directory.cleanup()


def _package(complexity=None, task_complexities=("simple",)):
    tasks = [Task(id=f"WP1-T{index}", title="t", description="d", complexity=value,
                  reasoning="r", context="", preferred_capability="write")
             for index, value in enumerate(task_complexities, 1)]
    return WorkPackage(id="WP1", title="WP", kind="technical-outcome", objective="o",
                       acceptance_criteria=["a"], requirement_refs=[], tasks=tasks,
                       complexity=complexity)


class PlannerConfigTests(unittest.TestCase):
    def test_default_config_has_independent_planner_tiers(self):
        config = _load()
        self.assertEqual(set(config.planner.tiers), {"low", "medium", "high", "complex"})
        self.assertNotIn("planner", config.role_floors)
        self.assertEqual(config.planner.tiers["low"][0], "claude-haiku")
        self.assertNotEqual(config.planner.tiers["low"], config.tiers["simple"])

    def test_low_planner_can_match_simple_coder_capability(self):
        def low_cost_planner(raw):
            raw["planner"]["tiers"]["low"] = ["claude-haiku"]
        self.assertEqual(_load(low_cost_planner).planner.tiers["low"], ("claude-haiku",))

    def test_planner_floor_remains_optional_and_enforced_when_set(self):
        def below_floor(raw):
            raw["roles"]["planner"] = {"min_capability": 3}
        with self.assertRaisesRegex(RoutingConfigError, "below the planner floor"):
            _load(below_floor)

    def test_reviewer_floor_is_separate_from_coder_floor(self):
        def strict_reviewer(raw):
            raw["roles"]["reviewer"]["min_capability"] = 3
        with self.assertRaisesRegex(RoutingConfigError, "below the reviewer floor"):
            _load(strict_reviewer)

    def test_narration_is_on_by_default_and_validated(self):
        self.assertTrue(_load().planner.narrate)
        self.assertTrue(_load(lambda raw: raw["planner"].pop("narrate")).planner.narrate)
        self.assertFalse(_load(lambda raw: raw["planner"].update(narrate=False)).planner.narrate)
        with self.assertRaises(RoutingConfigError):
            _load(lambda raw: raw["planner"].update(narrate="yes"))

    def test_file_read_budget_defaults_and_is_validated(self):
        self.assertEqual(_load().planner.max_file_reads, 25)
        self.assertEqual(_load(lambda raw: raw["planner"].pop("max_file_reads")
                               ).planner.max_file_reads, 25)
        self.assertEqual(_load(lambda raw: raw["planner"].update(max_file_reads=10)
                               ).planner.max_file_reads, 10)
        for bad in (0, True, "10"):
            with self.assertRaisesRegex(RoutingConfigError, "max_file_reads"):
                _load(lambda raw, bad=bad: raw["planner"].update(max_file_reads=bad))

    def test_discovery_defaults_to_advise_and_is_validated(self):
        discovery = _load().planner.discovery
        self.assertEqual(discovery.mode, "advise")
        self.assertEqual(discovery.small_file_bytes, 6000)
        self.assertEqual(discovery.thresholds["low"]["full_reads"], 2)
        default = _load(lambda raw: raw.pop("discovery")).planner.discovery
        self.assertEqual(default.mode, "advise")
        self.assertEqual(_load(lambda raw: raw["discovery"].update(mode="off")
                               ).planner.discovery.mode, "off")
        merged = _load(lambda raw: raw["discovery"].update(
            thresholds={"high": {"tool_calls": 20}})).planner.discovery.thresholds
        self.assertEqual(merged["high"]["tool_calls"], 20)
        self.assertEqual(merged["high"]["full_reads"], 6)
        self.assertEqual(merged["low"], discovery.thresholds["low"])
        for bad in ({"mode": "loud"}, {"small_file_bytes": 0}, {"small_file_bytes": True},
                    {"thresholds": {"huge": {}}}, {"thresholds": {"low": {"files": 3}}},
                    {"thresholds": {"low": {"tool_calls": -1}}}):
            with self.assertRaisesRegex(RoutingConfigError, "discovery"):
                _load(lambda raw, bad=bad: raw["discovery"].update(bad))

    def test_advise_mode_is_default(self):
        self.assertEqual(_load(lambda raw: raw["discovery"].update(mode="advise")
                               ).planner.discovery.mode, "advise")

    def test_discovery_role_thresholds_and_global_disable(self):
        discovery = _load().discovery
        self.assertTrue(discovery.enabled)
        self.assertEqual(discovery.policy, "kb_first")
        # kb_first (configured): every role is gated.
        self.assertEqual(discovery.mode_for("coder"), "advise")
        self.assertTrue(discovery.gated("reviewer"))
        # Selective: planners keep KB-first gates; coders and reviewers only observe.
        selective = _load(lambda raw: raw["discovery"].update(policy="selective")).discovery
        self.assertEqual(selective.mode_for("planner"), "advise")
        self.assertEqual(selective.mode_for("coder"), "observe")
        self.assertFalse(selective.gated("reviewer"))
        with self.assertRaisesRegex(RoutingConfigError, "policy"):
            _load(lambda raw: raw["discovery"].update(policy="always"))
        self.assertEqual(discovery.thresholds_for("coder", "simple")["discovery_calls"], 18)
        self.assertFalse(_load(lambda raw: raw["discovery"].update(enabled=False)
                               ).discovery.enabled)

    def test_config_without_planner_or_roles_still_loads(self):
        def legacy(raw):
            raw.pop("planner")
            raw.pop("roles")
            for model in raw["models"].values():
                model.pop("capability")
        config = _load(legacy)
        self.assertIsNone(config.planner)
        self.assertEqual(config.role_floors, {})


class PlannerRoutingTests(unittest.TestCase):
    def test_read_only_gateway_policy(self):
        self.assertEqual(gateway_attempts("preferred"), (True, False))
        self.assertEqual(gateway_attempts("required"), (True,))
        self.assertEqual(gateway_attempts("disabled"), (False,))
        self.assertEqual(gateway_kwargs("claude", "http://gateway", True),
                         {"execution_env": {"ANTHROPIC_BASE_URL": "http://gateway"}})
        self.assertEqual(gateway_kwargs("codex", "http://gateway", True),
                         {"gateway_url": "http://gateway"})
        self.assertEqual(gateway_kwargs("codex", "http://gateway", False), {})

    def test_wp_complexity_selects_the_planner_tier(self):
        router = PlannerRoutingService(_load())
        self.assertEqual(router.route("low").model, "claude-haiku")
        decision = router.route("complex")
        self.assertEqual(decision.model, "claude-opus")
        self.assertEqual(decision.reasoning_effort, "xhigh")

    def test_unhealthy_planner_falls_through_to_next_candidate(self):
        router = PlannerRoutingService(_load())
        for _ in range(router.config.failure_threshold):
            router.health.record("claude-haiku", "claude-cli", True, False)
            router.health.record("claude-haiku", "claude-cli", False, False)
        self.assertEqual(router.route("low").model, "codex-sol-medium")

    def test_gateway_failure_does_not_disqualify_healthy_direct_planner(self):
        router = PlannerRoutingService(_load())
        for _ in range(router.config.failure_threshold):
            router.health.record("claude-haiku", "claude-cli", True, False)
        self.assertEqual(router.route("low").model, "claude-haiku")

    def test_planner_routing_requires_a_planner_section(self):
        config = _load(lambda raw: raw.pop("planner"))
        with self.assertRaises(RoutingConfigError):
            PlannerRoutingService(config)


class WorkPackageComplexityTests(unittest.TestCase):
    def test_declared_complexity_wins(self):
        self.assertEqual(effective_wp_complexity(_package("complex")), ("complex", "declared"))

    def test_missing_complexity_is_derived_from_the_hardest_task(self):
        self.assertEqual(effective_wp_complexity(_package(None, ("simple", "complex"))),
                         ("high", "derived"))
        self.assertEqual(effective_wp_complexity(_package(None, ("simple",))),
                         ("low", "derived"))


if __name__ == "__main__":
    unittest.main()

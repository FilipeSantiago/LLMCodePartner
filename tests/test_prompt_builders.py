"""The prompt builders: one class per role, created by the factory, isolated from each other."""

import unittest

from agent.prompt import (ChecksPromptBuilder, ChecksPromptContext, CoderPromptBuilder,
                          CoderPromptContext, PlannerPromptBuilder, PlannerPromptContext,
                          PromptBuilder, PromptFactory, PromptRole, ReaderPromptBuilder,
                          ReaderPromptContext, ReviewerPromptBuilder, ReviewerPromptContext)
from agent.prompt.base import ParentPromptBuilder
from providers import subagents
from tests.prompt_scenarios import task


def _contexts():
    return {
        PromptRole.PLANNER: PlannerPromptContext("BRIEF", [], None, "go"),
        PromptRole.CODER: CoderPromptContext(task()),
        PromptRole.REVIEWER: ReviewerPromptContext(task(), "Change"),
        PromptRole.READER: ReaderPromptContext("task"),
        PromptRole.CHECKS: ChecksPromptContext(),
    }


def _build(role, builder=None) -> str:
    builder = builder or PromptFactory.create(role)
    return builder.with_context(_contexts()[role]).build().text


class FactoryTests(unittest.TestCase):
    def test_one_builder_class_per_role(self):
        expected = {PromptRole.PLANNER: PlannerPromptBuilder,
                    PromptRole.CODER: CoderPromptBuilder,
                    PromptRole.REVIEWER: ReviewerPromptBuilder,
                    PromptRole.READER: ReaderPromptBuilder,
                    PromptRole.CHECKS: ChecksPromptBuilder}
        for role, cls in expected.items():
            builder = PromptFactory.create(role)
            self.assertIs(type(builder), cls)
            self.assertIsInstance(builder, PromptBuilder)
            self.assertEqual(builder.role, role)
        self.assertIs(type(PromptFactory.create("reader")), ReaderPromptBuilder)

    def test_every_create_returns_a_new_builder(self):
        self.assertIsNot(PromptFactory.create(PromptRole.CODER),
                         PromptFactory.create(PromptRole.CODER))

    def test_unknown_role_or_purpose_raises(self):
        with self.assertRaisesRegex(ValueError, "no prompt builder registered"):
            PromptFactory.create("optimizer")
        with self.assertRaisesRegex(ValueError, "no prompt builder for subagent purpose"):
            PromptFactory.for_purpose("optimizer")

    def test_subagent_purpose_maps_to_the_parent_role(self):
        self.assertIs(type(PromptFactory.for_purpose("planner")), PlannerPromptBuilder)
        self.assertIs(type(PromptFactory.for_purpose("task")), CoderPromptBuilder)
        self.assertIs(type(PromptFactory.for_purpose("reviewer")), ReviewerPromptBuilder)


class InterfaceTests(unittest.TestCase):
    def test_build_needs_a_context(self):
        for role in PromptRole:
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "with_context"):
                PromptFactory.create(role).build()

    def test_every_role_builds_a_prompt(self):
        for role in PromptRole:
            with self.subTest(role=role):
                prompt = PromptFactory.create(role).with_context(_contexts()[role]).build()
                self.assertTrue(prompt.text)
                self.assertEqual(prompt.user_message(), {"role": "user", "content": prompt.text})

    def test_repair_only_where_a_role_has_one(self):
        self.assertIn("- gap", PromptFactory.create(PromptRole.PLANNER).repair(["gap"]))
        self.assertIn("bad JSON", PromptFactory.create(PromptRole.REVIEWER).repair("bad JSON"))
        for role in (PromptRole.CODER, PromptRole.READER, PromptRole.CHECKS):
            with self.subTest(role=role), self.assertRaises(NotImplementedError):
                PromptFactory.create(role).repair("gap")


class IsolationTests(unittest.TestCase):
    def test_planner_names_only_available_bulk_graph_tool(self):
        with_query = PlannerPromptBuilder().with_context(PlannerPromptContext(
            "BRIEF", [], None, "go", kb_tools=frozenset({"kb_query_graph"}))).build().text
        without_query = PlannerPromptBuilder().with_context(PlannerPromptContext(
            "BRIEF", [], None, "go", kb_tools=frozenset({"kb_search_graph"}))).build().text
        self.assertIn("one `kb_query_graph`", with_query)
        self.assertIn("`file_pattern` alternation", without_query)
        self.assertNotIn("kb_query_graph", without_query)

    def test_workspace_is_stated_for_native_roles(self):
        coder = CoderPromptBuilder().with_workspace(
            [{"role": "user", "content": "task"}], "/tmp/project", "a.py")
        self.assertIn("Working directory: /tmp/project", coder[0]["content"])
        reviewer = ReviewerPromptBuilder().with_context(ReviewerPromptContext(
            task(), "Change", native=True, workspace="/tmp/project")).build().text
        self.assertIn("Working directory: /tmp/project", reviewer)

    def test_overriding_a_shared_fragment_changes_only_that_role(self):
        before = {role: _build(role) for role in PromptRole}

        class CustomCoder(CoderPromptBuilder):
            def read_pipeline(self) -> str:
                return "CUSTOM READ PIPELINE"

        custom = _build(PromptRole.CODER, CustomCoder())
        self.assertIn("CUSTOM READ PIPELINE", custom)
        self.assertNotIn(CoderPromptBuilder().read_pipeline(), custom)
        self.assertIn(CoderPromptBuilder().read_pipeline(), before[PromptRole.CODER])
        self.assertIn(ReviewerPromptBuilder().read_pipeline(), before[PromptRole.REVIEWER])
        self.assertEqual({role: _build(role) for role in PromptRole}, before)

    def test_coder_and_reviewer_share_scope_rules_and_the_planner_explores(self):
        request = subagents.Request(roles=(subagents.READER,), purpose="task",
                                    scope=("src/a.py",))
        planner = PromptFactory.create(PromptRole.PLANNER).delegation_rules(request)
        coder = PromptFactory.create(PromptRole.CODER).delegation_rules(request)
        reviewer = PromptFactory.create(PromptRole.REVIEWER).delegation_rules(request)
        self.assertIn("as widely as the Work Package needs", planner)
        self.assertNotIn("src/a.py", planner)
        self.assertNotEqual(coder, reviewer)
        self.assertIn("- src/a.py", coder)
        self.assertIn("SCOPE EXCEPTION:", coder)
        for role in (PromptRole.PLANNER, PromptRole.CODER, PromptRole.REVIEWER):
            self.assertEqual(PromptFactory.create(role).delegation_rules(None), "")

    def test_every_parent_role_owns_its_bulk_and_read_rules(self):
        roles = (PromptRole.PLANNER, PromptRole.CODER, PromptRole.REVIEWER)
        own = {PromptFactory.create(role).own_reads_rule() for role in roles}
        self.assertEqual(len(own), 3)
        self.assertEqual(len({PromptFactory.create(role).bulk_rule() for role in roles}), 3)
        for role in roles:
            self.assertIn(PromptFactory.create(role).bulk_rule(), _build(role))

    def test_every_parent_rule_is_abstract_in_the_base(self):
        expected = {"bulk_rule", "read_pipeline", "own_reads_rule", "delegation_rules",
                    "scope_exception_rule", "scope_block", "checks_rule"}
        self.assertEqual(ParentPromptBuilder.__abstractmethods__, expected | {"sections"})

    def test_parent_builders_must_say_what_they_read(self):
        class Incomplete(ParentPromptBuilder):
            role = PromptRole.CODER

            def sections(self, context):
                return []

        with self.assertRaises(TypeError):
            Incomplete()
        self.assertNotIsInstance(PromptFactory.create(PromptRole.READER), ParentPromptBuilder)


if __name__ == "__main__":
    unittest.main()

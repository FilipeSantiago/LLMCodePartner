"""Planned ranges follow files that earlier tasks changed; new packages get their marker."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.planning.models import EditRange, ReferenceRange, TaskContext
from agent.planning.rebase import rebase_task, with_package_markers
from agent.routing.implementation import ImplementationService
from agent.routing.persistence import RoutingPersistence
from tests.test_precise_plans import approved

BEFORE = "".join(f"line {number}\n" for number in range(1, 21))


def task(*edits, references=()):
    return TaskContext("WP1-T2", "Change it", edits=list(edits), references=list(references))


def lines(context):
    return [(edit.start_line, edit.end_line) for edit in context.edits]


class RebaseTests(unittest.TestCase):
    def test_insertion_above_shifts_the_range(self):
        after = "new 1\nnew 2\nnew 3\n" + BEFORE
        moved = rebase_task(task(EditRange("a.py", "edit", 10, 12)), {"a.py": BEFORE},
                            {"a.py": after})
        self.assertEqual(lines(moved), [(13, 15)])

    def test_insertion_inside_grows_the_range(self):
        rows = BEFORE.splitlines(keepends=True)
        after = "".join(rows[:10] + ["inserted\n", "inserted\n"] + rows[10:])
        moved = rebase_task(task(EditRange("a.py", "edit", 10, 12)), {"a.py": BEFORE},
                            {"a.py": after})
        self.assertEqual(lines(moved), [(10, 14)])

    def test_deletion_above_and_replaced_range(self):
        rows = BEFORE.splitlines(keepends=True)
        deleted = "".join(rows[3:])
        self.assertEqual(lines(rebase_task(task(EditRange("a.py", "edit", 10, 12)),
                                           {"a.py": BEFORE}, {"a.py": deleted})), [(7, 9)])
        replaced = "".join(rows[:9] + ["rewritten\n"] + rows[12:])
        self.assertEqual(lines(rebase_task(task(EditRange("a.py", "edit", 10, 12)),
                                           {"a.py": BEFORE}, {"a.py": replaced})), [(10, 10)])

    def test_unchanged_files_new_files_and_references(self):
        after = "header\n" + BEFORE
        context = task(EditRange("b.py", "edit", 4, 6), EditRange("new.py", "create it"),
                       references=[ReferenceRange("a.py", "API", 2, 3),
                                   ReferenceRange("a.py", "whole file")])
        moved = rebase_task(context, {"a.py": BEFORE, "b.py": BEFORE},
                            {"a.py": after, "b.py": BEFORE})
        self.assertEqual(lines(moved), [(4, 6), (None, None)])
        self.assertEqual([(ref.start_line, ref.end_line) for ref in moved.references],
                         [(3, 4), (None, None)])
        self.assertIs(rebase_task(context, {}, {}), context)


class PackageMarkerTests(unittest.TestCase):
    def test_new_package_gets_a_marker_when_siblings_use_them(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "trainer").mkdir()
            Path(root, "trainer", "__init__.py").write_text("")
            context = task(EditRange("generation/factory.py", "create the factory"),
                           EditRange("generation/errors.py", "create errors"))
            marked = with_package_markers(context, set(), Path(root))
            self.assertEqual([edit.file for edit in marked.edits][-1],
                             "generation/__init__.py")
            self.assertEqual(len(marked.edits), 3)
            planned = with_package_markers(context, {"generation/__init__.py"}, Path(root))
            self.assertIs(planned, context)

    def test_namespace_packages_and_existing_markers_are_left_alone(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "trainer").mkdir()
            context = task(EditRange("generation/factory.py", "create the factory"))
            self.assertIs(with_package_markers(context, set(), Path(root)), context)
            Path(root, "trainer", "__init__.py").write_text("")
            Path(root, "generation").mkdir()
            Path(root, "generation", "__init__.py").write_text("")
            self.assertIs(with_package_markers(context, set(), Path(root)), context)


class FittedPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_later_task_uses_ranges_moved_by_an_earlier_task(self):
        with tempfile.TemporaryDirectory() as root:
            store = RoutingPersistence(Path(root) / "routing.sqlite3")
            plan_time = "".join(f"line {number}\n" for number in range(1, 80))
            await store.record_rollback_file("job", 0, "src/a.py", True, None, plan_time)
            Path(root, "src").mkdir()
            Path(root, "src", "a.py").write_text("added\n" * 5 + plan_time)
            service = SimpleNamespace(persistence=store)
            with mock.patch("agent.routing.implementation.native_project",
                            return_value=SimpleNamespace(workspace=root)):
                plan = approved()
                fitted = await ImplementationService._fitted_plan(
                    service, plan, "job", ["WP1-T1"])
            edit = fitted.tasks["WP1-T1"].edits[0]
            self.assertEqual((edit.file, edit.start_line, edit.end_line), ("src/a.py", 45, 63))
            self.assertEqual(fitted.tasks["WP1-T2"], plan.tasks["WP1-T2"])

    async def test_without_a_native_project_the_plan_is_unchanged(self):
        plan = approved()
        with mock.patch("agent.routing.implementation.native_project", return_value=None):
            fitted = await ImplementationService._fitted_plan(
                SimpleNamespace(persistence=None), plan, "job", None)
        self.assertIs(fitted, plan)


if __name__ == "__main__":
    unittest.main()

"""Chat presentation: blocks never run together and models still get raw data."""

import json
import tempfile
import unittest
from pathlib import Path

from agent.routing.implementation import ImplementationService, _without_handoff
from agent.routing.persistence import RoutingPersistence
from agent.routing.review import ReviewDecision, ReviewFinding
from agent.spec.models import CompletionReview, ModelIdentity
from agent.workflow.review_retry import menu_choices
from conversation import presentation as ui
from mcp_bridge.models import DoneEvent, TextEvent
from tests.test_routing import _Artifacts, _Reviewer

HANDOFF = {
    "schema_version": 1, "task_id": "WP1-T1", "summary": "Implemented the loader factory",
    "changed_files": [{"path": "data_loader/factory.py", "change": "created"}],
    "criteria": [{"criterion": "a", "status": "satisfied"},
                 {"criterion": "b", "status": "blocked"}],
    "remaining_gaps": ["docs not updated"], "execution_dispute": None,
}


def _identity(label):
    return ModelIdentity(label, "claude", label, None, label)


class HelperTests(unittest.TestCase):
    def test_blocks_end_with_a_blank_line_and_empty_sections_vanish(self):
        for text in (ui.block("x"), ui.heading("🛠", "T", "meta"), ui.section("S", ["a"]),
                     ui.quote("a\nb"), ui.menu(["1. Reply `a` to a.", "2. Reply `b` to b."]),
                     ui.coder_handoff(HANDOFF), ui.coder_handoff({})):
            self.assertTrue(text.endswith("\n\n"), text)
            self.assertFalse(text.endswith("\n\n\n"), text)
        self.assertEqual(ui.section("Empty", []), "")
        self.assertEqual(ui.paragraph("Why", " "), "")
        self.assertEqual(set(ui.STATUS), {"approved", "needs_work", "ask_user", "unverified",
                                          "failed", "interrupted"})

    def test_progress_lines_always_end_with_a_newline(self):
        for kind in ("header", "narration", "tool", "note", "separator"):
            self.assertTrue(ui.progress_line(kind, "a\nb").endswith("\n"))
        self.assertEqual(ui.progress_line("tool", "Reading `x`"), "- 🔎 Reading `x`\n")
        self.assertEqual(ui.progress_line("narration", "why\n now"), "\n💭 why now\n")

    def test_menus_still_parse(self):
        self.assertEqual(menu_choices(ui.menu(["1. Reply `accept` to keep it.",
                                               "2. Reply `undo` to restore."])),
                         {1: "accept", 2: "undo"})

    def test_handoff_summary_hides_the_json(self):
        text = ui.coder_handoff(HANDOFF)
        self.assertIn("Implemented the loader factory", text)
        self.assertIn("`data_loader/factory.py` (created)", text)
        self.assertIn("1 satisfied · 1 blocked", text)
        self.assertIn("docs not updated", text)
        self.assertNotIn("schema_version", text)
        prose = _without_handoff("I extended the registry.\n```json\n"
                                 + json.dumps(HANDOFF) + "\n```")
        self.assertEqual(prose, "I extended the registry.")

    def test_review_renders_findings_and_validation_table(self):
        finding = ReviewFinding("implementation", "Recurrent trainer missing",
                                "Register the GRU trainer", ("GRU is created by name",),
                                ("trainers/factory.py",))
        decision = ReviewDecision(
            "needs_work", "One trainer is missing", "codex-luna-low", None, None,
            ("Inspected the factory",), (), ("trainers/factory.py",), (finding,),
            validation_results=({"expectation": "Factory tests pass", "category": "required",
                                 "status": "failed"},))
        text = ImplementationService._decision_text(
            "WP1", "needs_work", decision,
            CompletionReview(_identity("Claude Haiku"), _identity("Codex Luna"), "r", []))
        self.assertIn("### 🔍 Review · WP1 — 🛠 Needs work", text)
        self.assertIn("coder: Claude Haiku → reviewer: Codex Luna", text)
        self.assertIn("1. **Recurrent trainer missing** — Register the GRU trainer", text)
        self.assertIn("   - Approve when: GRU is created by name", text)
        self.assertIn("| required | Factory tests pass | ❌ failed |", text)
        self.assertNotIn("```json", text)


class _HandoffExecution:
    async def execute(self, queued_task, change_id, messages, specs, **kwargs):
        yield TextEvent("I will extend the existing registry.")
        yield TextEvent(json.dumps({**HANDOFF, "task_id": queued_task.id}))
        yield DoneEvent("endTurn")


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_job_stream_is_readable_and_report_keeps_raw_json(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = RoutingPersistence(Path(directory) / "routing.sqlite3")

            async def actual_model(_decision_id):
                return "claude-haiku"

            persistence.actual_model_for = actual_model
            service = ImplementationService(_Artifacts(), _HandoffExecution(), persistence,
                                            _Reviewer(["done", "done"]))
            job = await service.create_job(["WP1-T1"])
            events = [event async for event in service.start(job, [], [])]
            texts = [event.text for event in events if isinstance(event, TextEvent)]
            self.assertTrue(all(text.endswith("\n") for text in texts), texts)
            stream = "".join(texts)
            self.assertIn(f"**Job {job.job_id[:8]}** queued 1 task(s)", stream)
            self.assertIn("### 🛠 WP1-T1 ·", stream)
            self.assertIn("> I will extend the existing registry.", stream)
            self.assertIn("**Coder handoff** (unverified until review)", stream)
            self.assertNotIn('"schema_version"', stream)
            self.assertNotRegex(stream, r"handoff\S*WP1|\.Starting|\)Reviewing")
            report = await persistence.coder_report_for(job.job_id, 0)
            self.assertIn('"schema_version": 1', report)


if __name__ == "__main__":
    unittest.main()

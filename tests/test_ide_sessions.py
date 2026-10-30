"""IDE session identity without the bootstrap tool call: disk config, conversation
marker, persistence across restarts.

    .venv/bin/python -m unittest tests.test_ide_sessions -v
"""

import asyncio
import json
from types import SimpleNamespace
from unittest import mock

from agent.spec import ide_mcp_sessions as sessions
from agent.spec import session_store
from tests import temp_db
from tests.test_workspace import _Fixture

CONFIG = json.dumps({"type": "streamable-http", "url": "http://127.0.0.1:64462/stream",
                     "headers": {"IJ_MCP_SERVER_PROJECT_PATH": "/home/u/demo"}})


def setUpModule():
    temp_db.start()


def tearDownModule():
    temp_db.stop()


def _message(role, content, tool_calls=None, tool_call_id=None):
    return SimpleNamespace(role=role, content=content, tool_calls=tool_calls,
                           tool_call_id=tool_call_id)


class NativeBootstrapConfigTests(_Fixture):
    def _with_config(self, project_id):
        path = self.project(project_id)
        (path / ".codepartner").mkdir()
        (path / ".codepartner" / "ide_mcp.json").write_text(CONFIG)
        return path

    def test_single_mounted_project_config_is_read_from_disk(self):
        self._with_config("demo")
        self.assertEqual(sessions.native_bootstrap_config(), CONFIG)

    def test_no_config_needs_the_bootstrap(self):
        self.project("demo")
        self.assertIsNone(sessions.native_bootstrap_config())

    def test_several_candidates_are_ambiguous(self):
        first = self._with_config("one")
        text = self.config.read_text()
        second = self._with_config("two")
        self.register(text + self.config.read_text().split("projects:\n", 1)[1])
        self.mounted.update({str(first), str(second)})
        self.assertEqual(len(sessions.project_registry().projects()), 2)
        self.assertIsNone(sessions.native_bootstrap_config())


class ConversationMarkerTests(_Fixture):
    def test_marker_in_assistant_history_identifies_the_conversation(self):
        conversation_id = sessions.new_conversation_id()
        request = SimpleNamespace(conversation_id=None, messages=[
            _message("user", "/project"),
            _message("assistant", "Project demo." + sessions.conversation_marker(conversation_id)),
            _message("user", "next"),
        ])
        self.assertEqual(sessions.conversation_for_request(request), conversation_id)

    def test_marker_in_a_user_message_is_ignored(self):
        request = SimpleNamespace(conversation_id=None, messages=[
            _message("user", sessions.conversation_marker(sessions.new_conversation_id()))])
        self.assertIsNone(sessions.conversation_for_request(request))

    def test_marker_is_stripped_before_the_model_sees_history(self):
        marker = sessions.conversation_marker(sessions.new_conversation_id())
        cleaned = sessions.strip_conversation_marker([
            {"role": "assistant", "content": "Done." + marker}, {"role": "user", "content": "x"}])
        self.assertEqual(cleaned[0]["content"], "Done.")

    def test_marker_is_appended_before_the_finish_chunk(self):
        from api.dialog_controller import _with_marker
        from conversation.streaming_responder import StreamingResponder

        async def body(response):
            return "".join([chunk if isinstance(chunk, str) else chunk.decode()
                            async for chunk in response.body_iterator])
        conversation_id = sessions.new_conversation_id()
        text = asyncio.run(body(_with_marker(
            StreamingResponder().reply("hello", "m", 1), conversation_id)))
        chunks = [json.loads(line[6:]) for line in text.split("\n\n")
                  if line.startswith("data: {")]
        contents = [c["choices"][0]["delta"].get("content") for c in chunks]
        self.assertEqual(contents[:2], ["hello", sessions.conversation_marker(conversation_id)])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")


class PersistenceTests(_Fixture):
    def test_bootstrap_ids_survive_a_restart(self):
        sessions.remember_bootstrap_call("call_1", "conv_" + "a" * 32)
        sessions._bootstrap_calls.clear()  # what a container restart loses
        request = SimpleNamespace(conversation_id=None, messages=[
            _message("assistant", None, tool_calls=[{"id": "call_1"}])])
        self.assertEqual(sessions.conversation_for_request(request), "conv_" + "a" * 32)

    def test_session_is_rebuilt_from_its_stored_config(self):
        conversation_id = "conv_" + "b" * 32
        session_store.save_config(conversation_id, CONFIG)
        session_store.save_binding(conversation_id, "demo")
        rebuilt = SimpleNamespace(workspace_id="demo")
        with mock.patch.object(sessions, "configure_session",
                               mock.AsyncMock(return_value=rebuilt)) as configure:
            self.assertIs(asyncio.run(sessions.restore_session(conversation_id)), rebuilt)
        configure.assert_awaited_once_with(conversation_id, CONFIG)
        self.assertEqual(sessions._workspace_bindings[conversation_id], "demo")
        self.assertIsNone(asyncio.run(sessions.restore_session("conv_" + "c" * 32)))

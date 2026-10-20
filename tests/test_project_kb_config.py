"""Project KB metadata is read through the resolved IDE MCP project."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.spec.ide_mcp_sessions import configure_session
from agent.spec.jetbrains_mcp import JetBrainsMcpError
from tests import temp_db


def setUpModule():
    temp_db.start()


def tearDownModule():
    temp_db.stop()


_IDE = {"type": "streamable-http", "url": "http://127.0.0.1:64462/stream",
        "headers": {"IJ_MCP_SERVER_PROJECT_PATH": "/target/repo"}}


class ProjectKbConfigTest(unittest.IsolatedAsyncioTestCase):
    async def _configure(self, yaml_text, legacy=None):
        class Client:
            def __init__(self, **_kwargs):
                pass

            async def file_exists(self, path):
                assert path == ".codepartner/config.yaml"
                return yaml_text is not None

            async def read_file(self, path):
                assert path == ".codepartner/config.yaml"
                return yaml_text

        class Tools:
            @classmethod
            async def discover(cls, _client):
                return SimpleNamespace(specs=(), handlers={})

        class Files:
            def __init__(self, _client, _specs):
                pass

            async def file_exists(self, path):
                return await Client().file_exists(path)

            async def read_file(self, path):
                return await Client().read_file(path)

        config = dict(_IDE)
        if legacy is not None:
            config["knowledge_base"] = legacy
        with patch("agent.spec.ide_mcp_sessions.JetBrainsMcpClient", Client), \
                patch("agent.spec.ide_mcp_sessions.DirectIdeTools", Tools), \
                patch("agent.spec.ide_mcp_sessions.DirectMcpFileClient", Files):
            return await configure_session("test_kb_config", json.dumps(config))

    async def test_yaml_mapping_wins_over_legacy(self):
        session = await self._configure(
            "project:\n  id: music-generation\n  knowledge_base:\n"
            "    provider: codebase-memory\n    project_id: exact-project\n",
            {"enabled": True, "provider": "codebase-memory", "project_id": "old"})
        self.assertEqual(session.project_name, "music-generation")
        self.assertEqual(session.project_path, "/target/repo")
        self.assertEqual(session.knowledge_base["project_id"], "exact-project")

    async def test_missing_yaml_keeps_legacy_mapping(self):
        session = await self._configure(None, {
            "enabled": True, "provider": "codebase-memory", "project_id": "old"})
        self.assertEqual(session.knowledge_base["project_id"], "old")

    async def test_invalid_yaml_provider_is_reported(self):
        with self.assertRaisesRegex(JetBrainsMcpError, "project.knowledge_base"):
            await self._configure("project:\n  id: demo\n  knowledge_base:\n"
                                  "    provider: other\n    project_id: x\n")

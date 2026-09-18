"""Tests for the codex provider and its bridged MCP endpoint.

Dependency-free (stdlib `unittest`, no pytest, no network, no `codex` process): run

    .venv/bin/python -m unittest tests.test_codex_events -v

Covers the three pieces that are ours rather than the CLI's: the JSONL event
mapping, the neutral-tier → codex-model mapping, and the guardrails the provider
claims — a read-only sandbox on every invocation, and an MCP endpoint that exposes
only what the run's role already allows.
"""
import asyncio
import os
import unittest
from unittest import mock

import mcp.types as types

from mcp_bridge import mcp_http
from mcp_bridge.models import ErrorEvent, Run, TextEvent
from mcp_bridge.registry import ROLE_CODER, ROLE_OPTIMIZER, ToolRegistry, ToolSpec, allowed_for
from providers.codex import CodexProvider, model_for, parse_event, usage_of

CODEX_ENV = ("CODEX_MODEL", "CODEX_MODEL_SIMPLE", "CODEX_MODEL_MEDIUM",
             "CODEX_MODEL_COMPLEX", "CODEX_CWD", "CODEX_BIN", "CODEX_ARGS")


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=name, schema={"type": "object", "properties": {}})


def _clean_env(**overrides):
    """Env with every CODEX_* var unset except the given overrides."""
    env = {k: v for k, v in os.environ.items() if k not in CODEX_ENV}
    env.update(overrides)
    return mock.patch.dict(os.environ, env, clear=True)


class ParseEvent(unittest.TestCase):
    def test_agent_message_becomes_text(self):
        event = parse_event({"type": "item.completed",
                             "item": {"type": "agent_message", "text": "done"}})
        self.assertEqual(event, TextEvent("done"))

    def test_empty_agent_message_is_dropped(self):
        # An empty TextEvent would be skipped by the SSE layer anyway; don't emit it.
        self.assertIsNone(parse_event({"type": "item.completed",
                                       "item": {"type": "agent_message", "text": ""}}))

    def test_error_item_becomes_error(self):
        event = parse_event({"type": "item.completed",
                             "item": {"type": "error", "message": "model metadata missing"}})
        self.assertEqual(event, ErrorEvent("model metadata missing"))

    def test_top_level_error_becomes_error(self):
        self.assertEqual(parse_event({"type": "error", "message": "400 bad request"}),
                         ErrorEvent("400 bad request"))

    def test_turn_failed_becomes_error(self):
        event = parse_event({"type": "turn.failed", "error": {"message": "not supported"}})
        self.assertEqual(event, ErrorEvent("not supported"))

    def test_turn_failed_without_message_still_reports(self):
        self.assertIsInstance(parse_event({"type": "turn.failed"}), ErrorEvent)

    def test_lifecycle_and_unknown_lines_are_ignored(self):
        for obj in ({"type": "thread.started", "thread_id": "x"},
                    {"type": "turn.started"},
                    {"type": "item.started", "item": {"type": "reasoning"}},
                    {"type": "item.completed", "item": {"type": "reasoning"}},
                    {"type": "item.completed", "item": {"type": "mcp_tool_call"}},
                    {"type": "something.new"},
                    {}):
            with self.subTest(obj=obj):
                self.assertIsNone(parse_event(obj))


class Usage(unittest.TestCase):
    def test_usage_from_turn_completed(self):
        usage = {"input_tokens": 10, "output_tokens": 3}
        self.assertEqual(usage_of({"type": "turn.completed", "usage": usage}), usage)

    def test_no_usage_elsewhere(self):
        self.assertIsNone(usage_of({"type": "turn.started", "usage": {"input_tokens": 1}}))
        self.assertIsNone(usage_of({"type": "turn.completed"}))
        self.assertIsNone(usage_of({"type": "turn.completed", "usage": "nope"}))


class ModelForTier(unittest.TestCase):
    def test_tiers_map_to_their_own_models(self):
        with _clean_env(CODEX_MODEL_SIMPLE="mini", CODEX_MODEL_MEDIUM="mid",
                        CODEX_MODEL_COMPLEX="max"):
            self.assertEqual(model_for("haiku"), "mini")
            self.assertEqual(model_for("sonnet"), "mid")
            self.assertEqual(model_for("opus"), "max")

    def test_tier_falls_back_to_the_shared_model(self):
        with _clean_env(CODEX_MODEL="fallback"):
            self.assertEqual(model_for("opus"), "fallback")
            self.assertEqual(model_for(None), "fallback")

    def test_nothing_configured_means_cli_default(self):
        with _clean_env():
            self.assertIsNone(model_for(None))
            self.assertIsNone(model_for("sonnet"))

    def test_unknown_tier_is_dropped_not_forwarded(self):
        # A bad routing decision should cost a default, not a CLI error.
        with _clean_env(CODEX_MODEL="fallback"):
            self.assertEqual(model_for("gpt-nonsense"), "fallback")
        with _clean_env():
            self.assertIsNone(model_for("gpt-nonsense"))


class SandboxArgs(unittest.TestCase):
    def test_every_invocation_is_read_only(self):
        with _clean_env():
            args = CodexProvider()._base_args()
        self.assertIn("exec", args)
        self.assertIn("--json", args)
        self.assertEqual(args[args.index("-s") + 1], "read-only")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", args)
        self.assertNotIn("-m", args)  # no model configured → CLI default

    def test_extra_args_cannot_loosen_the_sandbox(self):
        # CODEX_ARGS is a convenience (`--oss`, `--profile`, ...), not a way out of
        # the read-only sandbox: a repeated flag wins on the right, so ours is last.
        with _clean_env(CODEX_ARGS="--oss -s danger-full-access"):
            args = CodexProvider()._base_args()
        self.assertIn("--oss", args)
        self.assertEqual(args[-2:], ["-s", "read-only"])

    def test_configured_model_and_cwd_are_passed(self):
        with _clean_env(CODEX_MODEL_COMPLEX="big", CODEX_CWD="/tmp/project"):
            args = CodexProvider()._base_args("opus")
        self.assertEqual(args[args.index("-m") + 1], "big")
        self.assertEqual(args[args.index("--cd") + 1], "/tmp/project")


class BridgedEndpointHonorsTheRole(unittest.TestCase):
    """The endpoint codex talks to must expose the run's registry, not the raw specs
    JetBrains advertised — otherwise the per-role allowlist would stop at the door."""

    ADVERTISED = [_spec("get_file_text_by_path"), _spec("create_new_file"),
                  _spec("rename_refactoring")]

    def _server(self, role):
        run = Run(registry=ToolRegistry(allowed_for(role)))
        run.registry.register(self.ADVERTISED)
        return mcp_http._build_server(run)

    def _list(self, role):
        server = self._server(role)
        result = asyncio.run(
            server.request_handlers[types.ListToolsRequest](types.ListToolsRequest(method="tools/list"))
        )
        return sorted(t.name for t in result.root.tools)

    def test_optimizer_endpoint_exposes_reads_only(self):
        self.assertEqual(self._list(ROLE_OPTIMIZER), ["get_file_text_by_path"])

    def test_coder_endpoint_exposes_everything_advertised(self):
        self.assertEqual(self._list(ROLE_CODER),
                         sorted(s.name for s in self.ADVERTISED))

    def test_call_outside_the_role_is_refused(self):
        # Belt and braces: even if the model names an unlisted tool, it never reaches
        # the bridge (and so never reaches the IDE).
        server = self._server(ROLE_OPTIMIZER)
        result = asyncio.run(server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(name="create_new_file", arguments={}),
            )
        ))
        self.assertTrue(result.root.isError)
        self.assertIn("not available", result.root.content[0].text)


class SharedHistoryHelpers(unittest.TestCase):
    """`_flatten` / `_trailing_tool_results` moved onto `Provider`; both run-backed
    providers inherit the same behaviour."""

    def test_flatten_splits_system_and_drops_tool_plumbing(self):
        prompt, system = CodexProvider()._flatten([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file text"},
        ])
        self.assertEqual(system, "be brief")
        self.assertEqual(prompt, "fix the bug")

    def test_trailing_tool_results_are_found(self):
        results = CodexProvider()._trailing_tool_results([
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": "x"},
            {"role": "tool", "content": "no id"},
        ])
        self.assertEqual([m["tool_call_id"] for m in results], ["call_1"])


if __name__ == "__main__":
    unittest.main()

"""OpenTelemetry tracing: span tree, context propagation, privacy and resilience."""

import asyncio
import contextvars
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from claude_agent_sdk import AssistantMessage, ResultMessage
from mcp import types
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent.routing.config import load_routing_config
from agent.routing.executors.registry import ExecutorRegistry
from agent.routing.persistence import RoutingPersistence
from agent.routing.service import ModelRoutingService
from agent.routing.task_execution import TaskExecutionService
from agent.planning.routing import PlannerRoutingService
from mcp_bridge import bridge, mcp_http
from mcp_bridge import server as engine
from mcp_bridge.models import DoneEvent, ErrorEvent, TextEvent, ToolCallEvent
from mcp_bridge.registry import ToolSpec
from providers import claude as claude_provider
from providers.claude import ClaudeProvider
from providers.codex import CodexProvider
from telemetry import tracing
from tests import temp_db
from tests.test_routing import Router, task

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = {"type": "object", "properties": {"file_path": {"type": "string"},
                                           "path": {"type": "string"},
                                           "text": {"type": "string"}}}
SECRET = "sk-ant-api03-SUPERSECRETVALUE123456"
SOURCE = "def transfer(account, amount):  # confidential source line"


def setUpModule():
    temp_db.start()


def tearDownModule():
    temp_db.stop()


def spec(name: str) -> ToolSpec:
    return ToolSpec(name, "test tool", SCHEMA)


async def drain(run) -> list:
    events = []
    while True:
        event = await run.next_event()
        events.append(event)
        if isinstance(event, DoneEvent):
            return events


def fresh_task(coro):
    """A task with an empty context, like a separate HTTP request (Codex callback,
    IDE tool result): nothing from the Run's task is inherited."""
    return asyncio.create_task(coro, context=contextvars.Context())


class TracingCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tracing._reset_for_tests()
        self.exporter = InMemorySpanExporter()
        self.assertTrue(tracing.configure_tracing(self.exporter, force=True))

    def tearDown(self):
        tracing._reset_for_tests()

    def spans(self, name=None):
        found = self.exporter.get_finished_spans()
        return [span for span in found if name is None or span.name == name]

    def one(self, name):
        found = self.spans(name)
        self.assertEqual(len(found), 1, [span.name for span in self.spans()])
        return found[0]

    def assertChild(self, child, parent):
        self.assertEqual(child.context.trace_id, parent.context.trace_id)
        self.assertIsNotNone(child.parent)
        self.assertEqual(child.parent.span_id, parent.context.span_id)

    def all_attribute_text(self) -> str:
        return "\n".join(f"{key}={value}" for span in self.spans()
                         for key, value in (span.attributes or {}).items())


class RootExecutionTests(TracingCase):
    async def test_successful_run_produces_a_root_execution_span(self):
        tracing.set_session("conv_abc123")

        async def strategy(run, prompt, system, specs):
            await run.put(TextEvent("done"))
            return "endTurn"

        run = engine.start_run("prompt", None, [], strategy)
        run.metadata.update({"trace_id": "trace_deadbeef", "bridge_command": "confirm_implementation"})
        events = await drain(run)
        self.assertIsInstance(events[-1], DoneEvent)
        root = self.one(tracing.EXECUTION_SPAN)
        self.assertIsNone(root.parent)
        attrs = root.attributes
        self.assertEqual(attrs["codepartner.execution.id"], run.run_id)
        self.assertEqual(attrs["codepartner.execution.status"], "completed")
        self.assertEqual(attrs["codepartner.session.id"], "conv_abc123")
        self.assertEqual(attrs["langfuse.session.id"], "conv_abc123")
        self.assertEqual(attrs["langfuse.trace.metadata.execution_id"], run.run_id)
        self.assertEqual(attrs["codepartner.command"], "confirm_implementation")
        self.assertEqual(attrs["codepartner.diagnostic_trace_id"], "trace_deadbeef")
        self.assertEqual(root.status.status_code, StatusCode.OK)
        self.assertEqual(run.metadata["otel_trace_id"], format(root.context.trace_id, "032x"))

    async def test_failed_strategy_marks_trace_failed_without_message(self):
        async def strategy(run, prompt, system, specs):
            raise RuntimeError(f"provider exploded with {SECRET}")

        run = engine.start_run("", None, [], strategy)
        events = await drain(run)
        # Original error handling is unchanged: the message still reaches the stream.
        self.assertIn(SECRET, events[0].message)
        root = self.one(tracing.EXECUTION_SPAN)
        self.assertEqual(root.status.status_code, StatusCode.ERROR)
        self.assertEqual(root.attributes["error.type"], "RuntimeError")
        self.assertEqual(root.attributes["codepartner.execution.status"], "failed")
        self.assertEqual(root.attributes["codepartner.execution.errors"], 1)
        self.assertNotIn(SECRET, self.all_attribute_text())
        self.assertNotIn(SECRET, root.status.description or "")
        self.assertEqual(list(root.events), [])  # no exception event or stack trace

    async def test_cancelled_run_is_marked_cancelled(self):
        started = asyncio.Event()

        async def strategy(run, prompt, system, specs):
            started.set()
            await asyncio.sleep(30)
            return "endTurn"

        run = engine.start_run("", None, [], strategy)
        await started.wait()
        run.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run.task
        root = self.one(tracing.EXECUTION_SPAN)
        self.assertEqual(root.attributes["codepartner.execution.status"], "cancelled")
        self.assertEqual(root.attributes["error.type"], "cancelled")


class ContextPropagationTests(TracingCase):
    async def test_detached_task_carries_the_run_span_and_restores_the_caller(self):
        seen = {}
        caller = tracing._tracer.start_span("caller")
        token = trace.use_span(caller, end_on_exit=True)
        with token:
            async def strategy(run, prompt, system, specs):
                seen["current"] = trace.get_current_span().get_span_context()
                seen["execution"] = tracing._execution.get()
                return "endTurn"

            run = engine.start_run("", None, [], strategy)
            # The caller's own context is untouched by start_run.
            self.assertIs(trace.get_current_span(), caller)
            self.assertIsNone(tracing._execution.get())
        await drain(run)
        root = self.one(tracing.EXECUTION_SPAN)
        caller_span = self.one("caller")
        self.assertIsNone(root.parent)  # a Run is a new trace, not a request child
        self.assertNotEqual(root.context.trace_id, caller_span.context.trace_id)
        self.assertEqual(seen["current"].span_id, root.context.span_id)
        self.assertIs(seen["execution"], run.trace.execution)

    async def test_nested_provider_run_and_codex_style_callback_parent_on_llm_span(self):
        handled = []

        async def handler(arguments):
            handled.append(arguments)
            return "contents"

        async def provider_strategy(run, prompt, system, specs):
            # The Codex child calls back over HTTP: a different task, empty context.
            result = await fresh_task(bridge.call_tool(run, "read_file", {"file_path": "a.py"}))
            await run.put(TextEvent(result))
            return "endTurn"

        async def command_strategy(run, prompt, system, specs):
            child = engine.start_run("", None, [spec("read_file")], provider_strategy,
                                     direct_tool_handlers={"read_file": handler},
                                     trace_kind="llm", trace_provider="codex",
                                     trace_model="gpt-5.1-codex")
            async for event in _events(child):
                await run.put(event)
            return "endTurn"

        parent = engine.start_run("", None, [], command_strategy)
        await drain(parent)
        root = self.one(tracing.EXECUTION_SPAN)
        llm = self.one(tracing.LLM_SPAN)
        tool = self.one(tracing.TOOL_SPAN)
        self.assertChild(llm, root)
        self.assertChild(tool, llm)
        self.assertNotEqual(tool.parent.span_id, root.context.span_id)
        self.assertEqual(llm.attributes["codepartner.execution.id"], parent.run_id)
        self.assertEqual(tool.attributes["codepartner.execution.id"], parent.run_id)
        self.assertEqual(root.attributes["codepartner.execution.tool_calls"], 1)
        self.assertEqual(root.attributes["codepartner.provider"], "codex")
        self.assertEqual([call["file_path"] for call in handled], ["a.py"])

    async def test_ide_round_trip_resolved_by_another_request(self):
        async def strategy(run, prompt, system, specs):
            text = await bridge.call_tool(run, "get_file_problems", {"path": "a.py"})
            await run.put(TextEvent(text))
            return "endTurn"

        run = engine.start_run("", None, [spec("get_file_problems")], strategy,
                               trace_kind="llm", trace_provider="claude", trace_model="sonnet")
        call = await run.next_event()
        self.assertIsInstance(call, ToolCallEvent)

        async def later_request():
            return engine.resume(call.tool_call_id, "no problems")

        self.assertIs(await fresh_task(later_request()), run)
        await drain(run)
        root = self.one(tracing.EXECUTION_SPAN)
        llm = self.one(tracing.LLM_SPAN)
        tool = self.one(tracing.TOOL_SPAN)
        self.assertChild(llm, root)
        self.assertChild(tool, llm)
        self.assertEqual(tool.attributes["codepartner.mcp.route"], "ide_bridge")
        self.assertEqual(tool.attributes["codepartner.mcp.server"], "jetbrains-ide")
        self.assertEqual(tool.attributes["codepartner.mcp.status"], "ok")
        self.assertEqual(tool.attributes["gen_ai.tool.name"], "get_file_problems")

    async def test_concurrent_runs_never_mix_traces(self):
        async def make(index):
            async def handler(arguments):
                await asyncio.sleep(0.01 * (3 - index))
                return f"r{index}"

            async def strategy(run, prompt, system, specs):
                for _ in range(3):
                    await fresh_task(bridge.call_tool(run, "read_file", {"file_path": f"{index}.py"}))
                    await asyncio.sleep(0)
                return "endTurn"

            run = engine.start_run("", None, [spec("read_file")], strategy,
                                   direct_tool_handlers={"read_file": handler},
                                   trace_kind="llm", trace_provider="claude")
            await drain(run)
            return run

        runs = await asyncio.gather(*(make(index) for index in range(3)))
        llm_by_run = {span.attributes["codepartner.run.id"]: span
                      for span in self.spans(tracing.LLM_SPAN)}
        roots = {span.attributes["codepartner.execution.id"]: span
                 for span in self.spans(tracing.EXECUTION_SPAN)}
        self.assertEqual(len({span.context.trace_id for span in roots.values()}), 3)
        tools = self.spans(tracing.TOOL_SPAN)
        self.assertEqual(len(tools), 9)
        for tool in tools:
            run_id = tool.attributes["codepartner.run.id"]
            self.assertChild(tool, llm_by_run[run_id])
            self.assertChild(llm_by_run[run_id], roots[run_id])
        self.assertEqual({run.run_id for run in runs}, set(llm_by_run))


async def _events(run):
    while True:
        event = await run.next_event()
        yield event
        if isinstance(event, DoneEvent):
            return


class ToolStatusTests(TracingCase):
    async def _call(self, handler, name="read_file", arguments=None, call_name=None):
        captured = {}

        async def strategy(run, prompt, system, specs):
            try:
                captured["result"] = await bridge.call_tool(run, call_name or name, arguments or {"file_path": "a.py"})
            except BaseException as exc:  # noqa: BLE001 - asserting propagation
                captured["error"] = exc
            return "endTurn"

        handlers = {name: handler} if handler is not None else {}
        run = engine.start_run("", None, [spec(name)], strategy,
                               direct_tool_handlers=handlers, trace_kind="llm",
                               trace_provider="claude")
        await drain(run)
        return captured, self.one(tracing.TOOL_SPAN)

    async def test_error_result_is_returned_unchanged_and_marked(self):
        error_text = "Error: file not found"

        async def handler(arguments):
            return error_text

        captured, span = await self._call(handler)
        self.assertIs(captured["result"], error_text)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "tool_error")
        self.assertEqual(span.attributes["codepartner.mcp.status"], "error")
        self.assertEqual(span.attributes["codepartner.mcp.route"], "direct")

    async def test_handler_exception_keeps_existing_behaviour(self):
        async def handler(arguments):
            raise ValueError(f"backend leaked {SECRET}")

        captured, span = await self._call(handler)
        # _call_direct already converts handler failures into an error result.
        self.assertTrue(captured["result"].startswith("Error: direct PyCharm MCP tool"))
        self.assertEqual(span.attributes["error.type"], "tool_error")
        self.assertNotIn(SECRET, self.all_attribute_text())

    async def test_dispatch_exception_is_the_same_object(self):
        boom = ConnectionError("socket closed")

        async def failing(run, name, arguments, tool):
            raise boom

        with mock.patch.object(bridge, "_dispatch_backend", failing):
            captured, span = await self._call(None)
        self.assertIs(captured["error"], boom)
        self.assertEqual(span.attributes["error.type"], "ConnectionError")
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

    async def test_timeout_propagates_and_is_classified(self):
        with mock.patch.object(bridge, "TOOL_TIMEOUT", 0.05):
            captured, span = await self._call(None, name="get_file_problems")
        self.assertIsInstance(captured["error"], TimeoutError)
        self.assertEqual(span.attributes["error.type"], "timeout")
        self.assertEqual(span.attributes["codepartner.mcp.route"], "ide_bridge")

    async def test_invalid_call_is_a_refusal(self):
        captured, span = await self._call(None, call_name="unregistered_tool")
        self.assertTrue(captured["result"].startswith("Error: invalid MCP tool call"))
        self.assertEqual(span.attributes["codepartner.mcp.route"], "refused")
        self.assertEqual(span.attributes["error.type"], "invalid_call")

    async def test_kb_tool_is_attributed_to_codebase_memory(self):
        async def handler(arguments):
            return '{"results": []}'

        captured, span = await self._call(handler, name="kb_search_graph")
        self.assertEqual(span.attributes["codepartner.mcp.server"], "codebase-memory")
        self.assertEqual(span.attributes["codepartner.mcp.status"], "ok")


class ProviderTests(TracingCase):
    async def test_claude_turns_show_context_growth_without_double_counting(self):
        async def fake_query(prompt, options):
            for amount in (80, 240):
                yield AssistantMessage(content=[], model="claude-sonnet-4-5",
                                       usage={"input_tokens": amount, "output_tokens": 7,
                                              "cache_read_input_tokens": 20,
                                              "cache_creation_input_tokens": 0})
            yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                                is_error=False, num_turns=2, session_id="s",
                                usage={"input_tokens": 320, "output_tokens": 14,
                                       "cache_read_input_tokens": 40,
                                       "cache_creation_input_tokens": 0},
                                terminal_reason="endTurn")

        with mock.patch.object(claude_provider, "_query_messages", fake_query):
            [event async for event in ClaudeProvider().tools(
                [{"role": "user", "content": "go"}], [], model="sonnet")]
        turns = self.spans(tracing.TURN_SPAN)
        self.assertEqual([span.attributes["codepartner.turn.sequence"] for span in turns],
                         [1, 2])
        self.assertEqual([span.attributes["codepartner.turn.context_tokens"] for span in turns],
                         [100, 260])
        self.assertEqual(sum(span.attributes["gen_ai.usage.input_tokens"] for span in turns),
                         360)
        self.assertNotIn("gen_ai.usage.input_tokens", self.one(tracing.LLM_SPAN).attributes)

    async def test_claude_run_records_usage_and_in_process_tool_parenting(self):
        async def handler(arguments):
            return SOURCE

        async def fake_query(prompt, options):
            instance = options.mcp_servers[claude_provider.SERVER_NAME]["instance"]
            call = instance.request_handlers[types.CallToolRequest]
            await call(types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(
                name="read_file", arguments={"file_path": "bank.py"})))
            yield AssistantMessage(content=[], model="claude-sonnet-4-5",
                                   usage={"input_tokens": 100, "output_tokens": 20,
                                          "cache_read_input_tokens": 50,
                                          "cache_creation_input_tokens": 5})
            yield ResultMessage(
                subtype="success", duration_ms=10, duration_api_ms=8, is_error=False,
                num_turns=1, session_id="s", total_cost_usd=0.01,
                usage={"input_tokens": 100, "output_tokens": 20,
                       "cache_read_input_tokens": 50, "cache_creation_input_tokens": 5},
                model_usage={"claude-haiku-4-5": {"inputTokens": 30, "outputTokens": 2},
                             "claude-sonnet-4-5": {"inputTokens": 100, "outputTokens": 20}},
                terminal_reason="endTurn")

        with mock.patch.object(claude_provider, "_query_messages", fake_query):
            events = [event async for event in ClaudeProvider().tools(
                [{"role": "user", "content": SOURCE}], [spec("read_file")], model="sonnet",
                direct_tool_handlers={"read_file": handler})]
        self.assertIsInstance(events[-1], DoneEvent)
        root, llm = self.one(tracing.EXECUTION_SPAN), self.one(tracing.LLM_SPAN)
        turn = self.one(tracing.TURN_SPAN)
        tool = self.one(tracing.TOOL_SPAN)
        self.assertChild(llm, root)
        self.assertChild(turn, llm)
        self.assertChild(tool, turn)
        attrs = llm.attributes
        self.assertEqual(attrs["codepartner.provider"], "claude")
        self.assertEqual(attrs["gen_ai.provider.name"], "anthropic")
        self.assertEqual(attrs["gen_ai.request.model"], "sonnet")
        self.assertEqual(attrs["gen_ai.response.model"], "claude-sonnet-4-5")
        self.assertEqual(attrs["codepartner.aggregate.input_tokens"], 155)
        self.assertEqual(attrs["langfuse.observation.type"], "agent")
        self.assertFalse([key for key in attrs if key.startswith("gen_ai.usage")])
        self.assertEqual(turn.attributes["gen_ai.usage.input_tokens"], 155)
        self.assertEqual(turn.attributes["gen_ai.usage.cache_read.input_tokens"], 50)
        self.assertEqual(turn.attributes["codepartner.turn.context_tokens"], 155)
        self.assertEqual(turn.attributes["codepartner.turn.context_measurement"], "input_proxy")
        self.assertEqual(turn.attributes["codepartner.turn.tool_calls"], 1)
        self.assertNotIn("gen_ai.operation.name", attrs)
        self.assertNotIn(SOURCE, self.all_attribute_text())

    async def test_claude_without_reported_usage_records_none(self):
        async def fake_query(prompt, options):
            if False:
                yield None

        with mock.patch.object(claude_provider, "_query_messages", fake_query):
            [event async for event in ClaudeProvider().tools(
                [{"role": "user", "content": "hi"}], [], model="haiku")]
        attrs = self.one(tracing.LLM_SPAN).attributes
        self.assertFalse([key for key in attrs if key.startswith("gen_ai.usage")])
        self.assertEqual(self.one(tracing.TURN_SPAN).attributes[
            "codepartner.turn.context_measurement"], "unavailable")

    def test_missing_turn_usage_stays_unavailable(self):
        attributes = tracing._turn_usage_attributes({"input_tokens": 10}, "codex")
        self.assertEqual(attributes["gen_ai.usage.input_tokens"], 10)
        self.assertNotIn("gen_ai.usage.output_tokens", attributes)
        self.assertNotIn("gen_ai.usage.cache_read.input_tokens", attributes)
        self.assertEqual(attributes["codepartner.turn.context_measurement"], "unavailable")

    def test_codex_cache_write_is_counted_in_input_without_fabrication(self):
        attributes = tracing._turn_usage_attributes({
            "input_tokens": 100, "cached_input_tokens": 30,
            "cache_write_input_tokens": 10, "output_tokens": 5}, "codex")
        self.assertEqual(attributes["gen_ai.usage.input_tokens"], 100)
        self.assertEqual(attributes["gen_ai.usage.cache_creation.input_tokens"], 10)

    async def test_codex_run_and_real_http_handler_callback(self):
        async def handler(arguments):
            return "file text"

        async def fake_exec(self, args, prompt, env=None, report_exit=False):
            yield {"type": "turn.started"}
            run = next(iter(engine._live_runs.values()))
            server = mcp_http._build_server(run)
            call = server.request_handlers[types.CallToolRequest]
            # The codex child process calls the per-run endpoint: another request task.
            result = await fresh_task(call(types.CallToolRequest(
                method="tools/call", params=types.CallToolRequestParams(
                    name="read_file", arguments={"file_path": "a.py"}))))
            assert not result.root.isError, result
            yield {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}
            yield {"type": "turn.completed", "usage": {
                "input_tokens": 300, "cached_input_tokens": 100, "output_tokens": 40}}

        with mock.patch.object(CodexProvider, "_exec", fake_exec):
            events = [event async for event in CodexProvider().tools(
                [{"role": "user", "content": "go"}], [spec("read_file")],
                model="gpt-5.1-codex", concrete_model=True, reasoning_effort="high",
                direct_tool_handlers={"read_file": handler})]
        self.assertIsInstance(events[-1], DoneEvent)
        root, llm = self.one(tracing.EXECUTION_SPAN), self.one(tracing.LLM_SPAN)
        turn = self.one(tracing.TURN_SPAN)
        tool = self.one(tracing.TOOL_SPAN)
        self.assertChild(llm, root)
        self.assertChild(turn, llm)
        self.assertChild(tool, turn)
        attrs = llm.attributes
        self.assertEqual(attrs["codepartner.provider"], "codex")
        self.assertEqual(attrs["gen_ai.provider.name"], "openai")
        self.assertEqual(attrs["gen_ai.request.model"], "gpt-5.1-codex")
        self.assertEqual(attrs["codepartner.routing.reasoning_effort"], "high")
        self.assertEqual(attrs["codepartner.aggregate.input_tokens"], 300)
        self.assertEqual(turn.attributes["gen_ai.usage.input_tokens"], 300)
        self.assertEqual(turn.attributes["codepartner.turn.context_measurement"], "unavailable")
        self.assertEqual(turn.attributes["codepartner.turn.tool_calls"], 1)
        self.assertEqual(attrs["codepartner.execution.tool_calls"], 1)


class RoutingTests(TracingCase):
    async def test_coder_routing_span_is_a_child_with_decision_attributes(self):
        class Executor:
            async def execute(self, _task, _model, context):
                yield TextEvent("finished")
                yield DoneEvent("endTurn")

        captured = {}

        async def strategy(run, prompt, system, specs):
            with tempfile.TemporaryDirectory() as directory:
                config = load_routing_config()
                service = TaskExecutionService(
                    config=config, router=ModelRoutingService(config, Router("claude-haiku")),
                    persistence=RoutingPersistence(Path(directory) / "r.sqlite3"),
                    executors=ExecutorRegistry({"claude-cli": Executor(), "codex-cli": Executor()}))
                captured["events"] = [e async for e in service.execute(task("simple"), "CHG1", [], [])]
            return "endTurn"

        run = engine.start_run("", None, [], strategy)
        await drain(run)
        root, routing = self.one(tracing.EXECUTION_SPAN), self.one(tracing.ROUTING_SPAN)
        self.assertChild(routing, root)
        attrs = routing.attributes
        self.assertEqual(attrs["codepartner.routing.kind"], "coder")
        self.assertEqual(attrs["codepartner.routing.complexity"], "simple")
        self.assertEqual(attrs["codepartner.routing.strategy"], "semantic-router")
        self.assertEqual(attrs["codepartner.model"], "claude-haiku")
        self.assertEqual(attrs["codepartner.task.id"], "WP1-T1")
        self.assertTrue(attrs["codepartner.decision.id"].startswith("route_"))
        self.assertIn("codepartner.routing.duration_ms", attrs)
        self.assertEqual(root.attributes["codepartner.task.id"], "WP1-T1")
        self.assertEqual(root.attributes["codepartner.routing.strategy"], "semantic-router")
        self.assertEqual(root.attributes["langfuse.trace.metadata.task_id"], "WP1-T1")
        # The free-text routing reason is never exported.
        self.assertNotIn("chosen for test", self.all_attribute_text())

    async def test_routing_failure_marks_span_and_propagates_the_same_error(self):
        config = load_routing_config()
        service = TaskExecutionService(
            config=config, router=ModelRoutingService(config, Router("claude-haiku")),
            persistence=mock.Mock(), executors=ExecutorRegistry({}))
        boom = LookupError(f"no candidates for {SECRET}")
        with mock.patch.object(service.router, "route", side_effect=boom):
            with self.assertRaises(LookupError) as raised:
                [e async for e in service.execute(task("simple"), "CHG1", [], [])]
        self.assertIs(raised.exception, boom)
        routing = self.one(tracing.ROUTING_SPAN)
        self.assertEqual(routing.status.status_code, StatusCode.ERROR)
        self.assertEqual(routing.attributes["error.type"], "LookupError")
        self.assertNotIn(SECRET, self.all_attribute_text())

    def test_planner_routing_span(self):
        config = load_routing_config()
        decision = PlannerRoutingService(config).route("medium")
        routing = self.one(tracing.ROUTING_SPAN)
        self.assertEqual(routing.attributes["codepartner.routing.kind"], "planner")
        self.assertEqual(routing.attributes["codepartner.model"], decision.model)
        self.assertEqual(routing.attributes["codepartner.routing.strategy"], "planner-tier")


class PrivacyTests(TracingCase):
    def test_schema_drops_unknown_keys_free_text_and_secrets(self):
        cleaned = tracing.clean_attributes({
            "codepartner.unknown": "x",
            "user.prompt": "please fix",
            "codepartner.task.id": "WP1-T1",
            "codepartner.command": "fix the login bug please",   # free text: not an id
            "codepartner.decision.id": "sk-ant-api03-abcdefghijklmnop",  # secret-shaped id
            "codepartner.provider": "evilcorp",                  # not in the enum
            "error.type": "connection refused by 10.0.0.1",      # a message, not a type
            "codepartner.mcp.argument_keys": ["path", "Bearer xyz", "text"],
            "codepartner.execution.tool_calls": -3,
        })
        self.assertEqual(cleaned, {"codepartner.task.id": "WP1-T1",
                                   "codepartner.mcp.argument_keys": ["path", "text"]})

    async def test_no_arguments_results_or_prompts_are_exported(self):
        async def handler(arguments):
            return SOURCE * 3

        async def strategy(run, prompt, system, specs):
            await bridge.call_tool(run, "read_file", {"file_path": "secret/keys.py",
                                                     "text": f"Authorization: Bearer {SECRET}"})
            return "endTurn"

        run = engine.start_run(f"{SOURCE}\nkey={SECRET}", "system prompt", [spec("read_file")],
                               strategy, direct_tool_handlers={"read_file": handler},
                               trace_kind="llm", trace_provider="claude")
        run.metadata.update({"conversation_text": SOURCE, "trace_id": "trace_ok",
                             "task_id": SECRET})
        await drain(run)
        text = self.all_attribute_text()
        for forbidden in (SECRET, SOURCE, "secret/keys.py", "Bearer", "system prompt"):
            self.assertNotIn(forbidden, text)
        tool = self.one(tracing.TOOL_SPAN)
        self.assertEqual(tool.attributes["codepartner.mcp.argument_keys"], ("file_path", "text"))
        self.assertGreater(tool.attributes["codepartner.mcp.result_bytes"], 0)


class DisabledTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tracing._reset_for_tests()

    async def test_disabled_tracing_changes_nothing(self):
        with mock.patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            self.assertFalse(tracing.configure_tracing())

        async def handler(arguments):
            return "Error: nope"

        async def strategy(run, prompt, system, specs):
            text = await bridge.call_tool(run, "read_file", {"file_path": "a.py"})
            await run.put(TextEvent(text))
            await run.put(ErrorEvent("visible"))
            return "endTurn"

        run = engine.start_run("", None, [spec("read_file")], strategy,
                               direct_tool_handlers={"read_file": handler},
                               trace_kind="llm", trace_provider="claude")
        events = await drain(run)
        self.assertIsNone(run.trace)
        self.assertNotIn("otel_trace_id", run.metadata)
        self.assertEqual([type(event).__name__ for event in events],
                         ["TextEvent", "ErrorEvent", "DoneEvent"])
        self.assertEqual(events[0].text, "Error: nope")


class _BlackHole:
    """A TCP server that accepts connections and never answers (a hung collector)."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.conns = []
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self):
        try:
            while True:
                self.conns.append(self.sock.accept()[0])
        except OSError:
            pass

    def close(self):
        for conn in self.conns:
            conn.close()
        self.sock.close()


class CollectorUnavailableTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tracing._reset_for_tests()

    def tearDown(self):
        tracing._reset_for_tests()

    async def _execute_with_endpoint(self, endpoint: str, timeout: str):
        env = {"OTEL_ENABLED": "true", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint,
               "OTEL_EXPORTER_OTLP_TIMEOUT": timeout, "OTEL_BSP_SCHEDULE_DELAY": "10"}
        with mock.patch.dict(os.environ, env):
            self.assertTrue(tracing.configure_tracing())

            async def strategy(run, prompt, system, specs):
                await run.put(TextEvent("still works"))
                return "endTurn"

            started = time.monotonic()
            run = engine.start_run("", None, [], strategy, trace_kind="llm",
                                   trace_provider="claude")
            events = await drain(run)
            elapsed = time.monotonic() - started
        self.assertEqual(events[0].text, "still works")
        self.assertIsInstance(events[-1], DoneEvent)
        self.assertLess(elapsed, 1.0)  # export never sits on the execution path

    async def test_refused_connection_does_not_affect_execution_or_shutdown(self):
        await self._execute_with_endpoint("http://127.0.0.1:9/v1/traces", "1")
        started = time.monotonic()
        tracing.shutdown_tracing(timeout=3)
        self.assertLess(time.monotonic() - started, 3.5)

    async def test_hung_collector_shutdown_is_hard_bounded(self):
        hole = _BlackHole()
        try:
            # A 60s export timeout: force_flush/shutdown would block far longer than
            # the bound if shutdown were not bounded independently.
            await self._execute_with_endpoint(f"http://127.0.0.1:{hole.port}/v1/traces", "60")
            await asyncio.sleep(0.2)  # let the batch thread start an export that hangs
            started = time.monotonic()
            finished = tracing.shutdown_tracing(timeout=2)
            elapsed = time.monotonic() - started
        finally:
            hole.close()
        self.assertFalse(finished)
        self.assertLess(elapsed, 2.5)

    def test_process_exit_is_bounded_with_hung_collector(self):
        hole = _BlackHole()
        script = (
            "import time, telemetry.tracing as t\n"
            "assert t.configure_tracing()\n"
            "for i in range(50): t._tracer.start_span('x').end()\n"
            "time.sleep(0.3)\n"
            "t.shutdown_tracing()\n"
        )
        env = {**os.environ, "OTEL_ENABLED": "true", "OTEL_SHUTDOWN_TIMEOUT_SECONDS": "2",
               "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"http://127.0.0.1:{hole.port}/v1/traces",
               "OTEL_EXPORTER_OTLP_TIMEOUT": "60", "OTEL_BSP_SCHEDULE_DELAY": "10"}
        try:
            started = time.monotonic()
            done = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                                  capture_output=True, timeout=30)
            elapsed = time.monotonic() - started
        finally:
            hole.close()
        self.assertEqual(done.returncode, 0, done.stderr.decode()[-2000:])
        self.assertLess(elapsed, 6.0)


if __name__ == "__main__":
    unittest.main()

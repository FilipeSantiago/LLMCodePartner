"""Deterministic per-contract tool authorization and observation control."""

import hashlib
import json
import shlex
import time
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from agent.task_contracts.models import TaskContract
from agent.task_contracts.policy import path_excluded, protected_path, secret_path
from mcp_bridge.adapters import mutation_targets
from mcp_bridge.registry import (
    classify_tool, is_mutating_tool, OP_FILE_CREATE, OP_FILE_PATCH, OP_FILE_READ,
    OP_FILE_REPLACE, OP_FILE_SEARCH, OP_SHELL_EXECUTE, OP_SYMBOL_RENAME,
    ToolRegistry, ToolSpec,
)
from telemetry.models import AttemptSummary, TelemetryContext, ToolActivity
from telemetry.recorder import TelemetryRecorder


class ContractExecutionGuard:
    def __init__(self, contract: TaskContract, context: TelemetryContext,
                 specs: list[ToolSpec], recorder: TelemetryRecorder | None = None):
        self.contract = contract
        self.context = context
        self.recorder = recorder or TelemetryRecorder()
        self._specs = {spec.name: spec for spec in specs}
        self._registry = ToolRegistry()
        self._registry.register(specs)
        self._signatures: Counter[str] = Counter()
        self._categories: Counter[str] = Counter()
        self._started: dict[str, float] = {}
        self._call_details: dict[str, tuple[str, tuple[str, ...], str]] = {}
        self._files_read: set[str] = set()
        # Paths outside the Tasker's expected scope. Measured, never denied.
        self.unexpected_reads: list[str] = []
        self.unexpected_writes: list[str] = []
        self._output_chars = 0
        self._tool_calls = 0
        self._repeated = 0
        self._calls_without_progress = 0
        self._meaningful_progress = 0
        self._calls_before_first_edit = 0
        self._first_edit = False
        self._finished = False
        self.escalation_reason: str | None = None

    def authorize(self, call_id: str, name: str, arguments: dict) -> str | None:
        effective = self._effective_name(name, arguments)
        category = self._category(name, effective)
        normalized_arguments = self._dispatcher_arguments(name, arguments)
        paths = self._paths(normalized_arguments)
        if is_mutating_tool(name):
            paths = tuple(dict.fromkeys((*paths, *self._registry.mutation_targets(
                name, arguments))))
        elif name.endswith("execute_tool"):
            paths = tuple(dict.fromkeys((*paths, *self._dispatcher_mutation_paths(
                effective, normalized_arguments))))
        limits = self.contract.execution_limits
        if self.escalation_reason:
            return self._denial(self.escalation_reason)
        if name not in self.contract.allowed_tools:
            return self._deny(f"tool {name!r} is outside the contract")
        required_operation = self._required_operation(effective)
        if (required_operation is not None
                and required_operation not in self.contract.allowed_operations):
            return self._deny(
                f"operation {required_operation!r} is outside the contract")
        if required_operation == OP_SHELL_EXECUTE:
            check_error = self._validate_check(effective, normalized_arguments)
            if check_error:
                return self._deny(check_error)
        # Call-count budgets are intentionally not enforced; the counters below
        # are measurements for telemetry only.
        excluded = [path for path in paths if self._excluded(path)]
        if excluded:
            return self._deny("path excluded by user decision: " + ", ".join(excluded[:3]))
        # The deterministic Tasker cannot know every file a change needs, so its
        # files are expected scope, not a wall. Only genuinely unsafe paths stop.
        if category == "write":
            if self.contract.read_only:
                return self._deny("this read-only contract does not allow writes")
            if not paths:
                # Matches the fail-closed mutation journal: an unidentifiable target
                # (including .git internals, which adapters refuse) is never written.
                return self._deny("write target could not be identified")
            protected = [path for path in paths if protected_path(path)]
            if protected:
                return self._deny("protected path: " + ", ".join(protected[:3]))
        if category == "read":
            secrets = [path for path in paths if secret_path(path)]
            if secrets:
                return self._deny("secrets file: " + ", ".join(secrets[:3]))
        signature = hashlib.sha256(json.dumps(
            [effective, normalized_arguments], sort_keys=True, default=str).encode()).hexdigest()
        repeated = self._signatures[signature] > 0
        if repeated and self._signatures[signature] >= limits.repeated_calls:
            return self._deny("repeated identical tool-call limit reached")
        self._signatures[signature] += 1
        self._tool_calls += 1
        self._categories[category] += 1
        self._calls_without_progress += 1
        if repeated:
            self._repeated += 1
        if not self._first_edit and category != "write":
            self._calls_before_first_edit += 1
        if category == "read":
            self._files_read.update(paths)
        if category in {"read", "write"}:
            target = self.unexpected_writes if category == "write" else self.unexpected_reads
            target.extend(path for path in paths
                          if not self._expected(path) and path not in target)
        self._started[call_id] = time.monotonic()
        self._call_details[call_id] = (category, paths, signature)
        self.recorder.tool_started(ToolActivity(
            context=self.context, call_id=call_id, tool_name=effective,
            category=category, signature=signature, paths=paths, repeated=repeated,
            calls_before_first_edit=self._calls_before_first_edit,
        ))
        return None

    def complete(self, call_id: str, result: str) -> str:
        category, _, _ = self._call_details.pop(call_id, ("other", (), ""))
        started = self._started.pop(call_id, time.monotonic())
        success = not (result or "").lstrip().lower().startswith("error")
        progress = success and category in {"write", "test"}
        if progress:
            self._calls_without_progress = 0
            self._meaningful_progress += 1
            if category == "write":
                self._first_edit = True
        output = result or "Tool succeeded and produced no output."
        if category == "search":
            output = self._without_excluded(output)
        self._output_chars += len(output)
        self.recorder.tool_finished(
            call_id, success, len(output), int((time.monotonic() - started) * 1000),
            progress, None if success else "tool_error",
        )
        return output

    def finish(self, outcome: str, usage: dict | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        for call_id, started in tuple(self._started.items()):
            self.recorder.tool_finished(
                call_id, False, 0, int((time.monotonic() - started) * 1000),
                False, "unfinished_tool_call",
            )
        self._started.clear()
        self._call_details.clear()
        self.recorder.attempt_finished(AttemptSummary(
            context=self.context, outcome=outcome, tool_calls=self._tool_calls,
            calls_by_category=dict(self._categories),
            unique_files_read=tuple(sorted(self._files_read)),
            output_chars=self._output_chars, repeated_calls=self._repeated,
            calls_before_first_edit=self._calls_before_first_edit,
            meaningful_progress=self._meaningful_progress, usage=usage,
            escalation_reason=self.escalation_reason,
            extra={"unexpected_writes": list(self.unexpected_writes),
                   "unexpected_reads": list(self.unexpected_reads)},
        ))

    def _deny(self, reason: str) -> str:
        self.escalation_reason = reason
        return self._denial(reason)

    @staticmethod
    def _denial(reason: str) -> str:
        return "Error: TaskContract execution stopped: " + reason

    def _category(self, name: str, effective: str) -> str:
        lower = effective.lower()
        if any(token in lower for token in ("test", "run_configuration", "terminal")):
            return "test"
        if any(token in lower for token in ("search", "find_text", "find_file")):
            return "search"
        if any(token in lower for token in ("read_file", "get_file_text")):
            return "read"
        spec = self._specs.get(name)
        decision = classify_tool(spec) if spec else None
        if decision and "file_content_read" in decision.operations:
            return "read"
        if decision and ("file_search" in decision.operations
                         or "file_text_search" in decision.operations):
            return "search"
        if is_mutating_tool(effective) or is_mutating_tool(name):
            return "write"
        return "other"

    @staticmethod
    def _effective_name(name: str, arguments: dict) -> str:
        if name.endswith("execute_tool") and isinstance(arguments.get("command"), str):
            try:
                return shlex.split(arguments["command"])[0]
            except (ValueError, IndexError):
                return name
        return name

    @staticmethod
    def _paths(arguments: Any) -> tuple[str, ...]:
        found: list[str] = []
        path_keys = {"path", "filepath", "file_path", "file", "sourcepath", "targetpath"}
        def visit(value, key=""):
            if isinstance(value, dict):
                for child_key, child in value.items():
                    visit(child, child_key.lower())
            elif isinstance(value, list):
                for child in value:
                    visit(child, key)
            elif isinstance(value, str) and key in path_keys:
                parsed = PurePosixPath(value)
                if not parsed.is_absolute() and ".." not in parsed.parts:
                    found.append(value)
        visit(arguments)
        return tuple(dict.fromkeys(found))

    @staticmethod
    def _dispatcher_arguments(name: str, arguments: dict) -> dict:
        if not name.endswith("execute_tool") or not isinstance(arguments.get("command"), str):
            return arguments
        try:
            tokens = shlex.split(arguments["command"])
        except ValueError:
            return arguments
        parsed: dict[str, Any] = {}
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token.startswith("--") and index + 1 < len(tokens):
                key = token[2:]
                value: Any = tokens[index + 1]
                if key in {"limit", "offset", "maxLines", "startLine", "endLine"}:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                parsed[key] = value
                index += 2
            else:
                index += 1
        return parsed

    @staticmethod
    def _required_operation(effective: str) -> str | None:
        lower = effective.lower()
        if "terminal" in lower or "run_configuration" in lower:
            return OP_SHELL_EXECUTE
        if lower in {"read_file", "get_file_text_by_path"}:
            return OP_FILE_READ
        if "search" in lower or lower in {"find_text", "exact_search"}:
            return OP_FILE_SEARCH
        if lower in {"create_new_file", "create_new_file_with_text"}:
            return OP_FILE_CREATE
        if lower == "apply_patch":
            return OP_FILE_PATCH
        if lower in {"replace_file_text_by_path", "replace_specific_text",
                     "replace_text_in_file"}:
            return OP_FILE_REPLACE
        if lower == "rename_refactoring":
            return OP_SYMBOL_RENAME
        return None

    @staticmethod
    def _dispatcher_mutation_paths(effective: str, arguments: dict) -> tuple[str, ...]:
        lower = effective.lower()
        adapter = None
        if lower in {"create_new_file", "create_new_file_with_text"}:
            adapter = "create"
        elif lower == "apply_patch":
            adapter = "patch"
        elif lower in {"replace_file_text_by_path", "replace_specific_text",
                       "replace_text_in_file"}:
            adapter = "replace"
        elif lower == "rename_refactoring":
            adapter = "rename"
        return mutation_targets(adapter, arguments) if adapter else ()

    @staticmethod
    def _validate_check(effective: str, arguments: dict) -> str | None:
        if "run_configuration" in effective.lower():
            return None
        command = arguments.get("command")
        if not isinstance(command, str):
            return "test execution requires an explicit command"
        try:
            tokens = shlex.split(command)
        except ValueError:
            return "test command could not be parsed"
        if any(token in command for token in (";", "&&", "||", "`", "$(`", ">", "<")):
            return "test command contains unsupported shell control syntax"
        if tokens[:3] == ["uv", "run", "pytest"]:
            targets = tokens[3:]
        elif tokens[:3] and len(tokens) >= 3 and tokens[1:3] == ["-m", "pytest"]:
            targets = tokens[3:]
        elif tokens[:1] == ["pytest"]:
            targets = tokens[1:]
        else:
            return "only focused pytest checks are allowed by this contract"
        if not any(not token.startswith("-") and token not in {".", "tests"}
                   for token in targets):
            return "pytest check must name a focused file, node, or subdirectory"
        return None

    def _excluded(self, path: str) -> bool:
        return path_excluded(path, self.contract.excluded_files)

    def _without_excluded(self, output: str) -> str:
        """Remove search hits in excluded or secrets files so their content never leaks."""
        removed = 0

        def hidden(path: str) -> bool:
            path = path.strip()
            return bool(path) and (self._excluded(path) or secret_path(path))

        def mentions(value) -> bool:
            if isinstance(value, dict):
                return any(mentions(item) for item in value.values()
                           if isinstance(item, (str, dict)))
            # Matches both a bare path and "path:line: text" search output.
            return isinstance(value, str) and hidden(value.split(":", 1)[0])

        def prune(value):
            nonlocal removed
            if isinstance(value, list):
                kept = [prune(item) for item in value if not mentions(item)]
                removed += len(value) - len(kept)
                return kept
            if isinstance(value, dict):
                return {key: prune(item) for key, item in value.items()}
            return value

        try:
            filtered = json.dumps(prune(json.loads(output)))
        except (TypeError, ValueError):
            lines = output.splitlines()
            kept = [line for line in lines if not any(
                mentions(token.strip("\"'`,()[]")) for token in line.split())]
            removed = len(lines) - len(kept)
            filtered = "\n".join(kept)
        if removed:
            filtered += f"\n({removed} result(s) in excluded or secrets files were removed.)"
        return filtered

    def _expected(self, path: str) -> bool:
        candidate = PurePosixPath(path)
        return any(candidate == PurePosixPath(item)
                   for item in (*self.contract.primary_files, *self.contract.reference_files))

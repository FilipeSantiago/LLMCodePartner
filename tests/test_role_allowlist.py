"""Tests for the per-role MCP tool allowlist.

Dependency-free (stdlib `unittest`, no pytest): run with

    .venv/bin/python -m unittest tests.test_role_allowlist -v

Covers that `allowed_for` hands each agent role its own slice of the tool surface,
that an unrecognized role fails closed, and that `ToolRegistry` actually enforces the
slice it is given (the enforcement point — a spec outside the role never registers,
so no provider ever builds a handler for it).
"""
import unittest

from mcp_bridge.registry import (
    READ,
    SEARCH,
    REFACTOR,
    MUTATING,
    ROLE_CODER,
    ROLE_OPTIMIZER,
    SUPPORTED,
    WRITE,
    ToolRegistry,
    ToolSpec,
    allowed_for,
    classify_tool,
    is_mutating_tool,
    mutating_tools,
    OP_FILE_READ,
    OP_FILE_SEARCH,
    OP_FILE_CREATE,
)
from mcp_bridge.adapters import prepare_dispatcher_read


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=name, schema={"type": "object", "properties": {}})


class AllowedFor(unittest.TestCase):
    def test_optimizer_is_read_only(self):
        allowed = allowed_for(ROLE_OPTIMIZER)
        self.assertEqual(allowed, READ | SEARCH)
        self.assertFalse(allowed & WRITE, "optimizer must not hold any write tool")
        self.assertFalse(allowed & REFACTOR, "optimizer must not hold any refactor tool")

    def test_coder_holds_the_full_surface(self):
        allowed = allowed_for(ROLE_CODER)
        for group in (READ, WRITE, REFACTOR):
            self.assertTrue(group <= allowed)
        self.assertEqual(allowed, SUPPORTED)

    def test_unroled_gets_the_ceiling(self):
        # Back-compat: a caller that names no role behaves exactly as before.
        self.assertEqual(allowed_for(None), SUPPORTED)

    def test_unknown_role_fails_closed(self):
        # A typo'd role should cost capability, not safety.
        self.assertEqual(allowed_for("codre"), READ | SEARCH)
        self.assertFalse(allowed_for("codre") & WRITE)


class RegistryEnforcesTheRole(unittest.TestCase):
    ADVERTISED = [
        _spec("get_file_text_by_path"),      # READ
        _spec("list_directory_tree"),        # READ
        _spec("create_new_file"),            # WRITE
        _spec("replace_file_text_by_path"),  # WRITE
        _spec("rename_refactoring"),         # REFACTOR
    ]

    def test_optimizer_registry_drops_writes(self):
        registry = ToolRegistry(allowed_for(ROLE_OPTIMIZER))
        registry.register(self.ADVERTISED)
        self.assertEqual(
            sorted(registry.names()), ["get_file_text_by_path", "list_directory_tree"]
        )

    def test_coder_registry_keeps_everything_advertised(self):
        registry = ToolRegistry(allowed_for(ROLE_CODER))
        registry.register(self.ADVERTISED)
        self.assertEqual(
            sorted(registry.names()), sorted(s.name for s in self.ADVERTISED)
        )

    def test_unadvertised_tools_never_register(self):
        # The intersection cuts both ways: in the role's list but not advertised.
        registry = ToolRegistry(allowed_for(ROLE_CODER))
        registry.register([_spec("get_file_text_by_path")])
        self.assertEqual(registry.names(), ["get_file_text_by_path"])
        self.assertNotIn("apply_patch", registry.names())

    def test_unknown_tool_name_is_ignored(self):
        registry = ToolRegistry(allowed_for(ROLE_CODER))
        registry.register([_spec("run_terminal_command")])
        self.assertTrue(registry.is_empty())

    def test_mutating_tool_helpers_only_accept_supported_writes_or_refactors(self):
        self.assertIn("create_new_file", MUTATING)
        self.assertTrue(is_mutating_tool("apply_patch"))
        self.assertTrue(is_mutating_tool("rename_refactoring"))
        self.assertFalse(is_mutating_tool("read_file"))
        self.assertEqual(
            mutating_tools([_spec("read_file"), _spec("apply_patch"), _spec("rename_refactoring")]),
            ("apply_patch", "rename_refactoring"),
        )

    def test_invalid_schema_is_not_registered_as_a_mutating_capability(self):
        malformed = ToolSpec("apply_patch", "patch", {"type": "array"})
        self.assertFalse(classify_tool(malformed).accepted)
        registry = ToolRegistry(allowed_for(ROLE_CODER))
        registry.register([malformed])
        self.assertTrue(registry.is_empty())

    def test_artifact_operation_policy_keeps_exact_lookup_not_regex_fallback(self):
        registry = ToolRegistry(allowed_operations=frozenset({
            OP_FILE_READ, OP_FILE_SEARCH, OP_FILE_CREATE,
        }))
        registry.register([_spec("search_file"), _spec("search_regex"), _spec("read_file")])
        self.assertEqual(registry.candidates(OP_FILE_SEARCH)[0].name, "search_file")
        self.assertNotIn("search_regex", registry.names())

    def test_artifact_read_uses_advertised_dispatcher_without_default_limit(self):
        registry = ToolRegistry(allowed_operations=frozenset({OP_FILE_READ}))
        registry.register([_spec("execute_tool")])
        name, arguments = registry.prepare_call(OP_FILE_READ, {"file_path": ".codepartner/spec-index.json"})
        self.assertEqual(name, "execute_tool")
        self.assertEqual(arguments, {"command": "read_file --file_path .codepartner/spec-index.json"})
        self.assertNotIn("--limit", arguments["command"])

    def test_dispatcher_read_quotes_paths_and_validates_page_bounds(self):
        self.assertEqual(
            prepare_dispatcher_read({"file_path": "openspec/a b.json", "limit": 5000, "offset": 3}),
            {"command": "read_file --file_path 'openspec/a b.json' --limit 5000 --offset 3"},
        )
        with self.assertRaises(ValueError):
            prepare_dispatcher_read({"file_path": "openspec/x", "limit": 5001})


if __name__ == "__main__":
    unittest.main()

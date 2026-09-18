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
    REFACTOR,
    ROLE_CODER,
    ROLE_OPTIMIZER,
    SUPPORTED,
    WRITE,
    ToolRegistry,
    ToolSpec,
    allowed_for,
)


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=name, schema={"type": "object", "properties": {}})


class AllowedFor(unittest.TestCase):
    def test_optimizer_is_read_only(self):
        allowed = allowed_for(ROLE_OPTIMIZER)
        self.assertEqual(allowed, READ)
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
        self.assertEqual(allowed_for("codre"), READ)
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


if __name__ == "__main__":
    unittest.main()

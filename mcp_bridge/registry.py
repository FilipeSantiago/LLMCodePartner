"""Per-run MCP classification, policy, and operation selection."""
from dataclasses import dataclass
import logging
from typing import Any

from logger.diagnostic import debug

log = logging.getLogger("codepartner.bridge.registry")
CAP_READ, CAP_SEARCH, CAP_WRITE, CAP_REFACTOR, CAP_EXECUTE = "read", "search", "write", "refactor", "execute"
DOMAIN_SOURCE, DOMAIN_PROJECT, DOMAIN_NOTEBOOK = "source_files", "project_state", "notebook"
OP_FILE_READ, OP_FILE_SEARCH = "file_content_read", "file_search"
OP_FILE_CREATE, OP_FILE_REPLACE, OP_FILE_PATCH, OP_FILE_DELETE = "file_create", "file_replace", "file_patch", "file_delete"
OP_SYMBOL_RENAME, OP_FILE_FORMAT, OP_SHELL_EXECUTE = "symbol_rename", "file_format", "shell_execute"

@dataclass(frozen=True)
class Definition:
    aliases: frozenset[str]; capabilities: frozenset[str]; domain: str; operations: frozenset[str]; adapter: str
def _d(a, c, domain, ops, adapter): return Definition(frozenset(a), frozenset(c), domain, frozenset(ops), adapter)

DEFINITIONS = (
    _d(("read_file",), (CAP_READ,), DOMAIN_SOURCE, (OP_FILE_READ,), "native_read"),
    _d(("get_file_text_by_path",), (CAP_READ,), DOMAIN_SOURCE, (OP_FILE_READ,), "legacy_read"),
    _d(("search_file",), (CAP_SEARCH,), DOMAIN_SOURCE, (OP_FILE_SEARCH,), "file_search"),
    # Content/regex search can find source text, but cannot establish that a
    # particular catalog path exists.  Only the file-listing adapter provides
    # the exact ``file_search`` operation.
    _d(("find_text", "exact_search", "search_in_files_by_text", "search_regex", "search_text"), (CAP_READ, CAP_SEARCH), DOMAIN_SOURCE, ("file_text_search",), "search"),
    _d(("list_files_in_folder", "list_directory_tree", "get_symbol_info", "get_file_problems", "get_all_open_file_paths", "get_git_diff_all", "get_vcs_log", "open_commit", "documentation_search", "get_project_dependencies", "get_project_modules", "find_usages"), (CAP_READ,), DOMAIN_PROJECT, ("project_inspect",), "inspect"),
    _d(("create_new_file", "create_new_file_with_text"), (CAP_WRITE,), DOMAIN_SOURCE, (OP_FILE_CREATE,), "create"),
    _d(("replace_file_text_by_path", "replace_specific_text", "replace_text_in_file"), (CAP_WRITE,), DOMAIN_SOURCE, (OP_FILE_REPLACE,), "replace"),
    _d(("apply_patch",), (CAP_WRITE,), DOMAIN_SOURCE, (OP_FILE_PATCH,), "patch"),
    _d(("delete_file", "delete_file_by_path"), (CAP_WRITE,), DOMAIN_SOURCE, (OP_FILE_DELETE,), "delete"),
    _d(("rename_refactoring",), (CAP_REFACTOR,), DOMAIN_SOURCE, (OP_SYMBOL_RENAME,), "rename"),
    _d(("reformat_file",), (CAP_REFACTOR,), DOMAIN_SOURCE, (OP_FILE_FORMAT,), "format"),
    # A dispatcher is transport, never a blanket source capability.  Discovery
    # can only add operations to this request after its own bridge call succeeds.
    _d(("execute_tool",), (CAP_EXECUTE,), DOMAIN_PROJECT, (), "dispatcher"),
    _d(("read_notebook",), (CAP_READ,), DOMAIN_NOTEBOOK, ("notebook_content_read",), "notebook_read"),
    _d(("edit_notebook",), (CAP_WRITE,), DOMAIN_NOTEBOOK, ("notebook_replace", "notebook_patch"), "notebook_edit"),
)

PREFIXES = ("mcp__pycharm__", "mcp__jetbrains__")
READ = frozenset(a for d in DEFINITIONS if CAP_READ in d.capabilities for a in d.aliases)
SEARCH = frozenset(a for d in DEFINITIONS if CAP_SEARCH in d.capabilities for a in d.aliases)
WRITE = frozenset(a for d in DEFINITIONS if CAP_WRITE in d.capabilities for a in d.aliases)
REFACTOR = frozenset(a for d in DEFINITIONS if CAP_REFACTOR in d.capabilities for a in d.aliases)
MUTATING, SUPPORTED = WRITE | REFACTOR, READ | SEARCH | WRITE | REFACTOR | frozenset(a for d in DEFINITIONS if CAP_EXECUTE in d.capabilities for a in d.aliases)
ROLE_OPTIMIZER, ROLE_CODER = "optimizer", "coder"
ROLES = {ROLE_OPTIMIZER: READ | SEARCH, ROLE_CODER: SUPPORTED}
def allowed_for(role: str | None) -> frozenset[str]: return SUPPORTED if role is None else ROLES.get(role, READ | SEARCH)

@dataclass
class ToolSpec:
    name: str; description: str; schema: Any
@dataclass(frozen=True)
class CapabilityDecision:
    name: str; capability: str | None; accepted: bool; reason: str; domain: str | None = None; operations: frozenset[str] = frozenset(); executable: bool = False
def _definition(name: str) -> Definition | None:
    bare = next((name[len(p):] for p in PREFIXES if name.startswith(p)), name)
    return next((d for d in DEFINITIONS if bare in d.aliases), None)
def classify_tool(spec: ToolSpec) -> CapabilityDecision:
    d = _definition(spec.name)
    if d is None: return CapabilityDecision(spec.name, None, False, "unknown_alias")
    cap = next(iter(d.capabilities))
    if not isinstance(spec.schema, dict): return CapabilityDecision(spec.name, cap, False, "schema_missing_or_not_object", d.domain, d.operations)
    if spec.schema.get("type") not in (None, "object"): return CapabilityDecision(spec.name, cap, False, "schema_is_not_object", d.domain, d.operations)
    if "properties" in spec.schema and not isinstance(spec.schema["properties"], dict): return CapabilityDecision(spec.name, cap, False, "schema_properties_not_object", d.domain, d.operations)
    from mcp_bridge.adapters import validate_adapter
    check = validate_adapter(d.adapter, spec.schema)
    return CapabilityDecision(spec.name, cap, True, check.reason, d.domain, d.operations, check.ok)

@dataclass
class _Entry: spec: ToolSpec; decision: CapabilityDecision; definition: Definition
class ToolRegistry:
    """Request-local catalog; classification never makes a tool callable alone."""
    def __init__(self, allowed: frozenset[str] | None = None, allowed_operations: frozenset[str] | None = None):
        self._allowed = SUPPORTED if allowed is None else allowed
        self._allowed_operations = allowed_operations
        self._entries = {}
        self.rejections = {}
    def _allowed_name(self, name): return name in self._allowed or any(name.startswith(p) and name[len(p):] in self._allowed for p in PREFIXES)
    def register(self, specs):
        for spec in specs:
            decision, definition = classify_tool(spec), _definition(spec.name)
            if not self._allowed_name(spec.name): self.rejections[spec.name] = "excluded_by_role_policy"
            elif (self._allowed_operations is not None
                  and not decision.operations.intersection(self._allowed_operations)
                  # The dispatcher is admitted only as the transport for an
                  # explicitly requested source-file read; it gains no general
                  # operation capability from its name.
                  and not (definition is not None and definition.adapter == "dispatcher"
                           and OP_FILE_READ in self._allowed_operations)):
                self.rejections[spec.name] = "excluded_by_operation_policy"
            elif not decision.accepted or not decision.executable or definition is None: self.rejections[spec.name] = decision.reason
            elif spec.name not in self._entries: self._entries[spec.name] = _Entry(spec, decision, definition)
            debug(log, "registry.registration_considered", tool=vars(spec), decision=vars(decision), rejection=self.rejections.get(spec.name))
        return self.tools()
    def tools(self): return [e.spec for e in self._entries.values()]
    def names(self): return list(self._entries)
    def is_empty(self): return not self._entries
    def candidates(self, operation, domain=None): return [e.spec for e in self._entries.values() if operation in e.definition.operations and (domain is None or domain == e.definition.domain)]
    def require(self, operation, domain=DOMAIN_SOURCE):
        choices = self.candidates(operation, domain)
        if choices: return choices[0]
        rejected = ", ".join(f"{n}: {r}" for n, r in self.rejections.items()) or "none"
        raise LookupError(f"required operation {operation!r} in domain {domain!r} unavailable; advertised: {', '.join(self.names()) or 'none'}; rejected: {rejected}")
    def prepare_call(self, operation, arguments, domain=DOMAIN_SOURCE):
        if operation == OP_FILE_READ and domain == DOMAIN_SOURCE:
            dispatcher = next((entry for entry in self._entries.values()
                               if entry.definition.adapter == "dispatcher"), None)
            if dispatcher is not None:
                from mcp_bridge.adapters import prepare_dispatcher_read
                return dispatcher.spec.name, prepare_dispatcher_read(arguments)
        spec = self.require(operation, domain); entry = self._entries[spec.name]
        from mcp_bridge.adapters import prepare_arguments
        return spec.name, prepare_arguments(entry.definition.adapter, spec.schema, arguments)
    def prepare_advertised_call(self, name, arguments):
        entry = self._entries.get(name)
        if entry is None: raise LookupError(f"tool {name!r} is not executable in this request")
        from mcp_bridge.adapters import prepare_arguments
        return name, prepare_arguments(entry.definition.adapter, entry.spec.schema, arguments)
def source_mutation_decisions(specs):
    r = ToolRegistry(); r.register(specs)
    ops = {OP_FILE_CREATE, OP_FILE_REPLACE, OP_FILE_PATCH, OP_FILE_DELETE, OP_SYMBOL_RENAME, OP_FILE_FORMAT}
    return tuple(e.decision for e in r._entries.values() if e.definition.domain == DOMAIN_SOURCE and e.definition.operations & ops)
def has_existing_source_editor(specs):
    r = ToolRegistry(); r.register(specs)
    return bool(r.candidates(OP_FILE_REPLACE, DOMAIN_SOURCE) or r.candidates(OP_FILE_PATCH, DOMAIN_SOURCE))
def mutating_tools(specs): return tuple(d.name for d in source_mutation_decisions(specs))
def is_mutating_tool(name):
    d = _definition(name); return bool(d and d.domain == DOMAIN_SOURCE and d.operations & {OP_FILE_CREATE, OP_FILE_REPLACE, OP_FILE_PATCH, OP_FILE_DELETE, OP_SYMBOL_RENAME, OP_FILE_FORMAT})
def operations_for(name):
    d = _definition(name)
    return d.operations if d else frozenset()

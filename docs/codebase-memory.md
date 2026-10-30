# Optional Codebase Memory for planning and review

Code Partner can use an existing Codebase Memory v0.11.0 or v0.12.x index before OpenSpec
planning and Work Package planning, then refresh it after a passing WP review.
The project remains accessible through
IntelliJ MCP for exact source verification. Neither Code Partner nor the KB bridge
mounts the project repository.

This integration is off by default. Ordinary `docker compose up` and existing
`.env` settings continue to work without a KB installation.

## Find the host paths and project ID

Run `command -v codebase-memory-mcp` and `codebase-memory-mcp --version` on the
host. The binary must be a Linux v0.11.0 or v0.12.x executable for the container architecture.
Codebase Memory stores SQLite indexes in `CBM_CACHE_DIR`, defaulting to
`~/.cache/codebase-memory-mcp`; check your host environment for an override.
Confirm that the chosen directory already contains the project's `.db` file.
Run `codebase-memory-mcp cli --quiet list_projects --format json` and copy the
exact `name` of the intended indexed project. Codebase Memory v0.11.0 uses
per-project locks for graph mutations; if host/container coordination reports a
conflict, the refresh fails without changing the accepted review.

Set the following values in `.env`, using absolute host paths and the owner/group
of the storage directory (obtain the IDs with `stat -c '%u %g' /path/to/storage`):

```dotenv
KB_ENABLED=true
KB_PROVIDER=codebase-memory
KB_BINARY_PATH=/absolute/host/path/to/codebase-memory-mcp
KB_DATA_PATH=/absolute/host/path/to/codebase-memory-mcp-storage
KB_UID=1000
KB_GID=1000
```

Add `.codepartner/config.yaml` to the IDE project:

```yaml
project:
  id: music-generation
  knowledge_base:
    provider: codebase-memory
    project_id: the-exact-name-from-list_projects
```

Code Partner discovers this file through IntelliJ MCP and binds the exact KB
project before planning. `project.id` is a human-readable project label; existing
Code Partner job identity remains unchanged. A legacy `knowledge_base` entry in
`.codepartner/ide_mcp.json` remains supported when `config.yaml` is absent. If
both are present, the YAML mapping wins. The normal IDE connection file remains:

```json
{
  "type": "streamable-http",
  "url": "http://127.0.0.1:64462/stream",
  "headers": {
    "IJ_MCP_SERVER_PROJECT_PATH": "/absolute/path/to/the/project"
  }
}
```

Start the optional sidecar with
`docker compose -f compose.yaml -f compose.kb.yaml up --build -d`.
The bridge uses a shared Unix socket at `/run/kb-bridge/mcp.sock` and publishes
no host port. A missing binary or storage path is rejected; Compose will not
create an empty host directory. The KB storage bind mount is read-write because
Codebase Memory v0.11.0 starts a daemon that writes coordination files, logs,
and SQLite state even when the exposed tools are read-only. Its existing index
survives container recreation in the host directory.

After a Reviewer records a passing WP review, Code Partner asks IntelliJ MCP to
run the host-installed Codebase Memory CLI. It checks the indexed project's
`root_path` against the IDE project path before indexing, then verifies the
project's index status through MCP. The IDE must expose a terminal tool, and
`KB_BINARY_PATH` and `KB_DATA_PATH` must also be valid host paths. A failed or
timed-out refresh is logged and recorded without changing the accepted review.
Duplicate acceptance delivery does not launch another refresh for the same job
and WP. For manual host indexing, close other CBM sessions if the provider
reports a coordination conflict. The bridge exposes Codebase Memory's
native `scout` tool profile; both planning stages receive bounded graph context,
and the WP planner, coder, and reviewer can use project-bound graph and outline
tools on demand. Coder and reviewer prompts do not receive a KB prefetch; they
start from the approved Work Package plan. Agents can inspect a candidate file's
indexed outline, decide whether it is relevant, then read the needed current
source regions through IntelliJ MCP. The approved plan carries compact file,
symbol, and line-range discovery hints to coders and reviewers; hints are not
source evidence and may become stale after edits. Graph results may lag current
edits, so coders verify implementation details with IntelliJ MCP source reads
and reviewers base decisions on current source and captured execution evidence.
Small, recently modified, and non-code files may still need broader reads.
**Bulk queries.** When the KB server has `query_graph`, it is offered to the planner as
`kb_query_graph`, and the WP prefetch outlines every file holding a named symbol in one
`query_graph` call (falling back to one `get_file_outline` per file). The prompts tell
every agent to batch lookups: one `search_graph` with a `name_pattern` alternation or a
`semantic_query` array, one `query_graph` for several files, never one call per file.

**Refresh before planning.** Every new planning session reindexes the project first,
whether or not recorded edits are newer than the index; coding keeps the conditional
catch-up.

**KB-first rule** (`PromptBuilder.read_pipeline` in `agent/prompt/base.py`; see
[prompts](prompts.md)). Coder and reviewer prompts, including their subagents, get the
read pipeline: before the first read of each code file they consult Codebase Memory,
then read only the relevant current-source ranges. The planner has a separate single
read pipeline: query KB, choose minimal files and ranges, delegate source inspection to
its reader when offered, and plan from the cited brief. Direct bounded planner reads are
the disclosed fallback when the reader is unavailable or fails.

This is a prompt rule only. Nothing intercepts, replaces or refuses a read or a search,
so native execution and subagents work with it. Compliance is measured, not
enforced: native runs record `kb_calls` next to their reads in `native_tool_metrics`, and
the planner's advise-mode `discovery` telemetry still records retrieval levels and may
append a one-line advisory to a result.

**Index catch-up.** Before WP planning and at the start of each implementation job,
Code Partner compares the KB `indexed_at` with the project's applied or verified
mutations. If accepted edits are newer, it runs the same host reindex as the
post-review refresh. This covers refreshes that never ran. A failed catch-up is
logged (`kb_catch_up_failed`) and never blocks the run.

KB results never override current files.

`kb_get_code_snippet` returns the indexed source of a symbol. Like every KB result it
may lag current edits, so agents verify the lines they rely on in the current files
before editing or deciding. Reviewer evidence and inspected-file counts use only
current file reads, not KB outlines or graph results.

The WP planner also records discovery telemetry per turn: retrieval level,
direct or escalated full reads, the evidence strength of each read path, and
whether the KB was used or the planner fell back to IDE-only discovery
(`discovery`, see docs/wp-planner.md).
Existing telemetry records outline lookups, whole and ranged source reads,
repeated reads, source response bytes, and provider-reported input tokens when
available. Source bytes are measured separately from token usage.
When a task contract excludes files, KB tools and the policy are omitted from
that task and its Work Package review. A stale index is an architectural lead;
all agents verify current implementation details through IntelliJ MCP.

To disable the feature, set `KB_ENABLED=false`, omit the Compose override, and
remove the project `knowledge_base` mapping. If KB is unavailable, planners use
their existing IntelliJ MCP exploration path. An explicitly configured KB
project absent from `list_projects` is reported as a configuration error.

The KB bridge uses the read-only `analysis` tool profile so planners can use
`query_graph` for multiple file declarations in one request. Startup verifies
required tools; the planner names only the KB tools actually offered.

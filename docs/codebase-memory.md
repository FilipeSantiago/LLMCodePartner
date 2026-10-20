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
**Discovery policy.** `discovery.policy` in `config/model-routing.yaml` decides
which roles get the KB-first gates described below:
- **`selective`.** Only the WP planner gets the gates. Coders and
  reviewers start from the task, the approved plan context and current source.
  They query the KB on demand for architecture, dependency or impact questions.
  Their KB calls are still recorded, but nothing is intercepted and no advisories
  are added.
- **`kb_first` (configured).** All three roles use controlled IDE tools, even when
  native execution is configured. The first read of each file obtains its KB outline
  without returning source content. Subsequent reads require a line range of at most
  `discovery.outline_gate_lines` lines. When an outline is unavailable, a successful
  file-scoped search is required before a bounded read. This applies to code, tests,
  documentation, configuration, and notebooks. The planner always follows these rules.
  Coders and reviewers use the KB to locate code, not to stand in for files the KB
  cannot know. Their first read of such a file returns current source without an
  outline, and later reads need a line range. These files are plan files (the approved
  plan's shared and task `relevant_files`), files changed in the run or the job,
  files changed since the index (`recent_mutations`), and code files whose outline is
  empty. An empty outline's read carries a stale-index note. After an edit or a
  terminal command, files already located are read directly instead of being outlined
  again. `read_enforcement` counts `direct_plan`, `direct_changed` and
  `direct_stale_index`.

**Index catch-up.** Before WP planning and at the start of each implementation job,
Code Partner compares the KB `indexed_at` with the project's applied or verified
mutations. If accepted edits are newer, it runs the same host reindex as the
post-review refresh. This covers refreshes that never ran. A failed catch-up is
logged (`kb_catch_up_failed`) and never blocks the run.

KB results never override current files.

**Native agents** (docs/native-execution.md) remain available under `selective` or
when discovery is disabled. Their searches use their own tools, which the IDE gates
never see:
- Under `selective`, coder and reviewer prompts give KB-on-demand guidance. The
  planner remains KB-first for discovery questions, while source reads are not
  strictly mediated in this mode.
- Native **Claude** planner runs also get a one-time nudge (`providers/kb_gate.py`,
  a Claude Code `PreToolUse` hook):
  - until the agent's first `kb_*` call, a broad identifier-only search is not run;
    instead the agent is told to call `kb_search_graph`/`kb_trace_path`. Such a search
    means a Grep over the project or a directory, or a simple `grep -r`/`rg`/`git grep`
    command;
  - the first repository-wide Glob gets the same treatment, pointing to
    `kb_get_architecture`;
  - repeating the identical call runs it;
  - at most `discovery.search_gate_max` calls are advised per run;
  - searches inside a single file, literal, import and usage searches, tests/docs/config
    paths, pipelines, reads and edits always run.
- Native **Codex** runs search through their shell, which CodePartner cannot intercept,
  so they get the guidance only.
- Native metrics record `kb_calls` and, for Claude, `kb_gate: {advised, repeated_anyway,
  kb_used}` in `native_tool_metrics`.

When the gates apply to a role and KB tools are offered, the following happens. A bare
definition-style repository text search first shows indexed graph definitions
under a conspicuous "TEXT SEARCH NOT EXECUTED" header; repeating the identical
search executes it unchanged. Usage, import, literal, test, documentation and
configuration searches are not intercepted. A broad read of a large indexed
code file first returns its outline; repeating the read returns IDE source.
Both results are navigation hints. Changed reviewer files, files edited in the
current run and known post-index mutations bypass the outline gate.

The kb_get_code_snippet tool resolves a symbol's indexed path and line range;
Code Partner reads that range through the current IDE and verifies the
declaration and Python enclosing scope before returning source. If the location
is stale, it searches the current file for a unique verified relocation. It
never mounts the repository into the KB bridge. For KB-assisted coder edits,
the replaced region must first have been read through the IDE; a verified
snippet read counts. Reviewer evidence and inspected-file counts use only
current IDE reads, not KB outlines or graph results.

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

# Agent Tool Interface

Claude and Codex coder runs share the same interface at `mcp_bridge.bridge.call_tool`.
The providers still use their existing in-process and HTTP MCP transports. OpenSpec
artifact reads, mutation snapshots, and Reviewer tool calls keep their existing
trusted paths; only model-facing coder calls are shaped.

## Observations

`codepartner_read_file(path, start_line?, end_line?, refresh?)` returns the whole
file by default, or only the requested inclusive line range. The IDE reader accepts
at most 5,000 lines per request and may return head/tail portions for longer files;
the interface assembles distinct requests and verifies contiguous line numbers.
An incomplete or denied read produces an explicit error, not a false complete result.

Repository searches return all IDE-provided locations unless the caller supplies a limit.
`codepartner_search_in_file(path, query)` restricts discovery to one file when
the IDE supports path filters. Raw authorized results can be retrieved through
`codepartner_get_observation(observation_id, offset?)`. The special observation
ID `index` lists observed ranges and edits. Observations are retained for the run
without a count or byte cap; an ID from another run is unavailable.

Tool responses state completion, failure, truncation, and available expansion.
Commands expose an exit code when the IDE reports one. Unknown exit codes remain
unknown. Test output is reduced to totals and a small number of failure lines;
the authorized original output can be retrieved if more detail is needed.

Repeated reads reuse the run-local result without another IDE request. Pass
`refresh=true` to force a new read; mutations clear cached reads. The IDE does
not expose a reliable file revision token, so external edits require refresh.
Searches are compared after the IDE responds again.

## Mutations and boundaries

The mutation journal captures complete preimages and postimages independently
of the coder's bounded observations. Verified postimages undergo Python or JSON
parser validation before the edit result reaches the coder. Python files may
also receive a bounded IDE inspection result. Parser registration is in
`mcp_bridge.edit_validation.VALIDATORS`. Broad test suites are not run after
each edit.

The coder cannot directly read `.env` files, `.git`, CodePartner internals,
explicitly excluded paths, absolute paths, or parent traversal paths. `.env.example`
remains readable. Read-only task modes deny edits. Codex's built-in shell is
disabled so its reads pass through MCP. The IDE terminal dispatcher accepts a
small set of test and static-check commands; arbitrary shell commands are denied.
OpenSpec may mention protected files without granting the coder access to them.

The current IDE protocol does not supply canonical resolved paths or an isolated
test-runner filesystem. On deployments where the project is not mounted in the
CodePartner process, the bridge cannot independently resolve symlinks inside
the IDE project or guarantee that a permitted test script never opens a secret.
Those two guarantees require support from the IDE-side runner. The interface
denies direct path escapes and shell commands that explicitly name protected
files, and filters protected search and listing results.

## Measurement

`telemetry_aci_attempts` records one row per coder run, including agent calls,
backend calls, returned lines, visible and backend output bytes, duplicate
calls, time to first edit, provider usage, and available cost. It never stores
source text or complete tool results. `TelemetryAggregation.aci_comparison()`
groups rows by task and observation mode.

`CODEPARTNER_ACI_ENABLED=false` switches file reads to the legacy observation
shape for a measured baseline while retaining the guard. Use identical tasks,
models, IDE projects, and starting revisions for baseline and compact runs.
Check completion and Reviewer verdicts alongside tokens and cost. The current
Claude SDK and Codex CLI own their internal conversational histories; the bridge
can limit new observations but cannot replace older model messages. There is no
claim of active-history compaction until the provider runtimes expose a supported
history hook and measurements show a reduction in actual model context.

# Context compression

Coder tool results can optionally be compressed before they reach Claude or
Codex. The original always stays retrievable through
`codepartner_get_observation`, and every observation records where it came from
and what was done to it. Compression is off by default
(`CODEPARTNER_COMPRESS_ENABLED=false`).

The OpenSpec change `openspec/changes/add-context-compression/` holds the
requirements, the design decisions and the upstream spike results.

## Where it runs

```
bridge.call_tool → AgentToolInterface.execute → _execute (permissions, shaping, observation)
                                             → _compress_delivery (compression, provenance)
```

Compression runs after the existing permission checks and shaping, for both
providers. `bridge.py` routing, the tool registry and permissions are unchanged.
When compression is disabled, results are byte-identical to the uncompressed
interface.

## What is compressed

| Result | Treatment |
|---|---|
| File reads, edits | Never compressed. In compact mode a path-only `codepartner_read_file` call previews up to 200 lines; `full_file=true` requests the whole file. A large identical repeat read in the same run returns a reference to the original observation; refresh forces a new read. |
| Failed tools, denied calls | Never compressed. |
| Error output (traceback, `FAILED`, `ValueError:`, `error[E…]:`, non-zero exit) | `conservative` only, which removes ANSI codes and nothing else |
| JSON | Accepted only if it parses to the same value as the original |
| Other search, listing, command and generic results ≥ `CODEPARTNER_COMPRESS_MIN_BYTES` | The configured mode |

A compressed result is replaced by the original when any of these checks fail:

- an identifier is lost: a path, `path:line`, pytest node id, `File "…", line N` frame, `call_*` id or `obs-N` id
- an error-like line is lost
- the JSON value changed
- the output is empty
- the output saves less than `CODEPARTNER_COMPRESS_MIN_SAVING`

The original is also used when the compressor fails or times out.

These checks are a minimum. They do not prove a compressed result is still
useful: a traceback depends on surrounding lines that match no pattern. That is
why error output stays `conservative`, and why `balanced` is judged on task
outcomes (see Rollout).

A compressed result keeps its status header and adds one line:

```
TOOL COMPLETE
tool=get_symbol_info
truncated=false
compressed=balanced bytes=54491→37683 sha256=1f0c…; original: codepartner_get_observation(observation_id='obs-7')
{"results":[…]}
```

When a later large result has the same content as an earlier one that is still
current, it is replaced by `IDENTICAL RESULT … obs-N`.
Large exact repeat file reads use `IDENTICAL READ RESULT … obs-N` in the same run.
The first requested range is returned exactly; a path-only read returns up to
200 lines, clearly marks partial output, and never invents a total line count.
The original IDE output for that request is retrievable, but may itself be only
a window; request further ranges or `full_file=true` for more source. An edit or
`refresh=true` invalidates the repeat reference and marks
the earlier observation stale. The ACI attempt records `read_reference_hits`,
`read_reference_bytes_saved`, and `compress_attempts` alongside the existing
compression outcome counters.

## Compressor isolation

Only context-compress's output pipeline (`compressOutput`, pinned at
`2026.8.3`) is used. Its MCP server tools are never exposed: `execute`,
`execute_file`, `batch_execute` and `fetch_and_index` run code, read files or
use the network. Its `auto` mode is never used either, because it sends samples
to an LLM. `infra/compress-bridge/bridge.mjs` accepts only
`{op: "compress", mode: "conservative"|"balanced", text}` and `{op: "version"}`.
It refuses command hints, which turn on upstream filters that dropped all pytest
tracebacks in the spike. It also passes an explicit size cap, so upstream never
truncates.

| Layer | Sidecar (`compose.compress.yaml`) | Local backend |
|---|---|---|
| Network | `network_mode: none`; `fetch`/`net.connect` stubbed | `fetch`/`net.connect` stubbed |
| Files | Read-only rootfs; no project, home or `.env` mounts; reads limited to `/opt/compress-bridge` | Reads limited to `infra/compress-bridge` |
| Processes | Node `--permission` denies child processes, workers and addons | Same |
| Secrets | No `env_file`; empty environment | Empty environment except `PATH` |
| Privileges | uid/gid 10001, `cap_drop: ALL`, `no-new-privileges`, socket mode 0770 | Code Partner's user |

The optional Compose overlay adds GID 10001 as a supplemental group to Code
Partner. This allows a Code Partner UID/GID such as 1000:1000 to connect to the
sidecar's 0770 socket without broadening its permissions. Recreate the Code Partner
container after changing the overlay; an already running process keeps its old
supplemental groups.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CODEPARTNER_COMPRESS_ENABLED` | `false` | Master switch. It also requires `CODEPARTNER_ACI_ENABLED=true`. |
| `CODEPARTNER_COMPRESS_BACKEND` | `sidecar` | `sidecar`, `local` or `noop` |
| `CODEPARTNER_COMPRESS_SOCKET` | `/run/compress-bridge/compress.sock` | Sidecar socket |
| `CODEPARTNER_COMPRESS_BRIDGE_DIR` | `infra/compress-bridge` | Local backend directory. Run `npm ci --ignore-scripts` there first. |
| `CODEPARTNER_COMPRESS_MODE` | `conservative` | `conservative` or `balanced`. Any other value disables compression. |
| `CODEPARTNER_COMPRESS_MIN_BYTES` | `4096` | Smaller results are never sent to the compressor |
| `CODEPARTNER_COMPRESS_MIN_SAVING` | `0.05` | Below this saving ratio, the original is kept |
| `CODEPARTNER_COMPRESS_TIMEOUT_MS` | `1500` | Per request. On timeout the original is delivered. |
| `CODEPARTNER_COMPRESS_CATEGORIES` | `command,listing,other,search` | Eligible result categories |
| `CODEPARTNER_COMPRESS_PERSIST` | `false` | Also store the original text, which is searchable in SQLite |
| `CODEPARTNER_COMPRESS_PERSIST_DAYS` | `7` | Retention for persisted text |

To use the sidecar: `docker compose -f compose.yaml -f compose.compress.yaml up --build`.

## Provenance and retrieval

Each observation from a run with compression enabled is written to
`context_observations` in the routing database when the run ends. Each row holds:

- run, job, task and attempt ids, and the observation id
- the tool, its category and the resources it touched
- hashes of the arguments, the original content and the delivered text
- the mode, compressor version, byte counts and estimated token counts
- whether compression applied, the reason, `duplicate_of` and `stale`

Original text goes to `context_observation_text` (FTS5) only when
`CODEPARTNER_COMPRESS_PERSIST=true`, and only for compression-eligible
results. Whole-file reads and edits are never persisted. That keeps the default rule that
telemetry stores no source text.

Retrieval is limited to the run that produced the observation:

- `codepartner_get_observation(observation_id="obs-N")` returns the original.
- `codepartner_get_observation(observation_id="search", query="KeyError")` runs a
  BM25 search over this run's observations.
- An id from another run returns `OBSERVATION UNAVAILABLE`.
- After an edit, observations whose resources include an edited file are marked
  stale and reload with a `STALE:` header. They are also no longer used as
  duplicate targets.

Memoized results are keyed by content hash, backend, compressor version, mode,
command hint (always empty), category and effective config. Within a process, a
repeated key returns identical bytes. Provider history is never rewritten. Only
new results are compressed, so provider prompt caching of earlier turns is
unaffected.

## Measurement

`telemetry_aci_attempts.metrics_json` records, for each attempt:

- `first_source_read_bytes`, `bounded_preview_reads`, `bounded_preview_bytes`,
  `preview_followup_reads`, `preview_observation_expansions`,
  `preview_observation_bytes`, `explicit_full_file_reads`, `explicit_full_file_bytes`,
  `explicit_range_reads`, and `explicit_range_bytes`
- `compression_mode`
- `compress_applied`, `compress_fallbacks`, `compress_guard_rejections`
- `compress_bytes_before` and `compress_bytes_after`
- `dedup_hits` and `est_tokens_saved`

`TelemetryAggregation.aci_comparison(task_ids)` groups attempts by mode, for
example `compact` versus `compact+compress:balanced`, and reports two blocks:

- `estimated`: the compressor's own numbers (bytes/4). These are not provider tokens.
- `actual`: provider-reported usage, normalized by `normalize_usage`. Claude's
  `input_tokens` excludes cache reads and writes, while Codex's includes
  `cached_input_tokens`. Both are converted to
  `uncached_input_tokens`, `cache_read_tokens`, `cache_write_tokens` and
  `output_tokens`.

`cache_read_ratio` is a diagnostic only. Smaller tool results can lower it even
when total cost falls.

### Offline benchmark

```
uv run python scripts/bench_compression.py               # corpus generated from this repo
uv run python scripts/bench_compression.py --corpus DIR  # recorded outputs
```

Results on the generated corpus, with 3 repetitions per sample (estimated tokens):

| Sample | conservative | balanced | Notes |
|---|---|---|---|
| Coloured test-runner output (12 KB) | −22% | −22% | ANSI codes only |
| Structured JSON (54 KB) | 0% | −31% | Minified; parse-equal |
| Heartbeat log (144 KB) | 0% | −99.9% | Template folding |
| Log with a traceback (144 KB) | 0% | 0% | Forced to conservative, so kept intact |
| `git grep` search, `git log` | 0% | 0% | The upstream pipeline has nothing to remove |
| **Total** | **−0.6%** | **−35%** | Dominated by the synthetic heartbeat log |

Every sample gave byte-identical output across repetitions. The totals depend
entirely on the corpus: real coder traffic is mostly reads, which are excluded,
and already-shaped searches and command summaries. Expect a much smaller real
effect, and measure it with the online A/B.

## Rollout

1. **Baseline:** compression off, fixed task set.
2. **Stage A:** `CODEPARTNER_COMPRESS_MODE=conservative`.
3. **Stage B:** `balanced`. Error output stays conservative automatically.

Run every stage on the same tasks, models, project revision and IDE, for both
Claude and Codex. Move to the next stage only if:

- task success rate and Reviewer verdicts do not regress, and
- actual uncached input tokens and cost per completed task go down.

Report cache metrics alongside, as a diagnostic. Check `compress_guard_rejections`
and `compress_fallbacks` for unexpected rates.

## Known limits

- Log template folding replaces distinct values (ids, line numbers) in folded
  lines. It is accepted only when no error-like line is folded, so a log with a
  single traceback gets no compression at all.
- Error detection is pattern-based. Unusual failure formats may be compressed
  with the configured mode, although the identifier and error-line checks
  still apply.
- The memo is per process. A sidecar restart with a new version is detected,
  and those requests fail open.

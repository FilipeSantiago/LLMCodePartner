# Design: Provenance-aware context compression

## Context

Every Claude and Codex coder tool call already passes through `bridge.call_tool`, then `AgentToolInterface.execute` (`mcp_bridge/agent_tools.py`). That interface already shapes results, stores the raw text (`_remember`), serves authorized run-scoped retrieval (`codepartner_get_observation`), de-duplicates repeated reads and searches, redacts protected paths, and records per-attempt metrics in `telemetry_aci_attempts`. This change adds one step to that pipeline and introduces no new routing.

## Upstream spike findings (WP0, context-compress 2026.8.3)

Method: the corpus was eight real outputs:
- `pytest -q` (ANSI colour) and `pytest --tb=long -rA` failures
- `unittest -v` failures
- `git log --stat`
- two `grep -rn` sweeps of this repo
- a 168 KB noisy log with embedded tracebacks
- a 54 KB JSON result with paths, symbols and call ids

Each sample went through 3 modes × {no hint, realistic command hint}, 20 repetitions per process across 3 separate processes, with `env -i` and an empty `HOME` and cwd. Results were checked for byte-identical output, identifier retention (paths, `path:line`, pytest node ids, call ids, symbols, commit SHAs), error-line retention, and retention of traceback context that matches no error pattern (source frames, `>` markers, `E` lines, diff lines, captured output).

| Finding | Evidence | Consequence |
|---|---|---|
| **Deterministic** for a fixed (input, mode, hint, cap) | 48/48 combinations gave one SHA across 60 runs | Memoization is still used, because a version upgrade or config change can alter output. The key includes them. |
| **The `filter` path is a pure function** apart from `auto` | `compressOutput(stdout, cmd, mode, maxOutputBytes)` in `dist/cli/filter.js`. Only `auto` calls `pickModeAuto` (an LLM). | The sidecar imports this function. It does not spawn the CLI or `wrap` (which runs a shell), and it rejects `auto`. |
| **Even `conservative` truncates silently** above `maxOutputBytes` (default 100 KB) | The 168 KB log came back with a head/tail `smartTruncate` | Always pass an explicit `maxOutputBytes` equal to the input length, so upstream never truncates. Bounding is Code Partner's job. |
| **A hint turns on destructive command filters** | `pytest` hint + `balanced`: 11 KB became 649 B. All 26 node ids and 41 traceback-context lines were dropped, and failure reasons were cut to `KeyErro...` | **No command hint by default.** Error output never gets a hint and is `conservative` only. |
| **Upstream markers point at tools we don't expose** | `[+179 lines omitted — use execute(intent:…) or search()…]` | Strip upstream retrieval hints and append the Code Partner pointer (`codepartner_get_observation`). |
| **`balanced` with no hint is a no-op on grep, git log and unittest** | 0% change | Gains come only from ANSI stripping, JSON minification and log template folding. |
| **JSON minify (balanced) is lossless** | 54 KB became 37.7 KB (−31%) | Accept it only if `json.loads(out) == json.loads(in)`. |
| **`aggressive` collapses JSON** | 397/400 paths, ids and symbols lost | `aggressive` is unsupported. |
| **Log template folding is large but lossy** | 168 KB became 855 B (−99%). Distinct `job_N` ids and frame line numbers were folded into `×6 similar lines` | Allowed only when no error-like line is folded. If one is, fall back to the original. |
| **ANSI stripping** | Coloured pytest output shrank by 17%, losing nothing | This is the safe baseline. |

**Gate decision:** passed for the restricted surface: conservative everywhere, no-hint `balanced` for non-error logs and JSON, and the fallback guards. Command-hinted filters are out of scope until outcome data supports them.

**Expected impact (honest estimate):** file reads, which are the largest results, are excluded. Searches are already shaped, and test output is already reduced to totals plus failures. Real savings will come mostly from logs, coloured command output and structured JSON, so expect a modest overall reduction. The online benchmark (WP7) decides whether `balanced` is worth widening.

## Decisions

### D1. Hook in `_remember`, not `bridge.py`
`_remember` already sees both the shaped result and the raw result, owns the observation id, and runs for both providers. Retrieval goes through the normal `registry.prepare_advertised_call` path, so no permission code changes.

### D2. A sidecar that imports the function and runs nothing
`infra/compress-bridge/bridge.mjs` listens on a Unix socket and accepts `{mode, text, max_output_bytes}`. It calls `compressOutput(text, undefined, mode, max_output_bytes)` and returns `{text, version}`. It rejects any mode other than `conservative` or `balanced` and any request that carries a command.

Container settings follow the `kb-bridge` pattern:
- `network_mode: none`
- read-only rootfs, with tmpfs `/tmp`
- no project or home mounts
- empty `CONTEXT_COMPRESS_PASSTHROUGH_ENV`, and no `ANTHROPIC_API_KEY` or other secrets
- `CLAUDE_PROJECT_DIR` points at an empty directory

The `local` backend runs the same `bridge.mjs` as a long-lived subprocess speaking JSON lines over stdio, with an empty environment. The `noop` backend returns the input unchanged.

### D3. Mode policy
| Content | Mode |
|---|---|
| Error-heavy: `tool_failed`, non-zero exit, `Traceback`, `FAILED`, `Error:` | `conservative` |
| JSON | `balanced`, accepted only if parse-equal |
| Other non-error command, log or search output | the configured mode (`conservative` by default during benchmark stage A, `balanced` in stage B) |
| File reads, edits and mutations, OpenSpec and Reviewer trusted paths | never compressed |

### D4. Guards, a minimum check rather than proof
Compression falls back to the original when:
- an identifier is lost: paths, `path:line`, pytest node ids, symbols, observation ids or `call_*` ids
- an error-like line is lost
- the result is JSON and is no longer parse-equal
- the output is not shorter than `min_saving_ratio`
- the backend times out or fails

Passing the guards does not prove the output is still useful, because tracebacks depend on context lines that match no pattern. That is why error output stays `conservative`, and why correctness is judged on task outcomes (WP7).

### D5. Memoization
The cache key is `sha256(input) | compressor_version | mode | hint | json(effective_config)`, held in an in-process LRU for each service instance. Within the process, the same key always returns the same bytes, and only new results are ever compressed. Provider history is never rewritten.

### D6. Provenance
A new `context_observations` table is added to `TelemetryStorage`, holding:
- run, task and attempt ids, observation id
- tool, resource, args hash, content sha256
- mode, compressor version, byte counts and estimated token counts
- applied and reason, `duplicate_of`, stale, created_at

Raw text goes in an FTS5 table only when `CODEPARTNER_COMPRESS_PERSIST=true`, and otherwise in memory (the existing `observations` dict). That keeps the documented "telemetry never stores source text" default.

### D7. Measurement
Estimated tokens use bytes/4 and are labelled `estimated_*`. Actual usage comes from the provider's `usage` fields: `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens` and `cached_input_tokens`. The cache-read ratio is a diagnostic, not a success gate.

## Implementation notes

- **Error detection was narrowed** after the offline benchmark. A case-insensitive `error` pattern classified a `git grep` of `def …error…` signatures as error output. Detection now requires one of:
  - an exception name followed by `:` or the end of the line
  - a test-runner marker (`FAILED`, `FAIL:`, `ERROR`, `E   `)
  - a compiler or npm error prefix anchored to the start of the line or a `path:line:` prefix
  - a non-zero exit code

  Unit tests pin both sides of this rule.
- **Compression covers shaped bodies, not raw backend output.** Search results are already `path:line` lists, and command output is already reduced to totals and failures. In practice the gains come from generic tool results (JSON) and non-error logs.
- **Reads are tracked but never compressed**, so a reload after an edit carries a `STALE:` header.
- **Offline benchmark** (`scripts/bench_compression.py`, generated corpus): conservative −0.6%, balanced −35% estimated tokens, all byte-identical across repetitions. The balanced figure comes almost entirely from a synthetic heartbeat log, so actual savings must come from the online A/B (T7.3).

## Risks

- **Upstream behaviour changes between versions.** Mitigations: an exact version pin, the version in the memo key and provenance, and a re-run of the WP0 harness on every upgrade.
- **Folded or minified output confuses the model.** Mitigations: the marker plus a retrieval pointer, the guards, and the staged rollout.
- **Sidecar unavailable.** Mitigation: fail-open to the original, counted in `compress_fallbacks`.

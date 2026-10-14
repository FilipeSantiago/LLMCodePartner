# Tasks: Provenance-aware context compression

## 1. WP0 — Upstream spike

**Objective:** Pin context-compress and verify the `filter` contract, determinism and fidelity before building on it.

**Complexity:** low

- [x] T0.1 Pin the version and verify the filter contract
  - Description: Install `context-compress@2026.8.3`, read `dist/cli/filter.js`, and confirm that `compressOutput(stdout, cmd, mode, maxOutputBytes)` is pure apart from `auto`.
  - Status: completed
  - Complexity: low
  - Reasoning: the integration strategy depends on the real API surface.
  - Definition of done:
    - Findings recorded in design.md
- [x] T0.2 Test determinism and fidelity empirically
  - Description: Run 8 real outputs × 3 modes × {no hint, hint}, with 20 repetitions in each of 3 processes, and check identifier, error-line and traceback-context retention.
  - Status: completed
  - Complexity: low
  - Reasoning: determinism and fidelity must be measured, not assumed.
  - Definition of done:
    - Results table and gate decision in design.md

## 2. WP1 — Compressor sidecar

**Objective:** Run the filter pipeline isolated from the network, project files and secrets.

**Complexity:** medium

- [x] T1.1 Build the sidecar bridge
  - Description: Write `infra/compress-bridge/bridge.mjs` (a Unix socket in sidecar mode, JSON lines over stdio in local mode) that calls `compressOutput` with an explicit `maxOutputBytes`. It rejects `auto`, `aggressive` and any command hint.
  - Status: completed
  - Complexity: medium
  - Reasoning: importing the function avoids `wrap`/shell execution entirely.
  - Definition of done:
    - Requests for `auto` or with a command are rejected
    - The response includes the compressor version
- [x] T1.2 Write the Dockerfile and compose overlay
  - Description: Write `infra/compress-bridge/Dockerfile` (node:22-slim, `npm ci --ignore-scripts` from a pinned lockfile) and `compose.compress.yaml` (`network_mode: none`, read-only, tmpfs `/tmp`, socket volume only).
  - Status: completed
  - Complexity: low
  - Reasoning: reuses the kb-bridge isolation pattern.
  - Depends on: T1.1
  - Definition of done:
    - `docker inspect` shows `NetworkMode=none` and no project bind mount

## 3. WP2 — Compression service

**Objective:** Decide whether and how to compress, and never lose information silently.

**Complexity:** medium

- [x] T2.1 Write `ContextCompressionService` in `mcp_bridge/compression.py`
  - Description: Add backends (sidecar, local, noop), eligibility by category and threshold, the mode policy (error output conservative only), the identifier, error-line and JSON parse-equal guards, fail-open handling with a timeout, upstream marker rewriting and the memo LRU.
  - Status: completed
  - Complexity: medium
  - Reasoning: this is the core logic. Everything else wires it in.
  - Depends on: T1.1
  - Definition of done:
    - Every fallback path returns byte-identical original text
    - The memo key includes the version, mode, hint and effective config

## 4. WP3 — Interface integration

**Objective:** Apply compression to coder results and keep originals retrievable.

**Complexity:** medium

- [x] T3.1 Hook the service into `AgentToolInterface._remember`
  - Description: Compress the shaped result and append a marker plus the observation pointer. Wire the service through `mcp_bridge/server.py` and `agent/routing/task_execution.py` behind `CODEPARTNER_COMPRESS_ENABLED`.
  - Status: completed
  - Complexity: medium
  - Depends on: T2.1
  - Reasoning: the existing seam serves both Claude and Codex.
  - Definition of done:
    - With the feature disabled, results are byte-identical to the current behaviour
- [x] T3.2 Add `query` to `codepartner_get_observation`
  - Description: Add run-scoped search over observations, plus a STALE header on retrieval.
  - Status: completed
  - Complexity: low
  - Depends on: T4.1
  - Reasoning: covers the "searchable originals" requirement.
  - Definition of done:
    - An id or query from another run returns UNAVAILABLE or no results

## 5. WP4 — Provenance store

**Objective:** Record the origin and transformation of every observation.

**Complexity:** medium

- [x] T4.1 Add the provenance table to `TelemetryStorage`
  - Description: Add `context_observations` plus an optional FTS5 content table (only when `CODEPARTNER_COMPRESS_PERSIST=true`), dedup by content hash, and mark observations stale on mutation of the resource.
  - Status: completed
  - Complexity: medium
  - Reasoning: reuses the existing SQLite store and migration loop.
  - Definition of done:
    - By default no source text is persisted
    - Duplicates point at the first observation

## 6. WP5 — Token measurement

**Objective:** Keep estimated savings separate from actual provider usage.

**Complexity:** low

- [x] T5.1 Add per-attempt compression metrics and usage normalization
  - Description: Add `compress_applied`, `compress_fallbacks`, `est_tokens_saved` and bytes before and after to `telemetry_aci_attempts`. Normalize `cache_creation_input_tokens`, and group `aci_comparison()` by compression mode.
  - Status: completed
  - Complexity: low
  - Depends on: T3.1
  - Reasoning: proving actual savings is part of acceptance.
  - Definition of done:
    - The report shows estimated and actual columns separately
    - The cache ratio is shown as a diagnostic

## 7. WP6 — Configuration and docs

**Objective:** Make the feature configurable and documented.

**Complexity:** low

- [x] T6.1 Add `.env.example` keys and docs
  - Description: Add the `CODEPARTNER_COMPRESS_*` keys, write `docs/context-compression.md`, and update `docs/agent-tool-interface.md`.
  - Status: completed
  - Complexity: low
  - Depends on: T3.1
  - Reasoning: operators need the knobs and the limits.
  - Definition of done:
    - The default is off
    - The docs state the guard limitations and the rollout stages

## 8. WP7 — Tests and benchmark

**Objective:** Show no regression and measure the real effect.

**Complexity:** medium

- [x] T7.1 Write unit and regression tests
  - Description: Write `tests/test_context_compression.py` with a fake backend: threshold, exclusions, conservative for errors, identifier, error and JSON fallbacks, fail-open, memo key, dedup, staleness, run-scoped retrieval and rejection of `auto`. Run the existing suite with the feature on and off.
  - Status: completed
  - Complexity: medium
  - Depends on: T3.2, T4.1
  - Reasoning: proves there is no regression.
  - Definition of done:
    - All tests pass in both configurations
- [x] T7.2 Build the offline replay benchmark
  - Description: Write `scripts/bench_compression.py`, porting the WP0 harness. It reports bytes, estimated tokens and fallback reasons per category and mode.
  - Status: completed
  - Complexity: low
  - Depends on: T2.1
  - Reasoning: a cheap, repeatable signal before any online runs.
  - Definition of done:
    - The report is reproducible from a fixed corpus
- [ ] T7.3 Run the staged online A/B
  - Description: Run the same tasks, models, revision and IDE for Claude and Codex, first with compression off, then stage A (conservative), then stage B (balanced on non-error output).
  - Status: pending
  - Context: Needs a live IDE, provider credentials and a fixed task set, so it can't run in CI. The protocol is in docs/context-compression.md#rollout and the report is `TelemetryAggregation.aci_comparison()`.
  - Complexity: medium
  - Depends on: T5.1, T7.1
  - Reasoning: task outcomes, not guards, establish correctness.
  - Definition of done:
    - No drop in success rate or Reviewer verdicts
    - Lower actual uncached input tokens and cost per completed task
    - Cache metrics reported as a diagnostic

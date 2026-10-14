# Proposal: Provenance-aware context compression

## Epic

**EPIC1 — Provenance-aware context compression**

Reduce the model context taken up by large coder tool results in Claude and Codex runs. Compression must lose no information the coder needs, must keep originals retrievable through authorized access, and must leave the security boundary unchanged.

## Why

Large search, command, log and structured results go into the provider conversation in full, and they stay there for the rest of the run. `AgentToolInterface` already shapes those results and keeps the raw output retrievable. It does not remove noise such as ANSI codes, repeated log lines or pretty-printed JSON, it does not record where a shaped result came from, and it does not separate estimated savings from actual provider token usage.

## What Changes

- Add a `ContextCompressionService` that runs inside `AgentToolInterface._remember`, after the existing shaping. `bridge.py` routing, the registry and permissions stay unchanged.
- Run the `context-compress` `filter` pipeline (pinned at `2026.8.3`) in a sidecar with no network and no project mount. Never expose its `execute`, `execute_file`, `batch_execute` or `fetch_and_index` tools, or its `auto` mode.
- Choose the mode by content type. Error-heavy output uses `conservative` only. `balanced` applies only to non-error output, and with no command hint.
- Fall back to the original when any downstream identifier, error line or JSON value is lost, and when the compressor fails or times out.
- Memoize results by input hash, compressor version, mode, command hint and effective config.
- Keep run-scoped provenance (origin, content hash, transformation, duplicate and stale state) and FTS5 search through `codepartner_get_observation(query=…)`.
- Report estimated compressor savings separately from actual provider input, cache-read, cache-write and output tokens.

## Work Packages

- **WP0 — Upstream spike** (research): pin the version, verify the `filter` contract, test determinism and fidelity empirically.
- **WP1 — Compressor sidecar** (infrastructure): a locked-down sidecar plus local and noop backends.
- **WP2 — Compression service** (feature): eligibility rules, mode policy, guards, fail-open handling and memoization.
- **WP3 — Interface integration** (feature): the `_remember` hook, the marker and `get_observation(query)`.
- **WP4 — Provenance store** (feature): the provenance table, FTS, dedup and staleness.
- **WP5 — Token measurement** (feature): estimated savings versus actual usage, and `aci_comparison` grouping.
- **WP6 — Configuration and docs** (docs): `.env.example`, `docs/context-compression.md`, and an update to `docs/agent-tool-interface.md`.
- **WP7 — Tests and benchmark** (verification): unit, regression and security tests, plus an offline replay and a staged online A/B run.

## Capabilities

### New Capabilities

- `context-compression`: compress eligible coder tool results without losing information, keep provenance and retrieval for the originals, and measure the effect.

### Modified Capabilities

_None._ The `codepartner_get_observation` schema gains an optional `query` parameter, and existing calls behave as before.

## Impact

- `mcp_bridge/agent_tools.py`, `mcp_bridge/server.py`, `agent/routing/task_execution.py`
- `telemetry/storage.py`, `telemetry/aggregation.py`
- New: `mcp_bridge/compression.py`, `infra/compress-bridge/`, `compose.compress.yaml`
- `.env.example`, `docs/`
- The feature is off by default (`CODEPARTNER_COMPRESS_ENABLED=false`), so there is no behavioural change until it is enabled.

# Proposal: Optimize planner discovery

## Why

The Work Package planner sometimes reads files that have nothing to do with the WP, and sometimes reads whole files when an outline or a ranged read would have been enough. Every source byte it reads stays in its context for the rest of the run. Before any behaviour changes, we need to measure which retrieval level each read used and how much evidence linked the read path to the WP. Only then can advisories be shown to cut exploration without lowering plan quality.

## Scope

- **In:** the WP planner (`agent/planning/service.py`), which is the only planner that explores with tools.
- **Out:**
  - The OpenSpec change generator (`agent/spec/openspec_adapter.py`). It is a completion with no tools.
  - Coder, reviewer and handoff behaviour.
  - The shared `KB_READING_GUIDANCE`.
  - Provenance infrastructure, new services and databases.

## What Changes

- `planner.discovery` config, with `mode: off | observe | advise` and per-complexity advisory thresholds. The default is `observe`, and `advise` is opt-in.
- `DiscoveryTracker` (`agent/planning/discovery.py`) records each turn's:
  - retrieval level: L1 for outline, search and graph calls; L2 for ranged reads and small whole reads; L3 for large whole reads. L3 reads are split into direct and escalated.
  - evidence strength of each read path: strong, moderate, weak or none. Reads with weak or no evidence are counted as *unexplained*, not irrelevant.
  - estimated added context.
  
  All of this is stored under the `discovery` key of `wp_planning_turns.tool_metrics`.
- `scripts/bench_planner.py` compares benchmark arms. It uses provider usage normalised by the existing `normalize_usage`, measures changed-file recall against the WP's implementation diff, and includes a manual relevance rubric.
- The hard caps (`max_tool_calls`, `max_file_reads`), the KB prefetch and the existing telemetry keys are unchanged.

## Work Packages

- **WP1 — Observe** (feature): config, tracker, wiring, benchmark tooling.
- **WP2 — Baseline** (research): pinned WP3 runs in observe mode, N ≥ 3.
- **WP3 — Advise** (feature): planner-only progressive-retrieval guidance, a sufficiency checklist, and an advisory shown once per threshold. Opt-in.
- **WP4 — Evaluate** (research): advise vs. baseline, then the docs.

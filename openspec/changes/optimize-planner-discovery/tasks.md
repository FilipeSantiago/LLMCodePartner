# Tasks: Optimize planner discovery

## 1. WP1 — Observe

**Objective:** Measure retrieval level, evidence strength and estimated added context for each WP planner turn, without changing behaviour.

**Complexity:** medium

- [x] T1.1 Discovery config
  - Description: Add `planner.discovery`, with `mode`, `small_file_bytes` and per-tier thresholds, to `PlannerConfig`. Validate it, and reject `advise` for now.
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - Defaults and validation tested in `tests/test_planner_routing.py`
- [x] T1.2 Discovery tracker
  - Description: Write `agent/planning/discovery.py`. It covers levels, direct vs. escalated L3, evidence strength kept separate from progression, unexplained reads, and `est_added_tokens`. Share dispatcher and range parsing with `SourceReadStats` through a behaviour-preserving extraction.
  - Status: completed
  - Complexity: medium
  - Definition of done:
    - Unit tests in `tests/test_planner_discovery.py`
- [x] T1.3 Observe wiring
  - Description: Pass an optional tracker to `instrument_handlers`, and record the result under `tool_metrics.discovery` in `_attempt_planner`.
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - Test showing `off` and `observe` give identical specs, conversations and tool results
- [x] T1.4 Benchmark script
  - Description: Write `scripts/bench_planner.py`. It reports median and range per arm, provider usage through `normalize_usage`, changed-file recall, gaps, and the rubric.
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - Tests in `tests/test_bench_planner.py` for each usage shape, for n/a handling and for quality fields

## 2. WP2 — Baseline

**Objective:** Record the pinned WP3 baseline in observe mode.

**Complexity:** low

- [x] T2.1 Legacy WP3 baseline from recorded sessions
  - Description: Bench the historical WP3 planning sessions against the files changed by WP3's verified implementation (job ed63bf10).
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - Results and caveats recorded in design.md
- [ ] T2.2 Pinned WP3 observe runs
  - Description: Rebuild the container. Check out the pre-WP3 commit, with the KB index matching it. Run N ≥ 3 fresh sessions of WP3 fingerprint `dcee3214` with the model, effort and `narrate` held constant, and score the manual rubric.
  - Status: pending (needs the user's IDE project and live models)
  - Complexity: simple
  - Definition of done:
    - Bench output recorded in design.md

## 3. WP3 — Advise

**Objective:** Nudge the planner toward the cheapest sufficient retrieval, without hard blocking.

**Complexity:** medium

- [x] T3.1 Planner-only guidance and sufficiency checklist
  - Description: Add a constant in `agent/planning/prompts.py` covering progressive retrieval and a checklist (components, dependencies, responsibilities, verification). The shared `KB_READING_GUIDANCE` stays unchanged.
  - Status: completed
  - Complexity: simple
- [x] T3.2 Soft-threshold advisories
  - Description: The first time a tier threshold is crossed, append one short advisory line to that tool result. Record the crossing in `threshold_crossings`. The hard caps are unchanged.
  - Status: completed
  - Complexity: medium
  - Definition of done:
    - The advisory fires once per threshold, and the shared guidance is unchanged (tests)

## 4. WP4 — Evaluate

**Complexity:** low

- [x] T4.1 Docs
  - Description: Update `docs/wp-planner.md` and `docs/codebase-memory.md`.
  - Status: completed
- [ ] T4.2 Advise vs. observe benchmark
  - Description: Run N ≥ 3 pinned WP3 sessions in `advise` and compare them with T2.2 using `scripts/bench_planner.py`. Then tune the tier thresholds.
  - Status: pending (needs the user's IDE project and live models)
  - Complexity: simple

## 5. WP5 — KB-first

**Objective:** Make Codebase Memory the planner's primary discovery path whenever it is offered. IDE source stays authoritative.

**Complexity:** medium

- [x] T5.1 Outline gate
  - Description: Write `OutlineGate`, hooked through `instrument_handlers(gate=)`. A broad read of a large indexed code file returns its outline first, and repeating the read passes through. Add `outline_gate_lines` to the config.
  - Status: completed
  - Complexity: medium
  - Definition of done:
    - Gate and pass-through tests in `tests/test_planner_discovery.py`
- [x] T5.2 KB-first prompt and neutral budget wording
  - Description: Add `KB_FIRST_GUIDANCE` whenever `kb_*` tools are offered. The read-budget refusal and its chat note no longer point to KB as a fallback.
  - Status: completed
  - Complexity: simple
- [x] T5.3 Record stopped runs
  - Description: Catch `CancelledError` in `_attempt_planner`, record the turn as cancelled, then re-raise.
  - Status: completed
  - Complexity: simple
- [x] T5.4 Whole reads by `limit`
  - Description: The tracker treats a `limit` beyond the returned file as a whole read.
  - Status: completed
  - Complexity: simple
- [ ] T5.5 Live check
  - Description: Rebuild, plan a WP, and confirm early KB calls and `outline_gate.served > 0`. Compare KB calls, L3 reads and `est_added_tokens` with the stopped WP5 run.
  - Status: pending (user)

## 6. WP6 — KB-first for all agents

**Objective:** Make Codebase Memory the default discovery map for planner, coder
and reviewer while preserving current IDE source as the authority.

- [x] T6.1 Shared policy and single authorized dispatch
  - The run-local policy sits in the shared bridge. Generated calls use the same
    request-scoped registry and coder/reviewer/planner wrappers as direct calls.
- [x] T6.2 Explicit graph-first search gate
  - Definition-style identifier searches show graph hints with a warning;
    repeating the original search executes it. Usage, import and literal intent
    passes through.
- [x] T6.3 IDE-backed snippets and scope verification
  - Resolve indexed locations, reject paths outside the indexed root, read
    current IDE source, verify the symbol scope, and relocate stale hints.
- [x] T6.4 Region-aware edit verification
  - KB-assisted edits require an IDE-verified affected region; new files are exempt.
- [x] T6.5 Per-role wiring, telemetry and regression tests
  - Planner, coder and reviewer attach a run-local policy; record outcomes and
    separate redundant reads from legitimate escalation.
- [ ] T6.6 Downstream benchmark and live check
  - Compare per-task efficiency and guardrails against historical WP3, WP4 and
    WP5 runs. Rebuild and execute one new WP with plan, implementation and review.

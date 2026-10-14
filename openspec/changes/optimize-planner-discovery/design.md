# Design: Optimize planner discovery

## WP6 revision: shared KB-first discovery

The policy is attached to each agent run at bridge.call_tool, after the model
requests a tool but before the existing authorized dispatch. Generated graph
lookups, outlines, current IDE source reads and relocation searches traverse
the same request registry, adapters, stop check, journal, review path check and
planner metrics/budgets. A context-local generated-call flag prevents recursive
gating. No repository mount, service, index or dependency is introduced.

The graph gate intercepts only bare identifier definition searches. Its result
explicitly says that text search was not executed; an identical repeat runs
the original search. KB paths and line ranges are hints. A snippet is returned
only after a current IDE read verifies the declaration and enclosing Python
scope, or a unique current relocation does. Reviewer changed files always
bypass the outline gate, and only IDE reads count as review evidence.

Security dependency: production contract-guard enforcement of generated calls
relies on the separate handoff work wiring ContractExecutionGuard into the
call path. Until then, tasks with excluded_files receive no KB policy or KB
tools. The policy does not grant any additional tool authority.

Historical per-task references: WP3 4c819fe8 (truth ed63bf10); WP4 22bf4d17
and a7950cdd (truth 7c9b0d76, excluding temporary files); WP5 v1 07bed0b.
These are not a controlled A/B test. Compare each task's source bytes,
estimated added context, provider usage, unnecessary lookups and guardrails,
not raw cross-WP totals.

## Context

These pieces already exist:
- The WP planner receives a bounded KB prefetch (`planning_context(..., "implementation")`) and on-demand `kb_*` tools.
- `instrument_handlers` enforces hard caps: 60 tool calls and 25 file reads. KB calls are exempt.
- `ToolMetrics` and `SourceReadStats` already record kb calls, whole and ranged reads, repeated reads and source bytes for each planner attempt in `wp_planning_turns`.

This change adds to those pieces. It does not replace them.

## Decisions

| Decision | Rationale |
|---|---|
| Observe first, advise later | Behaviour changes need a measured baseline. `advise` is opt-in and the default stays `observe`. |
| The tracker is an optional argument to `instrument_handlers` | The planner is the only caller. With no tracker, behaviour is byte-for-byte unchanged. |
| Observe never changes prompts, tool schemas or tool responses | A regression test compares `off` and `observe` provider inputs and tool results. |
| Retrieval level is decided from the **returned** UTF-8 size | A whole read under `small_file_bytes` (6000) is L2. Small configs and `__init__` files should not look like full reads. |
| L3 is split into `l3_direct` and `l3_escalated` | An escalated read follows an L1 or L2 call that targeted the same path. Only L1 and L2 calls mark progression. A repeated whole read stays direct. |
| Evidence is separate from progression | Outlining or reading a path never raises that path's evidence strength. A search inside a file is not evidence for that file. |
| Evidence strength comes from deterministic sources | **Strong:** paths in the WP brief or user message, or KB prefetch symbol matches and their outlines. **Moderate:** results of search or graph calls aimed at a WP identifier or a strong path. **Weak:** broad listings, architecture, KB keyword matches, untargeted searches. **None:** the path was never mentioned. |
| Weak and none count as *unexplained* | A read without evidence may still be legitimate exploration. It is not labelled irrelevant. |
| `est_added_tokens` = delivered UTF-8 bytes / 4 (prompt + tool results, refusals included) | This estimates newly added context. Provider cumulative usage is a different quantity and is reported separately. |
| The bench normalises usage with `telemetry.aggregation.normalize_usage` | Anthropic reports cache tokens outside `input_tokens`; Codex includes `cached_input_tokens` in it. One normaliser keeps uncached, cache-read and cache-write disjoint. |
| Quality has two parts | (a) changed-file recall against the implementation diff; (b) plan relevance: `validate_draft` gaps plus a manual rubric covering completeness, correctness, scope and acceptance-criteria coverage. |

## Baseline protocol (WP2)

- **Pin:** the project commit, the KB index (`index_status`), the WP3 fingerprint (`wp_fingerprint`), the planner model and effort, and `narrate`.
- **Runs:** N ≥ 3 fresh planning sessions in `observe`. Do not use `discuss`.
- **Command:** `scripts/bench_planner.py --arm baseline=… --truth wp3_changed_files.txt --rubric wp3_rubric.json`.

## Advise mode

- `DISCOVERY_GUIDANCE` (in `agent/planning/prompts.py`) is added to the planner prompt only in `advise`. It covers progressive retrieval (outline → ranged read → full file) and a sufficiency checklist: components and files, dependencies, task responsibilities, verification. It puts correctness first.
- `DiscoveryTracker` takes the WP complexity tier's thresholds. Each tool call it re-checks:
  - `tool_calls`: executed calls;
  - `full_reads`: L3 reads;
  - `est_added_tokens`;
  - `repeated_or_unexplained_reads`: reads that are repeated, or have weak or no evidence.
  
  The first time any threshold is reached, one advisory line (`[Planning advisory, not a limit] …`) is appended to that tool result. If several thresholds are reached on the same call, they share one line, and no threshold fires twice. Crossings are recorded in `threshold_crossings`, and the advisory's bytes count toward `est_added_tokens`.
- The hard caps, their refusal messages and the shared `KB_READING_GUIDANCE` are unchanged. A test pins the guidance text by hash.

## KB-first (WP5)

A WP5 run on 2026-10-14 showed the KB acting as a fallback, not the map:
- It made 2 outline calls in 41 calls, with no graph searches or traces.
- It made 26 `read_file` calls, mostly `limit: 5000`.
- The only KB pointer the planner received was the read-budget refusal.

Decisions:
- KB-first applies whenever KB tools are offered, in any mode.
- The planner prompt gets `KB_FIRST_GUIDANCE`.
- The deterministic `OutlineGate` answers a broad read of a large, indexed, not-yet-located code file with its outline. Repeating the read returns the file, so this is not a block.
- The refusal wording is neutral.
- A stopped run records its turn as cancelled. Before this, `CancelledError` bypassed `except Exception` and stopped runs left no telemetry.

## Baseline results (legacy, recorded 2026-10-14)

**Source:** the live routing database (`wp_planning_turns`), analysed on a copy with
`scripts/bench_planner.py`.

**Truth set:** the 9 files whose mutations reached `postimage_verified` in WP3's only completed implementation job, `ed63bf10`:
- `generation/{config,errors,interface,factory,ngram_generator,torch_generator,__init__,canonical_conversion}.py`
- `tests/test_generation.py`

These runs predate observe mode, so there are no `discovery` fields. Older rows also lack the source-read fields.

**WP3 was redefined between the runs.** The Oct 9 sessions planned an *evaluation layer* (6 tasks, fingerprint `a0f12ef3`). The implemented WP3 is a *`generation/` package* (5 tasks, fingerprint `dcee3214`). The truth set is therefore valid only for the current-WP3 run. For the Oct 9 cohort, the recall figures do not measure quality; that cohort serves only as a cost reference.

| Metric (median [range]) | Current WP3 (`dcee3214`), N=1 | Oct 9 WP3 (`a0f12ef3`), opus + KB, N=3 |
|---|---|---|
| Tool calls | 40 | 49 [44–51] |
| KB calls | 0 | 10 [5–13] |
| File reads (cap 25) | 25, plus 1 refused | 25 [25–25], plus 3 [1–3] refused |
| Whole / ranged reads | 0 / 25 (artefact, see below) | n/a |
| Source response bytes | 134,689 | n/a |
| Total input tokens (provider) | 1,017,037 | 2,105,910 [1,791,271–2,332,780] |
| — of which cache-read | 923,692 | 1,990,631 |
| — of which cache-write | 93,305 | 115,219 |
| Output tokens | 44,624 | 50,366 |
| Latency | 513 s | 609 s [579–630] |
| Gaps / repair turns | 0 / 0 | 0 / 0 |
| Planned files | 33 | 34 [31–46] |
| Changed-file recall | 1.0 (9/9) | not comparable (different WP3) |
| Planned files outside the diff | 24 | not comparable |

**Observations:**
- Every run hit the 25-read cap.
- Cumulative provider input is dominated by cache reads: the context is re-sent on every tool round-trip. Each source byte read early in a run is therefore paid for many times.
- The current-WP3 run made no KB calls at all.
- **Correction (2026-10-14):** the planner reads with `limit: 5000` and no start line. `read_region` (shared with the coder) records that as the range 1–5000, so "0 whole / 25 ranged" over-counts ranged reads. The discovery tracker now treats a `limit` larger than the returned file as a whole read. `SourceReadStats` is unchanged.
- 24 of its 33 planned files fall outside the implementation diff. Many are legitimate reuse or context files (decoders, trainer, notebooks), so this is a relevance signal for the manual rubric, not an error count.

**Still required for a pinned comparison:** N ≥ 3 fresh `observe` runs and N ≥ 3 `advise` runs of the same WP3 fingerprint, on the same commit and KB index, with the model, effort and `narrate` held constant. WP3 is already implemented in the project, so these runs need a checkout of the pre-WP3 commit; otherwise the planner sees the finished `generation/` package. The running container must also be rebuilt to include this change.

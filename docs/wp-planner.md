# Work Package planner

Every Work Package implementation request ("Let's do WP4", "implement WP1 and WP2")
goes through a planning stage first, in every mode except review. A strong model
investigates and plans the whole WP once, then hands structured context to each
task's coder and to the reviewer. Task execution, coder routing, and review are
unchanged; they gain context. New implementation starts require Work Package
planning. An EPIC selector queues its unfinished Work Packages; an individual
task selector directs the user to its parent Work Package. Saved jobs still
resume and recover directly.

When a request names several WPs, they are planned **just in time**: plan WP1, you
approve, WP1 is implemented and reviewed, then WP2 is planned against the code WP1
actually produced. The remaining WPs are stored on the planning session
(`queued_wp_ids`). Planning of the next WP starts in the same reply that reports the
previous WP's job as completed.

```
user: "Let's do WP4"
  → OpenSpec WP + WP complexity
  → planner routing (planner tier for that complexity)
  → planner turn: read-only repository investigation
  → plan + structured state ── "discuss" ──┐
        ↑______________________________________┘
  → "approve" → freeze (no model call) → ApprovedWPPlan
  → existing job: task contract + WP context + task context → coder (task complexity routing)
  → WP review starting from ReviewerContext + execution record
```

## Work Package complexity

`WorkPackage.complexity` is one of `low | medium | high | complex`. It rates how hard
the *whole package* is to plan. OpenSpec only states the level; it never names a
model. Task complexity (`simple | medium | complex`) is independent and still routes
coders. When a catalog predates the field, the level is derived from the hardest
active task (simple→low, medium→medium, complex→high) and recorded as
`complexity_source: derived`.

## Routing

`config/model-routing.yaml` → `planner.tiers` maps WP complexity to planner
candidates. `agent/planning/routing.py` picks the first healthy candidate
deterministically, and the planner floor (`roles.planner.min_capability`) applies.
The selected model follows its gateway policy: `preferred` tries agentgateway
then retries directly on a gateway transport or HTTP 401/403 authorization
failure; `required` uses only the gateway; `disabled` uses only the direct CLI.
A provider account/session quota, direct-route authentication failure, or
explicit local CLI login failure skips another route to the same account and
tries the next configured planner candidate. If both gateway and direct routes
fail operationally, the planner also tries the next candidate. A malformed plan is
repaired by the model that produced it, not handed to a different provider.
The model that succeeds is saved on the planning session for repair, discussion,
and approval. Each attempt is recorded with its model, gateway choice, available
usage, cost, and full failure diagnostic; the chat displays only short failure
reasons. No global quota cooldown is applied.
See `docs/model-routing.md` for the floors and the role-relative check.

## Conversation

| User says | Effect |
|---|---|
| "Let's do WP4", "implement WP1 and WP2" | New planning session for the first WP; the others are queued |
| any other text | "Discuss more": next planner turn in the same session |
| `approve`, `1`, `yes`, "looks good, go ahead" | Approve the current revision and start the job |
| `discuss`, `2` | Asks what to change (no model call) |
| "Let's do WP5" | Supersedes the session and plans WP5 |
| `never mind`, `stop planning`, `reject <token>` | Abandons planning |

Every complete plan ends with the two actions **Approve and implement** and
**Discuss more**. Nothing is coded before approval.

## Live progress

While the planner works, the chat shows what it is doing:

```
**Investigating WP1 (planner: Claude Opus)**

💭 Checking how loaders are currently registered
- 🔎 Reading `data_loader/base_dataloader.py` (lines 1–200)
- 🔎 Searching for `DataLoaderFactory`

---

## Plan for WP1 (revision 1)
```

- The 🔎 lines come from the tool calls themselves and cost no tokens.
- The 💭 lines are one short reason the planner writes before each tool call. They cost about 15–25 output tokens per call, and `planner.narrate: false` turns them off. Each `wp_planning_turns.tool_metrics` row records `narrate`, so the cost can be compared.
- The plan ends with **What I inspected and why**, built from the verified findings at no extra cost.

All chat formatting goes through `conversation/presentation.py`, which planner, coder, and reviewer output share. Every message ends with a newline, so streamed messages never run together.

## The planner keeps structured state

Every planner turn returns one JSON object (`agent/planning/prompts.py`). It holds the
human-readable `plan_markdown` and the full structured state:

- `findings`: verified facts (with file evidence) or explicit assumptions, with
  stable ids, linked to tasks;
- `shared`: the `WPSharedContext`, information several tasks need;
- `tasks`: one `TaskContext` per OpenSpec task, holding objective, approach, files,
  symbols, existing implementation, constraints, do-not-modify, and expected tests;
- `reviewer`: the `ReviewerContext`, holding expected behavior, task boundaries,
  risks, assumptions, and validation expectations.

The planner brief includes the statement and scenarios of every requirement
referenced by the Work Package, as well as task descriptions and design context.

On each discuss turn the planner revises this state in place and keeps its ids
stable. `validate_draft` checks the state after every turn. On failure the planner
gets one repair request listing the exact gaps; if gaps remain, the plan is shown
without the Approve option.

## Approval freezes; it does not reinterpret

`WPPlannerService.approve` makes no model call. It checks that the WP is unchanged
in OpenSpec since planning began, then `projection.project` builds the frozen
`ApprovedWPPlan`:

- each `TaskContext` gets its own findings;
- the shared context gets the findings used by several tasks, or by none;
- the reviewer context gets the approved plan text plus the architecture,
  constraint, risk, and assumption findings.

The plan id is deterministic (`plan_<session>_r<revision>`), and the job created at
claim time carries it.

## Downstream use

- **Contract:** `TaskContractor.initial_contract(task_context=...)` adds planned
  files and symbols to the contract scope. Planner facts become `known_context`.
  `do_not_modify` is advisory context, never `excluded_files`, because excluded
  files cannot even be read.
- **Coder prompt:** `TaskExecutionService` adds an `APPROVED WP PLAN CONTEXT` block
  and tells the coder to verify the listed files first and search more broadly only
  when the context proves insufficient.
- **Reviewer:** after each task, CodePartner appends `wp_reviewer_context_events`
  (files changed, decisions, captured tests, retries, deviations from the planned
  files, unresolved gaps). The review receives the frozen `ReviewerContext` plus
  these events as a private task attribute, never as task text. Planned files are
  optional read candidates, because a planned file may legitimately not exist.

## Persistence and raw telemetry

All of this lives in the routing SQLite database:

| Table | Contents |
|---|---|
| `wp_planning_sessions` | WP, fingerprint, complexity (and source), planner model and effort, status, revision, current draft, discussion |
| `wp_planning_turns` | one row per planner call (`plan`, `discuss`, `repair`): model, usage, tool metrics (calls, reads, searches, unique files, refused over budget), latency, error |
| `wp_approved_plans` | the frozen `ApprovedWPPlan` JSON |
| `wp_reviewer_context_events` | the per-task execution record fed to the reviewer |
| `implementation_jobs.plan_id` | links a job to the plan it executed |

Coder usage and reads are already recorded per job (`execution_attempts`,
`telemetry_aci_attempts`). Joining these through `plan_id` and `job_id` gives the
planner-vs-downstream cost comparison.

### KB-first discovery

The shared run-local policy now serves planner, coder and reviewer through the
normal request-scoped tool dispatch. Generated KB lookups and IDE reads undergo
the same registry, adapter, stop, journal, budget and reviewer-path checks as
direct agent calls. Set top-level discovery.enabled to false to disable it.
discovery.search_gate_max defaults to five. See
[Codebase Memory](codebase-memory.md) for search interception, current-source
verification and edit freshness.

Whenever Codebase Memory tools are offered to the planner, in any discovery mode:
- The planner prompt carries a KB-first discovery order: prefetched context, then
  `kb_search_graph` / `kb_trace_path`, then `kb_get_file_outline` before reading a
  code file, then ranged `read_file` for the needed symbols. Text and regex search
  are reserved for what the graph cannot answer. Current IDE source stays
  authoritative; the KB is the map.
- **Outline gate.** A broad read (from the start, no end line, no `limit` or one
  above `discovery.outline_gate_lines`, default 120) of a code file that
  has not yet been outlined, range-read or gated returns the file's KB outline
  instead, with "Read not performed …". Repeating the same read returns the file.
  - Served outlines count as KB calls, not reads, and are served even when the
    read budget is exhausted.
  - These pass straight through to the IDE: non-code files, files the index lacks
    (e.g. `.venv`), files whose outline ends before `outline_gate_lines`, outline
    errors, and full reads after the planner's own outline or ranged read.
  - Each turn records `outline_gate: {served, passthrough: {repeat, after_outline,
    small, not_indexed, non_code, error}}` in `tool_metrics`.
- The read-budget refusal no longer points to Codebase Memory as an after-budget
  fallback.

A stopped planning run is recorded too: its turn row has `error: "cancelled"`,
`tool_metrics.cancelled: true`, and the metrics collected up to the stop.

### Discovery telemetry (`discovery`)

The top-level default is `mode: advise` for planner, coder and reviewer; soft
thresholds never refuse a call. `mode: observe` records without advisory text.
Set `enabled: false` to turn KB-first off for every role. Planner turns record
`tool_metrics.discovery`; coder attempts record `metrics_json.kb_first`, and
reviewer attempts record the same with plan and job identifiers. Actual
provider usage remains separate from estimated added-context tokens.

It records:
- `retrieval_levels`, counting calls at each level:
  - **L1:** KB, search, listing and symbol calls;
  - **L2:** ranged reads, and whole reads whose *returned* response is under `small_file_bytes`;
  - **L3:** other whole reads.
- `l3_direct` / `l3_escalated`: an L3 read is escalated when an earlier L1 or L2 call targeted the same path.
- `evidence_strength`: reads grouped as strong, moderate, weak or none, by why the path might matter:
  - **strong:** WP text, or KB prefetch symbol matches;
  - **moderate:** searches or graph calls aimed at a WP identifier or a strong path;
  - **weak:** broad listings and keyword matches.

  Reading or outlining a path never adds evidence for it. `unexplained_reads` counts weak and none reads, with up to 20 example `unexplained_paths`. "Unexplained" does not mean irrelevant.
- `kb_prefetch`, `kb_tools_offered`, `kb_used` and `kb_fallback`.
- `est_prompt_tokens`, `est_tool_result_tokens` and `est_added_tokens`: newly added context estimated from delivered UTF-8 bytes / 4. Provider-reported `usage` is a separate, cumulative figure.

With `mode: advise` (the default), the planner prompt also gets planner-only
discovery guidance: outline, then targeted ranges, then whole files only when
needed, plus a sufficiency checklist. The first time a per-tier soft threshold
is reached (`tool_calls`, `full_reads`, `est_added_tokens`,
`repeated_or_unexplained_reads`), one `[Planning advisory, not a limit]` line is
appended to that tool result and recorded in `threshold_crossings`. Advisories
never refuse a call; the hard caps above stay in force. The guidance shared with
coders and reviewers also receive the shared KB-first guidance. The baseline and rollout status are in
`openspec/changes/optimize-planner-discovery/design.md`. To compare benchmark arms:

```
uv run python scripts/bench_planner.py --arm baseline=wpplan_a,wpplan_b,wpplan_c \
    --truth changed_files.txt --rubric rubric.json
```

## Future compatibility

The artifacts in `agent/planning/models.py` are versioned (`schema_version`),
serializable dataclasses with stable ids. `ApprovedWPPlan.provenance` records the
session, revision, model, effort, and complexity source. A knowledge base or
provenance layer can index them without changing the planning flow.

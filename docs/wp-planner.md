# Work Package planner

Every Work Package implementation request ("Let's do WP4", "implement WP1 and WP2")
goes through a planning stage first, in every mode except review. A strong model
investigates and plans the whole WP once, then hands structured context to each
task's coder and to the reviewer. Task execution, coder routing, and review are
unchanged; they gain context. Task selectors (`WP4-T2`), EPIC selectors, and the
legacy `/implement` command keep the direct flow.

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

## Future compatibility

The artifacts in `agent/planning/models.py` are versioned (`schema_version`),
serializable dataclasses with stable ids. `ApprovedWPPlan.provenance` records the
session, revision, model, effort, and complexity source. A knowledge base or
provenance layer can index them without changing the planning flow.

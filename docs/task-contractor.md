# TaskContractor

TaskContractor is the Tasker-owned boundary between OpenSpec planning and coder
execution. It keeps OpenSpec, model routing, Claude/Codex providers, mutation
journals, and independent review intact while making each coder call finite and
auditable.

## Execution lifecycle

For a new implementation task, Tasker derives an immutable `TaskContract` from
the OpenSpec task. The contract identifies its goal, effective complexity,
acceptance criteria, files and symbols, reference context, exposed tools,
the identical-call limit, and escalation conditions.

The coder receives only that contract and its selected tools. It returns a
schema-v2 `CoderResult` with one of these outcomes:

- `SUCCESS`
- `BLOCKED_CONTEXT`
- `SCOPE_TOO_LARGE`
- `WRONG_COMPLEXITY`
- `IMPLEMENTATION_FAILURE`

Every result returns to Tasker. A non-success result can produce a new immutable
contract revision, up to the cumulative execution-attempt limit. A success result
continues to the existing review boundary. A reviewer failure is recorded as
evidence and Tasker converts it into a new contract; reviewer prose is not used as
an open-ended coder prompt. Existing `PASS`, `BLOCK`, and `UNVERIFIED` work-package
semantics remain unchanged.

Contracts, results, and structured escalations are stored in the routing SQLite
database. Jobs created before this feature can complete through the legacy
handoff path.

## User resolution boundary

A pre-review execution-policy violation enters `awaiting_contract_resolution`,
which is distinct from a Reviewer failure (`awaiting_review_resolution`). Replies
are interpreted against that state and rendered as a tokenized proposal; nothing
resumes until the user confirms it. The valid actions are `RETRY_CONTRACT`,
`EXPAND_CONTRACT_SCOPE`, and `STOP_JOB`; the review-only retry is not available
from this state.

On confirmation, `retry_contract` loads the previous contract and escalation, and
Tasker decides whether to preserve, narrow, or expand the scope: paths the user
explicitly excludes are removed, paths the user confirmed adding are included,
and files the failed attempt requested stay outside unless added. The child
revision records the guidance and scope decision in `known_context`, and the
resolution is audited in `implementation_resolution_actions`. It does not
fabricate or require a completed review. A confirmed retry also starts a fresh
execution-attempt budget.

See [workflow actions](workflow-actions.md) for interpretation, confirmation
tokens, revalidation, and migration of older paused jobs.

## Discovery contracts

These remain for existing OpenSpec tasks and investigations that are themselves
deliverables. New changes do not need preparatory discovery tasks before a
Work Package implementation; its planner performs that investigation.

Tasks whose OpenSpec `preferred_capability` is `read`, `research`, `inspect`,
`inspection`, `analysis`, `review`, or `verification` investigate rather than edit.
Proving that something is absent (for example "no MLflow integration exists")
needs a repository-wide content search, which a file-scoped contract cannot
provide. Tasker therefore gives these tasks a read-only discovery contract:

- `read_only=True` and no primary files: reads are allowed anywhere, writes nowhere;
- operations: file read, file-name search, content search (`file_text_search`),
  and project inspection such as directory trees; verification tasks also keep
  focused pytest checks;
- no generic `execute_tool` dispatcher unless shell checks are allowed;
- files named by the task are recorded as starting points, not boundaries.

The Coder is told to search file contents to establish presence or absence and
to report SUCCESS only with the searches it ran cited in `verified`. Revisions
after Coder results, reviews, or user confirmation keep the contract read-only.

User exclusions such as "ignore the .env" are stored in `excluded_files` and
accumulate across revisions. The guard denies reading or writing them in every
contract and removes search hits in excluded files, so their content never
reaches the Coder.

## Expected scope and guardrails

Tasker is deterministic: it has no model and does not inspect the repository. It
takes a task's files from paths written in the OpenSpec text, and OpenSpec tasks
rarely name every file a change needs. Treating that guess as a hard wall turned
each underestimate into a failed attempt or a user escalation. Contract files are
therefore *expected scope*: guidance for the Coder and the baseline for auditing,
not a limit.

The contract still filters the advertised MCP catalog before either provider
starts, and the shared MCP bridge enforces it for both direct PyCharm handlers and
IDE-client tool-call resumes:

- tool and operation allowlists, and focused pytest checks only;
- no writes to protected paths (`.codepartner/`, `openspec/`, `.git/`) and no
  reads or writes of secrets files (`.env*` other than `.env.example`);
- no access to files the user excluded;
- no writes at all in read-only discovery contracts, and no write whose target
  cannot be identified (matching the fail-closed mutation journal);
- the repeated identical-call limit and the four-attempt Tasker↔Coder cap.

Search results in excluded or secrets files are removed before the Coder sees
them. Write contracts include content search and project inspection so the Coder
can find what Tasker could not predict. Call-count budgets remain removed; every
call is measured in telemetry. A violation records an escalation and retains
journaled changes for inspection.

Writing outside the expected files is allowed and audited. Every write is already
journaled with a before-image, so it can be undone. Before review, CodePartner
compares each task's journaled changes with its latest contract and adds a
CodePartner scope audit to the Reviewer's context. The Reviewer judges whether each
file was needed for the active acceptance criteria; unrelated or unjustified
changes become implementation findings. Each case is recorded as `scope_drift` in
`implementation_resolution_actions` (contract, expected files, unexpected files),
and each attempt's `unexpected_writes`/`unexpected_reads` are stored in
`telemetry_attempt_summaries.extra_json`. This is the data for provenance and for
a planning Tasker that learns which task shapes need which files.

### Scope requests

The Coder can still return `scope_requests`
(`{"path", "operation": "create|modify", "reason"}`), and free-text
`requested_files` are reduced to their leading path. Tasker decides them by rule
(`TaskContractor.grant_scope`): paths must be relative, inside the project, not
excluded, not protected, not in a read-only contract, and at most five per
revision. Granted paths become expected files of the next revision. After a denied
tool or operation, Tasker handles a structured non-SUCCESS result instead of
escalating directly when it grants files or when the Coder reports
`SCOPE_TOO_LARGE`/`WRONG_COMPLEXITY` (its complexity step-up can add operations
such as symbol rename). Otherwise the task escalates with the reasons, as it does
for a SUCCESS claim after a denial, protected, secrets or excluded paths, shell
commands, repeated identical calls, and unparseable handoffs.

### Execution environment

Codex contract runs disable its native shell, web search, and MCP servers inherited
from user configuration. Claude already runs with built-in tools disabled. Both
providers therefore reach project state only through the request-scoped MCP tools.
The profiles in `agent/task_contracts/policy.py` set allowed operations and the
identical-call limit per complexity; they are code policy, not prompt suggestions.

## Telemetry

The top-level `telemetry` package owns telemetry models, SQLite tables, recording,
and aggregation. It uses `ROUTING_DB_PATH` or the existing default
`~/.codepartner/routing.sqlite3`.

Telemetry records identifiers and measurements: calls by category, unique files
read, returned character counts, repeats, calls before the first edit, meaningful
progress, attempt/review outcomes, latency, escalation reasons, and provider token
usage when available. It does not store prompts, tool arguments, source content,
credentials, or complete tool output. Verification output remains in the existing
evidence store and can be correlated through operational identifiers.

Telemetry is observational. Its recorder isolates storage failures so metrics can
never disable contract enforcement.

## Efficiency regression

Use `TelemetryAggregation.task_summary(task_id)` to compare repeated runs from the
same repository revision, model route, task, and acceptance criteria. Compare both
coder-only and complete workflow cost so Tasker or Reviewer work is not hidden.

The local routing history available during implementation contained no task marked
`simple`, so it could not supply an honest historical “previously expensive simple”
baseline. The focused `Update router` regression fixture currently verifies the
enforced side of the comparison (file scope, repeated-call limits, and telemetry
capture) and that in-scope calls are no longer budget-limited. Once an expensive simple task is observed,
replay its original revision and persist the before/after measurements rather than
inventing a baseline.

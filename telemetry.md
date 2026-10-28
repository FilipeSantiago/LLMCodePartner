# Code Partner telemetry

Code Partner stores local operational measurements so execution cost, boundedness,
reliability, and review outcomes can be evaluated. The dedicated `telemetry` package
owns measurement schemas and writes. Telemetry is failure-isolated: a telemetry write
failure is logged and must never fail a coding task.

By default, telemetry and routing audit records use the SQLite database at
`~/.codepartner/routing.sqlite3`. Set `ROUTING_DB_PATH` to choose another file. These
tables are never uploaded. Separately, optional OpenTelemetry tracing
(`OTEL_ENABLED=true`) exports execution traces to a self-hosted Langfuse; it adds to
this local telemetry and does not replace it (see
[docs/observability.md](docs/observability.md)).

## Dedicated telemetry package

### Tool activity

`telemetry_tool_activity` contains one row per bounded Coder tool call:

- job, task, contract, contract revision, attempt, and role identifiers;
- tool name and category;
- a deterministic call signature and repeated-call flag;
- project-relative paths identified by the execution guard;
- calls made before the first edit;
- success or failure class, output character count, duration, and whether the call
  made meaningful progress.

It intentionally does not contain prompts, source text, tool arguments, or complete
tool output.

### Attempt summaries

`telemetry_attempt_summaries` records one aggregate row for each Coder or Reviewer
attempt:

- final outcome;
- total calls and calls by category;
- unique files read;
- returned output size;
- repeated calls;
- calls before the first edit;
- meaningful-progress count;
- input, output, and cache tokens when the provider reports them;
- escalation reason and small structured metadata such as violated criteria.

These rows provide the main before/after measures for TaskContractor. In particular,
the regression coverage exercises a previously expensive simple-task shape. Reads,
searches, and no-progress calls are measured but not limited while call budgets are
removed; path scope and repeated identical calls are enforced and reported.

### Workflow events

`telemetry_workflow_events` records each step of the two-phase workflow
(interpret → propose → confirm/reject → execute) for both paused-job replies and
new requests such as `/implement`:

- stage: `interpreted`, `clarification`, `proposed`, `confirmed`, `rejected`,
  `expired`, `stale`, `superseded`, `corrected` (replaced by a `change <token>:`
  correction), or `executed`;
- proposal identifier, interpreted action, and source workflow state;
- whether the interpretation was deterministic or came from the model, plus the
  model name;
- confidence, ambiguity count, selector count, and whether guidance was present;
- interpretation latency and model input/output tokens when the provider reports
  them;
- confirmation latency (proposal creation to confirmation or rejection);
- the final transition attempted and its result (`started` or `failed: …`);
- the SHA-256 fingerprint of the message.

The raw user message, guidance text, and paths are never stored in this table.
The Ollama interpreter now returns `prompt_eval_count`/`eval_count` usage, so token
counts are populated for model interpretations and null for deterministic ones.
`IMPLEMENT_INTENT_MODEL` selects the model (default: `OLLAMA_MODEL`), and
`IMPLEMENT_INTENT_TIMEOUT_SECONDS` bounds the call (default: 15).

## Existing routing and execution audit data

The routing database predates the dedicated telemetry package. These records remain
the authoritative operational audit trail and can be joined to telemetry by job, task,
contract, or decision identifiers.

- `routing_decisions`: requested complexity, candidate routes, recommended and actual
  model/provider/executor, routing source and reason, fallback, gateway use, final
  status, timestamp, and latency.
- `execution_attempts`: each provider execution attempt, status, latency, error,
  gateway use, and provider-reported token usage.
- `implementation_jobs` and `implementation_job_tasks`: selected scope, execution
  instruction, conversation linkage, task order, current state, revision counters,
  errors, and Coder handoff report.
- `implementation_reviews`: Coder and Reviewer models, verdict, rationale, structured
  findings, and review evidence.
- `implementation_resolution_actions`: user choices after review or recovery.
- `implementation_evidence`: verification tool, arguments, success, result excerpt,
  payload hash/path, attempt, and metadata.
- `implementation_mutations` and `implementation_rollback_files`: changed targets,
  mutation stages, failures, expected hashes, and recoverable pre-edit snapshots.
- `blocked_operations`: capability checks that prevented an unsafe or unsupported
  operation.
- `refinement_sessions`: lifecycle and fingerprints for draft OpenSpec refinements.
- `task_contracts`, `task_contract_results`, and `task_contract_escalations`: contract
  revisions, structured Coder results, limit counters, unresolved criteria, and
  escalation evidence.

`workflow_action_proposals` is operational state rather than an analytics table.
It holds every proposal: the confirmation token, normalized action and guidance,
target job/position/task, expected workflow state and contract revision,
interpretation source, confidence, ambiguities, expiry, lifecycle status, the job
created by a confirmed `START_IMPLEMENTATION`, any proposal it corrected, and the
transition result. The original message is represented only by a SHA-256
fingerprint. Confirmation revalidates the expected state and contract revision in
the same transaction that claims the proposal, so repeated confirmation cannot
execute twice or create duplicate jobs. See [workflow actions](docs/workflow-actions.md).

## Diagnostic logs

Runtime logs also report transient diagnostics:

- safe request shape and tool-surface fingerprints;
- advertised and accepted tool names;
- run, call, trace, job, and decision identifiers;
- provider model usage when supplied;
- tool-result lengths, routing outcomes, and failures.

Logs are not the durable telemetry API and their retention depends on the process
environment. Debug logging may expose more operational detail than the measurement
tables, so production log access should be restricted.

## Data boundaries and retention

The dedicated telemetry tables are metadata-only. The operational audit tables have a
different purpose and may contain Coder reports, review rationale, verification
arguments/excerpts, and rollback file contents. Treat the routing database as
project-sensitive data.

There is currently no automatic retention, compaction, anonymization, or remote
export. Deleting or rotating the configured SQLite database removes both telemetry and
the operational recovery/audit history, so retention automation must distinguish
measurement rows from state needed by active jobs before it is introduced.

## Interpreting the measurements

Useful comparisons include:

- median tool calls, file reads, searches, and output characters by task complexity;
- calls before first edit and no-progress calls;
- repeated-call rate and contracts that reached a deterministic limit;
- execution attempts and Reviewer failures per completed task;
- token usage by role, model, and outcome when available;
- proposal confirmation/rejection/stale rates, corrections, deterministic versus
  model interpretation, and interpretation and confirmation latency;
- final outcome and escalation reason for tasks that did not converge.

Telemetry describes efficiency and control behavior; it does not replace acceptance
criteria, focused tests, or independent Reviewer evidence.

# Workflow actions and confirmation

`ConversationWorkflow` in `agent/workflow/controller.py` is the formal
conversation state machine. Its `ConversationSnapshot` derives one of four
phases from persisted facts: `ready`, `proposed`, `running`, or `paused`.
Execution is recorded by the confirmed proposal and the existing
implementation job.

| Current phase | Interpreted turn | Next phase | Effect before confirmation |
|---|---|---|---|
| `ready` or `paused` | Answer or clarification | unchanged | None |
| `ready` or `paused` | Action proposal | `proposed` | Save proposal only |
| `proposed` | Free-form correction | `proposed` | Supersede old proposal |
| `proposed` | Reject | `ready` or `paused` | Mark proposal rejected |
| `proposed` | Confirm | `running` (or immediate result) | Atomically claim and revalidate, then execute |
| `running` | Execution finishes or pauses | `ready` or `paused` | Persist job/review result |

The phase is derived from durable records rather than a mutable chat flag.
Confirmation tokens are conversation-scoped, expire, and are one-shot.

Every actionable chat message follows one two-phase workflow:

```text
User message → interpret against the current workflow state → show proposal
             → user confirms / rejects / changes → revalidate → execute
```

Before confirmation, CodePartner only reads workflow metadata, interprets the
message, and saves a pending proposal. It never invokes the Coder, Reviewer,
tests, or mutation tools before a proposal is confirmed. Explicit slash commands
(`/implement WP4 WP5`, `/apply`, `/spec …`) are proposals too. Ordinary questions
are answered through a read-only tool role and do not need confirmation.

The mode picker is a hint to the interpreter, not a required message format.
For example, `Retry the WP4 review`, `WP4`, and a numbered reply to an immediately
preceding menu can be interpreted in context. The model can propose only
registered actions; it cannot grant new permissions or bypass protected paths.
The user may confirm, reject, amend guidance with `change <token>: ...`, or send
a new free-form correction that replaces the pending proposal.

The code lives in `agent/workflow/`:

| Module | Responsibility |
|---|---|
| `states.py` | Paused states and the per-state action table |
| `intent.py` | `WorkflowIntent`, `WorkflowActionProposal`, and the bounded context |
| `interpreter.py` | Classifies one message into a valid action (deterministic phrases, then the cheap model) |
| `controller.py` | `ConversationWorkflow`: state snapshots, proposals, confirmation, revalidation, and dispatch |
| `migration.py` | Reclassifies paused jobs written by older releases |

## Responsibility boundaries

- **Intent interpreter**: identifies which workflow action the user wants. It
  never plans the implementation.
- **Tasker** (`TaskContractor`): decides contract scope and creates revisions.
- **Coder**: executes the confirmed contract.
- **Reviewer**: validates against acceptance criteria.
- **Workflow controller**: enforces valid transitions and confirmation.

## Paused states and valid actions

`awaiting_user_decision` is no longer written. Each pause has its own state:

| State | Meaning | Valid actions |
|---|---|---|
| `awaiting_contract_resolution` | Stopped before review: TaskContract violated or exhausted | `RETRY_CONTRACT`, `EXPAND_CONTRACT_SCOPE`, `STOP_JOB` |
| `awaiting_review_resolution` | Reviewer finished; a decision is needed | `RETRY_CODER_AFTER_REVIEW`, `ACCEPT_REVIEW`, `UNDO_IMPLEMENTATION`, `STOP_JOB` |
| `awaiting_coder_recovery` | Coder interrupted before finishing | `RESOLVE_CODER_RECOVERY` (retry Coder), `UNDO_IMPLEMENTATION`, `STOP_JOB` |
| `recovery_required` | Mutation journal incomplete | `RESOLVE_CODER_RECOVERY` (inspect, establish baseline), `STOP_JOB` |
| `awaiting_human_confirmation` | Reviewer asked a question | `ANSWER_REVIEWER_QUESTION`, `RETRY_REVIEW`, `STOP_JOB` |
| `awaiting_evidence_resolution` | Reviewer could not obtain enough evidence | `RETRY_REVIEW`, `STOP_JOB` |
| `awaiting_manual_resolution` | Legacy pause that could not be classified | `STOP_JOB` |

Outside a paused job, `START_IMPLEMENTATION` (OpenSpec-scoped work), `CODING`
(ad-hoc project work), `APPLY_SPEC`, `DISCARD_SPEC`, and `EXECUTE_COMMAND`
(other registered workflow commands) are available. `ANSWER` and
`ASK_CLARIFICATION` are never executable.

A contract escalation can never be routed to the review-only retry, because
`RETRY_CODER_AFTER_REVIEW` is not in its action table.

## Turn detection

Clients resend the whole conversation on every request, so tool results from
earlier turns stay in the history. Only tool results that *end* the history make
a request a tool continuation (`openai_request.trailing_tool_results`); a fresh
user message after them is a new turn and goes through the proposal boundary.

- When IDE MCP setup finishes, the user message that triggered it is processed
  exactly once (`_replay_original_turn`); a repeated delivery of the setup result
  reports that it was already processed.
- A tool result that no live run is waiting for (the run finished, timed out, or
  was stopped) is reported; it never starts a new agent.
- A fresh user turn never falls through to the raw coding agent. Without a
  conversation identity there is no proposal boundary, so CodePartner says so.

## Routing a fresh turn

Slash commands and confirmation replies are protocol. Every other user turn is
interpreted twice (`agent/workflow/reconciler.py`):

- **Context-free:** what the message requests on its own (`classify_fresh`; the
  classifier receives no workflow state).
- **Contextual:** whether it replies to waiting work, using the paused-task
  interpreter below. A message that names work outside the paused tasks
  (for example `implement WP5 and WP6` while WP4-T1 is paused) is not treated as
  a reply to them.

`reconcile` picks **new flow**, **continuation**, **question**, or **unclear**.
An explicit phrase or a reply naming its task is a continuation without a second
model call. When a model-read continuation competes with a clearly requested new
implementation, spec draft, or review, nothing is guessed: the user is asked to
name the selector or prefix the task id.

### Work Package planning

In every mode except review, a WP implementation request ("Let's do WP4",
"implement WP1 and WP2") starts an interactive planning session for the first WP
instead of a `START_IMPLEMENTATION` proposal. Further WPs are planned one at a time,
each after the previous WP's job completes.
Planning is read-only, so it runs without a token. While the session is active,
the conversation phase is `planning`, and:

- an approval (`approve`, `1`, `yes`, "looks good, go ahead", or
  `confirm <token>`) claims the pending `APPROVE_WP_PLAN` proposal for the
  *current* revision;
- `discuss` / `2` asks what to change; any other text is a discuss turn;
- a request naming a different WP supersedes the session; `never mind` / `stop
  planning` (or rejecting the proposal) abandons it.

Each revision supersedes the previous approval proposal. The claim marks it stale
unless the session is still active at the same revision. See `docs/wp-planner.md`.

## Interpretation of replies to paused tasks

1. Recognized phrases are mapped deterministically, relative to the task's state.
   For example, `submit to coder again: <guidance>` means `RETRY_CONTRACT` in a
   contract pause and `RETRY_CODER_AFTER_REVIEW` after a review. Existing phrases
   (`accept`, `undo`, `retry review`, `inspect recovery`,
   `establish new baseline: …`, `stop job`, `yes`, `no: …`) keep working.
2. Otherwise the cheap model (`IMPLEMENT_INTENT_MODEL`) receives only the current
   message, a bounded context block for the target task, the valid actions, and a
   strict JSON schema whose `action` enum contains only those actions. It does not
   receive the conversation history or repository access.

Example context for the `.env` case:

```text
Current job: 89fd…
Current task: WP4-T1
State: awaiting_contract_resolution (stopped before review: …)
Cause: secrets file: .env
Contract revision: 1
Files allowed: .env.example
Files requested: .env
Review completed: false
Valid actions:
- RETRY_CONTRACT: …
- EXPAND_CONTRACT_SCOPE: …
- STOP_JOB: …
```

The interpreter does not guess. It asks for clarification when:

- several tasks are paused and the message does not name one (for example
  `WP4-T1: try again: …`);
- the model reports low confidence or chooses `ASK_CLARIFICATION`;
- the chosen action is not valid for the task's state;
- a guidance-bearing action has no guidance, or scope expansion names no files;
- the model is unavailable or returns an invalid result.

## Proposals and tokens

A proposal lists the action, its deterministic expected effects, and a
confirmation token:

```text
I understood your request as:

- Action: Retry WP4-T1 with a revised TaskContract
- Tasker creates TaskContract revision 2 for WP4-T1
- Files in the contract: `.env.example`
- Keep `.env` outside the contract
- Coder guidance: Do not inspect or modify .env. Continue using .env.example.
- Resume job `89fd…`
- No completed review is required before the Coder runs

Nothing has executed yet.

Reply `confirm 7K4MQX` or `reject 7K4MQX`. To adjust it, reply `change 7K4MQX: <correction>`.
```

- `confirm <token>` executes, `reject <token>` cancels, and
  `change <token>: <text>` appends to the guidance and replaces the proposal.
- A bare `yes`/`y`/`confirm` or `no`/`n`/`reject`/`cancel` binds to exactly one
  waiting item: the single pending proposal, or the single Reviewer question when
  no proposal is pending. With several candidates (or none) CodePartner lists
  what is waiting and executes nothing.
- A new actionable message supersedes any pending proposal.
- Proposals expire after 30 minutes.

## Revalidation on confirmation

Confirmation claims the proposal in one SQLite transaction that checks:

- the proposal is still pending and not expired;
- the target task is still in the expected state;
- the latest TaskContract revision is unchanged;
- for `APPLY_SPEC`/`DISCARD_SPEC`, the referenced draft is still active.

For `START_IMPLEMENTATION`, the selectors are re-resolved first; if the task list
changed, nothing runs. Otherwise the claim and job creation happen in the same
transaction, so repeated confirmation can never create duplicate jobs.

If anything changed, nothing executes. When the action is still valid for the
task's current state, a refreshed proposal with a new token is shown; otherwise
the current paused tasks and their valid actions are listed. Repeating a confirmed
token reports that it was already confirmed and runs nothing.

## Switching between flows

Only one mutating flow may be active per IDE project. The project id is a hash
of the IDE project path from `ide_mcp.json`; jobs store it, so jobs started from
other chats in the same project are visible. Read-only actions (reviews, spec
explanations, `inspect recovery`) never displace anything.

When a mutating proposal is created, the project's other unfinished jobs are
attached to it:

- **live** (a run is executing in this process) or **orphaned** (recorded as
  active, but nothing executes it, for example after a restart) jobs are to be
  stopped. The proposal becomes one combined question:

  ```text
  CodePartner is currently working on WP4. Would you like to stop WP4 and start
  implementing WP5 and WP6? WP4's existing file changes will remain; they will
  not be rolled back.
  ```

  That reply is the confirmation for the whole switch, not the start of another
  confirmation loop. Until it is answered nothing is stopped or started.
- **paused** jobs are left paused and mentioned as such.

The claim re-checks, in the same transaction, that the project's active jobs are
exactly the ones the user agreed to stop. If one finished or another started
meanwhile, the proposal is stale and a refreshed one is shown.

A confirmed switch is recorded in `workflow_switches` and moves through
`switch_requested → stopping → stopped → started`
(`agent/workflow/switching.py`):

1. The old job's runs refuse new tool calls.
2. CodePartner waits (up to `CODEPARTNER_STOP_WAIT_SECONDS`, default 20) for
   writes the IDE may still be applying, so none escapes the mutation journal.
   If one does not finish, nothing is cancelled, the switch ends in
   `stop_failed`, the new job is cancelled, and the old job keeps running.
3. The job is marked `stopping`, its runs are cancelled and must acknowledge,
   and the job and its in-flight task are recorded as `stopped`. Files are kept;
   nothing is rolled back. A halted job never advances, reviews, or reopens,
   even if its original request finishes later.
4. Only then is the new job released (`pending_switch → queued`) and started.

Recovery is idempotent: a switch left unfinished by a crash is completed on the
next turn (stopping jobs become stopped, the never-started job is cancelled, the
switch becomes `interrupted`), and the conversation is told once.

## Turn telemetry

`telemetry_turn_events` records, per turn: conversation id, project id, turn type
(`user`, `tool_continuation`, `bootstrap_replay`), the route, summaries of both
classifications (action, confidence, counts, source), the proposal id, and the
outcome. The message is stored only as a SHA-256 fingerprint; prompts, guidance,
and file contents are never recorded.

## Contract retry

`RETRY_CONTRACT` and `EXPAND_CONTRACT_SCOPE` call
`ImplementationService.retry_contract`, which:

1. loads the previous contract and its latest escalation;
2. lets Tasker (`TaskContractor.revise_after_escalation`) decide whether to
   preserve, narrow (paths the user excludes, such as "ignore the .env"), or
   expand (paths the user confirmed adding) the scope;
3. saves the child contract revision with the guidance in `known_context`,
   including which files stay outside the contract;
4. records the decision in `implementation_resolution_actions`;
5. runs the Coder with the revised contract.

It does not require or fabricate a completed review.

Contract files are expected scope rather than a hard limit (see
[TaskContractor](task-contractor.md#expected-scope-and-guardrails)), so
`EXPAND_CONTRACT_SCOPE` is rarely needed. It adds expected files, which keeps them
out of the Reviewer's scope-drift audit. It cannot unlock protected, secrets, or
excluded paths.

## Migrating paused jobs

On startup, paused tasks from older releases are reclassified:

| Old status | Condition | New status |
|---|---|---|
| `awaiting_user_decision` | latest escalation is a pre-review escalation newer than any completed review | `awaiting_contract_resolution` |
| `awaiting_user_decision` | a completed review exists | `awaiting_review_resolution` |
| `awaiting_user_decision` | interrupted Coder record exists | `awaiting_coder_recovery` |
| `awaiting_user_decision` | otherwise | `awaiting_manual_resolution` |
| `awaiting_confirmation` | — | `awaiting_human_confirmation` |
| `review_pending` (task status) | — | `awaiting_evidence_resolution` |

The `review_pending` review *verdict* stored in `implementation_reviews` is not
changed. The migration is idempotent and runs from both the routing and the
TaskContract persistence initializers, so table creation order does not matter.
A job that was stuck after a contract escalation becomes resolvable through the
workflow above.

# Model routing

`/implement WP1` plans the Work Package before creating a job. After approval,
each task selects a model from its `simple`, `medium`, or `complex` candidate tier
and starts the selected authenticated coding CLI. `/implement EPIC1` plans its WPs
in sequence. Task selectors and the legacy `/run` command direct users to the
parent Work Package. Tasks remain the routing unit, and the assembled WP is the
primary validation and review boundary.

## Configuration

`config/model-routing.yaml` is the application policy:

- `routing.tiers` defines the only allowed model aliases for each complexity.
- `models` maps each alias to a provider, executor, configurable CLI model, and
  gateway policy.
- `gateway_mode: preferred` attempts agentgateway first, then retries the same
  model directly after an operational gateway failure. This applies to coding,
  WP planning, and review calls.
- `gateway_mode: required` never bypasses the gateway; `disabled` never uses it.
- The bundled subscription-only configuration disables gateway attempts for every
  Codex alias: direct Codex login works without a separately scoped Responses API
  credential. Claude retains gateway-first routing.

The vLLM configuration in `infra/semantic-router/config.yaml` mirrors these
aliases. At runtime Code Partner calls vLLM's route-preview management endpoint,
then rejects any selected alias outside the local tier.

### Roles and capability floors

Coder, reviewer, and WP planner route from separate tables. Each model carries a
role-independent `capability` rank. The configured `roles.<role>.min_capability`
sets a floor for coders and reviewers; the planner has no floor by default:

- **Coder:** `routing.tiers`, keyed by task complexity (`simple|medium|complex`).
- **Reviewer:** the `reviewers` pairing table; fallbacks below the reviewer floor
  are skipped. The opposite provider is preferred; if its eligible models cannot
  run, an eligible same-provider model reviews with the same evidence requirements.
- **Planner:** `planner.tiers`, keyed by Work Package complexity
  (`low|medium|high|complex`), with optional `reasoning_effort` overrides and a
  `max_tool_calls` investigation budget.

Planner tiers may use lower-capability models, including Haiku in the low tier.
An optional explicit `roles.planner.min_capability` is still honored. A config
without `roles` or `planner` still loads; only `codepartner-plan` needs the planner
section. See `docs/wp-planner.md`.

## Safety and fallback

The router uses configured order as a deterministic fallback when Semantic Router
is unavailable or returns an invalid selection. Failed gateway attempts retry the
same model directly before the next candidate is tried, except a Claude
subscription/session-limit denial, which skips the redundant same-credential retry.
Operational coder failures
then try an eligible agent from the other provider, even after a successful IDE
write or a tool-result round trip. The next agent first checks the mutation
journal and current files; it does not replay completed tool calls. If no agent
can finish, the job pauses for a chat retry. Reviewer provider and output-format
failures also try another eligible reviewer. Unfavorable findings and failed
evidence checks remain review results. If no reviewer can run, review remains
pending/UNVERIFIED.

Approved implementation-job Codex runs preapprove only their private, role-filtered
IDE MCP tool list. The bridge still enforces allowed operations and journals writes;
standalone Codex calls keep their prior approval behavior. A missing, invalid, or
blocked coder handoff pauses the saved task for recovery. It cannot be labeled
implemented or advance a dependent WP task.

A clear `retry` reply for one paused coder or reviewer runs that saved job in a
streaming IDE session. If several jobs match, CodePartner asks which task to retry.
The gateway routes individual provider requests; CodePartner owns CLI agent
switching and recovery from persisted IDE tool state.

To repair the historical WP3 run that advanced blocked handoffs, use
`python -m agent.routing.repair_blocked_handoffs --list`, then preview the exact
job with `--job-id JOB_ID`. Only after reviewing the preview, add `--apply`.
The command refuses active jobs or any later mutation/review history and preserves
the original reports and routing audit.

## Operations

The current stack uses host networking because the PyCharm MCP server is bound to
host loopback. `semantic-router` and `agentgateway` are therefore explicit
loopback aliases inside Code Partner rather than normal Compose DNS names. Their
ports must remain distinct from PyCharm and the API.

Routing audit data is stored in `/home/codepartner/.codepartner/routing.sqlite3`
inside the existing persistent `agent-home` volume. It records model decisions,
attempts, gateway usage, timing, and available token usage but never prompts or
credentials.

Implementation jobs, ordered task statuses, refinement drafts, routing evidence,
and WP review outcomes are persisted in the same database. A WP executes or
reuses all applicable tasks before review. The reviewer returns `PASS` only with
sufficient contract evidence, `BLOCK` only for a concrete contract violation or
regression, and `UNVERIFIED` when review or infrastructure could not establish an
answer. `UNVERIFIED` is not sent to the coder as an implementation failure. A
failed task still stops the remaining queue, avoiding implicit execution of
dependent work.

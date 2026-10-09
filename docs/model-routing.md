# Model routing

`/implement WP1-T1` resolves an existing OpenSpec task, selects a model from the
task's `simple`, `medium`, or `complex` candidate tier, and starts the selected
authenticated coding CLI. `/implement WP1` expands an entire work package in
dependency order, while `/implement EPIC1` expands its WPs. Tasks remain the
routing unit, but the assembled WP is the primary validation and review boundary.
The legacy `/run` syntax is normalized into the same durable implementation job;
there is no parallel executor workflow.

## Configuration

`config/model-routing.yaml` is the application policy:

- `routing.tiers` defines the only allowed model aliases for each complexity.
- `models` maps each alias to a provider, executor, configurable CLI model, and
  gateway policy.
- `gateway_mode: preferred` attempts agentgateway first, then retries the same
  model directly after an operational gateway failure. This applies to coding,
  WP planning, and review calls.
- `gateway_mode: required` never bypasses the gateway; `disabled` never uses it.

The vLLM configuration in `infra/semantic-router/config.yaml` mirrors these
aliases. At runtime Code Partner calls vLLM's route-preview management endpoint,
then rejects any selected alias outside the local tier.

### Roles and capability floors

Coder, reviewer, and WP planner route from separate tables. Each model carries a
role-independent `capability` rank, and `roles.<role>.min_capability` sets a floor
per role:

- **Coder:** `routing.tiers`, keyed by task complexity (`simple|medium|complex`).
- **Reviewer:** the `reviewers` pairing table; fallbacks below the reviewer floor
  are skipped. The opposite provider is preferred; if its eligible models cannot
  run, an eligible same-provider model reviews with the same evidence requirements.
- **Planner:** `planner.tiers`, keyed by Work Package complexity
  (`low|medium|high|complex`), with optional `reasoning_effort` overrides and a
  `max_tool_calls` investigation budget.

Complexity is role-relative. The loader rejects a config whose `low` planner
candidates are not strictly more capable than every `simple` coder. A config
without `roles` or `planner` still loads; only `codepartner-plan` needs the planner
section. See `docs/wp-planner.md`.

## Safety and fallback

The router uses configured order as a deterministic fallback when Semantic Router
is unavailable or returns an invalid selection. Failed gateway attempts retry the
same model directly before the next candidate is tried. A different coding model
is not started after a successful JetBrains write tool result, preventing a
fallback agent from replaying partial edits. Reviewer fallback is only for
provider execution failure, not unfavorable findings or failed evidence checks.
If no reviewer can run, review remains pending/UNVERIFIED.

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

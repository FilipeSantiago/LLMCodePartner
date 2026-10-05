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
- `gateway_mode: preferred` attempts agentgateway then retries direct execution
  with the same authenticated CLI session.
- `gateway_mode: required` never bypasses the gateway; `disabled` never uses it.

The vLLM configuration in `infra/semantic-router/config.yaml` mirrors these
aliases. At runtime Code Partner calls vLLM's route-preview management endpoint,
then rejects any selected alias outside the local tier.

## Safety and fallback

The router uses configured order as a deterministic fallback when Semantic Router
is unavailable or returns an invalid selection. Failed gateway attempts retry the
same model directly before the next candidate is tried. A different model is not
started after a successful JetBrains write tool result, preventing a fallback
agent from replaying partial edits.

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

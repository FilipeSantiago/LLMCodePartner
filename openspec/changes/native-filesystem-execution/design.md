# Design: Native filesystem execution and selective KB

## Project identity

The IDE MCP session remains the conversation anchor: OpenSpec artifacts, the
planner and the reviewer use it unchanged.

`IdeMcpSession.workspace_id` binds the conversation to a registry entry. It is
resolved, in order, from:
1. an explicit `/project use`;
2. `.codepartner/config.yaml` `project.id`;
3. the IDE project path matched against `host_path`.

`/project use` is refused:
- for a `host_path` that differs from the IDE project, which keeps artifacts and
  code in the same project;
- while the conversation has an unfinished job.

Each request activates the binding in a context variable. Background runs inherit
it from the request that starts them.

`agent.workspace.validate_workspace` accepts only:
- paths under `CODEPARTNER_WORKSPACE_ROOT`;
- paths whose realpath equals the configured path;
- existing directories that are mount points.

A registered entry that is not mounted reports that the container must be
recreated.

## Native attempt

`TaskExecutionService._run` calls `plan_native(provider, excluded_files)` for every
routed attempt, because fallback attempts may use another provider. A native
attempt receives:
- the native task prompt;
- an `ExecutionContext.native`, an `agent.workspace.NativeRun` carrying:
  - the workspace;
  - toolchain environment and notes;
  - `start` and `finish` hooks bound to a per-attempt `RollbackJournal`.

Providers copy the `NativeRun` onto `Run.native`. In native mode they:
- filter bridged specs with `native_bridged_specs`;
- disable the IDE `AgentToolInterface`;
- configure the CLI's working directory and built-in tools.

The engine (`mcp_bridge.server._run`) wraps the strategy:
- `start()` runs before the CLI. A failure ends the run before any write.
- `finish()` runs exactly once afterwards, including on cancellation. It emits a
  `MutationEvent` per changed path, so the existing "no blind retry after
  material output" rule still holds.

The hooks live in the run's background task because a run can pause for bridged
IDE tools across HTTP requests.

## Git baseline journal

The baseline holds:
- `git ls-files -s` (staged entries, all stages);
- every tracked and untracked, non-ignored regular file or symlink, with its
  `lstat` signature, permission bits and raw blob id
  (`git hash-object -w --no-filters`);
- `HEAD`.

Only objects are written: no ref, index or work-tree change.

After the run, a path is recorded when its existence, bytes, permissions or index
entry changed. The cheap `lstat` signature filters out untouched files first.
Rows go into `implementation_rollback_files`:
- `expected_hash` is the sha256 of the current bytes;
- the new `native_json` column holds the blob id, permissions, symlink flag,
  baseline index entries and whether the index changed.

`INSERT OR IGNORE` keeps the first preimage per task position, so revisions
restore the original state. A moved `HEAD` is recorded as an integrity issue,
because commits are not rolled back.

`restore()` first checks every native path's current hash against
`expected_hash`; on any conflict it changes nothing. It then:
- rewrites bytes and permissions, or recreates symlinks;
- removes files the task created;
- resets each changed path's index entries through `git update-index --index-info`.

IDE-journaled rows keep the existing restore path.

## Selective KB policy

`DiscoveryConfig.policy` gates `KbFirstPolicy` per role:

| Role | Gated under `selective` | Gated under `kb_first` |
|---|---|---|
| Planner | Yes | Yes |
| Coder | No | Yes |
| Reviewer | No | Yes |

An ungated policy keeps on-demand KB tools, snippet verification and telemetry,
but never intercepts searches or reads, blocks edits or adds advisories. Coder and
reviewer prompts use `KB_SELECTIVE_GUIDANCE`. `KB_READING_GUIDANCE` remains for
planners and for `kb_first`.

## Not changed

- The router.
- OpenSpec.
- KB infrastructure and refresh.
- JetBrains MCP and the conversation/bootstrap flow.
- The planner and reviewer tool paths.
- IDE-mode coder behaviour.

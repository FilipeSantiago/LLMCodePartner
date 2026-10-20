# Proposal: Native filesystem execution and selective KB

## Why

Today every coder read, search, edit and test run is a round trip through the
IDE bridge:
- Claude Code runs with no built-in tools.
- Codex runs read-only, with its shell disabled.

On top of that, the KB-first policy intercepts coder and reviewer searches and
reads. Both add tool calls and context to ordinary implementation work.

Claude Code and Codex already have capable native file and shell tools. Code
Partner should:
- let them use those tools directly in mounted projects;
- keep the IDE for IDE-specific work;
- treat the KB as an on-demand architectural source.

## Scope

**In:**
- multi-project bind mounts into the existing container;
- a project registry;
- native coder execution for Claude and Codex;
- native change journaling for review and undo;
- the selective KB policy for coders and reviewers;
- docs, tests and a smoke test.

**Out:**
- benchmarking, which the maintainer runs separately;
- host-side services, SSH and execution workers;
- new databases;
- Security Gateway, Provenance and Context Assembler work.

## What Changes

- `config/projects.yaml`, read by `agent/workspace.py`: a hot-reloaded registry
  mapping a project id to a validated `/workspace` mount, with
  `execution: auto | native | ide`.
- Conversations bind to a registry entry through the existing
  `.codepartner/config.yaml` `project.id` or the IDE project path. `/project`
  shows, lists and binds.
- Coder attempts run natively when the project is mounted, is a git work tree, the
  provider is allowed, and the contract excludes no files. In `auto` mode they
  otherwise use the IDE path; in explicit `native` mode they report an error.
- Claude runs with `cwd` and built-in tools. Codex runs with `--cd`, its shell and
  the `workspace-write` sandbox, which is never widened automatically.
  Overlapping IDE file tools are not bridged.
- Codex's bubblewrap sandbox cannot create namespaces in the default container
  (tested), so `CODEPARTNER_NATIVE_PROVIDERS` defaults to `claude`. Codex stays on
  the IDE path until you opt in.
- `RollbackJournal` gains a git baseline: exact pre-task bytes, permissions,
  index entries and HEAD. Native changes are recorded in the existing rollback and
  mutation tables, so review and undo are unchanged.
- `discovery.policy: selective | kb_first`, default `selective`. KB-first gates
  stay with the WP planner; coders and reviewers use the KB on demand.
- Docker: host UID/GID build arguments, `/workspace`, a live read-only `./config`
  mount, `compose.projects.yaml`, and `scripts/register_project.py`.

## Work Packages

- **WP1 — Project registry and mounts** (feature)
- **WP2 — Native Claude/Codex execution** (feature), depends on WP1
- **WP3 — Selective KB policy** (feature)
- **WP4 — Tests, smoke test and docs** (feature), depends on WP1–WP3

## Scope update (2026-10-14): filesystem first for all agents

The filesystem became the main path for every agent, not only coders:
- default execution mode `native`; `CODEPARTNER_NATIVE_PROVIDERS` defaults to
  `claude,codex`, and Codex runs without its own sandbox (the container is the only
  boundary, and every mounted project is reachable);
- the WP planner, reviewer, chat answers and `/review` run as read-only native
  sessions: Claude gets only read tools; changes the agent reported writing are
  restored afterwards and fail the run. This is rollback, not a sandbox;
- the reviewer service reads its snapshot from disk; OpenSpec artifacts are read and
  written on disk;
- native test commands become execution evidence, and Code Partner runs required
  tests in the workspace itself;
- excluded files no longer force the IDE path: changes to them are restored and fail
  the attempt;
- KB-first gates do not apply to native runs; the KB is used on demand.

The IDE MCP bridge remains for IDE-specific extras and for `execution: ide`.


## Scope update (2026-10-14): undo removed

Undo is removed for now. The git baseline no longer writes objects
(`hash-object` without `-w`) and keeps no preimages; it only detects which paths
changed, for the review manifest and evidence. Read-only runs and excluded-file
changes fail with the paths and are not reverted. Journal-recovery states
(`inspect recovery`, `establish new baseline`) are gone; unverifiable changes are
reported to the reviewer instead of halting the job. The "Exact native undo"
requirement in `specs/native-execution/spec.md` no longer applies.

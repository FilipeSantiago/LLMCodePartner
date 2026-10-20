# Tasks: Native filesystem execution and selective KB

## 1. WP1 — Project registry and mounts

**Objective:** Mount several host projects into the existing container and bind conversations to them through a validated registry.

**Complexity:** medium

- [x] T1.1 Registry
  - Description: Add `agent/workspace.py`, which loads `config/projects.yaml` with env expansion, reloads it on change, and provides validation and `match`.
  - Status: completed
  - Complexity: medium
  - Definition of done:
    - Tests in `tests/test_workspace.py` reject unregistered, outside-root, unmounted and symlinked workspaces
- [x] T1.2 Conversation binding
  - Description: Add `IdeMcpSession.workspace_id` and the `/project [list|use <id>]` command, guarded by IDE project and active job. Activate the binding per request.
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - Tests in `tests/test_workspace.py` cover command parsing and matching
- [x] T1.3 Docker and registration
  - Description: Add UID/GID build args, `/workspace`, `safe.directory`, the read-only `./config` mount, `compose.projects.example.yaml` and `scripts/register_project.py`.
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - `docker compose config` validates with the example override

## 2. WP2 — Native Claude/Codex execution

**Objective:** Coder attempts use native CLI tools in the bound workspace, with journaled changes.

**Complexity:** complex

**Depends on:** WP1

- [x] T2.1 Native plan per attempt
  - Description: Add `plan_native` and `TaskExecutionService._native_run`. `auto` falls back to the IDE; explicit `native` reports an error.
  - Status: completed
  - Complexity: medium
  - Definition of done:
    - Tests cover a native context, auto fallback, excluded files and the explicit error
- [x] T2.2 Provider flags
  - Description: Claude gets `cwd`, built-in tools and read-only variants. Codex gets `--cd`, its shell and `workspace-write`, never widened. Overlapping bridged IDE file tools are filtered out.
  - Status: completed
  - Complexity: medium
  - Definition of done:
    - Tests cover Claude options, Codex native and IDE args, and the overlap filter
- [x] T2.3 Git-baseline journal
  - Description: Add `RollbackJournal.record_git_baseline` and `record_native_changes`, the native `restore`, the `native_json` column, and engine start/finish hooks.
  - Status: completed
  - Complexity: complex
  - Definition of done:
    - Exact-restore tests pass for staged, modified, untracked, created, deleted and chmod changes, conflicts, revisions and moved HEAD

## 3. WP3 — Selective KB policy

**Objective:** Make the KB optional for implementation while keeping it for planning.

**Complexity:** medium

- [x] T3.1 Policy and gates
  - Description: Add `discovery.policy`, `DiscoveryConfig.gated`, and `KbFirstPolicy(gates=…)` for coder and reviewer.
  - Status: completed
  - Complexity: medium
  - Definition of done:
    - Tests show no interception when ungated, and planner gates unchanged
- [x] T3.2 Guidance
  - Description: Add `KB_SELECTIVE_GUIDANCE` for coder and reviewer prompts. Native prompts are always context-first.
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - Prompt tests cover both policies

## 4. WP4 — Tests, smoke test and docs

**Objective:** Regression coverage, a live smoke test, and migration documentation.

**Complexity:** simple

**Depends on:** WP1, WP2, WP3

- [x] T4.1 Smoke test
  - Description: Add `scripts/smoke_native.py`. It checks native read, write of a uniquely named file, and `git status`, then removes only that file.
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - Passes in the container for Claude: read, write, command, then cleanup
    - Native run, journal and exact undo verified end to end with the real Claude CLI
    - Codex: live run blocked by the account usage limit on 2026-10-14. Its sandbox
      cannot start in the default container, so Codex is opt-in
- [x] T4.2 Docs
  - Description: Add `docs/native-execution.md`, and update the README, `.env.example` and the KB and planner docs.
  - Status: completed
  - Complexity: simple
  - Definition of done:
    - The docs cover setup, recreation, UID migration, container limits, Codex sandbox, undo semantics and security

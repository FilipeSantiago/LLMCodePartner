# Spec Delta

## Purpose

Run coder tasks with native Claude Code and Codex filesystem tools in registered,
mounted projects. Preserve IDE-specific tools, review, undo, and the KB as a
selective source.

## ADDED Requirements

### Requirement: Registered execution roots only
Native execution SHALL use only a workspace resolved from the project registry. It
SHALL be inside the configured workspace root, have no symlink in its path, and be
a mounted directory. Paths supplied by a model or a request SHALL NOT be used as
execution roots.

#### Scenario: Unmounted registration
- **WHEN** a registered project's workspace is not a mount point
- **THEN** native execution is unavailable, and the reason says the container must be recreated.

### Requirement: Auto and explicit native modes
In `auto` mode, a task that cannot run natively SHALL use the existing IDE path.
In explicit `native` mode, it SHALL fail with the reason and SHALL NOT fall back
to the IDE path or widen a sandbox.

#### Scenario: Explicit native without git
- **WHEN** a project with `execution: native` is not a git work tree
- **THEN** the task reports that native execution is unavailable, and nothing is executed.

### Requirement: Exact native undo
Before a native attempt, the system SHALL record the work tree's bytes,
permissions, staged entries and HEAD without changing them. Undo SHALL return
every path the attempt changed to that exact state and SHALL leave unrelated
paths untouched. Undo SHALL refuse every change if a journaled file was edited
after the attempt.

#### Scenario: Pre-existing user changes
- **WHEN** a task edits a file that had unstaged and staged user changes
- **THEN** undo restores both the user's work-tree content and the staged entry.

### Requirement: Native tool surface
Native runs SHALL NOT be offered bridged IDE tools that duplicate native file
read, search, list or write. IDE-specific tools and KB tools SHALL remain
available.

### Requirement: Selective KB
Under `discovery.policy: selective`, coder and reviewer runs SHALL NOT have
searches or reads intercepted by KB-first gates. KB tools SHALL remain available
on demand. The WP planner SHALL keep KB prefetch and gates.

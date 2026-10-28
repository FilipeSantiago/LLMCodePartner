# Inline execution

An agentic coder works out its own context: it reads the planned files and ranges one tool
call per turn, and every turn re-reads the whole conversation as cached prefix. The
planner has already found those files and ranges, so inline execution puts their current
source straight into the coder's first prompt. The coder then gets **no tools** and
replies once with exact replacements, which Code Partner applies and checks itself. The
reviewer works the same way: the service's verified snapshot and the plan's reference
source are its whole evidence.

A task becomes a handful of single-response calls instead of an open-ended session:

| Call | When |
|---|---|
| coder reply | always: every edit of the task in one JSON object |
| corrective reply | per rejected reply or failing check, at most `inline_fix_rounds` |
| context reply | once, when the coder names something the source left out |
| reviewer decision | always: every finding in one decision |
| revision (coder, then reviewer) | only after `needs_work` |

## Turning it on

`config/model-routing.yaml`, `execution:`

| Setting | Meaning |
|---|---|
| `coder_mode` | `agentic` (default): always the tool-using session. `inline`: inline for every planned task that fits. `auto`: inline only when every planned edit of an existing file has lines. Env: `CODEPARTNER_CODER_MODE`. |
| `inline_fix_rounds` | corrective calls per attempt (default 2) |
| `inline_context_chars` | source budget; a larger task runs agentic (default 120000) |
| `inline_lint_command` | optional command after the edits; `{files}` becomes the changed paths |

A task always runs agentic without an approved plan, without a native project (IDE path),
without edit ranges, or when its source exceeds the budget. The chat says which mode
each task got and why (`_inline: …_` / `_agentic: …_`).

## What the coder sees

`agent/planning/snippets.py` reads, from the files as they are now:

- **EDIT** regions: each planned edit range, already moved past earlier tasks' changes by
  `agent/planning/rebase.py`, widened by `planner.edit_margin_lines`;
- **NEW** files the task creates (named only);
- **REFERENCE** regions: the plan's references (a whole-file reference of a large file
  is cut to its first 400 lines);
- **earlier tasks' code**: for each earlier task of the job this task depends on, or
  whose changed files it uses, what that task changed, from the job's rollback preimages.
  A created file is included whole; a modified one by its changed ranges, grown to the
  enclosing top-level definition in Python files.

Files up to 300 lines are sent whole. Overlapping regions merge. Each line carries its
current number (`  12| code`).

## The reply

One JSON object (`agent/prompt/coder.py`, `InlineCoderPromptBuilder`):

```json
{"edits": [{"file": "src/app.py", "old_text": "exact current text", "new_text": "replacement"}],
 "new_files": [{"path": "src/new.py", "content": "whole file"}],
 "need_context": [],
 "summary": "...", "criteria": [{"criterion": "...", "status": "satisfied", "paths": ["src/app.py"]}],
 "remaining_gaps": [], "context_handoff": {"files": [], "symbols": [], "decisions": []}}
```

`agent/routing/patch_apply.py` resolves every edit in memory before writing anything:
`old_text` must occur exactly once (copied line numbers are stripped), edits to one file
apply in order, and only the task's edit files may change. One bad edit means nothing is
written and the coder gets the exact problems plus the current source.

`need_context` (`{"file": path, "lines": "a-b"}` or `{"symbol": name}`) is answered
once: Code Partner reads the file or finds the definition (`git grep`) and sends it. A
second request, or a reply that stays unusable after the fix rounds, ends inline
execution: the same task continues as an agentic session from the files as they are.

## Checks and evidence

After each applied reply Code Partner runs the task's `expected_tests` (explicit commands
as written; test paths as one `pytest` run, through `uv run` when the project has a
`pyproject.toml` or `uv.lock`) and `inline_lint_command`. Each run is execution evidence
for the reviewer. Failures go back to the coder with the updated source; checks still
failing after the last round are stated in the handoff summary, and the reviewer judges.

The attempt is journaled like a native run: a git baseline before the first write and
`record_native_changes` afterwards, so the review diff, the change journal and rollback
records are the same as for an agentic attempt. The handoff is the usual versioned JSON
with `changed_files` taken from what was written.

## Review

With `coder_mode` other than `agentic`, a native project and references that fit the
budget, the reviewer's model pass is tool-free: no native session, no IDE tools, no
reader. Its prompt holds the service's verified snapshot (unified diffs with surrounding
lines, new files in full) and the current source of the reviewed tasks' references, and
asks for every finding in one decision. A question the evidence cannot settle becomes a
verification finding owned by `evidence_service`.

## Measuring it

Each inline attempt records a telemetry row (`execution_mode: inline`) with `llm_calls`,
`fix_rounds`, `patch_apply_failures`, `need_context_requests`, `fallback_to_agentic`,
`checks_failed_final`, the summed usage and Claude's cost.
`TelemetryAggregation.task_comparison(job_ids)` reports them under the mode `inline` next
to native runs' exploration modes, so one work package run in each mode compares directly
on calls, cache-read tokens and cost. `need_context_requests` and `fallback_to_agentic`
count planner misses: context the plan should have carried.

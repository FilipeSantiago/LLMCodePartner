# Agent prompts

Every agent prompt is built in `agent/prompt`, one builder class per role:

| Role | Builder | Built by |
|---|---|---|
| WP planner | `PlannerPromptBuilder` (`agent/prompt/planner.py`) | `agent/planning/service.py` |
| Coder | `CoderPromptBuilder` (`agent/prompt/coder.py`) | `agent/routing/task_execution.py` |
| Reviewer | `ReviewerPromptBuilder` (`agent/prompt/reviewer.py`) | `agent/routing/review.py` |
| `reader` subagent | `ReaderPromptBuilder` (`agent/prompt/subagents/reader.py`) | `providers/subagents.py` |
| `checks` subagent | `ChecksPromptBuilder` (`agent/prompt/subagents/checks.py`) | `providers/subagents.py` |

All five inherit from `PromptBuilder` (`agent/prompt/base.py`) and share one interface;
the three parent agents (planner, coder, reviewer) inherit from `ParentPromptBuilder`,
which requires each role to supply its own reading rules. Each role's context dataclass carries
only the data its prompt renders.

```python
prompt = (PromptFactory.create(PromptRole.CODER)
          .with_context(CoderPromptContext(task, plan_context=plan, native=True))
          .build())
messages = [*system_prefix, prompt.user_message()]
```

- `PromptFactory.create(role)` returns a new builder for the role.
- `with_context(...)` sets the role's context. `build()` joins the role's `sections()` into a
  `Prompt`.
- `repair(problem)` is the follow-up asking for a corrected reply. Only the planner and the
  reviewer have one.

## Changing one role's prompt

A role's own text lives in its builder module, so editing it cannot change another role's
prompt.

The base contains no parent prompt wording. Each parent implements `bulk_rule()`,
`read_pipeline()`, `own_reads_rule()`, `delegation_rules()`,
`scope_exception_rule()`, `scope_block()`, and `checks_rule()`:
- The coder reads its edit
  ranges in one batch; the reviewer judges from the diff snapshot without re-reading it;
  the planner delegates source reads to the reader.
- The planner's bulk rule names `kb_query_graph` only when it is offered; otherwise it
  uses one `kb_search_graph` call across several file paths.
- With native Codex and scope enforcement enabled, the planner gets `planner_reader`
  for bounded source questions and `planner_declarations` for bulk symbol ranges.
  Its own shell and direct source tools are disabled. The reader is a separate
  tool-free Codex run that returns a concise `path:line` brief.
- Native prompts state the absolute working directory. References and scope exceptions
  go to the reader when one is offered.

Change a role's wording in that role's builder only.

`providers.subagents.exploration_rules(request)` returns the delegation rules of the parent
role behind `request.purpose` (`planner`, `task` or `reviewer`).

## Golden prompts

`tests/test_prompt_golden.py` pins the rendered text of every role's prompt in
representative scenarios (`tests/prompt_scenarios.py`) against `tests/fixtures/prompts`. Any
prompt change, intended or not, fails this test. After an intended change, regenerate the
fixtures and review the diff:

```bash
.venv/bin/python scripts/capture_prompt_goldens.py && git diff tests/fixtures/prompts
```

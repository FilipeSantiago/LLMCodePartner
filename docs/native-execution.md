# Native filesystem execution

The mounted project filesystem is Code Partner's main execution path. Claude Code
and Codex work on a project with their own file and shell tools, and Code Partner
reads and writes the project's files itself, instead of routing every read, search,
edit and test run through the IDE bridge. Projects are bind-mounted into the
existing `codepartner` container; no host-side service, SSH or execution worker is
involved.

This covers every agent and Code Partner's own file work:

| Who | Native behaviour |
|---|---|
| WP planner | read-only session (`Read`/`Grep`/`Glob` for Claude) |
| Coder (`/implement`) | writes in the project, journaled for review (no undo) |
| Reviewer | the service reads its snapshot from disk; the model pass is a read-only session |
| Chat answers, `/review` | read-only session |
| Ad-hoc chat coding | writes in the project, unjournaled, as on the IDE path |
| OpenSpec artifacts | `openspec/` and `.codepartner/` read and written on disk (atomic writes) |
| Test evidence | native test commands and Code Partner's own test runs are captured |

The IDE MCP bridge stays available **only for IDE-specific extras**: inspections
(`get_file_problems`), symbol lookups, `find_usages`, refactorings, formatting, run
configurations and notebooks. The IDE chat also still identifies the conversation
and its project. Codebase Memory (`kb_*`) stays available to native runs. Under
`discovery.policy: kb_first` their prompt says they MUST consult it before the first
read of each code file; nothing intercepts their reads (docs/codebase-memory.md).

## How a conversation finds its project

1. `config/projects.yaml` (the registry) maps a project id to its container
   workspace under `/workspace`.
2. When the IDE MCP session is configured, the conversation is bound to the entry
   whose id equals `project.id` in the project's `.codepartner/config.yaml`, or
   whose `host_path` equals the IDE project path.
3. `/project` shows the binding, `/project list` lists the registered projects
   with their mount status, and `/project use <id>` binds an entry that could not
   be matched automatically. It is refused for another IDE project's directory
   and while an implementation job of the conversation is unfinished.

**No setup tool call.** The IDE chat request names neither the project nor the
conversation. When exactly one registered, mounted native project has a
`.codepartner/ide_mcp.json`, Code Partner reads it from disk and configures the
session without asking the IDE. The conversation id then travels as a hidden
`<!-- codepartner:conv_… -->` comment at the end of the first reply, which the IDE
resends with the history (it is stripped before any model sees it). Sessions,
bootstrap ids and `/project use` bindings are stored in the routing database, so a
container restart or an idle gap rebuilds the session without a tool call. With
several such projects, an unregistered project or `execution: ide`, the first turn
of a new chat still reads `ide_mcp.json` through the IDE once.

Each IDE window has its own conversation and binding, so several mounted
projects can be used at the same time without restarting the container.
Execution roots come only from the registry. A path from a model or a request is
never used as a working directory.

## Registering a project

On the host:

```bash
.venv/bin/python scripts/register_project.py music-generation ~/Documents/research/MusicGeneration
.venv/bin/python scripts/register_project.py volley-vision ~/Documents/research/VolleyVision
```

The script:
- writes `config/projects.yaml` and `compose.projects.yaml`, both gitignored;
  `config/projects.example.yaml` and `compose.projects.example.yaml` show the format;
- mounts only the directory you name, refusing `/`, your home directory and their
  parents;
- tells you when the container must be recreated.

In `.env`, chain the mounts, and run as the host user that owns the projects:

```dotenv
COMPOSE_FILE=compose.yaml:compose.projects.yaml   # add :compose.kb.yaml etc. if used
HOST_UID=1000   # id -u
HOST_GID=1000   # id -g
```

Then apply the changes:

```bash
docker compose build codepartner     # only after changing HOST_UID/HOST_GID
docker compose up -d codepartner     # needed whenever a bind mount is added or changed
```

**Registry edits take effect immediately.** `./config` is mounted read-only into
the container and re-read when it changes. **A new mount needs the container to
be recreated**; `/project list` reports an entry that is registered but not
mounted.

### Changing the container UID

The image used UID 10001. If you set `HOST_UID`/`HOST_GID`, the existing
`agent-home` volume (CLI logins, routing database) must be handed to the new user
once:

```bash
docker compose run --rm --no-deps --user root codepartner \
  chown -R "$(id -u):$(id -g)" /home/codepartner
```

## Execution modes

`execution` in the registry, or `CODEPARTNER_EXECUTION_MODE` for every project,
selects the mode:

| Mode | Behaviour |
|---|---|
| `native` (default) | Native only. If it is unavailable, the turn fails with the reason; it never falls back silently. |
| `auto` | Native when available. Otherwise the IDE path, with the reason logged. |
| `ide` | The IDE MCP path. |

A conversation that is not bound to a registered project fails in the default mode
and tells you to register the project. Set `CODEPARTNER_EXECUTION_MODE=auto` or
`ide` to keep using unregistered projects through the IDE.

Native execution is available when:
- the conversation is bound to a registered project;
- the workspace is mounted under `/workspace` and is not a symlink;
- the workspace is the root of a git work tree, which change detection needs;
- the provider is listed in `CODEPARTNER_NATIVE_PROVIDERS` (default `claude,codex`).

**Excluded files** do not prevent native execution, but native tools cannot be
stopped from touching them. After the attempt, a change to an excluded path fails
the attempt with the reason; the change is journaled and left in place. This is
detection, not prevention.

### What the CLIs get

**Claude Code**
- runs with `cwd` set to the workspace;
- gets the built-in `Read`, `Grep`, `Glob`, `Edit`, `MultiEdit`, `Write` and
  `Bash` tools, with edits auto-accepted;
- has `WebFetch`/`WebSearch` disabled.

**Codex**
- runs `codex exec --cd <workspace> --dangerously-bypass-approvals-and-sandbox`
  with its shell (see below).

**Bridged tools.** IDE tools that duplicate native file access are not offered:
reads, searches, listings, creates, replacements, patches and deletes.
IDE-specific tools stay available: inspections (`get_file_problems`), symbol
lookups, `find_usages`, refactorings, formatting, run configurations, notebooks
and the IDE dispatcher. Codebase Memory `kb_*` tools stay too.

**KB-first.** Native agents keep Codebase Memory `kb_*` tools under every policy. Under
`discovery.policy: kb_first` their prompt requires a KB lookup before the first read of
each code file; it is a prompt rule, not an interception, so native execution is
unaffected. See docs/codebase-memory.md.

**Limits.** The planner's `max_tool_calls` / `max_file_reads` budgets count IDE and
KB tool calls only; native tool calls are counted in `native_tool_metrics` but not
capped.

**Prompt.** The coder prompt tells the agent to:
- work in the current directory;
- use the project's own commands;
- not commit, switch branches or stash.

The coder prompt, IDE or native, is scoped to the task. It carries the task contract, then
the parts of the approved plan that bear on the task:
- the work-package objective, architecture and decisions;
- the task's own files, symbols and existing implementation;
- findings that are unscoped or scoped to this task, and discovery leads;
- earlier task results, with dependencies marked;
- the task's required changes, the combined constraints and do-not-modify areas, and its
  validation.

Other tasks' requirements, files and findings stay out. The shared `EXECUTION RULES`
(`agent/routing/task_execution.py`) ask for minimal, non-repetitive inspection and targeted
validation first. They never forbid the reads or checks needed to resolve uncertainty, and
the agent can search for more context when the given context is not enough.

### Subagents: Delegation and source-reading rules

Every tool result a native run receives stays in the CLI's transcript and is read again on
every later turn. With `CODEPARTNER_READER_SUBAGENT=true` (off by default, for A/B tests),
native planner, coder and reviewer runs follow explicit source-delegation rules
(each parent role's `delegation_rules` in `agent/prompt`, see [prompts](prompts.md)): reading
source goes to a `reader` subagent on
a cheap model whose transcript stays outside the parent's context.

A subagent is an **executor, not an investigator**. The parent locates the code itself
(Codebase Memory first, then path-only `Glob`/`Grep` or shell searches) and hands over the full
context: exact paths and line ranges plus the questions for a `reader`, exact commands for
`checks`. Each subagent has one basic tool, never searches and never consults the KB; when a
request lacks something, it reports it under "Missing context:" instead of looking for it.

| Run | Subagents | Scope (the parent's) |
|---|---|---|
| Planner (read-only) | `reader` | unrestricted: as wide as the WP needs |
| Coder and test tasks | `reader`, `checks` | the task's planner-assigned files (`render.task_files`) |
| Read-only tasks, reviewer | `reader` | planner-assigned files plus the journaled changed files |

**Delegation rules (parent).** Locate the code, then delegate the reading: related files and
questions of one objective go in one request, independent requests go together. Every reader
request lists project-relative paths with line ranges; every checks request gives the exact
commands. On "Missing context:", locate what it names and send one follow-up. Coders and
reviewers locate code within their planner-assigned files first.

**Coders work from the plan.** A coder treats the approved plan context as its
investigation. Its scope is split into "Files to change" and "Read-only context" (the plan's
do-not-modify paths, `render.task_read_only_files`). It reads the files it changes itself,
only the ranges it edits, because Edit needs the agent's own Read. It never has a reader
transcribe them. It asks the reader only questions the plan leaves open, does not reread
files after editing, and gets the project file list for locating paths. Shell source dumps
stay forbidden. Telemetry counts `out_of_scope_paths` (reader-request paths outside the
scope), `whole_reads_of_edited` and `post_edit_rereads`. Claude coder runs write a 5-minute
prompt cache (`execution.prompt_cache_ttl`, default `5m`).

**No direct source reads (planner and reviewer).** Do not use shell commands, scripts or
parent file tools to display project-file contents, including source, tests and
configuration, even for known ranges. This includes `rg`/`grep` matched lines,
`cat`/`sed`/`head`/`tail`, and source diff output. Path-only discovery such as `rg --files`,
`rg -l` and `git diff --name-only` is allowed. Send source-inspection questions to the reader;
if it is unavailable or fails, report the evidence gap. This rule is prompt guidance, not a
shell-enforced restriction; telemetry observes command categories without blocking them.

**Subagent procedures.** `reader` (tool: `Read`): read every listed range in one response,
answer each question (≤ 300 words, `path:line` references), list anything unlisted it would
need under "Missing context:". `checks` (tool: `Bash`): run exactly the given commands and
return only exit codes, counts and failures; its commands still become test evidence.

**KB paths.** Codebase Memory indexes the project on the host (`host_path` in
`config/projects.yaml`), while native agents see it at its `workspace`. The bridge
(`mcp_bridge/bridge.py`) translates `kb_*` arguments to host paths and `kb_*` results to
workspace paths (`agent/workspace.py` `to_host`/`to_workspace`), so every path the parent
hands to a subagent exists in the container. Telemetry stores paths project-relative
(`NativeRun.relative`).

**Model.** One cheap model per provider in `config/model-routing.yaml`:
`subagents.reader_model: {claude: claude-haiku, codex: codex-luna-low}` (entries of
`models`, so their `cli_model` and `reasoning_effort` are reused) and `subagents.max_turns`
(`CODEPARTNER_SUBAGENT_MAX_TURNS` overrides it; default 5, since a fully specified
one-tool request needs 1–3 turns).

- **Claude Code.** `ClaudeAgentOptions.agents` with `background=False`. That flag alone
  does not keep a call in the foreground: Claude Code 2.1.287 starts an `Agent` call whose
  input omits `run_in_background` in the background, answers "Async agent launched", and
  delivers the brief later as a notification that ends the parent's turn (one more
  context re-read per call). Every run with subagents therefore has a PreToolUse hook that
  sets `run_in_background: false` on each `Agent` call and refuses subagent types other
  than the offered `reader`/`checks`; `CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1` also
  hides the built-in agents. The `reader` gets only `Read` and `checks` only `Bash`
  (`SUBAGENT_TOOLS`); a subagent can only use tools that exist in the session, so `Bash`
  is in the parent's tool set whenever `checks` is offered. With
  `subagents.enforce_scope: true`, the parent is offered role-specific file tools,
  `Agent` and the KB navigation tools (`KB_NAVIGATION_TOOLS`: paths, symbols and line
  ranges, no source). A PreToolUse policy checks paths, bounded reads and writes, denies
  the parent's own `Bash`, unrestricted shell, search and MCP source tools, and refuses a
  coder's second `reader` request unless its prompt states `SCOPE EXCEPTION:`.
  The coder's prompt sets one work order, each step in one response: KB navigation of
  the context files, at most one reader request, all planned edit ranges read together,
  then edits and one checks request.
  Every subagent tool is in the parent's allow list, so no subagent call is
  permission-denied. Runs use `strict_mcp_config`, so only Code Partner's MCP server loads,
  never the account's claude.ai connectors or user/project MCP config.
- **Codex CLI.** Roles via `-c agents.<role>.description=…` / `config_file=…` and
  `-c agents.max_depth=1`; the role file (a temp file for the process's lifetime) sets the
  reader model, effort and prompt. Its shell stays, so the one-tool rule is prompt-only. Checked
  with Codex 0.164: `-c` roles work despite `--ignore-user-config`; the parent `wait`s for
  the child's reply; spawns and waits appear as `collab_tool_call` items, while the child's
  own tool calls and usage do not appear in the parent's stream. Native Codex runs have no
  sandbox, so a `reader` is read-only by instruction; a read-only session still fails if
  anything changed. Not yet checked: whether a child reaches the bridged `kb_*` tools and
  the gateway provider. A Codex 0.164 probe confirmed that disabling the parent's
  shell also removes the reader child's shell, while PreToolUse has no documented
  child identity. When scope enforcement and a reader are both enabled, CodePartner
  stops a Codex native coder or reviewer before model execution rather than claiming
  that a hook alone confines arbitrary shell reads.

**Telemetry:**
- `native_tool_metrics.exploration`, for every native run whatever the flag (so A/B
  compares like with like): `mode` (`off` | `reader`), `direct_reads`, `ranged_reads`,
  `whole_file_reads`, `read_turns` (parent responses with reads; the work order expects
  1), `batched_read_turns`, `reads_in_batches`, `exact_rereads`,
  `bash_exploration`, `bash_exact_reads` (shell commands classified for Claude `Bash` and
  Codex `command_execution` alike), `brief_file_reads`, `brief_full_rereads`.
- `native_tool_metrics.subagents` when offered: `reader_calls`, `checks_calls`,
  `objectives_paths`, `requests_without_paths` (a reader request with no path or a checks
  request with no quoted command: the parent under-specified), `missing_context_reported`,
  `empty_briefs` (a reader reply with no citation, typically its turn limit),
  `async_launches` (an `Agent` call the CLI ran in the background; expected 0),
  `child_tool_calls`, `child_kb_calls` (expected 0), `child_tools`,
  `child_bash_exploration`, `brief_bytes`, `cited_files`, `child_usage`,
  `foreground_calls`, `background_calls`, `background_requests`, and spawn
  `descriptions`.
- `native_tool_metrics.scope_denials` counts denied parent tool calls by tool name;
  Claude also records `result_segments` for usage reconciliation.
- Native Codex planner turns with scope enforcement record `hosted_reader.calls`,
  `rejected`, `failures`, and separate reader `usage` under `native_tool_metrics`.
- `native_tool_metrics.cost_attribution`: parent vs subagent cost and tokens with
  `method`/`tokens_method`. Claude tokens are `exact` when the `Agent` result reports usage;
  the cost split is always `estimated_by_model` (flagged `ambiguous` when the reader and
  parent share a model). Codex: `parent_only`.
- Per completed task: `TelemetryAggregation.task_comparison(job_ids)` sums every attempt of
  a task (retries and fallbacks, both providers; a WP review is its own row) by
  `exploration_mode`: normalized tokens including `cache_read_tokens`, `cost_usd` (None when
  any attempt reported no cost, as Codex does), and the counters above. Inline attempts
  (docs/inline-execution.md) appear under the mode `inline`, with their call counters.
  `aci_comparison` labels reader runs `+explore:reader`.

### What is and is not isolated

**Codex runs without its own sandbox.** Its Linux sandbox (bubblewrap) needs
unprivileged user, mount and network namespaces, which the `codepartner` container
does not allow:

| Container setting | Result |
|---|---|
| Default | "No permissions to create a new namespace" |
| `seccomp=unconfined` | Mount setup is denied |
| `seccomp=unconfined` and `apparmor=unconfined` | Loopback network setup is denied |

The only boundary is therefore the container and what is mounted into it. This is
**not** per-project isolation:
- every project mounted into the container is reachable by every native agent, not
  only the conversation's project;
- the container uses the host network, so agents can reach the network and local
  services on the host;
- the agent-home volume (CLI logins, the routing database) is inside the container.

Claude Code has no OS sandbox either; its `cwd` is the workspace, but its `Bash`
tool runs with the container's permissions. To opt Codex out, set
`CODEPARTNER_NATIVE_PROVIDERS=claude`; its attempts then fail in `native` mode or
use the IDE path in `auto` mode.

### Read-only runs: detection and tool restrictions

Planner, reviewer, chat-answer and `/review` runs are read-only sessions:
- Claude gets only `Read`, `Grep` and `Glob`, so it has no tool that writes. This is
  a real restriction at the tool level.
- A native Codex planner with `subagents.enforce_scope: true` has its shell,
  configured apps, plugins, and direct source tools disabled. It receives only
  restricted KB navigation tools, a bulk declaration tool, and `planner_reader`.
  CodePartner reads the exact requested ranges (up to 200 lines per file and
  1200 lines total) and passes them to a separate Codex reader run with no shell
  or project tools. The planner receives that run's `path:line` brief. If the
  hosted reader or its model is unavailable, the planner attempt fails closed.
- Other native Codex read-only runs may still have a shell and can write.

Before the run, Code Partner takes a git baseline of the work tree (as for coder
runs). After the run, even when it fails, times out or is cancelled, it compares,
and **nothing is reverted**:
- if the agent itself reported writing (Claude `Edit`/`Write`, Codex
  `file_change`), the run fails and names the paths;
- other changed paths are reported, because the baseline cannot tell them apart
  from your own IDE edits or another run on the same project. For Codex, whose
  shell could have made them, the run fails; for read-only Claude it only warns.

## Review and the change journal

Before a native attempt starts, `RollbackJournal.record_git_baseline` records the
pre-task state of the work tree for comparison: the object id, mode and stat of
every tracked and untracked, non-ignored file (`git hash-object --no-filters`
without `-w`, so nothing is written to the repository), the staged index entries
and `HEAD`. If the baseline cannot be taken, the run does not start.

After the CLI exits, even on stop or cancellation, the journal compares the current
state with the baseline. Only changed paths are recorded in the mutation tables,
with whether they existed before and their post-run hash. Changed means a different
file content, mode or existence, or a different index entry. The reviewer's mutation
manifest and the retry logic use this journal.

**Preimages, for the review diff only.** Each touched file's content before its first
touch is kept in `implementation_rollback_files.before_content`: the IDE path keeps the
content it already reads before a write; the native path reads the plan's edit files at
the baseline and, for other changed files, recovers the baseline blob when git already
stores it (clean tracked files). The reviewer then sees each change as a unified diff
with `planner.edit_margin_lines` of context, plus planned lines the coder left
unchanged; new files stay whole, and a file without a preimage is shown whole.

**Planned ranges follow earlier tasks.** The planner's ranges are line numbers of the
source it saw. Before a coder or reviewer starts, each planned edit and reference range
on a file an earlier task changed is mapped through the diff between that file's first
preimage in the job and its current content (`agent/planning/rebase.py`), so a later
task's scope and its range-bounded reads point at the planned code. A planned new Python
package also gets its `__init__.py` as a new-file edit when no task plans one and a
sibling package has one.

**There is no undo.** Preimages are never restored, and task changes stay in the
project; use git to revert them. If a run moved `HEAD` (a commit or checkout) or a change
could not be verified, the job continues and the reviewer is told.

**Limitations.**
- Ignored files (`.gitignore`) are not journaled.
- Edits you make in the IDE while a native task runs are attributed to that task.

## Test evidence

Shell commands that look like test or verification runs (`pytest`, `unittest`, ...)
are captured as execution evidence for review, with their output and exit status:
- Claude: each `Bash` call is paired with its result. Success has no exit code in
  the CLI's result (`is_error=false`, recorded as 0); a failure starts with
  `Exit code N`; an error without a code (interrupted, timed out) is recorded as
  unknown and judged from its output.
- Codex: `command_execution` items carry the command, output and `exit_code`.

When a work package needs a passing test run and none was captured, Code Partner
runs the tests itself in the workspace (`uv run pytest ...` for uv/pyproject
projects), without the IDE's `execute_tool`.

## Container environment

Bind mounts expose host files, not host interpreters or tools. The image contains:
- Python 3.13 and `uv`
- Node.js 22
- git and ripgrep

When a project needs a tool the container lacks (for example `go`, `cargo`,
`mvn`), the coder sees a note and is told not to claim the command ran.

A host-created `.venv` is not used. For Python projects, `UV_PROJECT_ENVIRONMENT`
points `uv run` at a separate container environment
(`~/.cache/codepartner-venvs/<id>` in the `agent-home` volume). The host `.venv`
is never modified.

**Security.** A native agent can read everything inside every mounted project,
including files such as `.env`. Mount only projects you want the agents to see.
Dedicated secret filtering is future work.

## Smoke test

Inside the container, with a registered and mounted git project:

```bash
docker compose exec codepartner python scripts/smoke_native.py --project music-generation
docker compose exec codepartner python scripts/smoke_native.py --project music-generation --provider codex
```

The real CLI must, using only its own tools:
1. read a tracked file;
2. create a uniquely named `codepartner-smoke-<id>.txt`;
3. run `git status` on it.

The script verifies all three, then deletes only that file.

`--e2e` drives the real services natively: the OpenSpec catalog read from disk, the
WP planner, a journaled coder attempt that writes a test file and runs pytest, the
reviewer, then deletes the test file. The Work Package is synthetic and the routing database
temporary, so the project's OpenSpec state is not changed:

```bash
docker compose exec codepartner python scripts/smoke_native.py --project music-generation --e2e
docker compose exec codepartner python scripts/smoke_native.py --project music-generation --e2e --provider codex
```

## Telemetry for comparisons

Every coder attempt carries `execution_mode` (`native` or `ide`) and
`workspace_project` in its run metadata. Native planner turns record
`execution_mode: native` and `native_tool_metrics` in their planning-turn
`tool_metrics`; the file paths there are best-effort observations, not evidence. Native attempts write a
`telemetry_aci_attempts` row whose `metrics_json` has `execution_mode: native`,
`tool_calls`, per-tool counts (`tools`) and the files touched (`files`), next to
the provider-reported usage and cost. With an approved plan, the run metadata carries
`scope_files` (the task's edit and reference files; for the reviewer, also the
journaled files), and `native_tool_metrics` counts `out_of_scope_reads` (parent reads
and content searches outside them; subagent reads are not counted) and
`scope_exceptions` (stated `SCOPE EXCEPTION:` lines). Both are telemetry only and appear
in the per-task aggregation and the `codepartner.native.*` trace attributes. IDE
attempts keep writing their existing metrics there. Native execution has not been benchmarked here. Compare
arms on your own runs before drawing conclusions.

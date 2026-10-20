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
and its project. Codebase Memory (`kb_*`) is used on demand; native runs have no
KB-first gates or interception. With `discovery.policy: kb_first`, project-file
access instead uses the controlled IDE path for every role; native CLI file and
shell access is disabled for that run.

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

**KB-first.** Native agents keep Codebase Memory `kb_*` tools in non-strict modes.
Under `discovery.policy: kb_first`, agents use the IDE path so that a KB outline
or file-scoped search precedes each file's first source read, and every source read
is bounded. See docs/codebase-memory.md.

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

### Context compaction

CodePartner does not summarize agent history itself. It sets each CLI's own automatic
compaction per invocation (`providers/compaction.py`) from
`CODEPARTNER_AUTO_COMPACT_TOKENS` (default `250000`; `off` keeps each CLI's default).

**Claude Code.** Gets `--settings` with `{"autoCompactWindow": N}`. For coder runs it also
gets a `PreCompact` command hook that prints the task's preservation brief: the task id,
goal, acceptance criteria, excluded files, and what to keep (modified files, outstanding
steps, validation results, unresolved errors, the handoff). Claude Code uses the hook's
output as custom instructions for its native summary. Limitations:
- `autoCompactWindow` sets the window. Claude Code compacts at the window minus its summary
  buffer. With Claude Code 2.1.303, measured with `get_context_usage()`, that buffer is
  33k, so a 250k window compacts at about 217k tokens.
- Claude Code uses `min(model context window, N)`. Haiku 4.5 has a 200k window, so it stays
  at 200k (compacting at 167k). Sonnet 5 and Opus 5 drop from about 1M to 250k.
- An operator-set `CLAUDE_CODE_AUTO_COMPACT_WINDOW` takes precedence over the setting.

**Codex CLI.** Gets `-c model_auto_compact_token_limit=N`. Codex already runs with
`--ignore-user-config`, so `~/.codex/config.toml` is neither read nor changed, and a `-c`
override in `CODEX_ARGS` still wins. Codex has no hook for preservation instructions: its
only knob, `compact_prompt`, would replace the whole native summarizer prompt, so it is not
set. Codex keeps recent user messages, including the task prompt, next to its summary. A
model whose context window is smaller than N cannot reach the threshold before its window
fills. In the local Codex 0.146.0 model cache, the gpt-5.6 models have a 272k window with
95% usable, about 258k, so 250k fits. Models missing from that cache (for example
`gpt-6.1-sol`) have not been checked.

**What is recorded.** Each run's metadata records
`context_compaction: {requested_auto_compact_tokens, mechanism}` (what was asked for). It
appends to `compactions` only when the CLI actually reports a compaction: Claude's
`compact_boundary` message with its trigger and `pre_tokens`, or Codex's
`context_compaction` item. The task contract and CodePartner's execution records stay
authoritative after a compaction.

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

### Read-only runs: detection, not a sandbox

Planner, reviewer, chat-answer and `/review` runs are read-only sessions:
- Claude gets only `Read`, `Grep` and `Glob`, so it has no tool that writes. This is
  a real restriction at the tool level.
- Codex has its shell, which can write. Nothing prevents that.

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

**There is no undo.** No preimages are kept, and task changes stay in the project;
use git to revert them. If a run moved `HEAD` (a commit or checkout) or a change
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
the provider-reported usage and cost. IDE attempts keep writing their existing
metrics there. Native execution has not been benchmarked here. Compare
arms on your own runs before drawing conclusions.

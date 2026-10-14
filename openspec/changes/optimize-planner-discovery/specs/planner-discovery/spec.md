# Spec Delta

## Purpose

Measure, and later reduce, unnecessary exploration by the WP planner without lowering plan quality.

## ADDED Requirements

### Requirement: Explicit graph-first search interception
For every agent role, a repository-wide definition-style identifier text
search SHALL first show Codebase Memory graph definitions with a clear warning
that text search was not executed. Repeating the identical search SHALL execute
it unchanged. Usage, import, literal, test, documentation and configuration
searches SHALL pass through. A missing graph result or KB failure SHALL execute
the original search in the same call.

#### Scenario: Search repeat
- **WHEN** an agent repeats an identical gated identifier search
- **THEN** the original IDE text search executes with unchanged arguments.

### Requirement: KB hints require current IDE verification
Graph matches, outlines and indexed snippet locations SHALL be navigation
hints only. A snippet's source SHALL be read through the request's IDE tool and
its qualified symbol scope verified before it can be returned as current
source. Reviewer evidence SHALL count only successful current IDE reads.

#### Scenario: Stale snippet
- **WHEN** the indexed location does not contain the scoped declaration
- **THEN** the policy searches the current file for one verified relocation or
returns a stale, ambiguous or unresolved status without presenting KB source.

### Requirement: Authorized generated dispatch
Every policy-generated call SHALL go through the same request-scoped registry,
adapter, stop check, task contract and role-specific handler as a direct call.
The policy SHALL NOT recursively gate its own calls, and SHALL NOT grant access
to a tool absent from the advertised request.

#### Scenario: Refused generated read
- **WHEN** a generated IDE read is not authorized by the request registry
- **THEN** it is refused, not retried through another tool or handler.

### Requirement: Observe mode is behaviour-neutral
In `observe` mode, the system SHALL NOT change the planner's prompt, tool schemas, tool responses, hard caps or KB prefetch.

#### Scenario: Off and observe runs
- **WHEN** the same scripted planner run executes in `off` and in `observe`
- **THEN** the provider receives identical conversations, specs and tool results, and only `tool_metrics.discovery` differs

### Requirement: Retrieval level classification
The system SHALL classify each executed planner call as L1, L2 or L3:
- L1: outline, search, listing, symbol and graph calls;
- L2: ranged reads, and whole reads whose returned response is under `small_file_bytes`;
- L3: other whole reads.

It SHALL count L3 reads as escalated only when an earlier L1 or L2 call targeted the same path.

#### Scenario: Outline then full read
- **WHEN** the planner outlines `src/a.py` and then reads it whole, receiving 20 KB
- **THEN** the read counts as L3 escalated

#### Scenario: Small whole read
- **WHEN** a whole read returns fewer bytes than `small_file_bytes`
- **THEN** it counts as L2

### Requirement: Evidence strength is independent of retrieval
The system SHALL assign each read path the strongest evidence from the WP text, the KB prefetch and the results of search or graph calls. It SHALL NOT derive evidence for a path from reading or outlining that path, and SHALL count reads with weak or no evidence as unexplained.

#### Scenario: Outline only
- **WHEN** the planner outlines and then reads a path that no WP text, prefetch or search result mentioned
- **THEN** the read has evidence `none` and is counted as unexplained

#### Scenario: Broad listing
- **WHEN** a path appears only in a directory listing
- **THEN** a read of it has evidence `weak`

### Requirement: Estimated and actual tokens are separate
The system SHALL record `est_added_tokens`, computed from the UTF-8 bytes delivered to the planner, separately from provider-reported usage. The benchmark SHALL normalise provider usage into disjoint uncached, cache-read and cache-write input.

#### Scenario: Codex usage
- **WHEN** Codex reports `input_tokens=1000` and `cached_input_tokens=800`
- **THEN** the benchmark reports 200 uncached, 800 cache-read and 1000 total input tokens

### Requirement: Advisories are opt-in and one-shot
The system SHALL default to `observe`. In `advise`, it SHALL add planner-only discovery guidance and SHALL append at most one advisory line per soft threshold, the first time that threshold is reached. It SHALL NOT refuse calls because of a soft threshold and SHALL NOT change the shared KB guidance used by coders and reviewers.

#### Scenario: Full-read threshold reached
- **WHEN** a `low` WP planner makes its second L3 read with `full_reads: 2`
- **THEN** that read's result ends with one advisory line, the crossing is recorded, and later L3 reads receive no further full-read advisory

#### Scenario: Observe
- **WHEN** the mode is `observe`
- **THEN** no advisory is appended and `threshold_crossings` is empty

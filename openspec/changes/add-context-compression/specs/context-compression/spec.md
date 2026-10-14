# Spec Delta

## Purpose

Compress eligible coder tool results without losing information the coder needs, keep provenance and authorized retrieval for the originals, and measure the effect on actual provider token usage.

## ADDED Requirements

### Requirement: Eligibility
The system SHALL compress only search, command, log, listing and structured results larger than `CODEPARTNER_COMPRESS_MIN_BYTES`. It SHALL NOT compress file reads, edit or mutation results, OpenSpec artifact reads or Reviewer trusted reads.

#### Scenario: Small result
- **WHEN** a search result is smaller than the threshold
- **THEN** it is returned unchanged and no compression is recorded

#### Scenario: File read
- **WHEN** the coder reads a source file of any size
- **THEN** the result is not compressed

### Requirement: Error output is conservative
The system SHALL use `conservative` mode for error-heavy output, whatever mode is configured, and SHALL NOT pass a command hint for it.

#### Scenario: Failing tests with balanced configured
- **WHEN** a test command fails and `CODEPARTNER_COMPRESS_MODE=balanced`
- **THEN** only ANSI codes are removed, and every traceback line is preserved

### Requirement: No silent loss
The system SHALL return the original result when compression drops any identifier (path, `path:line`, test node id, symbol, observation id, call id) or any error-like line, or when it changes the parsed value of a JSON result.

#### Scenario: Identifier dropped
- **WHEN** the compressed output is missing a `path:line` reference present in the input
- **THEN** the original is returned and the reason `identifier_lost` is recorded

#### Scenario: JSON minified
- **WHEN** a JSON result is minified and still parses to an equal value
- **THEN** the minified result is returned

### Requirement: Fail-open
The system SHALL return the original result when the compressor times out, errors or is unavailable, and SHALL count it as a fallback.

#### Scenario: Sidecar down
- **WHEN** the compressor socket is unreachable
- **THEN** the coder receives the original result and `compress_fallbacks` is incremented

### Requirement: Authorized retrieval
Every compressed result SHALL carry a marker and an observation id. `codepartner_get_observation` SHALL return the original only to the run that produced it, and SHALL support `query` search scoped to that run.

#### Scenario: Other run's id
- **WHEN** a run requests an observation id from another run
- **THEN** the response is `OBSERVATION UNAVAILABLE`

#### Scenario: Query search
- **WHEN** the coder calls `codepartner_get_observation(query="KeyError")`
- **THEN** matching observations from the same run are listed with their ids

### Requirement: Isolated compressor
The compressor SHALL run without network access or project mounts, SHALL reject `auto` mode and command execution, and SHALL receive only the text Code Partner sends it.

#### Scenario: Auto requested
- **WHEN** a request asks for mode `auto`
- **THEN** the compressor rejects it and Code Partner returns the original

### Requirement: Memoized results
The system SHALL key memoized results by input hash, compressor version, mode, command hint and effective configuration.

#### Scenario: Configuration change
- **WHEN** the mode or compressor version changes
- **THEN** a new memo entry is computed instead of reusing the old one

### Requirement: Provenance
The system SHALL record, for each observation, the run, task and attempt ids, tool, resource, argument hash, content hash, mode, compressor version, sizes, applied flag and reason. It SHALL mark duplicates and mark observations stale when their resource is mutated.

#### Scenario: Resource edited
- **WHEN** a file is edited after a search result that referenced it was observed
- **THEN** that observation is marked stale, and retrieving it carries a `STALE` header

### Requirement: Measurement
The system SHALL report estimated compressor savings separately from actual provider usage (input, cache-read, cache-write and output tokens) for each attempt and compression mode.

#### Scenario: A/B comparison
- **WHEN** `aci_comparison()` is run over attempts with compression off and on
- **THEN** it reports both estimated savings and actual token deltas per completed task

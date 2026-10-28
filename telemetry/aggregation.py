"""Read-only efficiency summaries for evaluation and regression reporting."""

import json

from telemetry.storage import TelemetryStorage

_COMPRESSION_KEYS = ("compress_applied", "compress_fallbacks", "compress_guard_rejections",
                     "compress_bytes_before", "compress_bytes_after", "dedup_hits",
                     "est_tokens_saved")


def normalize_usage(usage: dict | None) -> dict:
    """Provider usage as uncached input, cache read, cache write and output tokens.

    Anthropic reports `input_tokens` excluding cache reads and writes; OpenAI and
    Codex report `input_tokens` including `cached_input_tokens`.
    """
    usage = usage or {}
    input_tokens = usage.get("input_tokens", usage.get("inputTokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("outputTokens", 0)) or 0
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        read = usage.get("cache_read_input_tokens") or 0
        write = usage.get("cache_creation_input_tokens") or 0
        uncached = input_tokens
    else:
        read = usage.get("cached_input_tokens") or 0
        write = usage.get("cache_write_input_tokens") or 0
        uncached = max(input_tokens - read - write, 0)
    return {"uncached_input_tokens": uncached, "cache_read_tokens": read,
            "cache_write_tokens": write, "output_tokens": output_tokens,
            "total_input_tokens": uncached + read + write}


_TASK_USAGE_KEYS = ("uncached_input_tokens", "cache_read_tokens", "cache_write_tokens",
                    "output_tokens", "total_input_tokens")
_TASK_EXPLORATION_KEYS = ("direct_reads", "ranged_reads", "whole_file_reads",
                          "exact_rereads", "batched_read_turns", "bash_exploration",
                          "bash_exact_reads", "brief_file_reads", "brief_full_rereads")
_TASK_SUBAGENT_KEYS = ("reader_calls", "checks_calls", "requests_without_paths",
                       "missing_context_reported", "empty_briefs", "brief_bytes")
# Parent reads outside the plan's files, and the scope exceptions it stated.
_TASK_SCOPE_KEYS = ("out_of_scope_reads", "scope_exceptions")


class TelemetryAggregation:
    def __init__(self, storage: TelemetryStorage | None = None):
        self.storage = storage or TelemetryStorage()

    def task_summary(self, task_id: str) -> dict:
        with self.storage._database() as db:
            rows = db.execute("""SELECT role, tool_calls, calls_by_category_json,
              unique_files_read_json, output_chars, repeated_calls,
              calls_before_first_edit, meaningful_progress, outcome
              FROM telemetry_attempt_summaries WHERE task_id=?""", (task_id,)).fetchall()
        categories: dict[str, int] = {}
        files: set[str] = set()
        for row in rows:
            for key, value in json.loads(row[2]).items():
                categories[key] = categories.get(key, 0) + int(value)
            files.update(json.loads(row[3]))
        return {
            "task_id": task_id,
            "execution_attempts": sum(row[0] == "coder" for row in rows),
            "reviewer_failures": sum(
                row[0] == "reviewer" and row[8] == "FAIL" for row in rows),
            "total_tool_calls": sum(row[1] for row in rows),
            "calls_by_category": categories,
            "unique_files_read": sorted(files),
            "output_chars": sum(row[4] for row in rows),
            "repeated_calls": sum(row[5] for row in rows),
            "calls_before_first_edit": sum(row[6] for row in rows),
            "meaningful_progress": sum(row[7] for row in rows),
            "final_outcome": rows[-1][8] if rows else None,
        }

    def task_comparison(self, job_ids: list[str]) -> dict:
        """Per-task A/B totals of native runs, grouped by `exploration_mode`.

        One row per (job, task) sums every recorded attempt of that task (coder retries
        and fallbacks on either provider; a WP review is its own row): normalized token
        usage including cache reads, total cost, and the exploration counters. Cost is
        None when an attempt had no reported cost (Codex reports tokens only), so a
        partial sum is never presented as the total."""
        if not job_ids:
            return {}
        placeholders = ",".join("?" for _ in job_ids)
        with self.storage._database() as db:
            rows = db.execute(
                "SELECT job_id, task_id, provider, model, metrics_json, usage_json, cost_usd "
                "FROM telemetry_aci_attempts WHERE job_id IN (" + placeholders + ")",
                tuple(job_ids),
            ).fetchall()
        result: dict[str, dict] = {}
        for job_id, task_id, provider, model, metrics_json, usage_json, cost in rows:
            metrics = json.loads(metrics_json)
            if metrics.get("execution_mode") != "native":
                continue
            exploration = metrics.get("exploration") or {}
            mode = metrics.get("exploration_mode") or exploration.get("mode") or "unknown"
            row = result.setdefault(job_id, {}).setdefault(task_id or "?", {}).setdefault(
                mode, {"attempts": 0, "providers": [], "cost_usd": 0.0, "cost_complete": True,
                       **{key: 0 for key in _TASK_USAGE_KEYS},
                       **{key: 0 for key in _TASK_EXPLORATION_KEYS},
                       **{key: 0 for key in _TASK_SUBAGENT_KEYS},
                       **{key: 0 for key in _TASK_SCOPE_KEYS}})
            row["attempts"] += 1
            if f"{provider}:{model}" not in row["providers"]:
                row["providers"].append(f"{provider}:{model}")
            for key, value in normalize_usage(json.loads(usage_json) if usage_json
                                              else {}).items():
                row[key] += value
            if cost is None:
                row["cost_complete"] = False
            else:
                row["cost_usd"] += cost
            for key in _TASK_EXPLORATION_KEYS:
                row[key] += exploration.get(key) or 0
            for key in _TASK_SUBAGENT_KEYS:
                row[key] += (metrics.get("subagents") or {}).get(key) or 0
            row["out_of_scope_reads"] += len(metrics.get("out_of_scope_reads") or ())
            row["scope_exceptions"] += metrics.get("scope_exceptions") or 0
        for tasks in result.values():
            for modes in tasks.values():
                for row in modes.values():
                    if not row.pop("cost_complete"):
                        row["cost_usd"] = None
        return result

    def aci_comparison(self, task_ids: list[str]) -> dict:
        """Compare recorded coder attempts by observation mode without source data."""
        if not task_ids:
            return {}
        placeholders = ",".join("?" for _ in task_ids)
        with self.storage._database() as db:
            rows = db.execute(
                "SELECT task_id, metrics_json, usage_json, cost_usd "
                "FROM telemetry_aci_attempts WHERE task_id IN (" + placeholders + ")",
                tuple(task_ids),
            ).fetchall()
        result: dict[str, dict] = {}
        for task_id, metrics_json, usage_json, cost in rows:
            metrics = json.loads(metrics_json)
            usage = json.loads(usage_json) if usage_json else {}
            mode = metrics.get("observation_mode", "unknown")
            compression = metrics.get("compression_mode", "off")
            if compression != "off":
                mode = f"{mode}+compress:{compression}"
            if metrics.get("exploration_mode") == "reader":
                mode = f"{mode}+explore:reader"
            group = result.setdefault(task_id, {}).setdefault(mode, {
                "attempts": 0, "agent_calls": 0, "backend_calls": 0,
                "lines_returned": 0, "model_output_bytes": 0,
                "backend_output_bytes": 0, "input_tokens": 0,
                "cached_input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0,
                # Estimated by the compressor (bytes/4), never provider-reported.
                "estimated": {key: 0 for key in _COMPRESSION_KEYS},
                # Reported by the provider, normalized across Claude and Codex.
                "actual": {"uncached_input_tokens": 0, "cache_read_tokens": 0,
                           "cache_write_tokens": 0, "output_tokens": 0,
                           "total_input_tokens": 0},
            })
            group["attempts"] += 1
            for key in ("agent_calls", "backend_calls", "lines_returned",
                        "model_output_bytes", "backend_output_bytes"):
                group[key] += metrics.get(key, 0)
            group["input_tokens"] += usage.get("input_tokens", usage.get("inputTokens", 0)) or 0
            group["cached_input_tokens"] += (usage.get("cache_read_input_tokens")
                                              or usage.get("cached_input_tokens") or 0)
            group["output_tokens"] += usage.get("output_tokens", usage.get("outputTokens", 0)) or 0
            group["cost_usd"] += cost or 0.0
            for key in _COMPRESSION_KEYS:
                group["estimated"][key] += metrics.get(key, 0)
            for key, value in normalize_usage(usage).items():
                group["actual"][key] += value
        for modes in result.values():
            for group in modes.values():
                actual = group["actual"]
                # Diagnostic only: smaller tool results can lower this ratio
                # while total cost falls.
                group["cache_read_ratio"] = (
                    round(actual["cache_read_tokens"] / actual["total_input_tokens"], 4)
                    if actual["total_input_tokens"] else None)
        return result

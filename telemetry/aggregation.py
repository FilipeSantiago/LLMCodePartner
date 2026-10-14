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
        write = 0
        uncached = max(input_tokens - read, 0)
    return {"uncached_input_tokens": uncached, "cache_read_tokens": read,
            "cache_write_tokens": write, "output_tokens": output_tokens,
            "total_input_tokens": uncached + read + write}


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

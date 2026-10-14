"""Compare WP planner discovery across benchmark arms from recorded planning turns.

Each arm is a set of planning session ids (N >= 3 runs of the same pinned WP).
Only the selected revision's turns are counted (default 1: the initial plan and
its repair turn). Reports median and range per arm.

    uv run python scripts/bench_planner.py \\
        --arm baseline=wpplan_a,wpplan_b,wpplan_c \\
        --arm observe=wpplan_d,wpplan_e,wpplan_f \\
        --truth wp3_changed_files.txt --rubric wp3_rubric.json [--json out.json]

Token figures are kept apart: `est_added_tokens` is the discovery tracker's
estimate of newly added context (delivered UTF-8 bytes / 4); provider tokens are
cumulative usage reported by the provider, normalized by
`telemetry.aggregation.normalize_usage` (uncached, cache-read and cache-write
input are disjoint, so nothing is counted twice). A run whose turns lack usage
reports provider tokens as n/a.

Plan quality is reported in two parts:
- changed-file recall: planned files (shared, task, reviewer and discovery
  files) against `--truth`, the files the WP's implementation diff changed;
- plan relevance: `validate_draft` gaps plus the manual rubric (`--rubric`, a
  JSON object keyed by session id, e.g. {"completeness": 4, "correctness": 5,
  "scope": 4, "criteria_covered": 3, "criteria_total": 3, "notes": "..."}).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.kb_first import _same  # noqa: E402
from agent.routing.persistence import RoutingPersistence  # noqa: E402
from telemetry.aggregation import normalize_usage  # noqa: E402

_TOOL_KEYS = ("tool_calls", "kb_calls", "file_reads", "searches", "whole_file_reads",
              "ranged_source_reads", "repeated_source_reads", "source_response_bytes",
              "refused_over_budget", "refused_reads")
_PROVIDER_KEYS = ("uncached_input_tokens", "cache_read_tokens", "cache_write_tokens",
                  "total_input_tokens", "output_tokens")
_RUBRIC_KEYS = ("completeness", "correctness", "scope", "criteria_covered", "criteria_total")


def planned_files(draft: dict | None) -> list[str]:
    draft = draft or {}
    files = [*((draft.get("shared") or {}).get("relevant_files") or ()),
             *((draft.get("reviewer") or {}).get("relevant_files") or ())]
    for task in (draft.get("tasks") or {}).values():
        files += (task or {}).get("relevant_files") or ()
    files += [item.get("file") for item in draft.get("discovery") or ()
              if isinstance(item, dict)]
    return list(dict.fromkeys(path for path in files if isinstance(path, str) and path))


def run_metrics(turns: list[dict], session: dict | None, truth: list[str] | None,
                rubric: dict | None = None, revision: int = 1) -> dict:
    """One planning run (session) reduced to comparable numbers."""
    selected = [turn for turn in turns if turn["revision"] == revision]
    # Keys an older row never recorded are n/a, not 0.
    run: dict = {key: (sum(turn["tool_metrics"].get(key) or 0 for turn in selected)
                       if any(key in turn["tool_metrics"] for turn in selected) else None)
                 for key in _TOOL_KEYS}
    discovery = [turn["tool_metrics"].get("discovery") for turn in selected]
    if selected and all(discovery):
        levels = [item["retrieval_levels"] for item in discovery
                  if "retrieval_levels" in item]
        strengths = [item["evidence_strength"] for item in discovery
                     if "evidence_strength" in item]
        if levels and strengths:
            run.update({level: sum(item[level] for item in levels)
                        for level in ("L1", "L2", "L3")})
            run.update({key: sum(item.get(key, 0) for item in discovery)
                        for key in ("l3_direct", "l3_escalated", "unexplained_reads",
                                    "est_added_tokens")})
            run.update({f"evidence_{name}": sum(item[name] for item in strengths)
                        for name in ("strong", "moderate", "weak", "none")})
            run["kb_fallback"] = int(any(item.get("kb_fallback") for item in discovery))
        for key in ("kb_fallback_or_escalation_count", "redundant_source_reads",
                    "redundant_source_bytes", "edit_verification_blocks"):
            if any(key in item for item in discovery):
                run[key] = sum(item.get(key, 0) for item in discovery)
        for gate, outcomes in (("search_gate", ("kb_sufficient", "kb_insufficient")),
                               ("outline_gate", ("outline_sufficient",
                                                 "refined_to_range", "escalated_to_full"))):
            for outcome in outcomes:
                if any(outcome in item.get(gate, {}) for item in discovery):
                    run[outcome] = sum(item.get(gate, {}).get(outcome, 0)
                                       for item in discovery)
    gates = [turn["tool_metrics"].get("outline_gate") for turn in selected]
    if selected and all(gates):
        run["outline_gate_served"] = sum(item["served"] for item in gates)
    run["cancelled_attempts"] = sum(bool(turn["tool_metrics"].get("cancelled"))
                                    for turn in selected)
    usages = [turn["usage"] for turn in selected]
    if selected and all(usages):
        normalized = [normalize_usage(usage) for usage in usages]
        run.update({key: sum(item[key] for item in normalized) for key in _PROVIDER_KEYS})
    else:
        run.update(dict.fromkeys(_PROVIDER_KEYS))
    run["latency_ms"] = sum(turn["latency_ms"] or 0 for turn in selected)
    run["attempts"] = len(selected)
    run["repair_turns"] = sum(turn["kind"] == "repair" for turn in selected)
    if session is not None:
        run["gaps"] = len(session.get("gaps") or ())
        if session.get("revision", revision) != revision:
            run["warning"] = (f"session is at revision {session['revision']}; planned files "
                              f"reflect that revision, not {revision}")
        planned = planned_files(session.get("draft"))
        run["planned_files"] = len(planned)
        if truth:
            hits = [path for path in truth if any(_same(path, item) for item in planned)]
            run["changed_file_recall"] = round(len(hits) / len(truth), 3)
            run["missed_changed_files"] = [path for path in truth if path not in hits]
            run["planned_outside_changed"] = sum(
                not any(_same(item, path) for path in truth) for item in planned)
    for key in _RUBRIC_KEYS:
        value = (rubric or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            run[f"rubric_{key}"] = value
    return run


def summarize(runs: list[dict]) -> dict:
    """Median and range of each numeric metric across an arm's runs."""
    keys = list(dict.fromkeys(key for run in runs for key, value in run.items()
                              if isinstance(value, (int, float))
                              and not isinstance(value, bool)))
    summary = {}
    for key in keys:
        values = [run[key] for run in runs if isinstance(run.get(key), (int, float))]
        missing = len(runs) - len(values)
        summary[key] = {"median": statistics.median(values), "min": min(values),
                        "max": max(values), "n": len(values), "missing": missing}
    return summary


def downstream_rows(db: Path, session_id: str) -> dict:
    """Read-only job/coder/reviewer join by the approved plan for one session."""
    if db is None:
        return {}
    with sqlite3.connect(db) as connection:
        plans = [row[0] for row in connection.execute(
            "SELECT plan_id FROM wp_approved_plans WHERE session_id=?", (session_id,))]
        if not plans:
            return {}
        jobs = [row[0] for plan_id in plans for row in connection.execute(
            "SELECT job_id FROM implementation_jobs WHERE plan_id=?", (plan_id,))]
        if not jobs:
            return {"downstream_jobs": 0}
        reviews = [row[0] for job_id in jobs for row in connection.execute(
            "SELECT verdict FROM implementation_reviews WHERE job_id=?", (job_id,))]
        attempts = [row for job_id in jobs for row in connection.execute(
            """SELECT a.status, a.usage_json FROM execution_attempts a
               JOIN implementation_job_tasks t ON t.decision_id=a.decision_id
               WHERE t.job_id=?""", (job_id,))]
        has_aci = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='telemetry_aci_attempts'").fetchone()
        aci = ([json.loads(row[0]) for job_id in jobs for row in connection.execute(
            "SELECT metrics_json FROM telemetry_aci_attempts WHERE job_id=?", (job_id,))]
            if has_aci else [])
        coder_usage = [normalize_usage(json.loads(row[1])) for row in attempts if row[1]]
        inspection = [item["changed_files_inspected_ratio"] for item in aci
                      if item.get("role") == "reviewer"
                      and isinstance(item.get("changed_files_inspected_ratio"), (int, float))]
        return {"downstream_jobs": len(jobs), "coder_attempts": len(attempts),
                "coder_retries": sum(row[0] != "completed" for row in attempts),
                "review_done": reviews.count("done"),
                "review_needs_work": reviews.count("needs_work"),
                "coder_total_input_tokens": sum(item["total_input_tokens"]
                                                for item in coder_usage),
                "changed_files_inspected_ratio": (
                    statistics.mean(inspection) if inspection else None),
                "downstream_source_bytes": sum(
                    item.get("kb_first", {}).get("source_response_bytes", 0)
                    for item in aci)}


async def collect(db: Path, arms: dict[str, list[str]], truth: list[str] | None,
                  rubric: dict, revision: int, *,
                  arm_truth: dict[str, list[str]] | None = None,
                  downstream: bool = False) -> dict:
    persistence = RoutingPersistence(db)
    report = {}
    for label, sessions in arms.items():
        runs = []
        for session_id in sessions:
            turns = await persistence.planning_turns(session_id)
            if not turns:
                raise SystemExit(f"no planning turns recorded for {session_id}")
            session = await persistence.planning_session(session_id)
            runs.append({"session_id": session_id, **run_metrics(
                turns, session, (arm_truth or {}).get(label, truth),
                rubric.get(session_id), revision),
                **(downstream_rows(persistence.path, session_id) if downstream else {})})
        report[label] = {"runs": runs, "summary": summarize(runs)}
    return report


def _number(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() or abs(value) >= 100 else f"{value:.3g}"


def render(report: dict) -> str:
    keys = list(dict.fromkeys(key for arm in report.values() for key in arm["summary"]))
    labels = list(report)
    width = max(len(key) for key in keys) + 2 if keys else 10
    column = 34
    lines = ["metric".ljust(width) + "".join(label.rjust(column) for label in labels)]
    for key in keys:
        cells = []
        for label in labels:
            stat = report[label]["summary"].get(key)
            if stat is None:
                cells.append("n/a".rjust(column))
                continue
            text = (f"{_number(stat['median'])} "
                    f"[{_number(stat['min'])}-{_number(stat['max'])}]")
            if stat["missing"]:
                text += f" ({stat['missing']} n/a)"
            cells.append(text.rjust(column))
        lines.append(key.ljust(width) + "".join(cells))
    for label, arm in report.items():
        for run in arm["runs"]:
            if run.get("warning"):
                lines.append(f"warning {label}/{run['session_id']}: {run['warning']}")
            if run.get("missed_changed_files"):
                lines.append(f"missed {label}/{run['session_id']}: "
                             + ", ".join(run["missed_changed_files"]))
    return "\n".join(lines)


def _arm(value: str) -> tuple[str, list[str]]:
    label, _, sessions = value.partition("=")
    ids = [item.strip() for item in sessions.split(",") if item.strip()]
    if not label or not ids:
        raise argparse.ArgumentTypeError("use LABEL=session_id,session_id,...")
    return label, ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=None,
                        help="routing database (default: ROUTING_DB_PATH or "
                             "~/.codepartner/routing.sqlite3)")
    parser.add_argument("--arm", type=_arm, action="append", required=True)
    parser.add_argument("--truth", type=Path, help="changed files, one path per line")
    parser.add_argument("--arm-truth", action="append", default=[],
                        help="LABEL=path to changed-file truth for one arm")
    parser.add_argument("--downstream", action="store_true",
                        help="join coder and reviewer outcomes through plan_id/job_id")
    parser.add_argument("--rubric", type=Path, help="manual rubric JSON keyed by session id")
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)
    truth = ([line.strip() for line in args.truth.read_text().splitlines()
              if line.strip() and not line.startswith("#")] if args.truth else None)
    rubric = json.loads(args.rubric.read_text()) if args.rubric else {}
    arms = dict(args.arm)
    arm_truth = {}
    for item in args.arm_truth:
        label, separator, path = item.partition("=")
        if not separator or label not in arms:
            parser.error("--arm-truth needs an existing arm label and file path")
        arm_truth[label] = [line.strip() for line in Path(path).read_text().splitlines()
                            if line.strip() and not line.startswith("#")]
    for label, sessions in arms.items():
        if len(sessions) < 3:
            print(f"note: arm {label!r} has {len(sessions)} run(s); use N >= 3",
                  file=sys.stderr)
    report = asyncio.run(collect(args.db, arms, truth, rubric, args.revision,
                                 arm_truth=arm_truth, downstream=args.downstream))
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

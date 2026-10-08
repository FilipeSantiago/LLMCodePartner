"""Strict parsing for the Coder-to-Tasker handoff."""

import json
import re

from agent.task_contracts.models import CoderOutcome, CoderResult, TaskContract


class CoderResultError(ValueError):
    pass


_CRITERION_ID = re.compile(r"\s*(AC\d+)\s*[:\-–]\s*", re.IGNORECASE)


def _requested_path(item: str) -> str:
    """Reduce "mlflow/config.py (new module)" to its leading path token."""
    token = (item.strip().split() or [""])[0].strip("`'\",;:()")
    return token if ("/" in token or "." in token) else item.strip()


def _criterion(item: str) -> dict:
    """Normalize a plain-text criterion claim; no decision depends on its shape."""
    match = _CRITERION_ID.match(item)
    normalized = {"criterion": item.strip(), "status": "unknown"}
    if match:
        normalized["id"] = match.group(1).upper()
    return normalized


def parse_coder_result(report: str, contract: TaskContract) -> CoderResult:
    decoder = json.JSONDecoder()
    value = None
    for index, char in enumerate(report):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(report[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("schema_version") == 2:
            value = candidate
    if value is None:
        raise CoderResultError("coder did not return the required TaskContract result")
    if len(json.dumps(value, ensure_ascii=False)) > 64_000:
        raise CoderResultError("coder result exceeds the bounded handoff size")
    if value.get("task_id") != contract.task_id:
        raise CoderResultError("coder result task_id does not match the contract")
    if (value.get("contract_id") != contract.contract_id
            or value.get("contract_revision") != contract.revision):
        raise CoderResultError("coder result references a different contract")
    try:
        outcome = CoderOutcome(value.get("outcome"))
    except ValueError as exc:
        raise CoderResultError("coder result has an invalid outcome") from exc
    summary = value.get("summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 4000:
        raise CoderResultError("coder result requires a concise summary")
    def strings(name: str) -> tuple[str, ...]:
        raw = value.get(name, [])
        if (not isinstance(raw, list) or len(raw) > 20
                or not all(isinstance(item, str) and len(item) <= 2000 for item in raw)):
            raise CoderResultError(f"coder result {name} must be an array of strings")
        return tuple(raw)
    def objects(name: str, from_string) -> tuple[dict, ...]:
        raw = value.get(name, [])
        if (not isinstance(raw, list) or len(raw) > 50
                or not all(isinstance(item, dict)
                           or (isinstance(item, str) and len(item) <= 2000)
                           for item in raw)):
            raise CoderResultError(
                f"coder result {name} must be an array of objects or strings")
        return tuple(item if isinstance(item, dict) else from_string(item)
                     for item in raw)
    raw_requests = value.get("scope_requests", [])
    if (not isinstance(raw_requests, list) or len(raw_requests) > 20
            or not all(isinstance(item, dict) and isinstance(item.get("path"), str)
                       and item.get("operation") in {"create", "modify"}
                       and len(item["path"]) <= 500 for item in raw_requests)):
        raise CoderResultError(
            "coder result scope_requests must be objects with path and "
            "operation create|modify")
    scope_requests = tuple(
        {"path": item["path"].strip(), "operation": item["operation"],
         "reason": str(item.get("reason", ""))[:500]} for item in raw_requests)
    return CoderResult(
        schema_version=2, task_id=contract.task_id,
        contract_id=contract.contract_id, contract_revision=contract.revision,
        outcome=outcome, summary=summary.strip(),
        changed_files=objects("changed_files", lambda item: {"path": item.strip()}),
        criteria=objects("criteria", _criterion),
        verified=strings("verified"), discoveries=strings("discoveries"),
        missing_context=strings("missing_context"),
        requested_files=tuple(_requested_path(item) for item in strings("requested_files")),
        remaining_gaps=strings("remaining_gaps"),
        meaningful_progress=bool(value.get("meaningful_progress")),
        scope_requests=scope_requests,
    )

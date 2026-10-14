"""Chat presentation for planner, coder, and reviewer output.

Pure string helpers; nothing here changes what any model receives. Blocks end
with a blank line and lines end with a newline, so streamed messages never
run together.
"""

import json
import re

STATUS = {
    "approved": "✅ Approved",
    "needs_work": "🛠 Needs work",
    "ask_user": "❓ Needs your confirmation",
    "unverified": "⚠️ Unverified",
    "failed": "⛔ Failed",
    "interrupted": "⏸ Interrupted",
}
VALIDATION_ICONS = {"satisfied": "✅", "failed": "❌", "unverified": "⚠️",
                    "skipped": "⏭", "not_required": "➖"}


def one_line(text: str) -> str:
    return " ".join(str(text).split())


def first_sentence(text: str, limit: int = 200) -> str:
    """The opening sentence of a summary, on one line and at most `limit` characters."""
    text = one_line(text)
    match = re.search(r"[.!?](?=\s)", text)
    sentence = text[:match.end()] if match else text
    return sentence if len(sentence) <= limit else sentence[:limit - 1].rstrip() + "…"


def block(text: str) -> str:
    """A paragraph-level block: always ends with exactly one blank line."""
    return text.rstrip("\n") + "\n\n"


def ensure_newline(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"


def status(kind: str) -> str:
    return STATUS[kind]


def heading(icon: str, title: str, meta: str | None = None) -> str:
    return block(f"### {icon} {title}" + (f"\n_{meta}_" if meta else ""))


def step(text: str) -> str:
    """One progress line in a running list."""
    return f"- {one_line(text)}\n"


def section(title: str, bullets) -> str:
    items = [one_line(item) for item in bullets if str(item).strip()]
    if not items:
        return ""
    return block("\n".join([f"**{title}**", *(f"- {item}" for item in items)]))


def paragraph(label: str, text: str) -> str:
    return block(f"**{label}** {one_line(text)}") if str(text).strip() else ""


def quote(text: str) -> str:
    lines = [line.rstrip() for line in str(text).strip().splitlines()]
    return block("\n".join(f"> {line}" if line else ">" for line in lines)) if lines else ""


def paths(items) -> str:
    return ", ".join(f"`{item}`" for item in dict.fromkeys(items) if item)


def short_job(job_id: str | None) -> str:
    return (job_id or "")[:8]


def menu(lines) -> str:
    """Numbered reply menu; the `N. Reply \\`cmd\\`` shape is parsed by menu_choices."""
    return block("\n".join(["**What next?**", *lines]))


def progress_line(kind: str, text: str = "") -> str:
    """Live planner progress: header, narration, tool, note, or separator."""
    text = one_line(text)
    if kind == "header":
        return f"**{text}**\n\n"
    if kind == "narration":
        return f"\n💭 {text}\n"
    if kind == "tool":
        return f"- 🔎 {text}\n"
    if kind == "note":
        return f"- ⏹ {text}\n"
    if kind == "separator":
        return "\n---\n\n"
    raise ValueError(f"unknown progress kind {kind!r}")


def findings_list(findings) -> str:
    """Reviewer findings as a numbered list (objects with problem/suggested_task/...)."""
    lines = []
    for number, finding in enumerate(findings, 1):
        lines.append(f"{number}. **{one_line(finding.problem)}** — "
                     f"{one_line(finding.suggested_task)}")
        criteria = [one_line(item) for item in getattr(finding, "approval_criteria", ())]
        if criteria:
            lines.append("   - Approve when: " + "; ".join(criteria))
        files = paths(getattr(finding, "artifact_paths", ()))
        if files:
            lines.append(f"   - Files: {files}")
    return block("\n".join(lines)) if lines else ""


def validation_table(results) -> str:
    rows = [result for result in results or () if isinstance(result, dict)]
    if not rows:
        return ""
    lines = ["**Validation contract**", "| | Expectation | Result |", "|---|---|---|"]
    for result in rows:
        state = str(result.get("status", ""))
        icon = VALIDATION_ICONS.get(state, "•")
        expectation = one_line(result.get("expectation", "")).replace("|", "\\|")
        lines.append(f"| {result.get('category', '')} | {expectation} | {icon} {state} |")
    return block("\n".join(lines))


def coder_handoff(handoff: dict) -> str:
    """The coder's JSON handoff as a readable summary (the raw JSON stays in the report)."""
    if not handoff:
        return block("**Coder handoff** (unverified until review)\n"
                     "_No structured handoff was returned._")
    lines = ["**Coder handoff** (unverified until review)"]
    if handoff.get("summary"):
        lines.append(first_sentence(handoff["summary"]))
    changed = [entry for entry in handoff.get("changed_files") or []
               if isinstance(entry, dict) and entry.get("path")]
    if changed:
        lines.append("- Changed: " + ", ".join(
            f"`{entry['path']}`" + (f" ({entry['change']})" if entry.get("change") else "")
            for entry in changed))
    criteria = [item for item in handoff.get("criteria") or [] if isinstance(item, dict)]
    if criteria:
        satisfied = sum(1 for item in criteria if item.get("status") == "satisfied")
        lines.append(f"- Criteria: {satisfied} satisfied · {len(criteria) - satisfied} blocked")
    gaps = [one_line(gap) for gap in handoff.get("remaining_gaps") or [] if str(gap).strip()]
    if gaps:
        lines.append("- Remaining gaps: " + "; ".join(gaps))
    if handoff.get("execution_dispute"):
        lines.append("- Execution dispute raised: "
                     + one_line(json.dumps(handoff["execution_dispute"], ensure_ascii=False))[:300])
    return block("\n".join(lines))

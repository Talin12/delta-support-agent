"""Render artifacts/results.json into the markdown tables used in the report.

Generated rather than hand-transcribed so the report can't drift from the run.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

ORDER = ["trivial", "simple", "agent"]
LABEL = {"trivial": "Trivial", "simple": "Simple (no LLM)", "agent": "Agent"}


def ci(point: float, bounds, pct=True) -> str:
    scale = 100 if pct else 1
    suffix = "%" if pct else ""
    return f"{point*scale:.1f}{suffix} <sub>[{bounds[0]*scale:.1f}, {bounds[1]*scale:.1f}]</sub>"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def joint_metric(system: str) -> dict | None:
    """Auto-handled AND correctly so AND with a reply worth sending.

    Judged sendability alone is close to worthless here: Delta's own house style is
    a generic "DM us", so a generic deflection scores well on the rubric even when
    it is sent to someone alleging a safety incident. Routing accuracy alone is
    equally hollow — it says nothing about whether the reply was any good. Only the
    conjunction describes traffic actually taken off a human's queue.
    """
    pred_path = ARTIFACTS / f"predictions_{system}.jsonl"
    golden_path = REPO_ROOT / "data" / "golden" / "golden_set.jsonl"
    if not pred_path.exists():
        return None
    golden = {g["golden_id"]: g for g in _read_jsonl(golden_path)}
    rows = [r for r in _read_jsonl(pred_path) if golden.get(r["golden_id"], {}).get("stratum") == "representative"]
    if not rows or "judge" not in rows[0]:
        return None

    useful = harmful = 0
    for r in rows:
        g = golden[r["golden_id"]]
        chose_auto = r["route"] == "AUTO"
        if not chose_auto:
            continue
        if g["route"] == "AUTO" and r["judge"]["sendable"]:
            useful += 1
        if g["route"] == "ESCALATE":
            harmful += 1
    n = len(rows)
    return {
        "n": n,
        "useful": useful,
        "useful_rate": useful / n if n else 0.0,
        "harmful": harmful,
        "harmful_rate": harmful / n if n else 0.0,
    }


def main() -> None:
    results = {r["system"]: r for r in json.loads((ARTIFACTS / "results.json").read_text())}
    systems = [s for s in ORDER if s in results]
    lines: list[str] = []

    def emit(text=""):
        lines.append(text)

    # ---- intent -------------------------------------------------------
    emit("### 4.1 Intent classification\n")
    emit("| System | Accuracy (representative) | Macro-F1 (representative) | Accuracy (pooled) | Macro-F1 (pooled) |")
    emit("|---|---|---|---|---|")
    for s in systems:
        r = results[s]
        rep = r.get("by_stratum", {}).get("representative", {})
        emit(
            f"| {LABEL[s]} | {ci(rep.get('intent_accuracy', 0), rep.get('intent_accuracy_ci', [0, 0]))} "
            f"| {rep.get('intent_macro_f1', 0):.3f} "
            f"| {ci(r['intent']['accuracy'], r['intent']['accuracy_ci'])} "
            f"| {r['intent']['macro_f1']:.3f} |"
        )
    emit()

    # ---- routing ------------------------------------------------------
    emit("### 4.2 Routing\n")
    emit("Representative slice only (n=120), the one production-quotable stratum.\n")
    emit("| System | Automation rate | False-auto rate | Needless escalation | Safe automation |")
    emit("|---|---|---|---|---|")
    for s in systems:
        rt = results[s].get("by_stratum", {}).get("representative", {}).get("routing", {})
        if not rt:
            continue
        fa = rt.get("false_auto_rate")
        ne = rt.get("needless_escalation_rate")
        emit(
            f"| {LABEL[s]} | {rt.get('automation_rate', 0)*100:.1f}% "
            f"| {'n/a' if fa is None else f'{fa*100:.1f}%'} ({rt.get('false_auto_count', 0)}) "
            f"| {'n/a' if ne is None else f'{ne*100:.1f}%'} ({rt.get('needless_escalation_count', 0)}) "
            f"| {ci(rt.get('safe_automation_rate', 0), rt.get('safe_automation_ci', [0, 0]))} |"
        )
    emit()
    emit("Enriched slice (n=100), where the hard cases live.\n")
    emit("| System | Automation rate | False-auto rate | Needless escalation |")
    emit("|---|---|---|---|")
    for s in systems:
        rt = results[s].get("by_stratum", {}).get("enriched", {}).get("routing", {})
        if not rt:
            continue
        fa = rt.get("false_auto_rate")
        ne = rt.get("needless_escalation_rate")
        emit(
            f"| {LABEL[s]} | {rt.get('automation_rate', 0)*100:.1f}% "
            f"| {'n/a' if fa is None else f'{fa*100:.1f}%'} ({rt.get('false_auto_count', 0)}) "
            f"| {'n/a' if ne is None else f'{ne*100:.1f}%'} ({rt.get('needless_escalation_count', 0)}) |"
        )
    emit()

    # ---- reply quality ------------------------------------------------
    if any("reply_quality" in results[s] for s in systems):
        emit("### 4.3 Reply quality (blind LLM judge)\n")
        emit("| System | Sendable rate | Grounded | Addresses | Voice | Actionable | Safe | Mean |")
        emit("|---|---|---|---|---|---|---|---|")
        for s in systems:
            q = results[s].get("reply_quality")
            if not q:
                continue
            d = q["dimensions"]
            emit(
                f"| {LABEL[s]} | {ci(q['sendable_rate'], q['sendable_rate_ci'])} "
                f"| {d['grounded']['mean']:.2f} | {d['addresses']['mean']:.2f} "
                f"| {d['voice']['mean']:.2f} | {d['actionable']['mean']:.2f} "
                f"| {d['safe']['mean']:.2f} | **{q['mean_overall']:.2f}** |"
            )
        emit()

    # ---- the joint metric --------------------------------------------
    emit("### 4.4 Useful automation (the joint metric)\n")
    emit(
        "Reply quality and routing scored **together**. A reply can be perfectly\n"
        "good text and still be a catastrophic thing to auto-send, so neither number\n"
        "means anything alone. An example counts as useful automation only if the\n"
        "system chose AUTO, the human label agreed AUTO, and the judge called the\n"
        "reply sendable.\n"
    )
    emit("| System | Useful automation (representative) | Harmful automation (representative) |")
    emit("|---|---|---|")
    for s in systems:
        joint = joint_metric(s)
        if not joint:
            continue
        emit(
            f"| {LABEL[s]} | {joint['useful_rate']*100:.1f}% "
            f"({joint['useful']}/{joint['n']}) "
            f"| {joint['harmful_rate']*100:.1f}% ({joint['harmful']}/{joint['n']}) |"
        )
    emit()

    out = ARTIFACTS / "report_tables.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

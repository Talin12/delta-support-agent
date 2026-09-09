"""Extract concrete failure examples from a system's predictions.

Buckets errors by the mechanism that produced them rather than by intent label,
because "confused A for B" is a symptom and the mechanism is the thing you fix.
Prints real messages so the report can quote them verbatim instead of describing
failures in the abstract.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import policy  # noqa: E402
from prep import read_jsonl  # noqa: E402

GOLDEN = REPO_ROOT / "data" / "golden"
ARTIFACTS = REPO_ROOT / "artifacts"


def show(tag: str, items: list[tuple], limit: int) -> None:
    print(f"\n{'=' * 72}\n{tag}  (n={len(items)})\n{'=' * 72}")
    for g, p, note in items[:limit]:
        print(f"\n[{g['golden_id']} | {g['stratum']}/{g['enrichment_tag']}]")
        print(f"  MSG   : {g['message'][:200]}")
        print(f"  TRUTH : intent={g['intent']}  route={g['route']}")
        print(f"          {g['route_reason'][:120]}")
        print(f"  PRED  : intent={p['intent']} (conf {p['intent_confidence']})  route={p['route']} [{p['route_source']}]")
        print(f"          {p['route_reason'][:120]}")
        if p.get("reply"):
            print(f"  REPLY : {p['reply'][:160]}")
        if note:
            print(f"  NOTE  : {note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default="agent")
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    golden = {g["golden_id"]: g for g in read_jsonl(GOLDEN / "golden_set.jsonl")}
    preds = {p["golden_id"]: p for p in read_jsonl(ARTIFACTS / f"predictions_{args.system}.jsonl")}
    pairs = [(golden[k], preds[k]) for k in preds if k in golden]
    print(f"analysing {len(pairs)} predictions from '{args.system}'")

    false_auto, needless, rule_fp, intent_err, ungrounded = [], [], [], [], []

    for g, p in pairs:
        if g["route"] == policy.ESCALATE and p["route"] == policy.AUTO:
            false_auto.append((g, p, "auto-sent a case that needed a human"))
        if g["route"] == policy.AUTO and p["route"] == policy.ESCALATE:
            src = p["route_source"]
            note = f"escalated by {src}"
            needless.append((g, p, note))
            if src.startswith("rule:") and src != "rule:low_confidence":
                rule_fp.append((g, p, f"deterministic {src} misfired on a routine message"))
        if g["intent"] != p["intent"]:
            intent_err.append((g, p, f"{g['intent']} -> {p['intent']}"))
        reply = p.get("reply", "")
        if "<link>" in reply:
            ungrounded.append((g, p, "leaked the <link> cleaning placeholder into the reply"))

    # Signature fabrication: Delta agents sign with their real initials. A generated
    # reply carrying an initials sign-off is impersonating a specific employee.
    import re

    sig = re.compile(r"\*[A-Z]{2,3}\b")
    fabricated_sig = [(g, p, f"signed as {sig.search(p['reply']).group()}")
                      for g, p in pairs if p.get("reply") and sig.search(p["reply"])]

    print("\nfailure counts")
    for name, items in [
        ("false auto (needed human, auto-sent)", false_auto),
        ("needless escalation", needless),
        ("  of which deterministic rule false positives", rule_fp),
        ("intent errors", intent_err),
        ("replies leaking <link> placeholder", ungrounded),
        ("replies fabricating an agent signature", fabricated_sig),
    ]:
        print(f"  {name:48s} {len(items):4d}")

    print("\nmost common intent confusions")
    for (pair, n) in Counter(note for _, _, note in intent_err).most_common(8):
        print(f"  {pair:52s} {n:3d}")

    show("FALSE AUTO — the dangerous error", false_auto, args.limit)
    show("DETERMINISTIC RULE FALSE POSITIVES", rule_fp, args.limit)
    show("INTENT ERRORS", intent_err, args.limit)
    if fabricated_sig:
        show("FABRICATED AGENT SIGNATURE", fabricated_sig, 3)

    summary = {
        "system": args.system,
        "n": len(pairs),
        "false_auto": len(false_auto),
        "needless_escalation": len(needless),
        "rule_false_positives": len(rule_fp),
        "intent_errors": len(intent_err),
        "link_placeholder_leak": len(ungrounded),
        "fabricated_signature": len(fabricated_sig),
        "top_confusions": Counter(note for _, _, note in intent_err).most_common(10),
    }
    out = ARTIFACTS / f"failures_{args.system}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

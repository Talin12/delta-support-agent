"""How well does the LLM judge agree with a human?

Pairs the hand scores in data/golden/human_reply_scores.jsonl with the judge's
scores for the same replies. Reports weighted kappa per dimension, agreement on
the sendable call, and the judge's bias — a judge can correlate well and still sit
a point high, which would overstate absolute quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from prep import read_jsonl  # noqa: E402
from retrieve import ResolutionIndex  # noqa: E402
from prep import INTERIM  # noqa: E402

import judge as judging  # noqa: E402

GOLDEN = REPO_ROOT / "data" / "golden"
ARTIFACTS = REPO_ROOT / "artifacts"


def main() -> None:
    human_path = GOLDEN / "human_reply_scores.jsonl"
    if not human_path.exists():
        raise SystemExit(
            f"{human_path} not found. Run scripts/sample_for_human_scoring.py and "
            "score the replies first."
        )
    human_rows = [r for r in read_jsonl(human_path) if r.get("grounded") is not None]
    print(f"human-scored replies: {len(human_rows)}")

    golden = {g["golden_id"]: g for g in read_jsonl(GOLDEN / "golden_set.jsonl")}
    threads = read_jsonl(INTERIM / "Delta_threads.jsonl")
    index = ResolutionIndex(threads, exclude_thread_ids={g["thread_id"] for g in golden.values()})

    # Re-score with the judge. These calls hit the committed cache when the harness
    # has already judged the same (message, reply) pair, so this is usually free.
    machine_rows = []
    for i, row in enumerate(human_rows, 1):
        exemplars = index.search(row["message"], k=3)
        score = judging.judge_reply("Delta", row["message"], row["reply"], exemplars)
        machine_rows.append(score.to_dict())
        if i % 20 == 0:
            print(f"  judged {i}/{len(human_rows)}")

    report = judging.agreement_report(human_rows, machine_rows)

    # Per-system breakdown: a judge can look well-calibrated overall while being
    # systematically generous to one system.
    by_system: dict[str, dict] = {}
    for system in sorted({r["_system"] for r in human_rows}):
        idx = [i for i, r in enumerate(human_rows) if r["_system"] == system]
        if len(idx) < 3:
            continue
        h = [human_rows[i] for i in idx]
        m = [machine_rows[i] for i in idx]
        hm = sum(sum(x[d] for d in judging.DIMENSIONS) / 5 for x in h) / len(h)
        mm = sum(sum(x[d] for d in judging.DIMENSIONS) / 5 for x in m) / len(m)
        by_system[system] = {
            "n": len(idx),
            "human_mean": round(hm, 2),
            "judge_mean": round(mm, 2),
            "judge_bias": round(mm - hm, 2),
            "sendable_raw_agreement": round(
                sum(bool(a["sendable"]) == bool(b["sendable"]) for a, b in zip(h, m)) / len(h), 3
            ),
        }
    report["by_system"] = by_system

    out = ARTIFACTS / "judge_agreement.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nJudge vs human (n={report['n']})\n")
    print(f"{'dimension':12s} {'QWK':>6s}  {'label':16s} {'±1':>6s} {'human':>6s} {'judge':>6s} {'bias':>6s}")
    for dim, d in report["dimensions"].items():
        print(
            f"{dim:12s} {d['qwk']:6.3f}  {d['qwk_label']:16s} "
            f"{d['within_one']:6.2f} {d['human_mean']:6.2f} {d['judge_mean']:6.2f} {d['judge_bias']:+6.2f}"
        )
    s = report["sendable"]
    print(
        f"\nsendable: kappa {s['kappa']:.3f} ({s['kappa_label']}), "
        f"raw agreement {s['raw_agreement']:.3f}\n"
        f"  human says sendable {s['human_sendable_rate']:.1%} | "
        f"judge says {s['judge_sendable_rate']:.1%}"
    )
    print(f"\nspearman on mean score: {report['overall']['spearman_mean_score']:.3f}")
    print("\nper system:")
    for system, d in by_system.items():
        print(
            f"  {system:8s} n={d['n']:3d}  human {d['human_mean']:.2f}  "
            f"judge {d['judge_mean']:.2f}  bias {d['judge_bias']:+.2f}"
        )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

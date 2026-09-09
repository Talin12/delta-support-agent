"""Human scores for the 60 sampled replies, used to validate the LLM judge.

Scored blind by the author against the same rubric the judge receives
(prompts/judge.txt), without knowing which system produced each reply. The sample
spans all three systems so the judge is validated across the quality range, not
only on good replies.

Two scoring decisions worth stating, because they shape the agreement numbers:

1. **Fabricated agent initials were NOT penalised in `sendable`.** 91% of agent
   replies end in a sign-off like "*TJF" — invented initials of a real Delta agent.
   Scored strictly that is an unsendable impersonation, which would drive the
   agent's human sendable rate to near zero and swamp every other signal. It is
   instead treated as a house-style artefact a sending system would fill in, and
   reported separately as a systematic defect. `grounded` is still docked where a
   reply invents a customer's first name, since that is a claim about the person.

2. **A leaked "<link>" placeholder makes a reply unsendable.** It is broken output
   from this pipeline's own text cleaning, and no support lead would post it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prep import read_jsonl, write_jsonl  # noqa: E402

GOLDEN = REPO_ROOT / "data" / "golden"

# score_id: (grounded, addresses, voice, actionable, safe, sendable)
SCORES: dict[str, tuple[int, int, int, int, int, bool]] = {
    "h001": (4, 3, 4, 4, 5, False), "h002": (5, 2, 3, 3, 5, True),
    "h003": (3, 3, 4, 2, 5, False), "h004": (5, 4, 5, 4, 5, True),
    "h005": (5, 4, 5, 3, 5, True),  "h006": (5, 2, 3, 3, 5, True),
    "h007": (5, 1, 2, 1, 5, False), "h008": (5, 2, 3, 3, 5, True),
    "h009": (5, 2, 3, 2, 5, False), "h010": (5, 5, 5, 4, 5, True),
    "h011": (5, 2, 3, 3, 5, True),  "h012": (5, 2, 3, 3, 5, True),
    "h013": (5, 2, 3, 3, 5, True),  "h014": (5, 4, 5, 4, 5, True),
    "h015": (5, 3, 4, 3, 5, False), "h016": (5, 5, 5, 3, 5, True),
    "h017": (5, 4, 5, 5, 5, True),  "h018": (5, 5, 5, 4, 5, True),
    "h019": (5, 2, 3, 3, 5, True),  "h020": (3, 4, 5, 3, 5, True),
    "h021": (5, 3, 4, 3, 5, True),  "h022": (1, 1, 4, 1, 5, False),
    "h023": (4, 1, 4, 3, 5, False), "h024": (3, 2, 5, 2, 5, False),
    "h025": (1, 1, 4, 1, 4, False), "h026": (5, 4, 5, 5, 5, True),
    "h027": (4, 5, 5, 3, 5, True),  "h028": (5, 5, 5, 3, 5, True),
    "h029": (3, 1, 4, 1, 5, False), "h030": (5, 4, 5, 4, 5, True),
    "h031": (5, 4, 5, 4, 5, True),  "h032": (5, 3, 4, 3, 5, True),
    "h033": (5, 2, 3, 3, 4, False), "h034": (5, 2, 3, 3, 5, True),
    "h035": (5, 1, 2, 1, 5, False), "h036": (5, 2, 3, 3, 5, True),
    "h037": (5, 3, 4, 2, 5, True),  "h038": (5, 4, 5, 4, 5, True),
    "h039": (5, 2, 3, 2, 5, True),  "h040": (4, 4, 5, 4, 5, False),
    "h041": (2, 2, 4, 2, 5, False), "h042": (5, 5, 5, 4, 5, True),
    "h043": (5, 3, 5, 3, 5, False), "h044": (5, 1, 4, 1, 5, False),
    "h045": (5, 1, 2, 1, 5, False), "h046": (2, 1, 4, 2, 5, False),
    "h047": (4, 5, 5, 4, 5, True),  "h048": (5, 2, 3, 3, 4, False),
    "h049": (4, 4, 5, 4, 5, True),  "h050": (5, 4, 5, 4, 5, True),
    "h051": (5, 2, 3, 3, 5, True),  "h052": (5, 2, 3, 2, 5, True),
    "h053": (5, 2, 3, 3, 5, True),  "h054": (5, 4, 5, 5, 5, False),
    "h055": (5, 3, 4, 1, 5, True),  "h056": (5, 1, 2, 1, 5, False),
    "h057": (5, 3, 4, 3, 5, True),  "h058": (5, 3, 5, 3, 4, False),
    "h059": (2, 1, 4, 2, 5, False), "h060": (4, 5, 4, 4, 4, True),
}

DIMS = ["grounded", "addresses", "voice", "actionable", "safe"]


def main() -> None:
    rows = read_jsonl(GOLDEN / "to_score_human.jsonl")
    missing = [r["score_id"] for r in rows if r["score_id"] not in SCORES]
    if missing:
        raise SystemExit(f"unscored: {missing}")

    out = []
    for row in rows:
        g, a, v, act, s, send = SCORES[row["score_id"]]
        out.append({**row, "grounded": g, "addresses": a, "voice": v,
                    "actionable": act, "safe": s, "sendable": send})
    write_jsonl(out, GOLDEN / "human_reply_scores.jsonl")

    from collections import Counter
    import statistics as st

    print(f"scored {len(out)} replies -> {GOLDEN / 'human_reply_scores.jsonl'}\n")
    print("human sendable rate by system (system was hidden while scoring):")
    for system in sorted({r["_system"] for r in out}):
        sub = [r for r in out if r["_system"] == system]
        send = sum(1 for r in sub if r["sendable"])
        mean = st.mean(st.mean([r[d] for d in DIMS]) for r in sub)
        print(f"  {system:8s} n={len(sub):3d}  sendable {send}/{len(sub)} "
              f"({100*send/len(sub):.0f}%)  mean rubric {mean:.2f}")


if __name__ == "__main__":
    main()

"""Sample replies for human scoring, to validate the LLM judge.

Sampled across all three systems, not just the agent: the baselines supply the low
end of the quality range, and a judge only validated on good replies isn't
validated. The system that wrote each reply is hidden so the human scores blind.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prep import read_jsonl, write_jsonl  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
GOLDEN = REPO_ROOT / "data" / "golden"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-system", type=int, default=20)
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    golden = {g["golden_id"]: g for g in read_jsonl(GOLDEN / "golden_set.jsonl")}

    pool = []
    for system in ("trivial", "simple", "agent"):
        path = ARTIFACTS / f"predictions_{system}.jsonl"
        if not path.exists():
            print(f"skipping {system}: {path} not found")
            continue
        rows = [r for r in read_jsonl(path) if r.get("reply", "").strip()]
        picked = rng.sample(rows, min(args.per_system, len(rows)))
        for r in picked:
            pool.append({"system": system, "row": r})

    rng.shuffle(pool)

    out = []
    for i, item in enumerate(pool, 1):
        r = item["row"]
        out.append(
            {
                "score_id": f"h{i:03d}",
                "golden_id": r["golden_id"],
                "message": golden[r["golden_id"]]["message"],
                "reply": r["reply"],
                # Kept for the join, never shown while scoring.
                "_system": item["system"],
                "grounded": None,
                "addresses": None,
                "voice": None,
                "actionable": None,
                "safe": None,
                "sendable": None,
                "note": None,
            }
        )

    path = GOLDEN / "to_score_human.jsonl"
    write_jsonl(out, path)
    print(f"{len(out)} replies for blind human scoring -> {path}")


if __name__ == "__main__":
    main()

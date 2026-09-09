"""Sample threads for the golden evaluation set.

Two strata, tagged in the output so metrics can be reported separately:
  representative (120) - uniform random; the only slice a production estimate may
      be quoted from
  enriched (100) - oversamples rule-triggering, long-running, low-precedent and
      very short messages. At a ~1.8% base rate a uniform 220 would contain about
      four safety-critical messages, too few to measure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import policy  # noqa: E402
from prep import INTERIM, clean_text, read_jsonl, write_jsonl  # noqa: E402
from retrieve import ResolutionIndex, first_brand_reply  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "data" / "golden"


def eligible(thread: dict) -> bool:
    msg = clean_text(thread["first_customer_message"])
    return len(msg) >= 12 and first_brand_reply(thread) is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default="Delta")
    ap.add_argument("--n-representative", type=int, default=120)
    ap.add_argument("--n-enriched", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    threads = [t for t in read_jsonl(INTERIM / f"{args.brand}_threads.jsonl") if eligible(t)]
    print(f"eligible threads: {len(threads):,}")

    by_id = {t["thread_id"]: t for t in threads}

    # ---- Stratum A: representative -------------------------------------
    representative = rng.sample(threads, args.n_representative)
    chosen = {t["thread_id"] for t in representative}

    # ---- Stratum B: enriched tail --------------------------------------
    remaining = [t for t in threads if t["thread_id"] not in chosen]

    rule_hits = [
        t for t in remaining if policy.check_rules(clean_text(t["first_customer_message"]))[0]
    ]
    long_threads = [t for t in remaining if t["n_turns"] >= 6]
    short_vague = [
        t for t in remaining if len(clean_text(t["first_customer_message"])) < 60
    ]

    print("building retrieval index to find low-precedent messages")
    index = ResolutionIndex(threads)
    novel = []
    probe = rng.sample(remaining, min(3000, len(remaining)))
    for t in probe:
        hits = index.search(t["first_customer_message"], k=1)
        top = hits[0].score if hits else 0.0
        novel.append((top, t))
    novel.sort(key=lambda x: x[0])
    novel_threads = [t for _, t in novel[:400]]

    quotas = [
        ("rule_triggered", rule_hits, 40),
        ("long_thread", long_threads, 25),
        ("low_precedent", novel_threads, 20),
        ("short_vague", short_vague, 15),
    ]

    enriched: list[tuple[str, dict]] = []
    for tag, pool, quota in quotas:
        pool = [t for t in pool if t["thread_id"] not in chosen]
        picked = rng.sample(pool, min(quota, len(pool)))
        for t in picked:
            chosen.add(t["thread_id"])
            enriched.append((tag, t))
        print(f"  {tag}: pool {len(pool):,} -> took {len(picked)}")

    # ---- Emit ----------------------------------------------------------
    rows = []
    for i, t in enumerate(representative):
        rows.append(("representative", "uniform_random", t))
    for tag, t in enriched:
        rows.append(("enriched", tag, t))
    rng.shuffle(rows)

    out = []
    for i, (stratum, tag, t) in enumerate(rows, 1):
        rule_name, _ = policy.check_rules(clean_text(t["first_customer_message"]))
        out.append(
            {
                "golden_id": f"g{i:04d}",
                "thread_id": t["thread_id"],
                "stratum": stratum,
                "enrichment_tag": tag,
                "message": clean_text(t["first_customer_message"]),
                "n_turns": t["n_turns"],
                "rule_triggered": rule_name,
                # Reference material for the labeller. See codebook.md: labelling the
                # route WITH knowledge of what Delta actually did is a deliberate
                # choice, traded against some hindsight bias.
                "historical_brand_reply": clean_text(first_brand_reply(t) or ""),
                "intent": None,
                "route": None,
                "route_reason": None,
            }
        )

    path = GOLDEN_DIR / "to_label.jsonl"
    write_jsonl(out, path)
    print(f"\n{len(out)} examples -> {path}")
    print(f"  representative: {sum(1 for r in out if r['stratum'] == 'representative')}")
    print(f"  enriched:       {sum(1 for r in out if r['stratum'] == 'enriched')}")


if __name__ == "__main__":
    main()

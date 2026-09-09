"""Derive a candidate intent taxonomy from the brand's own traffic.

Clusters first-contact messages with TF-IDF + KMeans and dumps the top terms and
sample messages per cluster. Optionally asks an LLM to name the intents; with
--no-llm it stops at the evidence and the taxonomy is written by hand from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from llm import complete_json
from prep import INTERIM, clean_text, read_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

TAXONOMY_PROMPT = """You are designing an intent taxonomy for a customer support agent that handles \
incoming public messages for the brand "{brand}" on Twitter.

Below are {k} clusters discovered by unsupervised clustering of real first-contact customer \
messages. For each cluster you get its distinctive terms and a random sample of real messages.

{clusters}

Design a taxonomy of {target_min}-{target_max} intents that:
- covers the traffic above, with each intent grounded in the clusters you can point to
- is mutually exclusive: a support agent reading one message should rarely hesitate between two
- is actionable: two intents should differ in what the agent DOES, not just in wording
- includes exactly one catch-all intent for messages that fit nothing else
- splits clusters that mix different required actions, and merges clusters that need the same action

Return strict JSON, no prose:
{{
  "intents": [
    {{
      "name": "snake_case_name",
      "description": "one sentence a human labeller can apply consistently",
      "boundary": "the nearest neighbouring intent and the rule that separates them",
      "example_messages": ["verbatim example from the clusters above", "another"],
      "source_clusters": [0, 3]
    }}
  ],
  "rejected": [{{"idea": "intent considered", "why_rejected": "reason"}}]
}}"""


def cluster_messages(messages: list[str], k: int, seed: int = 13):
    vec = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.45,
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(messages)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(matrix)
    terms = vec.get_feature_names_out()
    top_terms = []
    for centre in km.cluster_centers_:
        idx = centre.argsort()[::-1][:12]
        top_terms.append([terms[i] for i in idx])
    return labels, top_terms


def format_clusters(messages: list[str], labels, top_terms, n_examples: int = 8) -> str:
    import random

    rng = random.Random(7)
    blocks = []
    for cid, terms in enumerate(top_terms):
        members = [m for m, lab in zip(messages, labels) if lab == cid]
        sample = rng.sample(members, min(n_examples, len(members)))
        quoted = "\n".join(f'    - "{s[:220]}"' for s in sample)
        blocks.append(
            f"CLUSTER {cid} (n={len(members)})\n"
            f"  distinctive terms: {', '.join(terms)}\n"
            f"  sample messages:\n{quoted}"
        )
    return "\n\n".join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", required=True)
    ap.add_argument("--k", type=int, default=14, help="KMeans clusters (over-cluster, then merge)")
    ap.add_argument("--sample", type=int, default=4000, help="messages to cluster")
    ap.add_argument("--target-min", type=int, default=6)
    ap.add_argument("--target-max", type=int, default=9)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="emit cluster evidence only; the taxonomy is then written by hand from it",
    )
    args = ap.parse_args()

    threads = read_jsonl(INTERIM / f"{args.brand}_threads.jsonl")
    import random

    rng = random.Random(args.seed)
    pool = [clean_text(t["first_customer_message"]) for t in threads]
    pool = [m for m in pool if len(m) >= 15]
    messages = rng.sample(pool, min(args.sample, len(pool)))
    print(f"clustering {len(messages):,} first-contact messages into k={args.k}")

    labels, top_terms = cluster_messages(messages, args.k, args.seed)
    evidence = format_clusters(messages, labels, top_terms)

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "cluster_evidence.txt").write_text(evidence, encoding="utf-8")
    print(f"cluster evidence -> {ARTIFACTS / 'cluster_evidence.txt'}")

    if args.no_llm:
        print("--no-llm: stopping after cluster evidence")
        return

    prompt = TAXONOMY_PROMPT.format(
        brand=args.brand,
        k=args.k,
        clusters=evidence,
        target_min=args.target_min,
        target_max=args.target_max,
    )
    proposal = complete_json(prompt, temperature=0.0, max_tokens=4096, model_tag="taxonomy")
    out = ARTIFACTS / "taxonomy_proposal.json"
    out.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"proposed {len(proposal.get('intents', []))} intents -> {out}")
    for intent in proposal.get("intents", []):
        print(f"  - {intent['name']}: {intent['description']}")


if __name__ == "__main__":
    main()

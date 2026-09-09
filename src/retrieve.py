"""Retrieve how the brand historically resolved similar issues.

Grounding the draft matters more than fluency here: the agent should reply the way
this brand actually replies, not the way a generic assistant would. We index the
brand's own first-contact messages and hand the model the replies that followed.

Leakage control: any thread in the golden set is excluded from the index, so an
evaluated message can never retrieve its own historical answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from prep import clean_text


@dataclass
class Exemplar:
    thread_id: int
    customer_message: str
    brand_reply: str
    score: float
    n_turns: int


def first_brand_reply(thread: dict) -> str | None:
    for turn in thread["turns"]:
        if not turn["inbound"]:
            return turn["text"]
    return None


class ResolutionIndex:
    """TF-IDF nearest-neighbour index over (customer message -> brand reply) pairs."""

    def __init__(self, threads: list[dict], *, exclude_thread_ids: set[int] | None = None):
        exclude = exclude_thread_ids or set()
        self.records: list[dict] = []
        for thread in threads:
            if thread["thread_id"] in exclude:
                continue
            reply = first_brand_reply(thread)
            if not reply:
                continue
            msg = clean_text(thread["first_customer_message"])
            if len(msg) < 15:
                continue
            self.records.append(
                {
                    "thread_id": thread["thread_id"],
                    "customer_message": msg,
                    "brand_reply": clean_text(reply, drop_handles=True),
                    "n_turns": thread["n_turns"],
                }
            )
        if not self.records:
            raise ValueError("no indexable threads after filtering")

        self.vectorizer = TfidfVectorizer(
            max_features=40000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(
            [r["customer_message"] for r in self.records]
        )

    def __len__(self) -> int:
        return len(self.records)

    def search(self, message: str, k: int = 4, min_score: float = 0.05) -> list[Exemplar]:
        query = self.vectorizer.transform([clean_text(message)])
        sims = cosine_similarity(query, self.matrix).ravel()
        order = np.argsort(sims)[::-1][:k]
        out = []
        for idx in order:
            score = float(sims[idx])
            if score < min_score:
                continue
            rec = self.records[idx]
            out.append(
                Exemplar(
                    thread_id=rec["thread_id"],
                    customer_message=rec["customer_message"],
                    brand_reply=rec["brand_reply"],
                    score=round(score, 4),
                    n_turns=rec["n_turns"],
                )
            )
        return out

    def save_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"n_indexed": len(self.records), "vocab": len(self.vectorizer.vocabulary_)},
                indent=2,
            ),
            encoding="utf-8",
        )


def format_exemplars(exemplars: list[Exemplar]) -> str:
    if not exemplars:
        return "(no similar historical case found)"
    blocks = []
    for i, ex in enumerate(exemplars, 1):
        blocks.append(
            f"[case {i} | similarity {ex.score:.2f} | thread ran {ex.n_turns} turns]\n"
            f"  customer: {ex.customer_message[:300]}\n"
            f"  {'brand replied'}: {ex.brand_reply[:300]}"
        )
    return "\n".join(blocks)

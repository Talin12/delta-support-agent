"""LLM-as-judge for reply quality, plus the agreement analysis that validates it.

A judge score is worthless until you know how far it drifts from a human. So this
module does two jobs: it scores replies, and it measures itself against the human
scores in data/golden/human_reply_scores.jsonl using quadratic-weighted Cohen's
kappa (the right statistic for ordered 1-5 anchors, since it penalises a 5-vs-1
disagreement far more than 5-vs-4).

Blinding: replies are scored one at a time with no system identity attached, so the
judge cannot favour the agent over a baseline it can recognise.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import llm  # noqa: E402
from retrieve import format_exemplars  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"

DIMENSIONS = ["grounded", "addresses", "voice", "actionable", "safe"]


@dataclass
class JudgeScore:
    grounded: int
    addresses: int
    voice: int
    actionable: int
    safe: int
    sendable: bool
    worst_problem: str
    error: str | None = None

    def mean_score(self) -> float:
        return float(np.mean([getattr(self, d) for d in DIMENSIONS]))

    def to_dict(self) -> dict:
        return {
            **{d: getattr(self, d) for d in DIMENSIONS},
            "sendable": self.sendable,
            "worst_problem": self.worst_problem,
            "mean_score": round(self.mean_score(), 3),
            "error": self.error,
        }


def _clamp(value, lo=1, hi=5) -> int:
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return lo


def judge_reply(brand: str, message: str, reply: str, exemplars) -> JudgeScore:
    if not reply.strip():
        return JudgeScore(1, 1, 1, 1, 1, False, "empty reply")
    template = (PROMPTS / "judge.txt").read_text(encoding="utf-8")
    prompt = template.format(
        brand=brand,
        message=message,
        exemplars=format_exemplars(exemplars) if not isinstance(exemplars, str) else exemplars,
        reply=reply,
    )
    try:
        out = llm.complete_json(prompt, temperature=0.0, max_tokens=2500, model_tag="judge", model_override=os.environ.get("JUDGE_MODEL", ""))
    except Exception as exc:  # noqa: BLE001
        return JudgeScore(1, 1, 1, 1, 1, False, "judge error", error=str(exc)[:200])
    return JudgeScore(
        grounded=_clamp(out.get("grounded")),
        addresses=_clamp(out.get("addresses")),
        voice=_clamp(out.get("voice")),
        actionable=_clamp(out.get("actionable")),
        safe=_clamp(out.get("safe")),
        sendable=bool(out.get("sendable", False)),
        worst_problem=str(out.get("worst_problem", ""))[:200],
    )


# --------------------------------------------------------------------------
# Agreement statistics
# --------------------------------------------------------------------------


def quadratic_weighted_kappa(a: list[int], b: list[int], min_rating=1, max_rating=5) -> float:
    """Cohen's kappa with quadratic weights, for ordinal ratings."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = max_rating - min_rating + 1
    observed = np.zeros((n, n))
    for x, y in zip(a, b):
        observed[x - min_rating, y - min_rating] += 1

    hist_a = np.bincount(a - min_rating, minlength=n)
    hist_b = np.bincount(b - min_rating, minlength=n)
    expected = np.outer(hist_a, hist_b) / len(a)

    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    denom = (weights * expected).sum()
    if denom == 0:
        return 1.0
    return float(1 - (weights * observed).sum() / denom)


def binary_kappa(a: list[bool], b: list[bool]) -> float:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    observed_agree = float((a == b).mean())
    p_a = a.mean()
    p_b = b.mean()
    expected_agree = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected_agree == 1.0:
        return 1.0
    return float((observed_agree - expected_agree) / (1 - expected_agree))


def spearman(a: list[float], b: list[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def interpret_kappa(k: float) -> str:
    if np.isnan(k):
        return "undefined"
    if k < 0.0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def agreement_report(human: list[dict], machine: list[dict]) -> dict:
    """Compare paired human and judge scorings of the same replies."""
    assert len(human) == len(machine), "human and judge scorings must be paired"
    report: dict = {"n": len(human), "dimensions": {}}

    for dim in DIMENSIONS:
        h = [int(x[dim]) for x in human]
        m = [int(x[dim]) for x in machine]
        exact = float(np.mean([hi == mi for hi, mi in zip(h, m)]))
        within_one = float(np.mean([abs(hi - mi) <= 1 for hi, mi in zip(h, m)]))
        report["dimensions"][dim] = {
            "qwk": round(quadratic_weighted_kappa(h, m), 3),
            "qwk_label": interpret_kappa(quadratic_weighted_kappa(h, m)),
            "exact_agreement": round(exact, 3),
            "within_one": round(within_one, 3),
            "human_mean": round(float(np.mean(h)), 2),
            "judge_mean": round(float(np.mean(m)), 2),
            "judge_bias": round(float(np.mean(m) - np.mean(h)), 2),
        }

    hs = [bool(x["sendable"]) for x in human]
    ms = [bool(x["sendable"]) for x in machine]
    tp = sum(1 for a, b in zip(hs, ms) if a and b)
    fp = sum(1 for a, b in zip(hs, ms) if not a and b)
    fn = sum(1 for a, b in zip(hs, ms) if a and not b)
    report["sendable"] = {
        "kappa": round(binary_kappa(hs, ms), 3),
        "kappa_label": interpret_kappa(binary_kappa(hs, ms)),
        "raw_agreement": round(float(np.mean([a == b for a, b in zip(hs, ms)])), 3),
        "human_sendable_rate": round(float(np.mean(hs)), 3),
        "judge_sendable_rate": round(float(np.mean(ms)), 3),
        "judge_precision_vs_human": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "judge_recall_vs_human": round(tp / (tp + fn), 3) if (tp + fn) else None,
    }

    h_mean = [float(np.mean([x[d] for d in DIMENSIONS])) for x in human]
    m_mean = [float(np.mean([x[d] for d in DIMENSIONS])) for x in machine]
    report["overall"] = {
        "spearman_mean_score": round(spearman(h_mean, m_mean), 3),
        "mean_abs_error": round(float(np.mean(np.abs(np.array(h_mean) - np.array(m_mean)))), 3),
    }
    return report


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

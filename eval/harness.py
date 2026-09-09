"""Run every system over the golden set and score it.

Reports three families of metric, because the three stages fail differently:

  intent  - accuracy and macro-F1. Macro-F1 is the one that matters: accuracy on a
            skewed taxonomy mostly measures the majority class, which is exactly
            what the trivial baseline exploits.

  routing - not just precision/recall. The two errors have wildly different costs:
            auto-sending something that needed a human (false auto) can be a legal
            or PR incident; escalating something routine just costs an agent a
            minute. So we report them separately and never average them away.

  reply   - blind LLM-judge rubric scores, reported with bootstrap CIs.

Every headline number carries a 95% bootstrap confidence interval. At n=200 a
three-point gap is not a result, and the intervals make that impossible to hide.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

import llm  # noqa: E402
import policy  # noqa: E402
from agent import SupportAgent, load_taxonomy  # noqa: E402
from baselines import SimpleBaseline, TrivialBaseline  # noqa: E402
from prep import INTERIM, clean_text, read_jsonl, write_jsonl  # noqa: E402
from retrieve import ResolutionIndex  # noqa: E402

import judge as judging  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "data" / "golden"
ARTIFACTS = REPO_ROOT / "artifacts"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 13) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def fmt_ci(point: float, ci: tuple[float, float], pct: bool = True) -> str:
    scale = 100 if pct else 1
    unit = "%" if pct else ""
    return f"{point * scale:.1f}{unit} [{ci[0] * scale:.1f}, {ci[1] * scale:.1f}]"


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict]:
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for t in y_true if t == label)
        per_class[label] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": support,
        }
        if support > 0:
            f1s.append(f1)
    return (float(np.mean(f1s)) if f1s else 0.0), per_class


def confusion(y_true: list[str], y_pred: list[str]) -> dict:
    table: dict[str, Counter] = {}
    for t, p in zip(y_true, y_pred):
        table.setdefault(t, Counter())[p] += 1
    return {k: dict(v) for k, v in table.items()}


def routing_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """ESCALATE is the positive class. The two error types are kept separate."""
    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == policy.ESCALATE and p == policy.ESCALATE)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == policy.AUTO and p == policy.ESCALATE)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == policy.ESCALATE and p == policy.AUTO)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == policy.AUTO and p == policy.AUTO)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    n_should_escalate = tp + fn
    n_should_auto = tn + fp

    correct_auto = [
        1.0 if (p == policy.AUTO and t == policy.AUTO) else 0.0
        for t, p in zip(y_true, y_pred)
    ]
    return {
        "n": n,
        "escalation_precision": round(prec, 3),
        "escalation_recall": round(rec, 3),
        "escalation_f1": round(2 * prec * rec / (prec + rec), 3) if (prec + rec) else 0.0,
        # The dangerous error: needed a human, got auto-sent anyway.
        "false_auto_rate": round(fn / n_should_escalate, 3) if n_should_escalate else None,
        "false_auto_count": fn,
        # The merely expensive error: routine ticket sent to a human.
        "needless_escalation_rate": round(fp / n_should_auto, 3) if n_should_auto else None,
        "needless_escalation_count": fp,
        # Share of all traffic the system chose to answer without a human.
        "automation_rate": round((tn + fn) / n, 3) if n else 0.0,
        "safe_automation_rate": round(tn / n, 3) if n else 0.0,
        "safe_automation_ci": bootstrap_ci(correct_auto),
        "accuracy": round((tp + tn) / n, 3) if n else 0.0,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def build_systems(brand: str, threads: list[dict], golden: list[dict], seed: int):
    """Construct agent + baselines, with golden threads excluded from retrieval."""
    golden_ids = {g["thread_id"] for g in golden}
    index = ResolutionIndex(threads, exclude_thread_ids=golden_ids)
    print(f"retrieval index: {len(index):,} historical cases (golden threads excluded)")

    taxonomy = load_taxonomy(GOLDEN_DIR / "taxonomy.json")
    agent = SupportAgent(brand, taxonomy, index)
    trivial = TrivialBaseline.fit([g["intent"] for g in golden])
    simple = SimpleBaseline(index, seed=seed)
    return {"trivial": trivial, "simple": simple, "agent": agent}, index


def run_system(name, system, golden: list[dict], seed: int = 13) -> list[dict]:
    """Run a system over the golden set.

    The simple baseline is special-cased: it is a supervised classifier with no
    separate training corpus, so it is evaluated by 5-fold cross-validation over
    the golden set. Every prediction therefore comes from a model that never saw
    that example, which keeps the comparison against the agent honest.
    """
    if isinstance(system, SimpleBaseline):
        return _run_simple_cv(system, golden, seed)

    workers = int(os.environ.get("EVAL_WORKERS", "6"))
    if workers <= 1:
        rows = []
        for i, item in enumerate(golden, 1):
            out = system.handle(item["message"])
            rows.append({"golden_id": item["golden_id"], **out.to_dict()})
            if i % 25 == 0:
                print(f"  [{name}] {i}/{len(golden)}")
        return rows

    # Order is restored by index afterwards so results stay deterministic despite
    # out-of-order completion.
    results: list[dict | None] = [None] * len(golden)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(system.handle, item["message"]): i
            for i, item in enumerate(golden)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = {"golden_id": golden[i]["golden_id"], **fut.result().to_dict()}
            done += 1
            if done % 25 == 0:
                print(f"  [{name}] {done}/{len(golden)}")
    return [r for r in results if r is not None]


def _run_simple_cv(system: SimpleBaseline, golden: list[dict], seed: int) -> list[dict]:
    from sklearn.model_selection import StratifiedKFold

    messages = [g["message"] for g in golden]
    intents = [g["intent"] for g in golden]

    # Classes with fewer members than n_splits cannot be stratified; fold them into
    # a placeholder for splitting purposes only, never for training labels.
    counts = Counter(intents)
    strat = [y if counts[y] >= 5 else "__rare__" for y in intents]

    rows: list[dict | None] = [None] * len(golden)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(messages, strat), 1):
        system.fit(
            [messages[i] for i in train_idx],
            [intents[i] for i in train_idx],
            routes=[golden[i]["route"] for i in train_idx],
        )
        for i in test_idx:
            out = system.handle(messages[i])
            rows[i] = {"golden_id": golden[i]["golden_id"], **out.to_dict()}
        print(f"  [simple] fold {fold}/5 done (train={len(train_idx)}, test={len(test_idx)})")
    return [r for r in rows if r is not None]


def score_system(name, rows, golden, index, brand, run_judge: bool,
                 judge_stratum: str = "") -> dict:
    by_id = {g["golden_id"]: g for g in golden}
    y_intent_true = [by_id[r["golden_id"]]["intent"] for r in rows]
    y_intent_pred = [r["intent"] for r in rows]
    y_route_true = [by_id[r["golden_id"]]["route"] for r in rows]
    y_route_pred = [r["route"] for r in rows]

    correct = [1.0 if t == p else 0.0 for t, p in zip(y_intent_true, y_intent_pred)]
    mf1, per_class = macro_f1(y_intent_true, y_intent_pred)

    # Integrity gate. A row whose LLM call failed is not a prediction; scoring it as
    # one silently converts an outage into a finding. An earlier run of this harness
    # did exactly that: 127 quota failures degraded to ESCALATE and presented as an
    # 80.9% "over-cautious escalation" result.
    n_errors = sum(1 for r in rows if r.get("error"))
    error_rate = n_errors / len(rows) if rows else 0.0
    if n_errors:
        print(
            f"  !! {n_errors}/{len(rows)} rows ({error_rate:.1%}) failed with an error. "
            f"Metrics below are NOT valid."
        )
        for sample in [r for r in rows if r.get("error")][:2]:
            print(f"     e.g. {sample['error'][:130]}")

    result = {
        "system": name,
        "valid": n_errors == 0,
        "n_errors": n_errors,
        "error_rate": round(error_rate, 4),
        "intent": {
            "accuracy": round(float(np.mean(correct)), 3),
            "accuracy_ci": bootstrap_ci(correct),
            "macro_f1": round(mf1, 3),
            "per_class": per_class,
            "confusion": confusion(y_intent_true, y_intent_pred),
        },
        "routing": routing_metrics(y_route_true, y_route_pred),
    }

    # The representative slice is the only one from which a production estimate may
    # be quoted; the enriched slice is ~20x over-weighted on hard cases by design.
    result["by_stratum"] = {}
    for stratum in ("representative", "enriched"):
        idx = [i for i, r in enumerate(rows) if by_id[r["golden_id"]]["stratum"] == stratum]
        if not idx:
            continue
        s_correct = [correct[i] for i in idx]
        s_mf1, _ = macro_f1(
            [y_intent_true[i] for i in idx], [y_intent_pred[i] for i in idx]
        )
        result["by_stratum"][stratum] = {
            "n": len(idx),
            "intent_accuracy": round(float(np.mean(s_correct)), 3),
            "intent_accuracy_ci": bootstrap_ci(s_correct),
            "intent_macro_f1": round(s_mf1, 3),
            "routing": routing_metrics(
                [y_route_true[i] for i in idx], [y_route_pred[i] for i in idx]
            ),
        }

    if run_judge:
        # Judging is the most quota-hungry stage. Scoping it to one stratum keeps
        # the cross-system comparison inside a single model's daily budget, which
        # matters because comparability requires every system judged by the SAME
        # model — splitting the work across models would silently confound it.
        judged_rows = (
            [r for r in rows if by_id[r["golden_id"]]["stratum"] == judge_stratum]
            if judge_stratum
            else rows
        )
        print(f"  [{name}] judging {len(judged_rows)} replies"
              + (f" ({judge_stratum} stratum only)" if judge_stratum else ""))
        workers = int(os.environ.get("EVAL_WORKERS", "6"))
        rows = judged_rows
        scores: list = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    judging.judge_reply,
                    brand,
                    row["message"],
                    row["reply"],
                    index.search(row["message"], k=3),
                ): i
                for i, row in enumerate(rows)
            }
            done = 0
            for fut in as_completed(futures):
                scores[futures[fut]] = fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"    judged {done}/{len(rows)}")
        for row, sc in zip(rows, scores):
            row["judge"] = sc.to_dict()
        n_judge_errors = sum(1 for s in scores if s.error)
        if n_judge_errors:
            print(
                f"  !! {n_judge_errors}/{len(scores)} judge calls failed. "
                f"Reply-quality metrics below are NOT valid."
            )
        sendable = [1.0 if s.sendable else 0.0 for s in scores]
        result["reply_quality"] = {
            "valid": n_judge_errors == 0,
            "n_judge_errors": n_judge_errors,
            "stratum": judge_stratum or "pooled",
            "n_judged": len(rows),
            "sendable_rate": round(float(np.mean(sendable)), 3),
            "sendable_rate_ci": bootstrap_ci(sendable),
            "mean_overall": round(float(np.mean([s.mean_score() for s in scores])), 3),
            "dimensions": {
                dim: {
                    "mean": round(float(np.mean([getattr(s, dim) for s in scores])), 2),
                    "ci": bootstrap_ci([float(getattr(s, dim)) for s in scores]),
                }
                for dim in judging.DIMENSIONS
            },
            "worst_problems": Counter(
                s.worst_problem.lower()[:60] for s in scores if s.worst_problem
            ).most_common(8),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", required=True)
    ap.add_argument("--systems", default="trivial,simple,agent")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N golden items")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judging (fast, free)")
    ap.add_argument("--judge-stratum", default="",
                    help="judge only this stratum (e.g. representative), to stay inside a "
                         "single judge model's daily quota while keeping systems comparable")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    print(llm.describe())

    golden = read_jsonl(GOLDEN_DIR / "golden_set.jsonl")
    if args.limit:
        golden = golden[: args.limit]
    print(f"golden set: {len(golden)} examples")

    threads = read_jsonl(INTERIM / f"{args.brand}_threads.jsonl")
    systems, index = build_systems(args.brand, threads, golden, args.seed)

    ARTIFACTS.mkdir(exist_ok=True)
    results = []
    for name in args.systems.split(","):
        name = name.strip()
        if name not in systems:
            print(f"skipping unknown system {name!r}")
            continue
        print(f"\n=== {name} ===")
        rows = run_system(name, systems[name], golden, args.seed)
        result = score_system(name, rows, golden, index, args.brand, not args.no_judge,
                              judge_stratum=args.judge_stratum)
        results.append(result)
        write_jsonl(rows, ARTIFACTS / f"predictions_{name}.jsonl")

        print(
            f"  intent acc {fmt_ci(result['intent']['accuracy'], result['intent']['accuracy_ci'])}"
            f" | macro-F1 {result['intent']['macro_f1']:.3f}"
        )
        r = result["routing"]
        print(
            f"  routing: false-auto {r['false_auto_rate']} ({r['false_auto_count']}) "
            f"| needless-escalation {r['needless_escalation_rate']} ({r['needless_escalation_count']}) "
            f"| safe automation {fmt_ci(r['safe_automation_rate'], r['safe_automation_ci'])}"
        )
        if "reply_quality" in result:
            q = result["reply_quality"]
            print(
                f"  reply: sendable {fmt_ci(q['sendable_rate'], q['sendable_rate_ci'])} "
                f"| mean rubric {q['mean_overall']:.2f}/5"
            )

    out = ARTIFACTS / "results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()

"""Turn the raw twcs.csv dump into per-brand conversation threads.

The raw file is a flat table of tweets linked by in_response_to_tweet_id, so a
conversation is a connected chain through those links. Resolve every tweet to its
thread root, then keep the threads the brand took part in.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "twcs.csv"
INTERIM = REPO_ROOT / "data" / "interim"

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
]

HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_data.sh first (needs ~/.kaggle/kaggle.json)."
        )
    df = pd.read_csv(path, usecols=USECOLS, dtype={"author_id": "str", "text": "str"})
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    )
    df = df.dropna(subset=["tweet_id", "text"])
    df["tweet_id"] = df["tweet_id"].astype("int64")
    return df


def brand_volumes(df: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    """Rank brands by how many outbound support replies they sent."""
    outbound = df[~df["inbound"].astype(bool)]
    counts = outbound["author_id"].value_counts().head(top)
    rows = []
    for brand, n_replies in counts.items():
        replies = outbound[outbound["author_id"] == brand]["text"]
        rows.append(
            {
                "brand": brand,
                "n_replies": int(n_replies),
                "median_reply_chars": int(replies.str.len().median()),
                "pct_reply_asks_dm": round(
                    100
                    * replies.str.contains(
                        r"\bDM\b|direct message|dm us|send us a", case=False, regex=True
                    ).mean(),
                    1,
                ),
                "pct_reply_has_link": round(
                    100 * replies.str.contains(r"https?://", regex=True).mean(), 1
                ),
            }
        )
    return pd.DataFrame(rows)


TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def _parse_ts(raw: str) -> float:
    """Twitter's created_at -> epoch seconds. Returns inf for unparseable stamps
    so malformed rows sort last instead of silently leading a thread."""
    try:
        return datetime.strptime(str(raw), TS_FORMAT).timestamp()
    except (ValueError, TypeError):
        return float("inf")


def _resolve_roots(parent: dict[int, int]) -> dict[int, int]:
    """Map every tweet to its thread root, with path compression."""
    root: dict[int, int] = {}

    def find(node: int) -> int:
        path = []
        while node in parent and node not in root:
            path.append(node)
            node = parent[node]
        top = root.get(node, node)
        for item in path:
            root[item] = top
        return top

    for node in list(parent.keys()):
        find(node)
    return root


def build_threads(df: pd.DataFrame, brand: str) -> list[dict]:
    """Reconstruct every conversation thread that the given brand took part in."""
    parent = {
        int(t): int(p)
        for t, p in zip(df["tweet_id"], df["in_response_to_tweet_id"])
        if pd.notna(p)
    }
    root_of = _resolve_roots(parent)

    df = df.copy()
    df["root"] = [root_of.get(int(t), int(t)) for t in df["tweet_id"]]

    brand_roots = set(df.loc[df["author_id"] == brand, "root"])
    threads_df = df[df["root"].isin(brand_roots)]

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in threads_df.itertuples(index=False):
        grouped[row.root].append(
            {
                "tweet_id": int(row.tweet_id),
                "author_id": row.author_id,
                "inbound": bool(row.inbound),
                "created_at": row.created_at,
                "ts": _parse_ts(row.created_at),
                "text": row.text,
            }
        )

    threads = []
    for root, turns in grouped.items():
        # tweet_id is assigned in REVERSE chronological order in this dump, so
        # sorting by it silently inverts every conversation. Sort by timestamp.
        turns.sort(key=lambda t: (t["ts"], -t["tweet_id"]))
        # Drop threads where another brand also replied: cross-brand mentions
        # (e.g. a customer tagging two airlines) muddy the "how does THIS brand
        # resolve it" signal we later retrieve over.
        outbound_authors = {t["author_id"] for t in turns if not t["inbound"]}
        if outbound_authors - {brand}:
            continue
        customer_turns = [t for t in turns if t["inbound"]]
        brand_turns = [t for t in turns if not t["inbound"]]
        if not customer_turns or not brand_turns:
            continue
        threads.append(
            {
                "thread_id": int(root),
                "brand": brand,
                "n_turns": len(turns),
                "n_customer_turns": len(customer_turns),
                "n_brand_turns": len(brand_turns),
                "created_at": customer_turns[0]["created_at"],
                "first_customer_message": customer_turns[0]["text"],
                "turns": turns,
            }
        )
    threads.sort(key=lambda t: t["thread_id"])
    return threads


def clean_text(text: str, *, drop_handles: bool = True) -> str:
    """Normalise tweet text for modelling: strip @handles, URLs, collapse space."""
    out = URL_RE.sub(" <link> ", text)
    if drop_handles:
        out = HANDLE_RE.sub(" ", out)
    out = out.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    return WS_RE.sub(" ", out).strip()


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", help="brand author_id, e.g. AppleSupport")
    ap.add_argument("--survey", action="store_true", help="print brand volume table and exit")
    ap.add_argument("--raw", type=Path, default=RAW_CSV)
    args = ap.parse_args()

    df = load_raw(args.raw)
    print(f"loaded {len(df):,} tweets")

    if args.survey or not args.brand:
        table = brand_volumes(df)
        print(table.to_string(index=False))
        table.to_csv(REPO_ROOT / "artifacts" / "brand_survey.csv", index=False)
        return

    threads = build_threads(df, args.brand)
    out = INTERIM / f"{args.brand}_threads.jsonl"
    write_jsonl(threads, out)
    n_turns = sum(t["n_turns"] for t in threads)
    print(f"{args.brand}: {len(threads):,} threads, {n_turns:,} turns -> {out}")


if __name__ == "__main__":
    main()

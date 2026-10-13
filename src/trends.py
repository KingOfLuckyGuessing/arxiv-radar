"""Detect emerging topics: which research areas are heating up right now?

Method: for each topic, compare its share of papers in the recent window
against its baseline share, and rank by z-score of the difference.

Usage:
    python -m src.trends --input data/topics.parquet --recent-weeks 2 --baseline-weeks 8
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def detect_emerging(topics: pd.DataFrame, recent_weeks: int, baseline_weeks: int,
                    top_n: int = 15) -> pd.DataFrame:
    required = {"window_start", "topic", "topic_name", "share"}
    missing = required - set(topics.columns)
    if missing:
        raise SystemExit(
            f"topics.parquet is missing columns {sorted(missing)} — it was likely "
            f"written by a failed run. Delete data/topics.parquet and re-run "
            f"`python -m src.topics`."
        )
    topics = topics.copy()
    # format="ISO8601": written timestamps mix "…T03:24:04+00:00" and
    # "…T03:24:04.437500+00:00" (isoformat drops zero microseconds), which
    # breaks pandas' strict format inference without this.
    topics["window_start"] = pd.to_datetime(topics["window_start"], format="ISO8601")
    latest = topics["window_start"].max()

    recent_cutoff = latest - pd.Timedelta(weeks=recent_weeks)
    baseline_cutoff = latest - pd.Timedelta(weeks=recent_weeks + baseline_weeks)

    recent = topics[topics["window_start"] >= recent_cutoff]
    baseline = topics[(topics["window_start"] >= baseline_cutoff)
                      & (topics["window_start"] < recent_cutoff)]

    recent_share = recent.groupby("topic")["share"].mean().rename("recent_share")
    baseline_share = baseline.groupby("topic")["share"].mean().rename("baseline_share")

    merged = pd.concat([recent_share, baseline_share], axis=1).fillna(0.0)
    merged["delta"] = merged["recent_share"] - merged["baseline_share"]

    # z-score of the delta across topics: how unusual is this surge?
    std = merged["delta"].std()
    merged["z"] = merged["delta"] / std if std > 0 else 0.0

    names = topics.drop_duplicates("topic").set_index("topic")["topic_name"]
    merged["topic_name"] = merged.index.map(names)
    return merged.sort_values("z", ascending=False).head(top_n).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Find emerging arXiv topics.")
    parser.add_argument("--input", default="data/topics.parquet")
    parser.add_argument("--out", default="data/emerging.parquet")
    parser.add_argument("--recent-weeks", type=int, default=2)
    parser.add_argument("--baseline-weeks", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    topics = pd.read_parquet(args.input)
    emerging = detect_emerging(topics, args.recent_weeks, args.baseline_weeks, args.top_n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emerging.to_parquet(out, index=False)

    print(f"\nTop {args.top_n} emerging topics "
          f"(last {args.recent_weeks}w vs prior {args.baseline_weeks}w):\n")
    for _, row in emerging.iterrows():
        print(f"  z={row['z']:+.2f}  {row['topic_name']} "
              f"({row['baseline_share']:.1%} -> {row['recent_share']:.1%})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

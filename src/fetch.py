"""Fetch paper metadata from the arXiv API and store it locally.

Usage:
    python -m src.fetch --categories cs.LG cs.CL cs.AI --max-results 2000 --days-back 90
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_papers(categories: list[str], max_results: int, days_back: int) -> pd.DataFrame:
    """Pull recent papers for the given arXiv categories."""
    query = " OR ".join(f"cat:{c}" for c in categories)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    # Be polite to the arXiv API: page slowly, retry on hiccups.
    client = arxiv.Client(page_size=500, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    records = []
    for result in client.results(search):
        published = result.published.replace(tzinfo=timezone.utc)
        if published < cutoff:
            continue
        records.append(
            {
                "id": result.entry_id.split("/abs/")[-1],
                "title": " ".join(result.title.split()),
                "abstract": " ".join(result.summary.split()),
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "published": published.isoformat(),
                "url": result.entry_id,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch recent arXiv papers.")
    parser.add_argument("--categories", nargs="+", default=["cs.LG", "cs.CL", "cs.AI"],
                        help="arXiv categories, e.g. cs.LG cs.CL")
    parser.add_argument("--max-results", type=int, default=2000)
    parser.add_argument("--days-back", type=int, default=90,
                        help="Only keep papers published within this window")
    parser.add_argument("--out", default=str(DATA_DIR / "papers.parquet"))
    args = parser.parse_args()

    df = fetch_papers(args.categories, args.max_results, args.days_back)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} papers -> {out}")


if __name__ == "__main__":
    main()

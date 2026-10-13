"""Bulk-fetch arXiv papers via the OpenAlex API (the multi-year path).

Why this exists: the arXiv API returns HTTP 500 past start=10000, capping any
single query around ~10k results. OpenAlex mirrors arXiv metadata with native
date filters and cursor pagination, so multi-year backfills are practical:
free, no API key, ~200 records per request.

Usage:
    python -m src.fetch_openalex --from-date 2023-01-01 --to-date 2026-01-01
    python -m src.fetch_openalex --from-date 2024-01-01 --to-date 2025-01-01 \\
        --max-results 20000 --out data/papers_2024.parquet --mailto you@example.com

Output schema matches src/fetch.py (id, title, abstract, authors, categories,
published, url) plus a cited_by_count bonus column. Feed it straight into
`python -m src.topics --input <out>`.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.openalex.org/works"
ARXIV_SOURCE_ID = "S4306400194"   # "arXiv (Cornell University)", type=repository
AI_SUBFIELD_ID = "1702"          # OpenAlex subfield "Artificial Intelligence"
PER_PAGE = 200
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# New-style (2401.12345) and legacy (hep-th/9901001) arXiv identifiers.
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?|([a-z-]+\/\d{7})(v\d+)?")


def published_from(arxiv_id: str | None, fallback_date: str) -> str:
    """Derive the true submission date from the arXiv ID.

    OpenAlex's publication_date can reflect re-indexing/DOI events rather than
    first submission (e.g. arXiv:1506.02216, a 2015 paper, showed up dated
    2026-05-02 after a Zenodo re-deposit). The arXiv ID itself encodes the
    original submission year+month, which is what trend windows need.
    When the ID agrees with OpenAlex's date we keep OpenAlex's day precision;
    when they disagree we trust the ID (day=01). Unparseable IDs fall back
    to OpenAlex's date unchanged.
    """
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", (arxiv_id or "").lower())
    if not m:
        m = re.match(r"^[a-z-]+\/(\d{2})(\d{2})\d{3}$", (arxiv_id or "").lower())
    if m:
        yy = int(m.group(1))
        yyyy = 2000 + yy if yy < 91 else 1900 + yy  # arXiv began 1991
        id_ym = f"{yyyy:04d}-{m.group(2)}"
        if fallback_date[:7] == id_ym:
            return f"{fallback_date}T00:00:00+0000"
        return f"{id_ym}-01T00:00:00+0000"
    return f"{fallback_date}T00:00:00+0000"


def reconstruct_abstract(inv_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild plain text."""
    if not inv_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inv_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in range(len(positions)))


def arxiv_id_from(work: dict) -> str | None:
    """Extract the arXiv id from the DOI or any arxiv.org URL (abs or pdf)."""
    doi = work.get("doi") or ""
    m = ARXIV_ID_RE.search(doi)
    if m:
        return (m.group(1) or m.group(3)).lower()
    landing = (work.get("primary_location") or {}).get("landing_page_url") or ""
    m = ARXIV_ID_RE.search(landing)
    if m:
        return (m.group(1) or m.group(3)).lower()
    return None


def work_to_record(work: dict) -> dict | None:
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if not abstract.strip():
        return None
    pub_date = work.get("publication_date")
    if not pub_date:
        return None
    topic = work.get("primary_topic") or {}
    authorships = work.get("authorships") or []
    arxiv_id = arxiv_id_from(work)
    landing = (work.get("primary_location") or {}).get("landing_page_url") or ""
    return {
        "id": arxiv_id or work["id"].rsplit("/", 1)[-1],
        "title": " ".join((work.get("title") or "").split()),
        "abstract": abstract,
        "authors": [a["author"]["display_name"] for a in authorships
                    if a.get("author", {}).get("display_name")],
        "categories": [topic.get("display_name")] if topic.get("display_name") else [],
        # Date from the arXiv ID, not OpenAlex (see published_from).
        "published": published_from(arxiv_id, pub_date),
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (landing or work.get("id")),
        "cited_by_count": work.get("cited_by_count", 0),
    }


def fetch_page(session: requests.Session, params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        r = session.get(API, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or 500 <= r.status_code < 600:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"OpenAlex kept failing (HTTP {r.status_code}); "
                       f"partial results were saved — re-run to resume.")


def save_partial(records: list[dict], out: Path) -> int:
    """Append records to the parquet, de-duplicated by id. Returns total rows."""
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(records)
    if out.exists():
        old = pd.read_parquet(out)
        new = pd.concat([old, new]).drop_duplicates("id", keep="last")
    new.to_parquet(out, index=False)
    return len(new)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-fetch arXiv papers via OpenAlex (multi-year backfills).")
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--subfield", default=AI_SUBFIELD_ID,
                        help="OpenAlex subfield id (default: 1702 = Artificial Intelligence)")
    parser.add_argument("--max-results", type=int, default=100_000)
    parser.add_argument("--out", default=str(DATA_DIR / "papers_openalex.parquet"))
    parser.add_argument("--mailto", default=None,
                        help="Your email: puts you in OpenAlex's polite pool (faster, fewer limits)")
    args = parser.parse_args()

    out = Path(args.out)
    cursor_path = out.with_suffix(".cursor")
    filter_str = (f"primary_location.source.id:{ARXIV_SOURCE_ID},"
                  f"primary_topic.subfield.id:{args.subfield},"
                  f"from_publication_date:{args.from_date},"
                  f"to_publication_date:{args.to_date}")

    # Resume support: the cursor sidecar lets an interrupted run pick up where
    # it left off (partial parquet is de-duplicated by id on every flush).
    cursor = "*"
    if cursor_path.exists():
        try:
            saved = json.loads(cursor_path.read_text())
            if saved.get("filter") == filter_str and saved.get("cursor"):
                cursor = saved["cursor"]
                print(f"Resuming from saved cursor ({out.name} keeps earlier pages).")
            else:
                cursor_path.unlink()
        except (json.JSONDecodeError, OSError):
            cursor_path.unlink()

    select = ("id,doi,title,abstract_inverted_index,publication_date,"
              "primary_location,authorships,cited_by_count,primary_topic")
    session = requests.Session()
    session.headers["User-Agent"] = "arxiv-radar/1.0 (research project)"

    total_seen = 0
    skipped_no_abstract = 0
    buffer: list[dict] = []
    try:
        while True:
            params = {"filter": filter_str, "per-page": PER_PAGE,
                      "cursor": cursor, "select": select}
            if args.mailto:
                params["mailto"] = args.mailto
            page = fetch_page(session, params)
            for work in page.get("results", []):
                rec = work_to_record(work)
                if rec is None:
                    skipped_no_abstract += 1
                    continue
                buffer.append(rec)
                total_seen += 1
                if total_seen >= args.max_results:
                    break

            n_total = save_partial(buffer, out)
            buffer = []
            cursor = page["meta"].get("next_cursor")
            cursor_path.write_text(json.dumps({"filter": filter_str, "cursor": cursor}))
            print(f"  ...{n_total} saved (skipped {skipped_no_abstract} without abstracts)")

            if not cursor or total_seen >= args.max_results:
                break
            time.sleep(0.15)  # stay comfortably inside the polite rate limit
    finally:
        if buffer:  # never lose a partial page on interrupt/error
            n_total = save_partial(buffer, out)
            print(f"Saved partial results: {n_total} rows -> {out}")

    cursor_path.unlink(missing_ok=True)
    print(f"\nDone: {total_seen} papers -> {out}")
    if total_seen >= args.max_results:
        print("WARNING: hit --max-results; narrow the date range for full coverage.")
    print(f"Next: python -m src.topics --input {out} --window-weeks 12 --step-weeks 4")


if __name__ == "__main__":
    main()

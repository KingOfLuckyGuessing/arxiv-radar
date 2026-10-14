"""Bulk-fetch arXiv papers via the OpenAlex API (the multi-year path).

Why this exists: the arXiv API returns HTTP 500 past start=10000, capping any
single query around ~10k results. OpenAlex mirrors arXiv metadata with native
date filters and cursor pagination, so multi-year backfills are practical.
At this scale OpenAlex requires a free API key (get one in ~30s at
https://openalex.org/settings/api); pass it via --api-key or the
OPENALEX_API_KEY environment variable.

Usage:
    python -m src.fetch_openalex --from-date 2023-01-01 --to-date 2026-01-01 --api-key $OPENALEX_API_KEY
    python -m src.fetch_openalex --from-date 2024-01-01 --to-date 2025-01-01 \\
        --max-results 20000 --out data/papers_2024.parquet

Output schema matches src/fetch.py (id, title, abstract, authors, categories,
published, url) plus a cited_by_count bonus column. Feed it straight into
`python -m src.topics --input <out>`.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.openalex.org/works"
ARXIV_SOURCE_ID = "S4306400194"   # "arXiv (Cornell University)", type=repository
AI_SUBFIELD_ID = "1702"          # OpenAlex subfield "Artificial Intelligence"
PER_PAGE = 100  # OpenAlex's documented per_page maximum (lowered from 200)
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
        if r.status_code == 401:
            # A 401 never heals with retries: OpenAlex requires a free API
            # key for cursor paging at this scale (keyless budget exhausted).
            raise SystemExit(
                "OpenAlex returned HTTP 401: large backfills need a free API key.\n"
                "Get one in ~30 seconds at https://openalex.org/settings/api\n"
                "then re-run with --api-key YOUR_KEY (or set OPENALEX_API_KEY).\n"
                "Partial results were saved — the re-run will resume."
            )
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
    parser.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"),
                        help="OpenAlex API key (free at openalex.org/settings/api). "
                             "Required for large backfills; or set OPENALEX_API_KEY.")
    parser.add_argument("--mailto", default=None,
                        help="Legacy: OpenAlex has ignored mailto since Feb 2026; "
                             "use --api-key instead.")
    args = parser.parse_args()

    out = Path(args.out)
    cursor_path = out.with_suffix(".cursor")
    filter_str = (f"primary_location.source.id:{ARXIV_SOURCE_ID},"
                  f"primary_topic.subfield.id:{args.subfield},"
                  f"from_publication_date:{args.from_date},"
                  f"to_publication_date:{args.to_date}")

    # Resume support: the cursor sidecar lets an interrupted run pick up where
    # it left off (partial parquet is de-duplicated by id on every flush).
    # The sidecar is versioned by per_page: a cursor minted for a different
    # page size is not reused (OpenAlex cursors are opaque; resuming across a
    # page-size change could silently skip rows — a clean re-walk is safe
    # because dedupe keeps what's already saved).
    cursor = "*"
    if cursor_path.exists():
        try:
            saved = json.loads(cursor_path.read_text())
            if (saved.get("filter") == filter_str
                    and saved.get("cursor")
                    and saved.get("per_page") == PER_PAGE):
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
    if args.api_key:
        session.headers["Authorization"] = f"Bearer {args.api_key}"
    else:
        print("Note: no --api-key given. OpenAlex requires a free API key for "
              "large backfills (get one at openalex.org/settings/api) — "
              "expect HTTP 401 once the keyless budget is spent.")

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
            cursor_path.write_text(json.dumps({"filter": filter_str,
                                               "cursor": cursor,
                                               "per_page": PER_PAGE}))
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

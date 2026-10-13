# arXiv Research Radar

Which research topics are heating up? This project ingests recent arXiv papers,
models their topics with BERTopic, and flags emerging areas by comparing recent
topic velocity against a baseline — all behind a Streamlit dashboard.

## Pipeline

```
arXiv API  →  fetch.py   →  data/papers.parquet
                         →  topics.py  →  data/topics.parquet   (BERTopic, sliding windows)
                         →  trends.py  →  data/emerging.parquet (velocity z-scores)
                         →  app/dashboard.py                    (Streamlit)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch papers (default: cs.LG, cs.CL, cs.AI, last 90 days)
python -m src.fetch --max-results 2000 --days-back 90

# 2. Topic-model them over 4-week sliding windows
python -m src.topics

# 3. Detect emerging topics (last 2 weeks vs prior 8)
python -m src.trends

# 4. Launch the dashboard
streamlit run app/dashboard.py
```

## Multi-year backfills

The arXiv API returns HTTP 500 past `start=10000`, so any single query caps
around ~10k papers. For trends over years, use the OpenAlex fetcher instead —
free, no API key, native date filters, and cursor pagination built for bulk:

```bash
# ~3 years of AI papers on arXiv (takes a few minutes; resumable if interrupted)
python -m src.fetch_openalex --from-date 2023-01-01 --to-date 2026-01-01

# Then run the same pipeline against it:
python -m src.topics --input data/papers_openalex.parquet --window-weeks 12 --step-weeks 4
python -m src.trends --input data/topics.parquet --recent-weeks 8 --baseline-weeks 32
streamlit run app/dashboard.py
```

Notes on the OpenAlex path:

- It filters to arXiv (`primary_location.source.id:S4306400194`) and the
  OpenAlex *Artificial Intelligence* subfield (`--subfield 1702`, configurable) —
  the bulk-mode analog of arXiv categories. Papers keep the same parquet schema
  as `src/fetch.py`, plus a `cited_by_count` bonus column.
- Embeddings are cached in `data/embeddings.npy` (keyed to the dataset), so the
  one-time CPU cost isn't repeated when you re-tune topics. Delete the file to
  force a recompute.
- Add `--mailto you@example.com` to join OpenAlex's polite pool (faster, fewer
  rate limits).

## Which fetcher should I use?

| | `src/fetch.py` (arXiv API) | `src/fetch_openalex.py` (OpenAlex) |
|---|---|---|
| Best for | Recent pulls (days–months) | Multi-year backfills |
| Category filter | Exact arXiv categories | OpenAlex AI subfield (configurable) |
| Limit | ~10k results per query (API 500s past `start=10000`) | Cursor pagination, no practical limit |
| Freshness | Real-time | Updated continuously (small lag) |
| Extras | — | `cited_by_count` per paper |

## Project structure

- `src/fetch.py` — arXiv API ingestion (polite paging, date cutoff) → parquet
- `src/topics.py` — BERTopic (UMAP + HDBSCAN + bigram vectorizer) over sliding time windows
- `src/trends.py` — emerging-topic detection via share-velocity z-scores
- `app/dashboard.py` — Streamlit: emerging topics table, trend lines, paper browser
- `data/` — local parquet artifacts (git-ignored; regenerate with the pipeline)

## How the trend detection works

For each topic, we compute its share of papers in a recent window vs. a baseline
window, then z-score the difference across topics. A high z-score means the topic
is surging unusually fast relative to everything else — not just growing in
absolute terms.

## Roadmap

- [ ] Citation graph: link papers via references, rank topics by influence (Neo4j)
- [ ] LLM-generated topic labels instead of keyword lists
- [ ] Weekly email/Slack digest of top emerging topics
- [ ] Compare embedding models (MiniLM vs. bge) and their effect on topic quality
- [ ] Scheduled refresh (GitHub Actions cron) so the dashboard is always current

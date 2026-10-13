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

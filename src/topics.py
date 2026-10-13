"""Fit topic models over sliding time windows and track how topics evolve.

Usage:
    python -m src.topics --input data/papers.parquet --window-weeks 4 --out data/topics.parquet

TODO (your call):
  - Tune min_topic_size / UMAP n_neighbors for your corpus size.
  - Try a KeyBERT-inspired or LLM representation model for nicer topic labels.
  - Experiment with embedding models (all-MiniLM-L6-v2 is fast; bge/mxbai are stronger).
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

EMBEDDINGS_CACHE = Path(__file__).resolve().parent.parent / "data" / "embeddings.npy"


def _docs_hash(ids: list[str]) -> str:
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:16]


def get_embeddings(ids: list[str], docs: list[str],
                   cache_path: Path = EMBEDDINGS_CACHE) -> np.ndarray:
    """Embed once, reuse forever.

    Embeddings are the dominant cost of the pipeline (especially on CPU), but
    they don't change when you re-tune UMAP/HDBSCAN. This caches them keyed by
    a hash of the document ids, so repeat runs skip straight to modeling.
    """
    hash_path = cache_path.with_suffix(".sha")
    want = _docs_hash(ids)
    if cache_path.exists() and hash_path.exists() \
            and hash_path.read_text().strip() == want:
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print("Computing embeddings (one-time cost; cached for future runs)...")
    from sentence_transformers import SentenceTransformer  # lazy: keeps --help fast
    model = SentenceTransformer("all-MiniLM-L6-v2")  # uses CUDA automatically if present
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    hash_path.write_text(want)
    return embeddings


def build_model(n_docs: int) -> BERTopic:
    """BERTopic with the classic UMAP + HDBSCAN stack.

    min_cluster_size scales with corpus size so small fetches don't
    collapse everything into noise (-1).
    """
    min_cluster_size = min(30, max(5, n_docs // 100))
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    return BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic-model arXiv papers over time.")
    parser.add_argument("--input", default="data/papers.parquet")
    parser.add_argument("--out", default="data/topics.parquet")
    parser.add_argument("--window-weeks", type=int, default=4,
                        help="Width of each sliding time window")
    parser.add_argument("--step-weeks", type=int, default=1,
                        help="How far the window slides each step")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    # format="ISO8601": arXiv timestamps mix fractional/non-fractional seconds.
    df["published"] = pd.to_datetime(df["published"], format="ISO8601")
    df = df.sort_values("published").reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(
            "No papers in the input file. Run `python -m src.fetch` first "
            "and check that it saved more than 0 papers."
        )
    docs = (df["title"] + ". " + df["abstract"]).tolist()

    embeddings = get_embeddings(df["id"].tolist(), docs)
    model = build_model(len(docs))
    topics, _ = model.fit_transform(docs, embeddings=embeddings)

    df["topic"] = topics
    topic_info = model.get_topic_info()
    names = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_name"] = df["topic"].map(names)

    # Sliding windows: which topics appear in each window?
    window = pd.Timedelta(weeks=args.window_weeks)
    step = pd.Timedelta(weeks=args.step_weeks)
    start, end = df["published"].min(), df["published"].max()
    span = end - start
    if span <= pd.Timedelta(0):
        raise SystemExit(
            "All papers share the same timestamp — cannot build time windows. "
            "Fetch a wider date range with `python -m src.fetch --days-back N`."
        )
    if span < window:
        # Common when --max-results caps a hot category to a few days of papers:
        # shrink the windows to fit the data instead of producing zero windows.
        window = span / 4
        step = window / 4
        print(f"Note: papers span only {span} — shrinking to {window} windows "
              f"sliding every {step}.")
    rows = []
    t = start
    while t + window <= end:
        chunk = df[(df["published"] >= t) & (df["published"] < t + window)]
        counts = chunk[chunk["topic"] != -1]["topic"].value_counts()
        for topic_id, count in counts.items():
            # Fixed strftime format (not isoformat): isoformat() drops zero
            # microseconds, producing mixed formats that break strict parsers.
            rows.append({
                "window_start": t.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "window_end": (t + window).strftime("%Y-%m-%dT%H:%M:%S%z"),
                "topic": int(topic_id),
                "topic_name": names.get(topic_id, str(topic_id)),
                "count": int(count),
                "share": float(count / max(len(chunk), 1)),
            })
        t += step

    if not rows:
        raise SystemExit(
            "No topics found in any window. Likely causes:\n"
            "  - every document was labeled noise (-1): lower HDBSCAN "
            "min_cluster_size in src/topics.py\n"
            "  - windows contain too few papers: fetch a wider date range "
            "(`python -m src.fetch --days-back N`)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    n_topics = len([t for t in names if t != -1])
    print(f"Found {n_topics} topics across {len(rows)} window-topic rows -> {out}")
    print("\nTop topics overall:")
    print(topic_info[topic_info["Topic"] != -1][["Topic", "Name", "Count"]].head(10)
          .to_string(index=False))


if __name__ == "__main__":
    main()

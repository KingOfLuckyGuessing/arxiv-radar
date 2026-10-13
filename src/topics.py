"""Fit topic models over sliding time windows and track how topics evolve.

Usage:
    python -m src.topics --input data/papers.parquet --window-weeks 4 --out data/topics.parquet

TODO (your call):
  - Tune min_topic_size / UMAP n_neighbors for your corpus size.
  - Try a KeyBERT-inspired or LLM representation model for nicer topic labels.
  - Experiment with embedding models (all-MiniLM-L6-v2 is fast; bge/mxbai are stronger).
"""

import argparse
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


def build_model() -> BERTopic:
    """BERTopic with the classic UMAP + HDBSCAN stack."""
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean",
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
    df["published"] = pd.to_datetime(df["published"])
    df = df.sort_values("published").reset_index(drop=True)
    docs = (df["title"] + ". " + df["abstract"]).tolist()

    model = build_model()
    topics, _ = model.fit_transform(docs)

    df["topic"] = topics
    topic_info = model.get_topic_info()
    names = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_name"] = df["topic"].map(names)

    # Sliding windows: which topics appear in each window?
    window = pd.Timedelta(weeks=args.window_weeks)
    step = pd.Timedelta(weeks=args.step_weeks)
    start, end = df["published"].min(), df["published"].max()
    rows = []
    t = start
    while t + window <= end:
        chunk = df[(df["published"] >= t) & (df["published"] < t + window)]
        counts = chunk[chunk["topic"] != -1]["topic"].value_counts()
        for topic_id, count in counts.items():
            rows.append({
                "window_start": t.isoformat(),
                "window_end": (t + window).isoformat(),
                "topic": int(topic_id),
                "topic_name": names.get(topic_id, str(topic_id)),
                "count": int(count),
                "share": float(count / max(len(chunk), 1)),
            })
        t += step

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

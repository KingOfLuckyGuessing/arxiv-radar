"""Streamlit dashboard: topic trends, emerging topics, paper browser.

Usage:
    streamlit run app/dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="arXiv Research Radar", layout="wide")
st.title("arXiv Research Radar")
st.caption("Which research topics are heating up? Topic trends from recent arXiv papers.")


@st.cache_data
def load():
    papers = pd.read_parquet(DATA_DIR / "papers.parquet")
    topics = pd.read_parquet(DATA_DIR / "topics.parquet")
    emerging_path = DATA_DIR / "emerging.parquet"
    emerging = pd.read_parquet(emerging_path) if emerging_path.exists() else None
    return papers, topics, emerging


try:
    papers, topics, emerging = load()
except FileNotFoundError:
    st.error("No data yet. Run the pipeline first: see README quickstart.")
    st.stop()

topics["window_start"] = pd.to_datetime(topics["window_start"])

# --- Emerging topics ---
st.header("Emerging topics")
if emerging is not None and not emerging.empty:
    st.dataframe(
        emerging[["topic_name", "z", "baseline_share", "recent_share"]]
        .rename(columns={"topic_name": "Topic", "z": "Trend z-score",
                         "baseline_share": "Baseline share",
                         "recent_share": "Recent share"}),
        use_container_width=True,
    )
else:
    st.info("Run `python -m src.trends` to compute emerging topics.")

# --- Trend lines ---
st.header("Topic trajectories")
top_topics = (
    topics.groupby("topic")["count"].sum().sort_values(ascending=False).head(8).index
)
plot_df = topics[topics["topic"].isin(top_topics)]
fig = px.line(
    plot_df, x="window_start", y="share", color="topic_name",
    labels={"window_start": "Week", "share": "Share of papers", "topic_name": "Topic"},
)
st.plotly_chart(fig, use_container_width=True)

# --- Paper browser ---
st.header("Browse papers")
query = st.text_input("Search titles and abstracts")
view = papers
if query:
    mask = (papers["title"].str.contains(query, case=False, na=False)
            | papers["abstract"].str.contains(query, case=False, na=False))
    view = papers[mask]
st.write(f"{len(view)} papers")
for _, row in view.sort_values("published", ascending=False).head(50).iterrows():
    with st.expander(row["title"]):
        st.write(row["abstract"][:1200] + ("..." if len(row["abstract"]) > 1200 else ""))
        st.caption(f"{row['published'][:10]} · {', '.join(row['categories'][:4])}")
        st.link_button("arXiv", row["url"])

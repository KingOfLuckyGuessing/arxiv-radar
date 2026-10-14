"""Streamlit dashboard: interactive topic trends for arXiv papers.

Views:
  - Momentum: which topics are accelerating (baseline vs recent share)
  - Topic river: how topic shares evolve over time (stacked area + heatmap)
  - Papers: most-cited papers, citation scatter, searchable browser

Usage:
    streamlit run app/dashboard.py
"""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def short_name(name: str, n: int = 38) -> str:
    """Make BERTopic-style names ('12_transformer_attention_bert') readable."""
    base = re.sub(r"^-?\d+_", "", str(name)).replace("_", " ")
    return base if len(base) <= n else base[: n - 1] + "…"


@st.cache_data
def load(suffix: str = ""):
    """Load one dataset. `suffix` groups papers/topics/emerging files, e.g.
    papers_quant.parquet + topics_quant.parquet + emerging_quant.parquet."""
    papers = pd.read_parquet(DATA_DIR / f"papers{suffix}.parquet")
    topics = pd.read_parquet(DATA_DIR / f"topics{suffix}.parquet")
    emerging_path = DATA_DIR / f"emerging{suffix}.parquet"
    emerging = pd.read_parquet(emerging_path) if emerging_path.exists() else None
    return papers, topics, emerging


def list_datasets() -> dict:
    """{pretty label: file suffix} for every papers*.parquet in data/."""
    out = {}
    for p in sorted(DATA_DIR.glob("papers*.parquet")):
        suffix = p.stem[len("papers"):]  # "" or "_quant"
        label = suffix.lstrip("_").replace("_", " ").title() or "Main"
        out[label] = suffix
    return out


def normalize(df_papers: pd.DataFrame, df_topics: pd.DataFrame) -> tuple:
    """Parse timestamps, ensure citation column exists, add short topic names."""
    papers = df_papers.copy()
    topics = df_topics.copy()
    papers["published"] = pd.to_datetime(papers["published"], format="ISO8601")
    topics["window_start"] = pd.to_datetime(topics["window_start"], format="ISO8601")
    if "cited_by_count" not in papers.columns:
        papers["cited_by_count"] = 0
    papers["cited_by_count"] = (
        pd.to_numeric(papers["cited_by_count"], errors="coerce").fillna(0).astype(int)
    )
    topics["short_name"] = topics["topic_name"].map(short_name)
    return papers, topics


def top_topics(topics: pd.DataFrame, n: int) -> list:
    """Topic ids of the n highest-volume topics, with stable short names."""
    totals = topics.groupby("topic")["count"].sum().sort_values(ascending=False)
    return totals.head(n).index.tolist()


def momentum_fig(emerging: pd.DataFrame, topics: pd.DataFrame) -> go.Figure:
    """Bubble chart: baseline share vs recent share, sized by volume, colored by z."""
    totals = (
        topics.groupby("topic")
        .agg(total=("count", "sum"), topic_name=("topic_name", "first"))
        .reset_index()
    )
    df = emerging.merge(totals, on=["topic", "topic_name"], how="left")
    df["total"] = df["total"].fillna(0).astype(int)

    fig = px.scatter(
        df,
        x="baseline_share",
        y="recent_share",
        size="total",
        color="z",
        hover_name="topic_name",
        hover_data={"baseline_share": ":.2%", "recent_share": ":.2%",
                    "z": ":.2f", "total": True},
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        size_max=60,
        labels={"baseline_share": "Baseline share", "recent_share": "Recent share",
                "z": "Trend z-score", "total": "Papers"},
        title="Topic momentum: recent share vs baseline share",
    )
    lim = float(max(df["baseline_share"].max(), df["recent_share"].max(), 1e-6)) * 1.05
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                  line=dict(dash="dash", color="gray"))
    fig.add_annotation(x=lim * 0.97, y=lim * 0.9, text="no change",
                       showarrow=False, font=dict(color="gray"))
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(height=560)
    return fig


def river_fig(topics: pd.DataFrame, topic_ids: list,
              value_col: str = "share") -> go.Figure:
    """Stacked area chart of topic share (or raw paper count) over time."""
    df = topics[topics["topic"].isin(topic_ids)].sort_values("window_start")
    is_share = value_col == "share"
    fig = px.area(
        df, x="window_start", y=value_col, color="short_name",
        labels={"window_start": "Week",
                "share": "Share of papers", "count": "Papers",
                "short_name": "Topic"},
        title=f"Topic river: {'share of papers' if is_share else 'paper count'} "
              f"over time",
    )
    if is_share:
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=480, legend=dict(orientation="h", y=-0.25))
    return fig


def heatmap_fig(topics: pd.DataFrame, topic_ids: list) -> go.Figure:
    """Topic x time heatmap: exactly when did each topic ignite?"""
    df = topics[topics["topic"].isin(topic_ids)]
    pivot = df.pivot_table(index="short_name", columns="window_start",
                           values="share", aggfunc="mean").fillna(0)
    # Order rows by total volume, columns chronologically.
    order = (df.groupby("short_name")["count"].sum()
               .reindex(pivot.index).sort_values(ascending=False).index)
    pivot = pivot.reindex(order).sort_index(axis=1)
    fig = px.imshow(
        pivot.values,
        x=[d.strftime("%Y-%m-%d") for d in pivot.columns],
        y=pivot.index,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(x="Week", y="Topic", color="Share"),
        title="Topic heatmap: when did each topic ignite?",
    )
    fig.update_layout(height=max(320, 28 * len(pivot)))
    return fig


def cited_bar_fig(papers: pd.DataFrame, n: int = 15) -> go.Figure:
    """Horizontal bars: most-cited papers."""
    df = papers.nlargest(n, "cited_by_count").copy()
    df["label"] = df["title"].str.slice(0, 70)
    fig = px.bar(
        df.sort_values("cited_by_count"),
        x="cited_by_count", y="label", orientation="h",
        hover_name="title",
        hover_data={"published": True, "cited_by_count": True, "label": False},
        labels={"cited_by_count": "Citations", "label": ""},
        title=f"Most-cited papers (top {n})",
    )
    fig.update_layout(height=max(360, 32 * n), yaxis_tickfont=dict(size=11))
    return fig


def citation_scatter_fig(papers: pd.DataFrame) -> go.Figure:
    """Every paper: publication date vs citations (log scale)."""
    df = papers.copy()
    if len(df) > 15000:
        df = df.sample(15000, random_state=7)
    df["authors_str"] = df["authors"].map(
        lambda a: ", ".join(a[:3]) if isinstance(a, list) else "")
    fig = px.scatter(
        df, x="published", y="cited_by_count",
        hover_name="title",
        hover_data={"authors_str": True, "published": True,
                    "cited_by_count": True},
        labels={"published": "Published", "cited_by_count": "Citations",
                "authors_str": "Authors"},
        title="Citation landscape: papers over time",
        opacity=0.55,
    )
    fig.update_yaxes(type="log", title="Citations (log scale)")
    fig.update_layout(height=480)
    return fig


def paper_browser(papers: pd.DataFrame):
    """Searchable paper list with citation counts."""
    st.subheader("Browse papers")
    query = st.text_input("Search titles and abstracts")
    sort = st.selectbox("Sort by", ["Newest first", "Most cited"])
    view = papers
    if query:
        mask = (papers["title"].str.contains(query, case=False, na=False)
                | papers["abstract"].str.contains(query, case=False, na=False))
        view = papers[mask]
    if sort == "Most cited":
        view = view.sort_values("cited_by_count", ascending=False)
    else:
        view = view.sort_values("published", ascending=False)
    st.write(f"{len(view)} papers")
    for _, row in view.head(50).iterrows():
        with st.expander(f"{row['title']}  ·  {int(row['cited_by_count'])} citations"):
            st.write(row["abstract"][:1200]
                     + ("..." if len(row["abstract"]) > 1200 else ""))
            cats = row["categories"]
            cats = ", ".join(cats[:4]) if isinstance(cats, list) else ""
            st.caption(f"{row['published'].date()} · {cats}")
            st.link_button("arXiv", row["url"])


def main():
    st.set_page_config(page_title="arXiv Research Radar", layout="wide")
    st.title("arXiv Research Radar")
    st.caption("Which research topics are heating up? "
               "Interactive trends from recent arXiv papers.")

    datasets = list_datasets()
    if not datasets:
        st.error("No data yet. Run the pipeline first: see README quickstart.")
        st.stop()
    if len(datasets) > 1:
        choice = st.sidebar.selectbox("Dataset", list(datasets))
    else:
        choice = list(datasets)[0]
    suffix = datasets[choice]

    try:
        papers, topics, emerging = load(suffix)
    except FileNotFoundError as e:
        st.error(f"Dataset '{choice}' is missing its topics file: {e}. "
                 f"Run `python -m src.topics --input data/papers{suffix}.parquet "
                 f"--out data/topics{suffix}.parquet`.")
        st.stop()
    papers, topics = normalize(papers, topics)

    st.sidebar.header("Display")
    n_topics = st.sidebar.slider("Topics to show", 5, 25, 10)
    topic_ids = top_topics(topics, n_topics)

    tab_momentum, tab_river, tab_papers = st.tabs(
        ["🔥 Momentum", "🌊 Topic river", "📄 Papers"])

    with tab_momentum:
        if emerging is not None and not emerging.empty:
            st.plotly_chart(momentum_fig(emerging, topics), use_container_width=True)
            st.caption("Above the dashed line = gaining share. "
                       "Bubble size = total papers, color = trend z-score.")
            st.subheader("Top movers")
            show = emerging.copy()
            show["topic_name"] = show["topic_name"].map(short_name)
            st.dataframe(
                show[["topic_name", "z", "baseline_share", "recent_share"]]
                .rename(columns={"topic_name": "Topic", "z": "Trend z-score",
                                 "baseline_share": "Baseline",
                                 "recent_share": "Recent"}),
                use_container_width=True,
            )
        else:
            st.info("Run `python -m src.trends` to compute emerging topics.")

    with tab_river:
        metric = st.radio("Show", ["Share of papers", "Paper count"],
                          horizontal=True, key="river_metric")
        value_col = "share" if metric == "Share of papers" else "count"
        st.plotly_chart(river_fig(topics, topic_ids, value_col),
                        use_container_width=True)
        st.plotly_chart(heatmap_fig(topics, topic_ids), use_container_width=True)

    with tab_papers:
        if papers["cited_by_count"].max() > 0:
            st.plotly_chart(cited_bar_fig(papers), use_container_width=True)
            st.plotly_chart(citation_scatter_fig(papers), use_container_width=True)
        else:
            st.info("Citation data comes from the OpenAlex fetcher "
                    "(`python -m src.fetch_openalex`). Papers fetched via the "
                    "arXiv API have no citation counts.")
        paper_browser(papers)


if __name__ == "__main__":
    main()

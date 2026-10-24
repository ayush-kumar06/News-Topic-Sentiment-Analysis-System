import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="News Intelligence Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = {
    "Positive": "#2E9E6B",
    "Neutral": "#8A94A6",
    "Negative": "#D9534F",
}
SENTIMENT_ICONS = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}
ACCENT = "#3B6FD4"

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}

    .hero {{
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #14213D 0%, #1F3A6E 60%, {ACCENT} 100%);
        color: #fff;
        margin-bottom: 1.4rem;
    }}
    .hero h1 {{ margin: 0; font-size: 1.9rem; font-weight: 700; color: #fff; }}
    .hero p  {{ margin: .35rem 0 0; opacity: .8; font-size: 1rem; }}

    .kpi {{
        border: 1px solid rgba(128,128,128,.25);
        border-left: 5px solid var(--accent);
        border-radius: 12px;
        padding: .9rem 1.1rem;
        background: rgba(128,128,128,.06);
    }}
    .kpi .label {{ font-size: .85rem; opacity: .7; }}
    .kpi .value {{ font-size: 1.9rem; font-weight: 700; line-height: 1.2; }}
    .kpi .sub   {{ font-size: .8rem; opacity: .6; }}

    .topic-card {{
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: .9rem 1.1rem;
        height: 100%;
        background: rgba(128,128,128,.05);
    }}
    .topic-card h4 {{ margin: 0 0 .5rem; font-size: 1rem; }}
    .chip {{
        display: inline-block;
        padding: .15rem .6rem;
        margin: 0 .3rem .35rem 0;
        border-radius: 999px;
        font-size: .78rem;
        background: rgba(59,111,212,.15);
        border: 1px solid rgba(59,111,212,.35);
    }}

    button[data-baseweb="tab"] {{ font-size: 1rem; font-weight: 600; }}
    section[data-testid="stSidebar"] {{ border-right: 1px solid rgba(128,128,128,.2); }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📰 News Topic & Sentiment Analysis</h1>
        <p>AI-powered news clustering and sentiment insights</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
file_path = Path("news_analysis.csv")

if not file_path.exists():
    st.warning("Analysis file not found. Run: `python train_model.py`")
    st.stop()


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def topic_keywords(texts: pd.Series, topics: pd.Series, top_n: int = 6) -> dict:
    """Most frequent meaningful words for each topic."""
    result = {}
    for topic in sorted(topics.unique()):
        docs = texts[topics == topic].fillna("").astype(str)
        if docs.str.strip().eq("").all():
            result[topic] = []
            continue
        vec = CountVectorizer(stop_words="english", max_features=200)
        counts = vec.fit_transform(docs).sum(axis=0).A1
        words = vec.get_feature_names_out()
        top = counts.argsort()[-top_n:][::-1]
        result[topic] = [words[i] for i in top]
    return result


df = load_data(str(file_path))

required = {"headline", "topic_id", "sentiment"}
if not required.issubset(df.columns):
    st.error(f"news_analysis.csv must contain: {', '.join(sorted(required))}")
    st.stop()

df["topic_label"] = "Topic " + (df["topic_id"].astype(int) + 1).astype(str)

# ------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

available_sentiments = [s for s in SENTIMENT_ORDER if s in df["sentiment"].unique()]
selected_sentiments = st.sidebar.multiselect(
    "Sentiment", available_sentiments, default=available_sentiments
)

available_topics = sorted(df["topic_id"].dropna().unique().tolist())
selected_topics = st.sidebar.multiselect(
    "Topic",
    available_topics,
    default=available_topics,
    format_func=lambda t: f"Topic {int(t) + 1}",
)

search_text = st.sidebar.text_input("Search headlines", placeholder="e.g. economy")

filtered_df = df[
    df["sentiment"].isin(selected_sentiments) & df["topic_id"].isin(selected_topics)
]
if search_text:
    filtered_df = filtered_df[
        filtered_df["headline"].str.contains(search_text, case=False, na=False)
    ]

st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** articles")

# ------------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------------
total = len(filtered_df)
counts = filtered_df["sentiment"].value_counts()


def kpi(col, label, value, color, sub=""):
    col.markdown(
        f"""
        <div class="kpi" style="--accent:{color}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pct(n):
    return f"{n / total * 100:.0f}% of results" if total else "—"


c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Total news", total, ACCENT, f"{filtered_df['topic_id'].nunique()} topics")
kpi(c2, "Positive", int(counts.get("Positive", 0)), SENTIMENT_COLORS["Positive"], pct(counts.get("Positive", 0)))
kpi(c3, "Negative", int(counts.get("Negative", 0)), SENTIMENT_COLORS["Negative"], pct(counts.get("Negative", 0)))
kpi(c4, "Neutral", int(counts.get("Neutral", 0)), SENTIMENT_COLORS["Neutral"], pct(counts.get("Neutral", 0)))

st.write("")

if filtered_df.empty:
    st.info("No articles match these filters. Try widening your selection or clearing the search.")
    st.stop()

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_overview, tab_topics, tab_articles = st.tabs(["📊 Overview", "🧩 Topics", "📄 Articles"])

color_scale = alt.Scale(
    domain=list(SENTIMENT_COLORS.keys()), range=list(SENTIMENT_COLORS.values())
)

# ---- Overview ----------------------------------------------------
with tab_overview:
    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment distribution")
        sent_df = counts.rename_axis("sentiment").reset_index(name="count")
        donut = (
            alt.Chart(sent_df)
            .mark_arc(innerRadius=70, outerRadius=120)
            .encode(
                theta="count:Q",
                color=alt.Color("sentiment:N", scale=color_scale, legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["sentiment", "count"],
            )
            .properties(height=320)
        )
        st.altair_chart(donut, use_container_width=True)

    with right:
        st.subheader("Sentiment by topic")
        mix = (
            filtered_df.groupby(["topic_label", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        stacked = (
            alt.Chart(mix)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("topic_label:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count:Q", title="Articles"),
                color=alt.Color("sentiment:N", scale=color_scale, legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["topic_label", "sentiment", "count"],
            )
            .properties(height=320)
        )
        st.altair_chart(stacked, use_container_width=True)

# ---- Topics ------------------------------------------------------
with tab_topics:
    st.subheader("Discovered topics")
    st.caption("Topic IDs are automatically discovered clusters, not predefined categories.")

    keywords = {}
    if "clean_text" in df.columns:
        keywords = topic_keywords(df["clean_text"], df["topic_id"])

    topic_ids = sorted(filtered_df["topic_id"].unique())
    cols = st.columns(min(len(topic_ids), 4) or 1)

    for i, t in enumerate(topic_ids):
        subset = filtered_df[filtered_df["topic_id"] == t]
        top_sent = subset["sentiment"].mode().iloc[0]
        chips = "".join(f'<span class="chip">{w}</span>' for w in keywords.get(t, []))
        cols[i % len(cols)].markdown(
            f"""
            <div class="topic-card">
                <h4>Topic {int(t) + 1} · {len(subset)} articles</h4>
                <div>{chips or '<span style="opacity:.6">No keywords available</span>'}</div>
                <div style="margin-top:.4rem;font-size:.85rem;opacity:.75">
                    Mostly {SENTIMENT_ICONS[top_sent]} {top_sent.lower()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Articles per topic")
    topic_counts = (
        filtered_df["topic_label"].value_counts().rename_axis("topic").reset_index(name="count")
    )
    bar = (
        alt.Chart(topic_counts)
        .mark_bar(color=ACCENT, cornerRadiusEnd=4)
        .encode(
            y=alt.Y("topic:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Articles"),
            tooltip=["topic", "count"],
        )
        .properties(height=max(120, 45 * len(topic_counts)))
    )
    st.altair_chart(bar, use_container_width=True)

# ---- Articles ----------------------------------------------------
with tab_articles:
    st.subheader("Analyzed news articles")

    table = filtered_df[["headline", "topic_label", "sentiment"]].copy()
    table["sentiment"] = table["sentiment"].map(lambda s: f"{SENTIMENT_ICONS.get(s, '')} {s}")
    table = table.rename(
        columns={"headline": "Headline", "topic_label": "Topic", "sentiment": "Sentiment"}
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Headline": st.column_config.TextColumn(width="large"),
            "Topic": st.column_config.TextColumn(width="small"),
            "Sentiment": st.column_config.TextColumn(width="small"),
        },
    )

    st.download_button(
        "⬇️ Download filtered results (CSV)",
        data=filtered_df.drop(columns=["topic_label"]).to_csv(index=False).encode("utf-8"),
        file_name="filtered_news_analysis.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Built with Streamlit · TF-IDF + K-Means for topics · VADER for sentiment")
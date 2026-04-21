"""
Sentiment + Topic Insight Platform
Live demo of the Sentiment140 sentiment classifier enhanced with topic discovery.

Run locally:    streamlit run app.py
Deploy free:    https://share.streamlit.io
"""

import io
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Page config ----------
st.set_page_config(
    page_title="Sentiment x Topic Insight",
    page_icon="💬",
    layout="wide",
)

# ---------- Constants ----------
ARTIFACTS_DIR = Path("artifacts")
REQUIRED_FILES = [
    "tfidf_vectorizer.joblib",
    "sentiment_model.joblib",
    "topic_vectorizer.joblib",
    "topic_model.joblib",
    "metadata.json",
]


# ---------- Cleaning function (matches Review 1) ----------
def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = text.replace("#", "").replace("'", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- Artifact loading ----------
@st.cache_resource
def load_artifacts():
    missing = [f for f in REQUIRED_FILES if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        return None, missing

    tfidf = joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(ARTIFACTS_DIR / "sentiment_model.joblib")
    topic_vec = joblib.load(ARTIFACTS_DIR / "topic_vectorizer.joblib")
    topic_model = joblib.load(ARTIFACTS_DIR / "topic_model.joblib")
    with open(ARTIFACTS_DIR / "metadata.json") as f:
        meta = json.load(f)

    return {
        "tfidf": tfidf,
        "model": model,
        "topic_vec": topic_vec,
        "topic_model": topic_model,
        "meta": meta,
    }, []


# ---------- Prediction helpers ----------
def predict_sentiment(texts, art):
    cleaned = [clean_tweet(t) for t in texts]
    X = art["tfidf"].transform(cleaned)
    if hasattr(art["model"], "predict_proba"):
        proba = art["model"].predict_proba(X)[:, 1]
    else:
        # fallback for raw LinearSVC — use decision function squashed
        decision = art["model"].decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-decision))
    pred = (proba >= 0.5).astype(int)
    return pred, proba, cleaned


def predict_topics(cleaned_texts, art):
    Xt = art["topic_vec"].transform(cleaned_texts)
    topic_dist = art["topic_model"].transform(Xt)
    # row-normalize
    s = topic_dist.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    topic_dist = topic_dist / s
    dominant = topic_dist.argmax(axis=1)
    return topic_dist, dominant


def topic_label_for(topic_id, art):
    return art["meta"]["topic_labels"].get(str(topic_id), f"Topic {topic_id}")


# ---------- UI: sidebar ----------
def render_sidebar(art):
    st.sidebar.markdown("### About this app")
    st.sidebar.markdown(
        "Twitter sentiment classification combined with topic discovery. "
        "Built on the Sentiment140 dataset (1.6M labeled tweets)."
    )

    if art is not None:
        meta = art["meta"]
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model")
        st.sidebar.markdown(f"**Sentiment:** {meta['baseline_model_name']}")
        st.sidebar.markdown(f"**Topics:** {meta['topic_model_name']} ({meta['n_topics']} topics)")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Test set performance")
        m = meta["metrics"]
        st.sidebar.metric("Baseline F1", f"{m['baseline_f1']:.3f}")
        st.sidebar.metric("Enhanced F1", f"{m['enhanced_f1']:.3f}",
                          delta=f"{(m['enhanced_f1'] - m['baseline_f1']):+.3f}")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Topics discovered")
        for tid, label in meta["topic_labels"].items():
            with st.sidebar.expander(f"Topic {tid}: {label[:30]}"):
                words = meta["topic_top_words"].get(str(tid), [])
                st.write(", ".join(words))


# ---------- UI: single tweet mode ----------
def render_single_mode(art):
    st.subheader("Analyze a single message")
    st.caption("Paste any tweet, customer review, or feedback comment.")

    default_examples = [
        "the new update is amazing! finally fixed the lag issue",
        "waited 45 min on hold and they hung up. worst customer service ever",
        "delivery was on time but the box was damaged",
    ]
    example_choice = st.selectbox(
        "Or pick an example",
        ["(write my own)"] + default_examples,
    )

    if example_choice != "(write my own)":
        default_text = example_choice
    else:
        default_text = ""

    text = st.text_area("Message", value=default_text, height=120, max_chars=500)

    if st.button("Analyze", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Type or pick a message first.")
            return

        pred, proba, cleaned = predict_sentiment([text], art)
        topic_dist, dominant = predict_topics(cleaned, art)
        topic_id = int(dominant[0])
        topic_score = float(topic_dist[0, topic_id])

        sentiment_label = "Positive 😊" if pred[0] == 1 else "Negative 😟"
        confidence = proba[0] if pred[0] == 1 else 1 - proba[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", sentiment_label)
        c2.metric("Confidence", f"{confidence:.1%}")
        c3.metric("Dominant topic", topic_label_for(topic_id, art),
                  delta=f"{topic_score:.0%} weight")

        # Topic distribution bar chart
        st.markdown("#### Full topic distribution")
        topic_df = pd.DataFrame({
            "Topic": [topic_label_for(i, art) for i in range(len(topic_dist[0]))],
            "Weight": topic_dist[0],
        }).sort_values("Weight", ascending=False)

        chart = (
            alt.Chart(topic_df)
            .mark_bar()
            .encode(
                x=alt.X("Weight:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Topic:N", sort="-x"),
                color=alt.Color("Weight:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["Topic", alt.Tooltip("Weight:Q", format=".1%")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Cleaned text used by the model"):
            st.code(cleaned[0] or "(empty after cleaning)")


# ---------- UI: bulk mode ----------
def render_bulk_mode(art):
    st.subheader("Bulk analyze a dataset")
    st.caption(
        "Upload a CSV with a column of text. The app will tag every row with sentiment + topic "
        "and produce the business dashboard."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    sample_button = st.button("Or load a built-in sample of 50 fake customer reviews")

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return
    elif sample_button:
        df = _build_sample_dataset()

    if df is None:
        st.info("Upload a file or click the sample button to begin.")
        return

    st.write(f"**Rows loaded:** {len(df):,}")
    st.dataframe(df.head(), use_container_width=True)

    text_col = st.selectbox("Which column contains the text?", df.columns.tolist())
    if not text_col:
        return

    if st.button("Run analysis", type="primary"):
        with st.spinner("Scoring rows..."):
            texts = df[text_col].fillna("").astype(str).tolist()
            pred, proba, cleaned = predict_sentiment(texts, art)
            topic_dist, dominant = predict_topics(cleaned, art)

            df_out = df.copy()
            df_out["sentiment"] = ["positive" if p == 1 else "negative" for p in pred]
            df_out["confidence"] = np.where(pred == 1, proba, 1 - proba).round(3)
            df_out["topic_id"] = dominant
            df_out["topic_label"] = [topic_label_for(int(t), art) for t in dominant]

        _render_bulk_dashboard(df_out, art)


def _render_bulk_dashboard(df_out, art):
    st.markdown("---")
    st.subheader("Dashboard")

    # KPI row
    n = len(df_out)
    n_neg = int((df_out["sentiment"] == "negative").sum())
    pct_neg = n_neg / n if n else 0
    n_topics_seen = df_out["topic_id"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Messages analyzed", f"{n:,}")
    k2.metric("Negative", f"{n_neg:,}", delta=f"{pct_neg:.1%}")
    k3.metric("Positive", f"{n - n_neg:,}", delta=f"{1 - pct_neg:.1%}")
    k4.metric("Distinct topics", n_topics_seen)

    # Topic-level summary
    summary = (
        df_out
        .groupby(["topic_id", "topic_label"])
        .agg(
            volume=("sentiment", "size"),
            negative_rate=("sentiment", lambda s: (s == "negative").mean()),
        )
        .reset_index()
        .sort_values("negative_rate", ascending=False)
    )

    # Bubble chart — the headline visual
    st.markdown("#### Where customers are unhappy")
    st.caption("Each bubble is a topic. Higher = more negative. Larger = more volume. Hot spots are top-right.")
    bubble = (
        alt.Chart(summary)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X("volume:Q", title="Volume of messages"),
            y=alt.Y("negative_rate:Q", title="Negative rate", scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("volume:Q", scale=alt.Scale(range=[100, 2000]), legend=None),
            color=alt.Color("negative_rate:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
            tooltip=[
                alt.Tooltip("topic_label:N", title="Topic"),
                alt.Tooltip("volume:Q", title="Volume"),
                alt.Tooltip("negative_rate:Q", title="Negative %", format=".1%"),
            ],
        )
        .properties(height=400)
    )
    text_overlay = (
        alt.Chart(summary)
        .mark_text(fontSize=10, fontWeight="bold")
        .encode(
            x="volume:Q",
            y="negative_rate:Q",
            text=alt.Text("topic_label:N"),
        )
        .transform_calculate(short_label="slice(datum.topic_label, 0, 25)")
    )
    st.altair_chart(bubble + text_overlay, use_container_width=True)

    # Table
    st.markdown("#### Topic-level breakdown")
    st.dataframe(
        summary
        .assign(negative_rate=lambda d: (d["negative_rate"] * 100).round(1).astype(str) + "%")
        .rename(columns={"topic_label": "Topic", "volume": "Volume", "negative_rate": "Negative %"})
        [["Topic", "Volume", "Negative %"]],
        use_container_width=True,
        hide_index=True,
    )

    # Drill-down
    st.markdown("#### Drill into a topic")
    chosen = st.selectbox(
        "Pick a topic to see example messages",
        options=summary["topic_label"].tolist(),
    )
    drill = df_out[df_out["topic_label"] == chosen].copy()
    neg_first = drill.sort_values("sentiment").head(20)
    st.dataframe(
        neg_first[["sentiment", "confidence", "topic_label"] + [c for c in df_out.columns if c not in ("sentiment", "confidence", "topic_id", "topic_label")]],
        use_container_width=True,
        hide_index=True,
    )

    # Download
    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    st.download_button(
        "Download tagged CSV",
        data=buf.getvalue(),
        file_name="tagged_messages.csv",
        mime="text/csv",
    )


def _build_sample_dataset():
    """A tiny fake customer-feedback dataset to demo the bulk mode."""
    samples = [
        "The food was cold and the waiter was rude",
        "Loved the staff! Super friendly and helpful",
        "App keeps crashing when I open the chat tab",
        "Delivery arrived 3 days late and box was damaged",
        "Best purchase I've made this year, totally worth it",
        "Customer service kept transferring me, took 2 hours",
        "Battery life on this phone is incredible",
        "The hotel room was dirty and smelled weird",
        "Quick refund, no questions asked, very professional",
        "WiFi at the hotel is unusable, can't even check email",
        "Excellent value for the price, highly recommend",
        "Returned the product, still waiting on refund 2 weeks later",
        "Movie was a masterpiece, perfect ending",
        "Flight got cancelled and they offered no compensation",
        "Pizza was perfect, exactly as ordered",
        "Update broke everything, please rollback",
        "Yoga class was relaxing and instructor was great",
        "Why does the website log me out every 5 minutes",
        "Coffee here is the best in town",
        "Charged me twice and refuses to refund",
        "Concert tonight was unforgettable",
        "Phone overheats during games, very disappointing",
        "Quick checkout, smooth experience overall",
        "The doctor was kind but the wait was 3 hours",
        "Subscription renewed without warning, sneaky",
        "Gym is clean and equipment is well maintained",
        "Driver was reckless and music was too loud",
        "Loved the dessert, will come back",
        "Package never arrived, tracking shows nothing",
        "Soundtrack is gorgeous, listening on repeat",
        "App is slow on older phones, please optimize",
        "Spa day was rejuvenating",
        "Bug in the latest version makes the app unusable",
        "Friendly support team helped me out quickly",
        "Server lag is unbearable, fix this please",
        "Wonderful weekend at the resort",
        "Coffee was burnt and lukewarm",
        "Picked up my order in 2 minutes, smooth",
        "Refund process is a nightmare",
        "Beautiful design, stunning screen quality",
        "Lost my data after the update, furious",
        "Flight was on time and crew was lovely",
        "Headphones broke after 2 weeks of use",
        "Best book I've read all year",
        "Internet drops every hour, cant work",
        "Massage therapist was incredible",
        "Late delivery again, third time this month",
        "Affordable and tastes great",
        "Slow shipping but product is excellent",
        "Charged hidden fees that werent disclosed",
    ]
    return pd.DataFrame({"message": samples})


# ---------- Main ----------
def main():
    st.title("Sentiment x Topic Insight Platform")
    st.markdown(
        "Predict sentiment AND surface what people are actually talking about — "
        "in a single dashboard. Built on Sentiment140 (1.6M tweets)."
    )

    art, missing = load_artifacts()

    if art is None:
        st.error(
            "Model artifacts not found. Run the modeling notebook first to produce them, "
            "then place them in an `artifacts/` folder next to `app.py`."
        )
        st.write("Missing files:")
        for m in missing:
            st.write(f"- `artifacts/{m}`")
        st.stop()

    render_sidebar(art)

    tab_single, tab_bulk = st.tabs(["Single message", "Bulk CSV upload"])
    with tab_single:
        render_single_mode(art)
    with tab_bulk:
        render_bulk_mode(art)

    st.markdown("---")
    st.caption(
        "Note: Sentiment140 labels were generated from emoticons, so model accuracy "
        "with classical methods plateaus around 80%. The strength of this app is the "
        "sentiment x topic insight layer, not raw accuracy."
    )


if __name__ == "__main__":
    main()

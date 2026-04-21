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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------- VADER lexicon safety net ----------
# VADER is a rule-based sentiment analyzer built for social media. It handles:
#   - formal English the model's TF-IDF doesn't know (dissatisfied, inadequate)
#   - negation ("not good" -> negative, "not bad" -> positive)
#   - intensifiers ("very bad" is stronger than "bad")
# We use it as a first-pass safety net: if VADER gives a strong signal, we trust it.
# Otherwise we fall back to the ML model trained on Sentiment140.

VADER_STRONG_THRESHOLD = 0.3   # |compound| above this -> trust VADER over ML
VADER_WEAK_THRESHOLD = 0.05    # |compound| below this -> VADER is uncertain, use ML
ML_MIN_VOCAB_FEATURES = 3      # below this, ML can't meaningfully predict — prefer VADER


@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()


def hybrid_predict(text, art, vader):
    """Two-stage sentiment prediction:
       1) VADER checks for lexical sentiment + handles negation
       2) ML model handles the long tail of informal tweet patterns

    Decision logic:
      - Strong VADER signal (|compound| >= 0.3) -> trust VADER
      - ML has few vocabulary matches (<3) AND VADER has any signal (|compound| > 0.05) -> trust VADER
      - Otherwise -> trust ML

    Returns: (pred, confidence, source, vader_compound, ml_proba, cleaned_text)
    """
    # VADER runs on the RAW text (it uses punctuation, capitals, etc. as signal)
    v = vader.polarity_scores(text)
    compound = v["compound"]

    # Always compute the ML prediction too — we show both in the UI for transparency
    cleaned = clean_tweet(text)
    X = art["tfidf"].transform([cleaned])
    ml_n_features = int(X.nnz)

    if hasattr(art["model"], "predict_proba"):
        ml_proba = float(art["model"].predict_proba(X)[0, 1])
    else:
        decision = float(art["model"].decision_function(X)[0])
        ml_proba = float(1.0 / (1.0 + np.exp(-decision)))

    # Decision rules (in order of priority)
    use_vader = False
    if abs(compound) >= VADER_STRONG_THRESHOLD:
        # Rule 1: VADER has a strong opinion — trust it
        use_vader = True
    elif ml_n_features < ML_MIN_VOCAB_FEATURES and abs(compound) >= VADER_WEAK_THRESHOLD:
        # Rule 2: ML has almost no vocabulary coverage AND VADER has any signal — trust VADER
        use_vader = True

    if use_vader:
        pred = 1 if compound > 0 else 0
        # Map compound magnitude to confidence:
        # |compound|=0.3 -> 65% confidence; |compound|=1.0 -> 100% confidence
        confidence = 0.5 + abs(compound) * 0.5
        source = "VADER"
    else:
        pred = 1 if ml_proba >= 0.5 else 0
        confidence = ml_proba if pred == 1 else 1 - ml_proba
        source = "ML"

    return pred, confidence, source, compound, ml_proba, cleaned


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
def predict_topics(cleaned_texts, art):
    Xt = art["topic_vec"].transform(cleaned_texts)
    topic_dist = art["topic_model"].transform(Xt)
    # Keep the RAW signal strength (how strongly NMF actually activated)
    raw_signal = topic_dist.sum(axis=1)
    # Row-normalize for relative comparison
    s = topic_dist.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    topic_dist_norm = topic_dist / s
    dominant = topic_dist_norm.argmax(axis=1)
    return topic_dist_norm, dominant, raw_signal


def explain_prediction(cleaned_text, art, top_k=5):
    """Return the words in the input that contributed most toward positive/negative.

    Works for LogisticRegression directly. For CalibratedClassifierCV (wrapping LinearSVC),
    we pull coefficients from the calibrated base estimator.
    """
    X = art["tfidf"].transform([cleaned_text])
    if X.nnz == 0:
        return [], [], 0

    model = art["model"]
    # Extract coefficient vector
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
    elif hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV wraps multiple base estimators
        coefs = [c.estimator.coef_[0] for c in model.calibrated_classifiers_
                 if hasattr(c, "estimator") and hasattr(c.estimator, "coef_")]
        if not coefs:
            return [], [], X.nnz
        coef = np.mean(coefs, axis=0)
    else:
        return [], [], X.nnz

    # Per-feature contribution = tfidf_value * coefficient
    feature_names = art["tfidf"].get_feature_names_out()
    contrib = X.multiply(coef).tocsr()
    row = contrib.getrow(0)
    indices = row.indices
    values = row.data

    word_contribs = [(feature_names[i], float(v)) for i, v in zip(indices, values)]
    pos_words = sorted([w for w in word_contribs if w[1] > 0], key=lambda x: -x[1])[:top_k]
    neg_words = sorted([w for w in word_contribs if w[1] < 0], key=lambda x: x[1])[:top_k]
    return pos_words, neg_words, X.nnz


def explain_topic(cleaned_text, dominant_topic, art, top_k=6):
    """Return the words in the input that pushed it into the dominant topic."""
    topic_vec = art["topic_vec"]
    topic_model = art["topic_model"]

    Xt = topic_vec.transform([cleaned_text])
    if Xt.nnz == 0:
        return []

    feature_names = topic_vec.get_feature_names_out()
    topic_weights = topic_model.components_[dominant_topic]

    row = Xt.getrow(0)
    indices = row.indices
    values = row.data

    # Weight each input word by how strongly it belongs to the dominant topic
    word_scores = [(feature_names[i], float(v * topic_weights[i]))
                   for i, v in zip(indices, values)]
    word_scores = [w for w in word_scores if w[1] > 0]
    word_scores.sort(key=lambda x: -x[1])
    return word_scores[:top_k]


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
        st.sidebar.markdown("### How predictions work")
        st.sidebar.markdown(
            "**Hybrid engine.** Every prediction first runs through **VADER** — a "
            "rule-based sentiment lexicon built for social media that handles "
            "negation, intensifiers, and formal English. If VADER has a strong "
            "opinion, we trust it. Otherwise we fall back to the **ML model** "
            "trained on Sentiment140."
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### ML Model")
        st.sidebar.markdown(f"**Sentiment:** {meta['baseline_model_name']}")
        st.sidebar.markdown(f"**Topics:** {meta['topic_model_name']} ({meta['n_topics']} topics)")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Test set performance (ML only)")
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

        vader = get_vader()
        pred_int, confidence, source, vader_compound, ml_proba, cleaned_text = hybrid_predict(text, art, vader)
        cleaned = [cleaned_text]  # downstream functions expect a list

        topic_dist, dominant, raw_signal = predict_topics(cleaned, art)
        topic_id = int(dominant[0])
        topic_score = float(topic_dist[0, topic_id])
        topic_raw = float(raw_signal[0])

        pos_words, neg_words, n_features = explain_prediction(cleaned[0], art)
        topic_words = explain_topic(cleaned[0], topic_id, art)

        # ---------- Hybrid engine banner — shows which engine decided ----------
        if source == "VADER":
            st.info(
                f"🛟 **Lexicon safety net used.** VADER detected strong sentiment "
                f"(compound score {vader_compound:+.2f}) and overrode the ML model. "
                f"The ML model would have said {('positive' if ml_proba >= 0.5 else 'negative')} "
                f"at {max(ml_proba, 1-ml_proba):.0%} confidence — but it only knows words it "
                f"saw during training, so we trust VADER for clear-cut cases like this."
            )
        else:
            st.success(
                f"🤖 **ML model used.** VADER had no strong opinion (compound {vader_compound:+.2f}), "
                f"so the decision came from the Sentiment140-trained classifier."
            )

        # ---------- Out-of-vocabulary note (only if ML was used and no features matched) ----------
        if source == "ML" and n_features == 0:
            st.error(
                "⚠️ **The ML model also doesn't know any of these words.** "
                "VADER couldn't find clear sentiment either. The prediction below is "
                "essentially a default guess. Try a longer message."
            )
        elif source == "ML" and n_features < 3:
            st.warning(
                f"Only {n_features} word(s) matched the ML vocabulary. Low reliability."
            )

        # ---------- Headline metrics ----------
        sentiment_label = "Positive 😊" if pred_int == 1 else "Negative 😟"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment", sentiment_label)
        c2.metric("Confidence", f"{confidence:.0%}")
        c3.metric("Engine", source)
        c4.metric(
            "Dominant topic",
            topic_label_for(topic_id, art),
            delta=f"signal: {topic_raw:.2f}",
            delta_color="off",
        )

        # ---------- Engine breakdown ----------
        with st.expander("🔬 See how each engine scored this"):
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("**VADER (lexicon-based)**")
                st.metric("Compound score", f"{vader_compound:+.3f}",
                          help="Range: -1 (very negative) to +1 (very positive). |score| > 0.5 = strong.")
                if abs(vader_compound) >= VADER_STRONG_THRESHOLD:
                    st.caption("✅ Strong enough to decide")
                else:
                    st.caption("⚪ Too weak to decide alone")
            with bc2:
                st.markdown("**ML model (Sentiment140)**")
                st.metric("Positive probability", f"{ml_proba:.1%}")
                if n_features == 0:
                    st.caption("❌ 0 words in vocabulary")
                elif n_features < 3:
                    st.caption(f"⚠️ Only {n_features} words in vocabulary")
                else:
                    st.caption(f"✅ {n_features} words in vocabulary")

        # Keep the original pred variable name compatible with the rest of the function
        pred = np.array([pred_int])
        proba = np.array([ml_proba])  # for the explainability section below, we use ML internals

        # ---------- Explainability: why did the ML model say what it said? ----------
        st.markdown("---")
        st.markdown("### 🔍 Why the ML model said what it said")
        st.caption(
            "Per-word contribution from the Sentiment140 classifier. Note: if VADER "
            "overrode the ML model above, the final answer is based on VADER, not these words."
        )

        if n_features == 0:
            st.info("No explanation available — none of your input words are in the model's vocabulary.")
        else:
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                st.markdown("**🟢 Pushed toward POSITIVE**")
                if pos_words:
                    df_pos = pd.DataFrame(pos_words, columns=["word", "strength"])
                    chart = (
                        alt.Chart(df_pos)
                        .mark_bar(color="#2E7D32")
                        .encode(
                            x=alt.X("strength:Q", title="Contribution"),
                            y=alt.Y("word:N", sort="-x", title=None),
                            tooltip=["word", alt.Tooltip("strength:Q", format=".3f")],
                        )
                        .properties(height=180)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("No positive-leaning words in your input.")

            with exp_col2:
                st.markdown("**🔴 Pushed toward NEGATIVE**")
                if neg_words:
                    df_neg = pd.DataFrame(neg_words, columns=["word", "strength"])
                    df_neg["strength_abs"] = df_neg["strength"].abs()
                    chart = (
                        alt.Chart(df_neg)
                        .mark_bar(color="#C62828")
                        .encode(
                            x=alt.X("strength_abs:Q", title="Contribution"),
                            y=alt.Y("word:N", sort="-x", title=None),
                            tooltip=["word", alt.Tooltip("strength:Q", format=".3f")],
                        )
                        .properties(height=180)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("No negative-leaning words in your input.")

        # ---------- Topic insight ----------
        st.markdown("---")
        st.markdown("### 🗂️ What is this message about?")

        if topic_raw < 0.05:
            st.warning(
                f"Weak topic signal ({topic_raw:.2f}). The input is too short or generic "
                "for the topic model to confidently place it. The match below is the best guess "
                "but should not be trusted."
            )

        tc1, tc2 = st.columns([1, 1])
        with tc1:
            st.markdown(f"**Best-match topic:** {topic_label_for(topic_id, art)}")
            words_in_topic = art["meta"]["topic_top_words"].get(str(topic_id), [])[:6]
            st.caption(f"Topic's defining words: {', '.join(words_in_topic)}")
            if topic_words:
                st.markdown("**Your words that triggered this match:**")
                for w, score in topic_words:
                    st.markdown(f"- `{w}` (score: {score:.2f})")
            else:
                st.caption("None of your words strongly match this topic's defining vocabulary.")

        with tc2:
            st.markdown("**All topic weights**")
            topic_df = pd.DataFrame({
                "Topic": [topic_label_for(i, art)[:25] for i in range(len(topic_dist[0]))],
                "Weight": topic_dist[0],
            }).sort_values("Weight", ascending=False)

            chart = (
                alt.Chart(topic_df)
                .mark_bar()
                .encode(
                    x=alt.X("Weight:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Topic:N", sort="-x", title=None),
                    color=alt.Color("Weight:Q", scale=alt.Scale(scheme="blues"), legend=None),
                    tooltip=["Topic", alt.Tooltip("Weight:Q", format=".1%")],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)

        with st.expander("🔧 Debug: cleaned text sent to the model"):
            st.code(cleaned[0] or "(empty after cleaning)")
            st.caption(
                f"{n_features} word(s) matched the model's TF-IDF vocabulary. "
                f"Raw topic signal: {topic_raw:.3f} (values below ~0.05 mean the topic model barely activated)."
            )


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
        with st.spinner("Scoring rows with VADER + ML hybrid..."):
            vader = get_vader()
            texts = df[text_col].fillna("").astype(str).tolist()

            preds, confs, sources, cleaned_list = [], [], [], []
            for t in texts:
                p, c, src, _, _, cl = hybrid_predict(t, art, vader)
                preds.append(p)
                confs.append(c)
                sources.append(src)
                cleaned_list.append(cl)

            preds = np.array(preds)
            topic_dist, dominant, _ = predict_topics(cleaned_list, art)

            df_out = df.copy()
            df_out["sentiment"] = ["positive" if p == 1 else "negative" for p in preds]
            df_out["confidence"] = np.round(confs, 3)
            df_out["engine"] = sources
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
    n_vader = int((df_out["engine"] == "VADER").sum()) if "engine" in df_out.columns else 0
    pct_vader = n_vader / n if n else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Messages analyzed", f"{n:,}")
    k2.metric("Negative", f"{n_neg:,}", delta=f"{pct_neg:.1%}")
    k3.metric("Positive", f"{n - n_neg:,}", delta=f"{1 - pct_neg:.1%}")
    k4.metric("Distinct topics", n_topics_seen)
    k5.metric("Decided by VADER", f"{n_vader:,}",
              delta=f"{pct_vader:.0%} of rows",
              delta_color="off",
              help="VADER handled rows with clear lexical sentiment. The rest went to the ML model.")

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
        "This app combines a rule-based lexicon (VADER) with a Sentiment140-trained "
        "ML classifier. VADER catches formal and negated language the ML model misses; "
        "the ML model handles the long tail of informal tweet patterns. The sentiment × "
        "topic dashboard is where the business insight lives — not in raw accuracy."
    )


if __name__ == "__main__":
    main()

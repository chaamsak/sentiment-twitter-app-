# Sentiment × Topic Insight Platform

A web app that classifies short-text sentiment **and** surfaces the topics people are actually talking about. Built on the 1.6M-tweet Sentiment140 corpus, deployed as a Streamlit app running a hybrid ML + lexicon ensemble.

**Live demo:** _(add your share.streamlit.io URL here after deploying)_

---

## What this app does

Turns *"people are unhappy"* into *"people are unhappy **about X**."*

Two modes:

- **Single message** — paste a tweet, review, or feedback comment → get sentiment, confidence, dominant topic, and a per-word explanation
- **Bulk CSV upload** — upload any text dataset → tag every row with sentiment + topic → see a sentiment × topic dashboard → download the tagged file

A built-in 50-row sample is included so anyone can try the bulk mode without uploading their own data.

---

## Architecture

The app runs a **hybrid ensemble** of two complementary sentiment components. Both components score every input, and a deterministic decision rule picks which one drives the final prediction. Both scores are always visible in the UI so users can see which component decided and why.

### Component 1: ML classifier (trained)

- **Vectorizer:** TF-IDF with 1–2 grams, `min_df=5`, `sublinear_tf=True`, ~43K features
- **Preprocessing (v2):** lowercase → strip URLs/mentions/HTML → drop apostrophes → negation marking with 3-token scope bounded by punctuation → WordNet lemmatization (verb form, then noun form)
- **Classifier:** Logistic Regression (`saga` solver, `C=1.0`), selected over LinearSVC on validation-set F1
- **Trained on:** 155K stratified Sentiment140 tweets (80% train / 10% val / 10% test, with per-user cap to prevent author-level leakage)
- **Strengths:** handles tweet-language patterns, slang, and common sentiment vocabulary
- **Weaknesses:** formal English (`dissatisfied`, `inadequate`) never appeared in 2009 Twitter, so the model has no signal for those words

### Component 2: Lexicon analyzer (rule-based)

- **Library:** VADER (Valence Aware Dictionary and sEntiment Reasoner) — a hand-curated lexicon of ~7,500 sentiment-bearing words with empirically-set valence scores
- **Strengths:** handles negation, intensifiers, and formal English natively. Reliable on clear-cut lexical cases the ML component's training corpus didn't cover
- **Weaknesses:** can't learn domain-specific patterns, struggles with sarcasm and context-dependent sentiment

### Ensemble decision rule

Both components always score the input. The lexicon takes precedence when:
- It produces a strong signal (`|compound| ≥ 0.3`), OR
- The ML component has fewer than 3 vocabulary matches AND the lexicon has any signal (`|compound| ≥ 0.05`)

Otherwise the ML component's prediction stands. This keeps the ML component as the default path while allowing the lexicon to handle cases the ML component demonstrably can't.

---

## Repo structure

This is the **deployment repo**. It contains only what the live app needs to run:

```
sentiment-twitter-app/
├── app.py                       # Streamlit application (hybrid ensemble)
├── requirements.txt             # Python dependencies
├── README.md                    # this file
└── artifacts/                   # trained models (produced by the research notebooks)
    ├── tfidf_vectorizer.joblib
    ├── sentiment_model.joblib
    ├── topic_vectorizer.joblib
    ├── topic_model.joblib
    ├── metadata.json
    └── preprocessing_config.json
```

The research notebooks (EDA, cleaning, model training, ablation studies, evaluation) live separately and are not required to run the app.

---

## Data

**Training data is not included in this repo.** This repo ships only the trained model artifacts needed to serve the app.

- **Dataset:** Sentiment140 (Go, Bhayani & Huang, 2009)
- **Source:** [kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size:** 1.6M tweets; this project was trained on a stratified 200K sample
- **Labels:** auto-generated from emoticons at collection time (not human-verified)

The cleaning and training code lives in the research notebooks. If you want to retrain from scratch, download the Sentiment140 CSV from Kaggle and run those notebooks to regenerate the `artifacts/` folder.

---

## Honest positioning

Sentiment140 labels come from emoticons, not human review. Classical ML methods therefore plateau around **80% accuracy** on this dataset — that's a ceiling, not a failure.

Adding topic features to a TF-IDF baseline rarely moves accuracy much because TF-IDF already captures most lexical signal on short text. The **ML accuracy delta from topic features is near zero**; the value of topic modeling here is the **interpretability layer** — the ability to say not just "this tweet is negative" but "this tweet is negative AND it's about customer service." The notebook reports baseline-vs-enhanced numbers honestly.

The **hybrid architecture** is what makes the app usable beyond the training corpus. The ML component was honest about its failure modes — formal English and unambiguous negation — so the deployed system uses the lexicon to cover them. This is a standard production pattern (ensemble with specialization), not a workaround.

---

## Methodology summary (research notebooks)

**Preprocessing (v2 — upgraded from Review 1):**
- Negation-aware cleaning: detect negation words, prefix next 3 tokens with `not_`, scope bounded by punctuation
- WordNet lemmatization: `frustrated / frustrating / frustration` all collapse to `frustrate`
- Ablation study in the notebook compares v1 (original) and v2 preprocessing with identical hyperparameters

**Sentiment classifier selection:**
- Tested Logistic Regression and LinearSVC on TF-IDF features
- Winner selected by F1 on validation set, final numbers reported on held-out test set

**Topic model selection:**
- Tested LDA and NMF at 10 topics
- Winner selected by a co-occurrence-based coherence score
- NMF won, consistent with literature on short-text topic modeling

**Leakage prevention:**
- Per-user tweet cap (30 max) prevents any single author from dominating training
- Stratified 60/20/20 train/validation/test split
- Test set touched exactly once for final reporting

**Reproducibility:**
- `RANDOM_STATE = 42` throughout
- All decisions validated with measurable criteria, not inspection alone

---

## Run the app locally

Requires Python 3.10+.

```bash
git clone <this repo URL>
cd sentiment-twitter-app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. First launch downloads the WordNet corpus (~10 MB) automatically.

---

## Deploy for free

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app → Deploy a public app from GitHub**
4. Select this repo, set **Main file path** to `app.py`, click **Deploy**

First build takes 3–5 minutes (installs dependencies + downloads WordNet). You'll get a permanent URL you can share.

---

## Next iterations

- Fine-tune DistilBERT as a third ensemble component for the long tail of cases neither TF-IDF nor VADER handles well
- Character n-gram features to improve OOV handling without leaving the classical-ML regime
- Scale to the full 1.6M with `HashingVectorizer` + `partial_fit` for streaming training

---

## Citation

If you use this work, please cite the original dataset:

> Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.* CS224N Project Report, Stanford.

And the VADER library:

> Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* Eighth International Conference on Weblogs and Social Media (ICWSM-14).

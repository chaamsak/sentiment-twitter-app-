# Sentiment × Topic Insight Platform

A web app that classifies short-text sentiment **and** surfaces the topics people are actually talking about. Built on the 1.6M-tweet Sentiment140 corpus, deployed as a Streamlit app running a TF-IDF + Logistic Regression classifier alongside NMF topic modeling.

**Live demo:** _(add your share.streamlit.io URL here after deploying)_

---

## What this app does

Turns *"people are unhappy"* into *"people are unhappy **about X**."*

Two modes:

- **Single message** — paste a tweet, review, or feedback comment → get sentiment, confidence, dominant topic, and a per-word explanation of the model's reasoning
- **Bulk CSV upload** — upload any text dataset → tag every row with sentiment + topic → see a sentiment × topic dashboard → download the tagged file

A built-in 50-row sample is included so anyone can try the bulk mode without uploading their own data.

---

## Architecture

The app runs a single ML pipeline with two complementary models working together:

### Sentiment classifier

- **Vectorizer:** TF-IDF with 1–2 grams, `min_df=5`, `sublinear_tf=True`, ~43K features
- **Preprocessing (v2):** lowercase → strip URLs/mentions/HTML → drop apostrophes → negation marking with 3-token scope bounded by punctuation → WordNet lemmatization (verb form, then noun form)
- **Classifier:** Logistic Regression (`saga` solver, `C=1.0`), selected over LinearSVC based on validation-set F1
- **Trained on:** 155K stratified Sentiment140 tweets (60% train / 20% val / 20% test, with per-user cap to prevent author-level leakage)

### Topic model

- **Method:** NMF on TF-IDF features, 10 topics (selected over LDA based on coherence scoring)
- **Preprocessing:** domain-specific stopword list plus content-word filtering to surface real themes instead of pronouns/fillers
- **Topics discovered:** Work & routine, Sleep & exhaustion, Sickness, Missing friends, Music & entertainment, Love & appreciation, Weekend plans, Good wishes, Twitter activity, Home & boredom, Future plans, Sadness & loss

### Explainability

Every prediction in the app shows which exact words from the input pushed the score toward positive or negative, with their TF-IDF × coefficient contribution values. Users see the model's reasoning, not just its verdict.

---

## Honest positioning

Sentiment140 labels come from emoticons, not human review. Classical ML methods therefore plateau around **80% accuracy** on this dataset — that's a ceiling, not a failure.

Adding topic features to the TF-IDF baseline rarely moves accuracy much because TF-IDF already captures most lexical signal on short text. The **accuracy delta from topic features is near zero**; the value of topic modeling here is the **interpretability layer** — saying not just "this tweet is negative" but "this tweet is negative AND it's about customer service." The notebook reports baseline-vs-enhanced numbers honestly.

### Known model limitations

The app surfaces these honestly rather than hiding them:

- **Out-of-vocabulary formal words.** Sentiment140 is 2009 Twitter language. Formal English like `dissatisfied`, `inadequate`, `unacceptable` barely appears in the training corpus, so the model has no learned signal for them. When input words don't appear in the model's vocabulary, the app shows a clear warning that the prediction is unreliable.
- **Low-vocab inputs.** Very short or unusual messages may match only 1–2 words in the vocabulary. The app flags these with a weak-signal warning.
- **Label noise.** Emoticon-based labels introduce 10–15% noise, which caps achievable accuracy.

The next iteration would swap TF-IDF for transformer embeddings (DistilBERT), which handle OOV natively through subword tokenization.

---

## Repo structure

This is the **deployment repo**. It contains only what the live app needs to run:

```
sentiment-twitter-app/
├── app.py                       # Streamlit application
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

## Methodology summary (research notebooks)

**Preprocessing (v2 — upgraded from Review 1):**
- Negation-aware cleaning: detect negation words, prefix next 3 tokens with `not_`, scope bounded by punctuation
- WordNet lemmatization: `frustrated / frustrating / frustration` all collapse to `frustrate`
- Ablation study in the notebook compares v1 (original) vs v2 preprocessing with identical hyperparameters, plus a targeted evaluation on 15 curated negation/word-family cases

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
- All modeling decisions validated with measurable criteria, not inspection alone

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

- Fine-tune DistilBERT to handle OOV words natively through subword tokenization
- Character n-gram features as a lighter-weight path to partial OOV handling
- Scale to the full 1.6M with `HashingVectorizer` + `partial_fit` for streaming training

---

## Citation

If you use this work, please cite the original dataset:

> Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.* CS224N Project Report, Stanford.

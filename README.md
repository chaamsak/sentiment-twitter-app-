# Sentiment × Topic Insight Platform

A web app that classifies short-text sentiment **and** surfaces the topics people are actually talking about — built on the 1.6M-tweet Sentiment140 dataset and deployed with Streamlit.

**Live demo:** _(add your share.streamlit.io URL here after deploying)_

---

## What this app does

Turns *"people are unhappy"* into *"people are unhappy **about X**."*

Two modes:

- **Single message** — paste a tweet, review, or feedback comment → get sentiment, confidence, and dominant topic
- **Bulk CSV upload** — upload any text dataset → tag every row with sentiment + topic → see a sentiment × topic dashboard → download the tagged file

A built-in 50-row sample is included so anyone can try the bulk mode without uploading their own data.

---

## Repo structure

This is the **deployment repo**. It contains only what the live app needs to run:

```
sentiment-twitter-app/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # this file
└── artifacts/              # trained models (produced by the research notebooks)
    ├── tfidf_vectorizer.joblib
    ├── sentiment_model.joblib
    ├── topic_vectorizer.joblib
    ├── topic_model.joblib
    └── metadata.json
```

The research notebooks (EDA, cleaning, model training, evaluation) live separately and are not required to run the app.

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

Sentiment140 labels come from emoticons, not human review. Classical methods therefore plateau around **80% accuracy** on this dataset — that's a ceiling, not a failure.

Adding topic features to a TF-IDF baseline rarely moves accuracy much because TF-IDF already captures most lexical signal on short text. **The point of this project is the interpretability layer**, not the accuracy delta. The notebook reports baseline-vs-enhanced honestly; the app is where the actual business value lives.

---

## Run the app locally

Requires Python 3.10+.

```bash
git clone <this repo URL>
cd sentiment-twitter-app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Deploy for free

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app → Deploy a public app from GitHub**
4. Select this repo, set **Main file path** to `app.py`, click **Deploy**

First build takes 3–5 minutes. You'll get a permanent URL you can share.

---

## Methodology summary

- **Sentiment classifier:** tested Logistic Regression vs LinearSVC on TF-IDF (1–2 grams, `min_df=5`, `sublinear_tf=True`). Winner picked by F1 on a held-out 20% test set.
- **Topic model:** tested LDA vs NMF at 10 topics. Winner picked by a topic-coherence proxy. NMF won, consistent with literature on short text.
- **Enhanced model:** TF-IDF matrix stacked with normalized topic distributions, refit with Logistic Regression.
- **Reproducibility:** `RANDOM_STATE = 42` throughout; test set never touched during training.

---

## What's next

- Fine-tune DistilBERT for the sentiment classifier
- Sentence-transformer embeddings for topic clustering
- Scale to the full 1.6M with `HashingVectorizer` + `partial_fit`

---

## Citation

If you use this work, please cite the original dataset:

> Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.* CS224N Project Report, Stanford.

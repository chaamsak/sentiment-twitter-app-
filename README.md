# Sentiment x Topic Insight Platform

Twitter sentiment classification combined with topic discovery, deployed as a usable web app.  
**Dataset:** Sentiment140 (1.6M tweets) — Kaggle.

## What this project actually is

A two-stage pipeline that turns "people are unhappy" into **"people are unhappy about X."**

1. **Stage 1 — Sentiment classifier** (TF-IDF + Logistic Regression / LinearSVC)  
   Predicts positive vs negative for any short text.

2. **Stage 2 — Topic discovery** (LDA + NMF, picks the more coherent one)  
   Surfaces 10 themes from the corpus and tags each message with its dominant topic.

3. **Stage 3 — Dashboard** (Streamlit)  
   Crosses sentiment with topic to show **negative rate by theme** — the actionable view a real CX team would use.

## Honest positioning

Sentiment140 labels were auto-generated from emoticons, so classical methods plateau around **80% accuracy**. Adding topic features to TF-IDF rarely moves accuracy meaningfully because TF-IDF already captures lexical signal. **The win of this project is the insight layer, not the accuracy delta.** The notebook reports the comparison honestly and lets the dashboard carry the business value.

## Project structure

```
sentiment_project/
├── sentiment140_review1_clean.csv       # produced by Review 1 notebook (you provide)
├── sentiment140_review2_modeling.ipynb  # this repo — modeling + topics + business layer
├── app.py                               # Streamlit platform
├── requirements.txt                     # for deployment
├── README.md                            # this file
└── artifacts/                           # produced by the notebook
    ├── tfidf_vectorizer.joblib
    ├── sentiment_model.joblib
    ├── topic_vectorizer.joblib
    ├── topic_model.joblib
    └── metadata.json
```

## Setup

### 1. Run the modeling notebook
Open `sentiment140_review2_modeling.ipynb` in Colab or Jupyter, point it at your cleaned CSV, run all cells.  
Expected runtime on Colab CPU: ~10–15 minutes for 200K rows.  
Output: the `artifacts/` folder.

### 2. Refine topic labels (recommended)
After the notebook prints the top words per topic, **scroll back to the labeling cell and overwrite `topic_labels[i]`** with human names like:
```python
topic_labels[0] = "Work & job complaints"
topic_labels[1] = "Music & entertainment"
```
Re-run the labeling cell + the save-artifacts cell. Better labels = sharper dashboard.

### 3. Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501.

### 4. Deploy publicly (free)
- Push this folder to a GitHub repo (include the `artifacts/` folder)
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect the repo, point to `app.py`, deploy

## What the app does

**Tab 1 — Single message:**  
Paste a tweet/review/comment → get sentiment, confidence, dominant topic, full topic distribution.

**Tab 2 — Bulk CSV upload:**  
Upload any CSV with a text column → tags every row → bubble-chart dashboard showing volume vs negative rate per topic → downloadable tagged CSV.

There's also a built-in 50-row sample dataset so reviewers can demo the bulk mode without uploading anything.

## For the academic review

- Reproducibility: `RANDOM_STATE = 42` everywhere
- Honest comparison: baseline vs enhanced reported with accuracy, F1, ROC-AUC
- Two algorithms tried per task (LR vs LinearSVC for sentiment; LDA vs NMF for topics)
- Topic coherence proxy used to pick the topic model rather than guessing
- Test set held out from all training including topic fitting

## For the portfolio

Two sentences for your CV / LinkedIn:

> Built an end-to-end NLP pipeline on the 1.6M-tweet Sentiment140 dataset, combining a TF-IDF + Logistic Regression sentiment classifier with NMF topic discovery to produce a sentiment×topic insight layer. Deployed as an interactive Streamlit dashboard that lets non-technical users tag any text dataset and see where negative sentiment is concentrated by theme.

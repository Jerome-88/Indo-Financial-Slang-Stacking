---
title: CS IDX30 Financial Sentiment API
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# CS-IDX30 Financial Sentiment Analysis

Sentiment analysis system for Indonesian stock market discourse, targeting slang-heavy texts from Twitter and Stockbit. Classifies text as **Negative**, **Neutral**, or **Positive** using traditional ML models with TF-IDF features.

## Project Structure

```
AOL_NLP/
├── README.md
├── Dockerfile                     # HF Spaces deployment
├── requirements.txt
├── requirements_light.txt         # Lightweight deps (no torch)
├── .gitignore
├── backend/
│   └── app.py                     # FastAPI inference server
├── notebooks/
│   ├── merge.ipynb                # Merge 4 raw datasets into one
│   ├── preprocess.ipynb           # Clean & normalize text
│   └── train_experiment.ipynb     # Hyperparameter tuning & evaluation
├── Dataset/
│   └── dataset_final_clean.csv    # Final preprocessed dataset (6,120 rows)
├── model_final/
│   ├── pipeline_svm.joblib        # Pipeline(TF-IDF + SVM)
│   ├── pipeline_lr.joblib         # Pipeline(TF-IDF + LR)
│   ├── pipeline_rf.joblib         # Pipeline(TF-IDF + RF)
│   ├── pipeline_xgboost.joblib    # Pipeline(TF-IDF + XGBoost)
│   ├── tfidf_vectorizer_30k.joblib
│   ├── oof_cache.joblib           # OOF predictions cache
│   └── model_info.json            # CV results & best params
├── results/                       # Visualization outputs (PNG)
└── frontend/                      # React + TypeScript dashboard
```

## Models

Hyperparameter tuning via **GridSearchCV** (SVM, LR) and **RandomizedSearchCV** (RF, XGBoost) with 5-Fold Stratified CV.

| Model | Tuning Method | OOF Macro F1 |
|---|---|---|
| SVM (LinearSVC) | GridSearchCV (9 combos) | 0.6241 |
| **Logistic Regression** | **GridSearchCV (16 combos)** | **0.6593** |
| Random Forest | RandomizedSearchCV (30/360) | 0.6351 |
| XGBoost | RandomizedSearchCV (30/46,080) | 0.6353 |
| Soft Voting Ensemble | — | 0.6552 |

> OOF (Out-of-Fold) scores from `cross_val_predict` with Pipeline(TF-IDF + model) — leak-free evaluation.

**TF-IDF Configuration:** 30k max features, min_df=3, unigram+bigram, sublinear_tf.

## Dataset

Merged from 4 sources (6,140 → 6,120 rows after cleaning):

| Source | Platform | Records |
|---|---|---|
| IDSMSA | Twitter | 3,288 |
| ARTO | Stockbit | 495 |
| ICCSCI | Stockbit | 1,842 |
| PTBA | Stockbit | 495 |

## Deployment

- **Backend:** [HF Spaces](https://huggingface.co/spaces/siomay88/aol_nlp) (Docker, FastAPI)
- **Frontend:** Vercel (React + Vite, proxies API to HF Spaces)

## Setup (Local)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install
```

## Running (Local)

**Backend (FastAPI):**
```bash
python backend/app.py
# http://localhost:8000
```

**Frontend (React dev server):**
```bash
cd frontend && npm run dev
# http://localhost:5173
```

## Notebook Workflow

Run notebooks in order:

1. `merge.ipynb` — merge raw datasets → `dataset_combined_final.csv`
2. `preprocess.ipynb` — clean & normalize → `Dataset/dataset_final_clean.csv`
3. `train_experiment.ipynb` — hyperparameter tuning, evaluation, ensemble → `model_final/`

Set `RETRAIN = True` in the tuning cell to re-run training, or `False` to load from saved `.joblib` files.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single-text prediction (one model) |
| `/predict/batch` | POST | Batch prediction |
| `/predict/multi` | POST | Multi-model prediction with ensemble |
| `/slang/search` | GET | Search Indonesian financial slang dictionary |

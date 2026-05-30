# FinSlang-AI: Indonesian Financial Sentiment Analysis

Sentiment analysis system for Indonesian stock market discourse, targeting slang-heavy texts from Twitter and Stockbit. Classifies text as **Negative**, **Neutral**, or **Positive**.

## Project Structure

```
AOL_NLP/
├── README.md
├── requirements.txt
├── .gitignore
├── backend/
│   ├── app.py              # FastAPI inference server
│   └── train.py            # Transformer training pipeline (K-Fold CV)
├── notebooks/
│   ├── 01_merge.ipynb              # Merge 4 raw datasets into one
│   ├── 02_preprocess.ipynb         # Clean & normalize text
│   └── 03_hyperparameter_search.ipynb  # Traditional ML experiments
├── Dataset/
│   └── dataset_final_clean.csv     # Final preprocessed dataset (6,119 rows)
├── model_final/
│   ├── pipeline_svm.joblib
│   ├── pipeline_lr.joblib
│   ├── pipeline_rf.joblib
│   ├── pipeline_xgboost.joblib
│   ├── tfidf_vectorizer_30k.joblib
│   └── model_info.json
├── results/                # Visualization outputs (PNG)
└── frontend/               # React + TypeScript dashboard
```

## Models

| Model | Type | Val Macro F1 |
|---|---|---|
| SVM (TF-IDF) | Traditional | 0.607 |
| Logistic Regression (TF-IDF) | Traditional | 0.618 |
| Random Forest (TF-IDF) | Traditional | 0.568 |
| XGBoost (TF-IDF) | Traditional | 0.609 |
| IndoBERT | Transformer | ~0.74 |
| IndoBERTweet | Transformer | ~0.74 |
| XLM-RoBERTa | Transformer | ~0.74 |
| **Ensemble (Transformers)** | **Transformer** | **0.743** |

> Transformer weights (`*.pt`) are not tracked by Git due to size. Run `train.py` to generate them.

## Dataset

Merged from 4 sources (6,140 → 6,119 rows after cleaning):

| Source | Platform | Records |
|---|---|---|
| IDSMSA | Twitter | 3,288 |
| ARTO | Stockbit | 499 |
| ICCSCI | Stockbit | 1,853 |
| PTBA | Stockbit | 500 |

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install
```

## Running

**Backend (FastAPI):**
```bash
python backend/app.py
# → http://localhost:8000
```

**Frontend (React dev server):**
```bash
cd frontend && npm run dev
# → http://localhost:5173
```

**Train transformers:**
```bash
python backend/train.py
# Saves weights to saved_models_run_<timestamp>/ at project root
```

## Notebook Workflow

Run notebooks in order:

1. `01_merge.ipynb` — merge raw datasets → `dataset_combined_final.csv`
2. `02_preprocess.ipynb` — clean & normalize → `Dataset/dataset_final_clean.csv`
3. `03_hyperparameter_search.ipynb` — tune traditional ML models → `model_final/`

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single-text prediction (one model) |
| `/predict/batch` | POST | Batch prediction |
| `/predict/multi` | POST | Multi-model prediction with XAI explanation |

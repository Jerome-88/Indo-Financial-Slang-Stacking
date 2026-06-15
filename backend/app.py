
import json
import os
import time
import joblib
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

try:
    import torch
    from torch.cuda.amp import autocast
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import hf_hub_download
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

#config

BASE_DIR = Path(__file__).resolve().parent.parent

LABEL_NAMES  = ["Negative", "Neutral", "Positive"]
LABEL_MAP    = {"Negative": 0, "Neutral": 1, "Positive": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

VECTORIZER_PATH  = str(BASE_DIR / "model_final/tfidf_vectorizer_30k.joblib")
TRADITIONAL_INFO = str(BASE_DIR / "model_final/model_info.json")

# model traditional
TRADITIONAL_MODEL_PATHS = {
    "svm"                 : str(BASE_DIR / "model_final/pipeline_svm.joblib"),
    "random_forest"       : str(BASE_DIR / "model_final/pipeline_rf.joblib"),
    "xgboost"             : str(BASE_DIR / "model_final/pipeline_xgboost.joblib"),
    "logistic_regression" : str(BASE_DIR / "model_final/pipeline_lr.joblib"),
}

# model transformer
# Set HF_MODEL_REPO env var on HF Spaces to enable auto-download of .pt weights.
# Example: HF_MODEL_REPO = "yourname/cs-idx30-models"
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")

TRANSFORMER_CONFIGS = {
    "indobert": {
        "checkpoint"  : "indobenchmark/indobert-base-p2",
        "weights"     : str(BASE_DIR / "saved_models_run_20260514_192555/indobert_fold2.pt"),
        "hf_filename" : "indobert_fold2.pt",
        "max_len"     : 128,
    },
    "indoroberta": {
        "checkpoint"  : "indolem/indobertweet-base-uncased",
        "weights"     : str(BASE_DIR / "saved_models_run_20260514_192555/indoroberta_fold1.pt"),
        "hf_filename" : "indoroberta_fold1.pt",
        "max_len"     : 128,
    },
    "xlmr": {
        "checkpoint"  : "xlm-roberta-base",
        "weights"     : str(BASE_DIR / "saved_models_run_20260514_192555/xlmr_fold1.pt"),
        "hf_filename" : "xlmr_fold1.pt",
        "max_len"     : 128,
    },
}



logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)



class ModelRegistry:
    def __init__(self):
        if TORCH_AVAILABLE:
            self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.use_amp = self.device.type == "cuda"
        else:
            self.device  = None
            self.use_amp = False

        self.vectorizer             = None
        self.traditional_models     = {}
        self.best_traditional_name  = "unknown"
        self.transformers           = {}
        self.loaded_models          = []

    def load_traditional(self):
        if Path(VECTORIZER_PATH).exists():
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("TF-IDF Vectorizer loaded")
        else:
            logger.warning(f"Standalone vectorizer not found {VECTORIZER_PATH}")

        # Load semua traditional model
        for name, path in TRADITIONAL_MODEL_PATHS.items():
            if Path(path).exists():
                self.traditional_models[name] = joblib.load(path)
                self.loaded_models.append(name)
                logger.info(f"Traditional model loaded: {name}")
            else:
                logger.warning(f"Not found {path}")

        # Tentukan best model berdasarkan val_f1 dari model_info.json
        NAME_MAP = {"SVM": "svm", "LR": "logistic_regression", "RF": "random_forest", "XGBoost": "xgboost"}
        if Path(TRADITIONAL_INFO).exists():
            with open(TRADITIONAL_INFO) as f:
                info = json.load(f)
            models_info = info.get("models", {})
            if models_info:
                best_key = max(models_info, key=lambda k: models_info[k].get("val_f1", 0))
                self.best_traditional_name = NAME_MAP.get(best_key, best_key.lower())
            logger.info(f"Best traditional: {self.best_traditional_name}")

    def load_transformer(self, name: str, cfg: dict):
        if not TORCH_AVAILABLE:
            logger.warning(f"torch/transformers not installed — skipping {name}")
            return

        weights_path = cfg["weights"]

        if not Path(weights_path).exists():
            if HF_MODEL_REPO:
                try:
                    logger.info(f"Downloading {name} weights from {HF_MODEL_REPO} ...")
                    weights_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=cfg["hf_filename"])
                    logger.info(f"Downloaded: {weights_path}")
                except Exception as e:
                    logger.warning(f"HF Hub download failed for {name}: {e}  skipping")
                    return
            else:
                logger.warning(f"Weights not found: {weights_path}  skipping {name}")
                return

        logger.info(f"Loading transformer: {name}...")
        tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
        model     = AutoModelForSequenceClassification.from_pretrained(
            cfg["checkpoint"], num_labels=3
        )
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        self.transformers[name] = {
            "tokenizer" : tokenizer,
            "model"     : model,
            "max_len"   : cfg["max_len"],
        }
        self.loaded_models.append(name)
        logger.info(f"{name} loaded on {self.device}")

    def load_all(self):
        logger.info(f"Device: {self.device}  |  torch_available: {TORCH_AVAILABLE}")
        self.load_traditional()
        if TORCH_AVAILABLE:
            for name, cfg in TRANSFORMER_CONFIGS.items():
                self.load_transformer(name, cfg)
        else:
            logger.warning("Transformer models skipped (torch not installed)")
        logger.info(f"Loaded models: {self.loaded_models}")

registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading all models...")
    registry.load_all()
    logger.info("All models ready!")
    yield
    logger.info("Shutting down...")



app = FastAPI(
    title       = "Financial Sentiment API",
    description = """
Sentiment Analysis untuk teks finansial Indonesia (slang-aware).

**Model Traditional (TF-IDF based, cepat < 5ms):**
- `svm` — SVM Linear
- `random_forest` — Random Forest
- `xgboost` — XGBoost
- `logistic_regression` — Logistic Regression
- `traditional` — Best model otomatis dari training

**Model Transformer (~100ms CPU / ~20ms GPU):**
- `indobert` — IndoBERT
- `indoroberta` — IndoBERTweet
- `xlmr` — XLM-RoBERTa

**Ensemble:**
- `ensemble` — Soft voting semua transformer
- `ensemble_all` — Soft voting 4 model traditional (SVM + RF + XGBoost + LR)
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

#skema

VALID_MODELS = (
    "svm", "random_forest", "xgboost", "logistic_regression", "traditional",
    "indobert", "indoroberta", "xlmr",
    "ensemble", "ensemble_all",
)

class PredictRequest(BaseModel):
    text  : str = Field(..., example="HAKA ARA lagi, cuan terus serok aja bro")
    model : str = Field(
        default = "traditional",
        example = "traditional",
        description = f"Pilih: {' | '.join(VALID_MODELS)}",
    )

class BatchPredictRequest(BaseModel):
    texts : list[str] = Field(..., example=["HAKA naik terus", "nyangkut parah nih"])
    model : str       = Field(default="traditional")

class PredictResponse(BaseModel):
    text          : str
    label         : str
    confidence    : float
    probabilities : dict
    model_used    : str
    latency_ms    : float
    xai_explanation: Optional[str] = None

class BatchPredictResponse(BaseModel):
    results    : list[PredictResponse]
    total      : int
    model_used : str
    latency_ms : float


class MultiPredictRequest(BaseModel):
    text   : str = Field(..., example="Harga saham turun, rugi banyak")
    models : list[str] = Field(
        default_factory=lambda: ["traditional"],
        example=["traditional", "indobert", "ensemble"],
        description="List model yang ingin diminta prediksinya",
    )


#fungsi prediksi

def predict_traditional(text: str, model_key: str) -> dict:
    resolved_key = registry.best_traditional_name if model_key == "traditional" else model_key

    if resolved_key not in registry.traditional_models:
        raise HTTPException(
            status_code = 503,
            detail      = f"Model '{resolved_key}' tidak loaded. "
                          f"Available: {list(registry.traditional_models.keys())}"
        )

    model = registry.traditional_models[resolved_key]  
    pred  = model.predict([text])[0]
    proba = model.predict_proba([text])[0]

    display_name = (
        f"traditional (best={resolved_key})"
        if model_key == "traditional"
        else model_key
    )

    return {
        "label"         : IDX_TO_LABEL[int(pred)],
        "confidence"    : round(float(proba.max()), 4),
        "probabilities" : {l: round(float(p), 4) for l, p in zip(LABEL_NAMES, proba)},
        "model_used"    : display_name,
    }

def predict_transformer(text: str, model_name: str) -> dict:
    if model_name not in registry.transformers:
        raise HTTPException(
            status_code = 503,
            detail      = f"Transformer '{model_name}' tidak loaded. "
                          f"Available: {list(registry.transformers.keys())}"
        )

    t         = registry.transformers[model_name]
    inputs    = t["tokenizer"](
        text,
        max_length     = t["max_len"],
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )
    inputs = {k: v.to(registry.device) for k, v in inputs.items()}

    with torch.no_grad():
        with autocast(enabled=registry.use_amp):
            outputs = t["model"](**inputs)

    proba = torch.softmax(outputs.logits.float(), dim=-1)[0].cpu().numpy()
    pred  = int(np.argmax(proba))

    return {
        "label"         : IDX_TO_LABEL[pred],
        "confidence"    : round(float(proba.max()), 4),
        "probabilities" : {l: round(float(p), 4) for l, p in zip(LABEL_NAMES, proba)},
        "model_used"    : model_name,
    }

def predict_ensemble(text: str) -> dict:
    all_proba   = []
    models_used = []

    for name in registry.transformers:
        res = predict_transformer(text, name)
        all_proba.append(list(res["probabilities"].values()))
        models_used.append(name)

    if not all_proba:
        raise HTTPException(status_code=503, detail="Tidak ada transformer yang loaded untuk ensemble")

    avg_proba = np.mean(all_proba, axis=0)
    pred      = int(np.argmax(avg_proba))

    return {
        "label"         : IDX_TO_LABEL[pred],
        "confidence"    : round(float(avg_proba.max()), 4),
        "probabilities" : {l: round(float(p), 4) for l, p in zip(LABEL_NAMES, avg_proba)},
        "model_used"    : f"ensemble ({' + '.join(models_used)})",
    }

def predict_ensemble_traditional(text: str) -> dict:
    all_proba   = []
    models_used = []

    for name in ["svm", "random_forest", "xgboost", "logistic_regression"]:
        if name in registry.traditional_models:
            res = predict_traditional(text, name)
            all_proba.append(list(res["probabilities"].values()))
            models_used.append(name)

    if not all_proba:
        raise HTTPException(status_code=503, detail="Tidak ada model traditional yang loaded untuk ensemble_all")

    avg_proba = np.mean(all_proba, axis=0)
    pred      = int(np.argmax(avg_proba))

    return {
        "label"         : IDX_TO_LABEL[pred],
        "confidence"    : round(float(avg_proba.max()), 4),
        "probabilities" : {l: round(float(p), 4) for l, p in zip(LABEL_NAMES, avg_proba)},
        "model_used"    : f"ensemble_all ({' + '.join(models_used)})",
    }


def estimate_slang_density(text: str) -> float:
    slang_tokens = {
        "bro", "cuy", "anjay", "gila", "cuan", "fomo", "santuy", "ngacir",
        "mabar", "saham", "fix", "banget", "nih", "deh", "wkwk",
    }
    tokens = [token.strip(".,!?:;\"'()[]{}").lower() for token in text.split() if token.strip()]
    if not tokens:
        return 0.0
    slang_count = sum(1 for token in tokens if token in slang_tokens)
    return round(slang_count / len(tokens), 4)


def generate_xai_explanation(
    models_scores: dict,
    slang_density: float,
    consensus: str,
    final_confidence: float,
) -> str:

    normalized = {
        model: {
            "sentiment": str(payload.get("sentiment", "UNKNOWN")).upper(),
            "confidence": float(payload.get("confidence", 0.0)),
        }
        for model, payload in models_scores.items()
    }

    indobert = normalized.get("indobert", {"sentiment": "UNKNOWN", "confidence": 0.0})
    indobertweet = normalized.get("indoroberta", {"sentiment": "UNKNOWN", "confidence": 0.0})
    xlmr = normalized.get("xlmr", {"sentiment": "UNKNOWN", "confidence": 0.0})

    labels = [indobert["sentiment"], indobertweet["sentiment"], xlmr["sentiment"]]
    unique_labels = set(labels)
    high_confidence_count = sum(1 for payload in normalized.values() if payload["confidence"] >= 0.75)
    consensus_label = str(consensus).upper()
    average_confidence = float(final_confidence)
    low_slang = slang_density < 0.05

    if low_slang and consensus_label in {"NEGATIVE", "POSITIVE"} and average_confidence >= 0.60:
        return (
            f"Meskipun teks menunjukkan slang rendah ({round(slang_density * 100, 1)}%), "
            f"sistem mendeteksi sinyal pasar yang kuat sehingga memilih {consensus_label}. "
            "Ini konsisten dengan kasus berita formal di mana frasa teknikal atau aksi pasar "
            "membawa bobot sentimen lebih besar daripada nada jurnalisnya."
        )

    if len(unique_labels) == 1 and high_confidence_count >= 2:
        domain_reason = (
            "reaksi panik dan tekanan jual yang tajam" if consensus_label == "NEGATIVE"
            else "optimisme kuat dan potensi FOMO" if consensus_label == "POSITIVE"
            else "kestabilan sentimen yang jelas"
        )
        return (
            f"Ketiga model sepakat pada kelas {consensus_label} dengan confidence tinggi, "
            f"menandakan konsensus yang solid dalam konteks pasar modal. "
            f"Frasa pasar menunjukkan {domain_reason}, sehingga hasil ini dikeluarkan dengan nada yang tegas dan dapat dipercaya."
        )

    if (
        indobertweet["sentiment"] == indobert["sentiment"] and xlmr["sentiment"] != indobertweet["sentiment"]
    ) or (
        indobertweet["sentiment"] == xlmr["sentiment"] and indobert["sentiment"] != indobertweet["sentiment"]
    ):
        partner = "IndoBERT" if indobertweet["sentiment"] == indobert["sentiment"] else "XLM-RoBERTa"
        missed = xlmr["sentiment"] if partner == "IndoBERT" else indobert["sentiment"]
        return (
            f"IndoBERTweet selaras dengan {partner} dan menunjukkan kekuatan pemahaman konteks informal. "
            f"Model ini lebih sensitif terhadap slang dan nuansa pasar media sosial, yang sering kali hilang oleh model multibahasa atau formal seperti XLM-RoBERTa. "
            f"Sementara itu, {missed} menunjukkan bahwa ada perbedaan gaya bahasa yang harus diwaspadai."
        )

    if len(unique_labels) == 3 or average_confidence < 0.60:
        return (
            f"Ketiga arsitektur menghasilkan sentimen berbeda (IndoBERT={indobert['sentiment']} {round(indobert['confidence']*100)}%, "
            f"IndoBERTweet={indobertweet['sentiment']} {round(indobertweet['confidence']*100)}%, "
            f"XLM-RoBERTa={xlmr['sentiment']} {round(xlmr['confidence']*100)}%). "
            "Ini menunjukkan ambiguitas tinggi, kemungkinan karena noise semantik, konteks berlapis, atau gaya bahasa yang ironis/sarkastik. "
            "Tafsir akhir harus diperlakukan dengan kehati-hatian."
        )

    return (
        f"Model ensemble memilih {consensus_label} dengan confidence {round(average_confidence * 100, 1)}%. "
        "IndoBERTweet membantu menangkap nuansa informal, sedangkan IndoBERT dan XLM-RoBERTa menyeimbangkan hasil dengan konteks pasar yang lebih luas."
    )


def get_transformer_scores(text: str) -> dict[str, dict]:
    scores = {}
    for model_name in ("indobert", "indoroberta", "xlmr"):
        if model_name in registry.transformers:
            scores[model_name] = predict_transformer(text, model_name)
    return scores


def get_traditional_scores(text: str) -> dict[str, dict]:
    scores = {}
    for model_name in ("svm", "random_forest", "xgboost", "logistic_regression"):
        if model_name in registry.traditional_models:
            scores[model_name] = predict_traditional(text, model_name)
    return scores


def generate_xai_explanation_traditional(
    models_scores: dict,
    consensus: str,
    final_confidence: float,
) -> str:
    normalized = {
        model: {
            "sentiment": str(payload.get("label", "UNKNOWN")).upper(),
            "confidence": float(payload.get("confidence", 0.0)),
        }
        for model, payload in models_scores.items()
    }

    labels = [v["sentiment"] for v in normalized.values()]
    unique_labels = set(labels)
    consensus_label = str(consensus).upper()
    average_confidence = float(final_confidence)
    high_confidence_count = sum(1 for v in normalized.values() if v["confidence"] >= 0.70)

    svm  = normalized.get("svm",  {"sentiment": "UNKNOWN", "confidence": 0.0})
    rf   = normalized.get("random_forest", {"sentiment": "UNKNOWN", "confidence": 0.0})
    xgb  = normalized.get("xgboost", {"sentiment": "UNKNOWN", "confidence": 0.0})
    lr   = normalized.get("logistic_regression", {"sentiment": "UNKNOWN", "confidence": 0.0})

    model_summary = (
        f"SVM={svm['sentiment']} {round(svm['confidence']*100)}%, "
        f"RF={rf['sentiment']} {round(rf['confidence']*100)}%, "
        f"XGBoost={xgb['sentiment']} {round(xgb['confidence']*100)}%, "
        f"LR={lr['sentiment']} {round(lr['confidence']*100)}%"
    )

    if len(unique_labels) == 1 and high_confidence_count >= 3:
        domain_reason = (
            "tekanan jual dan sentimen negatif pasar yang kuat" if consensus_label == "NEGATIVE"
            else "momentum beli dan optimisme investor yang tinggi" if consensus_label == "POSITIVE"
            else "stabilitas sentimen tanpa sinyal direksi yang dominan"
        )
        return (
            f"Seluruh model traditional (SVM, Random Forest, XGBoost, Logistic Regression) "
            f"sepakat pada kelas {consensus_label} dengan confidence rata-rata {round(average_confidence * 100, 1)}%. "
            f"Konsensus ini menandakan {domain_reason}. "
            f"Fitur TF-IDF menangkap sinyal leksikal yang konsisten di seluruh arsitektur."
        )

    if len(unique_labels) == 1 and high_confidence_count < 3:
        return (
            f"Model traditional sepakat pada {consensus_label}, namun confidence rata-rata "
            f"{round(average_confidence * 100, 1)}% menunjukkan sinyal yang moderat. "
            f"Detail: {model_summary}. "
            "Kemungkinan terdapat ambiguitas leksikal atau campuran sentimen dalam teks."
        )

    if average_confidence < 0.55 or len(unique_labels) >= 3:
        return (
            f"Model traditional menghasilkan prediksi yang divergen ({model_summary}). "
            "Confidence rendah dan ketidaksepakatan antar model menunjukkan teks memiliki sinyal sentimen yang lemah atau ambigu. "
            "Disarankan untuk mempertimbangkan konteks pasar tambahan sebelum mengambil keputusan."
        )

    majority = [l for l in labels if labels.count(l) >= 2]
    if majority:
        dissenter_items = [(m, v) for m, v in normalized.items() if v["sentiment"] != majority[0]]
        dissenter_str = ", ".join(
            f"{m.replace('_', ' ').title()} ({v['sentiment']} {round(v['confidence']*100)}%)"
            for m, v in dissenter_items
        )
        return (
            f"Mayoritas model traditional sepakat pada {consensus_label} "
            f"(confidence ensemble {round(average_confidence * 100, 1)}%). "
            f"Model yang berbeda pendapat: {dissenter_str}. "
            "Perbedaan ini wajar pada teks yang mengandung campuran frasa formal dan informal pasar modal."
        )

    return (
        f"Ensemble traditional memilih {consensus_label} dengan confidence {round(average_confidence * 100, 1)}% "
        f"melalui soft voting ({model_summary}). "
        "Representasi TF-IDF menangkap bobot token finansial secara statistik."
    )


def run_predict(text: str, model: str) -> dict:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")
    if model not in VALID_MODELS:
        raise HTTPException(
            status_code = 400,
            detail      = f"Model '{model}' tidak dikenal. Pilihan: {', '.join(VALID_MODELS)}"
        )

    if model in ("svm", "random_forest", "xgboost", "logistic_regression", "traditional"):
        return predict_traditional(text, model)
    elif model == "ensemble":
        result = predict_ensemble(text)
        transformer_scores = get_transformer_scores(text)
        if len(transformer_scores) == 3:
            result["xai_explanation"] = generate_xai_explanation(
                models_scores    = transformer_scores,
                slang_density    = estimate_slang_density(text),
                consensus        = result["label"],
                final_confidence = result["confidence"],
            )
        else:
            result["xai_explanation"] = (
                "XAI explanation unavailable karena data transformer tidak lengkap. "
                "Pastikan IndoBERT, IndoBERTweet, dan XLM-RoBERTa sudah dimuat."
            )
        return result
    elif model == "ensemble_all":
        result = predict_ensemble_traditional(text)
        traditional_scores = get_traditional_scores(text)
        result["xai_explanation"] = generate_xai_explanation_traditional(
            models_scores    = traditional_scores,
            consensus        = result["label"],
            final_confidence = result["confidence"],
        )
        return result
    else:
        return predict_transformer(text, model)

SLANG_DICTIONARY = [
    {"term": "Cuan", "meaning": "Keuntungan atau profit dari investasi/trading saham.", "example": "Saham BBCA naik 5% hari ini, cuan terus!", "category": "Profit & Rugi"},
    {"term": "Nyangkut", "meaning": "Kondisi saham yang sudah dibeli namun harganya turun terus sehingga susah dijual tanpa rugi besar.", "example": "Beli GOTO di harga 200 sekarang tinggal 80, nyangkut parah nih.", "category": "Profit & Rugi"},
    {"term": "Cut Loss", "meaning": "Menjual saham yang sedang merugi untuk membatasi kerugian lebih lanjut.", "example": "Daripada nyangkut makin dalam, mending cut loss sekarang.", "category": "Strategi"},
    {"term": "FOMO", "meaning": "Fear Of Missing Out – rasa takut ketinggalan momentum beli sehingga investor terburu-buru masuk tanpa analisis.", "example": "Jangan FOMO beli saham yang sudah naik 50% dalam sehari.", "category": "Sentimen"},
    {"term": "Serok", "meaning": "Membeli saham dalam jumlah besar sekaligus, biasanya saat harga dianggap murah.", "example": "Pas IHSG drop 3%, gue langsung serok BBRI sebanyak mungkin.", "category": "Aksi Pasar"},
    {"term": "ARA", "meaning": "Auto Reject Atas – batas maksimal kenaikan harga saham dalam sehari yang ditetapkan BEI.", "example": "Saham HAKA kena ARA tiga hari berturut-turut, euforia banget!", "category": "Mekanisme Bursa"},
    {"term": "ARB", "meaning": "Auto Reject Bawah – batas maksimal penurunan harga saham dalam sehari yang ditetapkan BEI.", "example": "Saham itu kena ARB lagi, hati-hati bisa terjebak di sana.", "category": "Mekanisme Bursa"},
    {"term": "Gorengan", "meaning": "Saham dengan harga murah dan likuiditas rendah yang mudah dimanipulasi untuk dipompa naik lalu dijual.", "example": "Jangan tergiur saham gorengan, risikonya tinggi banget.", "category": "Tipe Saham"},
    {"term": "Bandar", "meaning": "Pihak (biasanya institusi atau individu kaya) yang diduga memanipulasi pergerakan harga saham tertentu.", "example": "Harga tiba-tiba naik drastis, kayaknya bandar lagi masuk nih.", "category": "Pelaku Pasar"},
    {"term": "Mantul", "meaning": "Singkatan dari 'mantap betul' atau berarti harga saham yang memantul naik setelah menyentuh support.", "example": "Setelah turun ke support 1500, TLKM mantul naik lagi.", "category": "Kondisi Pasar"},
    {"term": "Ngacir", "meaning": "Harga saham yang turun dengan sangat cepat.", "example": "Berita negatif keluar, sahamnya langsung ngacir ke bawah.", "category": "Kondisi Pasar"},
    {"term": "Porto", "meaning": "Singkatan dari portofolio, kumpulan aset investasi yang dimiliki seseorang.", "example": "Porto gue lagi merah semua karena IHSG turun.", "category": "Investasi"},
    {"term": "Averaging Down", "meaning": "Strategi membeli lebih banyak saham saat harga turun untuk meratakan harga beli rata-rata.", "example": "Saham ini fundamentalnya bagus, gue averaging down dulu.", "category": "Strategi"},
    {"term": "Averaging Up", "meaning": "Menambah posisi saham saat harganya sedang naik untuk memaksimalkan keuntungan.", "example": "Tren naik kuat, gue averaging up terus selagi momentum.", "category": "Strategi"},
    {"term": "Bullish", "meaning": "Kondisi pasar atau pandangan optimis bahwa harga akan terus naik.", "example": "Analis besar pada bullish terhadap sektor perbankan.", "category": "Kondisi Pasar"},
    {"term": "Bearish", "meaning": "Kondisi pasar atau pandangan pesimis bahwa harga akan terus turun.", "example": "Sentimen global bearish bikin IHSG ikut melemah.", "category": "Kondisi Pasar"},
    {"term": "Sideway", "meaning": "Pergerakan harga yang datar dan tidak signifikan, tidak naik maupun turun jauh.", "example": "Saham ini udah sideway tiga bulan, bosan nunggunya.", "category": "Kondisi Pasar"},
    {"term": "Breakout", "meaning": "Harga menembus level resistance yang sebelumnya sulit ditembus.", "example": "BBCA baru breakout dari resistance 9500, sinyal beli kuat.", "category": "Analisis Teknikal"},
    {"term": "Support", "meaning": "Level harga di mana tekanan beli cukup kuat untuk menghentikan penurunan harga.", "example": "IHSG menguji support 7000, kalau jebol bisa turun lebih dalam.", "category": "Analisis Teknikal"},
    {"term": "Resistance", "meaning": "Level harga di mana tekanan jual cukup kuat untuk menghentikan kenaikan.", "example": "Resistance kuat di 8000, harga belum berhasil tembusnya.", "category": "Analisis Teknikal"},
    {"term": "Scalping", "meaning": "Strategi trading jangka sangat pendek (menit hingga jam) untuk mendapat profit kecil tapi sering.", "example": "Gue scalping BBRI pagi ini, in-out tiga kali dapat cuan tipis-tipis.", "category": "Strategi"},
    {"term": "Swing Trade", "meaning": "Trading dengan horizon menengah, biasanya beberapa hari hingga beberapa minggu.", "example": "Ini swing trade, targetnya naik 10% dalam dua minggu.", "category": "Strategi"},
    {"term": "All In", "meaning": "Menggunakan seluruh modal yang tersedia untuk membeli satu saham.", "example": "Gue all in TLKM karena yakin banget sama prospeknya.", "category": "Strategi"},
    {"term": "Panic Selling", "meaning": "Aksi jual massal karena kepanikan, biasanya saat ada berita negatif besar.", "example": "Pas corona 2020 banyak yang panic selling, padahal harusnya beli.", "category": "Aksi Pasar"},
    {"term": "Profit Taking", "meaning": "Aksi jual saham untuk merealisasikan keuntungan yang sudah didapat.", "example": "Saham naik 20%, banyak investor profit taking sehingga harga koreksi.", "category": "Aksi Pasar"},
    {"term": "Blue Chip", "meaning": "Saham perusahaan besar, terkenal, dan stabil secara finansial.", "example": "Buat investasi jangka panjang, pilih blue chip seperti BBCA atau TLKM.", "category": "Tipe Saham"},
    {"term": "Penny Stock", "meaning": "Saham dengan harga sangat murah, biasanya di bawah Rp 200 per lembar.", "example": "Penny stock memang murah, tapi likuiditasnya rendah dan risikonya tinggi.", "category": "Tipe Saham"},
    {"term": "Saham Tidur", "meaning": "Saham yang sangat jarang diperdagangkan sehingga harganya tidak bergerak.", "example": "Jangan beli saham tidur, susah jualnya nanti.", "category": "Tipe Saham"},
    {"term": "Lot", "meaning": "Satuan perdagangan saham di BEI, di mana 1 lot = 100 lembar saham.", "example": "Gue beli 10 lot BBRI, berarti 1000 lembar saham.", "category": "Mekanisme Bursa"},
    {"term": "IHSG", "meaning": "Indeks Harga Saham Gabungan – indeks utama yang mencerminkan kinerja seluruh saham di Bursa Efek Indonesia.", "example": "IHSG turun 2% hari ini karena sentimen global memburuk.", "category": "Indeks"},
    {"term": "IPO", "meaning": "Initial Public Offering – penawaran saham perdana perusahaan kepada publik melalui bursa.", "example": "IPO saham GOTO kemarin oversubscribed berkali-kali lipat.", "category": "Aksi Korporasi"},
    {"term": "Dividen", "meaning": "Pembagian keuntungan perusahaan kepada para pemegang saham.", "example": "BBCA bagi dividen Rp 340 per saham, lumayan cuan pasif.", "category": "Investasi"},
    {"term": "Pump and Dump", "meaning": "Manipulasi pasar: harga saham dipompa naik oleh sekelompok pihak lalu dijual saat sudah tinggi, merugikan investor lain.", "example": "Hati-hati dengan saham yang tiba-tiba naik 100% tanpa alasan, bisa jadi pump and dump.", "category": "Manipulasi Pasar"},
    {"term": "Margin Call", "meaning": "Peringatan dari broker untuk menambah dana jaminan karena posisi margin sudah mencapai batas risiko.", "example": "Harga turun terus, broker kasih margin call, terpaksa jual saham.", "category": "Risiko"},
    {"term": "TP (Take Profit)", "meaning": "Target harga jual di mana investor berencana merealisasikan keuntungan.", "example": "TP gue di harga 5000, kalau sudah nyentuh sana langsung jual.", "category": "Strategi"},
    {"term": "SL (Stop Loss)", "meaning": "Batas harga jual otomatis untuk membatasi kerugian jika harga turun terlalu jauh.", "example": "Pasang SL di 10% bawah harga beli biar tidak rugi terlalu banyak.", "category": "Strategi"},
    {"term": "Emot", "meaning": "Trading secara emosional, mengambil keputusan beli/jual berdasarkan perasaan bukan analisis.", "example": "Jangan emot saat porto merah, stick to the plan!", "category": "Sentimen"},
    {"term": "YOLO", "meaning": "You Only Live Once – ungkapan untuk keputusan trading berisiko tinggi dengan modal besar.", "example": "YOLO all in saham gorengan itu berbahaya banget bro.", "category": "Sentimen"},
    {"term": "Diamond Hands", "meaning": "Istilah untuk investor yang teguh memegang saham meski harga turun jauh.", "example": "Gue diamond hands GOTO dari IPO, yakin bakal balik.", "category": "Strategi"},
    {"term": "Nabung Saham", "meaning": "Strategi membeli saham secara rutin dan berkala dalam jumlah kecil terlepas dari kondisi pasar.", "example": "Gue nabung saham BBCA tiap bulan sejak 5 tahun lalu, sekarang hasilnya lumayan.", "category": "Strategi"},
    {"term": "Delisting", "meaning": "Pencabutan saham dari daftar perdagangan Bursa Efek Indonesia.", "example": "Perusahaan itu terancam delisting karena tidak memenuhi syarat BEI.", "category": "Mekanisme Bursa"},
    {"term": "Right Issue", "meaning": "Penawaran saham baru kepada pemegang saham lama dengan harga khusus (biasanya lebih murah).", "example": "BRIS right issue untuk ekspansi bisnis, dilusi tapi fundamentalnya masih kuat.", "category": "Aksi Korporasi"},
    {"term": "Stock Split", "meaning": "Pemecahan saham menjadi lebih banyak lembar dengan harga lebih rendah; nilai total tidak berubah.", "example": "Setelah stock split 1:5, harga BBCA jadi lebih terjangkau untuk ritel.", "category": "Aksi Korporasi"},
    {"term": "Volatilitas", "meaning": "Tingkat fluktuasi atau perubahan harga saham dalam periode tertentu.", "example": "Saham small cap punya volatilitas tinggi, bisa naik atau turun tajam.", "category": "Analisis"},
    {"term": "Likuiditas", "meaning": "Kemudahan membeli atau menjual saham tanpa mempengaruhi harga secara signifikan.", "example": "Blue chip punya likuiditas tinggi, mudah beli dan jual kapan saja.", "category": "Analisis"},
]

#endpoint

@app.get("/", tags=["Health"])
def root():
    return {
        "status"        : "running",
        "loaded_models" : registry.loaded_models,
        "device"        : str(registry.device),
        "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

@app.get("/models", tags=["Info"])
def list_models():
    """List semua model yang tersedia beserta status loaded-nya."""
    traditional = {}
    for name, path in TRADITIONAL_MODEL_PATHS.items():
        traditional[name] = {
            "status" : "loaded" if name in registry.traditional_models else "not loaded",
            "path"   : path,
            "type"   : "TF-IDF + Classical ML",
            "speed"  : "< 5ms",
        }

    transformers = {}
    for name, cfg in TRANSFORMER_CONFIGS.items():
        transformers[name] = {
            "status"     : "loaded" if name in registry.transformers else "not loaded",
            "checkpoint" : cfg["checkpoint"],
            "weights"    : cfg["weights"],
            "type"       : "Transformer",
            "speed"      : "~100ms (CPU) / ~20ms (GPU)",
        }

    return {
        "device"       : str(registry.device),
        "loaded_models": registry.loaded_models,
        "traditional"  : traditional,
        "transformers" : transformers,
        "ensemble"     : {
            "ensemble"     : "Soft voting semua transformer",
            "ensemble_all" : "Soft voting 4 model traditional (SVM + RF + XGBoost + LR)",
        },
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    start   = time.time()
    result  = run_predict(req.text, req.model)
    elapsed = round((time.time() - start) * 1000, 2)

    return PredictResponse(text=req.text, latency_ms=elapsed, **result)

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    if len(req.texts) > 50:
        raise HTTPException(status_code=400, detail="Maksimal 50 teks per request")

    start   = time.time()
    results = []

    for text in req.texts:
        t0     = time.time()
        result = run_predict(text, req.model)
        results.append(PredictResponse(
            text       = text,
            latency_ms = round((time.time() - t0) * 1000, 2),
            **result,
        ))

    return BatchPredictResponse(
        results    = results,
        total      = len(results),
        model_used = req.model,
        latency_ms = round((time.time() - start) * 1000, 2),
    )


@app.get("/slang/search", tags=["Slang Dictionary"])
def search_slang(q: str = ""):
    if not q.strip():
        return {"results": SLANG_DICTIONARY, "total": len(SLANG_DICTIONARY), "query": ""}
    q_lower = q.strip().lower()
    results = [
        entry for entry in SLANG_DICTIONARY
        if q_lower in entry["term"].lower()
        or q_lower in entry["meaning"].lower()
        or q_lower in entry["category"].lower()
    ]
    return {"results": results, "total": len(results), "query": q}


@app.post("/predict/multi", tags=["Inference"])
def predict_multi(req: MultiPredictRequest):
    start = time.time()
    results = {}

    for model_name in req.models:
        t0 = time.time()
        try:
            res = run_predict(req.text, model_name)
            latency = round((time.time() - t0) * 1000, 2)
            results[model_name] = {
                "text": req.text,
                "latency_ms": latency,
                **res,
            }
        except HTTPException as e:
            results[model_name] = {"error": str(e.detail)}

    total_elapsed = round((time.time() - start) * 1000, 2)
    return {"results": results, "model_count": len(req.models), "latency_ms": total_elapsed}

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="static-assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(_full_path: str = ""):
        return FileResponse(str(FRONTEND_DIST / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
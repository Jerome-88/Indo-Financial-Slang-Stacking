
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "data_path"   : str(BASE_DIR / "dataset_final_clean_no_selective.csv"),
    "text_col"    : "text_cleaned",
    "label_col"   : "sentiment",
    "num_labels"  : 3,

    #Models
    "models": {
        "indobert": {
            "checkpoint" : "indobenchmark/indobert-base-p2",
            "max_len"    : 128,
        },
        "indoroberta": {
            "checkpoint" : "indolem/indobertweet-base-uncased",
            "max_len"    : 128,
        },
        "xlmr": {
            "checkpoint" : "xlm-roberta-base",
            "max_len"    : 128,
        },
    },

    #Train
    "n_folds"             : 5,
    "epochs"              : 4,
    "batch_size"          : 32,
    "learning_rate"       : 2e-5,
    "weight_decay"        : 0.01,
    "warmup_ratio"        : 0.1,
    "early_stop_patience" : 2,
    "seed"                : 42,

    #Output
    "experiment_name" : None,
    "output_dir"  : None,
    "model_dir"   : None,
}

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_MAP   = {"Negative": 0, "Neutral": 1, "Positive": 2}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU — menggunakan CPU (akan lebih lambat)")
    return device

def compute_class_weights(labels: np.ndarray, device) -> torch.Tensor:
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)

def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class PreTokenizedDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        print(f"Pre-tokenizing {len(texts)} samples...", end=" ", flush=True)
        enc = tokenizer(
            texts,
            max_length     = max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)
        print(f"done. Shape: {self.input_ids.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"      : self.input_ids[idx],
            "attention_mask" : self.attention_mask[idx],
            "label"          : self.labels[idx],
        }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    class_weights, use_amp) -> float:
    model.train()
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss(weight=class_weights)

    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        with autocast(enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device, class_weights, use_amp,
             return_logits=False):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    all_logits = []
    criterion  = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            with autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs.logits.float(), dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    if return_logits:
        return avg_loss, acc, macro_f1, np.array(all_logits)
    return avg_loss, acc, macro_f1


def train_model_kfold(model_name, checkpoint, max_len, texts, labels,
                      device, config, use_amp) -> np.ndarray:

    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Max len    : {max_len} | Batch size: {config['batch_size']} | FP16: {use_amp}")
    print(f"{'='*70}")

    skf = StratifiedKFold(
        n_splits     = config["n_folds"],
        shuffle      = True,
        random_state = config["seed"],
    )

    oof_logits  = np.zeros((len(texts), config["num_labels"]))
    fold_scores = []

    # Load tokenizer sekali, reused di semua fold
    print(f"\n   Loading tokenizer: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/{config['n_folds']} ---")

        train_texts  = [texts[i] for i in train_idx]
        val_texts    = [texts[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels   = labels[val_idx]

        print(f"   Train: {len(train_texts)} | Val: {len(val_texts)}")

        #Pre-tokenisasi
        train_ds = PreTokenizedDataset(train_texts, train_labels.tolist(), tokenizer, max_len)
        val_ds   = PreTokenizedDataset(val_texts,   val_labels.tolist(),   tokenizer, max_len)

        pin = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                                  shuffle=True,  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"] * 2,
                                  shuffle=False, num_workers=0, pin_memory=pin)

        # Load model fresh per fold
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=config["num_labels"]
        ).to(device)

        class_weights = compute_class_weights(train_labels, device)

        optimizer    = AdamW(model.parameters(),
                             lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])
        total_steps  = len(train_loader) * config["epochs"]
        warmup_steps = int(total_steps * config["warmup_ratio"])
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
        )
        scaler = GradScaler(enabled=use_amp)

        best_f1      = 0.0
        best_logits  = None
        patience_ctr = 0
        best_path    = f"{config['model_dir']}/{model_name}_fold{fold_num}.pt"
        os.makedirs(config["model_dir"], exist_ok=True)

        for epoch in range(config["epochs"]):
            t_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     scaler, device, class_weights, use_amp)
            v_loss, v_acc, v_f1, v_logits = evaluate(
                model, val_loader, device, class_weights, use_amp, return_logits=True
            )

            print(
                f"Epoch {epoch+1}/{config['epochs']} | "
                f"Train Loss: {t_loss:.4f} | "
                f"Val Loss: {v_loss:.4f} | "
                f"Val Acc: {v_acc:.4f} | "
                f"Val F1: {v_f1:.4f}"
            )

            if v_f1 > best_f1:
                best_f1      = v_f1
                best_logits  = v_logits
                patience_ctr = 0
                torch.save(model.state_dict(), best_path)
                print(f"Best F1: {best_f1:.4f} — saved to {best_path}")
            else:
                patience_ctr += 1
                print(f"No improvement. Patience: {patience_ctr}/{config['early_stop_patience']}")
                if patience_ctr >= config["early_stop_patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        oof_logits[val_idx] = best_logits
        fold_scores.append(best_f1)
        print(f"   Fold {fold_num} Best F1: {best_f1:.4f}")

        del model, train_ds, val_ds, train_loader, val_loader
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} — CV Summary:")
    for i, s in enumerate(fold_scores):
        print(f"   Fold {i+1}: F1 = {s:.4f}")
    print(f"   Mean F1 : {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*70}")

    return oof_logits

#meta learner

def train_meta_learner(oof_logits_all, labels, config):
    print(f"\n{'='*70}")
    print("TIER-1: META-LEARNER (Logistic Regression)")
    print(f"{'='*70}")
    print(f"Feature matrix shape : {oof_logits_all.shape}")

    meta = LogisticRegression(
        max_iter     = 1000,
        C            = 1.0,
        class_weight = "balanced",
        random_state = config["seed"],
        multi_class  = "multinomial",
        solver       = "lbfgs",
    )
    meta.fit(oof_logits_all, labels)

    preds   = meta.predict(oof_logits_all)
    oof_acc = accuracy_score(labels, preds)
    oof_f1  = f1_score(labels, preds, average="macro")

    print(f"\nOOF Accuracy : {oof_acc:.4f}")
    print(f"OOF Macro F1 : {oof_f1:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=LABEL_NAMES)}")

    return meta

#evaluation

def full_evaluation(meta, oof_logits_all, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    preds  = meta.predict(oof_logits_all)
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average="macro")
    cm     = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, output_dict=True
    )

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Macro F1 : {f1:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=LABEL_NAMES)}")
    print(f"   Confusion Matrix:\n   {cm}")

    save_json({
        "timestamp"             : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_folds"               : CONFIG["n_folds"],
        "accuracy"              : round(acc, 4),
        "macro_f1"              : round(f1, 4),
        "confusion_matrix"      : cm.tolist(),
        "classification_report" : report,
    }, f"{output_dir}/final_results.json")

    print(f"\nSaved in {output_dir}/final_results.json")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CONFIG["experiment_name"] = f"run_{timestamp}"
    CONFIG["output_dir"] = str(BASE_DIR / f"results_{CONFIG['experiment_name']}")
    CONFIG["model_dir"]  = str(BASE_DIR / f"saved_models_{CONFIG['experiment_name']}")


    print(f"Starting new experiment: {CONFIG['experiment_name']}")
    print(f"Output folder : {CONFIG['output_dir']}/")
    print(f"Model folder  : {CONFIG['model_dir']}/\n")

    set_seed(CONFIG["seed"])
    device  = get_device()
    use_amp = device.type == "cuda"
    print(f"   Mixed Precision (FP16): {'ENABLED' if use_amp else 'DISABLED (CPU)'}")

    #load data
    df = pd.read_csv(CONFIG["data_path"])
    print(f"{len(df)} records")
    print(f"Columns  : {list(df.columns)}")

    texts  = df[CONFIG["text_col"]].fillna("").tolist()
    labels = np.array([LABEL_MAP[s] for s in df[CONFIG["label_col"]]])

    dist = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"   Label dist: { {LABEL_NAMES[k]: v for k, v in dist.items()} }")

    # Buat folder output
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)



    all_oof = []
    for model_name, cfg in CONFIG["models"].items():
        oof = train_model_kfold(
            model_name = model_name,
            checkpoint = cfg["checkpoint"],
            max_len    = cfg["max_len"],
            texts      = texts,
            labels     = labels,
            device     = device,
            config     = CONFIG,
            use_amp    = use_amp,
        )
        all_oof.append(oof)
        np.save(f"{CONFIG['output_dir']}/oof_{model_name}.npy", oof)
        print(f"OOF saved → {CONFIG['output_dir']}/oof_{model_name}.npy")

    # concat oof
    oof_combined = np.concatenate(all_oof, axis=1)
    print(f"Shape: {oof_combined.shape}  ← (samples, 3 models × 3 classes)")
    np.save(f"{CONFIG['output_dir']}/oof_combined.npy", oof_combined)
    print(f"Saved → {CONFIG['output_dir']}/oof_combined.npy")

    #meta learner
    print("\nTraining Tier-1 Meta-Learner")
    meta = train_meta_learner(oof_combined, labels, CONFIG)

    # final evaluation
    print("\nFinal Evaluation")
    results = full_evaluation(meta, oof_combined, labels, CONFIG["output_dir"])

    #Summary
    print("\n" + "="*70)
    print("summary")
    print("="*70)
    print(f"""
  Experiment    : {CONFIG['experiment_name']}
  Dataset       : {len(df)} records
  Strategy      : 5-Fold Stratified CV + OOF Stacking
  Models        : IndoBERT + IndoBERTweet + XLM-RoBERTa
  Accuracy      : {results['accuracy']:.4f}
  Macro F1      : {results['macro_f1']:.4f}
""")

if __name__ == "__main__":
    main()
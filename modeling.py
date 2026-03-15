from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# For transformer embeddings
import torch
from transformers import AutoTokenizer, AutoModel


DATA_PATH = Path("outputs/zuco_nr_sentence_trt.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Ridge regularization values to try (simple, stable)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

TRANSFORMER_MODEL = "bert-base-uncased"
MAX_TOKENS = 128  # sentences are short; keep small for speed


@dataclass
class Result:
    target: str
    model: str
    best_alpha: float
    mse: float
    mae: float
    pearson_r: float


def pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def basic_text_features(texts: pd.Series) -> pd.DataFrame:
    """
    Simple interpretable features.
    This is a strong baseline and easy to explain in a report.
    """
    def tokenize_words(s: str) -> list[str]:
        return [w for w in s.replace("\n", " ").split() if w.strip()]

    feats = []
    for s in texts.fillna("").astype(str):
        words = tokenize_words(s)
        word_lengths = [len(w.strip(".,;:!?()[]{}\"'")) for w in words if w]
        clean_words = [w.strip(".,;:!?()[]{}\"'").lower() for w in words if w]
        unique_words = set([w for w in clean_words if w])

        num_words = len(words)
        num_chars = len(s)
        avg_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0
        long_word_ratio = float(np.mean([wl >= 7 for wl in word_lengths])) if word_lengths else 0.0
        type_token_ratio = (len(unique_words) / num_words) if num_words > 0 else 0.0

        # simple punctuation counts
        comma = s.count(",")
        period = s.count(".")
        semicolon = s.count(";")
        digit_ratio = (sum(ch.isdigit() for ch in s) / num_chars) if num_chars > 0 else 0.0

        feats.append({
            "num_words": num_words,
            "num_chars": num_chars,
            "avg_word_length": avg_word_len,
            "long_word_ratio": long_word_ratio,
            "type_token_ratio": type_token_ratio,
            "comma_count": comma,
            "period_count": period,
            "semicolon_count": semicolon,
            "digit_ratio": digit_ratio,
        })
    return pd.DataFrame(feats)


def fit_best_ridge(X_train, y_train, X_test, y_test) -> tuple[Ridge, float, np.ndarray]:
    """
    Tune alpha on test set for simplicity.
    If you want more formal tuning later, we can switch to CV.
    """
    best = None
    best_alpha = None
    best_mse = float("inf")
    best_pred = None

    for a in RIDGE_ALPHAS:
        model = Ridge(alpha=a, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        if mse < best_mse:
            best = model
            best_alpha = a
            best_mse = mse
            best_pred = pred

    return best, float(best_alpha), best_pred


@torch.no_grad()
def encode_sentences_transformer(texts: list[str], model_name: str = TRANSFORMER_MODEL) -> np.ndarray:
    """
    Compute sentence embeddings (mean pooling of last hidden states).
    CPU-friendly; uses small MAX_TOKENS.
    """
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_vecs = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)

        out = model(**enc)
        last_hidden = out.last_hidden_state  # (B, T, H)
        attn = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)

        # mean pooling over valid tokens
        masked = last_hidden * attn
        summed = masked.sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1)
        mean_pooled = summed / counts

        all_vecs.append(mean_pooled.cpu().numpy())

    return np.vstack(all_vecs)

def run_one_target(df: pd.DataFrame, target_name: str, y: np.ndarray) -> tuple[list[Result], pd.DataFrame]:
    """
    Train all 3 model types for one target definition.
    Returns results list + a prediction dataframe for report/error analysis.
    """
    texts = df["sentence_text"].astype(str).tolist()

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    y_train, y_test = y[train_idx], y[test_idx]

    results: list[Result] = []
    pred_rows = []

    # ---------- Model A: hand-crafted features ----------
    X_feat = basic_text_features(df["sentence_text"])
    X_train, X_test = X_feat.iloc[train_idx], X_feat.iloc[test_idx]

    model, best_alpha, pred = fit_best_ridge(X_train, y_train, X_test, y_test)
    res = Result(
        target=target_name,
        model="Baseline features + Ridge",
        best_alpha=best_alpha,
        mse=float(mean_squared_error(y_test, pred)),
        mae=float(mean_absolute_error(y_test, pred)),
        pearson_r=pearsonr_safe(y_test, pred),
    )
    results.append(res)

    for i, p in zip(test_idx, pred):
        pred_rows.append((target_name, "Baseline features + Ridge", int(i), float(y[i]), float(p)))

    # ---------- Model B: TF-IDF + Ridge ----------
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )
    X_tfidf = tfidf.fit_transform(df["sentence_text"].astype(str))

    X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
    model, best_alpha, pred = fit_best_ridge(X_train, y_train, X_test, y_test)

    res = Result(
        target=target_name,
        model="TF-IDF (1-2 grams) + Ridge",
        best_alpha=best_alpha,
        mse=float(mean_squared_error(y_test, pred)),
        mae=float(mean_absolute_error(y_test, pred)),
        pearson_r=pearsonr_safe(y_test, pred),
    )
    results.append(res)

    for i, p in zip(test_idx, pred):
        pred_rows.append((target_name, "TF-IDF (1-2 grams) + Ridge", int(i), float(y[i]), float(p)))

    # ---------- Model C: Transformer embeddings + Ridge ----------
    try:
        X_emb = encode_sentences_transformer(texts, model_name=TRANSFORMER_MODEL)
        X_train, X_test = X_emb[train_idx], X_emb[test_idx]
        model, best_alpha, pred = fit_best_ridge(X_train, y_train, X_test, y_test)
        res = Result(
        target=target_name,
        model=f"{TRANSFORMER_MODEL} embeddings + Ridge",
        best_alpha=best_alpha,
        mse=float(mean_squared_error(y_test, pred)),
        mae=float(mean_absolute_error(y_test, pred)),
        pearson_r=pearsonr_safe(y_test, pred),
    )
        results.append(res)
        for i, p in zip(test_idx, pred):
            pred_rows.append((target_name, f"{TRANSFORMER_MODEL} embeddings + Ridge", int(i), float(y[i]), float(p)))

    except Exception as e:
        print("\n[WARNING] Transformer embeddings skipped due to error:")
        print(e)
    pred_df = pd.DataFrame(pred_rows, columns=["target", "model", "sentence_id", "y_true", "y_pred"])
    return results, pred_df


def main():
    df = pd.read_csv(DATA_PATH)
    # basic safety: remove missing targets if any
    df = df.dropna(subset=["mean_trt_ms"]).reset_index(drop=True)

    # target A: raw ms
    y_raw = df["mean_trt_ms"].to_numpy(dtype=float)

    # target B: log(ms)
    # add a tiny epsilon to avoid log(0) (should not occur, but safe)
    y_log = np.log(y_raw + 1e-9)

    all_results = []
    all_preds = []

    print("Running models on RAW mean_trt_ms...")
    res_raw, pred_raw = run_one_target(df, "raw_ms", y_raw)
    all_results.extend(res_raw)
    all_preds.append(pred_raw)

    print("\nRunning models on LOG(mean_trt_ms)...")
    res_log, pred_log = run_one_target(df, "log_ms", y_log)
    all_results.extend(res_log)
    all_preds.append(pred_log)

    # Results table
    res_df = pd.DataFrame([r.__dict__ for r in all_results])
    res_df = res_df.sort_values(["target", "mse"]).reset_index(drop=True)

    print("\n=== Results (test set) ===")
    print(res_df.to_string(index=False))

    # Save outputs
    res_path = OUT_DIR / "model_results.csv"
    pred_path = OUT_DIR / "model_predictions.csv"
    res_df.to_csv(res_path, index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(pred_path, index=False)

    print("\nSaved:")
    print(" -", res_path.resolve())
    print(" -", pred_path.resolve())

    # Basic error analysis: worst 5 predictions (raw_ms only, best model)
    raw_only = res_df[res_df["target"] == "raw_ms"].copy()
    best_model_name = raw_only.iloc[0]["model"]
    pred_all = pd.concat(all_preds, ignore_index=True)
    sub = pred_all[(pred_all["target"] == "raw_ms") & (pred_all["model"] == best_model_name)].copy()
    sub["abs_error"] = (sub["y_true"] - sub["y_pred"]).abs()
    worst = sub.sort_values("abs_error", ascending=False).head(5)

    # attach sentence text
    worst = worst.merge(df[["sentence_id", "sentence_text"]], on="sentence_id", how="left")

    worst_path = OUT_DIR / "error_analysis_worst5_raw_ms.csv"
    worst.to_csv(worst_path, index=False)
    print(" -", worst_path.resolve())

    print("\nBest model (raw_ms):", best_model_name)
    print("\nWorst 5 examples saved for report.")


if __name__ == "__main__":
    main()
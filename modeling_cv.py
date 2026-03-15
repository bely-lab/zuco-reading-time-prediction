import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, List

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import pearsonr


import torch
from transformers import AutoTokenizer, AutoModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "outputs", "zuco_nr_sentence_trt.csv")  
OUT_DIR = os.path.join(BASE_DIR, "outputs_cv")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

TRANSFORMER_MODEL = "bert-base-uncased"
MAX_LEN = 128  
BATCH_SIZE = 16


def pearson_corr(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return np.nan
    r, _ = pearsonr(y_true, y_pred)
    return float(r)

def safe_log_ms(y_ms: np.ndarray) -> np.ndarray:
    # TRT is positive; add small epsilon for safety
    return np.log(y_ms + 1e-6)

def basic_text_clean(s: str) -> str:
    # Keep simple; you can expand later if needed
    return " ".join(str(s).split())

def extract_baseline_features(texts: List[str]) -> pd.DataFrame:
    # Baseline features similar to what you already used + a few robust ones
    rows = []
    for t in texts:
        t = basic_text_clean(t)
        chars = len(t)
        words = t.split()
        num_words = len(words)
        word_lengths = [len(w.strip(".,;:!?\"'()[]{}")) for w in words if len(w) > 0]
        avg_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0
        long_word_ratio = float(np.mean([wl >= 7 for wl in word_lengths])) if word_lengths else 0.0
        unique_words = len(set([w.lower() for w in words]))
        type_token_ratio = (unique_words / num_words) if num_words > 0 else 0.0

        # crude sentence split
        num_sentences = max(1, sum([t.count(x) for x in [".", "!", "?"]]))
        avg_sentence_length = (num_words / num_sentences) if num_sentences > 0 else num_words

        rows.append({
            "num_chars": chars,
            "num_words": num_words,
            "avg_word_length": avg_word_len,
            "long_word_ratio": long_word_ratio,
            "unique_words": unique_words,
            "type_token_ratio": type_token_ratio,
            "num_sentences": num_sentences,
            "avg_sentence_length": avg_sentence_length,
        })
    return pd.DataFrame(rows)

@dataclass
class CVResult:
    target: str
    model: str
    alpha: float
    pearson_mean: float
    pearson_std: float
    mse_mean: float
    mse_std: float
    mae_mean: float
    mae_std: float


# ---------------------------
# BERT embeddings (CPU-friendly)
# ---------------------------
@torch.no_grad()
def encode_bert_embeddings(texts: List[str]) -> np.ndarray:
    """
    Returns sentence embeddings by mean pooling last hidden state.
    """
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = AutoModel.from_pretrained(TRANSFORMER_MODEL)
    model.eval()
    model.to(device)

    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [basic_text_clean(t) for t in texts[i:i+BATCH_SIZE]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last_hidden = out.last_hidden_state  # (B, T, H)
        attn = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)

        # mean pool over tokens that are not padding
        summed = (last_hidden * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1)
        mean_pooled = summed / counts
        all_embs.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embs)


# ---------------------------
# Cross-validation runners
# ---------------------------
def cv_eval_precomputed_X(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    target_name: str,
    alphas: List[float],
    standardize: bool = True
) -> Tuple[CVResult, Dict[str, np.ndarray]]:
    """
    CV for models where X is already numeric.
    Returns best CVResult by MSE and also per-fold predictions if needed.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    best = None
    best_fold_preds = None

    for a in alphas:
        pearsons, mses, maes = [], [], []
        fold_preds = np.empty_like(y, dtype=float)

        for train_idx, test_idx in kf.split(X):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            if standardize:
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

            model = Ridge(alpha=a, random_state=RANDOM_STATE)
            model.fit(Xtr, ytr)
            yp = model.predict(Xte)

            fold_preds[test_idx] = yp
            pearsons.append(pearson_corr(yte, yp))
            mses.append(mean_squared_error(yte, yp))
            maes.append(mean_absolute_error(yte, yp))

        res = CVResult(
            target=target_name,
            model=model_name,
            alpha=a,
            pearson_mean=float(np.mean(pearsons)),
            pearson_std=float(np.std(pearsons)),
            mse_mean=float(np.mean(mses)),
            mse_std=float(np.std(mses)),
            mae_mean=float(np.mean(maes)),
            mae_std=float(np.std(maes)),
        )

        if best is None or res.mse_mean < best.mse_mean:
            best = res
            best_fold_preds = fold_preds.copy()

    return best, {"oof_pred": best_fold_preds}


def cv_eval_tfidf(
    texts: List[str],
    y: np.ndarray,
    target_name: str,
    alphas: List[float]
) -> Tuple[CVResult, Dict[str, np.ndarray]]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    best = None
    best_fold_preds = None

    for a in alphas:
        pearsons, mses, maes = [], [], []
        fold_preds = np.empty_like(y, dtype=float)

        for train_idx, test_idx in kf.split(texts):
            Xtr = [texts[i] for i in train_idx]
            Xte = [texts[i] for i in test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("ridge", Ridge(alpha=a, random_state=RANDOM_STATE))
            ])
            pipe.fit(Xtr, ytr)
            yp = pipe.predict(Xte)

            fold_preds[test_idx] = yp
            pearsons.append(pearson_corr(yte, yp))
            mses.append(mean_squared_error(yte, yp))
            maes.append(mean_absolute_error(yte, yp))

        res = CVResult(
            target=target_name,
            model="TF-IDF (1-2 grams) + Ridge",
            alpha=a,
            pearson_mean=float(np.mean(pearsons)),
            pearson_std=float(np.std(pearsons)),
            mse_mean=float(np.mean(mses)),
            mse_std=float(np.std(mses)),
            mae_mean=float(np.mean(maes)),
            mae_std=float(np.std(maes)),
        )

        if best is None or res.mse_mean < best.mse_mean:
            best = res
            best_fold_preds = fold_preds.copy()

    return best, {"oof_pred": best_fold_preds}


def cv_eval_mean_baseline(y: np.ndarray, target_name: str) -> CVResult:
    # Predict global mean (a very strong sanity check baseline)
    pred = np.full_like(y, fill_value=float(np.mean(y)), dtype=float)
    return CVResult(
        target=target_name,
        model="Mean baseline",
        alpha=np.nan,
        pearson_mean=pearson_corr(y, pred),
        pearson_std=0.0,
        mse_mean=float(mean_squared_error(y, pred)),
        mse_std=0.0,
        mae_mean=float(mean_absolute_error(y, pred)),
        mae_std=0.0,
    )


# ---------------------------
# Plotting
# ---------------------------
def plot_true_vs_pred(y_true, y_pred, title, path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    r = pearson_corr(y_true, y_pred)
    plt.title(f"{title}\nPearson r={r:.3f}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_residuals(y_true, y_pred, title, path):
    resid = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.7)
    plt.axhline(0.0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Pred)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_cv_bar(df_results, target, metric, path):
    """
    metric: 'pearson_mean' or 'mse_mean' etc.
    """
    sub = df_results[df_results["target"] == target].copy()
    sub = sub.sort_values(metric, ascending=(metric != "pearson_mean"))

    means = sub[metric].values
    stds = sub[metric.replace("_mean", "_std")].values
    labels = sub["model"].values

    plt.figure(figsize=(9, 4.5))
    plt.bar(range(len(labels)), means, yerr=stds)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.title(f"{target}: {metric.replace('_mean','')} (mean ± std over {N_SPLITS}-fold CV)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_top_coefficients(feature_names, coefs, path, top_k=15):
    order = np.argsort(np.abs(coefs))[::-1][:top_k]
    names = [feature_names[i] for i in order][::-1]
    vals = [coefs[i] for i in order][::-1]

    plt.figure(figsize=(7, 5))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.xlabel("Coefficient")
    plt.title("Top coefficients (Baseline Ridge)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    df = pd.read_csv(INPUT_CSV)
    df["sentence_text"] = df["sentence_text"].astype(str).map(basic_text_clean)

    y_raw = df["mean_trt_ms"].values.astype(float)
    y_log = safe_log_ms(y_raw)

    texts = df["sentence_text"].tolist()

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    all_results: List[CVResult] = []

    # Mean baseline
    all_results.append(cv_eval_mean_baseline(y_raw, "raw_ms"))
    all_results.append(cv_eval_mean_baseline(y_log, "log_ms"))

    # Baseline features
    X_base_df = extract_baseline_features(texts)
    X_base = X_base_df.values.astype(float)
    feature_names = list(X_base_df.columns)

    # Length-only baseline (just num_words)
    X_len = X_base_df[["num_words"]].values.astype(float)

    best_base_raw, oof_base_raw = cv_eval_precomputed_X(
        X_base, y_raw, "Baseline features + Ridge", "raw_ms", alphas, standardize=True
    )
    best_base_log, oof_base_log = cv_eval_precomputed_X(
        X_base, y_log, "Baseline features + Ridge", "log_ms", alphas, standardize=True
    )

    best_len_raw, oof_len_raw = cv_eval_precomputed_X(
        X_len, y_raw, "Length-only (num_words) + Ridge", "raw_ms", alphas, standardize=True
    )
    best_len_log, oof_len_log = cv_eval_precomputed_X(
        X_len, y_log, "Length-only (num_words) + Ridge", "log_ms", alphas, standardize=True
    )

    all_results += [best_len_raw, best_len_log, best_base_raw, best_base_log]

    # TF-IDF
    best_tfidf_raw, oof_tfidf_raw = cv_eval_tfidf(texts, y_raw, "raw_ms", alphas)
    best_tfidf_log, oof_tfidf_log = cv_eval_tfidf(texts, y_log, "log_ms", alphas)
    all_results += [best_tfidf_raw, best_tfidf_log]

    # BERT embeddings
    print("Encoding BERT embeddings (may take a bit on CPU)...")
    X_bert = encode_bert_embeddings(texts)

    best_bert_raw, oof_bert_raw = cv_eval_precomputed_X(
        X_bert, y_raw, "bert-base-uncased embeddings + Ridge", "raw_ms", alphas, standardize=True
    )
    best_bert_log, oof_bert_log = cv_eval_precomputed_X(
        X_bert, y_log, "bert-base-uncased embeddings + Ridge", "log_ms", alphas, standardize=True
    )
    all_results += [best_bert_raw, best_bert_log]

    # Save CV results
    res_df = pd.DataFrame([r.__dict__ for r in all_results])
    res_path = os.path.join(OUT_DIR, "cv_results.csv")
    res_df.to_csv(res_path, index=False)

    # Choose best model for raw_ms by MSE
    raw_df = res_df[res_df["target"] == "raw_ms"].copy()
    best_row = raw_df.sort_values("mse_mean").iloc[0]
    best_model_name = best_row["model"]
    print("\nBest CV model (raw_ms) by MSE:", best_model_name)

    # Create OOF prediction table (for plots + error analysis)
    oof_pred_map = {
        ("raw_ms", "Baseline features + Ridge"): oof_base_raw["oof_pred"],
        ("log_ms", "Baseline features + Ridge"): oof_base_log["oof_pred"],
        ("raw_ms", "Length-only (num_words) + Ridge"): oof_len_raw["oof_pred"],
        ("log_ms", "Length-only (num_words) + Ridge"): oof_len_log["oof_pred"],
        ("raw_ms", "TF-IDF (1-2 grams) + Ridge"): oof_tfidf_raw["oof_pred"],
        ("log_ms", "TF-IDF (1-2 grams) + Ridge"): oof_tfidf_log["oof_pred"],
        ("raw_ms", "bert-base-uncased embeddings + Ridge"): oof_bert_raw["oof_pred"],
        ("log_ms", "bert-base-uncased embeddings + Ridge"): oof_bert_log["oof_pred"],
    }

    # Plots: CV bars
    plot_cv_bar(res_df, "raw_ms", "pearson_mean", os.path.join(OUT_DIR, "cv_raw_pearson.png"))
    plot_cv_bar(res_df, "raw_ms", "mse_mean", os.path.join(OUT_DIR, "cv_raw_mse.png"))
    plot_cv_bar(res_df, "log_ms", "pearson_mean", os.path.join(OUT_DIR, "cv_log_pearson.png"))
    plot_cv_bar(res_df, "log_ms", "mse_mean", os.path.join(OUT_DIR, "cv_log_mse.png"))

    # Best model OOF plots (raw)
    y_pred_best_raw = oof_pred_map[("raw_ms", best_model_name)]
    plot_true_vs_pred(y_raw, y_pred_best_raw, f"OOF True vs Pred ({best_model_name}) - raw_ms",
                      os.path.join(OUT_DIR, "best_raw_true_vs_pred.png"))
    plot_residuals(y_raw, y_pred_best_raw, f"OOF Residuals ({best_model_name}) - raw_ms",
                   os.path.join(OUT_DIR, "best_raw_residuals.png"))

    # Error analysis table (raw, best model)
    err = np.abs(y_raw - y_pred_best_raw)
    err_df = df.copy()
    err_df["y_true_raw_ms"] = y_raw
    err_df["y_pred_raw_ms"] = y_pred_best_raw
    err_df["abs_error"] = err
    err_df = err_df.sort_values("abs_error", ascending=False)

    err_df_out = os.path.join(OUT_DIR, "error_analysis_worst10_raw_ms.csv")
    err_df.head(10)[["sentence_text", "y_true_raw_ms", "y_pred_raw_ms", "abs_error"]].to_csv(err_df_out, index=False)

    # Extra: error by length bins (simple but informative)
    lengths = X_base_df["num_words"].values
    bins = pd.qcut(lengths, q=4, duplicates="drop")
    tmp = pd.DataFrame({"bin": bins, "abs_error": err})
    err_by_bin = tmp.groupby("bin")["abs_error"].agg(["mean", "median", "count"]).reset_index()
    err_by_bin.to_csv(os.path.join(OUT_DIR, "error_by_length_bin.csv"), index=False)

    # Feature weights plot (fit one final baseline model on full data for interpretability)
    # Note: This is for interpretation only, not evaluation.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_base)
    ridge = Ridge(alpha=float(best_base_raw.alpha), random_state=RANDOM_STATE)
    ridge.fit(Xs, y_raw)
    plot_top_coefficients(feature_names, ridge.coef_, os.path.join(OUT_DIR, "baseline_top_coeffs.png"), top_k=15)

    print("\nSaved CV + plots to:", OUT_DIR)
    print("Key files:")
    print(" - cv_results.csv")
    print(" - best_raw_true_vs_pred.png")
    print(" - best_raw_residuals.png")
    print(" - baseline_top_coeffs.png")
    print(" - error_analysis_worst10_raw_ms.csv")
    print(" - error_by_length_bin.csv")


if __name__ == "__main__":
    main()
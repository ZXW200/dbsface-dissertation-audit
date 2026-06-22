"""Exploratory decomposition: nested out-of-fold logistic analysis.

This script quantifies how much of the full-pixel MLP's discrimination on the
supplied PD-DBS test split is independent of low-level regional structure. It
reuses the exact 40-feature ROI low-level confound representation defined in
``run_roi_lowlevel_confound_baseline.py`` (per ROI: mean, standard deviation,
10th percentile, 90th percentile, and mean gradient magnitude), and the same
class-weighted, early-stopped NumPy logistic estimator that produced the
published ROI low-level baseline (AUROC ~0.787), keeping the comparison within
the same classifier family.

Two nested out-of-fold (OOF) logistic models are compared on the test set:

* Model A: OOF logistic on the 40 ROI low-level confound features      -> AUROC X
* Model B: OOF logistic on the 40 features + the MLP per-sample score   -> AUROC Y

The net increment Y - X summarises the added discriminative signal after the
low-level regional features are included. Everything runs single-threaded with a
fixed seed (42).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "scripts"))
from load_pd_dbs import load_pd_dbs  # noqa: E402
from train_baseline_mlp_numpy import sigmoid, stratified_split  # noqa: E402

SEED = 42
N_FOLDS = 5
DATA_PATH = REPO_ROOT / "data" / "raw" / "PD_DBS_Data.mat"
MASKS_PATH = REPO_ROOT / "outputs" / "roi" / "coarse_roi_masks.npy"
ROI_DEFS_PATH = REPO_ROOT / "outputs" / "roi" / "coarse_roi_definitions.csv"
PRED_PATH = REPO_ROOT / "outputs" / "baseline" / "predictions_test.csv"
OUT_DIR = REPO_ROOT / "outputs" / "confound_net_increment"


def mask_to_flat(mask: np.ndarray) -> np.ndarray:
    """Map a display-space (row, col) ROI mask onto the raw 1024 flat order."""
    return mask.T.reshape(-1).astype(bool)


def extract_features(x_flat: np.ndarray, masks: np.ndarray, roi_names: list[str]) -> pd.DataFrame:
    """Replicate the published 40-feature ROI low-level confound representation."""
    images = x_flat.reshape(x_flat.shape[0], 32, 32).transpose(0, 2, 1).astype(np.float32)
    gy, gx = np.gradient(images, axis=(1, 2))
    grad = np.sqrt(gx * gx + gy * gy)
    rows: dict[str, np.ndarray] = {}
    stats = ["mean", "std", "p10", "p90", "grad_mean"]
    columns = [f"{roi}__{stat}" for roi in roi_names for stat in stats]
    for mask, roi in zip(masks, roi_names):
        flat = mask_to_flat(mask)
        vals = x_flat[:, flat].astype(np.float32)
        gvals = grad[:, mask]
        rows[f"{roi}__mean"] = vals.mean(axis=1)
        rows[f"{roi}__std"] = vals.std(axis=1)
        rows[f"{roi}__p10"] = np.percentile(vals, 10, axis=1)
        rows[f"{roi}__p90"] = np.percentile(vals, 90, axis=1)
        rows[f"{roi}__grad_mean"] = gvals.mean(axis=1)
    return pd.DataFrame(rows, columns=columns)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = SEED,
    epochs: int = 1000,
    lr: float = 0.03,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Class-weighted, early-stopped logistic (mirrors the published baseline)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=x_train.shape[1]).astype(np.float32)
    b = np.float32(0.0)
    best_w, best_b, best_loss = w.copy(), float(b), float("inf")

    pos_weight = len(y_train) / max(2 * int(y_train.sum()), 1)
    neg_weight = len(y_train) / max(2 * int((1 - y_train).sum()), 1)
    sample_weight = np.where(y_train == 1, pos_weight, neg_weight).astype(np.float32)

    for _ in range(epochs):
        p = sigmoid(x_train @ w + b)
        err = (p - y_train.astype(np.float32)) * sample_weight
        grad_w = (x_train.T @ err) / len(x_train) + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w.astype(np.float32)
        b = np.float32(b - lr * grad_b)

        val_p = sigmoid(x_val @ w + b)
        eps = 1e-7
        val_loss = float(
            -np.mean(
                y_val * np.log(np.clip(val_p, eps, 1 - eps))
                + (1 - y_val) * np.log(np.clip(1 - val_p, eps, 1 - eps))
            )
        )
        if val_loss < best_loss:
            best_loss, best_w, best_b = val_loss, w.copy(), float(b)
    return best_w, best_b


def oof_auroc(x: np.ndarray, y: np.ndarray, seed: int, n_folds: int) -> tuple[float, np.ndarray]:
    """Out-of-fold logistic AUROC; per-fold standardisation and early stopping."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(x, y):
        x_tr_all, y_tr_all = x[train_idx], y[train_idx]
        inner_tr, inner_val = stratified_split(y_tr_all, val_fraction=0.2, seed=seed)
        mean, std = fit_standardizer(x_tr_all[inner_tr])
        x_inner = standardize(x_tr_all[inner_tr], mean, std)
        x_val = standardize(x_tr_all[inner_val], mean, std)
        x_eval = standardize(x[test_idx], mean, std)
        w, b = train_logistic(x_inner, y_tr_all[inner_tr], x_val, y_tr_all[inner_val], seed=seed)
        oof[test_idx] = sigmoid(x_eval @ w + b)
    return float(roc_auc_score(y, oof)), oof


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_pd_dbs(DATA_PATH)
    x_test_flat = data["x_test_flat"]
    y_test = data["y_test"].astype(int)

    masks = np.load(MASKS_PATH).astype(bool)
    roi_names = pd.read_csv(ROI_DEFS_PATH)["roi_name"].tolist()

    preds = pd.read_csv(PRED_PATH)
    if len(preds) != len(y_test):
        raise ValueError(f"Prediction count {len(preds)} != test count {len(y_test)}")
    if not np.array_equal(preds["y_true"].to_numpy().astype(int), y_test):
        raise ValueError("predictions_test.csv y_true does not match supplied y_test order")
    mlp_score = preds["p_class1"].to_numpy(dtype=np.float32)

    features = extract_features(x_test_flat, masks, roi_names)
    x_confound = features.to_numpy(dtype=np.float32)               # 40 features
    x_combined = np.column_stack([x_confound, mlp_score]).astype(np.float32)  # 40 + MLP score

    auroc_x, oof_a = oof_auroc(x_confound, y_test, SEED, N_FOLDS)   # Model A
    auroc_y, oof_b = oof_auroc(x_combined, y_test, SEED, N_FOLDS)   # Model B
    net = auroc_y - auroc_x

    def sig4(v: float) -> float:
        return float(f"{v:.4g}")

    result = {
        "n_test": int(len(y_test)),
        "n_features_confound": int(x_confound.shape[1]),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "model_a_confound_only_auroc": sig4(auroc_x),
        "model_b_confound_plus_mlp_auroc": sig4(auroc_y),
        "net_increment": sig4(net),
        "auroc_x_full": auroc_x,
        "auroc_y_full": auroc_y,
        "net_increment_full": net,
    }
    (OUT_DIR / "confound_net_increment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "sample_index_test": np.arange(len(y_test)),
            "y_true": y_test,
            "mlp_p_class1": mlp_score,
            "oof_model_a_confound_only": oof_a,
            "oof_model_b_confound_plus_mlp": oof_b,
        }
    ).to_csv(OUT_DIR / "confound_net_increment_oof_predictions.csv", index=False)

    print("# Confound-adjusted discrimination (nested OOF logistic)")
    print(f"Test samples: {len(y_test)}")
    print(f"Model A (40 confound features) AUROC X = {auroc_x:.4f}")
    print(f"Model B (40 features + MLP score) AUROC Y = {auroc_y:.4f}")
    print(f"Net increment Y - X = {net:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

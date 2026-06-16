"""Run a low-level ROI statistics baseline for the PD-DBS audit.

The model sees only per-ROI intensity and texture summaries, not raw pixels or
spatial face structure. This provides a stronger low-level confound comparator
than whole-image global statistics alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
from load_pd_dbs import load_pd_dbs
from train_baseline_mlp_numpy import metric_summary, sigmoid, stratified_split


def mask_to_flat(mask: np.ndarray) -> np.ndarray:
    return mask.T.reshape(-1).astype(bool)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def feature_names(roi_names: list[str]) -> list[str]:
    stats = ["mean", "std", "p10", "p90", "grad_mean"]
    return [f"{roi}__{stat}" for roi in roi_names for stat in stats]


def extract_features(x_flat: np.ndarray, masks: np.ndarray, roi_names: list[str]) -> pd.DataFrame:
    images = x_flat.reshape(x_flat.shape[0], 32, 32).transpose(0, 2, 1).astype(np.float32)
    gy, gx = np.gradient(images, axis=(1, 2))
    grad = np.sqrt(gx * gx + gy * gy)
    rows: dict[str, np.ndarray] = {}
    for mask, roi in zip(masks, roi_names):
        flat = mask_to_flat(mask)
        vals = x_flat[:, flat].astype(np.float32)
        gvals = grad[:, mask]
        rows[f"{roi}__mean"] = vals.mean(axis=1)
        rows[f"{roi}__std"] = vals.std(axis=1)
        rows[f"{roi}__p10"] = np.percentile(vals, 10, axis=1)
        rows[f"{roi}__p90"] = np.percentile(vals, 90, axis=1)
        rows[f"{roi}__grad_mean"] = gvals.mean(axis=1)
    return pd.DataFrame(rows, columns=feature_names(roi_names))


def train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=x_train.shape[1]).astype(np.float32)
    b = np.float32(0.0)
    best_w = w.copy()
    best_b = float(b)
    best_loss = float("inf")
    history: list[dict[str, float]] = []

    pos_weight = len(y_train) / max(2 * int(y_train.sum()), 1)
    neg_weight = len(y_train) / max(2 * int((1 - y_train).sum()), 1)
    sample_weight = np.where(y_train == 1, pos_weight, neg_weight).astype(np.float32)

    for epoch in range(1, epochs + 1):
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
        val_acc = float(((val_p >= 0.5).astype(int) == y_val).mean())
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc})
        if val_loss < best_loss:
            best_loss = val_loss
            best_w = w.copy()
            best_b = float(b)
    return best_w, best_b, history


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    n0, n1 = len(a), len(b)
    if n0 < 2 or n1 < 2:
        return float("nan")
    pooled = np.sqrt(((n0 - 1) * a.var(ddof=1) + (n1 - 1) * b.var(ddof=1)) / max(n0 + n1 - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def class_difference_table(features: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        roi, stat = col.split("__", 1)
        a = features.loc[y == 0, col].to_numpy(dtype=float)
        b = features.loc[y == 1, col].to_numpy(dtype=float)
        rows.append(
            {
                "roi_name": roi,
                "feature": stat,
                "mean_class0": float(a.mean()),
                "mean_class1": float(b.mean()),
                "diff_class1_minus_class0": float(b.mean() - a.mean()),
                "cohen_d": cohen_d(a, b),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("cohen_d", key=lambda s: s.abs(), ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--roi-masks", default="outputs/roi/coarse_roi_masks.npy")
    parser.add_argument("--roi-definitions", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument("--output-dir", default="outputs/lowlevel_roi_confound")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pd_dbs(args.data)
    masks = np.load(args.roi_masks).astype(bool)
    roi_defs = pd.read_csv(args.roi_definitions)
    roi_names = roi_defs["roi_name"].tolist()

    train_features = extract_features(data["x_train_flat"], masks, roi_names)
    test_features = extract_features(data["x_test_flat"], masks, roi_names)
    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)

    train_idx, val_idx = stratified_split(y_train, val_fraction=0.2, seed=args.seed)
    mean, std = fit_standardizer(train_features.iloc[train_idx].to_numpy(np.float32))
    x_train = standardize(train_features.iloc[train_idx].to_numpy(np.float32), mean, std)
    x_val = standardize(train_features.iloc[val_idx].to_numpy(np.float32), mean, std)
    x_test = standardize(test_features.to_numpy(np.float32), mean, std)

    w, b, history = train_logistic(
        x_train,
        y_train[train_idx],
        x_val,
        y_train[val_idx],
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )
    p_test = sigmoid(x_test @ w + b)
    metrics = metric_summary(y_test, p_test)

    pd.DataFrame(
        {
            "sample_index_test": np.arange(len(y_test)),
            "y_true": y_test,
            "p_class1_lowlevel_roi": p_test,
            "y_pred_lowlevel_roi": (p_test >= 0.5).astype(int),
        }
    ).to_csv(out_dir / "lowlevel_roi_predictions.csv", index=False)
    test_features.assign(sample_index_test=np.arange(len(y_test)), y_true=y_test).to_csv(
        out_dir / "lowlevel_roi_features_test.csv", index=False
    )
    train_features.assign(sample_index_train=np.arange(len(y_train)), y_true=y_train).to_csv(
        out_dir / "lowlevel_roi_features_train.csv", index=False
    )
    diff_df = class_difference_table(test_features, y_test)
    diff_df.to_csv(out_dir / "lowlevel_roi_class_differences.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    metadata = {
        "feature_family": "per-ROI mean, standard deviation, 10th percentile, 90th percentile, and gradient magnitude mean",
        "n_features": int(train_features.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "metrics": metrics,
        "strongest_lowlevel_feature_by_abs_d": diff_df.iloc[0].to_dict(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("# ROI Low-Level Confound Baseline")
    print(f"Features: {train_features.shape[1]} per image")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print("Strongest low-level feature:")
    print(diff_df.head(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

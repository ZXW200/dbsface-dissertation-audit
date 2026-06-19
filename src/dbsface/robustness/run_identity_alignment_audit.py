"""Additional leakage, global-feature, and coarse-alignment audits.

This script runs two sensitivity checks for the supplied image-level split:

1. High image-level performance may reflect subject/acquisition similarity.
2. Fixed coarse ROI masks only make anatomical sense if faces are roughly
   centred and similarly framed.

All outputs retain numeric Class 0 / Class 1 wording.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs
from train_baseline_mlp_numpy import metric_summary, stratified_split


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.05, size=x_train.shape[1]).astype(np.float64)
    b = 0.0
    best_w = w.copy()
    best_b = b
    best_loss = float("inf")
    y_train = y_train.astype(np.float64)
    y_val = y_val.astype(np.float64)
    for _ in range(epochs):
        p = sigmoid(x_train @ w + b)
        grad = p - y_train
        w -= lr * ((x_train.T @ grad) / len(x_train) + l2 * w)
        b -= lr * float(grad.mean())
        val_p = sigmoid(x_val @ w + b)
        val_loss = float(-(y_val * np.log(np.clip(val_p, 1e-7, 1)) + (1 - y_val) * np.log(np.clip(1 - val_p, 1e-7, 1))).mean())
        if val_loss < best_loss:
            best_loss = val_loss
            best_w = w.copy()
            best_b = b
    return best_w, best_b


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    sims = []
    for start in range(0, len(b_norm), chunk):
        sims.append(b_norm[start : start + chunk] @ a_norm.T)
    return np.vstack(sims)


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y_sorted = y[order].astype(int)
    pos = y_sorted.sum()
    neg = len(y_sorted) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.r_[0, tps / pos, 1]
    fpr = np.r_[0, fps / neg, 1]
    return float(np.trapezoid(tpr, fpr))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def global_features(x_flat: np.ndarray) -> tuple[np.ndarray, list[str]]:
    imgs = x_flat.reshape(len(x_flat), 32, 32).transpose(0, 2, 1).astype(np.float64)
    feats = []
    names = [
        "mean_intensity",
        "std_intensity",
        "p95_minus_p05",
        "upper_minus_lower_mean",
        "left_minus_right_mean",
        "center_minus_border_mean",
        "bright_fraction",
        "dark_fraction",
    ]
    for img in imgs:
        p05, p50, p95 = np.percentile(img, [5, 50, 95])
        border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
        center = img[10:22, 10:22].reshape(-1)
        feats.append(
            [
                float(img.mean()),
                float(img.std()),
                float(p95 - p05),
                float(img[:16, :].mean() - img[16:, :].mean()),
                float(img[:, :16].mean() - img[:, 16:].mean()),
                float(center.mean() - border.mean()),
                float((img > p50).mean()),
                float((img < p50).mean()),
            ]
        )
    return np.asarray(feats, dtype=np.float64), names


def otsu_threshold(img: np.ndarray) -> float:
    values = img.astype(np.float64).ravel()
    hist, edges = np.histogram(values, bins=64)
    mids = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    mean1 = np.cumsum(hist * mids) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * mids)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1))[::-1]
    var_between = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[1:]) ** 2
    return float(mids[int(np.argmax(var_between))])


def alignment_qc(x_flat: np.ndarray, y: np.ndarray, split: str) -> list[dict[str, float | int | str]]:
    imgs = x_flat.reshape(len(x_flat), 32, 32).transpose(0, 2, 1).astype(np.float64)
    rows = []
    yy, xx = np.mgrid[0:32, 0:32]
    for idx, img in enumerate(imgs):
        threshold = otsu_threshold(img)
        mask = img > threshold
        if mask.mean() < 0.08 or mask.mean() > 0.92:
            threshold = float(np.percentile(img, 55))
            mask = img > threshold
        coords = np.argwhere(mask)
        if len(coords) == 0:
            coords = np.argwhere(np.ones_like(mask, dtype=bool))
        weights = np.maximum(img - np.percentile(img, 10), 0)
        if weights.sum() <= 0:
            weights = mask.astype(float)
        cx = float((weights * xx).sum() / np.maximum(weights.sum(), 1e-12))
        cy = float((weights * yy).sum() / np.maximum(weights.sum(), 1e-12))
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        rows.append(
            {
                "sample_id": f"{split}_{idx:04d}",
                "split": split,
                "y_true": int(y[idx]),
                "centroid_x": cx,
                "centroid_y": cy,
                "centroid_dx_from_center": cx - 15.5,
                "centroid_dy_from_center": cy - 15.5,
                "bbox_x0": int(x0),
                "bbox_x1": int(x1),
                "bbox_y0": int(y0),
                "bbox_y1": int(y1),
                "bbox_width": int(x1 - x0),
                "bbox_height": int(y1 - y0),
                "foreground_fraction": float(mask.mean()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--model", default="models/baseline_mlp_numpy.npz")
    parser.add_argument("--predictions", default="outputs/baseline/predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/data_qc")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_pd_dbs(args.data)
    x_train = data["x_train_flat"].astype(np.float64)
    x_test = data["x_test_flat"].astype(np.float64)
    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)

    # Wider train-test similarity sensitivity using the frozen baseline predictions.
    pred_df = pd.read_csv(args.predictions)
    model_scores = pred_df["p_class1"].to_numpy(float)
    sim = cosine_similarity_matrix(x_train, x_test)
    max_cos = sim.max(axis=1)
    rows = []
    for threshold in [0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.90]:
        keep = max_cos < threshold
        if keep.sum() >= 10 and len(np.unique(y_test[keep])) == 2:
            metrics = metric_summary(y_test[keep], model_scores[keep])
            rows.append(
                {
                    "max_cosine_exclusion_threshold": threshold,
                    "removed_test_samples": int((~keep).sum()),
                    "remaining_test_samples": int(keep.sum()),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                }
            )
    pd.DataFrame(rows).to_csv(out / "similarity_threshold_sensitivity.csv", index=False)

    # Global image statistic correlations and a low-dimensional baseline.
    x_all_train, feature_names = global_features(x_train)
    x_all_test, _ = global_features(x_test)
    corr_rows = []
    for split, feats, labels in [("train", x_all_train, y_train), ("test", x_all_test, y_test)]:
        for j, name in enumerate(feature_names):
            corr_rows.append(
                {
                    "split": split,
                    "feature": name,
                    "pearson_with_label": pearson(feats[:, j], labels),
                    "class0_mean": float(feats[labels == 0, j].mean()),
                    "class1_mean": float(feats[labels == 1, j].mean()),
                    "class1_minus_class0": float(feats[labels == 1, j].mean() - feats[labels == 0, j].mean()),
                }
            )
    pd.DataFrame(corr_rows).to_csv(out / "global_image_statistics_by_label.csv", index=False)

    tr_idx, val_idx = stratified_split(y_train, 0.2, args.seed)
    mean_g = x_all_train[tr_idx].mean(axis=0, keepdims=True)
    std_g = x_all_train[tr_idx].std(axis=0, keepdims=True)
    std_g[std_g < 1e-8] = 1.0
    xg_tr = (x_all_train[tr_idx] - mean_g) / std_g
    xg_val = (x_all_train[val_idx] - mean_g) / std_g
    xg_test = (x_all_test - mean_g) / std_g
    w, b = train_logistic(xg_tr, y_train[tr_idx], xg_val, y_train[val_idx], epochs=1200, lr=0.03, l2=1e-3, seed=args.seed)
    global_p = sigmoid(xg_test @ w + b)
    global_metrics = metric_summary(y_test, global_p)
    (out / "global_statistics_baseline_metrics.json").write_text(json.dumps(global_metrics, indent=2), encoding="utf-8")

    # Coarse alignment/framing QC.
    align_rows = alignment_qc(x_train, y_train, "train") + alignment_qc(x_test, y_test, "test")
    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(out / "alignment_qc.csv", index=False)
    align_summary = (
        align_df.groupby("split")[
            [
                "centroid_x",
                "centroid_y",
                "centroid_dx_from_center",
                "centroid_dy_from_center",
                "bbox_width",
                "bbox_height",
                "foreground_fraction",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    align_summary.to_csv(out / "alignment_qc_summary.csv")

    # Human-readable summary.
    strongest_global = max(corr_rows, key=lambda r: abs(float(r["pearson_with_label"])))
    summary = f"""# Identity, Global-Confound, and Alignment Audit

## Similarity-threshold sensitivity

The frozen baseline was re-evaluated after excluding test samples whose maximum cosine similarity to any training image exceeded progressively lower thresholds. This is stricter than the original exact/near-duplicate check, but it is still not a substitute for patient identifiers.

{pd.DataFrame(rows).to_markdown(index=False)}

## Global image statistics

A logistic baseline using only eight global image statistics achieved:

- accuracy: {global_metrics['accuracy']:.4f}
- balanced accuracy: {global_metrics['balanced_accuracy']:.4f}
- AUROC: {global_metrics['auroc']:.4f}
- trapezoidal AUPRC: {global_metrics['auprc']:.4f}

The strongest single global-statistic/label correlation was `{strongest_global['feature']}` on the {strongest_global['split']} split, r = {float(strongest_global['pearson_with_label']):.4f}. These results test whether gross brightness/contrast/framing statistics alone can explain the main classifier.

## Coarse alignment/framing QC

Intensity-threshold foreground masks were used to estimate coarse centroid and bounding-box summaries. The output is a low-resolution framing check that gives limited support for the fixed-ROI assumption.

{align_summary.to_markdown()}

## Interpretation

These analyses report exact duplicate checks, broader similarity-threshold exclusions, global image statistics, and coarse alignment summaries. They provide risk context for the supplied image-level split while patient-level independence remains dependent on patient identifiers.
"""
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

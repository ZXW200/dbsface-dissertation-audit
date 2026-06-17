"""Train the Week 2 NumPy MLP baseline classifier for the PD-DBS data.

The intended baseline is a small CNN, but the bundled runtime currently has no
torch, tensorflow, or keras. To keep Week 2 unblocked, this script trains a
dependency-light NumPy MLP fallback and writes the same output contract expected
from a baseline trainer: metrics, per-sample predictions, curves, figures, and
a model checkpoint.

All outputs use numeric Class 0 / Class 1 naming. The current package treats
the classes only as supplied numeric labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent))
from dbsface.data.load_pd_dbs import load_pd_dbs


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def bce_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def stratified_split(y: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for cls in sorted(np.unique(y)):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def init_model(n_features: int, hidden: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": (rng.normal(0, np.sqrt(2 / n_features), size=(n_features, hidden))).astype(np.float32),
        "b1": np.zeros((1, hidden), dtype=np.float32),
        "w2": (rng.normal(0, np.sqrt(2 / hidden), size=(hidden, 1))).astype(np.float32),
        "b2": np.zeros((1, 1), dtype=np.float32),
    }


def forward(model: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = x @ model["w1"] + model["b1"]
    h1 = relu(z1)
    logits = h1 @ model["w2"] + model["b2"]
    return sigmoid(logits).reshape(-1), z1, h1


def adam_update(
    model: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    opt: dict[str, dict[str, np.ndarray]],
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    for key, grad in grads.items():
        opt["m"][key] = beta1 * opt["m"][key] + (1 - beta1) * grad
        opt["v"][key] = beta2 * opt["v"][key] + (1 - beta2) * (grad * grad)
        m_hat = opt["m"][key] / (1 - beta1**step)
        v_hat = opt["v"][key] / (1 - beta2**step)
        model[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    model = init_model(x_train.shape[1], hidden, seed)
    opt = {
        "m": {k: np.zeros_like(v) for k, v in model.items()},
        "v": {k: np.zeros_like(v) for k, v in model.items()},
    }
    history = []
    best_loss = float("inf")
    best_model = {k: v.copy() for k, v in model.items()}
    step = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx].astype(np.float32).reshape(-1, 1)

            p, z1, h1 = forward(model, xb)
            dz2 = (p.reshape(-1, 1) - yb) / len(xb)
            grads = {
                "w2": h1.T @ dz2 + l2 * model["w2"],
                "b2": dz2.sum(axis=0, keepdims=True),
            }
            dh1 = dz2 @ model["w2"].T
            dz1 = dh1 * (z1 > 0)
            grads["w1"] = xb.T @ dz1 + l2 * model["w1"]
            grads["b1"] = dz1.sum(axis=0, keepdims=True)

            step += 1
            adam_update(model, grads, opt, step, lr)

        train_p = forward(model, x_train)[0]
        val_p = forward(model, x_val)[0]
        train_loss = float(bce_loss(y_train, train_p).mean())
        val_loss = float(bce_loss(y_val, val_p).mean())
        train_acc = float(((train_p >= 0.5).astype(int) == y_train).mean())
        val_acc = float(((val_p >= 0.5).astype(int) == y_val).mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = {k: v.copy() for k, v in model.items()}

    return best_model, history


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y = y.astype(int)
    pred = pred.astype(int)
    return {
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "tp": int(((y == 1) & (pred == 1)).sum()),
    }


def roc_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(scores)[::-1]
    y_sorted = y[order].astype(int)
    pos = y_sorted.sum()
    neg = len(y_sorted) - pos
    if pos == 0 or neg == 0:
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.r_[0, tps / pos, 1]
    fpr = np.r_[0, fps / neg, 1]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def pr_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(scores)[::-1]
    y_sorted = y[order].astype(int)
    pos = y_sorted.sum()
    if pos == 0:
        return np.array([0, 1]), np.array([0, 0]), float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    recall = np.r_[0, tps / pos]
    precision = np.r_[1, tps / np.maximum(tps + fps, 1)]
    return recall, precision, float(np.trapezoid(precision, recall))


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | dict[str, int]]:
    pred = (p >= 0.5).astype(int)
    cm = confusion(y, pred)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    balanced = 0.5 * (recall + specificity)
    fpr, tpr, auroc = roc_curve(y, p)
    pr_recall, pr_precision, auprc = pr_curve(y, p)
    counts = np.bincount(y.astype(int), minlength=2)
    return {
        "n": int(len(y)),
        "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "majority_baseline_accuracy": float(counts.max() / len(y)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "precision_class1": float(precision),
        "recall_class1": float(recall),
        "specificity_class0": float(specificity),
        "f1_class1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier_score": float(np.mean((p - y) ** 2)),
        "mean_bce_loss": float(bce_loss(y, p).mean()),
        "confusion_matrix": cm,
    }


def write_predictions(path: Path, sample_ids: Iterable[str], split: str, y: np.ndarray, p: np.ndarray) -> None:
    pred = (p >= 0.5).astype(int)
    losses = bce_loss(y, p)
    df = pd.DataFrame(
        {
            "sample_id": list(sample_ids),
            "split": split,
            "y_true": y.astype(int),
            "p_class1": p.astype(float),
            "y_pred": pred.astype(int),
            "correct": (pred == y).astype(int),
            "loss": losses.astype(float),
        }
    )
    df.to_csv(path, index=False)


def save_curve_csv(path: Path, x_name: str, x: np.ndarray, y_name: str, y: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([x_name, y_name])
        writer.writerows(zip(x, y))


def draw_line_plot(path: Path, series: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int, int]]], title: str) -> None:
    width, height = 720, 480
    margin = 60
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 18), title, fill=(0, 0, 0))
    draw.rectangle([margin, margin, width - margin, height - margin], outline=(0, 0, 0))
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2] for s in series])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1

    def to_px(xv, yv):
        x = margin + (xv - xmin) / (xmax - xmin) * (width - 2 * margin)
        y = height - margin - (yv - ymin) / (ymax - ymin) * (height - 2 * margin)
        return x, y

    for idx, (label, xvals, yvals, color) in enumerate(series):
        pts = [to_px(float(a), float(b)) for a, b in zip(xvals, yvals)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
        lx = width - margin - 180
        ly = margin + 18 * idx
        draw.line([lx, ly + 6, lx + 24, ly + 6], fill=color, width=3)
        draw.text((lx + 30, ly), label, fill=(0, 0, 0))
    img.save(path)


def draw_confusion_matrix(path: Path, cm: dict[str, int], title: str = "Confusion Matrix") -> None:
    img = Image.new("RGB", (420, 360), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 20), title, fill=(0, 0, 0))
    labels = [["TN", "FP"], ["FN", "TP"]]
    values = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    max_v = max(max(row) for row in values) or 1
    for r in range(2):
        for c in range(2):
            x0 = 90 + c * 120
            y0 = 80 + r * 100
            val = values[r][c]
            shade = 255 - int(130 * val / max_v)
            draw.rectangle([x0, y0, x0 + 110, y0 + 90], fill=(shade, shade, 255), outline=(0, 0, 0))
            draw.text((x0 + 35, y0 + 22), labels[r][c], fill=(0, 0, 0))
            draw.text((x0 + 35, y0 + 50), str(val), fill=(0, 0, 0))
    draw.text((115, 300), "Pred 0        Pred 1", fill=(0, 0, 0))
    draw.text((10, 105), "True 0", fill=(0, 0, 0))
    draw.text((10, 205), "True 1", fill=(0, 0, 0))
    img.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    data = load_pd_dbs(args.data)
    x_source = data["x_train_flat"].astype(np.float32)
    y_source = data["y_train"].astype(np.int64)
    x_test_source = data["x_test_flat"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    train_idx, val_idx = stratified_split(y_source, args.val_fraction, args.seed)
    x_train_raw, y_train = x_source[train_idx], y_source[train_idx]
    x_val_raw, y_val = x_source[val_idx], y_source[val_idx]

    mean, std = fit_standardizer(x_train_raw)
    x_train = standardize(x_train_raw, mean, std)
    x_val = standardize(x_val_raw, mean, std)
    x_test = standardize(x_test_source, mean, std)

    model, history = train_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )

    train_p = forward(model, x_train)[0]
    val_p = forward(model, x_val)[0]
    test_p = forward(model, x_test)[0]

    metrics = {
        "model_type": "numpy_mlp_fallback_no_torch_or_keras",
        "label_semantics": "numeric_class_0_vs_class_1; clinical 0/1 mapping unresolved",
        "data_path": str(Path(args.data).resolve()),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "normalization": "per-feature z-score fitted on train split",
        "train": metric_summary(y_train, train_p),
        "val": metric_summary(y_val, val_p),
        "test": metric_summary(y_test, test_p),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_predictions(out_dir / "predictions_train.csv", [f"train_{i:04d}" for i in train_idx], "train", y_train, train_p)
    write_predictions(out_dir / "predictions_val.csv", [f"train_{i:04d}" for i in val_idx], "val", y_val, val_p)
    write_predictions(out_dir / "predictions_test.csv", [f"test_{i:04d}" for i in range(len(y_test))], "test", y_test, test_p)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_curve.csv", index=False)
    draw_line_plot(
        out_dir / "training_curve.png",
        [
            ("train_loss", hist_df["epoch"].to_numpy(), hist_df["train_loss"].to_numpy(), (30, 90, 180)),
            ("val_loss", hist_df["epoch"].to_numpy(), hist_df["val_loss"].to_numpy(), (200, 80, 50)),
        ],
        "Training Curve",
    )

    fpr, tpr, _ = roc_curve(y_test, test_p)
    save_curve_csv(out_dir / "roc_curve.csv", "fpr", fpr, "tpr", tpr)
    draw_line_plot(out_dir / "roc_curve.png", [("ROC", fpr, tpr, (30, 90, 180))], "Test ROC Curve")

    recall, precision, _ = pr_curve(y_test, test_p)
    save_curve_csv(out_dir / "pr_curve.csv", "recall", recall, "precision", precision)
    draw_line_plot(out_dir / "pr_curve.png", [("PR", recall, precision, (30, 130, 80))], "Test Precision-Recall Curve")

    draw_confusion_matrix(out_dir / "confusion_matrix.png", metrics["test"]["confusion_matrix"])

    np.savez(
        model_dir / "baseline_mlp_numpy.npz",
        **model,
        mean=mean,
        std=std,
        train_idx=train_idx,
        val_idx=val_idx,
        seed=np.array([args.seed], dtype=np.int64),
    )

    print(json.dumps(metrics["test"], indent=2))
    print(f"wrote outputs to {out_dir.resolve()}")
    print(f"wrote model to {(model_dir / 'baseline_mlp_numpy.npz').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



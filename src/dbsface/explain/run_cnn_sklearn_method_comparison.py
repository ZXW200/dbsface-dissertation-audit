"""Run three isolated method experiments with PyTorch and scikit-learn.

Outputs are deliberately separated by method:

1. outputs/model_comparison/logistic_regression_linear
2. outputs/model_comparison/cnn_aev
3. outputs/model_comparison/cnn_gradcam_projected_roi

The CNN model is trained once and then reused for the CNN+AEV and
CNN+Grad-CAM projected-to-ROI analyses. Each method folder still receives its
own workflow diagram and result files to avoid mixing outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))
from dbsface.data.load_pd_dbs import load_pd_dbs


ROI_SHORT_NAMES = [
    "upper brow",
    "left eye",
    "right eye",
    "nasal midface",
    "left cheek",
    "right cheek",
    "mouth",
    "chin",
]

PALETTE = {
    "navy": (15, 37, 69),
    "blue": (47, 102, 208),
    "teal": (0, 124, 120),
    "orange": (213, 106, 0),
    "red": (217, 72, 59),
    "gray": (83, 96, 120),
    "grid": (220, 226, 235),
    "panel": (247, 250, 254),
    "soft_teal": (234, 247, 246),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy, value: str, size=22, fill=None, bold=False, anchor=None) -> None:
    draw.text(xy, value, font=font(size, bold), fill=fill or PALETTE["navy"], anchor=anchor)


def wrap_text(draw: ImageDraw.ImageDraw, xy, value: str, size: int, max_width: int, fill=None) -> None:
    fnt = font(size)
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=fnt)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for i, line in enumerate(lines):
        draw.text((x, y + i * (size + 5)), line, font=fnt, fill=fill or PALETTE["gray"])


def workflow_diagram(title: str, steps: list[tuple[str, str, tuple[int, int, int]]], output: Path) -> None:
    w, h = 1650, 710
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (55, 42), title, 40, fill=PALETTE["navy"], bold=True)
    x0, y0 = 70, 230
    box_w, box_h, gap = 280, 220, 45
    for i, (head, body, color) in enumerate(steps):
        x = x0 + i * (box_w + gap)
        draw.rounded_rectangle((x, y0, x + box_w, y0 + box_h), radius=24, fill=PALETTE["panel"], outline=PALETTE["grid"], width=3)
        draw.ellipse((x + 22, y0 + 22, x + 78, y0 + 78), fill=color)
        draw_text(draw, (x + 50, y0 + 50), str(i + 1), 22, fill="white", bold=True, anchor="mm")
        draw_text(draw, (x + 95, y0 + 30), head, 24, fill=PALETTE["navy"], bold=True)
        wrap_text(draw, (x + 30, y0 + 100), body, 20, box_w - 60, fill=PALETTE["gray"])
        if i < len(steps) - 1:
            ax0 = x + box_w + 8
            ax1 = x + box_w + gap - 8
            ay = y0 + box_h / 2
            draw.line((ax0, ay, ax1, ay), fill=PALETTE["teal"], width=5)
            draw.polygon([(ax1, ay), (ax1 - 16, ay - 10), (ax1 - 16, ay + 10)], fill=PALETTE["teal"])
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output, dpi=(300, 300))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ece_score(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(y_true[mask]) - np.mean(p[mask])))
    return ece


def metric_dict(y_true: np.ndarray, p_class1: np.ndarray, model_name: str) -> dict[str, Any]:
    y_pred = (p_class1 >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "model": model_name,
        "n": int(len(y_true)),
        "positive_class": "Class 1",
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_class1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, p_class1)),
        "auprc_class1": float(average_precision_score(y_true, p_class1)),
        "brier": float(brier_score_loss(y_true, p_class1)),
        "ece_10bin": float(ece_score(y_true, p_class1, bins=10)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def save_predictions(path: Path, sample_ids: list[str], y_true: np.ndarray, p_class1: np.ndarray) -> None:
    y_pred = (p_class1 >= 0.5).astype(int)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "y_true": y_true.astype(int),
            "p_class0": (1.0 - p_class1).astype(float),
            "p_class1": p_class1.astype(float),
            "y_pred": y_pred.astype(int),
            "correct": (y_pred == y_true).astype(int),
        }
    ).to_csv(path, index=False)


def load_rois(root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    roi_defs = pd.read_csv(root / "outputs" / "roi" / "coarse_roi_definitions.csv")
    masks = np.load(root / "outputs" / "roi" / "coarse_roi_masks.npy").astype(bool)
    flat_masks = np.array([m.T.reshape(-1).astype(bool) for m in masks])
    return roi_defs, masks, flat_masks


def run_logistic(root: Path, data: dict[str, np.ndarray], out_dir: Path, seed: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train = data["x_train_flat"].astype(np.float32)
    y_train = data["y_train"].astype(int)
    x_test = data["x_test_flat"].astype(np.float32)
    y_test = data["y_test"].astype(int)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, val_idx = next(split.split(x_train, y_train))
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train[fit_idx])
    x_val = scaler.transform(x_train[val_idx])
    x_test_std = scaler.transform(x_test)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
        class_weight=None,
    )
    model.fit(x_fit, y_train[fit_idx])
    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test_std)[:, 1]

    metrics = metric_dict(y_test, p_test, "sklearn LogisticRegression")
    metrics["validation_auroc"] = float(roc_auc_score(y_train[val_idx], p_val))
    metrics["validation_auprc_class1"] = float(average_precision_score(y_train[val_idx], p_val))
    save_json(out_dir / "metrics.json", metrics)
    save_predictions(out_dir / "test_predictions.csv", [f"test_{i:04d}" for i in range(len(y_test))], y_test, p_test)
    np.savez(out_dir / "linear_model_arrays.npz", coef=model.coef_, intercept=model.intercept_, mean=scaler.mean_, scale=scaler.scale_)
    save_linear_coefficient_map(model.coef_.reshape(32, 32).T, out_dir / "linear_coefficient_map.png")
    roi_defs, masks, _ = load_rois(root)
    coefficient_roi_summary(model.coef_.reshape(32, 32).T, masks, roi_defs, out_dir / "linear_coefficient_roi_summary.csv")
    write_readme(
        out_dir / "README.md",
        "Logistic Regression / Linear Model",
        [
            "Flattened-pixel linear baseline using sklearn LogisticRegression.",
            "Outputs: metrics, test predictions, coefficient map, and ROI summary of coefficient magnitudes.",
            "Coefficient maps are linear diagnostics, not clinical or anatomical evidence.",
        ],
    )
    return metrics


def save_linear_coefficient_map(weight_map: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=220)
    vmax = float(np.max(np.abs(weight_map))) or 1.0
    im = ax.imshow(weight_map, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title("Logistic regression coefficient map", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("coefficient toward Class 1", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def coefficient_roi_summary(weight_map: np.ndarray, masks: np.ndarray, roi_defs: pd.DataFrame, output: Path) -> None:
    rows = []
    for i, name in enumerate(ROI_SHORT_NAMES):
        vals = weight_map[masks[i]]
        rows.append(
            {
                "roi_index": i + 1,
                "roi_name": name,
                "pixel_count": int(vals.size),
                "mean_coefficient": float(np.mean(vals)),
                "mean_abs_coefficient": float(np.mean(np.abs(vals))),
                "sum_abs_coefficient": float(np.sum(np.abs(vals))),
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)


class SmallFaceCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        out = self.pool(feat)
        logit = self.classifier(out).squeeze(1)
        if return_features:
            return logit, feat
        return logit


def make_cnn_arrays(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = data["x_train_images"][:, :, :, 0].astype(np.float32)
    x_test = data["x_test_images"][:, :, :, 0].astype(np.float32)
    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)
    return x_train, y_train, x_test, y_test


def normalize_cnn_images(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x - mean) / max(std, 1e-6)).astype(np.float32)[:, None, :, :]


def train_cnn(data: dict[str, np.ndarray], out_dir: Path, seed: int, device: torch.device) -> tuple[SmallFaceCNN, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = make_cnn_arrays(data)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, val_idx = next(split.split(x_train.reshape(len(x_train), -1), y_train))
    train_mean = float(x_train[fit_idx].mean())
    train_std = float(x_train[fit_idx].std())
    train_mean_image = x_train[fit_idx].mean(axis=0).astype(np.float32)

    x_fit = normalize_cnn_images(x_train[fit_idx], train_mean, train_std)
    x_val = normalize_cnn_images(x_train[val_idx], train_mean, train_std)
    x_test_norm = normalize_cnn_images(x_test, train_mean, train_std)
    y_fit = y_train[fit_idx].astype(np.float32)
    y_val = y_train[val_idx].astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=64,
        shuffle=True,
        drop_last=False,
    )
    val_x_t = torch.from_numpy(x_val).to(device)
    val_y_t = torch.from_numpy(y_val).to(device)
    model = SmallFaceCNN().to(device)
    n_pos = float(np.sum(y_fit == 1))
    n_neg = float(np.sum(y_fit == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    patience = 18
    wait = 0
    history = []
    for epoch in range(1, 101):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_t)
            val_loss = float(criterion(val_logits, val_y_t).detach().cpu())
            val_p = torch.sigmoid(val_logits).detach().cpu().numpy()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_accuracy": float(accuracy_score(y_val.astype(int), (val_p >= 0.5).astype(int))),
            "val_auroc": float(roc_auc_score(y_val.astype(int), val_p)),
            "val_auprc_class1": float(average_precision_score(y_val.astype(int), val_p)),
        }
        history.append(row)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    p_test = predict_cnn(model, x_test_norm, device)
    metrics = metric_dict(y_test, p_test, "PyTorch SmallFaceCNN")
    metrics["epochs_run"] = int(history[-1]["epoch"])
    metrics["best_validation_loss"] = best_val_loss
    metrics["normalization"] = {"train_mean": train_mean, "train_std": train_std}
    metrics["device"] = str(device)
    save_json(out_dir / "cnn_metrics.json", metrics)
    pd.DataFrame(history).to_csv(out_dir / "cnn_training_history.csv", index=False)
    save_predictions(out_dir / "cnn_test_predictions.csv", [f"test_{i:04d}" for i in range(len(y_test))], y_test, p_test)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_mean": train_mean,
            "train_std": train_std,
            "train_mean_image": train_mean_image,
            "seed": seed,
            "model": "SmallFaceCNN",
        },
        out_dir / "cnn_model.pt",
    )
    return model, {
        "train_mean": train_mean,
        "train_std": train_std,
        "train_mean_image": train_mean_image,
        "metrics": metrics,
        "x_test_norm": x_test_norm,
    }


def predict_cnn(model: SmallFaceCNN, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(outputs)


def true_confidence(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.where(y.astype(int) == 1, p, 1.0 - p)


def run_cnn_aev(root: Path, data: dict[str, np.ndarray], model: SmallFaceCNN, cnn_info: dict[str, Any], out_dir: Path, device: torch.device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, masks, flat_masks = load_rois(root)
    x_test_flat = data["x_test_flat"].astype(np.float32)
    y_test = data["y_test"].astype(int)
    train_mean = float(cnn_info["train_mean"])
    train_std = float(cnn_info["train_std"])
    mean_image = cnn_info["train_mean_image"].astype(np.float32)
    mean_flat = mean_image.T.reshape(-1)
    x_test_img = data["x_test_images"][:, :, :, 0].astype(np.float32)
    p_orig = predict_cnn(model, normalize_cnn_images(x_test_img, train_mean, train_std), device)
    pred_orig = (p_orig >= 0.5).astype(int)
    true_conf_orig = true_confidence(y_test, p_orig)
    long_rows = []
    wide = {
        "sample_id": [f"test_{i:04d}" for i in range(len(y_test))],
        "y_true": y_test.astype(int),
        "p_class1_original": p_orig.astype(float),
        "y_pred_original": pred_orig.astype(int),
        "true_conf_original": true_conf_orig.astype(float),
    }
    for roi_i, roi_name in enumerate(ROI_SHORT_NAMES):
        masked_flat = x_test_flat.copy()
        masked_flat[:, flat_masks[roi_i]] = mean_flat[flat_masks[roi_i]]
        masked_img = masked_flat.reshape(len(masked_flat), 32, 32).transpose(0, 2, 1).astype(np.float32)
        p_masked = predict_cnn(model, normalize_cnn_images(masked_img, train_mean, train_std), device)
        pred_masked = (p_masked >= 0.5).astype(int)
        true_conf_masked = true_confidence(y_test, p_masked)
        evidence = true_conf_orig - true_conf_masked
        wide[f"p_class1_masked__{roi_name}"] = p_masked.astype(float)
        wide[f"evidence_drop__{roi_name}"] = evidence.astype(float)
        for idx in range(len(y_test)):
            long_rows.append(
                {
                    "sample_id": f"test_{idx:04d}",
                    "roi_index": roi_i + 1,
                    "roi_name": roi_name,
                    "y_true": int(y_test[idx]),
                    "p_class1_original": float(p_orig[idx]),
                    "p_class1_masked": float(p_masked[idx]),
                    "y_pred_original": int(pred_orig[idx]),
                    "y_pred_masked": int(pred_masked[idx]),
                    "true_conf_original": float(true_conf_orig[idx]),
                    "true_conf_masked": float(true_conf_masked[idx]),
                    "evidence_drop": float(evidence[idx]),
                    "prediction_changed": bool(pred_orig[idx] != pred_masked[idx]),
                }
            )
    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide)
    long_df.to_csv(out_dir / "cnn_aev_test_long.csv", index=False)
    wide_df.to_csv(out_dir / "cnn_aev_test_wide.csv", index=False)
    summary = (
        long_df.groupby(["y_true", "roi_index", "roi_name"], as_index=False)
        .agg(
            n=("evidence_drop", "size"),
            mean_evidence_drop=("evidence_drop", "mean"),
            median_evidence_drop=("evidence_drop", "median"),
            std_evidence_drop=("evidence_drop", "std"),
            prediction_change_rate=("prediction_changed", "mean"),
        )
        .sort_values(["y_true", "mean_evidence_drop"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "cnn_aev_summary_by_class.csv", index=False)
    save_summary_bar(summary, "mean_evidence_drop", "CNN + AEV: mean confidence drop", out_dir / "cnn_aev_summary_by_class.png", diverging=True)
    selected_pair_cnn_aev(data, model, cnn_info, masks, flat_masks, out_dir, device)


def sample_image(data: dict[str, np.ndarray], sample_id: str) -> tuple[np.ndarray, int]:
    split, idx_text = sample_id.split("_")
    idx = int(idx_text)
    if split == "train":
        return data["x_train_images"][idx, :, :, 0].astype(np.float32), int(data["y_train"][idx])
    if split == "test":
        return data["x_test_images"][idx, :, :, 0].astype(np.float32), int(data["y_test"][idx])
    raise ValueError(sample_id)


def selected_pair_cnn_aev(
    data: dict[str, np.ndarray],
    model: SmallFaceCNN,
    cnn_info: dict[str, Any],
    masks: np.ndarray,
    flat_masks: np.ndarray,
    out_dir: Path,
    device: torch.device,
) -> None:
    train_mean = float(cnn_info["train_mean"])
    train_std = float(cnn_info["train_std"])
    mean_flat = cnn_info["train_mean_image"].astype(np.float32).T.reshape(-1)
    rows = []
    for sample_id in ["train_0436", "test_0440"]:
        raw, y_true = sample_image(data, sample_id)
        p = float(predict_cnn(model, normalize_cnn_images(raw[None, :, :], train_mean, train_std), device)[0])
        tc = p if y_true == 1 else 1.0 - p
        flat = raw.T.reshape(1, -1).astype(np.float32)
        vals = []
        for roi_i in range(len(flat_masks)):
            masked = flat.copy()
            masked[:, flat_masks[roi_i]] = mean_flat[flat_masks[roi_i]]
            masked_img = masked.reshape(1, 32, 32).transpose(0, 2, 1)
            pm = float(predict_cnn(model, normalize_cnn_images(masked_img, train_mean, train_std), device)[0])
            tcm = pm if y_true == 1 else 1.0 - pm
            vals.append(tc - tcm)
        row = {
            "sample_id": sample_id,
            "y_true": y_true,
            "p_class0": 1.0 - p,
            "p_class1": p,
            "y_pred": int(p >= 0.5),
        }
        row.update({f"aev_{name}": float(val) for name, val in zip(ROI_SHORT_NAMES, vals)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "selected_visual_pair_cnn_aev.csv", index=False)
    selected_pair_vector_figure(data, rows, "CNN + AEV selected visual pair", "AEV score", out_dir / "selected_visual_pair_cnn_aev.png")


def gradcam_for_sample(model: SmallFaceCNN, x: np.ndarray, target_class: int, device: torch.device) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(x[None, :, :, :]).to(device)
    xb.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logit, features = model(xb, return_features=True)
    features.retain_grad()
    score = logit[0] if target_class == 1 else -logit[0]
    score.backward()
    grads = features.grad[0].detach()
    acts = features[0].detach()
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu(torch.sum(weights[:, None, None] * acts, dim=0))
    cam = torch.nn.functional.interpolate(cam[None, None, :, :], size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
    cam_np = cam.detach().cpu().numpy().astype(np.float32)
    if float(cam_np.max()) > 0:
        cam_np /= float(cam_np.max())
    return cam_np


def run_gradcam_roi(root: Path, data: dict[str, np.ndarray], model: SmallFaceCNN, cnn_info: dict[str, Any], out_dir: Path, device: torch.device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, masks, _ = load_rois(root)
    x_test, y_test = make_cnn_arrays(data)[2:]
    train_mean = float(cnn_info["train_mean"])
    train_std = float(cnn_info["train_std"])
    x_test_norm = normalize_cnn_images(x_test, train_mean, train_std)
    p_test = predict_cnn(model, x_test_norm, device)
    rows_long = []
    rows_wide = []
    for idx in range(len(y_test)):
        if idx % 200 == 0:
            print(f"Grad-CAM test sample {idx}/{len(y_test)}")
        cam = gradcam_for_sample(model, x_test_norm[idx], int(y_test[idx]), device)
        total = float(cam.sum())
        row = {
            "sample_id": f"test_{idx:04d}",
            "y_true": int(y_test[idx]),
            "p_class1": float(p_test[idx]),
            "y_pred": int(p_test[idx] >= 0.5),
        }
        for roi_i, roi_name in enumerate(ROI_SHORT_NAMES):
            vals = cam[masks[roi_i]]
            mass = float(vals.sum() / total) if total > 1e-8 else 0.0
            mean_val = float(vals.mean())
            row[f"gradcam_mass_fraction__{roi_name}"] = mass
            row[f"gradcam_mean__{roi_name}"] = mean_val
            rows_long.append(
                {
                    "sample_id": f"test_{idx:04d}",
                    "roi_index": roi_i + 1,
                    "roi_name": roi_name,
                    "y_true": int(y_test[idx]),
                    "p_class1": float(p_test[idx]),
                    "y_pred": int(p_test[idx] >= 0.5),
                    "gradcam_mass_fraction": mass,
                    "gradcam_mean": mean_val,
                }
            )
        rows_wide.append(row)
    long_df = pd.DataFrame(rows_long)
    wide_df = pd.DataFrame(rows_wide)
    long_df.to_csv(out_dir / "cnn_gradcam_roi_test_long.csv", index=False)
    wide_df.to_csv(out_dir / "cnn_gradcam_roi_test_wide.csv", index=False)
    summary = (
        long_df.groupby(["y_true", "roi_index", "roi_name"], as_index=False)
        .agg(
            n=("gradcam_mean", "size"),
            mean_gradcam_mean=("gradcam_mean", "mean"),
            median_gradcam_mean=("gradcam_mean", "median"),
            mean_mass_fraction=("gradcam_mass_fraction", "mean"),
        )
        .sort_values(["y_true", "mean_gradcam_mean"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "cnn_gradcam_roi_summary_by_class.csv", index=False)
    save_summary_bar(summary, "mean_gradcam_mean", "CNN + Grad-CAM projected to ROI", out_dir / "cnn_gradcam_roi_summary_by_class.png", diverging=False)
    selected_pair_gradcam(data, model, cnn_info, masks, out_dir, device)


def selected_pair_gradcam(
    data: dict[str, np.ndarray],
    model: SmallFaceCNN,
    cnn_info: dict[str, Any],
    masks: np.ndarray,
    out_dir: Path,
    device: torch.device,
) -> None:
    train_mean = float(cnn_info["train_mean"])
    train_std = float(cnn_info["train_std"])
    rows = []
    for sample_id in ["train_0436", "test_0440"]:
        raw, y_true = sample_image(data, sample_id)
        x_norm = normalize_cnn_images(raw[None, :, :], train_mean, train_std)
        p = float(predict_cnn(model, x_norm, device)[0])
        cam = gradcam_for_sample(model, x_norm[0], y_true, device)
        save_gradcam_overlay(raw, cam, out_dir / f"{sample_id}_gradcam_overlay.png")
        total = float(cam.sum())
        row = {
            "sample_id": sample_id,
            "y_true": y_true,
            "p_class0": 1.0 - p,
            "p_class1": p,
            "y_pred": int(p >= 0.5),
        }
        for roi_i, roi_name in enumerate(ROI_SHORT_NAMES):
            vals = cam[masks[roi_i]]
            row[f"aev_{roi_name}"] = float(vals.mean())
            row[f"gradcam_mass_fraction_{roi_name}"] = float(vals.sum() / total) if total > 1e-8 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "selected_visual_pair_cnn_gradcam_roi.csv", index=False)
    selected_pair_vector_figure(data, rows, "CNN + Grad-CAM ROI selected visual pair", "Area-normalised Grad-CAM ROI intensity", out_dir / "selected_visual_pair_cnn_gradcam_roi.png")


def save_gradcam_overlay(raw: np.ndarray, cam: np.ndarray, output: Path) -> None:
    face = normalize_face(raw).resize((320, 320), Image.Resampling.NEAREST)
    heat = Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize((320, 320), Image.Resampling.BILINEAR)
    heat_arr = np.asarray(heat)
    red = np.zeros((320, 320, 3), dtype=np.uint8)
    red[..., 0] = heat_arr
    red[..., 1] = np.clip(heat_arr * 0.28, 0, 255).astype(np.uint8)
    overlay = Image.blend(face, Image.fromarray(red, mode="RGB"), 0.45)
    overlay.save(output, dpi=(300, 300))


def normalize_face(raw: np.ndarray) -> Image.Image:
    vmin = float(raw.min())
    vmax = float(raw.max())
    arr = np.clip((raw - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")


def save_summary_bar(summary: pd.DataFrame, value_col: str, title: str, output: Path, diverging: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), dpi=220, sharex=False)
    colors = {0: "#007c78", 1: "#d56a00"}
    for ax, cls in zip(axes, [0, 1]):
        df = summary[summary["y_true"] == cls].sort_values("roi_index")
        vals = df[value_col].astype(float).to_numpy()
        bar_colors = ["#d9483b" if v >= 0 else "#2f66d0" for v in vals] if diverging else [colors[cls]] * len(vals)
        ax.barh(df["roi_name"], vals, color=bar_colors)
        ax.invert_yaxis()
        ax.axvline(0, color="#7f8795", lw=0.8)
        ax.set_title(f"Class {cls}", color=colors[cls], fontsize=10, fontweight="bold")
        ax.grid(axis="x", color="#e3e8f0", lw=0.6)
        ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold", color="#0f2545")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def selected_pair_vector_figure(data: dict[str, np.ndarray], rows: list[dict[str, Any]], title: str, xlabel: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.2), dpi=220, gridspec_kw={"width_ratios": [1.0, 2.2]})
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#0f2545")
    faces = [sample_image(data, str(row["sample_id"]))[0] for row in rows]
    vmin = min(float(face.min()) for face in faces)
    vmax = max(float(face.max()) for face in faces)
    for i, row in enumerate(rows):
        raw = faces[i]
        axes[i, 0].imshow(raw, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"{row['sample_id']} | true {row['y_true']} | pred {row['y_pred']}", fontsize=9)
        axes[i, 0].axis("off")
        vals = np.array([float(row[f"aev_{name}"]) for name in ROI_SHORT_NAMES])
        bar_colors = ["#d9483b" if v >= 0 else "#2f66d0" for v in vals]
        axes[i, 1].barh(ROI_SHORT_NAMES, vals, color=bar_colors)
        axes[i, 1].invert_yaxis()
        axes[i, 1].axvline(0, color="#7f8795", lw=0.8)
        axes[i, 1].grid(axis="x", color="#e3e8f0", lw=0.6)
        axes[i, 1].set_xlabel(xlabel, fontsize=8)
        axes[i, 1].tick_params(labelsize=8)
        axes[i, 1].set_title(f"p0={row['p_class0']:.3f}, p1={row['p_class1']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_readme(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(f"- {line}" for line in lines) + "\n", encoding="utf-8")


def copy_cnn_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["cnn_model.pt", "cnn_metrics.json", "cnn_training_history.csv", "cnn_test_predictions.csv"]:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-root", default="outputs/model_comparison")
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    set_seed(args.seed)
    data = load_pd_dbs(root / args.data)
    output_root = (root / args.output_root).resolve()
    linear_dir = output_root / "logistic_regression_linear"
    cnn_aev_dir = output_root / "cnn_aev"
    gradcam_dir = output_root / "cnn_gradcam_projected_roi"
    for directory in [linear_dir, cnn_aev_dir, gradcam_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    workflow_diagram(
        "Logistic Regression / Linear Model Workflow",
        [
            ("Input", "Load 32 x 32 faces and numeric Class 0/Class 1 labels.", PALETTE["blue"]),
            ("Flatten", "Use the supplied 1024-pixel vector with train-only standardisation.", PALETTE["teal"]),
            ("Fit", "Train an L2 logistic regression classifier using scikit-learn.", PALETTE["orange"]),
            ("Report", "Export metrics, predictions, coefficient map, and ROI coefficient summary.", PALETTE["red"]),
        ],
        linear_dir / "workflow_logistic_regression_linear.png",
    )
    linear_metrics = run_logistic(root, data, linear_dir, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workflow_diagram(
        "CNN + AEV Workflow",
        [
            ("Input", "Use 32 x 32 image tensors with train-only intensity normalisation.", PALETTE["blue"]),
            ("CNN", "Train a compact PyTorch CNN with spatial convolutions.", PALETTE["teal"]),
            ("Mask ROIs", "Replace each ROI with the training-set mean face and re-run the CNN.", PALETTE["orange"]),
            ("AEV", "Compute true-class confidence drop for each anatomical ROI.", PALETTE["red"]),
        ],
        cnn_aev_dir / "workflow_cnn_aev.png",
    )
    model, cnn_info = train_cnn(data, cnn_aev_dir, args.seed, device)
    run_cnn_aev(root, data, model, cnn_info, cnn_aev_dir, device)

    workflow_diagram(
        "CNN + Grad-CAM Projected to ROI Workflow",
        [
            ("Input", "Reuse the trained CNN and the same 8 coarse ROI definitions.", PALETTE["blue"]),
            ("Grad-CAM", "Backpropagate target-class score to final convolutional activations.", PALETTE["teal"]),
            ("Project", "Upsample the heatmap and aggregate it into ROI means and mass fractions.", PALETTE["orange"]),
            ("Report", "Export ROI vectors, class summaries, overlays, and selected-pair figures.", PALETTE["red"]),
        ],
        gradcam_dir / "workflow_cnn_gradcam_projected_roi.png",
    )
    copy_cnn_files(cnn_aev_dir, gradcam_dir)
    run_gradcam_roi(root, data, model, cnn_info, gradcam_dir, device)
    write_readme(
        gradcam_dir / "README.md",
        "CNN + Grad-CAM Projected to ROI",
        [
            "Uses the same trained CNN as cnn_aev.",
            "Grad-CAM is computed for the true class and projected into the eight coarse ROI masks.",
            "ROI mean is area-normalised; ROI mass fraction is also exported.",
        ],
    )

    combined = {
        "device": str(device),
        "logistic_regression_linear": linear_metrics,
        "cnn_shared_for_aev_and_gradcam": cnn_info["metrics"],
        "output_root": str(output_root),
    }
    save_json(output_root / "combined_metrics_summary.json", combined)
    print(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""Run advanced CNN-family model baselines.

Outputs are isolated under outputs/advanced_models. The experiment log is
updated only after concrete files and metrics have been written.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.models as tv_models
from PIL import Image, ImageDraw
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))
from dbsface.data.load_pd_dbs import load_pd_dbs
from dbsface.explain.run_cnn_sklearn_method_comparison import PALETTE, ROI_SHORT_NAMES, font, load_rois


CLASS_MAP = {
    0: "Class 0",
    1: "Class 1",
}

MODEL_LABELS = {
    "resnet18": "ResNet18 scratch 32x32",
    "efficientnet_b0": "EfficientNet-B0 scratch 32x32",
    "convnext_tiny": "ConvNeXt-Tiny scratch 32x32",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ece_score(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p <= hi) if i == bins - 1 else (p < hi))
        if np.any(mask):
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
        "ece_10bin": float(ece_score(y_true, p_class1)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def init_log(output_root: Path, args: argparse.Namespace, device: torch.device, reset: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "EXPERIMENT_LOG.md"
    if reset or not log_path.exists():
        text = [
            "# Advanced Model Experiment Log",
            "",
            f"Created: {datetime.now().isoformat(timespec='seconds')}",
            f"Output root: `{output_root}`",
            f"Device: `{device}`",
            f"Seed: `{args.seed}`",
            "",
            "## Data",
            "",
            "- Source file: `data/raw/PD_DBS_Data.mat`",
            "- Inputs: 32 x 32 grayscale face images.",
            f"- {CLASS_MAP[0]}.",
            f"- {CLASS_MAP[1]}.",
            "- The original train/test split is preserved; validation is a stratified split from the original training split.",
            "",
            "## Planned Models",
            "",
        ]
        for model_key in args.models:
            text.append(f"- `{model_key}`: {MODEL_LABELS[model_key]}")
        text.extend(["", "## Completed Runs", "", "No model has completed in this log yet.", ""])
        log_path.write_text("\n".join(text), encoding="utf-8")


def append_completed_log(output_root: Path, model_key: str, metrics: dict[str, Any], files: list[Path]) -> None:
    log_path = output_root / "EXPERIMENT_LOG.md"
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Advanced Model Experiment Log\n\n"
    text = text.replace("No model has completed in this log yet.\n", "")
    lines = [
        "",
        f"### {model_key}",
        "",
        f"- Completed: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model: {metrics['model']}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- AUROC: {metrics['auroc']:.6f}",
        f"- AUPRC Class 1: {metrics['auprc_class1']:.6f}",
        f"- F1 Class 1: {metrics['f1_class1']:.6f}",
        f"- ECE 10-bin: {metrics['ece_10bin']:.6f}",
        f"- Confusion matrix: TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}",
        "- Saved files:",
    ]
    for path in files:
        lines.append(f"  - `{path}`")
    lines.append("")
    log_path.write_text(text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def update_benchmark_summary(output_root: Path, model_key: str, metrics: dict[str, Any]) -> None:
    path = output_root / "advanced_benchmark_summary.csv"
    row = {
        "model_key": model_key,
        "model": metrics["model"],
        "accuracy": metrics["accuracy"],
        "auroc": metrics["auroc"],
        "auprc_class1": metrics["auprc_class1"],
        "f1_class1": metrics["f1_class1"],
        "precision_class1": metrics["precision_class1"],
        "recall_class1": metrics["recall_class1"],
        "brier": metrics["brier"],
        "ece_10bin": metrics["ece_10bin"],
        "tn": metrics["confusion_matrix"]["tn"],
        "fp": metrics["confusion_matrix"]["fp"],
        "fn": metrics["confusion_matrix"]["fn"],
        "tp": metrics["confusion_matrix"]["tp"],
        "epochs_run": metrics.get("epochs_run"),
        "best_validation_loss": metrics.get("best_validation_loss"),
    }
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["model_key"] != model_key]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.sort_values("accuracy", ascending=False).to_csv(path, index=False)


def write_method_readme(out_dir: Path, model_key: str, metrics: dict[str, Any], extra_lines: list[str]) -> None:
    lines = [
        f"# {MODEL_LABELS[model_key]}",
        "",
        "## Status",
        "",
        f"- Completed: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model key: `{model_key}`",
        f"- Accuracy: `{metrics['accuracy']:.6f}`",
        f"- AUROC: `{metrics['auroc']:.6f}`",
        f"- AUPRC Class 1: `{metrics['auprc_class1']:.6f}`",
        f"- F1 Class 1: `{metrics['f1_class1']:.6f}`",
        "",
        "## Files",
        "",
        "- `metrics.json`: frozen test metrics.",
        "- `test_predictions.csv`: per-sample predictions.",
        "- `training_history.csv`: train/validation history.",
        "- `roi_aev_summary_by_class.csv`: class-wise ROI mask-out sensitivity.",
        "- `roi_aev_test_long.csv` and `roi_aev_test_wide.csv`: per-sample ROI mask-out sensitivity.",
        "- `selected_pair_aev.png`: train_0436 / test_0440 ROI evidence comparison.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {line}" for line in extra_lines)
    lines.append("")
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def make_arrays(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = data["x_train_images"][:, :, :, 0].astype(np.float32)
    x_test = data["x_test_images"][:, :, :, 0].astype(np.float32)
    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)
    return x_train, y_train, x_test, y_test


def normalise(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x - mean) / max(std, 1e-6)).astype(np.float32)[:, None, :, :]


def build_model(model_key: str, masks: np.ndarray | None = None) -> nn.Module:
    if model_key == "resnet18":
        model = tv_models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if model_key == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, old.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        return model
    if model_key == "convnext_tiny":
        model = tv_models.convnext_tiny(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, old.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
        return model
    raise ValueError(f"Unknown model key: {model_key}")



def roi_dropout(xb: torch.Tensor, masks: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return xb
    batch = xb.shape[0]
    device = xb.device
    out = xb.clone()
    apply = torch.rand(batch, device=device) < p
    if not torch.any(apply):
        return out
    roi_ids = torch.randint(0, masks.shape[0], (batch,), device=device)
    for i in torch.where(apply)[0]:
        out[i] = out[i] * (1.0 - masks[roi_ids[i]].to(device))
    return out


def model_logits(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    out = model(xb)
    if isinstance(out, tuple):
        out = out[0]
    return out.squeeze(-1)


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            batch_p = torch.sigmoid(model_logits(model, xb)).detach().cpu().numpy()
            values.append(np.atleast_1d(batch_p))
    return np.concatenate(values).reshape(-1)


def predict_attention(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray | None:
    return None


def train_model(
    model_key: str,
    data: dict[str, np.ndarray],
    masks: np.ndarray,
    seed: int,
    device: torch.device,
    out_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    x_train, y_train, x_test, y_test = make_arrays(data)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, val_idx = next(split.split(x_train.reshape(len(x_train), -1), y_train))
    mean = float(x_train[fit_idx].mean())
    std = float(x_train[fit_idx].std())
    train_mean_image = x_train[fit_idx].mean(axis=0).astype(np.float32)

    x_fit = normalise(x_train[fit_idx], mean, std)
    x_val = normalise(x_train[val_idx], mean, std)
    x_test_norm = normalise(x_test, mean, std)
    y_fit = y_train[fit_idx].astype(np.float32)
    y_val = y_train[val_idx].astype(np.float32)

    model = build_model(model_key, masks=None).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)), batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    n_pos = float(np.sum(y_fit == 1))
    n_neg = float(np.sum(y_fit == 0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device))
    lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    masks_t = torch.from_numpy(masks.astype(np.float32))[:, None, :, :].to(device)
    roi_dropout_p = 0.0

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    wait = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model_logits(model, xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model_logits(model, val_x)
            val_loss = float(criterion(val_logits, val_y).detach().cpu())
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
    p_test = predict(model, x_test_norm, device)
    metrics = metric_dict(y_test, p_test, MODEL_LABELS[model_key])
    metrics["epochs_run"] = int(history[-1]["epoch"])
    metrics["best_validation_loss"] = float(best_val_loss)
    metrics["device"] = str(device)
    metrics["normalization"] = {"train_mean": mean, "train_std": std}
    metrics["validation_final"] = history[-1]
    info = {
        "train_mean": mean,
        "train_std": std,
        "train_mean_image": train_mean_image,
        "x_test_norm": x_test_norm,
        "y_test": y_test,
        "history": history,
        "p_test": p_test,
    }
    return model, metrics, info


def save_predictions(path: Path, y_true: np.ndarray, p: np.ndarray) -> None:
    y_pred = (p >= 0.5).astype(int)
    pd.DataFrame(
        {
            "sample_id": [f"test_{i:04d}" for i in range(len(y_true))],
            "y_true": y_true.astype(int),
            "p_class0": (1.0 - p).astype(float),
            "p_class1": p.astype(float),
            "y_pred": y_pred.astype(int),
            "correct": (y_pred == y_true).astype(int),
        }
    ).to_csv(path, index=False)


def true_conf(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.where(y.astype(int) == 1, p, 1.0 - p)


def run_roi_aev(
    model: nn.Module,
    data: dict[str, np.ndarray],
    info: dict[str, Any],
    masks: np.ndarray,
    flat_masks: np.ndarray,
    out_dir: Path,
    device: torch.device,
) -> tuple[Path, Path, Path]:
    x_test_flat = data["x_test_flat"].astype(np.float32)
    y_test = data["y_test"].astype(int)
    mean_flat = info["train_mean_image"].astype(np.float32).T.reshape(-1)
    p_orig = info["p_test"].astype(float)
    tc_orig = true_conf(y_test, p_orig)
    wide: dict[str, Any] = {
        "sample_id": [f"test_{i:04d}" for i in range(len(y_test))],
        "y_true": y_test.astype(int),
        "p_class1_original": p_orig,
        "y_pred_original": (p_orig >= 0.5).astype(int),
        "true_conf_original": tc_orig,
    }
    long_rows = []
    for roi_i, roi_name in enumerate(ROI_SHORT_NAMES):
        masked_flat = x_test_flat.copy()
        masked_flat[:, flat_masks[roi_i]] = mean_flat[flat_masks[roi_i]]
        masked_img = masked_flat.reshape(len(masked_flat), 32, 32).transpose(0, 2, 1).astype(np.float32)
        x_masked = normalise(masked_img, float(info["train_mean"]), float(info["train_std"]))
        p_masked = predict(model, x_masked, device)
        tc_masked = true_conf(y_test, p_masked)
        evidence = tc_orig - tc_masked
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
                    "true_conf_original": float(tc_orig[idx]),
                    "true_conf_masked": float(tc_masked[idx]),
                    "evidence_drop": float(evidence[idx]),
                    "prediction_changed": bool((p_orig[idx] >= 0.5) != (p_masked[idx] >= 0.5)),
                }
            )
    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide)
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
    long_path = out_dir / "roi_aev_test_long.csv"
    wide_path = out_dir / "roi_aev_test_wide.csv"
    summary_path = out_dir / "roi_aev_summary_by_class.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_roi_summary_plot(summary, out_dir / "roi_aev_summary_by_class.png")
    save_selected_pair_plot(data, model, info, masks, flat_masks, out_dir / "selected_pair_aev.png", device)
    return long_path, wide_path, summary_path


def save_attention_outputs(
    model: nn.Module,
    info: dict[str, Any],
    out_dir: Path,
    device: torch.device,
) -> list[Path]:
    attention = predict_attention(model, info["x_test_norm"], device)
    if attention is None:
        return []
    y_test = info["y_test"].astype(int)
    rows = []
    for idx in range(attention.shape[0]):
        for roi_i, roi_name in enumerate(ROI_SHORT_NAMES):
            rows.append(
                {
                    "sample_id": f"test_{idx:04d}",
                    "y_true": int(y_test[idx]),
                    "roi_index": roi_i + 1,
                    "roi_name": roi_name,
                    "attention": float(attention[idx, roi_i]),
                }
            )
    long_df = pd.DataFrame(rows)
    summary = (
        long_df.groupby(["y_true", "roi_index", "roi_name"], as_index=False)
        .agg(n=("attention", "size"), mean_attention=("attention", "mean"), median_attention=("attention", "median"))
        .sort_values(["y_true", "mean_attention"], ascending=[True, False])
    )
    wide = pd.DataFrame({"sample_id": [f"test_{i:04d}" for i in range(attention.shape[0])], "y_true": y_test})
    for roi_i, roi_name in enumerate(ROI_SHORT_NAMES):
        wide[f"attention__{roi_name}"] = attention[:, roi_i]
    paths = [
        out_dir / "roi_attention_test_long.csv",
        out_dir / "roi_attention_test_wide.csv",
        out_dir / "roi_attention_summary_by_class.csv",
    ]
    long_df.to_csv(paths[0], index=False)
    wide.to_csv(paths[1], index=False)
    summary.to_csv(paths[2], index=False)
    save_attention_summary_plot(summary, out_dir / "roi_attention_summary_by_class.png")
    save_architecture_diagram(out_dir / "roi_attention_architecture.png")
    return paths


def save_training_curve(history: list[dict[str, Any]], output: Path) -> None:
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    ax1.plot(df["epoch"], df["train_loss"], label="train loss", color="#0f2545", lw=1.8)
    ax1.plot(df["epoch"], df["val_loss"], label="val loss", color="#d9483b", lw=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(axis="y", color="#e3e8f0", lw=0.6)
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_auroc"], label="val AUROC", color="#007c78", lw=1.4, ls="--")
    ax2.set_ylabel("Validation AUROC")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_roi_summary_plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=220, sharex=False)
    for ax, cls in zip(axes, [0, 1]):
        df = summary[summary["y_true"] == cls].sort_values("roi_index")
        vals = df["mean_evidence_drop"].astype(float).to_numpy()
        colors = ["#d9483b" if v >= 0 else "#2f66d0" for v in vals]
        ax.barh(df["roi_name"], vals, color=colors)
        ax.axvline(0, color="#7f8795", lw=0.8)
        ax.invert_yaxis()
        ax.set_title(f"Class {cls}", fontsize=10, fontweight="bold", color="#0f2545")
        ax.grid(axis="x", color="#e3e8f0", lw=0.6)
        ax.tick_params(labelsize=8)
    fig.suptitle("ROI mask-out perturbation sensitivity", fontsize=12, fontweight="bold", color="#0f2545")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_attention_summary_plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=220, sharex=False)
    for ax, cls in zip(axes, [0, 1]):
        df = summary[summary["y_true"] == cls].sort_values("roi_index")
        ax.barh(df["roi_name"], df["mean_attention"].astype(float), color="#007c78")
        ax.invert_yaxis()
        ax.set_title(f"Class {cls}", fontsize=10, fontweight="bold", color="#0f2545")
        ax.grid(axis="x", color="#e3e8f0", lw=0.6)
        ax.tick_params(labelsize=8)
    fig.suptitle("Internal ROI attention", fontsize=12, fontweight="bold", color="#0f2545")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def sample_image(data: dict[str, np.ndarray], sample_id: str) -> tuple[np.ndarray, int]:
    split, idx_text = sample_id.split("_")
    idx = int(idx_text)
    if split == "train":
        return data["x_train_images"][idx, :, :, 0].astype(np.float32), int(data["y_train"][idx])
    if split == "test":
        return data["x_test_images"][idx, :, :, 0].astype(np.float32), int(data["y_test"][idx])
    raise ValueError(sample_id)


def save_selected_pair_plot(
    data: dict[str, np.ndarray],
    model: nn.Module,
    info: dict[str, Any],
    masks: np.ndarray,
    flat_masks: np.ndarray,
    output: Path,
    device: torch.device,
) -> None:
    rows = []
    mean_flat = info["train_mean_image"].astype(np.float32).T.reshape(-1)
    for sample_id in ["train_0436", "test_0440"]:
        raw, y_true = sample_image(data, sample_id)
        x = normalise(raw[None, :, :], float(info["train_mean"]), float(info["train_std"]))
        p = float(predict(model, x, device)[0])
        tc = p if y_true == 1 else 1.0 - p
        flat = raw.T.reshape(1, -1).astype(np.float32)
        vals = []
        for roi_i in range(len(flat_masks)):
            masked = flat.copy()
            masked[:, flat_masks[roi_i]] = mean_flat[flat_masks[roi_i]]
            masked_img = masked.reshape(1, 32, 32).transpose(0, 2, 1).astype(np.float32)
            pm = float(predict(model, normalise(masked_img, float(info["train_mean"]), float(info["train_std"])), device)[0])
            tcm = pm if y_true == 1 else 1.0 - pm
            vals.append(tc - tcm)
        rows.append({"sample_id": sample_id, "y_true": y_true, "p": p, "pred": int(p >= 0.5), "vals": np.array(vals)})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.2), dpi=220, gridspec_kw={"width_ratios": [1.0, 2.2]})
    faces = [sample_image(data, row["sample_id"])[0] for row in rows]
    vmin = min(float(face.min()) for face in faces)
    vmax = max(float(face.max()) for face in faces)
    for i, row in enumerate(rows):
        axes[i, 0].imshow(faces[i], cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"{row['sample_id']} | true {row['y_true']} | pred {row['pred']}", fontsize=9)
        axes[i, 0].axis("off")
        vals = row["vals"]
        colors = ["#d9483b" if v >= 0 else "#2f66d0" for v in vals]
        axes[i, 1].barh(ROI_SHORT_NAMES, vals, color=colors)
        axes[i, 1].invert_yaxis()
        axes[i, 1].axvline(0, color="#7f8795", lw=0.8)
        axes[i, 1].grid(axis="x", color="#e3e8f0", lw=0.6)
        axes[i, 1].set_xlabel("AEV score")
        axes[i, 1].tick_params(labelsize=8)
        axes[i, 1].set_title(f"p0={1.0-row['p']:.3f}, p1={row['p']:.3f}", fontsize=9)
    fig.suptitle("Selected visual pair: ROI mask-out sensitivity", fontsize=13, fontweight="bold", color="#0f2545")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_architecture_diagram(output: Path) -> None:
    img = Image.new("RGB", (1500, 830), "white")
    draw = ImageDraw.Draw(img)
    draw.text((55, 40), "ROI-aware explainable CNN", font=font(38, True), fill=PALETTE["navy"])
    boxes = [
        ((70, 185, 300, 330), "32 x 32 face", "grayscale input"),
        ((380, 160, 640, 355), "Shared CNN", "spatial feature map"),
        ((725, 80, 1045, 230), "Global branch", "whole-face feature"),
        ((725, 300, 1045, 500), "ROI branch", "8 anatomical masks + masked pooling"),
        ((1120, 185, 1410, 405), "Classifier", "global + ROI attention context"),
        ((1120, 500, 1410, 635), "Outputs", "Class probability + ROI attention vector"),
    ]
    for box, title, body in boxes:
        draw.rounded_rectangle(box, radius=22, fill=(247, 250, 254), outline=PALETTE["grid"], width=3)
        draw.text((box[0] + 22, box[1] + 28), title, font=font(24, True), fill=PALETTE["navy"])
        draw.text((box[0] + 22, box[1] + 72), body, font=font(19), fill=PALETTE["gray"])
    arrows = [((310, 258), (370, 258)), ((650, 258), (715, 155)), ((650, 258), (715, 398)), ((1055, 155), (1110, 260)), ((1055, 398), (1110, 295)), ((1265, 415), (1265, 490))]
    for (x0, y0), (x1, y1) in arrows:
        draw.line((x0, y0, x1, y1), fill=PALETTE["teal"], width=5)
        draw.ellipse((x1 - 5, y1 - 5, x1 + 5, y1 + 5), fill=PALETTE["teal"])
    draw.text((70, 660), "Contribution: classification is not produced by a black-box CNN alone.", font=font(19), fill=PALETTE["gray"])
    draw.text((70, 695), "An anatomical ROI pathway exposes an attention vector that can be compared with external AEV mask-out evidence.", font=font(19), fill=PALETTE["gray"])
    img.save(output, dpi=(300, 300))


def save_performance_plot(output_root: Path) -> None:
    path = output_root / "advanced_benchmark_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=220)
    ax.barh(df["model_key"], df["accuracy"], color="#007c78")
    for y, value in enumerate(df["accuracy"]):
        ax.text(value + 0.004, y, f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.0, 1.03)
    ax.set_xlabel("Test accuracy")
    ax.set_title("Advanced model benchmark", fontsize=12, fontweight="bold", color="#0f2545")
    ax.grid(axis="x", color="#e3e8f0", lw=0.6)
    fig.tight_layout()
    fig.savefig(output_root / "advanced_benchmark_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def run_one_model(args: argparse.Namespace, model_key: str, data: dict[str, np.ndarray], masks: np.ndarray, flat_masks: np.ndarray, device: torch.device, output_root: Path) -> None:
    out_dir = output_root / model_key
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RUNNING.txt").write_text(f"Started {datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")

    print(f"Running {model_key}")
    model, metrics, info = train_model(
        model_key,
        data,
        masks,
        args.seed,
        device,
        out_dir,
        args.max_epochs,
        args.patience,
        args.batch_size,
    )
    save_json(out_dir / "metrics.json", metrics)
    pd.DataFrame(info["history"]).to_csv(out_dir / "training_history.csv", index=False)
    save_training_curve(info["history"], out_dir / "training_curve.png")
    save_predictions(out_dir / "test_predictions.csv", info["y_test"], info["p_test"])
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_key": model_key,
            "train_mean": info["train_mean"],
            "train_std": info["train_std"],
            "train_mean_image": info["train_mean_image"],
            "seed": args.seed,
        },
        out_dir / "model.pt",
    )
    aev_files = run_roi_aev(model, data, info, masks, flat_masks, out_dir, device)
    attention_files = save_attention_outputs(model, info, out_dir, device)
    write_method_readme(
        out_dir,
        model_key,
        metrics,
        [
            "All reported metrics are frozen test-split metrics.",
            "ROI evidence is computed by replacing one ROI at a time with the train-set mean image and re-running the same trained model.",
            "ROI evidence is exported as an external AEV validation analysis for the trained model.",
        ],
    )
    (out_dir / "RUNNING.txt").unlink(missing_ok=True)
    update_benchmark_summary(output_root, model_key, metrics)
    save_performance_plot(output_root)
    completed_files = [
        out_dir / "metrics.json",
        out_dir / "test_predictions.csv",
        out_dir / "training_history.csv",
        out_dir / "training_curve.png",
        out_dir / "model.pt",
        *aev_files,
        out_dir / "roi_aev_summary_by_class.png",
        out_dir / "selected_pair_aev.png",
        out_dir / "README.md",
    ]
    completed_files.extend(attention_files)
    append_completed_log(output_root, model_key, metrics, completed_files)
    print(json.dumps({"model_key": model_key, "metrics": metrics}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-root", default="outputs/advanced_models")
    parser.add_argument("--models", nargs="+", default=["resnet18", "efficientnet_b0", "convnext_tiny"], choices=list(MODEL_LABELS))
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--max-epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reset-log", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = (root / args.output_root).resolve()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_log(output_root, args, device, reset=args.reset_log)
    data = load_pd_dbs(root / args.data)
    _, masks, flat_masks = load_rois(root)

    for model_key in args.models:
        run_one_model(args, model_key, data, masks, flat_masks, device, output_root)
    save_performance_plot(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






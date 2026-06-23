"""Shared utilities for the retained ROI-AMoE model.

This module contains only the final AMoE architecture and generic helper
functions used by the final repeated-seed, export, and explanation scripts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from torch import nn


METRIC_COLS = ["accuracy", "balanced_accuracy", "f1_class1", "auroc", "auprc_class1", "brier", "ece_10bin"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_arrays(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        data["x_train_images"][:, :, :, 0].astype(np.float32),
        data["y_train"].astype(int),
        data["x_test_images"][:, :, :, 0].astype(np.float32),
        data["y_test"].astype(int),
    )


def normalise(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x - mean) / max(std, 1e-6)).astype(np.float32)[:, None, :, :]


def ece_score(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        idx = (p >= left) & ((p <= right) if i == bins - 1 else (p < right))
        if np.any(idx):
            ece += float(np.mean(idx) * abs(np.mean(y_true[idx]) - np.mean(p[idx])))
    return ece


def metric_dict(y_true: np.ndarray, p_class1: np.ndarray, model_key: str, model_name: str) -> dict[str, Any]:
    y_pred = (p_class1 >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model_key": model_key,
        "model": model_name,
        "n": int(len(y_true)),
        "positive_class": "Class 1",
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_class1": float(f1_score(y_true, y_pred, pos_label=1)),
        "auroc": float(roc_auc_score(y_true, p_class1)),
        "auprc_class1": float(average_precision_score(y_true, p_class1)),
        "brier": float(brier_score_loss(y_true, p_class1)),
        "ece_10bin": float(ece_score(y_true, p_class1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


class AMoEComponentModel(nn.Module):
    """Final AMoE implementation with optional branches for ablation runs."""

    def __init__(self, masks: np.ndarray, config: dict[str, Any], channels: int = 96) -> None:
        super().__init__()
        self.config = config
        self.use_global = bool(config["use_global"])
        self.use_roi = bool(config["use_roi"])
        self.learned_gate = bool(config["learned_gate"])
        self.final_mode = str(config.get("final_mode", "additive"))
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        mask_tensor = torch.from_numpy(masks.astype(np.float32))[:, None, :, :]
        mask_small = torch.nn.functional.interpolate(mask_tensor, size=(8, 8), mode="nearest")
        self.register_buffer("roi_masks_8x8", mask_small[:, 0])

        if self.use_global:
            self.global_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Dropout(0.20),
                nn.Linear(channels * 4 * 4, 160),
                nn.SiLU(inplace=True),
            )
            self.global_logit = nn.Linear(160, 1)

        if self.use_roi:
            self.roi_embed = nn.Sequential(nn.Linear(channels, 64), nn.SiLU(inplace=True), nn.Dropout(0.05))
            self.roi_expert = nn.Linear(64, 1)
            if self.learned_gate:
                self.roi_gate = nn.Sequential(nn.Linear(64, 32), nn.SiLU(inplace=True), nn.Linear(32, 1))
            fusion_dim = 64 + 8 + 8
            if self.use_global:
                fusion_dim += 160
            self.fusion = nn.Sequential(
                nn.Dropout(0.20),
                nn.Linear(fusion_dim, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(96, 1),
            )
            self.roi_scale = nn.Parameter(torch.tensor(0.5))
            if self.use_global:
                self.final_roi_scale = nn.Parameter(torch.tensor(float(config.get("final_roi_scale_init", 0.0))))
                self.final_mix = nn.Parameter(torch.tensor(float(config.get("final_mix_init", 0.0))))

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        feat = self.features(x)
        batch = x.shape[0]
        device = x.device

        if self.use_global:
            global_vec = self.global_head(feat)
            global_logit = self.global_logit(global_vec).squeeze(1)
        else:
            global_vec = torch.zeros(batch, 0, device=device, dtype=feat.dtype)
            global_logit = torch.zeros(batch, device=device, dtype=feat.dtype)

        if self.use_roi:
            masks = self.roi_masks_8x8.to(device)
            denom = masks.sum(dim=(1, 2)).clamp_min(1.0)
            roi_vecs = torch.einsum("bchw,rhw->brc", feat, masks) / denom[None, :, None]
            roi_emb = self.roi_embed(roi_vecs)
            roi_logits = self.roi_expert(roi_emb).squeeze(-1)
            if self.learned_gate:
                gate_logits = self.roi_gate(roi_emb).squeeze(-1)
                gates = torch.softmax(gate_logits, dim=1)
            else:
                gates = torch.full_like(roi_logits, 1.0 / roi_logits.shape[1])
            roi_context = torch.sum(gates[:, :, None] * roi_emb, dim=1)
            fusion_inputs = [roi_context, roi_logits, gates]
            if self.use_global:
                fusion_inputs.insert(0, global_vec)
            fusion_logit = self.fusion(torch.cat(fusion_inputs, dim=1)).squeeze(1)
            weighted_roi_logit = torch.sum(gates * roi_logits, dim=1)
            roi_logit = fusion_logit + torch.sigmoid(self.roi_scale) * weighted_roi_logit
        else:
            roi_logits = torch.zeros(batch, 8, device=device, dtype=feat.dtype)
            gates = torch.full((batch, 8), 1.0 / 8.0, device=device, dtype=feat.dtype)
            roi_logit = torch.zeros(batch, device=device, dtype=feat.dtype)

        if self.use_global and self.use_roi:
            if self.final_mode == "additive":
                final_logit = global_logit + roi_logit
            elif self.final_mode == "residual_sigmoid":
                final_logit = global_logit + torch.sigmoid(self.final_roi_scale) * roi_logit
            elif self.final_mode == "residual_tanh":
                final_logit = global_logit + torch.tanh(self.final_roi_scale) * roi_logit
            elif self.final_mode == "fusion_only":
                final_logit = roi_logit
            elif self.final_mode == "blend":
                mix = torch.sigmoid(self.final_mix)
                final_logit = (1.0 - mix) * global_logit + mix * roi_logit
            else:
                raise ValueError(f"Unsupported final_mode: {self.final_mode}")
        elif self.use_global:
            final_logit = global_logit
        else:
            final_logit = roi_logit

        if return_parts:
            return {
                "logit": final_logit,
                "global_logit": global_logit,
                "roi_logits": roi_logits,
                "gates": gates,
            }
        return final_logit


def predict_parts(model: AMoEComponentModel, x: np.ndarray, device: torch.device, batch_size: int = 128) -> dict[str, np.ndarray]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {"p": [], "gates": [], "roi_logits": [], "global_logit": []}
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            parts = model(xb, return_parts=True)
            outputs["p"].append(torch.sigmoid(parts["logit"]).detach().cpu().numpy().reshape(-1))
            outputs["gates"].append(parts["gates"].detach().cpu().numpy())
            outputs["roi_logits"].append(parts["roi_logits"].detach().cpu().numpy())
            outputs["global_logit"].append(parts["global_logit"].detach().cpu().numpy().reshape(-1))
    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}


def apply_image_augmentation(x: torch.Tensor, mode: str, mean: float, std: float) -> torch.Tensor:
    if mode == "none":
        return x
    out = x
    b = out.shape[0]
    device = out.device
    if mode == "mild":
        angles = (torch.rand(b, device=device) * 10.0 - 5.0) * np.pi / 180.0
        shifts = torch.rand(b, 2, device=device) * 4.0 - 2.0
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        theta = torch.zeros((b, 2, 3), device=device, dtype=out.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 0, 2] = shifts[:, 0] * 2.0 / 32.0
        theta[:, 1, 2] = shifts[:, 1] * 2.0 / 32.0
        grid = F.affine_grid(theta, out.shape, align_corners=False)
        out = F.grid_sample(out, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        contrast = 1.0 + (torch.rand((b, 1, 1, 1), device=device) * 0.16 - 0.08)
        brightness = torch.rand((b, 1, 1, 1), device=device) * 0.16 - 0.08
        out = out * contrast + brightness
        out = out + torch.randn_like(out) * 0.025
        blur_mask = torch.rand(b, device=device) < 0.20
        if torch.any(blur_mask):
            blurred = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
            out = torch.where(blur_mask[:, None, None, None], blurred, out)
    else:
        raise ValueError(f"Unsupported final AMoE augmentation mode: {mode}")
    return out


def save_predictions(path: Path, y_true: np.ndarray, p: np.ndarray) -> None:
    y_pred = (p >= 0.5).astype(int)
    pd.DataFrame(
        {
            "sample_id": [f"test_{i:04d}" for i in range(len(y_true))],
            "y_true": y_true,
            "p_class0": 1.0 - p,
            "p_class1": p,
            "y_pred": y_pred,
            "correct": (y_pred == y_true).astype(int),
        }
    ).to_csv(path, index=False)


def save_training_curve(history: list[dict[str, Any]], output: Path) -> None:
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    ax1.plot(df["epoch"], df["train_loss"], label="train loss", color="#0f2545", lw=1.7)
    ax1.plot(df["epoch"], df["val_loss"], label="val loss", color="#d9483b", lw=1.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(axis="y", color="#e3e8f0", lw=0.6)
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_auroc"], label="val AUROC", color="#007c78", ls="--", lw=1.3)
    ax2.set_ylabel("Validation AUROC")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in df.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "n_seeds": int(len(group))}
        for col in METRIC_COLS:
            values = group[col].astype(float)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)



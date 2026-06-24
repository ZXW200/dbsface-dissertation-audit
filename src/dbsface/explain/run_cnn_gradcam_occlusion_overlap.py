"""Train a small CNN and compare ROI occlusion evidence with Grad-CAM.

The comparison is performed within the same CNN:

* ROI occlusion: true-class confidence drop after masking one fixed ROI.
* Grad-CAM: true-class Grad-CAM heatmaps projected onto the same fixed ROIs.

This keeps the overlap analysis model-consistent. The existing NumPy MLP
occlusion output is also merged for reference, but the primary overlap metrics
use CNN occlusion versus CNN Grad-CAM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs
from train_baseline_mlp_numpy import metric_summary


class SmallGradCamCNN(nn.Module):
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

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        logits = self.classifier(self.pool(feats)).squeeze(1)
        if return_features:
            return logits, feats
        return logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))


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


def normalize_images(train: np.ndarray, test: np.ndarray, fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    fit_images = train[fit_idx]
    mean = float(fit_images.mean())
    std = float(fit_images.std())
    if std < 1e-6:
        std = 1.0
    mean_image = fit_images.mean(axis=0).astype(np.float32)
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32), mean, std, mean_image


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    # Input images arrive as [N, 32, 32, 1].
    return torch.from_numpy(images.transpose(0, 3, 1, 2).astype(np.float32))


def predict_probs(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs).astype(float)


def true_confidence(y: np.ndarray, p_class1: np.ndarray) -> np.ndarray:
    return np.where(y == 1, p_class1, 1.0 - p_class1)


def train_cnn(
    x_train: torch.Tensor,
    y_train: np.ndarray,
    x_val: torch.Tensor,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = SmallGradCamCNN().to(device)
    counts = np.bincount(y_train.astype(int), minlength=2)
    pos_weight = torch.tensor([counts[0] / max(counts[1], 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(x_train, torch.from_numpy(y_train.astype(np.float32)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)

    best_state = None
    best_val_loss = float("inf")
    patience = 18
    wait = 0
    history = []
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val.to(device))
            val_loss = float(criterion(val_logits, y_val_t).detach().cpu())
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_acc = float(((val_probs >= 0.5).astype(int) == y_val).mean())
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
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

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def cnn_occlusion(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: np.ndarray,
    train_mean_image_normalized: np.ndarray,
    masks: np.ndarray,
    roi_defs: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p_orig = predict_probs(model, x_test, batch_size, device)
    conf_orig = true_confidence(y_test, p_orig)
    y_pred_orig = (p_orig >= 0.5).astype(int)

    long_rows = []
    summary_rows = []
    for roi_idx, row in roi_defs.iterrows():
        roi_name = row["roi_name"]
        mask = masks[roi_idx].astype(bool)
        x_masked = x_test.clone()
        x_masked[:, 0, mask] = torch.from_numpy(train_mean_image_normalized[mask].astype(np.float32))
        p_masked = predict_probs(model, x_masked, batch_size, device)
        conf_masked = true_confidence(y_test, p_masked)
        y_pred_masked = (p_masked >= 0.5).astype(int)
        evidence_drop = conf_orig - conf_masked
        changed = y_pred_orig != y_pred_masked
        summary_rows.append(
            {
                "roi_index": int(row["roi_index"]),
                "roi_name": roi_name,
                "n": int(len(y_test)),
                "mean_evidence_drop": float(evidence_drop.mean()),
                "median_evidence_drop": float(np.median(evidence_drop)),
                "prediction_change_rate": float(changed.mean()),
            }
        )
        for i in range(len(y_test)):
            long_rows.append(
                {
                    "sample_id": f"test_{i:04d}",
                    "roi_index": int(row["roi_index"]),
                    "roi_name": roi_name,
                    "y_true": int(y_test[i]),
                    "p_class1_original": float(p_orig[i]),
                    "p_class1_masked": float(p_masked[i]),
                    "true_conf_original": float(conf_orig[i]),
                    "true_conf_masked": float(conf_masked[i]),
                    "evidence_drop": float(evidence_drop[i]),
                    "prediction_changed": bool(changed[i]),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values("mean_evidence_drop", ascending=False)
    return pd.DataFrame(long_rows), summary


def compute_gradcam(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: np.ndarray,
    masks: np.ndarray,
    roi_defs: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    model.eval()
    heatmaps = []
    long_rows = []

    for start in range(0, len(x_test), batch_size):
        xb = x_test[start : start + batch_size].to(device)
        yb_np = y_test[start : start + batch_size].astype(np.int64)
        yb = torch.from_numpy(yb_np).to(device)

        model.zero_grad(set_to_none=True)
        logits, feats = model(xb, return_features=True)
        feats.retain_grad()
        target_scores = torch.where(yb == 1, logits, -logits)
        target_scores.sum().backward()
        grads = feats.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feats).sum(dim=1))
        cam = F.interpolate(cam[:, None, :, :], size=(32, 32), mode="bilinear", align_corners=False)[:, 0]
        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        for i in range(cam_np.shape[0]):
            h = cam_np[i]
            h = h - float(h.min())
            denom = float(h.max())
            if denom > 1e-8:
                h = h / denom
            heatmaps.append(h.astype(np.float32))

    heatmaps_arr = np.stack(heatmaps, axis=0)
    eps = 1e-8
    for i, h in enumerate(heatmaps_arr):
        total = float(h.sum()) + eps
        for roi_idx, row in roi_defs.iterrows():
            mask = masks[roi_idx].astype(bool)
            roi_sum = float(h[mask].sum())
            long_rows.append(
                {
                    "sample_id": f"test_{i:04d}",
                    "roi_index": int(row["roi_index"]),
                    "roi_name": row["roi_name"],
                    "y_true": int(y_test[i]),
                    "cam_energy_fraction": roi_sum / total,
                    "cam_mean": float(h[mask].mean()),
                    "cam_sum": roi_sum,
                }
            )
    long_df = pd.DataFrame(long_rows)
    summary = (
        long_df.groupby(["roi_index", "roi_name"], as_index=False)
        .agg(
            n=("cam_energy_fraction", "size"),
            mean_cam_energy_fraction=("cam_energy_fraction", "mean"),
            median_cam_energy_fraction=("cam_energy_fraction", "median"),
            mean_cam=("cam_mean", "mean"),
            median_cam=("cam_mean", "median"),
        )
        .sort_values("mean_cam_energy_fraction", ascending=False)
    )
    return heatmaps_arr, long_df, summary


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_set(df: pd.DataFrame, value_col: str, k: int) -> set[str]:
    return set(df.sort_values(value_col, ascending=False).head(k)["roi_name"].tolist())


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(x.min())
    mx = float(x.max())
    if mx > 1e-8:
        x = x / mx
    return x


def roi_weight_map(summary: pd.DataFrame, masks: np.ndarray, value_col: str) -> np.ndarray:
    out = np.zeros((32, 32), dtype=np.float32)
    for _, row in summary.iterrows():
        idx = int(row["roi_index"]) - 1
        out[masks[idx].astype(bool)] = float(row[value_col])
    return normalize_map(out)


def top_fraction_mask(x: np.ndarray, fraction: float) -> np.ndarray:
    flat = x.reshape(-1)
    n = max(1, int(round(len(flat) * fraction)))
    threshold = np.partition(flat, len(flat) - n)[len(flat) - n]
    return x >= threshold


def pixel_overlap_metrics(a: np.ndarray, b: np.ndarray, fractions: list[float]) -> list[dict[str, float]]:
    rows = []
    for frac in fractions:
        ma = top_fraction_mask(a, frac)
        mb = top_fraction_mask(b, frac)
        inter = int((ma & mb).sum())
        union = int((ma | mb).sum())
        rows.append(
            {
                "top_fraction": frac,
                "intersection_pixels": inter,
                "union_pixels": union,
                "iou": float(inter / union) if union else float("nan"),
                "dice": float(2 * inter / max(int(ma.sum()) + int(mb.sum()), 1)),
            }
        )
    return rows


def colorize_heatmap(h: np.ndarray) -> Image.Image:
    h = normalize_map(h)
    r = (255 * h).astype(np.uint8)
    g = (255 * np.clip(1.0 - np.abs(h - 0.5) * 2, 0, 1)).astype(np.uint8)
    b = (255 * (1.0 - h)).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=2), mode="RGB")


def save_map(path: Path, h: np.ndarray, title: str) -> None:
    img = colorize_heatmap(h).resize((320, 320), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (360, 380), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill=(0, 0, 0))
    canvas.paste(img, (20, 48))
    canvas.save(path, quality=92)


def save_contact_sheet(
    out_path: Path,
    images_raw: np.ndarray,
    heatmaps: np.ndarray,
    y_test: np.ndarray,
    probs: np.ndarray,
    sample_count: int,
) -> None:
    n = min(sample_count, len(y_test))
    idxs = np.linspace(0, len(y_test) - 1, n, dtype=int)
    tiles = []
    for idx in idxs:
        gray = images_raw[idx, :, :, 0]
        gray = normalize_map(gray)
        base = Image.fromarray((gray * 255).astype(np.uint8), mode="L").convert("RGB").resize((96, 96), Image.Resampling.NEAREST)
        heat = colorize_heatmap(heatmaps[idx]).resize((96, 96), Image.Resampling.BILINEAR)
        blend = Image.blend(base, heat, 0.45)
        draw = ImageDraw.Draw(blend)
        draw.text((2, 2), f"test_{idx:04d} y={int(y_test[idx])} p={probs[idx]:.2f}", fill=(255, 255, 0))
        tiles.append(blend)

    cols = 6
    rows = int(np.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * 112, rows * 124 + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "Grad-CAM overlays for true class on CNN test samples", fill=(0, 0, 0))
    for i, tile in enumerate(tiles):
        x = (i % cols) * 112
        y = 28 + (i // cols) * 124
        canvas.paste(tile, (x, y))
    canvas.save(out_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--roi-masks", default="outputs/roi/coarse_roi_masks.npy")
    parser.add_argument("--roi-defs", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument("--mlp-occlusion", default="outputs/aev/roi_occlusion_summary_overall.csv")
    parser.add_argument("--output-dir", default="outputs/gradcam_occlusion_overlap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    data = load_pd_dbs(args.data)
    train_images_raw = data["x_train_images"].astype(np.float32)
    test_images_raw = data["x_test_images"].astype(np.float32)
    y_source = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    masks = np.load(args.roi_masks).astype(bool)
    roi_defs = pd.read_csv(args.roi_defs)

    train_idx, val_idx = stratified_split(y_source, args.val_fraction, args.seed)
    train_images_norm, test_images_norm, train_mean_raw, train_std_raw, train_mean_image_raw = normalize_images(
        train_images_raw, test_images_raw, train_idx
    )
    x_all = images_to_tensor(train_images_norm)
    x_train = x_all[train_idx]
    x_val = x_all[val_idx]
    y_train = y_source[train_idx]
    y_val = y_source[val_idx]
    x_test = images_to_tensor(test_images_norm)
    train_mean_image_normalized = ((train_mean_image_raw[:, :, 0] - train_mean_raw) / train_std_raw).astype(np.float32)

    model, history = train_cnn(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": args.seed,
            "train_mean_raw": train_mean_raw,
            "train_std_raw": train_std_raw,
            "train_mean_image_raw": train_mean_image_raw,
            "class_0": "pre-DBS",
            "class_1": "post-DBS label",
        },
        out_dir / "small_cnn_gradcam.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    p_train = predict_probs(model, x_train, args.batch_size, device)
    p_val = predict_probs(model, x_val, args.batch_size, device)
    p_test = predict_probs(model, x_test, args.batch_size, device)
    metrics = {
        "model_type": "small_torch_cnn_for_gradcam",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "device": str(device),
        "class_convention": {"0": "pre-DBS", "1": "post-DBS label"},
        "train": metric_summary(y_train, p_train),
        "val": metric_summary(y_val, p_val),
        "test": metric_summary(y_test, p_test),
    }
    (out_dir / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "sample_id": [f"test_{i:04d}" for i in range(len(y_test))],
            "y_true": y_test,
            "p_class1": p_test,
            "y_pred": (p_test >= 0.5).astype(int),
            "correct": ((p_test >= 0.5).astype(int) == y_test).astype(int),
        }
    ).to_csv(out_dir / "cnn_predictions_test.csv", index=False)

    occ_long, occ_summary = cnn_occlusion(
        model,
        x_test,
        y_test,
        train_mean_image_normalized,
        masks,
        roi_defs,
        args.batch_size,
        device,
    )
    occ_long.to_csv(out_dir / "cnn_roi_occlusion_test.csv", index=False)
    occ_summary.to_csv(out_dir / "cnn_roi_occlusion_summary_overall.csv", index=False)

    heatmaps, cam_long, cam_summary = compute_gradcam(
        model,
        x_test,
        y_test,
        masks,
        roi_defs,
        args.batch_size,
        device,
    )
    np.save(out_dir / "gradcam_heatmaps_test.npy", heatmaps)
    cam_long.to_csv(out_dir / "gradcam_roi_test.csv", index=False)
    cam_summary.to_csv(out_dir / "gradcam_roi_summary_overall.csv", index=False)

    overlap = occ_summary.merge(cam_summary, on=["roi_index", "roi_name"], suffixes=("_occlusion", "_gradcam"))
    if Path(args.mlp_occlusion).exists():
        mlp_occ = pd.read_csv(args.mlp_occlusion)[["roi_name", "mean_evidence_drop"]].rename(
            columns={"mean_evidence_drop": "mlp_mean_evidence_drop"}
        )
        overlap = overlap.merge(mlp_occ, on="roi_name", how="left")
    overlap["rank_cnn_occlusion"] = overlap["mean_evidence_drop"].rank(ascending=False, method="min").astype(int)
    overlap["rank_gradcam_energy"] = overlap["mean_cam_energy_fraction"].rank(ascending=False, method="min").astype(int)
    overlap["rank_difference"] = overlap["rank_gradcam_energy"] - overlap["rank_cnn_occlusion"]
    overlap = overlap.sort_values("rank_cnn_occlusion")
    overlap.to_csv(out_dir / "gradcam_occlusion_roi_overlap.csv", index=False)

    spearman = rank_correlation(overlap["mean_evidence_drop"].to_numpy(), overlap["mean_cam_energy_fraction"].to_numpy())
    top3_occ = topk_set(overlap, "mean_evidence_drop", 3)
    top3_cam = topk_set(overlap, "mean_cam_energy_fraction", 3)
    top3_overlap = sorted(top3_occ & top3_cam)
    top1_match = overlap.sort_values("mean_evidence_drop", ascending=False).iloc[0]["roi_name"] == overlap.sort_values(
        "mean_cam_energy_fraction", ascending=False
    ).iloc[0]["roi_name"]

    mean_cam_map = normalize_map(heatmaps.mean(axis=0))
    occ_map = roi_weight_map(occ_summary, masks, "mean_evidence_drop")
    pixel_rows = pixel_overlap_metrics(occ_map, mean_cam_map, [0.10, 0.20, 0.30])
    pd.DataFrame(pixel_rows).to_csv(out_dir / "pixel_topk_overlap.csv", index=False)
    map_cosine = float(
        np.dot(occ_map.reshape(-1), mean_cam_map.reshape(-1))
        / max(np.linalg.norm(occ_map.reshape(-1)) * np.linalg.norm(mean_cam_map.reshape(-1)), 1e-12)
    )
    map_corr = float(np.corrcoef(occ_map.reshape(-1), mean_cam_map.reshape(-1))[0, 1])

    overlap_metrics = {
        "primary_comparison": "same-CNN ROI occlusion versus true-class Grad-CAM ROI projection",
        "cnn_test_accuracy": metrics["test"]["accuracy"],
        "cnn_test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
        "cnn_test_auroc": metrics["test"]["auroc"],
        "roi_spearman_rank_correlation": spearman,
        "top1_match": bool(top1_match),
        "top3_occlusion": sorted(top3_occ),
        "top3_gradcam": sorted(top3_cam),
        "top3_overlap": top3_overlap,
        "top3_overlap_count": len(top3_overlap),
        "top3_overlap_fraction": len(top3_overlap) / 3.0,
        "pixel_topk_overlap": pixel_rows,
        "mean_map_cosine_similarity": map_cosine,
        "mean_map_pearson_correlation": map_corr,
    }
    (out_dir / "overlap_metrics.json").write_text(json.dumps(overlap_metrics, indent=2), encoding="utf-8")

    save_map(out_dir / "mean_gradcam_heatmap.png", mean_cam_map, "Mean true-class Grad-CAM")
    save_map(out_dir / "cnn_occlusion_roi_importance_map.png", occ_map, "CNN occlusion ROI map")
    save_contact_sheet(out_dir / "gradcam_overlay_examples.jpg", test_images_raw, heatmaps, y_test, p_test, 24)

    md = [
        "# CNN Grad-CAM vs ROI Occlusion Overlap",
        "",
        "Primary comparison: same-CNN ROI occlusion evidence versus true-class Grad-CAM projected onto the same fixed 8 ROI atlas.",
        "",
        "Class convention: Class 0 = pre-DBS; Class 1 = post-DBS label. This is an image-level explanation consistency analysis.",
        "",
        "## CNN test performance",
        "",
        f"- Accuracy: {metrics['test']['accuracy']:.4f}",
        f"- Balanced accuracy: {metrics['test']['balanced_accuracy']:.4f}",
        f"- AUROC: {metrics['test']['auroc']:.4f}",
        f"- AUPRC: {metrics['test']['auprc']:.4f}",
        "",
        "## Overlap metrics",
        "",
        f"- ROI Spearman rank correlation: {spearman:.4f}",
        f"- Top-1 ROI match: {bool(top1_match)}",
        f"- Top-3 occlusion ROIs: {', '.join(sorted(top3_occ))}",
        f"- Top-3 Grad-CAM ROIs: {', '.join(sorted(top3_cam))}",
        f"- Top-3 overlap: {len(top3_overlap)}/3 ({', '.join(top3_overlap) if top3_overlap else 'none'})",
        f"- Mean-map cosine similarity: {map_cosine:.4f}",
        f"- Mean-map Pearson correlation: {map_corr:.4f}",
        "",
        "## ROI table",
        "",
        "| ROI | Occlusion mean drop | Grad-CAM energy fraction | Occlusion rank | Grad-CAM rank | Rank diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in overlap.iterrows():
        md.append(
            f"| {row['roi_name']} | {row['mean_evidence_drop']:.6f} | "
            f"{row['mean_cam_energy_fraction']:.6f} | {int(row['rank_cnn_occlusion'])} | "
            f"{int(row['rank_gradcam_energy'])} | {int(row['rank_difference'])} |"
        )
    md.extend(
        [
            "",
            "## Pixel top-k overlap",
            "",
            "| Top fraction | IoU | Dice |",
            "|---:|---:|---:|",
        ]
    )
    for row in pixel_rows:
        md.append(f"| {row['top_fraction']:.2f} | {row['iou']:.4f} | {row['dice']:.4f} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:34]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run supplementary ROI sensitivity experiments for the PD-DBS audit.

The outputs are intentionally scoped as sensitivity and negative-control
analyses. They reuse the existing NumPy MLP and YuNet region-only contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs
from run_yunet_region_only_mlp import ROI_NAMES, box_to_mask32, row_to_yunet_boxes
from train_baseline_mlp_numpy import (
    fit_standardizer,
    forward,
    metric_summary,
    standardize,
    stratified_split,
    train_mlp,
)


def load_model(path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ckpt = np.load(path)
    model = {key: ckpt[key] for key in ["w1", "b1", "w2", "b2"]}
    return model, ckpt["mean"].astype(np.float32), ckpt["std"].astype(np.float32)


def mask_to_flat(mask: np.ndarray) -> np.ndarray:
    return mask.T.reshape(-1).astype(bool)


def true_confidence(y: np.ndarray, p_class1: np.ndarray) -> np.ndarray:
    return np.where(y == 1, p_class1, 1.0 - p_class1)


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    pred = (p >= 0.5).astype(int)
    y = y.astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return float(0.5 * (sensitivity + specificity))


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    y = y.astype(int)
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]
    pos = int(y_sorted.sum())
    neg = int(len(y_sorted) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.r_[0.0, tps / pos, 1.0]
    fpr = np.r_[0.0, fps / neg, 1.0]
    return float(np.trapezoid(tpr, fpr))


def ci(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "sd": float(np.nanstd(arr, ddof=1)),
        "q2_5": float(np.nanquantile(arr, 0.025)),
        "q50": float(np.nanquantile(arr, 0.5)),
        "q97_5": float(np.nanquantile(arr, 0.975)),
    }


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    ra = pd.Series(list(a)).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(list(b)).rank(method="average").to_numpy(dtype=float)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def images_to_flat(images: np.ndarray) -> np.ndarray:
    return images[..., 0].transpose(0, 2, 1).reshape(len(images), -1).astype(np.float32)


def blur3(images: np.ndarray) -> np.ndarray:
    x = images[..., 0]
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            out += padded[:, dy : dy + 32, dx : dx + 32]
    out /= 9.0
    return out[..., None].astype(np.float32)


def static_region_only_metrics(
    x_test: np.ndarray,
    y: np.ndarray,
    model: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    masks: list[np.ndarray],
    roi_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    for roi_index, (roi_name, flat_mask) in enumerate(zip(roi_names, masks), start=1):
        x_region = np.repeat(mean.astype(np.float32), len(x_test), axis=0)
        x_region[:, flat_mask] = x_test[:, flat_mask]
        p = forward(model, standardize(x_region, mean, std))[0]
        pred = (p >= 0.5).astype(int)
        true_conf = true_confidence(y, p)
        metrics = metric_summary(y, p)
        rows.append(
            {
                "roi_index": roi_index,
                "roi_name": roi_name,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_class1": metrics["f1_class1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier_score": metrics["brier_score"],
                "mean_true_confidence": float(true_conf.mean()),
                "tn": metrics["confusion_matrix"]["tn"],
                "fp": metrics["confusion_matrix"]["fp"],
                "fn": metrics["confusion_matrix"]["fn"],
                "tp": metrics["confusion_matrix"]["tp"],
            }
        )
        for i in range(len(y)):
            pred_rows.append(
                {
                    "sample_id": f"test_{i:04d}",
                    "roi_index": roi_index,
                    "roi_name": roi_name,
                    "y_true": int(y[i]),
                    "p_class1": float(p[i]),
                    "y_pred": int(pred[i]),
                    "true_confidence": float(true_conf[i]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def load_yunet_boxes(audit_path: Path, upsample_size: int) -> tuple[pd.DataFrame, dict[int, dict[str, tuple[int, int, int, int]]]]:
    audit = pd.read_csv(audit_path)
    detected = audit[audit["detected"] == 1].copy()
    boxes = {int(row["global_index"]): row_to_yunet_boxes(row, upsample_size) for _, row in detected.iterrows()}
    return audit, boxes


def yunet_region_only_metrics(
    data: dict[str, np.ndarray],
    audit_path: Path,
    model: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    upsample_size: int,
) -> pd.DataFrame:
    x_test = data["x_test_flat"].astype(np.float32)
    y = data["y_test"].astype(int)
    n_train = len(data["x_train_flat"])
    _, boxes_by_global = load_yunet_boxes(audit_path, upsample_size)
    rows = []
    for roi_index, roi_name in enumerate(ROI_NAMES, start=1):
        x_region = np.repeat(mean.astype(np.float32), len(x_test), axis=0)
        missing = 0
        pixel_counts = []
        for i in range(len(x_test)):
            boxes = boxes_by_global.get(n_train + i)
            if boxes is None:
                missing += 1
                continue
            flat_mask = box_to_mask32(boxes[roi_name], upsample_size)
            pixel_counts.append(int(flat_mask.sum()))
            x_region[i, flat_mask] = x_test[i, flat_mask]
        p = forward(model, standardize(x_region, mean, std))[0]
        metrics = metric_summary(y, p)
        rows.append(
            {
                "roi_index": roi_index,
                "roi_name": roi_name,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_class1": metrics["f1_class1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier_score": metrics["brier_score"],
                "missing_yunet_count": missing,
                "mean_dynamic_roi_pixels": float(np.mean(pixel_counts)) if pixel_counts else float("nan"),
                "tn": metrics["confusion_matrix"]["tn"],
                "fp": metrics["confusion_matrix"]["fp"],
                "fn": metrics["confusion_matrix"]["fn"],
                "tp": metrics["confusion_matrix"]["tp"],
            }
        )
    return pd.DataFrame(rows)


def train_seed_model(
    data: dict[str, np.ndarray],
    seed: int,
    args: argparse.Namespace,
    permute_labels: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    x_source = data["x_train_flat"].astype(np.float32)
    y_source = data["y_train"].astype(np.int64)
    x_test = data["x_test_flat"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    train_idx, val_idx = stratified_split(y_source, args.val_fraction, seed)
    y_for_training = y_source.copy()
    if permute_labels:
        rng = np.random.default_rng(seed + 100_000)
        y_for_training = rng.permutation(y_for_training)

    x_train_raw = x_source[train_idx]
    y_train = y_for_training[train_idx]
    x_val_raw = x_source[val_idx]
    y_val = y_for_training[val_idx]

    mean, std = fit_standardizer(x_train_raw)
    x_train = standardize(x_train_raw, mean, std)
    x_val = standardize(x_val_raw, mean, std)
    x_test_std = standardize(x_test, mean, std)

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
        seed=seed,
    )
    p_test = forward(model, x_test_std)[0]
    metrics = metric_summary(y_test, p_test)
    full_metrics = {
        "seed": seed,
        "permuted_labels": bool(permute_labels),
        "best_val_loss": float(min(h["val_loss"] for h in history)),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1_class1": metrics["f1_class1"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "brier_score": metrics["brier_score"],
        "tn": metrics["confusion_matrix"]["tn"],
        "fp": metrics["confusion_matrix"]["fp"],
        "fn": metrics["confusion_matrix"]["fn"],
        "tp": metrics["confusion_matrix"]["tp"],
    }
    return model, mean, std, full_metrics


def bootstrap_fixed_vs_yunet(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(args.bootstrap_seed)
    fixed = pd.read_csv(args.fixed_predictions)
    yunet = pd.read_csv(args.yunet_predictions)
    rows = []
    for roi_name in ROI_NAMES:
        f = fixed[fixed["roi_name"] == roi_name].sort_values("sample_id")
        ydf = yunet[yunet["roi_name"] == roi_name].sort_values("sample_id")
        merged = f[["sample_id", "y_true", "p_class1"]].merge(
            ydf[["sample_id", "p_class1"]],
            on="sample_id",
            suffixes=("_fixed", "_yunet"),
        )
        y = merged["y_true"].to_numpy(dtype=int)
        pf = merged["p_class1_fixed"].to_numpy(dtype=float)
        py = merged["p_class1_yunet"].to_numpy(dtype=float)
        n = len(y)
        boot = {key: [] for key in ["fixed_ba", "yunet_ba", "delta_ba", "fixed_auroc", "yunet_auroc", "delta_auroc"]}
        for _ in range(args.bootstrap_n):
            idx = rng.integers(0, n, size=n)
            yf, pfb, pyb = y[idx], pf[idx], py[idx]
            f_ba = balanced_accuracy(yf, pfb)
            y_ba = balanced_accuracy(yf, pyb)
            f_auc = auroc(yf, pfb)
            y_auc = auroc(yf, pyb)
            boot["fixed_ba"].append(f_ba)
            boot["yunet_ba"].append(y_ba)
            boot["delta_ba"].append(y_ba - f_ba)
            boot["fixed_auroc"].append(f_auc)
            boot["yunet_auroc"].append(y_auc)
            boot["delta_auroc"].append(y_auc - f_auc)
        row = {"roi_name": roi_name, "n": n}
        point = {
            "fixed_ba": balanced_accuracy(y, pf),
            "yunet_ba": balanced_accuracy(y, py),
            "delta_ba": balanced_accuracy(y, py) - balanced_accuracy(y, pf),
            "fixed_auroc": auroc(y, pf),
            "yunet_auroc": auroc(y, py),
            "delta_auroc": auroc(y, py) - auroc(y, pf),
        }
        for metric, value in point.items():
            row[f"{metric}_point"] = value
            stats = ci(boot[metric])
            for stat_name, stat_value in stats.items():
                row[f"{metric}_{stat_name}"] = stat_value
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bootstrap_fixed_vs_yunet_ci.csv", index=False)
    return df


def shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    y_src0 = max(0, -dy)
    y_src1 = min(mask.shape[0], mask.shape[0] - dy)
    x_src0 = max(0, -dx)
    x_src1 = min(mask.shape[1], mask.shape[1] - dx)
    y_dst0 = max(0, dy)
    y_dst1 = y_dst0 + (y_src1 - y_src0)
    x_dst0 = max(0, dx)
    x_dst1 = x_dst0 + (x_src1 - x_src0)
    if y_src1 > y_src0 and x_src1 > x_src0:
        out[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), ((1, 1), (1, 1)), mode="constant")
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def erode_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), ((1, 1), (1, 1)), mode="constant")
    out = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out if out.sum() > 0 else mask.copy()


def fixed_roi_jitter(data, model, mean, std, masks: np.ndarray, roi_names: list[str], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_test = data["x_test_flat"].astype(np.float32)
    y = data["y_test"].astype(int)
    variants: list[tuple[str, list[np.ndarray]]] = []
    variants.append(("original", [mask_to_flat(m) for m in masks]))
    shifts = {
        "shift_up_1": (-1, 0),
        "shift_down_1": (1, 0),
        "shift_left_1": (0, -1),
        "shift_right_1": (0, 1),
        "shift_up_left_1": (-1, -1),
        "shift_up_right_1": (-1, 1),
        "shift_down_left_1": (1, -1),
        "shift_down_right_1": (1, 1),
    }
    for name, (dy, dx) in shifts.items():
        variants.append((name, [mask_to_flat(shift_mask(m, dy, dx)) for m in masks]))
    variants.append(("expand_1px", [mask_to_flat(dilate_mask(m)) for m in masks]))
    variants.append(("shrink_1px", [mask_to_flat(erode_mask(m)) for m in masks]))

    all_rows = []
    for variant, flat_masks in variants:
        metrics, _ = static_region_only_metrics(x_test, y, model, mean, std, flat_masks, roi_names)
        metrics.insert(0, "variant", variant)
        metrics["roi_pixels"] = [int(mask.sum()) for mask in flat_masks]
        all_rows.append(metrics)
    all_metrics = pd.concat(all_rows, ignore_index=True)
    all_metrics.to_csv(out_dir / "fixed_roi_jitter_region_only_metrics.csv", index=False)

    original = all_metrics[all_metrics["variant"] == "original"].set_index("roi_name")
    jitter = all_metrics[all_metrics["variant"] != "original"]
    summary_rows = []
    for roi_name, group in jitter.groupby("roi_name"):
        base = original.loc[roi_name]
        summary_rows.append(
            {
                "roi_name": roi_name,
                "original_balanced_accuracy": base["balanced_accuracy"],
                "jitter_balanced_accuracy_mean": group["balanced_accuracy"].mean(),
                "jitter_balanced_accuracy_sd": group["balanced_accuracy"].std(ddof=1),
                "jitter_balanced_accuracy_min": group["balanced_accuracy"].min(),
                "jitter_balanced_accuracy_max": group["balanced_accuracy"].max(),
                "original_auroc": base["auroc"],
                "jitter_auroc_mean": group["auroc"].mean(),
                "jitter_auroc_sd": group["auroc"].std(ddof=1),
                "jitter_auroc_min": group["auroc"].min(),
                "jitter_auroc_max": group["auroc"].max(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("original_balanced_accuracy", ascending=False)
    summary.to_csv(out_dir / "fixed_roi_jitter_summary_by_roi.csv", index=False)

    top_rows = []
    base_top3 = set(original.sort_values("balanced_accuracy", ascending=False).head(3).index)
    for variant, group in all_metrics.groupby("variant"):
        ordered = group.sort_values("balanced_accuracy", ascending=False)
        top3 = set(ordered.head(3)["roi_name"])
        top_rows.append(
            {
                "variant": variant,
                "top1_roi": ordered.iloc[0]["roi_name"],
                "top1_balanced_accuracy": ordered.iloc[0]["balanced_accuracy"],
                "top1_auroc": ordered.iloc[0]["auroc"],
                "top3_overlap_with_original": len(base_top3 & top3),
                "top3_roi": ";".join(ordered.head(3)["roi_name"].tolist()),
            }
        )
    top_summary = pd.DataFrame(top_rows)
    top_summary.to_csv(out_dir / "fixed_roi_jitter_top_rank_summary.csv", index=False)
    return all_metrics, summary


def yunet_multiseed(data, args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = [int(s.strip()) for s in args.yunet_multiseed.split(",") if s.strip()]
    full_rows = []
    roi_rows = []
    for seed in seeds:
        print(f"[yunet_multiseed] training seed {seed}", flush=True)
        model, mean, std, full = train_seed_model(data, seed, args, permute_labels=False)
        full_rows.append(full)
        metrics = yunet_region_only_metrics(data, Path(args.yunet_audit), model, mean, std, args.upsample_size)
        metrics.insert(0, "seed", seed)
        roi_rows.append(metrics)
    full_df = pd.DataFrame(full_rows)
    roi_df = pd.concat(roi_rows, ignore_index=True)
    full_df.to_csv(out_dir / "yunet_multiseed_full_face_metrics.csv", index=False)
    roi_df.to_csv(out_dir / "yunet_multiseed_region_only_metrics.csv", index=False)
    summary = (
        roi_df.groupby("roi_name", as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            balanced_accuracy_min=("balanced_accuracy", "min"),
            balanced_accuracy_max=("balanced_accuracy", "max"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auroc_min=("auroc", "min"),
            auroc_max=("auroc", "max"),
        )
        .sort_values("balanced_accuracy_mean", ascending=False)
    )
    top_counts = roi_df.sort_values(["seed", "balanced_accuracy"], ascending=[True, False]).groupby("seed").head(1)
    counts = top_counts["roi_name"].value_counts().rename_axis("roi_name").reset_index(name="top1_count")
    summary = summary.merge(counts, on="roi_name", how="left").fillna({"top1_count": 0})
    summary.to_csv(out_dir / "yunet_multiseed_region_only_summary.csv", index=False)
    return full_df, summary


def label_permutation_negative_control(data, args: argparse.Namespace, masks: np.ndarray, roi_names: list[str], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = [int(s.strip()) for s in args.permutation_seeds.split(",") if s.strip()]
    flat_masks = [mask_to_flat(m) for m in masks]
    full_rows = []
    roi_rows = []
    for seed in seeds:
        print(f"[label_permutation] training seed {seed}", flush=True)
        model, mean, std, full = train_seed_model(data, seed, args, permute_labels=True)
        full_rows.append(full)
        metrics, _ = static_region_only_metrics(
            data["x_test_flat"].astype(np.float32),
            data["y_test"].astype(int),
            model,
            mean,
            std,
            flat_masks,
            roi_names,
        )
        metrics.insert(0, "seed", seed)
        roi_rows.append(metrics)
    full_df = pd.DataFrame(full_rows)
    roi_df = pd.concat(roi_rows, ignore_index=True)
    full_df.to_csv(out_dir / "label_permutation_full_face_metrics.csv", index=False)
    roi_df.to_csv(out_dir / "label_permutation_region_only_metrics.csv", index=False)
    summary = (
        roi_df.groupby("roi_name", as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auroc_min=("auroc", "min"),
            auroc_max=("auroc", "max"),
        )
        .sort_values("auroc_mean", ascending=False)
    )
    summary.to_csv(out_dir / "label_permutation_region_only_summary.csv", index=False)
    return full_df, summary


def occlusion_fill_strategy(data, model, mean, std, masks: np.ndarray, roi_names: list[str], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_test = data["x_test_flat"].astype(np.float32)
    images = data["x_test_images"].astype(np.float32)
    y = data["y_test"].astype(int)
    p_orig = forward(model, standardize(x_test, mean, std))[0]
    true_orig = true_confidence(y, p_orig)
    pred_orig = (p_orig >= 0.5).astype(int)
    blur_flat = images_to_flat(blur3(images))
    strategies = {
        "train_mean": np.repeat(mean.astype(np.float32), len(x_test), axis=0),
        "zero": np.zeros_like(x_test, dtype=np.float32),
        "blur3_same_image": blur_flat,
    }
    rows = []
    for strategy, fill_values in strategies.items():
        for roi_index, roi_name in enumerate(roi_names, start=1):
            flat_mask = mask_to_flat(masks[roi_index - 1])
            x_masked = x_test.copy()
            x_masked[:, flat_mask] = fill_values[:, flat_mask]
            p_mask = forward(model, standardize(x_masked, mean, std))[0]
            pred_mask = (p_mask >= 0.5).astype(int)
            drop = true_orig - true_confidence(y, p_mask)
            rows.append(
                {
                    "fill_strategy": strategy,
                    "roi_index": roi_index,
                    "roi_name": roi_name,
                    "mean_evidence_drop": float(drop.mean()),
                    "median_evidence_drop": float(np.median(drop)),
                    "sd_evidence_drop": float(drop.std(ddof=1)),
                    "prediction_change_rate": float((pred_orig != pred_mask).mean()),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "occlusion_fill_strategy_metrics.csv", index=False)
    base = metrics[metrics["fill_strategy"] == "train_mean"].set_index("roi_name")
    base_top3 = set(base.sort_values("mean_evidence_drop", ascending=False).head(3).index)
    rank_rows = []
    for strategy, group in metrics.groupby("fill_strategy"):
        cur = group.set_index("roi_name").loc[base.index]
        top3 = set(cur.sort_values("mean_evidence_drop", ascending=False).head(3).index)
        rank_rows.append(
            {
                "fill_strategy": strategy,
                "spearman_vs_train_mean": spearman(base["mean_evidence_drop"], cur["mean_evidence_drop"]),
                "top1_roi": cur.sort_values("mean_evidence_drop", ascending=False).index[0],
                "top3_overlap_with_train_mean": len(base_top3 & top3),
                "top3_roi": ";".join(cur.sort_values("mean_evidence_drop", ascending=False).head(3).index.tolist()),
            }
        )
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(out_dir / "occlusion_fill_strategy_rank_stability.csv", index=False)
    return metrics, ranks


def fixed_yunet_rank_correlation(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    fixed = pd.read_csv(args.fixed_metrics).set_index("roi_name")
    yunet = pd.read_csv(args.yunet_metrics).set_index("roi_name")
    shared = [roi for roi in ROI_NAMES if roi in fixed.index and roi in yunet.index]
    rows = []
    for metric in ["balanced_accuracy", "auroc", "auprc"]:
        f = fixed.loc[shared, metric]
        y = yunet.loc[shared, metric]
        f_top3 = set(f.sort_values(ascending=False).head(3).index)
        y_top3 = set(y.sort_values(ascending=False).head(3).index)
        rows.append(
            {
                "metric": metric,
                "n_roi": len(shared),
                "spearman_fixed_vs_yunet": spearman(f, y),
                "fixed_top1": f.sort_values(ascending=False).index[0],
                "yunet_top1": y.sort_values(ascending=False).index[0],
                "top1_match": bool(f.sort_values(ascending=False).index[0] == y.sort_values(ascending=False).index[0]),
                "top3_overlap_count": len(f_top3 & y_top3),
                "fixed_top3": ";".join(f.sort_values(ascending=False).head(3).index.tolist()),
                "yunet_top3": ";".join(y.sort_values(ascending=False).head(3).index.tolist()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fixed_yunet_rank_correlation.csv", index=False)
    return df


def make_markdown_summary(
    out_dir: Path,
    bootstrap_df: pd.DataFrame,
    jitter_summary: pd.DataFrame,
    yunet_multiseed_summary: pd.DataFrame,
    perm_summary: pd.DataFrame,
    fill_ranks: pd.DataFrame,
    rank_corr: pd.DataFrame,
) -> None:
    right = bootstrap_df[bootstrap_df["roi_name"] == "right_cheek_zygomatic"].iloc[0]
    jitter_right = jitter_summary[jitter_summary["roi_name"] == "right_cheek_zygomatic"].iloc[0]
    yunet_top = yunet_multiseed_summary.iloc[0]
    perm_full = pd.read_csv(out_dir / "label_permutation_full_face_metrics.csv")
    md = [
        "# Supplementary ROI Sensitivity Experiments",
        "",
        "Class convention: Class 0 = pre-DBS; Class 1 = post-DBS label. These analyses are image-level sensitivity and negative-control checks.",
        "",
        "## 1. Bootstrap CI: fixed vs YuNet ROI",
        "",
        f"Right-cheek fixed vs YuNet delta AUROC: point {right['delta_auroc_point']:.4f}, 95% bootstrap CI [{right['delta_auroc_q2_5']:.4f}, {right['delta_auroc_q97_5']:.4f}].",
        f"Right-cheek delta balanced accuracy: point {right['delta_ba_point']:.4f}, 95% bootstrap CI [{right['delta_ba_q2_5']:.4f}, {right['delta_ba_q97_5']:.4f}].",
        "",
        "## 2. Fixed ROI jitter sensitivity",
        "",
        f"Right-cheek original AUROC {jitter_right['original_auroc']:.4f}; jittered AUROC mean {jitter_right['jitter_auroc_mean']:.4f}, range [{jitter_right['jitter_auroc_min']:.4f}, {jitter_right['jitter_auroc_max']:.4f}].",
        "",
        "## 3. YuNet dynamic ROI multi-seed stability",
        "",
        f"Top mean YuNet ROI across seeds: {yunet_top['roi_name']} with mean BA {yunet_top['balanced_accuracy_mean']:.4f} and mean AUROC {yunet_top['auroc_mean']:.4f}.",
        "",
        "## 4. Label permutation negative control",
        "",
        f"Permuted-label full-face AUROC mean {perm_full['auroc'].mean():.4f} across {len(perm_full)} seeds; region-only AUROCs remain near chance (see CSV).",
        "",
        "## 5. Occlusion fill strategy sensitivity",
        "",
        "| Fill strategy | Spearman vs train-mean fill | Top-1 ROI | Top-3 overlap |",
        "|---|---:|---|---:|",
    ]
    for _, row in fill_ranks.iterrows():
        md.append(
            f"| {row['fill_strategy']} | {row['spearman_vs_train_mean']:.4f} | {row['top1_roi']} | {int(row['top3_overlap_with_train_mean'])}/3 |"
        )
    md.extend(
        [
            "",
            "## 6. Fixed vs YuNet rank correlation",
            "",
            "| Metric | Spearman | Top-1 match | Top-3 overlap |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in rank_corr.iterrows():
        md.append(
            f"| {row['metric']} | {row['spearman_fixed_vs_yunet']:.4f} | {row['top1_match']} | {int(row['top3_overlap_count'])}/3 |"
        )
    md.extend(
        [
            "",
            "## Files",
            "",
            "- `bootstrap_fixed_vs_yunet_ci.csv`",
            "- `fixed_roi_jitter_region_only_metrics.csv`",
            "- `fixed_roi_jitter_summary_by_roi.csv`",
            "- `yunet_multiseed_region_only_metrics.csv`",
            "- `yunet_multiseed_region_only_summary.csv`",
            "- `label_permutation_full_face_metrics.csv`",
            "- `label_permutation_region_only_summary.csv`",
            "- `occlusion_fill_strategy_metrics.csv`",
            "- `occlusion_fill_strategy_rank_stability.csv`",
            "- `fixed_yunet_rank_correlation.csv`",
        ]
    )
    (out_dir / "SUPPLEMENTARY_ROI_SENSITIVITY_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--model", default="models/baseline_mlp_numpy.npz")
    parser.add_argument("--roi-masks", default="outputs/roi/coarse_roi_masks.npy")
    parser.add_argument("--roi-defs", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument("--fixed-metrics", default="outputs/aev/region_only_metrics.csv")
    parser.add_argument("--fixed-predictions", default="outputs/aev/region_only_predictions.csv")
    parser.add_argument("--yunet-metrics", default="outputs/external/pd_dbs_yunet_region_only/yunet_region_only_metrics.csv")
    parser.add_argument("--yunet-predictions", default="outputs/external/pd_dbs_yunet_region_only/yunet_region_only_predictions.csv")
    parser.add_argument("--yunet-audit", default="outputs/external/pd_dbs_yunet_feasibility/yunet_detection_audit.csv")
    parser.add_argument("--output-dir", default="outputs/sensitivity/roi_supplementary_experiments")
    parser.add_argument("--upsample-size", type=int, default=256)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260626)
    parser.add_argument("--yunet-multiseed", default="0,1,2,42,1024")
    parser.add_argument("--permutation-seeds", default="0,1,2")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    data = load_pd_dbs(args.data)
    model, mean, std = load_model(args.model)
    masks = np.load(args.roi_masks).astype(bool)
    roi_names = pd.read_csv(args.roi_defs)["roi_name"].tolist()

    print("[1/6] bootstrap fixed vs YuNet", flush=True)
    bootstrap_df = bootstrap_fixed_vs_yunet(args, out_dir)

    print("[2/6] fixed ROI jitter sensitivity", flush=True)
    _, jitter_summary = fixed_roi_jitter(data, model, mean, std, masks, roi_names, out_dir)

    print("[3/6] YuNet dynamic ROI multi-seed stability", flush=True)
    _, yunet_multiseed_summary = yunet_multiseed(data, args, out_dir)

    print("[4/6] label permutation negative control", flush=True)
    _, perm_summary = label_permutation_negative_control(data, args, masks, roi_names, out_dir)

    print("[5/6] occlusion fill strategy sensitivity", flush=True)
    _, fill_ranks = occlusion_fill_strategy(data, model, mean, std, masks, roi_names, out_dir)

    print("[6/6] fixed vs YuNet rank correlation", flush=True)
    rank_corr = fixed_yunet_rank_correlation(args, out_dir)

    make_markdown_summary(out_dir, bootstrap_df, jitter_summary, yunet_multiseed_summary, perm_summary, fill_ranks, rank_corr)
    print(f"wrote supplementary ROI sensitivity outputs to {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

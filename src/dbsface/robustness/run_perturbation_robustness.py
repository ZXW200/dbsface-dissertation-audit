"""Evaluate baseline and ROI-evidence robustness under simple image perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_calibration_numpy import calibration_bins
from load_pd_dbs import load_pd_dbs
from train_baseline_mlp_numpy import forward, metric_summary, standardize


def load_model(path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ckpt = np.load(path)
    model = {key: ckpt[key] for key in ["w1", "b1", "w2", "b2"]}
    return model, ckpt["mean"], ckpt["std"]


def images_to_flat(images: np.ndarray) -> np.ndarray:
    """Inverse of loader orientation: [N, 32, 32, 1] -> [N, 1024]."""

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


def crop_resize(images: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    out = []
    for arr in images[..., 0]:
        crop = arr[top : top + size, left : left + size]
        im = Image.fromarray(crop.astype(np.float32), mode="F").resize((32, 32), Image.Resampling.BILINEAR)
        out.append(np.asarray(im, dtype=np.float32))
    return np.stack(out, axis=0)[..., None]


def true_confidence(y: np.ndarray, p_class1: np.ndarray) -> np.ndarray:
    return np.where(y == 1, p_class1, 1.0 - p_class1)


def mask_to_flat(mask: np.ndarray) -> np.ndarray:
    return mask.T.reshape(-1).astype(bool)


def roi_evidence_means(
    x_flat: np.ndarray,
    y: np.ndarray,
    model: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    masks: np.ndarray,
    roi_names: list[str],
) -> pd.DataFrame:
    p_orig = forward(model, standardize(x_flat, mean, std))[0]
    true_orig = true_confidence(y, p_orig)
    rows = []
    for idx, roi in enumerate(roi_names):
        flat_mask = mask_to_flat(masks[idx])
        x_mask = x_flat.copy()
        x_mask[:, flat_mask] = mean[:, flat_mask]
        p_mask = forward(model, standardize(x_mask, mean, std))[0]
        drop = true_orig - true_confidence(y, p_mask)
        rows.append({"roi_name": roi, "mean_evidence_drop": float(drop.mean()), "median_evidence_drop": float(np.median(drop))})
    return pd.DataFrame(rows)


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    ra = a.rank(method="average").to_numpy(dtype=float)
    rb = b.rank(method="average").to_numpy(dtype=float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate_set(name: str, x_flat: np.ndarray, y: np.ndarray, model, mean, std, n_bins: int) -> dict:
    p = forward(model, standardize(x_flat, mean, std))[0]
    metrics = metric_summary(y, p)
    _, ece = calibration_bins(y.astype(float), p.astype(float), n_bins)
    return {
        "perturbation": name,
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1_class1": metrics["f1_class1"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "brier_score": metrics["brier_score"],
        "ece": ece,
        "tn": metrics["confusion_matrix"]["tn"],
        "fp": metrics["confusion_matrix"]["fp"],
        "fn": metrics["confusion_matrix"]["fn"],
        "tp": metrics["confusion_matrix"]["tp"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--model", default="models/baseline_mlp_numpy.npz")
    parser.add_argument("--roi-masks", default="outputs/roi/coarse_roi_masks.npy")
    parser.add_argument("--roi-defs", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument("--output-dir", default="outputs/robustness")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_pd_dbs(args.data)
    images = data["x_test_images"].astype(np.float32)
    x_flat = data["x_test_flat"].astype(np.float32)
    y = data["y_test"].astype(int)
    model, mean, std = load_model(args.model)
    masks = np.load(args.roi_masks).astype(bool)
    roi_names = pd.read_csv(args.roi_defs)["roi_name"].tolist()

    perturbations = {
        "original": images,
        "mild_blur_3x3": blur3(images),
        "center_crop_28_resize": crop_resize(images, 2, 2, 28),
        "offset_crop_28_resize": crop_resize(images, 1, 3, 28),
    }

    metric_rows = []
    evidence_tables = {}
    for name, imgs in perturbations.items():
        xf = images_to_flat(imgs) if name != "original" else x_flat
        metric_rows.append(evaluate_set(name, xf, y, model, mean, std, args.n_bins))
        evidence_tables[name] = roi_evidence_means(xf, y, model, mean, std, masks, roi_names)
        evidence_tables[name].to_csv(out_dir / f"roi_evidence_{name}.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    base = metrics.loc[metrics["perturbation"] == "original"].iloc[0]
    for col in ["accuracy", "balanced_accuracy", "f1_class1", "auroc", "auprc", "brier_score", "ece"]:
        metrics[f"delta_{col}_vs_original"] = metrics[col] - base[col]
    metrics.to_csv(out_dir / "perturbation_metrics.csv", index=False)

    original = evidence_tables["original"].set_index("roi_name")
    stability_rows = []
    original_top3 = set(original.sort_values("mean_evidence_drop", ascending=False).head(3).index)
    for name, table in evidence_tables.items():
        if name == "original":
            continue
        cur = table.set_index("roi_name").loc[original.index]
        corr = rank_corr(original["mean_evidence_drop"], cur["mean_evidence_drop"])
        top3 = set(cur.sort_values("mean_evidence_drop", ascending=False).head(3).index)
        stability_rows.append(
            {
                "perturbation": name,
                "spearman_like_rank_correlation": corr,
                "top3_overlap_count": len(original_top3 & top3),
                "top3_overlap_fraction": len(original_top3 & top3) / 3,
                "original_top3": ";".join(sorted(original_top3)),
                "perturbed_top3": ";".join(sorted(top3)),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(out_dir / "perturbation_roi_stability.csv", index=False)

    md = [
        "# Perturbation Robustness Summary",
        "",
        "Model: NumPy MLP. Labels: Class 0 = pre-DBS; Class 1 = post-DBS label.",
        "",
        "## Performance Under Perturbation",
        "",
        "| Perturbation | Accuracy | Balanced accuracy | AUROC | Trapezoidal AUPRC | Brier | ECE | Delta accuracy | Delta AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        md.append(
            f"| {row['perturbation']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['auroc']:.4f} | {row['auprc']:.4f} | {row['brier_score']:.4f} | {row['ece']:.4f} | "
            f"{row['delta_accuracy_vs_original']:.4f} | {row['delta_auroc_vs_original']:.4f} |"
        )
    md.extend(
        [
            "",
            "## ROI Evidence Ranking Stability",
            "",
            "| Perturbation | Rank correlation | Top-3 overlap | Perturbed top-3 |",
            "|---|---:|---:|---|",
        ]
    )
    for _, row in stability.iterrows():
        md.append(
            f"| {row['perturbation']} | {row['spearman_like_rank_correlation']:.4f} | "
            f"{row['top3_overlap_count']}/3 | {row['perturbed_top3']} |"
        )
    (out_dir / "perturbation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(stability.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

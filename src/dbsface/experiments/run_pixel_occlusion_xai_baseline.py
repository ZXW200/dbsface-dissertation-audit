"""Pixel-level occlusion baseline aggregated into the fixed ROI atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dbsface.data.load_pd_dbs import load_pd_dbs
from dbsface.experiments.run_roi_occlusion_mlp import load_model, mask_to_flat, true_confidence
from dbsface.experiments.train_baseline_mlp_numpy import forward, standardize


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def topk(values: pd.DataFrame, col: str, k: int = 3) -> set[str]:
    return set(values.sort_values(col, ascending=False).head(k)["roi_name"].tolist())


def comparison_row(left_name: str, right_name: str, df: pd.DataFrame, left_col: str, right_col: str) -> dict[str, object]:
    left_top = topk(df, left_col)
    right_top = topk(df, right_col)
    return {
        "comparison": f"{left_name} vs {right_name}",
        "spearman_rho": rank_correlation(df[left_col].to_numpy(float), df[right_col].to_numpy(float)),
        "top3_left": "; ".join(sorted(left_top)),
        "top3_right": "; ".join(sorted(right_top)),
        "top3_overlap_count": len(left_top & right_top),
        "top3_overlap": "; ".join(sorted(left_top & right_top)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--model", default="models/baseline_mlp_numpy.npz")
    parser.add_argument("--roi-masks", default="outputs/roi/coarse_roi_masks.npy")
    parser.add_argument("--roi-defs", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument("--mlp-roi-aev", default="outputs/aev/roi_occlusion_summary_overall.csv")
    parser.add_argument("--gradcam-overlap", default="outputs/gradcam_occlusion_overlap/gradcam_occlusion_roi_overlap.csv")
    parser.add_argument("--output-dir", default="outputs/xai_baselines")
    parser.add_argument("--pixel-batch", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pd_dbs(args.data)
    x_test = data["x_test_flat"].astype(np.float32)
    y = data["y_test"].astype(int)
    model, mean, std = load_model(args.model)
    masks = np.load(args.roi_masks).astype(bool)
    roi_defs = pd.read_csv(args.roi_defs)

    p_orig = forward(model, standardize(x_test, mean, std))[0]
    conf_orig = true_confidence(y, p_orig)
    n, n_features = x_test.shape
    pixel_mean = np.zeros(n_features, dtype=np.float64)
    pixel_class0 = np.zeros(n_features, dtype=np.float64)
    pixel_class1 = np.zeros(n_features, dtype=np.float64)

    class0 = y == 0
    class1 = y == 1
    for start in range(0, n_features, args.pixel_batch):
        pix = np.arange(start, min(start + args.pixel_batch, n_features))
        x_rep = np.repeat(x_test[None, :, :], len(pix), axis=0)
        for j, pidx in enumerate(pix):
            x_rep[j, :, pidx] = mean[0, pidx]
        flat = x_rep.reshape(-1, n_features)
        p_masked = forward(model, standardize(flat, mean, std))[0].reshape(len(pix), n)
        conf_masked = true_confidence(np.tile(y, (len(pix), 1)), p_masked)
        drops = conf_orig[None, :] - conf_masked
        pixel_mean[pix] = drops.mean(axis=1)
        pixel_class0[pix] = drops[:, class0].mean(axis=1)
        pixel_class1[pix] = drops[:, class1].mean(axis=1)

    pixel_df = pd.DataFrame(
        {
            "flat_pixel_index": np.arange(n_features),
            "mean_pixel_evidence_drop": pixel_mean,
            "mean_pixel_evidence_drop_class0": pixel_class0,
            "mean_pixel_evidence_drop_class1": pixel_class1,
        }
    )
    pixel_df.to_csv(out_dir / "pixel_occlusion_scores.csv", index=False)

    roi_rows = []
    for roi_idx, row in roi_defs.iterrows():
        roi_name = str(row["roi_name"])
        flat_mask = mask_to_flat(masks[roi_idx])
        vals = pixel_mean[flat_mask]
        vals0 = pixel_class0[flat_mask]
        vals1 = pixel_class1[flat_mask]
        roi_rows.append(
            {
                "roi_index": int(row["roi_index"]),
                "roi_name": roi_name,
                "n_pixels": int(flat_mask.sum()),
                "pixel_occlusion_mean_drop": float(vals.mean()),
                "pixel_occlusion_sum_drop": float(vals.sum()),
                "pixel_occlusion_mean_drop_class0": float(vals0.mean()),
                "pixel_occlusion_mean_drop_class1": float(vals1.mean()),
                "pixel_occlusion_class1_minus_class0": float(vals1.mean() - vals0.mean()),
            }
        )
    roi_df = pd.DataFrame(roi_rows)

    region_aev = pd.read_csv(args.mlp_roi_aev)[["roi_name", "mean_evidence_drop"]].rename(
        columns={"mean_evidence_drop": "mlp_region_mask_aev"}
    )
    roi_df = roi_df.merge(region_aev, on="roi_name", how="left")

    grad_path = Path(args.gradcam_overlap)
    if grad_path.exists():
        grad = pd.read_csv(grad_path)[
            ["roi_name", "mean_evidence_drop", "mean_cam_energy_fraction", "rank_cnn_occlusion", "rank_gradcam_energy"]
        ].rename(
            columns={
                "mean_evidence_drop": "cnn_region_mask_aev",
                "mean_cam_energy_fraction": "cnn_gradcam_energy",
            }
        )
        roi_df = roi_df.merge(grad, on="roi_name", how="left")

    roi_df["rank_mlp_region_mask_aev"] = roi_df["mlp_region_mask_aev"].rank(ascending=False, method="min").astype(int)
    roi_df["rank_pixel_occlusion_mean"] = roi_df["pixel_occlusion_mean_drop"].rank(ascending=False, method="min").astype(int)
    roi_df["rank_pixel_occlusion_sum"] = roi_df["pixel_occlusion_sum_drop"].rank(ascending=False, method="min").astype(int)
    roi_df = roi_df.sort_values("rank_mlp_region_mask_aev")
    roi_df.to_csv(out_dir / "pixel_occlusion_roi_summary.csv", index=False)

    comparisons = [
        comparison_row(
            "MLP region-mask AEV",
            "pixel occlusion ROI mean",
            roi_df,
            "mlp_region_mask_aev",
            "pixel_occlusion_mean_drop",
        ),
        comparison_row(
            "MLP region-mask AEV",
            "pixel occlusion ROI sum",
            roi_df,
            "mlp_region_mask_aev",
            "pixel_occlusion_sum_drop",
        ),
    ]
    if "cnn_gradcam_energy" in roi_df.columns:
        comparisons.extend(
            [
                comparison_row(
                    "MLP region-mask AEV",
                    "CNN Grad-CAM ROI energy",
                    roi_df,
                    "mlp_region_mask_aev",
                    "cnn_gradcam_energy",
                ),
                comparison_row(
                    "MLP region-mask AEV",
                    "CNN region-mask AEV",
                    roi_df,
                    "mlp_region_mask_aev",
                    "cnn_region_mask_aev",
                ),
                comparison_row(
                    "CNN region-mask AEV",
                    "CNN Grad-CAM ROI energy",
                    roi_df,
                    "cnn_region_mask_aev",
                    "cnn_gradcam_energy",
                ),
            ]
        )
    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(out_dir / "xai_roi_rank_comparisons.csv", index=False)

    meta = {
        "description": "Pixel-level occlusion baseline aggregated into the same eight fixed ROIs.",
        "n_test": int(n),
        "n_pixels": int(n_features),
        "pixel_batch": int(args.pixel_batch),
        "comparison_table": str((out_dir / "xai_roi_rank_comparisons.csv").resolve()),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    md = [
        "# Pixel-Occlusion XAI Baseline",
        "",
        "Each pixel was replaced by its training-set mean value, true-class confidence drop was measured, and pixel drops were aggregated into the same eight ROIs.",
        "",
        "## ROI rank comparisons",
        "",
        "| Comparison | Spearman rho | Top-3 overlap |",
        "|---|---:|---:|",
    ]
    for _, row in comp_df.iterrows():
        md.append(f"| {row['comparison']} | {row['spearman_rho']:.4f} | {int(row['top3_overlap_count'])}/3 |")
    md.extend(
        [
            "",
            "## ROI summary",
            "",
            "| ROI | Region-mask AEV | Pixel mean | Pixel sum |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in roi_df.iterrows():
        md.append(
            f"| {row['roi_name']} | {row['mlp_region_mask_aev']:.6f} | "
            f"{row['pixel_occlusion_mean_drop']:.6f} | {row['pixel_occlusion_sum_drop']:.6f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

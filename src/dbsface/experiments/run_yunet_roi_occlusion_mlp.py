"""Run YuNet dynamic-ROI mask-out AEV analysis for the PD-DBS data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs
from run_yunet_region_only_mlp import ROI_NAMES, box_to_mask32, row_to_yunet_boxes
from train_baseline_mlp_numpy import forward, standardize


def load_model(path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ckpt = np.load(path)
    model = {key: ckpt[key] for key in ["w1", "b1", "w2", "b2"]}
    return model, ckpt["mean"].astype(np.float32), ckpt["std"].astype(np.float32)


def true_confidence(y: np.ndarray, p_class1: np.ndarray) -> np.ndarray:
    return np.where(y == 1, p_class1, 1.0 - p_class1)


def load_yunet_boxes(
    audit_path: str | Path,
    upsample_size: int,
) -> tuple[pd.DataFrame, dict[int, dict[str, tuple[int, int, int, int]]]]:
    audit = pd.read_csv(audit_path)
    detected = audit[audit["detected"] == 1].copy()
    boxes = {int(row["global_index"]): row_to_yunet_boxes(row, upsample_size) for _, row in detected.iterrows()}
    return audit, boxes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--model", default="models/baseline_mlp_numpy.npz")
    parser.add_argument("--yunet-audit", default="outputs/external/pd_dbs_yunet_feasibility/yunet_detection_audit.csv")
    parser.add_argument("--fixed-overall", default="outputs/aev/roi_occlusion_summary_overall.csv")
    parser.add_argument("--output-dir", default="outputs/external/pd_dbs_yunet_aev")
    parser.add_argument("--upsample-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pd_dbs(args.data)
    x_test = data["x_test_flat"].astype(np.float32)
    y = data["y_test"].astype(int)
    n_train = len(data["x_train_flat"])
    model, mean, std = load_model(args.model)
    audit, boxes_by_global = load_yunet_boxes(args.yunet_audit, args.upsample_size)

    p_orig = forward(model, standardize(x_test, mean, std))[0]
    true_conf_orig = true_confidence(y, p_orig)
    pred_orig = (p_orig >= 0.5).astype(int)

    long_rows = []
    wide = {
        "sample_id": [f"test_{i:04d}" for i in range(len(y))],
        "split": ["test"] * len(y),
        "y_true": y.astype(int),
        "p_class1_original": p_orig.astype(float),
        "y_pred_original": pred_orig.astype(int),
        "correct_original": (pred_orig == y).astype(int),
        "true_conf_original": true_conf_orig.astype(float),
    }

    for roi_index, roi_name in enumerate(ROI_NAMES, start=1):
        p_masked = np.full(len(y), np.nan, dtype=np.float32)
        pred_masked = np.full(len(y), -1, dtype=int)
        true_conf_masked = np.full(len(y), np.nan, dtype=np.float32)
        evidence_drop = np.full(len(y), np.nan, dtype=np.float32)
        pixel_counts = np.zeros(len(y), dtype=int)
        missing = np.zeros(len(y), dtype=bool)

        x_masked_all = x_test.copy()
        for i in range(len(y)):
            global_idx = n_train + i
            boxes = boxes_by_global.get(global_idx)
            if boxes is None:
                missing[i] = True
                continue
            flat_mask = box_to_mask32(boxes[roi_name], args.upsample_size)
            pixel_counts[i] = int(flat_mask.sum())
            x_masked_all[i, flat_mask] = mean[0, flat_mask]

        p = forward(model, standardize(x_masked_all, mean, std))[0]
        p_masked[:] = p.astype(np.float32)
        pred_masked[:] = (p >= 0.5).astype(int)
        true_conf_masked[:] = true_confidence(y, p).astype(np.float32)
        evidence_drop[:] = (true_conf_orig - true_conf_masked).astype(np.float32)

        wide[f"p_class1_masked__{roi_name}"] = p_masked.astype(float)
        wide[f"evidence_drop__{roi_name}"] = evidence_drop.astype(float)
        wide[f"dynamic_roi_pixels__{roi_name}"] = pixel_counts.astype(int)

        for i in range(len(y)):
            long_rows.append(
                {
                    "sample_id": f"test_{i:04d}",
                    "split": "test",
                    "roi_index": roi_index,
                    "roi_name": roi_name,
                    "y_true": int(y[i]),
                    "p_class1_original": float(p_orig[i]),
                    "p_class1_masked": float(p_masked[i]),
                    "y_pred_original": int(pred_orig[i]),
                    "y_pred_masked": int(pred_masked[i]),
                    "true_conf_original": float(true_conf_orig[i]),
                    "true_conf_masked": float(true_conf_masked[i]),
                    "evidence_drop": float(evidence_drop[i]),
                    "prediction_changed": bool(pred_orig[i] != pred_masked[i]),
                    "dynamic_roi_pixels": int(pixel_counts[i]),
                    "missing_yunet": bool(missing[i]),
                }
            )

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide)
    long_df.to_csv(out_dir / "yunet_roi_occlusion_test.csv", index=False)
    wide_df.to_csv(out_dir / "yunet_aev_test.csv", index=False)

    overall = (
        long_df.groupby(["roi_index", "roi_name"], as_index=False)
        .agg(
            n=("evidence_drop", "size"),
            mean_evidence_drop=("evidence_drop", "mean"),
            median_evidence_drop=("evidence_drop", "median"),
            std_evidence_drop=("evidence_drop", "std"),
            prediction_change_rate=("prediction_changed", "mean"),
            mean_dynamic_roi_pixels=("dynamic_roi_pixels", "mean"),
            missing_yunet_count=("missing_yunet", "sum"),
        )
        .sort_values("mean_evidence_drop", ascending=False)
    )
    overall.to_csv(out_dir / "yunet_roi_occlusion_summary_overall.csv", index=False)

    by_class = (
        long_df.groupby(["y_true", "roi_index", "roi_name"], as_index=False)
        .agg(
            n=("evidence_drop", "size"),
            mean_evidence_drop=("evidence_drop", "mean"),
            median_evidence_drop=("evidence_drop", "median"),
            std_evidence_drop=("evidence_drop", "std"),
            q25_evidence_drop=("evidence_drop", lambda s: float(np.quantile(s, 0.25))),
            q75_evidence_drop=("evidence_drop", lambda s: float(np.quantile(s, 0.75))),
            prediction_change_rate=("prediction_changed", "mean"),
            mean_true_conf_original=("true_conf_original", "mean"),
            mean_true_conf_masked=("true_conf_masked", "mean"),
            mean_dynamic_roi_pixels=("dynamic_roi_pixels", "mean"),
            missing_yunet_count=("missing_yunet", "sum"),
        )
        .sort_values(["y_true", "mean_evidence_drop"], ascending=[True, False])
    )
    by_class.to_csv(out_dir / "yunet_roi_occlusion_summary_by_class.csv", index=False)

    comparison = None
    fixed_path = Path(args.fixed_overall)
    if fixed_path.exists():
        fixed = pd.read_csv(fixed_path)
        comparison = fixed[["roi_name", "mean_evidence_drop", "median_evidence_drop", "prediction_change_rate"]].merge(
            overall[["roi_name", "mean_evidence_drop", "median_evidence_drop", "prediction_change_rate", "mean_dynamic_roi_pixels"]],
            on="roi_name",
            suffixes=("_fixed", "_yunet"),
        )
        comparison["delta_mean_evidence_drop"] = comparison["mean_evidence_drop_yunet"] - comparison["mean_evidence_drop_fixed"]
        comparison["delta_prediction_change_rate"] = comparison["prediction_change_rate_yunet"] - comparison["prediction_change_rate_fixed"]
        comparison = comparison.sort_values("mean_evidence_drop_yunet", ascending=False)
        comparison.to_csv(out_dir / "yunet_vs_fixed_aev_occlusion_comparison.csv", index=False)

    detection_rate = float((audit["detected"] == 1).mean())
    md = [
        "# YuNet Dynamic ROI AEV Occlusion Summary",
        "",
        "YuNet dynamic ROI boxes were used for mask-out AEV analysis. Each 32x32 image was upsampled to 256x256 for face and five-landmark detection; the resulting dynamic ROI box was projected back to the 32x32 grid and replaced by the training-set mean during occlusion.",
        "",
        f"Detection rate: {detection_rate:.3f} ({int((audit['detected'] == 1).sum())}/{len(audit)})",
        "",
        "Class convention: Class 0 = pre-DBS; Class 1 = post-DBS label. This is an image-level explanation sensitivity analysis.",
        "",
        "## YuNet dynamic ROI AEV ranking",
        "",
        "| ROI | Mean evidence drop | Median evidence drop | Prediction change rate | Mean ROI pixels |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        md.append(
            f"| {row['roi_name']} | {row['mean_evidence_drop']:.6f} | {row['median_evidence_drop']:.6f} | "
            f"{row['prediction_change_rate']:.4f} | {row['mean_dynamic_roi_pixels']:.1f} |"
        )
    if comparison is not None:
        md.extend(
            [
                "",
                "## Fixed vs YuNet AEV comparison",
                "",
                "| ROI | Fixed mean drop | YuNet mean drop | Delta | Fixed change rate | YuNet change rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in comparison.iterrows():
            md.append(
                f"| {row['roi_name']} | {row['mean_evidence_drop_fixed']:.6f} | "
                f"{row['mean_evidence_drop_yunet']:.6f} | {row['delta_mean_evidence_drop']:.6f} | "
                f"{row['prediction_change_rate_fixed']:.4f} | {row['prediction_change_rate_yunet']:.4f} |"
            )
    md.extend(
        [
            "",
            "Interpretation: YuNet dynamic AEV provides an ROI-definition sensitivity check. Dynamic and fixed ROI geometries can change the magnitude and ranking of evidence drops.",
        ]
    )
    (out_dir / "yunet_roi_occlusion_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    metadata = {
        "data": str(Path(args.data).resolve()),
        "model": str(Path(args.model).resolve()),
        "yunet_audit": str(Path(args.yunet_audit).resolve()),
        "output_dir": str(out_dir.resolve()),
        "upsample_size": args.upsample_size,
        "detection_rate": detection_rate,
        "n_images": int(len(audit)),
        "n_test": int(len(y)),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))
    print(f"wrote outputs to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

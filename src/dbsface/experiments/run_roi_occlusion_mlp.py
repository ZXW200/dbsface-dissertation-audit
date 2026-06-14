"""Run coarse-ROI mask-out evidence analysis with the NumPy MLP baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs
from train_baseline_mlp_numpy import forward, standardize


def mask_to_flat(mask: np.ndarray) -> np.ndarray:
    """Map image-orientation [y, x] ROI mask to original 1024-vector order."""

    return mask.T.reshape(-1).astype(bool)


def load_model(path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ckpt = np.load(path)
    model = {key: ckpt[key] for key in ["w1", "b1", "w2", "b2"]}
    return model, ckpt["mean"], ckpt["std"]


def true_confidence(y: np.ndarray, p_class1: np.ndarray) -> np.ndarray:
    return np.where(y == 1, p_class1, 1.0 - p_class1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--model", default="models/baseline_mlp_numpy.npz")
    parser.add_argument("--roi-masks", default="outputs/roi/coarse_roi_masks.npy")
    parser.add_argument("--roi-defs", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument("--output-dir", default="outputs/aev")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pd_dbs(args.data)
    x_test = data["x_test_flat"].astype(np.float32)
    y = data["y_test"].astype(int)
    model, mean, std = load_model(args.model)
    masks = np.load(args.roi_masks).astype(bool)
    roi_defs = pd.read_csv(args.roi_defs)
    roi_names = roi_defs["roi_name"].tolist()

    p_orig = forward(model, standardize(x_test, mean, std))[0]
    true_conf_orig = true_confidence(y, p_orig)
    y_pred = (p_orig >= 0.5).astype(int)

    long_records = []
    wide = {
        "sample_id": [f"test_{i:04d}" for i in range(len(y))],
        "split": ["test"] * len(y),
        "y_true": y.astype(int),
        "p_class1_original": p_orig.astype(float),
        "y_pred_original": y_pred.astype(int),
        "correct_original": (y_pred == y).astype(int),
        "true_conf_original": true_conf_orig.astype(float),
    }

    for roi_idx, roi_name in enumerate(roi_names):
        flat_mask = mask_to_flat(masks[roi_idx])
        x_masked = x_test.copy()
        x_masked[:, flat_mask] = mean[:, flat_mask]
        p_masked = forward(model, standardize(x_masked, mean, std))[0]
        y_pred_masked = (p_masked >= 0.5).astype(int)
        true_conf_masked = true_confidence(y, p_masked)
        evidence_drop = true_conf_orig - true_conf_masked

        wide[f"p_class1_masked__{roi_name}"] = p_masked.astype(float)
        wide[f"evidence_drop__{roi_name}"] = evidence_drop.astype(float)

        for i in range(len(y)):
            long_records.append(
                {
                    "sample_id": f"test_{i:04d}",
                    "split": "test",
                    "roi_index": roi_idx + 1,
                    "roi_name": roi_name,
                    "y_true": int(y[i]),
                    "p_class1_original": float(p_orig[i]),
                    "p_class1_masked": float(p_masked[i]),
                    "y_pred_original": int(y_pred[i]),
                    "y_pred_masked": int(y_pred_masked[i]),
                    "true_conf_original": float(true_conf_orig[i]),
                    "true_conf_masked": float(true_conf_masked[i]),
                    "evidence_drop": float(evidence_drop[i]),
                    "prediction_changed": bool(y_pred[i] != y_pred_masked[i]),
                }
            )

    long_df = pd.DataFrame(long_records)
    wide_df = pd.DataFrame(wide)
    long_df.to_csv(out_dir / "roi_occlusion_test.csv", index=False)
    wide_df.to_csv(out_dir / "aev_test.csv", index=False)
    long_df[["sample_id", "roi_name", "y_true", "evidence_drop"]].to_csv(
        out_dir / "roi_occlusion_boxplot_data.csv", index=False
    )

    summary = (
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
        )
        .sort_values(["y_true", "mean_evidence_drop"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "roi_occlusion_summary_by_class.csv", index=False)

    overall = (
        long_df.groupby(["roi_index", "roi_name"], as_index=False)
        .agg(
            n=("evidence_drop", "size"),
            mean_evidence_drop=("evidence_drop", "mean"),
            median_evidence_drop=("evidence_drop", "median"),
            prediction_change_rate=("prediction_changed", "mean"),
        )
        .sort_values("mean_evidence_drop", ascending=False)
    )
    overall.to_csv(out_dir / "roi_occlusion_summary_overall.csv", index=False)

    md = [
        "# ROI Occlusion AEV Summary",
        "",
        "Model: `models/baseline_mlp_numpy.npz`",
        "Labels: Class 0 = pre-DBS; Class 1 = post-DBS label.",
        "",
        "Evidence definition:",
        "",
        "`evidence_drop = true_confidence_original - true_confidence_masked`",
        "",
        "Positive evidence_drop means masking the ROI reduced true-class confidence.",
        "",
        "## Overall ROI ranking by mean evidence drop",
        "",
        "| ROI | Mean evidence drop | Median evidence drop | Prediction change rate |",
        "|---|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        md.append(
            f"| {row['roi_name']} | {row['mean_evidence_drop']:.6f} | "
            f"{row['median_evidence_drop']:.6f} | {row['prediction_change_rate']:.4f} |"
        )
    (out_dir / "roi_occlusion_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

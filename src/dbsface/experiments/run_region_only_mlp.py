"""Run region-only validation using the NumPy MLP baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs
from train_baseline_mlp_numpy import forward, metric_summary, standardize


def mask_to_flat(mask: np.ndarray) -> np.ndarray:
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

    metrics_rows = []
    prediction_rows = []
    json_metrics = {}

    for roi_idx, roi_name in enumerate(roi_names):
        flat_mask = mask_to_flat(masks[roi_idx])
        x_region = np.repeat(mean.astype(np.float32), len(x_test), axis=0)
        x_region[:, flat_mask] = x_test[:, flat_mask]
        p = forward(model, standardize(x_region, mean, std))[0]
        pred = (p >= 0.5).astype(int)
        true_conf = true_confidence(y, p)
        metrics = metric_summary(y, p)
        json_metrics[roi_name] = metrics
        metrics_rows.append(
            {
                "roi_index": roi_idx + 1,
                "roi_name": roi_name,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_class1": metrics["f1_class1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier_score": metrics["brier_score"],
                "mean_true_confidence": float(true_conf.mean()),
                "majority_baseline_accuracy": metrics["majority_baseline_accuracy"],
                "tn": metrics["confusion_matrix"]["tn"],
                "fp": metrics["confusion_matrix"]["fp"],
                "fn": metrics["confusion_matrix"]["fn"],
                "tp": metrics["confusion_matrix"]["tp"],
            }
        )
        for i in range(len(y)):
            prediction_rows.append(
                {
                    "sample_id": f"test_{i:04d}",
                    "split": "test",
                    "roi_index": roi_idx + 1,
                    "roi_name": roi_name,
                    "y_true": int(y[i]),
                    "p_class1": float(p[i]),
                    "y_pred": int(pred[i]),
                    "correct": bool(pred[i] == y[i]),
                    "true_confidence": float(true_conf[i]),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("balanced_accuracy", ascending=False)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df.to_csv(out_dir / "region_only_metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "region_only_predictions.csv", index=False)
    (out_dir / "region_only_metrics.json").write_text(json.dumps(json_metrics, indent=2), encoding="utf-8")

    md = [
        "# Region-Only Validation Summary",
        "",
        "Only one coarse ROI is retained at a time; all other pixels are replaced with the training mean.",
        "",
        "Labels follow the project convention: Class 0 = pre-DBS; Class 1 = post-DBS label.",
        "",
        "| ROI | Accuracy | Balanced accuracy | AUROC | Trapezoidal AUPRC | Mean true confidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.iterrows():
        md.append(
            f"| {row['roi_name']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['auroc']:.4f} | {row['auprc']:.4f} | {row['mean_true_confidence']:.4f} |"
        )
    (out_dir / "region_only_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate baseline sensitivity after removing near-duplicate test samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_calibration_numpy import calibration_bins, brier_score
from train_baseline_mlp_numpy import metric_summary


def flatten_metrics(name: str, y: np.ndarray, p: np.ndarray, n_bins: int) -> dict[str, float | int | str]:
    m = metric_summary(y, p)
    _, ece = calibration_bins(y.astype(float), p.astype(float), n_bins)
    return {
        "set": name,
        "n": int(len(y)),
        "class0": int((y == 0).sum()),
        "class1": int((y == 1).sum()),
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "f1_class1": m["f1_class1"],
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "brier_score": brier_score(y.astype(float), p.astype(float)),
        "ece": ece,
        "tn": m["confusion_matrix"]["tn"],
        "fp": m["confusion_matrix"]["fp"],
        "fn": m["confusion_matrix"]["fn"],
        "tp": m["confusion_matrix"]["tp"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/baseline/predictions_test.csv")
    parser.add_argument("--near-duplicates", default="outputs/data_qc/near_duplicate_check.csv")
    parser.add_argument("--output-dir", default="outputs/data_qc")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions)
    near = pd.read_csv(args.near_duplicates)

    near_cols = [c for c in ["near_mse_flag", "near_cosine_flag"] if c in near.columns]
    if not near_cols:
        raise ValueError("Near-duplicate file has no near_mse_flag or near_cosine_flag columns")
    near["remove_flag"] = near[near_cols].any(axis=1)
    remove_ids = set(near.loc[near["remove_flag"], "test_id"].tolist())

    full_y = pred["y_true"].to_numpy(dtype=int)
    full_p = pred["p_class1"].to_numpy(dtype=float)
    keep = ~pred["sample_id"].isin(remove_ids)
    filt_y = pred.loc[keep, "y_true"].to_numpy(dtype=int)
    filt_p = pred.loc[keep, "p_class1"].to_numpy(dtype=float)

    rows = [
        flatten_metrics("full_test", full_y, full_p, args.n_bins),
        flatten_metrics("filtered_test_remove_near_duplicates", filt_y, filt_p, args.n_bins),
    ]
    metrics = pd.DataFrame(rows)
    base = metrics.iloc[0]
    filtered = metrics.iloc[1]
    delta = {
        "set": "delta_filtered_minus_full",
        "n": int(filtered["n"] - base["n"]),
        "class0": int(filtered["class0"] - base["class0"]),
        "class1": int(filtered["class1"] - base["class1"]),
    }
    for col in ["accuracy", "balanced_accuracy", "f1_class1", "auroc", "auprc", "brier_score", "ece"]:
        delta[col] = float(filtered[col] - base[col])
    for col in ["tn", "fp", "fn", "tp"]:
        delta[col] = int(filtered[col] - base[col])
    metrics = pd.concat([metrics, pd.DataFrame([delta])], ignore_index=True)
    metrics.to_csv(out_dir / "near_duplicate_sensitivity_metrics.csv", index=False)

    removed = near.loc[near["remove_flag"]].copy()
    removed.to_csv(out_dir / "near_duplicate_removed_test_samples.csv", index=False)

    md = [
        "# Near-Duplicate Sensitivity Analysis",
        "",
        f"Predictions: `{Path(args.predictions).resolve()}`",
        f"Near-duplicate file: `{Path(args.near_duplicates).resolve()}`",
        "",
        f"Removed test samples: {len(remove_ids)}",
        "",
        "| Metric | Full test | Filtered test | Delta |",
        "|---|---:|---:|---:|",
    ]
    for col in ["n", "accuracy", "balanced_accuracy", "f1_class1", "auroc", "auprc", "brier_score", "ece"]:
        md.append(f"| {col} | {base[col]:.6f} | {filtered[col]:.6f} | {delta[col]:.6f} |")
    conclusion = "Removing near-duplicate flagged test samples did not materially change the baseline metrics."
    if abs(delta["accuracy"]) > 0.01 or abs(delta["auroc"]) > 0.01:
        conclusion = "Removing near-duplicate flagged samples changed at least one primary metric by over 0.01."
    md.extend(
        [
            "",
            f"Conclusion: {conclusion}",
        ]
    )
    (out_dir / "near_duplicate_sensitivity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

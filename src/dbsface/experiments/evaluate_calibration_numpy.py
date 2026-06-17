"""Evaluate binary calibration from saved prediction CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int) -> tuple[pd.DataFrame, float]:
    rows = []
    ece = 0.0
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        count = int(mask.sum())
        if count:
            conf = float(p[mask].mean())
            acc = float(y[mask].mean())
            gap = abs(acc - conf)
            contribution = gap * count / len(y)
        else:
            conf = float("nan")
            acc = float("nan")
            gap = float("nan")
            contribution = 0.0
        ece += contribution
        rows.append(
            {
                "bin": i,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_p_class1": conf,
                "fraction_class1": acc,
                "abs_gap": gap,
                "ece_contribution": contribution,
            }
        )
    return pd.DataFrame(rows), float(ece)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/baseline/predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/calibration")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    y = df["y_true"].to_numpy(dtype=float)
    p = df["p_class1"].to_numpy(dtype=float)

    bins, ece = calibration_bins(y, p, args.n_bins)
    brier = brier_score(y, p)

    bins.to_csv(out_dir / "reliability_bins.csv", index=False)
    (out_dir / "brier_score.txt").write_text(f"{brier:.10f}\n", encoding="utf-8")
    (out_dir / "ece.txt").write_text(f"{ece:.10f}\n", encoding="utf-8")

    metrics = {
        "predictions": str(Path(args.predictions).resolve()),
        "label_semantics": "numeric_class_0_vs_class_1; clinical 0/1 mapping unresolved",
        "n": int(len(df)),
        "n_bins": args.n_bins,
        "brier_score": brier,
        "ece_class1_equal_width": ece,
        "mean_p_class1": float(p.mean()),
        "observed_fraction_class1": float(y.mean()),
    }
    (out_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    md = [
        "# Calibration Summary",
        "",
        f"Predictions: `{Path(args.predictions).resolve()}`",
        "",
        f"N: {metrics['n']}",
        f"Brier score: {brier:.6f}",
        f"ECE, class-1 probability, {args.n_bins} equal-width bins: {ece:.6f}",
        f"Mean predicted p_class1: {metrics['mean_p_class1']:.6f}",
        f"Observed Class 1 fraction: {metrics['observed_fraction_class1']:.6f}",
        "",
        "Interpretation boundary: calibration is evaluated for numeric Class 0 vs Class 1.",
    ]
    (out_dir / "calibration_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



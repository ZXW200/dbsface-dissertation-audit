"""Run multi-seed robustness for the NumPy MLP baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_calibration_numpy import calibration_bins
from load_pd_dbs import load_pd_dbs
from train_baseline_mlp_numpy import (
    fit_standardizer,
    forward,
    metric_summary,
    standardize,
    stratified_split,
    train_mlp,
)


def run_seed(data: dict[str, np.ndarray], seed: int, args: argparse.Namespace) -> dict[str, float | int]:
    x_source = data["x_train_flat"].astype(np.float32)
    y_source = data["y_train"].astype(np.int64)
    x_test_source = data["x_test_flat"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    train_idx, val_idx = stratified_split(y_source, args.val_fraction, seed)
    x_train_raw, y_train = x_source[train_idx], y_source[train_idx]
    x_val_raw, y_val = x_source[val_idx], y_source[val_idx]

    mean, std = fit_standardizer(x_train_raw)
    x_train = standardize(x_train_raw, mean, std)
    x_val = standardize(x_val_raw, mean, std)
    x_test = standardize(x_test_source, mean, std)

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
    p_test = forward(model, x_test)[0]
    metrics = metric_summary(y_test, p_test)
    _, ece = calibration_bins(y_test.astype(float), p_test.astype(float), args.n_bins)
    return {
        "seed": seed,
        "epochs": args.epochs,
        "best_val_loss": min(h["val_loss"] for h in history),
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
    parser.add_argument("--output-dir", default="outputs/robustness")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    data = load_pd_dbs(args.data)
    rows = [run_seed(data, seed, args) for seed in seeds]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "multiseed_metrics.csv", index=False)

    metric_cols = ["accuracy", "balanced_accuracy", "f1_class1", "auroc", "auprc", "brier_score", "ece"]
    summary = []
    for col in metric_cols:
        summary.append({"metric": col, "mean": float(df[col].mean()), "sd": float(df[col].std(ddof=1)), "min": float(df[col].min()), "max": float(df[col].max())})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "multiseed_summary.csv", index=False)

    md = [
        "# Multi-Seed Robustness Summary",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        "Model: NumPy MLP. Labels: Class 0 = pre-DBS; Class 1 = post-DBS label.",
        "This combines seed-dependent train/validation splits with model initialisation variability.",
        "",
        "| Metric | Mean | SD | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        md.append(f"| {row['metric']} | {row['mean']:.6f} | {row['sd']:.6f} | {row['min']:.6f} | {row['max']:.6f} |")
    (out_dir / "multiseed_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Confidence-matched sensitivity analysis for class-specific AEV statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .analyze_aev_class_statistics import (
        bh_fdr,
        bootstrap_ci,
        cliffs_delta,
        cohen_d,
        permutation_p_value,
    )
except ImportError:
    from analyze_aev_class_statistics import (  # type: ignore
        bh_fdr,
        bootstrap_ci,
        cliffs_delta,
        cohen_d,
        permutation_p_value,
    )


def standardised_mean_difference(x0: np.ndarray, x1: np.ndarray) -> float:
    pooled = np.sqrt((np.var(x0, ddof=1) + np.var(x1, ddof=1)) / 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def confidence_match(
    class0: pd.DataFrame,
    class1: pd.DataFrame,
    caliper: float,
) -> list[tuple[int, int]]:
    """Maximise one-to-one matches within a confidence caliper.

    Because the matching variable is one-dimensional, sorted monotone matching
    avoids crossed pairs. Dynamic programming first maximises pair count and
    then minimises total absolute confidence difference.
    """

    c0 = class0["true_conf_original"].to_numpy(dtype=float)
    c1 = class1["true_conf_original"].to_numpy(dtype=float)
    n0, n1 = len(c0), len(c1)

    pair_count = np.zeros((n0 + 1, n1 + 1), dtype=np.int16)
    total_distance = np.zeros((n0 + 1, n1 + 1), dtype=float)
    choice = np.zeros((n0 + 1, n1 + 1), dtype=np.int8)

    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            candidates = [
                (pair_count[i - 1, j], total_distance[i - 1, j], 1),
                (pair_count[i, j - 1], total_distance[i, j - 1], 2),
            ]
            distance = abs(c0[i - 1] - c1[j - 1])
            if distance <= caliper:
                candidates.append(
                    (
                        pair_count[i - 1, j - 1] + 1,
                        total_distance[i - 1, j - 1] + distance,
                        3,
                    )
                )

            best = max(candidates, key=lambda item: (item[0], -item[1], item[2] == 3))
            pair_count[i, j], total_distance[i, j], choice[i, j] = best

    pairs: list[tuple[int, int]] = []
    i, j = n0, n1
    while i > 0 and j > 0:
        if choice[i, j] == 3:
            pairs.append((int(class0.index[i - 1]), int(class1.index[j - 1])))
            i -= 1
            j -= 1
        elif choice[i, j] == 1:
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aev", default="outputs/aev/aev_test.csv")
    parser.add_argument("--output-dir", default="outputs/aev/confidence_matched")
    parser.add_argument("--caliper", type=float, default=0.02)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.aev)
    required = {"sample_id", "y_true", "true_conf_original"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"AEV input is missing required columns: {sorted(missing)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class0 = df[df["y_true"] == 0].sort_values(
        ["true_conf_original", "sample_id"], kind="mergesort"
    )
    class1 = df[df["y_true"] == 1].sort_values(
        ["true_conf_original", "sample_id"], kind="mergesort"
    )
    pairs = confidence_match(class0, class1, args.caliper)
    if not pairs:
        raise RuntimeError("No confidence-matched pairs were found.")

    pair_rows = []
    matched_indices = []
    for pair_id, (idx0, idx1) in enumerate(pairs):
        row0 = df.loc[idx0]
        row1 = df.loc[idx1]
        matched_indices.extend([idx0, idx1])
        pair_rows.append(
            {
                "pair_id": pair_id,
                "class0_sample_id": str(row0["sample_id"]),
                "class1_sample_id": str(row1["sample_id"]),
                "class0_true_conf_original": float(row0["true_conf_original"]),
                "class1_true_conf_original": float(row1["true_conf_original"]),
                "absolute_confidence_difference": abs(
                    float(row0["true_conf_original"] - row1["true_conf_original"])
                ),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "matched_pairs.csv", index=False)

    matched = df.loc[matched_indices].copy()
    matched.to_csv(out_dir / "matched_aev_samples.csv", index=False)

    rng = np.random.default_rng(args.seed)
    roi_cols = [column for column in df.columns if column.startswith("evidence_drop__")]
    rows = []
    for column in roi_cols:
        roi = column.replace("evidence_drop__", "")
        x0 = matched.loc[matched["y_true"] == 0, column].to_numpy(dtype=float)
        x1 = matched.loc[matched["y_true"] == 1, column].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(x0, x1, args.n_bootstrap, rng)
        rows.append(
            {
                "roi": roi,
                "n_class0": int(len(x0)),
                "n_class1": int(len(x1)),
                "mean_class0": float(np.mean(x0)),
                "mean_class1": float(np.mean(x1)),
                "diff_class1_minus_class0": float(np.mean(x1) - np.mean(x0)),
                "cohen_d": cohen_d(x0, x1),
                "cliffs_delta": cliffs_delta(x0, x1),
                "p_perm": permutation_p_value(x0, x1, args.n_permutations, rng),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison["p_fdr"] = bh_fdr(comparison["p_perm"].to_numpy())
    comparison["direction"] = np.where(
        comparison["diff_class1_minus_class0"] > 0,
        "higher in Class 1",
        "higher in Class 0",
    )
    comparison = comparison.sort_values(["p_fdr", "roi"])
    comparison.to_csv(out_dir / "confidence_matched_roi_class_comparison.csv", index=False)

    original_c0 = class0["true_conf_original"].to_numpy(dtype=float)
    original_c1 = class1["true_conf_original"].to_numpy(dtype=float)
    matched_c0 = matched.loc[matched["y_true"] == 0, "true_conf_original"].to_numpy(dtype=float)
    matched_c1 = matched.loc[matched["y_true"] == 1, "true_conf_original"].to_numpy(dtype=float)
    summary = {
        "input": str(Path(args.aev).resolve()),
        "label_convention": {"0": "pre-DBS", "1": "post-DBS"},
        "matching_variable": "true_conf_original",
        "matching_method": "one-to-one monotone optimal matching",
        "caliper": args.caliper,
        "n_pairs": len(pairs),
        "n_class0": len(pairs),
        "n_class1": len(pairs),
        "mean_confidence_class0_before": float(np.mean(original_c0)),
        "mean_confidence_class1_before": float(np.mean(original_c1)),
        "confidence_smd_before": standardised_mean_difference(original_c0, original_c1),
        "mean_confidence_class0_after": float(np.mean(matched_c0)),
        "mean_confidence_class1_after": float(np.mean(matched_c1)),
        "confidence_smd_after": standardised_mean_difference(matched_c0, matched_c1),
        "mean_absolute_pair_difference": float(pair_df["absolute_confidence_difference"].mean()),
        "max_absolute_pair_difference": float(pair_df["absolute_confidence_difference"].max()),
        "n_fdr_positive": int((comparison["p_fdr"] < 0.05).sum()),
        "fdr_positive_rois": comparison.loc[comparison["p_fdr"] < 0.05, "roi"].tolist(),
    }
    (out_dir / "confidence_matching_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Confidence-Matched AEV Class Comparison",
        "",
        "Class 0 represents pre-DBS images and Class 1 represents post-DBS images.",
        "",
        f"Pairs: {len(pairs)} per class; confidence caliper: {args.caliper:.3f}.",
        f"Mean original true-class confidence: Class 0 {np.mean(matched_c0):.4f}, Class 1 {np.mean(matched_c1):.4f}.",
        f"Mean absolute within-pair difference: {pair_df['absolute_confidence_difference'].mean():.4f}.",
        "",
        "| ROI | Difference (Class 1 - Class 0) | Cohen d | Cliff delta | p FDR | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in comparison.iterrows():
        report.append(
            f"| {row['roi']} | {row['diff_class1_minus_class0']:.4f} | "
            f"{row['cohen_d']:.4f} | {row['cliffs_delta']:.4f} | {row['p_fdr']:.4f} | "
            f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}] |"
        )
    (out_dir / "confidence_matched_summary.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

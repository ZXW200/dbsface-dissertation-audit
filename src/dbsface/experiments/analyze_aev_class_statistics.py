"""Class 0 vs Class 1 statistical comparison for occlusion-based AEV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def cohen_d(x0: np.ndarray, x1: np.ndarray) -> float:
    n0, n1 = len(x0), len(x1)
    var0 = np.var(x0, ddof=1)
    var1 = np.var(x1, ddof=1)
    pooled = ((n0 - 1) * var0 + (n1 - 1) * var1) / max(n0 + n1 - 2, 1)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / np.sqrt(pooled))


def cliffs_delta(x0: np.ndarray, x1: np.ndarray) -> float:
    """Cliff's delta for x1 versus x0."""

    x0_sorted = np.sort(x0)
    greater = np.searchsorted(x0_sorted, x1, side="left").sum()
    less = (len(x0_sorted) - np.searchsorted(x0_sorted, x1, side="right")).sum()
    # greater counts x0 < x1; less counts x0 > x1. Ties contribute 0.
    return float((greater - less) / (len(x0) * len(x1)))


def permutation_p_value(x0: np.ndarray, x1: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(x1) - np.mean(x0)))
    pooled = np.concatenate([x0, x1])
    n1 = len(x1)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = abs(float(np.mean(perm[:n1]) - np.mean(perm[n1:])))
        if diff >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def bootstrap_ci(x0: np.ndarray, x1: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        s0 = rng.choice(x0, size=len(x0), replace=True)
        s1 = rng.choice(x1, size=len(x1), replace=True)
        diffs[i] = np.mean(s1) - np.mean(s0)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aev", default="outputs/aev/aev_test.csv")
    parser.add_argument("--output-dir", default="outputs/aev")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.aev)
    rng = np.random.default_rng(args.seed)

    roi_cols = [c for c in df.columns if c.startswith("evidence_drop__")]
    rows = []
    for col in roi_cols:
        roi = col.replace("evidence_drop__", "")
        x0 = df.loc[df["y_true"] == 0, col].to_numpy(dtype=float)
        x1 = df.loc[df["y_true"] == 1, col].to_numpy(dtype=float)
        diff = float(np.mean(x1) - np.mean(x0))
        ci_low, ci_high = bootstrap_ci(x0, x1, args.n_bootstrap, rng)
        rows.append(
            {
                "roi": roi,
                "n_class0": int(len(x0)),
                "n_class1": int(len(x1)),
                "mean_class0": float(np.mean(x0)),
                "mean_class1": float(np.mean(x1)),
                "median_class0": float(np.median(x0)),
                "median_class1": float(np.median(x1)),
                "diff_class1_minus_class0": diff,
                "cohen_d": cohen_d(x0, x1),
                "cliffs_delta": cliffs_delta(x0, x1),
                "p_perm": permutation_p_value(x0, x1, args.n_permutations, rng),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )

    result = pd.DataFrame(rows)
    result["p_fdr"] = bh_fdr(result["p_perm"].to_numpy())
    result = result.sort_values("p_fdr")
    result.to_csv(out_dir / "roi_class_comparison.csv", index=False)

    ranked = result.sort_values("diff_class1_minus_class0", key=lambda s: np.abs(s), ascending=False)
    ranked.to_csv(out_dir / "roi_class_comparison_ranked.csv", index=False)

    md = [
        "# Class 0 vs Class 1 AEV Statistical Comparison",
        "",
        f"Input: `{Path(args.aev).resolve()}`",
        "",
        "Evidence variable: occlusion-based `evidence_drop` for each coarse ROI.",
        "",
        "Project label convention: Class 0 = pre-DBS; Class 1 = post-DBS label.",
        "",
        "## Ranked By FDR-Adjusted Permutation Test",
        "",
        "| ROI | Mean class 0 | Mean class 1 | Diff class1-class0 | Cohen d | Cliff delta | p_perm | p_fdr | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in result.iterrows():
        md.append(
            f"| {row['roi']} | {row['mean_class0']:.6f} | {row['mean_class1']:.6f} | "
            f"{row['diff_class1_minus_class0']:.6f} | {row['cohen_d']:.4f} | {row['cliffs_delta']:.4f} | "
            f"{row['p_perm']:.6f} | {row['p_fdr']:.6f} | "
            f"[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}] |"
        )
    (out_dir / "roi_class_comparison_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    ranked_md = [
        "# ROI Class Difference Ranking",
        "",
        "Ranked by absolute mean difference in evidence_drop between Class 1 and Class 0.",
        "",
        "| ROI | Diff class1-class0 | p_fdr | Direction |",
        "|---|---:|---:|---|",
    ]
    for _, row in ranked.iterrows():
        direction = "higher in Class 1" if row["diff_class1_minus_class0"] > 0 else "higher in Class 0"
        ranked_md.append(f"| {row['roi']} | {row['diff_class1_minus_class0']:.6f} | {row['p_fdr']:.6f} | {direction} |")
    (out_dir / "roi_class_comparison_ranked.md").write_text("\n".join(ranked_md) + "\n", encoding="utf-8")

    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

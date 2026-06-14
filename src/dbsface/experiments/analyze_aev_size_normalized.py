"""Assess ROI-size sensitivity in fixed-atlas AEV mask-out scores."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rank_average(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, with ties assigned their mean rank."""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return 0.0
    return float(np.sum(x * y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rank_average(x), rank_average(y))


def permutation_p_value(x: np.ndarray, y: np.ndarray, fn) -> float:
    """Exact two-sided permutation p-value for the eight-ROI table."""

    observed = abs(fn(x, y))
    count = 0
    total = 0
    for perm in itertools.permutations(y):
        total += 1
        if abs(fn(x, np.asarray(perm, dtype=float))) >= observed - 1e-12:
            count += 1
    return float(count / total)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/aev/roi_occlusion_summary_overall.csv")
    parser.add_argument("--roi-defs", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument("--output-dir", default="outputs/aev")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    roi_defs = pd.read_csv(args.roi_defs)[["roi_index", "roi_name", "pixel_count"]]
    df = summary.merge(roi_defs, on=["roi_index", "roi_name"], how="left")
    if df["pixel_count"].isna().any():
        missing = df.loc[df["pixel_count"].isna(), "roi_name"].tolist()
        raise ValueError(f"Missing pixel_count for ROI(s): {missing}")

    df["drop_per_pixel"] = df["mean_evidence_drop"] / df["pixel_count"]
    df["drop_per_100_pixels"] = df["drop_per_pixel"] * 100.0
    df["raw_rank"] = df["mean_evidence_drop"].rank(method="min", ascending=False).astype(int)
    df["size_normalized_rank"] = df["drop_per_100_pixels"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values("size_normalized_rank")

    area = df["pixel_count"].to_numpy(dtype=float)
    raw_drop = df["mean_evidence_drop"].to_numpy(dtype=float)
    norm_drop = df["drop_per_100_pixels"].to_numpy(dtype=float)

    metrics = {
        "n_rois": int(len(df)),
        "area_vs_raw_mean_drop": {
            "pearson_r": pearson(area, raw_drop),
            "pearson_permutation_p_two_sided": permutation_p_value(area, raw_drop, pearson),
            "spearman_rho": spearman(area, raw_drop),
            "spearman_permutation_p_two_sided": permutation_p_value(area, raw_drop, spearman),
        },
        "area_vs_size_normalized_drop": {
            "pearson_r": pearson(area, norm_drop),
            "pearson_permutation_p_two_sided": permutation_p_value(area, norm_drop, pearson),
            "spearman_rho": spearman(area, norm_drop),
            "spearman_permutation_p_two_sided": permutation_p_value(area, norm_drop, spearman),
        },
        "interpretation": (
            "Raw fixed-atlas mask-out AEV measures total confidence change when a full ROI is masked. "
            "Size-normalized AEV is a sensitivity summary that divides that total change by ROI pixel count."
        ),
    }

    df.to_csv(out_dir / "roi_occlusion_size_normalized.csv", index=False)
    (out_dir / "roi_size_confound_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

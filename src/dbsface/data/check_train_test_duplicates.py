"""Check exact and near-duplicate leakage between the provided train/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from dbsface.data.load_pd_dbs import load_pd_dbs


def row_hashes(x: np.ndarray) -> list[str]:
    x = np.ascontiguousarray(x)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in x]


def nearest_train_rows(x_train: np.ndarray, x_test: np.ndarray, chunk_size: int) -> pd.DataFrame:
    train_sq = np.sum(x_train * x_train, axis=1)
    test_sq = np.sum(x_test * x_test, axis=1)
    train_norm = np.sqrt(train_sq) + 1e-12
    test_norm = np.sqrt(test_sq) + 1e-12

    records = []
    n_features = x_train.shape[1]
    for start in range(0, len(x_test), chunk_size):
        end = min(start + chunk_size, len(x_test))
        block = x_test[start:end]
        dot = block @ x_train.T
        mse = (test_sq[start:end, None] + train_sq[None, :] - 2 * dot) / n_features
        mse = np.maximum(mse, 0.0)
        cosine = dot / (test_norm[start:end, None] * train_norm[None, :])

        min_mse_idx = np.argmin(mse, axis=1)
        max_cos_idx = np.argmax(cosine, axis=1)
        for local_idx, test_idx in enumerate(range(start, end)):
            best_mse_train = int(min_mse_idx[local_idx])
            best_cos_train = int(max_cos_idx[local_idx])
            records.append(
                {
                    "test_id": f"test_{test_idx:04d}",
                    "test_index": test_idx,
                    "nearest_mse_train_id": f"train_{best_mse_train:04d}",
                    "nearest_mse_train_index": best_mse_train,
                    "nearest_mse": float(mse[local_idx, best_mse_train]),
                    "nearest_rmse": float(np.sqrt(mse[local_idx, best_mse_train])),
                    "nearest_cosine_at_mse": float(cosine[local_idx, best_mse_train]),
                    "nearest_cosine_train_id": f"train_{best_cos_train:04d}",
                    "nearest_cosine_train_index": best_cos_train,
                    "max_cosine": float(cosine[local_idx, best_cos_train]),
                    "mse_at_max_cosine": float(mse[local_idx, best_cos_train]),
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-dir", default="outputs/data_qc")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--mse-threshold", type=float, default=1e-5)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pd_dbs(args.data)
    x_train = data["x_train_flat"].astype(np.float32)
    x_test = data["x_test_flat"].astype(np.float32)
    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)

    train_hash = row_hashes(x_train)
    test_hash = row_hashes(x_test)
    train_lookup: dict[str, list[int]] = {}
    for idx, h in enumerate(train_hash):
        train_lookup.setdefault(h, []).append(idx)

    exact_records = []
    for test_idx, h in enumerate(test_hash):
        for train_idx in train_lookup.get(h, []):
            exact_records.append(
                {
                    "test_id": f"test_{test_idx:04d}",
                    "test_index": test_idx,
                    "train_id": f"train_{train_idx:04d}",
                    "train_index": train_idx,
                    "y_test": int(y_test[test_idx]),
                    "y_train": int(y_train[train_idx]),
                    "label_conflict": bool(y_test[test_idx] != y_train[train_idx]),
                    "sha256": h,
                }
            )
    exact_df = pd.DataFrame(
        exact_records,
        columns=["test_id", "test_index", "train_id", "train_index", "y_test", "y_train", "label_conflict", "sha256"],
    )
    exact_df.to_csv(out_dir / "exact_duplicate_check.csv", index=False)

    near_df = nearest_train_rows(x_train, x_test, args.chunk_size)
    near_df["y_test"] = y_test[near_df["test_index"].to_numpy()]
    near_df["y_train_nearest_mse"] = y_train[near_df["nearest_mse_train_index"].to_numpy()]
    near_df["label_conflict_nearest_mse"] = near_df["y_test"] != near_df["y_train_nearest_mse"]
    near_df["near_mse_flag"] = near_df["nearest_mse"] <= args.mse_threshold
    near_df["near_cosine_flag"] = near_df["max_cosine"] >= args.cosine_threshold
    near_df.to_csv(out_dir / "near_duplicate_check.csv", index=False)

    top_near = near_df.sort_values(["nearest_mse", "max_cosine"], ascending=[True, False]).head(50)
    top_near.to_csv(out_dir / "near_duplicate_top50.csv", index=False)

    summary = {
        "data_path": str(Path(args.data).resolve()),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "exact_duplicate_pairs": int(len(exact_df)),
        "exact_label_conflicts": int(exact_df["label_conflict"].sum()) if len(exact_df) else 0,
        "near_mse_threshold": args.mse_threshold,
        "near_mse_pairs": int(near_df["near_mse_flag"].sum()),
        "near_mse_label_conflicts": int((near_df["near_mse_flag"] & near_df["label_conflict_nearest_mse"]).sum()),
        "cosine_threshold": args.cosine_threshold,
        "near_cosine_pairs": int(near_df["near_cosine_flag"].sum()),
        "median_nearest_mse": float(near_df["nearest_mse"].median()),
        "min_nearest_mse": float(near_df["nearest_mse"].min()),
        "median_max_cosine": float(near_df["max_cosine"].median()),
        "max_cosine": float(near_df["max_cosine"].max()),
    }
    (out_dir / "duplicate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    conclusion = "No exact train-test duplicate pairs were found."
    if summary["exact_duplicate_pairs"]:
        conclusion = "Exact train-test duplicate pairs were found and must be treated as leakage risk."
    elif summary["near_mse_pairs"] or summary["near_cosine_pairs"]:
        conclusion = "No exact duplicates were found, but near-duplicate risk remains under the configured thresholds."

    md = [
        "# Train-Test Duplicate / Leakage QC",
        "",
        f"Data file: `{Path(args.data).resolve()}`",
        "",
        f"Train images: {summary['n_train']}",
        f"Test images: {summary['n_test']}",
        "",
        f"Exact duplicate train-test pairs: {summary['exact_duplicate_pairs']}",
        f"Exact duplicate label conflicts: {summary['exact_label_conflicts']}",
        f"Near-duplicate pairs by MSE <= {args.mse_threshold}: {summary['near_mse_pairs']}",
        f"Near-duplicate label conflicts by MSE threshold: {summary['near_mse_label_conflicts']}",
        f"Near-duplicate pairs by cosine >= {args.cosine_threshold}: {summary['near_cosine_pairs']}",
        "",
        f"Minimum nearest-train MSE: {summary['min_nearest_mse']:.8f}",
        f"Median nearest-train MSE: {summary['median_nearest_mse']:.8f}",
        f"Maximum train-test cosine similarity: {summary['max_cosine']:.8f}",
        f"Median maximum train-test cosine similarity: {summary['median_max_cosine']:.8f}",
        "",
        f"Conclusion: {conclusion}",
        "",
        "Residual limitation: patient IDs are absent, so patient-level independence still cannot be verified from this file alone.",
    ]
    (out_dir / "duplicate_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




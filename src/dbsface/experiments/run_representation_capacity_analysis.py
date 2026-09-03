"""Compare representation and estimator-family effects without touching frozen results.

This standalone post-hoc analysis completes a representation x model grid:

* 40 ROI low-level summaries -> logistic regression and one-hidden-layer MLP
* 1,024 raw pixels           -> logistic regression and one-hidden-layer MLP

It also retrains the raw-pixel MLP after independently permuting pixel values
within each named ROI (and within the unassigned-pixel group) for every image.
That control preserves each group's per-image intensity distribution while
destroying stable within-group pixel locations. It does not preserve texture or
gradient summaries.

The script is intentionally isolated from the dissertation pipeline. It reads
the frozen data and ROI masks, refuses to overwrite a non-empty output folder,
and writes only to the explicitly supplied new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent / "data"))

from load_pd_dbs import load_pd_dbs
from run_roi_lowlevel_confound_baseline import extract_features, mask_to_flat
from train_baseline_mlp_numpy import (
    fit_standardizer,
    forward,
    metric_summary,
    standardize,
    stratified_split,
    train_mlp,
)


SCALAR_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1_class1",
    "auroc",
    "auprc",
    "brier_score",
    "mean_bce_loss",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("positive comma-separated values are required")
    return values


def prepare_standardized(
    x_train_all: np.ndarray,
    x_test: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = fit_standardizer(x_train_all[train_idx].astype(np.float32))
    return (
        standardize(x_train_all[train_idx].astype(np.float32), mean, std),
        standardize(x_train_all[val_idx].astype(np.float32), mean, std),
        standardize(x_test.astype(np.float32), mean, std),
    )


def fit_logistic_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    c_grid: list[float],
) -> tuple[np.ndarray, float, float, int, bool]:
    """Select C on validation AUROC; leave the frozen test split untouched."""
    best_model: LogisticRegression | None = None
    best_c = float("nan")
    best_val_auroc = -float("inf")
    for c_value in c_grid:
        model = LogisticRegression(
            C=c_value,
            class_weight=None,
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        val_p = model.predict_proba(x_val)[:, 1]
        val_auroc = float(roc_auc_score(y_val, val_p))
        if val_auroc > best_val_auroc:
            best_model = model
            best_c = float(c_value)
            best_val_auroc = val_auroc
    if best_model is None:
        raise RuntimeError("logistic model selection failed")
    n_iter = int(best_model.n_iter_[0])
    converged = n_iter < int(best_model.max_iter)
    return (
        best_model.predict_proba(x_test)[:, 1].astype(float),
        best_c,
        best_val_auroc,
        n_iter,
        converged,
    )


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    model, history = train_mlp(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        hidden=hidden,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        l2=l2,
        seed=seed,
    )
    return forward(model, x_test)[0].astype(float), history


def pixel_groups(masks: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    if masks.ndim != 3 or masks.shape[1:] != (32, 32):
        raise ValueError(f"expected ROI masks with shape (n, 32, 32), got {masks.shape}")
    assigned = np.zeros(32 * 32, dtype=bool)
    groups: list[np.ndarray] = []
    for mask in masks.astype(bool):
        flat = mask_to_flat(mask)
        if np.any(assigned & flat):
            raise ValueError("ROI masks overlap; within-group shuffle requires disjoint masks")
        idx = np.flatnonzero(flat)
        if len(idx) < 2:
            raise ValueError("each shuffle group must contain at least two pixels")
        assigned |= flat
        groups.append(idx)
    unassigned = np.flatnonzero(~assigned)
    if len(unassigned) >= 2:
        groups.append(unassigned)
    return groups, unassigned


def shuffle_within_groups(
    x: np.ndarray,
    groups: list[np.ndarray],
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Independently permute each pixel group within every image."""
    rng = np.random.default_rng(seed)
    source = np.asarray(x, dtype=np.float32)
    shuffled = source.copy()
    for row_idx in range(len(shuffled)):
        for group in groups:
            shuffled[row_idx, group] = source[row_idx, group][rng.permutation(len(group))]

    max_mean_change = 0.0
    max_sumsq_change = 0.0
    for group in groups:
        source_group = source[:, group].astype(np.float64)
        shuffled_group = shuffled[:, group].astype(np.float64)
        max_mean_change = max(
            max_mean_change,
            float(np.max(np.abs(source_group.mean(axis=1) - shuffled_group.mean(axis=1)))),
        )
        max_sumsq_change = max(
            max_sumsq_change,
            float(
                np.max(
                    np.abs(
                        np.square(source_group).sum(axis=1)
                        - np.square(shuffled_group).sum(axis=1)
                    )
                )
            ),
        )
    changed_fraction = float(np.mean(source != shuffled))
    checks = {
        "max_abs_group_mean_change": max_mean_change,
        "max_abs_group_sumsq_change": max_sumsq_change,
        "changed_pixel_fraction": changed_fraction,
    }
    if max_mean_change > 1e-6 or max_sumsq_change > 1e-4:
        raise RuntimeError(f"shuffle invariance check failed: {checks}")
    return shuffled, checks


def append_metric_rows(
    rows: list[dict[str, float | int | str]],
    seed: int,
    model_name: str,
    y_test: np.ndarray,
    p_test: np.ndarray,
    runtime_seconds: float,
    selected_c: float | None = None,
    validation_auroc: float | None = None,
    logistic_n_iter: int | None = None,
    logistic_converged: bool | None = None,
) -> dict[str, float | int | dict[str, int]]:
    metrics = metric_summary(y_test, p_test)
    row: dict[str, float | int | str] = {
        "seed": seed,
        "model": model_name,
        "runtime_seconds": runtime_seconds,
        "selected_c": float("nan") if selected_c is None else selected_c,
        "validation_auroc": float("nan") if validation_auroc is None else validation_auroc,
        "logistic_n_iter": float("nan") if logistic_n_iter is None else logistic_n_iter,
        "logistic_converged": "" if logistic_converged is None else str(logistic_converged),
    }
    for metric in SCALAR_METRICS:
        row[metric] = float(metrics[metric])
    rows.append(row)
    return metrics


def summarise_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model_name, block in per_seed.groupby("model", sort=False):
        for metric in SCALAR_METRICS + ("runtime_seconds",):
            values = block[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_contrasts(
    y: np.ndarray,
    mean_predictions: dict[str, np.ndarray],
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Image-level paired bootstrap on seed-averaged predictions."""
    simple_contrasts = {
        "lowlevel_mlp_minus_lowlevel_logistic": ("lowlevel_mlp", "lowlevel_logistic"),
        "raw_logistic_minus_lowlevel_logistic": ("raw_logistic", "lowlevel_logistic"),
        "raw_mlp_minus_lowlevel_mlp": ("raw_mlp_reference", "lowlevel_mlp"),
        "raw_mlp_minus_raw_logistic": ("raw_mlp_reference", "raw_logistic"),
        "roi_shuffled_mlp_minus_raw_mlp": ("roi_shuffled_raw_mlp", "raw_mlp_reference"),
    }

    def estimates(index: np.ndarray) -> dict[str, float]:
        y_sample = y[index]
        aucs = {
            name: float(roc_auc_score(y_sample, values[index]))
            for name, values in mean_predictions.items()
        }
        result = {
            label: aucs[left] - aucs[right]
            for label, (left, right) in simple_contrasts.items()
        }
        result["representation_by_estimator_interaction"] = (
            (aucs["raw_mlp_reference"] - aucs["lowlevel_mlp"])
            - (aucs["raw_logistic"] - aucs["lowlevel_logistic"])
        )
        return result

    point = estimates(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in point}
    accepted = 0
    while accepted < n_resamples:
        index = rng.integers(0, len(y), size=len(y))
        if np.unique(y[index]).size < 2:
            continue
        current = estimates(index)
        for name, value in current.items():
            draws[name].append(value)
        accepted += 1
    rows = []
    for name, estimate in point.items():
        values = np.asarray(draws[name], dtype=float)
        rows.append(
            {
                "contrast": name,
                "estimate": estimate,
                "bootstrap_ci_low": float(np.quantile(values, 0.025)),
                "bootstrap_ci_high": float(np.quantile(values, 0.975)),
                "n_resamples": n_resamples,
                "unit": "test image",
                "prediction_aggregation": "mean probability across seeds",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--roi-masks", default="outputs/roi/coarse_roi_masks.npy")
    parser.add_argument("--roi-definitions", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/representation_capacity_analysis_20260718",
    )
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2,3,4"))
    parser.add_argument("--logistic-c-grid", type=parse_float_list, default=parse_float_list("0.001,0.01,0.1,1,10"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    masks_path = Path(args.roi_masks).resolve()
    roi_definitions_path = Path(args.roi_definitions).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data = load_pd_dbs(data_path)
    masks = np.load(masks_path).astype(bool)
    roi_definitions = pd.read_csv(roi_definitions_path)
    roi_names = roi_definitions["roi_name"].astype(str).tolist()
    groups, unassigned = pixel_groups(masks)

    x_raw_train = data["x_train_flat"].astype(np.float32)
    x_raw_test = data["x_test_flat"].astype(np.float32)
    y_train_all = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)
    lowlevel_train = extract_features(x_raw_train, masks, roi_names).to_numpy(np.float32)
    lowlevel_test = extract_features(x_raw_test, masks, roi_names).to_numpy(np.float32)

    metric_rows: list[dict[str, float | int | str]] = []
    delta_rows: list[dict[str, float | int]] = []
    history_rows: list[dict[str, float | int | str]] = []
    shuffle_check_rows: list[dict[str, float | int | str]] = []
    prediction_sets: dict[str, list[np.ndarray]] = {
        "lowlevel_logistic": [],
        "lowlevel_mlp": [],
        "raw_logistic": [],
        "raw_mlp_reference": [],
        "roi_shuffled_raw_mlp": [],
    }

    for seed in args.seeds:
        train_idx, val_idx = stratified_split(y_train_all, val_fraction=0.2, seed=seed)
        y_train = y_train_all[train_idx]
        y_val = y_train_all[val_idx]

        low_train, low_val, low_test = prepare_standardized(
            lowlevel_train, lowlevel_test, train_idx, val_idx
        )
        raw_train, raw_val, raw_test = prepare_standardized(
            x_raw_train, x_raw_test, train_idx, val_idx
        )

        predictions: dict[str, np.ndarray] = {}

        tick = time.perf_counter()
        p, selected_c, val_auroc, n_iter, converged = fit_logistic_with_validation(
            low_train, y_train, low_val, y_val, low_test, seed, args.logistic_c_grid
        )
        predictions["lowlevel_logistic"] = p
        append_metric_rows(
            metric_rows,
            seed,
            "lowlevel_logistic",
            y_test,
            p,
            time.perf_counter() - tick,
            selected_c,
            val_auroc,
            n_iter,
            converged,
        )

        tick = time.perf_counter()
        p, history = fit_mlp(
            low_train,
            y_train,
            low_val,
            y_val,
            low_test,
            seed,
            args.hidden,
            args.epochs,
            args.batch_size,
            args.lr,
            args.l2,
        )
        predictions["lowlevel_mlp"] = p
        append_metric_rows(
            metric_rows,
            seed,
            "lowlevel_mlp",
            y_test,
            p,
            time.perf_counter() - tick,
        )
        history_rows.extend({"seed": seed, "model": "lowlevel_mlp", **row} for row in history)

        tick = time.perf_counter()
        p, selected_c, val_auroc, n_iter, converged = fit_logistic_with_validation(
            raw_train, y_train, raw_val, y_val, raw_test, seed, args.logistic_c_grid
        )
        predictions["raw_logistic"] = p
        append_metric_rows(
            metric_rows,
            seed,
            "raw_logistic",
            y_test,
            p,
            time.perf_counter() - tick,
            selected_c,
            val_auroc,
            n_iter,
            converged,
        )

        tick = time.perf_counter()
        p, history = fit_mlp(
            raw_train,
            y_train,
            raw_val,
            y_val,
            raw_test,
            seed,
            args.hidden,
            args.epochs,
            args.batch_size,
            args.lr,
            args.l2,
        )
        predictions["raw_mlp_reference"] = p
        append_metric_rows(
            metric_rows,
            seed,
            "raw_mlp_reference",
            y_test,
            p,
            time.perf_counter() - tick,
        )
        history_rows.extend({"seed": seed, "model": "raw_mlp_reference", **row} for row in history)

        shuffled_train_all, train_checks = shuffle_within_groups(
            x_raw_train, groups, seed=seed * 100_003 + 17
        )
        shuffled_test, test_checks = shuffle_within_groups(
            x_raw_test, groups, seed=seed * 100_003 + 29
        )
        for split_name, checks in (("train", train_checks), ("test", test_checks)):
            shuffle_check_rows.append({"seed": seed, "split": split_name, **checks})
        shuffled_train, shuffled_val, shuffled_test_std = prepare_standardized(
            shuffled_train_all, shuffled_test, train_idx, val_idx
        )

        tick = time.perf_counter()
        p, history = fit_mlp(
            shuffled_train,
            y_train,
            shuffled_val,
            y_val,
            shuffled_test_std,
            seed,
            args.hidden,
            args.epochs,
            args.batch_size,
            args.lr,
            args.l2,
        )
        predictions["roi_shuffled_raw_mlp"] = p
        append_metric_rows(
            metric_rows,
            seed,
            "roi_shuffled_raw_mlp",
            y_test,
            p,
            time.perf_counter() - tick,
        )
        history_rows.extend({"seed": seed, "model": "roi_shuffled_raw_mlp", **row} for row in history)

        aurocs = {
            name: float(metric_summary(y_test, values)["auroc"])
            for name, values in predictions.items()
        }
        for name, values in predictions.items():
            prediction_sets[name].append(values.copy())
        delta_rows.append(
            {
                "seed": seed,
                "lowlevel_mlp_minus_lowlevel_logistic": aurocs["lowlevel_mlp"] - aurocs["lowlevel_logistic"],
                "raw_logistic_minus_lowlevel_logistic": aurocs["raw_logistic"] - aurocs["lowlevel_logistic"],
                "raw_mlp_minus_lowlevel_mlp": aurocs["raw_mlp_reference"] - aurocs["lowlevel_mlp"],
                "raw_mlp_minus_raw_logistic": aurocs["raw_mlp_reference"] - aurocs["raw_logistic"],
                "roi_shuffled_mlp_minus_raw_mlp": aurocs["roi_shuffled_raw_mlp"] - aurocs["raw_mlp_reference"],
                "representation_by_estimator_interaction": (
                    (aurocs["raw_mlp_reference"] - aurocs["lowlevel_mlp"])
                    - (aurocs["raw_logistic"] - aurocs["lowlevel_logistic"])
                ),
            }
        )

        prediction_frame = pd.DataFrame(
            {
                "sample_index_test": np.arange(len(y_test)),
                "y_true": y_test,
                **{f"p_{name}": values for name, values in predictions.items()},
            }
        )
        prediction_frame.to_csv(output_dir / f"predictions_seed_{seed}.csv", index=False)
        print(
            f"seed {seed}: "
            + ", ".join(f"{name}={value:.4f}" for name, value in aurocs.items())
        )

    per_seed = pd.DataFrame(metric_rows)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summarise_metrics(per_seed).to_csv(output_dir / "metric_summary.csv", index=False)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(output_dir / "auroc_deltas_by_seed.csv", index=False)
    delta_summary = deltas.drop(columns="seed").agg(["mean", "std", "min", "max"]).T.reset_index()
    delta_summary = delta_summary.rename(columns={"index": "contrast"})
    delta_summary.to_csv(output_dir / "auroc_delta_summary.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / "mlp_training_history.csv", index=False)
    pd.DataFrame(shuffle_check_rows).to_csv(output_dir / "shuffle_invariance_checks.csv", index=False)
    mean_predictions = {
        name: np.mean(np.vstack(values), axis=0)
        for name, values in prediction_sets.items()
    }
    paired_bootstrap_contrasts(
        y_test,
        mean_predictions,
        n_resamples=args.bootstrap_resamples,
        seed=20260718,
    ).to_csv(output_dir / "paired_bootstrap_auroc_contrasts.csv", index=False)

    config = {
        "analysis": "representation x estimator-family factorial comparison with within-ROI spatial shuffle",
        "status": "exploratory post-hoc protocol incorporated into the dissertation",
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "roi_masks": str(masks_path),
        "roi_masks_sha256": sha256_file(masks_path),
        "roi_definitions": str(roi_definitions_path),
        "roi_definitions_sha256": sha256_file(roi_definitions_path),
        "n_train_source": int(len(y_train_all)),
        "n_test": int(len(y_test)),
        "n_raw_features": int(x_raw_train.shape[1]),
        "n_lowlevel_features": int(lowlevel_train.shape[1]),
        "roi_names": roi_names,
        "n_shuffle_groups": len(groups),
        "n_unassigned_pixels": int(len(unassigned)),
        "seeds": args.seeds,
        "validation_fraction": 0.2,
        "logistic": {
            "implementation": "sklearn LogisticRegression",
            "class_weight": None,
            "solver": "lbfgs",
            "max_iter": 5000,
            "c_grid_selected_on_validation_auroc": args.logistic_c_grid,
        },
        "mlp": {
            "implementation": "existing NumPy one-hidden-layer MLP",
            "hidden": args.hidden,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "l2": args.l2,
            "checkpoint_selection": "minimum validation BCE within the fixed epoch budget",
        },
        "shuffle": {
            "unit": "independent per-image permutation within each of eight disjoint ROI groups and the unassigned-pixel group",
            "preserves": "per-image pixel multiset within each group",
            "does_not_preserve": "within-group pixel locations, gradients, or texture",
            "applied_to": "source train and frozen test arrays before standardisation; model retrained for each seed",
        },
        "interpretation_boundary": (
            "The analysis compares measured representation and estimator-family effects and tests dependence on "
            "within-ROI spatial arrangement. It cannot identify the physical origin of residual signal as anatomical, "
            "treatment-related, identity-related, session-related, or acquisition-related."
        ),
        "inference_note": (
            "Paired bootstrap intervals use the test image as the resampling unit and seed-averaged predictions. "
            "This is an exploratory post-hoc audit and not subject-level or external-cohort inference."
        ),
        "bootstrap_resamples": args.bootstrap_resamples,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote isolated post-hoc analysis to {output_dir}")
    print(f"Total runtime: {config['total_runtime_seconds']:.1f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

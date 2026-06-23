"""Run a clean matched-seed benchmark for manuscript reporting.

This script deliberately writes to a new output directory.  It does not reuse
the historical single-run benchmark table, so the manuscript can report a
fresh fixed-test comparison under one evaluation protocol:

* the supplied train/test split is preserved for every model;
* each seed uses the same stratified train/validation split;
* normalization is fit only on the seed-specific training fold;
* Class 1 is the post-DBS label and is the positive class;
* CNN-family baselines are trained from scratch on 32 x 32 single-channel
  inputs with low-resolution stems and mild train-only augmentation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dbsface.experiments.final_amoe_support import apply_image_augmentation, save_json, summarize
from dbsface.data.load_pd_dbs import load_pd_dbs
from run_advanced_model_baselines import (
    MODEL_LABELS,
    build_model,
    load_rois,
    make_arrays as make_image_arrays,
    metric_dict as image_metric_dict,
    model_logits,
    normalise,
    predict,
    save_predictions as save_image_predictions,
    set_seed,
)
from dbsface.explain.run_cnn_sklearn_method_comparison import run_logistic
from run_final_amoe_experiments import CANDIDATE_CONFIGS, train_candidate
from dbsface.robustness.run_identity_alignment_audit import markdown_table
from run_multiseed_robustness import run_seed as run_mlp_seed
from run_smallcnn_mild_multiseed import train_smallcnn_mild


METRIC_COLS = [
    "accuracy",
    "balanced_accuracy",
    "f1_class1",
    "auroc",
    "auprc_class1",
    "brier",
    "ece_10bin",
]

DEFAULT_SEEDS = "0,1,2,42,1024,2048"
DEFAULT_MODELS = "logistic,mlp,smallcnn_mild,resnet18,efficientnet_b0,convnext_tiny,roi_amoe"

MODEL_NAMES = {
    "logistic": "Logistic regression",
    "mlp": "MLP",
    "smallcnn_mild": "SmallCNN + mild augmentation",
    "resnet18": "ResNet18 + mild augmentation",
    "efficientnet_b0": "EfficientNet-B0 + mild augmentation",
    "convnext_tiny": "ConvNeXt-Tiny + mild augmentation",
    "roi_amoe": "ROI-AMoE",
}

CNN_FAMILY = {"resnet18", "efficientnet_b0", "convnext_tiny"}


def parse_csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def parse_model_list(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(values) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    if not values:
        raise ValueError("At least one model is required.")
    return values


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def confusion(metrics: dict[str, Any]) -> dict[str, int]:
    cm = metrics.get("confusion_matrix", {})
    return {
        "tn": int(cm.get("tn", 0)),
        "fp": int(cm.get("fp", 0)),
        "fn": int(cm.get("fn", 0)),
        "tp": int(cm.get("tp", 0)),
    }


def canonical_row(model: str, seed: int, metrics: dict[str, Any]) -> dict[str, Any]:
    cm = confusion(metrics)
    if "balanced_accuracy" in metrics:
        balanced_accuracy = float(metrics["balanced_accuracy"])
    else:
        sensitivity = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
        specificity = cm["tn"] / max(cm["tn"] + cm["fp"], 1)
        balanced_accuracy = float(0.5 * (sensitivity + specificity))
    return {
        "model": model,
        "seed": int(seed),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": balanced_accuracy,
        "f1_class1": float(metrics["f1_class1"]),
        "auroc": float(metrics["auroc"]),
        "auprc_class1": float(metrics.get("auprc_class1", metrics.get("auprc"))),
        "brier": float(metrics.get("brier", metrics.get("brier_score"))),
        "ece_10bin": float(metrics.get("ece_10bin", metrics.get("ece"))),
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "tp": cm["tp"],
        "epochs_run": int(metrics.get("epochs_run", metrics.get("epochs", 0))),
    }


def run_logistic_baseline(root: Path, data: dict[str, np.ndarray], seed: int, out_dir: Path, force: bool) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        metrics = load_json(metrics_path)
    else:
        metrics = run_logistic(root, data, out_dir, seed)
        metrics["seed"] = int(seed)
        metrics["label_convention"] = {
            "class_0": "pre-DBS state",
            "class_1": "post-DBS label",
            "positive_class": "Class 1",
        }
        save_json(metrics_path, metrics)
    return canonical_row(MODEL_NAMES["logistic"], seed, metrics)


def run_mlp_baseline(data: dict[str, np.ndarray], seed: int, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        metrics = load_json(metrics_path)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        mlp_args = argparse.Namespace(
            val_fraction=args.mlp_val_fraction,
            epochs=args.mlp_epochs,
            hidden=args.mlp_hidden,
            batch_size=args.mlp_batch_size,
            lr=args.mlp_lr,
            l2=args.mlp_l2,
            n_bins=args.n_bins,
        )
        metrics = dict(run_mlp_seed(data, seed, mlp_args))
        metrics["seed"] = int(seed)
        metrics["model"] = MODEL_NAMES["mlp"]
        metrics["label_convention"] = {
            "class_0": "pre-DBS state",
            "class_1": "post-DBS label",
            "positive_class": "Class 1",
        }
        save_json(metrics_path, metrics)
    return canonical_row(MODEL_NAMES["mlp"], seed, metrics)


def run_smallcnn_mild_baseline(data: dict[str, np.ndarray], seed: int, out_dir: Path, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        metrics = load_json(metrics_path)
    else:
        metrics = train_smallcnn_mild(
            data=data,
            out_dir=out_dir,
            seed=seed,
            device=device,
            batch_size=args.batch_size,
            max_epochs=args.smallcnn_epochs,
            patience=args.smallcnn_patience,
            force=args.force,
        )
        metrics["label_convention"] = {
            "class_0": "pre-DBS state",
            "class_1": "post-DBS label",
            "positive_class": "Class 1",
        }
        save_json(metrics_path, metrics)
    return canonical_row(MODEL_NAMES["smallcnn_mild"], seed, metrics)


def train_cnn_family_mild(
    model_key: str,
    data: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        return load_json(metrics_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    x_train, y_train, x_test, y_test = make_image_arrays(data)
    split = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=seed)
    fit_idx, val_idx = next(split.split(x_train.reshape(len(x_train), -1), y_train))
    train_mean = float(x_train[fit_idx].mean())
    train_std = float(x_train[fit_idx].std())

    x_fit = normalise(x_train[fit_idx], train_mean, train_std)
    x_val = normalise(x_train[val_idx], train_mean, train_std)
    x_test_norm = normalise(x_test, train_mean, train_std)
    y_fit = y_train[fit_idx].astype(np.float32)
    y_val = y_train[val_idx].astype(np.float32)

    model = build_model(model_key).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)), batch_size=args.batch_size, shuffle=True)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    n_pos = float(np.sum(y_fit == 1))
    n_neg = float(np.sum(y_fit == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.cnn_lr, weight_decay=args.weight_decay)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    wait = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.cnn_epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = apply_image_augmentation(xb, "mild", train_mean, train_std)
            optimizer.zero_grad(set_to_none=True)
            logits = model_logits(model, xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_logits = model_logits(model, val_x)
            val_loss = float(criterion(val_logits, val_y).detach().cpu())
            val_p = torch.sigmoid(val_logits).detach().cpu().numpy()
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_accuracy": float(accuracy_score(y_val.astype(int), (val_p >= 0.5).astype(int))),
            "val_auroc": float(roc_auc_score(y_val.astype(int), val_p)),
            "val_auprc_class1": float(average_precision_score(y_val.astype(int), val_p)),
        }
        history.append(row)
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.cnn_patience:
                break

    model.load_state_dict(best_state)
    p_test = predict(model, x_test_norm, device)
    metrics = image_metric_dict(y_test, p_test, MODEL_NAMES[model_key])
    metrics.update(
        {
            "seed": int(seed),
            "epochs_run": int(history[-1]["epoch"]),
            "best_validation_loss": float(best_val_loss),
            "device": str(device),
            "normalization": {"train_mean": train_mean, "train_std": train_std},
            "training": {
                "augmentation_mode": "mild",
                "optimizer": "AdamW",
                "lr": float(args.cnn_lr),
                "weight_decay": float(args.weight_decay),
                "batch_size": int(args.batch_size),
                "validation_fraction": float(args.val_fraction),
                "pos_weight": float(pos_weight.detach().cpu().item()),
            },
            "architecture": {
                "source": MODEL_LABELS[model_key],
                "input": "32 x 32 single-channel grayscale",
                "pretraining": "none",
                "stem": "single-channel low-resolution stem; no ImageNet weights and no upsampling",
            },
            "label_convention": {
                "class_0": "pre-DBS state",
                "class_1": "post-DBS label",
                "positive_class": "Class 1",
            },
        }
    )
    save_json(metrics_path, metrics)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    save_image_predictions(out_dir / "test_predictions.csv", y_test, p_test)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "model": MODEL_NAMES[model_key],
            "normalization": metrics["normalization"],
            "training": metrics["training"],
            "architecture": metrics["architecture"],
        },
        out_dir / "model.pt",
    )
    return metrics


def run_cnn_family_baseline(model_key: str, data: dict[str, np.ndarray], seed: int, out_dir: Path, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    metrics = train_cnn_family_mild(model_key, data, seed, device, out_dir, args)
    return canonical_row(MODEL_NAMES[model_key], seed, metrics)


def run_roi_amoe(data: dict[str, np.ndarray], masks: np.ndarray, seed: int, out_dir: Path, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    metrics = train_candidate(
        data=data,
        masks=masks,
        candidate="final_amoe",
        seed=seed,
        device=device,
        out_dir=out_dir,
        max_epochs=args.amoe_epochs,
        patience=args.amoe_patience,
        batch_size=args.batch_size,
        lr=args.amoe_lr,
        pos_weight_scale=args.amoe_pos_weight_scale,
        force=args.force,
    )
    metrics["label_convention"] = {
        "class_0": "pre-DBS state",
        "class_1": "post-DBS label",
        "positive_class": "Class 1",
    }
    save_json(out_dir / "metrics.json", metrics)
    return canonical_row(MODEL_NAMES["roi_amoe"], seed, metrics)


def exact_signed_rank_p(differences: np.ndarray) -> float:
    nonzero = np.asarray([d for d in differences if abs(float(d)) > 0.0], dtype=float)
    n = len(nonzero)
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(nonzero))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    observed_wplus = float(np.sum(ranks[nonzero > 0.0]))
    total = float(np.sum(ranks))
    observed = min(observed_wplus, total - observed_wplus)
    count = 0
    extreme = 0
    for signs in itertools.product([False, True], repeat=n):
        wplus = float(np.sum(ranks[list(signs)]))
        stat = min(wplus, total - wplus)
        count += 1
        if stat <= observed + 1e-12:
            extreme += 1
    return float(extreme / count)


def write_paired_tests(long_df: pd.DataFrame, output: Path, reference: str = "ROI-AMoE") -> pd.DataFrame:
    ref = long_df[long_df["model"] == reference].set_index("seed")
    rows: list[dict[str, Any]] = []
    for model in [m for m in long_df["model"].drop_duplicates() if m != reference]:
        other = long_df[long_df["model"] == model].set_index("seed")
        common = sorted(set(ref.index).intersection(other.index))
        if not common:
            continue
        for metric in ["accuracy", "balanced_accuracy", "f1_class1", "auroc", "auprc_class1"]:
            diff = ref.loc[common, metric].to_numpy(dtype=float) - other.loc[common, metric].to_numpy(dtype=float)
            rows.append(
                {
                    "comparison": f"{reference} minus {model}",
                    "metric": metric,
                    "n_seeds": int(len(common)),
                    "seeds": ";".join(str(seed) for seed in common),
                    "reference_mean": float(ref.loc[common, metric].mean()),
                    "other_mean": float(other.loc[common, metric].mean()),
                    "mean_difference": float(np.mean(diff)),
                    "median_difference": float(np.median(diff)),
                    "min_difference": float(np.min(diff)),
                    "max_difference": float(np.max(diff)),
                    "exact_signed_rank_p_two_sided": exact_signed_rank_p(diff),
                    "all_seed_differences_positive": bool(np.all(diff > 0.0)),
                }
            )
    paired = pd.DataFrame(rows)
    paired.to_csv(output, index=False)
    return paired


def write_summary_md(summary: pd.DataFrame, paired: pd.DataFrame, output: Path, seeds: list[int], models: list[str]) -> None:
    show_cols = [
        "model",
        "n_seeds",
        "accuracy_mean",
        "accuracy_sd",
        "balanced_accuracy_mean",
        "auroc_mean",
        "auprc_class1_mean",
        "f1_class1_mean",
        "brier_mean",
        "ece_10bin_mean",
    ]
    show = summary[show_cols].copy()
    acc_tests = paired[paired["metric"] == "accuracy"][
        [
            "comparison",
            "n_seeds",
            "reference_mean",
            "other_mean",
            "mean_difference",
            "all_seed_differences_positive",
            "exact_signed_rank_p_two_sided",
        ]
    ].copy()
    lines = [
        "# Clean Fair Matched-Seed Benchmark",
        "",
        f"Completed: {datetime.now().isoformat(timespec='seconds')}",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"Models: {', '.join(models)}",
        "",
        "Class convention: Class 0 is the pre-DBS state; Class 1 is the post-DBS label and the positive class.",
        "",
        "Evaluation protocol: the supplied train/test split is fixed; each seed uses the same stratified 80/20 training/validation split across models; normalization is fit on the seed-specific training fold; CNN-family baselines use 32 x 32 single-channel scratch training, low-resolution stems, mild train-only augmentation, and no ImageNet weights or upsampling.",
        "",
        "## Summary",
        "",
        markdown_table(show, float_digits=4),
        "",
        "## Accuracy Paired Comparisons",
        "",
        markdown_table(acc_tests, float_digits=4),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-dir", default="outputs/fair_benchmark_multiseed")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-val-fraction", type=float, default=0.2)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-l2", type=float, default=1e-4)
    parser.add_argument("--smallcnn-epochs", type=int, default=100)
    parser.add_argument("--smallcnn-patience", type=int, default=18)
    parser.add_argument("--cnn-epochs", type=int, default=90)
    parser.add_argument("--cnn-patience", type=int, default=16)
    parser.add_argument("--cnn-lr", type=float, default=3e-4)
    parser.add_argument("--amoe-epochs", type=int, default=120)
    parser.add_argument("--amoe-patience", type=int, default=18)
    parser.add_argument("--amoe-lr", type=float, default=8e-4)
    parser.add_argument("--amoe-pos-weight-scale", type=float, default=0.60)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_csv_ints(args.seeds)
    models = parse_model_list(args.models)
    data = load_pd_dbs(root / args.data)
    _, masks, _ = load_rois(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for model_key in models:
            model_out = out / "per_seed_artifacts" / f"{model_key}_seed_{seed}"
            print(f"Fair benchmark: {MODEL_NAMES[model_key]}, seed={seed}, device={device}", flush=True)
            if model_key == "logistic":
                row = run_logistic_baseline(root, data, seed, model_out, args.force)
            elif model_key == "mlp":
                row = run_mlp_baseline(data, seed, model_out, args)
            elif model_key == "smallcnn_mild":
                row = run_smallcnn_mild_baseline(data, seed, model_out, args, device)
            elif model_key in CNN_FAMILY:
                row = run_cnn_family_baseline(model_key, data, seed, model_out, args, device)
            elif model_key == "roi_amoe":
                row = run_roi_amoe(data, masks, seed, model_out, args, device)
            else:
                raise ValueError(f"Unhandled model key: {model_key}")
            rows.append(row)
            print(f"  acc={row['accuracy']:.4f}, auroc={row['auroc']:.4f}, f1={row['f1_class1']:.4f}", flush=True)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(out / "fair_benchmark_multiseed_long.csv", index=False)
    summary = summarize(long_df).sort_values("accuracy_mean", ascending=False)
    summary.to_csv(out / "fair_benchmark_multiseed_summary.csv", index=False)
    paired = write_paired_tests(long_df, out / "fair_benchmark_multiseed_paired_tests.csv")
    write_summary_md(summary, paired, out / "fair_benchmark_multiseed_summary.md", seeds, models)
    save_json(
        out / "run_config.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "root": str(root),
            "data": args.data,
            "output_dir": args.output_dir,
            "seeds": seeds,
            "models": models,
            "device": str(device),
            "class_convention": {
                "class_0": "pre-DBS state",
                "class_1": "post-DBS label",
                "positive_class": "Class 1",
            },
            "protocol": {
                "fixed_supplied_test_split": True,
                "stratified_validation_fraction": args.val_fraction,
                "normalization_fit": "seed-specific training fold only",
                "cnn_family": "32 x 32 single-channel scratch training, low-resolution stems, mild train-only augmentation, no ImageNet weights, no upsampling",
                "roi_amoe": CANDIDATE_CONFIGS["final_amoe"]["description"],
            },
            "args": vars(args),
        },
    )
    print((out / "fair_benchmark_multiseed_summary.md").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



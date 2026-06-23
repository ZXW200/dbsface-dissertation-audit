"""Evaluate the retained ROI-AMoE configuration.

The candidates are derived from fixed architectural ablations rather than tuned
by editing test outputs:

1. auxiliary losses did not improve the AMoE ablation;
2. mild train-only augmentation improved the SmallCNN baseline;
3. uniform gating was competitive with learned gating.

This script writes the final repeated-seed and ablation tables for the retained
AMoE line.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR.parent.name == "EXPLAINABLE_FACE2GENE_DIAGNOSIS_SCRIPTS_GROUPED":
    for _GROUPED_DIR in sorted(_SCRIPT_DIR.parent.glob("[0-9][0-9]_*")):
        _GROUPED_PATH = str(_GROUPED_DIR)
        if _GROUPED_PATH not in sys.path:
            sys.path.insert(0, _GROUPED_PATH)
sys.path.append(str(_SCRIPT_DIR))
from dbsface.data.load_pd_dbs import load_pd_dbs
from dbsface.experiments.final_amoe_support import (
    AMoEComponentModel,
    apply_image_augmentation,
    make_arrays,
    metric_dict,
    predict_parts,
    save_json,
    set_seed,
    summarize,
)
from dbsface.explain.run_cnn_sklearn_method_comparison import load_rois
from dbsface.robustness.run_identity_alignment_audit import markdown_table


MODEL_ORDER = [
    "final_amoe",
    "final_amoe_learned_gate",
    "final_amoe_global_only",
    "final_amoe_no_global",
    "final_amoe_no_roi_dropout",
    "final_amoe_no_augmentation",
]

CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {
    "final_amoe": {
        "label": "ROI-AMoE",
        "description": "Residual ROI-aware AMoE with global branch, ROI experts, uniform ROI gate, no auxiliary losses, ROI dropout, and mild train-only image augmentation.",
        "use_global": True,
        "use_roi": True,
        "learned_gate": False,
        "aux_global": 0.0,
        "aux_roi": 0.0,
        "roi_dropout": 0.12,
        "noise_sd": 0.0,
        "contrast_jitter": 0.0,
        "augmentation_mode": "mild",
        "final_mode": "residual_sigmoid",
        "final_roi_scale_init": -1.39,
    },
    "final_amoe_learned_gate": {
        "label": "Ablation: learned ROI gate",
        "description": "Final AMoE setting but with a learned ROI gate instead of uniform ROI weighting.",
        "use_global": True,
        "use_roi": True,
        "learned_gate": True,
        "aux_global": 0.0,
        "aux_roi": 0.0,
        "roi_dropout": 0.12,
        "noise_sd": 0.0,
        "contrast_jitter": 0.0,
        "augmentation_mode": "mild",
        "final_mode": "residual_sigmoid",
        "final_roi_scale_init": -1.39,
    },
    "final_amoe_global_only": {
        "label": "Ablation: global branch only",
        "description": "Final AMoE setting with ROI experts, ROI gate, and ROI fusion removed.",
        "use_global": True,
        "use_roi": False,
        "learned_gate": False,
        "aux_global": 0.0,
        "aux_roi": 0.0,
        "roi_dropout": 0.0,
        "noise_sd": 0.0,
        "contrast_jitter": 0.0,
        "augmentation_mode": "mild",
        "final_mode": "additive",
    },
    "final_amoe_no_global": {
        "label": "Ablation: ROI branch only",
        "description": "Final AMoE setting with the whole-face global branch removed.",
        "use_global": False,
        "use_roi": True,
        "learned_gate": False,
        "aux_global": 0.0,
        "aux_roi": 0.0,
        "roi_dropout": 0.12,
        "noise_sd": 0.0,
        "contrast_jitter": 0.0,
        "augmentation_mode": "mild",
        "final_mode": "additive",
    },
    "final_amoe_no_roi_dropout": {
        "label": "Ablation: no ROI dropout",
        "description": "Final AMoE setting with ROI dropout removed during training.",
        "use_global": True,
        "use_roi": True,
        "learned_gate": False,
        "aux_global": 0.0,
        "aux_roi": 0.0,
        "roi_dropout": 0.0,
        "noise_sd": 0.0,
        "contrast_jitter": 0.0,
        "augmentation_mode": "mild",
        "final_mode": "residual_sigmoid",
        "final_roi_scale_init": -1.39,
    },
    "final_amoe_no_augmentation": {
        "label": "Ablation: no augmentation",
        "description": "Final AMoE setting without mild training-time image augmentation.",
        "use_global": True,
        "use_roi": True,
        "learned_gate": False,
        "aux_global": 0.0,
        "aux_roi": 0.0,
        "roi_dropout": 0.12,
        "noise_sd": 0.0,
        "contrast_jitter": 0.0,
        "augmentation_mode": "none",
        "final_mode": "residual_sigmoid",
        "final_roi_scale_init": -1.39,
    },
}


METRIC_COLS = ["accuracy", "balanced_accuracy", "f1_class1", "auroc", "auprc_class1", "brier", "ece_10bin"]


def parse_int_list(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one seed.")
    return values


def parse_model_list(text: str) -> list[str]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    unknown = sorted(set(values).difference(CANDIDATE_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}. Allowed: {sorted(CANDIDATE_CONFIGS)}")
    return values


def apply_roi_dropout(x: torch.Tensor, masks: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    apply = torch.rand(x.shape[0], device=x.device) < p
    if not torch.any(apply):
        return x
    out = x.clone()
    roi_ids = torch.randint(0, masks.shape[0], (x.shape[0],), device=x.device)
    for idx in torch.where(apply)[0]:
        out[idx] = out[idx] * (1.0 - masks[roi_ids[idx]])
    return out


def canonical_metrics(model: str, seed: int, metrics: dict[str, Any]) -> dict[str, Any]:
    cm = metrics["confusion_matrix"]
    tn = int(cm["tn"])
    fp = int(cm["fp"])
    fn = int(cm["fn"])
    tp = int(cm["tp"])
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "model": model,
        "seed": int(seed),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "f1_class1": float(metrics["f1_class1"]),
        "auroc": float(metrics["auroc"]),
        "auprc_class1": float(metrics["auprc_class1"]),
        "brier": float(metrics["brier"]),
        "ece_10bin": float(metrics["ece_10bin"]),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "epochs_run": int(metrics["epochs_run"]),
    }


def train_candidate(
    data: dict[str, np.ndarray],
    masks: np.ndarray,
    candidate: str,
    seed: int,
    device: torch.device,
    out_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    pos_weight_scale: float,
    force: bool,
) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    set_seed(seed)
    config = CANDIDATE_CONFIGS[candidate]
    x_train, y_train, x_test, y_test = make_arrays(data)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, val_idx = next(split.split(x_train.reshape(len(x_train), -1), y_train))
    mean = float(x_train[fit_idx].mean())
    std = float(x_train[fit_idx].std())
    x_fit = ((x_train[fit_idx] - mean) / max(std, 1e-6)).astype(np.float32)[:, None, :, :]
    x_val = ((x_train[val_idx] - mean) / max(std, 1e-6)).astype(np.float32)[:, None, :, :]
    x_test_norm = ((x_test - mean) / max(std, 1e-6)).astype(np.float32)[:, None, :, :]
    y_fit = y_train[fit_idx].astype(np.float32)
    y_val = y_train[val_idx].astype(np.float32)

    loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)), batch_size=batch_size, shuffle=True)
    model = AMoEComponentModel(masks, config).to(device)
    n_pos = float(np.sum(y_fit == 1))
    n_neg = float(np.sum(y_fit == 0))
    pos_weight_value = (n_neg / max(n_pos, 1.0)) * pos_weight_scale
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    roi_masks_t = torch.from_numpy(masks.astype(np.float32))[:, None, :, :].to(device)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    wait = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = apply_image_augmentation(xb, str(config["augmentation_mode"]), mean, std)
            xb = apply_roi_dropout(xb, roi_masks_t, float(config["roi_dropout"]))
            optimizer.zero_grad(set_to_none=True)
            parts = model(xb, return_parts=True)
            loss = criterion(parts["logit"], yb)
            if float(config["aux_global"]) > 0:
                loss = loss + float(config["aux_global"]) * criterion(parts["global_logit"], yb)
            if float(config["aux_roi"]) > 0:
                roi_target = yb[:, None].expand_as(parts["roi_logits"])
                roi_loss = F.binary_cross_entropy_with_logits(parts["roi_logits"], roi_target, reduction="none")
                loss = loss + float(config["aux_roi"]) * roi_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_parts = model(val_x, return_parts=True)
            val_loss = float(criterion(val_parts["logit"], val_y).detach().cpu())
            val_p = torch.sigmoid(val_parts["logit"]).detach().cpu().numpy()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_accuracy": float(accuracy_score(y_val.astype(int), (val_p >= 0.5).astype(int))),
            "val_auroc": float(roc_auc_score(y_val.astype(int), val_p)),
            "val_auprc_class1": float(average_precision_score(y_val.astype(int), val_p)),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    test_parts = predict_parts(model, x_test_norm, device)
    metrics = metric_dict(y_test, test_parts["p"], candidate, str(config["label"]))
    cm = metrics["confusion_matrix"]
    metrics.update(
        {
            "candidate": candidate,
            "description": config["description"],
            "balanced_accuracy": float(
                0.5
                * (
                    cm["tp"] / max(cm["tp"] + cm["fn"], 1)
                    + cm["tn"] / max(cm["tn"] + cm["fp"], 1)
                )
            ),
            "epochs_run": int(history[-1]["epoch"]),
            "best_validation_loss": float(best_val_loss),
            "seed": int(seed),
            "device": str(device),
            "normalization": {"train_mean": mean, "train_std": std},
            "training": {
                "augmentation_mode": config["augmentation_mode"],
                "aux_global_weight": float(config["aux_global"]),
                "aux_roi_weight": float(config["aux_roi"]),
                "roi_dropout": float(config["roi_dropout"]),
                "pos_weight_scale": float(pos_weight_scale),
                "effective_pos_weight": float(pos_weight_value),
                "lr": float(lr),
                "batch_size": int(batch_size),
            },
            "architecture": {
                "use_global": bool(config["use_global"]),
                "use_roi": bool(config["use_roi"]),
                "learned_gate": bool(config["learned_gate"]),
            },
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_path, metrics)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(test_parts["gates"], columns=[f"roi_{i + 1}_gate" for i in range(8)]).to_csv(out_dir / "test_roi_gates.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": [f"test_{i:04d}" for i in range(len(y_test))],
            "y_true": y_test,
            "p_class1": test_parts["p"],
            "y_pred": (test_parts["p"] >= 0.5).astype(int),
            "correct": ((test_parts["p"] >= 0.5).astype(int) == y_test).astype(int),
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "candidate": candidate,
            "config": config,
            "seed": seed,
            "train_mean": mean,
            "train_std": std,
        },
        out_dir / "model.pt",
    )
    return metrics


def save_summary_md(summary: pd.DataFrame, out: Path, seeds: list[int]) -> None:
    show = summary[
        [
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
    ]
    lines = [
        "# ROI-AMoE Final Multi-Seed Experiment",
        "",
        f"Completed: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        "This run evaluates the retained residual ROI-AMoE configuration: uniform ROI gate, no auxiliary losses, ROI dropout, and mild train-only image augmentation.",
        "",
        markdown_table(show, float_digits=4),
        "",
        "Interpretation: this table reports the retained AMoE fixed-test result under repeated seeds. It is a split-level modelling result and should be read with the leakage and grouped-similarity audits.",
    ]
    (out / "final_amoe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    plot_df = summary.sort_values("accuracy_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7.8, 3.6), dpi=220)
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["accuracy_mean"], xerr=plot_df["accuracy_sd"], color="#19314f", alpha=0.92, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlim(0.90, 1.0)
    ax.set_xlabel("Accuracy, mean +/- SD")
    ax.set_title("ROI-AMoE final repeated-seed result", fontweight="bold")
    ax.grid(axis="x", color="#e3e8f0", lw=0.7)
    for pos, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(float(row["accuracy_mean"]) + float(row["accuracy_sd"]) + 0.003, pos, f"{row['accuracy_mean']:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-dir", default="outputs/robustness/final_amoe_multiseed")
    parser.add_argument("--seeds", default="0,1,2,42,1024,2048")
    parser.add_argument("--candidates", default="final_amoe")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--pos-weight-scale", type=float, default=0.60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    candidates = parse_model_list(args.candidates)
    data = load_pd_dbs(root / args.data)
    _, masks, _ = load_rois(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for candidate in candidates:
            label = str(CANDIDATE_CONFIGS[candidate]["label"])
            model_out = out / "per_seed_artifacts" / f"{candidate}_seed_{seed}"
            print(f"ROI-AMoE run: {label}, seed={seed}, device={device}")
            metrics = train_candidate(
                data=data,
                masks=masks,
                candidate=candidate,
                seed=seed,
                device=device,
                out_dir=model_out,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                pos_weight_scale=args.pos_weight_scale,
                force=args.force,
            )
            row = canonical_metrics(label, seed, metrics)
            rows.append(row)
            print(f"  acc={row['accuracy']:.4f}, auroc={row['auroc']:.4f}, f1={row['f1_class1']:.4f}")

    long_df = pd.DataFrame(rows)
    long_df.to_csv(out / "final_amoe_long.csv", index=False)
    summary = summarize(long_df)
    order = {str(CANDIDATE_CONFIGS[key]["label"]): i for i, key in enumerate(MODEL_ORDER)}
    summary = summary.sort_values("model", key=lambda s: s.map(order))
    summary.to_csv(out / "final_amoe_summary.csv", index=False)
    save_summary_md(summary, out, seeds)
    plot_summary(summary, out / "final_amoe_accuracy.png")
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "completed": datetime.now().isoformat(timespec="seconds"),
                "root": str(root),
                "device": str(device),
                **vars(args),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print((out / "final_amoe_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



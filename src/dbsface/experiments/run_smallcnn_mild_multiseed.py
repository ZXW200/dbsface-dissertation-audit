"""Run SmallCNN mild-augmentation repeated-seed benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dbsface.experiments.final_amoe_support import apply_image_augmentation, save_predictions, summarize
from dbsface.data.load_pd_dbs import load_pd_dbs
from dbsface.explain.run_cnn_sklearn_method_comparison import (
    SmallFaceCNN,
    make_cnn_arrays,
    metric_dict,
    normalize_cnn_images,
    predict_cnn,
    save_json,
    set_seed,
)
from dbsface.robustness.run_identity_alignment_audit import markdown_table


def parse_int_list(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one seed.")
    return values


def train_smallcnn_mild(
    data: dict[str, np.ndarray],
    out_dir: Path,
    seed: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    force: bool,
) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    x_train, y_train, x_test, y_test = make_cnn_arrays(data)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, val_idx = next(split.split(x_train.reshape(len(x_train), -1), y_train))
    train_mean = float(x_train[fit_idx].mean())
    train_std = float(x_train[fit_idx].std())
    train_mean_image = x_train[fit_idx].mean(axis=0).astype(np.float32)

    x_fit = normalize_cnn_images(x_train[fit_idx], train_mean, train_std)
    x_val = normalize_cnn_images(x_train[val_idx], train_mean, train_std)
    x_test_norm = normalize_cnn_images(x_test, train_mean, train_std)
    y_fit = y_train[fit_idx].astype(np.float32)
    y_val = y_train[val_idx].astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_x_t = torch.from_numpy(x_val).to(device)
    val_y_t = torch.from_numpy(y_val).to(device)
    model = SmallFaceCNN().to(device)
    n_pos = float(np.sum(y_fit == 1))
    n_neg = float(np.sum(y_fit == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    wait = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = apply_image_augmentation(xb, "mild", train_mean, train_std)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_t)
            val_loss = float(criterion(val_logits, val_y_t).detach().cpu())
            val_p = torch.sigmoid(val_logits).detach().cpu().numpy()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_accuracy": float(accuracy_score(y_val.astype(int), (val_p >= 0.5).astype(int))),
            "val_auroc": float(roc_auc_score(y_val.astype(int), val_p)),
            "val_auprc_class1": float(average_precision_score(y_val.astype(int), val_p)),
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
    p_test = predict_cnn(model, x_test_norm, device)
    metrics = metric_dict(y_test, p_test, "SmallCNN mild augmentation")
    metrics["epochs_run"] = int(history[-1]["epoch"])
    metrics["best_validation_loss"] = best_val_loss
    metrics["normalization"] = {"train_mean": train_mean, "train_std": train_std}
    metrics["device"] = str(device)
    metrics["augmentation_mode"] = "mild"
    metrics["seed"] = int(seed)
    save_json(out_dir / "metrics.json", metrics)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    save_predictions(out_dir / "test_predictions.csv", y_test, p_test)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_mean": train_mean,
            "train_std": train_std,
            "train_mean_image": train_mean_image,
            "seed": seed,
            "model": "SmallFaceCNN",
            "augmentation_mode": "mild",
        },
        out_dir / "model.pt",
    )
    return metrics


def row_from_metrics(seed: int, metrics: dict[str, Any]) -> dict[str, Any]:
    cm = metrics.get("confusion_matrix", {})
    return {
        "model": "SmallCNN mild augmentation",
        "seed": int(seed),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "f1_class1": float(metrics["f1_class1"]),
        "auroc": float(metrics["auroc"]),
        "auprc_class1": float(metrics["auprc_class1"]),
        "brier": float(metrics["brier"]),
        "ece_10bin": float(metrics["ece_10bin"]),
        "tn": int(cm.get("tn", 0)),
        "fp": int(cm.get("fp", 0)),
        "fn": int(cm.get("fn", 0)),
        "tp": int(cm.get("tp", 0)),
        "epochs_run": int(metrics["epochs_run"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-dir", default="outputs/augmentation_ablation/smallcnn_mild_matched_multiseed")
    parser.add_argument("--seeds", default="0,1,2,42,1024,2048")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = load_pd_dbs(root / args.data)
    seeds = parse_int_list(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for seed in seeds:
        seed_dir = out / "per_seed_artifacts" / f"smallcnn_mild_seed_{seed}"
        print(f"SmallCNN mild multiseed: seed={seed}, device={device}", flush=True)
        metrics = train_smallcnn_mild(
            data=data,
            out_dir=seed_dir,
            seed=seed,
            device=device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            force=args.force,
        )
        row = row_from_metrics(seed, metrics)
        rows.append(row)
        print(f"  acc={row['accuracy']:.4f}, auroc={row['auroc']:.4f}, f1={row['f1_class1']:.4f}", flush=True)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(out / "smallcnn_mild_matched_multiseed_long.csv", index=False)
    summary = summarize(long_df)
    summary.to_csv(out / "smallcnn_mild_matched_multiseed_summary.csv", index=False)
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
        "# SmallCNN Mild-Augmentation Matched Multi-Seed Experiment",
        "",
        f"Completed: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        markdown_table(show, float_digits=4),
        "",
        "Interpretation: this table reports fixed-test seed variation for SmallCNN with mild train-only augmentation under the same seed list as the final ROI-AMoE repeated-seed run.",
    ]
    (out / "smallcnn_mild_matched_multiseed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    print((out / "smallcnn_mild_matched_multiseed_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""Run region-only validation with YuNet-derived dynamic ROI boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs
from train_baseline_mlp_numpy import forward, metric_summary, standardize


ROI_NAMES = [
    "upper_brow_forehead",
    "left_periocular",
    "right_periocular",
    "nasal_midface",
    "left_cheek_zygomatic",
    "right_cheek_zygomatic",
    "perioral_mouth",
    "chin_mandible",
]


ROI_COLORS = [
    (220, 70, 70),
    (70, 120, 220),
    (70, 190, 120),
    (240, 190, 70),
    (170, 90, 200),
    (70, 190, 190),
    (230, 90, 150),
    (120, 120, 120),
]


def load_model(path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ckpt = np.load(path)
    model = {key: ckpt[key] for key in ["w1", "b1", "w2", "b2"]}
    return model, ckpt["mean"], ckpt["std"]


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr).squeeze().astype(np.float32)
    if x.max() <= 1.5:
        x = x * 255.0
    else:
        lo, hi = np.percentile(x, [1, 99])
        if hi > lo:
            x = (x - lo) * 255.0 / (hi - lo)
    return np.clip(x, 0, 255).astype(np.uint8)


def fixed_mask_to_flat(mask: np.ndarray) -> np.ndarray:
    return mask.T.reshape(-1).astype(bool)


def clip_box(box: tuple[float, float, float, float], size: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(round(max(0, min(size - 1, x1))))
    y1 = int(round(max(0, min(size - 1, y1))))
    x2 = int(round(max(x1 + 1, min(size, x2))))
    y2 = int(round(max(y1 + 1, min(size, y2))))
    return x1, y1, x2, y2


def row_to_yunet_boxes(row: pd.Series, size: int) -> dict[str, tuple[int, int, int, int]]:
    pts = np.array(
        [
            [row["left_eye_x"], row["left_eye_y"]],
            [row["right_eye_x"], row["right_eye_y"]],
            [row["nose_x"], row["nose_y"]],
            [row["left_mouth_x"], row["left_mouth_y"]],
            [row["right_mouth_x"], row["right_mouth_y"]],
        ],
        dtype=float,
    )
    eyes = sorted([pts[0], pts[1]], key=lambda p: p[0])
    mouths = sorted([pts[3], pts[4]], key=lambda p: p[0])
    left_eye, right_eye = eyes
    nose = pts[2]
    left_mouth, right_mouth = mouths

    eye_dist = max(float(right_eye[0] - left_eye[0]), size * 0.18)
    eye_y = float((left_eye[1] + right_eye[1]) / 2.0)
    mouth_y = float((left_mouth[1] + right_mouth[1]) / 2.0)
    mouth_w = max(float(right_mouth[0] - left_mouth[0]), eye_dist * 0.55)
    mouth_x = float((left_mouth[0] + right_mouth[0]) / 2.0)
    nose_x = float(nose[0])

    boxes = [
        (
            min(left_eye[0], right_eye[0]) - 0.45 * eye_dist,
            max(0.0, eye_y - 0.55 * size),
            max(left_eye[0], right_eye[0]) + 0.45 * eye_dist,
            eye_y - 0.10 * size,
        ),
        (
            left_eye[0] - 0.45 * eye_dist,
            left_eye[1] - 0.18 * size,
            left_eye[0] + 0.35 * eye_dist,
            left_eye[1] + 0.16 * size,
        ),
        (
            right_eye[0] - 0.35 * eye_dist,
            right_eye[1] - 0.18 * size,
            right_eye[0] + 0.45 * eye_dist,
            right_eye[1] + 0.16 * size,
        ),
        (
            nose_x - 0.18 * eye_dist,
            eye_y + 0.02 * size,
            nose_x + 0.18 * eye_dist,
            mouth_y + 0.10 * size,
        ),
        (
            max(0.0, left_eye[0] - 0.55 * eye_dist),
            eye_y + 0.12 * size,
            nose_x - 0.08 * eye_dist,
            mouth_y + 0.08 * size,
        ),
        (
            nose_x + 0.08 * eye_dist,
            eye_y + 0.12 * size,
            min(float(size), right_eye[0] + 0.55 * eye_dist),
            mouth_y + 0.08 * size,
        ),
        (
            mouth_x - 0.75 * mouth_w,
            mouth_y - 0.14 * size,
            mouth_x + 0.75 * mouth_w,
            mouth_y + 0.18 * size,
        ),
        (
            mouth_x - 0.70 * mouth_w,
            mouth_y + 0.14 * size,
            mouth_x + 0.70 * mouth_w,
            min(float(size), mouth_y + 0.50 * size),
        ),
    ]
    return {name: clip_box(box, size) for name, box in zip(ROI_NAMES, boxes)}


def box_to_mask32(box: tuple[int, int, int, int], upsample_size: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    mx1 = int(np.floor(x1 / upsample_size * 32))
    my1 = int(np.floor(y1 / upsample_size * 32))
    mx2 = int(np.ceil(x2 / upsample_size * 32))
    my2 = int(np.ceil(y2 / upsample_size * 32))
    mx1 = max(0, min(31, mx1))
    my1 = max(0, min(31, my1))
    mx2 = max(mx1 + 1, min(32, mx2))
    my2 = max(my1 + 1, min(32, my2))
    mask = np.zeros((32, 32), dtype=bool)
    mask[my1:my2, mx1:mx2] = True
    return fixed_mask_to_flat(mask)


def true_confidence(y: np.ndarray, p_class1: np.ndarray) -> np.ndarray:
    return np.where(y == 1, p_class1, 1.0 - p_class1)


def draw_yunet_roi_tile(gray32: np.ndarray, row: pd.Series, boxes: dict[str, tuple[int, int, int, int]], size: int) -> Image.Image:
    gray = to_uint8_image(gray32)
    up = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
    rgb = np.repeat(up[..., None], 3, axis=2)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), f"{row['sample_id']} y={int(row['label'])} s={row['score']:.2f}", fill=(255, 255, 0))
    draw.rectangle([row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]], outline=(0, 255, 0), width=2)
    points = [
        (row["left_eye_x"], row["left_eye_y"]),
        (row["right_eye_x"], row["right_eye_y"]),
        (row["nose_x"], row["nose_y"]),
        (row["left_mouth_x"], row["left_mouth_y"]),
        (row["right_mouth_x"], row["right_mouth_y"]),
    ]
    for px, py in points:
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 0, 0))
    for color, name in zip(ROI_COLORS, ROI_NAMES):
        x1, y1, x2, y2 = boxes[name]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return img


def make_contact_sheet(rows: pd.DataFrame, images: np.ndarray, boxes_by_global: dict[int, dict[str, tuple[int, int, int, int]]], out_path: Path, size: int) -> None:
    tiles = []
    for _, row in rows.iterrows():
        idx = int(row["global_index"])
        tile = draw_yunet_roi_tile(images[idx], row, boxes_by_global[idx], size).resize((128, 128), Image.Resampling.NEAREST)
        tiles.append(tile)

    cols = 6
    rows_n = max(1, int(np.ceil(len(tiles) / cols)))
    canvas = Image.new("RGB", (cols * 148, rows_n * 164 + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "YuNet landmark5-derived 8 ROI boxes on upsampled 32x32 PD-DBS", fill=(0, 0, 0))
    for i, tile in enumerate(tiles):
        x = (i % cols) * 148
        y = 28 + (i // cols) * 164
        canvas.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--model", default="models/baseline_mlp_numpy.npz")
    parser.add_argument("--yunet-audit", default="outputs/external/pd_dbs_yunet_feasibility/yunet_detection_audit.csv")
    parser.add_argument("--output-dir", default="outputs/external/pd_dbs_yunet_region_only")
    parser.add_argument("--upsample-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pd_dbs(args.data)
    x_test = data["x_test_flat"].astype(np.float32)
    y = data["y_test"].astype(int)
    n_train = len(data["x_train_flat"])
    model, mean, std = load_model(args.model)

    audit = pd.read_csv(args.yunet_audit)
    detected = audit[audit["detected"] == 1].copy()
    boxes_by_global = {int(row["global_index"]): row_to_yunet_boxes(row, args.upsample_size) for _, row in detected.iterrows()}

    test_rows = audit[audit["split"] == "test"].copy()
    metrics_rows = []
    prediction_rows = []
    json_metrics = {}
    roi_box_rows = []

    for _, row in detected.iterrows():
        global_idx = int(row["global_index"])
        for roi_index, roi_name in enumerate(ROI_NAMES, start=1):
            x1, y1, x2, y2 = boxes_by_global[global_idx][roi_name]
            roi_box_rows.append(
                {
                    "global_index": global_idx,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "label": int(row["label"]),
                    "roi_index": roi_index,
                    "roi_name": roi_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "source": "yunet_landmark5",
                    "score": float(row["score"]),
                }
            )
    pd.DataFrame(roi_box_rows).to_csv(out_dir / "yunet_roi_boxes.csv", index=False)

    for roi_idx, roi_name in enumerate(ROI_NAMES, start=1):
        x_region = np.repeat(mean.astype(np.float32), len(x_test), axis=0)
        pixel_counts = []
        missing_count = 0
        for i in range(len(x_test)):
            global_idx = n_train + i
            boxes = boxes_by_global.get(global_idx)
            if boxes is None:
                missing_count += 1
                continue
            flat_mask = box_to_mask32(boxes[roi_name], args.upsample_size)
            pixel_counts.append(int(flat_mask.sum()))
            x_region[i, flat_mask] = x_test[i, flat_mask]

        p = forward(model, standardize(x_region, mean, std))[0]
        pred = (p >= 0.5).astype(int)
        true_conf = true_confidence(y, p)
        metrics = metric_summary(y, p)
        json_metrics[roi_name] = metrics
        metrics_rows.append(
            {
                "roi_index": roi_idx,
                "roi_name": roi_name,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_class1": metrics["f1_class1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier_score": metrics["brier_score"],
                "mean_true_confidence": float(true_conf.mean()),
                "majority_baseline_accuracy": metrics["majority_baseline_accuracy"],
                "missing_yunet_count": missing_count,
                "mean_dynamic_roi_pixels": float(np.mean(pixel_counts)),
                "tn": metrics["confusion_matrix"]["tn"],
                "fp": metrics["confusion_matrix"]["fp"],
                "fn": metrics["confusion_matrix"]["fn"],
                "tp": metrics["confusion_matrix"]["tp"],
            }
        )
        for i in range(len(y)):
            prediction_rows.append(
                {
                    "sample_id": f"test_{i:04d}",
                    "split": "test",
                    "roi_index": roi_idx,
                    "roi_name": roi_name,
                    "y_true": int(y[i]),
                    "p_class1": float(p[i]),
                    "y_pred": int(pred[i]),
                    "correct": bool(pred[i] == y[i]),
                    "true_confidence": float(true_conf[i]),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("balanced_accuracy", ascending=False)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df.to_csv(out_dir / "yunet_region_only_metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "yunet_region_only_predictions.csv", index=False)
    (out_dir / "yunet_region_only_metrics.json").write_text(json.dumps(json_metrics, indent=2), encoding="utf-8")

    images = np.concatenate([data["x_train_images"], data["x_test_images"]], axis=0)
    sample_rows = audit.sample(min(24, len(audit)), random_state=42)
    make_contact_sheet(sample_rows, images, boxes_by_global, out_dir / "qc_yunet_roi_examples.jpg", args.upsample_size)

    md = [
        "# YuNet Region-Only Validation Summary",
        "",
        "Only one YuNet landmark5-derived dynamic ROI is retained at a time; all other pixels are replaced with the training mean.",
        "",
        "Class convention: Class 0 = pre-DBS; Class 1 = post-DBS label. This is an ROI-definition sensitivity analysis.",
        "",
        "| ROI | Accuracy | Balanced accuracy | AUROC | AUPRC | Mean true confidence | Missing n | Mean ROI pixels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.iterrows():
        md.append(
            f"| {row['roi_name']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['auroc']:.4f} | {row['auprc']:.4f} | {row['mean_true_confidence']:.4f} | "
            f"{int(row['missing_yunet_count'])} | {row['mean_dynamic_roi_pixels']:.1f} |"
        )
    (out_dir / "yunet_region_only_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

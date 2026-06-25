"""Audit whether YuNet landmark detection is usable on 32x32 PD-DBS images.

The original PD-DBS dataset already contains low-resolution 32x32 grayscale face
crops. This script upsamples those crops, runs OpenCV YuNet face detection, and
writes a feasibility report plus visual QC sheets. It does not replace the
fixed ROI atlas; it tests whether detector-driven ROIs are technically viable.
"""

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


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr).squeeze().astype(np.float32)
    if x.max() <= 1.5:
        x = x * 255.0
    else:
        lo, hi = np.percentile(x, [1, 99])
        if hi > lo:
            x = (x - lo) * 255.0 / (hi - lo)
    return np.clip(x, 0, 255).astype(np.uint8)


def run_yunet(detector: cv2.FaceDetectorYN, gray32: np.ndarray, upsample_size: int) -> tuple[np.ndarray | None, float]:
    gray = to_uint8_image(gray32)
    up = cv2.resize(gray, (upsample_size, upsample_size), interpolation=cv2.INTER_CUBIC)
    bgr = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    detector.setInputSize((upsample_size, upsample_size))
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None, float("nan")
    faces = np.asarray(faces, dtype=float)
    best = faces[np.argmax(faces[:, 14])]
    return best, float(best[14])


def draw_detection_tile(gray32: np.ndarray, face: np.ndarray | None, label: int, sample_id: str, upsample_size: int) -> Image.Image:
    gray = to_uint8_image(gray32)
    up = cv2.resize(gray, (upsample_size, upsample_size), interpolation=cv2.INTER_NEAREST)
    rgb = np.repeat(up[..., None], 3, axis=2)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    if face is not None:
        x, y, w, h = face[:4]
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
        pts = face[4:14].reshape(5, 2)
        for i, (px, py) in enumerate(pts):
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 0, 0))
            draw.text((px + 2, py + 2), str(i), fill=(255, 0, 0))
        draw.text((2, 2), f"{sample_id} y={label} s={face[14]:.2f}", fill=(255, 255, 0))
    else:
        draw.text((2, 2), f"{sample_id} y={label} no face", fill=(255, 255, 0))
    return img


def make_contact_sheet(rows: pd.DataFrame, images: np.ndarray, out_path: Path, upsample_size: int, title: str) -> None:
    tiles = []
    for _, row in rows.iterrows():
        idx = int(row["global_index"])
        face = None
        if int(row["detected"]):
            face = np.array(
                [
                    row["x"],
                    row["y"],
                    row["w"],
                    row["h"],
                    row["left_eye_x"],
                    row["left_eye_y"],
                    row["right_eye_x"],
                    row["right_eye_y"],
                    row["nose_x"],
                    row["nose_y"],
                    row["left_mouth_x"],
                    row["left_mouth_y"],
                    row["right_mouth_x"],
                    row["right_mouth_y"],
                    row["score"],
                ],
                dtype=float,
            )
        tile = draw_detection_tile(images[idx], face, int(row["label"]), str(row["sample_id"]), upsample_size)
        tiles.append(tile.resize((128, 128), Image.Resampling.NEAREST))

    cols = 6
    rows_n = max(1, int(np.ceil(len(tiles) / cols)))
    canvas = Image.new("RGB", (cols * 148, rows_n * 164 + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0))
    for i, tile in enumerate(tiles):
        x = (i % cols) * 148
        y = 28 + (i // cols) * 164
        canvas.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--yunet-model", default="models/external/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--output-dir", default="outputs/external/pd_dbs_yunet_feasibility")
    parser.add_argument("--upsample-size", type=int, default=256)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.yunet_model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    data = load_pd_dbs(args.data)
    train_images = data["x_train_images"]
    test_images = data["x_test_images"]
    images = np.concatenate([train_images, test_images], axis=0)
    labels = np.concatenate([data["y_train"], data["y_test"]], axis=0)
    split = np.array(["train"] * len(train_images) + ["test"] * len(test_images))
    sample_ids = [f"train_{i:04d}" for i in range(len(train_images))] + [f"test_{i:04d}" for i in range(len(test_images))]

    detector = cv2.FaceDetectorYN_create(
        str(model_path),
        "",
        (args.upsample_size, args.upsample_size),
        score_threshold=args.score_threshold,
        nms_threshold=0.30,
        top_k=20,
    )

    rows = []
    for idx, (img, label, sample_split, sample_id) in enumerate(zip(images, labels, split, sample_ids)):
        face, score = run_yunet(detector, img, args.upsample_size)
        row = {
            "global_index": idx,
            "sample_id": sample_id,
            "split": sample_split,
            "label": int(label),
            "detected": int(face is not None),
            "score": score,
            "x": float("nan"),
            "y": float("nan"),
            "w": float("nan"),
            "h": float("nan"),
            "left_eye_x": float("nan"),
            "left_eye_y": float("nan"),
            "right_eye_x": float("nan"),
            "right_eye_y": float("nan"),
            "nose_x": float("nan"),
            "nose_y": float("nan"),
            "left_mouth_x": float("nan"),
            "left_mouth_y": float("nan"),
            "right_mouth_x": float("nan"),
            "right_mouth_y": float("nan"),
        }
        if face is not None:
            row.update(
                {
                    "x": float(face[0]),
                    "y": float(face[1]),
                    "w": float(face[2]),
                    "h": float(face[3]),
                    "left_eye_x": float(face[4]),
                    "left_eye_y": float(face[5]),
                    "right_eye_x": float(face[6]),
                    "right_eye_y": float(face[7]),
                    "nose_x": float(face[8]),
                    "nose_y": float(face[9]),
                    "left_mouth_x": float(face[10]),
                    "left_mouth_y": float(face[11]),
                    "right_mouth_x": float(face[12]),
                    "right_mouth_y": float(face[13]),
                }
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "yunet_detection_audit.csv", index=False)

    summary = {
        "n_images": int(len(df)),
        "upsample_size": args.upsample_size,
        "score_threshold": args.score_threshold,
        "detected_total": int(df["detected"].sum()),
        "detected_fraction": float(df["detected"].mean()),
        "detected_by_split": df.groupby("split")["detected"].mean().to_dict(),
        "detected_by_label": df.groupby("label")["detected"].mean().to_dict(),
        "n_detected_by_label": df.groupby("label")["detected"].sum().astype(int).to_dict(),
        "median_score_detected": float(df.loc[df["detected"] == 1, "score"].median()) if df["detected"].any() else float("nan"),
    }
    (out_dir / "yunet_detection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rng = np.random.default_rng(42)
    detected = df[df["detected"] == 1].copy()
    failed = df[df["detected"] == 0].copy()
    detected_sample = detected.sample(min(24, len(detected)), random_state=42) if len(detected) else detected
    failed_sample = failed.sample(min(24, len(failed)), random_state=42) if len(failed) else failed
    make_contact_sheet(detected_sample, images, out_dir / "qc_detected_examples.jpg", args.upsample_size, "YuNet detections on upsampled 32x32 PD-DBS")
    make_contact_sheet(failed_sample, images, out_dir / "qc_failed_examples.jpg", args.upsample_size, "YuNet failures on upsampled 32x32 PD-DBS")

    lines = [
        "# PD-DBS YuNet Feasibility Audit",
        "",
        "This audit tests whether an automatic face detector plus five landmarks can be used directly on the original 32x32 grayscale PD-DBS images after upsampling.",
        "",
        f"- Images tested: {summary['n_images']}",
        f"- Upsample size: {args.upsample_size}",
        f"- YuNet score threshold: {args.score_threshold}",
        f"- Detected images: {summary['detected_total']} ({summary['detected_fraction']:.3f})",
        f"- Detection fraction by label: {summary['detected_by_label']}",
        f"- Median detected score: {summary['median_score_detected']:.3f}",
        "",
        "Interpretation: if detection is sparse or visually unstable, the original fixed ROI atlas should remain the primary method for the 32x32 DBS dataset. Detector-driven ROIs are better suited to the higher-resolution YouTubePD raw-video extraction.",
        "",
        "QC files:",
        "- `qc_detected_examples.jpg`",
        "- `qc_failed_examples.jpg`",
        "- `yunet_detection_audit.csv`",
    ]
    (out_dir / "yunet_detection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

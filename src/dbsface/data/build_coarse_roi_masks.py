"""Build 8 coarse anatomical ROI masks for 32x32 PD-DBS face images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent))
from dbsface.data.load_pd_dbs import load_pd_dbs


ROI_DEFS = [
    ("upper_brow_forehead", "Upper face / brow / forehead", 0, 8, 4, 28),
    ("left_periocular", "Image-left periocular region", 8, 15, 0, 14),
    ("right_periocular", "Image-right periocular region", 8, 15, 18, 32),
    ("nasal_midface", "Nasal bridge / central midface", 8, 22, 14, 18),
    ("left_cheek_zygomatic", "Image-left cheek / zygomatic region", 15, 24, 0, 14),
    ("right_cheek_zygomatic", "Image-right cheek / zygomatic region", 15, 24, 18, 32),
    ("perioral_mouth", "Perioral / mouth region", 24, 28, 8, 24),
    ("chin_mandible", "Chin / mandible region", 28, 32, 6, 26),
]

COLORS = [
    (230, 70, 70),
    (70, 130, 230),
    (70, 180, 130),
    (220, 160, 50),
    (160, 90, 200),
    (60, 190, 190),
    (240, 100, 170),
    (100, 100, 100),
]


def build_masks() -> tuple[list[str], np.ndarray, pd.DataFrame]:
    masks = []
    rows = []
    names = []
    for idx, (name, description, y0, y1, x0, x1) in enumerate(ROI_DEFS, start=1):
        mask = np.zeros((32, 32), dtype=bool)
        mask[y0:y1, x0:x1] = True
        masks.append(mask)
        names.append(name)
        rows.append(
            {
                "roi_index": idx,
                "roi_name": name,
                "description": description,
                "y_start": y0,
                "y_end_exclusive": y1,
                "x_start": x0,
                "x_end_exclusive": x1,
                "pixel_count": int(mask.sum()),
            }
        )
    stacked = np.stack(masks, axis=0)
    overlap = stacked.sum(axis=0)
    if int(overlap.max()) > 1:
        raise ValueError(f"Coarse ROI masks must be mutually exclusive; found {int((overlap > 1).sum())} overlapping pixels")
    return names, stacked, pd.DataFrame(rows)


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.squeeze()
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def overlay_masks(base: np.ndarray, masks: np.ndarray, names: list[str], scale: int = 10) -> Image.Image:
    gray = (normalize_image(base) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for mask, color in zip(masks, COLORS):
        color_arr = np.array(color, dtype=np.float32)
        rgb[mask] = 0.55 * rgb[mask] + 0.45 * color_arr
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((32 * scale, 32 * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    for idx, (mask, name, color) in enumerate(zip(masks, names, COLORS), start=1):
        ys, xs = np.where(mask)
        if len(xs):
            x = int(xs.mean() * scale)
            y = int(ys.mean() * scale)
            draw.rectangle([x, y, x + 13, y + 11], fill=(255, 255, 255))
            draw.text((x + 2, y), str(idx), fill=color)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-dir", default="outputs/roi")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, masks, definitions = build_masks()
    np.save(out_dir / "coarse_roi_masks.npy", masks)
    definitions.to_csv(out_dir / "coarse_roi_definitions.csv", index=False)

    data = load_pd_dbs(args.data)
    sample = data["x_test_images"][args.sample_index]
    overlay = overlay_masks(sample, masks, names)

    legend_width = 360
    canvas = Image.new("RGB", (overlay.width + legend_width, overlay.height), "white")
    canvas.paste(overlay, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((overlay.width + 12, 12), "Coarse ROI legend", fill=(0, 0, 0))
    for idx, row in definitions.iterrows():
        y = 38 + idx * 28
        color = COLORS[idx]
        draw.rectangle([overlay.width + 12, y, overlay.width + 28, y + 16], fill=color)
        draw.text((overlay.width + 36, y), f"{idx+1}. {row['roi_name']}", fill=(0, 0, 0))
    canvas.save(out_dir / "coarse_roi_overlay_examples.png")

    table_lines = [
        "| ROI | Name | Pixels | Rows | Cols |",
        "|---:|---|---:|---|---|",
    ]
    for _, row in definitions.iterrows():
        table_lines.append(
            f"| {row['roi_index']} | {row['roi_name']} | {row['pixel_count']} | "
            f"{row['y_start']}:{row['y_end_exclusive']} | {row['x_start']}:{row['x_end_exclusive']} |"
        )
    summary = [
        "# Coarse ROI Masks",
        "",
        "These 8 mutually exclusive coarse anatomical ROIs are a fallback for 32x32 grayscale images where fine landmark-based 18-ROI extraction may be unstable.",
        "",
        *table_lines,
    ]
    (out_dir / "coarse_roi_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(definitions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



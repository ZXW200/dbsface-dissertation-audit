"""Create dissertation figures from frozen CSV/PNG outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from dbsface.data.load_pd_dbs import load_pd_dbs
except ImportError:  # pragma: no cover - allows direct execution from package folders
    load_pd_dbs = None


OUT = Path("latex_project/figures/dissertation")


PALETTE = {
    "navy": (42, 67, 101),
    "blue": (76, 130, 180),
    "teal": (70, 150, 145),
    "green": (87, 148, 103),
    "orange": (204, 132, 67),
    "red": (190, 83, 83),
    "gray": (105, 112, 122),
    "light": (242, 245, 248),
    "grid": (218, 224, 230),
    "text": (35, 39, 45),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def text(draw: ImageDraw.ImageDraw, xy, s: str, size=24, fill=None, bold=False, anchor=None) -> None:
    draw.text(xy, s, font=font(size, bold), fill=fill or PALETTE["text"], anchor=anchor)


def vertical_text(img: Image.Image, xy, s: str, size=22, fill=None) -> None:
    """Draw a vertical axis label with enough internal margin to avoid clipping."""
    fnt = font(size)
    probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), s, font=fnt)
    width = bbox[2] - bbox[0] + 16
    height = bbox[3] - bbox[1] + 16
    label = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((8, 8), s, font=fnt, fill=fill or PALETTE["gray"])
    rotated = label.rotate(90, expand=True)
    x, y = xy
    img.paste(rotated, (int(x - rotated.width / 2), int(y - rotated.height / 2)), rotated)


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], x_ticks: list[float], x_min: float, x_max: float) -> None:
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=PALETTE["text"], width=2)
    for tick in x_ticks:
        x = x0 + (tick - x_min) / (x_max - x_min) * (x1 - x0)
        draw.line((x, y0, x, y1), fill=PALETTE["grid"], width=1)
        draw.line((x, y1, x, y1 + 8), fill=PALETTE["text"], width=2)
        text(draw, (x, y1 + 14), f"{tick:.2f}", 18, fill=PALETTE["gray"], anchor="ma")


def hbar(
    rows: list[tuple[str, float]],
    title: str,
    subtitle: str,
    x_label: str,
    path: Path,
    x_min: float | None = None,
    x_max: float | None = None,
    diverging: bool = False,
) -> None:
    w, h = 1400, 820
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (60, 38), title, 34, bold=True)
    text(draw, (60, 82), subtitle, 22, fill=PALETTE["gray"])
    left, top, right, bottom = 390, 135, 1320, 720
    vals = [v for _, v in rows]
    if x_min is None:
        x_min = min(0, min(vals) * 1.15)
    if x_max is None:
        x_max = max(vals) * 1.15
    if x_max == x_min:
        x_max = x_min + 1
    ticks = [x_min + i * (x_max - x_min) / 4 for i in range(5)]
    draw_axes(draw, (left, top, right, bottom), ticks, x_min, x_max)
    if diverging and x_min < 0 < x_max:
        zx = left + (0 - x_min) / (x_max - x_min) * (right - left)
        draw.line((zx, top, zx, bottom), fill=PALETTE["text"], width=2)
    row_h = (bottom - top) / len(rows)
    for i, (label, val) in enumerate(rows):
        y = top + i * row_h + row_h * 0.5
        text(draw, (left - 18, y), label.replace("_", " "), 20, anchor="rm")
        x_val = left + (val - x_min) / (x_max - x_min) * (right - left)
        x_zero = left + (0 - x_min) / (x_max - x_min) * (right - left)
        if diverging:
            color = PALETTE["blue"] if val >= 0 else PALETTE["orange"]
            x_start, x_end = sorted([x_zero, x_val])
        else:
            color = PALETTE["teal"]
            x_start, x_end = left, x_val
        draw.rounded_rectangle((x_start, y - 16, x_end, y + 16), radius=7, fill=color)
        value_anchor = "lm" if val >= 0 or not diverging else "rm"
        value_x = x_end + 10 if val >= 0 or not diverging else x_start - 10
        text(draw, (value_x, y), f"{val:.3f}", 18, fill=PALETTE["text"], anchor=value_anchor)
    text(draw, ((left + right) / 2, h - 45), x_label, 20, fill=PALETTE["gray"], anchor="ma")
    img.save(path)


def line_plot(points: list[tuple[float, float]], title: str, subtitle: str, x_label: str, y_label: str, path: Path) -> None:
    w, h = 1300, 850
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (60, 38), title, 34, bold=True)
    text(draw, (60, 82), subtitle, 22, fill=PALETTE["gray"])
    left, top, right, bottom = 190, 145, 1180, 720
    draw.rectangle((left, top, right, bottom), outline=PALETTE["text"], width=2)
    for i in range(1, 5):
        x = left + i * (right - left) / 5
        y = top + i * (bottom - top) / 5
        draw.line((x, top, x, bottom), fill=PALETTE["grid"], width=1)
        draw.line((left, y, right, y), fill=PALETTE["grid"], width=1)
    def px(x, y):
        return (left + x * (right - left), bottom - y * (bottom - top))
    draw.line((left, bottom, right, top), fill=PALETTE["gray"], width=2)
    pts = [px(x, y) for x, y in points]
    if len(pts) > 1:
        draw.line(pts, fill=PALETTE["blue"], width=4)
    for p in pts:
        draw.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), fill=PALETTE["blue"])
    text(draw, ((left + right) / 2, h - 55), x_label, 22, fill=PALETTE["gray"], anchor="ma")
    vertical_text(img, (70, (top + bottom) / 2), y_label, 22, fill=PALETTE["gray"])
    for v in [0, 0.5, 1.0]:
        text(draw, (left + v * (right - left), bottom + 16), f"{v:.1f}", 18, fill=PALETTE["gray"], anchor="ma")
        text(draw, (left - 18, bottom - v * (bottom - top)), f"{v:.1f}", 18, fill=PALETTE["gray"], anchor="rm")
    img.save(path)


def combine_images(paths: list[Path], labels: list[str], title: str, subtitle: str, output: Path) -> None:
    w, h = 1500, 1100
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (50, 35), title, 34, bold=True)
    text(draw, (50, 78), subtitle, 22, fill=PALETTE["gray"])
    boxes = [(50, 130, 720, 530), (780, 130, 1450, 530), (50, 620, 720, 1020), (780, 620, 1450, 1020)]
    for path, label, box in zip(paths, labels, boxes):
        panel = Image.open(path).convert("RGB")
        max_w, max_h = box[2] - box[0], box[3] - box[1] - 36
        panel.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        x = box[0] + (max_w - panel.width) // 2
        y = box[1] + 36 + (max_h - panel.height) // 2
        text(draw, (box[0], box[1]), label, 24, bold=True)
        draw.rectangle(box, outline=PALETTE["grid"], width=2)
        img.paste(panel, (x, y))
    img.save(output)


def pipeline_schematic(output: Path) -> None:
    w, h = 1850, 860
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (60, 42), "Facial ROI evidence audit workflow", 38, bold=True)
    text(
        draw,
        (60, 88),
        "The dissertation converts pre- and post-DBS facial-model behaviour into auditable ROI-level evidence.",
        22,
        fill=PALETTE["gray"],
    )
    boxes = [
        ("PD-DBS\n32\u00d732 faces", "Class 0 pre-DBS\nClass 1 post-DBS"),
        ("Data QC", "arrays, labels,\nsimilarity"),
        ("NumPy MLP", "p(Class 1) and\nheld-out metrics"),
        ("Calibration", "Brier, ECE,\nreliability bins"),
        ("ROI setup", "fixed atlas plus\nYuNet sensitivity"),
        ("Mask-out AEV", "true-class\nconfidence drop"),
        ("ROI tests", "region-only,\nclass statistics"),
        ("Audit", "robustness and\nstructured reporting"),
    ]
    x0, y0 = 70, 185
    box_w, box_h, gap = 195, 145, 26
    for i, (title, subtitle) in enumerate(boxes):
        x = x0 + i * (box_w + gap)
        fill = PALETTE["light"] if i % 2 == 0 else (236, 246, 244)
        draw.rounded_rectangle((x, y0, x + box_w, y0 + box_h), radius=12, fill=fill, outline=PALETTE["grid"], width=2)
        text(draw, (x + box_w / 2, y0 + 34), title, 22, bold=True, anchor="ma")
        for j, line in enumerate(subtitle.split("\n")):
            text(draw, (x + box_w / 2, y0 + 78 + j * 24), line, 18, fill=PALETTE["gray"], anchor="ma")
        if i < len(boxes) - 1:
            ax0 = x + box_w + 4
            ax1 = x + box_w + gap - 4
            ay = y0 + box_h / 2
            draw.line((ax0, ay, ax1, ay), fill=PALETTE["gray"], width=3)
            draw.polygon([(ax1, ay), (ax1 - 12, ay - 7), (ax1 - 12, ay + 7)], fill=PALETTE["gray"])
    branch_y = 390
    roi_x = x0 + 4 * (box_w + gap)
    draw.line((roi_x + box_w / 2, y0 + box_h, roi_x + box_w / 2, branch_y), fill=PALETTE["gray"], width=3)
    draw.rounded_rectangle((roi_x - 20, branch_y, roi_x + box_w + 20, branch_y + 112), radius=12, fill=(255, 249, 235), outline=PALETTE["orange"], width=2)
    text(draw, (roi_x + box_w / 2, branch_y + 27), "YuNet-derived ROI", 22, bold=True, anchor="ma")
    text(draw, (roi_x + box_w / 2, branch_y + 62), "automatic ROI\nsensitivity check", 17, fill=PALETTE["gray"], anchor="ma")

    note_y = 560
    draw.rounded_rectangle((120, note_y, 1730, note_y + 220), radius=16, fill=(250, 250, 252), outline=PALETTE["grid"], width=2)
    text(draw, (150, note_y + 30), "Evaluation outputs", 27, bold=True)
    bullets = [
        "Classification, calibration, and reliability metrics.",
        "Per-image AEV, class statistics, and retained-region responses.",
        "Similarity, perturbation, ROI-geometry, and cross-method sensitivity.",
    ]
    for j, bullet in enumerate(bullets):
        text(draw, (168, note_y + 78 + j * 29), f"- {bullet}", 20, fill=PALETTE["text"])
    img.save(output)


def _roi_label(name: str) -> str:
    return name.replace("_", " ")


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


def _short_roi_label(name: str) -> str:
    labels = {
        "upper_brow_forehead": "upper brow/forehead",
        "left_periocular": "left periocular",
        "right_periocular": "right periocular",
        "nasal_midface": "nasal midface",
        "left_cheek_zygomatic": "left cheek/zygomatic",
        "right_cheek_zygomatic": "right cheek/zygomatic",
        "perioral_mouth": "perioral/mouth",
        "chin_mandible": "chin/mandible",
    }
    return labels.get(name, name.replace("_", " "))


def _compact_roi_label(name: str) -> str:
    labels = {
        "upper_brow_forehead": "upper brow/forehead",
        "left_periocular": "left periocular",
        "right_periocular": "right periocular",
        "nasal_midface": "nasal midface",
        "left_cheek_zygomatic": "left cheek/zygomatic",
        "right_cheek_zygomatic": "right cheek/zygomatic",
        "perioral_mouth": "perioral/mouth",
        "chin_mandible": "chin/mandible",
    }
    return labels.get(name, name.replace("_", " "))


def _scaled_face(arr: np.ndarray, size: int, vmin: float | None = None, vmax: float | None = None) -> Image.Image:
    arr = arr.astype(float)
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())
    scaled = (arr - vmin) / max(vmax - vmin, 1e-6)
    scaled = np.clip(scaled, 0, 1)
    return Image.fromarray((scaled * 255).astype("uint8"), mode="L").resize(
        (size, size),
        Image.Resampling.NEAREST,
    ).convert("RGB")


def _draw_roi_outline(tile: Image.Image, roi_def: dict[str, str], colour: tuple[int, int, int], width: int = 3) -> None:
    draw = ImageDraw.Draw(tile)
    scale_x = tile.width / 32
    scale_y = tile.height / 32
    x0 = int(float(roi_def["x_start"]) * scale_x)
    x1 = int(float(roi_def["x_end_exclusive"]) * scale_x) - 1
    y0 = int(float(roi_def["y_start"]) * scale_y)
    y1 = int(float(roi_def["y_end_exclusive"]) * scale_y) - 1
    draw.rectangle((x0, y0, x1, y1), outline=colour, width=width)


def _draw_aev_bar_panel(
    draw: ImageDraw.ImageDraw,
    sample_row: dict[str, str],
    roi_defs: list[dict[str, str]],
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    label_x = x0 + 220
    axis_x0 = x0 + 245
    axis_x1 = x1 - 120
    top = y0 + 44
    bottom = y1 - 48
    x_min, x_max = -0.15, 0.85
    zero_x = axis_x0 + (0 - x_min) / (x_max - x_min) * (axis_x1 - axis_x0)

    text(draw, (x0, y0), "AEV components for this sample", 22, bold=True)
    text(
        draw,
        (x0, y1 - 24),
        "Evidence drop = true-class confidence(original) - true-class confidence(masked ROI)",
        15,
        fill=PALETTE["gray"],
    )

    for tick in [-0.10, 0.00, 0.40, 0.80]:
        x = axis_x0 + (tick - x_min) / (x_max - x_min) * (axis_x1 - axis_x0)
        draw.line((x, top, x, bottom), fill=PALETTE["grid"], width=1)
        text(draw, (x, bottom + 8), f"{tick:.1f}", 13, fill=PALETTE["gray"], anchor="ma")
    draw.line((zero_x, top, zero_x, bottom), fill=PALETTE["text"], width=2)

    row_h = (bottom - top) / len(roi_defs)
    for i, roi_def in enumerate(roi_defs):
        roi = roi_def["roi_name"]
        val = float(sample_row[f"evidence_drop__{roi}"])
        y = top + i * row_h + row_h * 0.5
        text(draw, (label_x, y), _compact_roi_label(roi), 17, anchor="rm")
        x_val = axis_x0 + (val - x_min) / (x_max - x_min) * (axis_x1 - axis_x0)
        colour = PALETTE["red"] if val >= 0 else PALETTE["blue"]
        bx0, bx1 = sorted((zero_x, x_val))
        draw.rectangle((bx0, y - 9, bx1, y + 9), fill=colour)
        text(draw, (axis_x1 + 12, y), f"{val:+.3f}", 15, fill=colour, anchor="lm")


def _draw_dual_aev_row(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    y: int,
    sample_row: dict[str, str],
    sample_image: np.ndarray,
    train_mean_image: np.ndarray,
    masks: np.ndarray,
    roi_defs: list[dict[str, str]],
) -> None:
    """Draw one full-width AEV example with print-readable labels."""
    sample_id = sample_row["sample_id"]
    y_true = int(sample_row["y_true"])
    y_pred = int(sample_row["y_pred_original"])
    p_class1 = float(sample_row["p_class1_original"])
    true_conf = float(sample_row["true_conf_original"])
    row_fill = (247, 250, 252) if y_true == 0 else (255, 250, 242)
    row_outline = PALETTE["grid"] if y_true == 0 else (225, 195, 150)
    draw.rounded_rectangle((48, y, 1852, y + 345), radius=18, fill=row_fill, outline=row_outline, width=2)

    # At 1900 px across an approximately 160 mm text width, 21 px is about 5 pt.
    text(draw, (75, y + 17), f"{sample_id} | true Class {y_true} | predicted Class {y_pred}", 24, bold=True)
    text(
        draw,
        (75, y + 49),
        f"p(Class 1) = {p_class1:.4f}, true-class confidence = {true_conf:.4f}",
        21,
        fill=PALETTE["gray"],
    )

    vmin, vmax = float(sample_image.min()), float(sample_image.max())
    face = _scaled_face(sample_image, 128, vmin, vmax)
    draw.rounded_rectangle((78, y + 88, 230, y + 270), radius=12, fill="white", outline=PALETTE["grid"], width=2)
    img.paste(face, (90, y + 100))
    text(draw, (154, y + 235), "original face", 21, fill=PALETTE["gray"], anchor="ma")

    conf_label = f"p(Class {y_true}) = {true_conf:.4f}"
    draw.rounded_rectangle((52, y + 281, 256, y + 329), radius=12, fill=(232, 247, 245), outline=PALETTE["teal"], width=2)
    text(draw, (154, y + 305), conf_label, 21, bold=True, fill=PALETTE["teal"], anchor="mm")

    text(draw, (270, y + 79), "Eight fixed-atlas mask-out inputs", 21, bold=True)
    tile_size = 72
    tile_gap_x = 33
    tile_gap_y = 33
    start_x = 275
    start_y = y + 108
    for i, roi_def in enumerate(roi_defs):
        row = i // 4
        col = i % 4
        tile_x = start_x + col * (tile_size + tile_gap_x)
        tile_y = start_y + row * (tile_size + tile_gap_y)
        masked = sample_image.copy()
        masked[masks[i]] = train_mean_image[masks[i]]
        tile = _scaled_face(masked, tile_size, vmin, vmax)
        _draw_roi_outline(tile, roi_def, PALETTE["red"], width=2)
        draw.rounded_rectangle(
            (tile_x - 5, tile_y - 5, tile_x + tile_size + 5, tile_y + tile_size + 29),
            radius=8,
            fill="white",
            outline=PALETTE["grid"],
            width=1,
        )
        img.paste(tile, (tile_x, tile_y))
        text(draw, (tile_x + tile_size / 2, tile_y + tile_size + 7), f"ROI {i + 1}", 21, bold=True, anchor="ma")

    draw.line((697, y + 136, 750, y + 136), fill=PALETTE["teal"], width=5)
    draw.polygon([(750, y + 136), (728, y + 124), (728, y + 148)], fill=PALETTE["teal"])

    bar_x0, label_x = 780, 1040
    axis_x0, axis_x1, value_x = 1060, 1690, 1815
    top, bottom = y + 111, y + 287
    x_min, x_max = -0.15, 0.85
    zero_x = axis_x0 + (0 - x_min) / (x_max - x_min) * (axis_x1 - axis_x0)
    text(draw, (bar_x0, y + 78), "AEV components", 22, bold=True)
    text(draw, (1005, y + 79), "Top four drops marked 1 to 4", 21, fill=PALETTE["gray"])

    ranked = sorted(
        [
            (roi_def["roi_name"], float(sample_row[f"evidence_drop__{roi_def['roi_name']}"]))
            for roi_def in roi_defs
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:4]
    rank_by_roi = {roi: rank for rank, (roi, _) in enumerate(ranked, start=1)}

    for tick in [-0.10, 0.00, 0.40, 0.80]:
        tx = axis_x0 + (tick - x_min) / (x_max - x_min) * (axis_x1 - axis_x0)
        draw.line((tx, top, tx, bottom), fill=PALETTE["grid"], width=1)
        text(draw, (tx, bottom + 8), f"{tick:.1f}", 21, fill=PALETTE["gray"], anchor="ma")
    draw.line((zero_x, top, zero_x, bottom), fill=PALETTE["text"], width=2)

    row_h = (bottom - top) / len(roi_defs)
    for i, roi_def in enumerate(roi_defs):
        roi = roi_def["roi_name"]
        value = float(sample_row[f"evidence_drop__{roi}"])
        yy = top + i * row_h + row_h * 0.5
        roi_label = _compact_roi_label(roi)
        if roi in rank_by_roi:
            roi_label = f"{rank_by_roi[roi]}. {roi_label}"
        text(draw, (label_x, yy), roi_label, 21, anchor="rm")
        x_value = axis_x0 + (value - x_min) / (x_max - x_min) * (axis_x1 - axis_x0)
        colour = PALETTE["red"] if value >= 0 else PALETTE["blue"]
        bx0, bx1 = sorted((zero_x, x_value))
        draw.rectangle((bx0, yy - 6, bx1, yy + 6), fill=colour)
        text(draw, (value_x, yy), f"{value:+.3f}", 21, fill=colour, anchor="rm")

    text(
        draw,
        (bar_x0, y + 321),
        "Positive value: masking lowered true-class confidence.",
        21,
        fill=PALETTE["gray"],
    )


def dual_aev_worked_example(output: Path) -> None:
    if load_pd_dbs is None:
        raise RuntimeError("load_pd_dbs is required to draw the AEV worked example")
    data = load_pd_dbs("data/raw/PD_DBS_Data.mat")
    masks = np.load("outputs/roi/coarse_roi_masks.npy").astype(bool)
    roi_defs = sorted(read_csv("outputs/roi/coarse_roi_definitions.csv"), key=lambda r: int(r["roi_index"]))
    aev_rows = {row["sample_id"]: row for row in read_csv("outputs/aev/aev_test.csv")}
    sample_ids = ["test_0475", "test_0861"]
    train_mean_image = data["x_train_images"][:, :, :, 0].mean(axis=0)

    w, h = 1900, 980
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (58, 30), "Occlusion-based AEV construction from matched Class 0 and Class 1 examples", 32, bold=True)
    text(
        draw,
        (58, 72),
        "Correct NumPy MLP predictions with true-class confidence between 0.85 and 0.95. Fixed ROI atlas and training-set-mean mask-out.",
        21,
        fill=PALETTE["gray"],
    )
    step_y = 103
    steps = [
        "full face",
        "mask one fixed ROI",
        "re-run MLP",
        "record true-class drop",
        "assemble 8 AEV components",
    ]
    x = 80
    step_width = 285
    step_gap = 65
    for i, step in enumerate(steps):
        current_step_width = 320 if i == len(steps) - 1 else step_width
        draw.rounded_rectangle((x, step_y, x + current_step_width, step_y + 42), radius=9, fill=PALETTE["light"], outline=PALETTE["grid"], width=1)
        step_font = font(21, bold=i in {0, 4})
        left, top, right, bottom = draw.textbbox((0, 0), step, font=step_font)
        text_x = x + current_step_width / 2 - (left + right) / 2
        text_y = step_y + 21 - (top + bottom) / 2
        draw.text((text_x, text_y), step, font=step_font, fill=PALETTE["text"])
        if i < len(steps) - 1:
            arrow_y = step_y + 21
            draw.line((x + step_width + 5, arrow_y, x + step_width + step_gap - 10, arrow_y), fill=PALETTE["gray"], width=3)
            draw.polygon(
                [
                    (x + step_width + step_gap - 10, arrow_y),
                    (x + step_width + step_gap - 23, arrow_y - 7),
                    (x + step_width + step_gap - 23, arrow_y + 7),
                ],
                fill=PALETTE["gray"],
            )
        x += step_width + step_gap

    for sample_id, panel_y in zip(sample_ids, [158, 515]):
        sample_idx = int(sample_id.split("_")[1])
        sample_image = data["x_test_images"][sample_idx, :, :, 0].astype(float)
        _draw_dual_aev_row(
            img,
            draw,
            panel_y,
            aev_rows[sample_id],
            sample_image,
            train_mean_image,
            masks,
            roi_defs,
        )

    text(
        draw,
        (60, h - 34),
        "Each panel uses the same fixed ROI atlas and the same NumPy MLP output table.",
        21,
        fill=PALETTE["gray"],
    )
    img.save(output)


def _x_pos(x0: int, x1: int, x_min: float, x_max: float, value: float) -> float:
    return x0 + (value - x_min) / (x_max - x_min) * (x1 - x0)


def _panel_axis(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    ticks: list[float],
    tick_fmt: str,
    tick_font_size: int = 16,
) -> None:
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=PALETTE["text"], width=2)
    for tick in ticks:
        x = _x_pos(x0, x1, x_min, x_max, tick)
        draw.line((x, y0, x, y1), fill=PALETTE["grid"], width=1)
        draw.line((x, y1, x, y1 + 7), fill=PALETTE["text"], width=2)
        text(draw, (x, y1 + 13), tick_fmt.format(tick), tick_font_size, fill=PALETTE["gray"], anchor="ma")


def roi_evidence_summary_figure(output: Path) -> None:
    """Integrated ROI figure replacing several repeated single-metric bar charts."""
    w, h = 1800, 1080
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (55, 34), "Integrated ROI-level evidence summary", 38, bold=True)
    text(
        draw,
        (55, 80),
        "Rows are ordered by fixed-atlas mask-out evidence. Columns show ROI size, evidence drop, class direction, and region-only AUROC.",
        22,
        fill=PALETTE["gray"],
    )

    aev = read_csv("outputs/aev/roi_occlusion_summary_overall.csv")
    aev = sorted(aev, key=lambda r: float(r["mean_evidence_drop"]), reverse=True)
    pixel = {r["roi_name"]: float(r["pixel_count"]) for r in read_csv("outputs/roi/coarse_roi_definitions.csv")}
    class_diff = {
        r["roi"]: float(r["diff_class1_minus_class0"])
        for r in read_csv("outputs/aev/roi_class_comparison_ranked.csv")
    }
    region_auroc = {
        r["roi_name"]: float(r["auroc"])
        for r in read_csv("outputs/aev/region_only_metrics.csv")
    }

    top, bottom = 185, 900
    label_x = 315
    boxes = {
        "pixels": (375, top, 565, bottom),
        "drop": (650, top, 910, bottom),
        "class": (1010, top, 1300, bottom),
        "auroc": (1405, top, 1665, bottom),
    }
    titles = [
        ("A. ROI size", "pixels", boxes["pixels"]),
        ("B. Mask-out AEV", "mean drop", boxes["drop"]),
        ("C. Class direction", "Class 1 - Class 0", boxes["class"]),
        ("D. Region-only", "AUROC", boxes["auroc"]),
    ]
    for title, subtitle, box in titles:
        text(draw, (box[0], top - 56), title, 24, bold=True)
        text(draw, (box[0], top - 28), subtitle, 17, fill=PALETTE["gray"])

    _panel_axis(draw, boxes["pixels"], 0, 210, [0, 105, 210], "{:.0f}")
    _panel_axis(draw, boxes["drop"], 0, 0.06, [0, 0.03, 0.06], "{:.2f}")
    _panel_axis(draw, boxes["class"], -0.08, 0.05, [-0.08, 0, 0.05], "{:.2f}")
    _panel_axis(draw, boxes["auroc"], 0.50, 0.75, [0.50, 0.60, 0.75], "{:.2f}")

    zero_x = _x_pos(boxes["class"][0], boxes["class"][2], -0.08, 0.05, 0)
    draw.line((zero_x, top, zero_x, bottom), fill=PALETTE["text"], width=2)
    chance_x = _x_pos(boxes["auroc"][0], boxes["auroc"][2], 0.50, 0.75, 0.50)
    draw.line((chance_x, top, chance_x, bottom), fill=PALETTE["grid"], width=2)

    row_h = (bottom - top) / len(aev)
    for i, row in enumerate(aev):
        roi = row["roi_name"]
        y = top + i * row_h + row_h / 2
        if i % 2 == 0:
            draw.rectangle((45, y - row_h / 2 + 5, 1715, y + row_h / 2 - 5), fill=(248, 250, 252))
        text(draw, (label_x, y), _short_roi_label(roi), 22, anchor="rm")

        # A. ROI size
        px0, py0, px1, _ = boxes["pixels"]
        p_val = pixel[roi]
        p_x = _x_pos(px0, px1, 0, 210, p_val)
        draw.rounded_rectangle((px0, y - 12, p_x, y + 12), radius=5, fill=(150, 190, 186))
        text(draw, (p_x + 8, y), f"{p_val:.0f}", 16, fill=PALETTE["gray"], anchor="lm")

        # B. Mask-out evidence drop
        dx0, _, dx1, _ = boxes["drop"]
        d_val = float(row["mean_evidence_drop"])
        d_x = _x_pos(dx0, dx1, 0, 0.06, d_val)
        draw.rounded_rectangle((dx0, y - 12, d_x, y + 12), radius=5, fill=PALETTE["teal"])
        text(draw, (d_x + 8, y), f"{d_val:.3f}", 16, fill=PALETTE["text"], anchor="lm")

        # C. Class-specific difference
        cx0, _, cx1, _ = boxes["class"]
        c_val = class_diff[roi]
        c_x = _x_pos(cx0, cx1, -0.08, 0.05, c_val)
        c_start, c_end = sorted([zero_x, c_x])
        c_color = PALETTE["blue"] if c_val >= 0 else PALETTE["orange"]
        draw.rounded_rectangle((c_start, y - 11, c_end, y + 11), radius=5, fill=c_color)
        c_anchor = "lm" if c_val >= 0 else "rm"
        c_label_x = c_end + 8 if c_val >= 0 else c_start - 8
        text(draw, (c_label_x, y), f"{c_val:+.3f}", 16, fill=PALETTE["text"], anchor=c_anchor)

        # D. Region-only AUROC
        ax0, _, ax1, _ = boxes["auroc"]
        a_val = region_auroc[roi]
        a_x = _x_pos(ax0, ax1, 0.50, 0.75, a_val)
        draw.line((ax0, y, a_x, y), fill=(190, 210, 224), width=6)
        draw.ellipse((a_x - 8, y - 8, a_x + 8, y + 8), fill=PALETTE["blue"])
        text(draw, (a_x + 11, y), f"{a_val:.3f}", 16, fill=PALETTE["text"], anchor="lm")

    text(draw, (label_x, top - 56), "ROI", 22, bold=True, anchor="rm")
    text(draw, (1010, bottom + 72), "orange = higher Class 0 evidence, blue = higher Class 1 evidence", 17, fill=PALETTE["gray"])
    text(draw, (55, h - 58), "Class 0 = pre-DBS. Class 1 = post-DBS. Region-only AUROC uses the same trained MLP with one retained ROI.", 19, fill=PALETTE["gray"])
    img.save(output)


def robustness_summary_figure(output: Path) -> None:
    """Combine seed stability and perturbation robustness in a readable stacked figure."""
    w, h = 1400, 1160
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (55, 34), "Robustness summary", 42, bold=True)
    text(
        draw,
        (55, 86),
        "Repeated runs report mean, SD, and range. Image perturbation sensitivity is summarised by AUROC change.",
        24,
        fill=PALETTE["gray"],
    )

    seed_rows = {
        r["metric"]: r
        for r in read_csv("outputs/robustness/multiseed_summary.csv")
    }
    metrics = [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("AUROC", "auroc"),
        ("Trap. AUPRC", "auprc"),
        ("Brier", "brier_score"),
        ("ECE", "ece"),
    ]
    left, top, table_right = 70, 190, 1330
    text(draw, (left, top - 60), "A. Five-seed stability", 30, bold=True)
    text(draw, (left, top - 27), "mean \u00b1 SD", 22, fill=PALETTE["gray"])
    col_x = [left, 560, 800, 1035]
    header_y = top + 5
    draw.rounded_rectangle((left - 8, header_y - 4, table_right, header_y + 51), radius=8, fill=PALETTE["navy"])
    for x, label in zip(col_x, ["Metric", "Mean", "SD", "Range"]):
        text(draw, (x + 6, header_y + 24), label, 25, bold=True, fill="white", anchor="lm")
    row_h = 62
    for i, (label, key) in enumerate(metrics):
        y = header_y + 55 + i * row_h
        fill = PALETTE["light"] if i % 2 == 0 else "white"
        draw.rectangle((left - 8, y, table_right, y + row_h), fill=fill)
        r = seed_rows[key]
        mean, sd = float(r["mean"]), float(r["sd"])
        min_v, max_v = float(r["min"]), float(r["max"])
        text(draw, (col_x[0] + 6, y + row_h / 2), label, 25, anchor="lm")
        text(draw, (col_x[1] + 6, y + row_h / 2), f"{mean:.4f}", 25, anchor="lm")
        text(draw, (col_x[2] + 6, y + row_h / 2), f"{sd:.4f}", 25, anchor="lm")
        text(draw, (col_x[3] + 6, y + row_h / 2), f"{min_v:.3f}\u2013{max_v:.3f}", 25, anchor="lm")
    draw.rectangle((left - 8, header_y - 4, table_right, header_y + 55 + row_h * len(metrics)), outline=PALETTE["grid"], width=2)

    p_rows = [
        r for r in read_csv("outputs/robustness/perturbation_metrics.csv")
        if r["perturbation"] != "original"
    ]
    p_labels = {
        "mild_blur_3x3": "mild blur",
        "center_crop_28_resize": "centre crop",
        "offset_crop_28_resize": "offset crop",
    }
    px0, py0, px1, py1 = 360, 790, 1310, 1050
    text(draw, (left, 690), "B. Perturbation effect", 30, bold=True)
    text(draw, (left, 728), "Delta AUROC versus original test images", 22, fill=PALETTE["gray"])
    _panel_axis(
        draw,
        (px0, py0, px1, py1),
        -0.15,
        0.01,
        [-0.15, -0.10, -0.05, 0.00],
        "{:.2f}",
        tick_font_size=21,
    )
    zero = _x_pos(px0, px1, -0.15, 0.01, 0)
    draw.line((zero, py0, zero, py1), fill=PALETTE["text"], width=2)
    row_h = (py1 - py0) / len(p_rows)
    for i, row in enumerate(p_rows):
        y = py0 + i * row_h + row_h / 2
        label = p_labels.get(row["perturbation"], row["perturbation"].replace("_", " "))
        text(draw, (px0 - 18, y), label, 25, anchor="rm")
        val = float(row["delta_auroc_vs_original"])
        x_val = _x_pos(px0, px1, -0.15, 0.01, val)
        x_start, x_end = sorted([zero, x_val])
        draw.rounded_rectangle((x_start, y - 16, x_end, y + 16), radius=7, fill=PALETTE["orange"])
        text(draw, (x_start - 10, y), f"{val:.3f}", 23, anchor="rm")

    text(draw, (360, 1110), "Original AUROC = 0.9824. Centre and offset crops produce the larger changes.", 22, fill=PALETTE["gray"])
    img.save(output)


def _draw_grouped_hbars(
    draw: ImageDraw.ImageDraw,
    rows: list[tuple[str, float, float]],
    box: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    label_a: str,
    label_b: str,
    title: str,
    x_label: str,
) -> None:
    left, top, right, bottom = box
    text(draw, (left, top - 60), title, 24, bold=True)
    text(draw, (right - 210, top - 58), label_a, 18, fill=PALETTE["blue"])
    draw.rectangle((right - 238, top - 49, right - 218, top - 39), fill=PALETTE["blue"])
    text(draw, (right - 95, top - 58), label_b, 18, fill=PALETTE["orange"])
    draw.rectangle((right - 123, top - 49, right - 103, top - 39), fill=PALETTE["orange"])
    draw.rectangle((left, top, right, bottom), outline=PALETTE["grid"], width=2)
    for i in range(5):
        tick = x_min + i * (x_max - x_min) / 4
        x = left + (tick - x_min) / (x_max - x_min) * (right - left)
        draw.line((x, top, x, bottom), fill=PALETTE["grid"], width=1)
        text(draw, (x, bottom + 10), f"{tick:.2f}", 16, fill=PALETTE["gray"], anchor="ma")
    row_h = (bottom - top) / len(rows)
    for i, (label, a_val, b_val) in enumerate(rows):
        y = top + i * row_h + row_h / 2
        text(draw, (left - 12, y), _short_roi_label(label), 16, anchor="rm")
        for value, color, dy in [(a_val, PALETTE["blue"], -8), (b_val, PALETTE["orange"], 8)]:
            x_val = left + (value - x_min) / (x_max - x_min) * (right - left)
            draw.rounded_rectangle((left, y + dy - 5, x_val, y + dy + 5), radius=4, fill=color)
        text(draw, (right + 10, y - 8), f"{a_val:.3f}", 14, fill=PALETTE["gray"], anchor="lm")
        text(draw, (right + 10, y + 10), f"{b_val:.3f}", 14, fill=PALETTE["gray"], anchor="lm")
    text(draw, ((left + right) / 2, bottom + 42), x_label, 17, fill=PALETTE["gray"], anchor="ma")


def yunet_dynamic_roi_figure(output: Path) -> None:
    w, h = 1700, 1100
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    roi_defs = sorted(
        read_csv("outputs/roi/coarse_roi_definitions.csv"),
        key=lambda row: int(row["roi_index"]),
    )
    roi_colours = [
        PALETTE["blue"],
        PALETTE["orange"],
        PALETTE["green"],
        PALETTE["teal"],
        PALETTE["red"],
        (120, 90, 170),
        (180, 120, 60),
        (80, 120, 180),
    ]
    text(draw, (55, 34), "YuNet-derived ROI sensitivity analysis", 36, bold=True)
    text(
        draw,
        (55, 80),
        "Automatic five-landmark ROI boxes are compared with the fixed atlas on the 32\u00d732 grid.",
        22,
        fill=PALETTE["gray"],
    )

    text(draw, (60, 135), "A. Detection-derived ROI boxes", 24, bold=True)
    source = Path("outputs/external/pd_dbs_yunet_region_only/qc_yunet_roi_examples.jpg")
    if source.exists():
        sheet = Image.open(source).convert("RGB")
        tile_positions = [(0, 0), (1, 1), (3, 1), (4, 2)]
        for idx, (col, row) in enumerate(tile_positions):
            crop = sheet.crop((col * 148, 28 + row * 164, col * 148 + 128, 28 + row * 164 + 128))
            crop = crop.resize((225, 225), Image.Resampling.NEAREST)
            x = 70 + (idx % 2) * 260
            y = 175 + (idx // 2) * 275
            draw.rounded_rectangle((x - 8, y - 8, x + 233, y + 255), radius=10, fill=PALETTE["light"], outline=PALETTE["grid"], width=2)
            img.paste(crop, (x, y))
            text(draw, (x + 112, y + 234), f"example {idx + 1}", 16, fill=PALETTE["gray"], anchor="ma")
    elif load_pd_dbs is not None and Path("outputs/external/pd_dbs_yunet_region_only/yunet_roi_boxes.csv").exists():
        data = load_pd_dbs("data/raw/PD_DBS_Data.mat")
        boxes = read_csv("outputs/external/pd_dbs_yunet_region_only/yunet_roi_boxes.csv")
        sample_rows: dict[str, list[dict[str, str]]] = {}
        for row in boxes:
            if row["split"] != "test":
                continue
            sample_rows.setdefault(row["sample_id"], []).append(row)
        selected = sorted(
            sample_rows,
            key=lambda sid: float(sample_rows[sid][0]["score"]),
            reverse=True,
        )[:4]
        for idx, sample_id in enumerate(selected):
            sample_idx = int(sample_id.split("_")[1])
            arr = data["x_test_images"][sample_idx, :, :, 0].astype(float)
            arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
            tile = Image.fromarray((arr * 255).astype("uint8"), mode="L").resize((225, 225), Image.Resampling.NEAREST).convert("RGB")
            tile_draw = ImageDraw.Draw(tile)
            scale = 225 / 256
            for row in sample_rows[sample_id]:
                colour = roi_colours[(int(row["roi_index"]) - 1) % len(roi_colours)]
                x1 = int(round(float(row["x1"]) * scale))
                y1 = int(round(float(row["y1"]) * scale))
                x2 = int(round(float(row["x2"]) * scale))
                y2 = int(round(float(row["y2"]) * scale))
                tile_draw.rectangle((x1, y1, x2, y2), outline=colour, width=2)
            x = 70 + (idx % 2) * 260
            y = 175 + (idx // 2) * 275
            draw.rounded_rectangle((x - 8, y - 8, x + 233, y + 255), radius=10, fill=PALETTE["light"], outline=PALETTE["grid"], width=2)
            img.paste(tile, (x, y))
            score = float(sample_rows[sample_id][0]["score"])
            text(draw, (x + 112, y + 234), f"{sample_id}, score {score:.3f}", 15, fill=PALETTE["gray"], anchor="ma")
    else:
        text(draw, (90, 210), "YuNet ROI examples unavailable", 20, fill=PALETTE["red"])

    draw.rounded_rectangle((65, 715, 600, 845), radius=10, fill=PALETTE["light"], outline=PALETTE["grid"], width=2)
    text(draw, (85, 728), "ROI key: colour and number", 20, bold=True)
    for i, roi_def in enumerate(roi_defs):
        key_col = i // 4
        key_row = i % 4
        key_x = 85 + key_col * 260
        key_y = 758 + key_row * 25
        colour = roi_colours[i % len(roi_colours)]
        draw.rounded_rectangle((key_x, key_y - 11, key_x + 30, key_y + 11), radius=5, fill=colour)
        text(draw, (key_x + 15, key_y), str(i + 1), 19, bold=True, fill="white", anchor="mm")
        text(draw, (key_x + 40, key_y), _short_roi_label(roi_def["roi_name"]), 20, anchor="lm")

    aev_rows_raw = read_csv("outputs/external/pd_dbs_yunet_aev/yunet_vs_fixed_aev_occlusion_comparison.csv")
    aev_rows = [
        (r["roi_name"], _as_float(r, "mean_evidence_drop_fixed"), _as_float(r, "mean_evidence_drop_yunet"))
        for r in aev_rows_raw
    ]
    aev_rows = sorted(aev_rows, key=lambda r: r[2], reverse=True)
    _draw_grouped_hbars(
        draw,
        aev_rows,
        (820, 190, 1530, 500),
        0.0,
        0.08,
        "fixed atlas",
        "YuNet-derived",
        "B. Mean true-class confidence drop",
        "Mean true-class confidence drop",
    )

    region_rows_raw = read_csv("outputs/external/pd_dbs_yunet_region_only/yunet_vs_fixed_roi_comparison.csv")
    region_rows = [
        (r["roi_name"], _as_float(r, "auroc_fixed"), _as_float(r, "auroc_yunet"))
        for r in region_rows_raw
    ]
    region_rows = sorted(region_rows, key=lambda r: r[2], reverse=True)
    _draw_grouped_hbars(
        draw,
        region_rows,
        (820, 700, 1530, 980),
        0.52,
        0.73,
        "fixed atlas",
        "YuNet-derived",
        "C. Region-only AUROC",
        "AUROC from a single retained ROI",
    )

    note = (
        "Detection: 2,343/2,343 faces. Median YuNet score: 0.923. "
        "The comparison evaluates ROI-geometry sensitivity."
    )
    draw.rounded_rectangle((65, 860, 600, 1018), radius=12, fill=(255, 249, 235), outline=PALETTE["orange"], width=2)
    text(draw, (88, 885), "Interpretation", 23, bold=True)
    for j, line in enumerate([
        "Right cheek/zygomatic AUROC changes little across ROI definitions.",
        "Perioral/mouth and nasal estimates increase with YuNet geometry.",
        "Ranking changes show sensitivity to ROI geometry.",
    ]):
        text(draw, (90, 925 + j * 27), f"- {line}", 18)
    text(draw, (60, h - 42), note, 18, fill=PALETTE["gray"])
    img.save(output)


def table_panel(headers: list[str], rows: list[list[str]], title: str, subtitle: str, output: Path) -> None:
    row_h = 64
    w, h = 1450, max(410, 150 + row_h * (len(rows) + 1) + 55)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (55, 35), title, 34, bold=True)
    text(draw, (55, 80), subtitle, 22, fill=PALETTE["gray"])
    left, top, right = 75, 150, 1375
    weights = [1.45] + [1.0] * (len(headers) - 1)
    unit = (right - left) / sum(weights)
    col_edges = [left]
    for weight in weights:
        col_edges.append(col_edges[-1] + weight * unit)
    draw.rectangle((left, top, right, top + row_h), fill=PALETTE["navy"])
    for i, header in enumerate(headers):
        x0, x1 = col_edges[i], col_edges[i + 1]
        text(draw, ((x0 + x1) / 2, top + row_h / 2), header, 20, fill="white", bold=True, anchor="mm")
    for r, row in enumerate(rows):
        y = top + row_h * (r + 1)
        fill = "white" if r % 2 == 0 else PALETTE["light"]
        draw.rectangle((left, y, right, y + row_h), fill=fill)
        for c, cell in enumerate(row):
            x0, x1 = col_edges[c], col_edges[c + 1]
            cx, cy = (x0 + x1) / 2, y + row_h / 2
            if "\n" in cell:
                lines = cell.splitlines()
                line_gap = 21
                start_y = cy - line_gap * (len(lines) - 1) / 2
                for j, line in enumerate(lines):
                    text(draw, (cx, start_y + j * line_gap), line, 17, fill=PALETTE["text"], anchor="mm")
            else:
                text(draw, (cx, cy), cell, 19, fill=PALETTE["text"], anchor="mm")
    draw.rectangle((left, top, right, top + row_h * (len(rows) + 1)), outline=PALETTE["grid"], width=2)
    for i in range(1, len(headers)):
        x = col_edges[i]
        draw.line((x, top, x, top + row_h * (len(rows) + 1)), fill=PALETTE["grid"], width=1)
    img.save(output)


def baseline_panel_from_outputs(output: Path) -> None:
    metrics = json.loads(Path("outputs/baseline/metrics.json").read_text(encoding="utf-8"))["test"]
    roc_rows = read_csv("outputs/baseline/roc_curve.csv")
    pr_rows = read_csv("outputs/baseline/pr_curve.csv")
    train_rows = read_csv("outputs/baseline/training_curve.csv")

    w, h = 1500, 1100
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (50, 35), "Baseline numeric classification", 34, bold=True)
    text(draw, (50, 78), "The NumPy MLP baseline establishes the Class 0 vs Class 1 evaluation loop.", 22, fill=PALETTE["gray"])
    panels = {
        "cm": (60, 140, 700, 520),
        "roc": (790, 140, 1440, 520),
        "pr": (60, 625, 700, 1005),
        "train": (790, 625, 1440, 1005),
    }
    for label, box in panels.items():
        draw.rectangle(box, outline=PALETTE["grid"], width=2)

    def draw_curve(box, points, title, x_label, y_label, color=PALETTE["blue"], diag=False):
        x0, y0, x1, y1 = box
        text(draw, (x0 + 20, y0 + 18), title, 23, bold=True)
        left, top, right, bottom = x0 + 80, y0 + 70, x1 - 35, y1 - 65
        draw.rectangle((left, top, right, bottom), outline=PALETTE["text"], width=2)
        for i in range(1, 5):
            xx = left + i * (right - left) / 5
            yy = top + i * (bottom - top) / 5
            draw.line((xx, top, xx, bottom), fill=PALETTE["grid"], width=1)
            draw.line((left, yy, right, yy), fill=PALETTE["grid"], width=1)
        if diag:
            draw.line((left, bottom, right, top), fill=PALETTE["gray"], width=2)
        pts = [(left + x * (right - left), bottom - y * (bottom - top)) for x, y in points]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4)
        text(draw, ((left + right) / 2, y1 - 28), x_label, 18, fill=PALETTE["gray"], anchor="ma")
        vertical_text(img, (x0 + 35, (top + bottom) / 2), y_label, 18, fill=PALETTE["gray"])
        for v in [0, 0.5, 1.0]:
            text(draw, (left + v * (right - left), bottom + 12), f"{v:.1f}", 15, fill=PALETTE["gray"], anchor="ma")
            text(draw, (left - 10, bottom - v * (bottom - top)), f"{v:.1f}", 15, fill=PALETTE["gray"], anchor="rm")

    # Confusion matrix.
    x0, y0, x1, y1 = panels["cm"]
    text(draw, (x0 + 20, y0 + 18), "A  Confusion matrix", 23, bold=True)
    cm = metrics["confusion_matrix"]
    grid_left, grid_top, cell = x0 + 160, y0 + 105, 135
    headers = [["", "Pred 0", "Pred 1"], ["True 0", str(cm["tn"]), str(cm["fp"])], ["True 1", str(cm["fn"]), str(cm["tp"])]]
    for r, row in enumerate(headers):
        for c, val in enumerate(row):
            gx = grid_left + c * cell
            gy = grid_top + r * 70
            fill = PALETTE["navy"] if r == 0 or c == 0 else ("white" if r == c else PALETTE["light"])
            draw.rectangle((gx, gy, gx + cell, gy + 70), fill=fill, outline=PALETTE["grid"], width=2)
            color = "white" if r == 0 or c == 0 else PALETTE["text"]
            text(draw, (gx + cell / 2, gy + 35), val, 21, fill=color, bold=(r == 0 or c == 0), anchor="mm")
    text(draw, (x0 + 30, y1 - 45), f"Accuracy {metrics['accuracy']:.4f}   AUROC {metrics['auroc']:.4f}", 20, fill=PALETTE["gray"])

    roc_points = [(float(r["fpr"]), float(r["tpr"])) for r in roc_rows]
    pr_points = [(float(r["recall"]), float(r["precision"])) for r in pr_rows]
    draw_curve(panels["roc"], roc_points, "B  ROC curve", "False positive rate", "True positive rate", PALETTE["blue"], diag=True)
    draw_curve(panels["pr"], pr_points, "C  Precision-recall curve", "Recall", "Precision", PALETTE["teal"], diag=False)

    train_points = [(float(r["epoch"]), float(r["train_loss"])) for r in train_rows]
    val_points = [(float(r["epoch"]), float(r["val_loss"])) for r in train_rows]
    x0, y0, x1, y1 = panels["train"]
    text(draw, (x0 + 20, y0 + 18), "D  Training curve", 23, bold=True)
    left, top, right, bottom = x0 + 80, y0 + 70, x1 - 35, y1 - 65
    max_epoch = max(x for x, _ in train_points)
    max_loss = max(max(y for _, y in train_points), max(y for _, y in val_points)) * 1.05
    draw.rectangle((left, top, right, bottom), outline=PALETTE["text"], width=2)
    for i in range(1, 5):
        xx = left + i * (right - left) / 5
        yy = top + i * (bottom - top) / 5
        draw.line((xx, top, xx, bottom), fill=PALETTE["grid"], width=1)
        draw.line((left, yy, right, yy), fill=PALETTE["grid"], width=1)
    def scale_train(points):
        return [(left + x / max_epoch * (right - left), bottom - y / max_loss * (bottom - top)) for x, y in points]
    draw.line(scale_train(train_points), fill=PALETTE["blue"], width=4)
    draw.line(scale_train(val_points), fill=PALETTE["orange"], width=4)
    text(draw, (left + 20, top + 20), "train loss", 17, fill=PALETTE["blue"])
    text(draw, (left + 20, top + 45), "validation loss", 17, fill=PALETTE["orange"])
    text(draw, ((left + right) / 2, y1 - 28), "Epoch", 18, fill=PALETTE["gray"], anchor="ma")
    vertical_text(img, (x0 + 35, (top + bottom) / 2), "Loss", 18, fill=PALETTE["gray"])
    img.save(output)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    baseline_panel_from_outputs(OUT / "fig_baseline_panel.png")

    rows = read_csv("outputs/aev/roi_occlusion_summary_overall.csv")
    hbar(
        [(r["roi_name"], float(r["mean_evidence_drop"])) for r in rows],
        "Occlusion-based AEV ranking",
        "Mean true-class confidence drop after masking each mutually exclusive coarse ROI.",
        "Mean true-class confidence drop",
        OUT / "fig_aev_overall_ranking.png",
        x_min=0,
        x_max=0.06,
    )

    rows = read_csv("outputs/aev/roi_class_comparison_ranked.csv")
    hbar(
        [(r["roi"], float(r["diff_class1_minus_class0"])) for r in rows],
        "Class-specific AEV differences",
        "Positive values indicate higher evidence in Class 1. Negative values indicate higher evidence in Class 0.",
        "Mean evidence difference: Class 1 minus Class 0",
        OUT / "fig_aev_class_difference.png",
        x_min=-0.08,
        x_max=0.05,
        diverging=True,
    )

    rows = read_csv("outputs/aev/region_only_metrics.csv")
    rows = sorted(rows, key=lambda r: float(r["auroc"]), reverse=True)
    hbar(
        [(r["roi_name"], float(r["auroc"])) for r in rows],
        "Region-only validation",
        "The same trained MLP is evaluated on single-ROI inputs, which remain below the full-face AUROC of 0.9824.",
        "Region-only AUROC",
        OUT / "fig_region_only_auroc.png",
        x_min=0.45,
        x_max=0.75,
    )

    rows = read_csv("outputs/calibration/reliability_bins.csv")
    points = [(float(r["mean_p_class1"]), float(r["fraction_class1"])) for r in rows if r["count"] != "0"]
    line_plot(
        points,
        "Reliability diagram",
        "Calibration is assessed for the Class 0 pre-DBS versus Class 1 post-DBS task.",
        "Mean predicted probability for Class 1",
        "Observed Class 1 fraction",
        OUT / "fig_calibration_reliability.png",
    )

    rows = read_csv("outputs/robustness/perturbation_metrics.csv")
    hbar(
        [(r["perturbation"], float(r["delta_auroc_vs_original"])) for r in rows if r["perturbation"] != "original"],
        "Perturbation robustness",
        "Mild blur has little effect. Crop/resize perturbations cause larger AUROC drops.",
        "Delta AUROC versus original test images",
        OUT / "fig_perturbation_robustness.png",
        x_min=-0.15,
        x_max=0.01,
        diverging=True,
    )

    pipeline_schematic(OUT / "fig_method_pipeline.png")
    dual_aev_worked_example(OUT / "fig_aev_dual_worked_example.png")
    yunet_dynamic_roi_figure(OUT / "fig_yunet_dynamic_roi_sensitivity.png")
    roi_evidence_summary_figure(OUT / "fig_roi_evidence_summary.png")
    robustness_summary_figure(OUT / "fig_robustness_summary.png")

    roi_rows = read_csv("outputs/roi/coarse_roi_definitions.csv")
    hbar(
        [(r["roi_name"], float(r["pixel_count"])) for r in roi_rows],
        "Coarse ROI pixel coverage",
        "The eight 32\u00d732 ROI masks are mutually exclusive and cover 840 pixels.",
        "Pixels per ROI",
        OUT / "fig_roi_pixel_counts.png",
        x_min=0,
        x_max=210,
    )

    seed_rows = read_csv("outputs/robustness/multiseed_summary.csv")
    seed_metrics = [r for r in seed_rows if r["metric"] in {"accuracy", "balanced_accuracy", "auroc", "auprc", "brier_score", "ece"}]
    hbar(
        [(r["metric"], float(r["mean"])) for r in seed_metrics],
        "Multi-seed robustness summary",
        "Repeated random validation splits and model initialisations show stable technical performance.",
        "Mean metric value across five seeds",
        OUT / "fig_multiseed_summary.png",
        x_min=0,
        x_max=1,
    )

    identity_rows = []
    baseline_metrics = json.loads(Path("outputs/baseline/metrics.json").read_text(encoding="utf-8"))["test"]
    identity_rows.append(
        [
            "original image-level\ntest split",
            str(baseline_metrics["n"]),
            f"{float(baseline_metrics['accuracy']):.4f}",
            f"{float(baseline_metrics['balanced_accuracy']):.4f}",
            f"{float(baseline_metrics['auroc']):.4f}",
            f"{float(baseline_metrics['auprc']):.4f}",
        ]
    )
    threshold_rows = read_csv("outputs/data_qc/similarity_threshold_sensitivity.csv")
    near_duplicate_row = next(r for r in threshold_rows if str(float(r["max_cosine_exclusion_threshold"])) == "0.999")
    identity_rows.append(
        [
            "max-cosine < 0.999\nsubset",
            near_duplicate_row["remaining_test_samples"],
            f"{float(near_duplicate_row['accuracy']):.4f}",
            f"{float(near_duplicate_row['balanced_accuracy']):.4f}",
            f"{float(near_duplicate_row['auroc']):.4f}",
            f"{float(near_duplicate_row['auprc']):.4f}",
        ]
    )
    for threshold in ["0.99", "0.98"]:
        row = next(r for r in threshold_rows if str(float(r["max_cosine_exclusion_threshold"])) == threshold)
        identity_rows.append(
            [
                f"max-cosine < {threshold}\nsubset",
                row["remaining_test_samples"],
                f"{float(row['accuracy']):.4f}",
                f"{float(row['balanced_accuracy']):.4f}",
                f"{float(row['auroc']):.4f}",
                f"{float(row['auprc']):.4f}",
            ]
        )
    global_metrics = json.loads(Path("outputs/data_qc/global_statistics_baseline_metrics.json").read_text(encoding="utf-8"))
    identity_rows.append(
        [
            "global image-statistics\nlogistic baseline",
            str(global_metrics["n"]),
            f"{float(global_metrics['accuracy']):.4f}",
            f"{float(global_metrics['balanced_accuracy']):.4f}",
            f"{float(global_metrics['auroc']):.4f}",
            f"{float(global_metrics['auprc']):.4f}",
        ]
    )
    lowlevel_metrics = json.loads(Path("outputs/lowlevel_roi_confound/metrics.json").read_text(encoding="utf-8"))
    identity_rows.append(
        [
            "ROI low-level\nbaseline",
            str(lowlevel_metrics["n"]),
            f"{float(lowlevel_metrics['accuracy']):.4f}",
            f"{float(lowlevel_metrics['balanced_accuracy']):.4f}",
            f"{float(lowlevel_metrics['auroc']):.4f}",
            f"{float(lowlevel_metrics['auprc']):.4f}",
        ]
    )
    table_panel(
        ["Audit", "N", "Accuracy", "Bal. acc.", "AUROC", "Trap. AUPRC"],
        identity_rows,
        "Similarity and low-level summary audit",
        "Similarity-threshold subsets, global statistics, and ROI low-level summaries describe image-level sensitivity.",
        OUT / "fig_similarity_lowlevel_audit.png",
    )

    print(f"Wrote figures to {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

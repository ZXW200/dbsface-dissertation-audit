from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


# Publication contract: editable SVG text, TrueType PDF text, 600 dpi raster preview.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.5

INK = "#272B33"
MUTED = "#66707C"
LINE = "#CBD5DF"
PALE_BLUE = "#F1F5F9"
PALE_TEAL = "#EAF5F4"
BLUE = "#3775BA"
BLUE_DARK = "#0F4D92"
TEAL = "#42949E"
ORANGE = "#D9822B"
PALE_ORANGE = "#FFF6E8"
RED = "#C95450"
WHITE = "#FFFFFF"

ROI_COLORS = [
    "#D94A4A", "#4A7ED9", "#45A66F", "#D79B2E",
    "#9A58C7", "#45B8B0", "#D95C91", "#666666",
]

OUT_DIR = Path(__file__).resolve().parents[1] / "latex_project" / "figures" / "concept_candidates"


def setup_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def rounded(ax, x, y, w, h, fc=WHITE, ec=LINE, lw=0.8, radius=0.025, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.1, ms=9, z=5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z,
    ))


def draw_face(ax, cx, cy, scale=1.0, roi=False, masked=None):
    """Draw a deliberately schematic face; no patient image or quantitative data."""
    w, h = 0.105 * scale, 0.145 * scale
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor="#D5D9DE", edgecolor="#929AA3", lw=0.7, zorder=3))
    ax.add_patch(Ellipse((cx - 0.021 * scale, cy + 0.017 * scale), 0.012 * scale, 0.006 * scale,
                         facecolor=INK, edgecolor="none", zorder=4))
    ax.add_patch(Ellipse((cx + 0.021 * scale, cy + 0.017 * scale), 0.012 * scale, 0.006 * scale,
                         facecolor=INK, edgecolor="none", zorder=4))
    ax.plot([cx, cx - 0.004 * scale, cx + 0.006 * scale],
            [cy + 0.010 * scale, cy - 0.018 * scale, cy - 0.020 * scale],
            color="#737B84", lw=0.6, zorder=4)
    ax.plot([cx - 0.020 * scale, cx, cx + 0.020 * scale],
            [cy - 0.040 * scale, cy - 0.047 * scale, cy - 0.040 * scale],
            color="#737B84", lw=0.7, zorder=4)

    if not roi and masked is None:
        return

    # Coarse region rectangles are schematic approximations of the eight named ROIs.
    regions = [
        (-0.042, 0.040, 0.084, 0.033),
        (-0.043, 0.005, 0.038, 0.030),
        (0.005, 0.005, 0.038, 0.030),
        (-0.008, -0.018, 0.016, 0.052),
        (-0.043, -0.030, 0.038, 0.037),
        (0.005, -0.030, 0.038, 0.037),
        (-0.030, -0.057, 0.060, 0.026),
        (-0.027, -0.075, 0.054, 0.020),
    ]
    for i, (dx, dy, rw, rh) in enumerate(regions):
        if masked is not None and i == masked:
            fc, alpha, ec = "#6F7780", 0.95, "#42484F"
        elif roi:
            fc, alpha, ec = ROI_COLORS[i], 0.28, ROI_COLORS[i]
        else:
            continue
        ax.add_patch(Rectangle(
            (cx + dx * scale, cy + dy * scale), rw * scale, rh * scale,
            facecolor=fc, edgecolor=ec, alpha=alpha, lw=0.7, zorder=5,
        ))


def draw_heatmap(ax, x, y, w, h):
    rng = np.random.default_rng(17)
    yy, xx = np.mgrid[-1:1:9j, -1:1:9j]
    values = (
        0.85 * np.exp(-((xx + 0.35) ** 2 + (yy - 0.15) ** 2) / 0.20)
        - 0.65 * np.exp(-((xx - 0.25) ** 2 + (yy + 0.25) ** 2) / 0.15)
        + 0.08 * rng.normal(size=xx.shape)
    )
    ax.imshow(values, extent=(x, x + w, y, y + h), origin="lower", cmap="coolwarm",
              interpolation="nearest", vmin=-1, vmax=1, zorder=3, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#8D96A0", lw=0.7, zorder=4))


def save_all(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        kwargs = {"facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(OUT_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def figure_pixel_vs_aev():
    fig, ax = plt.subplots(figsize=(160 / 25.4, 82 / 25.4))
    setup_ax(ax)
    ax.text(0.025, 0.945, "From pixel explanations to auditable regional evidence",
            fontsize=12.5, fontweight="bold", color=INK, va="top")
    ax.text(0.025, 0.885, "Conceptual comparison at 32×32 facial-image resolution",
            fontsize=7.8, color=MUTED, va="top")

    rounded(ax, 0.025, 0.13, 0.445, 0.68, fc=PALE_BLUE)
    rounded(ax, 0.530, 0.13, 0.445, 0.68, fc=PALE_TEAL, ec="#BBD7D4")
    ax.text(0.050, 0.765, "Pixel-level explanation", fontsize=9.2, fontweight="bold", color=INK)
    ax.text(0.555, 0.765, "Regional AEV audit", fontsize=9.2, fontweight="bold", color=INK)

    draw_face(ax, 0.105, 0.555, 1.15)
    arrow(ax, 0.172, 0.555, 0.215, 0.555)
    draw_heatmap(ax, 0.235, 0.485, 0.105, 0.145)
    arrow(ax, 0.350, 0.555, 0.392, 0.555)
    rounded(ax, 0.397, 0.485, 0.050, 0.145, fc=WHITE)
    for i, a in enumerate([0.45, 0.78, 0.30, 0.65, 0.52, 0.38]):
        ax.add_patch(Rectangle((0.406 + (i % 2) * 0.018, 0.505 + (i // 2) * 0.040),
                               0.012, 0.028, color=BLUE if a > 0.5 else "#B6C2CE", alpha=a + 0.2))
    ax.text(0.105, 0.435, "Image", ha="center", color=MUTED)
    ax.text(0.288, 0.435, "Pixel heatmap", ha="center", color=MUTED)
    ax.text(0.422, 0.435, "Image-specific\nimportance", ha="center", va="top", color=MUTED)
    ax.text(0.247, 0.265, "Fine spatial output", ha="center", fontsize=8.2, fontweight="bold", color=INK)
    ax.text(0.247, 0.205, "Useful for localisation, but harder to\ncompare consistently across samples",
            ha="center", va="top", color=MUTED, linespacing=1.3)

    draw_face(ax, 0.605, 0.555, 1.15, roi=True)
    arrow(ax, 0.672, 0.555, 0.714, 0.555)
    # Eight-value AEV as a compact diverging vector.
    x0, y0 = 0.727, 0.495
    vals = [0.70, -0.22, 0.48, -0.14, 0.32, 0.58, 0.20, 0.42]
    for i, v in enumerate(vals):
        yy = y0 + i * 0.018
        ax.plot([x0 + 0.045, x0 + 0.045], [yy - 0.005, yy + 0.010], color="#8D96A0", lw=0.5)
        if v >= 0:
            ax.add_patch(Rectangle((x0 + 0.045, yy - 0.004), 0.040 * v, 0.008, color=RED, lw=0))
        else:
            ax.add_patch(Rectangle((x0 + 0.045 + 0.040 * v, yy - 0.004), -0.040 * v, 0.008, color=BLUE, lw=0))
    arrow(ax, 0.825, 0.555, 0.857, 0.555)
    rounded(ax, 0.865, 0.475, 0.085, 0.165, fc=WHITE)
    for i, txt in enumerate(["Class", "Function", "Robustness"]):
        ax.add_patch(Circle((0.882, 0.605 - i * 0.050), 0.007, facecolor=TEAL, edgecolor="none"))
        ax.text(0.895, 0.605 - i * 0.050, txt, fontsize=6.7, va="center", color=INK)
    ax.text(0.605, 0.435, "Named ROIs", ha="center", color=MUTED)
    ax.text(0.772, 0.435, "Eight-value AEV", ha="center", color=MUTED)
    ax.text(0.908, 0.435, "Audit tests", ha="center", color=MUTED)
    ax.text(0.752, 0.265, "Comparable regional evidence", ha="center", fontsize=8.2,
            fontweight="bold", color=INK)
    ax.text(0.752, 0.205, "Named measurements support cross-sample\nstatistics and controlled validation",
            ha="center", va="top", color=MUTED, linespacing=1.3)

    ax.text(0.500, 0.070,
            "Representation: pixels → named regions     Output: heatmap → vector     Validation: visual → functional and statistical",
            ha="center", va="center", fontsize=7.0, color=BLUE_DARK, fontweight="bold")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_all(fig, "concept_1_pixel_vs_regional_aev")


def figure_aev_construction():
    fig, ax = plt.subplots(figsize=(160 / 25.4, 76 / 25.4))
    setup_ax(ax)
    ax.text(0.025, 0.945, "Conceptual construction of an Anatomical Evidence Vector",
            fontsize=12.5, fontweight="bold", color=INK, va="top")
    ax.text(0.025, 0.882, "Each AEV component records the true-class confidence change caused by masking one named ROI",
            fontsize=7.8, color=MUTED, va="top")

    xs = [0.025, 0.220, 0.415, 0.610, 0.805]
    ws = [0.155] * 5
    card_colors = [PALE_BLUE, PALE_TEAL, PALE_BLUE, PALE_ORANGE, PALE_TEAL]
    titles = ["1  Full image", "2  Mask one ROI", "3  Re-run classifier", "4  Compute Δp", "5  Assemble AEV"]
    subtitles = [
        "Obtain baseline\ntrue-class confidence",
        "Replace the selected\nregion with the baseline",
        "Keep model and all\nother pixels fixed",
        "p(original) − p(masked)",
        "Repeat over the\neight named regions",
    ]
    for x, w, fc, title, sub in zip(xs, ws, card_colors, titles, subtitles):
        rounded(ax, x, 0.18, w, 0.60, fc=fc, ec=LINE if fc != PALE_ORANGE else "#E5B772")
        ax.text(x + 0.014, 0.730, title, fontsize=8.4, fontweight="bold", color=INK, va="top")
        ax.text(x + w / 2, 0.270, sub, fontsize=6.8, color=MUTED, ha="center", va="top", linespacing=1.3)

    for i in range(4):
        arrow(ax, xs[i] + ws[i] + 0.006, 0.490, xs[i + 1] - 0.008, 0.490, ms=8)

    draw_face(ax, xs[0] + 0.078, 0.515, 1.25)
    rounded(ax, xs[0] + 0.027, 0.345, 0.102, 0.055, fc=WHITE)
    ax.text(xs[0] + 0.078, 0.373, r"$p_t(x)$", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=BLUE_DARK)

    draw_face(ax, xs[1] + 0.078, 0.515, 1.25, masked=4)
    ax.text(xs[1] + 0.078, 0.382, "one ROI replaced", ha="center", color=ORANGE, fontsize=6.8,
            fontweight="bold")

    # Compact classifier icon.
    cx, cy = xs[2] + 0.078, 0.520
    for col, n in enumerate([4, 3, 2]):
        xx = cx - 0.050 + col * 0.050
        ys = np.linspace(cy - 0.055, cy + 0.055, n)
        for yy in ys:
            ax.add_patch(Circle((xx, yy), 0.008, facecolor=BLUE if col == 1 else "#AAB7C4", edgecolor=WHITE, lw=0.4))
        if col < 2:
            next_ys = np.linspace(cy - 0.055, cy + 0.055, [3, 2][col])
            for yy in ys:
                for ny in next_ys:
                    ax.plot([xx + 0.008, xx + 0.042], [yy, ny], color="#B6C0CA", lw=0.35, zorder=1)
    rounded(ax, xs[2] + 0.027, 0.345, 0.102, 0.055, fc=WHITE)
    ax.text(xs[2] + 0.078, 0.373, r"$p_t(x^{(r)})$", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=BLUE_DARK)

    rounded(ax, xs[3] + 0.023, 0.420, 0.109, 0.120, fc=WHITE, ec="#E5B772")
    ax.text(xs[3] + 0.078, 0.500, r"$\Delta_r = p_t(x) - p_t(x^{(r)})$", ha="center", va="center",
            fontsize=8.2, color=INK, fontweight="bold")
    ax.text(xs[3] + 0.078, 0.448, "positive = confidence drop", ha="center", fontsize=6.2, color=RED)

    # Vector chips, intentionally without values because this is not a results figure.
    vx, vy = xs[4] + 0.025, 0.405
    for i, c in enumerate(ROI_COLORS):
        col, row = i % 4, i // 4
        rounded(ax, vx + col * 0.027, vy + (1 - row) * 0.080, 0.022, 0.055,
                fc=c, ec=c, radius=0.008)
        ax.text(vx + col * 0.027 + 0.011, vy + (1 - row) * 0.080 + 0.027,
                str(i + 1), ha="center", va="center", color=WHITE, fontsize=6.5, fontweight="bold")
    ax.text(xs[4] + 0.078, 0.365, r"$\mathbf{a}_i = (\Delta_1, \ldots, \Delta_8)$", ha="center", fontsize=8.3,
            fontweight="bold", color=BLUE_DARK)

    ax.text(0.500, 0.085,
            "Controlled intervention: only the selected ROI changes; model weights, comparison image and scoring rule remain fixed",
            ha="center", va="center", color=MUTED, fontsize=7.0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_all(fig, "concept_2_aev_construction")


def figure_evidence_layers():
    fig, ax = plt.subplots(figsize=(160 / 25.4, 80 / 25.4))
    setup_ax(ax)
    ax.text(0.025, 0.945, "Three evidence layers in the regional explanation audit",
            fontsize=12.5, fontweight="bold", color=INK, va="top")
    ax.text(0.025, 0.882, "The dissertation combines predictive validity, regional function and reliability checks",
            fontsize=7.8, color=MUTED, va="top")

    layers = [
        (0.095, 0.145, 0.690, 0.175, PALE_BLUE, BLUE_DARK,
         "1  Predictive validity",
         ["Data QC", "Baseline classifier", "Calibration"],
         "Defines whether model output is reliable enough to interpret"),
        (0.135, 0.360, 0.650, 0.175, PALE_TEAL, TEAL,
         "2  Regional functional evidence",
         ["Mask-out AEV", "Class statistics", "Region-only validation"],
         "Measures whether named regions influence or retain predictions"),
        (0.175, 0.575, 0.610, 0.175, PALE_ORANGE, ORANGE,
         "3  Reliability and consistency",
         ["Similarity", "ROI geometry", "Pixel occlusion", "Grad-CAM"],
         "Tests stability across samples, ROI definitions and methods"),
    ]

    for x, y, w, h, fc, accent, title, chips, note in layers:
        rounded(ax, x, y, w, h, fc=fc, ec=accent, lw=0.9)
        ax.add_patch(Rectangle((x, y), 0.010, h, facecolor=accent, edgecolor="none", zorder=2))
        ax.text(x + 0.028, y + h - 0.036, title, fontsize=8.7, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.028, y + h - 0.075, note, ha="left", va="top",
                fontsize=6.15, color=MUTED)
        chip_x = x + 0.028
        chip_y = y + 0.022
        gap = 0.010
        width = (w - 0.056 - gap * (len(chips) - 1)) / len(chips)
        for label in chips:
            rounded(ax, chip_x, chip_y, width, 0.045, fc=WHITE, ec=LINE, radius=0.012)
            ax.text(chip_x + width / 2, chip_y + 0.0225, label, ha="center", va="center",
                    fontsize=6.0, color=INK)
            chip_x += width + gap

    # Right-side synthesis panel.
    rounded(ax, 0.825, 0.180, 0.145, 0.555, fc="#F7F8FA", ec=LINE)
    ax.text(0.897, 0.690, "Synthesis", ha="center", fontsize=8.8, fontweight="bold", color=INK)
    for i, (label, c) in enumerate([
        ("Valid\npredictions", BLUE_DARK),
        ("Functional\nregions", TEAL),
        ("Stable\nevidence", ORANGE),
    ]):
        yy = 0.595 - i * 0.125
        ax.add_patch(Circle((0.862, yy), 0.018, facecolor=c, edgecolor="none"))
        ax.text(0.892, yy, label, va="center", ha="left", fontsize=7.0, color=INK, linespacing=1.05)
        if i < 2:
            arrow(ax, 0.862, yy - 0.028, 0.862, yy - 0.087, color="#9AA4AE", ms=7)
    rounded(ax, 0.846, 0.215, 0.103, 0.068, fc=WHITE, ec=BLUE_DARK, radius=0.015)
    ax.text(0.897, 0.249, "Auditable ROI\nevidence", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=BLUE_DARK, linespacing=1.1)

    ax.text(0.440, 0.095,
            "Evidence accumulates across layers; no single heatmap or metric carries the full interpretation",
            ha="center", va="center", color=MUTED, fontsize=7.0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_all(fig, "concept_3_evidence_layers")


def main():
    figure_pixel_vs_aev()
    figure_aev_construction()
    figure_evidence_layers()
    print(f"Wrote concept figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

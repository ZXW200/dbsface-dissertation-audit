"""Generate additional publication figures from frozen dissertation outputs.

The script is deliberately read-only with respect to analysis outputs: it does not
train models or recompute inferential statistics.  It renders three figures that
visualise values already reported in the dissertation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Publication contract: editable SVG text and TrueType text in PDF.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "latex_project" / "figures" / "dissertation"

INK = "#272727"
MUTED = "#6B7280"
GRID = "#D8DEE6"
BLUE = "#0F4D92"
BLUE_SOFT = "#7884B4"
TEAL = "#42949E"
ORANGE = "#D9822B"
RED = "#B64342"
LIGHT = "#F4F6F8"
MID = "#9AA4AF"

ROI_LABELS = {
    "upper_brow_forehead": "Upper brow/forehead",
    "left_periocular": "Left periocular",
    "right_periocular": "Right periocular",
    "nasal_midface": "Nasal midface",
    "left_cheek_zygomatic": "Left cheek/zygomatic",
    "right_cheek_zygomatic": "Right cheek/zygomatic",
    "perioral_mouth": "Perioral/mouth",
    "chin_mandible": "Chin/mandible",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save an editable vector bundle plus a high-resolution raster preview."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def _draw_forest_series(
    ax: plt.Axes,
    frame: pd.DataFrame,
    y_positions: np.ndarray,
    color: str,
    marker: str,
    label: str,
) -> None:
    for row, y in zip(frame.itertuples(index=False), y_positions):
        estimate = float(row.diff_class1_minus_class0)
        low = float(row.bootstrap_ci_low)
        high = float(row.bootstrap_ci_high)
        significant = float(row.p_fdr) < 0.05
        ax.plot([low, high], [y, y], color=color, linewidth=1.25, zorder=2)
        ax.plot([low, low], [y - 0.055, y + 0.055], color=color, linewidth=0.8, zorder=2)
        ax.plot([high, high], [y - 0.055, y + 0.055], color=color, linewidth=0.8, zorder=2)
        ax.scatter(
            [estimate],
            [y],
            s=28,
            marker=marker,
            facecolor=color if significant else "white",
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )
    # Legend handle is added once per series, independent of FDR fill state.
    ax.plot([], [], color=color, marker=marker, markerfacecolor=color, linewidth=1.25, label=label)


def make_class_difference_forest() -> None:
    """Full-split and confidence-matched AEV mean differences with image-level CIs."""
    full = pd.read_csv(ROOT / "outputs" / "aev" / "roi_class_comparison.csv")
    matched = pd.read_csv(
        ROOT
        / "outputs"
        / "aev"
        / "confidence_matched"
        / "confidence_matched_roi_class_comparison.csv"
    )

    # Preserve the confidence-matched result order: strongest persistent directions first.
    order = matched["roi"].tolist()
    full = full.set_index("roi").loc[order].reset_index()
    matched = matched.set_index("roi").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(160 / 25.4, 91 / 25.4))
    y = np.arange(len(order), dtype=float)
    _draw_forest_series(ax, full, y - 0.13, BLUE_SOFT, "o", "Full test split (675 / 496)")
    _draw_forest_series(ax, matched, y + 0.13, ORANGE, "s", "Confidence-matched (451 / 451)")

    ax.axvline(0, color=INK, linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([ROI_LABELS[r] for r in order])
    ax.invert_yaxis()
    ax.set_xlim(-0.09, 0.085)
    ax.set_xlabel("Mean AEV difference (Class 1 - Class 0)")
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=6.5,
        handlelength=2.6,
        columnspacing=1.8,
    )

    ax.text(
        0.01,
        1.11,
        "Filled marker: FDR-adjusted p < 0.05; horizontal line: image-level bootstrap 95% CI",
        transform=ax.transAxes,
        fontsize=6.4,
        color=MUTED,
        va="bottom",
    )
    ax.text(0.02, -0.15, "higher in Class 0", transform=ax.transAxes, color=BLUE, fontsize=6.4)
    ax.text(0.98, -0.15, "higher in Class 1", transform=ax.transAxes, color=RED, fontsize=6.4, ha="right")
    fig.subplots_adjust(left=0.30, right=0.985, top=0.86, bottom=0.18)
    save_figure(fig, "fig_aev_confidence_matched_forest")


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an empirical ROC curve without adding a plotting dependency."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = scores[order]
    true_positive = np.cumsum(y_sorted == 1)
    false_positive = np.cumsum(y_sorted == 0)
    threshold_ends = np.r_[np.where(np.diff(scores_sorted) != 0)[0], len(scores_sorted) - 1]
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)
    tpr = np.r_[0.0, true_positive[threshold_ends] / positives]
    fpr = np.r_[0.0, false_positive[threshold_ends] / negatives]
    return fpr, tpr


def make_signal_profile() -> None:
    """Visualise the shared-protocol OOF low-level/raw-pixel score comparison."""
    pred = pd.read_csv(
        ROOT
        / "outputs"
        / "confound_net_increment"
        / "confound_net_increment_oof_predictions.csv"
    )
    metrics = json.loads(
        (
            ROOT
            / "outputs"
            / "confound_net_increment"
            / "confound_net_increment.json"
        ).read_text(encoding="utf-8")
    )

    series = [
        ("ROI low-level summaries", "oof_lowlevel_only", float(metrics["lowlevel_only_auroc"]), MID),
        ("Raw-pixel MLP score", "oof_mlp_score_only", float(metrics["mlp_score_only_auroc"]), BLUE),
        ("Combined score blocks", "oof_combined", float(metrics["combined_auroc"]), ORANGE),
    ]

    fig, (ax_roc, ax_auc) = plt.subplots(
        1,
        2,
        figsize=(160 / 25.4, 84 / 25.4),
        gridspec_kw={"width_ratios": [1.18, 0.92]},
    )
    y_true = pred["y_true"].to_numpy(dtype=int)
    for name, column, auc, color in series:
        fpr, tpr = roc_curve_points(y_true, pred[column].to_numpy(dtype=float))
        linestyle = "--" if name == "Combined" else "-"
        ax_roc.plot(fpr, tpr, color=color, linewidth=1.6, linestyle=linestyle, label=f"{name}: {auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], color=GRID, linewidth=0.9, linestyle=":")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.015)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("Five-fold OOF ROC curves", loc="left", fontsize=8.2, fontweight="bold")
    ax_roc.grid(color=GRID, linewidth=0.45)
    ax_roc.legend(loc="lower right", fontsize=6.2)
    add_panel_label(ax_roc, "a")

    labels = [s[0] for s in series]
    values = np.array([s[2] for s in series])
    colors = [s[3] for s in series]
    y = np.arange(len(labels))[::-1]
    for yi, value, color in zip(y, values, colors):
        ax_auc.plot([0.72, value], [yi, yi], color=color, linewidth=2.0, alpha=0.75)
        ax_auc.scatter([value], [yi], s=34, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax_auc.text(value + 0.006, yi, f"{value:.4f}", va="center", fontsize=7.0, color=INK)
    ax_auc.set_yticks(y)
    ax_auc.set_yticklabels(labels)
    ax_auc.set_xlim(0.72, 1.015)
    ax_auc.set_ylim(-0.75, 2.65)
    ax_auc.set_xlabel("AUROC")
    ax_auc.set_title("Ranking performance", loc="left", fontsize=8.2, fontweight="bold")
    ax_auc.grid(axis="x", color=GRID, linewidth=0.45)
    ax_auc.tick_params(axis="y", length=0)
    ax_auc.spines["left"].set_visible(False)
    ax_auc.text(
        0.03,
        0.08,
        "Combined - low-level: +0.2043\nCombined - MLP score: -0.0042",
        transform=ax_auc.transAxes,
        fontsize=6.7,
        color=INK,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": GRID, "linewidth": 0.7},
    )
    add_panel_label(ax_auc, "b")

    fig.text(
        0.5,
        0.015,
        "Shared protocol: five-fold out-of-fold logistic analysis within the test split (n = 1,171)",
        ha="center",
        fontsize=6.6,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.20, wspace=0.40)
    save_figure(fig, "fig_lowlevel_fullpixel_oof_profile")


def make_intervention_geometry_stability() -> None:
    """Compare fill-strategy rank dependence with fixed-atlas jitter stability."""
    sensitivity_root = ROOT / "outputs" / "sensitivity" / "roi_supplementary_experiments"
    fill = pd.read_csv(sensitivity_root / "occlusion_fill_strategy_rank_stability.csv")
    jitter = pd.read_csv(sensitivity_root / "fixed_roi_jitter_summary_by_roi.csv")
    top_rank = pd.read_csv(sensitivity_root / "fixed_roi_jitter_top_rank_summary.csv")

    fill_order = ["train_mean", "zero", "blur3_same_image"]
    fill = fill.set_index("fill_strategy").loc[fill_order].reset_index()
    fill_labels = {
        "train_mean": "Training mean (reference)",
        "zero": "Zero fill",
        "blur3_same_image": "Blur fill",
    }

    jitter = jitter.sort_values("original_auroc", ascending=False).reset_index(drop=True)
    top_roi = "right_cheek_zygomatic"
    top_count = int((top_rank["top1_roi"] == top_roi).sum())

    fig, (ax_fill, ax_jitter) = plt.subplots(
        1,
        2,
        figsize=(160 / 25.4, 93 / 25.4),
        gridspec_kw={"width_ratios": [0.82, 1.35]},
    )

    y_fill = np.arange(len(fill_order))[::-1]
    fill_colors = [BLUE, TEAL, ORANGE]
    for yi, row, color in zip(y_fill, fill.itertuples(index=False), fill_colors):
        rho = float(row.spearman_vs_train_mean)
        overlap = int(float(row.top3_overlap_with_train_mean))
        ax_fill.plot([0, rho], [yi, yi], color=color, linewidth=2.2, alpha=0.75)
        ax_fill.scatter([rho], [yi], s=38, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        if rho > 0.85:
            text_x, text_ha = rho - 0.07, "right"
        else:
            text_x, text_ha = rho + 0.04, "left"
        ax_fill.text(
            text_x,
            yi,
            f"Spearman $\\rho$ = {rho:.3f}\nTop-three: {overlap}/3",
            va="center",
            ha=text_ha,
            fontsize=6.6,
        )
    ax_fill.set_yticks(y_fill)
    ax_fill.set_yticklabels([fill_labels[x] for x in fill_order])
    ax_fill.set_xlim(0, 1.08)
    ax_fill.set_ylim(-0.65, 2.65)
    ax_fill.set_xlabel("Spearman correlation vs training-mean fill")
    ax_fill.set_title("Replacement-value dependence", loc="left", fontsize=8.2, fontweight="bold")
    ax_fill.grid(axis="x", color=GRID, linewidth=0.45)
    ax_fill.tick_params(axis="y", length=0)
    ax_fill.spines["left"].set_visible(False)
    add_panel_label(ax_fill, "a")

    y_jitter = np.arange(len(jitter))[::-1]
    for yi, row in zip(y_jitter, jitter.itertuples(index=False)):
        highlight = row.roi_name == top_roi
        color = BLUE if highlight else MID
        ax_jitter.plot(
            [float(row.jitter_auroc_min), float(row.jitter_auroc_max)],
            [yi, yi],
            color=color,
            linewidth=1.6 if highlight else 1.1,
            alpha=0.95,
        )
        ax_jitter.scatter(
            [float(row.jitter_auroc_mean)],
            [yi],
            color=color,
            s=27 if highlight else 20,
            zorder=3,
            label="Jitter mean" if yi == y_jitter[0] else None,
        )
        ax_jitter.scatter(
            [float(row.original_auroc)],
            [yi],
            marker="D",
            facecolor="white",
            edgecolor=INK,
            linewidth=0.8,
            s=24,
            zorder=4,
            label="Original atlas" if yi == y_jitter[0] else None,
        )
    ax_jitter.set_yticks(y_jitter)
    ax_jitter.set_yticklabels([ROI_LABELS[r] for r in jitter["roi_name"]])
    low = min(float(jitter["jitter_auroc_min"].min()), float(jitter["original_auroc"].min())) - 0.015
    high = max(float(jitter["jitter_auroc_max"].max()), float(jitter["original_auroc"].max())) + 0.02
    ax_jitter.set_xlim(low, high)
    ax_jitter.set_ylim(-0.5, 8.7)
    ax_jitter.set_xlabel("Region-only AUROC")
    ax_jitter.set_title("Fixed-atlas jitter (10 shifted/scaled variants)", loc="left", fontsize=8.2, fontweight="bold")
    ax_jitter.grid(axis="x", color=GRID, linewidth=0.45)
    ax_jitter.tick_params(axis="y", length=0)
    ax_jitter.spines["left"].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color=MID, linewidth=1.2, markersize=4, label="Jitter mean and min\u2013max"),
        Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor=INK, color="none", markersize=4, label="Original atlas"),
    ]
    ax_jitter.legend(handles=handles, loc="lower right", fontsize=6.2)
    ax_jitter.text(
        0.98,
        0.94,
        f"Right cheek/zygomatic was top-ranked\nby balanced accuracy in {top_count} of {len(top_rank)} atlas conditions",
        transform=ax_jitter.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=INK,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": LIGHT, "edgecolor": GRID, "linewidth": 0.7},
    )
    add_panel_label(ax_jitter, "b")

    fig.subplots_adjust(left=0.16, right=0.985, top=0.88, bottom=0.18, wspace=0.62)
    save_figure(fig, "fig_fill_and_geometry_stability")


def main() -> int:
    make_class_difference_forest()
    make_signal_profile()
    make_intervention_geometry_stability()
    print(f"Wrote additional dissertation figures to {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

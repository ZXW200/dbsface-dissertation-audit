"""Render the frozen representation-by-estimator follow-up analysis.

The script reads the authoritative v2 per-seed metrics only. It does not train
models, resample predictions, or alter any analysis output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DATA_PATH = (
    ROOT
    / "outputs"
    / "representation_capacity_analysis_20260718_v2"
    / "per_seed_metrics.csv"
)
OUT_DIR = ROOT / "latex_project" / "figures" / "dissertation"

INK = "#272727"
MUTED = "#667085"
GRID = "#D8DEE6"
LOGISTIC = "#6B7280"
MLP = "#0F4D92"
SHUFFLED = "#D9822B"

EXPECTED_MODELS = {
    "lowlevel_logistic",
    "lowlevel_mlp",
    "raw_logistic",
    "raw_mlp_reference",
    "roi_shuffled_raw_mlp",
}
EXPECTED_SEEDS = {0, 1, 2, 3, 4}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def read_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    models = set(frame["model"].astype(str))
    seeds = set(frame["seed"].astype(int))
    if models != EXPECTED_MODELS:
        raise ValueError(f"Unexpected model set: {sorted(models)}")
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seed set: {sorted(seeds)}")
    counts = frame.groupby("model")["seed"].nunique()
    if not (counts == len(EXPECTED_SEEDS)).all():
        raise ValueError("Each model must contain exactly five distinct seeds")
    return frame


def model_values(frame: pd.DataFrame, model: str) -> np.ndarray:
    subset = frame.loc[frame["model"] == model].sort_values("seed")
    return subset["auroc"].to_numpy(dtype=float)


def draw_mean_sd(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    marker: str,
) -> None:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    ax.errorbar(
        [x],
        [mean],
        yerr=[sd],
        color=color,
        linewidth=1.4,
        capsize=3.0,
        capthick=1.0,
        marker=marker,
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.2,
        zorder=5,
    )


def make_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, (ax_factorial, ax_shuffle) = plt.subplots(
        1,
        2,
        figsize=(170 / 25.4, 86 / 25.4),
        gridspec_kw={"width_ratios": [1.30, 0.90]},
    )

    # Panel a: the seed-matched 2 x 2 representation-by-estimator comparison.
    groups = [("Low-level ROI\nsummaries", "lowlevel"), ("Raw pixels", "raw")]
    estimator_specs = [
        ("Logistic", "logistic", -0.16, LOGISTIC, "o"),
        ("MLP", "mlp", 0.16, MLP, "s"),
    ]
    jitter = np.array([-0.036, -0.018, 0.0, 0.018, 0.036])

    for estimator_label, estimator_key, offset, color, marker in estimator_specs:
        means = []
        for group_index, (_, representation_key) in enumerate(groups):
            model = f"{representation_key}_{estimator_key}"
            if model == "raw_mlp":
                model = "raw_mlp_reference"
            values = model_values(frame, model)
            x = float(group_index) + offset
            ax_factorial.scatter(
                x + jitter,
                values,
                s=17,
                color=color,
                alpha=0.56,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            draw_mean_sd(ax_factorial, x, values, color, marker)
            mean = float(np.mean(values))
            means.append(mean)
            ax_factorial.text(
                x,
                mean + 0.010,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=color,
            )
        ax_factorial.plot(
            np.arange(len(groups), dtype=float) + offset,
            means,
            color=color,
            linewidth=1.15,
            alpha=0.72,
            zorder=2,
            label=estimator_label,
        )

    ax_factorial.set_xticks(np.arange(len(groups), dtype=float))
    ax_factorial.set_xticklabels([label for label, _ in groups])
    ax_factorial.set_ylim(0.80, 1.005)
    ax_factorial.set_yticks(np.arange(0.80, 1.001, 0.05))
    ax_factorial.set_ylabel("Test AUROC")
    ax_factorial.set_title(
        "Representation × estimator family",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
    )
    ax_factorial.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax_factorial.legend(loc="lower right", fontsize=6.7, handlelength=2.0)
    ax_factorial.text(
        0.0,
        -0.20,
        "Small points: seeds 0–4; open markers/error bars: mean ± SD",
        transform=ax_factorial.transAxes,
        fontsize=6.2,
        color=MUTED,
    )
    add_panel_label(ax_factorial, "a")

    # Panel b: paired seed results before and after the within-group perturbation.
    raw = model_values(frame, "raw_mlp_reference")
    shuffled = model_values(frame, "roi_shuffled_raw_mlp")
    x_positions = np.array([0.0, 1.0])
    for seed_index, (raw_value, shuffled_value) in enumerate(zip(raw, shuffled)):
        ax_shuffle.plot(
            x_positions,
            [raw_value, shuffled_value],
            color="#AAB2BD",
            linewidth=0.9,
            alpha=0.82,
            zorder=1,
        )
        ax_shuffle.scatter(
            [0.0, 1.0],
            [raw_value, shuffled_value],
            s=19,
            color=[MLP, SHUFFLED],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.70,
            zorder=2,
        )
    draw_mean_sd(ax_shuffle, 0.0, raw, MLP, "s")
    draw_mean_sd(ax_shuffle, 1.0, shuffled, SHUFFLED, "s")

    raw_mean = float(np.mean(raw))
    shuffled_mean = float(np.mean(shuffled))
    delta = shuffled_mean - raw_mean
    ax_shuffle.text(0.0, raw_mean + 0.020, f"{raw_mean:.3f}", ha="center", color=MLP, fontsize=6.6)
    ax_shuffle.text(
        1.0,
        shuffled_mean - 0.035,
        f"{shuffled_mean:.3f}",
        ha="center",
        color=SHUFFLED,
        fontsize=6.6,
    )
    ax_shuffle.annotate(
        f"Mean paired change\n{delta:.3f} AUROC",
        xy=(0.58, 0.79),
        xycoords="data",
        ha="center",
        va="center",
        fontsize=6.7,
        color=INK,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": GRID},
    )
    ax_shuffle.set_xticks(x_positions)
    ax_shuffle.set_xticklabels(["Raw-pixel\nMLP", "Spatially permuted\nMLP"])
    ax_shuffle.set_xlim(-0.28, 1.28)
    ax_shuffle.set_ylim(0.60, 1.005)
    ax_shuffle.set_yticks(np.arange(0.60, 1.001, 0.10))
    ax_shuffle.set_title(
        "Within-group spatial permutation",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
    )
    ax_shuffle.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax_shuffle.text(
        0.0,
        -0.20,
        "Lines pair identical seeds; each MLP was retrained",
        transform=ax_shuffle.transAxes,
        fontsize=6.2,
        color=MUTED,
    )
    add_panel_label(ax_shuffle, "b")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.25, wspace=0.34)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, stem_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / stem_name
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stem", default="fig_representation_estimator_analysis")
    args = parser.parse_args()
    frame = read_metrics(args.metrics)
    save_figure(make_figure(frame), args.output_dir, args.stem)


if __name__ == "__main__":
    main()

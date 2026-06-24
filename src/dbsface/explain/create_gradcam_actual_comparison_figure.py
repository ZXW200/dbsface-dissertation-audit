"""Create a dissertation figure with actual Grad-CAM and ROI examples.

The figure deliberately separates two things:

* single-image visual sanity checks, shown as original / Grad-CAM / fixed ROI
  AEV / YuNet automatic ROI AEV for the same test image; and
* full-test-set ROI agreement metrics, reported as compact text.

This avoids treating an individual heatmap as the quantitative result.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs


SAMPLE_IDS = ["test_0508", "test_0856"]


ROI_SHORT_NAMES = {
    "upper_brow_forehead": "upper brow",
    "left_periocular": "left eye",
    "right_periocular": "right eye",
    "nasal_midface": "nasal midface",
    "left_cheek_zygomatic": "left cheek",
    "right_cheek_zygomatic": "right cheek",
    "perioral_mouth": "mouth",
    "chin_mandible": "chin",
}


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        return np.zeros_like(image, dtype=float)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x = x - float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx > 1e-12:
        x = x / mx
    return x


def roi_map_for_sample(occ_sample: pd.DataFrame, masks: np.ndarray) -> tuple[np.ndarray, str, float]:
    positive = occ_sample.copy()
    positive["positive_drop"] = positive["evidence_drop"].clip(lower=0)
    top = positive.sort_values("positive_drop", ascending=False).iloc[0]
    roi_map = np.zeros((32, 32), dtype=float)
    for _, row in positive.iterrows():
        roi_idx = int(row["roi_index"]) - 1
        roi_map[masks[roi_idx].astype(bool)] = float(row["positive_drop"])
    return normalize_map(roi_map), str(top["roi_name"]), float(top["positive_drop"])


def dynamic_roi_map_for_sample(
    occ_sample: pd.DataFrame, box_sample: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, str, float]:
    positive = occ_sample.copy()
    positive["positive_drop"] = positive["evidence_drop"].clip(lower=0)
    top = positive.sort_values("positive_drop", ascending=False).iloc[0]
    roi_map = np.zeros((32, 32), dtype=float)
    top_mask = np.zeros((32, 32), dtype=bool)

    for _, row in positive.iterrows():
        boxes = box_sample.loc[box_sample["roi_name"] == row["roi_name"]]
        if boxes.empty:
            continue
        box = boxes.iloc[0]
        x1 = max(0, min(31, int(np.floor(float(box["x1"]) / 8.0))))
        y1 = max(0, min(31, int(np.floor(float(box["y1"]) / 8.0))))
        x2 = max(x1 + 1, min(32, int(np.ceil(float(box["x2"]) / 8.0))))
        y2 = max(y1 + 1, min(32, int(np.ceil(float(box["y2"]) / 8.0))))
        roi_map[y1:y2, x1:x2] = np.maximum(roi_map[y1:y2, x1:x2], float(row["positive_drop"]))
        if row["roi_name"] == top["roi_name"]:
            top_mask[y1:y2, x1:x2] = True

    return normalize_map(roi_map), top_mask, str(top["roi_name"]), float(top["positive_drop"])


def add_image_axis(ax: plt.Axes, title: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=11, pad=6, color="#142b4a", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#c8d2df")


def main() -> int:
    data = load_pd_dbs("data/raw/PD_DBS_Data.mat")
    test_images = data["x_test_images"].astype(np.float32)
    y_test = data["y_test"].astype(int)

    out_dir = Path("outputs/gradcam_occlusion_overlap")
    heatmaps = np.load(out_dir / "gradcam_heatmaps_test.npy")
    predictions = pd.read_csv(out_dir / "cnn_predictions_test.csv")
    occlusion = pd.read_csv(out_dir / "cnn_roi_occlusion_test.csv")
    yunet_occlusion = pd.read_csv(out_dir / "cnn_yunet_roi_occlusion_test.csv")
    gradcam_roi = pd.read_csv(out_dir / "gradcam_roi_test.csv")
    overlap = pd.read_csv(out_dir / "gradcam_occlusion_roi_overlap.csv")
    metrics = json.loads((out_dir / "overlap_metrics.json").read_text(encoding="utf-8"))
    masks = np.load("outputs/roi/coarse_roi_masks.npy").astype(bool)
    yunet_boxes = pd.read_csv("outputs/external/pd_dbs_yunet_region_only/yunet_roi_boxes.csv")
    yunet_boxes = yunet_boxes.loc[yunet_boxes["split"] == "test"].copy()

    fig_height = 4.2 + 2.25 * len(SAMPLE_IDS)
    fig = plt.figure(figsize=(12.2, fig_height), dpi=180, facecolor="white")
    gs = fig.add_gridspec(
        nrows=len(SAMPLE_IDS) + 2,
        ncols=4,
        height_ratios=[0.42, *([1] * len(SAMPLE_IDS)), 0.55],
        hspace=0.54,
        wspace=0.14,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.72,
        "Actual test-image comparison",
        fontsize=19,
        fontweight="bold",
        color="#142b4a",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.0,
        0.08,
        "Rows show the same 32x32 test image across columns. Grad-CAM, fixed-ROI occlusion, and YuNet-ROI occlusion all use the same CNN.\nClass 0 = pre-DBS; Class 1 = post-DBS label.",
        fontsize=10.2,
        color="#536174",
        transform=title_ax.transAxes,
    )

    for row_idx, sample_id in enumerate(SAMPLE_IDS, start=1):
        test_idx = int(sample_id.split("_")[1])
        raw = contrast_stretch(test_images[test_idx, :, :, 0])
        cam = normalize_map(heatmaps[test_idx])
        pred_row = predictions.loc[predictions["sample_id"] == sample_id].iloc[0]

        occ_sample = occlusion.loc[occlusion["sample_id"] == sample_id]
        roi_map, top_occ_roi, top_occ_drop = roi_map_for_sample(occ_sample, masks)
        yunet_occ_sample = yunet_occlusion.loc[yunet_occlusion["sample_id"] == sample_id]
        yunet_box_sample = yunet_boxes.loc[yunet_boxes["sample_id"] == sample_id]
        yunet_map, top_yunet_mask, top_yunet_roi, top_yunet_drop = dynamic_roi_map_for_sample(
            yunet_occ_sample, yunet_box_sample
        )
        cam_sample = gradcam_roi.loc[gradcam_roi["sample_id"] == sample_id].sort_values(
            "cam_energy_fraction", ascending=False
        )
        top_cam_roi = str(cam_sample.iloc[0]["roi_name"])
        top_cam_fraction = float(cam_sample.iloc[0]["cam_energy_fraction"])

        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(raw, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        add_image_axis(ax0, "Original" if row_idx == 1 else None)
        ax0.text(
            0.02,
            -0.18,
            f"{sample_id} | Class {int(y_test[test_idx])} | p(Class 1)={float(pred_row['p_class1']):.2f}",
            transform=ax0.transAxes,
            fontsize=8.6,
            color="#142b4a",
            ha="left",
        )

        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.imshow(raw, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax1.imshow(cam, cmap="inferno", interpolation="bilinear", alpha=np.clip(0.12 + 0.55 * cam, 0.0, 0.62))
        add_image_axis(ax1, "Grad-CAM (CNN)" if row_idx == 1 else None)
        ax1.text(
            0.02,
            -0.18,
            f"top ROI: {ROI_SHORT_NAMES.get(top_cam_roi, top_cam_roi)} ({top_cam_fraction:.2f})",
            transform=ax1.transAxes,
            fontsize=8.6,
            color="#536174",
            ha="left",
        )

        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.imshow(raw, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax2.imshow(roi_map, cmap="Reds", interpolation="nearest", alpha=np.clip(0.05 + 0.58 * roi_map, 0.0, 0.62))
        top_mask = masks[int(occ_sample.loc[occ_sample["roi_name"] == top_occ_roi, "roi_index"].iloc[0]) - 1]
        ax2.contour(top_mask.astype(float), levels=[0.5], colors=["#00d0ff"], linewidths=1.15)
        add_image_axis(ax2, "Fixed ROI occ. (CNN)" if row_idx == 1 else None)
        ax2.text(
            0.02,
            -0.18,
            f"top ROI: {ROI_SHORT_NAMES.get(top_occ_roi, top_occ_roi)} (drop={top_occ_drop:.2f})",
            transform=ax2.transAxes,
            fontsize=8.6,
            color="#536174",
            ha="left",
        )

        ax3 = fig.add_subplot(gs[row_idx, 3])
        ax3.imshow(raw, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax3.imshow(yunet_map, cmap="Blues", interpolation="nearest", alpha=np.clip(0.05 + 0.58 * yunet_map, 0.0, 0.62))
        ax3.contour(top_yunet_mask.astype(float), levels=[0.5], colors=["#00d0ff"], linewidths=1.15)
        add_image_axis(ax3, "YuNet ROI occ. (CNN)" if row_idx == 1 else None)
        ax3.text(
            0.02,
            -0.18,
            f"top ROI: {ROI_SHORT_NAMES.get(top_yunet_roi, top_yunet_roi)} (drop={top_yunet_drop:.2f})",
            transform=ax3.transAxes,
            fontsize=8.6,
            color="#536174",
            ha="left",
        )

    bottom_ax = fig.add_subplot(gs[len(SAMPLE_IDS) + 1, :])
    bottom_ax.axis("off")
    top_roi_text = ", ".join(
        ROI_SHORT_NAMES.get(name, name) for name in overlap.sort_values("rank_cnn_occlusion").head(3)["roi_name"]
    )
    fixed_summary = pd.read_csv(out_dir / "cnn_roi_occlusion_summary_overall.csv")
    yunet_summary = pd.read_csv(out_dir / "cnn_yunet_roi_occlusion_summary_overall.csv")
    fixed_yunet = fixed_summary[["roi_name", "mean_evidence_drop"]].merge(
        yunet_summary[["roi_name", "mean_evidence_drop"]],
        on="roi_name",
        suffixes=("_fixed", "_yunet"),
    )
    fixed_rank = fixed_yunet["mean_evidence_drop_fixed"].rank(ascending=False, method="average")
    yunet_rank = fixed_yunet["mean_evidence_drop_yunet"].rank(ascending=False, method="average")
    fixed_yunet_spearman = float(np.corrcoef(fixed_rank, yunet_rank)[0, 1])
    fixed_yunet_top3 = set(
        fixed_yunet.sort_values("mean_evidence_drop_fixed", ascending=False).head(3)["roi_name"]
    )
    yunet_top3 = set(
        fixed_yunet.sort_values("mean_evidence_drop_yunet", ascending=False).head(3)["roi_name"]
    )
    fixed_yunet_top3_overlap = len(fixed_yunet_top3 & yunet_top3)
    bottom_ax.text(
        0.0,
        0.72,
        "Full test-set ROI agreement",
        fontsize=12,
        fontweight="bold",
        color="#142b4a",
        transform=bottom_ax.transAxes,
    )
    bottom_ax.text(
        0.0,
        0.25,
        (
            f"Fixed ROI Grad-CAM/occlusion agreement: CNN AUROC={metrics['cnn_test_auroc']:.4f}; ROI Spearman="
            f"{metrics['roi_spearman_rank_correlation']:.4f}; top-1 match="
            f"{'Yes' if metrics['top1_match'] else 'No'}; top-3 overlap="
            f"{metrics['top3_overlap_count']}/3 ({top_roi_text}).\n"
            f"Same-CNN fixed-vs-YuNet occlusion: ROI Spearman={fixed_yunet_spearman:.4f}; "
            f"top-3 overlap={fixed_yunet_top3_overlap}/3. Image rows are qualitative examples; metrics use full-test ROI aggregation."
        ),
        fontsize=9.2,
        color="#536174",
        transform=bottom_ax.transAxes,
    )

    out_path = Path("figures/dissertation/fig_gradcam_occlusion_overlap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    pd.DataFrame({"sample_id": SAMPLE_IDS}).to_csv(out_dir / "actual_comparison_figure_samples.csv", index=False)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

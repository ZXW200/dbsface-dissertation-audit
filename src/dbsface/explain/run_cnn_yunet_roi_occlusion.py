"""Run YuNet dynamic-ROI occlusion with the trained Grad-CAM CNN.

This creates the automatic-ROI counterpart to
``outputs/gradcam_occlusion_overlap/cnn_roi_occlusion_test.csv``. The fixed ROI
and YuNet ROI occlusion views can then be compared within the same CNN used for
Grad-CAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from load_pd_dbs import load_pd_dbs
from run_cnn_gradcam_occlusion_overlap import SmallGradCamCNN, images_to_tensor, predict_probs, true_confidence


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def boxes_to_masks(box_sample: pd.DataFrame) -> dict[str, tuple[np.ndarray, int]]:
    masks: dict[str, tuple[np.ndarray, int]] = {}
    for _, box in box_sample.iterrows():
        mask = np.zeros((32, 32), dtype=bool)
        x1 = max(0, min(31, int(np.floor(float(box["x1"]) / 8.0))))
        y1 = max(0, min(31, int(np.floor(float(box["y1"]) / 8.0))))
        x2 = max(x1 + 1, min(32, int(np.ceil(float(box["x2"]) / 8.0))))
        y2 = max(y1 + 1, min(32, int(np.ceil(float(box["y2"]) / 8.0))))
        mask[y1:y2, x1:x2] = True
        masks[str(box["roi_name"])] = (mask, int(mask.sum()))
    return masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--checkpoint", default="outputs/gradcam_occlusion_overlap/small_cnn_gradcam.pt")
    parser.add_argument("--boxes", default="outputs/external/pd_dbs_yunet_region_only/yunet_roi_boxes.csv")
    parser.add_argument("--roi-defs", default="outputs/roi/coarse_roi_definitions.csv")
    parser.add_argument("--output-dir", default="outputs/gradcam_occlusion_overlap")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    model = SmallGradCamCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = load_pd_dbs(args.data)
    test_images_raw = data["x_test_images"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)
    train_mean_raw = float(checkpoint["train_mean_raw"])
    train_std_raw = float(checkpoint["train_std_raw"])
    train_mean_image_raw = np.asarray(checkpoint["train_mean_image_raw"], dtype=np.float32)
    train_mean_image_normalized = ((train_mean_image_raw[:, :, 0] - train_mean_raw) / train_std_raw).astype(np.float32)
    test_images_norm = ((test_images_raw - train_mean_raw) / train_std_raw).astype(np.float32)
    x_test = images_to_tensor(test_images_norm)

    p_orig = predict_probs(model, x_test, args.batch_size, device)
    conf_orig = true_confidence(y_test, p_orig)
    y_pred_orig = (p_orig >= 0.5).astype(int)

    roi_defs = pd.read_csv(args.roi_defs)
    boxes = pd.read_csv(args.boxes)
    boxes = boxes.loc[boxes["split"] == "test"].copy()

    long_rows = []
    for i in range(len(y_test)):
        sample_id = f"test_{i:04d}"
        box_sample = boxes.loc[boxes["sample_id"] == sample_id]
        mask_by_roi = boxes_to_masks(box_sample)
        for _, roi in roi_defs.iterrows():
            roi_name = str(roi["roi_name"])
            roi_index = int(roi["roi_index"])
            missing = roi_name not in mask_by_roi
            if missing:
                mask = np.zeros((32, 32), dtype=bool)
                dynamic_roi_pixels = 0
            else:
                mask, dynamic_roi_pixels = mask_by_roi[roi_name]

            x_masked = x_test[i : i + 1].clone()
            if dynamic_roi_pixels:
                x_masked[0, 0, mask] = torch.from_numpy(train_mean_image_normalized[mask].astype(np.float32))
            p_masked = float(predict_probs(model, x_masked, args.batch_size, device)[0])
            conf_masked = float(p_masked if y_test[i] == 1 else 1.0 - p_masked)
            y_pred_masked = int(p_masked >= 0.5)
            long_rows.append(
                {
                    "sample_id": sample_id,
                    "roi_index": roi_index,
                    "roi_name": roi_name,
                    "y_true": int(y_test[i]),
                    "p_class1_original": float(p_orig[i]),
                    "p_class1_masked": p_masked,
                    "true_conf_original": float(conf_orig[i]),
                    "true_conf_masked": conf_masked,
                    "evidence_drop": float(conf_orig[i] - conf_masked),
                    "prediction_changed": bool(y_pred_orig[i] != y_pred_masked),
                    "dynamic_roi_pixels": int(dynamic_roi_pixels),
                    "missing_yunet": bool(missing),
                }
            )

    long_df = pd.DataFrame(long_rows)
    summary = (
        long_df.groupby(["roi_index", "roi_name"], as_index=False)
        .agg(
            n=("evidence_drop", "size"),
            mean_evidence_drop=("evidence_drop", "mean"),
            median_evidence_drop=("evidence_drop", "median"),
            std_evidence_drop=("evidence_drop", "std"),
            prediction_change_rate=("prediction_changed", "mean"),
            mean_dynamic_roi_pixels=("dynamic_roi_pixels", "mean"),
            missing_yunet_count=("missing_yunet", "sum"),
        )
        .sort_values("mean_evidence_drop", ascending=False)
    )
    long_df.to_csv(out_dir / "cnn_yunet_roi_occlusion_test.csv", index=False)
    summary.to_csv(out_dir / "cnn_yunet_roi_occlusion_summary_overall.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

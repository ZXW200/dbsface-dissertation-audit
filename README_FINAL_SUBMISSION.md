# Final Submission Package

This clean package contains the final dissertation PDF, LaTeX source, figures, a structured Python reproduction package, model checkpoint, structured outputs, local data file, and a SHA256 manifest.

Final dissertation: `DISSERTATION_FINAL.pdf`

## Included for Reproducibility

- `data/raw/PD_DBS_Data.mat`: local MATLAB data file used in the dissertation.
- `models/baseline_mlp_numpy.npz`: trained NumPy MLP checkpoint.
- `outputs/`: structured outputs for baseline, data QC, calibration, ROI masks, AEV, ROI-size sensitivity, ROI low-level confound baseline, pixel-occlusion comparison, region-only validation, robustness, YuNet sensitivity, compact CNN, and Grad-CAM consistency.
- `main.py`: primary command dispatcher for reproduction.
- `scripts/`: six small stage entry points.
- `src/dbsface/data/`: data loading, inspection, duplicate checks, and fixed ROI construction.
- `src/dbsface/experiments/`: baseline, calibration, fixed-ROI AEV, ROI-size sensitivity, ROI low-level confound baseline, pixel-occlusion, region-only, and YuNet ROI modules.
- `src/dbsface/robustness/`: leakage, multi-seed, perturbation, and sensitivity analyses.
- `src/dbsface/explain/`: YuNet feasibility, compact CNN, and Grad-CAM consistency modules.
- `src/dbsface/draw/`: figure-generation modules.
- `src/dbsface/build/`: document-build helper scripts retained for traceability.
- `latex_project/`: authoritative LaTeX source corresponding to the submitted PDF.
- `latex_project/figures/lu-logo.svg`: Lancaster logo source used in the title-page assets.
- `pyproject.toml`: package metadata for the `src/dbsface` code layout.
- `requirements.txt`: Python dependencies for the included reproduction code.
- `RUN_REPRODUCTION.md`: execution order and build instructions.
- `DATA_AVAILABILITY.md`: data-use and interpretation notes.
- `SHA256_MANIFEST.txt`: hashes for package files.

## Notes

- The LaTeX source is compiled directly from `latex_project/main.tex`.
- The included Python package and structured outputs reproduce the dissertation analyses reported in the final PDF.
- The final wording reports numeric Class 0 vs Class 1 technical results. Clinical DBS-state validation and full Face2Gene diagnostic validation are reserved for datasets with patient-level clinical evidence.

Archived planning/proposal/scope drafts are deliberately not included in this final submission package.

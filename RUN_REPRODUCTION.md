# Dissertation Package and Reproduction Guide

This package accompanies the MSc dissertation:

Region-Level Explainability within the Explainable Face2Gene Diagnosis Project: A Coarse-ROI Occlusion Audit on Low-Resolution PD-DBS Facial Images

Zixu Wang, supervised by Dr Richard Jiang, School of Computing and Communications, Lancaster University.

## What is submitted

`DISSERTATION_FINAL.pdf` is the submitted dissertation: 66 pages, A4, 12 pt. It is the only file uploaded to the DS591 Moodle submission point. Everything else here is supporting material retained for examination, the viva, and reproduction.

`PROJECT_SPECIFICATION.pdf` is the project specification. The same specification is bound into the dissertation as Appendix A.

## Contents

- `latex_project/` — authoritative LaTeX source, final figures and `ref.bib`. Built with XeLaTeX; `main.pdf` here is byte-identical to `DISSERTATION_FINAL.pdf`.
- `data/raw/PD_DBS_Data.mat` — the working data file.
- `models/` — the NumPy MLP checkpoint and the YuNet detector used by the reported analyses.
- `outputs/` — structured outputs behind the reported numbers: baseline, calibration, AEV, confidence matching, low-level comparators, robustness, YuNet and Grad-CAM results.
- `main.py`, `scripts/`, `src/dbsface/` — staged entry points and implementation modules.
- `requirements.txt` — the runtime dependencies, pinned to the versions used for verification.
The cited literature itself is `latex_project/ref.bib`. Copies of the source papers are kept outside this package, on the author's machine only, and are not redistributed.

## Data

The data file is:

```text
data/raw/PD_DBS_Data.mat
```

It contains four arrays: `x_train`, `x_test`, `y_train`, and `y_test`. The project convention maps Class 0 to pre-DBS images and Class 1 to post-DBS images. This convention is used by the dissertation, the scripts and every generated output.

The raw matrix is included so that authorised examiners can verify the computational pipeline. Public redistribution of the facial-image matrix remains subject to institutional governance and data-controller approval. A public-facing release should omit `data/raw/PD_DBS_Data.mat` and retain the analysis code, ROI definitions and non-identifying aggregate outputs. A repository DOI should be added once a deposited record is public and verified.

## Runtime

The experiments were implemented as a structured Python package. The primary entry point is `main.py`, six small stage wrappers are provided under `scripts/`, and the implementation modules are under `src/dbsface/`. The dissertation uses a NumPy MLP and occlusion-based AEV as the main reproducible audit workflow, with ROI-size sensitivity, an ROI low-level baseline, pixel-occlusion comparison, YuNet sensitivity, a compact CNN and Grad-CAM checks added as sensitivity analyses.

Install the dependencies. Versions are pinned to those used for verification on Windows with Python 3.13; `torch` is left unpinned because the verified CUDA build is not distributed on PyPI, and the CPU build runs every analysis here.

```powershell
py -m pip install -r requirements.txt
```

No installation of the package itself is required: `main.py` and the stage wrappers add `src/` to the import path.

## Reproduction order

From the package root, run the staged workflow:

```powershell
py main.py prepare
py main.py baseline --seed 42
py main.py core-audit
py main.py robustness
py main.py advanced-audit
py main.py figures
```

The same stages are also available as readable wrapper scripts:

```powershell
py scripts/01_prepare_data_and_roi.py
py scripts/02_train_baseline.py
py scripts/03_run_core_audit.py
py scripts/04_run_robustness.py
py scripts/05_run_dynamic_roi_and_gradcam.py
py scripts/06_make_figures.py
```

To rerun one implementation module directly:

```powershell
py main.py run train_baseline_mlp_numpy --seed 42
py main.py run run_multiseed_robustness --seeds 0,1,2,3,4
```

The exploratory representation-by-estimator and within-group shuffle analysis is a standalone post-hoc module. Write a reproduction run to a fresh directory, because the script refuses to overwrite non-empty outputs:

```powershell
py src/dbsface/experiments/run_representation_capacity_analysis.py --output-dir outputs/representation_capacity_analysis_reproduction
py scripts/generate_representation_capacity_figure.py --metrics outputs/representation_capacity_analysis_reproduction/per_seed_metrics.csv --output-dir outputs/representation_capacity_analysis_reproduction/figures
```

The submitted analysis outputs remain frozen under `outputs/representation_capacity_analysis_20260718_v2/`. Running `scripts/generate_representation_capacity_figure.py` without arguments reads that frozen metrics file and recreates the dissertation figure bundle.

A full rerun includes 300-epoch baseline training, fixed-ROI AEV, baseline-confidence matching, ROI-size sensitivity, the ROI low-level baseline and adjusted score-increment analysis, pixel-occlusion comparison, five-seed robustness, YuNet dynamic-ROI checks, compact CNN modelling and the Grad-CAM audit. For a fast rerun, use the included checkpoint and structured outputs, then rerun the non-training stages from `core-audit` onward.

## LaTeX build

From `latex_project/`:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
biber main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The build requires XeLaTeX and Biber. The source prefers Times New Roman and falls back to TeX Gyre Termes when that font is unavailable.

## Package maintenance, 2026-08-23

The following material was removed. No source, data, model, output or manuscript file used by the dissertation was altered, and `DISSERTATION_FINAL.pdf` is unchanged.

- `tmp/` — 596 page renders and contact sheets from proofreading passes of superseded builds; re-renderable from the PDFs.
- Six `__pycache__` directories.
- `outputs/representation_capacity_analysis_20260718/` — the first, withdrawn run of the representation analysis. It used a mismatched class weighting for logistic regression and was marked not for interpretation; the dissertation cites the corrected `_v2` outputs only.
- Two checksum listings and two overlapping top-level notes. This file now carries the package description, the data notes and the reproduction instructions that were previously split across three documents.
- A second, version-pinned dependency list, and `pyproject.toml`. The same eight dependencies had been declared in three places. `requirements.txt` now states them once, with the verified versions. `pyproject.toml` described an installable distribution that was never built or installed, and nothing in the package referred to it.
- `reference_pdfs/` — 61 downloaded source papers and their checking tables, moved to local storage outside the package. Publisher PDFs are licensed to the author and should not travel with a submitted or shared package. The bibliography in `latex_project/ref.bib` is unaffected.
- `DISSERTATION_FINAL_BEFORE_20260724.pdf` and `latex_project_BACKUP_before_20260724_edits/` — the build and source preceding the 2026-07-24 wording edits, which tightened phrasing in the abstract, methods, results, literature review and appendix without changing any reported value. The earlier `FINAL_SUBMISSION_SUBMIT_20260702` package, kept separately, remains available as the prior snapshot.

This package now contains exactly one dissertation build and one LaTeX source tree.

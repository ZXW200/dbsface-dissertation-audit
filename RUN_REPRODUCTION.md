# Reproduction Guide

This package accompanies the MSc dissertation:

Toward Explainable Face2Gene Diagnosis: A Coarse-ROI Occlusion Evidence Pipeline for Low-resolution Facial Phenotyping

## Runtime

The experiments were implemented as a structured Python package. The primary entry point is `main.py`, six small stage wrappers are provided under `scripts/`, and the implementation modules are under `src/dbsface/`. The completed dissertation uses a NumPy MLP and occlusion-based AEV as the main reproducible audit workflow, with ROI-size sensitivity, ROI low-level confound baseline, pixel-occlusion comparison, YuNet sensitivity, compact CNN, and Grad-CAM checks added as sensitivity analyses.

Install minimal dependencies:

```powershell
py -m pip install -r requirements.txt
```

## Data

The local data file used for the dissertation is included at:

```text
data/raw/PD_DBS_Data.mat
```

It contains four arrays: `x_train`, `x_test`, `y_train`, and `y_test`. The labels are numeric Class 0 / Class 1, with Class 0 treated as pre-DBS and Class 1 treated as post-DBS label under the project convention. This dissertation uses the image arrays and numeric class labels in the working file. The package reports classification and ROI evidence audit results for the supplied label task.

## Reproduction Order

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

To rerun one implementation module directly, use:

```powershell
py main.py run train_baseline_mlp_numpy --seed 42
py main.py run run_multiseed_robustness --seeds 0,1,2,3,4
```

The full rerun includes 300-epoch baseline training, fixed-ROI AEV, ROI-size sensitivity, ROI low-level confound baseline, pixel-occlusion comparison, five-seed robustness, YuNet dynamic-ROI checks, compact CNN modelling, and Grad-CAM audit. For a fast rerun, use the included checkpoint and structured outputs, then rerun the non-training stages from `core-audit` onward.

## LaTeX Build

To compile the LaTeX dissertation, run from `latex_project/`:

```powershell
cd latex_project
xelatex -interaction=nonstopmode -halt-on-error main.tex
biber main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The LaTeX build requires XeLaTeX and Biber. The generated `main.tex` prefers Times New Roman and SimSun on Windows, but includes fallbacks to TeX Gyre Termes and common TeX Live CJK fonts for non-Windows builds.

## Frozen Results

The submitted PDF uses structured outputs already included under:

```text
outputs/
models/
```

The final dissertation is `DISSERTATION_FINAL.pdf`.

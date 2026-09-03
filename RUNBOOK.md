# RUNBOOK — what still executes in the dissertation package

Scope: `C:\Users\1\Desktop\Lan_Msc\FINAL_SUBMISSION_REVISION_COPY_20260717` (the 17 July
revision) and nothing else. No file from `paper_ieee_20260825` or
`CLEANROOM_MATCHED_SUPPORT_20260710` is referenced here. Every runtime below was
measured on 2026-09-03 with
`C:\Users\1\AppData\Local\Programs\Python\Python313\python.exe` (CPython 3.13.5).

The dissertation tree was treated as read-only. All execution happened in a copy at
`...\scratchpad\defencecode\runnability\sandbox`, built from `main.py`, `src\`,
`scripts\`, plus a copy of `data\`, `models\` and `outputs\`. No file under
`outputs\`, `data\`, `models\` or `latex_project\` in the dissertation tree has been
modified (verified: nothing under those four directories has an mtime later than
2026-08-23).

---

## Headline

**The whole pipeline still runs, and it is fast.** `py main.py all` — 25 modules, from
raw `.mat` through training, AEV, robustness, YuNet, CNN/Grad-CAM and every figure —
completed in **57.6 s** on CPU with return code 0. Nothing is broken, nothing needs a
GPU, nothing needs a download.

**And it reproduces exactly.** Reruns of the frozen analyses produced byte-identical
files, including the model checkpoint:

| Rerun | Compared against | Result |
|---|---|---|
| `train_baseline_mlp_numpy --seed 42` (2.37 s) | `models/baseline_mlp_numpy.npz` | **byte-identical** (282,416 B) |
| same run | `outputs/baseline/predictions_test.csv`, `roc_curve.csv`, `training_curve.csv` | **byte-identical** |
| same run | `outputs/baseline/metrics.json` | identical except the recorded `data_path` string |
| `run_roi_occlusion_mlp` (0.69 s) | `outputs/aev/aev_test.csv`, `roi_occlusion_test.csv`, `roi_occlusion_summary_overall.csv` | **byte-identical** |
| `analyze_aev_class_statistics` (1.86 s) | `outputs/aev/roi_class_comparison.csv`, `..._ranked.csv` | **byte-identical** |
| `analyze_aev_confidence_matched` (2.02 s) | `outputs/aev/confidence_matched/confidence_matched_roi_class_comparison.csv` | **byte-identical** |
| `analyze_aev_size_normalized` (2.65 s) | `outputs/aev/roi_occlusion_size_normalized.csv` | **byte-identical** |
| `evaluate_calibration_numpy` (0.51 s) | `outputs/calibration/reliability_bins.csv` | **byte-identical** |
| `run_near_duplicate_sensitivity` (0.56 s) | `outputs/data_qc/near_duplicate_sensitivity_metrics.csv` | **byte-identical** |
| `build_coarse_roi_masks` (0.65 s) | `latex_project/figures/dissertation/coarse_roi_overlay_examples.png` | **byte-identical** |
| `run_representation_capacity_analysis` (32.8 s) | `outputs/representation_capacity_analysis_20260718_v2/per_seed_metrics.csv` | identical in every column except `runtime_seconds` |
| `scripts/06_make_figures.py` (2.65 s) | 24 files in `latex_project/figures/dissertation/` | 17 byte-identical; 6 differ only in embedded PDF/SVG creation-date metadata (same byte length); 1 genuine difference (see Gap 3) |

An examiner asking "can you show it running?" can be answered in under a second, and
"can you show it gives the same numbers?" in under three.

---

## Classes

| Class | Meaning |
|---|---|
| **A — RUNS ANYWHERE** | Self-contained. No data, no checkpoint, no `outputs/`. |
| **B — NEEDS THE OUTPUTS ONLY** | Reads recorded CSV/JSON under `outputs/`. No image data. Safe to demo on any machine, including one that must not hold the face matrix. |
| **C — NEEDS THE GOVERNANCE-LIMITED IMAGE MATRIX** | Reads `data/raw/PD_DBS_Data.mat`. Cannot run where the matrix is withheld. |
| **D — NEEDS AN EXTERNAL MODEL FILE** | Additionally reads `models/external/face_detection_yunet_2023mar.onnx`. |

Note on C: the NumPy MLP checkpoint `models/baseline_mlp_numpy.npz` and the CNN
checkpoint `outputs/gradcam_occlusion_overlap/small_cnn_gradcam.pt` **are** shipped in
the package, so "needs a checkpoint" is never itself a blocker — only the image matrix is.

---

## Every entry point

### `scripts/` — pipeline wrappers and figure generators

| Script | Class | Inputs | Writes to | Verified runtime |
|---|---|---|---|---|
| `01_prepare_data_and_roi.py` | C | `.mat` | `outputs/roi`, stdout | ~0.8 s (sum of its 2 modules) |
| `02_train_baseline.py` | C | `.mat` | `outputs/baseline`, `models/` | 2.4 s |
| `03_run_core_audit.py` | C | `.mat`, `models/baseline_mlp_numpy.npz`, `outputs/roi`, `outputs/baseline` | `outputs/{data_qc,calibration,aev,xai_baselines,lowlevel_roi_confound,confound_net_increment}` | ~17 s |
| `04_run_robustness.py` | C | `.mat`, checkpoint, `outputs/{roi,baseline,aev,data_qc,external}` | `outputs/{data_qc,robustness,sensitivity,aev/cluster_robust}` | ~64 s |
| `05_run_dynamic_roi_and_gradcam.py` | **D** | `.mat`, ONNX, checkpoint, `outputs/{roi,aev,external,gradcam_occlusion_overlap}` | `outputs/{external,gradcam_occlusion_overlap}`, `latex_project/figures/dissertation` | ~36 s |
| `06_make_figures.py` | C | `.mat` (for the AEV worked-example panel) + most of `outputs/` | `latex_project/figures/dissertation` | **2.65 s** |
| `generate_additional_dissertation_figures.py` | **B** | `outputs/aev/*`, `outputs/aev/confidence_matched/*`, `outputs/lowlevel_roi_confound/*`, `outputs/confound_net_increment/*`, `outputs/sensitivity/roi_supplementary_experiments/*` | `latex_project/figures/dissertation` (hard-coded, no CLI override) | **2.14 s** |
| `generate_method_concept_figures.py` | **A** | none — draws from constants in the file | `latex_project/figures/concept_candidates` (hard-coded) | **1.58 s** |
| `generate_representation_capacity_figure.py` | **B** | `outputs/representation_capacity_analysis_20260718_v2/per_seed_metrics.csv` | `--output-dir` (default `latex_project/figures/dissertation`) | **1.49 s** |

`main.py` itself is a dispatcher (`py main.py list` prints the command map; instant, class A).

### `src/dbsface/data` — loading and ROI construction

| Module | Class | Inputs | Verified runtime |
|---|---|---|---|
| `load_pd_dbs.py` | C | `.mat`. Bespoke MATLAB-5 reader, **not** `scipy.io`. Exposes `main()` that prints array shapes. | <0.2 s |
| `inspect_pd_dbs_mat.py` | C | `.mat`. Pure-stdlib header/variable dump. | **0.16 s** |
| `build_coarse_roi_masks.py` | C | `.mat` (only to draw the QC overlay); ROI geometry is hard-coded in the module | **0.65 s** |
| `check_train_test_duplicates.py` | C | `.mat` | **0.63 s** |
| `__init__.py` | — | not an entry point | — |

### `src/dbsface/experiments` — training, occlusion, AEV statistics

| Module | Class | Inputs | Verified runtime |
|---|---|---|---|
| `train_baseline_mlp_numpy.py` | C | `.mat` | **2.37 s** (300 epochs) |
| `evaluate_calibration_numpy.py` | **B** | `outputs/baseline/predictions_test.csv` | **0.51 s** |
| `run_roi_occlusion_mlp.py` | C | `.mat`, `models/baseline_mlp_numpy.npz`, `outputs/roi/*` | **0.69 s** |
| `analyze_aev_class_statistics.py` | **B** | `outputs/aev/aev_test.csv` | **1.86 s** |
| `analyze_aev_confidence_matched.py` | **B** | `outputs/aev/aev_test.csv` | **2.02 s** |
| `analyze_aev_size_normalized.py` | **B** | `outputs/aev/roi_occlusion_summary_overall.csv`, `outputs/roi/coarse_roi_definitions.csv` | **2.65 s** |
| `analyze_aev_cluster_robust_stats.py` | C | `outputs/aev/aev_test.csv` **and** `.mat` (cosine clustering of images) | **33.4 s** |
| `run_pixel_occlusion_xai_baseline.py` | C | `.mat`, checkpoint, `outputs/roi/*`, `outputs/aev/roi_occlusion_summary_overall.csv`, `outputs/gradcam_occlusion_overlap/gradcam_occlusion_roi_overlap.csv` | **4.90 s** |
| `run_roi_lowlevel_confound_baseline.py` | C | `.mat`, `outputs/roi/*` | **0.72 s** |
| `run_confound_net_increment.py` | C | `.mat`, `outputs/roi/*`, `outputs/baseline/predictions_test.csv`. **No argparse** — all paths hard-coded relative to a `parents[3]` repo root. | **1.97 s** |
| `run_region_only_mlp.py` | C | `.mat`, checkpoint, `outputs/roi/*` | **0.70 s** |
| `run_yunet_region_only_mlp.py` | C | `.mat`, checkpoint, `outputs/external/pd_dbs_yunet_feasibility/yunet_detection_audit.csv`. Uses `cv2` for resizing but **not** the ONNX file. | **2.11 s** |
| `run_yunet_roi_occlusion_mlp.py` | C | `.mat`, checkpoint, YuNet audit CSV, `outputs/aev/roi_occlusion_summary_overall.csv` | **0.91 s** |
| `run_representation_capacity_analysis.py` | C | `.mat`, `outputs/roi/*`. **Standalone** — not reachable through `main.py run` (absent from `MODULE_IMPORTS`); must be invoked by file path. Refuses to write into a non-empty directory. | **32.8 s** |
| `__init__.py` | — | not an entry point | — |

That is 15 files in `experiments/`, matching the brief's count.

### `src/dbsface/explain`

| Module | Class | Inputs | Verified runtime |
|---|---|---|---|
| `audit_pd_dbs_yunet_feasibility.py` | **D** | `.mat`, `models/external/face_detection_yunet_2023mar.onnx` (via `cv2.FaceDetectorYN`) | **9.12 s** (2,343 images, 100 % detection) |
| `run_cnn_gradcam_occlusion_overlap.py` | C | `.mat`, `outputs/roi/*`, `outputs/aev/roi_occlusion_summary_overall.csv`. Trains a small CNN in **torch**, CPU default, 120 epochs. | **15.60 s** |
| `run_cnn_yunet_roi_occlusion.py` | C | `.mat`, `outputs/gradcam_occlusion_overlap/small_cnn_gradcam.pt`, `outputs/external/pd_dbs_yunet_region_only/yunet_roi_boxes.csv`, `outputs/roi/coarse_roi_definitions.csv`. torch. | **6.65 s** |
| `create_gradcam_actual_comparison_figure.py` | C | `.mat` plus six files in `outputs/gradcam_occlusion_overlap/`, `outputs/roi/coarse_roi_masks.npy`, YuNet boxes | **1.73 s** |
| `__init__.py` | — | not an entry point | — |

### `src/dbsface/robustness`

| Module | Class | Inputs | Verified runtime |
|---|---|---|---|
| `run_near_duplicate_sensitivity.py` | **B** | `outputs/baseline/predictions_test.csv`, `outputs/data_qc/near_duplicate_check.csv` | **0.56 s** |
| `run_similarity_alignment_audit.py` | C | `.mat`, checkpoint, `outputs/baseline/predictions_test.csv` | **1.11 s** |
| `run_multiseed_robustness.py` | C | `.mat` (retrains 5 seeds × 300 epochs) | **9.79 s** |
| `run_perturbation_robustness.py` | C | `.mat`, checkpoint, `outputs/roi/*` | **0.95 s** |
| `run_roi_supplementary_experiments.py` | C | `.mat`, checkpoint, `outputs/roi/*`, `outputs/aev/region_only_*`, both YuNet output sets | **18.73 s** |
| `__init__.py` | — | not an entry point | — |

### `src/dbsface/draw`

| Module | Class | Inputs | Verified runtime |
|---|---|---|---|
| `create_dissertation_figures.py` | C | `.mat` (`dual_aev_worked_example`, called unconditionally from `main()` at line 1104) plus ~20 files across `outputs/`. Writes to `latex_project/figures/dissertation` — path is **relative to the working directory**, and `_bootstrap.add_project_paths()` chdirs to the package root, so it always writes inside the package. | **0.71 s** |

### Class tally

37 entry points in total (9 in `scripts/`, 28 modules under `src/dbsface/` excluding the
six `__init__.py` files), plus `main.py` itself. Note that `scripts/01`–`06` are thin
wrappers over modules also counted individually.

- **A (runs anywhere): 1** — `scripts/generate_method_concept_figures.py`. `main.py list`
  also needs nothing.
- **B (outputs only): 7** — `evaluate_calibration_numpy`, `analyze_aev_class_statistics`,
  `analyze_aev_confidence_matched`, `analyze_aev_size_normalized`,
  `run_near_duplicate_sensitivity`, `scripts/generate_additional_dissertation_figures.py`,
  `scripts/generate_representation_capacity_figure.py`
- **C (image matrix): 27** — 5 wrapper scripts, all 4 `data/` modules, 10 of the 14
  `experiments/` modules, 3 of the 4 `explain/` modules, 4 of the 5 `robustness/` modules,
  and `draw/create_dissertation_figures.py`
- **D (image matrix + external ONNX): 2** — `explain/audit_pd_dbs_yunet_feasibility.py`
  and, transitively, `scripts/05_run_dynamic_roi_and_gradcam.py`

The practical reading: **only 8 of 37 entry points can run at all without the governed
image matrix**, and 7 of those 8 are re-analysis of recorded results rather than the audit
itself.

---

## Dependencies actually imported

Third-party imports found across `main.py`, `scripts/` and `src/` — nothing else:

| Package | Installed here | Pinned in `requirements.txt` | Used by |
|---|---|---|---|
| `numpy` | 2.2.6 | 2.2.6 | 31 files — everything |
| `pandas` | 2.3.3 | 2.3.3 | 27 files |
| `matplotlib` | 3.10.7 | 3.10.7 | 4 files (all figure generators; `Agg` backend forced) |
| `PIL` (Pillow) | 12.2.0 | 12.2.0 | 7 files — all raster figure drawing is PIL, not matplotlib |
| `opencv-python` (`cv2`) | 4.13.0 | 4.13.0.92 | 2 files — `audit_pd_dbs_yunet_feasibility`, `run_yunet_region_only_mlp` |
| `scipy` | 1.15.3 | 1.15.3 | 1 file — `analyze_aev_cluster_robust_stats` (`sparse`, `csgraph.connected_components`, `stats.norm`) |
| `scikit-learn` | 1.7.2 | 1.7.2 | 2 files — `run_confound_net_increment`, `run_representation_capacity_analysis` |
| `torch` | 2.9.1+cu128 | unpinned (CPU build sufficient) | 2 files — `run_cnn_gradcam_occlusion_overlap`, `run_cnn_yunet_roi_occlusion` |

Everything else is stdlib: `argparse`, `csv`, `json`, `pathlib`, `hashlib`, `struct`,
`zlib`, `itertools`, `collections`, `math`, `random`, `time`, `importlib`, `os`, `sys`.

**Nothing unusual.** All eight are mainstream, all eight are already installed on this
machine at (or within a patch of) the pinned versions, and the environment as it stands
runs every module. Two points worth being able to say out loud:

1. **There is no `scipy.io.loadmat`.** `data/load_pd_dbs.py` reuses a hand-written
   MATLAB-5 parser in `inspect_pd_dbs_mat.py` (`struct` + `zlib`, ~160 lines) to read
   the `.mat`. That is a deliberate choice, it is the only reader in the package, and it
   is why `inspect_pd_dbs_mat` is a pure-stdlib module.
2. **`torch` is the only heavy dependency, and only two modules need it** — both in the
   Grad-CAM sensitivity analysis. The main audit workflow (NumPy MLP + occlusion AEV) has
   no deep-learning framework in it at all.

---

## Live demo — verified commands

All three were run exactly as written. Setup used for verification: the package tree
copied to a sandbox and executed there; on the candidate's own machine the same commands
run from the package root with `outputs/` in place.

### Demo 1 — the core measurement, from raw pixels (0.69 s) · class C

Recomputes the entire fixed-ROI Absolute Evidence Value table: 1,171 test images × 8 ROIs
occluded, evidence drop measured against the frozen checkpoint.

```powershell
py main.py run run_roi_occlusion_mlp
```

Real first lines of output:

```text
 roi_index              roi_name    n  mean_evidence_drop  median_evidence_drop  prediction_change_rate
         1   upper_brow_forehead 1171            0.049135              0.002061                0.062340
         6 right_cheek_zygomatic 1171            0.044752              0.002116                0.065756
         5  left_cheek_zygomatic 1171            0.040594              0.001107                0.059778
         8         chin_mandible 1171            0.031842              0.000550                0.048676
```

What it shows: the ranking that Chapter 5 reports, regenerated live in under a second,
and the resulting `aev_test.csv` / `roi_occlusion_test.csv` /
`roi_occlusion_summary_overall.csv` are **byte-identical** to the submitted files. This
is the strongest single demonstration available — it closes the loop from the raw matrix
to the headline table. It needs the governance-limited `.mat`.

### Demo 2 — the statistics, with no image data on the machine (1.86 s) · class B

Recomputes the per-ROI class contrast: Cohen's d, Cliff's delta, permutation p, bootstrap
CI, FDR — from the recorded AEV table alone.

```powershell
py main.py run analyze_aev_class_statistics
```

Real first lines of output:

```text
                  roi  n_class0  n_class1  mean_class0  mean_class1  ...  cohen_d  cliffs_delta   p_perm  bootstrap_ci_low  bootstrap_ci_high    p_fdr
        nasal_midface       675       496     0.011795     0.044561  ... 0.276945      0.188471 0.000200          0.018882           0.047972 0.000533
right_cheek_zygomatic       675       496     0.071083     0.008918  ... -0.378600    -0.294316 0.000200         -0.080487          -0.044042 0.000533
        chin_mandible       675       496     0.059157    -0.005331  ... -0.479229    -0.339701 0.000200         -0.079513          -0.049566 0.000533
```

(Columns elided above for width; the real run prints all 14.) Output is byte-identical to
`outputs/aev/roi_class_comparison.csv`. **This is the demo to use if the face matrix must
stay closed** — it touches only a recorded CSV, so it can be run on an examiner's laptop,
in a seminar room, or on a screen share, with no governed data present.

### Demo 3 — retrain the baseline from scratch and land on the same checkpoint (2.37 s) · class C

```powershell
py main.py run train_baseline_mlp_numpy --seed 42 --output-dir <scratch>/baseline --model-dir <scratch>/models
```

Real first lines of output:

```text
{
  "n": 1171,
  "class_counts": { "0": 675, "1": 496 },
  "majority_baseline_accuracy": 0.5764304013663536,
  "accuracy": 0.9521776259607173,
  "balanced_accuracy": 0.9472909199522103,
```

What it shows: 300 epochs of the NumPy MLP in 2.4 s, and the written
`baseline_mlp_numpy.npz` is **byte-for-byte the same file** as the one shipped in
`models/` — verified by full-file comparison and by max-absolute-difference 0.0 on all
nine arrays. Determinism is not being asserted, it is being demonstrated. Redirect the
output paths as shown so the demo does not touch the submitted package.

### Optional fourth — zero-input figure (1.58 s) · class A

```powershell
py scripts/generate_method_concept_figures.py
```

Prints `Wrote concept figures to ...\latex_project\figures\concept_candidates`, producing
three SVG/PDF/PNG triplets. The only script in the package that needs nothing at all: no
data, no checkpoint, no `outputs/`. Useful only as a "the environment is alive" opener;
it demonstrates nothing about the analysis. **It writes into
`latex_project/figures/concept_candidates` and has no CLI override**, so run it from a
copy, not from the submitted tree.

### Do not pick for a live demo

- `analyze_aev_cluster_robust_stats` (33.4 s) and `run_representation_capacity_analysis`
  (32.8 s) — too slow to hold a room, and the latter refuses to write into a non-empty
  directory, which will look like a failure if the target already exists.
- `run_roi_supplementary_experiments` (18.7 s) and `run_cnn_gradcam_occlusion_overlap`
  (15.6 s) — both fine, both too long.
- `audit_pd_dbs_yunet_feasibility` (9.1 s) — the only module that exercises
  `models/external`; worth naming, but slow for a live run and it is a sensitivity
  analysis, not the main result.

---

## Gaps and things that will be asked

**Gap 1 — `py main.py all` is not the union of the stage commands.** `COMMANDS["all"]`
contains 25 modules; the six stage commands together contain 26. The one missing from
`all` is `analyze_aev_cluster_robust_stats` — the cluster-robust AEV statistics that
produce `outputs/aev/cluster_robust/`. Verified programmatically. So `py main.py all`
alone does **not** regenerate the whole `outputs/` tree; the documented six-stage order in
`RUN_REPRODUCTION.md` does. Say "run the six stages", not "run all".

**Gap 2 — two output artefacts are consumed by the code but absent from the shipped
`outputs/`.** `create_dissertation_figures` looks for
`outputs/external/pd_dbs_yunet_region_only/qc_yunet_roi_examples.jpg`, and
`build_coarse_roi_masks` writes `coarse_roi_overlay_examples.png` into `outputs/roi/`.
Neither file is present in the package. Both are regenerated by rerunning their producer
(`run_yunet_region_only_mlp` and `build_coarse_roi_masks` respectively — confirmed by
running both), so this is a pruning artefact and an ordering dependency, not a broken
link. Consequence: `06_make_figures.py` must run *after* `05_run_dynamic_roi_and_gradcam.py`
in a clean rerun, or one figure panel silently degrades.

**Gap 3 — one committed figure does not reproduce byte-for-byte.**
`latex_project/figures/dissertation/fig_yunet_dynamic_roi_sensitivity.png`
(cited at `chapters/results.tex:247`). Regenerated: 297,862 B; committed: 183,095 B; same
1700×1100 canvas. Pixel comparison localises the difference to a rectangle spanning
y 175–701, x 70–554 — exactly the four example-image tiles of Panel A, which are cropped
out of `qc_yunet_roi_examples.jpg`. **11.0 % of pixels differ, all of them inside Panel A;
every quantitative panel of that figure is identical.** The honest statement is: the
example crops shown in Panel A came from a QC contact sheet that is not in the package,
and regenerating that sheet selects different example images. No reported number is
affected. Six further figures differ only in embedded PDF/SVG creation-date metadata
(identical byte lengths) and are not a real difference.

**Gap 4 — `run_confound_net_increment` has no CLI arguments.** Every path is hard-coded
relative to `Path(__file__).resolve().parents[3]`. It cannot be pointed at a different
data file or a different output directory without editing the module. Every other
experiment module accepts `--data` / `--output-dir`.

**Gap 5 — figure output paths are not overridable.**
`generate_additional_dissertation_figures.py`, `generate_method_concept_figures.py` and
`create_dissertation_figures.py` all write into `latex_project/figures/...` with no CLI
option. Combined with `_bootstrap.add_project_paths()` calling `os.chdir(package_root)`,
any figure command run inside the submitted tree writes into the submitted tree. Demo from
a copy.

**Gap 6 — `run_representation_capacity_analysis` is not reachable through the
dispatcher.** It is absent from `MODULE_IMPORTS`, so `py main.py run
run_representation_capacity_analysis` would fall through to `dbsface.<name>` and fail. It
must be invoked by file path, as `RUN_REPRODUCTION.md` in fact instructs. Its rerun matches
the frozen `_v2` outputs in every column except `runtime_seconds`.

**Observation, not a gap.** During this session one bytecode cache file appeared in the
dissertation tree: `scripts/__pycache__/generate_method_concept_figures.cpython-313.pyc`
(2026-09-03 16:27:54). Every command reported here was executed against the sandbox copy,
never the dissertation tree, so this was not written by this work — most likely a sibling
process. It is a `.pyc` only; no source, data, model, output or LaTeX file in the tree has
an mtime later than 2026-08-23. Two other `__pycache__` directories in the tree pre-date
this session (2026-09-01 and 2026-09-02).

**What could not be established.** Whether the committed
`fig_yunet_dynamic_roi_sensitivity.png` Panel A used a different sample-selection rule or
simply a differently-seeded contact sheet cannot be determined, because the source JPG is
not in the package and no run metadata for it survives. Stating which of the two it was
would be a guess.

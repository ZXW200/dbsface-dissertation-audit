# Coarse-ROI occlusion audit for low-resolution facial classification

Code for the MSc dissertation *Region-Level Explainability within the Explainable
Face2Gene Diagnosis Project: A Coarse-ROI Occlusion Audit on Low-Resolution
PD-DBS Facial Images*, Lancaster University, 2026.

Zixu Wang, supervised by Dr Richard Jiang, School of Computing and
Communications.

## What this does

A classifier is trained on 32x32 grayscale facial images labelled with a binary
pre- versus post-deep-brain-stimulation state. Eight named facial regions are
defined as a fixed rectangular atlas. Each region is masked in turn, the change
in the model's confidence in the *true* class is recorded per image, and those
eight numbers form an anatomical evidence vector. The audit then asks whether the
per-region differences between classes survive multiplicity correction,
confidence matching, dependence-aware re-estimation, and changes to the fill
value and to the atlas geometry.

The finding is that they largely do not survive the reparameterisations: the
regional ranking moves when the fill value or the region geometry changes, so the
ranking is a property of those choices rather than of the model.

## Layout

```
main.py                 entry point; `python main.py all` runs the pipeline
scripts/                numbered stages 01-06, plus the figure generators
src/dbsface/
    data/               loading, ROI atlas construction, duplicate checks
    experiments/        training, occlusion, AEV statistics
    explain/            Grad-CAM and pixel-attribution comparisons
    robustness/         perturbation, multi-seed and similarity audits
    draw/               dissertation figure generation
models/                 the trained baseline checkpoint
outputs/                written by the pipeline; empty in this repository
```

## Running it

```
pip install -r requirements.txt
python main.py prepare
python main.py baseline
python main.py core-audit
python main.py advanced-audit
python main.py robustness
python main.py figures
```

About 100 seconds end to end on a laptop, writing 137 files into `outputs/`.

Two things to know before running anything else.

**Run the stages in the order above, not the order the numbered scripts in
`scripts/` suggest.** `robustness` reads a table that `advanced-audit` writes, so
running `04_run_robustness.py` before `05_run_dynamic_roi_and_gradcam.py` fails on
a clean checkout. The numbered scripts are kept because they are what the
dissertation describes; the order above is what works from empty.

**`python main.py all` does not complete from a clean checkout**, for the same
reason: its module list has the consumer before the producer. It succeeds only
when `outputs/` is already populated.

`outputs/` ships almost empty. The one exception is
`outputs/external/pd_dbs_yunet_region_only/yunet_vs_fixed_roi_comparison.csv`, an
eight-row region-level comparison table that the figure stage reads and that no
script in this tree regenerates. It is committed so that `figures` can run.

## Data

`data/raw/PD_DBS_Data.mat` is the working dataset: 2,343 grayscale 32x32 facial
crops with a binary label, split into a training pool and a frozen test split of
1,171 images. Class 0 is the pre-stimulation condition and Class 1 the supplied
post-stimulation condition. No participant identifiers are distributed with it,
so every estimate in this work is an image-level estimate.

The images were collected by the source study, which drew them from public online
video of people with Parkinson's disease:

> R. Jiang, P. Chazot, N. Pavese, D. Crookes, A. Bouridane and M. E. Celebi,
> "Private facial prediagnosis as an edge service for Parkinson's DBS treatment
> valuation," *IEEE Journal of Biomedical and Health Informatics*, 26(6),
> 2703-2713, 2022.

Use of the data in this dissertation was covered by project ethics approval at
Lancaster University. No clinical, diagnostic or prognostic claim is made
anywhere in this work; the labels are treated as a supervised target. See
`DATA_AVAILABILITY.md` for the governance context.

`models/external/face_detection_yunet_2023mar.onnx` is *not* included. It is the
released YuNet model from the OpenCV Zoo and should be downloaded from there if
the detector-derived atlas comparison is to be re-run.

## Reproducibility

`RUN_REPRODUCTION.md` records the package structure, the stage contract and the
one withdrawn run (the representation-capacity analysis was re-run under a
corrected class weighting; the figure scripts read the corrected directory).

Every stage takes an explicit `--seed`, defaulting to 42. Given the same input
matrix, reruns reproduce the recorded outputs exactly, including the model
checkpoint.

## Note on comments

This copy carries explanatory comments on the load-bearing steps that the version
submitted for assessment does not. Only comments differ; no behaviour was
changed.

## Licence

Code under the MIT licence. Documentation and aggregate outputs under CC BY 4.0.
The image data is not covered by either and is not distributed here.

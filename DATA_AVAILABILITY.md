# Data Availability and Use Notes

The final submission package includes the local MATLAB data file used for the dissertation so that examiners can verify the computational pipeline:

```text
data/raw/PD_DBS_Data.mat
```

The dissertation treats this file as the working dataset. It contains image arrays and numeric Class 0 / Class 1 labels. The project convention is Class 0 = pre-DBS and Class 1 = post-DBS label. The package reports classification, calibration, robustness, and ROI evidence audit results for the supplied label task.

Before sharing the package outside the assessment route, confirm that redistribution of the raw facial data is permitted. If redistribution is not permitted, remove `data/raw/PD_DBS_Data.mat` before external sharing and retain the SHA256 manifest plus structured outputs for traceability.

# 3V Feature Selection & Visualization

Minimal utilities to build Parquet datasets from peak lists, run iterative L1‑SVM
feature selection, and visualize selected features over time.

## Contents
- `build_parquets_from_peaklists.py` — parse peak list CSVs and write star‑schema Parquet
  (`samples.parquet`, `features.parquet`, `abundances.parquet`).
- `query_and_preprocess.py` — helpers to query Parquet into ML‑ready matrices and run
  preprocessing (impute → log2(+ε) → scale). Includes optional Laplacian‑score shortlist.
- `L1SVM_selector.py` — iterative L1‑SVM (prox‑gradient, squared hinge) with 5‑fold CV;
  records selected features and mean CV accuracy to `results/selected_features.csv`.
  Supports **adaptive L1 penalties** (Option 2).
- `selected_feature_churn.py` — summarize per‑time feature churn (same/new/dropped) and
  save stacked‑bar plots.
- `PCA_vis.py` — PCA visualizations (single figure with subplots):
  - Intersection of feature sets across times.
  - Seed‑time feature set projected across all times.

## Requirements
Python 3.9+ and the following packages:
```
numpy
pandas
scikit-learn
matplotlib
pyarrow   # or fastparquet
```
(Optionally `scipy` if you use extras from your pipeline.)

## Results & Reproducibility
- Feature selections are appended to `results/selected_features.csv` with columns:
  `modality, polarity, condition, time_point, feature_id, iteration, mean_cv_accuracy`.
- Churn summaries and PCA figures are written under `results/` (ignored by git).


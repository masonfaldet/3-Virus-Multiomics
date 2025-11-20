# 3-Virus Multi-omics Feature Selection

This repository contains analysis code for a multi-omics time-course study of three viruses (CHIKV, DENV, ZIKV) compared with mock-infected controls.
All code is written in Python and organised so that each analysis has a single driver script with an editable configuration block.

---

## Repository layout

- `utilities/`
  - Tools to build and query a Parquet-based database of samples, features, and abundances.
  - Typical workflow:
    - Convert peak-list CSV files into a star-schema set of Parquet files (`samples.parquet`, `features.parquet`, `abundances.parquet`).
    - Query those Parquet files into analysis-ready matrices and apply standard preprocessing (imputation, log2 transform, scaling, groupwise missingness filters).
- `SVM_feature_selection/`
  - Iterative L1-penalised SVM feature selection comparing each virus against MOCK at each time point.
  - Produces:
    - Tables of selected features with cross-validated performance and per-feature weights.
    - “Churn” summaries showing which features are gained/lost across time.
    - PCA plots projecting samples onto selected features.
- `functional_selection/`
  - “Functional” feature selection based on time-course curves.
  - For each feature, constructs median abundance curves over time for MOCK and for each virus, then:
    - Computes a weighted L2 distance between virus and MOCK curves.
    - Estimates permutation p-values and applies FDR control.
    - Plots the highest-ranked features with per-sample scatter points and median curves.

---

## Data organisation

Analyses assume a Parquet “star schema” created from the MS peak-list files:

- `samples.parquet` – one row per sample (metadata such as virus, time point, modality, polarity).
- `features.parquet` – one row per feature (feature ID and basic annotations).
- `abundances.parquet` – long-format table linking samples and features to measured signal intensities.

These files are not tracked in the repository. They can be built using the scripts in `utilities/` (see inline comments in `peak_to_parquets_v1.py` for the expected input format).

By default, analysis drivers expect these Parquet files to live in a top-level `parquets/` directory.

---

## Main analysis entry points

### 1. L1-SVM feature selection

- Location: `SVM_feature_selection/driver.py`
- Purpose:
  - For each combination of (modality, polarity, virus, time point), fit an L1-penalised SVM to distinguish virus vs MOCK.
  - Record which features are selected, their mean weight across cross-validation folds, and the number of folds in which each feature appears.
  - Summarise feature churn over time and generate PCA plots on selected features.
- How it is structured:
  - The top of `driver.py` defines an `L1SVMSelectorConfig` block where all analysis choices can be edited:
    - Which modalities, polarities, viruses, and time points to include.
    - Missingness filters and preprocessing choices.
    - Number of cross-validation folds and SVM hyperparameters.
  - Running the script writes:
    - A timestamped CSV of selected features to `SVM_feature_selection/results/out_csvs/`.
    - A JSON “sidecar” file capturing the configuration used.
    - Churn summary tables and bar plots under `SVM_feature_selection/results/out_plots/`.
    - PCA plots showing separation of virus and MOCK samples projected onto selected features.

### 2. Functional time-course feature selection

- Location: `functional_selection/driver.py`
- Purpose:
  - Treat each feature as a function of time for MOCK and for each virus.
  - Compute a weighted L2 distance between virus and MOCK median curves and assess significance via permutations.
  - Apply false discovery rate (FDR) control and visualise the most separated features.
- How it is structured:
  - The top of `driver.py` defines a `FunctionalSelectionConfig` block:
    - Which modalities, polarities, viruses, and time points to use.
    - Missingness filters and whether to centre/scale the curves (to emphasise shape vs level).
    - Number of permutations and random seed.
    - Output filename pattern.
  - Block 1 runs the functional selection and writes:
    - A timestamped CSV of feature-level statistics to `functional_selection/results/out_csvs/`, with columns including:
      - `feature_id`, `modality`, `polarity`, `virus`, weighted L2 distance, p-value, and NaN counts.
    - A matching JSON sidecar with the configuration.
  - Block 2 applies FDR filtering and writes filtered tables.
  - Block 3 plots the top-ranked features per (virus, modality, polarity) to `functional_selection/results/out_plots/`.

---

## Software requirements

The code was developed with:

- Python 3.9 or later
- Recommended packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `pyarrow` (or `fastparquet`) for Parquet I/O
  - `scipy` for some statistical utilities

No environment file is included. A typical setup is to create a virtual environment and install these packages using `pip` or `conda`.

---

## Typical use

1. Build the Parquet database from peak-list files using the scripts in `utilities/` (once per dataset).
2. Run `SVM_feature_selection/driver.py` to obtain discriminative feature sets and visualisations for virus vs MOCK at each time point.
3. Run `functional_selection/driver.py` to obtain functionally selected features based on time-course behaviour and to visualise the highest-ranked candidates.

Both drivers are structured so that collaborators can adjust biological questions (which viruses, time points, or modalities to include) by editing a small configuration block at the top of each script, without having to modify the analysis code itself.

import os
import datetime
import json
from dataclasses import dataclass, asdict
import pandas as pd
from functional_selection_v1 import (
    FunctionalSelectionConfig,
    run_functional_selection
)
from feature_plots import plot_top_k_features
from utilities.query_and_preprocess_v1 import fdr_filter_functional_results


"""
================================================================================
Functional selection driver: overview
================================================================================

This script coordinates a two-stage “functional selection” pipeline:

  1. Functional time-course feature scoring (Block 1)
     - For each (modality, polarity, virus) combination, computes weighted
       L2 distances between MOCK and virus median time-course curves.
     - Estimates permutation-based p-values per feature.
     - Writes a timestamped CSV of feature-level statistics and a JSON sidecar
       capturing the configuration used.

  2. FDR-based feature filtering (Block 2)
     - Loads the functional-selection results from Block 1.
     - Applies Benjamini–Hochberg FDR control within each
       (virus, modality, polarity) block at a user-specified threshold.
     - Writes a filtered CSV containing only FDR-significant features
       for the requested subsets of viruses/modalities/polarities.

Typical usage:
  - Edit the `config` object in Block 1 to run a new functional selection.
  - In Block 2, set `csv_name` to the timestamped basename from Block 1 and
    choose the FDR level and subsets (viruses, modalities, polarities).
"""

"""
*******************************************************************************
BLOCK 1: Functional time-course feature scoring
*******************************************************************************
Configure and run the functional selection procedure over all specified
(modality, polarity, virus) combinations.

User-facing configuration:
  - `modalities`, `polarities`, `viruses`, `mock_label`, `time_points`
  - Groupwise missingness filtering (`group_cols`, `min_prop`, etc.)
  - Curve handling (`center_curves`, `scale_curves`)
  - Permutation test settings (`n_permutations`, `random_state`)
  - Output pattern for the result table

Outputs:
  - Timestamped CSV: functional_selection_YYYYMMDD_HHMMSS.csv
  - JSON sidecar:   ._functional_selection_YYYYMMDD_HHMMSS.json
"""
# # Edit this config block to control the functional selection run.
# config = FunctionalSelectionConfig(
#     root="../parquets",
#     modalities=("lipid", "metabolite"),
#     polarities=("+", "-"),
#     viruses=("CHIKV", "DENV", "ZIKV"),
#     mock_label="MOCK",
#     time_points=(0, 5, 7, 14, 21),
#
#     group_cols=("condition", "time_point"),
#     min_prop=0.65,
#     min_group_n=1,
#     require_all_groups=False,
#
#     center_curves=True,
#     scale_curves=True,
#
#     n_permutations=1000,
#     random_state=42,
#
#     log_fallback_eps=1e-3,
#
#     output_pattern="results/out_csvs/functional_selection.csv",
# )
#
# # NOTE: No need to touch the code in the rest of this block
#
# # Build timestamped output path
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# base_dir, base_name = os.path.split(config.output_pattern)
# stem, ext = os.path.splitext(base_name)
# if not ext:
#     ext = ".csv"
# dated_name = f"{stem}_{timestamp}{ext}"
# output_csv = os.path.join(base_dir, dated_name) if base_dir else dated_name
#
# # Ensure output directory exists
# if base_dir:
#     os.makedirs(base_dir, exist_ok=True)
#
# # Run analysis
# res = run_functional_selection(config)
# res.to_csv(output_csv, index=False)
#
# # Write sidecar JSON with config
# csv_filename = os.path.basename(output_csv)
# sidecar_stem, _ = os.path.splitext(csv_filename)
# sidecar_name = f"._{sidecar_stem}.json"
# sidecar_path = os.path.join(base_dir, sidecar_name) if base_dir else sidecar_name
#
# # Convert config to JSON-serializable dict (e.g. tuples -> lists)
# cfg_dict = asdict(config)
#
# with open(sidecar_path, "w") as f:
#     json.dump(cfg_dict, f, indent=2)
#
# print(f"Saved: {output_csv} ({len(res)} rows)")
# print(f"Wrote sidecar: {sidecar_path}")


"""
*******************************************************************************
BLOCK 2: FDR-based filtering of functional features
*******************************************************************************
Apply Benjamini–Hochberg FDR control to the functional selection results.

User-facing settings:
  - `csv_name`:
      Basename (without .csv) of the functional-selection run produced in
      Block 1 (e.g. "functional_selection_YYYYMMDD_HHMMSS").
  - `q_threshold`:
      Target FDR level (e.g. 0.05 or 0.1).
  - `modalities`, `polarities`, `viruses`:
      Subsets indicating which (virus, modality, polarity) blocks to include.

Behavior:
  - Restricts to rows matching the requested subsets.
  - Within each (virus, modality, polarity) block, applies BH-FDR at level
    `q_threshold`.
  - Writes a filtered CSV containing only FDR-significant features.
"""

# # Name of the functional-selection run (basename without ".csv")
# csv_name = "functional_selection_20251119_082149"
#
# # BH-FDR threshold
# q_threshold = 0.05
#
# # Subsets of interest
# modalities = ["metabolite", "lipid"]
# polarities = ["+", "-"]
# viruses = ["CHIKV", "DENV", "ZIKV"]
#
#
# # NOTE: No further edits are typically required below this line for Block 2.
# base_dir = "results/out_csvs"
# feat_df = pd.read_csv(os.path.join(base_dir, csv_name) + ".csv")
# filt_df = fdr_filter_functional_results(
#     df = feat_df,
#     viruses = viruses,
#     modalities = modalities,
#     polarities = polarities,
#     q = q_threshold,
# )
#
# filt_name = f"{csv_name}__q_{q_threshold}"
# filt_df.to_csv(os.path.join(base_dir, filt_name) + ".csv", index=False)
"""
*******************************************************************************
BLOCK 3: Plot top-k functional features
*******************************************************************************
Visualize the most functionally separated features per (virus, modality, polarity)
using the weighted L2 distances from functional selection.

User-facing settings:
  - `csv_name`:
      Basename (without ".csv") of the functional-selection run or an
      FDR-filtered version produced in Block 2.
  - `k`:
      Number of top-ranked features (by weighted_l2) to plot per block.
  - `modalities`, `polarities`, `viruses`:
      Subsets specifying which (virus, modality, polarity) groups to visualize.
  - `max_nans`: 
      Most a feature can be missing to be considered when ranking

Outputs:
  - For each (virus, modality, polarity) block with available features, up to
    `k` PNG plots written by `plot_feature`, one per feature:
        results/out_plots/{virus}_{feature_id}.png
"""

# Name of the functional-selection run or a filtered version (basename without ".csv")
csv_name = "functional_selection_20251118_232555__q_0.1"

# Number of top features to visualize per (virus, modality, polarity)
k = 5

# Most a feature can be missing to be considered for top-k
max_nans = 25

# Subsets of interest
modalities = ["metabolite", "lipid"]
polarities = ["+", "-"]
viruses = ["CHIKV", "DENV", "ZIKV"]

# NOTE: No further edits are typically required below this line for Block 3.

sf_df = pd.read_csv(f"results/out_csvs/{csv_name}.csv")

for virus in viruses:
    for modality in modalities:
        for polarity in polarities:
            print(
                f"[BLOCK 3] Plotting top {k} features for "
                f"virus={virus}, modality={modality}, polarity={polarity}"
            )
            plot_top_k_features(
                root="../parquets",
                sub_dir = csv_name,
                sf_df=sf_df,
                virus=virus,
                modality=modality,
                polarity=polarity,
                k=k,
                max_nan=max_nans,
            )

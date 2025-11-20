import os
import json
from datetime import datetime
from feature_selector_v2 import L1SVMSelectorConfig, run_iteration_block
from churn_v1 import  load_selections, build_summary, plot_group_bars
from pca_v1 import pca_seed, pca_intersection



"""
================================================================================
Driver script overview
================================================================================

  1. L1–SVM feature selection (Block 1)
     - Runs iterative L1–SVM–based feature selection across
       (modality, polarity, condition, time_point).
     - Writes a timestamped CSV of selected features and a JSON sidecar
       capturing the configuration.

  2. Feature churn summarization and visualization (Block 2)
     - Loads a selected–features CSV.
     - Builds a churn summary (how often features recur across settings).
     - Writes a summary CSV and bar plots.

  3. PCA projections on selected features (Block 3)
     - For each (modality, polarity, condition), projects data onto selected
       feature sets.
     - Produces intersection–based and seed–based PCA plots to visualize
       temporal separation between conditions.

Typical usage:
  - Comment out block 2 & 3
  - Edit the configuration object in Block 1 to run a new selection.
  - Uncomment block 2 & 3, comment block 1
  - Update the `csv_name` variables in Blocks 2 and 3 to point to the
    desired selection run.
"""

"""
*******************************************************************************
BLOCK 1: L1–SVM feature selection
*******************************************************************************
Configure and run iterative L1–SVM feature selection for each
(modality, polarity, condition, time_point) combination.

You typically only need to modify the `config` object below to:
  - Choose modalities, polarities, and conditions.
  - Adjust preprocessing, missingness filtering, and model hyperparameters.
  - Control CV behavior and the output location.

A timestamped CSV of selected features and a matching JSON sidecar
(recording the configuration) are written to disk.
"""

config = L1SVMSelectorConfig(
    # IO / loop settings
    root="../parquets",                    # directory where parquets live
    modalities=("lipid", "metabolite"),    # modalities
    polarities=("+", "-"),                 # polarity
    conditions=("CHIKV", "DENV", "ZIKV"),  # conditions
    time_points=(0, 5, 7, 14, 21),         # time points
    output_csv="results/out_csvs/selected_features.csv",  # base pattern for results

    # Accuracy threshold to keep iterating within a (modality,polarity,condition,time_point) block
    acc_threshold=0.75,

    # L1ProxSVM knobs
    lambda_=0.2,         # sparsity penalty (larger = fewer features less discriminatory, smaller = more features more discriminatory)
    step_size=None,      # leave None to use data driven heuristic
    delta_tol=1e-6,      # convergence criteria
    max_iter=20000,      # max iterations in training the feature selector
    fit_intercept=True,  # keep true to allow hyperplane to not go through the origin

    # CV / selection behaviour
    n_folds=5,           # number of folds in a single iteration
    cv_random_state=42,  # random state for reproducibility
    weight_tol=1e-8,     # threshold for "non-zero" weights

    # Groupwise missingness filter
    filter_groupwise_missingness=True,  # group filter
    min_prop=0.65,                      # what percentage must a feature be present within a group to be kept
    min_group_n=1,                      # min groups in filtering
    require_all_groups=False,           # False means min_prop in at least one group, True means min_prop in all groups

    # Preprocessing
    imputer="two_step_label_agnostic",  # imputation technique
    scale="standard",                   # standard means center data (optimal for fitting SVMs)
)

# NOTE: No changes are typically required below this line for Block 1.


# Build a timestamped output filename based on config.output_csv
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir, base_name = os.path.split(config.output_csv)
stem, ext = os.path.splitext(base_name)
dated_name = f"{stem}_{timestamp}{ext}"

# Resolve final CSV path and ensure directory exists
if base_dir:
    os.makedirs(base_dir, exist_ok=True)
    output_csv_path = os.path.join(base_dir, dated_name)
else:
    output_csv_path = dated_name

# Update config to hold the resolved CSV path
config.output_csv = output_csv_path

# Run selector and write CSV
res = run_iteration_block(config)
res.to_csv(config.output_csv, index=False)

# Build sidecar path: "._{filename}.json" where {filename} is the CSV basename
csv_filename = os.path.basename(config.output_csv)
sidecar_name = f"._{csv_filename[:-4]}.json"
sidecar_path = os.path.join(os.path.dirname(config.output_csv), sidecar_name) if base_dir else sidecar_name

# Convert config to a JSON-serializable dict (e.g. tuples -> lists)
cfg_dict = {}
for k, v in config.__dict__.items():
    if isinstance(v, tuple):
        cfg_dict[k] = list(v)
    else:
        cfg_dict[k] = v

with open(sidecar_path, "w") as f:
    json.dump(cfg_dict, f, indent=2)

print(f"Saved: {config.output_csv} ({len(res)} rows)")
print(f"Wrote sidecar: {sidecar_path}")



"""
*******************************************************************************
BLOCK 2: Feature churn summary and bar plots
*******************************************************************************
Summarize and visualize feature “churn” (re-use/recurrence) across
(modality, polarity, condition, time_point) based on a selected–features CSV.

User-facing settings:
  - Set `csv_name` to the timestamped basename (without .csv) of the
    selection run produced in Block 1.

Outputs:
  - A churn summary CSV whose name encodes the source selection run.
  - Grouped bar plots illustrating churn patterns, written to a dedicated
    subdirectory under results/out_plots/.
"""
#
# # Set this to the basename (without .csv) of the selected-features file from Block 1.
# csv_name = "selected_features_20251118_211050"
#
#
# # NOTE: No need to touch the rest of the code in this block
# csv_dir = "results/out_csvs/"
# csv = csv_dir + csv_name + ".csv"
#
# # Results folder
# outdir = "results"
# os.makedirs(outdir, exist_ok=True)
#
# # Plots written to subdirectory named after time of feature selection run
# plots_dir = os.path.join(outdir, f"out_plots/feature_churn_plots/{csv_name}")
# os.makedirs(plots_dir, exist_ok=True)
#
# # Load selected features from specified run
# df = load_selections(csv)
#
# # Use unique feature_ids per (mod, pol, cond, time) regardless of iteration
# df_u = df.drop_duplicates(["modality", "polarity", "condition", "time_point", "feature_id"]).copy()
#
# # Get churn summary and write to feature_churn_summary_{csv_time} so that the name of summary matches
# # feature selection run.
# summary = build_summary(df_u)
# summary_csv = os.path.join(outdir, f"out_csvs/churn_of_{csv_name}.csv")
# summary.to_csv(summary_csv, index=False)
#
# # Plot churn summary plots to subdirectory set above
# saved = plot_group_bars(summary, plots_dir)
#
# print(f"Summary written to: {summary_csv}")
# if saved:
#     print(f"Saved {len(saved)} plots to: {plots_dir}")
# else:
#     print("No plots generated (no groups with data).")


"""
*******************************************************************************
BLOCK 3: PCA projections on selected features
*******************************************************************************
Visualize structure in the selected feature space via PCA.

For each (modality, polarity, condition) triple:
  - `pca_intersection`:
      * Uses the intersection of features selected across time points.
  - `pca_seed`:
      * Uses the feature set selected at a chosen seed time and projects
        all time points onto that set.

User-facing settings:
  - `csv_name`: must match the selected–features CSV basename from Block 1.
  - `modalities`, `polarities`, `conditions`, `seeds`: control which groups
    and seed times are visualized.

Outputs:
  - PCA plots written to results/out_plots/pca_plots/{csv_name}/.
"""
# csv_name = "selected_features_20251118_211050"
# modalities = ["metabolite", "lipid"]
# polarities = ["+", "-"]
# conditions = ["CHIKV", "DENV", "ZIKV"]
# seeds = [0, 5, 7, 14, 21]
#
#
# #NOTE: You don't need to touch rest of the code in this block.
#
# csv_dir = "results/out_csvs/"
# csv      = csv_dir + csv_name + ".csv"
#
#
#
# for m in modalities:
#     for pol in polarities:
#         for c in conditions:
#             p = pca_intersection(
#                 csv_path     = csv,
#                 root         = "../parquets",
#                 modality     = m,
#                 polarity     = pol,
#                 condition    = c,
#                 times        = None,
#                 outdir       = f"results/out_plots/pca_plots/{csv_name}",
#                 min_features = 2,
#             )
#             print("Saved:", p)
#             for t in seeds:
#                 p = pca_seed(
#                     csv_path     = csv,
#                     root         = "../parquets",
#                     modality     = m,
#                     polarity     = pol,
#                     condition    = c,
#                     seed_time    = t,
#                     times        = None,
#                     outdir       = f"results/out_plots/pca_plots/{csv_name}",
#                     min_features =2,
#                 )
#                 print("Saved:", p)
# pass
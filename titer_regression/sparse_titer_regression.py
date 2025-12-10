#!/usr/bin/env python3
"""
sparse_titer_regression

Bootstrap-ElasticNet feature selection + L2 regression for virus titer.

For each (condition, modality, polarity):

  1. Query star-schema Parquets -> (X_df, y=titer).
  2. Drop samples with NaN titers.
  3. 80/20 train/test split stratified by time_point (if possible).
  4. Fit preprocessing pipeline (from query_and_preprocess_v1.make_preprocess_pipeline)
     on train and transform train/test.
  5. Bootstrap the *preprocessed* train data `config.bootstrap` times:
       - Fit ElasticNet (L1 ratio, alpha) each time.
       - Record which features are selected (|coef| > coef_threshold).
  6. Aggregate selection counts per feature → write selected_features CSV.
  7. Pick Ridge alpha from a grid via CV on the train split.
  8. For k = 0, 1, ..., max(n_selections):
       - Keep features with n_selections >= k.
       - Fit Ridge (L2) on train, compute train/test RMSE.
     → write RMSE CSV.
  9. For each (condition, modality, polarity), plot:
       min_selections vs train_RMSE and test_RMSE,
       with horizontal lines for baseline (timepoint-mean) RMSE.

Outputs (all names suffixed by config.run_id)
---------------------------------------------
  results/out_csvs/selected_features_<run_id>.csv
      columns: condition, modality, polarity, feature_id, n_selections

  results/out_csvs/rmse_vs_min_selections_<run_id>.csv
      columns: condition, modality, polarity, min_selections, n_features,
               train_RMSE, test_RMSE,
               baseline_train_RMSE, baseline_test_RMSE

  results/out_csvs/sparse_titer_regression_config_<run_id>.json
      serialized config (dataclass → dict)

  results/out_plots/rmse_vs_min_selections_<cond>_<mod>_<pol>_<run_id>.png
      line plot: min_selections vs RMSE (train + test + baseline lines)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

# ----------------------------------------------------------------------
# Adjust this import to match where your make_dataset / pipeline live.
# ----------------------------------------------------------------------
import utilities.query_and_preprocess_v1 as qp


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class SparseTiterRegressionConfig:
    # Unique run identifier; appended to all filenames
    run_id: str

    # Parquet root ("parquets" directory containing samples/features/abundances)
    parquet_root: str = "parquets"

    # Base results directory (CSV + plots go under subdirs)
    results_root: str = "results"

    # Experimental filters
    conditions: Sequence[str] = ("CHIKV", "DENV", "ZIKV")
    modalities: Sequence[str] = ("metabolite", "lipid")
    polarities: Sequence[str] = ("+", "-")
    time_points: Sequence[int] = (5, 7, 14, 21)  # typically titer > 0

    # Preprocessing (delegated to qp.make_preprocess_pipeline)
    imputer: str = "median"              # 'median', 'knn', or 'two_step_label_agnostic'
    imputer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Groupwise missingness filter
    group_wise_min_prop: float = 0.5

    # Train/test split
    test_size: float = 0.2               # 80/20 split
    random_state: int = 123

    # Bootstrap ElasticNet feature selection
    bootstrap: int = 50                  # number of bootstrap resamples
    l1_ratio: float = 0.9                # ElasticNet L1 ratio
    alpha: float = 0.1                   # ElasticNet overall penalty
    max_iter: int = 1000
    epsilon: float = 1e-4                # convergence tolerance (maps to tol)
    coef_threshold: float = 1e-6         # treat |coef| > this as "selected"

    # Final L2 regressor for RMSE curves
    # (ridge_alpha is kept as a fallback; actual alpha is chosen from ridge_alpha_grid via CV.)
    ridge_alpha: float = 1.0
    ridge_alpha_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0)


# ----------------------------------------------------------------------
# Core routine
# ----------------------------------------------------------------------

def run_sparse_titer_regression(config: SparseTiterRegressionConfig) -> None:
    """
    Main entry point. To be called from an external driver, e.g.:

        from sparse_titer_regression import SparseTiterRegressionConfig, run_sparse_titer_regression

        cfg = SparseTiterRegressionConfig(run_id="20251208_v1")
        run_sparse_titer_regression(cfg)
    """
    rng = np.random.default_rng(config.random_state)

    out_csv_dir = os.path.join(config.results_root, "out_csvs")
    out_plots_dir = os.path.join(config.results_root, "out_plots")
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_plots_dir, exist_ok=True)

    selected_rows: List[Dict[str, Any]] = []
    rmse_rows: List[Dict[str, Any]] = []

    for condition in config.conditions:
        for modality in config.modalities:
            for polarity in config.polarities:
                print(f"=== condition={condition}, modality={modality}, polarity={polarity} ===")

                # ----------------- 1. Query data (X_df, y=titer) -----------------
                try:
                    X_df, y, S, F = qp.make_dataset(
                        root=config.parquet_root,
                        conditions=[condition],
                        times=config.time_points,
                        omics=[modality],
                        polarity=[polarity],
                        label_col="titer",
                    )
                except Exception as e:
                    print(
                        f"[WARN] Skipping (condition={condition}, modality={modality}, polarity={polarity}) "
                        f"due to dataset error: {e}"
                    )
                    continue

                # Filter features by groupwise missingness
                X_df = qp.filter_groupwise_missingness(
                    X_df=X_df,
                    samples_df=S,
                    group_cols=("condition", "time_point"),
                    min_prop=config.group_wise_min_prop,
                    require_all_groups=False,
                )

                # Align sample metadata to X_df and drop NaN titers
                S_idx = S.set_index("sample_id").loc[X_df.index]
                mask_valid = ~np.isnan(y)

                if mask_valid.sum() < 3:
                    print(f"[WARN] Fewer than 3 samples with non-NaN titer; skipping this combo.")
                    continue

                X_df = X_df.iloc[mask_valid]
                y_vec = y[mask_valid]
                S_idx = S_idx.iloc[mask_valid]
                time_arr = S_idx["time_point"].to_numpy()

                # Log2-transform titers
                y_vec = np.log2(y_vec)

                # ----------------- 2. Train/test split (80/20) -----------------
                try:
                    split_seed = rng.integers(0, 1_000_000)
                    X_train_df, X_test_df, y_train, y_test, time_train, time_test = train_test_split(
                        X_df,
                        y_vec,
                        time_arr,
                        test_size=config.test_size,
                        random_state=int(split_seed),
                        stratify=time_arr,
                    )
                except ValueError as e:
                    # If stratified split fails (e.g. not enough samples in some time bin),
                    # fall back to unstratified split.
                    print(
                        f"[WARN] Stratified split failed ({e}); falling back to unstratified split."
                    )
                    split_seed = rng.integers(0, 1_000_000)
                    X_train_df, X_test_df, y_train, y_test, time_train, time_test = train_test_split(
                        X_df,
                        y_vec,
                        time_arr,
                        test_size=config.test_size,
                        random_state=int(split_seed),
                        stratify=None,
                    )

                if X_train_df.shape[0] < 3 or X_train_df.shape[1] < 2:
                    print(
                        f"[WARN] Not enough data after split (n_train={X_train_df.shape[0]}, "
                        f"p={X_train_df.shape[1]}); skipping."
                    )
                    continue

                feature_names = X_train_df.columns.to_numpy()

                # ----------------- Baseline: timepoint-mean regressor ---------
                # Compute mean log2-titer per timepoint on TRAIN only.
                train_means_by_time: Dict[Any, float] = {}
                for t in np.unique(time_train):
                    train_means_by_time[t] = float(y_train[time_train == t].mean())

                # Baseline predictions
                y_train_baseline = np.array(
                    [train_means_by_time[t] for t in time_train]
                )
                # For test, if a timepoint wasn't seen in train (unlikely), fall back to global mean
                global_mean_train = float(y_train.mean())
                y_test_baseline = np.array(
                    [train_means_by_time.get(t, global_mean_train) for t in time_test]
                )

                baseline_train_rmse = float(
                    np.sqrt(mean_squared_error(y_train, y_train_baseline))
                )
                baseline_test_rmse = float(
                    np.sqrt(mean_squared_error(y_test, y_test_baseline))
                )

                # ----------------- 3. Preprocess (train-aware) -----------------
                preproc = qp.make_preprocess_pipeline(
                    imputer=config.imputer,
                    scale="standard",
                    **(config.imputer_kwargs or {}),
                )
                preproc.fit(X_train_df.to_numpy(), y_train)
                X_train_proc = preproc.transform(X_train_df.to_numpy())
                X_test_proc = preproc.transform(X_test_df.to_numpy())

                n_train, n_features = X_train_proc.shape
                sel_counts = np.zeros(n_features, dtype=int)

                # ----------------- 4. Bootstrap ElasticNet -----------------
                for b in range(config.bootstrap):
                    boot_idx = rng.integers(0, n_train, size=n_train)
                    X_boot = X_train_proc[boot_idx]
                    y_boot = y_train[boot_idx]

                    model_seed = rng.integers(0, 1_000_000)
                    enet = ElasticNet(
                        alpha=config.alpha,
                        l1_ratio=config.l1_ratio,
                        max_iter=config.max_iter,
                        tol=config.epsilon,
                        random_state=int(model_seed),
                        fit_intercept=True,
                    )
                    enet.fit(X_boot, y_boot)

                    coefs = enet.coef_.ravel()
                    selected_mask = np.abs(coefs) > config.coef_threshold
                    sel_counts += selected_mask.astype(int)

                # Record selected features (only those selected at least once)
                for feat_name, count in zip(feature_names, sel_counts):
                    if count <= 0:
                        continue
                    selected_rows.append(
                        {
                            "condition": condition,
                            "modality": modality,
                            "polarity": polarity,
                            "feature_id": str(feat_name),
                            "n_selections": int(count),
                        }
                    )

                if not np.any(sel_counts > 0):
                    print(
                        "[WARN] No features selected by ElasticNet for this setting; "
                        "skipping RMSE curves."
                    )
                    continue

                # ----------------- 5. Choose Ridge alpha via CV ---------------
                candidate_mask = sel_counts > 0
                X_train_candidates = X_train_proc[:, candidate_mask]

                # If somehow only one feature, still do CV; KFold will handle as long as n_train >= 2.
                alpha_grid = list(config.ridge_alpha_grid)
                best_alpha = None
                best_cv_rmse = np.inf

                n_splits = min(5, X_train_candidates.shape[0])
                if n_splits < 2:
                    # Fallback: no real CV possible
                    best_alpha = config.ridge_alpha
                    print(
                        f"[WARN] Too few training samples for CV; using fallback ridge_alpha={best_alpha}."
                    )
                else:
                    kf = KFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=config.random_state,
                    )
                    for alpha in alpha_grid:
                        cv_rmses: List[float] = []
                        for tr_idx, val_idx in kf.split(X_train_candidates):
                            X_tr_cv = X_train_candidates[tr_idx]
                            X_val_cv = X_train_candidates[val_idx]
                            y_tr_cv = y_train[tr_idx]
                            y_val_cv = y_train[val_idx]

                            ridge_cv = Ridge(alpha=alpha, fit_intercept=True)
                            ridge_cv.fit(X_tr_cv, y_tr_cv)
                            y_val_pred = ridge_cv.predict(X_val_cv)
                            rmse_val = float(
                                np.sqrt(mean_squared_error(y_val_cv, y_val_pred))
                            )
                            cv_rmses.append(rmse_val)

                        mean_rmse = float(np.mean(cv_rmses))
                        if mean_rmse < best_cv_rmse:
                            best_cv_rmse = mean_rmse
                            best_alpha = alpha

                    if best_alpha is None:
                        best_alpha = config.ridge_alpha
                        print(
                            f"[WARN] CV did not set best_alpha; using fallback ridge_alpha={best_alpha}."
                        )

                print(
                    f"[INFO] Best Ridge alpha for ({condition}, {modality}, {polarity}) "
                    f"from grid {alpha_grid} is {best_alpha} (CV RMSE={best_cv_rmse:.3f})."
                )

                # ----------------- 6. RMSE curves vs min_selections (k) -------
                max_sel = int(sel_counts.max())
                k_values = list(range(0, max_sel + 1))  # include k=0 baseline

                for k in k_values:
                    mask_k = sel_counts >= k
                    if not np.any(mask_k):
                        continue

                    X_tr_k = X_train_proc[:, mask_k]
                    X_te_k = X_test_proc[:, mask_k]

                    ridge = Ridge(alpha=best_alpha, fit_intercept=True)
                    ridge.fit(X_tr_k, y_train)
                    y_tr_pred = ridge.predict(X_tr_k)
                    y_te_pred = ridge.predict(X_te_k)

                    train_rmse = float(
                        np.sqrt(mean_squared_error(y_train, y_tr_pred))
                    )
                    test_rmse = float(
                        np.sqrt(mean_squared_error(y_test, y_te_pred))
                    )

                    rmse_rows.append(
                        {
                            "condition": condition,
                            "modality": modality,
                            "polarity": polarity,
                            "min_selections": int(k),
                            "n_features": int(mask_k.sum()),
                            "train_RMSE": train_rmse,
                            "test_RMSE": test_rmse,
                            "baseline_train_RMSE": baseline_train_rmse,
                            "baseline_test_RMSE": baseline_test_rmse,
                            "ridge_alpha": float(best_alpha),
                        }
                    )

    # ------------------------------------------------------------------
    # 6. Write selected-features CSV
    # ------------------------------------------------------------------
    if selected_rows:
        sel_df = pd.DataFrame(selected_rows)
        sel_path = os.path.join(
            out_csv_dir,
            f"selected_features_{config.run_id}.csv",
        )
        sel_df.to_csv(sel_path, index=False)
        print(f"[INFO] Wrote selected features to {sel_path}")
    else:
        print("[WARN] No selected features to write.")

    # ------------------------------------------------------------------
    # 7. Write RMSE CSV
    # ------------------------------------------------------------------
    if rmse_rows:
        rmse_df = pd.DataFrame(rmse_rows)
        rmse_path = os.path.join(
            out_csv_dir,
            f"rmse_vs_min_selections_{config.run_id}.csv",
        )
        rmse_df.to_csv(rmse_path, index=False)
        print(f"[INFO] Wrote RMSE summary to {rmse_path}")
    else:
        print("[WARN] No RMSE results to write.")
        return

    # ------------------------------------------------------------------
    # 8. Write config sidecar (JSON)
    # ------------------------------------------------------------------
    cfg_dict = asdict(config)
    cfg_path = os.path.join(
        out_csv_dir,
        f"sparse_titer_regression_config_{config.run_id}.json",
    )
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[INFO] Wrote config sidecar to {cfg_path}")

    # ------------------------------------------------------------------
    # 9. Make plots per (condition, modality, polarity)
    # ------------------------------------------------------------------
    os.makedirs(f"{out_plots_dir}/run_id__{config.run_id}", exist_ok=True)
    rmse_df = pd.DataFrame(rmse_rows)
    for (condition, modality, polarity), grp in rmse_df.groupby(
        ["condition", "modality", "polarity"]
    ):
        grp_sorted = grp.sort_values("min_selections")

        # Baseline RMSEs (same within group)
        baseline_train = float(grp_sorted["baseline_train_RMSE"].iloc[0])
        baseline_test = float(grp_sorted["baseline_test_RMSE"].iloc[0])

        fig, ax = plt.subplots()
        ax.plot(
            grp_sorted["min_selections"],
            grp_sorted["train_RMSE"],
            marker="o",
            label="Train RMSE",
        )
        ax.plot(
            grp_sorted["min_selections"],
            grp_sorted["test_RMSE"],
            marker="o",
            label="Test RMSE",
        )

        # Add baseline as horizontal dashed lines
        ax.axhline(
            baseline_train,
            linestyle="--",
            linewidth=1.0,
            label="Baseline train RMSE",
        )
        ax.axhline(
            baseline_test,
            linestyle="--",
            linewidth=1.0,
            label="Baseline test RMSE",
        )

        ax.set_xlabel("min_selections (k)")
        ax.set_ylabel("RMSE (log2 titer)")
        ax.set_title(f"{condition} | {modality} | {polarity}")
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(
            out_plots_dir,
            f"run_id__{config.run_id}",
            f"rmse_vs_min_selections_{condition}_{modality}_{polarity}_{config.run_id}.png",
        )
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Wrote plot to {plot_path}")


# ----------------------------------------------------------------------
# Optional: quick manual test when run directly
# (You will normally call run_sparse_titer_regression from an external driver.)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    example_cfg = SparseTiterRegressionConfig(run_id="dev_test")
    run_sparse_titer_regression(example_cfg)

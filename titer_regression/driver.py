#!/usr/bin/env python3
"""
Minimal driver for sparse_titer_regression.

Edit the SparseTiterRegressionConfig instance in main(), then run:

    python run_sparse_titer_regression.py
"""

from __future__ import annotations

from sparse_titer_regression import (
    SparseTiterRegressionConfig,
    run_sparse_titer_regression,
)


def main() -> None:
    # ------------------------------------------------------------------
    # USER SETTINGS:
    # Edit any of these fields to control the run.
    # ------------------------------------------------------------------
    cfg = SparseTiterRegressionConfig(
        run_id="3",          # change per run to keep outputs distinct

        # Parquet star schema root (directory with samples/features/abundances)
        parquet_root="../parquets",

        # Results root: CSVs -> results/out_csvs, plots -> results/out_plots
        results_root="results",

        # Group-wise missingness filter
        group_wise_min_prop = 0.5,

        # Experimental filters
        conditions=("CHIKV", "DENV", "ZIKV"),
        modalities=("metabolite", "lipid"),
        polarities=("+", "-"),
        time_points=(5, 7, 14, 21),

        # Preprocessing
        imputer="two_step_label_agnostic",  # 'median', 'knn', or 'two_step_label_agnostic'
        imputer_kwargs={
            "noise_scale" : 0.001,
            "nan_threshold" : 0.4,
            "k" : 4
        },                  # e.g. {"noise_scale": 0.01, ...} for two_step_label_agnostic

        # Train/test split
        test_size=0.2,                   # 80/20
        random_state=123,

        # Bootstrap ElasticNet feature selection
        bootstrap=50,                    # number of bootstrap resamples
        l1_ratio=0.95,
        alpha=1,
        max_iter=10000,
        epsilon=1e-5,                    # convergence tolerance (tol)
        coef_threshold=1e-6,             # |coef| > this => "selected"

        # Final L2 regressor
        ridge_alpha=1.0,
    )

    # ------------------------------------------------------------------
    # Run the pipeline
    # ------------------------------------------------------------------
    run_sparse_titer_regression(cfg)


if __name__ == "__main__":
    main()

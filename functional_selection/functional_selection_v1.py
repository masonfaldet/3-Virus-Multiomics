#!/usr/bin/env python3
"""
Functional feature selection based on time-course differences between MOCK and virus.

For each (modality, polarity, virus) triple:
  - Query samples & abundances via query_and_preprocess_v1.make_dataset
  - Filter features by groupwise missingness (condition × time_point)
  - Apply log2(x + eps) using Log2WithPseudocount from query_and_preprocess_v1
  - For each feature f:
      * Build median time-course curves a_f^{MOCK}(t), a_f^{virus}(t)
      * Optionally center and/or L2-normalize each curve (across time)
      * Compute weighted L2 distance between the two curves over time
      * Estimate a permutation p-value for this distance
      * Count NaNs across all samples in {MOCK, virus}
  - Write one row per feature to a CSV with columns:
      feature_id, modality, polarity, virus, weighted_l2, p_value, n_NaNs

Output CSV:
  results/out_csvs/functional_selection_YYYYMMDD_HHMMSS.csv

A JSON sidecar "._functional_selection_YYYYMMDD_HHMMSS.json" records the run config.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Sequence, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from utilities.query_and_preprocess_v1 import (
    make_dataset,
    filter_groupwise_missingness,
    Log2WithPseudocount,
)


@dataclass
class FunctionalSelectionConfig:
    """Configuration for functional time-course feature selection.

    Edit an instance of this in the __main__ block to control all knobs.

    Attributes
    ----------
    root : str
        Root directory containing Parquet star-schema tables
        (samples.parquet, features.parquet, abundances.parquet).
    modalities : tuple[str, ...]
        Omics modalities to include, e.g. ("lipid", "metabolite").
    polarities : tuple[str, ...]
        Ionization polarities, e.g. ("+", "-").
    viruses : tuple[str, ...]
        Conditions to contrast against MOCK, e.g. ("CHIKV", "DENV", "ZIKV").
    mock_label : str
        Label used for the control condition (default "MOCK").
    time_points : tuple[int, ...]
        Time points (e.g. (0, 5, 7, 14, 21)) used to build time-course curves.
        Only time points present in the data will contribute to distances.
    group_cols : tuple[str, ...]
        Columns in the samples table to define groups for missingness filtering.
        Default is ("condition", "time_point").
    min_prop : float
        Minimum non-missing proportion within a group to keep a feature.
    min_group_n : int
        Minimum group size for a group to be considered in filtering.
    require_all_groups : bool
        If True, feature must pass min_prop in *every* eligible group.
        If False, passing in at least one group is enough.
    center_curves : bool
        If True, center each curve over time (weighted mean = 0) before computing distance.
        Use this to focus on *shape* rather than overall level.
    scale_curves : bool
        If True, scale each curve to unit weighted L2 norm before computing distance.
        Typically used together with center_curves for pure shape comparisons.
    n_permutations : int
        Number of label permutations for p-value estimation.
    random_state : int
        Seed for the RNG used in permutations.
    log_fallback_eps : float
        Fallback epsilon for Log2WithPseudocount transformer.
    output_pattern : str
        Base path for output CSV, before timestamp is appended.
    """

    # IO / data selection
    root: str = "parquets"
    modalities: Tuple[str, ...] = ("lipid", "metabolite")
    polarities: Tuple[str, ...] = ("+", "-")
    viruses: Tuple[str, ...] = ("CHIKV", "DENV", "ZIKV")
    mock_label: str = "MOCK"
    time_points: Tuple[int, ...] = (0, 5, 7, 14, 21)

    # Groupwise missingness filter
    group_cols: Tuple[str, ...] = ("condition", "time_point")
    min_prop: float = 0.6
    min_group_n: int = 3
    require_all_groups: bool = True

    # Curve processing
    center_curves: bool = False
    scale_curves: bool = False

    # Permutation testing
    n_permutations: int = 1000
    random_state: int = 42

    # Log transform
    log_fallback_eps: float = 1e-3

    # Output
    output_pattern: str = "results/out_csvs/functional_selection.csv"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def trapezoidal_weights(times: np.ndarray) -> np.ndarray:
    """Compute trapezoidal-rule weights on a sorted 1D array of time points."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D array.")
    if times.size == 1:
        return np.ones(1, dtype=float)
    t = np.sort(times)
    n = t.size
    w = np.zeros_like(t)
    w[0] = 0.5 * (t[1] - t[0])
    for i in range(1, n - 1):
        w[i] = 0.5 * (t[i + 1] - t[i - 1])
    w[n - 1] = 0.5 * (t[n - 1] - t[n - 2])
    return w


def center_and_scale_curve(values: np.ndarray,
                           weights: np.ndarray,
                           center: bool,
                           scale: bool) -> np.ndarray:
    """Optionally center and L2-normalize a curve over time.

    `values` and `weights` are assumed to be 1D arrays of the same length,
    restricted to time points where the curve is defined (no NaNs).
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if center:
        w_sum = w.sum()
        if w_sum > 0:
            mean = float((w * v).sum() / w_sum)
            v = v - mean
    if scale:
        norm = float(np.sqrt((w * (v ** 2)).sum()))
        if norm > 0:
            v = v / norm
    return v


def weighted_l2_distance(curve_a: np.ndarray,
                         curve_b: np.ndarray,
                         weights: np.ndarray,
                         center_curves: bool,
                         scale_curves: bool) -> float:
    """Compute weighted L2 distance between two curves with optional centering/scaling.

    Curves are 1D arrays of the same length. NaNs are allowed; we only use
    positions where *both* curves are finite.
    """
    a = np.asarray(curve_a, dtype=float)
    b = np.asarray(curve_b, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan

    a_valid = a[mask]
    b_valid = b[mask]
    w_valid = w[mask]

    a_valid = center_and_scale_curve(a_valid, w_valid, center_curves, scale_curves)
    b_valid = center_and_scale_curve(b_valid, w_valid, center_curves, scale_curves)

    diff = a_valid - b_valid
    d2 = float((w_valid * (diff ** 2)).sum())
    return float(np.sqrt(d2))


def build_curves_for_feature(
    x: np.ndarray,
    cond: np.ndarray,
    times: np.ndarray,
    times_grid: np.ndarray,
    mock_label: str,
    virus_label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a_f^{MOCK}(t) and a_f^{virus}(t) as median-over-samples curves.

    Parameters
    ----------
    x : array of shape (n_samples,)
        Log-transformed abundances for this feature.
    cond : array of shape (n_samples,)
        Condition labels (should include mock_label and virus_label only).
    times : array of shape (n_samples,)
        Time-point labels aligned with x.
    times_grid : array of unique sorted time points to evaluate curves on.
    mock_label : str
    virus_label : str

    Returns
    -------
    a_mock, a_virus : arrays of shape (len(times_grid),)
        Median curves; may contain NaNs where no samples are available.
    """
    x = np.asarray(x, dtype=float)
    cond = np.asarray(cond)
    times = np.asarray(times)
    times_grid = np.asarray(times_grid)

    a_mock = np.full(times_grid.shape, np.nan, dtype=float)
    a_virus = np.full(times_grid.shape, np.nan, dtype=float)

    for i, t in enumerate(times_grid):
        mask_t = (times == t)
        if not np.any(mask_t):
            continue

        # MOCK
        mask_mock = mask_t & (cond == mock_label)
        if np.any(mask_mock):
            vals = x[mask_mock]
            if np.any(np.isfinite(vals)):
                a_mock[i] = float(np.nanmedian(vals))

        # virus
        mask_virus = mask_t & (cond == virus_label)
        if np.any(mask_virus):
            vals = x[mask_virus]
            if np.any(np.isfinite(vals)):
                a_virus[i] = float(np.nanmedian(vals))

    return a_mock, a_virus


def build_permutation_label_sets(
    cond: np.ndarray,
    times: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Generate permuted condition label arrays, stratified by time point."""
    cond = np.asarray(cond)
    times = np.asarray(times)
    unique_times = np.unique(times)

    perms: List[np.ndarray] = []
    for _ in range(n_perms):
        perm = cond.copy()
        for t in unique_times:
            idx = np.where(times == t)[0]
            if idx.size > 1:
                perm[idx] = rng.permutation(perm[idx])
        perms.append(perm)
    return perms


def compute_stat_and_pvalue_for_feature(
    x: np.ndarray,
    cond: np.ndarray,
    times: np.ndarray,
    times_grid: np.ndarray,
    weights: np.ndarray,
    mock_label: str,
    virus_label: str,
    perm_labels: List[np.ndarray],
    center_curves: bool,
    scale_curves: bool,
) -> Tuple[float, float]:
    """Compute observed weighted L2 distance and permutation p-value for one feature."""
    # Observed statistic
    a_mock_obs, a_virus_obs = build_curves_for_feature(
        x=x,
        cond=cond,
        times=times,
        times_grid=times_grid,
        mock_label=mock_label,
        virus_label=virus_label,
    )
    d_obs = weighted_l2_distance(
        a_mock_obs, a_virus_obs, weights,
        center_curves=center_curves,
        scale_curves=scale_curves,
    )
    if not np.isfinite(d_obs):
        return np.nan, np.nan

    # Permutation distribution
    if not perm_labels:
        return d_obs, np.nan

    perm_stats = []
    for perm in perm_labels:
        a_mock_perm, a_virus_perm = build_curves_for_feature(
            x=x,
            cond=perm,
            times=times,
            times_grid=times_grid,
            mock_label=mock_label,
            virus_label=virus_label,
        )
        d_perm = weighted_l2_distance(
            a_mock_perm, a_virus_perm, weights,
            center_curves=center_curves,
            scale_curves=scale_curves,
        )
        if np.isfinite(d_perm):
            perm_stats.append(d_perm)

    if len(perm_stats) == 0:
        return d_obs, np.nan

    perm_stats = np.asarray(perm_stats, dtype=float)
    # One-sided p-value: P(D_perm >= D_obs) with +1 smoothing
    p_val = (1.0 + np.sum(perm_stats >= d_obs)) / (perm_stats.size + 1.0)
    return d_obs, float(p_val)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_functional_selection(config: FunctionalSelectionConfig) -> pd.DataFrame:
    """Run functional time-course feature selection for all configured settings.

    Returns
    -------
    pandas.DataFrame
        One row per (feature, modality, polarity, virus) with columns:
        feature_id, modality, polarity, virus, weighted_l2, p_value, n_NaNs.
    """
    rng = np.random.default_rng(config.random_state)
    all_rows: List[Dict] = []

    for modality in config.modalities:
        for polarity in config.polarities:
            for virus in config.viruses:
                print(f"[INFO] Processing modality={modality}, polarity={polarity}, virus={virus}")

                # 1) Query dataset for this modality/polarity/virus vs MOCK
                try:
                    X_df, y, S, F = make_dataset(
                        root=config.root,
                        conditions=[virus, config.mock_label],
                        times=list(config.time_points),
                        omics=[modality],
                        polarity=[polarity],
                        label_col="condition",
                    )
                except Exception as e:
                    print(f"  [WARN] make_dataset failed for ({modality}, {polarity}, {virus}): {e}")
                    continue

                if X_df.empty:
                    print("  [WARN] Empty X_df; skipping.")
                    continue

                # 2) Filter by groupwise missingness
                try:
                    X_filt = filter_groupwise_missingness(
                        X_df=X_df,
                        samples_df=S,
                        group_cols=config.group_cols,
                        min_prop=config.min_prop,
                        min_group_n=config.min_group_n,
                        require_all_groups=config.require_all_groups,
                    )
                except Exception as e:
                    print(f"  [WARN] filter_groupwise_missingness failed: {e}")
                    continue

                if X_filt.shape[1] == 0:
                    print("  [WARN] No features left after groupwise missingness filter; skipping.")
                    continue

                # Align metadata with filtered X
                S_idx = S.set_index("sample_id").loc[X_filt.index]
                cond = S_idx["condition"].to_numpy()
                times = S_idx["time_point"].to_numpy()

                # Enforce only virus vs MOCK
                mask_pair = (cond == virus) | (cond == config.mock_label)
                if not np.any(mask_pair):
                    print("  [WARN] No samples with virus vs MOCK labels after alignment; skipping.")
                    continue

                X_filt = X_filt.loc[mask_pair]
                cond = cond[mask_pair]
                times = times[mask_pair]

                # 3) Log2 transform (with pseudocount)
                log2_tf = Log2WithPseudocount(fallback_eps=config.log_fallback_eps)
                X_log_np = log2_tf.fit_transform(X_filt.values)
                X_log = pd.DataFrame(X_log_np, index=X_filt.index, columns=X_filt.columns)

                # 4) Time grid and weights
                unique_times = sorted(set(int(t) for t in times) & set(config.time_points))
                if not unique_times:
                    print("  [WARN] No overlapping time points between data and config.time_points; skipping.")
                    continue

                times_grid = np.array(unique_times, dtype=int)
                weights = trapezoidal_weights(times_grid.astype(float))

                # 5) Precompute permutation label sets (stratified by time)
                perm_labels: List[np.ndarray] = []
                if config.n_permutations > 0:
                    perm_labels = build_permutation_label_sets(
                        cond=cond,
                        times=times,
                        n_perms=config.n_permutations,
                        rng=rng,
                    )

                # 6) Per-feature statistics
                X_np = X_log.values
                n_samples, n_features = X_np.shape
                mask_vm = (cond == virus) | (cond == config.mock_label)

                for j, feature_id in enumerate(X_log.columns):
                    x = X_np[:, j]

                    # Count NaNs across all samples in {MOCK, virus}
                    n_nans = int(np.isnan(x[mask_vm]).sum())

                    d_obs, p_val = compute_stat_and_pvalue_for_feature(
                        x=x,
                        cond=cond,
                        times=times,
                        times_grid=times_grid,
                        weights=weights,
                        mock_label=config.mock_label,
                        virus_label=virus,
                        perm_labels=perm_labels,
                        center_curves=config.center_curves,
                        scale_curves=config.scale_curves,
                    )

                    all_rows.append(
                        {
                            "feature_id": feature_id,
                            "modality": modality,
                            "polarity": polarity,
                            "virus": virus,
                            "weighted_l2": d_obs,
                            "p_value": p_val,
                            "n_NaNs": n_nans,
                        }
                    )

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "feature_id",
                "modality",
                "polarity",
                "virus",
                "weighted_l2",
                "p_value",
                "n_NaNs",
            ]
        )

    df = pd.DataFrame(all_rows)
    return df

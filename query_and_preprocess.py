#!/usr/bin/env python3
"""
Query → Preprocess → ML-ready arrays from Parquet star schema.

Inputs (from build_parquets_from_peaklists.py outputs):
  parquets/samples.parquet     : sample_id, condition, time_point, ...
  parquets/features.parquet    : feature_id, omics, polarity, ...
  parquets/abundances.parquet  : sample_id, feature_id, area, rt, mz, omics, polarity, ...

Workflow
--------
1) Query samples by condition/time/omics/polarity → sample list
2) Pivot abundances to samples × features (areas)
3) (Optional) Feature filtering by missingness overall or per group
4) Train/test split
5) Preprocess pipeline (log2 + pseudocount) → impute → scale
6) Emit (X_train, y_train), (X_test, y_test), with aligned ids

This script exposes reusable functions and a runnable demo under __main__.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------- Query layer ----------------------------------

def load_tables(root: str):
    samples = pd.read_parquet(os.path.join(root, "samples.parquet"))
    abund   = pd.read_parquet(os.path.join(root, "abundances.parquet"))
    features= pd.read_parquet(os.path.join(root, "features.parquet"))
    return samples, abund, features


def query_samples(
    samples: pd.DataFrame,
    conditions: Optional[Sequence[str]] = None,
    times: Optional[Sequence[int]] = None,
    exclude_qc: bool = True,
) -> pd.DataFrame:
    S = samples.copy()
    if conditions is not None:
        S = S[S["condition"].isin(conditions)]
    if times is not None:
        S = S[S["time_point"].isin(times)]
    # omics/polarity live in abundances, not samples; leave for pivot stage
    if exclude_qc and "is_qc" in S.columns:
        S = S[~S["is_qc"]]
    if S.empty:
        raise ValueError("No samples match the filters provided.")
    return S

def query_features(
        features: pd.DataFrame,
        omics: Optional[Sequence[str]] = None,
        polarity: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    F = features.copy()
    if omics is not None:
        F = F[F["omics"].isin(omics)]
    if polarity is not None:
        F = F[F["polarity"].isin(polarity)]
    if F.empty:
        raise ValueError("No features match the filters provided.")

    return F


def pivot_matrix(
    abund: pd.DataFrame,
    sample_ids: Sequence[str],
    omics: Optional[Sequence[str]] = None,
    polarity: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    A = abund[abund["sample_id"].isin(sample_ids)]
    if omics is not None:
        A = A[A["omics"].isin(omics)]
    if polarity is not None:
        A = A[A["polarity"].isin(polarity)]
    if A.empty:
        raise ValueError("No abundances after applying omics/polarity filters.")
    Xw = A.pivot(index="sample_id", columns="feature_id", values="area").sort_index()
    # Ensure all requested samples appear (even if entirely NA for selected mods)
    Xw = Xw.reindex(index=pd.Index(sample_ids, name="sample_id"))
    return Xw


def make_dataset(
    root: str,
    conditions: Optional[Sequence[str]] = None,
    times: Optional[Sequence[int]] = None,
    omics: Optional[Sequence[str]] = None,
    polarity: Optional[Sequence[str]] = None,
    label_col: str = "condition",
):
    samples, abund, features = load_tables(root)
    F = query_features(features, omics, polarity)
    S = query_samples(samples, conditions, times)
    X_df = pivot_matrix(abund, S["sample_id"].tolist(), omics, polarity)
    # Align labels to rows of X_df
    y = S.set_index("sample_id").loc[X_df.index, label_col].to_numpy()
    return X_df, y, S, F


# ------------------------- Feature filtering --------------------------------

def filter_overall_missingness(
    X_df: pd.DataFrame,
    max_na_prop: float = 0.5,
) -> pd.Index:
    """Keep features with ≤ max_na_prop missing over ALL selected samples."""
    na_prop = X_df.isna().mean(axis=0)
    keep = na_prop <= max_na_prop
    return X_df.loc[:,X_df.columns[keep]]


def filter_groupwise_missingness(
    X_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    group_cols: Sequence[str] = ("condition", "time_point"),
    min_prop: float = 0.6,
    min_group_n: int = 3,
    require_all_groups: bool = True,
) -> pd.DataFrame:
    """Keep features with ≥ min_prop observed values in EACH sufficiently large group.

    Parameters
    ----------
    X_df : samples×features DataFrame.
    samples_df : DataFrame with rows for sample_id and group columns.
    group_cols : columns in samples_df to define groups.
    min_prop : minimum proportion of non-missing within a group.
    min_group_n : a group must have at least this many samples to be considered.
    require_all_groups : if True, feature must pass the threshold in *every* eligible group.
    """

    # align metadata to X_df rows (sample_id)
    meta = samples_df.set_index("sample_id").loc[X_df.index, list(group_cols)]

    keep_mask = None  # boolean array aligned to X_df.columns
    groups = meta.groupby(list(group_cols), observed=True)

    # Use positional indices (ints) to avoid label/int confusion
    for _, pos in groups.indices.items():  # `pos` is a numpy array of row positions
        if len(pos) < min_group_n:
            continue
        prop = X_df.iloc[pos].notna().mean(axis=0).to_numpy()  # per-feature proportion observed
        g_keep = prop >= min_prop

        if keep_mask is None:
            keep_mask = g_keep
        else:
            keep_mask = (keep_mask & g_keep) if require_all_groups else (keep_mask | g_keep)

    if keep_mask is None:
        # no eligible groups; default to keeping everything
        return X_df.copy()

    return X_df.loc[:, X_df.columns[keep_mask]]


# ------------------------- Preprocess pipeline -------------------------------
class Log2WithPseudocount(BaseEstimator, TransformerMixin):
    """Apply log2(x + eps) with per-feature eps.

    eps_j = 0.5 * min_positive_j computed on the data seen in `fit`.
    Use inside pipelines for strict train-aware behavior.
    """
    def __init__(self, fallback_eps: float = 1e-3):
        self.fallback_eps = float(fallback_eps)
        self.eps_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        mins = []
        for j in range(X.shape[1]):
            col = X[:, j]
            pos = col[col > 0]
            if pos.size:
                mins.append(0.5 * float(np.nanmin(pos)))
            else:
                mins.append(self.fallback_eps)
        eps = np.array(mins, dtype=np.float32)
        eps[~np.isfinite(eps)] = self.fallback_eps
        eps[eps <= 0] = self.fallback_eps
        self.eps_ = eps
        return self

    def transform(self, X):
        if self.eps_ is None:
            raise RuntimeError("Log2WithPseudocount must be fitted before transform.")
        X = np.asarray(X, dtype=np.float32)
        return np.log2(X + self.eps_)

class LabelAgnosticTwoStepImputer(BaseEstimator, TransformerMixin):
    """Two-step label-agnostic imputation compatible with scikit-learn.

    Step 1 (fit on train only): for features with missingness >= `nan_threshold`,
    fill missing entries with (min_observed/5) + U(0, noise_scale * (min_observed/5)).
    Step 2: fit a KNNImputer on the step-1-imputed TRAIN matrix. During transform,
    apply step-1 using the TRAIN min values, then use the fitted KNNImputer to transform.

    Parameters
    ----------
    noise_scale : float
        Scale for the uniform noise in step 1.
    nan_threshold : float
        Column missingness fraction threshold to trigger step 1 on that column.
    k : int
        Number of neighbors for the KNNImputer.
    random_state : Optional[int]
        Seed for reproducibility of the uniform noise draws.
    drop_all_nan_in_any_label : bool
        If True and `y` is provided in fit, drop features that are all-NaN in any label group in TRAIN.
        (Dimensionality reduction.)
    """
    def __init__(self, noise_scale: float = 0.01, nan_threshold: float = 0.5, k: int = 5,
                 random_state: Optional[int] = 42, drop_all_nan_in_any_label: bool = False):
        self.noise_scale = float(noise_scale)
        self.nan_threshold = float(nan_threshold)
        self.k = int(k)
        self.random_state = random_state
        self.drop_all_nan_in_any_label = bool(drop_all_nan_in_any_label)
        # Fitted attrs
        self.min_impute_: Optional[dict] = None  # {col_index: min_val}
        self.step1_cols_: Optional[np.ndarray] = None  # boolean mask of columns that got step1
        self.keep_cols_mask_: Optional[np.ndarray] = None  # columns kept after optional drop
        self.imputer_: Optional[KNNImputer] = None
        self.n_features_in_: Optional[int] = None

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        n, p = X.shape
        self.n_features_in_ = p

        # Optionally drop columns that are all-NaN in any label group (train-only decision)
        keep_mask = np.ones(p, dtype=bool)
        if self.drop_all_nan_in_any_label and y is not None:
            y_arr = np.asarray(y)
            for lab in np.unique(y_arr):
                grp = X[y_arr == lab]
                all_nan_cols = np.isnan(grp).all(axis=0)
                keep_mask &= ~all_nan_cols
        self.keep_cols_mask_ = keep_mask
        Xk = X[:, keep_mask]
        p_kept = Xk.shape[1]

        # Step 1: determine which columns qualify and compute per-col min
        frac_nan = np.mean(np.isnan(Xk), axis=0)
        step1_cols = frac_nan >= self.nan_threshold
        self.step1_cols_ = step1_cols
        self.min_impute_ = {}
        X_step1 = Xk.copy()
        rng = self._rng()
        for j in np.where(step1_cols)[0]:
            col = X_step1[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.all():
                continue  # no observed min, leave for KNN
            min_val = float(np.nanmin(col))
            base = min_val / 5.0
            noise = rng.uniform(0.0, self.noise_scale * base, size=nan_mask.sum())
            col[nan_mask] = base + noise
            X_step1[:, j] = col
            self.min_impute_[int(j)] = min_val

        # Step 2: KNNImputer fit on TRAIN (after step1)
        self.imputer_ = KNNImputer(n_neighbors=self.k, weights="distance")
        self.imputer_.fit(X_step1)
        return self

    def transform(self, X):
        if self.imputer_ is None or self.keep_cols_mask_ is None or self.step1_cols_ is None:
            raise RuntimeError("LabelAgnosticTwoStepImputer is not fitted.")
        X = np.asarray(X, dtype=np.float32)
        Xk = X[:, self.keep_cols_mask_]
        X_step1 = Xk.copy()
        rng = self._rng()
        for j, min_val in self.min_impute_.items():
            col = X_step1[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                base = float(min_val) / 5.0
                noise = rng.uniform(0.0, self.noise_scale * base, size=nan_mask.sum())
                col[nan_mask] = base + noise
                X_step1[:, j] = col
        X_imp = self.imputer_.transform(X_step1)
        return X_imp

def label_aware_two_step_impute(
    X_df: pd.DataFrame,
    y: np.ndarray,
    *,
    noise_scale: float = 0.01,
    group_missing_prop: float = 0.5,
    num_neighbors: int = 5,
    drop_features_missing_in_any_label: bool = True,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Two-step **label-aware** imputation with **no train/test split**.

    For each label class independently:
      1) If a feature has > `group_missing_prop` missing in that class, fill those NaNs with
         base = group_min/5 + U(0, noise_scale * group_min).
      2) Fit a KNNImputer **on that class only** and impute remaining NaNs.

    Optionally drop any feature that is all-NaN in *any* class (to mirror
    `_remove_features_missing_in_any_label(1)`). Returns a DataFrame aligned to
    the input index; if features are dropped, the returned columns are a subset
    of the originals.
    """
    rng = np.random.default_rng(random_state)
    labels = np.asarray(y)
    X = X_df.to_numpy(dtype=np.float32, copy=True)
    n_samples, n_features = X.shape

    # Drop features that are all-NaN in any label group
    keep_cols = np.ones(n_features, dtype=bool)
    if drop_features_missing_in_any_label:
        for k in np.unique(labels):
            grp_idx = labels == k
            if not np.any(grp_idx):
                continue
            all_nan_cols = np.isnan(X[grp_idx]).all(axis=0)
            keep_cols &= ~all_nan_cols
        if not keep_cols.all():
            X = X[:, keep_cols]
    cols = X_df.columns[keep_cols]

    # Step 1: group-wise thresholded min-imputation
    mask = np.isnan(X)
    for k in np.unique(labels):
        grp_idx = np.where(labels == k)[0]
        if grp_idx.size == 0:
            continue
        group_mask = mask[grp_idx, :]
        n_k = grp_idx.size
        missing_counts = group_mask.sum(axis=0)
        to_fill_step1 = missing_counts > (group_missing_prop * n_k)  # strict '>' per spec
        if not np.any(to_fill_step1):
            continue
        sub = X[grp_idx, :]
        # per-column minima within the group (ignoring NaNs)
        mins = np.nanmin(sub, axis=0)
        for j in np.where(to_fill_step1)[0]:
            col = sub[:, j]
            nan_local = np.isnan(col)
            if nan_local.all():
                continue
            gmin = float(mins[j])
            base = gmin / 5.0
            noise = rng.uniform(0.0, noise_scale * gmin, size=nan_local.sum())
            col[nan_local] = base + noise
            sub[:, j] = col
        X[grp_idx, :] = sub
        mask[grp_idx, :] = np.isnan(sub)

    # Step 2: KNN per label group
    for k in np.unique(labels):
        grp_idx = np.where(labels == k)[0]
        if grp_idx.size == 0:
            continue
        sub = X[grp_idx, :]
        if np.isnan(sub).any():
            imp = KNNImputer(n_neighbors=num_neighbors, weights="uniform")
            sub = imp.fit_transform(sub)
            X[grp_idx, :] = sub

    return pd.DataFrame(X, index=X_df.index, columns=cols)


def make_preprocess_pipeline(
    imputer: str = "median",  # 'median', 'knn', 'two_step_label_agnostic'
    scale: str = "standard",
    **imputer_kwargs,
):
    steps: List[Tuple[str, TransformerMixin]] = []
    if imputer == "two_step_label_agnostic":
        # Two-step operates on raw scale first, then we log/scale.
        steps.append(("two_step", LabelAgnosticTwoStepImputer(**imputer_kwargs)))
        steps.append(("log2", Log2WithPseudocount(fallback_eps=1e-3)))

    else:
        # Default: log first, then classic imputation, then scale
        steps.append(("log2", Log2WithPseudocount(fallback_eps=1e-3)))
        if imputer == "median":
            steps.append(("impute", SimpleImputer(strategy="median")))
        elif imputer == "knn":
            steps.append(("impute", KNNImputer(n_neighbors=5, weights="distance")))
        else:
            raise ValueError("imputer must be 'median', 'knn', or 'two_step_label_agnostic'")
    if scale == "standard":
        steps.append(("scale", StandardScaler(with_mean=True, with_std=True)))
    else:
        raise ValueError("scale must be 'standard'")
    return Pipeline(steps)

if __name__ == "__main__":

    X_df, y, s, features = make_dataset(
       root="parquets",
       conditions=["MOCK", "CHIKV"],
       times=[0],
       omics=["metabolite"],
       polarity=["+"]
    )
    print(X_df.shape)
    print(y)
    print(s.head())
    print(features.head(3))

    X_df_filt = filter_groupwise_missingness(
        X_df = X_df,
        samples_df= s,
        group_cols= ("condition", "time_point"),
        min_prop= 0.3,
        min_group_n=1,
        require_all_groups=False
    )

    print(X_df_filt.shape)


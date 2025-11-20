#!/usr/bin/env python3
"""
PCA visualizations for selected features (separate from churn script).

Two callable functions (each returns the saved figure path; optionally returns the
figure if `return_fig=True`). Each produces a **single figure** with subplots
(one per time).

1) pca_intersection(...): For a specified (modality, polarity, condition),
   - Extract selected features at **all** available time points from selected_features.csv
   - Compute the intersection across times
   - If intersection size >= min_features, for each time t:
       • Project that time's data onto the intersection
       • Run 2D PCA and place as a subplot in a single figure

2) pca_seed(...): For a specified (modality, polarity, condition, seed_time=t*),
   - Extract selected features at t*
   - For each time t with selections available:
       • Project that time's data onto the **seed** feature set
       • Run 2D PCA and place as a subplot in a single figure

Notes:
- Uses query_and_preprocess.make_dataset and make_preprocess_pipeline to
  impute + log2(+eps) + standardize within each time's matrix.
- Requires labels y ∈ {condition, MOCK} per time.
- Skips a subplot if < min_features remain after column alignment.
- Saves a single PNG per call under `outdir`.
"""
from __future__ import annotations
import os
import re
from math import ceil, sqrt
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utilities.query_and_preprocess_v1 import make_dataset, make_preprocess_pipeline

TIME_ORDER = [0, 5, 7, 14, 21]

# ------------------------------ Utilities -----------------------------------

def sanitize(s: str) -> str:
    if s == "+":
        return "P"
    if s == "-":
        return "N"
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_selections(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"modality", "polarity", "condition", "time_point", "feature_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    # Deduplicate per time to avoid repeated iterations
    return df.drop_duplicates(["modality","polarity","condition","time_point","feature_id"]).copy()


def _load_Xy_for_time(root: str, modality: str, polarity: str, condition: str, t: int):
    X_df, y, S, _ = make_dataset(
        root=root,
        conditions=[condition, "MOCK"],
        times=[t],
        omics=[modality],
        polarity=[polarity],
        label_col="condition",
    )
    return X_df, y


def _preprocess_matrix(X_df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    pre = make_preprocess_pipeline(imputer="two_step_label_agnostic", scale="standard")
    return pre.fit_transform(X_df.values, y)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    if X.shape[1] < 2:
        raise ValueError("Need at least 2 features to run 2D PCA.")
    return PCA(n_components=2, random_state=0).fit_transform(X)


def _grid(n: int) -> Tuple[int, int]:
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / cols))
    return rows, cols


def _scatter_ax(ax, Z: np.ndarray, y: np.ndarray, condition: str, t: int):
    y = np.asarray(y)
    pos = (y == condition)
    ax.scatter(Z[pos, 0], Z[pos, 1], label=condition, alpha=0.85)
    ax.scatter(Z[~pos, 0], Z[~pos, 1], label="MOCK", alpha=0.85, marker="x")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"t={t}")

# ------------------------------- API ----------------------------------------

def pca_intersection(
    *,
    csv_path: str,
    root: str = "parquets",
    modality: str,
    polarity: str,
    condition: str,
    times: Optional[Iterable[int]] = None,
    outdir: str = "results",
    min_features: int = 2,
    return_fig: bool = False,
):
    """Single-figure PCA using the **intersection** of selected features across times.

    Returns the saved figure path. If `return_fig=True`, also returns the figure.
    """
    df = load_selections(csv_path)
    g = df[(df.modality == modality) & (df.polarity == polarity) & (df.condition == condition)]
    if g.empty:
        print("No selections for this group.")
        return None if not return_fig else (None, None)

    requested_times = list(times) if times is not None else TIME_ORDER
    sets: Dict[int, set] = {}
    for t in requested_times:
        sub = g[g.time_point == int(t)]
        if not sub.empty:
            sets[int(t)] = set(sub.feature_id.astype(str).unique())
    if not sets:
        print("No time points with selections for this group.")
        return None if not return_fig else (None, None)

    inter = set.intersection(*sets.values()) if len(sets) > 1 else next(iter(sets.values()))
    if len(inter) < min_features:
        print(f"Intersection too small ({len(inter)} < {min_features}); aborting.")
        return None if not return_fig else (None, None)

    panels: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for t in sorted(sets.keys(), key=lambda x: TIME_ORDER.index(x) if x in TIME_ORDER else x):
        X_df, y = _load_Xy_for_time(root, modality, polarity, condition, t)
        cols = [c for c in X_df.columns if str(c) in inter]
        if len(cols) < min_features:
            print(f"t={t}: only {len(cols)} of {len(inter)} intersection feats present; skip.")
            continue
        X_sub = X_df.loc[:, cols]
        try:
            Xp = _preprocess_matrix(X_sub, y)
            Z = _pca_2d(Xp)
            panels.append((t, Z, y))
        except Exception as e:
            print(f"t={t}: PCA failed: {e}")

    if not panels:
        print("Nothing to plot after alignment/preprocess.")
        return None if not return_fig else (None, None)

    rows, cols = _grid(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0*cols, 4.0*rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (t, Z, y) in zip(axes, panels):
        _scatter_ax(ax, Z, y, condition, t)
    for ax in axes[len(panels):]:
        ax.axis('off')

    fig.suptitle(f"PCA on intersection — {modality} {polarity} — {condition}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])

    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f"intersection_{sanitize(modality)}_{sanitize(polarity)}_{sanitize(condition)}.png")
    fig.savefig(fpath, dpi=150)
    if return_fig:
        return fpath, fig
    plt.close(fig)
    return fpath


def pca_seed(
    *,
    csv_path: str,
    root: str = "parquets",
    modality: str,
    polarity: str,
    condition: str,
    seed_time: int,
    times: Optional[Iterable[int]] = None,
    outdir: str = "results",
    min_features: int = 2,
    return_fig: bool = False,
):
    """Single-figure PCA using the feature set selected at seed time t* across times.

    Returns the saved figure path. If `return_fig=True`, also returns the figure.
    """
    df = load_selections(csv_path)
    g = df[(df.modality == modality) & (df.polarity == polarity) & (df.condition == condition)]
    if g.empty:
        print("No selections for this group.")
        return None if not return_fig else (None, None)

    tstar = int(seed_time)
    feats_t = set(g.loc[g.time_point == tstar, "feature_id"].astype(str).unique())
    if len(feats_t) < min_features:
        print(f"Seed set at t*={tstar} too small ({len(feats_t)} < {min_features}); aborting.")
        return None if not return_fig else (None, None)

    requested_times = list(times) if times is not None else TIME_ORDER
    times_present = [int(t) for t in requested_times if not g[g.time_point == int(t)].empty]
    if not times_present:
        print("No time points with selections for this group.")
        return None if not return_fig else (None, None)

    panels: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for t in sorted(times_present, key=lambda x: TIME_ORDER.index(x) if x in TIME_ORDER else x):
        X_df, y = _load_Xy_for_time(root, modality, polarity, condition, t)
        cols = [c for c in X_df.columns if str(c) in feats_t]
        if len(cols) < min_features:
            print(f"t={t}: only {len(cols)} of {len(feats_t)} seed feats present; skip.")
            continue
        X_sub = X_df.loc[:, cols]
        try:
            Xp = _preprocess_matrix(X_sub, y)
            Z = _pca_2d(Xp)
            panels.append((t, Z, y))
        except Exception as e:
            print(f"t={t}: PCA failed: {e}")

    if not panels:
        print("Nothing to plot after alignment/preprocess.")
        return None if not return_fig else (None, None)

    rows, cols = _grid(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0*cols, 4.0*rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (t, Z, y) in zip(axes, panels):
        _scatter_ax(ax, Z, y, condition, t)
    for ax in axes[len(panels):]:
        ax.axis('off')

    fig.suptitle(f"PCA on seed t*={tstar} — {modality} {polarity} — {condition}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])

    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f"seed_{sanitize(modality)}_{sanitize(polarity)}_{sanitize(condition)}_seed{tstar}.png")
    fig.savefig(fpath, dpi=150)
    if return_fig:
        return fpath, fig
    plt.close(fig)
    return fpath

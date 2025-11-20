#!/usr/bin/env python3
"""
Feature churn visualization from iterative L1-SVM selections.

Input CSV (from iterative_l1_svm_selection.py):
  columns: modality, polarity, condition, time_point, feature_id, iteration, mean_cv_accuracy

For each (modality, polarity, condition), we track the selected feature set at each
available time_point and compute, relative to the PREVIOUS observed time_point:
  • same    = |curr ∩ prev|
  • new     = |curr \ prev|
  • dropped = |prev \ curr|

We then produce:
  1) A CSV summary: results/out_csvs/churn_{csv_name}.csv
  2) Stacked bar charts per group under results/out_plots/feature_churn_plots/{csv_name}
where csv_name is the name of the feature set .csv

Notes:
  - Time ordering is enforced as [0, 5, 7, 14, 21]. Missing time points are skipped.
  - For the first observed time point in a group, same=0, dropped=0, new=|curr|.
  - Matplotlib only, no seaborn; no explicit colors are set.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

TIME_ORDER = [0, 5, 7, 14, 21]


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
    return df


def sets_by_time(df_group: pd.DataFrame) -> List[Tuple[int, set]]:
    """Return list of (time_point, set_of_features) sorted by TIME_ORDER, skipping missing times."""
    out: List[Tuple[int, set]] = []
    for t in TIME_ORDER:
        sub = df_group[df_group["time_point"] == t]
        if sub.empty:
            continue
        feats = set(sub["feature_id"].astype(str).unique())
        out.append((t, feats))
    return out


def churn_rows_for_group(df_group: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    seq = sets_by_time(df_group)
    if not seq:
        return rows

    # First observed time
    t0, s0 = seq[0]
    rows.append({
        "modality": df_group["modality"].iloc[0],
        "polarity": df_group["polarity"].iloc[0],
        "condition": df_group["condition"].iloc[0],
        "time_point": t0,
        "same": 0,
        "new": len(s0),
        "dropped": 0,
        "current_n": len(s0),
        "previous_n": 0,
    })

    prev_t, prev_s = t0, s0
    for t, s in seq[1:]:
        same = len(prev_s & s)
        new = len(s - prev_s)
        dropped = len(prev_s - s)
        rows.append({
            "modality": df_group["modality"].iloc[0],
            "polarity": df_group["polarity"].iloc[0],
            "condition": df_group["condition"].iloc[0],
            "time_point": t,
            "same": same,
            "new": new,
            "dropped": dropped,
            "current_n": len(s),
            "previous_n": len(prev_s),
        })
        prev_t, prev_s = t, s
    return rows


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for (mod, pol, cond), g in df.groupby(["modality", "polarity", "condition"], sort=False):
        rows.extend(churn_rows_for_group(g))
    if rows:
        out = pd.DataFrame(rows)
        out.sort_values(["modality", "polarity", "condition", "time_point"], inplace=True)
    else:
        out = pd.DataFrame(columns=["modality","polarity","condition","time_point","same","new","dropped","current_n","previous_n"])
    return out


def plot_group_bars(summary: pd.DataFrame, outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []

    for (mod, pol, cond), g in summary.groupby(["modality","polarity","condition"], sort=False):
        # Ensure time order
        g = g.set_index("time_point").reindex(TIME_ORDER).dropna(how="all").reset_index()
        if g.empty:
            continue
        x = g["time_point"].astype(int).to_numpy()
        same = g["same"].fillna(0).to_numpy()
        new = g["new"].fillna(0).to_numpy()
        dropped = g["dropped"].fillna(0).to_numpy()

        fig, ax = plt.subplots(figsize=(8, 4))
        # Stacked bars for current timepoint relative to previous
        ax.bar(x, same, width=0.9, label="same")
        ax.bar(x, new, bottom=same, width=0.9, label="new")
        ax.bar(x, dropped, bottom=same+new, width=0.9, label="dropped")

        ax.set_title(f"Feature churn: {mod} {pol} — {cond}")
        ax.set_xlabel("time point")
        ax.set_ylabel("count")
        ax.legend()
        ax.set_xticks(x)

        fname = f"feature_churn_{sanitize(mod)}_{sanitize(pol)}_{sanitize(cond)}.png"
        fpath = os.path.join(outdir, fname)
        fig.tight_layout()
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        saved.append(fpath)

    return saved

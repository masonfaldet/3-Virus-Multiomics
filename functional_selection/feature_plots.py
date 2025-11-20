import os
import re
import numpy as np
import matplotlib.pyplot as plt

from utilities.query_and_preprocess_v1 import make_dataset, Log2WithPseudocount
from functional_selection_v1 import build_curves_for_feature

# -------------------------- Plotting functions ---------------------------------
def plot_feature(
    root,
    modality,
    polarity,
    virus,
    feature_id,
    mock_label="MOCK",
    sub_dir = "features",
    time_points=(0, 5, 7, 14, 21),
    weighted_l2=None,
):
    """
    Plot per-sample time-course for a single feature, comparing virus vs MOCK.

    Parameters
    ----------
    root : str
        Root directory containing the Parquet star-schema tables.
    modality : str
        Omics modality (e.g. "lipid" or "metabolite").
    polarity : str
        Ionization polarity ("+" or "-").
    virus : str
        Virus label to compare against mock_label (e.g. "CHIKV").
    feature_id : str
        Feature identifier (e.g. "LN_246", "MP_993") matching a column in abundances.
    mock_label : str, default "MOCK"
        Label for the mock/control condition.
    time_points : iterable of int, default (0, 5, 7, 14, 21)
        Time points to request from make_dataset.
    weighted_l2 : float or None, default None
        Optional weighted L2 distance for this feature (e.g. from functional
        selection). If provided and finite, it is included in the plot title.

    Behavior
    --------
    - Queries abundance data for the given (modality, polarity) and {virus, MOCK}.
    - Applies log2(+eps) transform via Log2WithPseudocount.
    - Drops NaN abundances at the sample level and uses only non-NaN values.
    - Creates a scatter plot of (time_point, log-abundance) for each sample,
      colored by condition.
    - Overlays median curves a_f^{MOCK}(t) and a_f^{virus}(t).
    - Saves the plot to:
        results/out_plots/{virus}_{feature_id}.png

    Returns
    -------
    str or None
        Path to the saved PNG, or None if plotting was skipped (e.g. no data).
    """


    # Query data for virus vs mock across the requested time points
    try:
        X_df, y, S, F = make_dataset(
            root=root,
            conditions=[virus, mock_label],
            times=list(time_points),
            omics=[modality],
            polarity=[polarity],
            label_col="condition",
        )
    except Exception as e:
        print(f"[plot_feature] make_dataset failed for {feature_id}: {e}")
        return None

    if X_df.empty:
        print(f"[plot_feature] Empty dataset for feature {feature_id}; skipping.")
        return None

    col = str(feature_id)
    if col not in X_df.columns:
        print(f"[plot_feature] Feature {col} not found in X_df; available columns: {len(X_df.columns)}")
        return None

    # Align sample metadata with X_df rows
    try:
        S_idx = S.set_index("sample_id").loc[X_df.index]
    except Exception:
        # Fallback: assume S is already aligned
        S_idx = S.loc[X_df.index]

    cond = S_idx["condition"].to_numpy()
    times = S_idx["time_point"].to_numpy()

    # Extract raw abundances and apply log2(+eps)
    x_raw = X_df[col].to_numpy(dtype=float).reshape(-1, 1)
    log2_tf = Log2WithPseudocount(fallback_eps=1e-3)
    x_log = log2_tf.fit_transform(x_raw).ravel()

    # Drop NaNs: only use samples with finite log-abundances
    mask_valid = np.isfinite(x_log)
    if not np.any(mask_valid):
        print(f"[plot_feature] All values NaN for feature {feature_id}; skipping.")
        return None

    x_use = x_log[mask_valid]
    cond_use = cond[mask_valid]
    times_use = times[mask_valid]

    # Build time grid from actually observed times (after NaN filtering)
    unique_times = sorted(set(int(t) for t in times_use))
    if not unique_times:
        print(f"[plot_feature] No time points for feature {feature_id}; skipping.")
        return None
    times_grid = np.array(unique_times, dtype=int)

    # Median curves using the same helper as functional selection
    a_mock, a_virus = build_curves_for_feature(
        x=x_use,
        cond=cond_use,
        times=times_use,
        times_grid=times_grid,
        mock_label=mock_label,
        virus_label=virus,
    )

    # Scatter plot of sample-level observations (only non-NaN)
    fig, ax = plt.subplots(figsize=(6, 4))

    mask_mock = (cond_use == mock_label)
    mask_virus = (cond_use == virus)

    if np.any(mask_mock):
        ax.scatter(
            times_use[mask_mock],
            x_use[mask_mock],
            alpha=0.7,
            label=f"{mock_label} samples",
        )
    if np.any(mask_virus):
        ax.scatter(
            times_use[mask_virus],
            x_use[mask_virus],
            alpha=0.7,
            marker="x",
            label=f"{virus} samples",
        )

    # Overlay median curves (NaNs in curves are okay; matplotlib will break the line)
    ax.plot(
        times_grid,
        a_mock,
        marker="o",
        linestyle="-",
        label=f"{mock_label} median",
    )
    ax.plot(
        times_grid,
        a_virus,
        marker="s",
        linestyle="-",
        label=f"{virus} median",
    )

    ax.set_xlabel("time_point")
    ax.set_ylabel("log2 abundance")

    # Title (optionally include weighted L2)
    import math as _math
    if (weighted_l2 is not None) and _math.isfinite(weighted_l2):
        ax.set_title(
            f"{feature_id} — {modality} {polarity} "
            f"({virus} vs {mock_label}), L2 = {weighted_l2:.3f}"
        )
    else:
        ax.set_title(f"{feature_id} — {modality} {polarity} ({virus} vs {mock_label})")

    ax.legend()
    fig.tight_layout()

    # Sanitize filename a bit
    def _sanitize(s):
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))

    outdir = os.path.join("results", "out_plots", sub_dir)
    os.makedirs(outdir, exist_ok=True)
    fname = f"{_sanitize(virus)}_{_sanitize(feature_id)}.png"
    fpath = os.path.join(outdir, fname)

    fig.savefig(fpath, dpi=150)
    plt.close(fig)

    print(f"[plot_feature] Saved plot to: {fpath}")
    return fpath

def plot_top_k_features(
    root,
    sf_df,
    modality,
    polarity,
    virus,
    k,
    sub_dir,
    max_nan=None,
    mock_label="MOCK",
):
    f"""
    Plot the top-k features (by weighted L2 distance) within a given block.

    Parameters
    ----------
    root : str
        Root directory containing the Parquet star-schema tables.
    sub_dir : str
        pngs will be written to results/out_plots/sub_dir/.
    sf_df : pd.DataFrame
        DataFrame produced by `run_functional_selection` or
        `fdr_filter_functional_results`, with at least columns:
        ["feature_id", "modality", "polarity", "virus",
         "weighted_l2", "p_value", "n_NaNs"].
    modality : str
        Modality to filter on (e.g. "lipid" or "metabolite").
    polarity : str
        Polarity to filter on ("+" or "-").
    virus : str
        Virus label to filter on (e.g. "CHIKV").
    k : int
        Number of top features (by weighted_l2) to plot.
    max_nan : int or None, default None
        If not None, discard candidate features with n_NaNs > max_nan
        before ranking by weighted_l2.
    mock_label : str, default "MOCK"
        Label for the mock/control condition (passed through to plot_feature).

    Returns
    -------
    list of str
        Paths to the saved PNGs for the top-k features (some entries may be
        None if individual plots were skipped).
    """
    import numpy as np

    # Restrict to the specified (virus, modality, polarity) block
    block = sf_df[
        (sf_df["modality"] == modality)
        & (sf_df["polarity"] == polarity)
        & (sf_df["virus"] == virus)
    ].copy()

    if block.empty:
        print(
            f"[plot_top_k_features] No rows found for modality={modality}, "
            f"polarity={polarity}, virus={virus}."
        )
        return []

    # Optional NaN filter: keep only features with n_NaNs <= max_nan
    if max_nan is not None:
        if "n_NaNs" not in block.columns:
            print("[plot_top_k_features] Column 'n_NaNs' not found; cannot apply max_nan filter.")
        else:
            block = block[block["n_NaNs"] <= max_nan]
            if block.empty:
                print(
                    f"[plot_top_k_features] No features remain after n_NaNs <= {max_nan} filter "
                    f"for modality={modality}, polarity={polarity}, virus={virus}."
                )
                return []

    # Drop rows with non-finite weighted_l2 and sort descending
    w = block["weighted_l2"].to_numpy(dtype=float)
    finite_mask = np.isfinite(w)
    block = block.loc[finite_mask]
    if block.empty:
        print(
            f"[plot_top_k_features] All weighted_l2 values are NaN/inf for "
            f"modality={modality}, polarity={polarity}, virus={virus}."
        )
        return []

    block_sorted = block.sort_values("weighted_l2", ascending=False)
    top = block_sorted.head(int(k))

    paths = []
    for _, row in top.iterrows():
        fid = str(row["feature_id"])
        w2 = float(row["weighted_l2"])
        if not np.isfinite(w2):
            w2 = None

        path = plot_feature(
            root=root,
            modality=modality,
            polarity=polarity,
            virus=virus,
            feature_id=fid,
            mock_label=mock_label,
            weighted_l2=w2,
            sub_dir=f"{sub_dir}_nan_{max_nan}"
        )
        paths.append(path)

    return paths

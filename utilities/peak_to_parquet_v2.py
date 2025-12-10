#!/usr/bin/env python3
"""
Build Parquet datasets (samples, features, abundances) from four peak-list CSVs
(one per modality/polarity) with headers like:

  datafile:{yyddmm}_3V_{Omic}{Polarity}B{Batch}_{inj_order}_{virus}{time}_{index}:{data}

Where:
  Omic ∈ {L (lipid), M (metabolite)}
  Polarity ∈ {P(+), N(-)}
  virus ∈ {ZIKV, MOCK, DENV, CHIKV}
  time ∈ {0,5,7,14,21}
  data ∈ {area, rt, mz}

Notes
-----
• Column 1 is the per-file unique integer feature id, named "id" (case-insensitive).
• We store blanks as NaN (pandas default). No imputation is done here.
• We create canonical identifiers:
    sample_id  = f"{virus}_{time}_{index}"
    feature_id = f"{omic_abbrev}{polarity}_{raw_id}"  # e.g., LP_1234, MN_987
• Output:
    samples.parquet     : sample_id, condition (virus), time_point (int), titer (float)
    features.parquet    : feature_id, omics, polarity, feature_raw_id, mz, rt, n_obs, peakset_version
    abundances.parquet  : sample_id, feature_id, area, rt, mz, omics, polarity,
                          batch, injection_order, peakset_version

Usage
-----
    python peak_to_parquets_v1.py \
        --data-dir path/to/peaklists \
        --out-dir ./parquets \
        --titer-csv "241203 3 virus mosquito titers format.csv"

"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OMICS_MAP = {"L": "lipid", "M": "metabolite"}
POL_MAP   = {"P": "+", "N": "-"}
OMICS_ABBR = {"lipid": "L", "metabolite": "M"}
POL_ABBR   = {"+": "P", "-": "N"}

# Example header:
# datafile:241203_3V_LPB1_012_ZIKV5_003:area
HEADER_RX = re.compile(
    r"^datafile:(?P<date>\d{6})_3V_"
    r"(?P<omic>[LM])(?P<polarity>[PN])B(?P<batch>\d+)_"
    r"(?P<inj>[A-Za-z0-9]+)_"
    r"(?P<virus>ZIKV|MOCK|DENV|CHIKV)"
    r"(?P<time>0|5|7|14|21)_"
    r"(?P<index>[0-9]+).mzML:"
    r"(?P<data>area|rt|mz)$"
)


def parse_header(col: str) -> Optional[Dict[str, str]]:
    m = HEADER_RX.match(col)
    if not m:
        return None
    d = m.groupdict()
    d["sample_key"] = col.rsplit(":", 1)[0]  # everything before :data
    d["col_data"] = d["data"]
    return d


def find_id_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower() == "id":
            return c
    raise ValueError("Could not find an 'id' column (case-insensitive) in the CSV.")


def _attach_titers(samples: pd.DataFrame, titer_csv_path: str) -> pd.DataFrame:
    """
    Attach a 'titer' column to samples using the titer CSV.

    The titer CSV is expected to have columns:
        Virus, DPI, Sample.Number, Copy.Number

    Mapping:
        Virus         <-> condition / parsed virus from sample_id
        DPI           <-> time_point / parsed time from sample_id
        Sample.Number <-> parsed index from sample_id
        Copy.Number   -> titer

    Samples with no matching entry get titer = NaN.
    """
    if not os.path.exists(titer_csv_path):
        raise FileNotFoundError(f"Titer CSV not found: {titer_csv_path!r}")

    titer_df = pd.read_csv(titer_csv_path)

    # Standardize column names just in case there is whitespace
    titer_df.columns = [c.strip() for c in titer_df.columns]

    required_cols = {"Virus", "DPI", "Sample.Number", "Copy.Number"}
    missing = required_cols - set(titer_df.columns)
    if missing:
        raise ValueError(
            f"Titer CSV is missing required columns: {sorted(missing)}; "
            f"found columns: {list(titer_df.columns)}"
        )

    # Keep only the columns we care about
    titer_df = titer_df[["Virus", "DPI", "Sample.Number", "Copy.Number"]].copy()

    # Coerce to numeric where appropriate
    titer_df["DPI"] = pd.to_numeric(titer_df["DPI"], errors="coerce")
    titer_df["Sample.Number"] = pd.to_numeric(titer_df["Sample.Number"], errors="coerce")
    titer_df["Copy.Number"] = pd.to_numeric(titer_df["Copy.Number"], errors="coerce")

    # Drop rows that can't be matched because of NaNs in key fields
    titer_df = titer_df.dropna(subset=["Virus", "DPI", "Sample.Number"])

    # Rename to match join keys we'll build from sample_id
    titer_df = titer_df.rename(
        columns={
            "Virus": "virus_join",
            "DPI": "time_join",
            "Sample.Number": "index_join",
            "Copy.Number": "titer",
        }
    )

    # If there are multiple rows per (virus, time, index), aggregate (mean is arbitrary but defined)
    titer_df = (
        titer_df[["virus_join", "time_join", "index_join", "titer"]]
        .groupby(["virus_join", "time_join", "index_join"], as_index=False)
        .agg({"titer": "mean"})
    )

    # Parse sample_id -> (virus, time, index)
    # sample_id is defined as f"{virus}_{time}_{index}"
    parts = samples["sample_id"].str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError(
            "Unexpected sample_id format; expected exactly two '_' characters "
            "as in 'ZIKV_5_003'."
        )

    samples = samples.copy()
    samples["virus_join"] = parts[0]
    samples["time_join"] = pd.to_numeric(parts[1], errors="coerce")
    samples["index_join"] = pd.to_numeric(parts[2], errors="coerce")

    # Left join: samples without titers keep NaN
    samples = samples.merge(
        titer_df,
        on=["virus_join", "time_join", "index_join"],
        how="left",
    )

    # Clean up helper columns
    samples = samples.drop(columns=["virus_join", "time_join", "index_join"])

    # Ensure titer is float
    if "titer" in samples.columns:
        samples["titer"] = samples["titer"].astype("float64")
    else:
        samples["titer"] = np.nan

    return samples


def build_parquets(data_dir: str, out_dir: str, titer_csv_path: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    long_rows: List[pd.DataFrame] = []

    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir!r}")

    for path in csv_paths:
        df = pd.read_csv(
            path,
            dtype={"id": "Int64"},  # if exact name; otherwise adjusted below
            keep_default_na=True,
            na_values=["", "NA", "NaN"],
        )
        id_col = find_id_col(df)

        # Collect all header parses
        parsed = [parse_header(c) for c in df.columns]
        parsed = [p for p in parsed if p is not None]
        if not parsed:
            print(f"[WARN] No matching datafile:* columns in {path}")
            continue

        meta_df = pd.DataFrame(parsed)
        # Group by unique sample_key -> (area, rt, mz) columns per sample
        for sample_key, g in meta_df.groupby("sample_key"):
            cols = {
                rec["col_data"]: f"{sample_key}:{rec['col_data']}"
                for rec in g.to_dict("records")
            }

            # Ensure at least 'area' exists; rt/mz may be absent in some exports
            if "area" not in cols:
                # If no area, skip this sample
                # (You can relax this if you want to keep rows with only rt/mz.)
                continue

            use_cols = [id_col] + [
                c for c in [cols.get("area"), cols.get("rt"), cols.get("mz")] if c
            ]
            sub = df[use_cols].copy()

            # Normalize column names to ['area','rt','mz'] when present
            rename_map = {cols[k]: k for k in ["area", "rt", "mz"] if k in cols}
            sub.rename(columns=rename_map, inplace=True)

            # Pull one representative metadata row from this sample_key
            meta = g.iloc[0].to_dict()
            sample_id = f"{meta['virus']}_{meta['time']}_{meta['index']}"
            omics = OMICS_MAP[meta["omic"]]
            polarity = POL_MAP[meta["polarity"]]

            sub.rename(columns={id_col: "feature_raw_id"}, inplace=True)
            sub["sample_id"] = sample_id
            sub["omics"] = omics
            sub["polarity"] = polarity
            sub["batch"] = (
                int(meta["batch"]) if str(meta["batch"]).isdigit() else meta["batch"]
            )
            sub["injection_order"] = meta["inj"]
            sub["condition"] = meta["virus"]
            sub["time_point"] = (
                int(meta["time"]) if str(meta["time"]).isdigit() else meta["time"]
            )
            sub["peakset_version"] = meta["date"]

            long_rows.append(sub)

    if not long_rows:
        raise RuntimeError("No data columns were parsed from the provided CSVs.")

    abund = pd.concat(long_rows, ignore_index=True)

    # Construct globally-unique feature_id per modality/polarity
    omic_abbrev = abund["omics"].map(OMICS_ABBR)
    pol_abbrev = abund["polarity"].map(POL_ABBR)
    abund["feature_id"] = (
        omic_abbrev + pol_abbrev + "_" + abund["feature_raw_id"].astype(str)
    )

    # Dtypes
    for c in ["area", "rt", "mz"]:
        if c in abund:
            abund[c] = abund[c].astype("float32")
    abund["sample_id"] = abund["sample_id"].astype("string")
    abund["feature_id"] = abund["feature_id"].astype("string")
    abund["omics"] = abund["omics"].astype("category")
    abund["polarity"] = abund["polarity"].astype("category")

    # ---- Build outputs ------------------------------------------------------
    # 1) samples.parquet (unique across modalities)
    samples = (
        abund[["sample_id", "condition", "time_point"]]
        .drop_duplicates()
        .sort_values(["condition", "time_point", "sample_id"])
        .reset_index(drop=True)
    )

    # Attach virus titers (titer column)
    samples = _attach_titers(samples, titer_csv_path=titer_csv_path)

    # 2) features.parquet (canonical mz/rt per feature via median across samples)
    # Ensure mz/rt exist even if absent in some files
    for c in ["mz", "rt"]:
        if c not in abund.columns:
            abund[c] = np.nan

    group_cols = ["feature_id", "omics", "polarity", "peakset_version"]
    # Use a robust named-aggregation pattern compatible across pandas versions
    features = (
        abund.groupby(group_cols, observed=True)
        .agg(
            feature_raw_id=("feature_raw_id", "first"),
            n_obs=("area", lambda s: s.notna().sum()),
            mz=("mz", "median"),
            rt=("rt", "median"),
        )
        .reset_index()
    )

    # 3) abundances.parquet (long, per sample-feature)
    keep_cols = [
        "sample_id",
        "feature_id",
        "area",
        "rt",
        "mz",
        "omics",
        "polarity",
        "batch",
        "injection_order",
        "peakset_version",
    ]
    keep_cols = [c for c in keep_cols if c in abund.columns]
    abundances = abund[keep_cols].copy()

    # Enforce uniqueness per (sample_id, feature_id)
    dup_mask = abundances.duplicated(subset=["sample_id", "feature_id"], keep=False)
    if dup_mask.any():
        # If duplicates exist, keep the first and warn.
        dup_count = int(dup_mask.sum())
        print(
            f"[WARN] Found {dup_count} duplicate (sample_id, feature_id) rows. "
            "Keeping first occurrences."
        )
        abundances = abundances.drop_duplicates(
            subset=["sample_id", "feature_id"], keep="first"
        )

    # ---- Write Parquets -----------------------------------------------------
    samples.to_parquet(
        os.path.join(out_dir, "samples.parquet"),
        compression="zstd",
        index=False,
    )
    features.to_parquet(
        os.path.join(out_dir, "features.parquet"),
        compression="zstd",
        index=False,
    )
    abundances.to_parquet(
        os.path.join(out_dir, "abundances.parquet"),
        compression="zstd",
        index=False,
    )

    # ---- Summary ------------------------------------------------------------
    print("\nWrote:")
    print(f"  samples.parquet    : {len(samples):>6} rows")
    print(f"  features.parquet   : {len(features):>6} rows")
    print(f"  abundances.parquet : {len(abundances):>6} rows")

    print("\nExamples:")
    print(samples.head(3).to_string(index=False))
    print(features.head(3).to_string(index=False))
    print(abundances.head(3).to_string(index=False))


if __name__ == "__main__":
    build_parquets(
        "../peaklists",
        "../parquets",
        "../3_Virus_Data/titer.csv"
    )

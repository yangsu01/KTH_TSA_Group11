"""
Step 1 – Data Loading & Preprocessing
======================================
Responsibility
--------------
Extract the Austria (AT) actual load column from the raw ENTSO-E CSV,
resample hourly values to daily means, handle missing values, and save a
clean series that every downstream step can load with a single pd.read_csv.

Person 1 owns this file.

Input
-----
  config.RAW_DATA_PATH  →  data_hourly.csv  (raw, ~50 k rows, semi-colon separated)

Output
------
  config.AT_LOAD_DAILY_PATH  →  data/processed/at_load_daily.csv

    Columns : date       – calendar date, YYYY-MM-DD (str)
              load_mw    – mean daily load in MW (float, no NaNs)
    Rows    : one per calendar day from TRAIN_START to VAL_END (~2192 rows)

Run
---
  python src/step1_load_data.py

Test
----
  pytest tests/test_step1.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Make project root importable regardless of where the script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Public API (used by tests and downstream steps) ───────────────────────────

def find_column_index(raw_path: str | os.PathLike) -> int:
    """
    Scan the first three header rows of the raw CSV to find the column index
    that matches (TARGET_COUNTRY, TARGET_VARIABLE, TARGET_TYPE).

    Parameters
    ----------
    raw_path : path to data_hourly.csv

    Returns
    -------
    int  –  0-based column index of the target series

    Raises
    ------
    ValueError  if the column is not found
    """
    with open(raw_path, "r", encoding="utf-8") as f:
        rows = [f.readline().strip().split(";") for _ in range(3)]

    countries, variables, meas_types = rows[0], rows[1], rows[2]

    for i, (c, v, m) in enumerate(zip(countries, variables, meas_types)):
        if (c == config.TARGET_COUNTRY
                and v == config.TARGET_VARIABLE
                and m == config.TARGET_TYPE):
            return i

    raise ValueError(
        f"Column not found for "
        f"{config.TARGET_COUNTRY}/{config.TARGET_VARIABLE}/{config.TARGET_TYPE}. "
        f"Check that config.py matches the CSV header."
    )


def load_and_resample(raw_path: str | os.PathLike) -> pd.Series:
    """
    Load the AT actual load column from the raw CSV and resample to daily means.

    Steps performed
    ---------------
    1. Locate the correct column via find_column_index().
    2. Parse the utc_timestamp column as UTC-aware datetimes.
    3. Coerce load values to float (handles European comma decimals).
    4. Filter to [TRAIN_START, VAL_END].
    5. Resample hourly → daily mean.
    6. Fill any remaining NaNs via linear interpolation.

    Parameters
    ----------
    raw_path : path to data_hourly.csv

    Returns
    -------
    pd.Series  –  daily mean load in MW, DatetimeIndex (UTC), name='load_mw'
                  index frequency 'D', no NaNs
    """
    col_idx = find_column_index(raw_path)

    # The raw CSV has 7 header rows before data:
    #   row 0 : country
    #   row 1 : variable
    #   row 2 : measurement type
    #   row 3 : source description
    #   row 4 : URL
    #   row 5 : unit
    #   row 6 : column names  (utc_timestamp, cet_cest_timestamp, ...)
    # We skip rows 1–5 and use row 6 as the header.
    df = pd.read_csv(
        raw_path,
        sep=";",
        skiprows=[1, 2, 3, 4, 5],   # keep row 0 as header
        header=0,
        usecols=[0, col_idx],
        na_values=["", "n/e", "N/E", "nan"],
        low_memory=False,
    )

    # Normalise column names
    df.columns = ["utc_timestamp", "load_mw"]

    # Parse timestamps (UTC-aware)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce",
                                          format="mixed")
    df = df.dropna(subset=["utc_timestamp"])
    df = df.set_index("utc_timestamp").sort_index()

    # Coerce load values – handle European decimal commas
    df["load_mw"] = (
        df["load_mw"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Filter to analysis window
    df = df.loc[config.TRAIN_START : config.VAL_END]

    # Resample to daily mean (MW is a power quantity → mean is correct)
    daily: pd.Series = df["load_mw"].resample("D").mean()
    daily.name = "load_mw"

    # Fill gaps (public holidays, data outages) via linear interpolation
    n_missing = daily.isna().sum()
    if n_missing > 0:
        print(f"[step1] Interpolating {n_missing} missing daily values.")
        daily = daily.interpolate(method="linear").ffill().bfill()

    return daily


def save_daily(series: pd.Series, output_path: str | os.PathLike) -> None:
    """
    Persist the daily series to CSV.

    Output schema
    -------------
    date      : YYYY-MM-DD string
    load_mw   : float
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = series.reset_index()
    df.columns = ["date", "load_mw"]
    df["date"] = df["date"].dt.date.astype(str)
    df.to_csv(output_path, index=False)
    print(f"[step1] Saved {len(df)} rows -> {output_path}")


# ── Helper for downstream steps ───────────────────────────────────────────────

def load_daily(path: str | os.PathLike = config.AT_LOAD_DAILY_PATH) -> pd.Series:
    """
    Convenience loader used by steps 2–5.

    Returns
    -------
    pd.Series  –  daily load in MW, DatetimeIndex, name='load_mw'
    """
    df = pd.read_csv(path, parse_dates=["date"])
    series = df.set_index("date")["load_mw"]
    series.index.freq = pd.infer_freq(series.index)
    return series


def get_train_val_split(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Split a daily series into training and validation portions using the
    date boundaries defined in config.py.

    Returns
    -------
    (train, val)  –  two pd.Series
    """
    train = series.loc[config.TRAIN_START : config.TRAIN_END]
    val   = series.loc[config.VAL_START   : config.VAL_END]
    return train, val


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[step1] Loading and resampling AT load data …")
    series = load_and_resample(config.RAW_DATA_PATH)
    save_daily(series, config.AT_LOAD_DAILY_PATH)

    train, val = get_train_val_split(series)
    print(f"[step1] Training days : {len(train)}  ({config.TRAIN_START} – {config.TRAIN_END})")
    print(f"[step1] Validation days: {len(val)}   ({config.VAL_START} – {config.VAL_END})")
    print(series.describe().to_string())

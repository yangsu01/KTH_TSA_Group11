"""
Step 2 – Stationarity Analysis & Preprocessing
================================================
Responsibility
--------------
Take the clean daily load series from step 1 and produce a stationary series
suitable for ARMA fitting.  Document every transformation so that step 5 can
reverse them when mapping forecasts back to MW.

Person 2 owns this file.

Input
-----
  config.AT_LOAD_DAILY_PATH  →  data/processed/at_load_daily.csv

Output
------
  config.AT_LOAD_STATIONARY_PATH  →  data/processed/at_load_stationary.csv

    Columns : date             – YYYY-MM-DD (str)
              load_stationary  – stationary series (float, no NaNs)

  config.DECOMPOSITION_PATH  →  results/decomposition.json

    Schema (all fields must be present; set to null if not applicable):
    {
      "transformations"     : ["log"],            // ordered list, e.g. ["log","seasonal_diff","diff"]
      "trend_type"          : "linear",           // "none" | "linear" | "polynomial" | "moving_average"
      "trend_coefficients"  : [a, b],             // [intercept, slope] on integer t=0,1,…
      "seasonal_period"     : 365,                // null if no seasonal component removed
      "seasonal_component"  : [...],              // array of length seasonal_period, or null
      "diff_orders"         : [1],                // list of differencing lags applied (e.g. [365,1])
      "train_end"           : "2019-12-31",
      "val_start"           : "2020-01-01",
      "val_end"             : "2020-12-31",
      "n_train"             : 1826,
      "n_val"               : 366
    }

  Plots saved to config.PLOTS_DIR:
    step2_original_series.png
    step2_decomposition.png
    step2_stationary_series.png
    step2_adf_kpss_summary.png   (optional – can be a text output instead)

Run
---
  python src/step2_stationarity.py

Test
----
  pytest tests/test_step2.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.step1_load_data import load_daily, get_train_val_split


# ── Public API ────────────────────────────────────────────────────────────────

def plot_original(series: pd.Series) -> None:
    """
    Plot the raw daily load series and save to PLOTS_DIR.

    TODO: implement
      - Time-series plot of the full series
      - Mark the train/val split with a vertical line
      - Label axes, add title, save as step2_original_series.png
    """
    raise NotImplementedError("TODO: plot raw series")


def decompose_series(train: pd.Series) -> dict:
    """
    Decompose the training series into trend + seasonal + residual components.
    Use only the training portion to avoid look-ahead bias.

    TODO: implement using one or more of:
      - statsmodels.tsa.seasonal.seasonal_decompose  (additive or multiplicative)
      - statsmodels.tsa.seasonal.STL
      - Manual polynomial / moving-average trend fit

    Parameters
    ----------
    train : pd.Series  –  training portion of the daily load series

    Returns
    -------
    dict with keys:
      "trend"    : pd.Series  –  estimated trend component (same index as train)
      "seasonal" : pd.Series  –  estimated seasonal component, or None
      "residual" : pd.Series  –  train - trend - seasonal (or equivalent)
      "period"   : int | None –  detected seasonal period (e.g. 365 or 7)

    Hint: electricity load typically shows both a weekly (period=7) and annual
    (period=365) seasonality.  You do not have to remove both; motivate your
    choice in the report.
    """
    raise NotImplementedError("TODO: decompose the training series")


def make_stationary(series: pd.Series, decomposition: dict) -> tuple[pd.Series, dict]:
    """
    Apply transformations (log, differencing, detrending …) to produce a
    stationary series.  Document every step in a metadata dict that step 5
    will use to reverse the transformations.

    Parameters
    ----------
    series       : pd.Series  –  full daily series (train + val)
    decomposition: dict       –  output of decompose_series()

    Returns
    -------
    (stationary, meta)
      stationary : pd.Series  –  stationary series, no NaNs
      meta       : dict       –  decomposition.json payload (see schema above)

    TODO: implement
      Suggested order of operations:
        1. Apply log transform if the variance grows with the level.
        2. Remove trend (subtract fitted trend or difference).
        3. Remove seasonal component (subtract or seasonally difference).
        4. Test for remaining unit roots (ADF / KPSS – see test_stationarity).
        5. If not yet stationary, apply first-order differencing.
      Record every applied transformation in meta["transformations"].
    """
    raise NotImplementedError("TODO: make the series stationary")


def test_stationarity(series: pd.Series) -> dict:
    """
    Run Augmented Dickey-Fuller and KPSS tests on the given series.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    dict with keys:
      "adf_statistic"  : float
      "adf_pvalue"     : float
      "adf_conclusion" : "stationary" | "non-stationary"
      "kpss_statistic" : float
      "kpss_pvalue"    : float
      "kpss_conclusion": "stationary" | "non-stationary"
      "conclusion"     : "stationary" | "non-stationary"   # both tests agree

    TODO: implement using
      from statsmodels.tsa.stattools import adfuller, kpss
    """
    raise NotImplementedError("TODO: run ADF and KPSS tests")


def plot_stationary(original: pd.Series, stationary: pd.Series) -> None:
    """
    Side-by-side plot: original series vs stationary series.
    Save to step2_stationary_series.png.

    TODO: implement
    """
    raise NotImplementedError("TODO: plot stationary series")


def save_stationary(series: pd.Series, output_path: str | os.PathLike) -> None:
    """Save the stationary series to CSV (contracts defined in module docstring)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = series.reset_index()
    df.columns = ["date", "load_stationary"]
    df["date"] = df["date"].dt.date.astype(str)
    df.to_csv(output_path, index=False)
    print(f"[step2] Saved {len(df)} rows → {output_path}")


def save_decomposition(meta: dict, output_path: str | os.PathLike) -> None:
    """Serialise the decomposition metadata to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert numpy types for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2, default=_convert)
    print(f"[step2] Saved decomposition → {output_path}")


# ── Helper for downstream steps ───────────────────────────────────────────────

def load_stationary(
    path: str | os.PathLike = config.AT_LOAD_STATIONARY_PATH,
) -> pd.Series:
    """
    Convenience loader used by steps 3–5.

    Returns
    -------
    pd.Series  –  stationary load series, DatetimeIndex, name='load_stationary'
    """
    df = pd.read_csv(path, parse_dates=["date"])
    series = df.set_index("date")["load_stationary"]
    return series


def load_decomposition(
    path: str | os.PathLike = config.DECOMPOSITION_PATH,
) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("[step2] Loading daily data …")
    series = load_daily()
    train, val = get_train_val_split(series)

    print("[step2] Plotting original series …")
    plot_original(series)

    print("[step2] Decomposing training series …")
    decomposition = decompose_series(train)

    print("[step2] Making series stationary …")
    stationary, meta = make_stationary(series, decomposition)

    print("[step2] Testing stationarity …")
    results = test_stationarity(stationary.loc[config.TRAIN_START : config.TRAIN_END])
    print(f"  ADF  : {results['adf_conclusion']}  (p={results['adf_pvalue']:.4f})")
    print(f"  KPSS : {results['kpss_conclusion']} (p={results['kpss_pvalue']:.4f})")

    print("[step2] Plotting stationary series …")
    plot_stationary(series, stationary)

    save_stationary(stationary, config.AT_LOAD_STATIONARY_PATH)
    save_decomposition(meta, config.DECOMPOSITION_PATH)
    print("[step2] Done.")

"""
Step 5 – Forecasting & Evaluation
===================================
Responsibility
--------------
Use the fitted ARMA model to forecast the validation period, reverse all
transformations applied in step 2, and evaluate forecast accuracy against
the true AT load values.  Produce the final plots for the report.

Person 5 owns this file.

Input
-----
  config.FITTED_MODEL_PATH        →  results/fitted_model.pkl
  config.AT_LOAD_STATIONARY_PATH  →  data/processed/at_load_stationary.csv
  config.DECOMPOSITION_PATH       →  results/decomposition.json
  config.AT_LOAD_DAILY_PATH       →  data/processed/at_load_daily.csv  (actuals)

Output
------
  config.FORECAST_PATH  →  results/forecast.csv

    Columns:
      date                 – YYYY-MM-DD (str)
      forecast_stationary  – forecast on the stationary scale (float)
      lower_95             – lower 95 % prediction interval (original MW scale)
      upper_95             – upper 95 % prediction interval (original MW scale)
      forecast_original    – point forecast on the original MW scale
      actual               – true load_mw from at_load_daily.csv

  Plots saved to config.PLOTS_DIR:
    step5_forecast_stationary.png   – forecast vs actual on stationary scale
    step5_forecast_original.png     – forecast vs actual in MW (main report plot)
    step5_prediction_intervals.png  – zoom on first ~60 days with PI ribbon

Run
---
  python src/step5_forecasting.py

Test
----
  pytest tests/test_step5.py
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
from src.step2_stationarity import load_stationary, load_decomposition
from src.step4_model_fitting import load_fitted_model


# ── Public API ────────────────────────────────────────────────────────────────

def forecast_stationary(
    fitted_model,
    n_steps: int,
    alpha: float = 1 - config.CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """
    Produce an h-step-ahead forecast on the stationary scale.

    Parameters
    ----------
    fitted_model : ARIMAResultsWrapper (from step 4)
    n_steps      : int    –  number of steps to forecast (= len(validation set))
    alpha        : float  –  significance level for prediction intervals

    Returns
    -------
    pd.DataFrame with columns:
      "forecast"  – point forecast (float)
      "lower"     – lower PI bound (float)
      "upper"     – upper PI bound (float)
    Index: integer 0..n_steps-1 (dates are attached later)

    TODO: implement
      forecast_result = fitted_model.get_forecast(steps=n_steps)
      summary = forecast_result.summary_frame(alpha=alpha)
      # summary has columns: mean, mean_se, mean_ci_lower, mean_ci_upper
    """
    raise NotImplementedError("TODO: forecast on stationary scale")


def back_transform(
    forecast_df: pd.DataFrame,
    decomposition: dict,
    train_series: pd.Series,
) -> pd.DataFrame:
    """
    Reverse the transformations documented in decomposition.json to map
    the stationary-scale forecast back to the original MW scale.

    Parameters
    ----------
    forecast_df   : pd.DataFrame  –  output of forecast_stationary()
    decomposition : dict          –  loaded from decomposition.json
    train_series  : pd.Series     –  original (non-stationary) training series
                                     needed to initialise differencing reversal

    Returns
    -------
    pd.DataFrame with columns:
      "forecast_original"  – point forecast in MW
      "lower_95"           – lower PI in MW
      "upper_95"           – upper PI in MW

    TODO: implement
      Apply the inverse of each transformation listed in
      decomposition["transformations"] in reverse order.  Common inverses:
        - "diff"           → cumulative sum (np.cumsum), seeded from last train value
        - "seasonal_diff"  → seasonal cumulative sum, seeded from last period values
        - "log"            → np.exp
        - "detrend_linear" → add back trend_coefficients evaluated at forecast times
        - "seasonal_sub"   → add back seasonal_component at the right phase

      Handle prediction intervals consistently (apply the same inverse transform
      to lower and upper bounds).
    """
    raise NotImplementedError("TODO: back-transform forecast to original scale")


def evaluate_forecast(
    forecast_original: pd.Series,
    actual: pd.Series,
) -> dict:
    """
    Compute forecast accuracy metrics.

    Parameters
    ----------
    forecast_original : pd.Series  –  point forecast in MW
    actual            : pd.Series  –  true load values in MW

    Returns
    -------
    dict with keys:
      "mae"   : float  –  Mean Absolute Error (MW)
      "rmse"  : float  –  Root Mean Squared Error (MW)
      "mape"  : float  –  Mean Absolute Percentage Error (%)
      "n_obs" : int

    TODO: implement
    """
    raise NotImplementedError("TODO: compute forecast accuracy metrics")


def plot_forecast_stationary(
    stationary: pd.Series,
    forecast_df: pd.DataFrame,
    val_dates: pd.DatetimeIndex,
) -> None:
    """
    Plot actual vs forecast on the stationary scale.
    Save to step5_forecast_stationary.png.

    TODO: implement
    """
    raise NotImplementedError("TODO: plot stationary forecast")


def plot_forecast_original(
    daily: pd.Series,
    forecast_original: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> None:
    """
    Main report plot: actual load (train + val) vs forecast with 95 % PI ribbon.
    Save to step5_forecast_original.png.

    TODO: implement
      - Plot full training series in one colour
      - Plot validation actuals in another colour
      - Plot point forecast as a dashed line
      - Fill between lower and upper PI
      - Mark the train/val boundary with a vertical line
      - Label axes (Date, Load [MW]), add legend and title
    """
    raise NotImplementedError("TODO: plot original-scale forecast")


def save_forecast(
    dates: pd.DatetimeIndex,
    forecast_df: pd.DataFrame,
    back_transformed: pd.DataFrame,
    actual: pd.Series,
    output_path: str | os.PathLike = config.FORECAST_PATH,
) -> None:
    """Assemble and save forecast.csv."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = pd.DataFrame({
        "date"                : dates.date.astype(str),
        "forecast_stationary" : forecast_df["forecast"].values,
        "forecast_original"   : back_transformed["forecast_original"].values,
        "lower_95"            : back_transformed["lower_95"].values,
        "upper_95"            : back_transformed["upper_95"].values,
        "actual"              : actual.values,
    })
    result.to_csv(output_path, index=False)
    print(f"[step5] Saved {len(result)} forecast rows → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("[step5] Loading inputs …")
    fitted_model  = load_fitted_model()
    stationary    = load_stationary()
    decomposition = load_decomposition()
    daily         = load_daily()
    train_daily, val_daily = get_train_val_split(daily)
    train_stat, val_stat   = get_train_val_split(stationary)

    n_steps = len(val_stat)
    print(f"[step5] Forecasting {n_steps} steps ahead …")

    forecast_df = forecast_stationary(fitted_model, n_steps)
    forecast_df.index = val_stat.index  # attach validation dates

    print("[step5] Back-transforming to MW scale …")
    back = back_transform(forecast_df, decomposition, train_stat)
    back.index = val_daily.index

    print("[step5] Evaluating forecast …")
    metrics = evaluate_forecast(back["forecast_original"], val_daily)
    print(f"  MAE  = {metrics['mae']:.1f} MW")
    print(f"  RMSE = {metrics['rmse']:.1f} MW")
    print(f"  MAPE = {metrics['mape']:.2f} %")

    print("[step5] Saving plots …")
    plot_forecast_stationary(stationary, forecast_df, val_stat.index)
    plot_forecast_original(daily, back["forecast_original"],
                           back["lower_95"], back["upper_95"])

    save_forecast(val_daily.index, forecast_df, back, val_daily)
    print("[step5] Done.")

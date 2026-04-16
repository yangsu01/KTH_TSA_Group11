"""
Tests for Step 5 – Forecasting & Evaluation
=============================================
Run:  pytest tests/test_step5.py

NOTE: Requires outputs from steps 1–4.
      Run all previous steps before running these tests.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.step1_load_data import load_daily, get_train_val_split
from src.step2_stationarity import load_stationary, load_decomposition
from src.step4_model_fitting import load_fitted_model, fit_arma, extract_params
from src.step5_forecasting import (
    forecast_stationary,
    back_transform,
    evaluate_forecast,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily_series():
    if not config.AT_LOAD_DAILY_PATH.exists():
        pytest.skip("Step 1 output not found.")
    return load_daily()


@pytest.fixture(scope="module")
def stationary_series():
    if not config.AT_LOAD_STATIONARY_PATH.exists():
        pytest.skip("Step 2 output not found.")
    return load_stationary()


@pytest.fixture(scope="module")
def decomposition():
    if not config.DECOMPOSITION_PATH.exists():
        pytest.skip("decomposition.json not found.")
    return load_decomposition()


@pytest.fixture(scope="module")
def fitted_model():
    if not config.FITTED_MODEL_PATH.exists():
        pytest.skip("fitted_model.pkl not found.")
    return load_fitted_model()


@pytest.fixture(scope="module")
def forecast_csv():
    if not config.FORECAST_PATH.exists():
        pytest.skip("forecast.csv not found. Run: python src/step5_forecasting.py")
    return pd.read_csv(config.FORECAST_PATH, parse_dates=["date"])


@pytest.fixture(scope="module")
def synthetic_fitted():
    """Fast AR(1) fixture for unit-testing forecast functions."""
    rng = np.random.default_rng(11)
    x = np.zeros(200)
    for t in range(1, 200):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
    series = pd.Series(x)
    model  = fit_arma(series, p=1, q=0)
    return model


# ── forecast_stationary() ────────────────────────────────────────────────────

def test_forecast_returns_dataframe(synthetic_fitted):
    result = forecast_stationary(synthetic_fitted, n_steps=30)
    assert isinstance(result, pd.DataFrame)


def test_forecast_correct_length(synthetic_fitted):
    n = 30
    result = forecast_stationary(synthetic_fitted, n_steps=n)
    assert len(result) == n, f"Expected {n} rows, got {len(result)}"


def test_forecast_required_columns(synthetic_fitted):
    result = forecast_stationary(synthetic_fitted, n_steps=10)
    for col in ("forecast", "lower", "upper"):
        assert col in result.columns, \
            f"forecast_stationary() must return column '{col}'"


def test_forecast_pi_ordering(synthetic_fitted):
    result = forecast_stationary(synthetic_fitted, n_steps=30)
    assert (result["lower"] <= result["forecast"]).all(), \
        "lower PI must be ≤ point forecast"
    assert (result["forecast"] <= result["upper"]).all(), \
        "point forecast must be ≤ upper PI"


def test_forecast_pi_widens_over_horizon(synthetic_fitted):
    """Prediction interval width should be non-decreasing."""
    result = forecast_stationary(synthetic_fitted, n_steps=30)
    widths = result["upper"] - result["lower"]
    assert (widths.diff().dropna() >= -1e-9).all(), \
        "Prediction interval width should be non-decreasing over the horizon"


# ── evaluate_forecast() ───────────────────────────────────────────────────────

def test_evaluate_returns_dict():
    rng = np.random.default_rng(99)
    y_true = pd.Series(rng.uniform(3000, 8000, 100))
    y_pred = y_true + rng.normal(0, 200, 100)
    result = evaluate_forecast(y_pred, y_true)
    assert isinstance(result, dict)


def test_evaluate_required_keys():
    rng = np.random.default_rng(99)
    y_true = pd.Series(rng.uniform(3000, 8000, 100))
    y_pred = y_true + rng.normal(0, 200, 100)
    result = evaluate_forecast(y_pred, y_true)
    for key in ("mae", "rmse", "mape", "n_obs"):
        assert key in result, f"evaluate_forecast() must return key '{key}'"


def test_evaluate_perfect_forecast_zero_error():
    y = pd.Series([5000.0] * 50)
    result = evaluate_forecast(y, y)
    assert result["mae"]  < 1e-9
    assert result["rmse"] < 1e-9
    assert result["mape"] < 1e-9


def test_evaluate_mae_leq_rmse():
    """MAE ≤ RMSE always holds (by Cauchy-Schwarz)."""
    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.uniform(3000, 8000, 200))
    y_pred = y_true + rng.normal(0, 500, 200)
    result = evaluate_forecast(y_pred, y_true)
    assert result["mae"] <= result["rmse"] + 1e-9


def test_evaluate_n_obs_correct():
    n = 80
    y = pd.Series(np.ones(n) * 5000)
    result = evaluate_forecast(y, y)
    assert result["n_obs"] == n


# ── Output file contracts ─────────────────────────────────────────────────────

def test_forecast_csv_exists():
    assert config.FORECAST_PATH.exists(), (
        "forecast.csv not found. Run: python src/step5_forecasting.py"
    )


def test_forecast_csv_columns(forecast_csv):
    expected = {"date", "forecast_stationary", "forecast_original",
                "lower_95", "upper_95", "actual"}
    assert expected.issubset(set(forecast_csv.columns)), \
        f"Missing columns: {expected - set(forecast_csv.columns)}"


def test_forecast_csv_length(forecast_csv):
    assert len(forecast_csv) >= 60, \
        f"Forecast should cover at least 60 days, got {len(forecast_csv)}"


def test_forecast_csv_no_nans(forecast_csv):
    assert forecast_csv.isna().sum().sum() == 0, \
        "forecast.csv must contain no NaN values"


def test_forecast_pi_valid(forecast_csv):
    assert (forecast_csv["lower_95"] <= forecast_csv["forecast_original"]).all()
    assert (forecast_csv["forecast_original"] <= forecast_csv["upper_95"]).all()


def test_forecast_mape_reasonable(forecast_csv):
    """MAPE should be below 50 % – a sanity check, not a quality threshold."""
    mape = (
        (forecast_csv["forecast_original"] - forecast_csv["actual"]).abs()
        / forecast_csv["actual"].abs()
    ).mean() * 100
    assert mape < 50, \
        f"MAPE = {mape:.1f} % is unreasonably high – check back-transformation"


def test_plots_created():
    for name in ("step5_forecast_stationary.png", "step5_forecast_original.png"):
        assert (config.PLOTS_DIR / name).exists(), f"Plot not found: {name}"

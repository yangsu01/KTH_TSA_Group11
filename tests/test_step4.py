"""
Tests for Step 4 – Model Fitting & Diagnostics
================================================
Run:  pytest tests/test_step4.py

NOTE: Requires step 2 and step 3 outputs.
      Run steps 1–3 before running these tests.
"""
import os
import sys
import json
import pickle
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.step1_load_data import get_train_val_split
from src.step2_stationarity import load_stationary
from src.step4_model_fitting import (
    fit_arma,
    extract_params,
    diagnose_residuals,
    load_fitted_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_stationary():
    if not config.AT_LOAD_STATIONARY_PATH.exists():
        pytest.skip("Step 2 output not found.")
    stationary = load_stationary()
    train, _ = get_train_val_split(stationary)
    return train


@pytest.fixture(scope="module")
def synthetic_ar1():
    """Simple AR(1) series for fast unit tests."""
    rng = np.random.default_rng(7)
    n = 300
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()
    return pd.Series(x)


@pytest.fixture(scope="module")
def fitted_model_and_params(synthetic_ar1):
    model  = fit_arma(synthetic_ar1, p=1, q=0)
    params = extract_params(model)
    return model, params


@pytest.fixture(scope="module")
def loaded_fitted_model():
    if not config.FITTED_MODEL_PATH.exists():
        pytest.skip("fitted_model.pkl not found. Run: python src/step4_model_fitting.py")
    return load_fitted_model()


@pytest.fixture(scope="module")
def model_params_json():
    if not config.MODEL_PARAMS_PATH.exists():
        pytest.skip("model_params.json not found. Run: python src/step4_model_fitting.py")
    with open(config.MODEL_PARAMS_PATH) as f:
        return json.load(f)


# ── fit_arma() ────────────────────────────────────────────────────────────────

def test_fit_arma_returns_object(synthetic_ar1):
    model = fit_arma(synthetic_ar1, p=1, q=0)
    assert model is not None


def test_fit_arma_11_returns_object(synthetic_ar1):
    model = fit_arma(synthetic_ar1, p=1, q=1)
    assert model is not None


def test_fit_arma_ar1_recovers_coefficient(synthetic_ar1):
    """Fitted AR(1) coefficient should be close to the true 0.6."""
    model  = fit_arma(synthetic_ar1, p=1, q=0)
    params = extract_params(model)
    ar     = params["ar_params"]
    assert len(ar) == 1, "AR(1) model must have exactly 1 AR parameter"
    assert abs(ar[0] - 0.6) < 0.15, \
        f"AR(1) coefficient {ar[0]:.3f} is too far from true value 0.6"


# ── extract_params() ──────────────────────────────────────────────────────────

def test_extract_params_required_keys(fitted_model_and_params):
    _, params = fitted_model_and_params
    required = [
        "order", "method", "ar_params", "ma_params", "sigma2",
        "aic", "bic", "log_likelihood", "n_obs",
    ]
    for key in required:
        assert key in params, f"extract_params() must return key '{key}'"


def test_extract_params_sigma2_positive(fitted_model_and_params):
    _, params = fitted_model_and_params
    assert params["sigma2"] > 0, "Noise variance σ² must be positive"


def test_extract_params_n_obs_correct(synthetic_ar1, fitted_model_and_params):
    _, params = fitted_model_and_params
    assert params["n_obs"] == len(synthetic_ar1), \
        "n_obs must equal the length of the training series"


def test_extract_params_order_matches(synthetic_ar1):
    model  = fit_arma(synthetic_ar1, p=2, q=1)
    params = extract_params(model)
    assert params["order"]["p"] == 2
    assert params["order"]["q"] == 1
    assert len(params["ar_params"]) == 2
    assert len(params["ma_params"]) == 1


def test_extract_params_aic_lower_than_null(synthetic_ar1, fitted_model_and_params):
    """AR(1) AIC should be lower than a simple Gaussian log-likelihood."""
    _, params = fitted_model_and_params
    assert params["aic"] < 0 or params["aic"] < 10_000, \
        "AIC value looks implausible"


# ── diagnose_residuals() ──────────────────────────────────────────────────────

def test_diagnose_residuals_returns_dict(synthetic_ar1, fitted_model_and_params):
    model, _ = fitted_model_and_params
    result = diagnose_residuals(model, synthetic_ar1)
    assert isinstance(result, dict)


def test_diagnose_residuals_required_keys(synthetic_ar1, fitted_model_and_params):
    model, _ = fitted_model_and_params
    result = diagnose_residuals(model, synthetic_ar1)
    for key in ("ljung_box_lag", "ljung_box_stat", "ljung_box_pvalue", "jb_pvalue"):
        assert key in result, f"diagnose_residuals() must return key '{key}'"


def test_ljung_box_pvalue_range(synthetic_ar1, fitted_model_and_params):
    model, _ = fitted_model_and_params
    result = diagnose_residuals(model, synthetic_ar1)
    assert 0.0 <= result["ljung_box_pvalue"] <= 1.0


def test_ar1_residuals_are_white_noise(synthetic_ar1):
    """Residuals of a correctly specified AR(1) should pass Ljung-Box (p>0.05)."""
    model  = fit_arma(synthetic_ar1, p=1, q=0)
    result = diagnose_residuals(model, synthetic_ar1)
    assert result["ljung_box_pvalue"] > 0.05, (
        f"AR(1) residuals failed Ljung-Box test "
        f"(p={result['ljung_box_pvalue']:.4f}). "
        "Consider a higher order model."
    )


# ── Output file contracts ─────────────────────────────────────────────────────

def test_fitted_model_file_exists():
    assert config.FITTED_MODEL_PATH.exists(), (
        "fitted_model.pkl not found. Run: python src/step4_model_fitting.py"
    )


def test_fitted_model_is_unpicklable(loaded_fitted_model):
    assert loaded_fitted_model is not None


def test_model_params_file_exists():
    assert config.MODEL_PARAMS_PATH.exists(), (
        "model_params.json not found. Run: python src/step4_model_fitting.py"
    )


def test_model_params_json_valid(model_params_json):
    assert isinstance(model_params_json, dict)


def test_model_params_ljung_box_passes(model_params_json):
    """The final fitted model's residuals must pass the Ljung-Box test."""
    p_val = model_params_json.get("ljung_box_pvalue", 0)
    assert p_val > 0.05, (
        f"Fitted model residuals failed Ljung-Box test (p={p_val:.4f}). "
        "Consider changing the ARMA order in step 3."
    )


def test_plots_created():
    for name in ("step4_residuals.png", "step4_residual_acf.png", "step4_qq_plot.png"):
        assert (config.PLOTS_DIR / name).exists(), f"Plot not found: {name}"

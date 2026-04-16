"""
Tests for Step 3 – Model Identification & Order Selection
==========================================================
Run:  pytest tests/test_step3.py

NOTE: Requires step 2 output.  Run `python src/step2_stationarity.py` first.
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.step1_load_data import get_train_val_split
from src.step2_stationarity import load_stationary
from src.step3_model_selection import (
    compute_acf_pacf,
    plot_acf_pacf,
    grid_search_arma,
    select_best_order,
    load_model_selection,
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
def model_selection():
    if not config.MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not found. Run: python src/step3_model_selection.py")
    return load_model_selection()


@pytest.fixture(scope="module")
def synthetic_stationary():
    """Small white-noise series for fast unit tests."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.standard_normal(300))


# ── compute_acf_pacf() ────────────────────────────────────────────────────────

def test_acf_pacf_returns_dict(synthetic_stationary):
    result = compute_acf_pacf(synthetic_stationary, nlags=20)
    assert isinstance(result, dict)


def test_acf_pacf_required_keys(synthetic_stationary):
    result = compute_acf_pacf(synthetic_stationary, nlags=20)
    for key in ("acf_values", "pacf_values", "acf_conf_int", "pacf_conf_int", "lags"):
        assert key in result, f"compute_acf_pacf() must return key '{key}'"


def test_acf_lag0_is_one(synthetic_stationary):
    result = compute_acf_pacf(synthetic_stationary, nlags=20)
    assert abs(result["acf_values"][0] - 1.0) < 1e-9, \
        "ACF at lag 0 must equal 1.0"


def test_acf_pacf_correct_length(synthetic_stationary):
    nlags = 20
    result = compute_acf_pacf(synthetic_stationary, nlags=nlags)
    assert len(result["acf_values"])  == nlags + 1
    assert len(result["pacf_values"]) == nlags + 1
    assert len(result["lags"])        == nlags + 1


def test_acf_conf_int_shape(synthetic_stationary):
    nlags = 20
    result = compute_acf_pacf(synthetic_stationary, nlags=nlags)
    assert len(result["acf_conf_int"]) == nlags + 1
    for ci in result["acf_conf_int"]:
        assert len(ci) == 2, "Each confidence interval must be [lower, upper]"
        assert ci[0] <= ci[1], "lower bound must be ≤ upper bound"


# ── grid_search_arma() ────────────────────────────────────────────────────────

def test_grid_search_returns_list(synthetic_stationary):
    candidates = grid_search_arma(synthetic_stationary, max_p=2, max_q=2)
    assert isinstance(candidates, list)
    assert len(candidates) > 0


def test_grid_search_candidate_keys(synthetic_stationary):
    candidates = grid_search_arma(synthetic_stationary, max_p=2, max_q=2)
    for c in candidates:
        for key in ("p", "q", "aic", "bic"):
            assert key in c, f"Each candidate must have key '{key}'"


def test_grid_search_no_trivial_model(synthetic_stationary):
    candidates = grid_search_arma(synthetic_stationary, max_p=2, max_q=2)
    for c in candidates:
        assert not (c["p"] == 0 and c["q"] == 0), \
            "ARMA(0,0) (white noise) must not be included in candidates"


def test_grid_search_sorted_by_aic(synthetic_stationary):
    candidates = grid_search_arma(synthetic_stationary, max_p=2, max_q=2)
    aics = [c["aic"] for c in candidates]
    assert aics == sorted(aics), "Candidates must be sorted by AIC (ascending)"


def test_grid_search_bounds(synthetic_stationary):
    max_p, max_q = 3, 2
    candidates = grid_search_arma(synthetic_stationary, max_p=max_p, max_q=max_q)
    for c in candidates:
        assert 0 <= c["p"] <= max_p
        assert 0 <= c["q"] <= max_q


# ── select_best_order() ───────────────────────────────────────────────────────

def test_select_best_order_returns_dict(synthetic_stationary):
    candidates = grid_search_arma(synthetic_stationary, max_p=2, max_q=2)
    result = select_best_order(candidates)
    assert isinstance(result, dict)


def test_select_best_order_required_keys(synthetic_stationary):
    candidates = grid_search_arma(synthetic_stationary, max_p=2, max_q=2)
    result = select_best_order(candidates)
    for key in ("best_by_aic", "best_by_bic", "selected_order", "selection_note"):
        assert key in result, f"select_best_order() must return key '{key}'"


def test_selected_order_valid_values(synthetic_stationary):
    candidates = grid_search_arma(synthetic_stationary, max_p=2, max_q=2)
    result = select_best_order(candidates)
    p = result["selected_order"]["p"]
    q = result["selected_order"]["q"]
    assert p >= 0 and q >= 0
    assert not (p == 0 and q == 0), "Selected model cannot be ARMA(0,0)"


def test_selection_note_is_nonempty_string(synthetic_stationary):
    candidates = grid_search_arma(synthetic_stationary, max_p=2, max_q=2)
    result = select_best_order(candidates)
    assert isinstance(result["selection_note"], str)
    assert len(result["selection_note"]) > 10, \
        "selection_note must be a meaningful justification string"


# ── Output file contracts ─────────────────────────────────────────────────────

def test_model_selection_file_exists():
    assert config.MODEL_SELECTION_PATH.exists(), (
        "model_selection.json not found. Run: python src/step3_model_selection.py"
    )


def test_model_selection_has_candidates(model_selection):
    assert "candidate_models" in model_selection
    assert len(model_selection["candidate_models"]) > 0


def test_model_selection_has_selected_order(model_selection):
    assert "selected_order" in model_selection
    p = model_selection["selected_order"]["p"]
    q = model_selection["selected_order"]["q"]
    assert isinstance(p, int) and isinstance(q, int)
    assert p + q > 0, "Selected model must have at least one parameter"


def test_model_selection_acf_present(model_selection):
    assert "acf_values" in model_selection
    assert len(model_selection["acf_values"]) >= 21


def test_plots_created():
    for name in ("step3_acf.png", "step3_pacf.png", "step3_aic_bic_heatmap.png"):
        assert (config.PLOTS_DIR / name).exists(), f"Plot not found: {name}"

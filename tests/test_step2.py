"""
Tests for Step 2 – Stationarity Analysis & Preprocessing
=========================================================
Run:  pytest tests/test_step2.py

NOTE: These tests require step 1 output to exist.
      Run `python src/step1_load_data.py` first.
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.step1_load_data import load_daily, get_train_val_split
from src.step2_stationarity import (
    decompose_series,
    make_stationary,
    test_stationarity,
    load_stationary,
    load_decomposition,
    save_stationary,
    save_decomposition,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily_series():
    return load_daily()


@pytest.fixture(scope="module")
def train_series(daily_series):
    train, _ = get_train_val_split(daily_series)
    return train


@pytest.fixture(scope="module")
def stationary_series():
    """Load the processed file; skip if step 2 hasn't been run yet."""
    if not config.AT_LOAD_STATIONARY_PATH.exists():
        pytest.skip("Step 2 output not found. Run: python src/step2_stationarity.py")
    return load_stationary()


@pytest.fixture(scope="module")
def decomposition():
    if not config.DECOMPOSITION_PATH.exists():
        pytest.skip("decomposition.json not found. Run: python src/step2_stationarity.py")
    return load_decomposition()


# ── decompose_series() ────────────────────────────────────────────────────────

def test_decompose_returns_dict(train_series):
    result = decompose_series(train_series)
    assert isinstance(result, dict)


def test_decompose_required_keys(train_series):
    result = decompose_series(train_series)
    for key in ("trend", "seasonal", "residual", "period"):
        assert key in result, f"decompose_series() must return key '{key}'"


def test_decompose_same_length_as_input(train_series):
    result = decompose_series(train_series)
    assert len(result["residual"]) <= len(train_series), \
        "Residual cannot be longer than the input"


# ── make_stationary() ─────────────────────────────────────────────────────────

def test_make_stationary_returns_tuple(daily_series, train_series):
    decomposition = decompose_series(train_series)
    result = make_stationary(daily_series, decomposition)
    assert isinstance(result, tuple) and len(result) == 2, \
        "make_stationary() must return (stationary_series, meta_dict)"


def test_make_stationary_no_nans(daily_series, train_series):
    decomposition = decompose_series(train_series)
    stationary, _ = make_stationary(daily_series, decomposition)
    assert stationary.isna().sum() == 0, \
        "Stationary series must contain no NaN values"


def test_make_stationary_meta_has_required_keys(daily_series, train_series):
    decomposition = decompose_series(train_series)
    _, meta = make_stationary(daily_series, decomposition)
    required = [
        "transformations", "trend_type", "seasonal_period",
        "diff_orders", "train_end", "val_start", "val_end",
        "n_train", "n_val",
    ]
    for key in required:
        assert key in meta, f"decomposition.json must contain key '{key}'"


def test_make_stationary_transformations_is_list(daily_series, train_series):
    decomposition = decompose_series(train_series)
    _, meta = make_stationary(daily_series, decomposition)
    assert isinstance(meta["transformations"], list), \
        "'transformations' must be a list"
    assert len(meta["transformations"]) >= 1, \
        "At least one transformation must be applied"


# ── test_stationarity() ───────────────────────────────────────────────────────

def test_stationarity_returns_dict(train_series):
    """Uses a simple synthetic stationary series to test the function contract."""
    rng = np.random.default_rng(42)
    white_noise = pd.Series(rng.standard_normal(500))
    result = test_stationarity(white_noise)
    assert isinstance(result, dict)


def test_stationarity_required_keys(train_series):
    rng = np.random.default_rng(42)
    white_noise = pd.Series(rng.standard_normal(500))
    result = test_stationarity(white_noise)
    for key in ("adf_pvalue", "kpss_pvalue", "adf_conclusion",
                "kpss_conclusion", "conclusion"):
        assert key in result, f"test_stationarity() must return key '{key}'"


def test_stationarity_detects_white_noise():
    """White noise should be identified as stationary."""
    rng = np.random.default_rng(42)
    white_noise = pd.Series(rng.standard_normal(500))
    result = test_stationarity(white_noise)
    assert result["adf_conclusion"] == "stationary", \
        "ADF should classify white noise as stationary"


def test_stationarity_detects_random_walk():
    """A random walk should be identified as non-stationary."""
    rng = np.random.default_rng(42)
    rw = pd.Series(np.cumsum(rng.standard_normal(500)))
    result = test_stationarity(rw)
    assert result["adf_conclusion"] == "non-stationary", \
        "ADF should classify a random walk as non-stationary"


# ── Output file contracts ─────────────────────────────────────────────────────

def test_stationary_file_exists():
    assert config.AT_LOAD_STATIONARY_PATH.exists(), (
        "Stationary CSV not found. Run: python src/step2_stationarity.py"
    )


def test_stationary_columns(stationary_series):
    assert stationary_series.name == "load_stationary"


def test_stationary_no_nans(stationary_series):
    assert stationary_series.isna().sum() == 0


def test_stationary_is_actually_stationary(stationary_series):
    """The training portion of the stationary series must pass ADF."""
    train_stat = stationary_series.loc[:config.TRAIN_END]
    result = test_stationarity(train_stat)
    assert result["adf_conclusion"] == "stationary", (
        f"Training stationary series failed ADF test "
        f"(p={result['adf_pvalue']:.4f}). Revisit step 2 transformations."
    )


def test_decomposition_file_exists():
    assert config.DECOMPOSITION_PATH.exists(), (
        "decomposition.json not found. Run: python src/step2_stationarity.py"
    )


def test_decomposition_json_valid(decomposition):
    assert isinstance(decomposition, dict)


def test_decomposition_date_fields(decomposition):
    assert decomposition["train_end"]  == config.TRAIN_END
    assert decomposition["val_start"]  == config.VAL_START
    assert decomposition["val_end"]    == config.VAL_END


def test_decomposition_n_train(decomposition):
    assert decomposition["n_train"] >= 1820, \
        f"n_train should be ~1826 days (5 years), got {decomposition['n_train']}"


def test_plots_created():
    for name in ("step2_original_series.png",
                 "step2_stationary_series.png"):
        path = config.PLOTS_DIR / name
        assert path.exists(), f"Plot not found: {path}"

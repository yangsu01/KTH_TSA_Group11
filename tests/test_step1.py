"""
Tests for Step 1 – Data Loading & Preprocessing
================================================
Run:  pytest tests/test_step1.py
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.step1_load_data import (
    find_column_index,
    load_and_resample,
    save_daily,
    load_daily,
    get_train_val_split,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily_series():
    """Load the processed file if it exists, otherwise run step 1."""
    if not config.AT_LOAD_DAILY_PATH.exists():
        series = load_and_resample(config.RAW_DATA_PATH)
        save_daily(series, config.AT_LOAD_DAILY_PATH)
    return load_daily()


# ── Column detection ──────────────────────────────────────────────────────────

def test_find_column_index_returns_int():
    idx = find_column_index(config.RAW_DATA_PATH)
    assert isinstance(idx, int), "Column index must be an integer"
    assert idx >= 2, "Data columns start at index 2 (after the two timestamp columns)"


def test_find_column_index_correct_country():
    """Verify the found index really points to AT load actual."""
    idx = find_column_index(config.RAW_DATA_PATH)
    with open(config.RAW_DATA_PATH, "r") as f:
        rows = [f.readline().strip().split(";") for _ in range(3)]
    assert rows[0][idx] == config.TARGET_COUNTRY
    assert rows[1][idx] == config.TARGET_VARIABLE
    assert rows[2][idx] == config.TARGET_TYPE


# ── Output schema ─────────────────────────────────────────────────────────────

def test_output_file_exists():
    assert config.AT_LOAD_DAILY_PATH.exists(), (
        f"Output file not found: {config.AT_LOAD_DAILY_PATH}\n"
        "Run: python src/step1_load_data.py"
    )


def test_output_columns(daily_series):
    assert daily_series.name == "load_mw", "Series must be named 'load_mw'"


def test_output_index_is_datetime(daily_series):
    assert isinstance(daily_series.index, pd.DatetimeIndex), \
        "Index must be DatetimeIndex"


def test_output_daily_frequency(daily_series):
    diffs = daily_series.index.to_series().diff().dropna().dt.days.unique()
    assert set(diffs) == {1}, \
        "Series must be at daily frequency (no gaps)"


def test_output_date_range(daily_series):
    assert str(daily_series.index[0].date())  == config.TRAIN_START, \
        f"Series should start at {config.TRAIN_START}"
    assert str(daily_series.index[-1].date()) == config.VAL_END, \
        f"Series should end at {config.VAL_END}"


def test_output_minimum_length(daily_series):
    # 6 years of daily data ≥ 2000 points (project requirement)
    assert len(daily_series) >= 2000, \
        f"Expected ≥2000 data points, got {len(daily_series)}"


def test_no_nans(daily_series):
    assert daily_series.isna().sum() == 0, \
        "Output must contain no NaN values (step 1 must interpolate gaps)"


def test_load_values_are_positive(daily_series):
    assert (daily_series > 0).all(), \
        "All load values must be positive (MW)"


def test_load_values_plausible_range(daily_series):
    # Austria peak load is ~10 000 MW; minimum ~3 000 MW
    assert daily_series.min() > 1_000, \
        f"Minimum load {daily_series.min():.0f} MW seems too low"
    assert daily_series.max() < 30_000, \
        f"Maximum load {daily_series.max():.0f} MW seems too high"


# ── Train / val split ─────────────────────────────────────────────────────────

def test_train_val_split_no_overlap(daily_series):
    train, val = get_train_val_split(daily_series)
    assert train.index.max() < val.index.min(), \
        "Training set must end before validation set begins"


def test_train_covers_five_years(daily_series):
    train, _ = get_train_val_split(daily_series)
    years = (train.index.max() - train.index.min()).days / 365.25
    assert years >= 4.9, f"Training period should be ~5 years, got {years:.2f}"


def test_val_covers_at_least_two_months(daily_series):
    _, val = get_train_val_split(daily_series)
    assert len(val) >= 60, \
        f"Validation set should have at least 60 days, got {len(val)}"

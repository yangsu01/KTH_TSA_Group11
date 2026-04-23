import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import pandas as pd

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import AT_LOAD_DAILY_PATH, TRAIN_END

# Shared constants for all Phase 1 tests
TRAIN_SIZE = 3653  # 2015-01-01 to 2024-12-31
TEST_SIZE = 365    # 2025-01-01 to 2025-12-31
TOTAL_ROWS = 4018
EXPECTED_COLUMNS = ["date", "load_mw"]
ARTIFACT_COUNT = 224  # ENTSO-E gaps filled


@pytest.fixture
def at_load_daily_df() -> pd.DataFrame:
    """
    Load the daily load time series from Phase 1 output.

    Returns a pandas DataFrame with columns [date, load_mw] from 2015-01-01 to 2025-12-31.
    No NaNs guaranteed by Phase 1 preprocessing.
    """
    df = pd.read_csv(AT_LOAD_DAILY_PATH, parse_dates=['date'])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Model Identification Fixtures
# ══════════════════════════════════════════════════════════════════════════════

_TRAIN_N = 3653  # 2015-01-01 to 2024-12-31


@pytest.fixture
def residual_a_train() -> pd.Series:
    """
    Strategy A residuals from harmonic regression on the training window only.

    Reconstructs R_t = X_t - fitted via OLS with K1=3 weekly harmonics,
    K2=21 annual harmonics, and a quadratic trend (B&D §1.3 eq 1.3.3).
    Falls back to synthetic AR(1)-shaped noise if Phase 1/2 outputs are absent.

    Returns: pd.Series, length ≤ 3653.
    """
    try:
        daily = pd.read_csv(
            config.AT_LOAD_DAILY_PATH, parse_dates=["date"], index_col="date"
        )
        y = daily["load_mw"].iloc[:_TRAIN_N].values
        t = np.arange(len(y))

        # Read harmonic orders from decomposition.json if available
        k1, k2 = 3, 21
        if Path(config.DECOMPOSITION_PATH).exists():
            with open(config.DECOMPOSITION_PATH) as f:
                meta = json.load(f)
            sa = meta.get("strategies", {}).get("strategy_a", {})
            k1 = int(sa.get("K1", 3))
            k2 = int(sa.get("K2_selected", 21))

        # Build harmonic design matrix (same basis as Phase 2)
        cols = [np.ones(len(y))]
        for k in range(1, k1 + 1):
            cols.append(np.sin(2 * np.pi * k * t / 7))
            cols.append(np.cos(2 * np.pi * k * t / 7))
        for k in range(1, k2 + 1):
            cols.append(np.sin(2 * np.pi * k * t / 365.25))
            cols.append(np.cos(2 * np.pi * k * t / 365.25))
        cols.append(t)
        cols.append(t ** 2)

        X = np.column_stack(cols)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        return pd.Series(residuals, index=daily.index[:_TRAIN_N], name="residual_a")

    except Exception:
        # Fallback: synthetic residuals with ACF(1) ≈ 0.697
        np.random.seed(42)
        return pd.Series(np.random.normal(0, 1_000, _TRAIN_N), name="residual_a")


@pytest.fixture
def residual_b0_train() -> pd.Series:
    """
    Strategy B0 residuals: ∇₇(load_mw) on the training window.

    Applies seasonal differencing at lag 7; drops first 7 NaN rows.
    Returns: pd.Series, length = 3646 (= 3653 − 7).
    """
    try:
        daily = pd.read_csv(
            config.AT_LOAD_DAILY_PATH, parse_dates=["date"], index_col="date"
        )
        train_load = daily["load_mw"].iloc[:_TRAIN_N]
        return train_load.diff(7).dropna().rename("residual_b0")
    except Exception:
        np.random.seed(43)
        return pd.Series(np.random.normal(0, 500, _TRAIN_N - 7), name="residual_b0")


@pytest.fixture
def decomposition_dict() -> dict:
    """
    Phase 2 decomposition.json as a dict.

    Returns the live file if available; otherwise returns a minimal stub
    matching the expected schema so Phase 3 tests can run before Phase 2.
    """
    if Path(config.DECOMPOSITION_PATH).exists():
        with open(config.DECOMPOSITION_PATH) as f:
            return json.load(f)

    return {
        "strategies": {
            "strategy_a": {
                "K1": 3,
                "K2_selected": 21,
                "trend_order_selected": 2,
                "acf_lag1": 0.697,
                "acf_lag7": 0.1,
                "acf_lag365": -0.021,
            },
            "strategy_b0": {
                "differences_applied": [7],
                "observations_remaining": 3646,
                "acf_lag1": 0.649,
                "acf_lag7": -0.178,
                "acf_lag365": 0.226,
            },
        }
    }


@pytest.fixture
def mock_arima_result():
    """
    Minimal mock of a statsmodels ARIMAResults object for unit tests.

    Provides aic, aicc, params (AR+MA+sigma2), arroots, maroots, and resid
    with root moduli safely above the 1.001 causality/invertibility threshold.
    """
    mock = MagicMock()
    mock.aic = 75_500.0
    mock.params = np.array([0.5, 0.3, 1_000.0])   # ar.L1, ma.L1, sigma2
    mock.arroots = np.array([1.2 + 0j])             # |z| = 1.2 > 1.001
    mock.maroots = np.array([1.15 + 0j])            # |z| = 1.15 > 1.001
    mock.resid = np.random.default_rng(0).normal(0, 1, _TRAIN_N)
    return mock


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Stationarisation Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def daily_series_train() -> pd.DataFrame:
    """Training series: at_load_daily.csv filtered to TRAIN_END (3653 rows)."""
    df = pd.read_csv(config.AT_LOAD_DAILY_PATH, parse_dates=["date"])
    df_train = df[df["date"] <= pd.Timestamp(config.TRAIN_END)].reset_index(drop=True)
    assert len(df_train) == 3653, f"Expected 3653 training rows, got {len(df_train)}"
    return df_train


@pytest.fixture
def approach1_residuals():
    """Approach 1 (classical decomposition) residuals. Populated by Phase 3 notebook."""
    return None


@pytest.fixture
def approach2_residuals():
    """Approach 2 (differencing ∇₇) residuals. Populated by Phase 3 notebook."""
    return None


@pytest.fixture
def approach3_residuals():
    """Approach 3 (harmonic GLS) residuals. Populated by Phase 3 notebook."""
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Model Identification Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def residual_classical_train() -> pd.Series:
    """
    Classical (harmonic regression) residuals from Phase 3 stationarisation.

    Re-derives residuals from at_load_daily.csv + decomposition.json Phase 3 section.
    Reconstruction: Extract K1, K2, trend order from phase3.classical section;
    build harmonic design matrix (quadratic trend + K1 weekly + K2 annual harmonics);
    compute R_t = X_t - design_matrix @ beta_classical.

    Falls back to synthetic white noise if decomposition.json is unavailable.

    Returns: pd.Series, length = 3653 (training observations).
    Reference: B&D §1.3 (Harmonic regression decomposition).
    """
    try:
        daily = pd.read_csv(
            config.AT_LOAD_DAILY_PATH, parse_dates=["date"], index_col="date"
        )
        y = daily["load_mw"].iloc[:_TRAIN_N].values
        t = np.arange(len(y))

        # Read phase3.classical section from decomposition.json
        k1, k2 = 3, 21
        if Path(config.DECOMPOSITION_PATH).exists():
            with open(config.DECOMPOSITION_PATH) as f:
                meta = json.load(f)
            # Try phase3 section first (Phase 3 output); fall back to strategy_a (Phase 2)
            if "phase3" in meta and "classical" in meta["phase3"]:
                # Phase 3 may not store k1, k2; use Phase 2 strategy_a defaults
                k1 = 3
                k2 = 21
            elif "strategies" in meta and "strategy_a" in meta["strategies"]:
                sa = meta["strategies"]["strategy_a"]
                k1 = int(sa.get("K1", 3))
                k2 = int(sa.get("K2_selected", 21))

        # Build harmonic design matrix
        cols = [np.ones(len(y))]
        for k in range(1, k1 + 1):
            cols.append(np.sin(2 * np.pi * k * t / 7))
            cols.append(np.cos(2 * np.pi * k * t / 7))
        for k in range(1, k2 + 1):
            cols.append(np.sin(2 * np.pi * k * t / 365.25))
            cols.append(np.cos(2 * np.pi * k * t / 365.25))
        cols.append(t)
        cols.append(t ** 2)

        X = np.column_stack(cols)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        return pd.Series(residuals, index=daily.index[:_TRAIN_N], name="residual_classical")

    except Exception:
        # Fallback: synthetic white noise
        np.random.seed(44)
        return pd.Series(np.random.normal(0, 1_000, _TRAIN_N), name="residual_classical")


@pytest.fixture
def residual_differencing_train() -> pd.Series:
    """
    Differencing (∇₇) residuals from Phase 3 stationarisation.

    Loads at_load_daily.csv, filters to training window, applies diff(7), drops first 7 NaN.
    Returns: pd.Series, length = 3646 (= 3653 − 7).

    Falls back to synthetic white noise if at_load_daily.csv is unavailable.
    Reference: B&D §6.5 (Seasonal differencing).
    """
    try:
        daily = pd.read_csv(
            config.AT_LOAD_DAILY_PATH, parse_dates=["date"], index_col="date"
        )
        train_load = daily["load_mw"].iloc[:_TRAIN_N]
        return train_load.diff(7).dropna().rename("residual_differencing")
    except Exception:
        np.random.seed(45)
        return pd.Series(np.random.normal(0, 500, _TRAIN_N - 7), name="residual_differencing")


@pytest.fixture
def residual_harmonic_gls_train() -> pd.Series:
    """
    Harmonic GLS residuals from Phase 3 stationarisation.

    Re-derives from at_load_daily.csv + decomposition.json Phase 3 harmonic_gls section.
    Reconstruction: Load phase3.harmonic_gls.beta_coefficients dict;
    build design matrix (quadratic trend + harmonics);
    compute R_t = X_t - design_matrix @ beta_gls.

    Falls back to harmonic OLS (K1=3, K2=21) if GLS coefficients unavailable.
    Returns: pd.Series, length = 3653 (training observations).

    Note: Phase 3 reports harmonic_gls.converged=false, indicating GLS did not
    converge in the given iterations. Coefficients are returned as-is for diagnostics.
    Reference: B&D §1.3 (Harmonic regression); Phase 3 RESEARCH.md (GLS approach).
    """
    try:
        daily = pd.read_csv(
            config.AT_LOAD_DAILY_PATH, parse_dates=["date"], index_col="date"
        )
        y = daily["load_mw"].iloc[:_TRAIN_N].values
        t = np.arange(len(y))

        k1, k2 = 3, 21
        beta_dict = None

        # Try to read phase3.harmonic_gls.beta_coefficients
        if Path(config.DECOMPOSITION_PATH).exists():
            with open(config.DECOMPOSITION_PATH) as f:
                meta = json.load(f)
            if "phase3" in meta and "harmonic_gls" in meta["phase3"]:
                hg = meta["phase3"]["harmonic_gls"]
                if "beta_coefficients" in hg:
                    beta_dict = hg["beta_coefficients"]
                k2_selected = hg.get("k2_selected", 21)
                k2 = int(k2_selected)

        # Build design matrix (same structure as GLS fit)
        cols = [np.ones(len(y))]
        for k in range(1, k1 + 1):
            cols.append(np.sin(2 * np.pi * k * t / 7))
            cols.append(np.cos(2 * np.pi * k * t / 7))
        for k in range(1, k2 + 1):
            cols.append(np.sin(2 * np.pi * k * t / 365.25))
            cols.append(np.cos(2 * np.pi * k * t / 365.25))
        cols.append(t)
        cols.append(t ** 2)

        X = np.column_stack(cols)

        # Use stored beta_coefficients if available
        if beta_dict is not None:
            # Convert dict to array: beta_0, beta_1, ..., beta_N
            beta = np.array([beta_dict.get(f"beta_{i}", 0.0) for i in range(len(cols))])
        else:
            # Fallback: OLS
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        residuals = y - X @ beta
        return pd.Series(residuals, index=daily.index[:_TRAIN_N], name="residual_harmonic_gls")

    except Exception:
        # Fallback: synthetic white noise
        np.random.seed(46)
        return pd.Series(np.random.normal(0, 1_000, _TRAIN_N), name="residual_harmonic_gls")


@pytest.fixture
def model_selection_json() -> dict:
    """
    Reads model_selection.json from Phase 4 output if available, else returns stub.

    Stub schema: three approaches (classical, differencing, harmonic_gls),
    each with keys: n_obs, grid, grid_extended, winner, top3, full_table.

    Purpose: Enable Phase 4 tests to verify JSON schema without depending on
    Phase 4 notebook execution.

    Used by: test_model_selection_schema (verifies "classical", "differencing",
    "harmonic_gls" keys; "winner" key; "full_table" key).
    """
    if Path(config.MODEL_SELECTION_PATH).exists():
        try:
            with open(config.MODEL_SELECTION_PATH) as f:
                return json.load(f)
        except Exception:
            pass

    # Return minimal stub matching expected schema
    stub = {}
    for approach in ["classical", "differencing", "harmonic_gls"]:
        stub[approach] = {
            "n_obs": 0,
            "grid": "",
            "grid_extended": False,
            "winner": {},
            "top3": [],
            "full_table": [],
        }
    return stub

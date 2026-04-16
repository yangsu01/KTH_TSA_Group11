"""
Step 4 – Model Fitting & Diagnostics
======================================
Responsibility
--------------
Fit the ARMA model selected in step 3 to the stationary training series.
Report parameter estimates, standard errors, and run residual diagnostics
to verify that the model is adequate.

Person 4 owns this file.

Input
-----
  config.AT_LOAD_STATIONARY_PATH  →  data/processed/at_load_stationary.csv
  config.MODEL_SELECTION_PATH     →  results/model_selection.json

Output
------
  config.FITTED_MODEL_PATH  →  results/fitted_model.pkl
    Serialised statsmodels ARIMAResultsWrapper (pickle).

  config.MODEL_PARAMS_PATH  →  results/model_params.json

    Schema:
    {
      "order"            : {"p": 2, "q": 1},
      "method"           : "MLE",            // "Yule-Walker" | "Innovations" | "MLE"
      "ar_params"        : [0.52, -0.18],    // AR coefficients phi_1, phi_2, ...
      "ma_params"        : [0.31],           // MA coefficients theta_1, ...
      "sigma2"           : 0.0041,           // estimated noise variance
      "ar_stderr"        : [...],
      "ma_stderr"        : [...],
      "ar_conf_int_95"   : [[lower,upper],...],
      "ma_conf_int_95"   : [[lower,upper],...],
      "aic"              : 1234.5,
      "bic"              : 1256.7,
      "log_likelihood"   : -612.3,
      "n_obs"            : 1826,
      "ljung_box_lag"    : 20,
      "ljung_box_stat"   : 18.4,
      "ljung_box_pvalue" : 0.56,             // >0.05 → residuals are white noise
      "jb_pvalue"        : 0.12              // Jarque-Bera normality test p-value
    }

  Plots saved to config.PLOTS_DIR:
    step4_residuals.png         – residual time series
    step4_residual_acf.png      – ACF of residuals (should be white noise)
    step4_qq_plot.png           – QQ-plot of standardised residuals

Run
---
  python src/step4_model_fitting.py

Test
----
  pytest tests/test_step4.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.step2_stationarity import load_stationary
from src.step3_model_selection import load_model_selection
from src.step1_load_data import get_train_val_split


# ── Public API ────────────────────────────────────────────────────────────────

def fit_arma(
    series: pd.Series,
    p: int,
    q: int,
    method: str = "MLE",
):
    """
    Fit ARMA(p, q) to the training series.

    Parameters
    ----------
    series : pd.Series  –  stationary training series (no NaNs)
    p      : int        –  AR order
    q      : int        –  MA order
    method : str        –  "MLE" (default), "Yule-Walker", or "Innovations"
                           For Yule-Walker / Innovations see the course book
                           Chapters 5.2–5.3.  For MLE use ARIMA with order=(p,0,q).

    Returns
    -------
    Fitted model result object.
      - For MLE      : statsmodels ARIMAResultsWrapper
      - For YW/Innov : dict with keys "ar_params", "ma_params", "sigma2"
                       (return a thin wrapper so step 5 can call .predict())

    TODO: implement
      from statsmodels.tsa.arima.model import ARIMA
      model = ARIMA(series, order=(p, 0, q))
      result = model.fit(method='innovations_mle')  # or 'statespace' for full MLE
    """
    raise NotImplementedError("TODO: fit ARMA model")


def extract_params(fitted_model) -> dict:
    """
    Extract parameter estimates and goodness-of-fit metrics from the fitted
    model into a JSON-serialisable dict (model_params.json schema).

    Parameters
    ----------
    fitted_model : ARIMAResultsWrapper (or equivalent)

    Returns
    -------
    dict  –  see model_params.json schema in module docstring

    TODO: implement
      Use fitted_model.params, fitted_model.bse, fitted_model.conf_int(),
      fitted_model.aic, fitted_model.bic, fitted_model.llf
    """
    raise NotImplementedError("TODO: extract model parameters")


def diagnose_residuals(fitted_model, series: pd.Series) -> dict:
    """
    Run residual diagnostics and return a summary dict.

    Diagnostics to perform
    ----------------------
    1. Ljung-Box test on residuals (use lag=20) – tests for remaining autocorrelation.
    2. Jarque-Bera test – tests normality of residuals.
    3. Plot: residual time series.
    4. Plot: ACF of residuals (should show no significant spikes).
    5. Plot: QQ-plot of standardised residuals.

    Parameters
    ----------
    fitted_model : fitted ARMA result
    series       : pd.Series  –  training series (for residual extraction)

    Returns
    -------
    dict with keys:
      "ljung_box_lag"    : int
      "ljung_box_stat"   : float
      "ljung_box_pvalue" : float
      "jb_pvalue"        : float

    TODO: implement
      from statsmodels.stats.diagnostic import acorr_ljungbox
      from statsmodels.stats.stattools import jarque_bera
    """
    raise NotImplementedError("TODO: diagnose residuals")


def save_fitted_model(fitted_model, output_path: str | os.PathLike) -> None:
    """Pickle the fitted model for use in step 5."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(fitted_model, f)
    print(f"[step4] Saved fitted model → {output_path}")


def load_fitted_model(path: str | os.PathLike = config.FITTED_MODEL_PATH):
    """Convenience loader used by step 5."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model_params(params: dict, output_path: str | os.PathLike) -> None:
    """Serialise parameter dict to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        raise TypeError(type(obj))

    with open(output_path, "w") as f:
        json.dump(params, f, indent=2, default=_convert)
    print(f"[step4] Saved model params → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("[step4] Loading inputs …")
    stationary     = load_stationary()
    train, _       = get_train_val_split(stationary)
    model_sel      = load_model_selection()

    p = model_sel["selected_order"]["p"]
    q = model_sel["selected_order"]["q"]
    print(f"[step4] Fitting ARMA({p},{q}) …")

    fitted = fit_arma(train, p, q)
    params = extract_params(fitted)
    diag   = diagnose_residuals(fitted, train)

    # Merge diagnostics into params
    params.update(diag)

    print(f"  AR params : {params.get('ar_params')}")
    print(f"  MA params : {params.get('ma_params')}")
    print(f"  σ²        : {params.get('sigma2'):.6f}")
    print(f"  AIC={params.get('aic'):.2f}  BIC={params.get('bic'):.2f}")
    print(f"  Ljung-Box p-value: {params.get('ljung_box_pvalue'):.4f}")

    save_fitted_model(fitted, config.FITTED_MODEL_PATH)
    save_model_params(params, config.MODEL_PARAMS_PATH)
    print("[step4] Done.")

"""
Step 3 – Model Identification & Order Selection
================================================
Responsibility
--------------
Use the stationary training series from step 2 to identify the ARMA(p, q)
order.  Produce ACF/PACF plots and an AIC/BIC grid search to justify the
chosen model order.

Person 3 owns this file.

Input
-----
  config.AT_LOAD_STATIONARY_PATH  →  data/processed/at_load_stationary.csv
    (use only the TRAINING portion: up to config.TRAIN_END)

Output
------
  config.MODEL_SELECTION_PATH  →  results/model_selection.json

    Schema:
    {
      "candidate_models" : [
          {"p": 0, "q": 1, "aic": 1234.5, "bic": 1256.7},
          ...
      ],
      "best_by_aic"   : {"p": 2, "q": 1},
      "best_by_bic"   : {"p": 1, "q": 1},
      "selected_order": {"p": 2, "q": 1},   // justified choice (may differ from best AIC/BIC)
      "selection_note": "Selected AIC-optimal model; BIC prefers parsimonious AR(1,1).",
      "acf_values"    : [...],               // list of float, lags 0..40
      "pacf_values"   : [...],
      "acf_conf_int"  : [[lower,upper], ...],
      "pacf_conf_int" : [[lower,upper], ...]
    }

  Plots saved to config.PLOTS_DIR:
    step3_acf.png
    step3_pacf.png
    step3_aic_bic_heatmap.png

Run
---
  python src/step3_model_selection.py

Test
----
  pytest tests/test_step3.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.step2_stationarity import load_stationary
from src.step1_load_data import get_train_val_split


# ── Public API ────────────────────────────────────────────────────────────────

def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 40,
) -> dict:
    """
    Compute ACF and PACF values with 95 % confidence intervals.

    Parameters
    ----------
    series : pd.Series  –  stationary training series (no NaNs)
    nlags  : int        –  number of lags to compute

    Returns
    -------
    dict with keys:
      "acf_values"    : list[float]          length nlags+1
      "pacf_values"   : list[float]          length nlags+1
      "acf_conf_int"  : list[[lower,upper]]  length nlags+1
      "pacf_conf_int" : list[[lower,upper]]  length nlags+1
      "lags"          : list[int]            0..nlags

    TODO: implement using
      from statsmodels.tsa.stattools import acf, pacf
    """
    raise NotImplementedError("TODO: compute ACF and PACF")


def plot_acf_pacf(acf_pacf: dict) -> None:
    """
    Plot ACF and PACF as bar charts with confidence-interval bands.
    Save to step3_acf.png and step3_pacf.png.

    TODO: implement
      - Horizontal dashed lines at ±1.96/√N (or use the conf_int from acf_pacf)
      - Clearly labelled x-axis (lags) and y-axis (correlation)
      - Indicate which lags are significant

    Hint: statsmodels.graphics.tsaplots.plot_acf / plot_pacf are convenient
    wrappers, but you can also build the bar chart manually for full control.
    """
    raise NotImplementedError("TODO: plot ACF and PACF")


def grid_search_arma(
    series: pd.Series,
    max_p: int = config.MAX_P,
    max_q: int = config.MAX_Q,
) -> list[dict]:
    """
    Fit ARMA(p, q) for all combinations 0 ≤ p ≤ max_p, 0 ≤ q ≤ max_q,
    and record AIC and BIC for each.

    Parameters
    ----------
    series : pd.Series  –  stationary training series
    max_p  : int
    max_q  : int

    Returns
    -------
    list of dicts, each with keys: "p", "q", "aic", "bic"
    Sorted by AIC (ascending).  Models that fail to converge are excluded.

    TODO: implement using
      from statsmodels.tsa.arima.model import ARIMA
      # ARMA(p,q) ≡ ARIMA(p,0,q)

    Note: skip (p=0, q=0) as it is a trivial white-noise model.
    """
    raise NotImplementedError("TODO: grid search ARMA orders")


def select_best_order(candidates: list[dict]) -> dict:
    """
    Choose a (p, q) order from the candidate list and return a selection dict.

    Parameters
    ----------
    candidates : list[dict]  –  output of grid_search_arma()

    Returns
    -------
    dict with keys:
      "best_by_aic"    : {"p": int, "q": int}
      "best_by_bic"    : {"p": int, "q": int}
      "selected_order" : {"p": int, "q": int}
      "selection_note" : str   –  1–2 sentence justification for the report

    TODO: implement
      - Identify the model with the lowest AIC and the model with the lowest BIC.
      - If they agree, select that order.
      - If they disagree, choose and justify (e.g. prefer BIC for parsimony).
      - Consider whether the ACF/PACF plots support the chosen order.
    """
    raise NotImplementedError("TODO: select best ARMA order")


def plot_aic_bic_heatmap(candidates: list[dict]) -> None:
    """
    Plot AIC and BIC as colour heatmaps over the (p, q) grid.
    Save to step3_aic_bic_heatmap.png.

    TODO: implement
      - Two subplots: one for AIC, one for BIC
      - p on y-axis, q on x-axis
      - Annotate the cell with the lowest value
    """
    raise NotImplementedError("TODO: plot AIC/BIC heatmap")


def save_model_selection(
    acf_pacf: dict,
    candidates: list[dict],
    selection: dict,
    output_path: str | os.PathLike = config.MODEL_SELECTION_PATH,
) -> None:
    """Merge all results into model_selection.json."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "candidate_models": candidates,
        **selection,
        "acf_values":    acf_pacf["acf_values"],
        "pacf_values":   acf_pacf["pacf_values"],
        "acf_conf_int":  acf_pacf["acf_conf_int"],
        "pacf_conf_int": acf_pacf["pacf_conf_int"],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[step3] Saved model selection → {output_path}")


def load_model_selection(
    path: str | os.PathLike = config.MODEL_SELECTION_PATH,
) -> dict:
    """Convenience loader used by step 4."""
    with open(path) as f:
        return json.load(f)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("[step3] Loading stationary series …")
    stationary = load_stationary()
    train, _ = get_train_val_split(stationary)

    print("[step3] Computing ACF / PACF …")
    acf_pacf = compute_acf_pacf(train)
    plot_acf_pacf(acf_pacf)

    print(f"[step3] Grid searching ARMA(0..{config.MAX_P}, 0..{config.MAX_Q}) …")
    candidates = grid_search_arma(train)
    plot_aic_bic_heatmap(candidates)

    print("[step3] Selecting best order …")
    selection = select_best_order(candidates)
    p = selection["selected_order"]["p"]
    q = selection["selected_order"]["q"]
    print(f"  Selected ARMA({p},{q})  –  {selection['selection_note']}")

    save_model_selection(acf_pacf, candidates, selection)
    print("[step3] Done.")

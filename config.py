"""
Shared configuration for the TSA pipeline.
All paths are relative to the project root.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Raw data ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH = ROOT / "data_hourly.csv"

# ── Output directories ────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"
PLOTS_DIR     = ROOT / "plots"

# ── Target series ─────────────────────────────────────────────────────────────
TARGET_COUNTRY  = "AT"                          # Austria
TARGET_VARIABLE = "load"
TARGET_TYPE     = "actual_entsoe_transparency"

# ── Time splits ───────────────────────────────────────────────────────────────
TRAIN_START = "2015-01-01"   # first full day in CET
TRAIN_END   = "2019-12-31"   # end of training window  (5 years)
VAL_START   = "2020-01-01"   # start of validation window
VAL_END     = "2020-09-30"   # end of validation window (data ends here)

# ── File contracts (exact paths every step reads / writes) ────────────────────
#
# Step 1 writes:
AT_LOAD_DAILY_PATH      = PROCESSED_DIR / "at_load_daily.csv"
#   Columns : date (YYYY-MM-DD), load_mw (float)
#   Rows    : one row per calendar day, 2015-01-01 – 2020-12-31 (~2192 rows)
#   No NaNs guaranteed (interpolated by step 1)

# Step 2 writes:
AT_LOAD_STATIONARY_PATH = PROCESSED_DIR / "at_load_stationary.csv"
#   Columns : date (YYYY-MM-DD), load_stationary (float)
#   Rows    : one row per day (may be shorter than daily if differencing was used)
#   No NaNs guaranteed
DECOMPOSITION_PATH      = RESULTS_DIR / "decomposition.json"
#   Schema described in src/step2_stationarity.py

# Step 3 writes:
MODEL_SELECTION_PATH    = RESULTS_DIR / "model_selection.json"
#   Schema described in src/step3_model_selection.py

# Step 4 writes:
MODEL_PARAMS_PATH       = RESULTS_DIR / "model_params.json"
#   Schema described in src/step4_model_fitting.py
FITTED_MODEL_PATH       = RESULTS_DIR / "fitted_model.pkl"
#   Serialised statsmodels ARIMAResultsWrapper

# Step 5 writes:
FORECAST_PATH           = RESULTS_DIR / "forecast.csv"
#   Columns : date, forecast_stationary, forecast_original,
#             lower_95, upper_95, actual
#   Rows    : one per day in the validation window

# ── Modelling hyper-parameters (shared defaults) ──────────────────────────────
MAX_P = 5          # maximum AR order to search in grid
MAX_Q = 5          # maximum MA order to search in grid
CONFIDENCE_LEVEL = 0.95

# ── Plot style ────────────────────────────────────────────────────────────────
FIGURE_DPI    = 150
FIGURE_SIZE   = (10, 4)   # (width, height) in inches

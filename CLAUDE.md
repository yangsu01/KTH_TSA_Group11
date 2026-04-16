# KTH TSA Group 11 – Claude Agent Instructions

## Project overview

This is a KTH Time Series Analysis course project (3 ECTS, pass/fail).
We analyse **Austria (AT) hourly electricity load** data from the ENTSO-E
open power system dataset.

- **Data file**: `data_hourly.csv` (ENTSO-E, semicolon-separated, ~50 k rows)
- **Target column**: `AT / load / actual_entsoe_transparency` (units: MW)
- **Analysis window**: 2015-01-01 → 2020-12-31  (~2192 daily observations after resampling)
- **Train / val split**: train = 2015–2019 (5 years), val = 2020 (1 year)
- **Language**: Python 3.10+, `.py` scripts only (no notebooks for production code)
- **Tests**: pytest

The deliverable is a set of **plots** for a written report, plus the analysis
code itself.  The report must include: data cleaning → ARMA fitting → forecasting.

---

## Repository layout

```
KTH_TSA_Group11/
├── data_hourly.csv              raw ENTSO-E data (do not modify)
├── data/
│   └── processed/
│       ├── at_load_daily.csv        ← written by step 1
│       └── at_load_stationary.csv   ← written by step 2
├── results/
│   ├── decomposition.json           ← written by step 2
│   ├── model_selection.json         ← written by step 3
│   ├── model_params.json            ← written by step 4
│   ├── fitted_model.pkl             ← written by step 4
│   └── forecast.csv                 ← written by step 5
├── plots/                           ← all .png files written by steps 2–5
├── src/
│   ├── step1_load_data.py
│   ├── step2_stationarity.py
│   ├── step3_model_selection.py
│   ├── step4_model_fitting.py
│   └── step5_forecasting.py
├── tests/
│   ├── test_step1.py
│   ├── test_step2.py
│   ├── test_step3.py
│   ├── test_step4.py
│   └── test_step5.py
├── config.py                        shared constants and paths
└── requirements.txt
```

---

## Pipeline – 5 equal steps

Each step is owned by one person.  Steps must be run **in order** (1 → 5)
because each reads the output of the previous step.  Steps 3–5 can only be
worked on after step 2 is complete; step 5 requires step 4.

```
step1  →  at_load_daily.csv
step2  →  at_load_stationary.csv + decomposition.json
step3  →  model_selection.json
step4  →  fitted_model.pkl + model_params.json
step5  →  forecast.csv + final plots
```

### Step 1 – Data Loading & Preprocessing  (Person 1)

**File**: `src/step1_load_data.py`
**Status**: fully implemented – run it first to unblock everyone else.

**What it does**:
- Finds the AT/load/actual column in `data_hourly.csv` by scanning the 3-row header.
- Resamples hourly → daily mean (MW is a power quantity, so mean is correct).
- Interpolates any missing days linearly.
- Saves `data/processed/at_load_daily.csv`.

**Run**:
```bash
python src/step1_load_data.py
```

**Test**:
```bash
pytest tests/test_step1.py
```

**Output schema** (`at_load_daily.csv`):

| column   | type  | notes                     |
|----------|-------|---------------------------|
| date     | str   | YYYY-MM-DD                |
| load_mw  | float | mean daily load, no NaNs  |

---

### Step 2 – Stationarity Analysis  (Person 2)

**File**: `src/step2_stationarity.py`

**Goal**: produce a stationary series for ARMA fitting, and document every
transformation so step 5 can reverse them.

**What to implement** (all functions contain `raise NotImplementedError`):

1. `plot_original(series)` – plot the raw series, mark train/val boundary.
2. `decompose_series(train)` – decompose into trend + seasonal + residual.
   - Use `statsmodels.tsa.seasonal.seasonal_decompose` or `STL`.
   - Electricity load has **weekly** (period=7) and **annual** (period=365)
     seasonality.  Motivate which you remove.
3. `make_stationary(series, decomposition)` – apply transformations in order:
   - Log transform (if variance is non-constant)
   - Remove trend (subtract or difference)
   - Remove seasonal component (subtract or seasonal differencing)
   - First-order differencing if a unit root remains
   - Return `(stationary_series, meta_dict)` where meta matches the
     `decomposition.json` schema in the docstring.
4. `test_stationarity(series)` – run ADF and KPSS tests.
   - `from statsmodels.tsa.stattools import adfuller, kpss`
5. `plot_stationary(original, stationary)` – side-by-side comparison.

**Key rule**: Only use the **training** series when fitting the decomposition
(no look-ahead into the validation period).  Apply the fitted transformation
to the full series.

**Run**:
```bash
python src/step2_stationarity.py
```

**Test**:
```bash
pytest tests/test_step2.py
```

**Output files**:
- `data/processed/at_load_stationary.csv`  (columns: `date`, `load_stationary`)
- `results/decomposition.json`  (schema in `src/step2_stationarity.py` docstring)
- `plots/step2_original_series.png`
- `plots/step2_stationary_series.png`

---

### Step 3 – Model Identification & Order Selection  (Person 3)

**File**: `src/step3_model_selection.py`

**Goal**: identify the best ARMA(p, q) order using ACF/PACF plots and
information criteria (AIC/BIC).

**What to implement**:

1. `compute_acf_pacf(series, nlags=40)` – return ACF and PACF arrays.
   - `from statsmodels.tsa.stattools import acf, pacf`
2. `plot_acf_pacf(acf_pacf)` – bar charts with 95 % CI bands.
3. `grid_search_arma(series, max_p=5, max_q=5)` – fit ARMA(p,q) for all
   combinations, record AIC and BIC, return list sorted by AIC.
   - `from statsmodels.tsa.arima.model import ARIMA`  (ARMA = ARIMA with d=0)
   - Skip (p=0, q=0).  Catch convergence failures silently.
4. `select_best_order(candidates)` – pick (p,q) and write a justification string.
5. `plot_aic_bic_heatmap(candidates)` – colour heatmap of AIC/BIC over p×q grid.

**Important**: use **only the training portion** of the stationary series
(`series.loc[:config.TRAIN_END]`).

**Run**:
```bash
python src/step3_model_selection.py
```

**Test**:
```bash
pytest tests/test_step3.py
```

**Output files**:
- `results/model_selection.json`  (schema in `src/step3_model_selection.py` docstring)
- `plots/step3_acf.png`
- `plots/step3_pacf.png`
- `plots/step3_aic_bic_heatmap.png`

---

### Step 4 – Model Fitting & Diagnostics  (Person 4)

**File**: `src/step4_model_fitting.py`

**Goal**: fit the selected ARMA model, report parameter estimates with
standard errors, and verify that residuals are white noise.

**What to implement**:

1. `fit_arma(series, p, q, method="MLE")` – fit ARMA(p,q) to training data.
   - Default to MLE via `statsmodels ARIMA(series, order=(p,0,q)).fit()`.
   - The course also covers Yule-Walker (Chapter 5.2) and innovations
     (Chapter 5.3) methods – you may implement these as alternatives.
2. `extract_params(fitted_model)` – pull out AR/MA coefficients, σ², SEs,
   95 % CI, AIC, BIC, log-likelihood, n_obs.
3. `diagnose_residuals(fitted_model, series)`:
   - Ljung-Box test at lag 20 (`statsmodels.stats.diagnostic.acorr_ljungbox`).
   - Jarque-Bera normality test (`statsmodels.stats.stattools.jarque_bera`).
   - Save three plots: residual time series, ACF of residuals, QQ-plot.
   - **Key acceptance criterion**: Ljung-Box p-value > 0.05 (residuals ≈ white noise).
     If this fails, go back to step 3 and choose a different order.

**Run**:
```bash
python src/step4_model_fitting.py
```

**Test**:
```bash
pytest tests/test_step4.py
```

**Output files**:
- `results/fitted_model.pkl`
- `results/model_params.json`  (schema in `src/step4_model_fitting.py` docstring)
- `plots/step4_residuals.png`
- `plots/step4_residual_acf.png`
- `plots/step4_qq_plot.png`

---

### Step 5 – Forecasting & Evaluation  (Person 5)

**File**: `src/step5_forecasting.py`

**Goal**: forecast the full 2020 validation year, reverse all step-2
transformations, compute accuracy metrics, and produce the main report plots.

**What to implement**:

1. `forecast_stationary(fitted_model, n_steps, alpha)` – multi-step forecast
   with prediction intervals on the stationary scale.
   - `fitted_model.get_forecast(steps=n_steps).summary_frame(alpha=alpha)`
2. `back_transform(forecast_df, decomposition, train_series)` – reverse every
   transformation in `decomposition["transformations"]` in **reverse order**.
   Common inverses:
   - `"diff"` → cumulative sum seeded from last training value
   - `"seasonal_diff"` → seasonal cumsum seeded from last `period` training values
   - `"log"` → `np.exp`
   - `"detrend_linear"` → add back `trend_coefficients` at forecast time indices
   - `"seasonal_sub"` → add back `seasonal_component` at correct phase
3. `evaluate_forecast(forecast_original, actual)` – compute MAE, RMSE, MAPE.
4. `plot_forecast_original(...)` – the main report plot:
   train series + validation actuals + forecast + PI ribbon.
5. `plot_forecast_stationary(...)` – same but on the stationary scale.

**Run**:
```bash
python src/step5_forecasting.py
```

**Test**:
```bash
pytest tests/test_step5.py
```

**Output files**:
- `results/forecast.csv`  (columns: `date`, `forecast_stationary`,
  `forecast_original`, `lower_95`, `upper_95`, `actual`)
- `plots/step5_forecast_stationary.png`
- `plots/step5_forecast_original.png`
- `plots/step5_prediction_intervals.png`

---

## Running the full pipeline

```bash
python src/step1_load_data.py
python src/step2_stationarity.py
python src/step3_model_selection.py
python src/step4_model_fitting.py
python src/step5_forecasting.py
```

Run all tests:
```bash
pytest tests/
```

Run a single step's tests:
```bash
pytest tests/test_step2.py -v
```

---

## Data contracts (exact file schemas)

### `at_load_daily.csv`
```
date,load_mw
2015-01-01,6823.5
...
```
- `date`: YYYY-MM-DD string
- `load_mw`: float, daily mean load in MW, no NaNs
- Row count: ~2192 (2015-01-01 to 2020-12-31 inclusive)

### `at_load_stationary.csv`
```
date,load_stationary
2015-01-02,-0.0312
...
```
- `date`: YYYY-MM-DD string
- `load_stationary`: float, no NaNs
- May be shorter than daily due to differencing

### `decomposition.json`
```json
{
  "transformations": ["log", "seasonal_diff"],
  "trend_type": "linear",
  "trend_coefficients": [8.45, -0.0001],
  "seasonal_period": 365,
  "seasonal_component": [...],
  "diff_orders": [365],
  "train_end": "2019-12-31",
  "val_start": "2020-01-01",
  "val_end": "2020-12-31",
  "n_train": 1826,
  "n_val": 366
}
```

### `model_selection.json`
```json
{
  "candidate_models": [{"p":2,"q":1,"aic":1234.5,"bic":1256.7}, ...],
  "best_by_aic": {"p":2,"q":1},
  "best_by_bic": {"p":1,"q":1},
  "selected_order": {"p":2,"q":1},
  "selection_note": "AIC-optimal; ACF/PACF support AR(2) and MA(1).",
  "acf_values": [...],
  "pacf_values": [...],
  "acf_conf_int": [[lower,upper], ...],
  "pacf_conf_int": [[lower,upper], ...]
}
```

### `model_params.json`
```json
{
  "order": {"p":2,"q":1},
  "method": "MLE",
  "ar_params": [0.52, -0.18],
  "ma_params": [0.31],
  "sigma2": 0.0041,
  "ar_stderr": [...],
  "ma_stderr": [...],
  "ar_conf_int_95": [[lower,upper],...],
  "ma_conf_int_95": [[lower,upper],...],
  "aic": 1234.5,
  "bic": 1256.7,
  "log_likelihood": -612.3,
  "n_obs": 1826,
  "ljung_box_lag": 20,
  "ljung_box_stat": 18.4,
  "ljung_box_pvalue": 0.56,
  "jb_pvalue": 0.12
}
```

### `forecast.csv`
```
date,forecast_stationary,forecast_original,lower_95,upper_95,actual
2020-01-01,-0.021,6810.2,5910.1,7810.3,6923.0
...
```

---

## Key dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
statsmodels>=0.14
scipy>=1.10
pytest>=7.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Agent guidance

- **Never modify** `data_hourly.csv`.
- **Never modify** `config.py` unless fixing a clear bug — all steps depend on it.
- When implementing a step, read the step file's module docstring first — it
  contains the full output schema and implementation hints.
- Each function that needs implementing contains `raise NotImplementedError("TODO: …")`.
  Replace the raise with the implementation; do not change the function signature.
- `config.PLOTS_DIR` and other directories are created automatically by each step
  via `os.makedirs(..., exist_ok=True)`.
- If a test skips with "Step N output not found", it means you need to run the
  previous step's script first.
- Plot style: use `config.FIGURE_SIZE` and `config.FIGURE_DPI` for consistency.
  Save with `plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')`.
- All floating-point values in JSON must be plain Python `float`, not `numpy.float64`.
  Use the `_convert` helper in `save_decomposition` as a template.

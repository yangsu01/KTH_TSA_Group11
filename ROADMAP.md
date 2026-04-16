# Project Roadmap — Austria Total Electricity Load Forecasting

**Course:** Time Series Analysis  
**Data:** Austria total load (hourly), ENTSO-E Transparency Platform  
**Training window:** 2015-01-01 to 2019-12-31 (43,824 hourly observations)  
**Test window:** 2020-01-01 to 2020-12-31 (8,784 hourly observations)  
**Target model:** ARMA(p, q)  
**Deadline for final report:** May 13 at 19:00 | **Oral presentation:** May 18

---

## Acceptance Criteria Checklist

Every box below must be ticked before final submission.

### Data and sourcing
- [ ] Data source stated explicitly: URL, dataset name, access date (ENTSO-E Transparency Platform, Actual Total Load for Austria / AT)
- [ ] N reported in the report; N ≥ 2000 confirmed (hourly 2015–2019 gives N = 43,824)
- [ ] Data confirmed as non-financial and not a temperature record
- [ ] Data file included with the submission

### Cleaning and stationarisation
- [ ] Raw series plotted and visually described (trend direction, seasonal cycles, any anomalies)
- [ ] Trend component $m_t$ identified and removed — explicit formula reported (e.g. linear/polynomial OLS or moving-average smoother)
- [ ] Intraday seasonal component ($d = 24$) identified — explicit seasonal indices $\hat{s}_1, \ldots, \hat{s}_{24}$ reported
- [ ] Weekly seasonal component ($d = 168$ hours) considered and handled
- [ ] Annual seasonal component ($d = 8760$ hours) considered and handled
- [ ] All cleaning steps reported in order, each with its own formula or procedure
- [ ] Stationarity of residual $Z_t$ verified after cleaning (ACF plot, or formal ADF/KPSS test with citation)
- [ ] Log transform decision explained (applied or not, with justification based on whether variance scales with level)

### Model identification
- [ ] Sample ACF of $Z_t$ plotted and interpreted (cut-off lag or decay pattern)
- [ ] Sample PACF of $Z_t$ plotted and interpreted (cut-off lag or decay pattern)
- [ ] Candidate ARMA(p, q) models listed (at least 3–5 candidates)
- [ ] AICC values reported in a table for all candidates
- [ ] Final model chosen with explicit reasoning (lowest AICC + ACF/PACF agreement)
- [ ] Causality verified: all roots of $\phi(z)$ lie strictly outside the unit circle
- [ ] Invertibility verified: all roots of $\theta(z)$ lie strictly outside the unit circle

### Parameter estimation
- [ ] Estimation method stated: Yule-Walker, Innovations algorithm, Burg, or MLE
- [ ] Preliminary estimates computed (Yule-Walker for AR part; Innovations for MA part) as starting values
- [ ] MLE estimates $\hat{\phi}_1, \ldots, \hat{\phi}_p$, $\hat{\theta}_1, \ldots, \hat{\theta}_q$, $\hat{\sigma}^2$ reported in the report
- [ ] 95% confidence intervals for all estimated parameters reported
- [ ] Software and functions used stated and briefly explained

### Diagnostic checking
- [ ] Residuals $\hat{\varepsilon}_t$ plotted vs. time — no visible pattern
- [ ] Sample ACF of $\hat{\varepsilon}_t$ plotted — all lags within $\pm 1.96/\sqrt{n}$
- [ ] Ljung-Box portmanteau test applied and result reported ($Q$ statistic, degrees of freedom, p-value)
- [ ] If diagnostics fail: model order revisited and revised model reported

### Forecasting
- [ ] Prediction method stated explicitly (linear h-step predictors via Innovations algorithm or Durbin-Levinson recursion)
- [ ] Forecasts produced for the full 2020 test window (8,784 hours)
- [ ] Forecasts back-transformed to original MW scale (trend + seasonal components added back; exponential applied if log transform was used)
- [ ] Prediction error variance $\sigma^2(h) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$ computed
- [ ] 95% prediction intervals computed and plotted
- [ ] Forecast vs. actual plot produced for 2020 with prediction bands
- [ ] Quantitative accuracy metrics reported on 2020 test set: MAE, RMSE, MAPE
- [ ] Accuracy discussed in words: what drove errors, COVID period impact, seasonal fit quality

### Report quality
- [ ] Every figure has a title, axis labels with units, and a caption
- [ ] Report length: 1–2 pages Part A + 1 page Part B, excluding figures and references
- [ ] No unexplained assumptions — every modelling decision justified
- [ ] Difficulties encountered described (triple seasonality at hourly resolution, 2020 demand shock)
- [ ] Robustness discussion included (sensitivity to p, q choice; sensitivity to cleaning procedure)
- [ ] Alternative approaches noted (e.g. SARIMA, state-space models, ARIMA with differencing)

### Schedule
- [ ] Preliminary report (Part A) sent to peer review group by **May 1**
- [ ] Peer review feedback submitted to assigned group by **May 7**
- [ ] Final report (Part A + B) submitted via Canvas by **May 13 at 19:00**
- [ ] All group members present at oral presentation on **May 18**, Rooms B2 and K1

---

## Phase 1 — Data acquisition and initial exploration
**Target dates:** Apr 10–14

### 1.1 Download the data

Source: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu)  
Navigation: Consumption > Actual Total Load > Bidding Zone: AT (Austria)  
Export format: CSV, hourly resolution  
Period: 2015-01-01 00:00 to 2020-12-31 23:00 UTC

Expected number of observations:
- Training set 2015–2019: 5 years x 8,760 hours/year + 1 leap-year day (2016) x 24 = **43,824 hours**
- Test set 2020: 2020 is a leap year, so **8,784 hours**
- Total: **52,608 hourly observations**

Record in the report: platform name, URL, download date, and the exact column used ("Actual Total Load [MW]").

> Reference: Project Description — "the source of the data must be given clearly"

### 1.2 Inspect and clean the raw CSV

Check for: missing timestamps, duplicate entries, NaN or zero values (partial outages in the ENTSO-E data feed). Short gaps of 1–2 hours can be filled with linear interpolation. Longer gaps must be flagged and the handling reported explicitly. Confirm the final N in the report.

### 1.3 Plot the raw series

Plot $X_t$ (MW) vs $t$ (hourly index) for the full training window 2015–2019. Also produce a one-week zoom-in and a one-year zoom-in. From these plots identify:

- Long-run trend: typically slow and flat for Austria in this period
- Annual seasonality: higher load in winter, lower in summer — period $d_1 = 8760$ hours
- Weekly seasonality: weekdays vs. weekends — period $d_2 = 168$ hours
- Intraday seasonality: morning peak, midday dip, evening peak — period $d_3 = 24$ hours
- Outliers: Christmas/New Year dips, public holidays, and the COVID onset in March 2020 which will appear in the test set

> Reference: Chapter 1, Sec 1.3–1.4; Brockwell & Davis p. 1–10

### 1.4 Compute the sample ACF

Compute the sample autocovariance function:
$$\hat{\gamma}(h) = \frac{1}{n} \sum_{t=1}^{n-h} (X_{t+h} - \bar{X})(X_t - \bar{X}), \quad h = 0, 1, \ldots, H$$

with $H = 500$ lags (about 3 weeks at hourly resolution). Plot $\hat{\rho}(h) = \hat{\gamma}(h)/\hat{\gamma}(0)$ with 95% confidence bands $\pm 1.96/\sqrt{n}$.

Expected pattern before cleaning: very slow decay (confirming non-stationarity), plus spikes at $h = 24, 48, 72, \ldots$ (intraday) and at $h = 168, 336, \ldots$ (weekly).

> Reference: Chapter 1, Sec 1.5; Chapter 2, Sec 2.1–2.2; Appendix C

---

## Phase 2 — Cleaning the data: making $Z_t$ stationary
**Target dates:** Apr 14–21

Hourly electricity load has a **triple seasonal structure**: intraday ($d = 24$), weekly ($d = 168$), and annual ($d = 8760$). This is the central technical difficulty of working at hourly resolution. The goal is a residual series $Z_t$ that is weakly stationary.

### 2.1 Decide on a log transform

If the seasonal amplitude grows with the load level (larger oscillations in winter than in summer), stabilise the variance first:
$$W_t = \log(X_t)$$

Check this by plotting the rolling standard deviation over 30-day windows. If the standard deviation is roughly constant across the year, skip the transform. Report the decision and the evidence for it.

> Reference: Project Description — "this may involve transforming the data (e.g. using a log transform)"

### 2.2 Remove the long-run trend $m_t$

Fit a polynomial trend by OLS. A linear model is usually sufficient for Austria 2015–2019:
$$m_t = \hat{\beta}_0 + \hat{\beta}_1 \cdot t$$

Report $\hat{\beta}_0$ and $\hat{\beta}_1$ with units. Subtract: $Y_t = W_t - m_t$.

A centred moving-average filter with window $M = 8760$ is an equivalent alternative. State which method was used and why.

> Reference: Chapter 1, Sec 1.3; Chapter 2, Sec 2.3

### 2.3 Remove the three seasonal components

**Step A — Annual component ($d_1 = 8760$).**  
For each hour-of-year index $k \in \{1, \ldots, 8760\}$ compute:
$$\hat{s}^{(1)}_k = \frac{1}{|\{t : (t \bmod 8760) = k\}|} \sum_{\{t : (t \bmod 8760) = k\}} Y_t$$
Normalise so $\sum_{k=1}^{8760} \hat{s}^{(1)}_k = 0$. Define $V_t = Y_t - \hat{s}^{(1)}_{t \bmod 8760}$.

**Step B — Weekly component ($d_2 = 168$).**  
Repeat the same averaging procedure on $V_t$ for each hour-of-week index $k \in \{1, \ldots, 168\}$. Normalise and subtract to obtain $U_t$.

**Step C — Intraday component ($d_3 = 24$).**  
Repeat on $U_t$ for each hour-of-day index $k \in \{1, \ldots, 24\}$. Normalise and subtract to obtain $Z_t$.

Report all three sets of seasonal indices — either as tables or as clearly labelled plots. This is a hard requirement from the project description.

> Reference: Chapter 1, Sec 1.3–1.4; Chapter 2, Sec 2.3 (periodicity filter)  
> Project Description — "provide explicit expressions for the trend and seasonal components"

**Software note:** In Python, `statsmodels.tsa.seasonal.STL` handles a single seasonal period. For the triple structure, either apply STL three times sequentially or compute the seasonal averages manually as shown above. The manual approach is easier to explain in the report. In R, `msts()` from the `forecast` package supports multiple seasonal periods natively.

### 2.4 Verify stationarity of $Z_t$

After all three steps, check $Z_t$:

1. **Time plot:** mean near 0, variance stable over time.
2. **Sample ACF:** recompute up to lag 500. The spikes at $h = 24, 168, 8760$ should be gone or negligible. What remains is the ARMA dependence structure.
3. **Formal test (optional — cite the reference if used):** Augmented Dickey-Fuller test, null hypothesis: unit root present. Reject at $p < 0.05$ to confirm stationarity.

If the ACF of $Z_t$ still shows significant spikes at $h = 24$ or $h = 168$, refine the seasonal indices in Step 2.3.

> Reference: Chapter 2, Sec 2.1; Appendix C

---

## Phase 3 — Model identification: choosing $p$ and $q$
**Target dates:** Apr 21–28

### 3.1 Read the ACF of $Z_t$

The sample ACF pattern identifies the model family:

| ACF pattern | PACF pattern | Suggested model |
|---|---|---|
| Cuts off after lag $q$ | Tails off (geometric decay) | MA($q$) |
| Tails off (geometric decay) | Cuts off after lag $p$ | AR($p$) |
| Both tail off | Both tail off | ARMA($p$, $q$) |

Mark the 95% bands $\pm 1.96/\sqrt{n}$ on the plot. Only lags outside the bands are significant.

> Reference: Chapter 3, Sec 3.2; Lecture 6 & 7

### 3.2 Compute and read the PACF of $Z_t$

The partial autocorrelation at lag $h$ is the coefficient $\hat{\phi}_{hh}$ from the AR($h$) Yule-Walker system. Compute it via the Durbin-Levinson recursion:
$$\hat{\phi}_{h+1,h+1} = \frac{\hat{\rho}(h+1) - \sum_{j=1}^{h} \hat{\phi}_{hj}\,\hat{\rho}(h+1-j)}{1 - \sum_{j=1}^{h} \hat{\phi}_{hj}\,\hat{\rho}(j)}$$

Plot up to lag 50 with 95% bands. A pure AR($p$) shows PACF cutting off after lag $p$.

> Reference: Chapter 3, Sec 3.2.3; Chapter 2, Sec 2.5; Lecture 8 & 9

### 3.3 Fit candidate models and compare AICC

Fit ARMA($p$, $q$) for $p \in \{0, 1, 2, 3\}$ and $q \in \{0, 1, 2, 3\}$ — 16 models in total. For each compute:
$$\text{AICC}(p, q) = -2\,\ell(\hat{\phi}, \hat{\theta}, \hat{\sigma}^2) + \frac{2(p + q + 1)\,n}{n - p - q - 2}$$

Present all 16 values in a $4 \times 4$ table. Select the model with the smallest AICC. If two models are within 2 AICC units of each other, pick the simpler one (lower $p + q$).

> Reference: Chapter 5, Sec 5.5; Project Description — "choice of a suitable model should be motivated"

### 3.4 Verify causality and invertibility

For the selected ARMA($p$, $q$) with estimated coefficients:

- **Causality:** all roots of $\phi(z) = 1 - \hat{\phi}_1 z - \cdots - \hat{\phi}_p z^p = 0$ must satisfy $|z| > 1$.
- **Invertibility:** all roots of $\theta(z) = 1 + \hat{\theta}_1 z + \cdots + \hat{\theta}_q z^q = 0$ must satisfy $|z| > 1$.

In Python: `np.abs(np.roots([1, -phi_1, ..., -phi_p])) > 1` should return all `True`.

> Reference: Chapter 3, Sec 3.1; Definition 3.1.1; equations (3.1.5)–(3.1.6)

---

## Phase 4 — Parameter estimation
**Target dates:** Apr 28 – May 1

### 4.1 Preliminary estimates

**Yule-Walker (pure AR($p$) models):**

Solve $\hat{\Gamma}_p \hat{\phi} = \hat{\gamma}_p$, where $\hat{\Gamma}_p = [\hat{\gamma}(i-j)]_{i,j=1}^{p}$ and $\hat{\gamma}_p = (\hat{\gamma}(1), \ldots, \hat{\gamma}(p))'$:
$$\hat{\phi} = \hat{\Gamma}_p^{-1} \hat{\gamma}_p, \qquad \hat{\sigma}^2 = \hat{\gamma}(0) - \hat{\phi}'\,\hat{\gamma}_p$$

> Reference: Chapter 5, Sec 5.1.1; equations (5.1.3)–(5.1.6)

**Innovations algorithm (models with MA terms):**

Use the recursion from Chapter 5, Sec 5.1.3 to produce preliminary estimates of $\theta_1, \ldots, \theta_q$ as starting values for MLE.

> Reference: Chapter 5, Sec 5.1.3; Lecture 9

### 4.2 Maximum likelihood estimation

Maximise the Gaussian log-likelihood:
$$\ell(\phi, \theta, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{j=1}^{n} \frac{(Z_j - \hat{Z}_j)^2}{v_j}$$

where $\hat{Z}_j$ and $v_j$ are the one-step-ahead predicted value and its variance from the Innovations algorithm.

In Python:
```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(Z, order=(p, 0, q))
result = model.fit()
print(result.summary())
```

In R:
```r
fit <- arima(Z, order = c(p, 0, q))
summary(fit)
```

Report: $\hat{\phi}_1, \ldots, \hat{\phi}_p$, $\hat{\theta}_1, \ldots, \hat{\theta}_q$, $\hat{\sigma}^2$, and 95% confidence intervals for every coefficient.

> Reference: Chapter 5, Sec 5.2; Project Description — "include estimates for the parameters and the method used"

### 4.3 Diagnostic checking

Compute standardised residuals $\hat{\varepsilon}_t = (Z_t - \hat{Z}_t)/\sqrt{v_t}$.

1. **Time plot of $\hat{\varepsilon}_t$:** should look like white noise with no trend or pattern.
2. **Sample ACF of $\hat{\varepsilon}_t$:** all lags within $\pm 1.96/\sqrt{n}$.
3. **Ljung-Box test:**
$$Q_H = n(n+2)\sum_{h=1}^{H} \frac{\hat{\rho}^2_{\hat{\varepsilon}}(h)}{n - h} \overset{H_0}{\sim} \chi^2(H - p - q)$$
Use $H = 20$. Reject $H_0$ (white noise) if $p$-value $< 0.05$. If the test fails, increase $p$ or $q$ and re-estimate.

> Reference: Chapter 5, Sec 5.3

---

## Phase 5 — Forecasting and evaluation
**Target dates:** May 1–7

### 5.1 Compute h-step linear predictors on $Z_t$

For a causal ARMA($p$, $q$), the best linear $h$-step predictor satisfies:
$$P_n Z_{n+h} = \sum_{i=1}^{p} \hat{\phi}_i\,P_n Z_{n+h-i} + \sum_{j=1}^{q} \hat{\theta}_j\,\hat{\varepsilon}_{n+h-j}$$

where $P_n Z_{n+k} = Z_{n+k}$ for $k \leq 0$, and $\hat{\varepsilon}_{n+k} = 0$ for $k > 0$.

The mean squared prediction error grows with horizon:
$$\sigma^2(h) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$$

The MA($\infty$) coefficients $\{\psi_j\}$ come from $\psi(z) = \theta(z)/\phi(z)$.

> Reference: Chapter 3, Sec 3.3; Chapter 2, Sec 2.5; Chapter 5, Sec 5.4; Lecture 9

### 5.2 Back-transform to original MW scale

For each forecast horizon $h = 1, \ldots, 8784$:
$$\hat{X}_{n+h} = \hat{Z}_{n+h} + \hat{m}_{n+h} + \hat{s}^{(1)}_{(n+h)\,\bmod\,8760} + \hat{s}^{(2)}_{(n+h)\,\bmod\,168} + \hat{s}^{(3)}_{(n+h)\,\bmod\,24}$$

If a log transform was applied, exponentiate first: $\hat{X}_{n+h} = \exp(\hat{W}_{n+h})$.

The trend extrapolation $\hat{m}_{n+h}$ evaluates the fitted polynomial at $t = n + h$.

### 5.3 Compute 95% prediction intervals

$$\hat{X}_{n+h} \pm 1.96\,\hat{\sigma}\sqrt{\sum_{j=0}^{h-1}\psi_j^2}$$

Report explicitly how quickly the interval widens with $h$, and what this implies about the practical forecast horizon.

> Reference: Chapter 3, Sec 3.3; Chapter 5, Sec 5.4

### 5.4 Evaluate against the 2020 test set

On the 8,784 test observations compute:
$$\text{MAE} = \frac{1}{H}\sum_{h=1}^{H} |X_{n+h} - \hat{X}_{n+h}|$$

$$\text{RMSE} = \sqrt{\frac{1}{H}\sum_{h=1}^{H} (X_{n+h} - \hat{X}_{n+h})^2}$$

$$\text{MAPE} = \frac{100\%}{H}\sum_{h=1}^{H} \frac{|X_{n+h} - \hat{X}_{n+h}|}{X_{n+h}}$$

Produce two plots: (1) forecast vs. actual for January 2020 (one month zoom-in), (2) forecast vs. actual for all of 2020. Discuss where the forecast degrades: the COVID demand shock from March–April 2020 will cause systematic over-prediction, because the ARMA model has no mechanism to anticipate a structural break not present in the training data. This is worth discussing explicitly.

> Reference: Project Description — "discuss the accuracy of your forecast"

---

## Phase 6 — Report and presentation
**Target dates:** May 1–18

### 6.1 Preliminary report — Part A only (due May 1)

Send to peer review group by May 1. Must contain:
- Data source, N, and resolution
- All cleaning steps with explicit formulas
- AICC table for candidate models
- Estimated parameters with confidence intervals
- One forecast plot with prediction bands

### 6.2 Peer review feedback (due May 7)

Review the other group's Part A report. Check: stationarity verified, seasonal formulas explicit, model choice motivated, prediction errors reported. Give honest written feedback.

### 6.3 Final report — Part A + B (due May 13 at 19:00)

Submit via Canvas with the data file attached. Length: 1–2 pages Part A + 1 page Part B, excluding figures and references.

Required content:
- Data source and N
- All cleaning formulas
- Model selection table (AICC)
- Parameter estimates with confidence intervals
- Accuracy table: MAE, RMSE, MAPE on 2020 test set
- Discussion of difficulties (triple seasonality, COVID shock)
- Software and functions used, with a one-line explanation of what each function does

> Reference: Project Description — "you must state what software and functions you used and what those functions do"

### 6.4 Oral presentation (May 18, Rooms B2 and K1)

6–8 minutes, followed by 3–4 minutes of questions. Cover:
- The data and why Austria total load
- How the triple seasonal structure was handled at hourly resolution
- Model selection: alternatives considered, what AICC showed
- Forecast plot for 2020 with honest commentary on accuracy
- One difficulty and how it was addressed

All group members must be present and speak. Prepare answers for: "Why this p and q?", "What would SARIMA give you that ARMA cannot?", "How does the COVID shock affect your forecast?"

---

## Key references from the course

| Topic | Source |
|---|---|
| Time series decomposition | Brockwell & Davis, Chapter 1, Sec 1.3–1.4 |
| Stationarity, ACF | Chapter 1, Sec 1.5; Chapter 2, Sec 2.1–2.2; Appendix C |
| Linear filters, seasonal extraction | Chapter 2, Sec 2.3 |
| ARMA(p,q): definition, causality, invertibility | Chapter 3, Sec 3.1, Definition 3.1.1 |
| ACF and PACF of ARMA | Chapter 3, Sec 3.2 |
| Forecasting ARMA | Chapter 3, Sec 3.3 |
| Linear prediction, Durbin-Levinson | Chapter 2, Sec 2.5; Lecture 8 |
| Yule-Walker estimation | Chapter 5, Sec 5.1.1 |
| Innovations algorithm | Chapter 5, Sec 5.1.3 |
| Maximum likelihood estimation | Chapter 5, Sec 5.2 |
| Diagnostic checking | Chapter 5, Sec 5.3 |
| Forecasting with fitted model | Chapter 5, Sec 5.4 |
| Order selection, AICC | Chapter 5, Sec 5.5 |
| PACF, h-step prediction | Lectures 8–9; Chapter 3, Sec 3.3; Chapter 5, Sec 5.4 |

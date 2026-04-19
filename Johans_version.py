import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

plt.rcParams.update({"font.size": 15, "axes.labelpad": 2})

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_data(path="data_hourly.csv"):
    df = pd.read_csv(path, sep=";", skiprows=7, header=0,
                     names=["utc", "local", "load_mw"], parse_dates=["local"])
    df = df.dropna(subset=["load_mw"])
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    return df


def print_summary_raw(df):
    raw = df["load_mw"]
    print("=== Raw hourly data (before cleaning) ===")
    print(f"Time range : {df['local'].min().date()} – {df['local'].max().date()}")
    print(f"N obs      : {len(df):,}")
    print(f"Mean       : {raw.mean():,.0f} MW")
    print(f"Median     : {raw.median():,.0f} MW")
    print(f"Std        : {raw.std():,.0f} MW")
    print(f"Min        : {raw.min():,.0f} MW")
    print(f"Max        : {raw.max():,.0f} MW")


def clean_hourly(df, m=12, sigma=3):
    df = df.copy()
    ma = df["load_mw"].rolling(window=2*m+1, center=True).mean()
    residuals = df["load_mw"] - ma
    anomaly_mask = residuals.abs() > sigma * residuals.std()
    df.loc[anomaly_mask, "load_mw"] = float("nan")
    df["load_mw"] = df.set_index("local")["load_mw"].interpolate(method="time").values
    return df


def get_daily(df):
    return df.set_index("local")["load_mw"].resample("D").mean().dropna()


def clean_daily(daily, threshold=4000, sigma=3):
    daily = daily.copy()

    # Step 1: hard threshold
    n_thresh = (daily < threshold).sum()
    daily[daily < threshold] = float("nan")
    daily = daily.interpolate(method="time")
    print(f"\n=== Daily cleaning ===")
    print(f"  Hard threshold (<{threshold} MW) : {n_thresh} days removed")

    # Step 2: holidays
    holiday_dates = []
    for y in daily.index.year.unique():
        holiday_dates += list(pd.date_range(f"{y}-12-22", f"{y}-12-27"))
        holiday_dates += list(pd.date_range(f"{y}-12-31", f"{y+1}-01-02"))
    mask = daily.index.isin(holiday_dates)
    n_hol = mask.sum()
    daily[mask] = float("nan")
    daily = daily.interpolate(method="time")
    print(f"  Holidays                        : {n_hol} days removed")

    # Step 3: 3σ pass using rolling mean residuals
    rolling_mean = daily.rolling(window=30, center=True).mean()
    resid = daily - rolling_mean
    anomaly_mask = resid.abs() > sigma * resid.std()
    n_sigma = anomaly_mask.sum()
    daily[anomaly_mask] = float("nan")
    daily = daily.interpolate(method="time")
    print(f"  3σ outlier pass                 : {n_sigma} days removed")

    return daily


def print_summary_daily(daily, label="Daily data"):
    print(f"\n=== {label} ===")
    print(f"N obs      : {len(daily):,}")
    print(f"Mean       : {daily.mean():,.0f} MW")
    print(f"Median     : {daily.median():,.0f} MW")
    print(f"Std        : {daily.std():,.0f} MW")
    print(f"Min        : {daily.min():,.0f} MW")
    print(f"Max        : {daily.max():,.0f} MW")


def plot_load(daily):
    plt.figure(figsize=(14, 4))
    plt.plot(daily.index, daily.values, linewidth=0.8)
    plt.title("Austria Daily Electricity Load")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.show()


def plot_december_january(daily):
    years = sorted(daily.index.year.unique())
    periods = [(y, y+1) for y in years if y+1 in years]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    _, ax = plt.subplots(figsize=(14, 5))
    ax.axvspan(2, 7,  color="red",   alpha=0.15, label="Christmas (22–27 Dec)")
    ax.axvspan(11, 13, color="green", alpha=0.15, label="New Year (31 Dec–2 Jan)")

    for (y1, y2), color in zip(periods, colors):
        start = pd.Timestamp(f"{y1}-12-20")
        end   = pd.Timestamp(f"{y2}-01-05")
        segment = daily[start:end].dropna()
        offsets = (segment.index - start).days
        ax.plot(offsets, segment.values, linewidth=1.2, marker="o",
                markersize=3, color=color, label=f"{y1}/{y2}")

    ax.set_xticks(range(17))
    ax.set_xticklabels(
        [f"Dec {20+i}" if i < 12 else f"Jan {i-11}" for i in range(17)],
        rotation=30
    )
    ax.axhline(daily.mean(), color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Load (MW)")
    ax.set_title("Holiday period load by year")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def decompose_daily(daily):
    # Trend: centered 7-day MA (m=3)
    trend = daily.rolling(window=7, center=True).mean()

    # Seasonal: day-of-week means, normalized to sum to zero
    detrended = daily - trend
    dow_means = detrended.groupby(detrended.index.dayofweek).mean()
    dow_means -= dow_means.mean()
    seasonal = pd.Series(
        detrended.index.dayofweek.map(dow_means).values, index=daily.index
    )

    # Residuals
    residuals = daily - trend - seasonal

    print("\n=== Seasonal estimates (s_j) ===")
    for dow, name in enumerate(WEEKDAY_NAMES):
        print(f"  {name:<12}: {dow_means[dow]:+.2f} MW")

    def _stats(label, s):
        s = s.dropna()
        print(f"\n=== {label} ===")
        print(f"  Mean : {s.mean():,.2f}  Std : {s.std():,.2f}  Min : {s.min():,.2f}  Max : {s.max():,.2f}")
        return s.std()

    std_x  = _stats("X_t", daily)
    std_dt = _stats("X_t - m_t", detrended)
    std_y  = _stats("Y_t", residuals)

    if std_dt < std_x and std_y < std_dt:
        print("\n✓ Std decreases at each stage.")
    else:
        print("\n⚠ Std did NOT decrease monotonically — check decomposition.")

    return trend, seasonal, residuals


def plot_decomposition(daily, trend, residuals):
    detrended = daily - trend
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(daily.index, daily.values, linewidth=0.8)
    axes[0].plot(trend.index, trend.values, linewidth=1.5, label="m_t (7-day MA)")
    axes[0].set_ylabel("X_t (MW)")
    axes[0].legend()
    axes[1].plot(detrended.index, detrended.values, linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("X_t − m_t (MW)")
    axes[2].plot(residuals.index, residuals.values, linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Y_t (MW)")
    axes[2].set_xlabel("Date")
    fig.suptitle("Decomposition: X_t = m_t + s_t + Y_t")
    plt.tight_layout()
    plt.show()


def plot_residuals_stationarity(residuals, window=30):
    roll_mean = residuals.rolling(window=window, center=True).mean()
    roll_std  = residuals.rolling(window=window, center=True).std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(residuals.index, residuals.values, linewidth=0.8, alpha=0.7, label="Y_t")
    axes[0].plot(roll_mean.index, roll_mean.values, linewidth=1.5, label=f"Rolling mean ({window}d)")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Y_t (MW)")
    axes[0].legend()
    axes[1].plot(roll_std.index, roll_std.values, linewidth=1.5, color="orange")
    axes[1].set_ylabel(f"Rolling std ({window}d)")
    axes[1].set_xlabel("Date")
    fig.suptitle("Visual stationarity check — constant mean and variance?")
    plt.tight_layout()
    plt.show()


def plot_acf_residuals(residuals, lags=200):
    _, ax = plt.subplots(figsize=(14, 4))
    plot_acf(residuals.dropna(), lags=lags, ax=ax)
    ax.set_title("ACF of residuals Y_t")
    plt.tight_layout()
    plt.show()

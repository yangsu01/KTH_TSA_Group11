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


def clean_data(df, m=12, sigma=3):
    df = df.copy()
    ma = df["load_mw"].rolling(window=2*m+1, center=True).mean()
    residuals = df["load_mw"] - ma
    anomaly_mask = residuals.abs() > sigma * residuals.std()
    df.loc[anomaly_mask, "load_mw"] = float("nan")
    df["load_mw"] = df.set_index("local")["load_mw"].interpolate(method="time").values
    return df


def get_daily(df):
    return df.set_index("local")["load_mw"].resample("D").mean().dropna()


def print_summary_daily(daily):
    print("\n=== Daily aggregated data (after cleaning) ===")
    print(f"N obs      : {len(daily):,}")
    print(f"Mean       : {daily.mean():,.0f} MW")
    print(f"Median     : {daily.median():,.0f} MW")
    print(f"Std        : {daily.std():,.0f} MW")
    print(f"Min        : {daily.min():,.0f} MW")
    print(f"Max        : {daily.max():,.0f} MW")


def plot_load(daily):
    plt.figure(figsize=(14, 4))
    plt.plot(daily.index, daily.values, linewidth=0.8)
    plt.title("Austria Daily Electricity Load (cleaned)")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.show()


def decompose_daily(daily):
    # Step 1: trend via centered 7-day MA (m=3)
    trend = daily.rolling(window=7, center=True).mean()

    # Step 2: day-of-week seasonal component
    detrended = daily - trend
    dow_means = detrended.groupby(detrended.index.dayofweek).mean()
    dow_means -= dow_means.mean()  # normalize to sum to zero
    seasonal = detrended.index.dayofweek.map(dow_means)
    seasonal = pd.Series(seasonal.values, index=daily.index)

    # Step 3: residuals
    residuals = daily - trend - seasonal

    # Step 4a: print seasonal estimates
    print("\n=== Seasonal estimates (s_j) ===")
    for dow, name in enumerate(WEEKDAY_NAMES):
        print(f"  {name:<12}: {dow_means[dow]:+.2f} MW")

    # Step 4b: summary statistics per stage
    def _print_stats(label, s):
        s = s.dropna()
        print(f"\n=== {label} ===")
        print(f"  Mean   : {s.mean():,.2f} MW")
        print(f"  Median : {s.median():,.2f} MW")
        print(f"  Std    : {s.std():,.2f} MW")
        print(f"  Min    : {s.min():,.2f} MW")
        print(f"  Max    : {s.max():,.2f} MW")
        return s.std()

    std_x  = _print_stats("X_t (daily cleaned)", daily)
    std_dt = _print_stats("X_t - m_t (detrended)", detrended)
    std_y  = _print_stats("Y_t (residuals)", residuals)

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


def plot_acf_residuals(residuals, lags=50):
    _, ax = plt.subplots(figsize=(14, 4))
    plot_acf(residuals.dropna(), lags=lags, ax=ax)
    ax.set_title("ACF of residuals Y_t")
    plt.tight_layout()
    plt.show()

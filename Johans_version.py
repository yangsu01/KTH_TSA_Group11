import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

plt.rcParams.update({"font.size": 15, "axes.labelpad": 2})


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
    df["load_ma"] = df["load_mw"].rolling(window=2*m+1, center=True).mean()
    return df, m


def print_summary_daily(df):
    daily = df.set_index("local")["load_mw"].resample("D").mean()
    print("\n=== Daily aggregated data (after cleaning) ===")
    print(f"N obs      : {len(daily):,}")
    print(f"Mean       : {daily.mean():,.0f} MW")
    print(f"Median     : {daily.median():,.0f} MW")
    print(f"Std        : {daily.std():,.0f} MW")
    print(f"Min        : {daily.min():,.0f} MW")
    print(f"Max        : {daily.max():,.0f} MW")


def plot_load(df, m):
    plt.figure(figsize=(14, 4))
    plt.plot(df["local"], df["load_mw"], linewidth=0.5, alpha=0.5, label="Raw")
    plt.plot(df["local"], df["load_ma"], linewidth=1.2, label=f"MA (m={m})")
    plt.legend()
    plt.title("Austria Hourly Electricity Load")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.show()


def plot_acf_load(df):
    _, ax = plt.subplots(figsize=(14, 4))
    plot_acf(df["load_mw"].dropna(), lags=100, ax=ax)
    ax.set_title("ACF – Austria Hourly Load")
    plt.tight_layout()
    plt.show()


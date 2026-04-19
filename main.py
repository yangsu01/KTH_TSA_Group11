from Johans_version import (
    load_data, print_summary_raw, clean_hourly,
    get_daily, clean_daily, print_summary_daily,
    plot_load, plot_december_january,
    decompose_daily, plot_decomposition,
    plot_residuals_stationarity, plot_acf_residuals,
)


def main():
    # Load and clean hourly data
    df_raw = load_data()
    print_summary_raw(df_raw)
    df = clean_hourly(df_raw)

    # Aggregate to daily
    daily = get_daily(df)
    print_summary_daily(daily, "Raw daily data")
    plot_load(daily)
    plot_december_january(daily)

    # Clean daily: hard threshold + holidays + 3σ pass
    daily = clean_daily(daily)
    print_summary_daily(daily, "Cleaned daily data")

    # Decompose once on fully cleaned data
    trend, _, residuals = decompose_daily(daily)
    plot_decomposition(daily, trend, residuals)
    plot_residuals_stationarity(residuals)
    plot_acf_residuals(residuals)


if __name__ == "__main__":
    main()

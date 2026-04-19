from Johans_version import (
    load_data, print_summary_raw, clean_data,
    get_daily, print_summary_daily, plot_load,
    decompose_daily, plot_decomposition, plot_acf_residuals,
)


def main():
    df_raw = load_data()
    print_summary_raw(df_raw)

    df = clean_data(df_raw)
    daily = get_daily(df)
    print_summary_daily(daily)

    plot_load(daily)
    trend, _, residuals = decompose_daily(daily)
    plot_decomposition(daily, trend, residuals)
    plot_acf_residuals(residuals)


if __name__ == "__main__":
    main()

from Johans_version import (
    load_data, print_summary_raw, clean_data, print_summary_daily,
    plot_load, plot_acf_load,
)


def main():
    df_raw = load_data()
    print_summary_raw(df_raw)

    df, m = clean_data(df_raw)
    print_summary_daily(df)

    plot_load(df, m)
    plot_acf_load(df)


if __name__ == "__main__":
    main()

import argparse
import os

from market_data_maintainer import (
    CHART_DATA_FILE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHART_YEARS,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    ensure_shared_data,
    load_cache,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update chart history cache only.")
    parser.add_argument(
        "--chart-years",
        type=int,
        default=DEFAULT_CHART_YEARS,
        help=f"Years of chart history to keep (default: {DEFAULT_CHART_YEARS})",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Price lookback used for eligibility checks (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Tickers per yfinance batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=f"Seconds between batches (default: {DEFAULT_COOLDOWN_SECONDS})",
    )
    args = parser.parse_args()

    ensure_shared_data(
        lookback_days=args.lookback_days,
        chart_years=args.chart_years,
        batch_size=args.batch_size,
        cooldown=args.cooldown,
        run_prices=False,
        run_charts=True,
    )

    df = load_cache(CHART_DATA_FILE)
    if df.empty:
        print("  Charts: (empty)")
    else:
        size_mb = os.path.getsize(CHART_DATA_FILE) / 1_048_576 if os.path.exists(CHART_DATA_FILE) else 0
        print(f"  Charts: {len(df.columns)} tickers, {len(df)} rows  ({size_mb:.1f} MB)")

    print("\nChart cache refresh done.")

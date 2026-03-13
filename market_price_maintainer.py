import argparse
import os

from market_data_maintainer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    ETF_DATA_FILE,
    STOCK_DATA_FILE,
    ensure_shared_data,
    load_cache,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update stock/ETF market data caches (no chart backfill).")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Calendar days of price history to keep (default: {DEFAULT_LOOKBACK_DAYS})",
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
        batch_size=args.batch_size,
        cooldown=args.cooldown,
        run_prices=True,
        run_charts=False,
    )

    for label, path in [("Stocks", STOCK_DATA_FILE), ("ETFs", ETF_DATA_FILE)]:
        df = load_cache(path)
        if df.empty:
            print(f"  {label}: (empty)")
        else:
            size_mb = os.path.getsize(path) / 1_048_576 if os.path.exists(path) else 0
            print(f"  {label}: {len(df.columns)} tickers, {len(df)} rows  ({size_mb:.1f} MB)")

    print("\nPrice cache refresh done.")

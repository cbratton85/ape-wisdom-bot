import os
import re
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm


STOCK_DATA_FILE = "stock_data.csv.gz"
ETF_DATA_FILE = "etf_data.csv.gz"


# ==========================================
# CONFIG
# ==========================================
# Cache files
CHART_DATA_FILE = "chart_data.csv.gz"
ETF_CSV_FILE = "ETFs.csv"
STOCK_CSV_FILE = "STOCKS.csv"

# Defaults
DEFAULT_LOOKBACK_DAYS = 504
DEFAULT_CHART_YEARS = 2
DEFAULT_BATCH_SIZE = 40
DEFAULT_COOLDOWN_SECONDS = 1.5

# Universe prefilter (set to 0 to disable)
MIN_PRICE_FILTER = 1.0
MIN_AVG_VOLUME_FILTER = 100000
PREFILTER_LOOKBACK_DAYS = 45
PREFILTER_VOL_DAYS = 30
PREFILTER_BATCH_SIZE = 80
PREFILTER_COOLDOWN_SECONDS = 0.25

# Internal tuning
MIN_HISTORY_COVERAGE = 0.8
TRIM_BUFFER = 1.2
CHART_TRIM_BUFFER = 1.1
YF_TIMEOUT_SECONDS = 20
DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 5
PARTIAL_RETRY_BATCH_SIZE = 8
PARTIAL_RETRY_SLEEP_SECONDS = 8
FULL_BATCH_RECOVERY_SLEEP_SECONDS = 15
SAVE_RETRIES = 10
SAVE_RETRY_SECONDS = 0.8


# ==========================================
# TICKER HELPERS
# ==========================================
def _is_ticker(s):
    return bool(re.match(r"^[A-Z]{1,6}$", str(s).strip().upper()))


def _read_tickers(path):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        return []
    if df.empty:
        return []
    raw = df.iloc[:, 0].dropna()
    vals = raw.astype(str).str.strip().str.upper().tolist()
    return [v for v in vals if _is_ticker(v) and v not in {"SYMBOL", "TICKER", "NAN", "NONE", "NULL"}]


def load_master_tickers_by_type():
    stocks = sorted(set(_read_tickers(STOCK_CSV_FILE)))
    etfs = sorted(set(_read_tickers(ETF_CSV_FILE)))
    return stocks, etfs


# ==========================================
# CACHE I/O
# ==========================================


def load_cache(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.loc[:, ~df.columns.duplicated()]
    except Exception:
        return pd.DataFrame()


def safe_save(df, path):
    # Use a process-unique temp name and retry atomic replace for transient
    # Windows/OneDrive file locks.
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, compression="gzip")

    last_err = None
    for _ in range(SAVE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as err:
            last_err = err
            time.sleep(SAVE_RETRY_SECONDS)

    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass

    raise PermissionError(
        f"Failed to replace '{path}' after {SAVE_RETRIES} retries; file may be locked by another process."
    ) from last_err


def _status_tag(label, fallback):
    return f"  [{label}]" if label else f"  [{fallback}]"


def _chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _merge_frames(frames):
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    merged = pd.concat(valid, axis=1)
    return merged.loc[:, ~merged.columns.duplicated(keep="last")]


def _missing_symbols(requested, available_cols):
    available = set(available_cols)
    return [ticker for ticker in requested if ticker not in available]


def _fallback_chunk_size(missing_count):
    if missing_count <= 4:
        return 1
    return min(PARTIAL_RETRY_BATCH_SIZE, max(1, missing_count // 2))


def _download_raw(clean_tickers, start_date, threads):
    return yf.download(
        clean_tickers,
        start=start_date,
        progress=False,
        group_by="ticker",
        auto_adjust=True,
        threads=threads,
        timeout=YF_TIMEOUT_SECONDS,
    )


def _extract_field_frame(df, clean_tickers, field):
    if df is None or df.empty:
        return pd.DataFrame()

    out_cols = {}
    for ticker in clean_tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.get_level_values(0):
                    ticker_df = df[ticker]
                    if field in ticker_df.columns:
                        out_cols[ticker] = ticker_df[field]
            else:
                if field in df.columns and len(clean_tickers) == 1:
                    out_cols[ticker] = df[field]
        except Exception:
            continue

    return pd.DataFrame(out_cols) if out_cols else pd.DataFrame()


def _extract_snapshot_frames(df, clean_tickers):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    close_cols = {}
    vol_cols = {}

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        for ticker in clean_tickers:
            if ticker not in lvl0:
                continue
            ticker_df = df[ticker]
            if "Close" in ticker_df.columns:
                close_cols[ticker] = ticker_df["Close"]
            if "Volume" in ticker_df.columns:
                vol_cols[ticker] = ticker_df["Volume"]
    elif len(clean_tickers) == 1:
        ticker = clean_tickers[0]
        if "Close" in df.columns:
            close_cols[ticker] = df["Close"]
        if "Volume" in df.columns:
            vol_cols[ticker] = df["Volume"]

    close_out = pd.DataFrame(close_cols) if close_cols else pd.DataFrame()
    vol_out = pd.DataFrame(vol_cols) if vol_cols else pd.DataFrame()
    return close_out, vol_out


# ==========================================
# DOWNLOAD / UPDATE PIPELINE
# ==========================================


def _download_batch(tickers, start_date, field="Close", retries=DOWNLOAD_RETRIES):
    clean = [t.replace("/", "-") for t in tickers]
    result = pd.DataFrame()

    for attempt in range(retries):
        try:
            df = _download_raw(clean, start_date, threads=False)
            if df is None or df.empty:
                if attempt < retries - 1:
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue
                break

            result = _extract_field_frame(df, clean, field)
            missing = _missing_symbols(clean, result.columns)
            if not missing:
                return result
            break
        except Exception:
            if attempt < retries - 1:
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                break

    missing = _missing_symbols(clean, result.columns)
    if missing and len(clean) > 1:
        time.sleep(PARTIAL_RETRY_SLEEP_SECONDS if not result.empty else FULL_BATCH_RECOVERY_SLEEP_SECONDS)
        retry_frames = [result] if not result.empty else []
        for chunk in _chunked(missing, _fallback_chunk_size(len(missing))):
            retry_frames.append(_download_batch(chunk, start_date, field=field, retries=max(1, retries - 1)))
        return _merge_frames(retry_frames)

    return result


def _download_prefilter_snapshot(tickers, start_date, retries=DOWNLOAD_RETRIES):
    """Download Close+Volume snapshot in one request for prefiltering."""
    clean = [t.replace("/", "-") for t in tickers]
    close_out = pd.DataFrame()
    vol_out = pd.DataFrame()

    for attempt in range(retries):
        try:
            df = _download_raw(clean, start_date, threads=False)
            if df is None or df.empty:
                if attempt < retries - 1:
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue
                break
            close_out, vol_out = _extract_snapshot_frames(df, clean)
            resolved = set(close_out.columns).union(vol_out.columns)
            if not _missing_symbols(clean, resolved):
                return close_out, vol_out
            break
        except Exception:
            if attempt < retries - 1:
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                break

    resolved = set(close_out.columns).union(vol_out.columns)
    missing = _missing_symbols(clean, resolved)
    if missing and len(clean) > 1:
        time.sleep(PARTIAL_RETRY_SLEEP_SECONDS if resolved else FULL_BATCH_RECOVERY_SLEEP_SECONDS)
        close_frames = [close_out] if not close_out.empty else []
        vol_frames = [vol_out] if not vol_out.empty else []
        for chunk in _chunked(missing, _fallback_chunk_size(len(missing))):
            retry_close, retry_vol = _download_prefilter_snapshot(chunk, start_date, retries=max(1, retries - 1))
            if not retry_close.empty:
                close_frames.append(retry_close)
            if not retry_vol.empty:
                vol_frames.append(retry_vol)
        return _merge_frames(close_frames), _merge_frames(vol_frames)

    return close_out, vol_out


def _backfill_missing(df, missing, start_date, save_path, batch_size=DEFAULT_BATCH_SIZE, cooldown=DEFAULT_COOLDOWN_SECONDS, label=""):
    if not missing:
        tag = _status_tag(label, "backfill")
        print(f"{tag} All tickers already in cache — skipping backfill.")
        return df
    batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
    desc = f"  Backfill {label}" if label else "  Backfill"
    with tqdm(total=len(missing), desc=desc, unit="ticker", ncols=80, file=sys.stdout) as bar:
        for batch in batches:
            batch_df = _download_batch(batch, start_date, field="Close")
            if not batch_df.empty:
                df = pd.concat([df, batch_df], axis=1)
                df = df.loc[:, ~df.columns.duplicated(keep="last")]  # fresh data wins
                safe_save(df, save_path)
            bar.update(len(batch))
            time.sleep(cooldown)
    return df


def _update_latest(df, save_path, batch_size=DEFAULT_BATCH_SIZE, cooldown=DEFAULT_COOLDOWN_SECONDS, label=""):
    if df.empty:
        return df
    last_date = df.index.max()
    tag = _status_tag(label, "update")
    if pd.isna(last_date) or last_date.date() >= datetime.now().date():
        print(f"{tag} Already up to date (last: {last_date.date()}) — skipping update.")
        return df

    print(f"{tag} Updating from {last_date.date()} …")
    start = last_date.strftime("%Y-%m-%d")
    tickers = df.columns.tolist()
    new_rows = []
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    desc = f"  Update  {label}" if label else "  Update"
    with tqdm(total=len(tickers), desc=desc, unit="ticker", ncols=80, file=sys.stdout) as bar:
        for i, batch in enumerate(batches):
            batch_df = _download_batch(batch, start, field="Close")
            if i == 0 and (batch_df is None or batch_df.empty):
                bar.update(len(tickers))
                break
            if not batch_df.empty:
                new_rows.append(batch_df)
            bar.update(len(batch))
            time.sleep(cooldown)

    if new_rows:
        upd = pd.concat(new_rows, axis=1)
        df = pd.concat([df, upd], axis=0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        safe_save(df, save_path)
    return df


def _prefilter_by_price_and_volume(tickers, label, batch_size=PREFILTER_BATCH_SIZE, cooldown=PREFILTER_COOLDOWN_SECONDS):
    if not tickers:
        return []
    if MIN_PRICE_FILTER <= 0 and MIN_AVG_VOLUME_FILTER <= 0:
        return tickers

    start = (datetime.now() - timedelta(days=PREFILTER_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    kept = []
    dropped_price = 0
    dropped_volume = 0
    unresolved = 0

    desc = f"  Prefilter {label}"
    with tqdm(total=len(tickers), desc=desc, unit="ticker", ncols=80, file=sys.stdout) as bar:
        for batch in batches:
            close_df, vol_df = _download_prefilter_snapshot(batch, start)

            for t in batch:
                key = t.replace("/", "-")

                close_s = close_df[key] if not close_df.empty and key in close_df.columns else pd.Series(dtype=float)
                vol_s = vol_df[key] if not vol_df.empty and key in vol_df.columns else pd.Series(dtype=float)

                close_clean = close_s.dropna()
                vol_clean = vol_s.dropna()
                last_close = float(close_clean.tail(1).mean()) if not close_clean.empty else float("nan")
                avg_vol = float(vol_clean.tail(max(1, PREFILTER_VOL_DAYS)).mean()) if not vol_clean.empty else float("nan")

                # If snapshot data is unavailable, keep ticker (avoid over-pruning on transient API gaps).
                if not np.isfinite(last_close) and not np.isfinite(avg_vol):
                    kept.append(t)
                    unresolved += 1
                    continue

                if MIN_PRICE_FILTER > 0 and np.isfinite(last_close) and last_close < MIN_PRICE_FILTER:
                    dropped_price += 1
                    continue

                if MIN_AVG_VOLUME_FILTER > 0 and np.isfinite(avg_vol) and avg_vol < MIN_AVG_VOLUME_FILTER:
                    dropped_volume += 1
                    continue

                kept.append(t)

            bar.update(len(batch))
            if cooldown > 0:
                time.sleep(cooldown)

    print(
        f"  [{label}] Prefilter kept {len(kept)}/{len(tickers)} "
        f"(dropped price={dropped_price}, dropped volume={dropped_volume}, unresolved-kept={unresolved})"
    )
    return kept


def _trim_cache(path, allowed_cols, keep_days):
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[[c for c in df.columns if c in allowed_cols]]
        df = df.tail(keep_days)
        safe_save(df, path)
    except Exception:
        pass


def ensure_shared_data(
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    chart_years=DEFAULT_CHART_YEARS,
    batch_size=DEFAULT_BATCH_SIZE,
    cooldown=DEFAULT_COOLDOWN_SECONDS,
):
    stocks, etfs = load_master_tickers_by_type()
    stock_df = load_cache(STOCK_DATA_FILE)
    etf_df = load_cache(ETF_DATA_FILE)

    # Tickers are treated as "missing" if absent OR if they have fewer than 80% of
    # the requested lookback days of actual data (truncated / corrupted columns).
    min_rows = int(lookback_days * MIN_HISTORY_COVERAGE)
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    def _is_well_covered(df, ticker):
        return ticker in df.columns and int(df[ticker].notna().sum()) >= min_rows

    # Fast path: skip prefilter checks for symbols that already have healthy cache coverage.
    stock_ready = [t for t in stocks if _is_well_covered(stock_df, t)]
    etf_ready = [t for t in etfs if _is_well_covered(etf_df, t)]
    stock_candidates = [t for t in stocks if t not in set(stock_ready)]
    etf_candidates = [t for t in etfs if t not in set(etf_ready)]

    stocks = sorted(set(stock_ready + _prefilter_by_price_and_volume(stock_candidates, "Stocks")))
    etfs = sorted(set(etf_ready + _prefilter_by_price_and_volume(etf_candidates, "ETFs  ")))
    all_tickers = sorted(set(stocks + etfs))

    def _needs_backfill(df, ticker):
        return ticker not in df.columns or int(df[ticker].notna().sum()) < min_rows

    missing = [t for t in stocks if _needs_backfill(stock_df, t)]
    stock_df = _backfill_missing(stock_df, missing, start, STOCK_DATA_FILE, batch_size, cooldown, label="Stocks")
    stock_df = _update_latest(stock_df, STOCK_DATA_FILE, batch_size, cooldown, label="Stocks")

    missing = [t for t in etfs if _needs_backfill(etf_df, t)]
    etf_df = _backfill_missing(etf_df, missing, start, ETF_DATA_FILE, batch_size, cooldown, label="ETFs  ")
    etf_df = _update_latest(etf_df, ETF_DATA_FILE, batch_size, cooldown, label="ETFs  ")

    chart_df = load_cache(CHART_DATA_FILE)
    chart_days = max(1, int(chart_years)) * 365
    chart_min_rows = int(chart_days * MIN_HISTORY_COVERAGE)
    start_chart = (datetime.now() - timedelta(days=chart_days)).strftime("%Y-%m-%d")

    def _needs_chart_backfill(ticker):
        return ticker not in chart_df.columns or int(chart_df[ticker].notna().sum()) < chart_min_rows

    missing = [t for t in all_tickers if _needs_chart_backfill(t)]
    chart_df = _backfill_missing(chart_df, missing, start_chart, CHART_DATA_FILE, batch_size, cooldown, label="Charts")
    chart_df = _update_latest(chart_df, CHART_DATA_FILE, batch_size, cooldown, label="Charts")

    # Trim to configured retention windows (+ buffers)
    _trim_cache(STOCK_DATA_FILE, set(stocks), int(lookback_days * TRIM_BUFFER))
    _trim_cache(ETF_DATA_FILE, set(etfs), int(lookback_days * TRIM_BUFFER))
    _trim_cache(CHART_DATA_FILE, set(all_tickers), int(chart_days * CHART_TRIM_BUFFER))


# ==========================================
# DATA INTEGRITY CHECK
# ==========================================
_dropped_tickers: set = set()

def validate_and_repair(df, label="data", min_trading_days=252):
    """Ensure continuous trading-day index, fill gaps, and drop sparse tickers."""
    global _dropped_tickers
    if df.empty:
        return df

    # Build a proper trading-day calendar (business days, Mon-Fri)
    full_idx = pd.bdate_range(start=df.index.min(), end=df.index.max())

    # Reindex to include any missing trading days
    missing_days = full_idx.difference(df.index)
    if len(missing_days) > 0:
        print(f"  [{label}] Filling {len(missing_days)} missing trading days in index.")
        df = df.reindex(full_idx)

    # Snapshot actual available data (before filling) to enforce min trading days
    non_na_counts = df.notna().sum()

    # Forward-fill gaps (carries last known price — standard for market data)
    # limit=10 covers holidays + long weekends; beyond that it's truly missing
    df = df.ffill(limit=10)

    # Back-fill leading NaNs for tickers that started mid-dataset (limit=5)
    df = df.bfill(limit=5)

    # Drop tickers with fewer than min_trading_days actual data points
    before_cols = set(df.columns)
    keep_cols = [c for c in df.columns if non_na_counts.get(c, 0) >= min_trading_days]
    df = df[keep_cols]
    dropped_cols = before_cols - set(df.columns)
    if dropped_cols:
        _dropped_tickers |= dropped_cols
        print(f"  [{label}] Dropped {len(dropped_cols)} tickers with <{min_trading_days} trading days.")

    # Drop any fully-empty rows that might remain
    df = df.dropna(how="all")

    return df


def load_combined_price_cache():
    stock = load_cache(STOCK_DATA_FILE)
    etf = load_cache(ETF_DATA_FILE)
    if stock.empty and etf.empty:
        return pd.DataFrame()
    if stock.empty:
        return etf
    if etf.empty:
        return stock
    out = pd.concat([stock, etf], axis=1)
    return out.loc[:, ~out.columns.duplicated()]


# ==========================================
# CLI ENTRYPOINT
# ==========================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and update shared market data caches.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"Calendar days of price history to keep (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--chart-years", type=int, default=DEFAULT_CHART_YEARS,
                        help=f"Years of chart history to keep (default: {DEFAULT_CHART_YEARS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Tickers per yfinance batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_SECONDS,
                        help=f"Seconds between batches (default: {DEFAULT_COOLDOWN_SECONDS})")
    args = parser.parse_args()

    stocks, etfs = load_master_tickers_by_type()
    print(f"Tickers loaded: {len(stocks)} stocks, {len(etfs)} ETFs")
    print(f"Settings: lookback={args.lookback_days}d, chart={args.chart_years}yr, "
          f"batch={args.batch_size}, cooldown={args.cooldown}s")
    print()

    ensure_shared_data(
        lookback_days=args.lookback_days,
        chart_years=args.chart_years,
        batch_size=args.batch_size,
        cooldown=args.cooldown,
    )

    # Report final cache sizes
    for label, path in [("Stocks", STOCK_DATA_FILE), ("ETFs", ETF_DATA_FILE), ("Charts", CHART_DATA_FILE)]:
        df = load_cache(path)
        if df.empty:
            print(f"  {label}: (empty)")
        else:
            size_mb = os.path.getsize(path) / 1_048_576 if os.path.exists(path) else 0
            print(f"  {label}: {len(df.columns)} tickers, {len(df)} rows  ({size_mb:.1f} MB)")

    print("\nDone.")

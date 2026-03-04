import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import json
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import multiprocessing as mp

# ==========================================
# CONFIG
# ==========================================
DATA_FILE        = "historical_data.csv.gz"
CHART_DATA_FILE  = "chart_data.csv.gz"        # Extended history for Z-score charts only
VOLUME_DATA_FILE = "volume_data.csv.gz"        # Average daily volume per ticker
MCAP_CACHE_FILE  = "market_cap.json"           # Market cap per ticker
TRADES_FILE      = "active_trades.json"        # Active trade tracker
BATCH_SIZE = 40
COOLDOWN = 1.5
LOOKBACK_DAYS       = 650   # Days used for scoring / correlation / perf
CHART_LOOKBACK_DAYS = 1825  # ~5 years used for Z-score chart history
VOL_AVG_DAYS        = 30    # Rolling window for average volume calculation
CACHE_UPDATE_COOLDOWN_HOURS = 1
VOL_MCAP_COOLDOWN_HOURS    = 168   # Volume & market cap refresh interval (168h = 1 week)
NUM_WORKERS = max(1, (mp.cpu_count() or 2) - 0)  # CPU cores for parallel pair analysis

CORR_SHORT = 35
CORR_LONG = 100
Z_LENGTH = 100
Z_LENGTH_SHORT = 30
Z_LENGTH_LONG  = 250
PERF_LENGTH = 300

MIN_CORR_FILTER = 0.60
Z_THRESHOLD = 1.5
Z_MAX       = 5.0      # Max |Z| — above this is likely a structural break, not mean-reversion
Z_STRONG = 2.0

ADF_CONFIDENCE  = 0.95   # Min cointegration confidence (0.90=90%, 0.95=95%, 0.99=99%)
ADF_LOOKBACK_YRS = 3     # Years of data for cointegration test (1, 2, 3, 5, etc.)
ADF_MIN_DAYS    = 252    # Min trading days of spread data required for ADF (252 ≈ 1yr)

MIN_EST_RETURN = 1       # Min estimated return % to include pair (0 = no filter)
MIN_PRICE      = 1.00    # Exclude pairs where either ticker is below this price
MIN_AVG_VOLUME = 0       # Exclude pairs where either ticker avg daily volume is below this
# Market cap tiers: "mega", "large", "mid", "small", "micro", "nano", "none"
# mega=200B+  large=10B+  mid=2B+  small=300M+  micro=50M+  nano=1M+  none=no filter
MCAP_TIERS = {"mega": 200_000_000_000, "large": 10_000_000_000, "mid": 2_000_000_000,
              "small": 300_000_000, "micro": 50_000_000, "nano": 1_000_000, "none": 0}
MIN_MCAP_STOCK = "none"   # Min market cap tier for stocks (see tiers above)
MIN_MCAP_ETF   = "none"   # Min market cap tier for ETFs (see tiers above)

MAX_RESULTS    = 0        # Max total pairs to show in HTML (0 = show all)
MAX_RESULTS_ETF   = 100   # Max Pure ETF pairs (0 = no per-category limit)
MAX_RESULTS_STOCK = 350   # Max Pure Stock pairs (0 = no per-category limit)
MAX_RESULTS_MIXED = 50   # Max Mixed pairs (0 = no per-category limit)
MAX_CHARTS     = 0        # Max pairs to compute Z-score charts for (0 = show all)

W_ZSCORE    = 0.25   # Z-score magnitude (how far from mean)
W_HALFLIFE  = 0.25   # Half-life speed (faster reversion = more tradeable)
W_CONFIRM   = 0.20   # Timeframe confirmation (alignment + confidence combined)
W_ANNRET    = 0.15   # Annualized return potential (higher = more profitable)
W_STATIONARY = 0.10  # Spread stationarity (ADF tiebreaker — already filtered)
W_CORR      = 0.05   # Base correlation level

# ==========================================
# LOAD MASTER TICKERS
# ==========================================
TICKER_TYPES    = {}   # ticker -> "Pure ETF" | "Pure Stock"
TICKER_NAMES    = {}   # ticker -> human-readable name
TICKER_INDUSTRY = {}   # ticker -> sector  (col 3)
TICKER_SUBIND   = {}   # ticker -> industry (col 4)
TICKER_SUBIND2  = {}   # ticker -> sub-industry (col 5, stocks only)
ETF_LEV_TYPES   = {}   # ticker -> type tag (see _ETF_TYPE_MAP values)

# Maps ETFs.csv col-5 type-strings -> internal tag
_ETF_TYPE_MAP = {
    "etf":                      "normal",
    "etf, leveraged":           "leveraged",
    "etf, inverse":             "inverse",
    "etf, leveraged, inverse":  "lev_inv",
    "etn":                      "etn",
    "etn, leveraged":           "etn_lev",
    "etn, leveraged, inverse":  "etn_lev_inv",
}

def load_master_tickers():
    global TICKER_TYPES, TICKER_NAMES, TICKER_INDUSTRY, TICKER_SUBIND, TICKER_SUBIND2, ETF_LEV_TYPES
    tickers = []

    if os.path.exists("ETFs.csv"):
        df_etf = pd.read_csv("ETFs.csv", header=None)
        etfs = df_etf[0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers += etfs
        ncols = df_etf.shape[1]
        for _, row in df_etf.iterrows():
            t = str(row.iloc[0]).strip().upper()
            if not t or t in ("", "NONE", "NAN", "SYMBOL", "TICKER"):
                continue
            TICKER_TYPES[t] = "Pure ETF"
            if ncols >= 2 and pd.notna(row.iloc[1]):
                TICKER_NAMES[t] = str(row.iloc[1]).strip()
            if ncols >= 3 and pd.notna(row.iloc[2]):
                TICKER_INDUSTRY[t] = str(row.iloc[2]).strip()
            if ncols >= 4 and pd.notna(row.iloc[3]):
                TICKER_SUBIND[t] = str(row.iloc[3]).strip()
            if ncols >= 5 and pd.notna(row.iloc[4]):
                raw_type = str(row.iloc[4]).strip().lower()
                ETF_LEV_TYPES[t] = _ETF_TYPE_MAP.get(raw_type, "normal")
            else:
                ETF_LEV_TYPES[t] = "normal"
        n_lev    = sum(1 for v in ETF_LEV_TYPES.values() if v == "leveraged")
        n_inv    = sum(1 for v in ETF_LEV_TYPES.values() if v == "inverse")
        n_levinv = sum(1 for v in ETF_LEV_TYPES.values() if v == "lev_inv")
        n_etn    = sum(1 for v in ETF_LEV_TYPES.values() if v.startswith("etn"))
        print(f"ETFs.csv: {len(etfs)} tickers  |  leveraged={n_lev}  inverse={n_inv}  lev+inv={n_levinv}  etn={n_etn}")

    if os.path.exists("STOCKS.csv"):
        df_stock = pd.read_csv("STOCKS.csv", header=None)
        stocks = df_stock[0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers += stocks
        ncols = df_stock.shape[1]
        for _, row in df_stock.iterrows():
            t = str(row.iloc[0]).strip().upper()
            if not t or t in ("", "NONE", "NAN", "SYMBOL", "TICKER"):
                continue
            TICKER_TYPES[t] = "Pure Stock"
            if ncols >= 2 and pd.notna(row.iloc[1]):
                TICKER_NAMES[t] = str(row.iloc[1]).strip()
            if ncols >= 3 and pd.notna(row.iloc[2]):
                TICKER_INDUSTRY[t] = str(row.iloc[2]).strip()
            if ncols >= 4 and pd.notna(row.iloc[3]):
                TICKER_SUBIND[t] = str(row.iloc[3]).strip()
            if ncols >= 5 and pd.notna(row.iloc[4]):
                TICKER_SUBIND2[t] = str(row.iloc[4]).strip()

    tickers = list(set(tickers))
    tickers = [t for t in tickers if t not in ["", "NONE", "NAN", "SYMBOL", "TICKER"]]
    print(f"Loaded {len(tickers)} tickers total.")
    return tickers


# ==========================================
# SAFE SAVE
# ==========================================
def safe_save(df):
    tmp = DATA_FILE + ".tmp"
    df.to_csv(tmp, compression='gzip')
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    os.rename(tmp, DATA_FILE)


# ==========================================
# LOAD CACHE
# ==========================================
def load_cache():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
            df = df.loc[:, ~df.columns.duplicated()]
            print("Cache loaded.")
            return df
        except:
            pass
    return pd.DataFrame()


# ==========================================
# DOWNLOAD BATCH  (Close prices, or Volume)
# ==========================================
def download_batch(tickers, start_date, field="Close"):
    clean = [t.replace("/", "-") for t in tickers]
    max_retries = 3

    for attempt in range(max_retries):
        try:
            df = yf.download(
                clean,
                start=start_date,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                threads=False,
                timeout=20
            )

            if df is None or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return pd.DataFrame()

            result = pd.DataFrame()
            for t in clean:
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        if t in df.columns.levels[0]:
                            result[t] = df[t][field]
                    else:
                        if not df[field].empty:
                            result[t] = df[field]
                except Exception as e:
                    continue

            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt+1} failed, retrying...")
                time.sleep(5)
            else:
                print(f"  Batch failed after {max_retries} attempts: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


# ==========================================
# ACTIVE TRADE SYMBOLS
# ==========================================
def get_active_trade_symbols():
    """Return set of ticker symbols from open trades in active_trades.json."""
    if not os.path.exists(TRADES_FILE):
        return set()
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
        syms = set()
        for t in trades:
            if t.get("status") == "open" and "/" in t.get("pair", ""):
                a, b = t["pair"].split("/")
                syms.add(a)
                syms.add(b)
        return syms
    except Exception:
        return set()


def _refresh_tickers(data, tickers):
    """Force-download latest prices for specific tickers and merge into data."""
    if not tickers:
        return data
    # Use the last cached date (not +1) to re-fetch today's data without going into the future
    today_str = data.index.max().strftime("%Y-%m-%d") if not data.empty else (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"  Refreshing {len(tickers)} trade tickers: {', '.join(sorted(tickers)[:10])}{'...' if len(tickers) > 10 else ''}")
    # Preserve the cache file timestamp so trade refresh doesn't reset the cooldown
    original_mtime = os.path.getmtime(DATA_FILE) if os.path.exists(DATA_FILE) else None
    fresh = download_batch(list(tickers), today_str)
    if not fresh.empty:
        for col in fresh.columns:
            if col in data.columns:
                data.loc[fresh.index, col] = fresh[col]
            else:
                data[col] = fresh[col]
        safe_save(data)
        if original_mtime is not None:
            os.utime(DATA_FILE, (original_mtime, original_mtime))
    return data


# ==========================================
# BUILD DATASET
# ==========================================
def build_dataset(master):
    data = load_cache()

    if os.path.exists(DATA_FILE) and not data.empty:
        file_time = os.path.getmtime(DATA_FILE)
        last_update = datetime.fromtimestamp(file_time)
        hours_since_update = (datetime.now() - last_update).total_seconds() / 3600

        if hours_since_update < CACHE_UPDATE_COOLDOWN_HOURS:
            print(f"--- Cache is fresh ({round(hours_since_update, 2)}h old). Skipping download. ---")
            # Force-refresh active trade symbols even when cache is fresh
            trade_syms = get_active_trade_symbols()
            trade_syms = trade_syms & set(data.columns)  # only refresh tickers we already have
            if trade_syms:
                data = _refresh_tickers(data, trade_syms)
            data = data[[c for c in data.columns if c in master]]
            data = data.tail(LOOKBACK_DAYS)
            data = data.ffill().bfill()
            data = data.dropna(axis=1, thresh=len(data) * 0.2)
            return data

    existing = data.columns.tolist() if not data.empty else []
    missing = [t for t in master if t not in existing]

    if missing:
        total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Backfilling {len(missing)} tickers in {total_batches} batches...")
        start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

        for i, idx in enumerate(range(0, len(missing), BATCH_SIZE)):
            batch_num = i + 1
            batch = missing[idx: idx + BATCH_SIZE]
            print(f"[{batch_num}/{total_batches}] Downloading: {batch[0]}...{batch[-1] if len(batch) > 1 else ''}")
            batch_df = download_batch(batch, start)
            if not batch_df.empty:
                data = pd.concat([data, batch_df], axis=1)
                data = data.loc[:, ~data.columns.duplicated()]
                safe_save(data)
            time.sleep(COOLDOWN)

    if not data.empty:
        last_date = data.index.max()
        today = datetime.now().date()

        if last_date.date() < today:
            start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            tickers_to_update = data.columns.tolist()
            total_update_batches = (len(tickers_to_update) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Updating {len(tickers_to_update)} tickers ({total_update_batches} batches) starting from {start}...")

            new_rows = []
            for i, idx in enumerate(range(0, len(tickers_to_update), BATCH_SIZE)):
                batch = tickers_to_update[idx: idx + BATCH_SIZE]
                print(f"  [{i+1}/{total_update_batches}] Updating: {batch[0]}...")
                
                batch_df = download_batch(batch, start)
                
                # Market Closed Optimization: If the first batch is empty, 
                # no new data exists for today. Stop immediately.
                if i == 0 and (batch_df is None or batch_df.empty):
                    print(">>> No new data available today (Market likely closed). Skipping.")
                    break
                
                if not batch_df.empty:
                    new_rows.append(batch_df)
                
                time.sleep(COOLDOWN)

            # --- SAVE ONLY ONCE AFTER THE LOOP ---
            if new_rows:
                print("Merging updates and saving to disk...")
                # Combine batches side-by-side
                update_df = pd.concat(new_rows, axis=1)
                
                # Append to bottom of main data
                data = pd.concat([data, update_df], axis=0)
                
                # Deduplicate and Sort
                data = data[~data.index.duplicated(keep="last")].sort_index()
                
                # Save once using your safe_save (which now has compression='gzip')
                safe_save(data)
                print("Historical data updated and saved.")

    # Final filtering and cleanup
    data = data[[c for c in data.columns if c in master]]
    data = data.tail(LOOKBACK_DAYS)
    data = data.ffill().bfill()
    data = data.dropna(axis=1, thresh=len(data) * 0.2)
    safe_save(data)

    print(f"Final dataset ready: {len(data.columns)} tickers.")
    return data


# ==========================================
# BUILD EXTENDED CHART DATASET  (~5 years)
# ==========================================
def build_chart_dataset(master):
    """Downloads ~5 years of Close data for Z-score chart history.
    Uses its own cache file so it does not interfere with scoring data."""
    if os.path.exists(CHART_DATA_FILE):
        try:
            chart_data = pd.read_csv(CHART_DATA_FILE, index_col=0, parse_dates=True)
            chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()]
            file_time  = os.path.getmtime(CHART_DATA_FILE)
            hours_old  = (datetime.now() - datetime.fromtimestamp(file_time)).total_seconds() / 3600
            if hours_old < CACHE_UPDATE_COOLDOWN_HOURS:
                print(f"--- Chart cache fresh ({round(hours_old,2)}h). Skipping chart download. ---")
                chart_data = chart_data[[c for c in chart_data.columns if c in master]]
                return chart_data
        except:
            chart_data = pd.DataFrame()
    else:
        chart_data = pd.DataFrame()

    existing = chart_data.columns.tolist() if not chart_data.empty else []
    missing  = [t for t in master if t not in existing]
    start    = (datetime.now() - timedelta(days=CHART_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    if missing:
        total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Downloading {CHART_LOOKBACK_DAYS}-day chart history for {len(missing)} tickers ({total_batches} batches)...")
        for i, idx in enumerate(range(0, len(missing), BATCH_SIZE)):
            batch = missing[idx: idx + BATCH_SIZE]
            print(f"  [{i+1}/{total_batches}] {batch[0]}...")
            batch_df = download_batch(batch, start, field="Close")
            if not batch_df.empty:
                chart_data = pd.concat([chart_data, batch_df], axis=1)
                chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()]
                tmp = CHART_DATA_FILE + ".tmp"
                chart_data.to_csv(tmp, compression='gzip')
                if os.path.exists(CHART_DATA_FILE): os.remove(CHART_DATA_FILE)
                os.rename(tmp, CHART_DATA_FILE)
            time.sleep(COOLDOWN)

    if not chart_data.empty:
        last_date = chart_data.index.max()
        if last_date.date() < datetime.now().date():
            upd_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            tickers_to_upd = chart_data.columns.tolist()
            total_upd = (len(tickers_to_upd) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Updating chart cache ({total_upd} batches) from {upd_start}...")
            
            new_batches = []
            for i, idx in enumerate(range(0, len(tickers_to_upd), BATCH_SIZE)):
                batch = tickers_to_upd[idx: idx + BATCH_SIZE]
                print(f"  [{i+1}/{total_upd}] Updating: {batch[0]}...")
                
                batch_df = download_batch(batch, upd_start, field="Close")
                
                # Market Closed Optimization: If the first batch returns nothing, 
                # it's likely a weekend or holiday. Stop immediately.
                if i == 0 and (batch_df is None or batch_df.empty):
                    print(">>> No new data available (Market likely closed). Skipping update.")
                    break
                
                if not batch_df.empty:
                    new_batches.append(batch_df)
                
                time.sleep(COOLDOWN)

            # --- SAVE ONLY ONCE AFTER THE LOOP FINISHES ---
            if new_batches:
                print("Combining batches and saving to disk...")
                # Join all batches side-by-side (axis=1)
                combined_new_data = pd.concat(new_batches, axis=1)
                
                # Append the new rows to the master dataframe (axis=0)
                chart_data = pd.concat([chart_data, combined_new_data], axis=0)
                
                # Clean up and sort
                chart_data = chart_data[~chart_data.index.duplicated(keep="last")].sort_index()
                
                # Save compressed file once
                tmp = CHART_DATA_FILE + ".tmp"
                chart_data.to_csv(tmp, compression='gzip')
                if os.path.exists(CHART_DATA_FILE): 
                    os.remove(CHART_DATA_FILE)
                os.rename(tmp, CHART_DATA_FILE)
                print("Chart cache updated and saved successfully.")

    # Final cleanup before returning.
    # ffill(limit=5): bridges weekends/holidays (up to 5 trading-day gaps).
    # NO bfill: a new ticker that only has 1 year of data must NOT have its
    # first real price back-propagated into all earlier NaN rows.
    chart_data = chart_data[[c for c in chart_data.columns if c in master]]
    chart_data = chart_data.ffill(limit=5)
    print(f"Chart dataset ready: {len(chart_data.columns)} tickers, {len(chart_data)} days.")
    return chart_data


# ==========================================
# BUILD VOLUME DATASET (UPDATED)
# ==========================================
def build_volume_dataset(master):
    """Returns a dict {ticker: avg_volume} using a VOL_AVG_DAYS rolling average."""
    vol_avg = {}
    all_vol = pd.DataFrame()

    # 1. Load existing cache if it exists
    if os.path.exists(VOLUME_DATA_FILE):
        try:
            all_vol = pd.read_csv(VOLUME_DATA_FILE, index_col=0, parse_dates=True)
            all_vol = all_vol.loc[:, ~all_vol.columns.duplicated()]
            
            # If the file is less than 12 hours old, just use it and skip updating
            file_time = os.path.getmtime(VOLUME_DATA_FILE)
            hours_old = (datetime.now() - datetime.fromtimestamp(file_time)).total_seconds() / 3600
            
            if hours_old < VOL_MCAP_COOLDOWN_HOURS:
                print(f"--- Volume cache fresh ({round(hours_old,1)}h / {VOL_MCAP_COOLDOWN_HOURS}h cooldown). Using cached volume. ---")
                for col in all_vol.columns:
                    series = all_vol[col].dropna()
                    if len(series) > 0:
                        vol_avg[col] = float(series.rolling(VOL_AVG_DAYS, min_periods=1).mean().iloc[-1])
                return vol_avg
        except Exception as e:
            print(f"Could not read volume cache: {e}")
            all_vol = pd.DataFrame()

    # 2. Find missing tickers that aren't in the cache at all
    existing = all_vol.columns.tolist() if not all_vol.empty else []
    missing  = [t for t in master if t not in existing]
    start    = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    # 3. Only download the missing tickers
    if missing:
        total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Downloading volume data for {len(missing)} missing tickers ({total_batches} batches)...")
        
        new_vol_data = []
        for i, idx in enumerate(range(0, len(missing), BATCH_SIZE)):
            batch = missing[idx: idx + BATCH_SIZE]
            print(f"  [{i+1}/{total_batches}] Downloading Volume: {batch[0]}...")
            batch_df = download_batch(batch, start, field="Volume")
            if not batch_df.empty:
                new_vol_data.append(batch_df)
            time.sleep(COOLDOWN)
            
        if new_vol_data:
            new_df = pd.concat(new_vol_data, axis=1)
            all_vol = pd.concat([all_vol, new_df], axis=1)

    # 4. Save and calculate averages
    if not all_vol.empty:
        all_vol = all_vol.loc[:, ~all_vol.columns.duplicated()]
        tmp = VOLUME_DATA_FILE + ".tmp"
        all_vol.to_csv(tmp, compression='gzip')
        if os.path.exists(VOLUME_DATA_FILE): 
            os.remove(VOLUME_DATA_FILE)
        os.rename(tmp, VOLUME_DATA_FILE)
        
        for col in all_vol.columns:
            if col in master:
                series = all_vol[col].dropna()
                if len(series) > 0:
                    vol_avg[col] = float(series.rolling(VOL_AVG_DAYS, min_periods=1).mean().iloc[-1])

    print(f"Volume data ready for {len(vol_avg)} tickers.")
    return vol_avg


def build_market_cap(master):
    """Returns a dict {ticker: market_cap} using yfinance .info, with JSON cache."""
    mcap = {}

    # Load cache
    if os.path.exists(MCAP_CACHE_FILE):
        try:
            with open(MCAP_CACHE_FILE, "r") as f:
                mcap = json.load(f)
            file_time = os.path.getmtime(MCAP_CACHE_FILE)
            hours_old = (datetime.now() - datetime.fromtimestamp(file_time)).total_seconds() / 3600
            if hours_old < VOL_MCAP_COOLDOWN_HOURS:
                print(f"--- Market-cap cache fresh ({round(hours_old,1)}h / {VOL_MCAP_COOLDOWN_HOURS}h cooldown). Using cached data. ---")
                return mcap
        except Exception:
            mcap = {}

    # Find tickers missing from cache
    missing = [t for t in master if t not in mcap]
    if missing:
        print(f"Fetching market cap for {len(missing)} tickers...")
        for i, t in enumerate(tqdm(missing, desc="Market Cap")):
            try:
                info = yf.Ticker(t).info
                mc = info.get("marketCap")
                if mc and mc > 0:
                    mcap[t] = mc
            except Exception:
                pass
            if (i + 1) % 100 == 0:
                time.sleep(1)

    # Save cache
    with open(MCAP_CACHE_FILE, "w") as f:
        json.dump(mcap, f)
    print(f"Market cap data ready for {len(mcap)} tickers.")
    return mcap


# ==========================================
# ADF TEST (lightweight, no statsmodels needed)
# ==========================================
def adf_pvalue(series, max_lag=None):
    """
    Lightweight Augmented Dickey-Fuller test.
    Returns approximate p-value (0-1). Lower = more stationary/cointegrated.
    Uses MacKinnon (1994) critical value interpolation for 'c' (constant) case.
    """
    try:
        y = np.asarray(series.dropna(), dtype=float)
        n = len(y)
        if n < 30:
            return 1.0
        if max_lag is None:
            max_lag = int(np.floor(12 * (n / 100) ** 0.25))
            max_lag = min(max_lag, n // 3)
        dy = np.diff(y)
        y_lag = y[:-1]
        # Build regression: dy_t = alpha + gamma * y_{t-1} + sum(beta_k * dy_{t-k}) + e
        nobs = len(dy) - max_lag
        if nobs < 10:
            return 1.0
        X_cols = [np.ones(nobs), y_lag[max_lag:]]
        for k in range(1, max_lag + 1):
            X_cols.append(dy[max_lag - k: -k if k < len(dy) else None][:nobs])
        X = np.column_stack(X_cols)
        Y = dy[max_lag:]
        try:
            coef, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            return 1.0
        gamma = coef[1]
        # Compute t-statistic for gamma
        Y_hat = X @ coef
        sse = np.sum((Y - Y_hat) ** 2)
        mse = sse / (nobs - X.shape[1])
        try:
            var_coef = mse * np.linalg.inv(X.T @ X)
            se_gamma = np.sqrt(var_coef[1, 1])
        except (np.linalg.LinAlgError, ValueError):
            return 1.0
        if se_gamma <= 0:
            return 1.0
        t_stat = gamma / se_gamma
        # MacKinnon critical values for ADF with constant (approximate)
        # p-value approximation using critical value table with sub-1% granularity
        # cv: 0.1%=-4.32, 1%=-3.43, 5%=-2.86, 10%=-2.57 (for n>250, 'c' case)
        if t_stat <= -4.32:
            return 0.001
        elif t_stat <= -3.43:
            return 0.001 + (t_stat - (-4.32)) / ((-3.43) - (-4.32)) * (0.01 - 0.001)
        elif t_stat <= -2.86:
            return 0.01 + (t_stat - (-3.43)) / ((-2.86) - (-3.43)) * (0.05 - 0.01)
        elif t_stat <= -2.57:
            return 0.05 + (t_stat - (-2.86)) / ((-2.57) - (-2.86)) * (0.10 - 0.05)
        elif t_stat <= -1.94:
            return 0.10 + (t_stat - (-2.57)) / ((-1.94) - (-2.57)) * (0.50 - 0.10)
        elif t_stat <= -0.7:
            return 0.50 + (t_stat - (-1.94)) / ((-0.7) - (-1.94)) * (0.90 - 0.50)
        else:
            return 0.99
    except Exception:
        return 1.0


# ==========================================
# ANALYZE PAIR
# ==========================================
def compute_half_life(spread_series):
    """
    Estimate mean-reversion half-life (in days) via OLS on the OU process:
        delta_y = lambda * y_lag + const
    half_life = -ln(2) / lambda
    Returns float('nan') if not mean-reverting or insufficient data.
    """
    try:
        spread = spread_series.dropna()
        if len(spread) < 20:
            return float('nan')
        delta  = spread.diff().dropna()
        lagged = spread.shift(1).dropna()
        # align to common index
        lagged, delta = lagged.align(delta, join='inner')
        if len(lagged) < 10:
            return float('nan')
        X = np.column_stack([lagged.values, np.ones(len(lagged))])
        coef, _, _, _ = np.linalg.lstsq(X, delta.values, rcond=None)
        lam = coef[0]
        if lam >= 0:          # not mean-reverting
            return float('nan')
        hl = -np.log(2) / lam
        if hl <= 0 or hl > 1000:  # sanity bounds
            return float('nan')
        return round(hl, 1)
    except Exception:
        return float('nan')


def analyze_pair(pair):
    a, b = pair

    cl = corr_long.loc[a, b]
    if cl < MIN_CORR_FILTER:
        return None

    cs = corr_short.loc[a, b]
    corr_brk = cl - cs

    # Z-score on last Z_LENGTH days for scoring (standard fixed window)
    ratio = log_prices[a] - log_prices[b]
    mean  = ratio.mean()
    std   = ratio.std()

    if std == 0:
        return None

    z  = (ratio.iloc[-1] - mean) / std
    rp = perf[a] - perf[b]
    spread_std = std  # store for EstRet calculation

    type_a = TICKER_TYPES.get(a, "Unknown")
    type_b = TICKER_TYPES.get(b, "Unknown")

    if type_a == "Pure ETF" and type_b == "Pure ETF":
        pair_category = "Pure ETF"
    elif type_a == "Pure Stock" and type_b == "Pure Stock":
        pair_category = "Pure Stock"
    else:
        pair_category = "Mixed"

    if any(np.isnan(v) for v in [z, cl, corr_brk, rp]):
        return None

    # ── Half-life of mean reversion (using full log-price spread in prices_raw) ──
    try:
        full_spread = np.log(prices_raw[a]) - np.log(prices_raw[b])
        hl = compute_half_life(full_spread)
    except Exception:
        hl = float('nan')

    if np.isnan(hl):
        return None

    # ── ADF cointegration test on the spread ──
    # Require minimum spread length for reliable ADF results
    if len(full_spread) < ADF_MIN_DAYS:
        return None
    # Use ADF_LOOKBACK_YRS of data (252 trading days/yr) if available
    adf_days = int(ADF_LOOKBACK_YRS * 252)
    try:
        adf_spread = full_spread.iloc[-adf_days:] if len(full_spread) > adf_days else full_spread
        adf_p = adf_pvalue(adf_spread)
    except Exception:
        adf_p = 1.0

    # Filter: reject pairs that don't meet cointegration confidence threshold
    max_p = 1.0 - ADF_CONFIDENCE   # e.g. 0.95 confidence → p must be ≤ 0.05
    if adf_p > max_p:
        return None

    # ── Adaptive Z-score window based on half-life ──
    adaptive_window = int(max(50, min(250, hl * 5)))
    lp_full = log_prices_full
    ratio_adapt = lp_full[a].iloc[-adaptive_window:] - lp_full[b].iloc[-adaptive_window:]
    std_adapt   = ratio_adapt.std()
    if std_adapt > 0:
        z_adaptive = round((ratio_adapt.iloc[-1] - ratio_adapt.mean()) / std_adapt, 2)
    else:
        z_adaptive = round(z, 2)

    # ── Annualized returns ──
    try:
        n_days = len(prices_raw)
        ann_a  = round(((prices_raw[a].iloc[-1] / prices_raw[a].iloc[0]) ** (252 / n_days) - 1) * 100, 1)
        ann_b  = round(((prices_raw[b].iloc[-1] / prices_raw[b].iloc[0]) ** (252 / n_days) - 1) * 100, 1)
    except Exception:
        ann_a = ann_b = float('nan')

    # ── Multi-timeframe Z-scores (30d short, 250d long) ──
    ratio_s = log_prices_short[a] - log_prices_short[b]
    std_s   = ratio_s.std()
    z30     = round((ratio_s.iloc[-1] - ratio_s.mean()) / std_s, 2) if std_s > 0 else 0.0

    ratio_l = log_prices_long[a] - log_prices_long[b]
    std_l   = ratio_l.std()
    z250    = round((ratio_l.iloc[-1] - ratio_l.mean()) / std_l, 2) if std_l > 0 else 0.0

    # ── Timeframe alignment (continuous 0-1 score) ──
    # Measures how tightly the 3 Z-scores agree in direction AND magnitude.
    # Z-scores near zero (< 0.3σ) are treated as neutral — they don't vote on direction.
    zs = [z30, z, z250]
    abs_zs = [abs(v) for v in zs]
    NEUTRAL_ZONE = 0.3
    directional = [(v > NEUTRAL_ZONE, v < -NEUTRAL_ZONE) for v in zs]  # (is_pos, is_neg)
    n_pos = sum(1 for p, _ in directional if p)
    n_neg = sum(1 for _, n in directional if n)
    n_neutral = sum(1 for p, n in directional if not p and not n)
    same_dir = (n_pos > 0 and n_neg == 0) or (n_neg > 0 and n_pos == 0)

    if same_dir:
        # All directional Z-scores point the same way
        z_spread = max(abs_zs) - min(abs_zs)
        closeness = max(0.0, 1.0 - z_spread / 2.0)
        weakest_strength = min(abs_zs) / max(Z_THRESHOLD, 1.0)
        weakest_factor = min(weakest_strength, 1.0)
        # Neutral Z-scores reduce alignment proportionally (3/3 strong > 2/3 + 1 neutral)
        dir_ratio = (3 - n_neutral) / 3.0
        align_score = 0.5 + 0.5 * closeness * weakest_factor * dir_ratio
    elif n_pos > 0 and n_neg > 0:
        # Genuine conflict — some positive, some negative
        align_score = 0.15
    else:
        # All neutral (very rare — all Z near zero)
        align_score = 0.25

    # Bucket labels for display
    if align_score >= 0.75:
        alignment = "Aligned"
    elif align_score >= 0.40:
        alignment = "Mixed"
    else:
        alignment = "Conflicting"

    # ── Confidence level (continuous, based on magnitude agreement) ──
    # Considers: avg magnitude relative to threshold + how tightly grouped + direction agreement
    avg_z = sum(abs_zs) / 3.0
    strength = min(avg_z / max(Z_THRESHOLD, 1.0), 2.0) / 2.0  # 0-1: how far above threshold on avg
    z_spread = max(abs_zs) - min(abs_zs)
    consistency = max(0.0, 1.0 - z_spread / 3.0)  # 0-1: how tightly grouped
    dir_agreement = 1.0 if same_dir else (0.5 if n_neutral == 3 else 0.0)
    conf_score = strength * 0.6 + consistency * 0.2 + dir_agreement * 0.2

    if conf_score >= 0.70:
        confidence = "High"
    elif conf_score >= 0.40:
        confidence = "Med"
    else:
        confidence = "Low"

    # ── NEW 5-FACTOR SCORING ──
    # 1) Z-score magnitude: higher |Z| = stronger signal
    z_norm = min(abs(z) / 3.0, 1.0)
    # 2) Half-life speed: faster reversion = better trade (log decay: 1d→1.0, 15d→0.49, 200d→0.0)
    hl_norm = max(0.0, 1.0 - np.log(max(hl, 1.0)) / np.log(200.0))
    # 3) Stationarity: lower ADF p-value = more cointegrated spread
    #    Scaled to the post-filter range: p=0.001 → 1.0, p=max_p → 0.0
    stat_norm = max(0.0, min(1.0, (max_p - adf_p) / max(max_p, 0.01)))
    # 4) Timeframe confirmation: blend alignment (direction+closeness) with confidence (magnitude+consistency)
    confirm_norm = align_score * 0.5 + conf_score * 0.5
    # 5) Base correlation level
    corr_norm = min(cl / 1.0, 1.0)

    # ── Estimated pairs trade return (gross spread return if fully reverts) ──
    est_ret = round(abs(z) * spread_std * 100, 2)   # in %
    if MIN_EST_RETURN > 0 and est_ret < MIN_EST_RETURN:
        return None
    if not np.isnan(hl) and hl > 0:
        # Full reversion cycle ≈ 3x half-life; assume 70% capture rate
        cycle_days = hl * 3.0
        trades_per_year = 252.0 / cycle_days
        ann_ret = round(est_ret * trades_per_year * 0.70, 1)
    else:
        ann_ret = None

    # 6) Annualized return potential: 0%/yr → 0.0, 100%+/yr → 1.0
    ann_ret_norm = min((ann_ret or 0.0) / 100.0, 1.0) if ann_ret is not None and ann_ret > 0 else 0.0

    score = (W_ZSCORE * z_norm + W_HALFLIFE * hl_norm + W_STATIONARY * stat_norm
             + W_ANNRET * ann_ret_norm + W_CONFIRM * confirm_norm + W_CORR * corr_norm)

    return {
        "Pair":       f"{a}/{b}",
        "Category":   pair_category,
        "Z":          round(z, 2),
        "Corr":       round(cl, 2),
        "CorrBrk":    round(corr_brk, 3),
        "PerfDiff":   round(rp, 2),
        "Score":      round(score, 3),
        "HalfLife":   hl if not np.isnan(hl) else None,
        "ADF_p":      round(adf_p, 4),
        "AnnRetA":    ann_a if not np.isnan(ann_a) else None,
        "AnnRetB":    ann_b if not np.isnan(ann_b) else None,
        "EstRet":     est_ret,
        "AnnRet":     ann_ret,
        "SpreadStd":  round(float(std), 6),
        "Z30":        z30,
        "Z250":       z250,
        "Alignment":  alignment,
        "Confidence": confidence,
        "AdaptiveWindow": adaptive_window,
        "ZAdaptive":      z_adaptive,
    }


# ==========================================
# ROLLING Z-SCORE HISTORY FOR CHART
# ==========================================
def compute_z_history(a, b, price_data, window_override=None):
    """Rolling Z-score over all available data.
    Uses an adaptive window: Z_LENGTH when sufficient history exists,
    falling back to half the available data (min 20 days) for shorter-
    history tickers such as leveraged ETFs.
    If window_override is given, uses that window length instead.
    """
    log_a = np.log(price_data[a].dropna())
    log_b = np.log(price_data[b].dropna())
    combined = pd.DataFrame({"a": log_a, "b": log_b}).dropna()
    spread   = combined["a"] - combined["b"]

    n_pts  = len(spread)
    if window_override:
        window = min(window_override, max(20, n_pts // 2))
    else:
        window = Z_LENGTH if n_pts >= Z_LENGTH * 2 else max(20, n_pts // 2)
    if n_pts < window + 5:
        return [], []                # truly insufficient data

    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std()
    z_series  = (spread - roll_mean) / roll_std
    z_series  = z_series.dropna()

    dates  = [d.strftime("%Y-%m-%d") for d in z_series.index]
    values = [round(float(v), 4) if not np.isnan(v) else None for v in z_series.values]
    return dates, values


# ==========================================
# NORMALIZED PRICE HISTORY FOR COMPARISON CHART
# ==========================================
def compute_price_history(a, b, price_data):
    """Returns aligned dates + normalized price series (rebased to 100) for both symbols."""
    p_a = price_data[a].dropna()
    p_b = price_data[b].dropna()
    combined = pd.DataFrame({"a": p_a, "b": p_b}).dropna()
    if combined.empty:
        return [], [], []
    base_a = combined["a"].iloc[0]
    base_b = combined["b"].iloc[0]
    norm_a = [round(float(v / base_a * 100), 4) for v in combined["a"]]
    norm_b = [round(float(v / base_b * 100), 4) for v in combined["b"]]
    dates  = [d.strftime("%Y-%m-%d") for d in combined.index]
    return dates, norm_a, norm_b


# ==========================================
# BUILD SYMBOLS PAGE
# ==========================================
def build_symbols_page(valid_tickers):

    # ── CSV column layouts ────────────────────────────────────────────────────
    # ETFs.csv:   Ticker, Name, Sector, Industry, Type
    # STOCKS.csv: Ticker, Name, Sector, Industry, Subindustry
    def read_csv_meta(path, is_etf=False):
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, header=None)
        n_cols = min(df.shape[1], 5)
        df = df.iloc[:, :n_cols]

        # ETFs.csv:   Ticker, Name, Sector, Industry, Type
        # STOCKS.csv: Ticker, Name, Sector, Industry, Subindustry
        if is_etf:
            col_names = ["Ticker", "Name", "Sector", "Industry", "Type"][:n_cols]
        else:
            col_names = ["Ticker", "Name", "Sector", "Industry", "Subindustry"][:n_cols]
        df.columns = col_names

        # Ensure all hierarchy columns exist
        for col in ["Sector", "Industry", "Subindustry"]:
            if col not in df.columns:
                df[col] = ""

        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

        # Drop rows with non-ticker values (e.g. numeric garbage, header rows)
        df = df[df["Ticker"].str.match(r'^[A-Z]{1,6}$', na=False)]
        df = df[df["Ticker"].isin(valid_tickers)]
        df = df.fillna("")

        # Sanitize: drop obviously numeric sector/industry values
        for col in ["Sector", "Industry", "Subindustry"]:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].where(~df[col].str.match(r'^[\d\.\-]+$'), "")

        # Fill remaining blanks
        df["Sector"]      = df["Sector"].replace("", "Other")
        df["Industry"]    = df["Industry"].replace("", "Other")
        df["Subindustry"] = df["Subindustry"].replace("", "Other")
        return df

    df_etf   = read_csv_meta("ETFs.csv",   is_etf=True)
    df_stock = read_csv_meta("STOCKS.csv", is_etf=False)

    def build_section_html(df, accent_color):
        if df.empty:
            return "<p style='color:#64748b;'>No data found.</p>"
        html = ""
        for sector, sg in df.groupby("Sector"):
            html += f'<div class="sector-block"><div class="sector-header" style="color:{accent_color};">{sector}</div>'
            for industry, ig in sg.groupby("Industry"):
                html += f'<div class="industry-block"><div class="industry-label">{industry}</div>'
                for subindustry, sub_ig in ig.groupby("Subindustry"):
                    if subindustry != "Other":
                        html += f'<div class="subindustry-block"><div class="subindustry-label">&#8627; {subindustry}</div>'
                    html += '<div class="ticker-grid">'
                    for _, row in sub_ig.sort_values("Ticker").iterrows():
                        html += (
                            f'<div class="ticker-card">'
                            f'<span class="ticker-sym" style="color:{accent_color};">{row["Ticker"]}</span>'
                            f'<span class="ticker-name">{row.get("Name","")}</span>'
                            f'</div>'
                        )
                    html += "</div>"
                    if subindustry != "Other":
                        html += "</div>"
                html += "</div>"
            html += "</div>"
        return html

    etf_html   = build_section_html(df_etf,   "#38bdf8")
    stock_html = build_section_html(df_stock, "#f59e0b")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Symbol Reference | Pairs Scanner</title>

<link rel="icon" type="image/x-icon" href="favicon.ico?v=1">
<link rel="icon" type="image/png" sizes="32x32" href="favicon-32x32.png?v=1">
<link rel="icon" type="image/png" sizes="16x16" href="favicon-16x16.png?v=1">
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png?v=1">
<link rel="manifest" href="site.webmanifest">

<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #08090d;
    --surface: #0e1117;
    --surface2:#151821;
    --border:  #1e2535;
    --text:    #cbd5e1;
    --muted:   #64748b;
    --cyan:    #38bdf8;
    --amber:   #f59e0b;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Syne', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; font-size: 14px; }}

  /* TOPBAR */
  .topbar {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 14px 32px; display: flex; align-items: center;
    justify-content: space-between; position: sticky; top: 0; z-index: 100;
  }}
  .topbar h1 {{ font-size: 18px; font-weight: 800; letter-spacing: 0.04em; color: white; }}
  .topbar a {{
    color: var(--cyan); text-decoration: none; font-size: 12px; font-weight: 600;
    border: 1px solid rgba(56,189,248,0.3); padding: 6px 12px; border-radius: 4px;
    transition: all 0.15s; letter-spacing: 0.05em;
  }}
  .topbar a:hover {{ background: rgba(56,189,248,0.08); border-color: var(--cyan); }}

  /* STATS BAR */
  .stats-bar {{
    background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 8px 32px; font-family: var(--mono); font-size: 11px;
    color: var(--muted); display: flex; gap: 30px; flex-wrap: wrap;
  }}
  .stats-bar span {{ color: var(--text); font-weight: 600; }}

  /* SEARCH */
  .search-bar {{
    padding: 14px 32px; background: var(--surface); border-bottom: 1px solid var(--border);
  }}
  .search-bar input {{
    background: var(--surface2); border: 1px solid var(--border); color: white;
    padding: 8px 14px; border-radius: 5px; font-family: var(--mono);
    font-size: 12px; width: 360px; outline: none; transition: border 0.2s;
  }}
  .search-bar input:focus {{ border-color: var(--cyan); }}
  .search-bar input::placeholder {{ color: var(--muted); }}

  /* COLUMNS */
  .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
  .column {{ padding: 20px 32px; border-right: 1px solid var(--border); }}
  .column:last-child {{ border-right: none; }}

  .col-header {{
    font-size: 11px; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px;
  }}
  .col-header .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

  /* SECTOR */
  .sector-block {{ margin-bottom: 22px; }}
  .sector-header {{
    font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 10px; padding: 7px 12px; border-radius: 5px; border-left: 3px solid;
  }}
  .sector-header[style*="38bdf8"] {{ border-left-color: #38bdf8; background: rgba(56,189,248,0.06); }}
  .sector-header[style*="f59e0b"] {{ border-left-color: #f59e0b; background: rgba(245,158,11,0.06); }}

  /* INDUSTRY */
  .industry-block {{ margin: 10px 0 10px 14px; }}
  .industry-label {{
    font-size: 11px; font-weight: 700; color: #6b7f9a; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 6px; padding-bottom: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}

  /* SUBINDUSTRY */
  .subindustry-block {{ margin: 8px 0 8px 12px; }}
  .subindustry-label {{
    font-size: 10px; color: #4a5e72; font-family: var(--mono);
    margin-bottom: 6px; letter-spacing: 0.04em;
  }}

  /* TICKER CARDS */
  .ticker-grid {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
  .ticker-card {{
    background: var(--surface2); border: 1px solid var(--border); border-radius: 5px;
    padding: 6px 10px; display: flex; flex-direction: column; min-width: 140px; max-width: 210px;
    transition: border-color 0.15s, background 0.15s; cursor: default;
  }}
  .ticker-card:hover {{ border-color: #334155; background: #1e2535; }}
  .ticker-sym  {{ font-family: var(--mono); font-size: 12px; font-weight: 700; line-height: 1.2; }}
  .ticker-name {{
    font-size: 10px; color: #94a3b8; line-height: 1.3; margin-top: 2px;
    white-space: normal; overflow: visible;
  }}

  /* FLAT ALPHA LAYOUT (used when no sector data) */
  .flat-grid {{ padding: 0; }}
  .alpha-block {{ margin-bottom: 18px; }}
  .alpha-label {{
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    color: #3a4f66; letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 6px; padding-bottom: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }}

  @media (max-width: 900px) {{
    .columns {{ grid-template-columns: 1fr; }}
    .column {{ border-right: none; border-bottom: 1px solid var(--border); }}
  }}
</style>
</head>
<body>

<div class="topbar">
  <h1>Symbol Reference</h1>
  <a href="pairs_scanner.html">&#8592; Back to Dashboard</a>
</div>

<div class="stats-bar">
  <div>Total Active: <span>{len(valid_tickers)}</span></div>
  <div>ETFs: <span>{len(df_etf)}</span></div>
  <div>Stocks: <span>{len(df_stock)}</span></div>
  <div>Generated: <span id="gen-time"></span></div>
</div>

<div class="search-bar">
  <input type="text" id="searchBox" placeholder="Search ticker or name..." oninput="filterSymbols()" />
</div>

<div class="columns">
  <div class="column">
    <div class="col-header" style="color:#38bdf8;">
      <span class="dot" style="background:#38bdf8;"></span>Exchange Traded Funds
    </div>
    <div id="etf-section">{etf_html}</div>
  </div>
  <div class="column">
    <div class="col-header" style="color:#f59e0b;">
      <span class="dot" style="background:#f59e0b;"></span>Stocks
    </div>
    <div id="stock-section">{stock_html}</div>
  </div>
</div>

<script>
document.getElementById("gen-time").textContent = new Date({int(time.time() * 1000)}).toLocaleString();
function filterSymbols() {{
  const q = document.getElementById("searchBox").value.toUpperCase().trim();
  document.querySelectorAll(".ticker-card").forEach(card => {{
    const sym  = card.querySelector(".ticker-sym")?.textContent || "";
    const name = card.querySelector(".ticker-name")?.textContent || "";
    card.style.display = (sym.includes(q) || name.toUpperCase().includes(q)) ? "" : "none";
  }});
  ["subindustry-block","industry-block","sector-block"].forEach(cls => {{
    document.querySelectorAll("." + cls).forEach(block => {{
      block.style.display = [...block.querySelectorAll(".ticker-card")].some(c => c.style.display !== "none") ? "" : "none";
    }});
  }});
}}
</script>
</body>
</html>"""

    with open("symbols.html", "w", encoding="utf-8") as f:
        f.write(page)
    print("symbols.html created.")


# ==========================================
# MULTIPROCESSING WORKER INITIALIZERS
# ==========================================
def _init_analyze_worker(cl, cs, lp, lp_short, lp_long, lp_full, pr, pf, tt, elt):
    """Set shared read-only data as globals in each worker process."""
    global corr_long, corr_short, log_prices, log_prices_short, log_prices_long
    global log_prices_full, prices_raw, perf, TICKER_TYPES, ETF_LEV_TYPES
    corr_long        = cl
    corr_short       = cs
    log_prices       = lp
    log_prices_short = lp_short
    log_prices_long  = lp_long
    log_prices_full  = lp_full
    prices_raw       = pr
    perf             = pf
    TICKER_TYPES     = tt
    ETF_LEV_TYPES    = elt


def _init_chart_worker(cd, sd):
    """Set shared read-only data as globals in each chart worker process."""
    global _w_chart_data, _w_scoring_data
    _w_chart_data    = cd
    _w_scoring_data  = sd


def _compute_chart_for_pair(r):
    """Worker: compute Z-score/price chart history for one pair."""
    a, b = r["Pair"].split("/")
    try:
        src = _w_chart_data if (not _w_chart_data.empty and a in _w_chart_data.columns and b in _w_chart_data.columns) else _w_scoring_data
        dates, z_vals = compute_z_history(a, b, src)
        # Snap the last chart point to match the authoritative stats Z value
        stats_z = r.get("Z")
        if z_vals and stats_z is not None:
            z_vals[-1] = round(float(stats_z), 4)
        r["ZDates"]   = dates
        r["ZHistory"] = z_vals
        # Adaptive Z-score history using half-life-based window
        adapt_win = r.get("AdaptiveWindow")
        if adapt_win and adapt_win != Z_LENGTH:
            adapt_dates, adapt_vals = compute_z_history(a, b, src, window_override=adapt_win)
            # Snap adaptive chart endpoint to match stats adaptive Z
            stats_za = r.get("ZAdaptive")
            if adapt_vals and stats_za is not None:
                adapt_vals[-1] = round(float(stats_za), 4)
            r["ZDatesAdaptive"]   = adapt_dates
            r["ZHistoryAdaptive"] = adapt_vals
        else:
            r["ZDatesAdaptive"]   = dates
            r["ZHistoryAdaptive"] = z_vals
        pdates, pa, pb = compute_price_history(a, b, src)
        r["PriceDates"] = pdates
        r["PriceA"]     = pa
        r["PriceB"]     = pb
    except Exception:
        r["ZDates"]     = []
        r["ZHistory"]   = []
        r["ZDatesAdaptive"]   = []
        r["ZHistoryAdaptive"] = []
        r["PriceDates"] = []
        r["PriceA"]     = []
        r["PriceB"]     = []

    return r


# ==========================================
# TRADE TRACKER
# ==========================================
def update_active_trades(data, chart_data=None):
    """Read active_trades.json, update current Z-scores/prices and chart history from latest data."""
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
    except Exception:
        return []

    updated = []
    for t in trades:
        if t.get("status") != "open":
            updated.append(t)
            continue
        pair = t.get("pair", "")
        if "/" not in pair:
            updated.append(t)
            continue
        a, b = pair.split("/")
        if a not in data.columns or b not in data.columns:
            updated.append(t)
            continue

        # Update current prices
        t["currentPriceA"] = round(float(data[a].iloc[-1]), 2)
        t["currentPriceB"] = round(float(data[b].iloc[-1]), 2)

        # Recalculate current Z-score (100d window)
        log_r = np.log(data[[a, b]].dropna().tail(Z_LENGTH))
        if len(log_r) >= 20:
            ratio = log_r[a] - log_r[b]
            std = ratio.std()
            if std > 0:
                t["currentZ"] = round(float((ratio.iloc[-1] - ratio.mean()) / std), 2)

        # Update days held
        try:
            entry = datetime.strptime(t["entryDate"], "%Y-%m-%d")
            t["daysHeld"] = (datetime.now() - entry).days
        except Exception:
            pass

        # Estimate P&L (simple: price change on each leg relative to entry)
        try:
            dir = t.get("direction", "")
            pa_entry, pb_entry = t["entryPriceA"], t["entryPriceB"]
            pa_now, pb_now = t["currentPriceA"], t["currentPriceB"]
            if dir == "short_a_long_b":
                pnl_a = (pa_entry - pa_now) / pa_entry * 100  # short A gains when price drops
                pnl_b = (pb_now - pb_entry) / pb_entry * 100  # long B gains when price rises
            else:
                pnl_a = (pa_now - pa_entry) / pa_entry * 100  # long A
                pnl_b = (pb_entry - pb_now) / pb_entry * 100  # short B
            t["pnlPct"] = round((pnl_a + pnl_b) / 2, 2)
        except Exception:
            t["pnlPct"] = 0.0

        # Compute Z-score chart history for this trade
        try:
            src = chart_data if (chart_data is not None and not chart_data.empty and a in chart_data.columns and b in chart_data.columns) else data
            z_dates, z_vals = compute_z_history(a, b, src)
            # Snap last chart point to match authoritative currentZ
            if z_vals and t.get("currentZ") is not None:
                z_vals[-1] = round(float(t["currentZ"]), 4)
            t["chartDates"] = z_dates
            t["chartZ"] = z_vals
        except Exception:
            t["chartDates"] = []
            t["chartZ"] = []

        updated.append(t)

    # Save updated trades back
    with open(TRADES_FILE, "w") as f:
        json.dump(updated, f, indent=2)

    return updated


def generate_trades_page(trades):
    """Generate active_trades.html with current trade status."""
    open_trades = [t for t in trades if t.get("status") == "open"]
    closed_trades = [t for t in trades if t.get("status") == "closed"]

    def trade_card(t):
        pair = t.get("pair", "?")
        a, b = pair.split("/") if "/" in pair else (pair, "?")
        direction = t.get("direction", "")
        if direction == "short_a_long_b":
            dir_label = f"Short {a} / Long {b}"
            dir_class = "dir-short"
        elif direction == "long_a_short_b":
            dir_label = f"Long {a} / Short {b}"
            dir_class = "dir-long"
        else:
            dir_label = "Neutral"
            dir_class = "dir-neutral"

        entry_z = t.get("entryZ", 0)
        cur_z   = t.get("currentZ", 0)
        days    = t.get("daysHeld", 0)
        pnl     = t.get("pnlPct", 0)
        pnl_class = "pnl-pos" if pnl >= 0 else "pnl-neg"

        # Z progress toward 0 (target) — negative means Z moved further away
        if abs(entry_z) > 0:
            progress = min(100, (1 - abs(cur_z) / abs(entry_z)) * 100)
        else:
            progress = 0

        entry_pa = t.get("entryPriceA", 0)
        entry_pb = t.get("entryPriceB", 0)
        cur_pa   = t.get("currentPriceA", 0)
        cur_pb   = t.get("currentPriceB", 0)
        shares_a = t.get("sharesA", 0)
        shares_b = t.get("sharesB", 0)

        # Dollar P&L calculation
        dollar_pnl = 0
        if shares_a > 0 and shares_b > 0:
            if direction == "short_a_long_b":
                dollar_pnl = (entry_pa - cur_pa) * shares_a + (cur_pb - entry_pb) * shares_b
            else:
                dollar_pnl = (cur_pa - entry_pa) * shares_a + (entry_pb - cur_pb) * shares_b
        dollar_class = "pnl-pos" if dollar_pnl >= 0 else "pnl-neg"
        # P&L % per leg — flip sign for short leg so positive = profitable
        raw_chg_a = (cur_pa - entry_pa) / entry_pa * 100 if entry_pa > 0 else 0
        raw_chg_b = (cur_pb - entry_pb) / entry_pb * 100 if entry_pb > 0 else 0
        chg_a = -raw_chg_a if direction == "short_a_long_b" else raw_chg_a
        chg_b = raw_chg_b if direction == "short_a_long_b" else -raw_chg_b
        chg_a_class = "pnl-pos" if chg_a >= 0 else "pnl-neg"
        chg_b_class = "pnl-pos" if chg_b >= 0 else "pnl-neg"

        if progress > 0:
            pbar_style = f"width:{progress:.0f}%;background:var(--green);"
        elif progress < 0:
            pbar_style = f"width:{min(abs(progress), 100):.0f}%;background:var(--red);"
        else:
            pbar_style = "background:var(--muted);width:2px;"

        # Chart payload
        chart_dates = t.get("chartDates", [])
        chart_z = t.get("chartZ", [])
        has_chart = len(chart_dates) > 0
        chart_payload = json.dumps({"pair": pair, "dates": chart_dates, "z": chart_z, "currentZ": cur_z, "entryZ": entry_z, "zWindow": Z_LENGTH})
        chart_payload_esc = chart_payload.replace("&", "&amp;").replace("'", "&#39;")
        chart_btn = f"""<button class="tc-chart" onclick="openTradeChart(this)" data-chart='{chart_payload_esc}'>&#9657; Z-Chart</button>""" if has_chart else ""
        tid = t.get('id', '')
        if t.get('status') == 'open':
            action_btns = f'<button class="tc-edit" onclick="openEditModal(\'{tid}\')">&#9998; Edit</button><button class="tc-close" onclick="closeTrade(\'{tid}\')">&#10005; Close</button>'
        else:
            action_btns = f'<button class="tc-reopen" onclick="reopenTrade(\'{tid}\')">&#8634; Reopen</button><button class="tc-delete" onclick="deleteTrade(\'{tid}\')">&#10005;</button>'

        return f"""
        <div class="trade-card">
          <div class="tc-header">
            <span class="tc-pair">{pair}</span>
            <span class="tc-dir {dir_class}">{dir_label}</span>
            <span class="tc-days">{days}d held</span>
            {chart_btn}
            {action_btns}
          </div>
          <div class="tc-body">
            <div class="tc-stat">
              <div class="tc-label">Entry Z</div>
              <div class="tc-val">{entry_z:+.2f}&sigma;</div>
            </div>
            <div class="tc-stat">
              <div class="tc-label">Current Z</div>
              <div class="tc-val" style="color:{'#34d399' if abs(cur_z) < abs(entry_z) else '#ef4444'}">{cur_z:+.2f}&sigma;</div>
            </div>
            <div class="tc-stat">
              <div class="tc-label">Est P&L</div>
              <div class="tc-val {pnl_class}">{pnl:+.1f}%</div>
            </div>
            <div class="tc-stat tc-prices">
              <div class="tc-label">{a} <span style="color:var(--cyan);font-size:9px;">{'-' if direction == 'short_a_long_b' else '+'}{shares_a} shares</span></div>
              <div class="tc-val">${entry_pa:.2f} &rarr; ${cur_pa:.2f} <span class="{chg_a_class}">{chg_a:+.1f}%</span></div>
            </div>
            <div class="tc-stat tc-prices">
              <div class="tc-label">{b} <span style="color:var(--cyan);font-size:9px;">{'+' if direction == 'short_a_long_b' else '-'}{shares_b} shares</span></div>
              <div class="tc-val">${entry_pb:.2f} &rarr; ${cur_pb:.2f} <span class="{chg_b_class}">{chg_b:+.1f}%</span></div>
            </div>
            <div class="tc-stat">
              <div class="tc-label">$ P&L</div>
              <div class="tc-val {dollar_class}">{'+' if dollar_pnl >= 0 else ''}${abs(dollar_pnl):.0f}</div>
            </div>
          </div>
          <div class="tc-progress">
            <div class="tc-pbar-track">
              <div class="tc-pbar-fill" style="{pbar_style}"></div>
            </div>
            <span class="tc-pbar-label" style="color:{'var(--red)' if progress < 0 else 'var(--green)' if progress > 0 else 'var(--muted)'}">{progress:+.0f}% to Z=0</span>
          </div>
        </div>"""

    open_cards = "\n".join(trade_card(t) for t in open_trades) if open_trades else '<div class="no-trades">No open trades. Track a pair from the scanner to get started.</div>'
    closed_cards = "\n".join(trade_card(t) for t in closed_trades) if closed_trades else '<div class="no-trades">No closed trades yet.</div>'

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Active Trades</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #060a10; --surface: #0a0e17; --border: #1a2233;
    --cyan: #38bdf8; --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
    --muted: #64748b; --mono: 'JetBrains Mono', monospace; --sans: 'Syne', sans-serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: #e2e8f0; font-family: var(--mono); }}

  .topbar {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 10px 28px; display: flex; align-items: center; justify-content: space-between;
  }}
  .topbar .brand {{ font-family: var(--sans); font-size: 20px; font-weight: 800; color: white; letter-spacing: 0.08em; }}
  .topbar .brand span {{ color: var(--cyan); }}
  .nav-links {{ display: flex; gap: 10px; }}
  .nav-links a {{
    font-size: 12px; font-weight: 600; color: var(--cyan); text-decoration: none;
    letter-spacing: 0.05em; padding: 6px 12px; border: 1px solid rgba(56,189,248,0.3);
    border-radius: 4px; transition: all 0.15s; white-space: nowrap;
  }}
  .nav-links a:hover {{ background: rgba(56,189,248,0.08); border-color: var(--cyan); }}

  .content {{ max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
  h2 {{ font-family: var(--sans); font-size: 18px; font-weight: 700; color: white; margin: 20px 0 12px; letter-spacing: 0.05em; }}
  .section-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }}

  .trade-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 14px; }}

  .trade-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 18px; transition: border-color 0.2s;
  }}
  .trade-card:hover {{ border-color: var(--cyan); }}

  .tc-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: nowrap; }}
  .tc-pair {{ font-size: 16px; font-weight: 700; color: white; }}
  .tc-dir {{ font-size: 10px; padding: 2px 8px; border-radius: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
  .dir-short {{ background: rgba(239,68,68,0.12); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }}
  .dir-long  {{ background: rgba(34,197,94,0.12); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }}
  .dir-neutral {{ background: rgba(100,116,139,0.15); color: var(--muted); border: 1px solid var(--border); }}
  .tc-days {{ font-size: 11px; color: var(--muted); margin-left: auto; white-space: nowrap; }}
  .tc-header button {{ flex-shrink: 0; }}
  .tc-close {{
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); color: var(--red);
    font-size: 10px; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-family: var(--mono);
  }}
  .tc-close:hover {{ background: rgba(239,68,68,0.25); }}
  .tc-reopen {{
    background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24;
    font-size: 10px; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-family: var(--mono);
  }}
  .tc-reopen:hover {{ background: rgba(251,191,36,0.25); }}
  .tc-delete {{
    background: none; border: none; color: #ef4444; opacity: 0.4;
    font-size: 13px; padding: 2px 4px; cursor: pointer; line-height: 1;
  }}
  .tc-delete:hover {{ opacity: 1; }}
  .tc-edit {{
    background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.3); color: var(--cyan);
    font-size: 10px; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-family: var(--mono);
  }}
  .tc-edit:hover {{ background: rgba(56,189,248,0.25); }}
  .tc-chart {{
    background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.3); color: #a855f7;
    font-size: 10px; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-family: var(--mono);
  }}
  .tc-chart:hover {{ background: rgba(168,85,247,0.25); }}

  /* CHART MODAL */
  .chart-overlay {{
    display: none; position: fixed; inset: 0; z-index: 1100;
    background: rgba(0,0,0,0.8); backdrop-filter: blur(8px);
    align-items: center; justify-content: center;
  }}
  .chart-overlay.open {{ display: flex; }}
  .chart-modal {{
    background: #0a0e17; border: 1px solid var(--border); border-radius: 14px;
    padding: 22px 28px; width: min(1500px, 99vw); max-height: 95vh;
    box-shadow: 0 40px 100px rgba(0,0,0,0.7), 0 0 0 1px rgba(56,189,248,0.06);
  }}
  .chart-modal-header {{
    display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;
  }}
  .chart-modal-title {{ font-family: var(--sans); font-size: 15px; font-weight: 700; color: white; }}
  .chart-modal-title .cm-pair {{ color: var(--cyan); }}
  .chart-modal-stats {{
    display: flex; gap: 16px; font-size: 11px; color: var(--muted);
  }}
  .chart-modal-stats .cm-stat {{ display: flex; flex-direction: column; align-items: center; }}
  .chart-modal-stats .cm-val {{ font-weight: 600; font-size: 13px; }}
  .chart-modal-close {{
    background: none; border: 1px solid var(--border); color: var(--muted);
    font-size: 16px; padding: 4px 10px; border-radius: 6px; cursor: pointer;
  }}
  .chart-modal-close:hover {{ color: white; border-color: var(--muted); }}
  .chart-canvas-wrap {{ position: relative; height: 500px; }}

  /* EDIT MODAL */
  .edit-overlay {{
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(6px);
    align-items: center; justify-content: center;
  }}
  .edit-overlay.open {{ display: flex; }}
  .edit-modal {{
    background: #0a0e17; border: 1px solid var(--border); border-radius: 12px;
    padding: 24px; width: min(420px, 95vw);
  }}
  .edit-modal h3 {{ font-family: var(--sans); font-size: 16px; color: white; margin-bottom: 16px; }}
  .edit-field {{ margin-bottom: 12px; }}
  .edit-field label {{ display: block; font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
  .edit-field input {{
    width: 100%; background: #0d1520; border: 1px solid var(--border); color: #e2e8f0;
    font-family: var(--mono); font-size: 13px; padding: 8px 10px; border-radius: 6px;
  }}
  .edit-field input:focus {{ border-color: var(--cyan); outline: none; }}
  .edit-actions {{ display: flex; gap: 10px; margin-top: 18px; }}
  .edit-actions button {{
    flex: 1; padding: 8px; border-radius: 6px; font-family: var(--mono); font-size: 12px;
    font-weight: 600; cursor: pointer; border: 1px solid var(--border);
  }}
  .edit-save {{ background: rgba(34,197,94,0.15); color: var(--green); border-color: rgba(34,197,94,0.3) !important; }}
  .edit-save:hover {{ background: rgba(34,197,94,0.3); }}
  .edit-cancel {{ background: transparent; color: var(--muted); }}
  .edit-cancel:hover {{ background: rgba(100,116,139,0.15); }}

  .tc-body {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 10px; }}
  .tc-stat {{ }}
  .tc-label {{ font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
  .tc-val {{ font-size: 14px; font-weight: 600; color: #e2e8f0; }}
  .tc-prices .tc-val {{ font-size: 11px; }}
  .pnl-pos {{ color: var(--green) !important; }}
  .pnl-neg {{ color: var(--red) !important; }}

  .tc-progress {{ display: flex; align-items: center; gap: 10px; }}
  .tc-pbar-track {{ flex: 1; height: 4px; background: #1a2233; border-radius: 2px; overflow: hidden; }}
  .tc-pbar-fill {{ height: 100%; background: var(--cyan); border-radius: 2px; transition: width 0.3s; }}
  .tc-pbar-label {{ font-size: 10px; color: var(--muted); white-space: nowrap; }}

  .no-trades {{ color: var(--muted); font-size: 13px; padding: 30px; text-align: center; }}

  .actions {{ display: flex; gap: 10px; margin: 14px 0; }}
  .action-btn {{
    background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.25);
    color: var(--cyan); font-family: var(--mono); font-size: 12px; font-weight: 600;
    padding: 6px 14px; border-radius: 6px; cursor: pointer; transition: all 0.15s;
  }}
  .action-btn:hover {{ background: rgba(56,189,248,0.2); border-color: var(--cyan); }}
  .action-btn.red {{ color: var(--red); border-color: rgba(239,68,68,0.25); background: rgba(239,68,68,0.08); }}
  .action-btn.red:hover {{ background: rgba(239,68,68,0.2); border-color: var(--red); }}

  .footer {{ text-align: center; padding: 20px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); margin-top: 30px; }}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">ACTIVE <span>TRADES</span></div>
  <div class="nav-links">
    <a href="pairs_scanner.html">&larr; Scanner</a>
    <a href="symbols.html">Symbols</a>
  </div>
</div>

<div class="content">

  <div class="actions">
    <button class="action-btn" onclick="exportTrades()">&#8681; Export Trades</button>
    <label class="action-btn" style="cursor:pointer;">&#8679; Import Trades
      <input type="file" accept=".json" onchange="importTrades(event)" style="display:none;">
    </label>
    <button class="action-btn red" onclick="if(confirm('Close ALL open trades?'))closeAllTrades()">Close All</button>
    <span style="margin-left:auto;font-size:11px;color:var(--muted);">Last updated: <span id="update-time"></span></span>
  </div>

  <div class="section-label">Open Trades ({len(open_trades)})</div>
  <div class="trade-cards" id="openTrades">
    {open_cards}
  </div>

  <h2 style="margin-top:30px;">Closed Trades</h2>
  <div class="section-label">History ({len(closed_trades)})</div>
  <div class="trade-cards" id="closedTrades">
    {closed_cards}
  </div>
</div>

<div class="footer">
  Trades are stored in localStorage and active_trades.json &middot;
  Re-run scanner to refresh Z-scores and prices
</div>

<!-- EDIT MODAL -->
<div class="edit-overlay" id="editModal" onclick="if(event.target===this)closeEditModal()">
  <div class="edit-modal">
    <h3 id="editTitle">Edit Trade</h3>
    <input type="hidden" id="editTradeId">
    <div class="edit-field">
      <label>Entry Date</label>
      <input type="date" id="editDate">
    </div>
    <div class="edit-field">
      <label id="editLabelA">Entry Price A</label>
      <input type="number" id="editPriceA" step="0.01" min="0">
    </div>
    <div class="edit-field">
      <label id="editLabelB">Entry Price B</label>
      <input type="number" id="editPriceB" step="0.01" min="0">
    </div>
    <div class="edit-field">
      <label id="editLabelSharesA">Shares A</label>
      <input type="number" id="editSharesA" step="1" min="0">
    </div>
    <div class="edit-field">
      <label id="editLabelSharesB">Shares B</label>
      <input type="number" id="editSharesB" step="1" min="0">
    </div>
    <div class="edit-actions">
      <button class="edit-cancel" onclick="closeEditModal()">Cancel</button>
      <button class="edit-save" onclick="saveTradeEdit()">Save Changes</button>
    </div>
  </div>
</div>

<!-- CHART MODAL -->
<div class="chart-overlay" id="chartModal" onclick="if(event.target===this)closeTradeChart()">
  <div class="chart-modal">
    <div class="chart-modal-header">
      <div>
        <div class="chart-modal-title" id="chartTitle"></div>
        <div class="chart-modal-stats" id="chartStats"></div>
      </div>
      <button class="chart-modal-close" onclick="closeTradeChart()">&#x2715;</button>
    </div>
    <div class="chart-canvas-wrap">
      <canvas id="tradeZChart"></canvas>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
// Load annotation + zoom plugins
(function() {{
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js";
  s.onload = () => {{ Chart.register(window["chartjs-plugin-annotation"]); }};
  document.head.appendChild(s);
  const h = document.createElement("script");
  h.src = "https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js";
  h.onload = () => {{
    const z = document.createElement("script");
    z.src = "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js";
    document.head.appendChild(z);
  }};
  document.head.appendChild(h);
}})();

// Current-value marker plugin — draws a price tag on the right edge of the chart
const currentValueMarkerPlugin = {{
  id: "currentValueMarker",
  afterDraw(chart) {{
    const ctx = chart.ctx;
    const chartArea = chart.chartArea;
    chart.data.datasets.forEach((ds, i) => {{
      if (!chart.isDatasetVisible(i)) return;
      let lastVal = null;
      for (let j = ds.data.length - 1; j >= 0; j--) {{
        if (ds.data[j] !== null && ds.data[j] !== undefined) {{ lastVal = ds.data[j]; break; }}
      }}
      if (lastVal === null) return;
      const yAxisID = ds.yAxisID || "y";
      const scale = chart.scales[yAxisID];
      if (!scale) return;
      const yPx = scale.getPixelForValue(lastVal);
      if (yPx < chartArea.top - 5 || yPx > chartArea.bottom + 5) return;

      // Format label based on axis type
      let label;
      const ticks = chart.options.scales[yAxisID]?.ticks;
      if (ticks && ticks.callback) {{
        label = ticks.callback(lastVal, 0, []);
      }} else {{
        label = lastVal.toFixed(2);
      }}

      const color = ds.borderColor || "#38bdf8";
      const x = chartArea.right + 4;
      const font = "bold 10px 'JetBrains Mono', monospace";
      ctx.save();
      ctx.font = font;
      const textW = ctx.measureText(label).width;
      const padX = 5, padY = 3;
      const boxW = textW + padX * 2;
      const boxH = 14 + padY * 2;

      // Draw connector line from chart edge to marker
      ctx.beginPath();
      ctx.setLineDash([2, 2]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.moveTo(chartArea.right, yPx);
      ctx.lineTo(x, yPx);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw marker background
      ctx.fillStyle = color;
      ctx.beginPath();
      const r = 3;
      const bx = x, by = yPx - boxH / 2;
      ctx.roundRect(bx, by, boxW, boxH, [0, r, r, 0]);
      ctx.fill();

      // Arrow notch on left side
      ctx.beginPath();
      ctx.moveTo(bx, yPx - 5);
      ctx.lineTo(bx - 4, yPx);
      ctx.lineTo(bx, yPx + 5);
      ctx.closePath();
      ctx.fill();

      // Draw text
      ctx.fillStyle = "#0a0e17";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      ctx.fillText(label, bx + padX, yPx);
      ctx.restore();
    }});
  }}
}};
Chart.register(currentValueMarkerPlugin);

const crosshairPlugin = {{
  id: "crosshairLine",
  afterDraw(chart) {{
    const active = chart.tooltip?.getActiveElements?.();
    if (!active || !active.length) return;
    const {{ ctx, chartArea: {{ top, bottom }} }} = chart;
    const x = active[0].element.x;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(148,163,184,0.85)";
    ctx.setLineDash([]);
    ctx.stroke();
    ctx.restore();
  }},
}};
Chart.register(crosshairPlugin);

let activeTradeChart = null;

function openTradeChart(btn) {{
  const raw = btn.getAttribute("data-chart").replace(/&amp;/g, "&").replace(/&#39;/g, "'");
  const p = JSON.parse(raw);
  if (!p.dates || p.dates.length === 0) {{
    alert("No chart data available for this pair.");
    return;
  }}
  const [a, b] = p.pair.split("/");
  document.getElementById("chartTitle").innerHTML = `<span class="cm-pair">${{a}}</span><span style="color:#4a5568;margin:0 6px;">/</span><span class="cm-pair">${{b}}</span> Z-Score History`;
  const zAbs = Math.abs(p.currentZ);
  const zColor = zAbs >= 3 ? "#ef4444" : zAbs >= 2 ? "#f59e0b" : zAbs >= 1 ? "#38bdf8" : "#94a3b8";
  const eAbs = Math.abs(p.entryZ);
  const eColor = eAbs >= 3 ? "#ef4444" : eAbs >= 2 ? "#f59e0b" : eAbs >= 1 ? "#38bdf8" : "#94a3b8";
  document.getElementById("chartStats").innerHTML = `
    <div class="cm-stat"><span style="color:var(--muted);">Entry Z</span><span class="cm-val" style="color:${{eColor}}">${{p.entryZ >= 0 ? "+" : ""}}${{p.entryZ.toFixed(2)}}&sigma;</span></div>
    <div class="cm-stat"><span style="color:var(--muted);">Current Z</span><span class="cm-val" style="color:${{zColor}}">${{p.currentZ >= 0 ? "+" : ""}}${{p.currentZ.toFixed(2)}}&sigma;</span></div>
    <div class="cm-stat"><span style="color:var(--muted);">Window</span><span class="cm-val" style="color:#94a3b8;">${{p.zWindow}}d</span></div>`;
  document.getElementById("chartModal").classList.add("open");
  document.body.style.overflow = "hidden";
  setTimeout(() => buildTradeZChart(p.dates, p.z, p.entryZ, p.currentZ), 40);
}}

function closeTradeChart() {{
  document.getElementById("chartModal").classList.remove("open");
  document.body.style.overflow = "";
  if (activeTradeChart) {{ activeTradeChart.destroy(); activeTradeChart = null; }}
}}

function buildTradeZChart(dates, z, entryZ, currentZ) {{
  if (activeTradeChart) {{ activeTradeChart.destroy(); activeTradeChart = null; }}
  const ctx = document.getElementById("tradeZChart").getContext("2d");
  const grad = ctx.createLinearGradient(0, 0, 0, 380);
  grad.addColorStop(0,   "rgba(56,189,248,0.20)");
  grad.addColorStop(0.45,"rgba(56,189,248,0.06)");
  grad.addColorStop(1,   "rgba(56,189,248,0.00)");

  const hLine = (y, color, width, dash, lbl) => ({{
    type: "line", yMin: y, yMax: y,
    borderColor: color, borderWidth: width, borderDash: dash,
    label: {{ display: !!lbl, content: lbl, color, position: "end",
            font: {{ size: 10, family: "'JetBrains Mono',monospace", weight: "600" }},
            xAdjust: -10, yAdjust: y > 0 ? -10 : 8, backgroundColor: "transparent", borderWidth: 0 }},
  }});

  const annotations = {{
    zero: hLine(0,  "rgba(148,163,184,0.30)", 1,   [4,4], "0"),
    p1:   hLine(1,  "rgba(34,197,94,0.55)",   1,   [5,4], "+1\u03c3"),
    n1:   hLine(-1, "rgba(34,197,94,0.55)",   1,   [5,4], "-1\u03c3"),
    p2:   hLine(2,  "rgba(245,158,11,0.75)",  1.5, [5,3], "+2\u03c3"),
    n2:   hLine(-2, "rgba(245,158,11,0.75)",  1.5, [5,3], "-2\u03c3"),
    p3:   hLine(3,  "rgba(239,68,68,0.85)",   1.5, [],    "+3\u03c3"),
    n3:   hLine(-3, "rgba(239,68,68,0.85)",   1.5, [],    "-3\u03c3"),
  }};

  // Add entry Z horizontal line
  if (entryZ != null && Math.abs(entryZ) > 0.1) {{
    annotations.entryLine = {{
      type: "line", yMin: entryZ, yMax: entryZ,
      borderColor: "rgba(168,85,247,0.7)", borderWidth: 1.5, borderDash: [3, 3],
      label: {{ display: true, content: "Entry " + (entryZ >= 0 ? "+" : "") + entryZ.toFixed(2) + "\u03c3",
              color: "#a855f7", position: "start",
              font: {{ size: 9, family: "'JetBrains Mono',monospace", weight: "600" }},
              xAdjust: 10, yAdjust: -10, backgroundColor: "rgba(10,14,23,0.8)", borderWidth: 0,
              padding: 3 }},
    }};
  }}

  activeTradeChart = new Chart(ctx, {{
    type: "line",
    data: {{
      labels: dates,
      datasets: [{{
        label: "Z-Score",
        data: z,
        borderColor: "#38bdf8",
        borderWidth: 1.8,
        pointRadius: 0,
        pointHoverRadius: 3,
        pointBorderWidth: 0,
        fill: true,
        backgroundColor: grad,
        tension: 0.3,
        spanGaps: true,

      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      layout: {{ padding: {{ right: 60 }} }},
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: "#0d1520", borderColor: "#242d40", borderWidth: 1,
          titleColor: "#64748b", bodyColor: "#e2e8f0",
          titleFont: {{ family: "'JetBrains Mono',monospace", size: 11 }},
          bodyFont:  {{ family: "'JetBrains Mono',monospace", size: 14 }},
          padding: 14, caretSize: 5, caretPadding: 50,
          usePointStyle: true, pointStyle: "rectRounded", displayColors: true,
          callbacks: {{
            labelColor: c => ({{
              borderColor: c.dataset.borderColor,
              backgroundColor: c.dataset.borderColor,
              borderWidth: 0, borderRadius: 2,
            }}),
            label: c => {{
              const v = c.raw;
              if (v === null) return "";
              const lv = Math.abs(v) >= 3 ? "EXTREME" : Math.abs(v) >= 2 ? "STRONG" : Math.abs(v) >= 1 ? "SIGNAL" : "neutral";
              return ` Z = ${{v >= 0 ? "+" : ""}}${{v.toFixed(3)}}\u03c3   [${{lv}}]`;
            }},
          }},
        }},
        annotation: {{ annotations }},
        zoom: {{
          pan: {{ enabled: true, mode: "x" }},
          zoom: {{ wheel: {{ enabled: true, speed: 0.1 }}, pinch: {{ enabled: true }}, mode: "x" }},
        }},
      }},
      scales: {{
        x: {{
          ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }}, maxRotation: 0, maxTicksLimit: 10, autoSkip: true }},
          grid: {{ color: "rgba(28,35,51,0.7)" }}, border: {{ color: "#1c2333" }},
        }},
        y: {{
          ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }},
            callback: v => (v >= 0 ? "+" : "") + v.toFixed(2) + "\u03c3" }},
          grid: {{ color: "rgba(28,35,51,0.6)" }}, border: {{ color: "#1c2333" }},
        }},
      }},
    }},
  }});
}}

document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeTradeChart(); }});
</script>

<script>
const TRADES_INIT = {json.dumps(trades, default=str)};

function loadTrades() {{
  const local = JSON.parse(localStorage.getItem("activeTrades") || "[]");
  // Merge: localStorage entry data is authoritative (user may have edited prices/dates)
  // TRADES_INIT only provides fresh currentZ and currentPrices from last Python run
  const initMap = {{}};
  TRADES_INIT.forEach(t => {{ initMap[t.id] = t; }});
  const merged = local.map(t => {{
    if (initMap[t.id]) {{
      // Only take live market data from Python — NOT entry fields the user may have edited
      t.currentZ      = initMap[t.id].currentZ;
      t.currentPriceA = initMap[t.id].currentPriceA;
      t.currentPriceB = initMap[t.id].currentPriceB;
      // Chart data always comes from Python
      t.chartDates    = initMap[t.id].chartDates || [];
      t.chartZ        = initMap[t.id].chartZ || [];
    }}
    // Always recalculate days held and P&L from localStorage entry data
    try {{
      const entry = new Date(t.entryDate);
      t.daysHeld = Math.max(0, Math.floor((Date.now() - entry.getTime()) / 86400000));
    }} catch(e) {{}}
    const dir = t.direction || "";
    if (t.entryPriceA > 0 && t.entryPriceB > 0 && t.currentPriceA > 0 && t.currentPriceB > 0) {{
      let pnlA, pnlB;
      if (dir === "short_a_long_b") {{
        pnlA = (t.entryPriceA - t.currentPriceA) / t.entryPriceA * 100;
        pnlB = (t.currentPriceB - t.entryPriceB) / t.entryPriceB * 100;
      }} else {{
        pnlA = (t.currentPriceA - t.entryPriceA) / t.entryPriceA * 100;
        pnlB = (t.entryPriceB - t.currentPriceB) / t.entryPriceB * 100;
      }}
      t.pnlPct = Math.round((pnlA + pnlB) / 2 * 100) / 100;
    }}
    return t;
  }});
  // Add any TRADES_INIT entries not in local (skip deleted ones)
  const deleted = JSON.parse(localStorage.getItem("deletedTrades") || "[]");
  TRADES_INIT.forEach(t => {{
    if (!merged.some(m => String(m.id) === String(t.id)) && !deleted.includes(String(t.id))) merged.push(t);
  }});
  localStorage.setItem("activeTrades", JSON.stringify(merged));
  return merged;
}}

function renderTrades() {{
  const trades = loadTrades();
  const open = trades.filter(t => t.status === "open");
  const closed = trades.filter(t => t.status === "closed");

  const makeCard = (t) => {{
    const [a, b] = t.pair.split("/");
    const dir = t.direction === "short_a_long_b" ? `Short ${{a}} / Long ${{b}}` : `Long ${{a}} / Short ${{b}}`;
    const dirClass = t.direction === "short_a_long_b" ? "dir-short" : "dir-long";
    const progress = Math.abs(t.entryZ) > 0 ? Math.min(100, (1 - Math.abs(t.currentZ) / Math.abs(t.entryZ)) * 100) : 0;
    const pnl = t.pnlPct || 0;
    const pnlClass = pnl >= 0 ? "pnl-pos" : "pnl-neg";
    const zColor = Math.abs(t.currentZ) < Math.abs(t.entryZ) ? "#34d399" : "#ef4444";
    const sA = t.sharesA || 0;
    const sB = t.sharesB || 0;
    // Dollar P&L per leg
    let dollarPnl = 0;
    if (sA > 0 && sB > 0) {{
      if (t.direction === "short_a_long_b") {{
        dollarPnl = (t.entryPriceA - t.currentPriceA) * sA + (t.currentPriceB - t.entryPriceB) * sB;
      }} else {{
        dollarPnl = (t.currentPriceA - t.entryPriceA) * sA + (t.entryPriceB - t.currentPriceB) * sB;
      }}
    }}
    const dollarClass = dollarPnl >= 0 ? "pnl-pos" : "pnl-neg";
    const rawChgA = t.entryPriceA > 0 ? (t.currentPriceA - t.entryPriceA) / t.entryPriceA * 100 : 0;
    const rawChgB = t.entryPriceB > 0 ? (t.currentPriceB - t.entryPriceB) / t.entryPriceB * 100 : 0;
    const chgA = t.direction === "short_a_long_b" ? -rawChgA : rawChgA;
    const chgB = t.direction === "short_a_long_b" ? rawChgB : -rawChgB;
    const chgAClass = chgA >= 0 ? "pnl-pos" : "pnl-neg";
    const chgBClass = chgB >= 0 ? "pnl-pos" : "pnl-neg";
    const cDates = t.chartDates || [];
    const cZ = t.chartZ || [];
    const hasChart = cDates.length > 0;
    const chartPayload = hasChart ? JSON.stringify({{pair: t.pair, dates: cDates, z: cZ, currentZ: t.currentZ, entryZ: t.entryZ, zWindow: {Z_LENGTH}}}).replace(/&/g, "&amp;").replace(/'/g, "&#39;") : "";
    const chartBtn = hasChart ? `<button class="tc-chart" onclick="openTradeChart(this)" data-chart='${{chartPayload}}'>&#9657; Z-Chart</button>` : "";
    return `<div class="trade-card">
      <div class="tc-header">
        <span class="tc-pair">${{t.pair}}</span>
        <span class="tc-dir ${{dirClass}}">${{dir}}</span>
        <span class="tc-days">${{t.daysHeld || 0}}d held</span>
        ${{hasChart ? chartBtn : ""}}
        ${{t.status === "open" ? `<button class="tc-edit" onclick="openEditModal('${{t.id}}')">&#9998; Edit</button><button class="tc-close" onclick="closeTrade('${{t.id}}')">&#10005; Close</button>` : `<button class="tc-reopen" onclick="reopenTrade('${{t.id}}')">&#8634; Reopen</button><button class="tc-delete" onclick="deleteTrade('${{t.id}}')">&#10005;</button>`}}
      </div>
      <div class="tc-body">
        <div class="tc-stat"><div class="tc-label">Entry Z</div><div class="tc-val">${{t.entryZ >= 0 ? "+" : ""}}${{t.entryZ.toFixed(2)}}&sigma;</div></div>
        <div class="tc-stat"><div class="tc-label">Current Z</div><div class="tc-val" style="color:${{zColor}}">${{t.currentZ >= 0 ? "+" : ""}}${{t.currentZ.toFixed(2)}}&sigma;</div></div>
        <div class="tc-stat"><div class="tc-label">Est P&L</div><div class="tc-val ${{pnlClass}}">${{pnl >= 0 ? "+" : ""}}${{pnl.toFixed(1)}}%</div></div>
        <div class="tc-stat tc-prices"><div class="tc-label">${{a}} <span style="color:var(--cyan);font-size:9px;">${{t.direction === "short_a_long_b" ? "-" : "+"}}${{sA}} shares</span></div><div class="tc-val">$${{t.entryPriceA.toFixed(2)}} &rarr; $${{t.currentPriceA.toFixed(2)}} <span class="${{chgAClass}}">${{chgA >= 0 ? "+" : ""}}${{chgA.toFixed(1)}}%</span></div></div>
        <div class="tc-stat tc-prices"><div class="tc-label">${{b}} <span style="color:var(--cyan);font-size:9px;">${{t.direction === "short_a_long_b" ? "+" : "-"}}${{sB}} shares</span></div><div class="tc-val">$${{t.entryPriceB.toFixed(2)}} &rarr; $${{t.currentPriceB.toFixed(2)}} <span class="${{chgBClass}}">${{chgB >= 0 ? "+" : ""}}${{chgB.toFixed(1)}}%</span></div></div>
        <div class="tc-stat"><div class="tc-label">$ P&L</div><div class="tc-val ${{dollarClass}}">${{dollarPnl >= 0 ? "+":""}}$${{Math.abs(dollarPnl).toFixed(0)}}</div></div>
      </div>
      <div class="tc-progress">
        <div class="tc-pbar-track"><div class="tc-pbar-fill" style="width:${{progress > 0 ? progress.toFixed(0) : progress < 0 ? Math.min(Math.abs(progress), 100).toFixed(0) : 0}}%;${{progress > 0 ? 'background:var(--green);' : progress < 0 ? 'background:var(--red);' : 'background:var(--muted);width:2px;'}}"></div></div>
        <span class="tc-pbar-label" style="color:${{progress < 0 ? 'var(--red)' : progress > 0 ? 'var(--green)' : 'var(--muted)'}}">${{(progress >= 0 ? "+" : "") + progress.toFixed(0)}}% to Z=0</span>
      </div>
    </div>`;
  }};

  document.getElementById("openTrades").innerHTML = open.length
    ? open.map(makeCard).join("") : '<div class="no-trades">No open trades.</div>';
  document.getElementById("closedTrades").innerHTML = closed.length
    ? closed.map(makeCard).join("") : '<div class="no-trades">No closed trades yet.</div>';
}}

function closeTrade(id) {{
  const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
  const t = trades.find(t => t.id === id);
  if (t) {{
    t.status = "closed";
    t.closeDate = new Date().toISOString().slice(0,10);
  }}
  localStorage.setItem("activeTrades", JSON.stringify(trades));
  renderTrades();
}}

function reopenTrade(id) {{
  const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
  const t = trades.find(t => t.id === id);
  if (t) {{
    t.status = "open";
    delete t.closeDate;
  }}
  localStorage.setItem("activeTrades", JSON.stringify(trades));
  renderTrades();
}}

function deleteTrade(id) {{
  if (!confirm("Delete this trade permanently?")) return;
  let trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
  trades = trades.filter(t => String(t.id) !== String(id));
  localStorage.setItem("activeTrades", JSON.stringify(trades));
  // Track deleted IDs so loadTrades() won't re-add from TRADES_INIT
  const deleted = JSON.parse(localStorage.getItem("deletedTrades") || "[]");
  if (!deleted.includes(String(id))) deleted.push(String(id));
  localStorage.setItem("deletedTrades", JSON.stringify(deleted));
  renderTrades();
}}

function closeAllTrades() {{
  const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
  trades.forEach(t => {{
    if (t.status === "open") {{
      t.status = "closed";
      t.closeDate = new Date().toISOString().slice(0,10);
    }}
  }});
  localStorage.setItem("activeTrades", JSON.stringify(trades));
  renderTrades();
}}

function exportTrades() {{
  const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
  if (trades.length === 0) {{ alert("No trades to export."); return; }}
  const blob = new Blob([JSON.stringify(trades, null, 2)], {{ type: "application/json" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "active_trades.json";
  a.click();
  URL.revokeObjectURL(a.href);
}}

function importTrades(e) {{
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      const imported = JSON.parse(reader.result);
      if (!Array.isArray(imported)) {{ alert("Invalid file format."); return; }}
      const existing = JSON.parse(localStorage.getItem("activeTrades") || "[]");
      const ids = new Set(existing.map(t => t.id));
      imported.forEach(t => {{ if (!ids.has(t.id)) existing.push(t); }});
      localStorage.setItem("activeTrades", JSON.stringify(existing));
      renderTrades();
    }} catch(err) {{ alert("Error reading file: " + err.message); }}
  }};
  reader.readAsText(file);
}}

function openEditModal(id) {{
  const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
  const t = trades.find(x => x.id === id);
  if (!t) return;
  const [a, b] = t.pair.split("/");
  document.getElementById("editTradeId").value = id;
  document.getElementById("editTitle").textContent = "Edit Trade: " + t.pair;
  document.getElementById("editDate").value = t.entryDate || "";
  document.getElementById("editPriceA").value = t.entryPriceA || "";
  document.getElementById("editPriceB").value = t.entryPriceB || "";
  document.getElementById("editLabelA").textContent = "Entry Price " + a;
  document.getElementById("editLabelB").textContent = "Entry Price " + b;
  document.getElementById("editSharesA").value = t.sharesA || 0;
  document.getElementById("editSharesB").value = t.sharesB || 0;
  document.getElementById("editLabelSharesA").textContent = "Shares " + a;
  document.getElementById("editLabelSharesB").textContent = "Shares " + b;
  document.getElementById("editModal").classList.add("open");
}}

function closeEditModal() {{
  document.getElementById("editModal").classList.remove("open");
}}

function saveTradeEdit() {{
  const id = document.getElementById("editTradeId").value;
  const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
  const t = trades.find(x => x.id === id);
  if (!t) return;

  const newDate = document.getElementById("editDate").value;
  const newPriceA = parseFloat(document.getElementById("editPriceA").value);
  const newPriceB = parseFloat(document.getElementById("editPriceB").value);
  const newSharesA = parseInt(document.getElementById("editSharesA").value);
  const newSharesB = parseInt(document.getElementById("editSharesB").value);

  if (newDate) t.entryDate = newDate;
  if (!isNaN(newPriceA) && newPriceA > 0) t.entryPriceA = newPriceA;
  if (!isNaN(newPriceB) && newPriceB > 0) t.entryPriceB = newPriceB;
  if (!isNaN(newSharesA) && newSharesA >= 0) t.sharesA = newSharesA;
  if (!isNaN(newSharesB) && newSharesB >= 0) t.sharesB = newSharesB;

  // Recalculate days held
  try {{
    const entry = new Date(t.entryDate);
    t.daysHeld = Math.floor((Date.now() - entry.getTime()) / 86400000);
  }} catch(e) {{}}

  // Recalculate entry Z from chart data if entry date or prices changed
  if (t.chartDates && t.chartDates.length && t.chartZ && t.chartZ.length) {{
    // Find the Z value closest to the entry date
    const idx = t.chartDates.indexOf(t.entryDate);
    if (idx >= 0 && t.chartZ[idx] != null) {{
      t.entryZ = Math.round(t.chartZ[idx] * 100) / 100;
    }} else {{
      // Find nearest date if exact match not found
      let closest = 0;
      let minDiff = Infinity;
      const entryTime = new Date(t.entryDate).getTime();
      for (let i = 0; i < t.chartDates.length; i++) {{
        const diff = Math.abs(new Date(t.chartDates[i]).getTime() - entryTime);
        if (diff < minDiff) {{ minDiff = diff; closest = i; }}
      }}
      if (t.chartZ[closest] != null) {{
        t.entryZ = Math.round(t.chartZ[closest] * 100) / 100;
      }}
    }}
  }}

  // Recalculate P&L with new entry prices
  const dir = t.direction || "";
  if (dir === "short_a_long_b") {{
    const pnlA = (t.entryPriceA - t.currentPriceA) / t.entryPriceA * 100;
    const pnlB = (t.currentPriceB - t.entryPriceB) / t.entryPriceB * 100;
    t.pnlPct = Math.round((pnlA + pnlB) / 2 * 100) / 100;
  }} else {{
    const pnlA = (t.currentPriceA - t.entryPriceA) / t.entryPriceA * 100;
    const pnlB = (t.entryPriceB - t.currentPriceB) / t.entryPriceB * 100;
    t.pnlPct = Math.round((pnlA + pnlB) / 2 * 100) / 100;
  }}

  localStorage.setItem("activeTrades", JSON.stringify(trades));
  closeEditModal();
  renderTrades();
}}

document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeEditModal(); }});

window.addEventListener("DOMContentLoaded", () => {{
  document.getElementById("update-time").textContent = new Date({int(time.time() * 1000)}).toLocaleString();
  renderTrades();
}});
</script>
</body>
</html>"""

    with open("active_trades.html", "w", encoding="utf-8") as f:
        f.write(page)
    print(f"active_trades.html created. ({len(open_trades)} open, {len(closed_trades)} closed)")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    TICKERS = load_master_tickers()
    data = build_dataset(TICKERS)

    if data is None or data.empty:
        print("\n[!] No data available. Check internet or ticker list.")
        exit()
    if len(data) < 2:
        print(f"\n[!] Not enough history ({len(data)} days).")
        exit()

    # Build extended chart history (~5 years, separate cache)
    print("Building extended chart dataset...")
    chart_data = build_chart_dataset(TICKERS)

    # Build volume data (avg daily volume per ticker)
    print("Building volume dataset...")
    vol_avg = build_volume_dataset(TICKERS)

    # Build market cap data
    print("Building market cap dataset...")
    mcap_data = build_market_cap(TICKERS)

    valid = list(data.columns)
    pre_filter_count = len(valid)

    # Pre-filter tickers by price, volume, market cap BEFORE building combinations
    if MIN_PRICE > 0:
        valid = [t for t in valid if data[t].iloc[-1] >= MIN_PRICE]
    if MIN_AVG_VOLUME > 0:
        valid = [t for t in valid if vol_avg.get(t, 0) >= MIN_AVG_VOLUME]
    mcap_stock_min = MCAP_TIERS.get(MIN_MCAP_STOCK.lower(), 0) if isinstance(MIN_MCAP_STOCK, str) else MIN_MCAP_STOCK
    mcap_etf_min   = MCAP_TIERS.get(MIN_MCAP_ETF.lower(), 0)   if isinstance(MIN_MCAP_ETF, str)   else MIN_MCAP_ETF
    if mcap_stock_min > 0 or mcap_etf_min > 0:
        def _mcap_ok(t):
            mc = mcap_data.get(t, 0)
            tt = TICKER_TYPES.get(t, "Unknown")
            if tt == "Pure Stock" and mcap_stock_min > 0:
                return mc >= mcap_stock_min
            if tt == "Pure ETF" and mcap_etf_min > 0:
                return mc >= mcap_etf_min
            return True
        valid = [t for t in valid if _mcap_ok(t)]

    stock_label = MIN_MCAP_STOCK if isinstance(MIN_MCAP_STOCK, str) else f"${MIN_MCAP_STOCK:,}"
    etf_label   = MIN_MCAP_ETF   if isinstance(MIN_MCAP_ETF, str)   else f"${MIN_MCAP_ETF:,}"
    print(f"Tickers: {pre_filter_count} → {len(valid)} after filters (price>=${MIN_PRICE}, vol>={MIN_AVG_VOLUME:,}, stock mcap>={stock_label}, etf mcap>={etf_label})")
    print(f"Cointegration: ADF {int(ADF_CONFIDENCE*100)}% confidence over {ADF_LOOKBACK_YRS}yr lookback")
    print(f"Computing matrices for {len(valid)} symbols...")

    # Pre-compute number of pairs (used in HTML template regardless of cache)
    n_combos = len(valid) * (len(valid) - 1) // 2

    print(f"--- Computing matrices and analyzing {len(valid)} symbols... ---")
    returns    = data.pct_change().dropna(how="all")
    log_prices       = np.log(data.tail(Z_LENGTH))
    log_prices_short = np.log(data.tail(Z_LENGTH_SHORT))
    log_prices_long  = np.log(data.tail(Z_LENGTH_LONG))
    log_prices_full  = np.log(data)              # full history for adaptive windowing
    prices_raw = data

    corr_short = returns.tail(CORR_SHORT).corr()
    corr_long  = returns.tail(CORR_LONG).corr()

    perf_len = min(PERF_LENGTH, len(data) - 1)
    perf = (data.iloc[-1] / data.iloc[-(perf_len + 1)] - 1) * 100

    print("Building combinations...")
    combos = list(itertools.combinations(valid, 2))

    print(f"Analyzing pairs using {NUM_WORKERS} CPU cores...")
    chunksize = max(1, len(combos) // (NUM_WORKERS * 4))
    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_init_analyze_worker,
        initargs=(corr_long, corr_short, log_prices, log_prices_short,
                  log_prices_long, log_prices_full, prices_raw, perf,
                  dict(TICKER_TYPES), dict(ETF_LEV_TYPES))
    ) as pool:
        results = [
            r for r in tqdm(
                pool.imap_unordered(analyze_pair, combos, chunksize=chunksize),
                total=len(combos),
                desc="Analyzing Pairs"
            )
            if r is not None
        ]

    results = sorted(results, key=lambda x: x["Score"], reverse=True)
    total_valid = len(results)
    # Count totals per category before any capping
    total_by_cat = {"Pure ETF": 0, "Pure Stock": 0, "Mixed": 0}
    for r in results:
        total_by_cat[r.get("Category", "Mixed")] += 1

    # The code below runs EVERY time, regardless of whether calculations were cached
    # Apply per-category limits first, then overall limit
    cat_limits = {"Pure ETF": MAX_RESULTS_ETF, "Pure Stock": MAX_RESULTS_STOCK, "Mixed": MAX_RESULTS_MIXED}
    if any(v > 0 for v in cat_limits.values()):
        cat_counts = {"Pure ETF": 0, "Pure Stock": 0, "Mixed": 0}
        capped_results = []
        for r in results:
            cat = r.get("Category", "Mixed")
            limit = cat_limits.get(cat, 0)
            if limit > 0 and cat_counts[cat] >= limit:
                continue
            cat_counts[cat] += 1
            capped_results.append(r)
        results = capped_results
        print(f"Per-category caps applied: ETF={cat_counts['Pure ETF']}"
              f"{'/' + str(MAX_RESULTS_ETF) if MAX_RESULTS_ETF > 0 else ''}, "
              f"Stock={cat_counts['Pure Stock']}"
              f"{'/' + str(MAX_RESULTS_STOCK) if MAX_RESULTS_STOCK > 0 else ''}, "
              f"Mixed={cat_counts['Mixed']}"
              f"{'/' + str(MAX_RESULTS_MIXED) if MAX_RESULTS_MIXED > 0 else ''}")
    top_results = results[:MAX_RESULTS] if MAX_RESULTS > 0 else results
    chart_results = top_results[:MAX_CHARTS] if MAX_CHARTS > 0 else top_results

    # Compute rolling Z-score histories for top 500 pairs (parallel)
    print(f"Computing Z-score chart histories for top {len(chart_results)} pairs using {NUM_WORKERS} CPU cores...")
    chart_chunksize = max(1, len(chart_results) // (NUM_WORKERS * 2))
    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_init_chart_worker,
        initargs=(chart_data, data)
    ) as pool:
        chart_results = list(tqdm(
            pool.imap(_compute_chart_for_pair, chart_results, chunksize=chart_chunksize),
            total=len(chart_results),
            desc="Computing Charts"
        ))
    # Merge chart data back and remove pairs with no chart data
    chart_map = {r["Pair"]: r for r in chart_results}
    for r in top_results:
        if r["Pair"] in chart_map:
            r.update(chart_map[r["Pair"]])
    before_chart_filter = len(top_results)
    top_results = [r for r in top_results if r.get("ZDates") and len(r["ZDates"]) > 0]
    dropped = before_chart_filter - len(top_results)
    if dropped:
        print(f"Removed {dropped} pairs with insufficient data for Z-chart.")

    # Count shown per category (after all filtering, matching HTML generation skips)
    shown_by_cat = {"Pure ETF": 0, "Pure Stock": 0, "Mixed": 0}
    for r in top_results:
        z = r["Z"]
        if any(np.isnan(v) for v in [z, r["Score"], r["Corr"]]):
            continue
        if abs(z) < Z_THRESHOLD:
            continue
        if Z_MAX > 0 and abs(z) > Z_MAX:
            continue
        shown_by_cat[r.get("Category", "Mixed")] += 1

    # Helper: format volume as human-readable string
    def fmt_vol(v):
        if v <= 0: return "—"
        if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
        if v >= 1_000:     return f"{v/1_000:.0f}K"
        return str(int(v))

    print("Generating symbols.html...")
    build_symbols_page(valid)

    # ==========================================
    # GENERATE MAIN DASHBOARD
    # ==========================================
    print("Generating pairs_scanner.html...")

    # Count alignment categories for stats row
    n_aligned     = sum(1 for r in top_results if r.get("Alignment") == "Aligned")
    n_mixed       = sum(1 for r in top_results if r.get("Alignment") == "Mixed")
    n_conflicting = sum(1 for r in top_results if r.get("Alignment") == "Conflicting")
    n_conf_high   = sum(1 for r in top_results if r.get("Confidence") == "High")
    n_conf_med    = sum(1 for r in top_results if r.get("Confidence") == "Med")
    n_conf_low    = sum(1 for r in top_results if r.get("Confidence") == "Low")

    rows_html = ""
    for i, r in enumerate(tqdm(top_results, desc="Generating HTML")):
        z = r["Z"]
        a, b = r["Pair"].split("/")

        if any(np.isnan(v) for v in [z, r["Score"], r["Corr"]]):
            continue
        if abs(z) < Z_THRESHOLD:
            continue  # skip pairs without a tradeable signal
        if Z_MAX > 0 and abs(z) > Z_MAX:
            continue  # likely a structural break, not mean-reversion

        name_a = TICKER_NAMES.get(a, "")
        name_b = TICKER_NAMES.get(b, "")

        if z > Z_STRONG:
            sig_line1, sig_line2, sig_class = f"\u25bc\u25bc SHORT {a}", f"\u25b2\u25b2 LONG {b}", "sig-strong-short"
        elif z > Z_THRESHOLD:
            sig_line1, sig_line2, sig_class = f"\u25bc SHORT {a}", f"\u25b2 LONG {b}", "sig-short"
        elif z < -Z_STRONG:
            sig_line1, sig_line2, sig_class = f"\u25b2\u25b2 LONG {a}", f"\u25bc\u25bc SHORT {b}", "sig-strong-long"
        elif z < -Z_THRESHOLD:
            sig_line1, sig_line2, sig_class = f"\u25b2 LONG {a}", f"\u25bc SHORT {b}", "sig-long"
        else:
            sig_line1, sig_line2, sig_class = "NEUTRAL", "", "sig-neutral"

        cat_class = {"Pure ETF": "cat-etf", "Pure Stock": "cat-stock"}.get(r["Category"], "cat-mixed")

        # Build sector/industry/subindustry tooltip text for each ticker
        def _info_tip(t):
            parts = []
            sec = TICKER_INDUSTRY.get(t, "")
            ind = TICKER_SUBIND.get(t, "")
            sub = TICKER_SUBIND2.get(t, "")
            if sec: parts.append(f"Sector: {sec}")
            if ind: parts.append(f"Industry: {ind}")
            if sub: parts.append(f"Sub-industry: {sub}")
            return " &#10;".join(parts) if parts else t
        tip_a = _info_tip(a)
        tip_b = _info_tip(b)

        price_a    = round(data[a].iloc[-1], 2)
        price_b    = round(data[b].iloc[-1], 2)
        avgvol_a   = round(vol_avg.get(a, 0))
        avgvol_b   = round(vol_avg.get(b, 0))
        mcap_a     = mcap_data.get(a, 0)
        mcap_b     = mcap_data.get(b, 0)
        mcap_min   = min(mcap_a, mcap_b) if mcap_a > 0 and mcap_b > 0 else max(mcap_a, mcap_b)
        z_bar_pct  = min(max((abs(z) / 3.0) * 100, 0), 100)
        z_pos      = z >= 0
        score_pct  = round(min(max(r["Score"] * 100, 0), 100))
        hl         = r.get("HalfLife")
        est_ret    = r.get("EstRet") if r.get("EstRet") is not None else 0.0
        ann_ret    = r.get("AnnRet")
        z30        = r.get("Z30")
        z250       = r.get("Z250")
        alignment  = r.get("Alignment", "Mixed")
        align_class = {"Aligned": "align-yes", "Mixed": "align-mix", "Conflicting": "align-conf"}.get(alignment, "align-mix")
        align_label = alignment
        confidence  = r.get("Confidence", "Low")
        conf_class  = {"High": "conf-high", "Med": "conf-med", "Low": "conf-low"}.get(confidence, "conf-low")
        adapt_win   = r.get("AdaptiveWindow", Z_LENGTH)
        z_adaptive  = r.get("ZAdaptive")

        # Exit price estimates when Z reverts to 0 (equal attribution)
        spread_std = r.get("SpreadStd") or 0.0
        half_move  = z * spread_std / 2.0
        exit_a     = round(price_a * np.exp(-half_move), 2)
        exit_b     = round(price_b * np.exp(+half_move), 2)
        exit_a_chg = round((exit_a / price_a - 1) * 100, 1) if price_a > 0 else 0.0
        exit_b_chg = round((exit_b / price_b - 1) * 100, 1) if price_b > 0 else 0.0

        # Tag tickers — ETF_LEV_TYPES from col 5; stocks default to "normal"
        # Returns 4-state tag: "normal", "leveraged", "inverse", or "lev_inv"
        def _lev_tag(t):
            return ETF_LEV_TYPES.get(t, "normal")
        
        lev_a = _lev_tag(a)
        lev_b = _lev_tag(b)
        
        # Generate badges for leverage/inverse indicators
        badge_map = {
            "normal":      "",
            "leveraged":   '<span class="type-badge type-lev">LEV</span>',
            "inverse":     '<span class="type-badge type-inv">INV</span>',
            "lev_inv":     '<span class="type-badge type-levinv">LEV·INV</span>',
            "etn":         '<span class="type-badge type-etn">ETN</span>',
            "etn_lev":     '<span class="type-badge type-etn type-lev">ETN·LEV</span>',
            "etn_lev_inv": '<span class="type-badge type-etn type-levinv">ETN·LEV·INV</span>',
        }
        badge_a = badge_map.get(lev_a, "")
        badge_b = badge_map.get(lev_b, "")

        chart_payload = json.dumps({
            "pair":      r["Pair"],
            "nameA":     name_a,
            "nameB":     name_b,
            "dates":     r.get("ZDates",     []),
            "z":         r.get("ZHistory",   []),
            "zWindow":   Z_LENGTH,
            "datesAdaptive":  r.get("ZDatesAdaptive",   []),
            "zAdaptive":      r.get("ZHistoryAdaptive", []),
            "adaptiveWindow": r.get("AdaptiveWindow", Z_LENGTH),
            "zAdaptiveCurrent": r.get("ZAdaptive"),
            "currentZ":  r["Z"],
            "halfLife":  r.get("HalfLife"),
            "estRet":    r.get("EstRet"),
            "annRet":    r.get("AnnRet"),
            "priceDates": r.get("PriceDates", []),
            "priceA":    r.get("PriceA",     []),
            "priceB":    r.get("PriceB",     []),
            "priceANow": price_a,
            "priceBNow": price_b,
            "exitA":     exit_a,
            "exitB":     exit_b,
            "exitAChg":  exit_a_chg,
            "exitBChg":  exit_b_chg,
        })
        chart_payload_esc = chart_payload.replace("&", "&amp;").replace("'", "&#39;")

        rows_html += f"""
        <tr class="data-row" data-category="{r['Category']}" data-z="{z}"
            data-price-a="{price_a}" data-price-b="{price_b}"
            data-vol-a="{avgvol_a}" data-vol-b="{avgvol_b}"
            data-lev-a="{lev_a}" data-lev-b="{lev_b}"
            data-mcap="{mcap_min}"
            data-alignment="{alignment}"
            data-confidence="{confidence}">
          <td class="rank-cell">{i+1}</td>
          <td class="pair-cell">
            <div class="pair-names">
              <div class="pair-ticker-row">
                <span class="ticker-a" title="{tip_a}">{a}</span>{badge_a}
                <span class="pair-sep">/</span>
                <span class="ticker-b" title="{tip_b}">{b}</span>{badge_b}
                <span class="{cat_class} cat-badge">{r['Category'].replace('Pure ', '')}</span>
              </div>
              <div class="pair-fullnames">
                <div class="name-a" title="{tip_a}">{name_a}</div>
                <div class="name-b" title="{tip_b}">{name_b}</div>
              </div>
            </div>
          </td>
          <td class="z-cell">
            <div class="z-wrapper">
              <span class="z-value {'z-pos' if z_pos else 'z-neg'}"><span class="z-sub" style="font-size:9px;vertical-align:baseline;margin-right:2px;">100d:</span>{z:+.2f}&sigma;</span>
              <div class="z-bar-track">
                <div class="z-bar-fill {'z-bar-pos' if z_pos else 'z-bar-neg'}" style="width:{z_bar_pct}%;"></div>
              </div>
              <div class="z-sub-row">
                <span class="z-sub" title="30-day Z-score">30d:{f'{z30:+.1f}' if z30 is not None else '\u2014'}</span>
                <span class="z-sub" title="250-day Z-score">250d:{f'{z250:+.1f}' if z250 is not None else '\u2014'}</span>
                <span class="z-sub z-adaptive" title="Adaptive Z ({adapt_win}d window based on half-life)">A{adapt_win}d:{f'{z_adaptive:+.1f}' if z_adaptive is not None else '\u2014'}</span>
              </div>
              <div class="z-badge-row">
                <span class="conf-badge {conf_class}">{confidence}</span>
                <span class="align-badge {align_class}">{align_label}</span>
              </div>
            </div>
          </td>
          <td class="corr-cell">
            <span class="corr-value">{r['Corr']:.2f}</span>
            <span class="adf-value" title="Cointegration confidence ({(1-r['ADF_p'])*100:.1f}%, p={r['ADF_p']:.3f})">{(1-r['ADF_p'])*100:.0f}% Coint</span>
          </td>
          <td class="hl-cell">
            {f'<span class="hl-value">{hl:.0f}d</span>' if hl else '<span class="hl-na">—</span>'}
          </td>
          <td class="est-cell">
            <span class="est-ret">{est_ret:+.1f}%</span>
            {f'<span class="ann-ret">{ann_ret:+.0f}%/yr</span>' if ann_ret is not None else ''}
          </td>
          <td class="score-cell">
            <div class="score-bar-wrap">
              <span class="score-num">{r['Score']:.3f}</span>
              <div class="score-bar-track">
                <div class="score-bar-fill" style="width:{score_pct}%;"></div>
              </div>
            </div>
          </td>
          <td class="sig-cell" data-price-a="{price_a}" data-price-b="{price_b}">
            <div class="signal-badge {sig_class}">
              <div>{sig_line1} <span class="share-count sharesA"></span></div>
              {'<div>' + sig_line2 + ' <span class="share-count sharesB"></span></div>' if sig_line2 else ''}
            </div>
          </td>
          <td class="chart-cell">
            <button class="chart-btn price-btn" onclick="openChart(this,'price')" data-chart='{chart_payload_esc}'>&#9724; Price</button>
            <button class="chart-btn" onclick="openChart(this,'z')" data-chart='{chart_payload_esc}'>&#9657; Z-Chart</button>
          </td>
          <td class="track-cell">
            <button class="track-btn" onclick="trackTrade(this)"
              data-pair="{r['Pair']}" data-z="{z}" data-price-a="{price_a}" data-price-b="{price_b}"
              data-direction="{'short_a_long_b' if z > 0 else 'long_a_short_b' if z < 0 else 'neutral'}"
              data-sig="{sig_line1} / {sig_line2 if sig_line2 else ''}">&#9733; Track</button>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pairs Trading Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
      :root {{
        --bg:        #08090d;
        --surface:   #0d1117;
        --surface2:  #131720;
        --surface3:  #181f2e;
        --border:    #1c2333;
        --border2:   #242d40;
        --text:      #c9d1d9;
        --muted:     #4a5568;
        --faint:     #2d3748;
        --cyan:      #38bdf8;
        --cyan-dim:  rgba(56,189,248,0.12);
        --green:     #22c55e;
        --green-dim: rgba(34,197,94,0.12);
        --red:       #ef4444;
        --red-dim:   rgba(239,68,68,0.12);
        --amber:     #f59e0b;
        --orange:    #f97316;
        --purple:    #a78bfa;
        --mono: 'JetBrains Mono', monospace;
        --sans: 'Syne', sans-serif;
      }}
      *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
      html {{ scroll-behavior: smooth; }}
      body {{ background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; font-size: 14px; }}

      /* TOPBAR */
      .topbar {{
        position: sticky; top: 0; z-index: 200;
        background: rgba(8,9,13,0.92); backdrop-filter: blur(16px);
        border-bottom: 1px solid var(--border);
        padding: 0 32px; height: 56px;
        display: flex; align-items: center; justify-content: space-between; gap: 24px;
      }}
      .topbar-left {{ display: flex; align-items: center; gap: 16px; }}
      .brand {{ font-size: 16px; font-weight: 800; letter-spacing: 0.04em; color: white; }}
      .brand span {{ color: var(--cyan); }}
      .live-dot {{
        width: 7px; height: 7px; background: var(--green); border-radius: 50%;
        box-shadow: 0 0 6px var(--green); animation: pulse 2s ease-in-out infinite;
      }}
      @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1);}} 50%{{opacity:.5;transform:scale(.8);}} }}
      .topbar-meta {{ font-family: var(--mono); font-size: 11px; color: var(--muted); display: flex; gap: 20px; white-space: nowrap; }}
      .topbar-meta em {{ color: var(--text); font-style: normal; }}
      .nav-link {{
        font-size: 12px; font-weight: 600; color: var(--cyan); text-decoration: none;
        letter-spacing: 0.05em; padding: 6px 12px; border: 1px solid rgba(56,189,248,0.3);
        border-radius: 4px; transition: all 0.15s; white-space: nowrap;
      }}
      .nav-link:hover {{ background: var(--cyan-dim); border-color: var(--cyan); }}

      /* STATS ROW */
      .stats-row {{
        background: var(--surface); border-bottom: 1px solid var(--border);
        padding: 6px 20px; display: flex; flex-wrap: wrap;
        overflow: visible;
        position: relative; z-index: 50;
      }}
      .stat-item {{
        padding: 4px 14px 4px 0; margin-right: 14px;
        border-right: 1px solid var(--border); white-space: nowrap; flex-shrink: 0;
      }}
      .stat-item:last-child {{ border-right: none; }}
      .stat-label {{ font-size: 9px; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted); margin-bottom: 1px; }}
      .stat-value {{ font-family: var(--mono); font-size: 14px; font-weight: 600; color: white; }}
      .stat-value.cyan {{ color: var(--cyan); }}
      .stat-value.green {{ color: var(--green); }}
      .stat-value.amber {{ color: var(--amber); }}

      /* CONTROLS */
      .controls {{
        background: var(--surface2); border-bottom: 1px solid var(--border);
        padding: 6px 16px; display: flex; flex-direction: column; gap: 3px;
      }}
      .filter-row {{
        display: flex; gap: 4px; align-items: center; flex-wrap: wrap;
      }}
      .filter-row-label {{
        font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--muted); opacity: 0.5;
        min-width: 42px; white-space: nowrap;
      }}
      .control-group {{
        display: flex; align-items: center; gap: 4px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 5px; padding: 4px 7px;
      }}
      .control-group label {{
        font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--muted); white-space: nowrap;
      }}
      .control-group select,
      .control-group input[type="number"],
      .control-group input[type="text"] {{
        background: transparent; border: none; outline: none;
        color: white; font-family: var(--mono); font-size: 13px; min-width: 0;
      }}
      /* Hide native spinner arrows — we use custom +/− buttons instead */
      .control-group input[type="number"] {{
        -moz-appearance: textfield; width: 64px; text-align: center;
      }}
      .control-group input[type="number"]::-webkit-outer-spin-button,
      .control-group input[type="number"]::-webkit-inner-spin-button {{ -webkit-appearance: none; margin: 0; }}
      .control-group input[type="text"]   {{ width: 90px; }}
      .control-group select {{ cursor: pointer; }}
      .control-group select option {{ background: #0d1117; }}
      /* Custom ± stepper buttons */
      .step-btn {{
        background: var(--surface2); border: 1px solid var(--border2); color: var(--text);
        font-family: var(--mono); font-size: 15px; font-weight: 700;
        width: 24px; height: 24px; border-radius: 4px;
        cursor: pointer; line-height: 1; padding: 0; display: flex;
        align-items: center; justify-content: center; flex-shrink: 0;
        transition: background 0.12s, border-color 0.12s, color 0.12s;
      }}
      .step-btn:hover {{ background: var(--surface3); border-color: var(--cyan); color: var(--cyan); }}
      .step-btn:active {{ transform: scale(0.93); }}

      /* TABLE */
      .table-wrapper {{ padding: 24px 32px; overflow-x: auto; }}
      table {{
        width: 100%; border-collapse: separate; border-spacing: 0;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; overflow: hidden;
      }}
      thead tr {{ background: var(--surface2); }}
      th {{
        padding: 11px 14px; text-align: left; font-size: 10px; font-weight: 700;
        letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted);
        border-bottom: 1px solid var(--border); white-space: nowrap;
        user-select: none; cursor: pointer; transition: color 0.15s;
      }}
      th:hover {{ color: var(--text); }}
      tbody tr {{ transition: background 0.12s; border-bottom: 1px solid var(--border); }}
      tbody tr:last-child {{ border-bottom: none; }}
      tbody tr:hover {{ background: var(--surface3); }}
      tbody tr.row-hidden {{ display: none; }}
      td {{ padding: 10px 14px; vertical-align: middle; white-space: nowrap; }}

      /* PAIR CELL */
      .rank-cell {{ font-family: var(--mono); font-size: 11px; color: var(--muted); width: 38px; text-align: center; }}
      .pair-cell {{ min-width: 260px; }}
      .pair-names {{ display: flex; flex-direction: column; gap: 3px; }}
      .pair-ticker-row {{ display: flex; align-items: center; gap: 4px; }}
      .ticker-a {{ font-family: var(--mono); font-size: 14px; font-weight: 700; color: var(--cyan); cursor: help; }}
      .pair-sep  {{ color: var(--muted); margin: 0 2px; font-family: var(--mono); }}
      .ticker-b  {{ font-family: var(--mono); font-size: 14px; font-weight: 700; color: white; cursor: help; }}
      .pair-fullnames {{ display: flex; flex-direction: column; gap: 1px; margin-top: 2px; }}
      .name-a  {{ font-size: 12px; color: #6ab0cc; white-space: normal; line-height: 1.35; cursor: help; }}
      .name-b  {{ font-size: 12px; color: #8fa8be; white-space: normal; line-height: 1.35; cursor: help; }}

      /* BADGES */
      .cat-badge {{
        display: inline-block; font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
        padding: 2px 6px; border-radius: 3px; margin-left: 6px;
        text-transform: uppercase; vertical-align: middle;
      }}
      .cat-etf   {{ background: rgba(56,189,248,0.1);  color: var(--cyan);   border: 1px solid rgba(56,189,248,0.25); }}
      .cat-stock {{ background: rgba(245,158,11,0.1);  color: var(--amber);  border: 1px solid rgba(245,158,11,0.25); }}
      .cat-mixed {{ background: rgba(167,139,250,0.1); color: var(--purple); border: 1px solid rgba(167,139,250,0.25); }}

      /* Z-SCORE */
      .z-cell {{ min-width: 110px; }}
      .z-wrapper {{ display: flex; flex-direction: column; gap: 4px; }}
      .z-value {{ font-family: var(--mono); font-size: 14px; font-weight: 700; }}
      .z-pos {{ color: var(--red); }}
      .z-neg {{ color: var(--green); }}
      .z-bar-track {{ height: 3px; background: var(--faint); border-radius: 2px; overflow: hidden; width: 80px; }}
      .z-bar-fill  {{ height: 100%; border-radius: 2px; }}
      .z-bar-pos {{ background: var(--red); }}
      .z-bar-neg {{ background: var(--green); }}
      .z-sub-row {{ display: flex; gap: 6px; align-items: center; margin-top: 1px; }}
      .z-sub {{ font-family: var(--mono); font-size: 9px; color: #cbd5e1; }}
      .z-adaptive {{ color: #2dd4bf; }}
      .z-badge-row {{ display: flex; gap: 4px; align-items: center; margin-top: 2px; }}
      .align-badge {{ display: inline-flex; padding: 1px 5px; border-radius: 3px; font-size: 7px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }}
      .align-yes  {{ background: rgba(56,189,248,0.18);  color: #7dd3fc; }}
      .align-mix  {{ background: rgba(245,158,11,0.15);  color: #fcd34d; }}
      .align-conf {{ background: rgba(239,68,68,0.15);   color: #fca5a5; }}
      .conf-badge {{ display: inline-flex; padding: 1px 5px; border-radius: 3px; font-size: 7px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }}
      .conf-high  {{ background: rgba(74,222,128,0.18);  color: #86efac; }}
      .conf-med   {{ background: rgba(245,158,11,0.15);  color: #fcd34d; }}
      .conf-low   {{ background: rgba(239,68,68,0.12);   color: #fca5a5; }}

      /* CORR / PERF / SCORE */
      .corr-cell  {{ min-width: 90px; }}
      .corr-value {{ font-family: var(--mono); font-size: 13px; color: white; display: block; }}
      .adf-value  {{ font-family: var(--mono); font-size: 10px; color: #94a3b8; }}

      /* HALF-LIFE */
      .hl-cell   {{ min-width: 70px; text-align: center; }}
      .hl-value  {{ font-family: var(--mono); font-size: 13px; color: var(--purple); font-weight: 600; }}
      .hl-na     {{ font-family: var(--mono); font-size: 13px; color: var(--muted); }}

      /* EST RETURN CELL */
      .est-cell  {{ min-width: 0; text-align: right; white-space: nowrap; }}
      .est-ret   {{ font-family: var(--mono); font-size: 13px; font-weight: 600; color: #34d399; display: block; }}
      .ann-ret   {{ font-family: var(--mono); font-size: 10px; color: #059669; display: block; }}
    /* ETF TYPE BADGES */
      .type-badge  {{ display: inline-block; font-size: 8px; font-weight: 700; letter-spacing: 0.07em;
        padding: 1px 4px; border-radius: 3px; text-transform: uppercase;
        font-family: var(--mono); vertical-align: middle; }}
      .type-lev    {{ background: rgba(249,115,22,0.12); color: #fb923c; border: 1px solid rgba(249,115,22,0.3); }}
      .type-inv    {{ background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.3);  }}
      .type-levinv {{ background: rgba(239,68,68,0.2);   color: #fca5a5; border: 1px solid rgba(239,68,68,0.5); }}
      .type-etn    {{ background: rgba(167,139,250,0.12); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.3); }}

      .score-cell {{ min-width: 0; white-space: nowrap; }}
      .score-bar-wrap {{ display: flex; flex-direction: column; gap: 3px; }}
      .score-num  {{ font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--amber); }}
      .score-bar-track {{ height: 3px; background: var(--faint); border-radius: 2px; width: 70px; overflow: hidden; }}
      .score-bar-fill  {{ height: 100%; background: linear-gradient(90deg, var(--amber), var(--orange)); border-radius: 2px; }}

      /* SIGNAL */
      .sig-cell {{ min-width: 0; white-space: nowrap; }}
      .signal-badge {{
        display: inline-flex; flex-direction: column; align-items: flex-start; gap: 1px;
        padding: 3px 7px; border-radius: 4px; font-size: 10px;
        font-weight: 700; letter-spacing: 0.04em; white-space: nowrap; font-family: var(--mono);
      }}
      .sig-strong-short {{ background: var(--red-dim);          color: var(--red);    border: 1px solid rgba(239,68,68,0.4); }}
      .sig-short        {{ background: rgba(249,115,22,0.1);    color: var(--orange); border: 1px solid rgba(249,115,22,0.4); }}
      .sig-strong-long  {{ background: var(--green-dim);        color: var(--green);  border: 1px solid rgba(34,197,94,0.4); }}
      .sig-long         {{ background: rgba(132,204,22,0.1);    color: #84cc16;       border: 1px solid rgba(132,204,22,0.4); }}
      .sig-neutral      {{ background: rgba(71,85,105,0.2);     color: var(--muted);  border: 1px solid var(--border); }}

      /* CHART BUTTON */
      .chart-cell {{ min-width: 0; text-align: center; display: flex; flex-direction: column; gap: 3px; align-items: center; justify-content: center; }}
      .chart-btn {{
        background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.25);
        color: var(--cyan); font-family: var(--mono); font-size: 11px; font-weight: 600;
        padding: 4px 8px; border-radius: 4px; cursor: pointer; letter-spacing: 0.05em;
        transition: background 0.15s, border-color 0.15s; white-space: nowrap;
      }}
      .chart-btn:hover {{ background: rgba(56,189,248,0.18); border-color: var(--cyan); }}
      .track-cell {{ text-align: center; }}
      .track-btn {{
        background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25);
        color: var(--green); font-family: var(--mono); font-size: 10px; font-weight: 600;
        padding: 4px 8px; border-radius: 4px; cursor: pointer; letter-spacing: 0.05em;
        transition: background 0.15s, border-color 0.15s; white-space: nowrap;
      }}
      .track-btn:hover {{ background: rgba(34,197,94,0.2); border-color: var(--green); }}
      .track-btn.tracked {{ background: rgba(245,158,11,0.15); border-color: rgba(245,158,11,0.4); color: var(--amber); }}
      .price-btn {{
        background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.25);
        color: var(--purple);
      }}
      .price-btn:hover {{ background: rgba(167,139,250,0.18); border-color: var(--purple); }}

      /* TOGGLE SWITCH */
      .toggle-switch {{ position: relative; display: inline-block; cursor: pointer; }}
      .toggle-switch input {{ opacity: 0; width: 0; height: 0; position: absolute; }}
      .toggle-track {{
        display: inline-flex; align-items: center; width: 36px; height: 20px;
        background: var(--faint); border-radius: 10px; transition: background 0.2s;
        position: relative;
      }}
      .toggle-switch input:checked + .toggle-track {{ background: rgba(56,189,248,0.5); }}
      .toggle-thumb {{
        position: absolute; left: 2px; width: 16px; height: 16px;
        background: var(--muted); border-radius: 50%; transition: left 0.2s, background 0.2s;
      }}
      .toggle-switch input:checked + .toggle-track .toggle-thumb {{ left: 18px; background: var(--cyan); }}

      /* SORT ACTIVE */
      th.sort-active {{ color: var(--cyan); }}
      .sort-indicator {{ color: var(--cyan); font-size: 11px; margin-left: 3px; }}

      /* SHARES (inline in signal) */
      .share-count {{ font-size: 10px; color: #ffffff; margin-left: 3px; }}


      /* MODAL */
      .modal-overlay {{
        display: none; position: fixed; inset: 0; z-index: 1000;
        background: rgba(0,0,0,0.78); backdrop-filter: blur(8px);
        align-items: center; justify-content: center;
      }}
      .modal-overlay.open {{ display: flex; }}
      .modal {{
        background: #0a0e17;
        border: 1px solid #242d40; border-radius: 14px;
        width: min(1500px, 99vw); max-height: 95vh;
        display: flex; flex-direction: column;
        box-shadow: 0 40px 100px rgba(0,0,0,0.7), 0 0 0 1px rgba(56,189,248,0.06);
        overflow: hidden; animation: modal-in 0.2s ease;
      }}
      @keyframes modal-in {{ from{{opacity:0;transform:scale(0.96) translateY(12px);}} to{{opacity:1;transform:none;}} }}

      .modal-header {{
        padding: 22px 28px 18px;
        border-bottom: 1px solid #1c2333;
        display: flex; align-items: flex-start; justify-content: space-between; gap: 20px;
        background: linear-gradient(180deg, #0d1520 0%, #0a0e17 100%);
        flex-shrink: 0;
      }}
      .modal-title {{ display: flex; flex-direction: column; gap: 5px; }}
      .modal-pair  {{
        font-family: var(--mono); font-size: 24px; font-weight: 700;
        color: white; letter-spacing: -0.01em;
      }}
      .modal-pair .ma {{ color: var(--cyan); }}
      .modal-pair .mb {{ color: #e2e8f0; }}
      .modal-pair-names {{ font-size: 12px; color: #4a6080; font-family: var(--mono); }}

      .modal-stats {{ display: flex; gap: 28px; align-items: center; flex-shrink: 0; }}
      .mstat {{ text-align: right; }}
      .mstat-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 3px; }}
      .mstat-value {{ font-family: var(--mono); font-size: 20px; font-weight: 700; }}

      .modal-close {{
        background: none; border: none; color: var(--muted); font-size: 22px;
        cursor: pointer; padding: 0 4px; line-height: 1; transition: color 0.15s; flex-shrink: 0;
        margin-top: 2px;
      }}
      .modal-close:hover {{ color: white; }}

      .modal-body {{ padding: 20px 28px 22px; flex: 1; overflow: hidden; display: flex; flex-direction: column; }}
      .chart-tabs {{ display: flex; gap: 4px; margin-bottom: 14px; flex-shrink: 0; }}
      .chart-tab {{
        background: var(--surface2); border: 1px solid var(--border2); color: var(--muted);
        font-family: var(--mono); font-size: 11px; font-weight: 600; padding: 6px 14px;
        border-radius: 4px; cursor: pointer; letter-spacing: 0.06em; transition: all 0.15s;
      }}
      .chart-tab.active {{ background: rgba(56,189,248,0.12); border-color: var(--cyan); color: var(--cyan); }}
      .chart-tab:hover:not(.active) {{ background: var(--surface3); color: var(--text); }}

      /* MODAL TABS (in header) */
      .modal-tabs {{ display: flex; gap: 6px; align-items: center; flex-shrink: 0; }}
      .modal-tab {{
        background: rgba(30,37,53,0.8); border: 1px solid var(--border2);
        color: var(--muted); font-family: var(--mono); font-size: 11px; font-weight: 600;
        padding: 6px 14px; border-radius: 5px; cursor: pointer; letter-spacing: 0.05em;
        transition: all 0.15s; white-space: nowrap;
      }}
      .modal-tab:hover {{ color: var(--text); border-color: #3a4a66; }}
      .modal-tab.active {{ background: rgba(56,189,248,0.12); border-color: rgba(56,189,248,0.4); color: var(--cyan); }}
      .modal-track-btn {{ color: var(--green) !important; background: rgba(34,197,94,0.08) !important; border-color: rgba(34,197,94,0.25) !important; margin-left: 6px; }}
      .modal-track-btn:hover {{ background: rgba(34,197,94,0.2) !important; border-color: var(--green) !important; }}
      .modal-track-btn.tracked {{ background: rgba(245,158,11,0.15) !important; border-color: rgba(245,158,11,0.4) !important; color: var(--amber) !important; }}

      .chart-legend {{ display: flex; gap: 22px; margin-bottom: 14px; flex-shrink: 0; flex-wrap: wrap; }}
      .leg-item {{ display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); font-family: var(--mono); }}
      .leg-line  {{ width: 24px; height: 2px; border-radius: 1px; flex-shrink: 0; }}

      .chart-container {{ position: relative; flex: 1; min-height: 460px; }}

      .modal-footer {{
        padding: 12px 28px;
        border-top: 1px solid #1c2333;
        font-family: var(--mono); font-size: 11px; color: var(--muted);
        flex-shrink: 0; display: flex; gap: 28px; flex-wrap: wrap;
        background: #080c14;
      }}
      .modal-footer em {{ color: #64748b; font-style: normal; }}

      /* PAGINATION */
      .pagination-bar {{
        display: flex; align-items: center; justify-content: center; gap: 4px;
        padding: 10px 20px; background: var(--surface); border-top: 1px solid var(--border);
        font-family: var(--mono); font-size: 12px;
      }}
      .pagination-bar:empty {{ display: none; }}
      .pg-btn {{
        background: rgba(30,37,53,0.8); border: 1px solid var(--border2);
        color: var(--muted); padding: 4px 10px; border-radius: 4px; cursor: pointer;
        font-family: var(--mono); font-size: 11px; transition: all 0.15s;
      }}
      .pg-btn:hover {{ color: var(--text); border-color: #3a4a66; }}
      .pg-btn.active {{ background: rgba(56,189,248,0.12); border-color: rgba(56,189,248,0.4); color: var(--cyan); font-weight: 700; }}
      .pg-btn:disabled {{ opacity: 0.3; cursor: default; }}
      .pg-info {{ color: var(--muted); font-size: 11px; margin: 0 8px; }}
      tr.pg-hidden {{ display: none; }}

      /* FOOTER */
      .footer {{
        padding: 20px 32px; border-top: 1px solid var(--border);
        background: var(--surface); font-size: 11px; color: var(--muted);
        display: flex; justify-content: space-between; align-items: center; gap: 16px;
      }}
      .footer a {{ color: var(--cyan); text-decoration: none; }}
      .leg-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }}

      ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
      ::-webkit-scrollbar-track {{ background: var(--bg); }}
      ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}
      ::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}
    </style>
    </head>
    <body>

    <!-- TOPBAR -->
    <div class="topbar">
      <div class="topbar-left">
        <div class="live-dot"></div>
        <div class="brand">PAIRS <span>SCANNER</span></div>
      </div>
      <div class="topbar-meta">
        <span>Updated: <em id="update-time"></em></span>
        <span>Scanned: <em>{n_combos:,} pairs</em></span>
        <span>Valid Setups: <em>{total_valid:,}</em></span>
        <span>Showing: <em>{shown_by_cat['Pure Stock']} Stock, {shown_by_cat['Pure ETF']} ETF, {shown_by_cat['Mixed']} Mixed</em></span>
      </div>
      <div style="display:flex;gap:16px;align-items:center;">
        <a href="active_trades.html" class="nav-link" style="color:var(--green);">&#9733; My Trades</a>
        <a href="symbols.html" class="nav-link">Symbols &#8594;</a>
      </div>
    </div>

    <!-- STATS ROW -->
    <div class="stats-row">
      <div class="stat-item"><div class="stat-label">Pairs Scanned</div><div class="stat-value cyan">{n_combos:,}</div></div>
      <div class="stat-item"><div class="stat-label">Valid Setups</div><div class="stat-value green">{total_valid:,}</div></div>
      <div class="stat-item"><div class="stat-label">Active Symbols</div><div class="stat-value">{len(valid)}</div></div>
      <div class="stat-item"><div class="stat-label">Lookback</div><div class="stat-value">{LOOKBACK_DAYS}d</div></div>
      <div class="stat-item"><div class="stat-label">Z Threshold</div><div class="stat-value">&plusmn;{Z_THRESHOLD:.1f}&sigma;</div></div>
      <div class="stat-item"><div class="stat-label">Z Windows</div><div class="stat-value">{Z_LENGTH_SHORT}d / {Z_LENGTH}d / {Z_LENGTH_LONG}d</div></div>
      <div class="stat-item"><div class="stat-label">Corr Windows</div><div class="stat-value">{CORR_SHORT}d / {CORR_LONG}d</div></div>
      <div class="stat-item"><div class="stat-label">Min Corr</div><div class="stat-value">{MIN_CORR_FILTER:.2f}</div></div>
      <div class="stat-item"><div class="stat-label">Cointegration</div><div class="stat-value">{int(ADF_CONFIDENCE*100)}% / {ADF_LOOKBACK_YRS}yr</div></div>
      <div class="stat-item"><div class="stat-label">Alignment</div><div class="stat-value"><span style="color:var(--cyan)">{n_aligned}</span> / <span style="color:var(--amber)">{n_mixed}</span> / <span style="color:var(--red)">{n_conflicting}</span></div></div>
      <div class="stat-item"><div class="stat-label">Confidence</div><div class="stat-value"><span style="color:var(--green)">{n_conf_high}</span> / <span style="color:var(--amber)">{n_conf_med}</span> / <span style="color:var(--red)">{n_conf_low}</span></div></div>
    </div>

    <!-- CONTROLS -->
    <div class="controls">
      <!-- Row 1: Ticker search & market filters -->
      <div class="filter-row">
        <span class="filter-row-label">Search</span>
        <div class="control-group">
          <label>Ticker</label>
          <input type="text" id="tickerSearch" placeholder="SPY, AAPL&hellip;" oninput="applyFilters()">
        </div>
        <div class="control-group">
          <label>Min Price ($)</label>
          <button class="step-btn" onclick="stepValue('minPrice',-1)">−</button>
          <input type="number" id="minPrice" value="0" min="0" step="1" oninput="applyFilters()" style="width:48px;">
          <button class="step-btn" onclick="stepValue('minPrice',1)">+</button>
        </div>
        <div class="control-group">
          <label>Min Avg Vol</label>
          <select id="minVol" onchange="applyFilters()">
            <option value="0">Any</option>
            <option value="100000">&gt; 100K</option>
            <option value="500000">&gt; 500K</option>
            <option value="1000000">&gt; 1M</option>
            <option value="5000000">&gt; 5M</option>
            <option value="10000000">&gt; 10M</option>
          </select>
        </div>
        <div class="control-group">
          <label>Min Mkt Cap</label>
          <select id="minMcap" onchange="applyFilters()">
            <option value="0">Any</option>
            <option value="1000000">Nano (1M+)</option>
            <option value="50000000">Micro (50M+)</option>
            <option value="300000000">Small (300M+)</option>
            <option value="2000000000">Mid (2B+)</option>
            <option value="10000000000">Large (10B+)</option>
            <option value="200000000000">Mega (200B+)</option>
          </select>
        </div>
      </div>
      <!-- Row 2: Pair type & signal quality filters -->
      <div class="filter-row">
        <span class="filter-row-label">Signal</span>
        <div class="control-group">
          <label>Pair Type</label>
          <select id="typeFilter" onchange="applyFilters()">
            <option value="All">All</option>
            <option value="Pure ETF">ETF / ETF</option>
            <option value="Pure Stock">Stock / Stock</option>
            <option value="Mixed">Mixed</option>
          </select>
        </div>
        <div class="control-group">
          <label>ETF Type</label>
          <select id="levFilter" onchange="applyFilters()">
            <option value="all">All</option>
            <option value="exclude_both" selected>Excl Lev &amp; Inv</option>
            <option value="exclude_lev">Excl Lev</option>
            <option value="exclude_inv">Excl Inv</option>
            <option value="exclude_etn">Excl ETN</option>
            <option value="only_lev">Only Lev</option>
            <option value="only_inv">Only Inv</option>
            <option value="only_both">Only Lev &amp; Inv</option>
            <option value="only_etn">Only ETN</option>
          </select>
        </div>
        <div class="control-group">
          <label>Z Align</label>
          <select id="alignFilter" onchange="applyFilters()">
            <option value="all">All</option>
            <option value="Aligned">Aligned</option>
            <option value="Mixed">Mixed</option>
            <option value="Conflicting">Conflicting</option>
            <option value="not_conflicting">Excl Conflicting</option>
          </select>
        </div>
        <div class="control-group">
          <label>Confidence</label>
          <select id="confFilter" onchange="applyFilters()">
            <option value="all">All</option>
            <option value="High">High</option>
            <option value="Med">Med+</option>
            <option value="Low">Low</option>
          </select>
        </div>
        <div class="control-group">
          <label>Min |Z|</label>
          <button class="step-btn" onclick="stepValue('minZ',-0.5)">−</button>
          <input type="number" id="minZ" value="0" min="0" max="5" step="0.5" oninput="applyFilters()" style="width:42px;">
          <button class="step-btn" onclick="stepValue('minZ',0.5)">+</button>
        </div>
      </div>
      <!-- Row 3: Sort, display & trade sizing -->
      <div class="filter-row">
        <span class="filter-row-label">Display</span>
        <div class="control-group">
          <label>Sort</label>
          <select id="sortBy" onchange="sortTable()">
            <option value="score">Score</option>
            <option value="z_abs">|Z-Score|</option>
            <option value="hl">Half-Life</option>
            <option value="adf">Cointegration</option>
            <option value="est_ret">Est Return</option>
            <option value="ann_ret">Ann Return</option>
            <option value="corr">Correlation</option>
            <option value="alignment">Alignment</option>
            <option value="confidence">Confidence</option>
          </select>
        </div>
        <div class="control-group" title="When on, each symbol can appear at most once — only the highest-scored pair for that symbol is shown">
          <label>Unique Syms</label>
          <label class="toggle-switch">
            <input type="checkbox" id="uniqueSymFilter" onchange="applyFilters()" checked>
            <span class="toggle-track"><span class="toggle-thumb"></span></span>
          </label>
        </div>
        <div class="control-group">
          <label>Per Page</label>
          <select id="perPage" onchange="changePerPage()">
            <option value="25">25</option>
            <option value="50" selected>50</option>
            <option value="100">100</option>
            <option value="200">200</option>
            <option value="0">All</option>
          </select>
        </div>
        <div class="control-group">
          <label>Capital ($)</label>
          <button class="step-btn" onclick="stepValue('capitalInput',-1000)">−</button>
          <input type="number" id="capitalInput" value="10000" min="0" step="1000" oninput="calcShares()">
          <button class="step-btn" onclick="stepValue('capitalInput',1000)">+</button>
        </div>
      </div>
    </div>

    <!-- TABLE -->
    <div class="table-wrapper">
    <table id="mainTable">
    <thead>
    <tr>
      <th>#</th>
      <th>Pair / Name</th>
      <th onclick="setSort('z_abs')">Z-Score &#8597;</th>
      <th onclick="setSort('corr')">Corr / Coint &#8597;</th>
      <th onclick="setSort('hl')">Half-Life &#8597;</th>
      <th onclick="setSort('est_ret')" style="text-align:right;">Est Return &#8597;</th>
      <th onclick="setSort('score')">Score &#8597;</th>
      <th>Signal/Shares</th>
      <th style="text-align:center;">Charts</th>
      <th style="text-align:center;">Track</th>
    </tr>
    </thead>
    <tbody id="tableBody">
    {rows_html}
    </tbody>
    </table>
    </div>

    <!-- PAGINATION -->
    <div id="pagination" class="pagination-bar"></div>

    <!-- FOOTER -->
    <div class="footer">
      <div>
        <span class="leg-dot" style="background:var(--red)"></span>Short A / Long B &nbsp;
        <span class="leg-dot" style="background:var(--green)"></span>Long A / Short B &nbsp;
        <span class="leg-dot" style="background:var(--muted)"></span>Neutral
      </div>
      <div>
        Score = {int(W_ZSCORE*100)}% |Z| + {int(W_HALFLIFE*100)}% HL Speed + {int(W_CONFIRM*100)}% Confirm + {int(W_ANNRET*100)}% AnnRet + {int(W_STATIONARY*100)}% Coint + {int(W_CORR*100)}% Corr
        &nbsp;&middot;&nbsp; 50/50 capital sizing
        &nbsp;&middot;&nbsp; <a href="symbols.html">Symbol Reference</a>
      </div>
    </div>

    <!-- Z-SCORE CHART MODAL -->
    <div class="modal-overlay" id="chartModal" onclick="closeOnBg(event)">
      <div class="modal">
        <div class="modal-header">
          <div class="modal-title">
            <div class="modal-pair" id="modalPairLabel"></div>
            <div class="modal-pair-names" id="modalPairNames"></div>
          </div>
          <div class="modal-stats" id="modalStats"></div>
          <div class="modal-tabs" id="modalTabs">
            <button class="modal-tab active" id="tabZ" onclick="switchTab('z')">&#9657; Z-Score</button>
            <button class="modal-tab" id="tabP" onclick="switchTab('price')">&#9724; Price Overlay</button>
            <button class="modal-tab" id="tabB" onclick="switchTab('both')">&#9670; Both</button>
          </div>
          <button class="modal-tab modal-track-btn" id="modalTrackBtn" onclick="trackFromChart()">&#9733; Track</button>
          <button class="modal-close" onclick="closeChart()">&#x2715;</button>
        </div>
        <div class="modal-body">
          <div id="legendZ" class="chart-legend">
            <div class="leg-item"><div class="leg-line" style="background:#38bdf8;height:2px;"></div>Z-Score</div>
            <div class="leg-item"><div class="leg-line" style="background:#22c55e;opacity:.8;"></div>&plusmn;1&sigma;</div>
            <div class="leg-item"><div class="leg-line" style="background:#f59e0b;opacity:.8;"></div>&plusmn;2&sigma;</div>
            <div class="leg-item"><div class="leg-line" style="background:#ef4444;opacity:.9;"></div>&plusmn;3&sigma;</div>
            <div class="leg-item"><div class="leg-line" style="background:#94a3b8;opacity:.35;"></div>Zero</div>
          </div>
          <div id="legendP" class="chart-legend" style="display:none;">
            <div class="leg-item"><div class="leg-line" style="background:#38bdf8;"></div><span id="legLabelA" style="color:#38bdf8;">Leg A</span></div>
            <div class="leg-item"><div class="leg-line" style="background:#a78bfa;"></div><span id="legLabelB" style="color:#a78bfa;">Leg B</span></div>
            <div class="leg-item" style="color:#64748b;font-size:11px;">Normalized to 100 at first shared date</div>
          </div>
          <div id="legendB" class="chart-legend" style="display:none;">
            <div class="leg-item"><div class="leg-line" style="background:#38bdf8;height:2px;"></div>Z-Score</div>
            <div class="leg-item"><div class="leg-line" style="background:#22c55e;opacity:.8;"></div>&plusmn;1&sigma;</div>
            <div class="leg-item"><div class="leg-line" style="background:#f59e0b;opacity:.8;"></div>&plusmn;2&sigma;</div>
            <div style="width:1px;background:#2d3748;margin:0 6px;"></div>
            <div class="leg-item"><div class="leg-line" style="background:#38bdf8;"></div><span id="legBLabelA" style="color:#38bdf8;">A</span></div>
            <div class="leg-item"><div class="leg-line" style="background:#a78bfa;"></div><span id="legBLabelB" style="color:#a78bfa;">B</span></div>
            <div class="leg-item" style="color:#4a5568;font-size:11px;">Price normalized to 100</div>
          </div>
          <div class="chart-container">
            <canvas id="zChart" style="display:block;"></canvas>
            <canvas id="pChart" style="display:none;position:absolute;inset:0;width:100%;height:100%;"></canvas>
            <canvas id="bChart" style="display:none;position:absolute;inset:0;width:100%;height:100%;"></canvas>
          </div>
        </div>
        <div class="modal-footer" id="modalFooter"></div>
      </div>
    </div>

    <script>
    // ─── CHART STATE ──────────────────────────────────────────────────────────────
    let activeChart     = null;
    let activePChart    = null;
    let activeBChart    = null;
    let currentChartData = null;

    // Load annotation + zoom plugins async
    (function() {{
      const s = document.createElement("script");
      s.src = "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js";
      s.onload = () => {{ Chart.register(window["chartjs-plugin-annotation"]); }};
      document.head.appendChild(s);
      const h = document.createElement("script");
      h.src = "https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js";
      h.onload = () => {{
        const z = document.createElement("script");
        z.src = "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js";
        document.head.appendChild(z);
      }};
      document.head.appendChild(h);
    }})();

    // Vertical crosshair line plugin
    const crosshairPlugin = {{
      id: "crosshairLine",
      afterDraw(chart) {{
        const active = chart.tooltip?.getActiveElements?.();
        if (!active || !active.length) return;
        const {{ ctx, chartArea: {{ top, bottom }} }} = chart;
        const x = active[0].element.x;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(148,163,184,0.85)";
        ctx.setLineDash([]);
        ctx.stroke();
        ctx.restore();
      }},
    }};
    Chart.register(crosshairPlugin);

    // Current-value marker plugin — draws a price tag on the right edge
    const currentValueMarkerPlugin = {{
      id: "currentValueMarker",
      afterDraw(chart) {{
        const ctx = chart.ctx;
        const chartArea = chart.chartArea;
        chart.data.datasets.forEach((ds, i) => {{
          if (!chart.isDatasetVisible(i)) return;
          let lastVal = null;
          for (let j = ds.data.length - 1; j >= 0; j--) {{
            if (ds.data[j] !== null && ds.data[j] !== undefined) {{ lastVal = ds.data[j]; break; }}
          }}
          if (lastVal === null) return;
          const yAxisID = ds.yAxisID || "y";
          const scale = chart.scales[yAxisID];
          if (!scale) return;
          const yPx = scale.getPixelForValue(lastVal);
          if (yPx < chartArea.top - 5 || yPx > chartArea.bottom + 5) return;

          let label;
          const ticks = chart.options.scales[yAxisID]?.ticks;
          if (ticks && ticks.callback) {{
            label = ticks.callback(lastVal, 0, []);
          }} else {{
            label = lastVal.toFixed(2);
          }}

          const color = ds.borderColor || "#38bdf8";
          const x = chartArea.right + 4;
          const font = "bold 10px 'JetBrains Mono', monospace";
          ctx.save();
          ctx.font = font;
          const textW = ctx.measureText(label).width;
          const padX = 5, padY = 3;
          const boxW = textW + padX * 2;
          const boxH = 14 + padY * 2;

          ctx.beginPath();
          ctx.setLineDash([2, 2]);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.moveTo(chartArea.right, yPx);
          ctx.lineTo(x, yPx);
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.fillStyle = color;
          ctx.beginPath();
          const r = 3;
          const bx = x, by = yPx - boxH / 2;
          ctx.roundRect(bx, by, boxW, boxH, [0, r, r, 0]);
          ctx.fill();

          ctx.beginPath();
          ctx.moveTo(bx, yPx - 5);
          ctx.lineTo(bx - 4, yPx);
          ctx.lineTo(bx, yPx + 5);
          ctx.closePath();
          ctx.fill();

          ctx.fillStyle = "#0a0e17";
          ctx.textBaseline = "middle";
          ctx.textAlign = "left";
          ctx.fillText(label, bx + padX, yPx);
          ctx.restore();
        }});
      }}
    }};
    Chart.register(currentValueMarkerPlugin);

    // ─── TAB SWITCH ──────────────────────────────────────────────────────────────
    function switchTab(mode) {{
      const isZ = mode === 'z', isP = mode === 'price', isB = mode === 'both';
      document.getElementById("tabZ").classList.toggle("active", isZ);
      document.getElementById("tabP").classList.toggle("active", isP);
      document.getElementById("tabB").classList.toggle("active", isB);
      document.getElementById("legendZ").style.display = isZ ? "" : "none";
      document.getElementById("legendP").style.display = isP ? "" : "none";
      document.getElementById("legendB").style.display = isB ? "" : "none";
      document.getElementById("zChart").style.display  = isZ ? "block" : "none";
      document.getElementById("pChart").style.display  = isP ? "block" : "none";
      document.getElementById("bChart").style.display  = isB ? "block" : "none";
      if (isP && currentChartData && !activePChart) buildPriceChart(currentChartData);
      if (isB && currentChartData && !activeBChart) buildBothChart(currentChartData);
    }}

    // ─── OPEN CHART MODAL ────────────────────────────────────────────────────────
    function openChart(btn, mode) {{
      const raw = btn.getAttribute("data-chart")
        .replace(/&amp;/g, "&").replace(/&#39;/g, "'")
        .replace(/&lt;/g, "<").replace(/&gt;/g, ">");
      const p = JSON.parse(raw);
      if (!p.dates || p.dates.length === 0) {{ alert("No chart data available for this pair."); return; }}
      currentChartData = p;
      // Update modal Track button state
      const trk = document.getElementById("modalTrackBtn");
      const _trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
      if (_trades.some(t => t.pair === p.pair && t.status === "open")) {{
        trk.classList.add("tracked"); trk.innerHTML = "&#10003; Tracked";
      }} else {{
        trk.classList.remove("tracked"); trk.innerHTML = "&#9733; Track";
      }}
      const a = p.pair.split("/")[0], b = p.pair.split("/")[1];

      // Header
      document.getElementById("modalPairLabel").innerHTML =
        `<span class="ma">${{a}}</span><span style="color:#4a5568;margin:0 8px;">/</span><span class="mb">${{b}}</span>`;
      document.getElementById("modalPairNames").textContent =
        [p.nameA, p.nameB].filter(Boolean).join("  ·  ");
      document.getElementById("legLabelA").textContent = a;
      document.getElementById("legLabelB").textContent = b;

      // Stats
      const zAbs = Math.abs(p.currentZ);
      const zColor = zAbs >= 3 ? "#ef4444" : zAbs >= 2 ? "#f59e0b" : zAbs >= 1 ? "#38bdf8" : "#94a3b8";
      const zaVal = p.zAdaptiveCurrent;
      const zaAbs = zaVal != null ? Math.abs(zaVal) : 0;
      const zaColor = zaAbs >= 3 ? "#ef4444" : zaAbs >= 2 ? "#f59e0b" : zaAbs >= 1 ? "#38bdf8" : "#94a3b8";
      const zaStr = zaVal != null ? (zaVal >= 0 ? "+" : "") + zaVal.toFixed(2) + "\u03c3" : "\u2014";
      const hlStr  = p.halfLife != null ? Math.round(p.halfLife) + "d" : "—";
      const estStr = p.estRet  != null ? (p.estRet  >= 0 ? "+" : "") + p.estRet.toFixed(1)  + "%" : "—";
      const annStr = p.annRet  != null ? (p.annRet  >= 0 ? "+" : "") + p.annRet.toFixed(0)  + "%/yr" : "—";
      document.getElementById("modalStats").innerHTML = `
        <div class="mstat">
          <div class="mstat-label">Standard Z <span style="font-size:9px;color:#4a6080;">${{p.zWindow}}d</span></div>
          <div class="mstat-value" style="color:${{zColor}};">${{p.currentZ >= 0 ? "+" : ""}}${{p.currentZ.toFixed(2)}}&sigma;</div>
        </div>
        <div class="mstat">
          <div class="mstat-label">Adaptive Z <span style="font-size:9px;color:#4a6080;">${{p.adaptiveWindow}}d</span></div>
          <div class="mstat-value" style="color:${{zaColor}};">${{zaStr}}</div>
        </div>
        <div class="mstat">
          <div class="mstat-label">Half-Life</div>
          <div class="mstat-value" style="color:#a78bfa;">${{hlStr}}</div>
        </div>
        <div class="mstat">
          <div class="mstat-label">Est Return</div>
          <div class="mstat-value" style="color:#34d399;">${{estStr}}</div>
        </div>
        <div class="mstat">
          <div class="mstat-label">Ann Return</div>
          <div class="mstat-value" style="color:#059669;font-size:15px;">${{annStr}}</div>
        </div>
        <div class="mstat">
          <div class="mstat-label">History</div>
          <div class="mstat-value" style="color:#4a6080;">${{(p.priceDates||p.dates||[]).length}}d</div>
        </div>`;

      // Footer — includes exit price estimates when Z reverts to 0
      const footerDates = p.priceDates && p.priceDates.length ? p.priceDates : (p.dates || []);
      const fmtPx  = v => v != null ? "$" + v.toLocaleString("en-US",{{minimumFractionDigits:2,maximumFractionDigits:2}}) : "—";
      const fmtChg = v => v != null ? (v >= 0 ? "+" : "") + v.toFixed(1) + "%" : "";
      const exitAStr = p.exitA != null ? `${{fmtPx(p.exitA)}} (${{fmtChg(p.exitAChg)}})` : "—";
      const exitBStr = p.exitB != null ? `${{fmtPx(p.exitB)}} (${{fmtChg(p.exitBChg)}})` : "—";
      document.getElementById("modalFooter").innerHTML =
        `<span>Z window: <em>${{p.zWindow}}d (std)</em> / <em>${{p.adaptiveWindow}}d (adapt)</em></span>` +
        `<span>Data from: <em>${{footerDates[0] || "—"}}</em></span>` +
        `<span>Last: <em>${{footerDates[footerDates.length-1] || "—"}}</em></span>` +
        `<span style="border-left:1px solid #1c2333;padding-left:16px;font-size:13px;" title="Estimated exit prices if Z reverts to 0">Exit Z=0 &bull; <span style="color:#38bdf8;">${{a}}</span>&nbsp;&#x2248;&nbsp;<em style="color:#b8cedd;font-size:13px;">${{exitAStr}}</em></span>` +
        `<span style="font-size:13px;"><span style="color:#a78bfa;">${{b}}</span>&nbsp;&#x2248;&nbsp;<em style="color:#b8cedd;font-size:13px;">${{exitBStr}}</em></span>` +
        `<span style="margin-left:auto;font-size:10px;color:#2d3748;">Equal attribution &middot; ESC to close</span>`;

      // Destroy old charts
      if (activeChart)  {{ activeChart.destroy();  activeChart  = null; }}
      if (activePChart) {{ activePChart.destroy(); activePChart = null; }}
      if (activeBChart) {{ activeBChart.destroy(); activeBChart = null; }}

      // Reset to Z tab
      document.getElementById("zChart").style.display = "block";
      document.getElementById("pChart").style.display = "none";
      document.getElementById("bChart").style.display = "none";
      document.getElementById("tabZ").classList.add("active");
      document.getElementById("tabP").classList.remove("active");
      document.getElementById("tabB").classList.remove("active");
      document.getElementById("legendZ").style.display = "";
      document.getElementById("legendP").style.display = "none";
      document.getElementById("legendB").style.display = "none";
      document.getElementById("legBLabelA").textContent = a;
      document.getElementById("legBLabelB").textContent = b;

      // Open modal then build charts (slight delay for canvas visibility)
      document.getElementById("chartModal").classList.add("open");
      document.body.style.overflow = "hidden";

      setTimeout(() => {{
        buildZChart(p.dates, p.z, p.zWindow, p.datesAdaptive, p.zAdaptive, p.adaptiveWindow, p.currentZ, p.zAdaptiveCurrent);
        if (mode === 'price') switchTab('price');
      }}, 40);
    }}

    function buildZChart(dates, z, zWindow, datesAdapt, zAdapt, adaptWin, currentZ, zAdaptiveCurrent) {{
      if (activeChart) {{ activeChart.destroy(); activeChart = null; }}
      const ctx = document.getElementById("zChart").getContext("2d");

      const grad = ctx.createLinearGradient(0, 0, 0, 380);
      grad.addColorStop(0,   "rgba(56,189,248,0.20)");
      grad.addColorStop(0.45,"rgba(56,189,248,0.06)");
      grad.addColorStop(1,   "rgba(56,189,248,0.00)");

      // Merge dates from both series into a common axis
      const allDatesSet = new Set([...dates, ...(datesAdapt || [])]);
      const allDates = [...allDatesSet].sort();

      // Map standard Z onto common axis
      const stdMap = {{}};
      if (dates) dates.forEach((d,i) => {{ stdMap[d] = z[i]; }});
      const zStdAligned = allDates.map(d => stdMap[d] !== undefined ? stdMap[d] : null);

      // Map adaptive Z onto common axis
      const adaptMap = {{}};
      if (datesAdapt) datesAdapt.forEach((d,i) => {{ adaptMap[d] = zAdapt[i]; }});
      const zAdaptAligned = allDates.map(d => adaptMap[d] !== undefined ? adaptMap[d] : null);

      const hasAdaptive = adaptWin && adaptWin !== zWindow && datesAdapt && datesAdapt.length > 0;

      const hLine = (y, color, width, dash, lbl) => ({{
        type: "line", yMin: y, yMax: y,
        borderColor: color, borderWidth: width, borderDash: dash,
        label: {{ display: !!lbl, content: lbl, color, position: "end",
                  font: {{ size: 10, family: "'JetBrains Mono',monospace", weight: "600" }},
                  xAdjust: -10, yAdjust: y > 0 ? -10 : 8, backgroundColor: "transparent", borderWidth: 0 }},
      }});

      const datasets = [{{
        label: "Standard Z (" + zWindow + "d)",
        data: zStdAligned,
        borderColor: "#38bdf8",
        borderWidth: 1.8,
        pointRadius: 0,
        pointHoverRadius: 0,
        pointBorderWidth: 0,
        fill: true,
        backgroundColor: grad,
        tension: 0.3,
        spanGaps: true,

      }}];
      if (hasAdaptive) {{
        datasets.push({{
          label: "Adaptive Z (" + adaptWin + "d)",
          data: zAdaptAligned,
          borderColor: "#2dd4bf",
          borderWidth: 1.5,
          borderDash: [5, 3],
          pointRadius: 0,
          pointHoverRadius: 0,
          pointBorderWidth: 0,
          fill: false,
          tension: 0.3,
          spanGaps: true,

        }});
      }}

      activeChart = new Chart(ctx, {{
        type: "line",
        data: {{
          labels: allDates,
          datasets: datasets,
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          layout: {{ padding: {{ right: 60 }} }},
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{
            legend: {{ display: hasAdaptive, position: "top",
              labels: {{ color: "#e2e8f0", font: {{ family: "'JetBrains Mono',monospace", size: 10 }}, padding: 12,
                usePointStyle: true, pointStyle: "rectRounded", pointStyleWidth: 20,
                generateLabels: (chart) => chart.data.datasets.map((ds, i) => ({{
                  text: ds.label, fontColor: "#e2e8f0",
                  fillStyle: ds.borderColor, strokeStyle: ds.borderColor,
                  pointStyle: "rectRounded", lineWidth: 0,
                  lineDash: ds.borderDash || [], datasetIndex: i,
                  hidden: !chart.isDatasetVisible(i),
                }})),
              }} }},
            tooltip: {{
              backgroundColor: "#0d1520", borderColor: "#242d40", borderWidth: 1,
              titleColor: "#64748b", bodyColor: "#e2e8f0",
              titleFont: {{ family: "'JetBrains Mono',monospace", size: 11 }},
              bodyFont:  {{ family: "'JetBrains Mono',monospace", size: 14 }},
              padding: 14, caretSize: 5, caretPadding: 50,
              usePointStyle: true, pointStyle: "rectRounded", displayColors: true,
              callbacks: {{
                labelColor: c => ({{
                  borderColor: c.dataset.borderColor,
                  backgroundColor: c.dataset.borderColor,
                  borderWidth: 0,
                  borderRadius: 2,
                }}),
                label: c => {{
                  const v = c.raw;
                  if (v === null) return "";
                  const lv = Math.abs(v) >= 3 ? "EXTREME" : Math.abs(v) >= 2 ? "STRONG" : Math.abs(v) >= 1 ? "SIGNAL" : "neutral";
                  const prefix = c.datasetIndex === 0 ? "Std" : "Adpt";
                  return ` ${{prefix}} Z = ${{v >= 0 ? "+" : ""}}${{v.toFixed(3)}}\u03C3   [${{lv}}]`;
                }},
              }},
            }},
            annotation: {{
              annotations: {{
                zero: hLine(0,  "rgba(148,163,184,0.30)", 1,   [4,4], "0"),
                p1:   hLine(1,  "rgba(34,197,94,0.55)",   1,   [5,4], "+1\u03C3"),
                n1:   hLine(-1, "rgba(34,197,94,0.55)",   1,   [5,4], "-1\u03C3"),
                p2:   hLine(2,  "rgba(245,158,11,0.75)",  1.5, [5,3], "+2\u03C3"),
                n2:   hLine(-2, "rgba(245,158,11,0.75)",  1.5, [5,3], "-2\u03C3"),
                p3:   hLine(3,  "rgba(239,68,68,0.85)",   1.5, [],    "+3\u03C3"),
                n3:   hLine(-3, "rgba(239,68,68,0.85)",   1.5, [],    "-3\u03C3"),
              }},
            }},
            zoom: {{
              pan: {{ enabled: true, mode: "x" }},
              zoom: {{ wheel: {{ enabled: true, speed: 0.1 }}, pinch: {{ enabled: true }}, mode: "x" }},
            }},
          }},
          scales: {{
            x: {{
              ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }}, maxRotation: 0, maxTicksLimit: 10, autoSkip: true }},
              grid: {{ color: "rgba(28,35,51,0.7)" }}, border: {{ color: "#1c2333" }},
            }},
            y: {{
              ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }},
                callback: v => (v >= 0 ? "+" : "") + v.toFixed(2) + "\u03C3" }},
              grid: {{ color: "rgba(28,35,51,0.6)" }}, border: {{ color: "#1c2333" }},
            }},
          }},
        }},
      }});
    }}

    // ─── PRICE OVERLAY CHART ─────────────────────────────────────────────────────
    function buildPriceChart(p) {{
      if (activePChart) {{ activePChart.destroy(); activePChart = null; }}
      const {{ priceDates, priceA, priceB }} = p;
      const a = p.pair.split("/")[0], b = p.pair.split("/")[1];
      if (!priceDates || !priceDates.length) return;

      const ctx = document.getElementById("pChart").getContext("2d");

      const gradA = ctx.createLinearGradient(0, 0, 0, 380);
      gradA.addColorStop(0,   "rgba(56,189,248,0.18)");
      gradA.addColorStop(0.5, "rgba(56,189,248,0.04)");
      gradA.addColorStop(1,   "rgba(56,189,248,0.00)");

      const gradB = ctx.createLinearGradient(0, 0, 0, 380);
      gradB.addColorStop(0,   "rgba(167,139,250,0.14)");
      gradB.addColorStop(0.5, "rgba(167,139,250,0.03)");
      gradB.addColorStop(1,   "rgba(167,139,250,0.00)");

      // Compute correlation for y-axis label
      let corr = null;
      if (priceA.length > 10 && priceB.length > 10) {{
        const n = Math.min(priceA.length, priceB.length);
        const pa = priceA.slice(-n), pb = priceB.slice(-n);
        const ma = pa.reduce((s,v)=>s+v,0)/n, mb = pb.reduce((s,v)=>s+v,0)/n;
        let num=0, da2=0, db2=0;
        for(let i=0;i<n;i++){{num+=(pa[i]-ma)*(pb[i]-mb);da2+=(pa[i]-ma)**2;db2+=(pb[i]-mb)**2;}}
        corr = da2&&db2 ? (num/Math.sqrt(da2*db2)).toFixed(3) : null;
      }}

      activePChart = new Chart(ctx, {{
        type: "line",
        data: {{
          labels: priceDates,
          datasets: [
            {{
              label: a, data: priceA,
              borderColor: "#38bdf8", borderWidth: 2,
              pointRadius: 0, pointHoverRadius: 0,
              fill: true, backgroundColor: gradA,
              tension: 0.25, spanGaps: true,
            }},
            {{
              label: b, data: priceB,
              borderColor: "#a78bfa", borderWidth: 2,
              pointRadius: 0, pointHoverRadius: 0,
              fill: true, backgroundColor: gradB,
              tension: 0.25, spanGaps: true,
            }},
          ],
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          layout: {{ padding: {{ right: 60 }} }},
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: "#0d1520", borderColor: "#242d40", borderWidth: 1,
              titleColor: "#64748b", bodyColor: "#e2e8f0",
              titleFont: {{ family: "'JetBrains Mono',monospace", size: 11 }},
              bodyFont:  {{ family: "'JetBrains Mono',monospace", size: 13 }},
              padding: 14, caretSize: 5, caretPadding: 50,
              usePointStyle: true, pointStyle: "rectRounded",
              callbacks: {{
                label: c => {{
                  const pct = (c.raw - 100).toFixed(2);
                  return ` ${{c.dataset.label}}: ${{c.raw.toFixed(2)}}  (${{pct >= 0 ? "+" : ""}}${{pct}}%)`;
                }},
                labelColor: c => ({{ borderColor: c.dataset.borderColor, backgroundColor: c.dataset.borderColor, borderWidth: 0, borderRadius: 2 }}),
              }},
            }},
            annotation: {{
              annotations: {{
                baseline: {{ type: "line", yMin: 100, yMax: 100,
                  borderColor: "rgba(148,163,184,0.25)", borderWidth: 1, borderDash: [4,4] }},
              }},
            }},
            zoom: {{
              pan: {{ enabled: true, mode: "x" }},
              zoom: {{ wheel: {{ enabled: true, speed: 0.1 }}, pinch: {{ enabled: true }}, mode: "x" }},
            }},
          }},
          scales: {{
            x: {{
              ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }}, maxRotation: 0, maxTicksLimit: 10, autoSkip: true }},
              grid: {{ color: "rgba(28,35,51,0.7)" }}, border: {{ color: "#1c2333" }},
            }},
            y: {{
              ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }}, callback: v => v.toFixed(2) }},
              grid: {{ color: "rgba(28,35,51,0.6)" }}, border: {{ color: "#1c2333" }},
              title: {{
                display: true,
                text: corr ? `Normalized Price (base 100)  |  Corr: ${{corr}}` : "Normalized Price (base 100)",
                color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }},
              }},
            }},
          }},
        }},
      }});
    }}


    // ─── BOTH CHART (dual Y-axis: normalized price + Z-score) ────────────────────
    function buildBothChart(p) {{
      if (activeBChart) {{ activeBChart.destroy(); activeBChart = null; }}
      const {{ priceDates, priceA, priceB, dates: zDates, z: zVals }} = p;
      const a = p.pair.split("/")[0], b = p.pair.split("/")[1];

      const labels = priceDates && priceDates.length ? priceDates : (zDates || []);
      if (!labels.length) return;

      // Align Z onto price date axis
      const zDateSet = {{}};
      if (zDates) zDates.forEach((d,i) => {{ zDateSet[d] = zVals[i]; }});
      const zAligned  = labels.map(d => zDateSet[d] !== undefined ? zDateSet[d] : null);
      const paAligned = labels.map((d,i) => priceA && priceA[i] !== undefined ? priceA[i] : null);
      const pbAligned = labels.map((d,i) => priceB && priceB[i] !== undefined ? priceB[i] : null);

      const ctx = document.getElementById("bChart").getContext("2d");
      const gradA = ctx.createLinearGradient(0,0,0,400);
      gradA.addColorStop(0,"rgba(56,189,248,0.15)"); gradA.addColorStop(1,"rgba(56,189,248,0)");
      const gradB = ctx.createLinearGradient(0,0,0,400);
      gradB.addColorStop(0,"rgba(167,139,250,0.12)"); gradB.addColorStop(1,"rgba(167,139,250,0)");

      const zPtColors = zAligned.map(v => {{
        if (v === null) return "transparent";
        const av = Math.abs(v);
        return av >= 3 ? "#ef4444" : av >= 2 ? "#f59e0b" : av >= 1 ? "#38bdf8" : "rgba(148,163,184,0.35)";
      }});

      const hLine = (y,color,w,dash) => ({{type:"line",yMin:y,yMax:y,yScaleID:"yZ",
        borderColor:color,borderWidth:w,borderDash:dash}});

      activeBChart = new Chart(ctx, {{
        type: "line",
        data: {{ labels, datasets: [
          {{ label: a+" price", data: paAligned, yAxisID:"yP",
            borderColor:"#38bdf8", borderWidth:1.8, pointRadius:0, pointHoverRadius:0,
            fill:true, backgroundColor:gradA, tension:0.25, spanGaps:true, order:2 }},
          {{ label: b+" price", data: pbAligned, yAxisID:"yP",
            borderColor:"#a78bfa", borderWidth:1.8, pointRadius:0, pointHoverRadius:0,
            fill:true, backgroundColor:gradB, tension:0.25, spanGaps:true, order:3 }},
          {{ label:"Z-Score", data: zAligned, yAxisID:"yZ",
            borderColor:"rgba(248,215,80,0.9)", borderWidth:1.5,
            pointRadius:0, pointHoverRadius:0,
            pointBorderWidth:0,
            fill:false, tension:0.3, spanGaps:true, order:1 }},
        ]}},
        options: {{
          responsive:true, maintainAspectRatio:false,
          layout:{{padding:{{right:60}}}},
          interaction:{{mode:"index",intersect:false}},
          plugins:{{
            legend:{{display:false}},
            tooltip:{{
              backgroundColor:"#0d1520", borderColor:"#242d40", borderWidth:1,
              titleColor:"#64748b", bodyColor:"#e2e8f0",
              titleFont:{{family:"'JetBrains Mono',monospace",size:11}},
              bodyFont:{{family:"'JetBrains Mono',monospace",size:13}}, padding:14, caretPadding:20,
              usePointStyle:true, pointStyle:"rectRounded",
              callbacks:{{
                label: c => {{
                  if (c.datasetIndex < 2) {{
                    const pct = c.raw != null ? (c.raw-100).toFixed(2) : null;
                    return ` ${{c.dataset.label.replace(" price","")}}: ${{c.raw?.toFixed(2)??"—"}}  (${{pct!=null&&pct>=0?"+":""}}${{pct??"—"}}%)`;
                  }}
                  const v = c.raw;
                  if (v===null) return " Z = \u2014";
                  const lv = Math.abs(v)>=3?"EXTREME":Math.abs(v)>=2?"STRONG":Math.abs(v)>=1?"SIGNAL":"neutral";
                  return ` Z = ${{v>=0?"+":""}}${{v.toFixed(3)}}\u03C3  [${{lv}}]`;
                }},
                labelColor: c => ({{ borderColor: c.dataset.borderColor, backgroundColor: c.dataset.borderColor, borderWidth: 0, borderRadius: 2 }}),
              }},
            }},
            annotation:{{ annotations:{{
              z0:  hLine(0, "rgba(148,163,184,0.25)",1,[4,4]),
              zp1: hLine(1, "rgba(34,197,94,0.45)",  1,[5,4]),
              zn1: hLine(-1,"rgba(34,197,94,0.45)",  1,[5,4]),
              zp2: hLine(2, "rgba(245,158,11,0.65)", 1,[5,3]),
              zn2: hLine(-2,"rgba(245,158,11,0.65)", 1,[5,3]),
              zp3: hLine(3, "rgba(239,68,68,0.75)",  1,[]),
              zn3: hLine(-3,"rgba(239,68,68,0.75)",  1,[]),
              base:{{type:"line",yMin:100,yMax:100,yScaleID:"yP",
                borderColor:"rgba(148,163,184,0.2)",borderWidth:1,borderDash:[4,4]}},
            }} }},
            zoom: {{
              pan: {{ enabled: true, mode: "x" }},
              zoom: {{ wheel: {{ enabled: true, speed: 0.1 }}, pinch: {{ enabled: true }}, mode: "x" }},
            }},
          }},
          scales:{{
            x:{{ ticks:{{color:"#374151",font:{{family:"'JetBrains Mono',monospace",size:10}},
                  maxRotation:0,maxTicksLimit:10,autoSkip:true}},
                 grid:{{color:"rgba(28,35,51,0.7)"}},border:{{color:"#1c2333"}} }},
            yP:{{ position:"left",
                 ticks:{{color:"#38bdf8",font:{{family:"'JetBrains Mono',monospace",size:10}},callback:v=>v.toFixed(2)}},
                 grid:{{color:"rgba(28,35,51,0.5)"}},border:{{color:"#1c2333"}},
                 title:{{display:true,text:"Norm. Price (base 100)",color:"rgba(56,189,248,0.5)",
                   font:{{family:"'JetBrains Mono',monospace",size:10}}}} }},
            yZ:{{ position:"right",
                 ticks:{{color:"rgba(248,215,80,0.7)",font:{{family:"'JetBrains Mono',monospace",size:10}},
                   callback:v=>(v>=0?"+":"")+v.toFixed(2)+"\u03C3"}},
                 grid:{{drawOnChartArea:false}},border:{{color:"#1c2333"}},
                 title:{{display:true,text:"Z-Score",color:"rgba(248,215,80,0.5)",
                   font:{{family:"'JetBrains Mono',monospace",size:10}}}} }},
          }},
        }},
      }});
    }}

    // ─── CLOSE MODAL ─────────────────────────────────────────────────────────────
    function closeChart() {{
      document.getElementById("chartModal").classList.remove("open");
      document.body.style.overflow = "";
      if (activeChart)  {{ activeChart.destroy();  activeChart  = null; }}
      if (activePChart) {{ activePChart.destroy(); activePChart = null; }}
      if (activeBChart) {{ activeBChart.destroy(); activeBChart = null; }}
      currentChartData = null;
    }}
    function closeOnBg(e) {{ if (e.target.id === "chartModal") closeChart(); }}
    document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeChart(); }});

    // ─── STEP VALUE (custom ± stepper) ────────────────────────────────────────────
    function stepValue(id, delta) {{
      const el  = document.getElementById(id);
      const val = parseFloat(el.value) || 0;
      const min = parseFloat(el.min) ?? 0;
      const max = el.max !== "" ? parseFloat(el.max) : Infinity;
      el.value  = Math.min(max, Math.max(min, val + delta));
      el.dispatchEvent(new Event("input"));
    }}

    // ─── SHARE CALCULATOR ─────────────────────────────────────────────────────────
    function calcShares() {{
      const total = parseFloat(document.getElementById("capitalInput").value) || 0;
      const leg   = total / 2;
      document.querySelectorAll("tr.data-row:not(.row-hidden)").forEach(row => {{
        const sigCell = row.querySelector(".sig-cell");
        const cA = row.querySelector(".sharesA");
        const cB = row.querySelector(".sharesB");
        if (!sigCell || !cA) return;
        const pA = parseFloat(sigCell.dataset.priceA);
        const pB = parseFloat(sigCell.dataset.priceB);
        if (total > 0 && pA > 0 && pB > 0) {{
          const sA = Math.round(leg / pA);
          cA.textContent = sA.toLocaleString();
          if (cB) cB.textContent = Math.round((sA * pA) / pB).toLocaleString();
        }} else {{
          cA.textContent = "";
          if (cB) cB.textContent = "";
        }}
      }});
    }}

    // ─── FILTERS ──────────────────────────────────────────────────────────────────
    function applyFilters() {{
      const catF      = document.getElementById("typeFilter").value;
      const levF      = document.getElementById("levFilter").value;
      const alignF    = document.getElementById("alignFilter").value;
      const confF     = document.getElementById("confFilter").value;
      const minZv     = parseFloat(document.getElementById("minZ").value) || 0;
      const searchV   = document.getElementById("tickerSearch").value.toUpperCase().trim();
      const minPriceV = parseFloat(document.getElementById("minPrice").value) || 0;
      const minVolV   = parseFloat(document.getElementById("minVol").value) || 0;
      const minMcapV  = parseFloat(document.getElementById("minMcap").value) || 0;
      const uniqueSym = document.getElementById("uniqueSymFilter").checked;

      // First pass: standard filters
      document.querySelectorAll("tr.data-row").forEach(row => {{
        const z        = parseFloat(row.dataset.z);
        const cat      = row.dataset.category;
        const priceA   = parseFloat(row.dataset.priceA);
        const priceB   = parseFloat(row.dataset.priceB);
        const volA     = parseFloat(row.dataset.volA);
        const volB     = parseFloat(row.dataset.volB);
        const levA     = row.dataset.levA || "normal";
        const levB     = row.dataset.levB || "normal";
        const pairText = row.querySelector(".pair-cell").textContent.toUpperCase();

        // ETF/ETN type tags
        const isLev     = levA === "leveraged" || levB === "leveraged" || levA === "etn_lev" || levB === "etn_lev";
        const isInv     = levA === "inverse"   || levB === "inverse";
        const isLevInv  = levA === "lev_inv"   || levB === "lev_inv"  || levA === "etn_lev_inv" || levB === "etn_lev_inv";
        const isEtn     = levA.startsWith("etn") || levB.startsWith("etn");
        const isSpecial = isLev || isInv || isLevInv || isEtn;

        let show = true;
        if (catF !== "All" && cat !== catF)          show = false;
        if (Math.abs(z) < minZv)                     show = false;
        if (searchV && !pairText.includes(searchV))  show = false;
        if (minPriceV > 0 && (priceA < minPriceV || priceB < minPriceV)) show = false;
        if (minVolV > 0 && volA > 0 && volB > 0 && (volA < minVolV || volB < minVolV)) show = false;
        if (minMcapV > 0) {{
          const mcap = parseFloat(row.dataset.mcap) || 0;
          if (mcap < minMcapV) show = false;
        }}

        // ETF Type filter — driven by ETFs.csv col 5 via data-lev-a/b attributes
        if      (levF === "exclude_both" && isSpecial)              show = false;
        else if (levF === "exclude_lev"  && (isLev || isLevInv))   show = false;
        else if (levF === "exclude_inv"  && (isInv || isLevInv))   show = false;
        else if (levF === "exclude_etn"  && isEtn)                  show = false;
        else if (levF === "only_lev"     && !isLev)                 show = false;
        else if (levF === "only_inv"     && !isInv)                 show = false;
        else if (levF === "only_both"    && !(isLev || isInv || isLevInv)) show = false;
        else if (levF === "only_etn"     && !isEtn)                 show = false;

        // Alignment filter
        if (alignF !== "all") {{
          const align = row.dataset.alignment || "Mixed";
          if (alignF === "not_conflicting") {{
            if (align === "Conflicting") show = false;
          }} else {{
            if (align !== alignF) show = false;
          }}
        }}

        // Confidence filter
        if (confF !== "all") {{
          const conf = row.dataset.confidence || "Low";
          if (confF === "High" && conf !== "High") show = false;
          else if (confF === "Med" && conf === "Low") show = false;
          else if (confF === "Low" && conf !== "Low") show = false;
        }}

        row.dataset.baseHidden = show ? "0" : "1";
        row.classList.toggle("row-hidden", !show);
      }});

      // Second pass: unique symbol filter
      if (uniqueSym) {{
        const seenSymbols = new Set();
        const rows = [...document.querySelectorAll("tr.data-row:not(.row-hidden)")];
        rows.sort((a, b) => {{
          const sa = parseFloat(a.querySelector(".score-num")?.textContent) || 0;
          const sb = parseFloat(b.querySelector(".score-num")?.textContent) || 0;
          return sb - sa;
        }});
        rows.forEach(row => {{
          const tickers = row.querySelector(".pair-cell").textContent.match(/[A-Z]{{1,6}}/g) || [];
          const symA = tickers[0], symB = tickers[1];
          if (!symA || !symB) return;
          if (seenSymbols.has(symA) || seenSymbols.has(symB)) {{
            row.classList.add("row-hidden");
          }} else {{
            seenSymbols.add(symA);
            seenSymbols.add(symB);
          }}
        }});
      }}

      currentPage = 1;
      paginateTable();
    }}

    // ─── SORT ─────────────────────────────────────────────────────────────────────
    // ─── PAGINATION ────────────────────────────────────────────────────────────
    let currentPage = 1;
    let rowsPerPage = 50;

    function changePerPage() {{
      rowsPerPage = parseInt(document.getElementById("perPage").value) || 0;
      currentPage = 1;
      paginateTable();
    }}

    function paginateTable() {{
      const rows = [...document.querySelectorAll("tr.data-row")].filter(r => !r.classList.contains("row-hidden"));
      const total = rows.length;
      const pgBar = document.getElementById("pagination");

      // Show all mode
      if (rowsPerPage === 0 || total === 0) {{
        rows.forEach(r => r.classList.remove("pg-hidden"));
        pgBar.innerHTML = total > 0 ? `<span class="pg-info">Showing all ${{total}} pairs</span>` : "";
        calcShares();
        return;
      }}

      const totalPages = Math.ceil(total / rowsPerPage);
      if (currentPage > totalPages) currentPage = totalPages;
      if (currentPage < 1) currentPage = 1;
      const start = (currentPage - 1) * rowsPerPage;
      const end = start + rowsPerPage;

      rows.forEach((r, i) => {{
        r.classList.toggle("pg-hidden", i < start || i >= end);
      }});

      // Build page bar
      let html = `<button class="pg-btn" onclick="goPage(${{currentPage - 1}})" ${{currentPage === 1 ? "disabled" : ""}}>&laquo; Prev</button>`;

      // Smart page range: show first, last, and nearby pages
      const maxBtns = 7;
      let pages = [];
      if (totalPages <= maxBtns) {{
        for (let i = 1; i <= totalPages; i++) pages.push(i);
      }} else {{
        pages.push(1);
        let lo = Math.max(2, currentPage - 2);
        let hi = Math.min(totalPages - 1, currentPage + 2);
        if (lo > 2) pages.push(-1); // ellipsis
        for (let i = lo; i <= hi; i++) pages.push(i);
        if (hi < totalPages - 1) pages.push(-1); // ellipsis
        pages.push(totalPages);
      }}

      pages.forEach(p => {{
        if (p === -1) {{
          html += `<span class="pg-info">&hellip;</span>`;
        }} else {{
          html += `<button class="pg-btn ${{p === currentPage ? "active" : ""}}" onclick="goPage(${{p}})">${{p}}</button>`;
        }}
      }});

      html += `<button class="pg-btn" onclick="goPage(${{currentPage + 1}})" ${{currentPage === totalPages ? "disabled" : ""}}">Next &raquo;</button>`;
      html += `<span class="pg-info">${{start + 1}}&ndash;${{Math.min(end, total)}} of ${{total}}</span>`;
      pgBar.innerHTML = html;
      calcShares();
    }}

    function goPage(p) {{
      currentPage = p;
      paginateTable();
      document.querySelector(".table-wrapper").scrollIntoView({{ behavior: "smooth", block: "start" }});
    }}

    let currentSort = {{ key: "score", asc: false }};

    function setSort(key) {{
      currentSort.asc = (currentSort.key === key) ? !currentSort.asc : false;
      if (key === "hl" && currentSort.key !== key) currentSort.asc = true;
      currentSort.key = key;
      const dd = document.getElementById("sortBy");
      if (dd) dd.value = key;
      sortTable();
    }}

    function sortTable() {{
      const key   = currentSort.key || document.getElementById("sortBy").value;
      const asc   = currentSort.asc;
      const tbody = document.getElementById("tableBody");
      const rows  = [...tbody.querySelectorAll("tr.data-row")];

      const numOf = (row, sel) => {{
        const txt = row.querySelector(sel)?.textContent?.replace(/[^0-9.+-]/g, "") || "";
        return parseFloat(txt) || 0;
      }};

      rows.sort((a, b) => {{
        let va, vb;
        if      (key === "score")   {{ va = numOf(a,".score-num");               vb = numOf(b,".score-num"); }}
        else if (key === "z_abs")   {{ va = Math.abs(parseFloat(a.dataset.z));   vb = Math.abs(parseFloat(b.dataset.z)); }}
        else if (key === "corr")    {{ va = numOf(a,".corr-value");              vb = numOf(b,".corr-value"); }}
        else if (key === "adf")     {{ va = numOf(a,".adf-value") || 0;            vb = numOf(b,".adf-value") || 0; }}
        else if (key === "hl")      {{ va = numOf(a,".hl-value") || 99999;       vb = numOf(b,".hl-value") || 99999; }}
        else if (key === "est_ret") {{ va = numOf(a,".est-ret");                 vb = numOf(b,".est-ret"); }}
        else if (key === "ann_ret") {{ va = numOf(a,".ann-ret") || 0;            vb = numOf(b,".ann-ret") || 0; }}
        else if (key === "alignment") {{
          const alignRank = {{"Aligned": 3, "Mixed": 2, "Conflicting": 1}};
          va = alignRank[a.dataset.alignment] || 2;
          vb = alignRank[b.dataset.alignment] || 2;
        }}
        else if (key === "confidence") {{
          const confRank = {{"High": 3, "Med": 2, "Low": 1}};
          va = confRank[a.dataset.confidence] || 1;
          vb = confRank[b.dataset.confidence] || 1;
        }}
        else return 0;
        return asc ? (va - vb) : (vb - va);
      }});

      rows.forEach((r, i) => {{ r.querySelector(".rank-cell").textContent = i + 1; tbody.appendChild(r); }});
      updateSortIndicators(key, asc);
      paginateTable();
    }}

    function updateSortIndicators(key, asc) {{
      document.querySelectorAll("th[onclick]").forEach(th => {{
        th.querySelector(".sort-indicator")?.remove();
        th.classList.remove("sort-active");
        const onclick = th.getAttribute("onclick") || "";
        if (onclick.includes(`'${{key}}'`)) {{
          th.classList.add("sort-active");
          const span = document.createElement("span");
          span.className = "sort-indicator";
          span.textContent = asc ? " ↑" : " ↓";
          th.appendChild(span);
        }}
      }});
    }}

    // ─── TRADE TRACKER ─────────────────────────────────────────────────────────
    function trackTrade(btn) {{
      const pair = btn.dataset.pair;
      const id   = pair + "_" + new Date().toISOString().slice(0,10);
      const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
      // Toggle: if already tracked, untrack it
      const existingIdx = trades.findIndex(t => t.pair === pair && t.status === "open");
      if (existingIdx !== -1) {{
        trades.splice(existingIdx, 1);
        localStorage.setItem("activeTrades", JSON.stringify(trades));
        btn.classList.remove("tracked");
        btn.textContent = "\u2733 Track";
        showToast("Trade untracked: " + pair);
        return;
      }}
      const priceA = parseFloat(btn.dataset.priceA);
      const priceB = parseFloat(btn.dataset.priceB);
      // Calculate shares from current capital setting (50/50 split)
      const capital = parseFloat(document.getElementById("capitalInput").value) || 0;
      const leg = capital / 2;
      const sharesA = priceA > 0 ? Math.round(leg / priceA) : 0;
      const sharesB = priceB > 0 ? Math.round((sharesA * priceA) / priceB) : 0;
      const trade = {{
        id: id,
        pair: pair,
        direction: btn.dataset.direction,
        signal: btn.dataset.sig,
        entryDate: new Date().toISOString().slice(0,10),
        entryZ: parseFloat(btn.dataset.z),
        entryPriceA: priceA,
        entryPriceB: priceB,
        currentZ: parseFloat(btn.dataset.z),
        currentPriceA: priceA,
        currentPriceB: priceB,
        sharesA: sharesA,
        sharesB: sharesB,
        capital: capital,
        daysHeld: 0,
        status: "open",
      }};
      trades.push(trade);
      localStorage.setItem("activeTrades", JSON.stringify(trades));
      btn.classList.add("tracked");
      btn.textContent = "\u2713 Tracked";
      // Show toast
      showToast("Trade tracked: " + pair + " (" + sharesA + " / " + sharesB + " shares, $" + capital + ")");
    }}

    function trackFromChart() {{
      if (!currentChartData) return;
      const p = currentChartData;
      const pair = p.pair;
      const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
      const btn = document.getElementById("modalTrackBtn");
      // Toggle: if already tracked, untrack it
      const existingIdx = trades.findIndex(t => t.pair === pair && t.status === "open");
      if (existingIdx !== -1) {{
        trades.splice(existingIdx, 1);
        localStorage.setItem("activeTrades", JSON.stringify(trades));
        btn.classList.remove("tracked");
        btn.innerHTML = "&#9733; Track";
        // Also update the table row Track button
        document.querySelectorAll(".track-btn").forEach(tb => {{
          if (tb.dataset.pair === pair) {{ tb.classList.remove("tracked"); tb.textContent = "\u2733 Track"; }}
        }});
        showToast("Trade untracked: " + pair);
        return;
      }}
      const priceA = p.priceANow;
      const priceB = p.priceBNow;
      const z = p.currentZ;
      const direction = z > 0 ? "short_a_long_b" : z < 0 ? "long_a_short_b" : "neutral";
      const capital = parseFloat(document.getElementById("capitalInput").value) || 0;
      const leg = capital / 2;
      const sharesA = priceA > 0 ? Math.round(leg / priceA) : 0;
      const sharesB = priceB > 0 ? Math.round((sharesA * priceA) / priceB) : 0;
      const id = pair + "_" + new Date().toISOString().slice(0,10);
      const trade = {{
        id: id, pair: pair, direction: direction,
        signal: "",
        entryDate: new Date().toISOString().slice(0,10),
        entryZ: z, entryPriceA: priceA, entryPriceB: priceB,
        currentZ: z, currentPriceA: priceA, currentPriceB: priceB,
        sharesA: sharesA, sharesB: sharesB, capital: capital,
        daysHeld: 0, status: "open",
      }};
      trades.push(trade);
      localStorage.setItem("activeTrades", JSON.stringify(trades));
      btn.classList.add("tracked");
      btn.innerHTML = "&#10003; Tracked";
      // Also update the table row Track button
      document.querySelectorAll(".track-btn").forEach(tb => {{
        if (tb.dataset.pair === pair) {{ tb.classList.add("tracked"); tb.textContent = "\u2713 Tracked"; }}
      }});
      showToast("Trade tracked: " + pair + " (" + sharesA + " / " + sharesB + " shares, $" + capital + ")");
    }}

    function exportTrades() {{
      const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
      if (trades.length === 0) {{ showToast("No trades to export"); return; }}
      const blob = new Blob([JSON.stringify(trades, null, 2)], {{ type: "application/json" }});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "active_trades.json";
      a.click();
      URL.revokeObjectURL(a.href);
      showToast("Exported " + trades.length + " trade(s)");
    }}

    function showToast(msg) {{
      let t = document.getElementById("toast");
      if (!t) {{
        t = document.createElement("div");
        t.id = "toast";
        t.style.cssText = "position:fixed;bottom:20px;right:20px;background:#0d1520;border:1px solid var(--cyan);color:#e2e8f0;padding:10px 20px;border-radius:8px;font-family:var(--mono);font-size:12px;z-index:9999;opacity:0;transition:opacity 0.3s;";
        document.body.appendChild(t);
      }}
      t.textContent = msg;
      t.style.opacity = "1";
      setTimeout(() => {{ t.style.opacity = "0"; }}, 2500);
    }}

    // Mark already-tracked pairs on load
    function markTrackedPairs() {{
      const trades = JSON.parse(localStorage.getItem("activeTrades") || "[]");
      const openPairs = new Set(trades.filter(t => t.status === "open").map(t => t.pair));
      document.querySelectorAll(".track-btn").forEach(btn => {{
        if (openPairs.has(btn.dataset.pair)) {{
          btn.classList.add("tracked");
          btn.textContent = "\u2713 Tracked";
        }}
      }});
    }}

    window.addEventListener("DOMContentLoaded", () => {{
      document.getElementById("update-time").textContent = new Date({int(time.time() * 1000)}).toLocaleString();
      applyFilters();
      markTrackedPairs();
      document.getElementById("sortBy").addEventListener("change", () => {{
        currentSort.key = document.getElementById("sortBy").value;
        currentSort.asc = (currentSort.key === "hl");
        sortTable();
      }});
    }});
    </script>
    </body>
    </html>"""

    with open("pairs_scanner.html", "w", encoding="utf-8") as f:
            f.write(html)

    print(f"pairs_scanner.html created. ({len(top_results)} pairs rendered)")

    # ── Active Trades ──
    print("Updating active trades...")
    active_trades = update_active_trades(data, chart_data)
    generate_trades_page(active_trades)

    print("\nDone. Open pairs_scanner.html in your browser.")

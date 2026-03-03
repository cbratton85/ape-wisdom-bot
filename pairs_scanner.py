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
BATCH_SIZE = 40
COOLDOWN = 3
LOOKBACK_DAYS       = 650   # Days used for scoring / correlation / perf
CHART_LOOKBACK_DAYS = 1825  # ~5 years used for Z-score chart history
VOL_AVG_DAYS        = 30    # Rolling window for average volume calculation
CACHE_UPDATE_COOLDOWN_HOURS = 4
NUM_WORKERS = max(1, (mp.cpu_count() or 2) - 0)  # CPU cores for parallel pair analysis

CORR_SHORT = 35
CORR_LONG = 100
Z_LENGTH = 100
Z_LENGTH_SHORT = 30
Z_LENGTH_LONG  = 250
PERF_LENGTH = 300

MIN_CORR_FILTER = 0.60
Z_THRESHOLD = 2.0
Z_STRONG = 2.0

W_ZSCORE = 0.50
W_CORR_BRK = 0.25
W_REL_PERF = 0.25

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
            data = data[[c for c in data.columns if c in master]]
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
            
            if hours_old < 12:  # Extended cooldown for volume to prevent rate limits
                print(f"--- Volume cache fresh ({round(hours_old,2)}h). Using cached volume. ---")
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
            if hours_old < 72:
                print(f"--- Market-cap cache fresh ({round(hours_old,1)}h). Using cached data. ---")
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

    # Z-score on last Z_LENGTH days for scoring
    ratio = log_prices[a] - log_prices[b]
    mean  = ratio.mean()
    std   = ratio.std()

    if std == 0:
        return None

    z  = (ratio.iloc[-1] - mean) / std
    rp = perf[a] - perf[b]
    spread_std = std  # store for EstRet calculation

    z_norm    = min(abs(z) / 3.0, 1.0)
    corr_norm = min(max(corr_brk, 0) / 0.5, 1.0)
    perf_norm = min(abs(rp) / 10.0, 1.0)

    score = W_ZSCORE * z_norm + W_CORR_BRK * corr_norm + W_REL_PERF * perf_norm

    type_a = TICKER_TYPES.get(a, "Unknown")
    type_b = TICKER_TYPES.get(b, "Unknown")

    if type_a == "Pure ETF" and type_b == "Pure ETF":
        pair_category = "Pure ETF"
    elif type_a == "Pure Stock" and type_b == "Pure Stock":
        pair_category = "Pure Stock"
    else:
        pair_category = "Mixed"

    if any(np.isnan(v) for v in [z, cl, corr_brk, rp, score]):
        return None

    # ── Half-life of mean reversion (using full log-price spread in prices_raw) ──
    # Pairs without detectable mean-reversion are not viable pairs trades.
    try:
        full_spread = np.log(prices_raw[a]) - np.log(prices_raw[b])
        hl = compute_half_life(full_spread)
    except Exception:
        hl = float('nan')

    if np.isnan(hl):
        return None

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

    # ── Timeframe alignment ──
    signs = (z30 > 0, z > 0, z250 > 0)
    if (all(signs) or not any(signs)) and abs(z) >= Z_THRESHOLD:
        alignment = "Aligned"
    elif (z30 > 0) != (z250 > 0):
        alignment = "Conflicting"
    else:
        alignment = "Mixed"

    # ── Confidence level (how many timeframes confirm at threshold) ──
    above = sum(1 for v in (abs(z30), abs(z), abs(z250)) if v >= Z_THRESHOLD)
    same_dir = all(signs) or not any(signs)
    if above == 3 and same_dir:
        confidence = "High"
    elif above >= 2 and same_dir:
        confidence = "Med"
    else:
        confidence = "Low"

    # ── Estimated pairs trade return (gross spread return if fully reverts) ──
    est_ret = round(abs(z) * spread_std * 100, 2)   # in %
    # Annualized: one trade cycle = one half-life period, so trades/year = 252/hl.
    # This gives ann_ret > est_ret whenever hl < 252 days (i.e. sub-year mean reversion).
    if not np.isnan(hl) and hl > 0:
        ann_ret = round(est_ret * (252 / hl), 1)
    else:
        ann_ret = None

    return {
        "Pair":       f"{a}/{b}",
        "Category":   pair_category,
        "Z":          round(z, 2),
        "Corr":       round(cl, 2),
        "CorrBrk":    round(corr_brk, 3),
        "PerfDiff":   round(rp, 2),
        "Score":      round(score, 3),
        "HalfLife":   hl if not np.isnan(hl) else None,
        "AnnRetA":    ann_a if not np.isnan(ann_a) else None,
        "AnnRetB":    ann_b if not np.isnan(ann_b) else None,
        "EstRet":     est_ret,
        "AnnRet":     ann_ret,
        "SpreadStd":  round(float(std), 6),
        "Z30":        z30,
        "Z250":       z250,
        "Alignment":  alignment,
        "Confidence": confidence,
    }


# ==========================================
# ROLLING Z-SCORE HISTORY FOR CHART
# ==========================================
def compute_z_history(a, b, price_data):
    """Rolling Z-score over all available data.
    Uses an adaptive window: Z_LENGTH when sufficient history exists,
    falling back to half the available data (min 20 days) for shorter-
    history tickers such as leveraged ETFs.
    """
    log_a = np.log(price_data[a].dropna())
    log_b = np.log(price_data[b].dropna())
    combined = pd.DataFrame({"a": log_a, "b": log_b}).dropna()
    spread   = combined["a"] - combined["b"]

    n_pts  = len(spread)
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
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; font-size: 16px; }}

  /* TOPBAR */
  .topbar {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 20px 40px; display: flex; align-items: center;
    justify-content: space-between; position: sticky; top: 0; z-index: 100;
  }}
  .topbar h1 {{ font-size: 24px; font-weight: 800; letter-spacing: 0.04em; color: white; }}
  .topbar a {{
    color: var(--cyan); text-decoration: none; font-size: 15px; font-weight: 600;
    border: 1px solid var(--cyan); padding: 9px 20px; border-radius: 5px;
    transition: background 0.2s;
  }}
  .topbar a:hover {{ background: rgba(56,189,248,0.1); }}

  /* STATS BAR */
  .stats-bar {{
    background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 13px 40px; font-family: var(--mono); font-size: 14px;
    color: var(--muted); display: flex; gap: 40px; flex-wrap: wrap;
  }}
  .stats-bar span {{ color: var(--text); font-weight: 600; }}

  /* SEARCH */
  .search-bar {{
    padding: 22px 40px; background: var(--surface); border-bottom: 1px solid var(--border);
  }}
  .search-bar input {{
    background: var(--surface2); border: 1px solid var(--border); color: white;
    padding: 13px 20px; border-radius: 7px; font-family: var(--mono);
    font-size: 15px; width: 400px; outline: none; transition: border 0.2s;
  }}
  .search-bar input:focus {{ border-color: var(--cyan); }}
  .search-bar input::placeholder {{ color: var(--muted); }}

  /* COLUMNS */
  .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
  .column {{ padding: 30px 40px; border-right: 1px solid var(--border); }}
  .column:last-child {{ border-right: none; }}

  .col-header {{
    font-size: 14px; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 26px; padding-bottom: 16px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px;
  }}
  .col-header .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}

  /* SECTOR */
  .sector-block {{ margin-bottom: 34px; }}
  .sector-header {{
    font-size: 16px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 14px; padding: 10px 16px; border-radius: 6px; border-left: 4px solid;
  }}
  .sector-header[style*="38bdf8"] {{ border-left-color: #38bdf8; background: rgba(56,189,248,0.06); }}
  .sector-header[style*="f59e0b"] {{ border-left-color: #f59e0b; background: rgba(245,158,11,0.06); }}

  /* INDUSTRY */
  .industry-block {{ margin: 16px 0 16px 18px; }}
  .industry-label {{
    font-size: 13px; font-weight: 700; color: #6b7f9a; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}

  /* SUBINDUSTRY */
  .subindustry-block {{ margin: 12px 0 12px 16px; }}
  .subindustry-label {{
    font-size: 12px; color: #4a5e72; font-family: var(--mono);
    margin-bottom: 10px; letter-spacing: 0.04em;
  }}

  /* TICKER CARDS */
  .ticker-grid {{ display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 12px; }}
  .ticker-card {{
    background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
    padding: 9px 14px; display: flex; flex-direction: column; min-width: 160px; max-width: 240px;
    transition: border-color 0.15s, background 0.15s; cursor: default;
  }}
  .ticker-card:hover {{ border-color: #334155; background: #1e2535; }}
  .ticker-sym  {{ font-family: var(--mono); font-size: 15px; font-weight: 700; line-height: 1.2; }}
  .ticker-name {{
    font-size: 12px; color: #94a3b8; line-height: 1.4; margin-top: 4px;
    white-space: normal; overflow: visible;
  }}

  /* FLAT ALPHA LAYOUT (used when no sector data) */
  .flat-grid {{ padding: 0; }}
  .alpha-block {{ margin-bottom: 28px; }}
  .alpha-label {{
    font-family: var(--mono); font-size: 13px; font-weight: 700;
    color: #3a4f66; letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 10px; padding-bottom: 6px;
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
def _init_analyze_worker(cl, cs, lp, lp_short, lp_long, pr, pf, tt, elt):
    """Set shared read-only data as globals in each worker process."""
    global corr_long, corr_short, log_prices, log_prices_short, log_prices_long
    global prices_raw, perf, TICKER_TYPES, ETF_LEV_TYPES
    corr_long        = cl
    corr_short       = cs
    log_prices       = lp
    log_prices_short = lp_short
    log_prices_long  = lp_long
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
        r["ZDates"]   = dates
        r["ZHistory"] = z_vals
        pdates, pa, pb = compute_price_history(a, b, src)
        r["PriceDates"] = pdates
        r["PriceA"]     = pa
        r["PriceB"]     = pb
    except Exception:
        r["ZDates"]     = []
        r["ZHistory"]   = []
        r["PriceDates"] = []
        r["PriceA"]     = []
        r["PriceB"]     = []

    return r


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
    print(f"Computing matrices for {len(valid)} symbols...")

    # Pre-compute number of pairs (used in HTML template regardless of cache)
    n_combos = len(valid) * (len(valid) - 1) // 2

    print(f"--- Computing matrices and analyzing {len(valid)} symbols... ---")
    returns    = data.pct_change().dropna(how="all")
    log_prices       = np.log(data.tail(Z_LENGTH))
    log_prices_short = np.log(data.tail(Z_LENGTH_SHORT))
    log_prices_long  = np.log(data.tail(Z_LENGTH_LONG))
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
                  log_prices_long, prices_raw, perf,
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

    # The code below runs EVERY time, regardless of whether calculations were cached
    top_results = results[:500]

    # Compute rolling Z-score histories for top pairs (parallel)
    print(f"Computing Z-score chart histories for top pairs using {NUM_WORKERS} CPU cores...")
    chart_chunksize = max(1, len(top_results) // (NUM_WORKERS * 2))
    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_init_chart_worker,
        initargs=(chart_data, data)
    ) as pool:
        top_results = list(tqdm(
            pool.imap(_compute_chart_for_pair, top_results, chunksize=chart_chunksize),
            total=len(top_results),
            desc="Computing Charts"
        ))

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
    for i, r in enumerate(top_results):
        z = r["Z"]
        a, b = r["Pair"].split("/")

        if any(np.isnan(v) for v in [z, r["Score"], r["Corr"], r["PerfDiff"]]):
            continue

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
              <span class="z-value {'z-pos' if z_pos else 'z-neg'}">{z:+.2f}&sigma;</span>
              <div class="z-bar-track">
                <div class="z-bar-fill {'z-bar-pos' if z_pos else 'z-bar-neg'}" style="width:{z_bar_pct}%;"></div>
              </div>
              <div class="z-sub-row">
                <span class="z-sub" title="30-day Z-score">30d:{f'{z30:+.1f}' if z30 is not None else '\u2014'}</span>
                <span class="z-sub" title="250-day Z-score">250d:{f'{z250:+.1f}' if z250 is not None else '\u2014'}</span>
                <span class="align-badge {align_class}">{align_label}</span>
                <span class="conf-badge {conf_class}">{confidence}</span>
              </div>
            </div>
          </td>
          <td class="corr-cell">
            <span class="corr-value">{r['Corr']:.2f}</span>
            <span class="corr-brk">&Delta;{r['CorrBrk']:+.3f}</span>
          </td>
          <td class="hl-cell">
            {f'<span class="hl-value">{hl:.0f}d</span>' if hl else '<span class="hl-na">—</span>'}
          </td>
          <td class="est-cell">
            <span class="est-ret">{est_ret:+.1f}%</span>
            {f'<span class="ann-ret">{ann_ret:+.0f}%/yr</span>' if ann_ret is not None else ''}
          </td>
          <td class="perf-cell">
            <span class="{'perf-pos' if r['PerfDiff'] >= 0 else 'perf-neg'}">{r['PerfDiff']:+.1f}%</span>
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
            <button class="chart-btn" onclick="openChart(this,'z')" data-chart='{chart_payload_esc}'>&#9657; Z-Chart</button>
            <button class="chart-btn price-btn" onclick="openChart(this,'price')" data-chart='{chart_payload_esc}'>&#9724; Price</button>
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
        padding: 12px 32px; display: flex; flex-wrap: wrap;
        overflow: visible;
        position: relative; z-index: 50;
      }}
      .stat-item {{
        padding: 6px 28px 6px 0; margin-right: 28px;
        border-right: 1px solid var(--border); white-space: nowrap; flex-shrink: 0;
      }}
      .stat-item:last-child {{ border-right: none; }}
      .stat-label {{ font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }}
      .stat-value {{ font-family: var(--mono); font-size: 18px; font-weight: 600; color: white; }}
      .stat-value.cyan {{ color: var(--cyan); }}
      .stat-value.green {{ color: var(--green); }}
      .stat-value.amber {{ color: var(--amber); }}

      /* CONTROLS */
      .controls {{
        background: var(--surface2); border-bottom: 1px solid var(--border);
        padding: 9px 20px; display: flex; gap: 7px; align-items: center; flex-wrap: wrap;
      }}
      .control-group {{
        display: flex; align-items: center; gap: 5px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 6px; padding: 5px 9px;
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
      .corr-brk   {{ font-family: var(--mono); font-size: 10px; color: #cbd5e1; }}
      .perf-cell  {{ min-width: 75px; }}
      .perf-pos {{ font-family: var(--mono); font-size: 13px; color: var(--green); font-weight: 500; }}
      .perf-neg {{ font-family: var(--mono); font-size: 13px; color: var(--red);   font-weight: 500; }}

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
        <span>Setups: <em>{len(results):,}</em></span>
        <span>Showing: <em>Top {len(top_results)}</em></span>
      </div>
      <div><a href="symbols.html" class="nav-link">Symbol Reference &#8594;</a></div>
    </div>

    <!-- STATS ROW -->
    <div class="stats-row">
      <div class="stat-item"><div class="stat-label">Pairs Scanned</div><div class="stat-value cyan">{n_combos:,}</div></div>
      <div class="stat-item"><div class="stat-label">Valid Setups</div><div class="stat-value green">{len(results):,}</div></div>
      <div class="stat-item"><div class="stat-label">Active Symbols</div><div class="stat-value">{len(valid)}</div></div>
      <div class="stat-item"><div class="stat-label">Z Threshold</div><div class="stat-value">&plusmn;{Z_THRESHOLD:.1f}&sigma;</div></div>
      <div class="stat-item"><div class="stat-label">Min Correlation</div><div class="stat-value">{MIN_CORR_FILTER:.2f}</div></div>
      <div class="stat-item"><div class="stat-label">Corr Window</div><div class="stat-value">{CORR_SHORT}d / {CORR_LONG}d</div></div>
      <div class="stat-item"><div class="stat-label">Z Window</div><div class="stat-value">{Z_LENGTH}d</div></div>
      <div class="stat-item"><div class="stat-label">Perf Window</div><div class="stat-value amber">{PERF_LENGTH}d</div></div>
      <div class="stat-item"><div class="stat-label">Aligned</div><div class="stat-value cyan">{n_aligned}</div></div>
      <div class="stat-item"><div class="stat-label">Mixed</div><div class="stat-value amber">{n_mixed}</div></div>
      <div class="stat-item"><div class="stat-label">Conflicting</div><div class="stat-value" style="color:var(--red)">{n_conflicting}</div></div>
      <div class="stat-item"><div class="stat-label">High Conf</div><div class="stat-value green">{n_conf_high}</div></div>
      <div class="stat-item"><div class="stat-label">Med Conf</div><div class="stat-value amber">{n_conf_med}</div></div>
      <div class="stat-item"><div class="stat-label">Low Conf</div><div class="stat-value" style="color:var(--red)">{n_conf_low}</div></div>
    </div>

    <!-- CONTROLS -->
    <div class="controls">
      <div class="control-group">
        <label>Capital ($)</label>
        <button class="step-btn" onclick="stepValue('capitalInput',-1000)">−</button>
        <input type="number" id="capitalInput" value="5000" min="0" step="1000" oninput="calcShares()">
        <button class="step-btn" onclick="stepValue('capitalInput',1000)">+</button>
      </div>
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
          <option value="exclude_both">Excl Lev &amp; Inv</option>
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
          <option value="100000000">&gt; 100M</option>
          <option value="500000000">&gt; 500M</option>
          <option value="1000000000">&gt; 1B</option>
          <option value="5000000000">&gt; 5B</option>
          <option value="10000000000">&gt; 10B</option>
          <option value="50000000000">&gt; 50B</option>
        </select>
      </div>
      <div class="control-group">
        <label>Sort</label>
        <select id="sortBy" onchange="sortTable()">
          <option value="score">Score</option>
          <option value="z_abs">|Z-Score|</option>
          <option value="hl">Half-Life</option>
          <option value="est_ret">Est Return</option>
          <option value="ann_ret">Ann Return</option>
          <option value="corr">Correlation</option>
          <option value="perf">Perf Diff</option>
          <option value="alignment">Alignment</option>
          <option value="confidence">Confidence</option>
        </select>
      </div>
      <div class="control-group" title="When on, each symbol can appear at most once — only the highest-scored pair for that symbol is shown">
        <label>Unique Syms</label>
        <label class="toggle-switch">
          <input type="checkbox" id="uniqueSymFilter" onchange="applyFilters()">
          <span class="toggle-track"><span class="toggle-thumb"></span></span>
        </label>
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
      <th onclick="setSort('corr')">Corr / &Delta; &#8597;</th>
      <th onclick="setSort('hl')">Half-Life &#8597;</th>
      <th onclick="setSort('est_ret')" style="text-align:right;">Est Return &#8597;</th>
      <th onclick="setSort('perf')">Perf Diff &#8597;</th>
      <th onclick="setSort('score')">Score &#8597;</th>
      <th>Signal/Shares</th>
      <th style="text-align:center;">Charts</th>
    </tr>
    </thead>
    <tbody id="tableBody">
    {rows_html}
    </tbody>
    </table>
    </div>

    <!-- FOOTER -->
    <div class="footer">
      <div>
        <span class="leg-dot" style="background:var(--red)"></span>Short A / Long B &nbsp;
        <span class="leg-dot" style="background:var(--green)"></span>Long A / Short B &nbsp;
        <span class="leg-dot" style="background:var(--muted)"></span>Neutral
      </div>
      <div>
        Score = {int(W_ZSCORE*100)}% |Z| + {int(W_CORR_BRK*100)}% Corr Break + {int(W_REL_PERF*100)}% Rel Perf
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

    // Load annotation plugin async
    (function() {{
      const s = document.createElement("script");
      s.src = "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js";
      s.onload = () => {{ Chart.register(window["chartjs-plugin-annotation"]); }};
      document.head.appendChild(s);
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
        ctx.strokeStyle = "rgba(148,163,184,0.4)";
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.restore();
      }},
    }};
    Chart.register(crosshairPlugin);

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
      currentChartData = p;
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
      const hlStr  = p.halfLife != null ? Math.round(p.halfLife) + "d" : "—";
      const estStr = p.estRet  != null ? (p.estRet  >= 0 ? "+" : "") + p.estRet.toFixed(1)  + "%" : "—";
      const annStr = p.annRet  != null ? (p.annRet  >= 0 ? "+" : "") + p.annRet.toFixed(0)  + "%/yr" : "—";
      document.getElementById("modalStats").innerHTML = `
        <div class="mstat">
          <div class="mstat-label">Current Z</div>
          <div class="mstat-value" style="color:${{zColor}};">${{p.currentZ >= 0 ? "+" : ""}}${{p.currentZ.toFixed(2)}}&sigma;</div>
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
        `<span>Z window: <em>${{p.zWindow}} days</em></span>` +
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
        buildZChart(p.dates, p.z, p.zWindow);
        if (mode === 'price') switchTab('price');
      }}, 40);
    }}

    function buildZChart(dates, z, zWindow) {{
      if (activeChart) {{ activeChart.destroy(); activeChart = null; }}
      const ctx = document.getElementById("zChart").getContext("2d");

      const grad = ctx.createLinearGradient(0, 0, 0, 380);
      grad.addColorStop(0,   "rgba(56,189,248,0.20)");
      grad.addColorStop(0.45,"rgba(56,189,248,0.06)");
      grad.addColorStop(1,   "rgba(56,189,248,0.00)");

      const ptColors = z.map(v => {{
        if (v === null) return "transparent";
        const av = Math.abs(v);
        if (av >= 3) return "#ef4444";
        if (av >= 2) return "#f59e0b";
        if (av >= 1) return "#38bdf8";
        return "rgba(148,163,184,0.5)";
      }});

      const hLine = (y, color, width, dash, lbl) => ({{
        type: "line", yMin: y, yMax: y,
        borderColor: color, borderWidth: width, borderDash: dash,
        label: {{ display: !!lbl, content: lbl, color, position: "end",
                  font: {{ size: 10, family: "'JetBrains Mono',monospace", weight: "600" }},
                  xAdjust: -10, yAdjust: y > 0 ? -10 : 8, backgroundColor: "transparent", borderWidth: 0 }},
      }});

      activeChart = new Chart(ctx, {{
        type: "line",
        data: {{
          labels: dates,
          datasets: [{{
            label: "Z-Score",
            data: z,
            borderColor: "#38bdf8",
            borderWidth: 1.8,
            pointRadius: 0,
            pointHoverRadius: 0,
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
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: "#0d1520", borderColor: "#242d40", borderWidth: 1,
              titleColor: "#64748b", bodyColor: "#e2e8f0",
              titleFont: {{ family: "'JetBrains Mono',monospace", size: 11 }},
              bodyFont:  {{ family: "'JetBrains Mono',monospace", size: 14 }},
              padding: 14, caretSize: 5, caretPadding: 20,
              usePointStyle: false, displayColors: false,
              callbacks: {{
                label: c => {{
                  const v = c.raw;
                  if (v === null) return " Z = \u2014";
                  const lv = Math.abs(v) >= 3 ? "EXTREME" : Math.abs(v) >= 2 ? "STRONG" : Math.abs(v) >= 1 ? "SIGNAL" : "neutral";
                  return ` Z = ${{v >= 0 ? "+" : ""}}${{v.toFixed(3)}}\u03C3   [${{lv}}]`;
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
          }},
          scales: {{
            x: {{
              ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }}, maxRotation: 0, maxTicksLimit: 10, autoSkip: true }},
              grid: {{ color: "rgba(28,35,51,0.7)" }}, border: {{ color: "#1c2333" }},
            }},
            y: {{
              ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }},
                callback: v => (v >= 0 ? "+" : "") + v.toFixed(1) + "\u03C3" }},
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
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: "#0d1520", borderColor: "#242d40", borderWidth: 1,
              titleColor: "#64748b", bodyColor: "#e2e8f0",
              titleFont: {{ family: "'JetBrains Mono',monospace", size: 11 }},
              bodyFont:  {{ family: "'JetBrains Mono',monospace", size: 13 }},
              padding: 14, caretSize: 5, caretPadding: 20,
              usePointStyle: false,
              callbacks: {{
                label: c => {{
                  const pct = (c.raw - 100).toFixed(2);
                  return ` ${{c.dataset.label}}: ${{c.raw.toFixed(2)}}  (${{pct >= 0 ? "+" : ""}}${{pct}}%)`;
                }},
                labelColor: c => ({{ borderColor: c.dataset.borderColor, backgroundColor: c.dataset.borderColor }}),
              }},
            }},
            annotation: {{
              annotations: {{
                baseline: {{ type: "line", yMin: 100, yMax: 100,
                  borderColor: "rgba(148,163,184,0.25)", borderWidth: 1, borderDash: [4,4] }},
              }},
            }},
          }},
          scales: {{
            x: {{
              ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }}, maxRotation: 0, maxTicksLimit: 10, autoSkip: true }},
              grid: {{ color: "rgba(28,35,51,0.7)" }}, border: {{ color: "#1c2333" }},
            }},
            y: {{
              ticks: {{ color: "#374151", font: {{ family: "'JetBrains Mono',monospace", size: 10 }}, callback: v => v.toFixed(0) }},
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
          interaction:{{mode:"index",intersect:false}},
          plugins:{{
            legend:{{display:false}},
            tooltip:{{
              backgroundColor:"#0d1520", borderColor:"#242d40", borderWidth:1,
              titleColor:"#64748b", bodyColor:"#e2e8f0",
              titleFont:{{family:"'JetBrains Mono',monospace",size:11}},
              bodyFont:{{family:"'JetBrains Mono',monospace",size:13}}, padding:14, caretPadding:20,
              usePointStyle:false,
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
                labelColor: c => ({{ borderColor: c.dataset.borderColor, backgroundColor: c.dataset.borderColor }}),
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
          }},
          scales:{{
            x:{{ ticks:{{color:"#374151",font:{{family:"'JetBrains Mono',monospace",size:10}},
                  maxRotation:0,maxTicksLimit:10,autoSkip:true}},
                 grid:{{color:"rgba(28,35,51,0.7)"}},border:{{color:"#1c2333"}} }},
            yP:{{ position:"left",
                 ticks:{{color:"#38bdf8",font:{{family:"'JetBrains Mono',monospace",size:10}},callback:v=>v.toFixed(0)}},
                 grid:{{color:"rgba(28,35,51,0.5)"}},border:{{color:"#1c2333"}},
                 title:{{display:true,text:"Norm. Price (base 100)",color:"rgba(56,189,248,0.5)",
                   font:{{family:"'JetBrains Mono',monospace",size:10}}}} }},
            yZ:{{ position:"right",
                 ticks:{{color:"rgba(248,215,80,0.7)",font:{{family:"'JetBrains Mono',monospace",size:10}},
                   callback:v=>(v>=0?"+":"")+v.toFixed(1)+"\u03C3"}},
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

      calcShares();
    }}

    // ─── SORT ─────────────────────────────────────────────────────────────────────
    let currentSort = {{ key: "score", asc: false }};

    function setSort(key) {{
      currentSort.asc = (currentSort.key === key) ? !currentSort.asc : false;
      if (key === "hl" && currentSort.key !== "hl") currentSort.asc = true;
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
        else if (key === "perf")    {{ va = Math.abs(numOf(a,".perf-cell span")); vb = Math.abs(numOf(b,".perf-cell span")); }}
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
      calcShares();
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

    window.addEventListener("DOMContentLoaded", () => {{
      document.getElementById("update-time").textContent = new Date({int(time.time() * 1000)}).toLocaleString();
      calcShares();
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
    print("\nDone. Open pairs_scanner.html in your browser.")
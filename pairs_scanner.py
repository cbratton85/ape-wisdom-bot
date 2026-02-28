import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import json
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
DATA_FILE        = "historical_data.csv.gz"
CHART_DATA_FILE  = "chart_data.csv.gz"        # Extended history for Z-score charts only
VOLUME_DATA_FILE = "volume_data.csv.gz"        # Average daily volume per ticker
ANALYSIS_CACHE = "analysis_results.json"
BATCH_SIZE = 25
COOLDOWN = 1
LOOKBACK_DAYS       = 650   # Days used for scoring / correlation / perf
CHART_LOOKBACK_DAYS = 1825  # ~5 years used for Z-score chart history
VOL_AVG_DAYS        = 30    # Rolling window for average volume calculation
CACHE_UPDATE_COOLDOWN_HOURS = 1

CORR_SHORT = 35
CORR_LONG = 100
Z_LENGTH = 100
PERF_LENGTH = 300

MIN_CORR_FILTER = 0.60
Z_THRESHOLD = 1.0
Z_STRONG = 2.0

W_ZSCORE = 0.50
W_CORR_BRK = 0.25
W_REL_PERF = 0.25

# ==========================================
# LOAD MASTER TICKERS
# ==========================================
TICKER_TYPES = {}
TICKER_NAMES = {}   # maps ticker -> human-readable name

# ==========================================
# LEVERAGED / INVERSE ETF CLASSIFICATION
# Tickers added here are flagged in the UI so the user can filter them out.
# Add new tickers to the appropriate set as you expand your ETF list.
# ==========================================
LEVERAGED_ETFS = {
    # Broad market leveraged
    "SSO","UPRO","SPXL","QLD","TQQQ","DDM","UDOW","TNA","URTY",
    # Sector leveraged
    "TECL","LABU","FAS","NUGT","JNUG","NAIL","ERX","GUSH","CURE","DPST","BNKU",
    # Bond leveraged
    "TMF","TYD","UBT","UST",
    # Commodity leveraged
    "UGL","AGQ",
    # Vol leveraged
    "UVXY",
}
INVERSE_ETFS = {
    # Broad market inverse
    "SDS","SPXU","SH","PSQ","QID","SQQQ","DXD","SDOW","SPXS","TZA","SRTY",
    # Sector inverse
    "TECS","LABD","FAZ","DUST","JDST","ERY","DRIP",
    # Bond inverse
    "TMV","TYO","TBT","PST",
    # Commodity inverse
    "GLL","ZSL",
    # Vol inverse
    "SVXY","SVOL","ZIVB",
}

def load_master_tickers():
    global TICKER_TYPES, TICKER_NAMES
    tickers = []

    if os.path.exists("ETFs.csv"):
        df_etf = pd.read_csv("ETFs.csv", header=None)
        etfs = df_etf[0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers += etfs
        for t in etfs:
            TICKER_TYPES[t] = "Pure ETF"
        if df_etf.shape[1] >= 2:
            for _, row in df_etf.iterrows():
                t = str(row.iloc[0]).strip().upper()
                TICKER_NAMES[t] = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""

    if os.path.exists("STOCKS.csv"):
        df_stock = pd.read_csv("STOCKS.csv", header=None)
        stocks = df_stock[0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers += stocks
        for t in stocks:
            TICKER_TYPES[t] = "Pure Stock"
        if df_stock.shape[1] >= 2:
            for _, row in df_stock.iterrows():
                t = str(row.iloc[0]).strip().upper()
                TICKER_NAMES[t] = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""

    tickers = list(set(tickers))
    tickers = [t for t in tickers if t not in ["", "NONE", "NAN", "SYMBOL", "TICKER"]]
    print(f"Loaded {len(tickers)} tickers.")
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
                threads=True,
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
                    if len(clean) > 1:
                        if t in df.columns.levels[0]:
                            result[t] = df[t][field]
                    else:
                        if not df[field].empty:
                            result[t] = df[field]
                except:
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
    try:
        full_spread = np.log(prices_raw[a]) - np.log(prices_raw[b])
        hl = compute_half_life(full_spread)
    except Exception:
        hl = float('nan')

    # ── Annualized returns ──
    try:
        n_days = len(prices_raw)
        ann_a  = round(((prices_raw[a].iloc[-1] / prices_raw[a].iloc[0]) ** (252 / n_days) - 1) * 100, 1)
        ann_b  = round(((prices_raw[b].iloc[-1] / prices_raw[b].iloc[0]) ** (252 / n_days) - 1) * 100, 1)
    except Exception:
        ann_a = ann_b = float('nan')

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
    }


# ==========================================
# ROLLING Z-SCORE HISTORY FOR CHART
# ==========================================
def compute_z_history(a, b, price_data):
    """Rolling Z-score over all available data using Z_LENGTH-day window."""
    log_a = np.log(price_data[a].dropna())
    log_b = np.log(price_data[b].dropna())
    combined = pd.DataFrame({"a": log_a, "b": log_b}).dropna()
    spread   = combined["a"] - combined["b"]

    roll_mean = spread.rolling(Z_LENGTH).mean()
    roll_std  = spread.rolling(Z_LENGTH).std()
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

    # ── Built-in ETF category lookup ─────────────────────────────────────────
    # Provides Sector / Industry / Subindustry for well-known ETFs so that
    # even a 2-column ETFs.csv (Ticker, Name) gets a proper hierarchy.
    ETF_CATEGORIES = {
        # ── US EQUITY – BROAD MARKET ──────────────────────────────────────────
        "SPY":  ("US Equity", "Broad Market", "Large Cap Blend"),
        "IVV":  ("US Equity", "Broad Market", "Large Cap Blend"),
        "VOO":  ("US Equity", "Broad Market", "Large Cap Blend"),
        "VTI":  ("US Equity", "Broad Market", "Total Market"),
        "ITOT": ("US Equity", "Broad Market", "Total Market"),
        "SCHB": ("US Equity", "Broad Market", "Total Market"),
        "IWB":  ("US Equity", "Broad Market", "Large Cap Blend"),
        "RSP":  ("US Equity", "Broad Market", "Equal Weight"),
        "OEF":  ("US Equity", "Broad Market", "Large Cap 100"),
        "FXAIX":("US Equity", "Broad Market", "Large Cap Blend"),
        # Large Cap Growth / Value
        "IVW":  ("US Equity", "Broad Market", "Large Cap Growth"),
        "VUG":  ("US Equity", "Broad Market", "Large Cap Growth"),
        "VOOG": ("US Equity", "Broad Market", "Large Cap Growth"),
        "SCHG": ("US Equity", "Broad Market", "Large Cap Growth"),
        "QQQ":  ("US Equity", "Broad Market", "Large Cap Growth"),
        "QQQM": ("US Equity", "Broad Market", "Large Cap Growth"),
        "IWF":  ("US Equity", "Broad Market", "Large Cap Growth"),
        "IVE":  ("US Equity", "Broad Market", "Large Cap Value"),
        "VTV":  ("US Equity", "Broad Market", "Large Cap Value"),
        "VOOV": ("US Equity", "Broad Market", "Large Cap Value"),
        "SCHV": ("US Equity", "Broad Market", "Large Cap Value"),
        "IWD":  ("US Equity", "Broad Market", "Large Cap Value"),
        # Mid Cap
        "IJH":  ("US Equity", "Broad Market", "Mid Cap Blend"),
        "VO":   ("US Equity", "Broad Market", "Mid Cap Blend"),
        "MDY":  ("US Equity", "Broad Market", "Mid Cap Blend"),
        "IJJ":  ("US Equity", "Broad Market", "Mid Cap Value"),
        "IJK":  ("US Equity", "Broad Market", "Mid Cap Growth"),
        "VOE":  ("US Equity", "Broad Market", "Mid Cap Value"),
        "VOT":  ("US Equity", "Broad Market", "Mid Cap Growth"),
        "IWR":  ("US Equity", "Broad Market", "Mid Cap Blend"),
        # Small Cap
        "IJR":  ("US Equity", "Broad Market", "Small Cap Blend"),
        "VB":   ("US Equity", "Broad Market", "Small Cap Blend"),
        "IWM":  ("US Equity", "Broad Market", "Small Cap Blend"),
        "SLY":  ("US Equity", "Broad Market", "Small Cap Blend"),
        "IJS":  ("US Equity", "Broad Market", "Small Cap Value"),
        "VBR":  ("US Equity", "Broad Market", "Small Cap Value"),
        "IWN":  ("US Equity", "Broad Market", "Small Cap Value"),
        "IJT":  ("US Equity", "Broad Market", "Small Cap Growth"),
        "VBK":  ("US Equity", "Broad Market", "Small Cap Growth"),
        "IWO":  ("US Equity", "Broad Market", "Small Cap Growth"),
        # Dividend / Income
        "VYM":  ("US Equity", "Dividend & Income", "High Dividend"),
        "DVY":  ("US Equity", "Dividend & Income", "High Dividend"),
        "HDV":  ("US Equity", "Dividend & Income", "High Dividend"),
        "SCHD": ("US Equity", "Dividend & Income", "Dividend Growth"),
        "DGRO": ("US Equity", "Dividend & Income", "Dividend Growth"),
        "SDY":  ("US Equity", "Dividend & Income", "Dividend Growth"),
        "VIG":  ("US Equity", "Dividend & Income", "Dividend Growth"),
        "NOBL": ("US Equity", "Dividend & Income", "Dividend Aristocrats"),
        "SPYD": ("US Equity", "Dividend & Income", "High Dividend"),
        # Factor / Smart Beta
        "MTUM": ("US Equity", "Factor / Smart Beta", "Momentum"),
        "QUAL": ("US Equity", "Factor / Smart Beta", "Quality"),
        "VLUE": ("US Equity", "Factor / Smart Beta", "Value"),
        "USMV": ("US Equity", "Factor / Smart Beta", "Min Volatility"),
        "SIZE": ("US Equity", "Factor / Smart Beta", "Size"),
        "EFAV": ("US Equity", "Factor / Smart Beta", "Min Volatility"),
        "SPLV": ("US Equity", "Factor / Smart Beta", "Min Volatility"),
        "SPHQ": ("US Equity", "Factor / Smart Beta", "Quality"),
        "SPHB": ("US Equity", "Factor / Smart Beta", "High Beta"),
        "FDVV": ("US Equity", "Dividend & Income", "Dividend Growth"),

        # ── US EQUITY – SECTORS ───────────────────────────────────────────────
        # Technology
        "XLK":  ("US Equity", "Sector – Technology", "Broad Tech"),
        "VGT":  ("US Equity", "Sector – Technology", "Broad Tech"),
        "IYW":  ("US Equity", "Sector – Technology", "Broad Tech"),
        "FTEC": ("US Equity", "Sector – Technology", "Broad Tech"),
        "IGV":  ("US Equity", "Sector – Technology", "Software"),
        "WCLD": ("US Equity", "Sector – Technology", "Cloud / Software"),
        "BUG":  ("US Equity", "Sector – Technology", "Cybersecurity"),
        "CIBR": ("US Equity", "Sector – Technology", "Cybersecurity"),
        "HACK": ("US Equity", "Sector – Technology", "Cybersecurity"),
        "SOXX": ("US Equity", "Sector – Technology", "Semiconductors"),
        "SMH":  ("US Equity", "Sector – Technology", "Semiconductors"),
        "USD":  ("US Equity", "Sector – Technology", "Semiconductors"),
        "QTUM": ("US Equity", "Sector – Technology", "Quantum / AI"),
        "BOTZ": ("US Equity", "Sector – Technology", "Robotics / AI"),
        "ROBT": ("US Equity", "Sector – Technology", "Robotics / AI"),
        "IRBO": ("US Equity", "Sector – Technology", "Robotics / AI"),
        "ARKK": ("US Equity", "Sector – Technology", "Disruptive Innovation"),
        "ARKG": ("US Equity", "Sector – Technology", "Disruptive Innovation"),
        "ARKW": ("US Equity", "Sector – Technology", "Disruptive Innovation"),
        "ARKQ": ("US Equity", "Sector – Technology", "Robotics / AI"),
        "ARKF": ("US Equity", "Sector – Technology", "Fintech"),
        "FINX": ("US Equity", "Sector – Technology", "Fintech"),
        "IPAY": ("US Equity", "Sector – Technology", "Fintech"),
        "SKYY": ("US Equity", "Sector – Technology", "Cloud / Software"),
        # Healthcare
        "XLV":  ("US Equity", "Sector – Healthcare", "Broad Healthcare"),
        "VHT":  ("US Equity", "Sector – Healthcare", "Broad Healthcare"),
        "IYH":  ("US Equity", "Sector – Healthcare", "Broad Healthcare"),
        "FHLC": ("US Equity", "Sector – Healthcare", "Broad Healthcare"),
        "IBB":  ("US Equity", "Sector – Healthcare", "Biotech"),
        "XBI":  ("US Equity", "Sector – Healthcare", "Biotech"),
        "BBH":  ("US Equity", "Sector – Healthcare", "Biotech"),
        "IHI":  ("US Equity", "Sector – Healthcare", "Medical Devices"),
        "IHF":  ("US Equity", "Sector – Healthcare", "Managed Care / Insurance"),
        "PPH":  ("US Equity", "Sector – Healthcare", "Pharmaceuticals"),
        "PJP":  ("US Equity", "Sector – Healthcare", "Pharmaceuticals"),
        # Financials
        "XLF":  ("US Equity", "Sector – Financials", "Broad Financials"),
        "VFH":  ("US Equity", "Sector – Financials", "Broad Financials"),
        "IYF":  ("US Equity", "Sector – Financials", "Broad Financials"),
        "FNCL": ("US Equity", "Sector – Financials", "Broad Financials"),
        "KBE":  ("US Equity", "Sector – Financials", "Banks"),
        "KRE":  ("US Equity", "Sector – Financials", "Regional Banks"),
        "IAT":  ("US Equity", "Sector – Financials", "Regional Banks"),
        "KIE":  ("US Equity", "Sector – Financials", "Insurance"),
        "IAK":  ("US Equity", "Sector – Financials", "Insurance"),
        "KBWB": ("US Equity", "Sector – Financials", "Banks"),
        # Energy
        "XLE":  ("US Equity", "Sector – Energy", "Broad Energy"),
        "VDE":  ("US Equity", "Sector – Energy", "Broad Energy"),
        "IYE":  ("US Equity", "Sector – Energy", "Broad Energy"),
        "FENY": ("US Equity", "Sector – Energy", "Broad Energy"),
        "OIH":  ("US Equity", "Sector – Energy", "Oil Services"),
        "XES":  ("US Equity", "Sector – Energy", "Oil Services"),
        "FCG":  ("US Equity", "Sector – Energy", "Natural Gas"),
        "ICLN": ("US Equity", "Sector – Energy", "Clean Energy"),
        "QCLN": ("US Equity", "Sector – Energy", "Clean Energy"),
        "FAN":  ("US Equity", "Sector – Energy", "Wind"),
        "TAN":  ("US Equity", "Sector – Energy", "Solar"),
        "CNRG": ("US Equity", "Sector – Energy", "Clean Energy"),
        # Consumer Discretionary
        "XLY":  ("US Equity", "Sector – Consumer Disc.", "Broad Consumer Disc."),
        "VCR":  ("US Equity", "Sector – Consumer Disc.", "Broad Consumer Disc."),
        "IYC":  ("US Equity", "Sector – Consumer Disc.", "Broad Consumer Disc."),
        "FDIS": ("US Equity", "Sector – Consumer Disc.", "Broad Consumer Disc."),
        "RTH":  ("US Equity", "Sector – Consumer Disc.", "Retail"),
        "XRT":  ("US Equity", "Sector – Consumer Disc.", "Retail"),
        "ONLN": ("US Equity", "Sector – Consumer Disc.", "E-Commerce"),
        "IBUY": ("US Equity", "Sector – Consumer Disc.", "E-Commerce"),
        # Consumer Staples
        "XLP":  ("US Equity", "Sector – Consumer Staples", "Broad Staples"),
        "VDC":  ("US Equity", "Sector – Consumer Staples", "Broad Staples"),
        "IYK":  ("US Equity", "Sector – Consumer Staples", "Broad Staples"),
        "FSTA": ("US Equity", "Sector – Consumer Staples", "Broad Staples"),
        "PBJ":  ("US Equity", "Sector – Consumer Staples", "Food & Beverage"),
        # Industrials
        "XLI":  ("US Equity", "Sector – Industrials", "Broad Industrials"),
        "VIS":  ("US Equity", "Sector – Industrials", "Broad Industrials"),
        "IYJ":  ("US Equity", "Sector – Industrials", "Broad Industrials"),
        "FIDU": ("US Equity", "Sector – Industrials", "Broad Industrials"),
        "ITA":  ("US Equity", "Sector – Industrials", "Aerospace & Defense"),
        "PPA":  ("US Equity", "Sector – Industrials", "Aerospace & Defense"),
        "XAR":  ("US Equity", "Sector – Industrials", "Aerospace & Defense"),
        "JETS": ("US Equity", "Sector – Industrials", "Airlines"),
        "XTN":  ("US Equity", "Sector – Industrials", "Transportation"),
        # Materials
        "XLB":  ("US Equity", "Sector – Materials", "Broad Materials"),
        "VAW":  ("US Equity", "Sector – Materials", "Broad Materials"),
        "IYM":  ("US Equity", "Sector – Materials", "Broad Materials"),
        "FMAT": ("US Equity", "Sector – Materials", "Broad Materials"),
        # Utilities
        "XLU":  ("US Equity", "Sector – Utilities", "Broad Utilities"),
        "VPU":  ("US Equity", "Sector – Utilities", "Broad Utilities"),
        "IDU":  ("US Equity", "Sector – Utilities", "Broad Utilities"),
        "FUTY": ("US Equity", "Sector – Utilities", "Broad Utilities"),
        # Real Estate
        "XLRE": ("US Equity", "Sector – Real Estate", "Broad REIT"),
        "VNQ":  ("US Equity", "Sector – Real Estate", "Broad REIT"),
        "IYR":  ("US Equity", "Sector – Real Estate", "Broad REIT"),
        "FREL": ("US Equity", "Sector – Real Estate", "Broad REIT"),
        "REM":  ("US Equity", "Sector – Real Estate", "Mortgage REIT"),
        "MORT": ("US Equity", "Sector – Real Estate", "Mortgage REIT"),
        "KBWR": ("US Equity", "Sector – Real Estate", "Regional Banks"),
        # Communication Services
        "XLC":  ("US Equity", "Sector – Communication", "Broad Communication"),
        "VOX":  ("US Equity", "Sector – Communication", "Broad Communication"),
        "IYZ":  ("US Equity", "Sector – Communication", "Telecom"),
        "FCOM": ("US Equity", "Sector – Communication", "Broad Communication"),

        # ── INTERNATIONAL EQUITY ─────────────────────────────────────────────
        "EFA":  ("Intl Equity", "Developed Markets", "Broad Developed ex-US"),
        "VEA":  ("Intl Equity", "Developed Markets", "Broad Developed ex-US"),
        "SCHF": ("Intl Equity", "Developed Markets", "Broad Developed ex-US"),
        "IDEV": ("Intl Equity", "Developed Markets", "Broad Developed ex-US"),
        "VEU":  ("Intl Equity", "Developed Markets", "All-World ex-US"),
        "VXUS": ("Intl Equity", "Developed Markets", "All-World ex-US"),
        "IXUS": ("Intl Equity", "Developed Markets", "All-World ex-US"),
        "EWJ":  ("Intl Equity", "Developed Markets", "Japan"),
        "DXJ":  ("Intl Equity", "Developed Markets", "Japan"),
        "EWG":  ("Intl Equity", "Developed Markets", "Germany"),
        "EWU":  ("Intl Equity", "Developed Markets", "United Kingdom"),
        "EWC":  ("Intl Equity", "Developed Markets", "Canada"),
        "EWA":  ("Intl Equity", "Developed Markets", "Australia"),
        "EWL":  ("Intl Equity", "Developed Markets", "Switzerland"),
        "EWQ":  ("Intl Equity", "Developed Markets", "France"),
        "EWI":  ("Intl Equity", "Developed Markets", "Italy"),
        "EWP":  ("Intl Equity", "Developed Markets", "Spain"),
        "EWD":  ("Intl Equity", "Developed Markets", "Sweden"),
        "EWN":  ("Intl Equity", "Developed Markets", "Netherlands"),
        "EWH":  ("Intl Equity", "Developed Markets", "Hong Kong"),
        "EWS":  ("Intl Equity", "Developed Markets", "Singapore"),
        "EWK":  ("Intl Equity", "Developed Markets", "Belgium"),
        "EWO":  ("Intl Equity", "Developed Markets", "Austria"),
        # Emerging Markets
        "EEM":  ("Intl Equity", "Emerging Markets", "Broad EM"),
        "VWO":  ("Intl Equity", "Emerging Markets", "Broad EM"),
        "IEMG": ("Intl Equity", "Emerging Markets", "Broad EM"),
        "SCHE": ("Intl Equity", "Emerging Markets", "Broad EM"),
        "EWZ":  ("Intl Equity", "Emerging Markets", "Brazil"),
        "EWW":  ("Intl Equity", "Emerging Markets", "Mexico"),
        "EWT":  ("Intl Equity", "Emerging Markets", "Taiwan"),
        "EWY":  ("Intl Equity", "Emerging Markets", "South Korea"),
        "MCHI": ("Intl Equity", "Emerging Markets", "China"),
        "FXI":  ("Intl Equity", "Emerging Markets", "China Large Cap"),
        "KWEB": ("Intl Equity", "Emerging Markets", "China Internet"),
        "CQQQ": ("Intl Equity", "Emerging Markets", "China Tech"),
        "INDA": ("Intl Equity", "Emerging Markets", "India"),
        "PIN":  ("Intl Equity", "Emerging Markets", "India"),
        "EPI":  ("Intl Equity", "Emerging Markets", "India"),
        "GXC":  ("Intl Equity", "Emerging Markets", "China Broad"),
        "ERUS": ("Intl Equity", "Emerging Markets", "Russia"),
        "RSX":  ("Intl Equity", "Emerging Markets", "Russia"),
        "TUR":  ("Intl Equity", "Emerging Markets", "Turkey"),
        "ECH":  ("Intl Equity", "Emerging Markets", "Chile"),
        "EWX":  ("Intl Equity", "Emerging Markets", "EM Small Cap"),
        # Intl thematic
        "ACWI": ("Intl Equity", "Developed Markets", "All-Country World"),
        "VT":   ("Intl Equity", "Developed Markets", "All-Country World"),

        # ── FIXED INCOME ─────────────────────────────────────────────────────
        "AGG":  ("Fixed Income", "US Aggregate", "Broad Bond Market"),
        "BND":  ("Fixed Income", "US Aggregate", "Broad Bond Market"),
        "SCHZ": ("Fixed Income", "US Aggregate", "Broad Bond Market"),
        "IUSB": ("Fixed Income", "US Aggregate", "Broad Bond Market"),
        "FBND": ("Fixed Income", "US Aggregate", "Broad Bond Market"),
        # Treasury
        "IEF":  ("Fixed Income", "US Treasuries", "7-10 Year"),
        "TLT":  ("Fixed Income", "US Treasuries", "20+ Year"),
        "VGLT": ("Fixed Income", "US Treasuries", "20+ Year"),
        "TLH":  ("Fixed Income", "US Treasuries", "10-20 Year"),
        "IEI":  ("Fixed Income", "US Treasuries", "3-7 Year"),
        "SHY":  ("Fixed Income", "US Treasuries", "1-3 Year"),
        "VGSH": ("Fixed Income", "US Treasuries", "1-3 Year"),
        "VGIT": ("Fixed Income", "US Treasuries", "3-10 Year"),
        "SCHO": ("Fixed Income", "US Treasuries", "1-3 Year"),
        "SCHR": ("Fixed Income", "US Treasuries", "3-10 Year"),
        "SCHQ": ("Fixed Income", "US Treasuries", "20+ Year"),
        "GOVT": ("Fixed Income", "US Treasuries", "Broad Treasury"),
        "EDV":  ("Fixed Income", "US Treasuries", "Extended Duration"),
        "ZROZ": ("Fixed Income", "US Treasuries", "Zero Coupon"),
        "BIL":  ("Fixed Income", "US Treasuries", "1-3 Month T-Bill"),
        "SHV":  ("Fixed Income", "US Treasuries", "Short-Term"),
        "CLTL": ("Fixed Income", "US Treasuries", "Short-Term"),
        # TIPS
        "TIP":  ("Fixed Income", "Inflation-Protected", "TIPS Broad"),
        "SCHP": ("Fixed Income", "Inflation-Protected", "TIPS Broad"),
        "STIP": ("Fixed Income", "Inflation-Protected", "Short TIPS"),
        "VTIP": ("Fixed Income", "Inflation-Protected", "Short TIPS"),
        # Corporate
        "LQD":  ("Fixed Income", "Corporate Bonds", "Investment Grade"),
        "VCIT": ("Fixed Income", "Corporate Bonds", "Intermediate IG"),
        "VCSH": ("Fixed Income", "Corporate Bonds", "Short-Term IG"),
        "SPSB": ("Fixed Income", "Corporate Bonds", "Short-Term IG"),
        "SPIB": ("Fixed Income", "Corporate Bonds", "Intermediate IG"),
        "SPLB": ("Fixed Income", "Corporate Bonds", "Long-Term IG"),
        "IGIB": ("Fixed Income", "Corporate Bonds", "Intermediate IG"),
        "IGSB": ("Fixed Income", "Corporate Bonds", "Short-Term IG"),
        "IGLB": ("Fixed Income", "Corporate Bonds", "Long-Term IG"),
        "VCLT": ("Fixed Income", "Corporate Bonds", "Long-Term IG"),
        # High Yield
        "HYG":  ("Fixed Income", "High Yield", "Broad High Yield"),
        "JNK":  ("Fixed Income", "High Yield", "Broad High Yield"),
        "USHY": ("Fixed Income", "High Yield", "Broad High Yield"),
        "SHYG": ("Fixed Income", "High Yield", "Short-Term HY"),
        "SJNK": ("Fixed Income", "High Yield", "Short-Term HY"),
        "FALN": ("Fixed Income", "High Yield", "Fallen Angels"),
        "ANGL": ("Fixed Income", "High Yield", "Fallen Angels"),
        "BKLN": ("Fixed Income", "High Yield", "Bank Loans"),
        "SRLN": ("Fixed Income", "High Yield", "Bank Loans"),
        # Muni
        "MUB":  ("Fixed Income", "Municipal Bonds", "Broad Muni"),
        "VTEB": ("Fixed Income", "Municipal Bonds", "Broad Muni"),
        "SUB":  ("Fixed Income", "Municipal Bonds", "Short Muni"),
        "SCMB": ("Fixed Income", "Municipal Bonds", "Short Muni"),
        "TFI":  ("Fixed Income", "Municipal Bonds", "Broad Muni"),
        "SHM":  ("Fixed Income", "Municipal Bonds", "Short Muni"),
        "HYD":  ("Fixed Income", "Municipal Bonds", "High Yield Muni"),
        # International / EM
        "EMB":  ("Fixed Income", "Emerging Market Bonds", "USD EM Sovereign"),
        "PCY":  ("Fixed Income", "Emerging Market Bonds", "USD EM Sovereign"),
        "LEMB": ("Fixed Income", "Emerging Market Bonds", "Local Currency EM"),
        "EMLC": ("Fixed Income", "Emerging Market Bonds", "Local Currency EM"),
        "IAGG": ("Fixed Income", "International Bonds", "Global ex-US Aggregate"),
        "BNDX": ("Fixed Income", "International Bonds", "Global ex-US Aggregate"),
        # Multi-Sector / Other
        "BIV":  ("Fixed Income", "US Aggregate", "Intermediate Bond"),
        "BSV":  ("Fixed Income", "US Aggregate", "Short-Term Bond"),
        "BLV":  ("Fixed Income", "US Aggregate", "Long-Term Bond"),
        "VMBS": ("Fixed Income", "Mortgage-Backed", "MBS Broad"),
        "MBB":  ("Fixed Income", "Mortgage-Backed", "MBS Broad"),
        "CMBS": ("Fixed Income", "Mortgage-Backed", "CMBS"),

        # ── COMMODITIES ───────────────────────────────────────────────────────
        "GLD":  ("Commodities", "Precious Metals", "Gold"),
        "IAU":  ("Commodities", "Precious Metals", "Gold"),
        "GLDM": ("Commodities", "Precious Metals", "Gold"),
        "SGOL": ("Commodities", "Precious Metals", "Gold"),
        "BAR":  ("Commodities", "Precious Metals", "Gold"),
        "SLV":  ("Commodities", "Precious Metals", "Silver"),
        "SIVR": ("Commodities", "Precious Metals", "Silver"),
        "PPLT": ("Commodities", "Precious Metals", "Platinum"),
        "PALL": ("Commodities", "Precious Metals", "Palladium"),
        "DBP":  ("Commodities", "Precious Metals", "Gold + Silver"),
        # Energy Commodities
        "USO":  ("Commodities", "Energy Commodities", "Crude Oil"),
        "BNO":  ("Commodities", "Energy Commodities", "Brent Crude"),
        "UCO":  ("Commodities", "Energy Commodities", "Crude Oil 2x"),
        "SCO":  ("Commodities", "Energy Commodities", "Crude Oil -2x"),
        "UNG":  ("Commodities", "Energy Commodities", "Natural Gas"),
        "KOLD": ("Commodities", "Energy Commodities", "Natural Gas -2x"),
        "BOIL": ("Commodities", "Energy Commodities", "Natural Gas 2x"),
        "DBO":  ("Commodities", "Energy Commodities", "Oil Fund"),
        "DBE":  ("Commodities", "Energy Commodities", "Energy Basket"),
        # Agriculture
        "CORN": ("Commodities", "Agriculture", "Corn"),
        "WEAT": ("Commodities", "Agriculture", "Wheat"),
        "SOYB": ("Commodities", "Agriculture", "Soybeans"),
        "CANE": ("Commodities", "Agriculture", "Sugar"),
        "NIB":  ("Commodities", "Agriculture", "Cocoa"),
        "CAFE": ("Commodities", "Agriculture", "Coffee"),
        "DBA":  ("Commodities", "Agriculture", "Broad Agriculture"),
        "MOO":  ("Commodities", "Agriculture", "Agribusiness"),
        # Metals / Industrial
        "COPX": ("Commodities", "Industrial Metals", "Copper Miners"),
        "JJC":  ("Commodities", "Industrial Metals", "Copper"),
        "CPER": ("Commodities", "Industrial Metals", "Copper"),
        "DBB":  ("Commodities", "Industrial Metals", "Base Metals Basket"),
        "PDBC": ("Commodities", "Diversified Commodities", "Broad Commodity"),
        "DJP":  ("Commodities", "Diversified Commodities", "Broad Commodity"),
        "GSG":  ("Commodities", "Diversified Commodities", "Broad Commodity"),
        "BCI":  ("Commodities", "Diversified Commodities", "Broad Commodity"),
        # Miners
        "GDX":  ("Commodities", "Precious Metals", "Gold Miners"),
        "GDXJ": ("Commodities", "Precious Metals", "Junior Gold Miners"),
        "GOEX": ("Commodities", "Precious Metals", "Gold Miners"),
        "SIL":  ("Commodities", "Precious Metals", "Silver Miners"),
        "SILJ": ("Commodities", "Precious Metals", "Junior Silver Miners"),
        "RING": ("Commodities", "Precious Metals", "Gold Miners"),
        "PICK": ("Commodities", "Industrial Metals", "Diversified Miners"),
        "REMX": ("Commodities", "Industrial Metals", "Rare Earth & Strategic Metals"),

        # ── VOLATILITY & DERIVATIVES ──────────────────────────────────────────
        "VXX":  ("Volatility", "VIX Products", "Short-Term VIX"),
        "VIXY": ("Volatility", "VIX Products", "Short-Term VIX"),
        "UVXY": ("Volatility", "VIX Products", "Short-Term VIX 1.5x"),
        "SVXY": ("Volatility", "VIX Products", "Short VIX"),
        "VIXM": ("Volatility", "VIX Products", "Mid-Term VIX"),
        "VXZ":  ("Volatility", "VIX Products", "Mid-Term VIX"),
        "SVOL": ("Volatility", "VIX Products", "Short VIX"),
        "ZIVB": ("Volatility", "VIX Products", "Short VIX"),
        "PUTW": ("Volatility", "Options Strategy", "Put Write"),
        "BXMX": ("Volatility", "Options Strategy", "Buy-Write"),
        "CEFS": ("Volatility", "Options Strategy", "CEF Income"),

        # ── LEVERAGED / INVERSE – EQUITY ─────────────────────────────────────
        "SSO":  ("Leveraged / Inverse", "US Equity Leveraged", "S&P 500 2x"),
        "UPRO": ("Leveraged / Inverse", "US Equity Leveraged", "S&P 500 3x"),
        "SDS":  ("Leveraged / Inverse", "US Equity Inverse", "S&P 500 -2x"),
        "SPXU": ("Leveraged / Inverse", "US Equity Inverse", "S&P 500 -3x"),
        "SH":   ("Leveraged / Inverse", "US Equity Inverse", "S&P 500 -1x"),
        "PSQ":  ("Leveraged / Inverse", "US Equity Inverse", "Nasdaq -1x"),
        "QID":  ("Leveraged / Inverse", "US Equity Inverse", "Nasdaq -2x"),
        "SQQQ": ("Leveraged / Inverse", "US Equity Inverse", "Nasdaq -3x"),
        "QLD":  ("Leveraged / Inverse", "US Equity Leveraged", "Nasdaq 2x"),
        "TQQQ": ("Leveraged / Inverse", "US Equity Leveraged", "Nasdaq 3x"),
        "TNA":  ("Leveraged / Inverse", "US Equity Leveraged", "Small Cap 3x"),
        "TZA":  ("Leveraged / Inverse", "US Equity Inverse", "Small Cap -3x"),
        "URTY": ("Leveraged / Inverse", "US Equity Leveraged", "Small Cap 3x"),
        "SRTY": ("Leveraged / Inverse", "US Equity Inverse", "Small Cap -3x"),
        "DDM":  ("Leveraged / Inverse", "US Equity Leveraged", "Dow 2x"),
        "UDOW": ("Leveraged / Inverse", "US Equity Leveraged", "Dow 3x"),
        "DXD":  ("Leveraged / Inverse", "US Equity Inverse", "Dow -2x"),
        "SDOW": ("Leveraged / Inverse", "US Equity Inverse", "Dow -3x"),
        "SPXL": ("Leveraged / Inverse", "US Equity Leveraged", "S&P 500 3x"),
        "SPXS": ("Leveraged / Inverse", "US Equity Inverse", "S&P 500 -3x"),
        # Sector Leveraged / Inverse
        "TECL": ("Leveraged / Inverse", "Sector Leveraged", "Tech 3x"),
        "TECS": ("Leveraged / Inverse", "Sector Inverse", "Tech -3x"),
        "LABU": ("Leveraged / Inverse", "Sector Leveraged", "Biotech 3x"),
        "LABD": ("Leveraged / Inverse", "Sector Inverse", "Biotech -3x"),
        "FAS":  ("Leveraged / Inverse", "Sector Leveraged", "Financials 3x"),
        "FAZ":  ("Leveraged / Inverse", "Sector Inverse", "Financials -3x"),
        "NUGT": ("Leveraged / Inverse", "Sector Leveraged", "Gold Miners 2x"),
        "DUST": ("Leveraged / Inverse", "Sector Inverse", "Gold Miners -2x"),
        "JNUG": ("Leveraged / Inverse", "Sector Leveraged", "Junior Gold Miners 2x"),
        "JDST": ("Leveraged / Inverse", "Sector Inverse", "Junior Gold Miners -2x"),
        "NAIL": ("Leveraged / Inverse", "Sector Leveraged", "Homebuilders 3x"),
        "ERX":  ("Leveraged / Inverse", "Sector Leveraged", "Energy 2x"),
        "ERY":  ("Leveraged / Inverse", "Sector Inverse", "Energy -2x"),
        "GUSH": ("Leveraged / Inverse", "Sector Leveraged", "Oil & Gas 2x"),
        "DRIP": ("Leveraged / Inverse", "Sector Inverse", "Oil & Gas -2x"),
        "CURE": ("Leveraged / Inverse", "Sector Leveraged", "Healthcare 3x"),
        "DPST": ("Leveraged / Inverse", "Sector Leveraged", "Regional Banks 3x"),
        "BNKU": ("Leveraged / Inverse", "Sector Leveraged", "Banks 3x"),
        # Leveraged Bonds
        "TMF":  ("Leveraged / Inverse", "Bond Leveraged", "20yr Treasury 3x"),
        "TMV":  ("Leveraged / Inverse", "Bond Inverse", "20yr Treasury -3x"),
        "TYD":  ("Leveraged / Inverse", "Bond Leveraged", "10yr Treasury 3x"),
        "TYO":  ("Leveraged / Inverse", "Bond Inverse", "10yr Treasury -3x"),
        "UBT":  ("Leveraged / Inverse", "Bond Leveraged", "20yr Treasury 2x"),
        "TBT":  ("Leveraged / Inverse", "Bond Inverse", "20yr Treasury -2x"),
        "PST":  ("Leveraged / Inverse", "Bond Inverse", "7-10yr Treasury -2x"),
        "UST":  ("Leveraged / Inverse", "Bond Leveraged", "7-10yr Treasury 2x"),
        # Leveraged Commodities
        "UGL":  ("Leveraged / Inverse", "Commodity Leveraged", "Gold 2x"),
        "GLL":  ("Leveraged / Inverse", "Commodity Inverse", "Gold -2x"),
        "AGQ":  ("Leveraged / Inverse", "Commodity Leveraged", "Silver 2x"),
        "ZSL":  ("Leveraged / Inverse", "Commodity Inverse", "Silver -2x"),

        # ── CURRENCIES / FOREX ────────────────────────────────────────────────
        "UUP":  ("Currencies", "US Dollar", "Dollar Bullish"),
        "UDN":  ("Currencies", "US Dollar", "Dollar Bearish"),
        "FXE":  ("Currencies", "Major Currencies", "Euro"),
        "FXB":  ("Currencies", "Major Currencies", "British Pound"),
        "FXY":  ("Currencies", "Major Currencies", "Japanese Yen"),
        "FXF":  ("Currencies", "Major Currencies", "Swiss Franc"),
        "FXC":  ("Currencies", "Major Currencies", "Canadian Dollar"),
        "FXA":  ("Currencies", "Major Currencies", "Australian Dollar"),
        "CYB":  ("Currencies", "Emerging Currencies", "Chinese Yuan"),
        "FXCH": ("Currencies", "Emerging Currencies", "Chinese Yuan"),

        # ── MULTI-ASSET / ALLOCATION ──────────────────────────────────────────
        "AOM":  ("Multi-Asset", "Asset Allocation", "Conservative"),
        "AOR":  ("Multi-Asset", "Asset Allocation", "Moderate"),
        "AOA":  ("Multi-Asset", "Asset Allocation", "Aggressive"),
        "AOK":  ("Multi-Asset", "Asset Allocation", "Conservative"),
        "GAL":  ("Multi-Asset", "Asset Allocation", "Moderate"),
        "NTSX": ("Multi-Asset", "Asset Allocation", "90/60 Leveraged"),
        "GDE":  ("Multi-Asset", "Asset Allocation", "Gold + Equity"),
        "RPAR": ("Multi-Asset", "Asset Allocation", "Risk Parity"),
        "UPAR": ("Multi-Asset", "Asset Allocation", "Risk Parity"),
        "SWAN": ("Multi-Asset", "Asset Allocation", "Defined Outcome"),
    }
    # ─────────────────────────────────────────────────────────────────────────
    def read_csv_meta(path, is_etf=False):
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, header=None)
        n_cols = min(df.shape[1], 5)
        df = df.iloc[:, :n_cols]

        # ETFs.csv has 4 cols: Ticker, Name, Industry, SubIndustry
        # Stocks.csv may have 5 cols: Ticker, Name, Sector, Industry, SubIndustry
        if is_etf:
            col_names = ["Ticker", "Name", "Industry", "Subindustry", "Extra"][:n_cols]
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

        if is_etf:
            # For ETFs: Sector always comes from the built-in lookup dict.
            # Industry/Subindustry from CSV used only when dict has no entry.
            for idx, row in df.iterrows():
                t = row["Ticker"]
                if t in ETF_CATEGORIES:
                    sec, ind, sub = ETF_CATEGORIES[t]
                    df.at[idx, "Sector"]      = sec
                    df.at[idx, "Industry"]    = ind
                    df.at[idx, "Subindustry"] = sub
                else:
                    # Not in dict: use CSV Industry/Subindustry, group under "Other"
                    df.at[idx, "Sector"] = "Other"
                    # Industry and Subindustry already read from CSV columns

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
                    html += f'<div class="subindustry-block"><div class="subindustry-label">&#8627; {subindustry}</div><div class="ticker-grid">'
                    for _, row in sub_ig.sort_values("Ticker").iterrows():
                        html += (
                            f'<div class="ticker-card">'
                            f'<span class="ticker-sym" style="color:{accent_color};">{row["Ticker"]}</span>'
                            f'<span class="ticker-name">{row.get("Name","")}</span>'
                            f'</div>'
                        )
                    html += "</div></div>"
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
    padding: 9px 14px; display: flex; flex-direction: column; min-width: 95px;
    transition: border-color 0.15s, background 0.15s; cursor: default;
  }}
  .ticker-card:hover {{ border-color: #334155; background: #1e2535; }}
  .ticker-sym  {{ font-family: var(--mono); font-size: 15px; font-weight: 700; line-height: 1.2; }}
  .ticker-name {{
    font-size: 12px; color: #4a5568; line-height: 1.4; margin-top: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px;
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

    valid = list(data.columns)
    print(f"Computing matrices for {len(valid)} symbols...")

    # Pre-compute number of pairs (used in HTML template regardless of cache)
    n_combos = len(valid) * (len(valid) - 1) // 2

    # 1. Check if analysis cache is newer than the actual data file
    data_file = DATA_FILE # Use the variable from your CONFIG
    cache_exists = os.path.exists(ANALYSIS_CACHE)
    
    # Get the actual last-modified times
    data_time = os.path.getmtime(data_file) if os.path.exists(data_file) else 0
    cache_time = os.path.getmtime(ANALYSIS_CACHE) if cache_exists else 0

    if cache_exists and cache_time > data_time:
        print(f"--- Loading analyzed pairs from cache ({ANALYSIS_CACHE}) ---")
        with open(ANALYSIS_CACHE, "r") as f:
            results = json.load(f)
        # Still need these globals for price history / re-score in HTML generation
        returns    = data.pct_change().dropna(how="all")
        log_prices = np.log(data.tail(Z_LENGTH))
        prices_raw = data
        corr_short = returns.tail(CORR_SHORT).corr()
        corr_long  = returns.tail(CORR_LONG).corr()
        perf_len   = min(PERF_LENGTH, len(data) - 1)
        perf       = (data.iloc[-1] / data.iloc[-(perf_len + 1)] - 1) * 100
    else:
        # This only runs if data_file was updated or cache is missing
        print(f"--- Computing matrices and analyzing {len(valid)} symbols... ---")
        returns    = data.pct_change().dropna(how="all")
        log_prices = np.log(data.tail(Z_LENGTH))
        prices_raw = data                      # full price series for half-life & ann returns
        all_returns_for_hl = np.log(data)      # kept for reference

        corr_short = returns.tail(CORR_SHORT).corr()
        corr_long  = returns.tail(CORR_LONG).corr()

        perf_len = min(PERF_LENGTH, len(data) - 1)
        perf = (data.iloc[-1] / data.iloc[-(perf_len + 1)] - 1) * 100

        print("Building combinations...")
        combos = list(itertools.combinations(valid, 2))
        
        results = []
        for pair in tqdm(combos, desc="Analyzing Pairs"):
            r = analyze_pair(pair)
            if r:
                results.append(r)

        results = sorted(results, key=lambda x: x["Score"], reverse=True)
        
        # Save to cache
        with open(ANALYSIS_CACHE, "w") as f:
            json.dump(results, f)
        print(f"Analysis saved to {ANALYSIS_CACHE}")

    # The code below runs EVERY time, regardless of whether calculations were cached
    top_results = results[:500]

    # Compute rolling Z-score histories for top pairs
    print("Computing Z-score chart histories for top pairs...")
    for r in tqdm(top_results):
        a, b = r["Pair"].split("/")
        try:
            src = chart_data if (not chart_data.empty and a in chart_data.columns and b in chart_data.columns) else data
            dates, z_vals = compute_z_history(a, b, src)
            r["ZDates"]   = dates
            r["ZHistory"] = z_vals
            # Normalized price comparison series (rebased to 100)
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

        # Back-fill half-life / annualized returns for cache-loaded results
        if r.get("HalfLife") is None and "HalfLife" not in r:
            try:
                full_spread = np.log(prices_raw[a]) - np.log(prices_raw[b])
                r["HalfLife"] = compute_half_life(full_spread)
                if isinstance(r["HalfLife"], float) and np.isnan(r["HalfLife"]):
                    r["HalfLife"] = None
            except Exception:
                r["HalfLife"] = None
        if r.get("AnnRetA") is None and "AnnRetA" not in r:
            try:
                nd = len(prices_raw)
                r["AnnRetA"] = round(((prices_raw[a].iloc[-1] / prices_raw[a].iloc[0]) ** (252/nd) - 1)*100, 1)
                r["AnnRetB"] = round(((prices_raw[b].iloc[-1] / prices_raw[b].iloc[0]) ** (252/nd) - 1)*100, 1)
            except Exception:
                r["AnnRetA"] = r["AnnRetB"] = None
        # Back-fill EstRet / AnnRet (pairs trade est. return) for cache-loaded results
        if "EstRet" not in r or r.get("EstRet") is None:
            try:
                spread     = log_prices[a] - log_prices[b]
                spread_std = spread.std()
                est_r      = round(abs(r["Z"]) * float(spread_std) * 100, 2)
                r["EstRet"] = est_r
                hl = r.get("HalfLife")
                if hl and hl > 0:
                    r["AnnRet"] = round(est_r * (252 / hl), 1)
                else:
                    r["AnnRet"] = None
            except Exception:
                r["EstRet"] = None
                r["AnnRet"] = None

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

    rows_html = ""
    for i, r in enumerate(top_results):
        z = r["Z"]
        a, b = r["Pair"].split("/")

        if any(np.isnan(v) for v in [z, r["Score"], r["Corr"], r["PerfDiff"]]):
            continue

        name_a = TICKER_NAMES.get(a, "")
        name_b = TICKER_NAMES.get(b, "")

        if z > Z_STRONG:
            sig_label, sig_class, sig_arrow = f"SHORT {a} \u00b7 LONG {b}",  "sig-strong-short", "\u25bc\u25bc"
        elif z > Z_THRESHOLD:
            sig_label, sig_class, sig_arrow = f"SHORT {a} \u00b7 LONG {b}",  "sig-short",        "\u25bc"
        elif z < -Z_STRONG:
            sig_label, sig_class, sig_arrow = f"LONG {a} \u00b7 SHORT {b}",  "sig-strong-long",  "\u25b2\u25b2"
        elif z < -Z_THRESHOLD:
            sig_label, sig_class, sig_arrow = f"LONG {a} \u00b7 SHORT {b}",  "sig-long",         "\u25b2"
        else:
            sig_label, sig_class, sig_arrow = "NEUTRAL", "sig-neutral", "\u2014"

        cat_class = {"Pure ETF": "cat-etf", "Pure Stock": "cat-stock"}.get(r["Category"], "cat-mixed")

        price_a    = round(data[a].iloc[-1], 2)
        price_b    = round(data[b].iloc[-1], 2)
        avgvol_a   = round(vol_avg.get(a, 0))
        avgvol_b   = round(vol_avg.get(b, 0))
        z_bar_pct  = min(max((abs(z) / 3.0) * 100, 0), 100)
        z_pos      = z >= 0
        score_pct  = round(min(max(r["Score"] * 100, 0), 100))
        hl         = r.get("HalfLife")
        est_ret    = r.get("EstRet") if r.get("EstRet") is not None else 0.0
        ann_ret    = r.get("AnnRet")

        # Tag tickers as leveraged / inverse for JS filter
        def _lev_tag(t):
            if t in LEVERAGED_ETFS: return "leveraged"
            if t in INVERSE_ETFS:   return "inverse"
            return "normal"
        lev_a = _lev_tag(a)
        lev_b = _lev_tag(b)

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
        })
        chart_payload_esc = chart_payload.replace("&", "&amp;").replace("'", "&#39;")

        rows_html += f"""
        <tr class="data-row" data-category="{r['Category']}" data-z="{z}"
            data-price-a="{price_a}" data-price-b="{price_b}"
            data-vol-a="{avgvol_a}" data-vol-b="{avgvol_b}"
            data-lev-a="{lev_a}" data-lev-b="{lev_b}">
          <td class="rank-cell">{i+1}</td>
          <td class="pair-cell">
            <div class="pair-names">
              <div class="pair-ticker-row">
                <span class="ticker-a">{a}</span>
                <span class="pair-sep">/</span>
                <span class="ticker-b">{b}</span>
                <span class="{cat_class} cat-badge">{r['Category'].replace('Pure ', '')}</span>
              </div>
              <div class="pair-fullnames">
                <div class="name-a">{name_a}</div>
                <div class="name-b">{name_b}</div>
              </div>
            </div>
          </td>
          <td class="z-cell">
            <div class="z-wrapper">
              <span class="z-value {'z-pos' if z_pos else 'z-neg'}">{z:+.2f}&sigma;</span>
              <div class="z-bar-track">
                <div class="z-bar-fill {'z-bar-pos' if z_pos else 'z-bar-neg'}" style="width:{z_bar_pct}%;"></div>
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
          <td class="sig-cell">
            <span class="signal-badge {sig_class}">{sig_arrow} {sig_label}</span>
          </td>
          <td class="chart-cell">
            <button class="chart-btn" onclick="openChart(this,'z')" data-chart='{chart_payload_esc}'>&#9657; Z-Chart</button>
            <button class="chart-btn price-btn" onclick="openChart(this,'price')" data-chart='{chart_payload_esc}'>&#9724; Price</button>
          </td>
          <td class="shares-cell sharesA" data-price="{price_a}">
            <div class="share-price">${price_a:,.2f}</div>
            <div class="share-vol">{fmt_vol(avgvol_a)}</div>
          </td>
          <td class="shares-cell sharesB" data-price="{price_b}">
            <div class="share-price">${price_b:,.2f}</div>
            <div class="share-vol">{fmt_vol(avgvol_b)}</div>
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
  .ticker-a {{ font-family: var(--mono); font-size: 14px; font-weight: 700; color: var(--cyan); }}
  .pair-sep  {{ color: var(--muted); margin: 0 2px; font-family: var(--mono); }}
  .ticker-b  {{ font-family: var(--mono); font-size: 14px; font-weight: 700; color: white; }}
  .pair-fullnames {{ display: flex; flex-direction: column; gap: 1px; margin-top: 2px; }}
  .name-a  {{ font-size: 10px; color: #4a8aaa; white-space: normal; line-height: 1.35; }}
  .name-b  {{ font-size: 10px; color: #6b7f9a; white-space: normal; line-height: 1.35; }}

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

  /* CORR / PERF / SCORE */
  .corr-cell  {{ min-width: 90px; }}
  .corr-value {{ font-family: var(--mono); font-size: 13px; color: white; display: block; }}
  .corr-brk   {{ font-family: var(--mono); font-size: 10px; color: var(--muted); }}
  .perf-cell  {{ min-width: 75px; }}
  .perf-pos {{ font-family: var(--mono); font-size: 13px; color: var(--green); font-weight: 500; }}
  .perf-neg {{ font-family: var(--mono); font-size: 13px; color: var(--red);   font-weight: 500; }}

  /* HALF-LIFE */
  .hl-cell   {{ min-width: 70px; text-align: center; }}
  .hl-value  {{ font-family: var(--mono); font-size: 13px; color: var(--purple); font-weight: 600; }}
  .hl-na     {{ font-family: var(--mono); font-size: 13px; color: var(--muted); }}

  /* EST RETURN CELL */
  .est-cell  {{ min-width: 95px; text-align: right; }}
  .est-ret   {{ font-family: var(--mono); font-size: 13px; font-weight: 600; color: #34d399; display: block; }}
  .ann-ret   {{ font-family: var(--mono); font-size: 10px; color: #059669; display: block; }}

  .score-cell {{ min-width: 100px; }}
  .score-bar-wrap {{ display: flex; flex-direction: column; gap: 3px; }}
  .score-num  {{ font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--amber); }}
  .score-bar-track {{ height: 3px; background: var(--faint); border-radius: 2px; width: 70px; overflow: hidden; }}
  .score-bar-fill  {{ height: 100%; background: linear-gradient(90deg, var(--amber), var(--orange)); border-radius: 2px; }}

  /* SIGNAL */
  .sig-cell {{ min-width: 220px; }}
  .signal-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 4px; font-size: 11px;
    font-weight: 700; letter-spacing: 0.05em; white-space: nowrap; font-family: var(--mono);
  }}
  .sig-strong-short {{ background: var(--red-dim);          color: var(--red);    border: 1px solid rgba(239,68,68,0.4); }}
  .sig-short        {{ background: rgba(249,115,22,0.1);    color: var(--orange); border: 1px solid rgba(249,115,22,0.4); }}
  .sig-strong-long  {{ background: var(--green-dim);        color: var(--green);  border: 1px solid rgba(34,197,94,0.4); }}
  .sig-long         {{ background: rgba(132,204,22,0.1);    color: #84cc16;       border: 1px solid rgba(132,204,22,0.4); }}
  .sig-neutral      {{ background: rgba(71,85,105,0.2);     color: var(--muted);  border: 1px solid var(--border); }}

  /* CHART BUTTON */
  .chart-cell {{ min-width: 100px; text-align: center; display: flex; flex-direction: column; gap: 5px; align-items: center; justify-content: center; }}
  .chart-btn {{
    background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.25);
    color: var(--cyan); font-family: var(--mono); font-size: 11px; font-weight: 600;
    padding: 5px 11px; border-radius: 4px; cursor: pointer; letter-spacing: 0.05em;
    transition: background 0.15s, border-color 0.15s; white-space: nowrap; width: 88px;
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

  /* SHARES */
  .shares-cell {{ font-family: var(--mono); font-size: 12px; color: var(--text); min-width: 90px; text-align: right; vertical-align: middle; }}
  .share-price {{ font-size: 11px; color: #64748b; }}
  .share-vol   {{ font-size: 10px; color: #3d4f62; letter-spacing: 0.03em; }}

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
  <div class="stat-item"><div class="stat-label">Z Threshold</div><div class="stat-value">{Z_THRESHOLD:.1f}&sigma; / {Z_STRONG:.1f}&sigma;</div></div>
  <div class="stat-item"><div class="stat-label">Min Correlation</div><div class="stat-value">{MIN_CORR_FILTER:.2f}</div></div>
  <div class="stat-item"><div class="stat-label">Corr Window</div><div class="stat-value">{CORR_SHORT}d / {CORR_LONG}d</div></div>
  <div class="stat-item"><div class="stat-label">Z Window</div><div class="stat-value">{Z_LENGTH}d</div></div>
  <div class="stat-item"><div class="stat-label">Perf Window</div><div class="stat-value amber">{PERF_LENGTH}d</div></div>
</div>

<!-- CONTROLS -->
<div class="controls">
  <div class="control-group">
    <label>Capital ($)</label>
    <button class="step-btn" onclick="stepValue('capitalInput',-1000)">−</button>
    <input type="number" id="capitalInput" value="10000" min="0" step="1000" oninput="calcShares()">
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
    <label>Lev / Inv</label>
    <select id="levFilter" onchange="applyFilters()">
      <option value="all">All</option>
      <option value="exclude_both">Exclude Lev &amp; Inv</option>
      <option value="exclude_lev">Exclude Leveraged</option>
      <option value="exclude_inv">Exclude Inverse</option>
      <option value="only_lev">Only Leveraged</option>
      <option value="only_inv">Only Inverse</option>
      <option value="only_both">Only Lev &amp; Inv</option>
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
    <label>Sort</label>
    <select id="sortBy" onchange="sortTable()">
      <option value="score">Score</option>
      <option value="z_abs">|Z-Score|</option>
      <option value="hl">Half-Life</option>
      <option value="est_ret">Est Return</option>
      <option value="corr">Correlation</option>
      <option value="perf">Perf Diff</option>
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
  <th onclick="setSort('est_ret')">Est Return &#8597;</th>
  <th onclick="setSort('perf')">Perf Diff &#8597;</th>
  <th onclick="setSort('score')">Score &#8597;</th>
  <th>Signal</th>
  <th style="text-align:center;">Charts</th>
  <th style="text-align:right;">Leg A &nbsp;Shares</th>
  <th style="text-align:right;">Leg B &nbsp;Shares</th>
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
      <div class="chart-container">
        <canvas id="zChart" style="display:block;"></canvas>
        <canvas id="pChart" style="display:none;position:absolute;inset:0;width:100%;height:100%;"></canvas>
      </div>
    </div>
    <div class="modal-footer" id="modalFooter"></div>
  </div>
</div>

<script>
// ─── CHART STATE ──────────────────────────────────────────────────────────────
let activeChart     = null;
let activePChart    = null;
let currentChartData = null;

// Load annotation plugin async
(function() {{
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js";
  s.onload = () => {{ Chart.register(window["chartjs-plugin-annotation"]); }};
  document.head.appendChild(s);
}})();

// ─── TAB SWITCH ──────────────────────────────────────────────────────────────
function switchTab(mode) {{
  const isZ = mode === 'z';
  document.getElementById("tabZ").classList.toggle("active", isZ);
  document.getElementById("tabP").classList.toggle("active", !isZ);
  document.getElementById("legendZ").style.display = isZ ? "" : "none";
  document.getElementById("legendP").style.display = isZ ? "none" : "";
  document.getElementById("zChart").style.display  = isZ ? "block" : "none";
  document.getElementById("pChart").style.display  = isZ ? "none" : "block";
  if (!isZ && currentChartData && !activePChart) buildPriceChart(currentChartData);
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

  // Footer
  const footerDates = p.priceDates && p.priceDates.length ? p.priceDates : (p.dates || []);
  document.getElementById("modalFooter").innerHTML =
    `<span>Z window: <em>${{p.zWindow}} days</em></span>` +
    `<span>Data from: <em>${{footerDates[0] || "—"}}</em></span>` +
    `<span>Last: <em>${{footerDates[footerDates.length-1] || "—"}}</em></span>` +
    `<span>Z pts: <em>${{(p.dates||[]).length}}</em></span>` +
    `<span>Price pts: <em>${{(p.priceDates||[]).length}}</em></span>` +
    `<span style="margin-left:auto;color:#2d3748;">ESC or click outside to close</span>`;

  // Destroy old charts
  if (activeChart)  {{ activeChart.destroy();  activeChart  = null; }}
  if (activePChart) {{ activePChart.destroy(); activePChart = null; }}

  // Reset to Z tab
  document.getElementById("zChart").style.display = "block";
  document.getElementById("pChart").style.display = "none";
  document.getElementById("tabZ").classList.add("active");
  document.getElementById("tabP").classList.remove("active");
  document.getElementById("legendZ").style.display = "";
  document.getElementById("legendP").style.display = "none";

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
        pointRadius: dates.length > 250 ? 0 : 2.5,
        pointHoverRadius: 6,
        pointBackgroundColor: ptColors,
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
          padding: 14, caretSize: 5,
          callbacks: {{
            label: c => {{
              const v = c.raw;
              if (v === null) return " Z = —";
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
          pointRadius: priceDates.length > 300 ? 0 : 2, pointHoverRadius: 5,
          pointBackgroundColor: "#38bdf8", fill: true, backgroundColor: gradA,
          tension: 0.25, spanGaps: true,
        }},
        {{
          label: b, data: priceB,
          borderColor: "#a78bfa", borderWidth: 2,
          pointRadius: priceDates.length > 300 ? 0 : 2, pointHoverRadius: 5,
          pointBackgroundColor: "#a78bfa", fill: true, backgroundColor: gradB,
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
          padding: 14, caretSize: 5,
          callbacks: {{
            label: c => {{
              const pct = (c.raw - 100).toFixed(2);
              return ` ${{c.dataset.label}}: ${{c.raw.toFixed(2)}}  (${{pct >= 0 ? "+" : ""}}${{pct}}%)`;
            }},
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

// ─── CLOSE MODAL ─────────────────────────────────────────────────────────────
function closeChart() {{
  document.getElementById("chartModal").classList.remove("open");
  document.body.style.overflow = "";
  if (activeChart)  {{ activeChart.destroy();  activeChart  = null; }}
  if (activePChart) {{ activePChart.destroy(); activePChart = null; }}
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
    const cA = row.querySelector(".sharesA");
    const cB = row.querySelector(".sharesB");
    const pA = parseFloat(cA.dataset.price);
    const pB = parseFloat(cB.dataset.price);
    if (total > 0 && pA > 0 && pB > 0) {{
      const sA = Math.round(leg / pA);
      cA.textContent = sA.toLocaleString();
      cB.textContent = Math.round((sA * pA) / pB).toLocaleString();
    }} else {{ cA.textContent = cB.textContent = "\u2014"; }}
  }});
}}

// ─── FILTERS ──────────────────────────────────────────────────────────────────
function applyFilters() {{
  const catF      = document.getElementById("typeFilter").value;
  const levF      = document.getElementById("levFilter").value;
  const minZv     = parseFloat(document.getElementById("minZ").value) || 0;
  const searchV   = document.getElementById("tickerSearch").value.toUpperCase().trim();
  const minPriceV = parseFloat(document.getElementById("minPrice").value) || 0;
  const minVolV   = parseFloat(document.getElementById("minVol").value) || 0;
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

    // Lev/Inv classification: pair is "leveraged" if either leg is leveraged,
    // "inverse" if either leg is inverse, "both" if pair includes both types.
    const isLev = levA === "leveraged" || levB === "leveraged";
    const isInv = levA === "inverse"   || levB === "inverse";

    let show = true;
    if (catF !== "All" && cat !== catF)          show = false;
    if (Math.abs(z) < minZv)                     show = false;
    if (searchV && !pairText.includes(searchV))  show = false;
    if (minPriceV > 0 && (priceA < minPriceV || priceB < minPriceV)) show = false;
    if (minVolV > 0 && volA > 0 && volB > 0 && (volA < minVolV || volB < minVolV)) show = false;

    // Leveraged / Inverse filter
    if (levF === "exclude_both" && (isLev || isInv)) show = false;
    else if (levF === "exclude_lev" && isLev)        show = false;
    else if (levF === "exclude_inv" && isInv)        show = false;
    else if (levF === "only_lev"  && !isLev)         show = false;
    else if (levF === "only_inv"  && !isInv)         show = false;
    else if (levF === "only_both" && !(isLev || isInv)) show = false;

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

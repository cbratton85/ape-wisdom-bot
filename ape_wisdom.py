import requests
import yfinance as yf
import pandas as pd
import time
import datetime
import os
import sys
import math
import random
import json
import re
from bs4 import BeautifulSoup
import shutil
import numpy as np
from typing import Any

# ==============================================================================
#                               SECTION 1: CONFIGURATION
# ==============================================================================
# Paths and Environment Settings
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(SCRIPT_DIR, "public")
LOGOS_DIR = os.path.join(PUBLIC_DIR, "logos")
os.makedirs(LOGOS_DIR, exist_ok=True)
CACHE_FILE = os.path.join(SCRIPT_DIR, "ape_cache.json")
MARKET_DATA_CACHE_FILE = os.path.join(SCRIPT_DIR, "market_data.pkl")
HISTORY_FILE = os.path.join(SCRIPT_DIR, "market_history.json")
DELISTED_CACHE_FILE = os.path.join(SCRIPT_DIR, "delisted_cache.json")
GEKKO_SCREENER_FILE = os.path.join(SCRIPT_DIR, "gekko_screener.csv")

# Timeouts and Retention
CACHE_EXPIRY_SECONDS = 43200  # 12 hours
RETENTION_DAYS = 3
DELISTED_RETRY_DAYS = 1
TOOLTIP_HISTORY_DAYS = 24

# ------------------------------------------------------------------------------
# Filters & Algorithm Tuning
# ------------------------------------------------------------------------------
MIN_PRICE = 1.00
MIN_AVG_VOLUME = 50000
AVG_VOLUME_DAYS = 30
NAME_MAX_WIDTH = 50
LOTTERY_SIZE = 1
REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.0
TICKER_FIXES = {}
PERMANENT_BLACKLIST = set()  # Use set for O(1) membership checking

# ------------------------------------------------------------------------------
# Console Presentation (ANSI Colors)
# ------------------------------------------------------------------------------
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_RESET = '\033[0m'

# Network Session Setup
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})


# ==============================================================================
#                               SECTION 2: DATA PERSISTENCE CLASS
# ==============================================================================
class HistoryTracker:
    """
    Manages the historical JSON data, calculating velocity, acceleration,
    and maintaining the sliding window of data points.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError, ValueError) as e:
                print(f"{C_YELLOW}[!] Warning: Could not load cache: {e}{C_RESET}")
                return {}
        return {}

    def save(self, df):
        now = datetime.datetime.now(datetime.timezone.utc)
        now_ts = now.strftime("%Y-%m-%d %H:%M")
        cutoff = now.replace(tzinfo=None) - datetime.timedelta(days=RETENTION_DAYS)
        
        # --- EXCLUDE LIST ---
        # Keeps calculated fields safe from being overwritten by raw API data
        exclude_list = [
            'sym', 'name', 'meta', 'history', 'desc', 'type', 'avgvol', 'mcap', 'rolling',
            'z_rank_plus', 'z_surge', 'z_mnt_perc', 'z_upvotes', 'z_accel', 'z_upv_plus', 
            'z_ment', 'z_squeeze', 'type_tag', 'industry', 'heat', 'velocity', 'accel',
            'streak', 'upv_chg', 'day_perc', 'price', 'rsi', 'di_plus', 'di-', 'stoch_k',
            'stoch_d', 'curvol', 'raw_sctr', 'raw_ibd', 'ibd_rs', 'raw_spy', 'spy_rs'] 

        no_round_list = ['rank', 'rank_plus', 'ment', 'upvotes', 'upv_plus', 'streak']

        precision_map = {
            'price': 2, 'surge': 0, 'mnt_perc': 0, 'squeeze': 0, 
            'conv': 1, 'eff': 1, 'accel': 0, 'velocity': 0, 'heat': 1,
            'spy_rs': 1, 'gi': 1
        }

        # 3. Main Loop: Process each row in the DataFrame
        for _, row in df.iterrows():
            ticker = row['Sym']
            if ticker not in self.data:
                self.data[ticker] = {}

            entry = {}
            # Translate special characters: % → _perc, + → _plus
            trans_table = str.maketrans({'%': '_perc', '+': '_plus'})
            for col, val in row.items():
                col_clean = col.lower().translate(trans_table)
                
                if col_clean in exclude_list:
                    continue

                if col_clean in no_round_list:
                    entry[col_clean] = val
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    decimals = precision_map.get(col_clean, 2)
                    if decimals == 0:
                        entry[col_clean] = int(round(float(val)))
                    else:
                        entry[col_clean] = round(float(val), decimals)
                else:
                    entry[col_clean] = str(val)
                
            # Save this ticker's current snapshot
            self.data[ticker][now_ts] = entry

        # 4. Cleanup old data
        new_data_cleaned = {}
        for ticker, entries in self.data.items():
            valid_entries = {}
            for d, v in entries.items():
                try:
                    dt = datetime.datetime.strptime(d, "%Y-%m-%d %H:%M")
                    if dt > cutoff:
                        valid_entries[d] = v
                except ValueError:
                    continue 
            
            if valid_entries:
                new_data_cleaned[ticker] = valid_entries
        
        # 5. Update Memory (Wait for flush() to write to disk)
        self.data = new_data_cleaned

    def get_metrics(self, ticker, current_price, current_mnt, current_rank_plus, current_upvotes, current_surge, current_gi):
        """
        Calculates momentum metrics and builds history strings for tooltips.
        Passes 'current_surge' to ensure tooltip history matches the live SRG column.
        """
        if ticker not in self.data or not self.data[ticker]:
            return {"vel": 0, "accel": 0, "upv_chg": 0, "streak": 0, "rolling_trend": 0, "hist": {}}

        dates = sorted(self.data[ticker].keys())
        
        # --- CALCULATION LOGIC ---
        current_entry = self.data[ticker][dates[-1]]
        prev_entry = self.data[ticker][dates[-2]] if len(dates) > 1 else current_entry

        # === LIVE DATA SYNC ===
        try:
            current_entry['upvotes'] = int(current_upvotes)
            current_entry['price'] = float(current_price)
            current_entry['surge'] = int(current_surge)  # Syncs real-time SRG to history
            if current_gi is not None and not pd.isna(current_gi):
                current_entry['gi'] = round(float(current_gi), 1)
        except (ValueError, TypeError) as e:
            print(f"{C_YELLOW}[!] Warning: Could not sync live data for {ticker}: {e}{C_RESET}") 

        curr_rank = current_rank_plus 
        prev_rank = prev_entry.get('rank_plus', 0)
        velocity = int(curr_rank - prev_rank)

        prev_upv = prev_entry.get('upvotes', 0)
        if prev_upv == 0 and 'Upvotes' in prev_entry:
            # Data normalization issue - mixed case keys detected
            prev_upv = prev_entry.get('Upvotes', 0)
            print(f"{C_YELLOW}[!] Data quality: {ticker} has mixed-case keys{C_RESET}")
        upv_chg = int(current_upvotes - prev_upv)

        accel = 0
        if len(dates) >= 3:
            prev_2_entry = self.data[ticker][dates[-3]]
            prev_2_rank = prev_2_entry.get('rank_plus', 0)
            prev_vel = int(prev_rank - prev_2_rank)
            accel = velocity - prev_vel

        rolling_trend = 0
        for d in dates:
            val = self.data[ticker][d].get('rank_plus', 0)
            if val > 0: rolling_trend = rolling_trend + 1 if rolling_trend >= 0 else 1
            elif val < 0: rolling_trend = rolling_trend - 1 if rolling_trend <= 0 else -1

        # --- UPDATE THE ENTRY IN MEMORY ---
        current_entry['velocity'] = velocity
        current_entry['accel'] = accel
        current_entry['upv_plus'] = upv_chg
        current_entry['streak'] = rolling_trend
        
        # --- BUILD THE HISTORY MAP ---
        recent_dates = dates[-TOOLTIP_HISTORY_DAYS:]
        history_map = {
            'rank': [], 'rank_plus': [], 'price': [], 'ment': [], 'upvotes': [], 
            'accel': [], 'velocity': [], 'streak': [], 'upv_plus': [],
            'eff': [], 'conv': [], 'surge': [], 'mnt_perc': [], 'squeeze': [], 'master_score': [],
            'spy_rs': [], 'gi': []
        }

        def get_val(entry, key, signed=False, is_perc=False, decimals=2):
            val = entry.get(key, 0)
            if isinstance(val, (float, np.floating)): 
                val = round(float(val), decimals)
            if is_perc: 
                return f"{val}%"
            return f"{'+' if signed and val > 0 else ''}{val}"

        for d in recent_dates:
            entry = self.data[ticker][d]
            history_map['rank'].append(get_val(entry, 'rank'))
            history_map['rank_plus'].append(get_val(entry, 'rank_plus', signed=True))
            history_map['price'].append(get_val(entry, 'price'))
            history_map['ment'].append(get_val(entry, 'ment'))
            history_map['upvotes'].append(get_val(entry, 'upvotes'))
            history_map['accel'].append(get_val(entry, 'accel', signed=True))
            history_map['velocity'].append(get_val(entry, 'velocity', signed=True)) 
            history_map['streak'].append(get_val(entry, 'streak', signed=True))
            history_map['upv_plus'].append(get_val(entry, 'upv_plus', signed=True))
            history_map['eff'].append(get_val(entry, 'eff'))
            history_map['conv'].append(get_val(entry, 'conv'))
            history_map['surge'].append(get_val(entry, 'surge', is_perc=True))
            history_map['mnt_perc'].append(get_val(entry, 'mnt_perc', is_perc=True))
            history_map['squeeze'].append(get_val(entry, 'squeeze'))
            history_map['master_score'].append(get_val(entry, 'master_score', decimals=1))
            history_map['spy_rs'].append(get_val(entry, 'spy_rs'))
            history_map['gi'].append(get_val(entry, 'gi', decimals=1))
    
        final_histories = {k: " → ".join(v) for k, v in history_map.items()}

        return {
            "vel": velocity, 
            "accel": accel, 
            "upv_chg": upv_chg, 
            "streak": rolling_trend, 
            "hist": final_histories 
        }

    def flush(self):
        """Force saves the current data state to disk."""
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=4)


# ==============================================================================
#                               SECTION 3: UTILITY & HELPER FUNCTIONS
# ==============================================================================
def load_cache(filepath):
    """Generic loader for any JSON cache file"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_cache(filepath, cache_data):
    """Generic saver for any JSON cache file"""
    try:
        with open(filepath, 'w') as f: json.dump(cache_data, f, indent=4)
    except: pass

def load_gekko_scores(path=GEKKO_SCREENER_FILE):
    """Loads ticker -> GI score map from the local Gekko CSV export."""
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"{C_YELLOW}[!] Warning: Could not read GI CSV ({path}): {e}{C_RESET}")
        return {}

    required = {"ticker", "gi_score"}
    if not required.issubset(set(df.columns)):
        print(f"{C_YELLOW}[!] Warning: GI CSV missing required columns {required}{C_RESET}")
        return {}

    out = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        raw_gi = row.get("gi_score")
        if raw_gi is None:
            continue
        try:
            gi_val = float(raw_gi)
        except (TypeError, ValueError):
            continue
        if np.isnan(gi_val):
            continue
        gi_clamped = max(0.0, min(100.0, gi_val))
        # Store aliases so BRK.B/BRK-B style symbols match regardless of source format.
        out[ticker] = gi_clamped
        out[ticker.replace('.', '-')] = gi_clamped
        out[ticker.replace('-', '.')] = gi_clamped
    return out

def fetch_meta_data_robust(ticker):
    """Fetches descriptive metadata and exchange info from yfinance."""
    name, meta, quote_type = ticker, "Unknown", "EQUITY"
    mcap, currency, description, exchange = 0, "USD", "", "Unknown"

    try:
        dat = yf.Ticker(ticker)
        info = dat.info
        if info:
            quote_type = info.get('quoteType', 'EQUITY')
            name = info.get('shortName') or info.get('longName') or ticker
            mcap = info.get('marketCap', 0)
            currency = info.get('currency', 'USD')
            description = ""
            
            exchange = info.get('exchange', 'Unknown')
            
            if quote_type == 'ETF':
                meta = info.get('category', 'Unknown')
            else:
                meta = info.get('industry', 'Unknown')
                meta = meta.replace('\r', '').replace('\n', '').strip()
                if not meta or meta == name or meta == "Unknown - Unknown":
                    meta = "Unknown"
    except Exception:
        pass

    return {
        'ticker': ticker,
        'name': name,
        'meta': meta,
        'type': quote_type,
        'mcap': mcap,
        'currency': currency,
        'description': "",
        'exchange': exchange  # Now correctly populated from yfinance
    }

def calculate_rsi(series, period=14):
    """
    Calculates RSI using Wilder's Smoothing (EMA-based) to match TradingView.
    """
    try:
        if series is None or len(series) < period + 1:
            return 0
        
        # Ensure data is numeric and drop empty rows
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(series) < period + 1:
            return 0

        delta = series.diff()
        
        # Ups and Downs
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        # Wilder's Smoothing uses alpha = 1/period. 
        # In Pandas EWM, 'com' (center of mass) = period - 1
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        final_val = rsi.iloc[-1]
        
        # Return 50.0 if the price was completely flat (NaN result)
        if np.isnan(final_val):
            return 50.0
            
        return round(float(final_val), 1)

    except Exception:
        return 0

def calculate_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    """
    Calculates the Stochastic Oscillator accurately, preventing NaN warmup pollution.
    """
    try:
        # 1. Ensure numeric data and drop bad rows to maintain calendar continuity
        close = pd.to_numeric(df['Close'], errors='coerce')
        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        
        # Make sure we have enough days to satisfy all rolling windows
        required_periods = k_period + smooth_k + d_period - 2
        if len(df) < required_periods:
            return 50.0, 50.0

        # 2. Highest Highs & Lowest Lows (The Lookback)
        low_min = low.rolling(window=k_period, min_periods=k_period).min()
        high_max = high.rolling(window=k_period, min_periods=k_period).max()
        
        # 3. Raw Fast %K
        denom = high_max - low_min
        
        # Avoid divide-by-zero on flatline days. 
        # Leave it as NaN so it doesn't skew the moving averages!
        fast_k = 100 * (close - low_min) / denom.replace(0, np.nan)
        
        # 4. Slow %K (Smooth the Fast %K)
        slow_k = fast_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
        
        # 5. %D (Smooth the Slow %K)
        slow_d = slow_k.rolling(window=d_period, min_periods=d_period).mean() if d_period > 1 else slow_k
        
        # 6. Extract the final values
        final_k = slow_k.iloc[-1]
        final_d = slow_d.iloc[-1]
        
        # ONLY apply the 50.0 fallback at the very end if the stock completely flatlined
        if pd.isna(final_k): final_k = 50.0
        if pd.isna(final_d): final_d = 50.0
        
        return float(final_k), float(final_d)
        
    except Exception:
        return 50.0, 50.0

def calculate_raw_sctr(hist, ticker_name="Unknown"):
    """
    Calculates the raw technical score based on the SCTR formula.
    """
    try:
        # Clean the data: Drop any blank days yfinance might have padded
        close = pd.to_numeric(hist['Close'], errors='coerce').dropna()

        if close.empty or len(close) < 200:
            print(f"  > SCTR Skipped for {ticker_name}: Not enough data ({len(close)} days)")
            return -9999.0  # <--- Impossible number for failures

        # 1. Long-Term Indicators
        ema200 = close.ewm(span=200, adjust=False).mean()
        pct_above_ema200 = ((close.iloc[-1] - ema200.iloc[-1]) / ema200.iloc[-1]) * 100
        roc125 = ((close.iloc[-1] - close.iloc[-125]) / close.iloc[-125]) * 100

        # 2. Medium-Term Indicators
        ema50 = close.ewm(span=50, adjust=False).mean()
        pct_above_ema50 = ((close.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]) * 100
        roc20 = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100

        # 3. Short-Term Indicators
        rsi14 = calculate_rsi(close)
        
        # PPO Histogram Slope
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        ppo = ((ema12 - ema26) / ema26) * 100
        ppo_ema9 = ppo.ewm(span=9, adjust=False).mean()
        ppo_hist = ppo - ppo_ema9
        ppo_slope = (ppo_hist.iloc[-1] - ppo_hist.iloc[-4]) / 3

        # Final Raw Score
        raw_score = (pct_above_ema200 * 0.30) + (roc125 * 0.30) + \
                    (pct_above_ema50 * 0.15) + (roc20 * 0.15) + \
                    (rsi14 * 0.05) + (ppo_slope * 0.05)

        return float(raw_score)
    except Exception as e:
        print(f"  > SCTR Math Error for {ticker_name}: {e}")
        return -9999.0

def calculate_raw_ibd_rs(hist, ticker_name="Unknown"):
    """
    Calculates a raw Relative Strength score approximating the IBD RS formula.
    Uses 12-month data with heavier weighting on the most recent 3 months.
    """
    try:
        close = pd.to_numeric(hist['Close'], errors='coerce').dropna()
        if close.empty or len(close) < 63:
            return -9999.0
        
        c_now = close.iloc[-1]
        
        # Helper to safely grab historical prices without crashing on newer stocks
        def get_hist_price(days_back):
            return close.iloc[-days_back - 1] if len(close) > days_back else close.iloc[0]

        c_63 = get_hist_price(63)
        c_126 = get_hist_price(126)
        c_189 = get_hist_price(189)
        c_252 = get_hist_price(252)

        if 0 in [c_63, c_126, c_189, c_252]: 
            return -9999.0

        # IBD Formula Approximation: 40% Q1, 20% Q2, 20% Q3, 20% Q4
        rs_raw = ((c_now - c_63) / c_63) * 0.4 + \
                 ((c_63 - c_126) / c_126) * 0.2 + \
                 ((c_126 - c_189) / c_189) * 0.2 + \
                 ((c_189 - c_252) / c_252) * 0.2
        return float(rs_raw * 100)
    except Exception:
        return -9999.0

def calculate_raw_spy_rs(hist, spy_hist, ticker_name="Unknown"):
    """
    Calculates the raw relative return compared to SPY, strictly time-aligned.
    Strips timezones to prevent intersection failures between yf.Ticker and yf.download.
    """
    try:
        stock_close = pd.to_numeric(hist['Close'], errors='coerce').dropna()
        spy_close = pd.to_numeric(spy_hist, errors='coerce').dropna()

        if stock_close.empty or spy_close.empty or len(stock_close) < 63:
            return -9999.0
        
        # --- TIMEZONE FIX ---
        # Strip all timezone data so we are purely matching on YYYY-MM-DD
        if stock_close.index.tz is not None:
            stock_close.index = stock_close.index.tz_localize(None)
        if spy_close.index.tz is not None:
            spy_close.index = spy_close.index.tz_localize(None)
            
        stock_close.index = stock_close.index.normalize()
        spy_close.index = spy_close.index.normalize()
        
        # Find the exact overlapping calendar days
        common_dates = stock_close.index.intersection(spy_close.index)
        
        if len(common_dates) < 63:
            return -9999.0
        
        # Lock both arrays to identical dates and sort them
        # Note: using ~ .duplicated() prevents crashes if yfinance returns duplicate days
        stock_aligned = stock_close.loc[common_dates]
        stock_aligned = stock_aligned[~stock_aligned.index.duplicated(keep='first')].sort_index()
        
        spy_aligned = spy_close.loc[common_dates]
        spy_aligned = spy_aligned[~spy_aligned.index.duplicated(keep='first')].sort_index()
        
        lookback = min(len(stock_aligned), 252)
        
        c_now = stock_aligned.iloc[-1]
        c_past = stock_aligned.iloc[-lookback]
        stock_perf = (c_now - c_past) / c_past
        
        spy_now = spy_aligned.iloc[-1]
        spy_past = spy_aligned.iloc[-lookback]
        spy_perf = (spy_now - spy_past) / spy_past
        
        return float((stock_perf - spy_perf) * 100)
    except Exception:
        return -9999.0

def get_cached_logo(ticker):
    token = os.environ.get("LOGO_DEV_TOKEN")
    
    file_name = f"{ticker.upper()}.png"
    local_path = os.path.join(LOGOS_DIR, file_name)
    html_src_path = f"logos/{file_name}"
    url = f"https://img.logo.dev/ticker/{ticker.lower()}?token={token}"

    # --- TTL CONFIGURATION ---
    # 30 days in seconds = 30 * 24 * 60 * 60
    LOGO_TTL_SECONDS = 2592000 
    
    needs_download = True

    # 1. Check if we have the file, and if it's old enough to re-verify
    if os.path.exists(local_path):
        needs_download = False # Assume we are good by default
        
        file_age = time.time() - os.path.getmtime(local_path)
        
        # If the file is older than 30 days, check the server
        if file_age > LOGO_TTL_SECONDS:
            try:
                local_size = os.path.getsize(local_path)
                head_req = requests.head(url, timeout=5)
                
                if head_req.status_code == 200:
                    remote_size = int(head_req.headers.get('Content-Length', 0))
                    
                    if remote_size > 0 and local_size != remote_size:
                        needs_download = True
                    else:
                        # Sizes match. The logo hasn't changed.
                        # Update the file's 'modified' timestamp to right now 
                        # so we don't check again for another 30 days.
                        os.utime(local_path, None)
            except Exception:
                # If the network check fails, stick with our cached version
                pass

    # 2. Return the cached path if no update is needed
    if not needs_download:
        return html_src_path

    # 3. Download / Overwrite the file if it's new or the size changed
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(r.content)
            return html_src_path
        else:
            return html_src_path if os.path.exists(local_path) else "https://s3-symbol-logo.tradingview.com/indices/nasdaq-100.svg"
            
    except Exception:
        return html_src_path if os.path.exists(local_path) else "https://s3-symbol-logo.tradingview.com/indices/nasdaq-100.svg"

# ==============================================================================
#                               SECTION 4: CORE ANALYSIS ENGINE
# ==============================================================================
def filter_and_process(stocks):
    """
    The main logic pipeline:
    1. Check Lottery (retries banned stocks)
    2. Filter valid tickers
    3. Fetch Metadata
    4. Fetch Market Data (Batch Mode)
    5. Construct DataFrame & Calculate Indicators (RSI, Surge, Squeeze)
    6. Score and Sort
    """
    if not stocks: return pd.DataFrame()

    gi_map = load_gekko_scores()
    print(f"{C_CYAN}[#] Loaded {len(gi_map):,} GI ticker mappings from CSV.{C_RESET}")

    # --- LOAD CACHES SEPARATELY ---
    local_cache = load_cache(CACHE_FILE)            
    delisted_cache = load_cache(DELISTED_CACHE_FILE) 
    
    now = datetime.datetime.now(datetime.UTC)
    updated_delisted = False 

    # -----------------------------------------------------
    # Step 1. THE LOTTERY (Random Retry)
    # -----------------------------------------------------
    tickers_to_retry = []
    banned_tickers = list(delisted_cache.keys())
    
    if banned_tickers:
        draw_count = min(len(banned_tickers), LOTTERY_SIZE)
        tickers_to_retry = random.sample(banned_tickers, draw_count)
        if tickers_to_retry:
            print(f"{C_GREEN}[+] 🎰 LOTTERY TIME: Re-checking {len(tickers_to_retry)} banned tickers...{C_RESET}")
            for t in tickers_to_retry:
                if t in delisted_cache: del delisted_cache[t]
            updated_delisted = True

    # -----------------------------------------------------
    # Step 2. MAIN FILTER LOOP 
    # -----------------------------------------------------
    us_tickers = []
    for s in stocks:
        raw_ticker = str(s.get('ticker', '')).strip().upper()
        t = TICKER_FIXES.get(raw_ticker, raw_ticker.replace('.', '-'))
        t = str(t).strip().upper().replace('.', '-')
        if t in PERMANENT_BLACKLIST: continue
        if t in delisted_cache: continue
        us_tickers.append(t)
    
    us_tickers = list(set(us_tickers))
    tracker = HistoryTracker(HISTORY_FILE)
    
    # -----------------------------------------------------
    # Step 3. METADATA FETCHING
    # -----------------------------------------------------
    missing = [t for t in us_tickers if t not in local_cache and t not in delisted_cache]
    if missing:
        print(f"{C_YELLOW}Fetching metadata for {len(missing)} NEW items...{C_RESET}")
        
        for i, t in enumerate(missing):
            print(f"  > [{i+1}/{len(missing)}] Fetching: {t}") 
            
            res = fetch_meta_data_robust(t)
            
            if res: 
                local_cache[res['ticker']] = res
            else:
                print(f"{C_RED}  > {t} metadata 404/Not Found. Adding to DELISTED cache.{C_RESET}")
                delisted_cache[t] = {
                    'delisted': True, 
                    'last_checked': now.strftime("%Y-%m-%d"), 
                    'reason': 'Metadata 404'
                }
                updated_delisted = True
            time.sleep(0.75) 
        save_cache(CACHE_FILE, local_cache)

    # -----------------------------------------------------
    # Fetch SPY baseline for relative strength calculation
    # -----------------------------------------------------
    print(f"{C_CYAN}[#] Fetching SPY baseline...{C_RESET}")
    try:
        spy_hist = yf.Ticker("SPY").history(period="2y")['Close'].dropna()
    except Exception as e:
        print(f"{C_RED}[!] Error fetching SPY: {e}{C_RESET}")
        spy_hist = pd.Series(dtype=float)

    def _as_dataframe(obj: Any) -> pd.DataFrame:
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()

    # -----------------------------------------------------
    # Step 4. MARKET DATA FETCHING (Batch Mode)
    # -----------------------------------------------------
    valid_tickers = [t for t in us_tickers if t not in delisted_cache]
    market_data: pd.DataFrame = pd.DataFrame()
    use_cache = os.path.exists(MARKET_DATA_CACHE_FILE) and (time.time() - os.path.getmtime(MARKET_DATA_CACHE_FILE)) < CACHE_EXPIRY_SECONDS
    
    if use_cache:
        print(f"{C_CYAN}[#] Loading market data from cache...{C_RESET}")
        try: market_data = _as_dataframe(pd.read_pickle(MARKET_DATA_CACHE_FILE))
        except: use_cache = False

    if not use_cache:
        print(f"{C_YELLOW}[!] Downloading data for {len(valid_tickers)} tickers...{C_RESET}")
        CHUNK_SIZE = 40
        for i in range(0, len(valid_tickers), CHUNK_SIZE):
            batch = valid_tickers[i:i + CHUNK_SIZE]
            print(f"    > Processing Batch { (i//CHUNK_SIZE) + 1} ({len(batch)} tickers)...")
            try:
                if len(batch) == 1:
                    # Single ticker handling
                    batch_data: pd.DataFrame = _as_dataframe(yf.download(batch[0], period="2y", interval="1d", progress=False, auto_adjust=False))
                    if not batch_data.empty:
                        # Normalize format to match multi-index batch
                        batch_data.columns = pd.MultiIndex.from_product([[batch[0]], batch_data.columns])
                else:
                    # Multi ticker handling
                    batch_data: pd.DataFrame = _as_dataframe(yf.download(batch, period="2y", interval="1d", group_by='ticker', progress=False, threads=True, auto_adjust=False))

                if not batch_data.empty:
                    if market_data.empty: market_data = batch_data
                    else: market_data = pd.concat([market_data, batch_data], axis=1)
                
                # Increased sleep to prevent rate-limit "misses"
                if i + CHUNK_SIZE < len(valid_tickers): time.sleep(1.5) 
            except Exception as e:
                print(f"{C_RED}[!] Batch Error: {e}{C_RESET}")
                continue

        if not market_data.empty: market_data.to_pickle(MARKET_DATA_CACHE_FILE)

    # -----------------------------------------------------
    # Step 5. BUILD THE DATAFRAME
    # -----------------------------------------------------
    final_list = []

    for stock in stocks:
        raw_ticker = str(stock.get('ticker', '')).strip().upper()
        t = TICKER_FIXES.get(raw_ticker, raw_ticker.replace('.', '-'))
        t = str(t).strip().upper().replace('.', '-')
        if t in PERMANENT_BLACKLIST or t in delisted_cache: continue
        
        try:
            hist: pd.DataFrame = pd.DataFrame()
            
            # 1. EXTRACT FROM BATCH DATA
            if isinstance(market_data.columns, pd.MultiIndex):
                lvl0 = market_data.columns.get_level_values(0)
                lvl1 = market_data.columns.get_level_values(1)
                if t in lvl0:
                    # Standard (ticker, field) layout -- group_by='ticker'
                    selected = market_data[t]
                    if isinstance(selected, pd.DataFrame):
                        hist = selected[['High', 'Low', 'Close', 'Volume']].dropna()
                elif t in lvl1:
                    # Newer yfinance (field, ticker) layout
                    selected = market_data.xs(t, axis=1, level=1)
                    if isinstance(selected, pd.DataFrame):
                        hist = selected[['High', 'Low', 'Close', 'Volume']].dropna()
            else:
                if t in market_data.columns:
                    hist = market_data[[t]].dropna() # Fallback for single-column DF

            # 2. DOUBLE VERIFICATION / RETRY
            if hist.empty or len(hist) < 2:
                try:
                    # If batch failed, we need 20 days for a healthy 14-period RSI
                    retry_data: pd.DataFrame = _as_dataframe(yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=False))
                    if not retry_data.empty:
                        hist = retry_data
                        if isinstance(hist.columns, pd.MultiIndex):
                            hist.columns = hist.columns.droplevel(1)
                except:
                    pass

            # 3. SAFETY CHECK & RSI CALCULATION
            if hist.empty:
                print(f"{C_RED}  > {t} confirmed NO DATA. Adding to DELISTED cache.{C_RESET}")
                delisted_cache[t] = {
                    'delisted': True, 
                    'last_checked': now.strftime("%Y-%m-%d"), 
                    'reason': 'No Price Data'
                }
                updated_delisted = True
                continue

            # INITIAL DATA GATHERING
            info = local_cache.get(t, {})
            if info.get('currency') not in ['USD', None, '']: continue

            # Extract current price safely
            curr_p = float(hist['Close'].iloc[-1])
            if isinstance(curr_p, pd.Series): curr_p = curr_p.iloc[0]

            # 4. CALCULATE RSI & STOCHASTIC
            if not hist.empty and len(hist) >= 15:
                rsi_val = calculate_rsi(hist['Close'])
                stoch_k, stoch_d = calculate_stochastic(hist, k_period=14, d_period=3)

                # --- NEW: EMA Calculations ---
                ema9 = hist['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
                ema21 = hist['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
                ema50 = hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
                
                # Bullish: 9 > 21 > 50 | Bearish: 9 < 21 < 50
                if ema9 > ema21 > ema50: crossover_signal = 1
                elif ema9 < ema21 < ema50: crossover_signal = -1
                else: crossover_signal = 0
                
                raw_sctr = calculate_raw_sctr(hist, t)
                raw_ibd = calculate_raw_ibd_rs(hist, t)
                raw_spy = calculate_raw_spy_rs(hist, spy_hist, t)
            else:
                # Fallback if < 50 days of data
                rsi_val, stoch_k, stoch_d, ema9, ema21, ema50, crossover_signal = 0, 50.0, 50.0, 0, 0, 0, 0
                raw_sctr, raw_ibd, raw_spy = -9999.0, -9999.0, -9999.0

            clean_hist = hist['Volume'] 
            actual_vol_days = min(len(clean_hist), AVG_VOLUME_DAYS)
            avg_v = clean_hist.tail(actual_vol_days).mean()
            curr_v = int(hist['Volume'].iloc[-1])
            if isinstance(avg_v, pd.Series): avg_v = avg_v.iloc[0]
            if isinstance(curr_v, pd.Series): curr_v = curr_v.iloc[0]
            
            if curr_p < MIN_PRICE or avg_v < MIN_AVG_VOLUME: continue

            name = str(info.get('name', t)).replace('"', '').strip()[:NAME_MAX_WIDTH]
            cur_m = int(stock.get('mentions') or 0)
            old_m = int(stock.get('mentions_24h_ago') or 0)
            
            v_now_raw = hist['Volume'].iloc[-1]
            if isinstance(v_now_raw, pd.Series):
                v_now_raw = v_now_raw.iloc[0]
            v_now = float(v_now_raw)
            v_avg = float(avg_v)

            m_perc = int(((cur_m - old_m) / (old_m if old_m > 0 else 1) * 100))
            s_perc = int((v_now / v_avg * 100)) if v_avg > 0 else 0
            
            try: mcap = float(info.get('mcap', 0) or 0)
            except: mcap = 0
            
            log_mcap = math.log(mcap if mcap > 0 else 10**9, 10)
            squeeze_score = (cur_m * s_perc) / max(log_mcap, 1)

            rank_now = int(stock.get('rank') or 0)
            rank_old = int(stock.get('rank_24h_ago') or 0)
            rank_plus = (rank_old - rank_now) if rank_old != 0 else 0

            upvotes_raw = stock.get('upvotes')
            current_upvotes = int(upvotes_raw) if upvotes_raw is not None else 0
            
            conviction = (current_upvotes / cur_m) if cur_m > 0 else 0
            safe_surge = s_perc if s_perc > 10 else 10 
            efficiency = rank_plus / (safe_surge / 100.0) 

            # --- IMPROVED PRICE & DAY% CALCULATION ---
            day_chg_pct = 0.0
            
            try:
                if not hist.empty and len(hist) >= 2:
                    curr_p = float(hist['Close'].iloc[-1])
                    prev_day_close = float(hist['Close'].iloc[-2])
                    if prev_day_close > 0:
                        day_chg_pct = ((curr_p - prev_day_close) / prev_day_close) * 100
                
                    # 2. FORCE LIVE DATA FETCH (Always runs)
                    live_dat = yf.Ticker(t)
                    inf = live_dat.info

                    # Update Price
                    live_p = inf.get('currentPrice') or inf.get('regularMarketPrice')
                    prev_p = inf.get('previousClose') or inf.get('regularMarketPreviousClose')

                    if live_p and prev_p:
                        curr_p = float(live_p)
                        day_chg_pct = ((live_p - prev_p) / prev_p) * 100

                    live_v = inf.get('regularMarketVolume') or inf.get('volume')
                    if live_v:
                        curr_v = int(live_v)

            except Exception as e:
                pass

            # --- RECONCILE EMAs WITH THE FINAL DISPLAYED PRICE ---
            # ema9/21/50 above were computed from hist['Close'].iloc[-1], which can be
            # stale relative to curr_p (the live quote we may have just fetched, or the
            # close from a market_data cache that's reused for up to 12h). That mismatch
            # is what made the Trend arrows (Price vs EMA9 vs EMA21 vs EMA50) look wrong
            # intermittently. Recompute the EMAs with curr_p standing in for "today's
            # close" so Price is always self-consistent with the EMAs it's compared to.
            if not hist.empty and len(hist) >= 15:
                close_live = hist['Close'].copy()
                close_live.iloc[-1] = curr_p
                ema9 = close_live.ewm(span=9, adjust=False).mean().iloc[-1]
                ema21 = close_live.ewm(span=21, adjust=False).mean().iloc[-1]
                ema50 = close_live.ewm(span=50, adjust=False).mean().iloc[-1]

                if ema9 > ema21 > ema50: crossover_signal = 1
                elif ema9 < ema21 < ema50: crossover_signal = -1
                else: crossover_signal = 0

            s_perc = int((curr_v / avg_v * 100)) if avg_v > 0 else 0
            
            final_list.append({
                "Rank": rank_now, 
                "Name": name, 
                "Sym": t, 
                "Rank+": rank_plus,
                "Price": float(curr_p), 
                "Day%": float(day_chg_pct),
                "CurVol": int(curr_v),
                "AvgVol": int(avg_v), 
                "Surge": s_perc,
                "MENT": cur_m, 
                "Mnt%": m_perc, 
                "Type": info.get('type', 'EQUITY'),
                "Upvotes": current_upvotes, 
                "Meta": info.get('meta', '-'),
                "Desc": info.get('description', ''), 
                "Squeeze": squeeze_score,
                "MCap": mcap, 
                "Conv": conviction, 
                "Eff": efficiency,
                "Accel": 0, 
                "Upv+": 0, 
                "Velocity": 0, 
                "Streak": 0, 
                "Rolling": 0, 
                "History": "New",
                "RSI": rsi_val,
                "Stoch_K": stoch_k,
                "Stoch_D": stoch_d,
                "EMA9": ema9,
                "EMA21": ema21,
                "EMA50": ema50,
                "Trend": crossover_signal,
                "Raw_SCTR": raw_sctr,
                "Raw_IBD": raw_ibd,
                "IBD_RS": 0.0,
                "Raw_SPY": raw_spy,
                "SPY_RS": 0.0,
                "GI": gi_map.get(t) if gi_map.get(t) is not None else gi_map.get(t.replace('-', '.'))
            })
            
        except Exception as e:
            print(f"{C_RED}[!] Error processing {t}: {e}{C_RESET}")
            continue

    if updated_delisted:
        print(f"{C_GREEN}[+] Saving updated delisted cache...{C_RESET}")
        save_cache(DELISTED_CACHE_FILE, delisted_cache)

    # -----------------------------------------------------
    # Step 6. SCORING & SAVING
    # -----------------------------------------------------
    df = pd.DataFrame(final_list)
    if not df.empty and 'Sym' in df.columns:
        df = df.drop_duplicates(subset=['Sym'], keep='first')

    if not df.empty and 'GI' in df.columns:
        gi_hits = int(df['GI'].notna().sum())
        print(f"{C_CYAN}[#] GI coverage in output: {gi_hits}/{len(df)} tickers.{C_RESET}")

    if not df.empty:
        cols = ['Rank+', 'Surge', 'Mnt%', 'Upvotes', 'Accel', 'Upv+', 'MENT']
        weights = {'Rank+': 1.1, 'Surge': 1.1, 'Mnt%': 0.7, 'Upvotes': 1.0, 'Accel': 1.2, 'Upv+': 1.0, 'MENT': 0.8}
        
        # Ensure columns exist to prevent errors
        for col in cols:
            if col not in df.columns: df[col] = 0

        for col in cols:
            clean_series = df[col].clip(lower=0).astype(float)
            log_data = np.log1p(clean_series)
            mean = log_data.mean(); std = log_data.std(ddof=0)
            df[f'z_{col}'] = 0 if std == 0 else (log_data - mean) / std

        df['Master_Score'] = 0
        for col in cols:
            df['Master_Score'] += df[f'z_{col}'].clip(lower=0) * weights.get(col, 1.0)
        
        if 'Squeeze' not in df.columns: df['Squeeze'] = 0
        sq_series = df['Squeeze'].clip(lower=0).astype(float)
        log_sq = np.log1p(sq_series)
        mean_sq = log_sq.mean(); std_sq = log_sq.std(ddof=0)

        df['z_Squeeze'] = 0 if std_sq == 0 else (log_sq - mean_sq) / std_sq
        df['Heat'] = df['Master_Score']

        # --- TRUE PERCENTILE HELPER ---
        def true_percentile(series):
            ranks = series.rank(method='min')
            n = len(ranks)
            if n <= 1: return pd.Series(99.9, index=series.index)
            return ((ranks - 1) / (n - 1) * 99.9).round(1)

        # --- RANK THE SCTR UNIVERSE (UPDATED) ---
        df['SCTR'] = 0.0
        if 'Raw_SCTR' in df.columns:
            valid_mask = df['Raw_SCTR'] > -9000.0
            if valid_mask.any():
                df.loc[valid_mask, 'SCTR'] = true_percentile(df.loc[valid_mask, 'Raw_SCTR'])

        # --- RANK THE IBD RS UNIVERSE ---
        df['IBD_RS'] = 0.0
        if 'Raw_IBD' in df.columns:
            valid_mask_ibd = df['Raw_IBD'] > -9000.0
            if valid_mask_ibd.any():
                df.loc[valid_mask_ibd, 'IBD_RS'] = true_percentile(df.loc[valid_mask_ibd, 'Raw_IBD'])

        # --- RANK THE SPY RS UNIVERSE ---
        df['SPY_RS'] = 0.0
        if 'Raw_SPY' in df.columns:
            valid_mask_spy = df['Raw_SPY'] > -9000.0
            if valid_mask_spy.any():
                df.loc[valid_mask_spy, 'SPY_RS'] = true_percentile(df.loc[valid_mask_spy, 'Raw_SPY'])

        tracker.save(df)

        for index, row in df.iterrows():
            m = tracker.get_metrics(
                row['Sym'], 
                row['Price'], 
                row['MENT'], 
                row['Rank+'], 
                row['Upvotes'], 
                row['Surge'],
                row['GI']
            )
            
            df.at[index, 'Accel'] = m.get('accel', 0)
            df.at[index, 'Upv+'] = m.get('upv_chg', 0)
            df.at[index, 'Velocity'] = m.get('vel', 0)
            df.at[index, 'Streak'] = m.get('streak', 1)
            df.at[index, 'Rolling'] = m.get('streak', 0) 

            histories = m.get('hist', {})
            df.at[index, 'h_rank'] = histories.get('rank', '')
            df.at[index, 'h_rank_plus'] = histories.get('rank_plus', '')
            df.at[index, 'h_price'] = histories.get('price', '')
            df.at[index, 'h_ment'] = histories.get('ment', '')
            df.at[index, 'h_upvotes'] = histories.get('upvotes', '')
            df.at[index, 'h_velocity'] = histories.get('velocity', '') 
            df.at[index, 'h_accel'] = histories.get('accel', '')
            df.at[index, 'h_streak'] = histories.get('streak', '')
            df.at[index, 'h_upv_plus'] = histories.get('upv_plus', '')
            df.at[index, 'h_eff'] = histories.get('eff', '')
            df.at[index, 'h_conv'] = histories.get('conv', '')
            df.at[index, 'h_surge'] = histories.get('surge', '')
            df.at[index, 'h_mnt_perc'] = histories.get('mnt_perc', '')
            df.at[index, 'h_squeeze'] = histories.get('squeeze', '')
            df.at[index, 'h_heat'] = histories.get('master_score', '')
            df.at[index, 'h_spy_rs'] = histories.get('spy_rs', '')
            df.at[index, 'h_gi'] = histories.get('gi', '')

        # --- STEP 3: FLUSH TO DISK ---
        tracker.flush()
        
        return df
    
    return pd.DataFrame()


# ==============================================================================
#                               SECTION 5: DATA INGESTION (API)
# ==============================================================================
def get_all_trending_stocks():
    all_results, page = [], 1
    max_retries = 3
    print(f"{C_CYAN}--- API: Fetching list of trending stocks ---{C_RESET}")
    
    # Using a common browser header helps prevent 403 Forbidden errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    while True:
        success = False
        for attempt in range(max_retries):
            try:
                # Increased timeout to 20s as Ape Wisdom can be slow under load
                r = session.get(
                    f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}", 
                    headers=headers,
                    timeout=20
                )
                
                if r.status_code == 200:
                    data = r.json()
                    results = data.get('results', [])
                    
                    if not results:
                        # End of pages reached
                        return all_results
                    
                    all_results.extend(results)
                    print(f"  > Page {page} fetched ({len(results)} items)")
                    page += 1
                    success = True
                    break # Break retry loop, go to next page
                
                elif r.status_code == 429:
                    print(f"{C_RED}Rate limited (429). Waiting to retry...{C_RESET}")
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"{C_YELLOW}Warning: Page {page} returned status {r.status_code}{C_RESET}")
                    time.sleep(2)

            except Exception as e:
                print(f"{C_RED}Error fetching page {page} (Attempt {attempt+1}/{max_retries}): {e}{C_RESET}")
                time.sleep(5)
        
        # If we exhausted retries for a page without success, return what we have
        if not success:
            print(f"{C_RED}Critical: Failed to fetch page {page} after {max_retries} attempts.{C_RESET}")
            break

    return all_results


# ==============================================================================
#                        SECTION 6: FRONTEND GENERATION (HTML/CSS/JS)
# ==============================================================================
def export_interactive_html(df):
    # Prebind so exception logging is always safe even if setup fails early.
    C_RED = "#ff4d5a"
    try:
        export_df = df.copy()
        if not os.path.exists(PUBLIC_DIR): os.makedirs(PUBLIC_DIR)

        # --- TOOLTIP CONFIGURATION ---
        # 1. Use class="d-tooltip" for custom CSS styling (Instant, Black Box)
        # 2. tabindex="0" ensures it works on Mobile taps
        def with_hist(val_str, history_str):
            if not history_str or history_str == "New" or history_str == "": 
                return val_str
            safe_hist = history_str.replace('"', '&quot;')
            return f'<span class="d-tooltip" data-tooltip="{safe_hist}" tabindex="0">{val_str}</span>'

        for c in ['Accel', 'Velocity', 'Rolling', 'Squeeze', 'Upvotes', 'Rank+', 'Surge', 'Mnt%', 'Master_Score', 'z_Upvotes', 'z_Surge', 'z_Squeeze']:
            if c not in export_df.columns: export_df[c] = 0

        export_df.rename(columns={
            'Accel': 'Acc', 
            'Velocity': 'Vel', 
            'Rolling': 'Strk', 
            'Squeeze': 'Sqz',
            'Upvotes': 'Upvs',
            'Surge': 'Srg'
        }, inplace=True)

        export_df = export_df.astype(object)

        def color_span(text, color_hex): return f'<span style="color: {color_hex}; font-weight: bold;">{text}</span>'
        def format_vol(v):
            try:
                v = float(v)
                if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
                if v >= 1_000: return f"{v/1_000:.0f}K"
                return str(int(v))
            except: return "0"

        C_GREEN, C_YELLOW, C_RED, C_CYAN, C_WHITE = "#00d97e", "#f5c518", "#ff4d5a", "#00c8d7", "#e8e8f0"
        
        if 'AvgVol' not in export_df.columns: export_df['AvgVol'] = 0
        if 'CurVol' not in export_df.columns: export_df['CurVol'] = 0
        
        export_df['Vol_Display'] = export_df['AvgVol'].apply(format_vol)
        export_df['CurVol_Disp'] = export_df['CurVol'].apply(format_vol)
        export_df['Type_Tag'] = 'STOCK'

        if 'MENT' not in export_df.columns: export_df['MENT'] = 0

        def clean_vol_num(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return 0.0
            if isinstance(x, (int, float, np.integer, np.floating)):
                return float(x)
            
            # If it's a string (like "65.0M" or "$1,200")
            s = str(x).upper().replace(',', '').replace('$', '').replace('%', '').strip()
            if s == "" or s == "-": 
                return 0.0
            
            try:
                if 'M' in s:
                    return float(s.replace('M', '')) * 1_000_000
                if 'K' in s:
                    return float(s.replace('K', '')) * 1_000
                if 'B' in s:
                    return float(s.replace('B', '')) * 1_000_000_000
                return float(s)
            except ValueError:
                return 0.0

       # 1. SMART COLUMN FINDER
        # We prioritize raw numeric columns ('CurVol', 'AvgVol') set in filter_and_process
        clean_columns = [c.strip() for c in export_df.columns]
        col_map = dict(zip(clean_columns, export_df.columns))

        # Priority search for Current Volume
        current_vol_col = None
        for candidate in ['CurVol', 'VOL', 'Vol', 'Volume', 'Current Volume']:
            if candidate in col_map:
                current_vol_col = col_map[candidate]
                break

        # Priority search for Average Volume
        avg_vol_col = None
        for candidate in ['AvgVol', 'VOL(30)', 'Vol(30)', 'Avg Volume', 'Average Volume']:
            if candidate in col_map:
                avg_vol_col = col_map[candidate]
                break

        # 2. CALCULATE SURGE (Direct & Fresh)
        # This forces the math: Current Volume / 30-Day Average
        if 'CurVol' in export_df.columns and 'AvgVol' in export_df.columns:
            # Use the raw numbers directly (fastest and most accurate)
            c_series = export_df['CurVol'].astype(float)
            a_series = export_df['AvgVol'].astype(float).replace(0, 1) # Prevent div/0 errors
        
            surge_calc = (c_series / a_series) * 100
            
            export_df['Srg'] = surge_calc.fillna(0).astype(float)
            
            export_df['SRG'] = export_df['Srg'].astype(int).astype(str) + '%'
            
            export_df['Srg'] = export_df['Srg'].astype(object)

        elif current_vol_col and avg_vol_col:
            # Fallback: Parse text columns if raw data is missing
            curr_series = export_df[current_vol_col].apply(clean_vol_num)
            avg_series  = export_df[avg_vol_col].apply(clean_vol_num)
            
            surge_calc = (curr_series / (avg_series + 0.000001)) * 100
            surge_calc = surge_calc.replace([np.inf, -np.inf], 0).fillna(0)
            
            export_df['Srg'] = surge_calc.astype(float)
            export_df['SRG'] = surge_calc.astype(int).astype(str) + '%'
            export_df['Srg'] = export_df['Srg'].astype(object)
        else:
            print(f"[!] Warning: Missing volume columns for Surge calculation.")
            export_df['Srg'] = 0.0
            export_df['SRG'] = "0%"
            export_df['Srg'] = export_df['Srg'].astype(object)

        for index, row in export_df.iterrows():
            v_raw = row.get('CurVol_Disp', '0')
            export_df.at[index, 'CurVol_Disp'] = f'<div style="text-align: right; padding-right: 10px; color: #e8e8f0; font-weight:600;">{v_raw}</div>'
            
            avg_v_raw = row.get('Vol_Display', '0')
            export_df.at[index, 'Vol_Display'] = f'<div style="text-align: right; padding-right: 10px; color: #8888a0;">{avg_v_raw}</div>'
            m_val = row.get('MENT', 0)
            z_score = row.get('z_MENT', 0)
            
            if z_score >= 2.0: m_clr = "#f5c518"
            elif z_score >= 1.0: m_clr = "#00d97e"
            else: m_clr = "#e8e8f0"  
            
            export_df.at[index, 'MENT'] = color_span(f"{int(m_val)}", m_clr)

            # --- 1. VELOCITY (Vel) ---
            v_val = row.get('Vel', 0)
            v_hist = row.get('h_velocity', '') 
            v_color = C_GREEN if v_val > 0 else (C_RED if v_val < 0 else "#666")
            v_str = color_span(f"{v_val:+d}", v_color)
            export_df.at[index, 'Vel'] = with_hist(v_str, v_hist)
            
            # --- 2. ACCELERATION (Acc) ---
            ac_val = row.get('Acc', 0)
            ac_hist = row.get('h_accel', '')
            if ac_val >= 5: ac_clr = "#c45cf6"
            elif ac_val > 0: ac_clr = "#00c8d7"
            elif ac_val < 0: ac_clr = "#ff4d5a"
            else: ac_clr = "#555568"
            export_df.at[index, 'Acc'] = with_hist(color_span(f"{ac_val:+d}", ac_clr), ac_hist)

            # --- 3. EFFICIENCY (Eff) ---
            eff_val = float(row.get('Eff', 0)) 
            eff_hist = row.get('h_eff', '') 
            if eff_val >= 1.0: eff_clr = "#00d97e"       # Strong Green
            elif eff_val >= 0.5: eff_clr = "#f5c518"     # Yellow
            elif eff_val >= 0.1: eff_clr = "#e8e8f0"     # Neutral
            elif eff_val > -0.1: eff_clr = "#555568"     # Grey (Flat/Zero)
            else: eff_clr = "#ff4d5a"                    # Red (Negative)
            export_df.at[index, 'Eff'] = with_hist(color_span(f"{eff_val:.1f}", eff_clr), eff_hist)

            # --- 4. CONVICTION (Conv) ---
            conv_val = float(row.get('Conv', 0)) 
            conv_hist = row.get('h_conv', '') 
            conv_clr = "#f5c518" if conv_val > 1.0 else "#e8e8f0"
            export_df.at[index, 'Conv'] = with_hist(color_span(f"{conv_val:.1f}x", conv_clr), conv_hist)

            # --- 5. UPVOTE CHANGE (Upv+) ---
            upchg_val = row.get('Upv+', 0)
            upchg_hist = row.get('h_upv_plus', '') 
            upchg_clr = C_GREEN if upchg_val > 0 else (C_RED if upchg_val < 0 else "#666")
            export_df.at[index, 'Upv+'] = with_hist(color_span(f"{upchg_val:+d}", upchg_clr), upchg_hist)

            # --- 6. STREAK (Strk) - CALCULATION & FORMATTING ---
            rank_change_str = str(row.get('RANK+', '0')).replace(',', '')
            if '▲' in rank_change_str:
                rank_delta = float(rank_change_str.replace('▲', '').strip())
            elif '▼' in rank_change_str:
                rank_delta = -abs(float(rank_change_str.replace('▼', '').strip()))
            else:
                try: rank_delta = float(rank_change_str)
                except: rank_delta = 0.0
            old_streak = float(row.get('Strk', 0))
            if rank_delta > 0:
                new_streak = 1 if old_streak < 0 else old_streak + 1
            elif rank_delta < 0:
                new_streak = -1 if old_streak > 0 else old_streak - 1
            else:
                new_streak = old_streak
            trend_val = int(new_streak)
            trend_hist = row.get('h_streak', '') 
            sig_text = f"{trend_val:+d}"
            if trend_val >= 3: sig_color = "#00d97e"   
            elif trend_val > 0: sig_color = "#6eddb0"  
            elif trend_val <= -2: sig_color = "#ff4d5a" 
            else: sig_color = "#e8e8f0"              
            export_df.at[index, 'Strk'] = with_hist(color_span(sig_text, sig_color), trend_hist)

            # --- 7. HEAT SCORE ---
            score = float(row.get('Master_Score', 0))
            heat_hist = row.get('h_heat', '') 
            if score > 10: h_clr = "#ff4d5a"
            elif score > 5: h_clr = "#ff8840"
            elif score > 2: h_clr = "#f5c518"
            else: h_clr = "#555568"
            heat_span = f'<span style="color:{h_clr}; font-weight:bold;">{score:.1f}</span>'
            export_df.at[index, 'Heat'] = with_hist(heat_span, heat_hist)

            # --- 8. NAME & LOGO ---
            t_raw = row['Sym']
            clean_ticker = t_raw.replace('-', '.')
            
            logo_src = get_cached_logo(clean_ticker) 
            
            exchange_name = row.get("exchange", "Unknown")

            html_name = (
                f'<div class="symbol-container" style="display: flex; align-items: center; gap: 8px;" '
                f'onmouseenter="loadSymbolProfile(\'{clean_ticker}\', \'profile-{index}\', \'{exchange_name}\', event)" '
                f'onmouseleave="hideSymbolProfile(\'profile-{index}\')">' 
                f'<img src="{logo_src}" style="width: 22px; height: 22px; border-radius: 50%; background: #1e1e2a; flex-shrink: 0; object-fit: contain;" '
                f'onerror="this.src=\'https://s3-symbol-logo.tradingview.com/indices/nasdaq-100.svg\'">'
                f'<span class="text-content"><b>{row.get("Name", clean_ticker)}</b></span>'
                f'<div id="profile-{index}" class="chart-popup"></div>'
                f'</div>'
            )
            export_df.at[index, 'Name'] = html_name

            # --- 9. RANK+ ---
            r_val = row.get('Rank+', 0)
            r_hist = row.get('h_rank_plus', '')
            
            if r_val != 0:
                r_color = C_GREEN if r_val > 0 else C_RED
                r_arrow = "▲" if r_val > 0 else "▼"
                r_str = color_span(f"{r_val} {r_arrow}", r_color)
                export_df.at[index, 'Rank+'] = with_hist(r_str, r_hist)
            else:
                export_df.at[index, 'Rank+'] = with_hist('<span style="color:#555568">0</span>', r_hist)

            # --- 10. RANK ---
            rank_val = str(row.get('Rank', 0))
            rank_hist = row.get('h_rank', '')
            export_df.at[index, 'Rank'] = with_hist(rank_val, rank_hist)

            # --- 11. SURGE (SRG) - Linear Color Logic ---
            srg_raw = float(row.get('Srg', 0))
            srg_val_str = f"{int(srg_raw)}%"
            srg_hist = row.get('h_surge', '')
            
            # Linear thresholding: makes the dashboard much more intuitive
            if srg_raw >= 300:
                srg_clr = "#f5c518"  # Yellow: Extreme Surge (3x+ average)
            elif srg_raw >= 100:
                srg_clr = "#00d97e"  # Green: Above Average (1x+ average)
            elif srg_raw >= 50:
                srg_clr = "#e8e8f0"  # Neutral: Significant Progress
            else:
                srg_clr = "#555568"  # Grey: Low relative volume
                
            export_df.at[index, 'Srg'] = with_hist(color_span(srg_val_str, srg_clr), srg_hist)

            mnt_raw = row.get('Mnt%', 0)
            mnt_val_str = f"{int(mnt_raw)}%"
            mnt_hist = row.get('h_mnt_perc', '')
            mnt_z = row.get('z_Mnt%', 0)
            mnt_clr = C_YELLOW if mnt_z >= 2.0 else (C_GREEN if mnt_z >= 1.0 else C_WHITE)
            export_df.at[index, 'Mnt%'] = with_hist(color_span(mnt_val_str, mnt_clr), mnt_hist)

            # --- 12. SQUEEZE ---
            sq_val = int(row.get('Sqz', 0))
            sq_hist = row.get('h_squeeze', '') 
            sq_z = row.get('z_Squeeze', 0)
            sq_color = C_CYAN if sq_z > 1.5 else C_WHITE
            export_df.at[index, 'Sqz'] = with_hist(color_span(sq_val, sq_color), sq_hist)
            
            # --- 13. UPVOTES ---
            upvs_val = row.get('Upvs', 0)
            upvs_hist = row.get('h_upvotes', '')
            z_up = row.get('z_Upvotes', 0)
            upvs_clr = C_GREEN if z_up > 1.5 else C_WHITE
            upvs_str = color_span(upvs_val, upvs_clr)
            export_df.at[index, 'Upvs'] = with_hist(upvs_str, upvs_hist)
            
            # --- 14. MENTIONS ---
            ment_val = str(row.get('MENT', 0))
            ment_hist = row.get('h_ment', '')
            export_df.at[index, 'MENT'] = with_hist(ment_val, ment_hist)

            # --- STOCHASTIC VISUALIZATION ---
            stoch_k_v = float(row.get('Stoch_K', 50.0))
            stoch_d_v = float(row.get('Stoch_D', 50.0))

            stoch_tooltip = f"%K (5): {stoch_k_v:.1f}&#10;%D (1): {stoch_d_v:.1f}"
            
            # Color Logic: <= 20 Green, >= 80 Red
            if stoch_k_v <= 20.0:
                stoch_clr = "#00d97e" # Green (Oversold)
            elif stoch_k_v >= 80.0:
                stoch_clr = "#ff4d5a" # Red (Overbought)
            else:
                stoch_clr = "#e8e8f0" # Neutral

            # Create the HTML span WITHOUT the tooltip
            stoch_str = f'<span style="color:{stoch_clr}; font-weight:bold;">{stoch_k_v:.0f}</span>'
            export_df.at[index, 'STOCH'] = stoch_str

            # --- GI SCORE LOGIC ---
            gi_raw = row.get('GI', None)
            gi_hist = row.get('h_gi', '')
            try:
                gi_val = float(gi_raw)
            except (TypeError, ValueError):
                gi_val = None

            if gi_val is None or np.isnan(gi_val):
                export_df.at[index, 'GI'] = with_hist('<span style="color:#666; font-weight:600;">--</span>', gi_hist)
            else:
                if gi_val >= 75:
                    gi_clr = "#00d97e"
                elif gi_val >= 60:
                    gi_clr = "#6eddb0"
                elif gi_val >= 43:
                    gi_clr = "#f5c518"
                elif gi_val >= 28:
                    gi_clr = "#ff9f4f"
                else:
                    gi_clr = "#ff4d5a"
                export_df.at[index, 'GI'] = with_hist(color_span(f"{gi_val:.1f}", gi_clr), gi_hist)

            # --- RSI COLOR LOGIC ---
            rsi_raw = float(row.get('RSI', 0))
            if rsi_raw >= 70: 
                rsi_clr = "#ff4d5a" # Red (Overbought)
            elif rsi_raw <= 30 and rsi_raw > 0: 
                rsi_clr = "#00d97e" # Green (Oversold)
            else: 
                rsi_clr = "#e8e8f0"
            
            # Render RSI
            rsi_str = color_span(f"{rsi_raw:.1f}", rsi_clr)
            export_df.at[index, 'RSI'] = rsi_str

            # --- 3-ARROW EMA SYSTEM ---
            p_clean = float(row.get('Price', 0))
            ema9 = float(row.get('EMA9', 0))
            ema21 = float(row.get('EMA21', 0))
            ema50 = float(row.get('EMA50', 0))

            if ema9 == 0 and ema21 == 0 and ema50 == 0:
                # Sentinel from the <15-day-history fallback in filter_and_process -
                # there's no real trend to show, so don't fake a bullish ▲▲▲.
                export_df.at[index, 'Trend'] = '<div style="font-size:11px; font-weight:bold; text-align:center; color:#555568;">N/A</div>'
            else:
                # Determine individual arrow colors
                a1_clr = "#00d97e" if p_clean >= ema9 else "#ff4d5a"
                a2_clr = "#00d97e" if ema9 >= ema21 else "#ff4d5a"
                a3_clr = "#00d97e" if ema21 >= ema50 else "#ff4d5a"

                # Determine individual arrow directions
                a1_sym = "▲" if p_clean >= ema9 else "▼"
                a2_sym = "▲" if ema9 >= ema21 else "▼"
                a3_sym = "▲" if ema21 >= ema50 else "▼"

                # Construct the final 3-part badge
                trend_str = (
                    f'<span style="color:{a1_clr};">{a1_sym}</span>'
                    f'<span style="color:{a2_clr};">{a2_sym}</span>'
                    f'<span style="color:{a3_clr};">{a3_sym}</span>'
                )
                export_df.at[index, 'Trend'] = f'<div style="font-size:11px; font-weight:bold; letter-spacing:1px; text-align:center;">{trend_str}</div>'

            # --- SCTR COLOR LOGIC ---
            sctr_global = float(row.get('SCTR', 0.0))
            sctr_raw_math = float(row.get('Raw_SCTR', -9999.0)) # Grab the hidden raw math
            
            if sctr_global >= 80: sctr_clr = "#00d97e"       
            elif sctr_global >= 40: sctr_clr = "#f5c518"     
            elif sctr_global > 0: sctr_clr = "#ff4d5a"       
            else: sctr_clr = "#555568"                    
            
            # We add data-global and data-raw attributes so JavaScript can read them
            sctr_str = f'<span class="sctr-val" data-global="{sctr_global:.1f}" data-raw="{sctr_raw_math:.1f}" style="color:{sctr_clr}; font-weight:bold;">{sctr_global:.1f}</span>'
            export_df.at[index, 'SCTR'] = sctr_str

            # --- IBD RS COLOR LOGIC ---
            ibd_global = float(row.get('IBD_RS', 0.0))
            ibd_raw_math = float(row.get('Raw_IBD', -9999.0))
            
            if ibd_global >= 80: ibd_clr = "#00d97e"       
            elif ibd_global >= 40: ibd_clr = "#f5c518"     
            elif ibd_global > 0: ibd_clr = "#ff4d5a"       
            else: ibd_clr = "#555568"                     
            
            ibd_str = f'<span class="ibd-val" data-global="{ibd_global:.1f}" data-raw="{ibd_raw_math:.1f}" style="color:{ibd_clr}; font-weight:bold;">{ibd_global:.1f}</span>'
            export_df.at[index, 'IBD_RS'] = ibd_str

            # --- SPY RS COLOR LOGIC ---
            spy_global = float(row.get('SPY_RS', 0.0))
            spy_raw_math = float(row.get('Raw_SPY', -9999.0))
            
            if spy_global >= 80: spy_clr = "#00d97e"       
            elif spy_global >= 40: spy_clr = "#f5c518"     
            elif spy_global > 0: spy_clr = "#ff4d5a"       
            else: spy_clr = "#555568"                      
            
            spy_str = f'<span class="spy-val" data-global="{spy_global:.1f}" data-raw="{spy_raw_math:.1f}" style="color:{spy_clr}; font-weight:bold;">{spy_global:.1f}</span>'
            export_df.at[index, 'SPY_RS'] = with_hist(spy_str, row.get('h_spy_rs', ''))

            # --- 15. Percent Change ---
            d_val = row.get('Day%', 0)
            d_clr = "#00d97e" if d_val > 0 else ("#ff4d5a" if d_val < 0 else "#555568")
            export_df.at[index, 'Day%'] = color_span(f"{d_val:+.1f}%", d_clr)

            # --- 16. ETF BADGE & META ---
            is_fund = row.get('Type', 'EQUITY') == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
            meta_val = row.get('Meta', '-')
            if is_fund:
                badge = '<span style="background:rgba(196,92,246,0.18); color:#c45cf6; border:1px solid rgba(196,92,246,0.4); padding:1px 5px; border-radius:4px; font-size:9px; font-weight:800; letter-spacing:0.08em; margin-right:5px; vertical-align:middle; font-family:\'Inter\',sans-serif;">ETF</span>'
            else:
                badge = ""
            
            export_df.at[index, 'Meta'] = f"{badge}{color_span(meta_val, C_WHITE)}"
            export_df.at[index, 'Type_Tag'] = 'ETF' if is_fund else 'STOCK'
            
            # --- 17. SYMBOL & PRICE ---
            t = row['Sym']
            tv_ticker = t.replace('-', '.')
            export_df.at[index, 'Sym'] = (
                f'<div class="symbol-container" '
                f'onmouseenter="loadMiniChart(\'{tv_ticker}\', \'{index}\', \'{row.get("exchange", "")}\', event)" '
                f'onmouseleave="hideSymbolProfile(\'chart-tooltip-{index}\')">' 
                f'<a href="https://www.tradingview.com/chart/?symbol={tv_ticker}" target="_blank" style="color: #3b9eff; text-decoration: none; font-weight:700;">{t}</a>'
                f'<div id="chart-tooltip-{index}" class="chart-popup"></div>'
                f'</div>'
            )
            
            p_clean = row.get('Price', 0)
            
            export_df.at[index, 'Price'] = f'<div style="text-align: right; padding-right: 8px; color: #e8e8f0;">${p_clean:.2f}</div>'

            vol_raw = export_df.at[index, 'Vol_Display']
            export_df.at[index, 'Vol_Display'] = f'<div style="text-align: right; color: #8888a0;">{vol_raw}</div>'

        export_df.rename(columns={'Meta': 'INDUSTRY', 'Vol_Display': 'VOL(30)', 'CurVol_Disp': 'VOL'}, inplace=True)

        cols = [
            'Rank', 'Rank+', 'Heat', 'Name', 'Sym', 'Price', 'Day%', 'Acc', 'Eff', 'Conv', 'Upvs', 
            'Upv+', 'VOL', 'VOL(30)', 'Srg', 'Vel', 'Strk', 'MENT', 'Mnt%', 'Sqz', 'INDUSTRY',
            'Trend', 'GI', 'RSI', 'STOCH', 'SCTR', 'IBD_RS', 'SPY_RS', 'Type_Tag', 'AvgVol', 'MCap'
        ]
        for c in cols:
            if c not in export_df.columns:
                export_df[c] = 0

        # --- 1. GENERATE RAW HTML TABLE ---
        raw_table = export_df[cols].to_html(classes='table table-dark table-hover', index=False, escape=False)

        # --- 2. INJECT FAST TOOLTIPS (Find & Replace Headers) ---
        header_map = {
            '<th>Rank</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Current rank in popularity list.">RANK</span></th>',
            '<th>Rank+</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Rank(Yest) - Rank(Today)\nGreen: Climbing | Red: Falling">&nbsp;RANK+</span></th>',
            '<th>Heat</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Weighted aggregate momentum score.\nRed: >2.0σ | Orange: >1.5σ | Yellow: >1σ">&nbsp;HEAT</span></th>',
            '<th>Name</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Official security or ETF name.">&nbsp;NAME</span></th>',
            '<th>Sym</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Ticker symbol for trading.">&nbsp;SYM</span></th>',
            '<th>Price</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Real-time trading price.">&nbsp;&nbsp;PRICE</span></th>',
            '<th>Day%</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Daily % change since last close.\nGreen: Positive | Red: Negative">&nbsp;&nbsp;DAY%</span></th>',
            '<th>Acc</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Vel(Now) - Vel(1h ago)\nMag: Expl. | Cyan: Fast | Red: Slow">ACC</span></th>',
            '<th>Eff</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Rank gain per unit of volume.\nGrn: >1.0 | Yel: >0.5 | Red: <0">&nbsp;&nbsp;EFF</span></th>',
            '<th>Conv</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Upvotes / Mentions ratio.\nGold: >1.0x | White: Diluted">&nbsp;CONV</span></th>',
            '<th>Upvs</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Total upvotes (24h).\nGreen: High Activity (>1.5σ)">UPVS</span></th>',
            '<th>Upv+</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Upv(Now) - Upv(1h ago)\nGreen: Positive | Red: Negative">&nbsp;UPV+</span></th>',
            '<th>VOL</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Current session trading volume.">&nbsp;&nbsp;VOL</span></th>',
            '<th>VOL(30)</th>': '<th><span class="d-tooltip header-fix" data-tooltip="30-day average volume baseline.">&nbsp;VOL(30)</span></th>',
            '<th>Srg</th>': '<th><span class="d-tooltip header-fix" data-tooltip="(Vol / Avg) * 100\nYel: Anomaly | Green: High Surge">&nbsp;SRG</span></th>',
            '<th>Vel</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Hourly change in Rank+.\nGreen: Speeding Up | Red: Slowing">VEL</span></th>',
            '<th>Strk</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Consecutive hours in direction.\nBrgt: 3+ | Pale: >0 | Red: Cold">STRK</span></th>',
            '<th>MENT</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Total comments/posts (24h).">MENT</span></th>',
            '<th>Mnt%</th>': '<th><span class="d-tooltip header-fix" data-tooltip="% change in mentions (24h).\nYel: >2σ | Green: >1σ">&nbsp;MNT%</span></th>',
            '<th>Sqz</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Mentions * Surge / log(MCap)\nCyan: >1.5σ | White: Normal">&nbsp;SQZ</span></th>',
            '<th>INDUSTRY</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Industry category group.">INDUSTRY</span></th>',
            '<th>Trend</th>': '<th><span class="d-tooltip header-fix" data-tooltip="3-Arrow Trend System (9/21/50)\nArrow 1: Price vs 9 EMA\nArrow 2: 9 EMA vs 21 EMA\nArrow 3: 21 EMA vs 50 EMA\n▲▲▲ = Full Bullish Alignment">TREND</span></th>',
            '<th>GI</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Gekko GI score (0-100).\nGreen: strong accumulation | Red: heavy distribution" style="margin-left:4px;">GI</span></th>',
            '<th>RSI</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Relative Strength Index (14d).\nRed: Overbought | Green: Oversold" style="margin-left:3px;">RSI</span></th>',
            '<th>STOCH</th>': '<th><span class="d-tooltip header-fix" data-tooltip="Slow Stochastic Oscillator (%K14, %D3) developed by George Lane.\nLogic: Measures momentum by comparing the closing price to the 14-day price range. It assumes prices tend to close near their highs in an uptrend and lows in a downtrend.\nZones: &le; 20 is Oversold (Buy Zone, Green) | &ge; 80 is Overbought (Sell Zone, Red).">&nbsp;STOCH</span></th>',
            '<th>SCTR</th>': '<th style="text-align:center; padding:2px !important;"><div style="display:flex; flex-direction:column; align-items:center; justify-content:center; gap:2px;"><div id="sctr-toggle" class="d-tooltip" data-tooltip="Toggle Ranking Mode:\nGLOBAL: Ranks against the entire table.\nDYNAMIC: Re-ranks only the visable." onclick="toggleColumnMode(event, 25, \'sctr-toggle\')" style="background:#111118; border:1px solid #00d97e; border-radius:4px; padding:1px 5px; font-size:9px; cursor:pointer; color:#00d97e; line-height:1; transition:all 0.2s; font-family:Inter,sans-serif; font-weight:700; letter-spacing:0.06em;">GLOBAL</div><span class="d-tooltip header-fix" data-tooltip="StockCharts Technical Rank (SCTR) created by John Murphy.\nLogic: A percentile ranking (0-99.9) of a stock\'s technical strength versus its peers.\nFormula: Heavily weights long-term trends (200d EMA, 125d ROC), while factoring in medium-term (50d EMA, 20d ROC) and short-term (RSI, PPO slope) momentum." style="line-height:1;">SCTR</span></div></th>',
            '<th>IBD_RS</th>': '<th style="text-align:center; padding:2px !important;"><div style="display:flex; flex-direction:column; align-items:center; justify-content:center; gap:2px;"><div id="ibd-toggle" class="d-tooltip" data-tooltip="Toggle Ranking Mode:\nGLOBAL: Ranks against the entire table.\nDYNAMIC: Re-ranks only the visable." onclick="toggleColumnMode(event, 26, \'ibd-toggle\')" style="background:#111118; border:1px solid #00d97e; border-radius:4px; padding:1px 5px; font-size:9px; cursor:pointer; color:#00d97e; line-height:1; transition:all 0.2s; font-family:Inter,sans-serif; font-weight:700; letter-spacing:0.06em;">GLOBAL</div><span class="d-tooltip header-fix" data-tooltip="Relative Strength (RS) Rating developed by William O\'Neil (IBD).\nLogic: A percentile rank (0-99.9) of a stock\'s 52-week price performance.\nFormula: Emphasizes recent momentum by weighting the most recent quarter (3 months) at 40%, and the prior three quarters at 20% each." style="line-height:1;">IBD</span></div></th>',
            '<th>SPY_RS</th>': '<th style="text-align:center; padding:2px !important;"><div style="display:flex; flex-direction:column; align-items:center; justify-content:center; gap:2px;"><div id="spy-toggle" class="d-tooltip" data-tooltip="Toggle Ranking Mode:\nGLOBAL: Ranks against the entire table.\nDYNAMIC: Re-ranks only the visable." onclick="toggleColumnMode(event, 27, \'spy-toggle\')" style="background:#111118; border:1px solid #00d97e; border-radius:4px; padding:1px 5px; font-size:9px; cursor:pointer; color:#00d97e; line-height:1; transition:all 0.2s; font-family:Inter,sans-serif; font-weight:700; letter-spacing:0.06em;">GLOBAL</div><span class="d-tooltip header-fix" data-tooltip="Relative Strength against SPY (0-99.9).\nLogic: A percentile rank of the stock\'s 1-year performance compared to the SPY baseline." style="line-height:1;">vsSPY</span></div></th>'
        }
        for old_tag, new_tag in header_map.items():
            raw_table = raw_table.replace(old_tag, new_tag)

        table_html = f'<div class="table-scroll-container">{raw_table}</div>'
        utc_timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # ---------------------------------------------------------
        #   HTML TEMPLATE (Embedded CSS/JS)
        # ---------------------------------------------------------
        html_content = f"""<!DOCTYPE html><html lang="en"><head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="300">
        <title>Ape Wisdom Analysis</title>
        <link rel="icon" type="image/x-icon" href="favicon.ico?v=1">
        <link rel="icon" type="image/png" sizes="32x32" href="favicon-32x32.png?v=1">
        <link rel="icon" type="image/png" sizes="16x16" href="favicon-16x16.png?v=1">
        <link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png?v=1">
        <link rel="manifest" href="site.webmanifest">
        <link rel="icon" type="image/png" href="favicon.png?v=1">
        <link rel="shortcut icon" href="favicon.ico?v=1">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
        :root {{
            --bg-primary:    #0a0a0f;
            --bg-secondary:  #111118;
            --bg-card:       #16161f;
            --bg-hover:      #1e1e2a;
            --border-dim:    #252530;
            --border-mid:    #333344;
            --accent-green:  #00d97e;
            --accent-blue:   #3b9eff;
            --accent-yellow: #f5c518;
            --accent-red:    #ff4d5a;
            --accent-cyan:   #00c8d7;
            --accent-purple: #c45cf6;
            --text-primary:  #e8e8f0;
            --text-muted:    #8888a0;
            --text-dim:      #555568;
            --font-ui:       'Inter', system-ui, sans-serif;
            --font-mono:     'JetBrains Mono', 'Consolas', monospace;
        }}

        * {{ box-sizing: border-box; }}

        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border-mid); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #4a4a5a; }}

        .container-fluid {{
            visibility: hidden;
            opacity: 0;
            transition: visibility 0s, opacity 0.4s ease-in-out;
        }}

        body.loaded .container-fluid {{
            visibility: visible;
            opacity: 1;
        }}

        /* ── PAGE LOADER ──────────────────────────────────────────── */
        #page-loader {{
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: var(--bg-primary);
            z-index: 99999;
            gap: 20px;
        }}

        #page-loader::before {{
            content: '';
            width: 48px; height: 48px;
            border: 3px solid var(--border-mid);
            border-top-color: var(--accent-green);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}

        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        #page-loader-text {{
            font-family: var(--font-mono);
            font-size: 13px;
            font-weight: 500;
            color: var(--accent-green);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        /* ── BASE ─────────────────────────────────────────────────── */
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-mono);
            padding: 0; margin: 0;
            overflow-x: hidden;
        }}

        table {{ table-layout: fixed; width: 100%; }}
        .table-dark {{ --bs-table-bg: var(--bg-secondary); color: var(--text-primary); }}
            
        /* ── TOOLTIP SYSTEM ───────────────────────────────────────── */
        th, .d-tooltip {{ position: relative; cursor: help; }}

        th[data-tooltip]:not(.sorting):not(.sorting_asc):not(.sorting_desc)::after, .d-tooltip::after {{
            content: attr(data-tooltip);
            position: absolute;
            top: 130%; left: 50%;
            font-family: var(--font-ui);
            font-size: 12px;
            line-height: 1.6;
            font-weight: 400;
            text-align: left;
            color: var(--text-primary);
            background-color: #0d0d15;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid var(--border-mid);
            text-transform: none;
            white-space: normal;
            width: max-content;
            max-width: 800px;
            z-index: 999999;
            opacity: 0; visibility: hidden;
            transition: opacity 0.15s ease-in-out;
            pointer-events: none; margin-top: 5px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,255,255,0.03);
        }}

        th:hover::after, .d-tooltip:hover::after,
        th:focus::after, .d-tooltip:focus::after {{
            opacity: 1; visibility: visible;
        }}

        .tooltip-inner {{
            max-width: 400px !important;
            background-color: #0d0d15 !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-mid);
            font-family: var(--font-ui);
            font-size: 12px;
        }}
        .tooltip {{ z-index: 10000000 !important; }}

        .table-responsive, .container-fluid {{ overflow: visible !important; }}

        /* ── TABLE HEADERS ────────────────────────────────────────── */
        th {{
            vertical-align: middle;
            font-family: var(--font-ui);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 6px 4px !important;
            background-color: var(--bg-card);
            border-bottom: 2px solid var(--border-mid) !important;
            border-top: none !important;
            color: var(--text-muted);
            z-index: 10;
            white-space: nowrap;
        }}

        th:hover {{ color: var(--text-primary) !important; z-index: 100 !important; }}

        /* ── TABLE CELLS ──────────────────────────────────────────── */
        td {{
            vertical-align: middle;
            white-space: nowrap;
            border-bottom: 1px solid var(--border-dim) !important;
            padding: 2px 8px !important;
            line-height: 1.5;
            font-size: 13px;
            font-family: var(--font-mono);
        }}

        table.dataTable {{
            width: auto !important;
            margin: 0 auto;
            border-right: 1px solid var(--border-mid) !important;
            border-left: 1px solid var(--border-mid) !important;
            border-collapse: separate !important;
            border-spacing: 0;
        }}

        .dataTables_wrapper > .row {{ --bs-gutter-y: 0 !important; }}
        .dataTables_wrapper > div.row:first-child {{ margin-bottom: 0 !important; padding-bottom: 0 !important; }}
        .dataTables_wrapper > div.row:nth-child(2) > div {{ padding-top: 0 !important; margin-top: 0 !important; }}


        /* ── COLUMN ALIGNMENTS ────────────────────────────────────── */
        th:nth-child(1), td:nth-child(1) {{ width: 1%; text-align: center; font-weight: 700; color: var(--text-muted); }}
        th:nth-child(2), td:nth-child(2) {{ width: 1%; text-align: center; }}
        th:nth-child(3), td:nth-child(3) {{ width: 1%; text-align: center; font-weight: 700; }}

        th:nth-child(4), td:nth-child(4) {{
            width: 1%; min-width: 60px; max-width: 12vw;
            overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; text-align: left;
            padding: 0 5px; vertical-align: middle;
        }}

        td:nth-child(4) .d-tooltip {{ position: relative; width: 100%; }}
        td:nth-child(4) .d-tooltip .text-content {{ display: block; width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

        td:nth-child(4) .d-tooltip::after {{
            content: attr(data-tooltip);
            position: absolute; top: 100%; left: 0; margin-top: 5px;
            background-color: #0d0d15; color: var(--text-primary);
            width: 600px; max-width: 600px;
            padding: 12px 14px; border-radius: 8px;
            border: 1px solid var(--border-mid);
            font-size: 12px; font-weight: normal;
            font-family: var(--font-ui);
            white-space: normal; line-height: 1.5;
            z-index: 999999; box-shadow: 0 12px 32px rgba(0,0,0,0.9);
            opacity: 0; visibility: hidden;
            transition: opacity 0.2s ease-in-out; pointer-events: none;
        }}
        td:nth-child(4) .d-tooltip:hover::after {{ opacity: 1; visibility: visible; }}

        th:nth-child(5), td:nth-child(5) {{ width: 1%; text-align: left; }}
        th:nth-child(6), td:nth-child(6) {{ width: 1%; text-align: right; }}
        th:nth-child(7), td:nth-child(7) {{ width: 1%; text-align: right; }}
        th:nth-child(8), td:nth-child(8) {{ width: 1%; text-align: center; }}
        th:nth-child(9), td:nth-child(9) {{ width: 1%; text-align: center; }}
        th:nth-child(10), td:nth-child(10) {{ width: 1%; text-align: center; }}
        th:nth-child(11), td:nth-child(11) {{ width: 1%; text-align: center; }}
        th:nth-child(12), td:nth-child(12) {{ width: 1%; text-align: center; }}
        th:nth-child(13), td:nth-child(13) {{ width: 1%; text-align: right; font-weight: 600 !important; }}
        th:nth-child(14), td:nth-child(14) {{ width: 1%; text-align: right; letter-spacing: -0.5px; }}
        th:nth-child(15), td:nth-child(15) {{ width: 1%; text-align: center; }}
        th:nth-child(16), td:nth-child(16) {{ width: 1%; text-align: center; }}
        th:nth-child(17), td:nth-child(17) {{ width: 1%; text-align: center; }}
        th:nth-child(18), td:nth-child(18) {{ width: 1%; text-align: center; }}
        th:nth-child(19), td:nth-child(19) {{ width: 1%; text-align: center; }}
        th:nth-child(20), td:nth-child(20) {{ width: 1%; text-align: center; }}

        th:nth-child(21), td:nth-child(21) {{
            width: auto; min-width: 60px;
            overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; text-align: left;
        }}

        th:nth-child(22), td:nth-child(22) {{ width: 1%; text-align: center; font-weight: 700; }}
        th:nth-child(23), td:nth-child(23) {{ width: 1%; text-align: center; font-weight: 600; }}
        th:nth-child(24), td:nth-child(24) {{ width: 1%; text-align: center; font-weight: 600; }}
        th:nth-child(25), td:nth-child(25) {{ width: 1%; text-align: center; font-weight: 600; }}
        th:nth-child(26), td:nth-child(26) {{ width: 1%; text-align: center; font-weight: 600; }}
        th:nth-child(27), td:nth-child(27) {{ width: 1%; text-align: center; font-weight: 600; }}
        th:nth-child(28), td:nth-child(28) {{ width: 1%; text-align: center; font-weight: 600; border-right: 1px solid var(--border-mid) !important; }}

        /* ── LINKS & COLORS ───────────────────────────────────────── */
        a {{ color: var(--accent-blue); text-decoration: none; }}
        a:hover {{ color: #6bb8ff; text-decoration: underline; }}
        table.no-colors span {{ color: var(--text-muted) !important; font-weight: normal !important; }}
        table.no-colors a {{ color: var(--accent-blue) !important; }}

        /* ── FILTER BAR ───────────────────────────────────────────── */
        .filter-bar {{
            display: flex;
            gap: 6px;
            align-items: center;
            background: var(--bg-card);
            padding: 4px 8px;
            border-radius: 0;
            margin-bottom: 0;
            border-bottom: 1px solid var(--border-mid);
            border-top: none;
            font-size: 0.8rem;
            flex-wrap: nowrap;
            overflow-x: auto;
            white-space: nowrap;
            -ms-overflow-style: none;
            scrollbar-width: none;
            z-index: 9998 !important;
        }}

            .dataTables_wrapper > .row:first-child {{
                margin-bottom: 0px !important; 
                padding-bottom: 0px !important;
                min-height: 30px;
            }}
            
            .dataTables_length {{
                margin-bottom: 0px !important;
                margin-top: 0px !important; /* Pulls the "Show entries" box slightly up */
            }}
            
            .dataTables_filter {{
                top: 0px !important; /* Moves your floating search bar up to match */
            }}
            
            .filter-bar {{
                margin-bottom: 0px !important; /* Shrinks the gap below your custom buttons */
            }}

            .dataTables_wrapper .dataTables_filter {{
                position: sticky !important;
                top: 1px;
            }}

        .filter-bar::-webkit-scrollbar {{ display: none; }}
        .filter-group {{ display: flex; align-items: center; gap: 4px; }}

        .form-control-sm {{
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-mid) !important;
            color: var(--text-primary) !important;
            height: 26px; font-size: 0.78rem;
            padding: 2px 8px; outline: none;
            border-radius: 6px !important;
            font-family: var(--font-mono);
        }}
        .form-control-sm::placeholder {{ color: var(--text-dim) !important; opacity: 1; }}
        .form-control-sm:focus {{ border-color: var(--accent-cyan) !important; background: var(--bg-card) !important; box-shadow: 0 0 0 2px rgba(0,200,215,0.12) !important; }}

        .btn-reset {{
            border: 1px solid var(--border-mid);
            color: var(--text-muted);
            font-size: 0.75rem;
            background: var(--bg-secondary);
            border-radius: 6px;
            transition: all 0.15s;
        }}
        .btn-reset:hover {{ background: var(--bg-hover); color: var(--text-primary); border-color: #4a4a5a; }}

        #stockCounter {{
            color: var(--accent-green);
            font-weight: 700;
            margin-left: auto;
            border: 1px solid rgba(0,217,126,0.35);
            background: rgba(0,217,126,0.07);
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            letter-spacing: 0.04em;
            font-family: var(--font-ui);
        }}

        .mode-toggle {{ position: relative; z-index: 1 !important; }}

        /* ── HEADER ───────────────────────────────────────────────── */
        .header-flex {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 48px; width: 100%;
            padding: 0 16px;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-mid);
            box-sizing: border-box;
            z-index: 2000;
            position: relative;
        }}

        .header-left {{ flex: 0 0 220px; display: flex; align-items: center; gap: 12px; z-index: 1; }}

        .header-right {{
            flex: 0 0 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-end;
            z-index: 10;
        }}

        .update-label {{
            font-size: 9px;
            font-weight: 700;
            color: var(--text-dim);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-family: var(--font-ui);
        }}

        .header-center {{
            position: absolute;
            left: 50%; top: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 2px;
            width: auto;
            white-space: nowrap;
            z-index: 101;
        }}

        .header-fix {{ display: inline-block; max-width: 100%; white-space: nowrap; }}

        .summary-row {{
            display: flex;
            align-items: center;
            gap: 6px;
            line-height: 1;
            font-size: 10px;
            height: 10px;
        }}
        .summary-row:nth-child(1) {{ z-index: 40; }}
        .summary-row:nth-child(2) {{ z-index: 30; }}
        .summary-row:nth-child(3) {{ z-index: 20; }}
        .summary-row:nth-child(4) {{ z-index: 10; }}

        .row-label {{
            font-family: var(--font-ui);
            font-size: 9px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            text-align: right;
            cursor: help;
            border-bottom: none !important;
            position: relative;
            z-index: 50;
            margin-right: 10px;
            color: var(--text-dim);
        }}

        .row-content {{ font-size: 10px; font-weight: 600; color: var(--text-primary); font-family: var(--font-ui); }}

        .crumb-sep {{ color: var(--text-dim); margin: 0 3px; font-weight: bold; }}
        .crumb-num {{ color: var(--text-dim); margin-right: 3px; font-size: 10px; }}

        .clr-rank {{ color: var(--accent-cyan); }}
        .clr-surge {{ color: var(--accent-yellow); }}
        .clr-buzz {{ color: var(--accent-purple); }}

        .sector-tooltip {{ white-space: nowrap; }}

        .row-label::after {{
            content: attr(data-tooltip);
            position: absolute;
            top: 160%; left: 50%;
            transform: translateX(-50%);
            background-color: #0d0d15;
            color: var(--text-primary);
            padding: 7px 11px;
            border-radius: 7px;
            border: 1px solid var(--border-mid);
            font-family: var(--font-ui);
            font-size: 11px;
            font-weight: normal;
            text-transform: none;
            white-space: nowrap;
            z-index: 999999 !important;
            opacity: 0; visibility: hidden;
            transition: opacity 0.1s;
            pointer-events: none;
            margin-top: 5px;
            box-shadow: 0 8px 24px rgba(0,0,0,1);
        }}

        .row-label:hover::after {{ opacity: 1; visibility: visible; }}
        .clr-rank:hover::after {{ color: var(--accent-cyan) !important; border-color: var(--accent-cyan) !important; }}
        .clr-upv:hover::after {{ color: var(--accent-green) !important; border-color: var(--accent-green) !important; }}
        .clr-surge:hover::after {{ color: var(--accent-yellow) !important; border-color: var(--accent-yellow) !important; }}
        .clr-buzz:hover::after {{ color: var(--accent-purple) !important; border-color: var(--accent-purple) !important; }}

        /* ── DATATABLE CONTROLS ───────────────────────────────────── */
        .dataTables_wrapper .data_tables_header {{
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;
        }}

        .dataTables_filter {{
            position: fixed !important;
            left: 50vw !important;
            transform: translateX(-50%) !important;
            pointer-events: none !important;
            width: max-content !important;
            top: 0px !important;
            z-index: 10 !important;
            margin: 0 !important; padding: 0 !important;
        }}

        .d-tooltip::after, th[data-tooltip]::after {{ z-index: 999999 !important; }}

        .dataTables_filter input {{
            pointer-events: auto !important;
            width: 25vw !important;
            min-width: 150px !important;
            max-width: 350px !important;
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-mid) !important;
            border-radius: 20px !important;
            padding: 0 12px !important;
            outline: none !important;
            text-align: center !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            height: 22px !important;
            font-family: var(--font-ui) !important;
            transition: border-color 0.15s, box-shadow 0.15s !important;
        }}

        .dataTables_filter input::placeholder {{ color: var(--text-muted) !important; opacity: 1; }}
        .dataTables_filter input:focus::placeholder {{ color: transparent !important; }}
        .dataTables_filter input:focus {{ border-color: var(--accent-cyan) !important; box-shadow: 0 0 0 3px rgba(0,200,215,0.12) !important; }}
        .dataTables_filter label {{ color: transparent !important; font-size: 0 !important; display: flex !important; justify-content: center; width: 100%; }}

        /* ── PAGINATION ───────────────────────────────────────────── */
        .page-link {{
            background-color: var(--bg-secondary);
            border-color: var(--border-mid);
            color: var(--accent-green);
            font-family: var(--font-ui);
            font-size: 12px;
            transition: all 0.15s;
        }}
        .page-link:hover {{ background-color: var(--bg-hover); color: var(--accent-green); }}
        .page-item.active .page-link {{ background-color: var(--accent-green); border-color: var(--accent-green); color: #000; font-weight: 700; }}
        .page-item.disabled .page-link {{ background-color: var(--bg-primary); border-color: var(--border-dim); color: var(--text-dim); }}

        .mode-toggle label {{
            margin-left: 0;
            display: flex; align-items: center;
            background: var(--bg-secondary);
            padding: 2px;
            border-radius: 6px;
            cursor: pointer;
            border: 1px solid var(--border-mid);
        }}
        #modeSwitch {{ display: none; }}
        #modeSwitch:checked + label .e-label {{ color: var(--text-primary); background: var(--bg-hover); }}
        #modeSwitch:not(:checked) + label .s-label {{ color: var(--text-primary); background: var(--bg-hover); }}

        /* ── ROW HOVER ────────────────────────────────────────────── */
        tr:hover {{
            position: relative; z-index: 100;
            background-color: var(--bg-hover) !important;
            cursor: pointer;
        }}
        tr:hover td {{ background-color: transparent !important; color: #ffffff !important; }}

        /* ── TOOLTIP POSITIONING ──────────────────────────────────── */
        th:nth-child(-n+5) .d-tooltip::after,
        td:nth-child(-n+5) .d-tooltip::after {{
            left: 0 !important; right: auto !important;
            transform: none !important; text-align: left !important;
        }}
        th:nth-last-child(-n+7) .d-tooltip::after,
        td:nth-last-child(-n+7) .d-tooltip::after {{
            right: 0 !important; left: auto !important;
            transform: none !important; text-align: left !important;
        }}

        /* ── TIME DISPLAY ─────────────────────────────────────────── */
        #time {{
            font-family: var(--font-mono);
            font-size: 12px !important;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: 0.02em;
        }}

        /* ── SORTING ARROWS ───────────────────────────────────────── */
        table.dataTable thead > tr > th.sorting:before,
        table.dataTable thead .sorting:after,
        table.dataTable thead .sorting_asc:after,
        table.dataTable thead .sorting_desc:after {{
            display: inline-block !important; visibility: visible !important;
            opacity: 0.5 !important; position: relative !important; top: 0 !important;
        }}
        th.sorting::after, th.sorting_asc::after, th.sorting_desc::after {{ content: none !important; }}
        th.sorting::before, th.sorting_asc::before, th.sorting_desc::before {{ content: none !important; }}

        /* ── DATATABLE LENGTH SELECT ──────────────────────────────── */
        .dataTables_length select {{
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-mid) !important;
            border-radius: 6px;
            padding: 0 10px 0 4px !important;
            height: 26px !important;
            font-size: 12px !important;
            outline: none !important;
            box-shadow: none !important;
            min-width: 55px !important;
            font-family: var(--font-ui) !important;
        }}

        .dataTables_filter label, .dataTables_length label {{
            color: var(--text-muted) !important;
            font-size: 12px !important;
            line-height: 28px !important;
            display: flex;
            align-items: center !important;
            gap: 5px;
            font-family: var(--font-ui) !important;
        }}

        .dataTables_wrapper > .row {{
            padding-top: 2px !important;
            padding-bottom: 2px !important;
        }}
        .dataTables_length label {{ margin: 0 !important; padding: 0 !important; line-height: 1 !important; }}
        .dataTables_length {{ margin: 0 !important; }}
        .dataTables_filter {{ top: 0px !important; }}

        /* ── SYMBOL CONTAINER ─────────────────────────────────────── */
        .symbol-container {{
            position: relative;
            display: flex !important;
            align-items: center;
            gap: 8px;
            width: 100%;
            overflow: visible !important;
        }}

        /* ── INLINE TOOLTIP (d-tooltip) ───────────────────────────── */
        .d-tooltip {{ position: relative; cursor: help; }}
        .d-tooltip::after {{
            content: attr(data-tooltip);
            position: absolute; left: 50%;
            transform: translateX(-50%);
            background-color: #0d0d15 !important;
            color: var(--text-primary) !important;
            display: block; height: auto !important;
            min-width: max-content;
            padding: 7px 11px;
            border-radius: 8px;
            border: 1px solid var(--border-mid);
            font-family: var(--font-mono);
            font-size: 11px;
            white-space: pre-wrap !important;
            text-align: left !important;
            line-height: 1.4;
            width: max-content; max-width: none;
            opacity: 0; visibility: hidden;
            transition: opacity 0.1s ease-in-out;
            pointer-events: none;
            box-shadow: 0 12px 32px rgba(0,0,0,0.95);
            z-index: 99999 !important;
        }}
        .d-tooltip:hover::after, .d-tooltip:focus::after {{ opacity: 1; visibility: visible; }}
        td .d-tooltip::after {{ top: 110%; bottom: auto; }}
        th .d-tooltip::after {{ bottom: 110%; top: auto; }}

        /* ── CHART POPUP ──────────────────────────────────────────── */
        .chart-popup {{
            display: none;
            position: fixed;
            width: 400px; height: 400px;
            background: #08080f;
            border: 1px solid var(--border-mid);
            border-radius: 10px;
            z-index: 9999999 !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.95), 0 0 0 1px rgba(255,255,255,0.04);
            padding: 0 !important;
            pointer-events: auto;
            overflow: hidden;
            flex-direction: column;
        }}
        .chart-popup.large-chart {{ width: 75vw; height: 80vh; }}
        .symbol-container:hover .chart-popup {{ display: flex; }}

        /* ── COLUMN OVERFLOW CONTROL ──────────────────────────────── */
        th:nth-child(4), th:nth-child(21) {{ overflow: visible !important; z-index: 50 !important; }}
        th:nth-child(4) span, th:nth-child(21) span {{ white-space: nowrap; }}
        th:hover {{ z-index: 100 !important; }}

        .header-flex {{ z-index: 10 !important; position: relative; }}
        .table-scroll-container, .table-responsive, .container-fluid {{ overflow: visible !important; }}

        /* ── BTN-GROUP OVERRIDES ──────────────────────────────────── */
        .btn-outline-light {{
            border-color: var(--border-mid) !important;
            color: var(--text-muted) !important;
            background: var(--bg-secondary) !important;
            font-family: var(--font-ui) !important;
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.04em !important;
            transition: all 0.15s !important;
        }}
        .btn-check:checked + .btn-outline-light {{
            background: rgba(0,217,126,0.15) !important;
            border-color: var(--accent-green) !important;
            color: var(--accent-green) !important;
        }}
        .btn-outline-light:hover {{
            background: var(--bg-hover) !important;
            color: var(--text-primary) !important;
        }}

        /* ── HEATMAP MODAL ────────────────────────────────────────── */
        #heatmapModal {{ backdrop-filter: blur(4px); }}

        </style>
        </head>
        <body>
        <div id="page-loader"><span id="page-loader-text">LOADING MARKET DATA</span></div>
        <div class="container-fluid" style="width: auto; display: inline-block; min-width: 100%; margin: 0 auto;">

            <div class="header-flex">
    <div class="header-left">
        <a href="https://apewisdom.io" target="_blank" style="display:flex; align-items:center; opacity:0.9; transition:opacity 0.15s;">
            <img src="https://apewisdom.io/apewisdom-logo.svg" alt="Ape Wisdom" title="apewisdom.io" style="height: 32px; filter: brightness(1.1);">
        </a>
        <div class="mode-toggle">
            <input type="checkbox" id="modeSwitch" onclick="updateSummary()">
            <label for="modeSwitch">
                <span class="mode-label s-label" style="font-family:var(--font-ui); font-size:10px; font-weight:700; letter-spacing:0.08em; padding:4px 8px; border-radius:4px; color:var(--text-dim); transition:all 0.15s;">STOCKS</span>
                <span class="mode-label e-label" style="font-family:var(--font-ui); font-size:10px; font-weight:700; letter-spacing:0.08em; padding:4px 8px; border-radius:4px; color:var(--text-dim); transition:all 0.15s;">ETFs</span>
            </label>
        </div>
    </div>

    <div class="header-center">
        <div class="summary-row">
            <span class="row-label clr-rank" data-tooltip="Total Rank Change by Industry.">RANK:</span>
            <span id="rankBreadcrumb" class="row-content">...</span>
        </div>
        <div class="summary-row">
            <span class="row-label clr-upv" style="color: var(--accent-green);" data-tooltip="Total New Upvotes by Industry.">UPVOTES:</span>
            <span id="upvBreadcrumb" class="row-content">...</span>
        </div>
        <div class="summary-row">
            <span class="row-label clr-surge" data-tooltip="Total Volume Surge by Industry.">SURGE:</span>
            <span id="surgeBreadcrumb" class="row-content">...</span>
        </div>
        <div class="summary-row">
            <span class="row-label clr-buzz" data-tooltip="Total Social Buzz (Mentions) by Industry.">BUZZ:</span>
            <span id="mntBreadcrumb" class="row-content">...</span>
        </div>
    </div>

    <div class="header-right">
        <span class="update-label">LAST UPDATED</span>
        <span id="time" data-utc="{utc_timestamp}" style="font-size:11px;">Loading...</span>
    </div>
</div>

            <div class="filter-bar">
                <span style="font-family:var(--font-ui); font-size:9px; font-weight:800; letter-spacing:0.12em; color:var(--text-dim); text-transform:uppercase; margin-right:4px;">FILTERS</span>
                <div style="width:1px; height:16px; background:var(--border-mid); margin:0 4px;"></div>
                <button id="btnColors" class="btn btn-sm btn-reset" onclick="toggleColors()" title="Toggle Colors">🎨</button>
                <button class="btn btn-sm btn-reset" onclick="resetFilters()" title="Reset All Filters">↺ Reset</button>

                <div style="width:1px; height:16px; background:var(--border-mid); margin:0 4px;"></div>

                <div class="filter-group">
                    <label style="font-family:var(--font-ui); font-size:9px; font-weight:700; letter-spacing:0.06em; color:var(--text-dim); text-transform:uppercase;">Price</label>
                    <input type="text" id="minPrice" class="form-control form-control-sm" placeholder="Min" style="width: 48px;">
                    <span style="color:var(--text-dim); font-size:10px;">–</span>
                    <input type="text" id="maxPrice" class="form-control form-control-sm" placeholder="Max" style="width: 48px;">
                </div>

                <div style="width:1px; height:16px; background:var(--border-mid); margin:0 4px;"></div>

                <div class="filter-group">
                    <label style="font-family:var(--font-ui); font-size:9px; font-weight:700; letter-spacing:0.06em; color:var(--text-dim); text-transform:uppercase;">Vol(30)</label>
                    <input type="text" id="minVol" class="form-control form-control-sm" placeholder="Min" style="width: 48px;">
                    <span style="color:var(--text-dim); font-size:10px;">–</span>
                    <input type="text" id="maxVol" class="form-control form-control-sm" placeholder="Max" style="width: 48px;">
                </div>

                <div style="width:1px; height:16px; background:var(--border-mid); margin:0 4px;"></div>

                <div class="filter-group">
                    <div class="btn-group" role="group">
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio1" checked onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio1">ALL</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio2" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio2">STOCKS</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio3" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio3">ETFs</label>
                    </div>
                </div>

                <div style="width:1px; height:16px; background:var(--border-mid); margin:0 4px;"></div>

                <div class="filter-group">
                    <div class="btn-group" role="group">
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapAll" checked onclick="toggleMcap('all')">
                        <label class="btn btn-outline-light btn-sm" for="mcapAll" title="Show All Market Caps">ALL</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMega" onclick="toggleMcap('mega')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMega" title="Mega Cap: > $200B">MEGA</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapLarge" onclick="toggleMcap('large')">
                        <label class="btn btn-outline-light btn-sm" for="mcapLarge" title="Large Cap: $10B – $200B">LRG</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMid" onclick="toggleMcap('mid')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMid" title="Mid Cap: $2B – $10B">MID</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapSmall" onclick="toggleMcap('small')">
                        <label class="btn btn-outline-light btn-sm" for="mcapSmall" title="Small Cap: $250M – $2B">SML</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMicro" onclick="toggleMcap('micro')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMicro" title="Micro Cap: < $250M">MIC</label>
                    </div>
                </div>

                <div style="width:1px; height:16px; background:var(--border-mid); margin:0 4px;"></div>

                <button class="btn btn-sm btn-reset" onclick="openHeatmapModal('stock')" title="Stock Heatmap" style="background: linear-gradient(135deg, #c43000, #ff4422); color: white; font-weight: 700; border: none; font-family:var(--font-ui); font-size:0.7rem; letter-spacing:0.06em;">🔥 STOCKS</button>
                <button class="btn btn-sm btn-reset" onclick="openHeatmapModal('etf')" title="ETF Heatmap" style="background: linear-gradient(135deg, #c43000, #ff4422); color: white; font-weight: 700; border: none; font-family:var(--font-ui); font-size:0.7rem; letter-spacing:0.06em;">📈 ETFs</button>
                <button class="btn btn-sm btn-reset" onclick="exportTickers()" title="Download Ticker List" style="font-family:var(--font-ui); font-size:0.7rem;">.TXT</button>
                <button class="btn btn-sm btn-reset" onclick="copyTableToClipboard(event)" title="Copy Table" style="font-family:var(--font-ui); font-size:0.7rem;">📋 Copy</button>
                <button class="btn btn-sm btn-reset" onclick="copySymbolsToClipboard(event)" title="Copy Symbols Only" style="font-family:var(--font-ui); font-size:0.7rem;">📋 Syms</button>

                <span id="stockCounter">Loading...</span>
            </div>

            {table_html}
            
            <!-- Heatmap Modal -->
            <div id="heatmapModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.88); z-index: 9999; backdrop-filter: blur(4px);">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 99vw; height: 96vh; background: #08080f; display: flex; flex-direction: column; border: 1px solid #252530; box-shadow: 0 32px 80px rgba(0,0,0,0.95), 0 0 0 1px rgba(255,255,255,0.03); border-radius: 10px; overflow: hidden;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 14px; background: #111118; border-bottom: 1px solid #252530; flex-shrink: 0;">
                    <h2 id="heatmapTitle" style="color: #ff5533; margin: 0; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;">🔥 Stock Heatmap</h2>
                    <button onclick="closeHeatmapModal()" style="background: #252530; color: #8888a0; border: 1px solid #333344; padding: 3px 10px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; font-family: 'Inter', sans-serif; transition: all 0.15s;" onmouseover="this.style.background='#333344'; this.style.color='#e8e8f0';" onmouseout="this.style.background='#252530'; this.style.color='#8888a0';">✕ Close</button>
                </div>
                <div id="heatmapContainer" style="flex: 1; width: 100%; height: 100%; position: relative;"></div>
            </div>
    </div>
        
        <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
        <script>
    var table;

    // 1. UPDATED: Added 'heightOverride' to parameter list so it actually works
    function positionPopup(container, event, heightOverride, preferRight, centerVertically) {{
        if (!event || !container) return;

        container.style.display = "flex";

        const mouseY = event.clientY;
        const mouseX = event.clientX;
        const screenHeight = window.innerHeight;
        const screenWidth = window.innerWidth;

        const actualWidth = container.offsetWidth || 700;
        const actualHeight = heightOverride || container.offsetHeight || 400;

        const gapBelow = 1;
        const gapAbove = 1;

        const screenPadding = 20;

        const spaceBelow = screenHeight - mouseY - screenPadding;
        const spaceAbove = mouseY - screenPadding;
        let topPos;

        // If centerVertically is true, center on mouse Y position
        if (centerVertically) {{
            topPos = mouseY - (actualHeight / 2);
            // Clamp to screen bounds
            if (topPos < screenPadding) topPos = screenPadding;
            if (topPos + actualHeight > screenHeight - screenPadding) {{
                topPos = screenHeight - actualHeight - screenPadding;
            }}
        }} else {{
            // Prefer placing below if there's enough space or more space below than above
            if (spaceBelow >= actualHeight + gapBelow || spaceBelow >= spaceAbove) {{
                topPos = mouseY + gapBelow;
                if (topPos + actualHeight > screenHeight - screenPadding) {{
                    topPos = screenHeight - actualHeight - screenPadding;
                }}
            }} else {{
                // Place above; clamp so it never goes off-screen at the top
                topPos = mouseY - actualHeight - gapAbove;
                if (topPos < screenPadding) {{
                    topPos = screenPadding;
                }}
            }}
        }}

        container.style.top = topPos + "px";

        // Horizontal placement
        let leftPos = mouseX - (actualWidth / 2);
        const gapRight = 20;
        const spaceRight = screenWidth - mouseX - screenPadding;
        const spaceLeft = mouseX - screenPadding;

        if (preferRight) {{
            if (spaceRight >= actualWidth + gapRight) {{
                leftPos = mouseX + gapRight;
            }} else if (spaceLeft >= actualWidth + gapRight) {{
                leftPos = mouseX - actualWidth - gapRight;
            }} else {{
                // fallback to clamped center
                leftPos = mouseX - (actualWidth / 2);
            }}
        }}

        if (leftPos < screenPadding) leftPos = screenPadding;
        if (leftPos + actualWidth > screenWidth - screenPadding) {{
            leftPos = screenWidth - actualWidth - screenPadding;
        }}

        container.style.left = leftPos + "px";
    }}

    function getFinalSymbol(symbol, yfExchange) {{
        const ex = String(yfExchange || "").toUpperCase();
        const s = String(symbol || "").toUpperCase().replace('-', '.');

        const manualOverrides = {{
            'SPY': 'AMEX:SPY', 'VOO': 'AMEX:VOO', 'IVV': 'AMEX:IVV',
            'TQQQ': 'NASDAQ:TQQQ', 'SQQQ': 'NASDAQ:SQQQ', 'VPN': 'NASDAQ:VPN',
            'AM': 'NYSE:AM', 'DIA': 'AMEX:DIA', 'IWM': 'AMEX:IWM', 'DTE': 'NYSE:DTE'
        }};

        if (manualOverrides[s]) return manualOverrides[s];

        if (ex.includes('NMS') || ex.includes('NGM') || ex.includes('NCM') || ex.includes('NASDAQ')) {{
            return 'NASDAQ:' + s;
        }}
        if (ex.includes('NYQ') || ex.includes('NYSE')) {{
            return 'NYSE:' + s;
        }}
        if (ex.includes('ASE') || ex.includes('AMEX') || ex.includes('PCX') || ex.includes('ARCA')) {{
            return 'AMEX:' + s;
        }}
        if (ex.includes('BATS') || ex.includes('BZX')) {{
            return 'BATS:' + s;
        }}
        if (ex.includes('LSE')) {{
            return 'LSE:' + s;
        }}
        return s;
    }}

    function loadMiniChart(symbol, index, yfExchange, event) {{
        const container = document.getElementById('chart-tooltip-' + index);
        if (!container) return;
        
        container.innerHTML = "";
        
        // 1. Reset any previous inline styles
        container.style.width = ""; 
        container.classList.add('large-chart'); // Base preference (e.g., 75vw)
        
        // --- SMART SIZING LOGIC ---
        const mouseX = (event || window.event).clientX;
        const screenW = window.innerWidth;
        const spaceOnRight = screenW - mouseX - 40; // 40px buffer for scrollbar/padding
        
        // Define your ideal width (75% of screen) and minimum readable width (e.g., 500px)
        const idealWidth = screenW * 0.75; 
        const minWidth = 500; 

        // Check if ideal width fits on the right
        if (idealWidth > spaceOnRight) {{
            // It doesn't fit! 
            // If the available space is decent (>500px), shrink the chart to fit that space.
            if (spaceOnRight > minWidth) {{
                container.style.width = spaceOnRight + "px";
            }} 
            // If space is too small (<500px), we leave it big and let positionPopup flip it to the left.
        }}
        // ---------------------------

        const dynamicHeight = window.innerHeight * 0.8;
        positionPopup(container, event || window.event, dynamicHeight, true, true);
        
        const finalSymbol = getFinalSymbol(symbol, yfExchange);

        const widgetContainer = document.createElement('div');
        widgetContainer.className = 'tradingview-widget-container';
        widgetContainer.style.width = "100%";
        widgetContainer.style.height = "100%";
        
        const widgetDiv = document.createElement('div');
        widgetDiv.className = 'tradingview-widget-container__widget';
        widgetContainer.appendChild(widgetDiv);

        const script = document.createElement('script');
        script.type = 'text/javascript';
        script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
        script.async = true;
        
        script.innerHTML = JSON.stringify({{
            "autosize": true,
            "symbol": finalSymbol,
            "interval": "D",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "calendar": false,
            "details": true,
            "hotlist": false,
            "hide_side_toolbar": true,
            "hide_top_toolbar": true,
            "hide_legend": false,
            "hide_volume": false,
            "withdateranges": true,
            "range": "12M",
            "save_image": false,
            "backgroundColor": "#0F0F0F",
            "gridColor": "rgba(242, 242, 242, 0.06)",
            "watchlist": [],
            "compareSymbols": [],
            "studies": [
                {{"id": "MAExp@tv-basicstudies", "inputs": {{"length": 9}}, "overrides": {{"Plot.color": "#4CAF50", "Plot.linewidth": 1}}}},
                {{"id": "MAExp@tv-basicstudies", "inputs": {{"length": 21}}, "overrides": {{"Plot.color": "#00BCD4", "Plot.linewidth": 1}}}},
                {{"id": "MAExp@tv-basicstudies", "inputs": {{"length": 50}}, "overrides": {{"Plot.color": "#2979FF", "Plot.linewidth": 2}}}}
            ]
        }});

        widgetContainer.appendChild(script);
        container.appendChild(widgetContainer);
    }}

    function loadSymbolProfile(symbol, containerId, yfExchange, event) {{
        const container = document.getElementById(containerId);
        if (!container) return;

        if (container.innerHTML !== "") {{
            container.style.display = "flex";
            positionPopup(container, event, 400, true, true);
            return;
        }}

        container.style.display = "flex"; 
        positionPopup(container, event || window.event, 400, true, true);
        
        const finalSymbol = getFinalSymbol(symbol, yfExchange);
        const widgetContainer = document.createElement('div');
        widgetContainer.className = 'tradingview-widget-container';
        widgetContainer.style.width = "100%";
        widgetContainer.style.height = "100%";
        
        const script = document.createElement('script');
        script.type = 'text/javascript';
        script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js';
        script.async = true;
        script.innerHTML = JSON.stringify({{
            "symbol": finalSymbol,
            "width": "100%",
            "height": "100%",
            "colorTheme": "dark",
            "isTransparent": false,
            "showHeadline": true,
            "locale": "en"
        }});

        widgetContainer.appendChild(script);
        container.appendChild(widgetContainer);
    }}

    function hideSymbolProfile(containerId) {{
        const container = document.getElementById(containerId);
        if (container) {{
            container.style.display = "none";
            container.innerHTML = "";
            container.classList.remove('large-chart');
        }}
    }}

    function parseVal(str) {{
        if (!str || str === null) return 0;
        
        // 1. Convert to string and STRIP ALL HTML TAGS first
        // This removes the <span class="d-tooltip" ...> but leaves the value behind
        var clean = str.toString().replace(/<[^>]+>/g, '').trim();

        // 2. Remove symbols that aren't numbers ($, %, Arrows, Plus, 'x' for conviction)
        clean = clean.replace(/[$,%▲▼+x]/g, '').toLowerCase();

        // 3. Handle K/M/B Multipliers
        let mult = 1;
        if (clean.endsWith('k')) {{ mult = 1000; clean = clean.slice(0, -1); }}
        else if (clean.endsWith('m')) {{ mult = 1000000; clean = clean.slice(0, -1); }}
        else if (clean.endsWith('b')) {{ mult = 1000000000; clean = clean.slice(0, -1); }}

        // 4. Convert to number and multiply
        var result = parseFloat(clean) * mult;
        return isNaN(result) ? 0 : result;
    }}

    function updateSummary() {{
        if (!$.fn.DataTable.isDataTable('.table')) return;
        var api = $('.table').DataTable();
        var topSwitchIsETF = $('#modeSwitch').is(':checked');
        var allData = api.rows({{ search: 'none', order: 'index' }}).data();
        
        function getTopSectors(metricIdx) {{
            var sectorData = {{}};
            allData.each(function(row) {{
                var rawType = row[28].toString().replace(/<[^>]+>/g, ''); 
                if (topSwitchIsETF && !rawType.includes('ETF')) return;
                if (!topSwitchIsETF && rawType.includes('ETF')) return;

                var sector = row[20].toString().replace(/<[^>]+>/g, '').trim().replace(/^ETF/i, '');
                if (!topSwitchIsETF && sector === 'Exchange Traded Fund') return;
                var val = parseVal(row[metricIdx]); 
                var sym = row[4].replace(/<[^>]+>/g, '').trim();
                var name = row[3].replace(/<[^>]+>/g, '').trim();

                if (!sectorData[sector]) {{ sectorData[sector] = {{ totalSum: 0, count: 0, stocks: [] }}; }}
                sectorData[sector].totalSum += val;
                sectorData[sector].count += 1;
                sectorData[sector].stocks.push({{ s: sym, n: name, v: val }});
            }});

            var sorted = Object.keys(sectorData).map(function(s) {{
                return {{ name: s, total: sectorData[s].totalSum, count: sectorData[s].count, stocks: sectorData[s].stocks }};
            }});

            sorted = sorted.filter(function(s) {{ return s.count >= 2; }});

            sorted.sort(function(a, b) {{ 
                if (b.total !== a.total) {{ return b.total - a.total; }}
                return a.name.localeCompare(b.name); 
            }});

            if (sorted.length === 0) return '<span style="color:#666;">---</span>';

            var topThree = sorted.slice(0, 5);
            return topThree.map(function(s, i) {{
                s.stocks.sort(function(a, b) {{ return b.v - a.v; }});
                var topStocks = s.stocks; 
                
                var tipRows = topStocks.map(function(st) {{
                    var val = Math.round(st.v);
                    var numStr = val > 0 ? '+' + val : val;
                    var color = val > 0 ? '#00d97e' : (val < 0 ? '#ff4d5a' : '#8888a0');
                    return "<div style='display:flex; justify-content:flex-start; align-items:center; font-size:11px; margin-bottom:1px;'>" +
                                "<span style='min-width:45px; text-align:left; color:" + color + "; font-weight:bold;'>" + numStr + "</span>" +
                                "<span style='color:#fff; white-space:nowrap;'><b>" + st.s + "</b>: " + st.n + "</span>" +
                            "</div>";
                }}).join('');

                var tooltipHTML = "<div style='text-align:left; padding:2px;'>" + tipRows + "</div>";
                return '<span class="crumb-num">' + (i+1) + '.</span>' + 
                       '<span class="sector-tooltip" data-bs-title="' + tooltipHTML + '" style="cursor:help;">' + s.name + '</span>';
            }}).join('<span class="crumb-sep"> > </span>');
        }}

        $('.sector-tooltip').each(function() {{ var old = bootstrap.Tooltip.getInstance(this); if (old) old.dispose(); }});
        $('#rankBreadcrumb').html(getTopSectors(1));
        $('#upvBreadcrumb').html(getTopSectors(11));
        $('#surgeBreadcrumb').html(getTopSectors(14)); 
        $('#mntBreadcrumb').html(getTopSectors(18));  

        $('.sector-tooltip').each(function() {{
            new bootstrap.Tooltip(this, {{
                html: true,
                sanitize: false,
                animation: false,
                container: 'body',
                placement: 'bottom',
                boundary: 'viewport'
                }});
        }});

        // Evaluate dynamic vs global columns on every draw
        recalculateSCTR();
    }}

    function toggleColors() {{
        var t = document.querySelector('table'); var btn = document.getElementById('btnColors');
        t.classList.toggle('no-colors');
        if (t.classList.contains('no-colors')) {{ btn.innerHTML = "🎨"; btn.style.opacity = "0.6"; }} else {{ btn.innerHTML = "🎨"; btn.style.opacity = "1.0"; }}
    }}

    function resetFilters() {{ 
        $('#minPrice, #maxPrice, #minVol, #maxVol').val(''); 
        $('#btnradio1').prop('checked', true); 
        $('input[name="mcapFilter"]').prop('checked', false); 
        $('#mcapAll').prop('checked', true); 
        table.draw(); 
    }}

    function exportTickers() {{
        var data = table.rows({{ search: 'applied', order: 'current', page: 'current' }}).data();
        var tickers = []; 
        data.each(function (value) {{ var clean = value[4].replace(/<[^>]+>/g, '').trim(); if(clean) tickers.push(clean); }});
        if (tickers.length === 0) {{ alert("No visible tickers!"); return; }}
        var blob = new Blob([tickers.join(", ")], {{ type: "text/plain;charset=utf-8" }}); 
        var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "ape_tickers.txt"; document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }}

    function openHeatmapModal(type) {{
        const modal = document.getElementById('heatmapModal');
        const container = document.getElementById('heatmapContainer');
        const title = document.getElementById('heatmapTitle');
        modal.style.display = 'block';
        
        // Set title based on type
        if (type === 'etf') {{
            title.innerHTML = '📈 ETF HEATMAP &nbsp;<span style="font-weight:400; color:#8888a0; font-size:11px;">(End of Day)</span>';
        }} else {{
            title.innerHTML = '🔥 STOCK HEATMAP &nbsp;<span style="font-weight:400; color:#8888a0; font-size:11px;">(End of Day)</span>';
        }}
        
        // Clear previous content
        container.innerHTML = '';
        
        // Create TradingView heatmap widget
        const widgetContainer = document.createElement('div');
        widgetContainer.className = 'tradingview-widget-container';
        widgetContainer.style.width = '100%';
        widgetContainer.style.height = '100%';
        
        const widgetDiv = document.createElement('div');
        widgetDiv.className = 'tradingview-widget-container__widget';
        widgetContainer.appendChild(widgetDiv);
        
        const script = document.createElement('script');
        script.type = 'text/javascript';
        script.async = true;
        
        let config;
        if (type === 'etf') {{
            script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-etf-heatmap.js';
            config = {{
                "dataSource": "AllUSEtf",
                "blockSize": "volume",
                "blockColor": "change",
                "grouping": "asset_class",
                "locale": "en",
                "symbolUrl": "",
                "colorTheme": "dark",
                "hasTopBar": true,
                "isDataSetEnabled": true,
                "isZoomEnabled": true,
                "hasSymbolTooltip": true,
                "isMonoSize": false,
                "width": "100%",
                "height": "100%"
            }};
        }} else {{
            script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js';
            config = {{
                "dataSource": "SPX500",
                "blockSize": "Value.Traded",
                "blockColor": "change",
                "grouping": "sector",
                "locale": "en",
                "symbolUrl": "",
                "colorTheme": "dark",
                "exchanges": [],
                "hasTopBar": true,
                "isDataSetEnabled": true,
                "isZoomEnabled": true,
                "hasSymbolTooltip": true,
                "isMonoSize": false,
                "width": "100%",
                "height": "100%"
            }};
        }}
        
        script.innerHTML = JSON.stringify(config);
        
        widgetContainer.appendChild(script);
        container.appendChild(widgetContainer);
    }}

    function closeHeatmapModal() {{
        const modal = document.getElementById('heatmapModal');
        modal.style.display = 'none';
        const container = document.getElementById('heatmapContainer');
        container.innerHTML = '';
    }}

    // Close modal when clicking outside the modal content
    window.addEventListener('click', function(event) {{
        const modal = document.getElementById('heatmapModal');
        if (event.target === modal) {{
            modal.style.display = 'none';
            const container = document.getElementById('heatmapContainer');
            container.innerHTML = '';
        }}
    }});

    // Track modes for columns 25 (SCTR), 26 (IBD), and 27 (SPY) independently
    let columnModes = {{
        25: "global",
        26: "global",
        27: "global"
    }};

    function toggleColumnMode(event, colIdx, btnId) {{
        if (event) {{
            event.stopPropagation();
            event.preventDefault();
        }}

        const btn = document.getElementById(btnId);
        if (!btn) return;

        // Flip the mode for this specific column
        if (columnModes[colIdx] === "global") {{
            columnModes[colIdx] = "dynamic";
            btn.innerText = "DYNAMIC";
            btn.style.color = "#f5c518";
            btn.style.borderColor = "#f5c518";
        }} else {{
            columnModes[colIdx] = "global";
            btn.innerText = "GLOBAL";
            btn.style.color = "#00d97e";
            btn.style.borderColor = "#00d97e";
        }}
        
        recalculateSCTR();
        
        if ($.fn.DataTable.isDataTable('.table')) {{
            $('.table').DataTable().draw(false);
        }}
    }}

    function recalculateSCTR() {{
        if (!$.fn.DataTable.isDataTable('.table')) return;
        var api = $('.table').DataTable();
        
        // Target columns: 25 (SCTR), 26 (IBD RS), 27 (SPY RS)
        [25, 26, 27].forEach(function(colIdx) {{
            let valClass = (colIdx === 25) ? 'sctr-val' : (colIdx === 26 ? 'ibd-val' : 'spy-val');
            
            // Look up the specific mode for this column
            let currentMode = columnModes[colIdx];
            
            if (currentMode === "dynamic") {{
                // Get all rows that survive the current filters
                var validRows = api.rows({{ filter: 'applied' }}).indexes();
                let visibleSpans = [];

                validRows.each(function(idx) {{
                    var cellHtml = api.cell(idx, colIdx).data(); 
                    if (!cellHtml) return;
                    var rawMatch = cellHtml.match(/data-raw="([^"]+)"/);
                    var globalMatch = cellHtml.match(/data-global="([^"]+)"/);
                    
                    if (rawMatch && globalMatch) {{
                        var rawVal = parseFloat(rawMatch[1]);
                        var globalVal = parseFloat(globalMatch[1]);
                        
                        if (rawVal > -9000) {{
                            visibleSpans.push({{ rowIdx: idx, raw: rawVal, global: globalVal }});
                        }} else {{
                            var newHtml = '<span class="' + valClass + '" data-global="' + globalVal + '" data-raw="' + rawVal + '" style="color:#555568; font-weight:bold;">0.0</span>';
                            api.cell(idx, colIdx).data(newHtml); // Update Memory
                            var node = api.cell(idx, colIdx).node();
                            if (node) node.innerHTML = newHtml; // Update Visuals
                        }}
                    }}
                }});

                if (visibleSpans.length > 0) {{
                    // Sort math lowest to highest
                    visibleSpans.sort(function(a, b) {{ return a.raw - b.raw; }});
                    
                    let total = visibleSpans.length;
                    visibleSpans.forEach(function(item, index) {{
                        // True Percentile Formula in JS: (Rank - 1) / (N - 1)
                        let newSctr = (total > 1) ? (index / (total - 1)) * 99.9 : 99.9;
                        let clr = "#ff4d5a";
                        
                        if (newSctr >= 80) clr = "#00d97e";
                        else if (newSctr >= 40) clr = "#f5c518";
                        
                        var newHtml = '<span class="' + valClass + '" data-global="' + item.global + '" data-raw="' + item.raw + '" style="color:' + clr + '; font-weight:bold;">' + newSctr.toFixed(1) + '</span>';
                        api.cell(item.rowIdx, colIdx).data(newHtml);
                        var node = api.cell(item.rowIdx, colIdx).node();
                        if (node) node.innerHTML = newHtml;
                    }});
                }}
                
            }} else {{
                // Reset ALL rows back to Global for this specific column
                api.rows().indexes().each(function(idx) {{
                    var cellHtml = api.cell(idx, colIdx).data(); 
                    if (!cellHtml) return;
                    var rawMatch = cellHtml.match(/data-raw="([^"]+)"/);
                    var globalMatch = cellHtml.match(/data-global="([^"]+)"/);
                    
                    if (rawMatch && globalMatch) {{
                        var rawVal = parseFloat(rawMatch[1]);
                        var globalVal = parseFloat(globalMatch[1]);
                        
                        let clr = "#ff4d5a";
                        if (rawVal <= -9000) clr = "#555568";
                        else if (globalVal >= 80) clr = "#00d97e";
                        else if (globalVal >= 40) clr = "#f5c518";

                        var newHtml = '<span class="' + valClass + '" data-global="' + globalVal + '" data-raw="' + rawVal + '" style="color:' + clr + '; font-weight:bold;">' + globalVal.toFixed(1) + '</span>';
                        api.cell(idx, colIdx).data(newHtml);
                        var node = api.cell(idx, colIdx).node();
                        if (node) node.innerHTML = newHtml;
                    }}
                }});
            }}
        }});
    }}

    function toggleMcap(type) {{
        if (type === 'all') {{ 
            // Force 'All' to stay checked and uncheck others
            $('#mcapAll').prop('checked', true);
            $('input[name="mcapFilter"]').not('#mcapAll').prop('checked', false); 
        }} 
        else {{ 
            // Uncheck 'All' when a specific cap is selected
            $('#mcapAll').prop('checked', false); 
            
            // If everything is unchecked, default back to 'All'
            if ($('input[name="mcapFilter"]:checked').length === 0) {{ 
                $('#mcapAll').prop('checked', true); 
            }}
        }}
        table.draw(); 
    }}

    function copyTableToClipboard(event) {{ 
        const btn = event.currentTarget; 
        if (!table) return;

        // 1. Get Header Names (only visible columns)
        let headers = [];
        table.columns().every(function() {{
            if (this.visible()) {{
                // Extract text and strip HTML tags
                let headerText = this.header().innerHTML.replace(/<[^>]+>/g, '').trim();
                headers.push(headerText);
            }}
        }});
        
        let textToCopy = headers.join("\\t") + "\\n";

        // 2. Get all row data passing the current filters (across all pages)
        let rowData = table.rows({{ search: 'applied' }}).data();
        
        rowData.each(function(row) {{
            let rowVals = [];
            table.columns().every(function(colIdx) {{
                if (this.visible()) {{
                    // Strip HTML tags from the data cell
                    let cellText = String(row[colIdx]).replace(/<[^>]+>/g, '').trim();
                    rowVals.push(cellText);
                }}
            }});
            textToCopy += rowVals.join("\\t") + "\\n";
        }});

        navigator.clipboard.writeText(textToCopy).then(() => {{
            const originalText = btn.innerHTML; 
            btn.innerHTML = "✅ Copied!"; 
            btn.style.color = "#00d97e"; // Matches your Ape Wisdom theme green
            setTimeout(() => {{ btn.innerHTML = originalText; btn.style.color = ""; }}, 2000);
        }});
    }}

    function copySymbolsToClipboard(event) {{
        const btn = event.currentTarget;
        if (!table) return;
        
        // Grabs all symbols currently passing the active filters
        var data = table.rows({{ search: 'applied' }}).data();
        var tickers = []; 
        data.each(function (value) {{ 
            var clean = String(value[4]).replace(/<[^>]+>/g, '').trim(); 
            if(clean) tickers.push(clean); 
        }});
        
        if (tickers.length === 0) {{ 
            alert("No visible tickers!"); 
            return; 
        }}
        
        // Joins them with a comma and space for easy TradingView pasting
        navigator.clipboard.writeText(tickers.join(", ")).then(() => {{
            const originalText = btn.innerHTML; 
            btn.innerHTML = "✅ Copied!"; 
            btn.style.color = "#00d97e"; // Matches your Ape Wisdom theme green
            setTimeout(() => {{ btn.innerHTML = originalText; btn.style.color = ""; }}, 2000);
        }});
    }}

    $(document).ready(function(){{ 
        table = $('.table').DataTable({{
            "order":[[0,"asc"]], 
            "pageLength": 25,
            "lengthMenu": [[25, 50, 100, 150, 200, 250, -1], [25, 50, 100, 150, 200, 250, "All"]],

            "language": {{
                "search": "",
                "searchPlaceholder": "🔍 Search Symbol, Industry, or Value..."
            }},

            "initComplete": function(settings, json) {{
                $('#page-loader').remove();
                $('body').addClass('loaded');
                this.api().columns.adjust();
            }},

            "columnDefs": [ 
                // Metadata: Type_Tag (28), AvgVol (29), MCap (30) hidden
                {{ "visible": false, "targets": [28, 29, 30] }}, 
                
                // Numeric sorting: Includes GI and shifted technical columns
                {{ "targets": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27], 
                   "type": "num", 
                   "render": function(data, type) {{ 
                       if (type === 'sort' || type === 'type') {{ return parseVal(data); }} 
                       return data; 
                   }} 
                }},

                // Industry (20) as string
                {{ "targets": [20], "type": "string", "render": function(data, type) {{ 
                    if (type === 'sort' || type === 'type') {{ return data.toString().replace(/<[^>]+>/g, '').trim(); }} 
                    return data; 
                }} }}
            ],
            "drawCallback": function() {{ 
                var api = this.api(); 
                $("#stockCounter").text("" + api.rows({{filter:'applied'}}).count() + " / " + api.rows().count() + " Tickers"); 
            }}
        }});

        // --- CUSTOM FILTERING LOGIC ---
        $.fn.dataTable.ext.search.push(function(settings, data) {{
            // UPDATED INDICES:
            var typeTag = data[28] || "";
            var avgVol  = parseVal(data[29]);
            var mcap    = parseVal(data[30]);
            
            var viewMode = $('input[name="btnradio"]:checked').attr('id');
            var isETF = typeTag.includes("ETF");

            // 1. Stock/ETF Toggle
            if (viewMode == 'btnradio2' && isETF) return false; 
            if (viewMode == 'btnradio3' && !isETF) return false; 
            
            // 2. Market Cap Buttons
            if (!$('#mcapAll').is(':checked')) {{
                var match = false;
                if ($('#mcapMega').is(':checked') && mcap >= 200000000000) match = true;
                if ($('#mcapLarge').is(':checked') && (mcap >= 10000000000 && mcap < 200000000000)) match = true;
                if ($('#mcapMid').is(':checked') && (mcap >= 2000000000 && mcap < 10000000000)) match = true;
                if ($('#mcapSmall').is(':checked') && (mcap >= 250000000 && mcap < 2000000000)) match = true;
                if ($('#mcapMicro').is(':checked') && mcap < 250000000) match = true;
                if (!match) return false; 
            }}

            // 3. Price Filter (Index 5 = Price)
            var minP = parseVal($('#minPrice').val()), maxP = parseVal($('#maxPrice').val()); 
            var p = parseVal(data[5]);
            if (minP > 0 && p < minP) return false; 
            if (maxP > 0 && p > maxP) return false;
            
            // 4. Volume Filter (CORRECTED Index 22)
            var minV = parseVal($('#minVol').val()), maxV = parseVal($('#maxVol').val()); 
            if (minV > 0 && avgVol < minV) return false; 
            if (maxV > 0 && avgVol > maxV) return false;
            
            return true;
        }});

        $('#minPrice, #maxPrice, #minVol, #maxVol').on('keyup change', function() {{ table.draw(); }});
        window.redraw = function() {{ table.draw(); }};

        var d = new Date($("#time").data("utc")); 
        $("#time").text(d.toLocaleString('en-US', {{ hour12: false }}).replace(',', ''));
        
        table.on('draw', updateSummary);
        setTimeout(updateSummary, 100);
    }});
</script></body></html>"""
        
        filename = "momentum.html"
        momentum_path = os.path.join(PUBLIC_DIR, filename)
        
        with open(momentum_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        stale_gz = momentum_path + ".gz"
        if os.path.exists(stale_gz):
            os.remove(stale_gz)

        print(f"{C_GREEN}[+] Main dashboard updated at: {momentum_path}{C_RESET}")
        
        return filename
        
    except Exception as e:
        print(f"{C_RED}[!] Error generating HTML: {e}{C_RESET}")
        return None

# ==============================================================================
#                               SECTION 7: MAINTENANCE
# ==============================================================================
def cleanup_old_html_files(hours_to_keep=24):
    """
    Scans the PUBLIC_DIR for scan_*.html files.
    Deletes any that are older than 'hours_to_keep'.
    
    Args:
        hours_to_keep: Number of hours to retain files (default: 24)
    """
    print(f"{C_CYAN}--- Checking for old HTML files to clean up... ---{C_RESET}")
    
    if not os.path.exists(PUBLIC_DIR):
        return

    now = datetime.datetime.now()
    count = 0
    cutoff_seconds = hours_to_keep * 3600
    
    for filename in os.listdir(PUBLIC_DIR):
        # Only target files that match our specific pattern: scan_YYYY-MM-DD_HH-MM.html
        if filename.startswith("scan_") and filename.endswith(".html"):
            try:
                # Extract the date part from the filename
                # format is: scan_2025-01-30_10-00.html
                parts = filename.replace("scan_", "").replace(".html", "").split("_")
                
                if len(parts) >= 2:
                    date_str = parts[0]  # "2025-01-30"
                    file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    
                    # Calculate age in seconds for more precise control
                    age_seconds = (now - file_date).total_seconds()
                    
                    if age_seconds > cutoff_seconds:
                        file_path = os.path.join(PUBLIC_DIR, filename)
                        os.remove(file_path)
                        hours_old = age_seconds / 3600
                        print(f"  > Deleted: {filename} ({hours_old:.1f} hours old)")
                        count += 1
            except Exception as e:
                print(f"  > Skipping {filename}: {e}")
                continue

    if count == 0:
        print(f"{C_GREEN}  > No old files to delete.{C_RESET}")
    else:
        print(f"{C_GREEN}  > Cleanup complete. Removed {count} files.{C_RESET}")

# ==============================================================================
#                               SECTION 8: MAIN EXECUTION FLOW
# ==============================================================================
if __name__ == "__main__":
    # --- CONFIGURATION FOR DATA PERSISTENCE ---
    LATEST_DATA_FILE = os.path.join(SCRIPT_DIR, "latest_scan_data.pkl")
    
    is_auto = "--auto" in sys.argv
    if is_auto:
        print(f"{C_CYAN}Starting Auto Scan...{C_RESET}")

    df = pd.DataFrame()
    skip_fetch = False

    # 1. CHECK COOLDOWN STATUS (Currently disabled/commented out by you)
    # ... (Keep your commented code here if you want) ...

    # 2. LOAD OR FETCH DATA
    if skip_fetch:
        try:
            df = pd.read_pickle(LATEST_DATA_FILE)
            print(f"{C_GREEN}[+] Loaded cached data from {LATEST_DATA_FILE}{C_RESET}")
        except Exception as e:
            print(f"{C_RED}[!] Error loading saved data: {e}. Fetching fresh...{C_RESET}")
            skip_fetch = False

    if not skip_fetch:
        # Fetch fresh data from APIs
        raw = get_all_trending_stocks()
        
        if not raw: 
            print(f"{C_RED}[!] API returned no data.{C_RESET}")
            if df.empty: 
                sys.exit(1)
        
        else:
            new_df = filter_and_process(raw)
            
            if new_df is not None and not new_df.empty:
                df = new_df
                
                # SAVE SUCCESSFUL DATA FOR NEXT TIME
                try:
                    df.to_pickle(LATEST_DATA_FILE)
                except Exception as e:
                    print(f"{C_YELLOW}[!] Warning: Could not save cache: {e}{C_RESET}")
            else:
                print(f"{C_YELLOW}[!] filter_and_process returned no valid data.{C_RESET}")

    # 3. SAFETY CHECK
    if df.empty:
        print(f"{C_RED}[!] Table is empty. Check filters or API connection.{C_RESET}")
        sys.exit(0)

    # 4. GENERATE HTML
    fname = export_interactive_html(df)
    
    # Keep files for 5 hours (cleanup broken file every 5 hours instead of accumulating)
    cleanup_old_html_files(hours_to_keep=5)
    print(f"{C_GREEN}Script execution complete.{C_RESET}")

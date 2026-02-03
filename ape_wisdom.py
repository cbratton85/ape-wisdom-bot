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

# ==========================================
#                   CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(SCRIPT_DIR, "public")
CACHE_FILE = os.path.join(SCRIPT_DIR, "ape_cache.json")
MARKET_DATA_CACHE_FILE = os.path.join(SCRIPT_DIR, "market_data.pkl")
HISTORY_FILE = os.path.join(SCRIPT_DIR, "market_history.json")
DELISTED_CACHE_FILE = os.path.join(SCRIPT_DIR, "delisted_cache.json")
CACHE_EXPIRY_SECONDS = 1800 # 30 minutes
RETENTION_DAYS = 14
DELISTED_RETRY_DAYS = 1
TOOLTIP_HISTORY_DAYS = 12

# --- FILTERS & LAYOUT ---
MIN_PRICE = 1.00
MIN_AVG_VOLUME = 100000
AVG_VOLUME_DAYS = 30
NAME_MAX_WIDTH = 50
LOTTERY_SIZE = 1
REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.0
TICKER_FIXES = {}
PERMANENT_BLACKLIST = ['']

# ANSI COLORS
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_RESET = '\033[0m'

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

class HistoryTracker:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def save(self, df):
        now_ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M")
        cutoff = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - datetime.timedelta(days=RETENTION_DAYS)
        
        # --- EXCLUDE LIST ---
        # Keeps calculated fields safe from being overwritten by raw API data
        exclude_list = [
            'sym', 'name', 'meta', 'history', 'desc', 'type', 'avgvol', 'mcap', 'rolling',
            'z_rank_plus', 'z_surge', 'z_mnt_perc', 'z_upvotes', 'z_accel', 'z_upv_plus', 
            'z_ment', 'z_squeeze', 'type_tag', 'industry/sector', 'heat',
            'velocity', 'accel', 'streak', 'upv_chg'] 

        no_round_list = ['rank', 'rank_plus', 'ment', 'upvotes', 'upv_plus', 'streak']

        precision_map = {
            'price': 2, 'surge': 0, 'mnt_perc': 0, 'squeeze': 0, 
            'conv': 1, 'eff': 1, 'accel': 0, 'velocity': 0, 'heat': 1
        }

        # 3. Main Loop: Process each row in the DataFrame
        for _, row in df.iterrows():
            ticker = row['Sym']
            if ticker not in self.data:
                self.data[ticker] = {}

            entry = {}
            for col, val in row.items():
                col_clean = col.lower().replace('%', '_perc').replace('+', '_plus')
                
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

    def get_metrics(self, ticker, current_price, current_mnt, current_rank_plus, current_upvotes):
        if ticker not in self.data or not self.data[ticker]:
            return {"vel": 0, "accel": 0, "upv_chg": 0, "streak": 0, "rolling_trend": 0, "hist": {}}

        dates = sorted(self.data[ticker].keys())
        
        # --- CALCULATION LOGIC ---
        current_entry = self.data[ticker][dates[-1]]
        prev_entry = self.data[ticker][dates[-2]] if len(dates) > 1 else current_entry

        # === FORCE UPDATE PRICE ===
        # This guarantees the history file gets the fresh price regardless of save() logic
        try:
            current_entry['price'] = round(float(current_price), 2)
            current_entry['upvotes'] = int(current_upvotes)
        except:
            pass # Keep existing if conversion fails

        curr_rank = current_rank_plus 
        prev_rank = prev_entry.get('rank_plus', 0)
        velocity = int(curr_rank - prev_rank)

        prev_upv = prev_entry.get('upvotes', prev_entry.get('Upvotes', 0))
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

        # --- UPDATE THE ENTRY IN MEMORY NOW ---
        current_entry['velocity'] = velocity
        current_entry['accel'] = accel
        current_entry['upv_plus'] = upv_chg
        current_entry['streak'] = rolling_trend
        
        # --- BUILD THE HISTORY MAP ---
        recent_dates = dates[-TOOLTIP_HISTORY_DAYS:]
        history_map = {
            'rank': [], 'rank_plus': [], 'price': [], 'ment': [], 'upvotes': [], 
            'accel': [], 'velocity': [], 'streak': [], 'upv_plus': [],
            'eff': [], 'conv': [], 'surge': [], 'mnt_perc': [], 'squeeze': [], 'master_score': []
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

def fetch_meta_data_robust(ticker):
    name, meta, quote_type, mcap, currency, description = ticker, "Unknown", "EQUITY", 0, "USD", ""

    try:
        dat = yf.Ticker(ticker)
        info = dat.info
        if info:
            quote_type = info.get('quoteType', 'EQUITY')
            name = info.get('shortName') or info.get('longName') or ticker
            mcap = info.get('marketCap', 0)
            currency = info.get('currency', 'USD')
            description = info.get('longBusinessSummary', '')
            
            if quote_type == 'ETF':
                meta = info.get('category', 'Unknown')
            else:
                meta = info.get('industry', 'Unknown')
                meta = meta.replace('\r', '').replace('\n', '').strip()
                if not meta or meta == name or meta == "Unknown - Unknown":
                    meta = "Unknown"
    except Exception as e:
        pass
    return {'ticker': ticker,
            'name': name,
            'meta': meta,
            'type': quote_type,
            'mcap': mcap,
            'currency': currency,
            'description': description
            }

def filter_and_process(stocks):
    if not stocks: return pd.DataFrame()

    # --- LOAD CACHES SEPARATELY ---
    local_cache = load_cache(CACHE_FILE)           
    delisted_cache = load_cache(DELISTED_CACHE_FILE) 
    
    now = datetime.datetime.now(datetime.UTC)
    updated_delisted = False # Track if we need to save the delisted file

    # 1. THE LOTTERY (Random Retry)
    # Instead of checking dates, we pick a few random prisoners to set free and re-test.
    tickers_to_retry = []
    banned_tickers = list(delisted_cache.keys())
    
    if banned_tickers:
        # Don't try to pick more than we have
        draw_count = min(len(banned_tickers), LOTTERY_SIZE)
        
        # Pick random winners
        tickers_to_retry = random.sample(banned_tickers, draw_count)
        
        if tickers_to_retry:
            print(f"{C_GREEN}[+] 🎰 LOTTERY TIME: Re-checking {len(tickers_to_retry)} banned tickers: {tickers_to_retry}{C_RESET}")
            
            # Remove them from the cache so the script treats them as "new" and checks them
            for t in tickers_to_retry:
                if t in delisted_cache:
                    del delisted_cache[t]
            
            # We must save immediately so the loop below sees them as 'valid'
            # If they fail again later, they will be re-added to this file/dict
            updated_delisted = True

    # 2. MAIN FILTER LOOP 
    us_tickers = []
    
    for s in stocks:
        raw_ticker = s['ticker']
        t = TICKER_FIXES.get(raw_ticker, raw_ticker.replace('.', '-'))
        
        if t in PERMANENT_BLACKLIST: continue
        
        # If it's in the delisted cache (and didn't win the lottery), SKIP IT.
        if t in delisted_cache: continue
        
        us_tickers.append(t)
    
    us_tickers = list(set(us_tickers))
    tracker = HistoryTracker(HISTORY_FILE)
    
    # 3. METADATA FETCHING
    missing = [t for t in us_tickers if t not in local_cache and t not in delisted_cache]
    
    if missing:
        print(f"{C_YELLOW}Fetching metadata for {len(missing)} NEW items...{C_RESET}")
        for i, t in enumerate(missing):
            if i % 10 == 0 and i > 0: print(f"  > Progress: {i}/{len(missing)} metadata items fetched...")
            
            res = fetch_meta_data_robust(t)
            
            if res: 
                local_cache[res['ticker']] = res
            else:
                # 404 Error - Metadata Missing
                print(f"{C_RED}  > {t} metadata 404/Not Found. Adding to DELISTED cache.{C_RESET}")
                delisted_cache[t] = {
                    'delisted': True, 
                    'last_checked': now.strftime("%Y-%m-%d"), 
                    'reason': 'Metadata 404'
                }
                updated_delisted = True

            time.sleep(0.75) 

        save_cache(CACHE_FILE, local_cache)

    # 4. MARKET DATA FETCHING
    valid_tickers = [t for t in us_tickers if t not in delisted_cache]
    
    market_data = pd.DataFrame()
    use_cache = os.path.exists(MARKET_DATA_CACHE_FILE) and (time.time() - os.path.getmtime(MARKET_DATA_CACHE_FILE)) < CACHE_EXPIRY_SECONDS
    
    if use_cache:
        print(f"{C_CYAN}[#] Loading market data from cache...{C_RESET}")
        try:
            market_data = pd.read_pickle(MARKET_DATA_CACHE_FILE)
        except:
            print(f"{C_RED}[!] Cache corrupt, re-downloading.{C_RESET}")
            use_cache = False

    if not use_cache:
        print(f"{C_YELLOW}[!] Downloading data for {len(valid_tickers)} tickers...{C_RESET}")
        CHUNK_SIZE = 100
        for i in range(0, len(valid_tickers), CHUNK_SIZE):
            batch = valid_tickers[i:i + CHUNK_SIZE]
            print(f"    > Processing Batch { (i//CHUNK_SIZE) + 1} ({len(batch)} tickers)...")
            try:
                batch_data = yf.download(batch, period="40d", interval="1d", group_by='ticker', progress=False, threads=True)
                if not batch_data.empty:
                    if len(batch) == 1: batch_data.columns = pd.MultiIndex.from_product([batch, batch_data.columns])
                    if market_data.empty: market_data = batch_data
                    else: market_data = pd.concat([market_data, batch_data], axis=1)
                if i + CHUNK_SIZE < len(valid_tickers): time.sleep(2.5) 
            except Exception as e:
                print(f"{C_RED}[!] Error downloading batch {i}: {e}{C_RESET}")

        if not market_data.empty: market_data.to_pickle(MARKET_DATA_CACHE_FILE)

    # 5. BUILD THE DATAFRAME
    final_list = []

    for stock in stocks:
        t = TICKER_FIXES.get(stock['ticker'], stock['ticker'].replace('.', '-'))
        if t in PERMANENT_BLACKLIST or t in delisted_cache: continue
        
        try:
            hist = pd.DataFrame()
            if isinstance(market_data.columns, pd.MultiIndex):
                if t in market_data.columns.levels[0]:
                    hist = market_data[t].dropna()
            else:
                if t in market_data.columns:
                    hist = market_data[t].dropna()

            # --- DEAD TICKER CHECK (LOTTERY FAILURE CATCHER) ---
            # If a lottery winner has NO price data, this line catches it 
            # and immediately throws it back into the delisted_cache.
            if hist.empty: 
                print(f"{C_RED}  > {t} has NO price data. Adding to DELISTED cache.{C_RESET}")
                delisted_cache[t] = {
                    'delisted': True, 
                    'last_checked': now.strftime("%Y-%m-%d"), 
                    'reason': 'No Price Data'
                }
                updated_delisted = True
                continue

            curr_p = hist['Close'].iloc[-1]
            clean_hist = hist['Volume'] 
            actual_vol_days = min(len(clean_hist), AVG_VOLUME_DAYS)
            avg_v = clean_hist.tail(actual_vol_days).mean()
            
            if curr_p < MIN_PRICE or avg_v < MIN_AVG_VOLUME: continue

            info = local_cache.get(t, {})
            if info.get('currency') not in ['USD', None, '']: continue

            name = str(info.get('name', t)).replace('"', '').strip()[:NAME_MAX_WIDTH]
            cur_m = int(stock.get('mentions') or 0)
            old_m = int(stock.get('mentions_24h_ago') or 0)
            
            m_perc = int(((cur_m - old_m) / (old_m if old_m > 0 else 1) * 100))
            s_perc = int((hist['Volume'].iloc[-1] / avg_v * 100)) if avg_v > 0 else 0
            
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

            final_list.append({
                "Rank": rank_now, "Name": name, "Sym": t, "Rank+": rank_plus,
                "Price": float(curr_p), "AvgVol": int(avg_v), "Surge": s_perc,
                "MENT": cur_m, "Mnt%": m_perc, "Type": info.get('type', 'EQUITY'),
                "Upvotes": current_upvotes, "Meta": info.get('meta', '-'),
                "Desc": info.get('description', ''), "Squeeze": squeeze_score,
                "MCap": mcap, "Conv": conviction, "Eff": efficiency,
                "Accel": 0, "Upv+": 0, "Velocity": 0, "Streak": 0, 
                "Rolling": 0, "History": "New"
            })
            
        except Exception as e:
            continue

    # --- SAVE UPDATED DELISTED CACHE IF ANYTHING CHANGED ---
    if updated_delisted:
        print(f"{C_GREEN}[+] Saving updated delisted cache...{C_RESET}")
        save_cache(DELISTED_CACHE_FILE, delisted_cache)

    # 6. SCORING & SAVING (Identical to before)
    df = pd.DataFrame(final_list)
    if not df.empty and 'Sym' in df.columns:
        df = df.drop_duplicates(subset=['Sym'], keep='first')

    if not df.empty:
        cols = ['Rank+', 'Surge', 'Mnt%', 'Upvotes', 'Accel', 'Upv+', 'MENT']
        weights = {'Rank+': 1.1, 'Surge': 1.1, 'Mnt%': 0.7, 'Upvotes': 1.0, 'Accel': 1.2, 'Upv+': 1.0, 'MENT': 0.8}
        
        for col in cols:
            clean_series = df[col].clip(lower=0).astype(float)
            log_data = np.log1p(clean_series)
            mean = log_data.mean(); std = log_data.std(ddof=0)
            df[f'z_{col}'] = 0 if std == 0 else (log_data - mean) / std

        df['Master_Score'] = 0
        for col in cols:
            df['Master_Score'] += df[f'z_{col}'].clip(lower=0) * weights.get(col, 1.0)
        
        sq_series = df['Squeeze'].clip(lower=0).astype(float)
        log_sq = np.log1p(sq_series)
        mean_sq = log_sq.mean(); std_sq = log_sq.std(ddof=0)
        df['z_Squeeze'] = 0 if std_sq == 0 else (log_sq - mean_sq) / std_sq
        df['Heat'] = df['Master_Score']

        tracker.save(df) 

        for index, row in df.iterrows():
            m = tracker.get_metrics(row['Sym'], row['Price'], row['MENT'], row['Rank+'], row['Upvotes'])
            
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

        tracker.flush()
        return df
    
    return pd.DataFrame()

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

def export_interactive_html(df, ai_summary=""):
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

        C_GREEN, C_YELLOW, C_RED, C_CYAN, C_WHITE = "#00ff00", "#ffff00", "#ff4444", "#00ffff", "#ffffff"
        
        if 'AvgVol' not in export_df.columns: export_df['AvgVol'] = 0
        export_df['Vol_Display'] = export_df['AvgVol'].apply(format_vol)
        export_df['Type_Tag'] = 'STOCK'

        if 'MENT' not in export_df.columns: export_df['MENT'] = 0
        
        for index, row in export_df.iterrows():
            m_val = row.get('MENT', 0)
            z_score = row.get('z_MENT', 0)
            
            if z_score >= 2.0: m_clr = "#ffff00"
            elif z_score >= 1.0: m_clr = "#00ff00"
            else: m_clr = "#ffffff"  
            
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
            if ac_val >= 5: ac_clr = "#ff00ff"
            elif ac_val > 0: ac_clr = "#00ffff"
            elif ac_val < 0: ac_clr = "#ff4444"
            else: ac_clr = "#ffffff"
            export_df.at[index, 'Acc'] = with_hist(color_span(f"{ac_val:+d}", ac_clr), ac_hist)

            # --- 3. EFFICIENCY (Eff) ---
            eff_val = float(row.get('Eff', 0)) 
            eff_hist = row.get('h_eff', '') 
            if eff_val >= 1.0: eff_clr = "#00ff00"
            elif eff_val >= 0.5: eff_clr = "#ffff00"
            elif eff_val < 0.1 and eff_val > -0.1: eff_clr = "#666"
            else: eff_clr = "#ff4444"
            export_df.at[index, 'Eff'] = with_hist(color_span(f"{eff_val:.1f}", eff_clr), eff_hist)

            # --- 4. CONVICTION (Conv) ---
            conv_val = float(row.get('Conv', 0)) 
            conv_hist = row.get('h_conv', '') 
            conv_clr = "#ffcc00" if conv_val > 1.0 else "#ffffff"
            export_df.at[index, 'Conv'] = with_hist(color_span(f"{conv_val:.1f}x", conv_clr), conv_hist)

            # --- 5. UPVOTE CHANGE (Upv+) ---
            upchg_val = row.get('Upv+', 0)
            upchg_hist = row.get('h_upv_plus', '') 
            upchg_clr = C_GREEN if upchg_val > 0 else (C_RED if upchg_val < 0 else "#666")
            export_df.at[index, 'Upv+'] = with_hist(color_span(f"{upchg_val:+d}", upchg_clr), upchg_hist)

            # --- 6. STREAK (Strk) ---
            trend_val = row.get('Strk', 0)
            trend_hist = row.get('h_streak', '') 
            sig_text = f"{trend_val:+d}"
            if trend_val >= 3: sig_color = "#00ff00"
            elif trend_val > 0: sig_color = "#99ff99"
            elif trend_val <= -2: sig_color = "#ff4444"
            else: sig_color = "#ffffff"
            export_df.at[index, 'Strk'] = with_hist(color_span(sig_text, sig_color), trend_hist)

            # --- 7. HEAT SCORE ---
            score = float(row.get('Master_Score', 0))
            heat_hist = row.get('h_heat', '') 
            if score > 10: h_clr = "#ff0000"
            elif score > 5: h_clr = "#ff8800"
            elif score > 2: h_clr = "#ffff00"
            else: h_clr = "#888888"
            heat_span = f'<span style="color:{h_clr}; font-weight:bold;">{score:.1f}</span>'
            export_df.at[index, 'Heat'] = with_hist(heat_span, heat_hist)
            
            # --- 8. NAME (Fixed Description Tooltip) ---
            raw_desc = str(row.get('Desc', 'No description available.'))
            desc_text = raw_desc.replace('"', '&quot;').replace("'", "&apos;")
            # Using d-tooltip class here guarantees it uses our new CSS (Single Row, High Z-Index)
            export_df.at[index, 'Name'] = f'<span class="d-tooltip" data-tooltip="{desc_text}" tabindex="0" style="border-bottom:none;"><b>{row.get("Name","")}</b></span>'

            # --- 9. RANK+ ---
            r_val = row.get('Rank+', 0)
            r_hist = row.get('h_rank_plus', '')
            
            if r_val != 0:
                r_color = C_GREEN if r_val > 0 else C_RED
                r_arrow = "▲" if r_val > 0 else "▼"
                r_str = color_span(f"{r_val} {r_arrow}", r_color)
                export_df.at[index, 'Rank+'] = with_hist(r_str, r_hist)
            else:
                export_df.at[index, 'Rank+'] = with_hist('<span style="color:#888">0</span>', r_hist)

            # --- 10. RANK ---
            rank_val = str(row.get('Rank', 0))
            rank_hist = row.get('h_rank', '')
            export_df.at[index, 'Rank'] = with_hist(rank_val, rank_hist)

            # --- 11. SURGE & MNT% ---
            srg_val = f"{export_df.at[index, 'Srg']:.0f}%"
            srg_hist = row.get('h_surge', '')
            srg_z = row.get('z_Surge', 0)
            srg_clr = C_YELLOW if srg_z >= 2.0 else (C_GREEN if srg_z >= 1.0 else C_WHITE)
            export_df.at[index, 'Srg'] = with_hist(color_span(srg_val, srg_clr), srg_hist)

            mnt_val = f"{export_df.at[index, 'Mnt%']:.0f}%"
            mnt_hist = row.get('h_mnt_perc', '')
            mnt_z = row.get('z_Mnt%', 0)
            mnt_clr = C_YELLOW if mnt_z >= 2.0 else (C_GREEN if mnt_z >= 1.0 else C_WHITE)
            export_df.at[index, 'Mnt%'] = with_hist(color_span(mnt_val, mnt_clr), mnt_hist)

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

            # --- 15. ETF BADGE & META ---
            is_fund = row.get('Type', 'EQUITY') == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
            meta_val = row.get('Meta', '-')
            if is_fund:
                badge = '<span style="background-color:#ff00ff; color:black; padding:2px 5px; border-radius:4px; font-size:11px; font-weight:bold; margin-right:6px; vertical-align:middle;">ETF</span>'
            else:
                badge = ""
            
            export_df.at[index, 'Meta'] = f"{badge}{color_span(meta_val, C_WHITE)}"
            export_df.at[index, 'Type_Tag'] = 'ETF' if is_fund else 'STOCK'
            
            # --- 16. SYMBOL & PRICE ---
            t = row['Sym']
            tv_ticker = t.replace('-', '.')
            export_df.at[index, 'Sym'] = f'<a href="https://www.tradingview.com/chart/?symbol={tv_ticker}" target="_blank" style="color: #4da6ff; text-decoration: none;">{t}</a>'
            
            p_val = f"${row.get('Price', 0):.2f}"
            p_hist = row.get('h_price', '')
            export_df.at[index, 'Price'] = with_hist(p_val, p_hist)

            vol_raw = export_df.at[index, 'Vol_Display']
            export_df.at[index, 'Vol_Display'] = f'<div style="text-align: right; padding-right: 25px; color: #ccc;">{vol_raw}</div>'

        export_df.rename(columns={'Meta': 'INDUSTRY/SECTOR', 'Vol_Display': 'VOL(30)'}, inplace=True)

        cols = [
            'Rank', 'Rank+', 'Heat', 'Name', 'Sym', 'Price', 'Acc', 'Eff', 'Conv', 'Upvs', 
            'Upv+', 'VOL(30)', 'Srg', 'Vel', 'Strk', 'MENT', 'Mnt%', 'Sqz', 'INDUSTRY/SECTOR', 
            'Type_Tag', 'AvgVol', 'MCap'
        ]
        for c in cols:
            if c not in export_df.columns:
                export_df[c] = 0

        # --- 1. GENERATE RAW HTML TABLE ---
        raw_table = export_df[cols].to_html(classes='table table-dark table-hover', index=False, escape=False)

        # --- 2. INJECT FAST TOOLTIPS (Find & Replace Headers) ---
        header_map = {
            '<th>Rank</th>': '<th data-tooltip="Current Rank Position">RANK</th>',
            '<th>Rank+</th>': '<th data-tooltip="Rank Change vs 24h ago">RANK+</th>',
            '<th>Heat</th>': '<th data-tooltip="Master Momentum Score (Weighted)">HEAT</th>',
            '<th>Name</th>': '<th data-tooltip="Company Name">NAME</th>',
            '<th>Sym</th>': '<th data-tooltip="Ticker Symbol">SYM</th>',
            '<th>Price</th>': '<th data-tooltip="Current Stock Price">PRICE</th>',
            '<th>Acc</th>': '<th data-tooltip="Acceleration: Speed Change vs 1h ago">ACC</th>',
            '<th>Eff</th>': '<th data-tooltip="Efficiency: Rank gain per unit of volume">EFF</th>',
            '<th>Conv</th>': '<th data-tooltip="Conviction: Upvotes per Mention ratio">CONV</th>',
            '<th>Upvs</th>': '<th data-tooltip="Total Upvotes (24h)">UPVS</th>',
            '<th>Upv+</th>': '<th data-tooltip="New Upvotes gained in last hour">UPV+</th>',
            '<th>VOL(30)</th>': '<th data-tooltip="30-Day Average Volume">VOL(30)</th>',
            '<th>Srg</th>': '<th data-tooltip="Surge: Current Vol vs 30d Avg">SRG</th>',
            '<th>Vel</th>': '<th data-tooltip="Velocity: Rank change speed (1h)">VEL</th>',
            '<th>Strk</th>': '<th data-tooltip="Streak: Hours maintaining direction">STRK</th>',
            '<th>MENT</th>': '<th data-tooltip="Total Mentions (24h)">MENT</th>',
            '<th>Mnt%</th>': '<th data-tooltip="Mention % Change vs 24h ago">MNT%</th>',
            '<th>Sqz</th>': '<th data-tooltip="Short Squeeze Score">SQZ</th>',
            '<th>INDUSTRY/SECTOR</th>': '<th data-tooltip="Sector / Industry Group">INDUSTRY/SECTOR</th>'
        }
        for old_tag, new_tag in header_map.items():
            raw_table = raw_table.replace(old_tag, new_tag)

        table_html = f'<div class="table-scroll-container">{raw_table}</div>'
        utc_timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # --- 3. FINAL HTML TEMPLATE ---
        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Ape Wisdom Analysis</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <style>
            * {{ box-sizing: border-box; }}
            body {{ background-color: #101010; color: #e0e0e0; font-family: 'Consolas', 'Monaco', monospace; padding: 0; margin: 0; overflow-x: hidden; }}
            .table-dark {{ --bs-table-bg:#18181b; color:#ccc; }}
            
            /* GLOBAL TOOLTIP SETTINGS (Fixed High Z-Index & Single Row) */
            th, .d-tooltip {{
                position: relative;
                cursor: help;
            }}

            /* The Black Tooltip Box */
            th[data-tooltip]:not(.sorting):not(.sorting_asc):not(.sorting_desc)::after, .d-tooltip::after {{
                content: attr(data-tooltip); 
                position: absolute;
                top: 100%;    /* Ensures it pops up below the text */
                right: 0;     /* Aligns the right edge of the box to the right edge of the cell */
                left: auto;
                background-color: #000; color: #fff; 
                padding: 8px 12px; border-radius: 6px; border: 1px solid #444;
                font-size: 13px; font-weight: normal; 
                text-transform: none; 
                white-space: nowrap; 
                width: auto; max-width: none;
                z-index: 999999; 
                opacity: 0; visibility: hidden; 
                transition: opacity 0.1s; 
                pointer-events: none; margin-top: 5px;
                box-shadow: 0 4px 15px rgba(0,0,0,1);
            }}

            /* Show on Hover/Focus */
            th:hover::after, .d-tooltip:hover::after,
            th:focus::after, .d-tooltip:focus::after {{ 
                opacity: 1; visibility: visible; 
            }}
            
            /* BOOTSTRAP TOOLTIP OVERRIDES (For the Industry Breadcrumbs at Top) */
            .tooltip-inner {{
                max-width: none !important;
                white-space: nowrap !important;
                background-color: #000 !important;
                color: #fff !important;
                border: 1px solid #444;
            }}
            
            td {{ vertical-align:middle; white-space: nowrap; border-bottom:1px solid #333; padding: 4px 5px !important; font-size: 15px; }}
            table.dataTable {{ width: auto !important; margin: 0 auto; }}
            
            /* COLUMN WIDTHS */
            th:nth-child(1), td:nth-child(1) {{ width: 1%; text-align: center; font-weight: bold; }}
            th:nth-child(2), td:nth-child(2) {{ width: 1%; text-align: center; }}
            th:nth-child(3), td:nth-child(3) {{ width: 1%; text-align: center; font-weight: bold; }}
            th:nth-child(4), td:nth-child(4) {{
                white-space: normal !important;
                width: 350px; /* Set a specific width so it knows where to wrap */
                line-height: 1.4;
                text-align: left;
            }}
            th:nth-child(5), td:nth-child(5) {{ width: 1%; text-align: left; }}
            th:nth-child(6), td:nth-child(6) {{ width: 1%; text-align: right; }}
            th:nth-child(12), td:nth-child(12) {{ width: 1%; text-align: right; }}
            th:nth-child(7), td:nth-child(7), th:nth-child(8), td:nth-child(8), th:nth-child(9), td:nth-child(9),
            th:nth-child(10), td:nth-child(10), th:nth-child(11), td:nth-child(11), th:nth-child(13), td:nth-child(13),
            th:nth-child(14), td:nth-child(14), th:nth-child(15), td:nth-child(15), th:nth-child(16), td:nth-child(16),
            th:nth-child(17), td:nth-child(17), th:nth-child(18), td:nth-child(18) {{ width: 1%; text-align: center; }}
            th:nth-child(19), td:nth-child(19) {{
                min-width: 260px; 
                white-space: nowrap !important;
                overflow: hidden;    
                text-overflow: ellipsis;    
                text-align: left; 
                padding-left: 10px !important; 
                border-right: 1px solid #333; 
            }}
            
            a {{ color:#4da6ff; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
            table.no-colors span {{ color: #ddd !important; font-weight: normal !important; }}
            table.no-colors a {{ color: #4da6ff !important; }}
            
            /* FILTER BAR */
            .filter-bar {{ 
                display: flex; gap: 8px; align-items: center; background: #2a2a2a; padding: 8px; 
                border-radius: 5px; margin-bottom: 15px; border: 1px solid #444; font-size: 0.85rem;
                flex-wrap: nowrap; overflow-x: auto; white-space: nowrap; -ms-overflow-style: none; scrollbar-width: none;
            }}
            .filter-bar::-webkit-scrollbar {{ display: none; }} 
            .filter-group {{ display:flex; align-items:center; gap:4px; }}
            .form-control-sm {{ background: #111; border: 1px solid #555; color: #fff !important; height: 28px; font-size: 0.85rem; padding: 2px 8px; outline: none; }}

            .form-control-sm {{ background: #111; border: 1px solid #555; color: #fff !important; height: 28px; font-size: 0.85rem; padding: 2px 8px; outline: none; }}
            .form-control-sm::placeholder {{ color: #ccc !important; opacity: 1; }}
            .form-control-sm:focus {{ border-color: #00ffff; background: #1a1a1a; }}
            
            .form-control-sm:focus {{ border-color: #00ffff; background: #1a1a1a; }}
            .btn-reset {{ border: 1px solid #555; color: #fff; font-size: 0.8rem; background: #333; }}
            .btn-reset:hover {{ background: #444; color: #fff; }}
            #stockCounter {{ color: #00ff00; font-weight: bold; margin-left: auto; border: 1px solid #00ff00; padding: 2px 8px; border-radius: 4px; }}

            /* HEADER */
            .header-flex {{ display: flex; justify-content: space-between; align-items: center; height: 68px; width: 100%; padding: 0 15px; background: #111; margin-bottom: 5px; box-sizing: border-box; overflow: hidden; }}
            .header-left {{ flex: 0 0 200px; display: flex; align-items: center; z-index: 10; }}
            .header-right {{ flex: 0 0 400px; display: flex; justify-content: flex-end; align-items: center; z-index: 10; }}

            .header-center {{
                position: absolute; left: 50%; transform: translateX(-50%); display: grid; 
                grid-template-columns: max-content max-content; /* Forces labels and content to stay tight */
                gap: 0px 8px; max-width: 90%; /* Increased max-width to give it more horizontal room */
            }}

            .summary-row {{ display: contents; }}
            .row-label {{ font-size: 11px; font-weight: bold; text-transform: uppercase; text-align: right; cursor: help; border-bottom: none !important; text-decoration: underline dotted #555; position: relative; }}


            .row-content {{ font-size: 12px; font-weight: 600; color: #fff; }}

            .crumb-sep {{
                color: #555; 
                margin: 0 8px; 
                font-weight: bold; 
            }}

            .crumb-num {{
                color: #666; 
                margin-right: 4px; 
                font-size: 11px; 
            }}

            .clr-rank {{ color: #00ffff; }} .clr-surge {{ color: #ffcc00; }} .clr-buzz {{ color: #ff00ff; }}

            .sector-tooltip {{
                white-space: nowrap;
            }}
            
            /* LEGEND - RESTORED FULL VERSION */
            .legend-container {{ background-color: #222; border: 1px solid #444; border-radius: 6px; margin-bottom: 10px; overflow: hidden; }}
            .legend-header {{ background: #2a2a2a; padding: 4px 12px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: bold; color: #fff; }}
            .legend-box {{ padding: 5px; display: none; background-color: #1a1a1a; }}
            .legend-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; width: 100%; }}
            .legend-col {{ background: #222; border: 1px solid #333; padding: 6px; border-radius: 5px; }}
            .legend-title {{ color: #00ff00; font-weight: bold; border-bottom: 1px solid #444; margin-bottom: 5px; font-size: 0.85rem; text-transform: uppercase; padding-bottom: 2px; }}
            .legend-row {{ display: flex; align-items: flex-start; margin-bottom: 0px; font-size: 13px; border-bottom: 1px dashed #333; padding: 2px 0; line-height: 1.2; }}
            .metric-name {{ color: #00ffff; font-weight: bold; width: 60px; flex-shrink: 0; }}
            .metric-math {{ color: #888; font-family: monospace; font-size: 0.75rem; margin-right: 10px; flex-shrink: 0; }}
            .metric-desc {{ color: #ccc; }}
            .color-key {{ width: 80px; font-weight: bold; flex-shrink: 0; }}
            .color-desc {{ color: #bbb; }}
            @media (max-width: 900px) {{ .legend-grid {{ grid-template-columns: 1fr; }} }}

            /* TOP LABELS TOOLTIPS (Header Summary) */
            .row-label::after {{
                content: attr(data-tooltip); position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
                background-color: #000; color: #fff; padding: 8px 12px; border-radius: 6px; border: 1px solid #444;
                font-size: 11px; font-weight: normal; text-transform: none; 
                white-space: nowrap; width: auto; max-width: none;
                z-index: 999999 !important; opacity: 0; visibility: hidden; position: fixed; top: auto; transition: opacity 0.1s; pointer-events: none; margin-top: 5px;
            }}

            .row-label:hover::after {{ opacity: 1; visibility: visible; }}

            /* DATATABLES SEARCH */
            .dataTables_wrapper .data_tables_header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
            .dataTables_filter {{ position: absolute; left: 50%; transform: translateX(-50%); margin: 0 !important; float: none !important; }}
            .dataTables_filter input {{ width: 100% !important; max-width: 450px !important; background: #181818 !important; color: #fff !important; border: 1px solid #333 !important; border-radius: 20px !important; padding: 6px 20px !important; outline: none !important; }}
            .dataTables_filter input:focus {{ border-color: #00ffff !important; }}
            
            /* PAGINATION */
            .page-link {{ background-color: #222; border-color: #444; color: #00ff00; }}
            .page-item.active .page-link {{ background-color: #00ff00; border-color: #00ff00; color: #000; }}
            .page-item.disabled .page-link {{ background-color: #111; border-color: #333; color: #555; }}
            
            .mode-toggle label {{ margin-left: 15px; display: flex; align-items: center; background: #222; padding: 3px; border-radius: 4px; cursor: pointer; border: 1px solid #444; }}
            #modeSwitch {{ display: none; }}
            #modeSwitch:checked + label .e-label {{ color: #fff; background: #333; }}
            #modeSwitch:not(:checked) + label .s-label {{ color: #fff; background: #333; }}

            td:nth-child(4) .d-tooltip::after {{
                white-space: normal !important;
                width: 1000px !important;
                text-align: left;
                text-justify: none !important;
                word-spacing: normal;
                line-height: 1.2;
            }}

            tr:hover {{
                position: relative;
                z-index: 100;
            }}

            th:nth-child(-n+5)::after, td:nth-child(-n+5) .d-tooltip::after {{
                left: 0 !important;
                right: auto !important;
            }}

            #time {{
                font-family: monospace; 
                font-size: 14px !important; 
                font-weight: bold; 
                color: #00ff00; /* Changed to Green to match your theme */
            }}

            table.dataTable thead > tr > th.sorting:before,
            table.dataTable thead .sorting:after, 
            table.dataTable thead .sorting_asc:after, 
            table.dataTable thead .sorting_desc:after {{
                display: inline-block !important;
                visibility: visible !important;
                opacity: 0.6 !important;
                position: relative !important;
                margin-left: 10px !important;
                top: 0 !important;
            }}

            th.sorting::after, th.sorting_asc::after, th.sorting_desc::after {{ content: none !important; }}
            th.sorting::before, th.sorting_asc::before, th.sorting_desc::before {{ content: none !important; }}

        </style>
        </head>
        <body>
        <div class="container-fluid" style="width: 98%; max-width: 2500px; margin: 0 auto;">
            
            <div class="header-flex">
                <div class="header-left">
                    <a href="https://apewisdom.io" target="_blank">
                        <img src="https://apewisdom.io/apewisdom-logo.svg" alt="Ape Wisdom" style="height: 54px;">
                    </a>
                    <div class="mode-toggle">
                        <input type="checkbox" id="modeSwitch" onclick="updateSummary()">
                        <label for="modeSwitch">
                            <span class="mode-label s-label" style="font-size:12px; font-weight:bold; padding:5px 12px; color:#666;">STOCKS</span>
                            <span class="mode-label e-label" style="font-size:12px; font-weight:bold; padding:5px 12px; color:#666;">ETFS</span>
                        </label>
                    </div>
                </div>

                <div class="header-center">
                    <div class="summary-row">
                        <span class="row-label clr-rank" data-tooltip="Total Rank Change by Industry. Shows the sum of all position gains in the sector.">RANK CLIMBERS:</span>
                        <span id="rankBreadcrumb" class="row-content">...</span>
                    </div>
                    <div class="summary-row">
                        <span class="row-label clr-surge" data-tooltip="Total Volume Surge by Industry. Shows the combined volume pressure of all stocks in the sector.">VOL SURGE:</span>
                        <span id="surgeBreadcrumb" class="row-content">...</span>
                    </div>
                    <div class="summary-row">
                        <span class="row-label clr-buzz" data-tooltip="Total Mention Growth by Industry. Sum of all new chatter in the sector.">SOCIAL BUZZ:</span>
                        <span id="mntBreadcrumb" class="row-content">...</span>
                    </div>
                </div>

                <div class="header-right">
                    <span id="time" data-utc="{utc_timestamp}" style="font-family:monospace; font-size:11px; color:#666;">Loading...</span>
                </div>
            </div>

            <div class="filter-bar">
                <span style="color:#fff; font-weight:bold; margin-right:5px;">⚡ FILTERS:</span>
                <button id="btnColors" class="btn btn-sm btn-reset" onclick="toggleColors()" style="margin-right: 5px;">🎨 Colors: ON</button>
                <button class="btn btn-sm btn-reset" onclick="resetFilters()" title="Reset Filters">🔄</button>

                <div class="filter-group" style="margin-left: 10px; margin-right: 10px;">
                    <label>Price:</label>
                    <input type="text" id="minPrice" class="form-control form-control-sm" placeholder="Min" style="width: 50px;">
                    <span style="color:#666">-</span>
                    <input type="text" id="maxPrice" class="form-control form-control-sm" placeholder="Max" style="width: 50px;">
                </div>
                
                <div class="filter-group" style="margin-right: 10px;">
                    <label>Avg Vol:</label>
                    <input type="text" id="minVol" class="form-control form-control-sm" placeholder="Min" style="width: 50px;">
                    <span style="color:#666">-</span>
                    <input type="text" id="maxVol" class="form-control form-control-sm" placeholder="Max" style="width: 50px;">
                </div>

                <div class="filter-group">
                    <div class="btn-group" role="group" style="margin-right: 10px;">
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio1" checked onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio1" style="font-size: 0.8rem; padding: 4px 8px;">All</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio2" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio2" style="font-size: 0.8rem; padding: 4px 8px;">Stocks</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio3" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio3" style="font-size: 0.8rem; padding: 4px 8px;">ETFs</label>
                    </div>

                    <div class="btn-group" role="group">
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapAll" checked onclick="toggleMcap('all')">
                        <label class="btn btn-outline-light btn-sm" for="mcapAll" style="font-size: 0.8rem; padding: 4px 8px;">All</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMega" onclick="toggleMcap('mega')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMega" style="font-size: 0.8rem; padding: 4px 8px;">Mega</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapLarge" onclick="toggleMcap('large')">
                        <label class="btn btn-outline-light btn-sm" for="mcapLarge" style="font-size: 0.8rem; padding: 4px 8px;">Lrg</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMid" onclick="toggleMcap('mid')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMid" style="font-size: 0.8rem; padding: 4px 8px;">Mid</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapSmall" onclick="toggleMcap('small')">
                        <label class="btn btn-outline-light btn-sm" for="mcapSmall" style="font-size: 0.8rem; padding: 4px 8px;">Sml</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMicro" onclick="toggleMcap('micro')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMicro" style="font-size: 0.8rem; padding: 4px 8px;">Mic</label>
                    </div>
                </div>

                <button class="btn btn-sm btn-reset" onclick="exportTickers()" title="Download Ticker List" style="margin-left: 10px;">TXT File</button>
                <button class="btn btn-sm btn-reset" onclick="copyTableToClipboard(event)" title="Copy Table" style="margin-left: 10px;">📋 Copy</button>
                
                <span id="stockCounter">Loading...</span>
            </div>

            <div class="legend-container">
                <div class="legend-header" onclick="toggleLegend()">
                    <span>ℹ️ DATA DEFINITIONS & COLOR GUIDE (Click to Toggle)</span>
                    <span id="legendArrow">▼</span>
                </div>
                <div class="legend-box" id="legendContent">
                    <div class="legend-grid">
                        
                        <div class="legend-col">
                            <div class="legend-title">📉 Column Definitions</div>
                            
                            <div class="legend-row">
                                <span class="metric-name">RANK</span>
                                <span class="metric-math">Current Pos</span>
                                <span class="metric-desc">Current rank in the popularity list.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">RANK+</span>
                                <span class="metric-math">Rank(Yest) - Rank(Today)</span>
                                <span class="metric-desc">Positions changed vs 24h ago.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">HEAT</span>
                                <span class="metric-math">Master Score</span>
                                <span class="metric-desc">Weighted aggregate of all momentum signals.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">ACC</span>
                                <span class="metric-math">Vel(Now) - Vel(1h ago)</span>
                                <span class="metric-desc">Acceleration: (Rate of change of speed).</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">EFF</span>
                                <span class="metric-math">Rank+ / Surge</span>
                                <span class="metric-desc">Efficiency: Rank gain per unit of volume.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">CONV</span>
                                <span class="metric-math">Upvotes / Mentions</span>
                                <span class="metric-desc">Conviction: Sentiment Quality Ratio.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">UPVS</span>
                                <span class="metric-math">Raw Count</span>
                                <span class="metric-desc">Total upvotes in last 24h.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">UPV+</span>
                                <span class="metric-math">Upv(Now) - Upv(1hr ago)</span>
                                <span class="metric-desc">New upvotes gained since <b>Last Scan (1h)</b>.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">VOL</span>
                                <span class="metric-math">30-Day Mean</span>
                                <span class="metric-desc">Average daily trading volume.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">SRG</span>
                                <span class="metric-math">(Vol / Avg) * 100</span>
                                <span class="metric-desc">Surge: Current volume as % of 30-day Avg.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">VEL</span>
                                <span class="metric-math">Rank+(Now) - Rank+(1hr ago)</span>
                                <span class="metric-desc">Hourly change in Rank+. (Speeding up vs 1h ago?)</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">STRK</span>
                                <span class="metric-math">Hourly Streak</span>
                                <span class="metric-desc">Streak: Consecutive <b>HOURS</b> sustaining direction.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">MNT%</span>
                                <span class="metric-math">% Change</span>
                                <span class="metric-desc">Percent change in mentions vs 24h ago.</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">SQZ</span>
                                <span class="metric-math">Mnt * Surge / log(MCap)</span>
                                <span class="metric-desc">Short Squeeze Score (Vol+Chatter/Cap).</span>
                            </div>
                        </div>

                        <div class="legend-col">
                            <div class="legend-title">🎨 Color Indicators</div>

                            <div class="legend-row">
                                <span class="color-key">RANK</span>
                                <span class="color-desc">White (Standard).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">RANK+</span>
                                <span class="color-desc"><span style="color:#00ff00">Green</span> (Climbing), <span style="color:#ff4444">Red</span> (Falling).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">HEAT</span>
                                <span class="color-desc"><span style="color:#ff0000">Red</span> (> 2.0σ), <span style="color:#ff8800">Orange</span> (> 1.5σ), <span style="color:#ffff00">Yellow</span> (> 1σ).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">ACC</span>
                                <span class="color-desc">
                                    <span style="color:#ff00ff">Magenta</span> (Expl. ≥5), 
                                    <span style="color:#00ffff">Cyan</span> (Fast >0), 
                                    <span style="color:#ffffff">White</span> (Steady 0), 
                                    <span style="color:#ff4444">Red</span> (Slow <0).
                                </span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">EFF</span>
                                <span class="color-desc"><span style="color:#00ff00">Green</span> (> 1.0), <span style="color:#ffff00">Yellow</span> (> 0.5), <span style="color:#ff4444">Red</span> (Low).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">CONV</span>
                                <span class="color-desc"><span style="color:#ffcc00">Gold</span> (> 1.0x), White (Diluted).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">UPVS</span>
                                <span class="color-desc"><span style="color:#00ff00">Green</span> (High Activity > 1.5σ).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">UPV+</span>
                                <span class="color-desc"><span style="color:#00ff00">Green</span> (Positive), <span style="color:#ff4444">Red</span> (Negative).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">VOL</span>
                                <span class="color-desc"><span style="color:#ccc">Gray</span> (Static Stat).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">SRG</span>
                                <span class="color-desc"><span style="color:#ffff00">Yellow</span> (Anomaly > 2σ), <span style="color:#00ff00">Green</span> (High > 1σ).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">VEL</span>
                                <span class="color-desc"><span style="color:#00ff00">Green</span> (Speeding Up), <span style="color:#ff4444">Red</span> (Slowing).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">STRK</span>
                                <span class="color-desc"><span style="color:#00ff00">Green</span> (3+ Hours), <span style="color:#ff4444">Red</span> (Reversing).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">MNT%</span>
                                <span class="color-desc"><span style="color:#ffff00">Yellow</span> (> 2σ), <span style="color:#00ff00">Green</span> (> 1σ).</span>
                            </div>
                            <div class="legend-row">
                                <span class="color-key">SQZ</span>
                                <span class="color-desc"><span style="color:#00ffff">Cyan</span> (Score > 1.5σ), White (Normal).</span>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
            
            {table_html}
        </div>
        
        <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
        <script>
    var table;
    function parseVal(str) {{
        if (!str) return 0;
        var clean = str.toString().replace(/<[^>]+>/g, '').replace(/[$,%▲▼+]/g, '').trim().toLowerCase();
        let mult = 1;
        if (clean.endsWith('k')) {{ mult = 1000; clean = clean.replace('k', ''); }}
        else if (clean.endsWith('m')) {{ mult = 1000000; clean = clean.replace('m', ''); }}
        else if (clean.endsWith('b')) {{ mult = 1000000000; clean = clean.replace('b', ''); }}
        return parseFloat(clean) * mult || 0;
    }}

    function updateSummary() {{
        if (!$.fn.DataTable.isDataTable('.table')) return;
        var api = $('.table').DataTable();
        var topSwitchIsETF = $('#modeSwitch').is(':checked');
        var allData = api.rows().data();

        function getTopSectors(metricIdx) {{
            var sectorData = {{}};
            allData.each(function(row) {{
                var rawType = row[19].toString().replace(/<[^>]+>/g, ''); 
                if (topSwitchIsETF && !rawType.includes('ETF')) return;
                if (!topSwitchIsETF && rawType.includes('ETF')) return;

                var sector = row[18].toString().replace(/<[^>]+>/g, '').trim().replace(/^ETF/i, ''); 
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
            sorted.sort(function(a, b) {{ return b.total - a.total; }});
            if (sorted.length === 0) return '<span style="color:#666;">---</span>';

            var topThree = sorted.slice(0, 5);
            return topThree.map(function(s, i) {{
                s.stocks.sort(function(a, b) {{ return b.v - a.v; }});
                // SHOW ALL STOCKS
                var topStocks = s.stocks; 
                
                var tipRows = topStocks.map(function(st) {{
                    var numStr = st.v > 0 ? '+' + Math.round(st.v) : Math.round(st.v);
                    var color = st.v >= 0 ? '#00ff00' : '#ff4444';
                    return "<div style='display:flex; justify-content:flex-start; align-items:center; font-size:11px; margin-bottom:1px;'>" +
                                "<span style='min-width:45px; text-align:left; color:" + color + "; font-weight:bold;'>" + numStr + "</span>" +
                                "<span style='color:#fff; white-space:nowrap;'><b>" + st.s + "</b>: " + st.n + "</span>" +
                            "</div>";
                }}).join('');

                var tooltipHTML = "<div style='text-align:left; padding:2px;'>" + tipRows + "</div>";
                return '<span class="crumb-num">' + (i+1) + '.</span>' + 
                       '<span class="sector-tooltip" data-bs-title="' + tooltipHTML + '" style="cursor:help; border-bottom:1px dotted #555;">' + s.name + '</span>';
            }}).join('<span class="crumb-sep"> > </span>');
        }}

        $('.sector-tooltip').each(function() {{ var old = bootstrap.Tooltip.getInstance(this); if (old) old.dispose(); }});
        $('#rankBreadcrumb').html(getTopSectors(1));  
        $('#surgeBreadcrumb').html(getTopSectors(12)); 
        $('#mntBreadcrumb').html(getTopSectors(16));  

        $('.sector-tooltip').each(function() {{
            new bootstrap.Tooltip(this, {{ html: true, sanitize: false, animation: false, container: 'body' }});
        }});
    }}

    function toggleLegend() {{
        var x = document.getElementById("legendContent"); var arrow = document.getElementById("legendArrow");
        if (x.style.display === "block") {{ x.style.display = "none"; arrow.innerText = "▼"; }} else {{ x.style.display = "block"; arrow.innerText = "▲"; }}
    }}

    function toggleColors() {{
        var t = document.querySelector('table'); var btn = document.getElementById('btnColors');
        t.classList.toggle('no-colors');
        if (t.classList.contains('no-colors')) {{ btn.innerHTML = "🎨 Colors: OFF"; btn.style.opacity = "0.6"; }} else {{ btn.innerHTML = "🎨 Colors: ON"; btn.style.opacity = "1.0"; }}
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

    function toggleMcap(type) {{
        if (type === 'all') {{ $('input[name="mcapFilter"]').not('#mcapAll').prop('checked', false); }} 
        else {{ $('#mcapAll').prop('checked', false); if ($('input[name="mcapFilter"]:checked').length === 0) {{ $('#mcapAll').prop('checked', true); }} }}
        table.draw(); 
    }}

    function copyTableToClipboard(event) {{ 
        const btn = event.currentTarget; const table = document.querySelector(".table");
        if (!table) return;
        let rows = Array.from(table.querySelectorAll("tr"));
        let textToCopy = rows.map(row => {{
            let cells = Array.from(row.querySelectorAll("th, td"));
            return cells.map(cell => cell.innerText.trim()).join("\\t");
        }}).join("\\n");
        navigator.clipboard.writeText(textToCopy).then(() => {{
            const originalText = btn.innerHTML; btn.innerHTML = "✅ Copied!"; btn.style.color = "#00ff00";
            setTimeout(() => {{ btn.innerHTML = originalText; btn.style.color = ""; }}, 2000);
        }});
    }}

    $(document).ready(function(){{ 
        table = $('.table').DataTable({{
            "order":[[0,"asc"]], "pageLength": 15, "lengthMenu": [[15, 25, 50, 100, 250, -1], [15, 25, 50, 100, 250, "All"]],
            "columnDefs": [ 
                {{ "visible": false, "targets": [19, 20, 21] }}, 
                {{ "targets": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "type": "num", "render": function(data, type) {{ if (type === 'sort' || type === 'type') {{ return parseVal(data); }} return data; }} }},
                {{ "targets": [18], "type": "string", "render": function(data, type) {{ if (type === 'sort' || type === 'type') {{ return data.toString().replace(/<[^>]+>/g, '').trim(); }} return data; }} }}
            ],
            "drawCallback": function() {{ var api = this.api(); $("#stockCounter").text("Showing " + api.rows({{filter:'applied'}}).count() + " / " + api.rows().count() + " Tickers"); }}
        }});

        $.fn.dataTable.ext.search.push(function(settings, data) {{
            var typeTag = data[19] || ""; var viewMode = $('input[name="btnradio"]:checked').attr('id');
            var isETF = typeTag.includes("ETF");
            if (viewMode == 'btnradio2' && isETF) return false; 
            if (viewMode == 'btnradio3' && !isETF) return false; 
            
            if (!$('#mcapAll').is(':checked')) {{
                var mcap = parseVal(data[21]); var match = false;
                if ($('#mcapMega').is(':checked') && mcap >= 200000000000) match = true;
                if ($('#mcapLarge').is(':checked') && (mcap >= 10000000000 && mcap < 200000000000)) match = true;
                if ($('#mcapMid').is(':checked') && (mcap >= 2000000000 && mcap < 10000000000)) match = true;
                if ($('#mcapSmall').is(':checked') && (mcap >= 250000000 && mcap < 2000000000)) match = true;
                if ($('#mcapMicro').is(':checked') && mcap < 250000000) match = true;
                if (!match) return false; 
            }}

            var minP = parseVal($('#minPrice').val()), maxP = parseVal($('#maxPrice').val()); var p = parseVal(data[5]);
            if (minP > 0 && p < minP) return false; if (maxP > 0 && p > maxP) return false;
            
            var minV = parseVal($('#minVol').val()), maxV = parseVal($('#maxVol').val()); var v = parseVal(data[20]);
            if (minV > 0 && v < minV) return false; if (maxV > 0 && v > maxV) return false;
            
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
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"scan_{timestamp}.html"
        filepath = os.path.join(PUBLIC_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)
        index_path = os.path.join(PUBLIC_DIR, "index.html")
        shutil.copy(filepath, index_path)
        print(f"{C_GREEN}[+] Dashboard generated at: {filepath}{C_RESET}")
        return filename
    except Exception as e:
        print(f"{C_RED}[!] Error generating HTML: {e}{C_RESET}")
        return None

def send_discord_link(filename, ai_summary):
    print(f"\n{C_YELLOW}--- Sending Link to Discord... ---{C_RESET}")
    DISCORD_URL = os.environ.get('DISCORD_WEBHOOK')
    REPO_NAME = os.environ.get('GITHUB_REPOSITORY') 

    if not DISCORD_URL:
        print(f"{C_RED}[!] Error: DISCORD_WEBHOOK is missing. Check GitHub Secrets.{C_RESET}")
        return
    if not REPO_NAME:
        print(f"{C_RED}[!] Error: GITHUB_REPOSITORY is missing.{C_RESET}")
        return

    try:
        user, repo = REPO_NAME.split('/')
        website_url = f"https://{user}.github.io/{repo}/{filename}"
        
        msg = (f"🦍 **APE Wisdom Scanner**\n"
               f"🔗 **[Click Here to Open Dashboard]({website_url})**\n"
               f"*(Note: It may take ~30s for the link to go live)*")

        response = requests.post(DISCORD_URL, json={"content": msg})
        
        if response.status_code == 204:
            print(f"{C_GREEN}[+] Discord Link Sent Successfully!{C_RESET}")
        else:
            print(f"{C_RED}[!] Discord Failed: {response.status_code} - {response.text}{C_RESET}")

    except Exception as e:
        print(f"{C_RED}[!] Exception sending Discord link: {e}{C_RESET}")

def cleanup_old_html_files(days_to_keep=14):
    """
    Scans the PUBLIC_DIR for scan_*.html files.
    Deletes any that are older than 'days_to_keep'.
    """
    print(f"{C_CYAN}--- Checking for old HTML files to clean up... ---{C_RESET}")
    
    if not os.path.exists(PUBLIC_DIR):
        return

    now = datetime.datetime.now()
    count = 0
    
    for filename in os.listdir(PUBLIC_DIR):
        # Only target files that match our specific pattern: scan_YYYY-MM-DD_HH-MM.html
        if filename.startswith("scan_") and filename.endswith(".html"):
            try:
                # Extract the date part from the filename
                # format is: scan_2025-01-30_10-00.html
                # We split by '_' and take the middle part (date) and last part (time)
                parts = filename.replace("scan_", "").replace(".html", "").split("_")
                
                if len(parts) >= 2:
                    date_str = parts[0] # "2025-01-30"
                    file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    
                    # Calculate age
                    age_days = (now - file_date).days
                    
                    if age_days > days_to_keep:
                        file_path = os.path.join(PUBLIC_DIR, filename)
                        os.remove(file_path)
                        print(f"  > Deleted old file: {filename} ({age_days} days old)")
                        count += 1
            except Exception as e:
                # If a file has a weird name, just skip it
                print(f"  > Skipping check for {filename}: {e}")
                continue

    if count == 0:
        print(f"{C_GREEN}  > No old files found to delete.{C_RESET}")
    else:
        print(f"{C_GREEN}  > Cleanup complete. Removed {count} files.{C_RESET}")

if __name__ == "__main__":
    # --- CONFIGURATION FOR DATA PERSISTENCE ---
    LOCK_FILE = os.path.join(SCRIPT_DIR, "last_scan_time.txt")
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
            # Only exit if we don't have a backup dataframe loaded
            if df.empty: 
                sys.exit(1)
        
        else:
            new_df = filter_and_process(raw)
            
            # --- CRITICAL FIX: Check for None before checking .empty ---
            if new_df is not None and not new_df.empty:
                df = new_df
                
                # SAVE SUCCESSFUL DATA FOR NEXT TIME
                try:
                    df.to_pickle(LATEST_DATA_FILE)
                    # UPDATE TIMESTAMP
                    with open(LOCK_FILE, "w") as f: 
                        f.write(str(time.time()))
                except Exception as e:
                    print(f"{C_YELLOW}[!] Warning: Could not save cache: {e}{C_RESET}")
            else:
                print(f"{C_YELLOW}[!] filter_and_process returned no valid data.{C_RESET}")

    # 3. SAFETY CHECK
    if df.empty:
        print(f"{C_RED}[!] Table is empty. Check filters or API connection.{C_RESET}")
        sys.exit(0)

    # 4. GENERATE HTML
    # (Note: tracker.save(df) happens INSIDE filter_and_process now, so we don't call it here)
    fname = export_interactive_html(df)
    
    if fname:
        status_msg = "🚀 **Market Scan Updated**" if not skip_fetch else "🔄 **Dashboard Refreshed (Cached)**"
        send_discord_link(fname, status_msg)
        
    cleanup_old_html_files(days_to_keep=7)
    print(f"{C_GREEN}Done.{C_RESET}")

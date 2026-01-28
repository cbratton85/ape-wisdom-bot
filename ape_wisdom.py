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
CACHE_EXPIRY_SECONDS = 3600 
RETENTION_DAYS = 14          
DELISTED_RETRY_DAYS = 7        

# --- FILTERS & LAYOUT ---
MIN_PRICE = 0.01             
MIN_AVG_VOLUME = 100        
AVG_VOLUME_DAYS = 30
PAGE_SIZE = 30

# --- UPDATED WIDTHS ---
NAME_MAX_WIDTH = 50       
INDUSTRY_MAX_WIDTH = 60  
COL_WIDTHS = [50, 8, 8, 10, 8, 8, 8, 8, INDUSTRY_MAX_WIDTH] 
DASH_LINE = "-" * 170    

REQUEST_DELAY_MIN = 1.5 
REQUEST_DELAY_MAX = 3.0
TICKER_FIXES = {'GPS': 'GAP', 'FB': 'META', 'APE': 'AMC'}

# ANSI COLORS
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_RESET = '\033[0m'
C_MAGENTA = '\033[95m'
C_BOLD = '\033[1m'

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
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        for _, row in df.iterrows():
            ticker = row['Sym']
            if ticker not in self.data: self.data[ticker] = {}
            # --- CHANGE: Added 'upvotes' to saved data ---
            self.data[ticker][today] = {
                "rank_plus": int(row.get('Rank+', 0)),
                "price": float(row.get('Price', 0)),
                "mnt_perc": float(row.get('Mnt%', 0)),
                "upvotes": int(row.get('Upvotes', 0))
            }
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=RETENTION_DAYS)
        for ticker in list(self.data.keys()):
            self.data[ticker] = {d: v for d, v in self.data[ticker].items() if datetime.datetime.strptime(d, "%Y-%m-%d") > cutoff}
            if not self.data[ticker]: del self.data[ticker]
        with open(self.filepath, 'w') as f: json.dump(self.data, f, indent=4)

    def get_metrics(self, ticker, current_price, current_mnt):
        if ticker not in self.data or len(self.data[ticker]) < 2: 
            return {"vel": 0, "accel": 0, "upv_chg": 0, "div": False, "streak": 0, "rolling_trend": 0}

        dates = sorted(self.data[ticker].keys())
        today_data = self.data[ticker][dates[-1]]
        prev_data = self.data[ticker][dates[-2]]
        
        # --- ROLLING TREND (STREAK) ---
        rolling_trend = 0
        for d in dates:
            r_plus = self.data[ticker][d].get('rank_plus', 0)
            if r_plus > 0:
                rolling_trend = rolling_trend + 1 if rolling_trend >= 0 else 1
            elif r_plus < 0:
                rolling_trend = rolling_trend - 1 if rolling_trend <= 0 else -1
            else:
                rolling_trend = 0
        
        # --- NEW METRICS ---
        velocity = int(today_data.get('rank_plus', 0) - prev_data.get('rank_plus', 0))
        upv_chg = int(today_data.get('upvotes', 0) - prev_data.get('upvotes', 0))
        
        # Acceleration (Velocity Today - Velocity Yesterday)
        accel = 0
        if len(dates) >= 3:
            prev_2_data = self.data[ticker][dates[-3]]
            prev_vel = int(prev_data.get('rank_plus', 0) - prev_2_data.get('rank_plus', 0))
            accel = velocity - prev_vel

        return {"vel": velocity, "accel": accel, "upv_chg": upv_chg, "div": False, "streak": len(dates), "rolling_trend": rolling_trend}

def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}
def save_cache(cache_data):
    try:
        with open(CACHE_FILE, 'w') as f: json.dump(cache_data, f, indent=4)
    except: pass

def fetch_meta_data_robust(ticker):
    name, meta, quote_type, mcap, currency = ticker, "Unknown", "EQUITY", 0, "USD"
    try:
        dat = yf.Ticker(ticker) 
        info = dat.info
        if info:
            quote_type = info.get('quoteType', 'EQUITY')
            name = info.get('shortName') or info.get('longName') or ticker
            mcap = info.get('marketCap', 0)
            currency = info.get('currency', 'USD')
            
            if quote_type == 'ETF':
                meta = info.get('category', 'Unknown')
            else:
                s = info.get('sector', 'Unknown')
                i = info.get('industry', 'Unknown')
                # 1. Strip the junk characters
                meta = meta.replace('\r', '').replace('\n', '').strip()
            
                # 2. Fallback to 'Unknown' if it's empty or just the company name
                if not meta or meta == name or meta == "Unknown - Unknown":
                    meta = "Unknown"
                # ------------------------
    except: pass
    return {'ticker': ticker, 'name': name, 'meta': meta, 'type': quote_type, 'mcap': mcap, 'currency': currency}

def load_delisted():
    if os.path.exists(DELISTED_CACHE_FILE):
        try:
            with open(DELISTED_CACHE_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_delisted(data):
    try:
        with open(DELISTED_CACHE_FILE, 'w') as f: json.dump(data, f, indent=4)
    except: pass

def filter_and_process(stocks):
    if not stocks: return pd.DataFrame()
    
    # 1. Setup and Deduplication
    us_tickers = list(set([TICKER_FIXES.get(s['ticker'], s['ticker'].replace('.', '-')) for s in stocks]))
    local_cache = load_cache()

    tracker = HistoryTracker(HISTORY_FILE)

    now = datetime.datetime.utcnow()
    valid_tickers = []
    
    print(f"Processing {len(us_tickers)} tickers...")
    
    # 2. Clean Cache & Heal Unknown Metadata
    # This identifies tickers that are either delisted OR have "Unknown" in their meta field
    blacklist = [
        t for t in us_tickers 
        if local_cache.get(t, {}).get('delisted') or local_cache.get(t, {}).get('meta') == "Unknown"
    ]
    
    if blacklist:
        t = random.choice(blacklist)
        stock_data = local_cache[t]
        
        # Determine why it's in the blacklist
        is_unknown = stock_data.get('meta') == "Unknown"
        
        # Check timing for delisted retries
        last_checked_str = stock_data.get('last_checked', '2000-01-01')
        last_checked = datetime.datetime.strptime(last_checked_str, "%Y-%m-%d")

        # Healing logic: Unknowns get retried immediately; Delisted wait for the retry window
        if is_unknown or (now - last_checked).days >= DELISTED_RETRY_DAYS:
            reason = "Healing Metadata" if is_unknown else "Retry Delisted"
            print(f"{C_YELLOW}[!] Lottery Check ({reason}): Retrying {t}...{C_RESET}")
            del local_cache[t]

    # Re-calculate valid tickers to exclude current delisted ones
    valid_tickers = [t for t in us_tickers if not local_cache.get(t, {}).get('delisted')]
    
    # 3. Fetch Missing Metadata (This now handles the "Healed" tickers too)
    missing = [t for t in valid_tickers if t not in local_cache]
    if missing:
        print(f"Fetching metadata for {len(missing)} items...")
        for t in missing:
            try:
                res = fetch_meta_data_robust(t)
                if res: 
                    # Add a timestamp so the retry logic knows when this was last fetched
                    res['last_checked'] = datetime.datetime.utcnow().strftime("%Y-%m-%d")
                    local_cache[res['ticker']] = res
            except: 
                pass
        save_cache(local_cache)

    # 4. Download Market Data
    market_data = None
    use_cache = os.path.exists(MARKET_DATA_CACHE_FILE) and (time.time() - os.path.getmtime(MARKET_DATA_CACHE_FILE)) < CACHE_EXPIRY_SECONDS
    
    if use_cache: 
        market_data = pd.read_pickle(MARKET_DATA_CACHE_FILE)
    else:
        print(f"Downloading data for {len(valid_tickers)} tickers...")
        market_data = yf.download(valid_tickers, period="40d", interval="1d", group_by='ticker', progress=False, threads=True)
        if not market_data.empty: market_data.to_pickle(MARKET_DATA_CACHE_FILE)

    if len(valid_tickers) == 1 and not market_data.empty:
        market_data.columns = pd.MultiIndex.from_product([valid_tickers, market_data.columns])

    # 5. Build the List
    final_list = []
    for stock in stocks:
        t = TICKER_FIXES.get(stock['ticker'], stock['ticker'].replace('.', '-'))
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                if t not in market_data.columns.levels[0]: 
                    local_cache[t] = {'delisted': True, 'last_checked': datetime.datetime.utcnow().strftime("%Y-%m-%d")}
                    continue
                hist = market_data[t].dropna()
            else: 
                hist = market_data.dropna()

            if hist.empty: continue

            curr_p = hist['Close'].iloc[-1]
            avg_v = hist['Volume'].tail(AVG_VOLUME_DAYS).mean()
            
            if curr_p < MIN_PRICE or avg_v < MIN_AVG_VOLUME: continue

            info = local_cache.get(t, {})
            if info.get('currency', 'USD') != 'USD': continue

            name = str(info.get('name', t)).replace('"', '').replace('\r', '').replace('\n', '').strip()[:NAME_MAX_WIDTH]
            
            cur_m = int(stock.get('mentions', 0))
            old_m = int(stock.get('mentions_24h_ago', 0))
            m_perc = int(((cur_m - old_m) / (old_m if old_m > 0 else 1) * 100))
            s_perc = int((hist['Volume'].iloc[-1] / avg_v * 100)) if avg_v > 0 else 0
            
            try:
                mcap = float(info.get('mcap', 0) or 0)
            except: 
                mcap = 0
            
            log_mcap = math.log(mcap if mcap > 0 else 10**9, 10)
            squeeze_score = (cur_m * s_perc) / max(log_mcap, 1)

            rank_now = int(stock.get('rank', 0))
            rank_old = int(stock.get('rank_24h_ago', 0))
            rank_plus = (rank_old - rank_now) if rank_old != 0 else 0

            # --- NEW METRIC CALCULATIONS ---
            conviction = (int(stock.get('upvotes', 0)) / cur_m) if cur_m > 0 else 0
            safe_surge = s_perc if s_perc > 0 else 1
            efficiency = rank_plus / safe_surge

            m = tracker.get_metrics(t, float(curr_p), m_perc)

            final_list.append({
                "Rank": rank_now, "Name": name, "Sym": t, "Rank+": rank_plus,
                "Price": float(curr_p), 
                "AvgVol": int(avg_v),    # <--- IMPORTANT: Ensure this line exists
                "Surge": s_perc, "Mnt%": m_perc, "Type": info.get('type', 'EQUITY'),
                "Upvotes": int(stock.get('upvotes', 0)), "Meta": info.get('meta', '-'), 
                "Squeeze": squeeze_score, 
                "MCap": mcap,
                "Conv": conviction, 
                "Eff": efficiency,

                "Accel": m['accel'],
                "Upv+": m['upv_chg'],
                "Velocity": m['vel'],
                "Streak": m['streak'],
                "Rolling": m['rolling_trend'],
                "Divergence": m['div'] 
            })
        except Exception as e: 
            continue
    
    # 6. Scoring
    df = pd.DataFrame(final_list)
    if not df.empty:
        cols = ['Rank+', 'Surge', 'Mnt%', 'Upvotes', 'Accel', 'Upv+']
        weights = {
            'Rank+': 1.1, 
            'Surge': 1.1, 
            'Mnt%': 0.7, 
            'Upvotes': 1.0,
            'Accel': 1.2,
            'Upv+': 1.0 
        }
        
        for col in cols:
            # np.log1p smooths out massive outliers so they don't crush the mean
            clean_series = df[col].clip(lower=0).astype(float)
            log_data = np.log1p(clean_series)
            
            mean = log_data.mean()
            std = log_data.std(ddof=0)
            
            # Create the Z-score (standard deviations from the mean)
            df[f'z_{col}'] = 0 if std == 0 else (log_data - mean) / std

        # --- 3. MASTER SCORE CALCULATION ---
        df['Master_Score'] = 0
        for col in cols:
            # We clip(lower=0) so that being "worse than average" doesn't 
            # actively subtract from the heat score; it just adds 0.
            df['Master_Score'] += df[f'z_{col}'].clip(lower=0) * weights.get(col, 1.0)

        # Handle Squeeze Score Z-indexing
        if 'Squeeze' in df.columns:
            sq_series = df['Squeeze'].clip(lower=0).astype(float)
            log_sq = np.log1p(sq_series)
            mean_sq = log_sq.mean()
            std_sq = log_sq.std(ddof=0)
            df['z_Squeeze'] = 0 if std_sq == 0 else (log_sq - mean_sq) / std_sq
        else:
            df['z_Squeeze'] = 0

    # ==========================================
    #   FINALIZATION (Outside the 'if' block)
    # ==========================================
    # 1. Save the history with the new metrics
    tracker.save(df)
    
    # 2. Update the local metadata cache
    save_cache(local_cache)
    
    # 3. CRITICAL: Return the DataFrame so the main script can use it
    # This fixes the 'NoneType' AttributeError
    return df

def get_all_trending_stocks():
    all_results, page = [], 1
    print(f"{C_CYAN}--- API: Fetching list of trending stocks ---{C_RESET}")
    while True:
        try:
            r = requests.get(f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                results = data.get('results', [])
                if not results: break
                all_results.extend(results)
                page += 1
            else: break
        except: break
    return all_results

def export_interactive_html(df):
    try:
        export_df = df.copy().astype(object)
        if not os.path.exists(PUBLIC_DIR): os.makedirs(PUBLIC_DIR)

        def color_span(text, color_hex): return f'<span style="color: {color_hex}; font-weight: bold;">{text}</span>'
        def format_vol(v):
            if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
            if v >= 1_000: return f"{v/1_000:.0f}K"
            return str(v)

        C_GREEN, C_YELLOW, C_RED, C_CYAN, C_MAGENTA, C_WHITE = "#00ff00", "#ffff00", "#ff4444", "#00ffff", "#ff00ff", "#ffffff"
        export_df['Type_Tag'] = 'STOCK'
        export_df['Sig'] = ""
        export_df['Vol_Display'] = export_df['AvgVol'].apply(format_vol)

        for index, row in export_df.iterrows():
            # Velocity
            v_val = row['Velocity']
            if v_val != 0:
                v_color = C_GREEN if v_val > 0 else C_RED
                export_df.at[index, 'Velocity'] = color_span(f"{v_val:+d}", v_color)
            
            # 1. Acceleration (Sector-Relative)
            # We pulled this from the 'Rel_Acc' column we created in Step 6
            ac_val = float(row.get('Rel_Acc', 0)) # Using the new sector-relative math

            # Color Logic: Magnitude of outperformance vs Industry peers
            if ac_val >= 25.0: ac_clr = "#ff00ff"   # Magenta: Explosive breakout
            elif ac_val > 5.0: ac_clr = "#00ffff"   # Cyan: Strong outperformance
            elif ac_val < -5.0: ac_clr = "#ff4444"  # Red: Sector laggard
            else: ac_clr = "#ffffff"                # White: Moving with the crowd

            export_df.at[index, 'Accel'] = f'<span style="color:{ac_clr};">{ac_val:+.1f}%</span>'

            # 2. Efficiency
            eff_val = row['Eff']
            if eff_val >= 1.0: eff_clr = "#00ff00"
            elif eff_val >= 0.5: eff_clr = "#ffff00"
            elif eff_val < 0.1 and eff_val > -0.1: eff_clr = "#666"
            else: eff_clr = "#ff4444"
            export_df.at[index, 'Eff'] = color_span(f"{eff_val:.1f}", eff_clr)

            # 3. Conviction
            conv_val = row['Conv']
            conv_clr = "#ffcc00" if conv_val > 1.0 else "#ffffff"
            export_df.at[index, 'Conv'] = color_span(f"{conv_val:.1f}x", conv_clr)

            # 4. Upvote Change
            upchg_val = row['Upv+']
            upchg_clr = C_GREEN if upchg_val > 0 else (C_RED if upchg_val < 0 else "#666")
            export_df.at[index, 'Upv+'] = color_span(f"{upchg_val:+d}", upchg_clr)

            # Signals (The +/- 3 Streak)
            trend_val = row['Rolling']
            sig_text = f"{trend_val:+d}"
            if trend_val >= 3: sig_color = "#00ff00"
            elif trend_val > 0: sig_color = "#99ff99"
            elif trend_val <= -2: sig_color = "#ff4444"
            else: sig_color = "#ffffff"
            
            export_df.at[index, 'Sig'] = color_span(sig_text, sig_color)

            # Heat Score
            score = row['Master_Score']
            mean_s, std_s = df['Master_Score'].mean(), df['Master_Score'].std()
            z_heat = (score - mean_s) / std_s if std_s > 0 else 0
            
            if z_heat > 2.0: h_clr = "#ff0000"
            elif z_heat > 1.5: h_clr = "#ff8800"
            elif z_heat > 1.0: h_clr = "#ffff00"
            else: h_clr = "#888888"

            z_cols = [
                ('Rank+', row.get('z_Rank+', 0)),
                ('Surge', row.get('z_Surge', 0)),
                ('Mnt%', row.get('z_Mnt%', 0)),
                ('Upvs', row.get('z_Upvotes', 0)),
                ('Accel', row.get('z_Accel', 0))
            ]
            
            mvp_pair = max(z_cols, key=lambda x: x[1])
            mvp_metric = mvp_pair[0]
            mvp_val = mvp_pair[1]
            
            heat_html = f'<span style="color:{h_clr}; font-weight:bold;">{score:.1f}</span>'
            
            export_df.at[index, 'Name'] = f"<b>{row['Name']}</b>"
            export_df.at[index, 'Heat'] = heat_html
            
            # Rank+
            r_val = row['Rank+']
            if r_val != 0:
                r_color = C_GREEN if r_val > 0 else C_RED
                r_arrow = "▲" if r_val > 0 else "▼"
                export_df.at[index, 'Rank+'] = color_span(f"{r_val} {r_arrow}", r_color)
            else:
                export_df.at[index, 'Rank+'] = ""

            # Columns
            for col, z_col in [('Surge', 'z_Surge'), ('Mnt%', 'z_Mnt%')]:
                val = f"{row[col]:.0f}%"
                clr = C_YELLOW if row[z_col] >= 2.0 else (C_GREEN if row[z_col] >= 1.0 else C_WHITE)
                export_df.at[index, col] = color_span(val, clr)
            
            # Squeeze
            sq_z = row.get('z_Squeeze', 0)
            sq_color = C_CYAN if sq_z > 1.5 else C_WHITE
            export_df.at[index, 'Squeeze'] = color_span(int(row['Squeeze']), sq_color)
            
            export_df.at[index, 'Upvotes'] = color_span(row['Upvotes'], C_GREEN if row['z_Upvotes']>1.5 else C_WHITE)
            
            # ETF Badge
            is_fund = row['Type'] == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
            if is_fund:
                badge = '<span style="background-color:#ff00ff; color:black; padding:2px 5px; border-radius:4px; font-size:11px; font-weight:bold; margin-right:6px; vertical-align:middle;">ETF</span>'
                meta_text = color_span(row['Meta'], C_WHITE)
            else:
                badge = ""
                meta_text = color_span(row['Meta'], C_WHITE)

            export_df.at[index, 'Meta'] = f"{badge}{meta_text}"
            export_df.at[index, 'Type_Tag'] = 'ETF' if is_fund else 'STOCK'
            
            t = row['Sym']
            export_df.at[index, 'Sym'] = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color: #4da6ff; text-decoration: none;">{t}</a>'
            export_df.at[index, 'Price'] = f"${row['Price']:.2f}"
            export_df.at[index, 'Vol_Display'] = color_span(export_df.at[index, 'Vol_Display'], "#ccc")

        # 1. Drop the original raw integer 'Streak' column so it doesn't collide
        if 'Streak' in export_df.columns:
            export_df.drop(columns=['Streak'], inplace=True)

        # 2. Rename columns (Now 'Sig' becomes the ONLY 'Streak' column)
        export_df.rename(columns={
            'Meta': 'INDUSTRY/SECTOR',
            'Velocity': 'Vel', 
            'Vol_Display': 'Vol',
            'Sig': 'Strk',
            'Accel': 'Acc',
            'Squeeze': 'Sqz',
            'Upvotes': 'Upvs',
            'Surge': 'Srg'
            }, inplace=True)

        # The 'Shopping List'
        cols = [
            'Rank', 'Rank+', 'Heat', 'Name', 'Sym', 'Price', 
            'Acc', 'Eff', 'Conv', 'Upvs', 'Upv+',
            'Vol', 'Srg', 'Vel', 'Strk', 'Mnt%', 
            'Sqz', 'INDUSTRY/SECTOR', 'Type_Tag', 'AvgVol', 'MCap'
        ]
        final_df = export_df[cols]
        table_html = final_df.to_html(classes='table table-dark table-hover', index=False, escape=False)
        utc_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # HTML TEMPLATE
        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Ape Wisdom Analysis</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <style>
            body{{
                background-color:#101010; color:#e0e0e0; font-family:'Consolas','Monaco',monospace; padding:20px;
            }}
            .table-dark{{--bs-table-bg:#18181b;color:#ccc}}
            
            th{{
                color:#00ff00; border-bottom:2px solid #444;
                font-size: 15px;
                text-transform: uppercase;
                /* Changed padding to: Top:8px, Right:20px, Bottom:8px, Left:4px */
                vertical-align: middle !important; padding: 8px 22px 8px 6px !important; line-height: 1.2 !important;
            }}

            td{{
                vertical-align:middle; white-space: nowrap; border-bottom:1px solid #333; 
                padding: 4px 5px !important;
                font-size: 15px;
            }}

            table.dataTable {{ width: auto !important; margin: 0 auto; }}
            
            th:nth-child(1), td:nth-child(1) {{ width: 1%; text-align: center; }} /*RANK*/

            th:nth-child(2), td:nth-child(2) {{ width: 1%; text-align: center; }} /*RANK+*/

            th:nth-child(3), td:nth-child(3) {{ width: 1%; text-align: center; font-weight: bold; }} /*HEAT*/

            th:nth-child(4), td:nth-child(4) {{ max-width: 260px; overflow: hidden; text-overflow: ellipsis; }} /*NAME*/
            
            th:nth-child(5), td:nth-child(5) {{ width: 1%; text-align: left; }} /*SYM*/

            th:nth-child(6), td:nth-child(6) {{ width: 1%; text-align: right; }} /*PRICE*/

            th:nth-child(7), td:nth-child(7) {{ width: 1%; text-align: center; }} /*ACC*/

            th:nth-child(8), td:nth-child(8) {{ width: 1%; text-align: center; }} /*EFF*/

            th:nth-child(9), td:nth-child(9) {{ width: 1%; text-align: center; }} /*CONV*/

            th:nth-child(10), td:nth-child(10) {{ width: 1%; text-align: center; }} /*UPVS*/

            th:nth-child(11), td:nth-child(11) {{ width: 1%; text-align: center; }} /*UPV+*/

            th:nth-child(12), td:nth-child(12) {{ width: 1%; text-align: right; }} /*VOL*/

            th:nth-child(13), td:nth-child(13) {{ width: 1%; text-align: center; }} /*SRG*/

            th:nth-child(14), td:nth-child(14) {{ width: 1%; text-align: center; }} /*VEL*/

            th:nth-child(15), td:nth-child(15) {{ width: 1%; text-align: center; }} /*STRK*/
            
            th:nth-child(16), td:nth-child(16) {{ width: 1%; text-align: center; }} /*MNT%*/

            th:nth-child(17), td:nth-child(17) {{ width: 1%; text-align: center; }} /*SQZ*/

            th:nth-child(18), td:nth-child(18) {{ 
                max-width: 210px; 
                overflow: hidden; 
                text-overflow: ellipsis; 
                text-align: left; 
                padding-left: 10px !important; 
                border-right: 1px solid #333; 
            }}
            
            td{{vertical-align:middle; white-space: nowrap; border-bottom:1px solid #333;}} 
            a{{color:#4da6ff; text-decoration:none;}} a:hover{{text-decoration:underline;}}
            table.no-colors span {{ color: #ddd !important; font-weight: normal !important; }}
            table.no-colors a {{ color: #4da6ff !important; }}
            
            .legend-container {{ background-color: #222; border: 1px solid #444; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }}
            .legend-header {{ background: #2a2a2a; padding: 10px 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: bold; color: #fff; }}
            .legend-box {{
                padding: 8px;
                display: none;
                background-color: #1a1a1a;
            }}
            
            /* --- 2-COLUMN GRID LEGEND --- */
            
            .legend-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; width: 100%; }}
            .legend-col {{ background: #222;
                border: 1px solid #333;
                padding: 6px;
                border-radius: 5px;
            }}
            .legend-title {{ color: #00ff00; font-weight: bold; border-bottom: 1px solid #444; margin-bottom: 8px; font-size: 0.95rem; text-transform: uppercase; }}
            .legend-row {{
                display: flex;
                align-items: flex-start;
                margin-bottom: 1px;
                font-size: 15px;
                border-bottom: 1px dashed #333;
                padding-bottom: 1px;
            }}
            
            .metric-name {{ color: #00ffff; font-weight: bold; width: 60px; flex-shrink: 0; }}
            .metric-math {{ color: #888; font-family: monospace; font-size: 0.75rem; margin-right: 10px; flex-shrink: 0; }}
            .metric-desc {{ color: #ccc; }}
            
            .color-key {{ width: 80px; font-weight: bold; flex-shrink: 0; }}
            .color-desc {{ color: #bbb; }}
            
            @media (max-width: 900px) {{ .legend-grid {{ grid-template-columns: 1fr; }} }}

            .filter-bar {{ display: flex; gap: 8px; align-items: center; background: #2a2a2a; padding: 8px; border-radius: 5px; margin-bottom: 15px; border: 1px solid #444; flex-wrap: wrap; font-size: 0.85rem; }}
            .filter-group {{ display:flex; align-items:center; gap:4px; }}
            .form-control-sm {{ background:#111; border:1px solid #555; color:#fff; height: 28px; font-size: 0.8rem; padding: 2px 5px; }}
            .btn-reset {{ border: 1px solid #555; color: #fff; font-size: 0.8rem; background: #333; }}
            .btn-reset:hover {{ background: #444; color: #fff; }}
            
            #stockCounter {{ color: #00ff00; font-weight: bold; margin-left: auto; border: 1px solid #00ff00; padding: 2px 8px; border-radius: 4px;}}
            .header-flex {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
            .page-link {{ background-color: #222; border-color: #444; color: #00ff00; }}
            .page-item.active .page-link {{ background-color: #00ff00; border-color: #00ff00; color: #000; }}
            .page-item.disabled .page-link {{ background-color: #111; border-color: #333; color: #555; }}

            .dataTables_filter input {{
                width: 350px !important; background: #111 !important; border: 1px solid #555 !important;
                color: #fff !important; height: 28px !important; border-radius: 4px; margin-left: 10px;
            }}
        </style>
        </head>
        <body>
        <div class="container-fluid" style="width: fit-content; margin: 0 auto;">
            <div class="header-flex">
                <a href="https://apewisdom.io" target="_blank" style="text-decoration: none;">
                <img src="https://apewisdom.io/apewisdom-logo.svg" alt="Ape Wisdom" style="height: 60px; vertical-align: middle; margin-right: 15px;">
                </a>
                <span id="time" data-utc="{utc_timestamp}" style="font-size: 0.9rem; color: #888;">Loading...</span>
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
                    <input type="text" id="minVol" class="form-control form-control-sm" placeholder="500k" style="width: 50px;">
                    <span style="color:#666">-</span>
                    <input type="text" id="maxVol" class="form-control form-control-sm" placeholder="10m" style="width: 50px;">
                </div>

                <div class="filter-group">
                    <div class="btn-group" role="group" style="margin-right: 10px;">
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio1" checked onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio1" style="font-size: 0.75rem; padding: 2px 8px;">All</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio2" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio2" style="font-size: 0.75rem; padding: 2px 8px;">Stocks</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio3" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio3" style="font-size: 0.75rem; padding: 2px 8px;">ETFs</label>
                    </div>

                    <div class="btn-group" role="group">
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapAll" checked onclick="toggleMcap('all')">
                        <label class="btn btn-outline-light btn-sm" for="mcapAll" style="font-size: 0.75rem; padding: 2px 6px;">All</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMega" onclick="toggleMcap('mega')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMega" style="font-size: 0.75rem; padding: 2px 6px;" title="> $200B">Mega</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapLarge" onclick="toggleMcap('large')">
                        <label class="btn btn-outline-light btn-sm" for="mcapLarge" style="font-size: 0.75rem; padding: 2px 6px;" title="$10B - $200B">Lrg</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMid" onclick="toggleMcap('mid')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMid" style="font-size: 0.75rem; padding: 2px 6px;" title="$2B - $10B">Mid</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapSmall" onclick="toggleMcap('small')">
                        <label class="btn btn-outline-light btn-sm" for="mcapSmall" style="font-size: 0.75rem; padding: 2px 6px;" title="$250M - $2B">Sml</label>
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMicro" onclick="toggleMcap('micro')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMicro" style="font-size: 0.75rem; padding: 2px 6px;" title="< $250M">Mic</label>
                    </div>
                </div>

                <button class="btn btn-sm btn-reset" onclick="exportTickers()" title="Download Ticker List" style="margin-left: 10px;">Download TXT File</button>
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
                                <span class="metric-math">(Accel / Sector_Avg_Rank+) * 100</span>
                                <span class="metric-desc">Sector-Relative Acceleration: Speed gain vs industry peers.</span>
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
                                <span class="metric-math">Upv(Today) - Upv(Yest)</span>
                                <span class="metric-desc">Net change in upvotes vs 24h ago.</span>
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
                                <span class="metric-math">Rank+(Today) - Rank+(Yest)</span>
                                <span class="metric-desc">Velocity: Change in climb speed?</span>
                            </div>
                            <div class="legend-row">
                                <span class="metric-name">STRK</span>
                                <span class="metric-math">Consecutive Days</span>
                                <span class="metric-desc">Streak: Days sustaining current direction.</span>
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
                                <span class="color-desc"><span style="color:#00ff00">Green</span> (3+ Days), <span style="color:#ff4444">Red</span> (Reversing).</span>
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
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
        <script>
        
        function toggleMcap(type) {{
            if (type === 'all') {{
                $('input[name="mcapFilter"]').not('#mcapAll').prop('checked', false);
            }} else {{
                $('#mcapAll').prop('checked', false);
                if ($('input[name="mcapFilter"]:checked').length === 0) {{ $('#mcapAll').prop('checked', true); }}
            }}
            table.draw();
        }}
        function toggleLegend() {{
            var x = document.getElementById("legendContent"); var arrow = document.getElementById("legendArrow");
            if (x.style.display === "block") {{ x.style.display = "none"; arrow.innerText = "▼"; }} else {{ x.style.display = "block"; arrow.innerText = "▲"; }}
        }}
        function toggleColors() {{
            var table = document.querySelector('table'); var btn = document.getElementById('btnColors');
            table.classList.toggle('no-colors');
            if (table.classList.contains('no-colors')) {{ btn.innerHTML = "🎨 Colors: OFF"; btn.style.opacity = "0.6"; }} else {{ btn.innerHTML = "🎨 Colors: ON"; btn.style.opacity = "1.0"; }}
        }}
        function parseVal(str) {{
            if (!str) return 0;
            str = str.toString().toLowerCase().replace(/,/g, '').trim();

            let mult = 1;
            if (str.endsWith('k')) {{
                mult = 1000;
                str = str.replace('k', '');
            }} else if (str.endsWith('m')) {{
                mult = 1000000;
                str = str.replace('m', '');
            }} else if (str.endsWith('b')) {{
                mult = 1000000000;
                str = str.replace('b', '');
        }}
            // 3. Convert to float and multiply
            return parseFloat(str) * mult || 0;
        }}
        function resetFilters() {{ $('#minPrice, #maxPrice, #minVol, #maxVol').val(''); $('#btnradio1').prop('checked', true); $('input[name="mcapFilter"]').prop('checked', false); $('#mcapAll').prop('checked', true); table.draw();  }}
        function exportTickers() {{
            var table = $('.table').DataTable(); var data = table.rows({{ search: 'applied', order: 'current', page: 'current' }}).data();
            var tickers = []; data.each(function (value) {{ var div = document.createElement("div"); div.innerHTML = value[4]; var text = div.textContent || div.innerText || ""; if(text) tickers.push(text.trim()); }});
            if (tickers.length === 0) {{ alert("No visible tickers!"); return; }}
            var blob = new Blob([tickers.join(" ")], {{ type: "text/plain;charset=utf-8" }}); var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "ape_tickers_page.txt"; document.body.appendChild(a); a.click(); document.body.removeChild(a);
        }}
        $(document).ready(function(){{ 
            table=$('.table').DataTable({{
                "order":[[0,"asc"]], "pageLength": 25, "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                "columnDefs": [ 
                    {{ "visible": false, "targets": [18, 19, 20] }},

                    {{
                        "targets": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                        "type": "num",
                        "render": function(data, type) {{ if (type === 'sort' || type === 'type') {{ var clean = data.toString().replace(/<[^>]+>/g, '').replace(/[$,%+,x]/g, ''); return parseVal(clean); }} return data; }} }},
                    {{ "targets": [13], "type": "num", "render": function(data, type) {{ var clean = data.toString().replace(/<[^>]+>/g, '').replace(/,/g, ''); var val = parseFloat(clean) || 0; if (type === 'sort' || type === 'type') return val; if (val < 0) return '<span style="color:#ff4444">' + val + '</span>'; else return '<span style="color:#00ff00">' + val + '</span>'; }} }}
                ],
                "drawCallback": function() {{ var api = this.api(); $("#stockCounter").text("Showing " + api.rows({{filter:'applied'}}).count() + " / " + api.rows().count() + " Tickers"); }}
            }});
            
            $.fn.dataTable.ext.search.push(function(settings, data) {{
                // STOCK TYPE FILTER (Type_Tag)
                var typeTag = data[18] || ""; 
                var viewMode = $('input[name="btnradio"]:checked').attr('id');
                if (viewMode == 'btnradio2' && typeTag == 'ETF') return false;
                if (viewMode == 'btnradio3' && typeTag == 'STOCK') return false;
                
                // MARKET CAP FILTER (MCap)
                if (!$('#mcapAll').is(':checked')) {{
                    var mcap = parseFloat(data[20]) || 0; 
                    var match = false;
                    if ($('#mcapMega').is(':checked') && mcap >= 200000000000) match = true;
                    if ($('#mcapLarge').is(':checked') && (mcap >= 10000000000 && mcap < 200000000000)) match = true;
                    if ($('#mcapMid').is(':checked') && (mcap >= 2000000000 && mcap < 10000000000)) match = true;
                    if ($('#mcapSmall').is(':checked') && (mcap >= 250000000 && mcap < 2000000000)) match = true;
                    if ($('#mcapMicro').is(':checked') && mcap < 250000000) match = true;
                    if (!match) return false; 
                }}

                // PRICE & VOL FILTERS
                var minP = parseVal($('#minPrice').val()), maxP = parseVal($('#maxPrice').val());
                var p = parseFloat((data[5] || "0").replace(/[$,]/g, '')) || 0; // Price is col 4
                if (minP > 0 && p < minP) return false;
                if (maxP > 0 && p > maxP) return false;
                
                var minV = parseVal($('#minVol').val()), maxV = parseVal($('#maxVol').val());
                var v = parseFloat(data[19]) || 0; // Raw Vol is col 18
                if (minV > 0 && v < minV) return false;
                if (maxV > 0 && v > maxV) return false;
                return true;
            }});
            
            $('#minPrice, #maxPrice, #minVol, #maxVol').on('keyup change', function() {{ table.draw(); }});
            window.redraw = function() {{ 
                var mode = $('input[name="btnradio"]:checked').attr('id');
                var headerTxt = "INDUSTRY/SECTOR";
                if (mode == 'btnradio2') headerTxt = "INDUSTRY"; else if (mode == 'btnradio3') headerTxt = "SECTOR";
                $(table.column(17).header()).text(headerTxt);
                table.draw(); 
            }};
            var d=new Date($("#time").data("utc")); $("#time").text("Last Updated: " + d.toLocaleString());
        }});
        </script></body></html>"""
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        filename = f"scan_{timestamp}.html"
        filepath = os.path.join(PUBLIC_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)
        index_path = os.path.join(PUBLIC_DIR, "index.html")
        shutil.copy(filepath, index_path)
        print(f"{C_GREEN}[+] Dashboard generated at: {filepath}{C_RESET}")
        return filename
    except Exception as e:
        print(f"\n{C_RED}[!] Export Failed: {e}{C_RESET}")
        return None

def send_discord_link(filename):
    print(f"\n{C_YELLOW}--- Sending Link to Discord... ---{C_RESET}")
    
    # 1. Check if the environment variable exists
    DISCORD_URL = os.environ.get('DISCORD_WEBHOOK')
    REPO_NAME = os.environ.get('GITHUB_REPOSITORY') 

    if not DISCORD_URL:
        print(f"{C_RED}[!] Error: DISCORD_WEBHOOK is missing. Check GitHub Secrets.{C_RESET}")
        return
    if not REPO_NAME:
        print(f"{C_RED}[!] Error: GITHUB_REPOSITORY is missing.{C_RESET}")
        return

    try:
        # 2. Construct the URL
        user, repo = REPO_NAME.split('/')
        website_url = f"https://{user}.github.io/{repo}/{filename}"
        
        msg = (f"🚀 **Market Scan Complete**\n"
               f"The dashboard has been updated.\n\n"
               f"🔗 **[Click Here to Open Dashboard]({website_url})**\n"
               f"*(Note: It may take ~30s for the link to go live)*")

        # 3. Send and CHECK the status code
        response = requests.post(DISCORD_URL, json={"content": msg})
        
        if response.status_code == 204:
            print(f"{C_GREEN}[+] Discord Link Sent Successfully!{C_RESET}")
        else:
            # This logs the specific error from Discord (e.g., Rate Limit, 404)
            print(f"{C_RED}[!] Discord Failed: {response.status_code} - {response.text}{C_RESET}")

    except Exception as e:
        print(f"{C_RED}[!] Exception sending Discord link: {e}{C_RESET}")

if __name__ == "__main__":
    # Handle the --auto flag or standard run
    if "--auto" in sys.argv:
        print("Starting Auto Scan...")
    
    # 1. Fetch Data
    raw = get_all_trending_stocks()
    if not raw:
        print(f"{C_RED}[!] No data returned from ApeWisdom API. Exiting without Discord post.{C_RESET}")
        sys.exit(0) # Exit cleanly, but log that we found nothing

    # 2. Process Data
    df = filter_and_process(raw)
    if df.empty:
        print(f"{C_RED}[!] Data fetched, but all tickers were filtered out. Exiting.{C_RESET}")
        sys.exit(0)

    # 3. Generate HTML
    fname = export_interactive_html(df)
    
    # 4. Send to Discord
    if fname:
        send_discord_link(fname)
    else:
        print(f"{C_RED}[!] HTML generation failed. Skipping Discord.{C_RESET}")
    
    print("Done.")

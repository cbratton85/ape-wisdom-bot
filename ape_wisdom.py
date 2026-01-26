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
#                  CONFIGURATION
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
AVG_VOLUME_DAYS = 30     # Using 30-Day Average
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
            self.data[ticker][today] = {
                "rank_plus": int(row.get('Rank+', 0)),
                "price": float(row.get('Price', 0)),
                "mnt_perc": float(row.get('Mnt%', 0))
            }
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=RETENTION_DAYS)
        for ticker in list(self.data.keys()):
            self.data[ticker] = {d: v for d, v in self.data[ticker].items() if datetime.datetime.strptime(d, "%Y-%m-%d") > cutoff}
            if not self.data[ticker]: del self.data[ticker]
        with open(self.filepath, 'w') as f: json.dump(self.data, f, indent=4)

    def get_metrics(self, ticker, current_price, current_mnt):
        # FIX: Ensure this block is indented 8 spaces (2 levels) from the left margin
        if ticker not in self.data or len(self.data[ticker]) < 2: 
            return {"vel": 0, "div": False, "streak": 0}

        dates = sorted(self.data[ticker].keys())
        today_data = self.data[ticker][dates[-1]]
        prev_data = self.data[ticker][dates[-2]]

        # VELOCITY calculation
        velocity = int(today_data['rank_plus'] - prev_data['rank_plus'])
        
        # DIVERGENCE (ACCUM) calculation
        mnt_surge = current_mnt > (prev_data['mnt_perc'] + 10)
        price_stable = abs((current_price - prev_data['price']) / (prev_data['price'] or 1)) < 0.02
        divergence = mnt_surge and price_stable
        
        return {"vel": velocity, "div": divergence, "streak": len(dates)}

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
            currency = info.get('currency', 'USD') # Check Currency
            
            if quote_type == 'ETF':
                meta = info.get('category', 'Unknown')
            else:
                s = info.get('sector', 'Unknown')
                i = info.get('industry', 'Unknown')
                meta = f"{s} - {i}" if s != 'Unknown' else 'Unknown'
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
    
    # Normalize tickers
    us_tickers = list(set([TICKER_FIXES.get(s['ticker'], s['ticker'].replace('.', '-')) for s in stocks]))
    
    local_cache = load_cache()
    
    # --- SMART BLACKLIST (THE LOTTERY SYSTEM) ---
    now = datetime.datetime.utcnow()
    valid_tickers = []
    
    print(f"Processing {len(us_tickers)} tickers...")
    
    for t in us_tickers:
        # Check if currently blacklisted
        if t in local_cache and local_cache[t].get('delisted') == True:
            last_checked_str = local_cache[t].get('last_checked', '2000-01-01')
            try:
                last_checked = datetime.datetime.strptime(last_checked_str, "%Y-%m-%d")
                days_since = (now - last_checked).days
                
                # If it's been less than 7 days, normally we skip.
                if days_since < DELISTED_RETRY_DAYS:
                    # THE LOTTERY: 10% chance to retry anyway
                    # This allows 'dead' stocks to naturally come back to life over time
                    if random.random() > 0.10: 
                        continue # You lost the lottery. Stay blacklisted.
                    else:
                        # You won! Forget the blacklist so we fetch fresh MetaData
                        print(f"🎰 Lottery Win: Retrying blacklisted ticker {t}")
                        del local_cache[t] 
            except: pass
            
        valid_tickers.append(t)
    
    # Update our list to only the valid ones (plus the lottery winners)
    # The 'missing' check below will now see the Lottery Winners as "missing" 
    # because we deleted them from local_cache above.
    
    # Metadata healing
    missing = [t for t in valid_tickers if t not in local_cache]
    if missing:
        print(f"Fetching metadata for {len(missing)} items...")
        for i, t in enumerate(missing):
            try:
                res = fetch_meta_data_robust(t)
                if res: local_cache[res['ticker']] = res
            except: pass
        save_cache(local_cache)

    # Market Data Loading
    market_data = None
    use_cache = False
    if os.path.exists(MARKET_DATA_CACHE_FILE):
        if (time.time() - os.path.getmtime(MARKET_DATA_CACHE_FILE)) < CACHE_EXPIRY_SECONDS: use_cache = True

    if use_cache: market_data = pd.read_pickle(MARKET_DATA_CACHE_FILE)
    else:
        print(f"Downloading data for {len(valid_tickers)} tickers...")
        # Download everything in the valid list
        market_data = yf.download(valid_tickers, period="40d", interval="1d", group_by='ticker', progress=False, threads=True)
        if not market_data.empty: market_data.to_pickle(MARKET_DATA_CACHE_FILE)

    if len(valid_tickers) == 1 and not market_data.empty:
        market_data.columns = pd.MultiIndex.from_product([valid_tickers, market_data.columns])

    final_list = []
    for stock in stocks:
        t = TICKER_FIXES.get(stock['ticker'], stock['ticker'].replace('.', '-'))
        try:
            # --- MARKET DATA CHECKS ---
            if isinstance(market_data.columns, pd.MultiIndex):
                if t in market_data.columns.levels[0]: hist = market_data[t].dropna()
                else: 
                    # Download failed? Mark as delisted for next time
                    local_cache[t] = {'delisted': True, 'last_checked': datetime.datetime.utcnow().strftime("%Y-%m-%d")}
                    save_cache(local_cache)
                    continue
            else: hist = market_data.dropna()

            if hist.empty: 
                local_cache[t] = {'delisted': True, 'last_checked': datetime.datetime.utcnow().strftime("%Y-%m-%d")}
                save_cache(local_cache)
                continue

            curr_p = hist['Close'].iloc[-1]
            avg_v = hist['Volume'].tail(AVG_VOLUME_DAYS).mean()
            
            # Note: Filters are disabled (0.0) based on your previous request
            if curr_p < MIN_PRICE: continue
            if avg_v < MIN_AVG_VOLUME: continue

            info = local_cache.get(t, {})
            if info.get('currency', 'USD') != 'USD': continue

            name = str(info.get('name', t)).replace('"', '').strip()[:NAME_MAX_WIDTH]
            
            # --- MATH LOGIC ---
            cur_m = int(stock.get('mentions', 0))
            old_m = int(stock.get('mentions_24h_ago', 0))
            m_perc = int(((cur_m - old_m) / (old_m if old_m > 0 else 1) * 100))
            
            s_perc = int((hist['Volume'].iloc[-1] / avg_v * 100)) if avg_v > 0 else 0
            mcap = info.get('mcap', 10**9) or 10**9
            squeeze_score = (cur_m * s_perc) / max(math.log(mcap, 10), 1)

            rank_now = int(stock.get('rank', 0))
            rank_old = int(stock.get('rank_24h_ago', 0))
            
            if rank_old == 0: rank_plus = 0
            else: rank_plus = rank_old - rank_now

            final_list.append({
                "Rank": rank_now, 
                "Name": name, "Sym": t, 
                "Rank+": rank_plus,
                "Price": float(curr_p), 
                "AvgVol": int(avg_v),
                "Surge": s_perc, "Mnt%": m_perc, "Type": info.get('type', 'EQUITY'),
                "Upvotes": int(stock.get('upvotes', 0)), "Meta": info.get('meta', '-'), "Squeeze": squeeze_score
            })
        except Exception as e: 
            continue
    
    df = pd.DataFrame(final_list)
    if not df.empty:
        # --- NEW CONFIGURATION: THE BIG 4 ---
        # 1. Rank+  (Momentum/Speed)
        # 2. Surge  (Volume Strength)
        # 3. Mnt%   (Viral Growth)
        # 4. Upvotes(Raw Popularity)
        
        # We dropped 'Squeeze' to reduce noise/double-counting.
        cols = ['Rank+', 'Surge', 'Mnt%', 'Upvotes']
        
        # Weights: Give slightly more power to actual Rank movement and Raw Upvotes
        weights = {'Rank+': 1.1, 'Surge': 1.0, 'Mnt%': 0.8, 'Upvotes': 1.1}

        for col in cols:
            # 1. PRE-PROCESS: Clip negatives to 0. 
            clean_series = df[col].clip(lower=0).astype(float)
            
            # 2. LOG TRANSFORM: Compress outliers (The "Tesla Fix")
            # np.log1p(x) = log(1 + x)
            log_data = np.log1p(clean_series)
            
            # 3. STATS: Calculate Mean/Std on the COMPRESSED data
            mean = log_data.mean()
            std = log_data.std(ddof=0)
            
            # 4. Z-SCORE CALCULATION
            if std == 0:
                df[f'z_{col}'] = 0
            else:
                df[f'z_{col}'] = (log_data - mean) / std

        # 5. MASTER SCORE SUMMATION
        df['Master_Score'] = 0
        for col in cols:
            df['Master_Score'] += df[f'z_{col}'].clip(lower=0) * weights[col]

    tracker = HistoryTracker(HISTORY_FILE)
    vel, div, strk = [], [], []
    for _, row in df.iterrows():
        m = tracker.get_metrics(row['Sym'], row['Price'], row['Mnt%'])
        vel.append(m['vel']); div.append(m['div']); strk.append(m['streak'])
    df['Velocity'] = vel; df['Divergence'] = div; df['Streak'] = strk
    tracker.save(df)
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
        # Convert to object immediately so we can overwrite numbers with HTML strings
        export_df = df.copy().astype(object)
        
        if not os.path.exists(PUBLIC_DIR):
            os.makedirs(PUBLIC_DIR)

        def color_span(text, color_hex): return f'<span style="color: {color_hex}; font-weight: bold;">{text}</span>'
        def format_vol(v):
            if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
            if v >= 1_000: return f"{v/1_000:.0f}K"
            return str(v)

        C_GREEN, C_YELLOW, C_RED, C_CYAN, C_MAGENTA, C_WHITE = "#00ff00", "#ffff00", "#ff4444", "#00ffff", "#ff00ff", "#ffffff"
        export_df['Type_Tag'] = 'STOCK'
        tracker = HistoryTracker(HISTORY_FILE)
        
        # Initialize Vel as String to avoid FutureWarning
        export_df['Vel'] = ""; export_df['Sig'] = ""

        # Create Readable Volume Column
        export_df['Vol_Display'] = export_df['AvgVol'].apply(format_vol)

        for index, row in export_df.iterrows():
            # --- 1. VELOCITY (Momentum) ---
            m = tracker.get_metrics(row['Sym'], row['Price'], row['Mnt%'])
            v_val = m['vel']
            
            if v_val == 0:
                export_df.at[index, 'Vel'] = "" # Hide if 0
            else:
                v_color = C_GREEN if v_val > 0 else C_RED
                export_df.at[index, 'Vel'] = color_span(f"{v_val:+d}", v_color)
            
            # --- 2. SIGNAL (Accum/Trend) ---
            sig_text = ""
            if m['div']: sig_text = "ACCUM" 
            elif m['streak'] > 5: sig_text = "TREND"
            sig_color = C_CYAN if "ACCUM" in sig_text else C_YELLOW
            export_df.at[index, 'Sig'] = color_span(sig_text, sig_color)
            
            # --- 3. NAME (Heat Status) ---
            # Red > 4.0, Yellow > 2.0 (Based on 4 metrics)
            nm_clr = C_RED if row['Master_Score'] > 4.0 else (C_YELLOW if row['Master_Score'] > 2.0 else C_WHITE)
            export_df.at[index, 'Name'] = color_span(row['Name'], nm_clr)
            
            # --- 4. RANK+ (Directional Move) ---
            r_val = row['Rank+']
            
            if r_val == 0:
                export_df.at[index, 'Rank+'] = "" # Hide if 0
            else:
                r_color = C_GREEN if r_val > 0 else C_RED
                r_arrow = "▲" if r_val > 0 else "▼"
                export_df.at[index, 'Rank+'] = color_span(f"{r_val} {r_arrow}", r_color)

            # --- 5. METRICS (Surge & Mnt%) ---
            for col, z_col in [('Surge', 'z_Surge'), ('Mnt%', 'z_Mnt%')]:
                val = f"{row[col]:.0f}%"
                clr = C_YELLOW if row[z_col] >= 2.0 else (C_GREEN if row[z_col] >= 1.0 else C_WHITE)
                export_df.at[index, col] = color_span(val, clr)
                
            # Squeeze is now raw (White)
            export_df.at[index, 'Squeeze'] = color_span(int(row['Squeeze']), C_WHITE)
            
            export_df.at[index, 'Upvotes'] = color_span(row['Upvotes'], C_GREEN if row['z_Upvotes']>1.5 else C_WHITE)
            
            is_fund = row['Type'] == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
            export_df.at[index, 'Meta'] = color_span(row['Meta'], C_MAGENTA if is_fund else C_WHITE)
            export_df.at[index, 'Type_Tag'] = 'ETF' if is_fund else 'STOCK'
            
            t = row['Sym']
            export_df.at[index, 'Sym'] = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color: #4da6ff; text-decoration: none;">{t}</a>'
            export_df.at[index, 'Price'] = f"${row['Price']:.2f}"
            export_df.at[index, 'Vol_Display'] = color_span(export_df.at[index, 'Vol_Display'], "#ccc")

        export_df.rename(columns={'Meta': 'Industry/Sector', 'Vol_Display': 'Avg Vol'}, inplace=True)

        cols = ['Rank', 'Rank+', 'Name', 'Sym', 'Sig', 'Vel', 'Price', 'Avg Vol', 'Surge', 'Mnt%', 'Upvotes', 'Squeeze', 'Industry/Sector', 'Type_Tag', 'AvgVol']
        final_df = export_df[cols]
        table_html = final_df.to_html(classes='table table-dark table-hover', index=False, escape=False)
        utc_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # INSERTED BASE64 IMAGE DATA BELOW
        logo_data = "data:image/png;base64,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"

        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Ape Wisdom Analysis</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <style>
            body {{
                /* Exact match for the main background */
                background-color: #101010; 
                color: #e0e0e0;
                font-family: 'Consolas', 'Monaco', monospace;
                padding: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            .table-dark {{ --bs-table-bg: #18181b; color: #ccc; }}
            th{{color:#00ff00;border-bottom:2px solid #444; font-size: 14px;}}
            
            /* --- TABLE LAYOUT --- */
            table.dataTable {{ width: auto !important; margin: 0 auto; }}

            /* --- COLUMN WIDTHS --- */
            th:nth-child(1), td:nth-child(1), th:nth-child(2), td:nth-child(2),
            th:nth-child(4), td:nth-child(4), th:nth-child(5), td:nth-child(5),
            th:nth-child(6), td:nth-child(6) {{
                width: 1%; white-space: nowrap; text-align: center; padding: 0 8px;
            }}
            th:nth-child(7), td:nth-child(7), th:nth-child(8), td:nth-child(8),
            th:nth-child(9), td:nth-child(9), th:nth-child(10), td:nth-child(10),
            th:nth-child(11), td:nth-child(11), th:nth-child(12), td:nth-child(12) {{
                width: 1%; white-space: nowrap; text-align: right; padding: 0 10px;
            }}
            th:nth-child(3), td:nth-child(3) {{
                max-width: 230px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 15px;
            }}
            th:nth-child(13), td:nth-child(13) {{
                max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-left: 15px;
            }}
            
            td{{vertical-align:middle; white-space: nowrap; border-bottom:1px solid #333;}} 
            a{{color:#4da6ff; text-decoration:none;}} a:hover{{text-decoration:underline;}}

            /* --- COLOR TOGGLE OVERRIDE --- */
            /* When this class is active, force all spans to be light gray */
            table.no-colors span {{
                color: #ddd !important;
                font-weight: normal !important;
            }}
            /* Keep links blue though */
            table.no-colors a {{ color: #4da6ff !important; }}
            
            .legend-container {{ background-color: #222; border: 1px solid #444; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }}
            .legend-header {{ background: #2a2a2a; padding: 10px 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: bold; color: #fff; }}
            .legend-box {{ padding: 15px; display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; font-size: 0.85rem; border-top: 1px solid #444; }}
            .legend-section h5 {{ color: #00ff00; font-size: 1rem; border-bottom: 1px solid #444; padding-bottom: 5px; }}
            .legend-item {{ margin-bottom: 6px; }}
            .legend-key {{ font-weight: bold; display: inline-block; width: 100px; }}
            
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
        </style>
        </head>
        <body>
        <div class="container-fluid" style="width: fit-content; margin: 0 auto;">
            
            <div class="header-flex">
                <a href="https://apewisdom.io" target="_blank" style="text-decoration: none;">
                <img src="{logo_data}" alt="Ape Wisdom" style="height: 50px; vertical-align: middle; object-fit: contain;">
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
                    <label>Vol:</label>
                    <input type="text" id="minVol" class="form-control form-control-sm" placeholder="500k" style="width: 50px;">
                    <span style="color:#666">-</span>
                    <input type="text" id="maxVol" class="form-control form-control-sm" placeholder="10m" style="width: 50px;">
                </div>

                <div class="filter-group">
                    <div class="btn-group" role="group">
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio1" checked onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio1" style="font-size: 0.75rem; padding: 2px 8px;">All</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio2" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio2" style="font-size: 0.75rem; padding: 2px 8px;">Stocks</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio3" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio3" style="font-size: 0.75rem; padding: 2px 8px;">ETFs</label>
                    </div>
                </div>

                <button class="btn btn-sm btn-reset" onclick="exportTickers()" title="Download Ticker List" style="margin-left: 10px;">📋 Download TXT</button>
                <span id="stockCounter">Loading...</span>
            </div>

            <div class="legend-container">
                <div class="legend-header" onclick="toggleLegend()">
                    <span>ℹ️ STRATEGY GUIDE & LEGEND (Click to Toggle)</span>
                    <span id="legendArrow">▼</span>
                </div>
                <div class="legend-box" id="legendContent" style="display:none;">
                    
                    <div class="legend-section">
                        <h5>🔥 Heat Status (Name Color)</h5>
                        <div class="legend-item"><span class="legend-key" style="color:#ff4444">RED NAME</span> <b>Extreme (>4.0):</b> Massive outlier across all metrics.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ffff00">YEL NAME</span> <b>Elevated (>2.0):</b> Activity is well above normal.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ffffff">WHT NAME</span> <b>Normal:</b> Standard activity levels.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ff00ff">MAGENTA</span> Exchange Traded Fund (ETF).</div>
                    </div>

                    <div class="legend-section">
                        <h5>🚀 Significance Signals</h5>
                        <div class="legend-item"><span class="legend-key" style="color:#00ffff">ACCUM</span> <b>Accumulation:</b> Mentions Rising (>10%) + Price Flat.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ffff00">TREND</span> <b>Trending:</b> In Top 100 list for 5+ consecutive days.</div>
                    </div>
                    
                    <div class="legend-section">
                        <h5>📊 Metrics Explained</h5>
                        <div class="legend-item"><span class="legend-key">Rank+</span> Positions climbed in the last 24 hours.</div>
                        <div class="legend-item"><span class="legend-key">Vel</span> <b>Velocity:</b> Acceleration of climb vs yesterday.</div>
                        <div class="legend-item"><span class="legend-key">Surge</span> Volume increase vs 30-Day Average.</div>
                        <div class="legend-item"><span class="legend-key">Mnt%</span> Change in Mentions vs 24 hours ago.</div>
                        <div class="legend-item"><span class="legend-key">Upvotes</span> Net Upvotes on Reddit (Green=Pos, Red=Neg).</div>
                        <div class="legend-item"><span class="legend-key">Squeeze</span> (Mentions × Vol) / MarketCap (Ratio).</div>
                    </div>

                </div>
            </div>

            {table_html}
        </div>
        
        <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
        <script>
        function toggleLegend() {{
            var x = document.getElementById("legendContent");
            var arrow = document.getElementById("legendArrow");
            if (x.style.display === "none") {{ x.style.display = "grid"; arrow.innerText = "▲"; }} 
            else {{ x.style.display = "none"; arrow.innerText = "▼"; }}
        }}

        // UPDATED: Toggle Colors Function
        function toggleColors() {{
            var table = document.querySelector('table');
            var btn = document.getElementById('btnColors');
            
            table.classList.toggle('no-colors');
            
            if (table.classList.contains('no-colors')) {{
                btn.innerHTML = "🎨 Colors: OFF";
                btn.style.opacity = "0.6";
            }} else {{
                btn.innerHTML = "🎨 Colors: ON";
                btn.style.opacity = "1.0";
            }}
        }}

        function parseVal(str) {{
            if (!str) return 0;
            str = str.toString().toLowerCase().replace(/,/g, '');
            let mult = 1;
            if (str.endsWith('k')) mult = 1000;
            else if (str.endsWith('m')) mult = 1000000;
            else if (str.endsWith('b')) mult = 1000000000;
            return parseFloat(str) * mult || 0;
        }}

        function resetFilters() {{
            $('#minPrice, #maxPrice, #minVol, #maxVol').val(''); 
            $('#btnradio1').prop('checked', true); 
            redraw(); 
        }}

        function exportTickers() {{
            var table = $('.table').DataTable();
            var data = table.rows({{ search: 'applied', order: 'current', page: 'current' }}).data();
            var tickers = [];
            data.each(function (value) {{
                var div = document.createElement("div");
                div.innerHTML = value[3];
                var text = div.textContent || div.innerText || "";
                if(text) tickers.push(text.trim());
            }});
            if (tickers.length === 0) {{ alert("No visible tickers!"); return; }}
            var blob = new Blob([tickers.join(" ")], {{ type: "text/plain;charset=utf-8" }});
            var a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "ape_tickers_page.txt";
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
        }}

        $(document).ready(function(){{ 
            var table=$('.table').DataTable({{
                "order":[[0,"asc"]],
                "pageLength": 25,
                "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                "columnDefs": [ 
                    {{ "visible": false, "targets": [13, 14] }}, 
                    {{ "orderData": [14], "targets": [7] }},
                    {{ "targets": [1, 5, 6, 8, 9], "type": "num", "render": function(data, type) {{
                        if (type === 'sort' || type === 'type') {{
                            var clean = data.toString().replace(/<[^>]+>/g, '').replace(/[$,%+,]/g, '');
                            return parseFloat(clean) || 0;
                        }}
                        return data;
                    }} }},
                    {{ "targets": [10], "type": "num", "render": function(data, type, row) {{
                        // 1. STRIP HTML TAGS (<...>) to get the raw number string
                        var clean = data.toString().replace(/<[^>]+>/g, '').replace(/,/g, '');
                        var val = parseFloat(clean) || 0;
                        
                        // 2. SORT LOGIC: Return the clean number
                        if (type === 'sort' || type === 'type') return val;
                        
                        // 3. DISPLAY LOGIC: Color Red if negative, Green if positive
                        if (val < 0) {{
                            return '<span style="color:#ff4444">' + val + '</span>'; 
                        }} else {{
                            return '<span style="color:#00ff00">' + val + '</span>'; 
                        }}
                    }} }}
                ],
                "drawCallback": function() {{
                    var api = this.api();
                    $("#stockCounter").text("Showing " + api.rows({{filter:'applied'}}).count() + " / " + api.rows().count() + " Tickers");
                }}
            }});
            
            $.fn.dataTable.ext.search.push(function(settings, data) {{
                var typeTag = data[13] || ""; 
                var viewMode = $('input[name="btnradio"]:checked').attr('id');
                if (viewMode == 'btnradio2' && typeTag == 'ETF') return false;
                if (viewMode == 'btnradio3' && typeTag == 'STOCK') return false;

                var minP = parseVal($('#minPrice').val()), maxP = parseVal($('#maxPrice').val());
                var p = parseFloat((data[6] || "0").replace(/[$,]/g, '')) || 0;
                if (minP > 0 && p < minP) return false;
                if (maxP > 0 && p > maxP) return false;

                var minV = parseVal($('#minVol').val()), maxV = parseVal($('#maxVol').val());
                var v = parseFloat(data[14]) || 0; 
                if (minV > 0 && v < minV) return false;
                if (maxV > 0 && v > maxV) return false;

                return true;
            }});

            $('#minPrice, #maxPrice, #minVol, #maxVol').on('keyup change', function() {{ table.draw(); }});
            
            window.redraw = function() {{ 
                var mode = $('input[name="btnradio"]:checked').attr('id');
                var headerTxt = "Industry/Sector";
                if (mode == 'btnradio2') headerTxt = "Industry";
                else if (mode == 'btnradio3') headerTxt = "Sector";
                $(table.column(12).header()).text(headerTxt);
                table.draw(); 
            }};
            
            var d=new Date($("#time").data("utc"));
            $("#time").text("Last Updated: " + d.toLocaleString());
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
    DISCORD_URL = os.environ.get('DISCORD_WEBHOOK')
    REPO_NAME = os.environ.get('GITHUB_REPOSITORY') 
    
    if not DISCORD_URL or not REPO_NAME: 
        print("Missing Discord URL or Repo Name")
        return

    try:
        user, repo = REPO_NAME.split('/')
        website_url = f"https://{user}.github.io/{repo}/{filename}"
        
        msg = (f"🚀 **Market Scan Complete**\n"
               f"The dashboard has been updated.\n\n"
               f"🔗 **[Click Here to Open Dashboard]({website_url})**\n"
               f"*(Note: It may take ~30s for the link to go live)*")

        requests.post(DISCORD_URL, json={"content": msg})
        print(f"{C_GREEN}[+] Discord Link Sent!{C_RESET}")
    except Exception as e:
        print(f"Error sending Discord link: {e}")

if __name__ == "__main__":
    if "--auto" in sys.argv:
        print("Starting Auto Scan...")
        raw = get_all_trending_stocks()
        if raw:
            df = filter_and_process(raw)
            fname = export_interactive_html(df)
            if fname and 'send_discord_link' in globals():
                send_discord_link(fname)
        sys.exit()
    
    raw = get_all_trending_stocks()
    df = filter_and_process(raw)
    export_interactive_html(df)
    print("Done.")

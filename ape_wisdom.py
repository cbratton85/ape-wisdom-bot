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
    now = datetime.datetime.utcnow()
    valid_tickers = []
    
    print(f"Processing {len(us_tickers)} tickers...")
    
    # 2. Clean Cache (Lottery Check: Only retry ONE delisted ticker per run)
    # --------------------------
    blacklist = [t for t in us_tickers if local_cache.get(t, {}).get('delisted')]
    
    if blacklist:
        t = random.choice(blacklist) # Pick the "lottery" candidate
        last_checked = datetime.datetime.strptime(local_cache[t].get('last_checked', '2000-01-01'), "%Y-%m-%d")
        
        if (now - last_checked).days >= DELISTED_RETRY_DAYS:
            print(f"{C_YELLOW}[!] Lottery Check: Retrying {t}...{C_RESET}")
            del local_cache[t] # Remove to force a fresh metadata fetch

    # Re-build valid_tickers excluding those still marked as delisted
    valid_tickers = [t for t in us_tickers if not local_cache.get(t, {}).get('delisted')]
    
    # 3. Fetch Missing Metadata
    missing = [t for t in valid_tickers if t not in local_cache]
    if missing:
        print(f"Fetching metadata for {len(missing)} items...")
        for t in missing:
            try:
                res = fetch_meta_data_robust(t)
                if res: local_cache[res['ticker']] = res
            except: pass
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
            
            # --- Squeeze & MCap Logic ---
            try:
                mcap = float(info.get('mcap', 0) or 0)
            except: 
                mcap = 0
            
            log_mcap = math.log(mcap if mcap > 0 else 10**9, 10)
            squeeze_score = (cur_m * s_perc) / max(log_mcap, 1)

            rank_now = int(stock.get('rank', 0))
            rank_old = int(stock.get('rank_24h_ago', 0))
            rank_plus = (rank_old - rank_now) if rank_old != 0 else 0

            final_list.append({
                "Rank": rank_now, "Name": name, "Sym": t, "Rank+": rank_plus,
                "Price": float(curr_p), "AvgVol": int(avg_v),
                "Surge": s_perc, "Mnt%": m_perc, "Type": info.get('type', 'EQUITY'),
                "Upvotes": int(stock.get('upvotes', 0)), "Meta": info.get('meta', '-'), 
                "Squeeze": squeeze_score, 
                "MCap": mcap
            })
        except Exception as e: 
            continue
    
    # 6. Scoring
    df = pd.DataFrame(final_list)
    if not df.empty:
        cols = ['Rank+', 'Surge', 'Mnt%', 'Upvotes']
        weights = {'Rank+': 1.1, 'Surge': 1.0, 'Mnt%': 0.8, 'Upvotes': 1.1}
        
        for col in cols:
            clean_series = df[col].clip(lower=0).astype(float)
            log_data = np.log1p(clean_series)
            mean = log_data.mean(); std = log_data.std(ddof=0)
            df[f'z_{col}'] = 0 if std == 0 else (log_data - mean) / std

        df['Master_Score'] = 0
        for col in cols: df['Master_Score'] += df[f'z_{col}'].clip(lower=0) * weights[col]

        # --- Visual Only Z-Score for Squeeze ---
        if 'Squeeze' in df.columns:
            sq_series = df['Squeeze'].clip(lower=0).astype(float)
            log_sq = np.log1p(sq_series)
            mean_sq = log_sq.mean(); std_sq = log_sq.std(ddof=0)
            df['z_Squeeze'] = 0 if std_sq == 0 else (log_sq - mean_sq) / std_sq
        else:
            df['z_Squeeze'] = 0

    # 7. History
    tracker = HistoryTracker(HISTORY_FILE)
    vel, div, strk = [], [], []
    for _, row in df.iterrows():
        m = tracker.get_metrics(row['Sym'], row['Price'], row['Mnt%'])
        vel.append(m['vel']); div.append(m['div']); strk.append(m['streak'])
    
    df['Velocity'] = vel; df['Divergence'] = div; df['Streak'] = strk
    tracker.save(df)
    save_cache(local_cache)
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
        tracker = HistoryTracker(HISTORY_FILE)
        export_df['Vel'] = ""; export_df['Sig'] = ""
        export_df['Vol_Display'] = export_df['AvgVol'].apply(format_vol)

        for index, row in export_df.iterrows():
            # Metrics
            m = tracker.get_metrics(row['Sym'], row['Price'], row['Mnt%'])
            v_val = m['vel']
            if v_val != 0:
                v_color = C_GREEN if v_val > 0 else C_RED
                export_df.at[index, 'Vel'] = color_span(f"{v_val:+d}", v_color)
            
            # Signals
            sig_text = ""; sig_color = C_WHITE
            if m['div']: sig_text = "ACCUM"; sig_color = C_CYAN
            elif m['streak'] > 5: sig_text = "TREND"; sig_color = C_YELLOW
            export_df.at[index, 'Sig'] = color_span(sig_text, sig_color)
            
            # Heatmap Name
            nm_clr = C_RED if row['Master_Score'] > 4.0 else (C_YELLOW if row['Master_Score'] > 2.0 else C_WHITE)
            export_df.at[index, 'Name'] = color_span(row['Name'], nm_clr)
            
            # Rank+
            r_val = row['Rank+']
            if r_val != 0:
                r_color = C_GREEN if r_val > 0 else C_RED
                r_arrow = "▲" if r_val > 0 else "▼"
                export_df.at[index, 'Rank+'] = color_span(f"{r_val} {r_arrow}", r_color)

            # Columns
            for col, z_col in [('Surge', 'z_Surge'), ('Mnt%', 'z_Mnt%')]:
                val = f"{row[col]:.0f}%"
                clr = C_YELLOW if row[z_col] >= 2.0 else (C_GREEN if row[z_col] >= 1.0 else C_WHITE)
                export_df.at[index, col] = color_span(val, clr)
            
            # --- NEW: Squeeze with Cyan Logic ---
            sq_z = row.get('z_Squeeze', 0)
            sq_color = C_CYAN if sq_z > 1.5 else C_WHITE
            export_df.at[index, 'Squeeze'] = color_span(int(row['Squeeze']), sq_color)
            
            export_df.at[index, 'Upvotes'] = color_span(row['Upvotes'], C_GREEN if row['z_Upvotes']>1.5 else C_WHITE)
            
            # ETF Badge logic
            is_fund = row['Type'] == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
            if is_fund:
                badge = '<span style="background-color:#ff00ff; color:black; padding:2px 5px; border-radius:4px; font-size:11px; font-weight:bold; margin-right:6px; vertical-align:middle;">ETF</span>'
                meta_text = color_span(row['Meta'], C_MAGENTA)
            else:
                badge = ""
                meta_text = color_span(row['Meta'], C_WHITE)

            export_df.at[index, 'Meta'] = f"{badge}{meta_text}"
            export_df.at[index, 'Type_Tag'] = 'ETF' if is_fund else 'STOCK'
            
            t = row['Sym']
            export_df.at[index, 'Sym'] = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color: #4da6ff; text-decoration: none;">{t}</a>'
            export_df.at[index, 'Price'] = f"${row['Price']:.2f}"
            export_df.at[index, 'Vol_Display'] = color_span(export_df.at[index, 'Vol_Display'], "#ccc")

        export_df.rename(columns={'Meta': 'Industry/Sector', 'Vol_Display': 'Avg Vol'}, inplace=True)

        cols = ['Rank', 'Rank+', 'Name', 'Sym', 'Sig', 'Vel', 'Price', 'Avg Vol', 'Surge', 'Mnt%', 'Upvotes', 'Squeeze', 'Industry/Sector', 'Type_Tag', 'AvgVol', 'MCap']
        final_df = export_df[cols]
        table_html = final_df.to_html(classes='table table-dark table-hover', index=False, escape=False)
        utc_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Ape Wisdom Analysis</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <style>
            body{{
                background-color:#101010; color:#e0e0e0; font-family:'Consolas','Monaco',monospace; padding:20px;
                max-width: 1400px; margin: 0 auto;
            }}
            .table-dark{{--bs-table-bg:#18181b;color:#ccc}}
            
            th{{
                color:#00ff00; border-bottom:2px solid #444; font-size: 14px;
                vertical-align: middle !important; padding-top: 12px !important; padding-bottom: 12px !important; line-height: 1.4 !important;
            }}
            
            table.dataTable {{ width: auto !important; margin: 0 auto; }}
            
            /* Columns */
            th:nth-child(1), td:nth-child(1), th:nth-child(2), td:nth-child(2),
            th:nth-child(4), td:nth-child(4), th:nth-child(5), td:nth-child(5),
            th:nth-child(6), td:nth-child(6) {{ width: 1%; white-space: nowrap; text-align: center; padding: 0 8px; }}

            th:nth-child(7), td:nth-child(7), th:nth-child(8), td:nth-child(8),
            th:nth-child(9), td:nth-child(9), th:nth-child(10), td:nth-child(10),
            th:nth-child(11), td:nth-child(11), th:nth-child(12), td:nth-child(12) {{ width: 1%; white-space: nowrap; text-align: right; padding: 0 10px; }}

            th:nth-child(3), td:nth-child(3) {{ max-width: 230px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 15px; }}
            
            th:nth-child(13), td:nth-child(13) {{
                max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-left: 15px;
                border-right: 1px solid #333; 
            }}
            
            td{{vertical-align:middle; white-space: nowrap; border-bottom:1px solid #333;}} 
            a{{color:#4da6ff; text-decoration:none;}} a:hover{{text-decoration:underline;}}
            table.no-colors span {{ color: #ddd !important; font-weight: normal !important; }}
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
                <img src="https://apewisdom.io/apewisdom-logo.svg" alt="Ape Wisdom" style="height: 50px; vertical-align: middle; margin-right: 15px;">
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
                        <label class="btn btn-outline-light btn-sm" for="mcapAll">All</label>
    
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMega" onclick="toggleMcap('mega')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMega">Mega</label>

                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapLarge" onclick="toggleMcap('large')">
                        <label class="btn btn-outline-light btn-sm" for="mcapLarge">Lrg</label>

                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMid" onclick="toggleMcap('mid')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMid">Mid</label>
    
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapSmall" onclick="toggleMcap('small')">
                        <label class="btn btn-outline-light btn-sm" for="mcapSmall">Sml</label>
    
                        <input type="checkbox" class="btn-check" name="mcapFilter" id="mcapMicro" onclick="toggleMcap('micro')">
                        <label class="btn btn-outline-light btn-sm" for="mcapMicro">Mic</label>
                    </div>
                </div>

                <button class="btn btn-sm btn-reset" onclick="exportTickers()" title="Download Ticker List" style="margin-left: 10px;">TXT File</button>
                <span id="stockCounter">Loading...</span>
            </div>

            <div class="legend-container">
                <div class="legend-header" onclick="toggleLegend()">
                    <span>ℹ️ STRATEGY GUIDE & LEGEND (Click to Toggle)</span>
                    <span id="legendArrow">▼</span>
                </div>
                <div class="legend-box" id="legendContent" style="display:none;">
                    
                    <div class="legend-section">
                        <h5>🔥 Heat Status</h5>
                        <div class="legend-item"><span class="legend-key" style="color:#ff4444">RED NAME</span> <b>Extreme:</b> Massive outlier.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ffff00">YEL NAME</span> <b>Elevated:</b> Activity above normal.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ffffff">WHT NAME</span> <b>Normal:</b> Standard activity.</div>
                    </div>

                    <div class="legend-section">
                        <h5>🚀 Significance Signals (Sig)</h5>
                        <div class="legend-item"><span class="legend-key" style="color:#00ffff">ACCUM</span> Mentions Rising (>10%) + Price Flat.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ffff00">TREND</span> In Top list for 5+ consecutive days.</div>
                    </div>
                    
                    <div class="legend-section">
                        <h5>📊 Metrics Explained</h5>
                        <div class="legend-item"><span class="legend-key">Rank+</span> Positions climbed in the last 24 hours.</div>
                        <div class="legend-item"><span class="legend-key">Vel</span> <b>Velocity:</b> Change in climb speed vs 24h ago.</div>
                        <div class="legend-item"><span class="legend-key">Surge</span> Volume increase vs 30-Day Average.</div>
                        <div class="legend-item"><span class="legend-key">Mnt%</span> Change in Mentions vs 24h ago.</div>
                        <div class="legend-item"><span class="legend-key">Upvotes</span> Net Upvotes on Reddit.</div>
                        <div class="legend-item"><span class="legend-key">&Sigma; Squeeze</span> (Mentions × Vol) / MarketCap.</div>
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
                // If "All" is clicked, uncheck the others
                $('input[name="mcapFilter"]').not('#mcapAll').prop('checked', false);
            }} else {{
                // If a specific cap is clicked, uncheck "All"
                $('#mcapAll').prop('checked', false);
                // If everything is unchecked, turn "All" back on
                if ($('input[name="mcapFilter"]:checked').length === 0) {{
                    $('#mcapAll').prop('checked', true);
                }}
            }}
            // THIS is what actually triggers the table to re-filter
            table.draw();
        }}
        function toggleLegend() {{
            var x = document.getElementById("legendContent"); var arrow = document.getElementById("legendArrow");
            if (x.style.display === "none") {{ x.style.display = "grid"; arrow.innerText = "▲"; }} else {{ x.style.display = "none"; arrow.innerText = "▼"; }}
        }}
        function toggleColors() {{
            var table = document.querySelector('table'); var btn = document.getElementById('btnColors');
            table.classList.toggle('no-colors');
            if (table.classList.contains('no-colors')) {{ btn.innerHTML = "🎨 Colors: OFF"; btn.style.opacity = "0.6"; }} else {{ btn.innerHTML = "🎨 Colors: ON"; btn.style.opacity = "1.0"; }}
        }}
        function parseVal(str) {{
            if (!str) return 0; str = str.toString().toLowerCase().replace(/,/g, '');
            let mult = 1; if (str.endsWith('k')) mult = 1000; else if (str.endsWith('m')) mult = 1000000; else if (str.endsWith('b')) mult = 1000000000;
            return parseFloat(str) * mult || 0;
        }}
        function resetFilters() {{ $('#minPrice, #maxPrice, #minVol, #maxVol').val(''); $('#btnradio1').prop('checked', true); $('#mcapAll').prop('checked', true); redraw(); }}
        function exportTickers() {{
            table = $('.table').DataTable(); var data = table.rows({{ search: 'applied', order: 'current', page: 'current' }}).data();
            var tickers = []; data.each(function (value) {{ var div = document.createElement("div"); div.innerHTML = value[3]; var text = div.textContent || div.innerText || ""; if(text) tickers.push(text.trim()); }});
            if (tickers.length === 0) {{ alert("No visible tickers!"); return; }}
            var blob = new Blob([tickers.join(" ")], {{ type: "text/plain;charset=utf-8" }}); var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "ape_tickers_page.txt"; document.body.appendChild(a); a.click(); document.body.removeChild(a);
        }}
        $(document).ready(function(){{ 
            var table=$('.table').DataTable({{
                "order":[[0,"asc"]], "pageLength": 25, "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                "columnDefs": [ 
                    {{ "visible": false, "targets": [13, 14, 15] }}, // Hide Industry, RawVol, and NEW MCap (Col 15)
                    {{ "orderData": [14], "targets": [7] }},
                    {{ "targets": [1, 5, 6, 8, 9], "type": "num", "render": function(data, type) {{ if (type === 'sort' || type === 'type') {{ var clean = data.toString().replace(/<[^>]+>/g, '').replace(/[$,%+,]/g, ''); return parseFloat(clean) || 0; }} return data; }} }},
                    {{ "targets": [10], "type": "num", "render": function(data, type) {{ var clean = data.toString().replace(/<[^>]+>/g, '').replace(/,/g, ''); var val = parseFloat(clean) || 0; if (type === 'sort' || type === 'type') return val; if (val < 0) return '<span style="color:#ff4444">' + val + '</span>'; else return '<span style="color:#00ff00">' + val + '</span>'; }} }}
                ],
                "drawCallback": function() {{ var api = this.api(); $("#stockCounter").text("Showing " + api.rows({{filter:'applied'}}).count() + " / " + api.rows().count() + " Tickers"); }}
            }});
            
            $.fn.dataTable.ext.search.push(function(settings, data) {{
                // STOCK TYPE FILTER
                var typeTag = data[13] || ""; 
                var viewMode = $('input[name="btnradio"]:checked').attr('id');
                if (viewMode == 'btnradio2' && typeTag == 'ETF') return false;
                if (viewMode == 'btnradio3' && typeTag == 'STOCK') return false;
                
                // MARKET CAP FILTER (Multi-Select) - Braces Doubled for Python
                if (!$('#mcapAll').is(':checked')) {{
                    var mcap = parseFloat(data[15]) || 0; 
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
                if (mode == 'btnradio2') headerTxt = "Industry"; else if (mode == 'btnradio3') headerTxt = "Sector";
                $(table.column(12).header()).text(headerTxt);
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

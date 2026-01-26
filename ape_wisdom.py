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
MIN_PRICE = 0.50             
MIN_AVG_VOLUME = 5000        
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
        if ticker not in self.data or len(self.data[ticker]) < 2: return {"vel": 0, "div": False, "streak": 0}
        dates = sorted(self.data[ticker].keys())
        prev_data = self.data[ticker][dates[-2]]
        velocity = int(self.data[ticker][dates[-1]]['rank_plus'] - prev_data['rank_plus'])
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
    us_tickers = list(set([TICKER_FIXES.get(s['ticker'], s['ticker'].replace('.', '-')) for s in stocks]))
    
    # Load the existing cache (which we know persists correctly)
    local_cache = load_cache()
    
    # --- INTEGRATED BLACKLIST CHECK ---
    now = datetime.datetime.utcnow()
    valid_tickers = []
    
    for t in us_tickers:
        # Check if we previously flagged this ticker as delisted in the main cache
        if t in local_cache and local_cache[t].get('delisted') == True:
            last_checked_str = local_cache[t].get('last_checked', '2000-01-01')
            try:
                last_checked = datetime.datetime.strptime(last_checked_str, "%Y-%m-%d")
                # If it's been less than 7 days (or your setting), SKIP IT
                if (now - last_checked).days < DELISTED_RETRY_DAYS:
                    continue 
            except: pass
        valid_tickers.append(t)
    
    us_tickers = valid_tickers
    
    # Check for missing metadata for the valid ones
    missing = [t for t in us_tickers if t not in local_cache]
    if missing:
        print(f"Healing {len(missing)} metadata items...")
        for i, t in enumerate(missing):
            try:
                res = fetch_meta_data_robust(t)
                if res: local_cache[res['ticker']] = res
            except: pass
        save_cache(local_cache)

    market_data = None
    use_cache = False
    if os.path.exists(MARKET_DATA_CACHE_FILE):
        if (time.time() - os.path.getmtime(MARKET_DATA_CACHE_FILE)) < CACHE_EXPIRY_SECONDS: use_cache = True

    if use_cache: market_data = pd.read_pickle(MARKET_DATA_CACHE_FILE)
    else:
        # Download only the Valid tickers
        market_data = yf.download(us_tickers, period="40d", interval="1d", group_by='ticker', progress=False, threads=True)
        if not market_data.empty: market_data.to_pickle(MARKET_DATA_CACHE_FILE)

    if len(us_tickers) == 1 and not market_data.empty:
        idx = pd.MultiIndex.from_product([us_tickers, market_data.columns])
        market_data.columns = idx

    final_list = []
    for stock in stocks:
        t = TICKER_FIXES.get(stock['ticker'], stock['ticker'].replace('.', '-'))
        try:
            # --- CHECK IF DATA EXISTS ---
            if isinstance(market_data.columns, pd.MultiIndex):
                if t in market_data.columns.levels[0]: 
                    hist = market_data[t].dropna()
                else: 
                    # FIX: Ticker completely missing from download -> Save to MAIN CACHE
                    local_cache[t] = {'delisted': True, 'last_checked': datetime.datetime.utcnow().strftime("%Y-%m-%d")}
                    save_cache(local_cache)
                    continue
            else: hist = market_data.dropna()

            if hist.empty: 
                # FIX: Ticker has no history -> Save to MAIN CACHE
                local_cache[t] = {'delisted': True, 'last_checked': datetime.datetime.utcnow().strftime("%Y-%m-%d")}
                save_cache(local_cache)
                continue
            # -----------------------------

            curr_p = hist['Close'].iloc[-1]
            avg_v = hist['Volume'].tail(AVG_VOLUME_DAYS).mean()
            
            if curr_p < MIN_PRICE: continue
            if avg_v < MIN_AVG_VOLUME: continue

            info = local_cache.get(t, {})
            if info.get('currency', 'USD') != 'USD': continue

            name = str(info.get('name', t)).replace('"', '').strip()[:NAME_MAX_WIDTH]
            cur_m, old_m = int(stock.get('mentions', 0)), int(stock.get('mentions_24h_ago', 1))
            m_perc = int(((cur_m - (old_m or 1)) / (old_m or 1) * 100))
            s_perc = int((hist['Volume'].iloc[-1] / avg_v * 100)) if avg_v > 0 else 0
            mcap = info.get('mcap', 10**9) or 10**9
            squeeze_score = (cur_m * s_perc) / max(math.log(mcap, 10), 1)

            final_list.append({
                "Rank": int(stock['rank']), # <--- NEW CAPTURE
                "Name": name, "Sym": t, "Rank+": int(stock['rank_24h_ago']) - int(stock['rank']),
                "Price": float(curr_p), 
                "AvgVol": int(avg_v),
                "Surge": s_perc, "Mnt%": m_perc, "Type": info.get('type', 'EQUITY'),
                "Upvotes": int(stock.get('upvotes', 0)), "Meta": info.get('meta', '-'), "Squeeze": squeeze_score
            })
        except: continue
    
    df = pd.DataFrame(final_list)
    if not df.empty:
        cols = ['Rank+', 'Surge', 'Mnt%', 'Squeeze', 'Upvotes']
        for col in cols:
            mean, std = df[col].mean(), df[col].std(ddof=0)
            df[f'z_{col}'] = (df[col] - mean) / (std if std > 0 else 1)
        df['Master_Score'] = (df['z_Rank+'].clip(0) + df['z_Surge'].clip(0) + df['z_Mnt%'].clip(0) + df['z_Upvotes'].clip(0) + (df['z_Squeeze'].clip(0) * 0.5))

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
        # Create a copy to manipulate for display
        export_df = df.copy().astype(object)
        
        if not os.path.exists(PUBLIC_DIR):
            os.makedirs(PUBLIC_DIR)

        # --- "NANO BANANA" COLOR PALETTE & HELPERS ---
        C_BG_DARK = "#0d1117"      
        C_PANEL   = "#161b22"      
        C_BORDER  = "#30363d"      
        C_TEXT_MAIN = "#c9d1d9"    
        C_TEXT_MUTED = "#8b949e"   
        
        C_GREEN  = "#3fb950"       
        C_RED    = "#f85149"       
        C_YELLOW = "#d29922"       
        C_BLUE   = "#58a6ff"       
        C_PURPLE = "#bc8cff"       
        C_CYAN   = "#39c5cf"       

        def make_badge(text, bg_color):
            return f'<span style="background-color: {bg_color}20; color: {bg_color}; border: 1px solid {bg_color}40; padding: 2px 6px; border-radius: 4px; font-size: 0.70rem; font-weight: 700; letter-spacing: 0.5px;">{text}</span>'

        def format_vol(v):
            if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
            if v >= 1_000: return f"{v/1_000:.0f}K"
            return str(v)

        tracker = HistoryTracker(HISTORY_FILE)
        
        # --- PRE-PROCESS COLUMNS ---
        export_df['Type_Tag'] = 'STOCK'
        export_df['Vel'] = ""
        export_df['Sig'] = ""
        export_df['Vol_Display'] = export_df['AvgVol'].apply(format_vol)
        
        # FIX: Round Squeeze to whole number
        export_df['Squeeze'] = pd.to_numeric(export_df['Squeeze']).fillna(0).astype(int)

        for index, row in export_df.iterrows():
            # 1. VELOCITY
            m = tracker.get_metrics(row['Sym'], row['Price'], row['Mnt%'])
            v_val = m['vel']
            if v_val > 0:   arrow = f'<span style="color:{C_GREEN}">▲ {v_val}</span>'
            elif v_val < 0: arrow = f'<span style="color:{C_RED}">▼ {abs(v_val)}</span>'
            else:           arrow = f'<span style="color:{C_TEXT_MUTED}">-</span>'
            export_df.at[index, 'Vel'] = arrow
            
            # 2. SIGNALS (BADGES)
            sigs = []
            if m['div']: sigs.append(make_badge("ACCUM", C_CYAN))
            if m['streak'] > 5: sigs.append(make_badge("TREND", C_YELLOW))
            export_df.at[index, 'Sig'] = " ".join(sigs)
            
            # 3. NAME STYLING
            name_style = f"color: {C_TEXT_MAIN}; font-weight: 600;"
            if row['Master_Score'] > 3.0: 
                name_style = f"color: {C_RED}; font-weight: 700;" 
            elif row['Master_Score'] > 1.5:
                name_style = f"color: {C_YELLOW};"
                
            export_df.at[index, 'Name'] = f'<span style="{name_style}" title="{row["Name"]}">{row["Name"]}</span>'
            
            # 4. METRICS FORMATTING
            r_plus = row['Rank+']
            r_clr = C_GREEN if r_plus > 0 else (C_RED if r_plus < 0 else C_TEXT_MUTED)
            export_df.at[index, 'Rank+'] = f'<span style="color:{r_clr}">{r_plus}</span>'
            
            for col, z_col in [('Surge', 'z_Surge'), ('Mnt%', 'z_Mnt%')]:
                val = f"{row[col]:.0f}%"
                style = ""
                if row[z_col] >= 2.0: style = f"color: {C_YELLOW}; font-weight:bold;"
                elif row[z_col] >= 1.0: style = f"color: {C_GREEN};"
                else: style = f"color: {C_TEXT_MUTED};"
                export_df.at[index, col] = f'<span style="{style}">{val}</span>'

            # 5. META / ETF
            is_fund = row['Type'] == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
            meta_txt = row['Meta']
            if is_fund:
                export_df.at[index, 'Type_Tag'] = 'ETF'
                export_df.at[index, 'Meta'] = make_badge("ETF", C_PURPLE) + f" <span style='font-size:0.85em'>{meta_txt}</span>"
            else:
                export_df.at[index, 'Meta'] = f"<span style='color:{C_TEXT_MUTED}; font-size:0.85em' title='{meta_txt}'>{meta_txt}</span>"

            # 6. TICKER LINK
            t = row['Sym']
            export_df.at[index, 'Sym'] = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" class="ticker-link">{t}</a>'
            export_df.at[index, 'Price'] = f"${row['Price']:.2f}"

        export_df.rename(columns={'Meta': 'Industry/Sector', 'Vol_Display': 'Avg Vol'}, inplace=True)
        cols = ['Rank', 'Name', 'Sym', 'Vel', 'Sig', 'Rank+', 'Price', 'Avg Vol', 'Surge', 'Mnt%', 'Upvotes', 'Squeeze', 'Industry/Sector', 'Type_Tag', 'AvgVol']
        final_df = export_df[cols]
        
        table_html = final_df.to_html(classes='table table-dark table-hover', index=False, escape=False, border=0)
        utc_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # LOGO DATA (Keep your string here)
        logo_data = "data:image/png;base64,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"

        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Ape Wisdom | Market Intelligence</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        
        <style>
            :root {{
                --bg-dark: {C_BG_DARK};
                --panel-bg: {C_PANEL};
                --border-color: {C_BORDER};
                --text-main: {C_TEXT_MAIN};
                --text-muted: {C_TEXT_MUTED};
                --accent-blue: {C_BLUE};
                --accent-green: {C_GREEN};
                --accent-red: {C_RED};
            }}
            body {{
                background-color: var(--bg-dark);
                color: var(--text-main);
                font-family: 'Inter', sans-serif;
                font-size: 0.9rem;
                padding-bottom: 50px;
            }}
            
            h1, h2, h3, h4, h5 {{ font-weight: 600; letter-spacing: -0.02em; }}
            .mono {{ font-family: 'JetBrains Mono', monospace; }}
            
            .main-container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            
            .top-bar {{ 
                display: flex; justify-content: space-between; align-items: center; 
                padding: 15px 25px; background: var(--panel-bg); 
                border-bottom: 1px solid var(--border-color);
                border-radius: 8px 8px 0 0;
            }}
            .logo-section {{ display: flex; align-items: center; gap: 15px; }}
            .logo-section img {{ height: 45px; }}
            .logo-section h2 {{ margin: 0; font-size: 1.25rem; color: #fff; }}
            .meta-info {{ text-align: right; color: var(--text-muted); font-size: 0.8rem; }}

            .control-panel {{
                background: var(--panel-bg);
                border: 1px solid var(--border-color);
                border-top: none;
                padding: 15px 25px;
                display: flex; gap: 20px; align-items: center; flex-wrap: wrap;
                margin-bottom: 25px;
                border-radius: 0 0 8px 8px;
            }}
            .filter-group {{ display: flex; flex-direction: column; gap: 4px; }}
            .filter-group label {{ font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
            .form-control-dark {{
                background: #0d1117; border: 1px solid #30363d; color: #fff;
                padding: 6px 10px; font-size: 0.9rem; border-radius: 6px;
                width: 120px;
            }}
            .form-control-dark:focus {{ outline: none; border-color: var(--accent-blue); box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2); }}

            .btn-group-segment {{ background: #0d1117; padding: 3px; border-radius: 6px; border: 1px solid #30363d; display: inline-flex; }}
            .btn-check:checked + .btn-segment {{ background: var(--accent-blue); color: #fff; border-color: transparent; }}
            .btn-segment {{
                color: var(--text-muted); padding: 5px 12px; font-size: 0.85rem; border-radius: 4px; border: none; font-weight: 500; transition: all 0.2s;
            }}
            .btn-segment:hover {{ color: #fff; }}

            .table-container {{ 
                background: var(--panel-bg); border: 1px solid var(--border-color); 
                border-radius: 8px; overflow: hidden; padding: 0;
            }}
            .table-dark {{ --bs-table-bg: var(--panel-bg); color: var(--text-main); margin-bottom: 0; }}
            
            .table thead th {{
                background-color: #0d1117; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;
                border-bottom: 1px solid var(--border-color); padding: 12px 10px; vertical-align: middle;
            }}
            
            /* CSS FIX: Force single line and truncate */
            .table tbody td {{
                vertical-align: middle; border-bottom: 1px solid var(--border-color);
                padding: 10px 12px; font-size: 0.9rem; font-family: 'JetBrains Mono', monospace;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;
            }}
            .table tbody tr:hover {{ background-color: #21262d; }}

            .ticker-link {{ color: var(--accent-blue); font-weight: 700; text-decoration: none; }}
            .ticker-link:hover {{ text-decoration: underline; }}
            
            .btn-reset {{ 
                background: transparent; border: 1px solid var(--accent-red); color: var(--accent-red); 
                font-size: 0.8rem; padding: 5px 12px; border-radius: 4px; transition: 0.2s;
            }}
            .btn-reset:hover {{ background: var(--accent-red); color: #fff; }}
            
            .dataTables_info {{ color: var(--text-muted) !important; font-size: 0.85rem; padding: 15px; }}
            .dataTables_paginate {{ padding: 10px; }}
            .page-link {{ background: var(--bg-dark); border-color: var(--border-color); color: var(--text-main); }}
            .page-item.active .page-link {{ background: var(--accent-blue); border-color: var(--accent-blue); color: #fff; }}
            .page-item.disabled .page-link {{ background: var(--bg-dark); opacity: 0.5; }}

            .legend-toggle {{ cursor: pointer; color: var(--accent-blue); font-size: 0.85rem; font-weight: 500; }}
            .legend-grid {{ 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;
                padding: 20px; background: #0d1117; border-top: 1px solid var(--border-color);
                font-size: 0.8rem; color: var(--text-muted);
            }}
            .legend-key {{ font-weight: 700; color: #fff; margin-right: 5px; }}
            .legend-desc {{ display: block; margin-top: 2px; color: #8b949e; font-size: 0.75rem; }}
        </style>
        </head>
        <body>
        <div class="main-container">
            
            <div class="top-bar">
                <div class="logo-section">
                    <img src="{logo_data}" alt="Ape Wisdom">
                    <div>
                        <h2>Market Intelligence</h2>
                        <span style="color: var(--accent-blue); font-size: 0.8rem; font-weight: 600;">LIVE SCANNER</span>
                    </div>
                </div>
                <div class="meta-info">
                    <div id="time" data-utc="{utc_timestamp}">Loading...</div>
                    <div id="stockCounter">Scanning...</div>
                </div>
            </div>

            <div class="control-panel">
                <div class="filter-group">
                    <label>View Mode</label>
                    <div class="btn-group-segment" role="group">
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio1" autocomplete="off" checked onclick="redraw()">
                        <label class="btn btn-segment" for="btnradio1">All</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio2" autocomplete="off" onclick="redraw()">
                        <label class="btn btn-segment" for="btnradio2">Stocks</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio3" autocomplete="off" onclick="redraw()">
                        <label class="btn btn-segment" for="btnradio3">ETFs</label>
                    </div>
                </div>

                <div class="filter-group">
                    <label>Min Price</label>
                    <input type="number" id="minPrice" class="form-control-dark" placeholder="$0.00" step="0.5">
                </div>
                
                <div class="filter-group">
                    <label>Min Avg Vol</label>
                    <input type="number" id="minVol" class="form-control-dark" placeholder="0" step="10000">
                </div>

                <div style="flex-grow: 1;"></div>
                
                <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 5px;">
                     <button class="btn-reset" onclick="resetFilters()">Reset Filters ⟲</button>
                     <div class="legend-toggle" onclick="toggleLegend()">Show Strategy Guide ▼</div>
                </div>
            </div>
            
            <div id="legendPanel" style="display:none; margin-bottom: 25px; border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color);">
                <div class="legend-grid">
                     <div>
                        <div style="margin-bottom:8px; color: #fff; font-weight:600; border-bottom: 1px solid #333; padding-bottom: 4px;">🔥 Heat / Name Signals</div>
                        <div style="margin-bottom:6px;"><span class="legend-key" style="color:{C_RED}">RED NAME</span> <b>Extreme (>3σ):</b> <span class="legend-desc">Massive outlier. Very high volume/mentions relative to normal.</span></div>
                        <div style="margin-bottom:6px;"><span class="legend-key" style="color:{C_YELLOW}">YEL NAME</span> <b>Elevated (>1.5σ):</b> <span class="legend-desc">Activity is significantly above average.</span></div>
                        <div><span class="legend-key" style="color:#fff">WHT NAME</span> <b>Normal:</b> <span class="legend-desc">Standard activity levels.</span></div>
                     </div>
                     <div>
                        <div style="margin-bottom:8px; color: #fff; font-weight:600; border-bottom: 1px solid #333; padding-bottom: 4px;">🚀 Badge Signals</div>
                        <div style="margin-bottom:6px;"><span class="legend-key" style="color:{C_CYAN}">ACCUM</span> <b>Accumulation:</b> <span class="legend-desc">Mentions RISING (>10%) while Price is FLAT/STABLE. Often precedes a move.</span></div>
                        <div><span class="legend-key" style="color:{C_YELLOW}">TREND</span> <b>Consistent:</b> <span class="legend-desc">Has been in the Top Trending list for 5+ consecutive days.</span></div>
                     </div>
                     <div>
                        <div style="margin-bottom:8px; color: #fff; font-weight:600; border-bottom: 1px solid #333; padding-bottom: 4px;">📊 Metrics Defined</div>
                        <div style="margin-bottom:6px;"><span class="legend-key">Squeeze:</span> <span class="legend-desc">(Mentions × Vol Surge) / Log(MktCap). High score = High viral potential vs size.</span></div>
                        <div style="margin-bottom:6px;"><span class="legend-key">Surge:</span> <span class="legend-desc">Current Volume vs 30-Day Average (e.g., 100% = Normal, 200% = Double).</span></div>
                        <div><span class="legend-key">Vel:</span> <span class="legend-desc">Velocity. How many rank spots climbed since yesterday.</span></div>
                     </div>
                </div>
            </div>

            <div class="table-container">
                {table_html}
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: var(--text-muted); font-size: 0.8rem;">
                Generated by Ape Wisdom Bot | Data is delayed. Not financial advice.
            </div>
        </div>
        
        <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
        <script>
        function toggleLegend() {{
            var x = document.getElementById("legendPanel");
            if (x.style.display === "none") x.style.display = "block";
            else x.style.display = "none";
        }}

        function resetFilters() {{
            $('#minPrice').val(''); $('#minVol').val('');
            $('#btnradio1').prop('checked', true); redraw(); 
        }}

        $(document).ready(function(){{ 
            var table=$('.table').DataTable({{
                "order":[[0,"asc"]],
                "pageLength": 50,
                "lengthMenu": [[25, 50, 100, -1], [25, 50, 100, "All"]],
                "dom": 'rtip', 
                "columnDefs": [ 
                    {{ "visible": false, "targets": [13, 14] }},
                    {{ "orderData": [14], "targets": [7] }}
                ],
                "drawCallback": function(settings) {{
                    var api = this.api();
                    $("#stockCounter").text(api.rows({{filter:'applied'}}).count() + " Tickers Active");
                }}
            }});
            
            $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {{
                var typeTag = data[13] || ""; 
                var viewMode = $('input[name="btnradio"]:checked').attr('id');
                if (viewMode == 'btnradio2' && typeTag == 'ETF') return false;
                if (viewMode == 'btnradio3' && typeTag == 'STOCK') return false;

                var minPrice = parseFloat($('#minPrice').val()) || 0;
                var priceStr = data[6] || "0"; 
                var price = parseFloat(priceStr.replace(/[$,]/g, '')) || 0;
                if (price < minPrice) return false;

                var minVol = parseFloat($('#minVol').val()) || 0;
                var rawVol = parseFloat(data[14]) || 0; 
                if (rawVol < minVol) return false;

                return true;
            }});

            $('#minPrice, #minVol').on('keyup change', function() {{ table.draw(); }});
            
            window.redraw = function() {{ 
                var mode = $('input[name="btnradio"]:checked').attr('id');
                var headerTxt = (mode == 'btnradio3') ? "SECTOR" : "INDUSTRY";
                $(table.column(12).header()).text(headerTxt);
                table.draw(); 
            }};
            
            var d=new Date($("#time").data("utc"));
            $("#time").text(d.toLocaleString());
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

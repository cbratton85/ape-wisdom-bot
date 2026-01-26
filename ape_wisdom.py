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
        # Convert to object immediately so we can overwrite numbers with HTML strings
        export_df = df.copy().astype(object)
        
    if not os.path.exists(PUBLIC_DIR):
        os.makedirs(PUBLIC_DIR)

    # MODERN TEXT STYLE: Now creates "Pill" badges instead of just colored text
    def color_span(text, color_hex): 
        return f'<span style="background:{color_hex}15; color:{color_hex}; border:1px solid {color_hex}44; padding:2px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; white-space:nowrap;">{text}</span>'

    def format_vol(v):
        if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
        if v >= 1_000: return f"{v/1_000:.0f}K"
        return str(v)

    # Palette adjustments for a more premium look
    C_GREEN, C_YELLOW, C_RED, C_CYAN, C_MAGENTA, C_WHITE = "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7", "#f8fafc"
    export_df['Type_Tag'] = 'STOCK'
    tracker = HistoryTracker(HISTORY_FILE)
    
    export_df['Vel'] = ""; export_df['Sig'] = ""
    export_df['Vol_Display'] = export_df['AvgVol'].apply(format_vol)

    for index, row in export_df.iterrows():
        m = tracker.get_metrics(row['Sym'], row['Price'], row['Mnt%'])
        v_val = m['vel']
        v_color = C_GREEN if v_val > 0 else (C_RED if v_val < 0 else C_WHITE)
        v_arrow = "↑" if v_val > 5 else ("↓" if v_val < -5 else "")
        export_df.at[index, 'Vel'] = color_span(f"{v_val} {v_arrow}", v_color)
        
        sig_text = ""
        if m['div']: sig_text = "💎 ACCUM"
        elif m['streak'] > 5: sig_text = "🔥 TREND"
        sig_color = C_CYAN if "ACCUM" in sig_text else C_YELLOW
        export_df.at[index, 'Sig'] = color_span(sig_text, sig_color) if sig_text else ""
        
        nm_clr = C_RED if row['Master_Score'] > 3.0 else (C_YELLOW if row['Master_Score'] > 1.5 else C_WHITE)
        export_df.at[index, 'Name'] = color_span(row['Name'], nm_clr)
        
        for col, z_col in [('Rank+', 'z_Rank+'), ('Surge', 'z_Surge'), ('Mnt%', 'z_Mnt%')]:
            val = f"{row[col]:.0f}%" if '%' in col or 'Surge' in col else row[col]
            clr = C_YELLOW if row[z_col] >= 2.0 else (C_GREEN if row[z_col] >= 1.0 else C_WHITE)
            export_df.at[index, col] = color_span(val, clr)
            
        export_df.at[index, 'Squeeze'] = color_span(int(row['Squeeze']), C_CYAN if row['z_Squeeze']>1.5 else C_WHITE)
        export_df.at[index, 'Upvotes'] = color_span(row['Upvotes'], C_GREEN if row['z_Upvotes']>1.5 else C_WHITE)
        
        is_fund = row['Type'] == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
        export_df.at[index, 'Meta'] = color_span(row['Meta'], C_MAGENTA if is_fund else C_WHITE)
        export_df.at[index, 'Type_Tag'] = 'ETF' if is_fund else 'STOCK'
        
        t = row['Sym']
        export_df.at[index, 'Sym'] = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color: #6366f1; font-weight:bold; text-decoration: none;">{t}</a>'
        export_df.at[index, 'Price'] = f"<b>${row['Price']:.2f}</b>"
        export_df.at[index, 'Vol_Display'] = f'<span style="color:#94a3b8">{export_df.at[index, "Vol_Display"]}</span>'

    export_df.rename(columns={'Meta': 'Industry/Sector', 'Vol_Display': 'Avg Vol'}, inplace=True)
    cols = ['Rank', 'Name', 'Sym', 'Vel', 'Sig', 'Rank+', 'Price', 'Avg Vol', 'Surge', 'Mnt%', 'Upvotes', 'Squeeze', 'Industry/Sector', 'Type_Tag', 'AvgVol']
    final_df = export_df[cols]
    table_html = final_df.to_html(classes='table table-hover', index=False, escape=False)
    utc_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # LOGO DATA (Kept from your original)
    logo_data = "data:image/png;base64,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"

    html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>AlphaScan Market Terminal</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        body {{
            background-color: #0f172a; 
            color: #f1f5f9;
            font-family: 'Inter', system-ui, sans-serif;
            padding: 20px;
        }}
        
        /* Table Modernization */
        .table {{ 
            border-collapse: separate; 
            border-spacing: 0 8px; 
            margin-top: 10px;
        }}

        .table thead th {{
            background: transparent !important;
            color: #94a3b8 !important;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: none !important;
            padding: 15px !important;
        }}

        .table tbody tr {{
            background-color: #1e293b !important;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .table tbody tr:hover {{
            background-color: #334155 !important;
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }}

        .table td {{
            border: none !important;
            padding: 14px 15px !important;
            vertical-align: middle !important;
        }}

        .table tbody tr td:first-child {{ border-radius: 10px 0 0 10px; }}
        .table tbody tr td:last-child {{ border-radius: 0 10px 10px 0; }}

        /* Filter Bar & Legend Styling */
        .filter-bar {{ 
            background: #1e293b; 
            padding: 20px; 
            border: 1px solid #334155;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex; gap: 15px; align-items: center; flex-wrap: wrap;
        }}

        .legend-container {{
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            margin-bottom: 25px;
            overflow: hidden;
        }}

        .legend-header {{ background: #334155; padding: 12px 20px; cursor: pointer; display: flex; justify-content: space-between; font-weight: 600; }}
        .legend-section h5 {{ color: #818cf8; font-size: 1rem; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 10px; }}

        .form-control-sm {{
            background: #0f172a !important;
            border: 1px solid #475569 !important;
            color: #fff !important;
            border-radius: 6px;
        }}

        #stockCounter {{ 
            color: #818cf8; 
            font-weight: 600; 
            margin-left: auto; 
            border: 1px solid #334155; 
            padding: 4px 12px; 
            border-radius: 8px;
            background: #0f172a;
        }}

        .btn-outline-light {{ border-color: #475569; color: #94a3b8; }}
        .btn-check:checked + .btn-outline-light {{ background-color: #6366f1; border-color: #6366f1; }}
    </style>
    </head>
    <body>
    <div class="container-fluid" style="max-width:98%;">
        <div class="header-flex d-flex justify-content-between align-items-center mb-3">
            <img src="{logo_data}" alt="Ape Wisdom" style="height: 60px;">
            <span id="time" data-utc="{utc_timestamp}" style="font-size: 0.9rem; color: #888;">Loading...</span>
        </div>

        <div class="filter-bar">
            <span style="color:#fff; font-weight:bold; margin-right:10px;">⚡ FILTERS:</span>
            <button class="btn btn-sm btn-outline-light" onclick="resetFilters()">🔄 RESET</button>
            <div class="filter-group d-flex align-items-center gap-2">
                <label style="font-size:0.85rem; color:#94a3b8;">Min Price ($):</label>
                <input type="number" id="minPrice" class="form-control form-control-sm" style="width:80px;">
            </div>
            <div class="filter-group d-flex align-items-center gap-2">
                <label style="font-size:0.85rem; color:#94a3b8;">Min Avg Vol:</label>
                <input type="number" id="minVol" class="form-control form-control-sm" style="width:100px;">
            </div>
            <div class="btn-group" role="group">
                <input type="radio" class="btn-check" name="btnradio" id="btnradio1" autocomplete="off" checked onclick="redraw()">
                <label class="btn btn-outline-light btn-sm" for="btnradio1">All</label>
                <input type="radio" class="btn-check" name="btnradio" id="btnradio2" autocomplete="off" onclick="redraw()">
                <label class="btn btn-outline-light btn-sm" for="btnradio2">Stocks</label>
                <input type="radio" class="btn-check" name="btnradio" id="btnradio3" autocomplete="off" onclick="redraw()">
                <label class="btn btn-outline-light btn-sm" for="btnradio3">ETFs</label>
            </div>
            <span id="stockCounter">Loading...</span>
        </div>

        <div class="legend-container">
            <div class="legend-header" onclick="toggleLegend()">
                <span>ℹ️ STRATEGY GUIDE & LEGEND</span>
                <span id="legendArrow">▼</span>
            </div>
            <div class="p-4 row" id="legendContent" style="display:none; background:#1e293b;">
                <div class="col-md-4 legend-section">
                    <h5>🔥 Heat Status</h5>
                    <p style="font-size:0.85rem;"><span style="color:#ef4444; font-weight:bold;">RED:</span> Extreme Outlier (>3σ)</p>
                    <p style="font-size:0.85rem;"><span style="color:#f59e0b; font-weight:bold;">YELLOW:</span> Elevated Activity (>1.5σ)</p>
                </div>
                <div class="col-md-4 legend-section">
                    <h5>🚀 Signals</h5>
                    <p style="font-size:0.85rem;"><span style="color:#06b6d4; font-weight:bold;">ACCUM:</span> Mentions Up / Price Flat</p>
                    <p style="font-size:0.85rem;"><span style="color:#f59e0b; font-weight:bold;">TREND:</span> 5+ Days on Top List</p>
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
            if (x.style.display === "none") {{ x.style.display = "flex"; arrow.innerText = "▲"; }}
            else {{ x.style.display = "none"; arrow.innerText = "▼"; }}
        }}
        function resetFilters() {{
            $('#minPrice, #minVol').val('');
            $('#btnradio1').prop('checked', true);
            redraw();
        }}
        $(document).ready(function(){{ 
            var table=$('.table').DataTable({{
                "order":[[0,"asc"]],
                "pageLength": 25,
                "columnDefs": [{{ "visible": false, "targets": [13, 14] }}, {{ "orderData": [14], "targets": [7] }}],
                "drawCallback": function() {{
                    var api = this.api();
                    $("#stockCounter").text("Showing " + api.rows({{filter:'applied'}}).count() + " / " + api.rows().count() + " Tickers");
                }}
            }});
            $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {{
                var typeTag = data[13] || "";
                var viewMode = $('input[name="btnradio"]:checked').attr('id');
                if (viewMode == 'btnradio2' && typeTag == 'ETF') return false;
                if (viewMode == 'btnradio3' && typeTag == 'STOCK') return false;
                var minPrice = parseFloat($('#minPrice').val()) || 0;
                var price = parseFloat(data[6].replace(/[$,]/g, '')) || 0;
                if (price < minPrice) return false;
                var minVol = parseFloat($('#minVol').val()) || 0;
                if (parseFloat(data[14]) < minVol) return false;
                return true;
            }});
            $('#minPrice, #minVol').on('keyup change', function() {{ table.draw(); }});
            window.redraw = function() {{ table.draw(); }};
            var d=new Date($("#time").data("utc"));
            $("#time").text("Last Updated: " + d.toLocaleString());
        }});
    </script></body></html>"""

    filename = f"scan_{time.strftime('%Y-%m-%d_%H-%M')}.html"
    filepath = os.path.join(PUBLIC_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)
    shutil.copy(filepath, os.path.join(PUBLIC_DIR, "index.html"))
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

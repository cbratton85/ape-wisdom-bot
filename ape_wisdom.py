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
# The script will save the HTML here for the website
PUBLIC_DIR = os.path.join(SCRIPT_DIR, "public") 

CACHE_FILE = os.path.join(SCRIPT_DIR, "ape_cache.json")
MARKET_DATA_CACHE_FILE = os.path.join(SCRIPT_DIR, "market_data.pkl")
HISTORY_FILE = os.path.join(SCRIPT_DIR, "market_history.json")
CACHE_EXPIRY_SECONDS = 3600 
RETENTION_DAYS = 14          

# --- FILTERS ---
MIN_PRICE = 5        
MIN_AVG_VOLUME = 250000        
AVG_VOLUME_DAYS = 10
PAGE_SIZE = 30
NAME_MAX_WIDTH = 35
INDUSTRY_MAX_WIDTH = 41     
COL_WIDTHS = [35, 8, 8, 10, 8, 8, 8, 8, INDUSTRY_MAX_WIDTH] 
DASH_LINE = "-" * 135
REQUEST_DELAY_MIN = 1.5 
REQUEST_DELAY_MAX = 3.0
TICKER_FIXES = {'GPS': 'GAP', 'FB': 'META', 'SVRE': 'SaverOne', 'MMTLP': 'MMTLP', 'SAVERONE': 'SVRE', 'DTC': 'SOLO', 'APE': 'AMC'} 

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
STATS = {"total": 0, "filtered_price": 0, "filtered_vol": 0, "failed_data": 0}

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
    name, meta, quote_type, mcap = ticker, "Unknown", "EQUITY", 0
    try:
        dat = yf.Ticker(ticker) 
        info = dat.info
        if info:
            quote_type = info.get('quoteType', 'EQUITY')
            name = info.get('shortName') or info.get('longName') or ticker
            mcap = info.get('marketCap', 0)
            if quote_type == 'ETF':
                meta = info.get('category', 'Unknown')
            else:
                s = info.get('sector', 'Unknown')
                i = info.get('industry', 'Unknown')
                meta = f"{s} - {i}" if s != 'Unknown' else 'Unknown'
    except: pass
    return {'ticker': ticker, 'name': name, 'meta': meta, 'type': quote_type, 'mcap': mcap}

def filter_and_process(stocks):
    if not stocks: return pd.DataFrame()
    us_tickers = list(set([TICKER_FIXES.get(s['ticker'], s['ticker'].replace('.', '-')) for s in stocks]))
    local_cache = load_cache()
    
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
        market_data = yf.download(us_tickers, period="40d", interval="1d", group_by='ticker', progress=False, threads=True)
        if not market_data.empty: market_data.to_pickle(MARKET_DATA_CACHE_FILE)

    if len(us_tickers) == 1 and not market_data.empty:
        idx = pd.MultiIndex.from_product([us_tickers, market_data.columns])
        market_data.columns = idx

    final_list = []
    for stock in stocks:
        t = TICKER_FIXES.get(stock['ticker'], stock['ticker'].replace('.', '-'))
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                if t in market_data.columns.levels[0]: hist = market_data[t].dropna()
                else: continue
            else: hist = market_data.dropna()

            if hist.empty: continue
            curr_p = hist['Close'].iloc[-1]
            avg_v = hist['Volume'].tail(AVG_VOLUME_DAYS).mean()
            if curr_p < MIN_PRICE: continue
            if avg_v < MIN_AVG_VOLUME: continue

            info = local_cache.get(t, {})
            name = str(info.get('name', t)).replace('"', '').strip()[:NAME_MAX_WIDTH]
            cur_m, old_m = int(stock.get('mentions', 0)), int(stock.get('mentions_24h_ago', 1))
            m_perc = int(((cur_m - (old_m or 1)) / (old_m or 1) * 100))
            s_perc = int((hist['Volume'].iloc[-1] / avg_v * 100)) if avg_v > 0 else 0
            mcap = info.get('mcap', 10**9) or 10**9
            squeeze_score = (cur_m * s_perc) / max(math.log(mcap, 10), 1)

            final_list.append({
                "Name": name, "Sym": t, "Rank+": int(stock['rank_24h_ago']) - int(stock['rank']),
                "Price": float(curr_p), "Surge": s_perc, "Mnt%": m_perc, "Type": info.get('type', 'EQUITY'),
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
        export_df = df.copy()
        
        # --- CREATE PUBLIC FOLDER FOR WEBSITE ---
        if not os.path.exists(PUBLIC_DIR):
            os.makedirs(PUBLIC_DIR)

        def color_span(text, color_hex): return f'<span style="color: {color_hex}; font-weight: bold;">{text}</span>'
        C_GREEN, C_YELLOW, C_RED, C_CYAN, C_MAGENTA, C_WHITE = "#00ff00", "#ffff00", "#ff4444", "#00ffff", "#ff00ff", "#ffffff"
        export_df['Type_Tag'] = 'STOCK'
        tracker = HistoryTracker(HISTORY_FILE)
        export_df['Vel'] = 0; export_df['Sig'] = ""

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
            export_df.at[index, 'Sig'] = color_span(sig_text, sig_color)
            
            nm_clr = C_RED if row['Master_Score'] > 3.0 else (C_YELLOW if row['Master_Score'] > 1.5 else C_WHITE)
            export_df.at[index, 'Name'] = color_span(row['Name'], nm_clr)
            
            # Simple Column Coloring
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
            export_df.at[index, 'Sym'] = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color: #4da6ff; text-decoration: none;">{t}</a>'
            export_df.at[index, 'Price'] = f"${row['Price']:.2f}"

        cols = ['Name', 'Sym', 'Vel', 'Sig', 'Rank+', 'Price', 'Surge', 'Mnt%', 'Upvotes', 'Squeeze', 'Meta', 'Type_Tag']
        final_df = export_df[cols]
        table_html = final_df.to_html(classes='table table-dark table-hover', index=False, escape=False)
        utc_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # HTML Template
        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Ape Wisdom</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <style>body{{background-color:#121212;color:#e0e0e0;font-family:monospace;padding:20px}}
        .table-dark{{--bs-table-bg:#1e1e1e;color:#ccc}} th{{color:#00ff00}} td{{vertical-align:middle}} a{{color:#4da6ff}}
        .legend-box{{background:#2a2a2a;padding:10px;margin:15px 0;display:flex;gap:15px;border-radius:5px}}</style></head>
        <body><div class="container-fluid" style="max-width:1400px">
        <div class="d-flex justify-content-between"><h2>🦍 Ape Wisdom</h2><span id="time" data-utc="{utc_timestamp}"></span></div>
        <div class="legend-box">
            <span>🔴=Hot 🟡=Warm 🟢=Good</span>
            <span>💎 ACCUM = Price Flat/Mentions Up</span>
            <button class="btn btn-sm btn-outline-light" onclick="filterTable('all')">All</button>
            <button class="btn btn-sm btn-outline-light" onclick="filterTable('stock')">Stocks</button>
            <button class="btn btn-sm btn-outline-light" onclick="filterTable('etf')">ETFs</button>
        </div>
        {table_html}</div>
        <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
        <script>
        $(document).ready(function(){{ var table=$('.table').DataTable({{"order":[[4,"desc"]],"pageLength":25}});
        window.filterTable=function(t){{$.fn.dataTable.ext.search.pop();if(t!='all'){{$.fn.dataTable.ext.search.push(
        function(s,d,i){{return (d[11]||"")==(t=='etf'?'ETF':'STOCK')}})}};table.draw()}};
        var d=new Date($("#time").data("utc"));$("#time").text(d.toLocaleString());}});
        </script></body></html>"""

        # 1. Save with timestamp (History)
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        filename = f"scan_{timestamp}.html"
        filepath = os.path.join(PUBLIC_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)

        # 2. Save as Index (Current Link)
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
    REPO_NAME = os.environ.get('GITHUB_REPOSITORY') # e.g. "username/my-repo"
    
    if not DISCORD_URL or not REPO_NAME: 
        print("Missing Discord URL or Repo Name")
        return

    # Construct the Website URL
    try:
        user, repo = REPO_NAME.split('/')
        # This is the standard GitHub Pages URL format
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
    
    # Interactive Mode (Local)
    raw = get_all_trending_stocks()
    df = filter_and_process(raw)
    export_interactive_html(df)
    print("Done.")

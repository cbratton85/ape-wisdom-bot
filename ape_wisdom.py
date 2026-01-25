import requests
import yfinance as yf
import pandas as pd
import time
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
import sys
import math
import random
import json
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
#                 CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "ape_cache.json")
MARKET_DATA_CACHE_FILE = os.path.join(SCRIPT_DIR, "market_data.pkl")
CACHE_EXPIRY_SECONDS = 3600  # 1 Hour

# --- UPDATED FILTERS (Change 3) ---
MIN_PRICE = 5      
MIN_AVG_VOLUME = 250000      
AVG_VOLUME_DAYS = 10
PAGE_SIZE = 30

# --- LAYOUT SETTINGS (Change 5) ---
NAME_MAX_WIDTH = 35
INDUSTRY_MAX_WIDTH = 41     
# Widths: [Name, Sym, Rank, Price, Surge, Mnt, Upvt, Squeeze, Industry]
COL_WIDTHS = [35, 8, 8, 10, 8, 8, 8, 8, INDUSTRY_MAX_WIDTH] 
DASH_LINE = "-" * 135

# --- SAFETY SETTINGS ---
# Time to wait between requests (in seconds). 
# 2.0 is safe. 0.1 is fast but risky.
REQUEST_DELAY_MIN = 1.5 
REQUEST_DELAY_MAX = 3.0

TICKER_FIXES = {
    'GPS': 'GAP', 'FB': 'META', 'SVRE': 'SaverOne', 'MMTLP': 'MMTLP',
    'SAVERONE': 'SVRE', 'DTC': 'SOLO', 'APE': 'AMC'
} 

# ANSI COLORS
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_RESET = '\033[0m'
C_MAGENTA = '\033[95m'
C_BOLD = '\033[1m'
# ==========================================

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

STATS = {"total": 0, "filtered_price": 0, "filtered_vol": 0, "failed_data": 0}

def clear_screen():
    if os.name == 'nt': os.system('cls')
    else: sys.stdout.write("\033[H\033[2J\033[3J"); sys.stdout.flush()

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_cache(cache_data):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=4)
    except:
        pass

def fetch_sector_fallback(ticker):
    """
    Tries to fetch Sector/Industry from StockAnalysis.
    Now checks BOTH 'ETF' and 'Stocks' pages to catch Trusts like PSLV/PHYS.
    """
    clean_ticker = ticker.replace('-', '.')
    
    # 1. Try StockAnalysis (ETF Page)
    try:
        url = f"https://stockanalysis.com/etf/{clean_ticker}/"
        r = session.get(url, timeout=4)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            # Look for Category
            rows = soup.find_all('div', class_='px-4 py-3')
            for row in rows:
                label = row.find('span', class_='text-sm')
                if label and 'Category' in label.text:
                    val = row.find('div', class_='text-lg font-bold')
                    if val: return val.get_text(strip=True)
                    val_alt = row.find('a')
                    if val_alt: return val_alt.get_text(strip=True)
    except: pass

    # 2. Try StockAnalysis (Stock Page) - CRITICAL FIX FOR PSLV/PHYS
    try:
        # Trusts often live here instead of the ETF section
        url = f"https://stockanalysis.com/stocks/{clean_ticker}/"
        r = session.get(url, timeout=4)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            # Stocks page layout is different. Look for "Sector" and "Industry" labels.
            # Usually found in a table or grid.
            info_table = soup.find('div', class_='grid grid-cols-2 gap-x-4 gap-y-2')
            if info_table:
                # StockAnalysis usually puts Sector/Industry clearly in the overview
                # We will grab Industry as it is more descriptive for Trusts
                txt = info_table.get_text()
                if "Industry" in txt:
                    # Parse the specific industry link text
                    ind_link = soup.find('a', href=re.compile(r'/stocks/industry/'))
                    if ind_link:
                        return ind_link.get_text(strip=True)
    except: pass

    # 3. Fallback to Finviz (Last Resort)
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        r = session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=4)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            links = soup.find_all('a', class_='tab-link') 
            if len(links) >= 2:
                return f"{links[0].text} - {links[1].text}"
    except: pass
    
    return "Unknown"

def fetch_google_fallback(ticker):
    try:
        url = f"https://www.google.com/finance?q={ticker}"
        r = session.get(url, timeout=5)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        
        page_title = soup.title.string if soup.title else ""
        name = ticker 
        match = re.search(r"^(.*?)\s\(", page_title)
        if match: name = match.group(1).strip()
        
        meta = "Unknown"
        target_labels = ["Sector", "Industry", "Sector / Industry"]
        for label in target_labels:
            label_node = soup.find(string=re.compile(f"^{label}$", re.IGNORECASE))
            if label_node and label_node.parent:
                value_div = label_node.parent.find_next_sibling("div")
                if value_div:
                    meta = value_div.get_text(strip=True)
                    break 

        return {'ticker': ticker, 'name': name, 'meta': meta, 'type': 'EQUITY', 'mcap': 0}
    except: return None

def fetch_meta_data_robust(ticker):
    # Prepare standard outputs
    name, meta, quote_type, mcap = ticker, "Unknown", "EQUITY", 0
    
    # SLOW DOWN: Sleep before every single request to prevent bans
    time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

    # 1. Try YFinance API
    with open(os.devnull, 'w') as f:
        old_stderr = sys.stderr
        sys.stderr = f
        try:
            dat = yf.Ticker(ticker) 
            info = dat.info
            
            if info:
                quote_type = info.get('quoteType', 'EQUITY')
                name = info.get('shortName') or info.get('longName') or ticker
                mcap = info.get('marketCap', 0)

                if quote_type == 'ETF':
                    category = info.get('category')
                    if category and 'unknown' not in category.lower():
                        meta = category
                    else:
                        meta = 'Unknown' 
                else:
                    s = info.get('sector', 'Unknown')
                    i = info.get('industry', 'Unknown')
                    if s != 'Unknown' and i != 'Unknown':
                        meta = f"{s} - {i}"
                    else:
                        meta = 'Unknown'
        except: pass
        finally: sys.stderr = old_stderr 

    # 2. External Fallback (StockAnalysis/Finviz) if YFinance failed
    if meta in ['Unknown', 'General ETF', None]:
        # Sleep again before hitting the fallback site
        time.sleep(random.uniform(1.0, 2.0))
        found_meta = fetch_sector_fallback(ticker)
        if found_meta != "Unknown":
            meta = found_meta

    # 3. Apply Truncation
    if len(meta) > INDUSTRY_MAX_WIDTH:
        meta = meta[:INDUSTRY_MAX_WIDTH-3] + "..."

    # 4. Last Resort: Google Fallback for Name/Meta
    if name == ticker and meta == "Unknown":
        time.sleep(1.0) # Sleep before Google
        g_res = fetch_google_fallback(ticker)
        if g_res: return g_res

    return {'ticker': ticker, 'name': name, 'meta': meta, 'type': quote_type, 'mcap': mcap}

def print_ape_ui(current_sort, df, page, total_pages, view_mode):
    hidden_count = STATS["filtered_price"] + STATS["filtered_vol"]
    view_str = f"VIEW: {C_YELLOW}{view_mode.upper()}{C_RESET}"
    sort_str = f"SORT: {C_YELLOW}{current_sort}{C_RESET}"
    
    # Get current time for the header
    now = time.strftime("%H:%M:%S")

    # Modified Header with Timestamp
    print(f"\n {C_CYAN}{C_BOLD}APE WISDOM TRENDING DASHBOARD {C_RESET}| {C_YELLOW}Last Updated: {now}{C_RESET}")
    
    print(f" [ FILTERS: Price > ${MIN_PRICE:.2f} | Vol > {MIN_AVG_VOLUME:,} | {view_str} | {sort_str} ]")
    print(f" [ STATS: Found: {STATS['total']} | {C_RED}Hidden: {hidden_count}{C_RESET} | Showing: {len(df)} | {C_YELLOW}Page {page+1} of {total_pages}{C_RESET} ]")
    
    # Correctly prints a full line of equals signs (=) matching the dash width
    print(DASH_LINE.replace("-", "="))

def filter_and_process(stocks):
    if not stocks: return pd.DataFrame()
    STATS["total"] = len(stocks)
    STATS["filtered_price"] = 0
    STATS["filtered_vol"] = 0
    STATS["failed_data"] = 0
    
    us_tickers = list(set([TICKER_FIXES.get(s['ticker'], s['ticker'].replace('.', '-')) for s in stocks]))
    local_cache = load_cache()
    
    # --- METADATA HEALING (Slow Mode) ---
    missing = [t for t in us_tickers if t not in local_cache]
    if missing:
        print(f"{C_YELLOW}--- HEALING METADATA: {len(missing)} items (Sequential Mode) ---{C_RESET}")
        for i, t in enumerate(missing):
            try:
                pct = ((i + 1) / len(missing)) * 100
                print(f"\r    [{i+1}/{len(missing)}] {pct:.1f}% Fetching: {t:<8}", end="", flush=True)
                res = fetch_meta_data_robust(t)
                if res:
                    local_cache[res['ticker']] = res
                    if i % 5 == 0: save_cache(local_cache)
            except Exception as e:
                print(f" (Error: {e})", end="")
        save_cache(local_cache)
        print(f"\n{C_GREEN}    Done.{C_RESET}")
    else:
        print(f"{C_GREEN}--- CACHE LOADED (Metadata Current) ---{C_RESET}")

    # --- MARKET DATA DOWNLOAD ---
    market_data = None
    use_cache = False
    if os.path.exists(MARKET_DATA_CACHE_FILE):
        if (time.time() - os.path.getmtime(MARKET_DATA_CACHE_FILE)) < CACHE_EXPIRY_SECONDS:
            use_cache = True

    if use_cache:
        print(f"{C_GREEN}--- CACHE: Loading recent market data from file... ---{C_RESET}")
        market_data = pd.read_pickle(MARKET_DATA_CACHE_FILE)
    else:
        print(f"--- API: Downloading Market Data for {len(us_tickers)} symbols ---")
        market_data = yf.download(us_tickers, period="40d", interval="1d", group_by='ticker', progress=True, threads=True)
        if not market_data.empty:
            market_data.to_pickle(MARKET_DATA_CACHE_FILE)

    if len(us_tickers) == 1 and not market_data.empty:
        idx = pd.MultiIndex.from_product([us_tickers, market_data.columns])
        market_data.columns = idx

    # --- FILTERING & SCORING ---
    final_list = []
    for stock in stocks:
        t = TICKER_FIXES.get(stock['ticker'], stock['ticker'].replace('.', '-'))
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                if t in market_data.columns.levels[0]: hist = market_data[t].dropna()
                else: continue
            else:
                hist = market_data.dropna()

            if hist.empty: continue

            curr_p = hist['Close'].iloc[-1]
            avg_v = hist['Volume'].tail(AVG_VOLUME_DAYS).mean()
            
            if curr_p < MIN_PRICE: STATS["filtered_price"] += 1; continue
            if avg_v < MIN_AVG_VOLUME: STATS["filtered_vol"] += 1; continue

            info = local_cache.get(t, {})
            name = str(info.get('name', t)).replace('"', '').strip()[:NAME_MAX_WIDTH]
            
            cur_m, old_m = int(stock.get('mentions', 0)), int(stock.get('mentions_24h_ago', 1))
            m_perc = int(((cur_m - (old_m or 1)) / (old_m or 1) * 100))
            s_perc = int((hist['Volume'].iloc[-1] / avg_v * 100)) if avg_v > 0 else 0
            
            mcap = info.get('mcap', 10**9) or 10**9
            squeeze_score = (cur_m * s_perc) / max(math.log(mcap, 10), 1)

            final_list.append({
                "Name": name, "Sym": t, 
                "Rank+": int(stock['rank_24h_ago']) - int(stock['rank']),
                "Price": float(curr_p), "Surge": s_perc, "Mnt%": m_perc, "Type": info.get('type', 'EQUITY'),
                "Upvotes": int(stock.get('upvotes', 0)), "Meta": info.get('meta', '-'), 
                "Squeeze": squeeze_score
            })
        except: continue
    
    # --- CALCULATE STATISTICAL SIGNIFICANCE (Z-SCORES) ---
    df = pd.DataFrame(final_list)
    if not df.empty:
        cols_to_score = ['Rank+', 'Surge', 'Mnt%', 'Squeeze', 'Upvotes']
        for col in cols_to_score:
            col_mean = df[col].mean()
            col_std = df[col].std(ddof=0)
            if pd.isna(col_std) or col_std == 0: col_std = 1
            df[f'z_{col}'] = (df[col] - col_mean) / col_std

        # --- MASTER SCORE (GAINERS ONLY) ---
        # We use .clip(lower=0) so negative numbers count as 0.
        # This prevents a drop in one area from canceling out a gain in another.
        df['Master_Score'] = (
            df['z_Rank+'].clip(lower=0) + 
            df['z_Surge'].clip(lower=0) + 
            df['z_Mnt%'].clip(lower=0) + 
            df['z_Upvotes'].clip(lower=0) + 
            (df['z_Squeeze'].clip(lower=0) * 0.5) 
        )

    print(f"{C_GREEN}--- ANALYSIS COMPLETE ---{C_RESET}")
    return df

def get_all_trending_stocks():
    all_results, page, total_pages = [], 1, 1
    print(f"{C_CYAN}--- API: Fetching list of trending stocks ---{C_RESET}")
    while page <= total_pages:
        try:
            # Live status update on one line
            print(f"\r    Fetching page {page} of {total_pages}...", end="", flush=True)
            r = requests.get(f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                results = data.get('results', [])
                if not results: break
                all_results.extend(results)
                # On the first successful fetch, update total_pages
                if page == 1:
                    total_pages = data.get('pages', 1)
                page += 1
            else: break
        except: break
    print(f"\n{C_GREEN}    Done. Found {len(all_results)} total symbols.{C_RESET}")
    return all_results

def export_to_txt(df):
    try:
        # 1. Desktop Detection
        export_folder = os.path.join(SCRIPT_DIR, "exports")
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
            print(f"\n{C_CYAN}Created new folder for exports: {export_folder}{C_RESET}")

        filename = os.path.join(export_folder, "Trending_Symbols.txt")
        
        # 2. Ask for Limit
        print(f"\n {C_YELLOW}--- EXPORT SYMBOLS ONLY ---{C_RESET}")
        limit_input = input(f" How many symbols to include? (1-{len(df)}) [Default: all]: ")
        limit = int(limit_input) if limit_input.strip() else len(df)

        # 3. Extract just the 'Sym' column and join them with a space
        symbols_list = df['Sym'].head(limit).tolist()
        content = " ".join(symbols_list)
        
        # 4. Write to File (No headers, just the symbols)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"\n{C_GREEN}[+] Success! {len(symbols_list)} symbols saved to: {filename}{C_RESET}")
        time.sleep(2)
        
    except Exception as e:
        print(f"\n{C_RED}[!] Export Failed: {e}{C_RESET}")
        input("Press Enter to return to dashboard...")

def export_interactive_html(df):
    try:
        # 1. Work on a copy
        export_df = df.copy()

        # --- HELPER: Color Wrapper ---
        def color_span(text, color_hex):
            return f'<span style="color: {color_hex}; font-weight: bold;">{text}</span>'

        # Colors (Dark Mode)
        C_GREEN_HTML = "#00ff00"
        C_YELLOW_HTML = "#ffff00"
        C_RED_HTML = "#ff4444"
        C_CYAN_HTML = "#00ffff"
        C_MAGENTA_HTML = "#ff00ff"
        C_WHITE_HTML = "#ffffff"

        # Initialize the new "Type_Tag" column
        export_df['Type_Tag'] = 'STOCK'

        # 2. APPLY LOGIC ROW BY ROW
        for index, row in export_df.iterrows():
            
            # Name Coloring
            name_color = C_WHITE_HTML
            if row['Master_Score'] > 3.0:   name_color = C_RED_HTML
            elif row['Master_Score'] > 1.5: name_color = C_YELLOW_HTML
            export_df.at[index, 'Name'] = color_span(row['Name'], name_color)

            # Rank+ Coloring
            z = row['z_Rank+']
            val = row['Rank+']
            c = C_WHITE_HTML
            if z >= 2.0: c = C_YELLOW_HTML
            elif z >= 1.0: c = C_GREEN_HTML
            export_df.at[index, 'Rank+'] = color_span(val, c)

            # Surge Coloring
            z = row['z_Surge']
            val = f"{row['Surge']:.0f}%"
            c = C_WHITE_HTML
            if z >= 2.0: c = C_YELLOW_HTML
            elif z >= 1.0: c = C_GREEN_HTML
            export_df.at[index, 'Surge'] = color_span(val, c)

            # Mentions Coloring
            z = row['z_Mnt%']
            val = f"{row['Mnt%']:.0f}%"
            c = C_WHITE_HTML
            if z >= 2.0: c = C_YELLOW_HTML
            elif z >= 1.0: c = C_GREEN_HTML
            export_df.at[index, 'Mnt%'] = color_span(val, c)

            # Upvotes Coloring
            z = row['z_Upvotes']
            c = C_GREEN_HTML if z > 1.5 else C_WHITE_HTML
            export_df.at[index, 'Upvotes'] = color_span(row['Upvotes'], c)

            # Squeeze Coloring
            z = row['z_Squeeze']
            c = C_CYAN_HTML if z > 1.5 else C_WHITE_HTML
            export_df.at[index, 'Squeeze'] = color_span(int(row['Squeeze']), c)

            # Meta/Industry Coloring & Type Tagging
            is_fund = row['Type'] == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
            c = C_MAGENTA_HTML if is_fund else C_WHITE_HTML
            export_df.at[index, 'Meta'] = color_span(row['Meta'], c)
            
            if is_fund:
                export_df.at[index, 'Type_Tag'] = 'ETF'
            else:
                export_df.at[index, 'Type_Tag'] = 'STOCK'

            # Ticker Link
            t = row['Sym']
            link = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color: #4da6ff; text-decoration: none;">{t}</a>'
            export_df.at[index, 'Sym'] = link
            
            # Price Formatting
            export_df.at[index, 'Price'] = f"${row['Price']:.2f}"

        # 3. Select Columns
        cols_to_keep = ['Name', 'Sym', 'Rank+', 'Price', 'Surge', 'Mnt%', 'Upvotes', 'Squeeze', 'Meta', 'Master_Score', 'Type_Tag']
        final_df = export_df[cols_to_keep]

        # 4. Generate HTML Table
        table_html = final_df.to_html(classes='table table-dark table-hover', index=False, escape=False)

        # UTC Time
        utc_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # 5. Create HTML Template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ape Wisdom - Dark Mode</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
            <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
            <style>
                body {{ background-color: #121212; color: #e0e0e0; font-family: 'Consolas', 'Monaco', monospace; padding: 20px; }}
                .container {{ max-width: 95%; background: #1e1e1e; padding: 25px; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.7); }}
                
                h2 {{ color: #00ff00; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0; }}
                #time-display {{ color: #888; font-size: 0.9em; }}

                /* LEGEND BOX */
                .legend-box {{
                    background-color: #2a2a2a;
                    border: 1px solid #444;
                    border-radius: 6px;
                    padding: 10px 15px;
                    margin-top: 15px;
                    margin-bottom: 20px;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    font-size: 0.85rem;
                    align-items: center;
                    justify-content: space-between;
                }}
                .legend-left {{ display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }}
                .legend-group {{ display: flex; gap: 10px; align-items: center; border-right: 1px solid #444; padding-right: 20px; }}
                .legend-group:last-child {{ border-right: none; }}
                .legend-label {{ color: #888; text-transform: uppercase; font-size: 0.75rem; margin-right: 5px; }}

                .btn-group-xs > .btn, .btn-xs {{ padding: .25rem .4rem; font-size: .875rem; line-height: .5; border-radius: .2rem; }}
                .btn-check:checked + .btn-outline-light {{ background-color: #00ff00; color: black; border-color: #00ff00; }}
                
                table.dataTable {{ border-collapse: collapse !important; }}
                .table-dark {{ --bs-table-bg: #1e1e1e; color: #ccc; }}
                .table-dark th {{ color: #00ff00; border-bottom: 2px solid #444; }}
                .table-dark td {{ border-bottom: 1px solid #333; vertical-align: middle; }}
                
                .dataTables_filter input, .dataTables_length select {{ background-color: #333; color: white; border: 1px solid #555; }}
                .page-link {{ background-color: #333; border-color: #444; color: #00ff00; }}
                .page-item.active .page-link {{ background-color: #00ff00; border-color: #00ff00; color: black; }}
            </style>
        </head>
        <body>

        <div class="container">
            <div class="d-flex justify-content-between align-items-end">
                <h2>🦍 Ape Wisdom Dashboard</h2>
                <span id="time-display" data-utc="{utc_timestamp}">Loading time...</span>
            </div>

            <div class="legend-box">
                <div class="legend-left">
                    <div class="legend-group">
                        <span class="legend-label">STATUS:</span>
                        <span><span style="color:#ff4444; font-weight:bold;">RED</span> = Hot</span>
                        <span><span style="color:#ffff00; font-weight:bold;">YEL</span> = Warm</span>
                        <span><span style="color:#ff00ff; font-weight:bold;">MAG</span> = ETF</span>
                    </div>

                    <div class="legend-group">
                        <span class="legend-label">METRICS:</span>
                        <span><span style="color:#ffff00;">120%</span> = Extreme (>2&sigma;)</span>
                        <span><span style="color:#00ff00;">45%</span> = Strong (>1&sigma;)</span>
                    </div>
                </div>

                <div class="legend-group" style="border:none; padding-right:0;">
                    <span class="legend-label">VIEW:</span>
                    <div class="btn-group" role="group">
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio1" autocomplete="off" checked onclick="filterTable('all')">
                        <label class="btn btn-outline-light btn-sm" for="btnradio1">All</label>

                        <input type="radio" class="btn-check" name="btnradio" id="btnradio2" autocomplete="off" onclick="filterTable('stock')">
                        <label class="btn btn-outline-light btn-sm" for="btnradio2">Stocks Only</label>

                        <input type="radio" class="btn-check" name="btnradio" id="btnradio3" autocomplete="off" onclick="filterTable('etf')">
                        <label class="btn btn-outline-light btn-sm" for="btnradio3">ETFs Only</label>
                    </div>
                </div>
            </div>
            
            {table_html}
        </div>

        <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

        <script>
            var table;

            $(document).ready(function() {{
                table = $('.table').DataTable({{
                    "pageLength": 10,
                    "order": [[ 2, "desc" ]],
                    "columnDefs": [ 
                        {{ "visible": false, "targets": [9, 10] }} 
                    ],
                    "language": {{ "search": "SEARCH TICKER:", "lengthMenu": "Show _MENU_ entries" }}
                }});

                // Time Converter
                const timeSpan = document.getElementById('time-display');
                if (timeSpan) {{
                    const rawUtc = timeSpan.getAttribute('data-utc');
                    const localDate = new Date(rawUtc);
                    const dateString = localDate.toLocaleString(undefined, {{ 
                        year: 'numeric', month: 'numeric', day: 'numeric', 
                        hour: '2-digit', minute: '2-digit', second: '2-digit',
                        timeZoneName: 'short' 
                    }});
                    timeSpan.textContent = "Last Updated: " + dateString;
                }}
            }});

            // FILTER LOGIC
            function filterTable(type) {{
                $.fn.dataTable.ext.search.pop(); 
                
                if (type === 'all') {{
                    table.draw();
                    return;
                }}

                $.fn.dataTable.ext.search.push(
                    function(settings, data, dataIndex) {{
                        var typeTag = data[10] || ""; 
                        if (type === 'etf') return typeTag === 'ETF';
                        if (type === 'stock') return typeTag === 'STOCK';
                        return true;
                    }}
                );
                table.draw();
            }}
        </script>
        </body>
        </html>
        """

        # 6. Save with Timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(SCRIPT_DIR, f"ape_dashboard_{timestamp}.html")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n{C_GREEN}[+] HTML Dashboard generated: {filename}{C_RESET}")
        time.sleep(2)

    except Exception as e:
        print(f"\n{C_RED}[!] Export Failed: {e}{C_RESET}")
        time.sleep(2)

def send_email(filename):
    print(f"\n{C_YELLOW}--- Sending Email... ---{C_RESET}")
    
    # 1. Get Credentials from Environment Variables
    SMTP_SERVER = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('EMAIL_PORT', 587))
    SENDER_EMAIL = os.environ.get('EMAIL_USER')
    SENDER_PASSWORD = os.environ.get('EMAIL_PASSWORD')
    RECIPIENT_EMAIL = os.environ.get('EMAIL_TO')

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print(f"{C_RED}[!] Error: Missing Email Credentials.{C_RESET}")
        return

    # 2. Create the Email
    msg = MIMEMultipart()
    msg['Subject'] = f"🦍 Ape Wisdom Dashboard - {time.strftime('%Y-%m-%d %H:%M')}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL

    body = "Here is your latest market scan. Download the attachment to view the interactive dashboard."
    msg.attach(MIMEText(body, 'plain'))

    # 3. Attach the HTML File
    try:
        with open(filename, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(filename))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(filename)}"'
        msg.attach(part)

        # 4. Connect and Send
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            
        print(f"{C_GREEN}[+] Email sent successfully to {RECIPIENT_EMAIL}{C_RESET}")

    except Exception as e:
        print(f"{C_RED}[!] Email Failed: {e}{C_RESET}")

if __name__ == "__main__":
    
    # =========================================
    # 1. AUTO MODE (For GitHub/Cron/Email)
    # =========================================
    # This detects if the script is running in "Headless" mode
    if "--auto" in sys.argv:
        print(f"{C_YELLOW}--- STARTING AUTO-MODE SCAN ---{C_RESET}")
        
        # A. Fetch
        raw_data = get_all_trending_stocks()
        if not raw_data:
            print("No data found. Exiting.")
            sys.exit()
            
        # B. Process
        df = filter_and_process(raw_data)
        
        # C. Generate HTML (Silent)
        filename = export_interactive_html(df)
        
        # D. Send Email (Or Telegram if you prefer)
        if filename:
            send_email(filename)
            
        print(f"{C_GREEN}--- AUTO SCAN COMPLETE ---{C_RESET}")
        sys.exit() # <--- Important: Stops here so it doesn't open the menu

    # =========================================
    # 2. INTERACTIVE MODE (For You)
    # =========================================
    raw_data = get_all_trending_stocks()
    CACHED_DF = filter_and_process(raw_data)
    
    # Change 4: Added "0" for Master Score
    sort_map = {
        "T": ("Master_Score", False),
        "0": ("Master_Score", False),   # <--- Press 0 for Hot Stocks
        "1": ("Name", True),
        "2": ("Sym", True),
        "3": ("Rank+", False),
        "4": ("Price", False),
        "5": ("Surge", False),
        "6": ("Mnt%", False),
        "7": ("Upvotes", False),
        "8": ("Squeeze", False),
        "9": ("Meta", True)
    }
    
    current_key, current_page, view_mode = "3", 0, "all"

    while True:
        clear_screen()
        sort_col, sort_asc = sort_map.get(current_key, ("Rank+", False))

        # --- Apply View Filter ---
        if view_mode == 'stocks':
            df_view = CACHED_DF[CACHED_DF['Type'] == 'EQUITY']
        elif view_mode == 'etfs':
            df_view = CACHED_DF[CACHED_DF['Type'] == 'ETF']
        else: # 'all'
            df_view = CACHED_DF
        
        df_sorted = df_view.sort_values(by=sort_col, ascending=sort_asc)
        
        total_pages = max(1, math.ceil(len(df_sorted) / PAGE_SIZE))
        df_page = df_sorted.iloc[current_page*PAGE_SIZE : (current_page+1)*PAGE_SIZE]
        
        # We can remove print_ape_ui() call if the code below does the printing manually
        # OR keep it if it just prints the title banner. Assuming it prints title:
        # print_ape_ui(sort_col, df_sorted, current_page, total_pages, view_mode)
        
        # --- HEADER & DATA PRINTING ---
        if not df_page.empty:
            numbers_header = ""
            labels_header = ""
            labels = ["Name", "Sym", "Rank+", "Price", "R_Vol", "Mnt%", "Votes", "Squeeze", "Industry/Sector"]
            
            for i, (label, width) in enumerate(zip(labels, COL_WIDTHS)):
                numbers_header += f"{i+1:<{width}}"
                labels_header += f"{label:<{width}}"
                if label == "Price": # Fix alignment for Price column
                    numbers_header += " "
                    labels_header += " "
            
            print(f" {C_BOLD}{C_YELLOW}{numbers_header}{C_RESET}") 
            print(f" {C_BOLD}{labels_header}{C_RESET}\n" + DASH_LINE.replace("-", "="))

            for _, row in df_page.iterrows():
                # 1. Calculate Colors
                z_r = row['z_Rank+']
                r_color = C_YELLOW if z_r >= 2.0 else (C_GREEN if z_r >= 1.0 else "")

                z_s = row['z_Surge']
                s_color = C_YELLOW if z_s >= 2.0 else (C_GREEN if z_s >= 1.0 else "")

                z_m = row['z_Mnt%']
                m_color = C_YELLOW if z_m >= 2.0 else (C_GREEN if z_m >= 1.0 else "")

                sq_color = C_CYAN if row['z_Squeeze'] > 1.5 else ""
                v_color = C_GREEN if row['z_Upvotes'] > 1.5 else ""
                
                is_fund = row['Type'] == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
                meta_color = C_MAGENTA if is_fund else ""

                if row['Master_Score'] > 3.0:   name_color = C_RED
                elif row['Master_Score'] > 1.5: name_color = C_YELLOW
                else:                           name_color = ""

                # 2. Prepare Strings (Do this HERE to avoid the SyntaxError)
                clean_name = str(row['Name']).replace('\n', '').strip()[:NAME_MAX_WIDTH]
                clean_meta = str(row['Meta']).replace('\n', '').strip()[:INDUSTRY_MAX_WIDTH]
                surge_str = f"{row['Surge']:.0f}%"  # <--- Calculating this here fixes the error
                mnt_str = f"{row['Mnt%']:.0f}%"     # <--- Calculating this here fixes the error

                # 3. Print
                print(
                    f" {name_color}{clean_name:<{COL_WIDTHS[0]}}{C_RESET}"
                    f"{row['Sym']:<{COL_WIDTHS[1]}}"
                    f"{r_color}{row['Rank+']:<{COL_WIDTHS[2]}}{C_RESET}"
                    f"${row['Price']:<{COL_WIDTHS[3]}.2f} " 
                    f"{s_color}{surge_str:<{COL_WIDTHS[4]}}{C_RESET}"  # <--- Use the variable
                    f"{m_color}{mnt_str:<{COL_WIDTHS[5]}}{C_RESET}"    # <--- Use the variable
                    f"{v_color}{row['Upvotes']:<{COL_WIDTHS[6]}}{C_RESET}"
                    f"{sq_color}{int(row['Squeeze']):<{COL_WIDTHS[7]}}{C_RESET}"
                    f"{meta_color}{clean_meta:<{COL_WIDTHS[8]}}{C_RESET}"
                )
        
        print(DASH_LINE)
        nav = (f"[,] Prev | " if current_page > 0 else "") + (f"[.] Next | " if current_page < total_pages - 1 else "")
        print(f" {C_BOLD}{nav}[T]op | [3] Rank+ | [5] Surge | [6] Mnt% | [8] Squeeze | [V]iew | [P]rint .txt | [U]pdate Meta | [R]eload | [H]tml Export | [Q]uit{C_RESET}")        
        
        try:
            choice = input(f"\n {C_CYAN}Choice:{C_RESET} ").upper()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if choice == 'H':
            # Interactive HTML export
            export_interactive_html(CACHED_DF)
            input("Press Enter to continue...")
            continue
        
        if choice == 'P':
            export_to_txt(df_sorted)
            continue 
            
        elif choice in sort_map:
            current_key = choice
            current_page = 0
            
        elif choice == '.': current_page = min(current_page + 1, total_pages - 1)
        elif choice == ',': current_page = max(0, current_page - 1)
        elif choice == 'V':
            if view_mode == 'all':      view_mode = 'stocks'
            elif view_mode == 'stocks': view_mode = 'etfs'
            else:                       view_mode = 'all'
            current_page = 0

        elif choice == 'U':
            print(f"\n{C_YELLOW}--- SCANNING FOR UNKNOWN METADATA... ---{C_RESET}")
            local_cache = load_cache()
            keys_to_reset = [k for k, v in local_cache.items() if v.get('meta') in ['Unknown', 'General ETF', '-', None]]
            
            if keys_to_reset:
                print(f" Found {len(keys_to_reset)} incomplete records. Clearing them...")
                for k in keys_to_reset: del local_cache[k]
                save_cache(local_cache)
                print(f"{C_GREEN} Cache updated. Retrying fetches...{C_RESET}")
                time.sleep(1)
                CACHED_DF = filter_and_process(raw_data)
                current_page = 0
            else:
                print(f"{C_GREEN} No 'Unknown' records found!{C_RESET}")
                time.sleep(2)
            continue

        elif choice == 'R':
            print(f"\n{C_YELLOW}--- RELOADING DATA... ---{C_RESET}")
            raw_data = get_all_trending_stocks()
            CACHED_DF = filter_and_process(raw_data)
            current_page = 0
            continue

        elif choice == 'Q': 
            break

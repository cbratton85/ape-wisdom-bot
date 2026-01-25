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
CACHE_EXPIRY_SECONDS = 3600 
RETENTION_DAYS = 14          

# --- INTERNAL DATA COLLECTION LIMITS ---
# We set these LOW so the HTML gets data to work with.
# The User filters these visually in the Dashboard.
MIN_PRICE = 0.50             
MIN_AVG_VOLUME = 5000        
AVG_VOLUME_DAYS = 30
PAGE_SIZE = 30

# --- LAYOUT WIDTHS ---
NAME_MAX_WIDTH = 45      
INDUSTRY_MAX_WIDTH = 55  
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

def export_interactive_html(df):
    try:
        export_df = df.copy()
        
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
        export_df['Vel'] = 0; export_df['Sig'] = ""

        # Create Readable Volume Column
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
            export_df.at[index, 'Sig'] = color_span(sig_text, sig_color)
            
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
            export_df.at[index, 'Sym'] = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color: #4da6ff; text-decoration: none;">{t}</a>'
            export_df.at[index, 'Price'] = f"${row['Price']:.2f}"
            export_df.at[index, 'Vol_Display'] = color_span(export_df.at[index, 'Vol_Display'], "#ccc")

        export_df.rename(columns={'Meta': 'Industry/Sector', 'Vol_Display': 'Avg Vol'}, inplace=True)

        cols = ['Name', 'Sym', 'Vel', 'Sig', 'Rank+', 'Price', 'Avg Vol', 'Surge', 'Mnt%', 'Upvotes', 'Squeeze', 'Industry/Sector', 'Type_Tag', 'AvgVol']
        final_df = export_df[cols]
        table_html = final_df.to_html(classes='table table-dark table-hover', index=False, escape=False)
        utc_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # HTML + JS Logic
        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Ape Wisdom Analysis</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <style>
            body{{background-color:#121212;color:#e0e0e0;font-family:'Consolas','Monaco',monospace;padding:20px}}
            .table-dark{{--bs-table-bg:#1e1e1e;color:#ccc}} 
            th{{color:#00ff00;border-bottom:2px solid #444; font-size: 14px;}} 
            td{{vertical-align:middle; white-space: nowrap; border-bottom:1px solid #333;}} 
            a{{color:#4da6ff; text-decoration:none;}} a:hover{{text-decoration:underline;}}
            
            /* LEGEND STYLES */
            .legend-container {{ background-color: #222; border: 1px solid #444; border-radius: 8px; margin-bottom: 20px; overflow: hidden; transition: all 0.3s ease; }}
            .legend-header {{ background: #2a2a2a; padding: 10px 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: bold; color: #fff; }}
            .legend-header:hover {{ background: #333; }}
            .legend-box {{
                padding: 15px;
                display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; font-size: 0.85rem;
                border-top: 1px solid #444;
            }}
            .legend-section h5 {{ color: #00ff00; font-size: 1rem; border-bottom: 1px solid #444; padding-bottom: 5px; margin-bottom: 10px; }}
            .legend-item {{ margin-bottom: 6px; }}
            .legend-key {{ font-weight: bold; display: inline-block; width: 100px; }}
            
            /* FILTER BAR STYLES */
            .filter-bar {{ display:flex; gap:15px; align-items:center; background:#2a2a2a; padding:10px; border-radius:5px; margin-bottom:15px; border:1px solid #444; flex-wrap:wrap;}}
            .filter-group {{ display:flex; align-items:center; gap:5px; }}
            .filter-group label {{ font-size:0.9rem; color:#aaa; }}
            .form-control-sm {{ background:#111; border:1px solid #555; color:#fff; width: 100px;}}
            
            #stockCounter {{ color: #00ff00; font-weight: bold; margin-left: auto; font-family: 'Consolas', monospace; border: 1px solid #00ff00; padding: 2px 8px; border-radius: 4px;}}
            .header-flex {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 15px; }}
            
            .page-link {{ background-color: #222; border-color: #444; color: #00ff00; }}
            .page-item.active .page-link {{ background-color: #00ff00; border-color: #00ff00; color: #000; }}
            .page-item.disabled .page-link {{ background-color: #111; border-color: #333; color: #555; }}
            
            .btn-reset {{ border: 1px solid #555; color: #fff; font-size: 0.8rem; background: #333; }}
            .btn-reset:hover {{ background: #444; color: #fff; }}
        </style>
        </head>
        <body>
        <div class="container-fluid" style="max-width:98%;">
            
            <div class="header-flex">
                <h2>🦍 Ape Wisdom Market Scan</h2>
                <span id="time" data-utc="{utc_timestamp}">Loading...</span>
            </div>

            <div class="filter-bar">
                <span style="color:#fff; font-weight:bold; margin-right:10px;">⚡ FILTERS:</span>
                
                <button class="btn btn-sm btn-reset" onclick="resetFilters()">🔄 RESET</button>

                <div class="filter-group">
                    <label>Min Price ($):</label>
                    <input type="number" id="minPrice" class="form-control form-control-sm" placeholder="Any" step="0.5">
                </div>
                
                <div class="filter-group">
                    <label>Min Avg Vol:</label>
                    <input type="number" id="minVol" class="form-control form-control-sm" placeholder="Any" step="10000">
                </div>

                <div class="filter-group">
                    <div class="btn-group" role="group">
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio1" autocomplete="off" checked onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio1">All</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio2" autocomplete="off" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio2">Stocks</label>
                        <input type="radio" class="btn-check" name="btnradio" id="btnradio3" autocomplete="off" onclick="redraw()">
                        <label class="btn btn-outline-light btn-sm" for="btnradio3">ETFs</label>
                    </div>
                </div>

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
                        <div class="legend-item"><span class="legend-key" style="color:#ff4444">RED NAME</span> Hot (>3σ). High Volatility.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ffff00">YEL NAME</span> Warm (>1.5σ). Active.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ff00ff">MAGENTA</span> Exchange Traded Fund (ETF).</div>
                    </div>

                    <div class="legend-section">
                        <h5>🚀 Momentum Signals</h5>
                        <div class="legend-item"><span class="legend-key" style="color:#00ffff">💎 ACCUM</span> <b>Accumulation:</b> Mentions RISING (>10%) + Price FLAT.</div>
                        <div class="legend-item"><span class="legend-key" style="color:#ffff00">🔥 TREND</span> <b>Persistence:</b> Has remained in the Top Trending list for 5+ consecutive days.</div>
                        <div class="legend-item"><span class="legend-key">Vel</span> <b>Acceleration:</b> Difference in Rank+ vs yesterday.</div>
                    </div>
                    
                    <div class="legend-section">
                        <h5>📊 Metrics</h5>
                        <div class="legend-item"><span class="legend-key">Rank+</span> <b>Speed:</b> Spots climbed in last 24h.</div>
                        <div class="legend-item"><span class="legend-key">Surge</span> Volume vs 30-Day Avg.</div>
                        <div class="legend-item"><span class="legend-key">Mnt%</span> Change in Mentions vs 24h ago.</div>
                        <div class="legend-item"><span class="legend-key">Upvotes</span> Raw upvote count on Reddit.</div>
                        <div class="legend-item"><span class="legend-key">Squeeze</span> (Mentions × Vol) / MarketCap.</div>
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
            if (x.style.display === "none") {{
                x.style.display = "grid";
                arrow.innerText = "▲";
            }} else {{
                x.style.display = "none";
                arrow.innerText = "▼";
            }}
        }}

        // RESET FILTERS FUNCTION
        function resetFilters() {{
            $('#minPrice').val(''); // Clear inputs
            $('#minVol').val('');
            $('#btnradio1').prop('checked', true); // Reset to All
            redraw(); // Redraw table
        }}

        $(document).ready(function(){{ 
            var table=$('.table').DataTable({{
                "order":[[4,"desc"]],
                "pageLength": 25,
                "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                "columnDefs": [ {{ "visible": false, "targets": [12, 13] }} ],
                
                "drawCallback": function(settings) {{
                    var api = this.api();
                    var total = api.rows().count();
                    var shown = api.rows({{filter:'applied'}}).count();
                    $("#stockCounter").text("Showing " + shown + " / " + total + " Tickers");
                }}
            }});
            
            $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {{
                // 1. View Type
                var typeTag = data[12] || ""; 
                var viewMode = $('input[name="btnradio"]:checked').attr('id');
                if (viewMode == 'btnradio2' && typeTag == 'ETF') return false;
                if (viewMode == 'btnradio3' && typeTag == 'STOCK') return false;

                // 2. Price Filter (Col 5)
                var minPrice = parseFloat($('#minPrice').val()) || 0; // Default to 0 if empty
                var priceStr = data[5] || "0"; 
                var price = parseFloat(priceStr.replace(/[$,]/g, '')) || 0;
                if (price < minPrice) return false;

                // 3. Volume Filter (Hidden Col 13)
                var minVol = parseFloat($('#minVol').val()) || 0; // Default to 0 if empty
                var rawVol = parseFloat(data[13]) || 0; 
                if (rawVol < minVol) return false;

                return true;
            }});

            $('#minPrice, #minVol').on('keyup change', function() {{ table.draw(); }});
            
            window.redraw = function() {{ 
                var mode = $('input[name="btnradio"]:checked').attr('id');
                var headerTxt = "Industry/Sector";
                
                if (mode == 'btnradio2') headerTxt = "Industry";
                else if (mode == 'btnradio3') headerTxt = "Sector";
                
                $(table.column(11).header()).text(headerTxt);
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
    
    # Interactive Mode (Local)
    raw = get_all_trending_stocks()
    df = filter_and_process(raw)
    export_interactive_html(df)
    print("Done.")

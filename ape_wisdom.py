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
import shutil
import numpy as np
import google.generativeai as genai

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
NAME_MAX_WIDTH = 50

REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.0
TICKER_FIXES = {'GPS': 'GAP', 'FB': 'META', 'APE': 'AMC'}

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
        now_ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        
        for _, row in df.iterrows():
            ticker = row['Sym']
            if ticker not in self.data:
                self.data[ticker] = {}
            
            self.data[ticker][now_ts] = {
                "rank": int(row.get('Rank', 0)),
                "rank_plus": int(row.get('Rank+', 0)),
                "price": float(row.get('Price', 0)),
                "mnt_perc": float(row.get('Mnt%', 0)),
                "upvotes": int(row.get('Upvotes', 0)),
                "conv": float(row.get('Conv', 0))
            }
            
        # Clean up old data
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=RETENTION_DAYS)
        new_data_cleaned = {}
        
        for ticker, entries in self.data.items():
            valid_entries = {}
            for d, v in entries.items():
                try:
                    dt = datetime.datetime.strptime(d, "%Y-%m-%d %H:%M")
                except ValueError:
                    dt = datetime.datetime.strptime(d, "%Y-%m-%d") # Fallback
                
                if dt > cutoff:
                    valid_entries[d] = v
            
            if valid_entries:
                new_data_cleaned[ticker] = valid_entries
        
        self.data = new_data_cleaned
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get_metrics(self, ticker, current_price, current_mnt, current_rank_plus, current_upvotes):
        # 1. Handle case with no history
        if ticker not in self.data or not self.data[ticker]:
            # If we have no history, the current run IS the streak
            streak = 1 if current_rank_plus > 0 else (-1 if current_rank_plus < 0 else 0)
            return {"vel": 0, "accel": 0, "upv_chg": 0, "div": False, "streak": 1, "rolling_trend": streak}

        dates = sorted(self.data[ticker].keys())
        # "prev_data" is the last SAVED snapshot (Run -1)
        prev_data = self.data[ticker][dates[-1]]
        
        # --- 1. CALCULATE LIVE STREAK ---
        # First, calculate streak from history
        rolling_trend = 0
        for d in dates:
            r_plus = self.data[ticker][d].get('rank_plus', 0)
            if r_plus > 0:
                rolling_trend = rolling_trend + 1 if rolling_trend >= 0 else 1
            elif r_plus < 0:
                rolling_trend = rolling_trend - 1 if rolling_trend <= 0 else -1
            else:
                rolling_trend = 0
        
        # Now, apply the CURRENT (Live) Rank+ to that streak
        # This fixes the "Streak +4 but Rank -12" issue
        if current_rank_plus > 0:
            rolling_trend = rolling_trend + 1 if rolling_trend >= 0 else 1
        elif current_rank_plus < 0:
            rolling_trend = rolling_trend - 1 if rolling_trend <= 0 else -1
        else:
            rolling_trend = 0

        # --- 2. CALCULATE LIVE VELOCITY & ACCEL ---
        # Velocity = (Live Rank+) - (Last Run Rank+)
        prev_rank_plus = prev_data.get('rank_plus', 0)
        velocity = int(current_rank_plus - prev_rank_plus)
        
        # Upvote Change = (Live Upvotes) - (Last Run Upvotes)
        prev_upvotes = prev_data.get('upvotes', 0)
        upv_chg = int(current_upvotes - prev_upvotes)
        
        # Acceleration = (Current Velocity) - (Previous Velocity)
        accel = 0
        if len(dates) >= 2:
            prev_2_data = self.data[ticker][dates[-2]]
            prev_vel = int(prev_rank_plus - prev_2_data.get('rank_plus', 0))
            accel = velocity - prev_vel

        return {"vel": velocity, "accel": accel, "upv_chg": upv_chg, "div": False, "streak": len(dates) + 1, "rolling_trend": rolling_trend}

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
                meta = info.get('industry', 'Unknown') # Simplified
                meta = meta.replace('\r', '').replace('\n', '').strip()
                if not meta or meta == name or meta == "Unknown - Unknown":
                    meta = "Unknown"
    except: pass
    return {'ticker': ticker, 'name': name, 'meta': meta, 'type': quote_type, 'mcap': mcap, 'currency': currency}

def filter_and_process(stocks):
    if not stocks: return pd.DataFrame()
    
    us_tickers = list(set([TICKER_FIXES.get(s['ticker'], s['ticker'].replace('.', '-')) for s in stocks]))
    local_cache = load_cache()
    tracker = HistoryTracker(HISTORY_FILE)
    
    # Clean Cache of old delisted
    now = datetime.datetime.utcnow()
    valid_tickers = [t for t in us_tickers if not local_cache.get(t, {}).get('delisted')]
    
    # Fetch Missing Metadata
    missing = [t for t in valid_tickers if t not in local_cache]
    if missing:
        print(f"Fetching metadata for {len(missing)} items...")
        for t in missing:
            res = fetch_meta_data_robust(t)
            if res: local_cache[res['ticker']] = res
        save_cache(local_cache)

    # Download Market Data
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

    final_list = []
    for stock in stocks:
        t = TICKER_FIXES.get(stock['ticker'], stock['ticker'].replace('.', '-'))
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                if t not in market_data.columns.levels[0]:
                    local_cache[t] = {'delisted': True, 'last_checked': now.strftime("%Y-%m-%d")}
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
            
            try: mcap = float(info.get('mcap', 0) or 0)
            except: mcap = 0
            
            log_mcap = math.log(mcap if mcap > 0 else 10**9, 10)
            squeeze_score = (cur_m * s_perc) / max(log_mcap, 1)

            rank_now = int(stock.get('rank', 0))
            rank_old = int(stock.get('rank_24h_ago', 0))
            rank_plus = (rank_old - rank_now) if rank_old != 0 else 0

            conviction = (int(stock.get('upvotes', 0)) / cur_m) if cur_m > 0 else 0
            safe_surge = s_perc if s_perc > 0 else 1
            efficiency = rank_plus / safe_surge

            current_upvotes = int(stock.get('upvotes', 0))
            m = tracker.get_metrics(t, float(curr_p), m_perc, rank_plus, current_upvotes)

            final_list.append({
                "Rank": rank_now, "Name": name, "Sym": t, "Rank+": rank_plus,
                "Price": float(curr_p), "AvgVol": int(avg_v),
                "Surge": s_perc, "Mnt%": m_perc, "Type": info.get('type', 'EQUITY'),
                "Upvotes": int(stock.get('upvotes', 0)), "Meta": info.get('meta', '-'),
                "Squeeze": squeeze_score, "MCap": mcap, "Conv": conviction, "Eff": efficiency,
                "Accel": m['accel'], "Upv+": m['upv_chg'], "Velocity": m['vel'],
                "Streak": m['streak'], "Rolling": m['rolling_trend']
            })
        except Exception as e: continue
    
    # Scoring
    df = pd.DataFrame(final_list)
    if not df.empty and 'Sym' in df.columns:
        df = df.drop_duplicates(subset=['Sym'], keep='first')

    if not df.empty:
        cols = ['Rank+', 'Surge', 'Mnt%', 'Upvotes', 'Accel', 'Upv+']
        weights = {'Rank+': 1.1, 'Surge': 1.1, 'Mnt%': 0.7, 'Upvotes': 1.0, 'Accel': 1.2, 'Upv+': 1.0}
        
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

def export_interactive_html(df, ai_summary=""):
    try:
        # --- 1. PREPARE THE DATA ---
        export_df = df.copy()
        if not os.path.exists(PUBLIC_DIR): os.makedirs(PUBLIC_DIR)
        
        # Ensure necessary columns exist before we process them
        for c in ['Accel', 'Velocity', 'Rolling', 'Squeeze', 'Upvotes', 'Rank+', 'Surge', 'Mnt%', 'Master_Score', 'z_Upvotes', 'z_Surge', 'z_Squeeze']:
            if c not in export_df.columns: export_df[c] = 0

        # RENAME COLUMNS TO MATCH DISPLAY LOGIC
        # This fixes the issue where abbreviated columns were missing
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

        # Define Colors
        C_GREEN, C_YELLOW, C_RED, C_CYAN, C_WHITE = "#00ff00", "#ffff00", "#ff4444", "#00ffff", "#ffffff"
        
        if 'AvgVol' not in export_df.columns: export_df['AvgVol'] = 0
        export_df['Vol_Display'] = export_df['AvgVol'].apply(format_vol)
        export_df['Type_Tag'] = 'STOCK'

        # --- ROW-BY-ROW FORMATTING ---
        for index, row in export_df.iterrows():
            # Velocity
            v_val = row.get('Vel', 0)
            v_color = C_GREEN if v_val > 0 else (C_RED if v_val < 0 else "#666")
            export_df.at[index, 'Vel'] = color_span(f"{v_val:+d}", v_color)
            
            # Accel (Acc)
            ac_val = row.get('Acc', 0)
            if ac_val >= 5: ac_clr = "#ff00ff"
            elif ac_val > 0: ac_clr = "#00ffff"
            elif ac_val < 0: ac_clr = "#ff4444"
            else: ac_clr = "#ffffff"
            export_df.at[index, 'Acc'] = color_span(f"{ac_val:+d}", ac_clr)

            # Efficiency
            eff_val = row.get('Eff', 0)
            if eff_val >= 1.0: eff_clr = "#00ff00"
            elif eff_val >= 0.5: eff_clr = "#ffff00"
            elif eff_val < 0.1 and eff_val > -0.1: eff_clr = "#666"
            else: eff_clr = "#ff4444"
            export_df.at[index, 'Eff'] = color_span(f"{eff_val:.1f}", eff_clr)

            # Conviction
            conv_val = row.get('Conv', 0)
            conv_clr = "#ffcc00" if conv_val > 1.0 else "#ffffff"
            export_df.at[index, 'Conv'] = color_span(f"{conv_val:.1f}x", conv_clr)

            # Upvote Change
            upchg_val = row.get('Upv+', 0)
            upchg_clr = C_GREEN if upchg_val > 0 else (C_RED if upchg_val < 0 else "#666")
            export_df.at[index, 'Upv+'] = color_span(f"{upchg_val:+d}", upchg_clr)

            # Streak (Strk)
            trend_val = row.get('Strk', 0)
            sig_text = f"{trend_val:+d}"
            if trend_val >= 3: sig_color = "#00ff00"
            elif trend_val > 0: sig_color = "#99ff99"
            elif trend_val <= -2: sig_color = "#ff4444"
            else: sig_color = "#ffffff"
            export_df.at[index, 'Strk'] = color_span(sig_text, sig_color)

            # Heat Score (Calculated from Master_Score)
            score = row.get('Master_Score', 0)
            if score > 10: h_clr = "#ff0000"
            elif score > 5: h_clr = "#ff8800"
            elif score > 2: h_clr = "#ffff00"
            else: h_clr = "#888888"
            export_df.at[index, 'Heat'] = f'<span style="color:{h_clr}; font-weight:bold;">{score:.1f}</span>'
            
            # Name
            export_df.at[index, 'Name'] = f"<b>{row.get('Name','')}</b>"

            # Rank+
            r_val = row.get('Rank+', 0)
            if r_val != 0:
                r_color = C_GREEN if r_val > 0 else C_RED
                r_arrow = "▲" if r_val > 0 else "▼"
                export_df.at[index, 'Rank+'] = color_span(f"{r_val} {r_arrow}", r_color)
            else:
                export_df.at[index, 'Rank+'] = ""

            # Surge & Mnt% Colors (Uses z-scores for coloring)
            z_cols = [('Srg', 'z_Surge'), ('Mnt%', 'z_Mnt%')]
            for col, z_col in z_cols:
                val = f"{export_df.at[index, col]:.0f}%"
                z_val = row.get(z_col, 0)
                clr = C_YELLOW if z_val >= 2.0 else (C_GREEN if z_val >= 1.0 else C_WHITE)
                export_df.at[index, col] = color_span(val, clr)
            
            # Squeeze (Sqz)
            sq_z = row.get('z_Squeeze', 0)
            sq_color = C_CYAN if sq_z > 1.5 else C_WHITE
            export_df.at[index, 'Sqz'] = color_span(int(row.get('Sqz', 0)), sq_color)
            
            # Upvotes (Upvs)
            z_up = row.get('z_Upvotes', 0)
            export_df.at[index, 'Upvs'] = color_span(row.get('Upvs', 0), C_GREEN if z_up > 1.5 else C_WHITE)
            
            # ETF Badge & Meta
            is_fund = row.get('Type', 'EQUITY') == 'ETF' or 'Trust' in str(row['Name']) or 'Fund' in str(row['Name'])
            meta_val = row.get('Meta', '-')
            if is_fund:
                badge = '<span style="background-color:#ff00ff; color:black; padding:2px 5px; border-radius:4px; font-size:11px; font-weight:bold; margin-right:6px; vertical-align:middle;">ETF</span>'
                meta_text = color_span(meta_val, C_WHITE)
            else:
                badge = ""
                meta_text = color_span(meta_val, C_WHITE)

            export_df.at[index, 'Meta'] = f"{badge}{meta_text}"
            export_df.at[index, 'Type_Tag'] = 'ETF' if is_fund else 'STOCK'
            
            # Sym Link & Price
            t = row['Sym']
            export_df.at[index, 'Sym'] = f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color: #4da6ff; text-decoration: none;">{t}</a>'
            export_df.at[index, 'Price'] = f"${row.get('Price', 0):.2f}"
            export_df.at[index, 'Vol_Display'] = color_span(export_df.at[index, 'Vol_Display'], "#ccc")

        # Rename Meta to final header
        export_df.rename(columns={'Meta': 'INDUSTRY/SECTOR', 'Vol_Display': 'Vol'}, inplace=True)

        # DEFINE EXACT COLUMN ORDER (21 Columns)
        # Javascript DataTables relies on this exact index order
        cols = [
            'Rank', 'Rank+', 'Heat', 'Name', 'Sym', 'Price', 'Acc', 'Eff', 'Conv', 'Upvs', 
            'Upv+', 'Vol', 'Srg', 'Vel', 'Strk', 'Mnt%', 'Sqz', 'INDUSTRY/SECTOR', 
            'Type_Tag', 'AvgVol', 'MCap'
        ]
        
        # Safety fill
        for c in cols:
            if c not in export_df.columns: export_df[c] = 0

        final_df = export_df[cols]
        
        # NOTE: table-dark class + table-hover
        table_html = final_df.to_html(classes='table table-dark table-hover', index=False, escape=False)
        utc_timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        ai_box_html = ""
        if ai_summary:
            ai_box_html = f"""
            <div class="ai-box-wrapper" style="margin-bottom: 15px;">
                <div style="background: #18181b; border: 1px solid #00ff00; border-radius: 5px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,255,0,0.1);">
                    <div onclick="toggleAI()" style="padding: 8px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; background: #2a2a2a; border-bottom: 1px solid #333; font-weight: bold; color: #fff; font-size: 0.85rem; text-transform: uppercase;">
                        <span style="color: #00ff00;">✨ Gemini AI Intelligence Report (CLICK TO TOGGLE)</span>
                        <span id="aiArrow" style="color: #00ff00;">▼</span>
                    </div>
                    <div id="aiContent" style="display: none; padding: 15px; color: #e0e0e0; font-size: 14px; line-height: 1.6; white-space: pre-wrap; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">{ai_summary}</div>
                </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Ape Wisdom Analysis</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <style>
            body{{ background-color:#101010; color:#e0e0e0; font-family:'Consolas','Monaco',monospace; padding:20px; }}
            .master-container {{ max-width: 1600px; margin: 0 auto; width: 100%; }}
            .table-dark{{--bs-table-bg:#18181b;color:#ccc}}
            th{{ color:#00ff00; border-bottom:2px solid #444; font-size: 15px; text-transform: uppercase; vertical-align: middle !important; padding: 8px 22px 8px 6px !important; line-height: 1.2 !important; }}
            td{{ vertical-align:middle; white-space: nowrap; border-bottom:1px solid #333; padding: 4px 5px !important; font-size: 15px; }}
            table.dataTable {{ width: 100% !important; margin: 0 auto; }}
            
            /* Column Widths */
            th:nth-child(1), td:nth-child(1) {{ width: 1%; text-align: center; }} 
            th:nth-child(2), td:nth-child(2) {{ width: 1%; text-align: center; }}
            th:nth-child(3), td:nth-child(3) {{ width: 1%; text-align: center; font-weight: bold; }}
            th:nth-child(4), td:nth-child(4) {{ max-width: 260px; overflow: hidden; text-overflow: ellipsis; }}
            th:nth-child(5), td:nth-child(5) {{ width: 1%; text-align: left; }}
            th:nth-child(6), td:nth-child(6) {{ width: 1%; text-align: right; }}
            th:nth-child(18), td:nth-child(18) {{ max-width: 260px; overflow: hidden; text-overflow: ellipsis; text-align: left; padding-left: 10px !important; border-right: 1px solid #333; }}
            
            a{{color:#4da6ff; text-decoration:none;}} a:hover{{text-decoration:underline;}}
            table.no-colors span {{ color: #ddd !important; font-weight: normal !important; }}
            table.no-colors a {{ color: #4da6ff !important; }}
            
            .legend-container {{ background-color: #222; border: 1px solid #444; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }}
            .legend-header {{ background: #2a2a2a; padding: 8px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: bold; color: #fff; font-size: 0.85rem; text-transform: uppercase; border-bottom: 1px solid #333; }}
            .legend-box {{ padding: 8px; display: none; background-color: #1a1a1a; }}
            .legend-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; width: 100%; }}
            .legend-col {{ background: #222; border: 1px solid #333; padding: 6px; border-radius: 5px; }}
            .legend-title {{ color: #00ff00; font-weight: bold; border-bottom: 1px solid #444; margin-bottom: 8px; font-size: 0.95rem; text-transform: uppercase; }}
            .legend-row {{ display: flex; align-items: flex-start; margin-bottom: 1px; font-size: 15px; border-bottom: 1px dashed #333; padding-bottom: 1px; }}
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
            .dataTables_filter input {{ width: 350px !important; background: #111 !important; border: 1px solid #555 !important; color: #fff !important; height: 28px !important; border-radius: 4px; margin-left: 10px; }}
            /* --- PAGINATION DARK MODE FIX --- */
            .page-link {{ background-color: #1a1a1a !important; border-color: #444 !important; color: #ccc !important; }}
            .page-link:hover {{ background-color: #333 !important; color: #fff !important; }}
            .page-item.active .page-link {{ background-color: #00ff00 !important; border-color: #00ff00 !important; color: #000 !important; font-weight: bold; }}
            .page-item.disabled .page-link {{ background-color: #111 !important; border-color: #333 !important; color: #555 !important; }}
        </style>
        </head>
        <body>
        <div class="master-container">
            {ai_box_html}
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
                            <div class="legend-row"><span class="metric-name">RANK</span><span class="metric-math">Current Pos</span><span class="metric-desc">Current rank in the popularity list.</span></div>
                            <div class="legend-row"><span class="metric-name">RANK+</span><span class="metric-math">Rank(Yest) - Rank(Today)</span><span class="metric-desc">Positions changed vs 24h ago.</span></div>
                            <div class="legend-row"><span class="metric-name">HEAT</span><span class="metric-math">Master Score</span><span class="metric-desc">Weighted aggregate of all momentum signals.</span></div>
                            <div class="legend-row"><span class="metric-name">ACC</span><span class="metric-math">Vel(Today) - Vel(Yest)</span><span class="metric-desc">Acceleration: (Rate of change of speed).</span></div>
                            <div class="legend-row"><span class="metric-name">EFF</span><span class="metric-math">Rank+ / Surge</span><span class="metric-desc">Efficiency: Rank gain per unit of volume.</span></div>
                            <div class="legend-row"><span class="metric-name">CONV</span><span class="metric-math">Upvotes / Mentions</span><span class="metric-desc">Conviction: Sentiment Quality Ratio.</span></div>
                            <div class="legend-row"><span class="metric-name">UPVS</span><span class="metric-math">Raw Count</span><span class="metric-desc">Total upvotes in last 24h.</span></div>
                            <div class="legend-row"><span class="metric-name">UPV+</span><span class="metric-math">Upv(Today) - Upv(Yest)</span><span class="metric-desc">Net change in upvotes vs 24h ago.</span></div>
                            <div class="legend-row"><span class="metric-name">VOL</span><span class="metric-math">30-Day Mean</span><span class="metric-desc">Average daily trading volume.</span></div>
                            <div class="legend-row"><span class="metric-name">SRG</span><span class="metric-math">(Vol / Avg) * 100</span><span class="metric-desc">Surge: Current volume as % of 30-day Avg.</span></div>
                            <div class="legend-row"><span class="metric-name">VEL</span><span class="metric-math">Rank+(Today) - Rank+(Yest)</span><span class="metric-desc">Velocity: Change in climb speed?</span></div>
                            <div class="legend-row"><span class="metric-name">STRK</span><span class="metric-math">Consecutive Days</span><span class="metric-desc">Streak: Number of script runs sustaining direction.</span></div>
                            <div class="legend-row"><span class="metric-name">MNT%</span><span class="metric-math">% Change</span><span class="metric-desc">Percent change in mentions vs 24h ago.</span></div>
                            <div class="legend-row"><span class="metric-name">SQZ</span><span class="metric-math">Mnt * Surge / log(MCap)</span><span class="metric-desc">Short Squeeze Score (Vol+Chatter/Cap).</span></div>
                        </div>
                        <div class="legend-col">
                            <div class="legend-title">🎨 Color Indicators</div>
                            <div class="legend-row"><span class="color-key">RANK</span><span class="color-desc">White (Standard).</span></div>
                            <div class="legend-row"><span class="color-key">RANK+</span><span class="color-desc"><span style="color:#00ff00">Green</span> (Climbing), <span style="color:#ff4444">Red</span> (Falling).</span></div>
                            <div class="legend-row"><span class="color-key">HEAT</span><span class="color-desc"><span style="color:#ff0000">Red</span> (> 2.0σ), <span style="color:#ff8800">Orange</span> (> 1.5σ), <span style="color:#ffff00">Yellow</span> (> 1σ).</span></div>
                            <div class="legend-row"><span class="color-key">ACC</span><span class="color-desc"><span style="color:#ff00ff">Magenta</span> (Expl. ≥5), <span style="color:#00ffff">Cyan</span> (Fast >0), <span style="color:#ffffff">White</span> (Steady 0), <span style="color:#ff4444">Red</span> (Slow <0).</span></div>
                            <div class="legend-row"><span class="color-key">EFF</span><span class="color-desc"><span style="color:#00ff00">Green</span> (> 1.0), <span style="color:#ffff00">Yellow</span> (> 0.5), <span style="color:#ff4444">Red</span> (Low).</span></div>
                            <div class="legend-row"><span class="color-key">CONV</span><span class="color-desc"><span style="color:#ffcc00">Gold</span> (> 1.0x), White (Diluted).</span></div>
                            <div class="legend-row"><span class="color-key">UPVS</span><span class="color-desc"><span style="color:#00ff00">Green</span> (High Activity > 1.5σ).</span></div>
                            <div class="legend-row"><span class="color-key">UPV+</span><span class="color-desc"><span style="color:#00ff00">Green</span> (Positive), <span style="color:#ff4444">Red</span> (Negative).</span></div>
                            <div class="legend-row"><span class="color-key">VOL</span><span class="color-desc"><span style="color:#ccc">Gray</span> (Static Stat).</span></div>
                            <div class="legend-row"><span class="color-key">SRG</span><span class="color-desc"><span style="color:#ffff00">Yellow</span> (Anomaly > 2σ), <span style="color:#00ff00">Green</span> (High > 1σ).</span></div>
                            <div class="legend-row"><span class="color-key">VEL</span><span class="color-desc"><span style="color:#00ff00">Green</span> (Speeding Up), <span style="color:#ff4444">Red</span> (Slowing).</span></div>
                            <div class="legend-row"><span class="color-key">STRK</span><span class="color-desc"><span style="color:#00ff00">Green</span> (3+ Runs.), <span style="color:#ff4444">Red</span> (Reversing).</span></div>
                            <div class="legend-row"><span class="color-key">MNT%</span><span class="color-desc"><span style="color:#ffff00">Yellow</span> (> 2σ), <span style="color:#00ff00">Green</span> (> 1σ).</span></div>
                            <div class="legend-row"><span class="color-key">SQZ</span><span class="color-desc"><span style="color:#00ffff">Cyan</span> (Score > 1.5σ), White (Normal).</span></div>
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
            if (type === 'all') {{ $('input[name="mcapFilter"]').not('#mcapAll').prop('checked', false); }} 
            else {{ $('#mcapAll').prop('checked', false); if ($('input[name="mcapFilter"]:checked').length === 0) {{ $('#mcapAll').prop('checked', true); }} }}
            table.draw();
        }}
        function toggleLegend() {{
            var x = document.getElementById("legendContent"); var arrow = document.getElementById("legendArrow");
            if (x.style.display === "block") {{ x.style.display = "none"; arrow.innerText = "▼"; }} else {{ x.style.display = "block"; arrow.innerText = "▲"; }}
        }}
        function toggleAI() {{
            var x = document.getElementById("aiContent"); var arrow = document.getElementById("aiArrow");
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
            if (str.endsWith('k')) {{ mult = 1000; str = str.replace('k', ''); }} 
            else if (str.endsWith('m')) {{ mult = 1000000; str = str.replace('m', ''); }} 
            else if (str.endsWith('b')) {{ mult = 1000000000; str = str.replace('b', ''); }}
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
                    {{ "targets": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "type": "num", "render": function(data, type) {{ if (type === 'sort' || type === 'type') {{ var clean = data.toString().replace(/<[^>]+>/g, '').replace(/[$,%+,x]/g, ''); return parseVal(clean); }} return data; }} }}
                ],
                "drawCallback": function() {{ var api = this.api(); $("#stockCounter").text("Showing " + api.rows({{filter:'applied'}}).count() + " / " + api.rows().count() + " Tickers"); }}
            }});
            
            $.fn.dataTable.ext.search.push(function(settings, data) {{
                var typeTag = data[18] || ""; 
                var viewMode = $('input[name="btnradio"]:checked').attr('id');
                if (viewMode == 'btnradio2' && typeTag == 'ETF') return false;
                if (viewMode == 'btnradio3' && typeTag == 'STOCK') return false;
                
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

                var minP = parseVal($('#minPrice').val()), maxP = parseVal($('#maxPrice').val());
                var p = parseFloat((data[5] || "0").replace(/[$,]/g, '')) || 0; 
                if (minP > 0 && p < minP) return false;
                if (maxP > 0 && p > maxP) return false;
                
                var minV = parseVal($('#minVol').val()), maxV = parseVal($('#maxVol').val());
                var v = parseFloat(data[19]) || 0; 
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

def get_ai_analysis(df, history_data):
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return "AI analysis skipped: No API key."

    try:
        genai.configure(api_key=api_key)
        # --- SMART MODEL SELECTOR ---
        target_model = None
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name.lower() and 'gemini' in m.name.lower():
                        target_model = m.name; break
            if not target_model: target_model = 'gemini-pro'
            print(f"🤖 Connected to AI Model: {target_model}")
            model = genai.GenerativeModel(target_model)
        except:
            model = genai.GenerativeModel('gemini-pro')

        # --- DATA PREPARATION ---
        ai_df = df.reset_index(drop=True).copy()
        ai_df = ai_df.loc[:, ~ai_df.columns.duplicated()]
        
        # Standardize Names for AI
        mapping = {
            'Surge': 'Srg', 'Master_Score': 'Heat', 'Accel': 'Acc', 
            'Velocity': 'Vel', 'Rolling': 'Strk', 'Upvotes': 'Upvs'
        }
        ai_df.rename(columns=mapping, inplace=True)
        
        # --- GOD MODE: ADDED 'Eff' (Efficiency) FOR RULE #5 ---
        cols_for_ai = ['Sym', 'Rank', 'Price', 'Srg', 'Vel', 'Acc', 'Strk', 'Upv+', 'Eff', 'Heat']
        
        # Safety fill for missing cols
        for c in cols_for_ai:
            if c not in ai_df.columns: ai_df[c] = 0

        # DATASET 1: LEADERS
        leaders_df = ai_df.head(10)[cols_for_ai]
        leaders_str = leaders_df.to_string(index=False)

        # DATASET 2: STEALTH
        rest_of_market = ai_df.iloc[10:].copy()
        stealth_candidates = rest_of_market[ 
            (rest_of_market['Srg'] > 100) | (rest_of_market['Vel'] >= 2) 
        ].sort_values(by='Heat', ascending=False).head(5)
        stealth_str = stealth_candidates[cols_for_ai].to_string(index=False) if not stealth_candidates.empty else "None"

        # DATASET 3: 3-HOUR FORENSIC HISTORY
        top_20 = df.head(20)['Sym'].tolist()
        comparison_context = []
        for ticker in top_20:
            if ticker in history_data:
                snapshots = sorted(history_data[ticker].items())
                if len(snapshots) > 1:
                    curr = snapshots[-1][1]
                    prev = snapshots[-6][1] if len(snapshots) > 6 else snapshots[0][1]
                    
                    p_now = curr.get('price', 0)
                    p_old = prev.get('price', 0)
                    price_delta = round(((p_now - p_old) / p_old * 100), 2) if p_old > 0 else 0
                    
                    comparison_context.append({
                        "Sym": ticker,
                        "Rank_Chg": prev.get('rank', 0) - curr.get('rank', 0), # Positive = Climbing
                        "Price_%": f"{price_delta}%",
                        "Conv_Chg": round(curr.get('conv', 0) - prev.get('conv', 0), 2),
                        "Upv_Delta": int(curr.get('upvotes', 0) - prev.get('upvotes', 0))
                    })

        # --- THE 5-RULE PROMPT ---
        prompt = f"""
        You are a Senior Technical Analyst for a quant trading desk. Analyze the market data provided below to produce a "Morning Momentum Report."

        ### DATASET 1: LEADERBOARD (Top 10)
        {leaders_str}

        ### DATASET 2: DEEP SCAN (Stealth/Squeeze Candidates)
        {stealth_str}

        ### DATASET 3: 3-HOUR DELTAS (Trend Changes)
        {comparison_context}

        ### REPORT GUIDELINES:
        1. **Tone:** Professional, punchy, and narrative-driven. No robotic lists.
        2. **Formatting:** Use Markdown tables, **bold** for tickers, and emojis for section headers.

        ### REQUIRED OUTPUT SECTIONS:

        **1. 🌍 Executive Summary**
        * Provide a 2-sentence summary of the market's current mood. (e.g., "Significant rotation into Semiconductors, while Gold cools off.")

        **2. 🔝 Top Market Leaders (Table)**
        * Create a Markdown table with columns: **Symbol**, **Price**, **Sector**, and **Analyst Note**.
        * Select the top 4 strongest stocks from DATASET 1 based on 'Heat' and 'Rank'.

        **3. 🔥 High-Volatility & Squeeze Alerts**
        * Analyze DATASET 2. Identify stocks with high 'Srg' (Surge) or 'Sqz' (Squeeze) scores.
        * Format as bullet points: "**Ticker (Price):** Description of the move."
        * *Example:* "**CVNA ($45.20):** Massive 442% squeeze score indicates forced buy-ins."

        **4. 🏗️ Sector Strength Analysis**
        * Look at the 'Industry/Sector' of the top performers. Group them.
        * Identify the top 2 hot sectors. (e.g., "1. Technology", "2. Energy").

        **5. 📉 Notable Laggards / Divergences**
        * Look at DATASET 3. Identify any stock dropping in Rank or Price, OR any stock where Price is dropping but Rank is rising (Divergence).

        Produce the report now.
        """

        response = model.generate_content(prompt)
        clean_text = response.text.replace("Here's your analysis:", "").replace("Here is the summary:", "").strip()
        return clean_text

    except Exception as e:
        print(f"AI Error: {e}")
        return f"AI Analysis Failed: {str(e)}"

if __name__ == "__main__":
    if "--auto" in sys.argv:
        print("Starting Auto Scan...")
    
    # 1. Fetch Data
    raw = get_all_trending_stocks()
    if not raw:
        print(f"{C_RED}[!] No data returned from ApeWisdom API. Exiting.{C_RESET}")
        sys.exit(0)

    # --- SAFETY LIMIT: Only process top 500 to avoid Yahoo 429 Errors ---
    raw = raw[:500] 

    # 2. Process Data
    df = filter_and_process(raw)
    if df.empty:
        print(f"{C_RED}[!] Data fetched, but all tickers were filtered out. Exiting.{C_RESET}")
        sys.exit(0)

    # 3. Generate the AI Summary
    print("--- Generating AI Analysis ---")
    tracker = HistoryTracker(HISTORY_FILE)
    ai_summary = get_ai_analysis(df, tracker.data)

    # 4. Generate HTML
    fname = export_interactive_html(df, ai_summary)
    
    # 5. Send to Discord
    if fname:
        send_discord_link(fname, "")
    else:
        print(f"{C_RED}[!] HTML generation failed. Skipping Discord.{C_RESET}")
    
    print("Done.")

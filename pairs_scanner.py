import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import json
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
DATA_FILE        = "historical_data.csv.gz"
CHART_DATA_FILE  = "chart_data.csv.gz"        # Extended history for Z-score charts only
VOLUME_DATA_FILE = "volume_data.csv.gz"        # Average daily volume per ticker
BATCH_SIZE = 25
COOLDOWN = 1
LOOKBACK_DAYS       = 650   # Days used for scoring / correlation / perf
CHART_LOOKBACK_DAYS = 1825  # ~5 years used for Z-score chart history
VOL_AVG_DAYS        = 30    # Rolling window for average volume calculation
CACHE_UPDATE_COOLDOWN_HOURS = 4

CORR_SHORT = 35
CORR_LONG = 100
Z_LENGTH = 100
PERF_LENGTH = 300

MIN_CORR_FILTER = 0.60
Z_THRESHOLD = 1.0
Z_STRONG = 2.0

W_ZSCORE = 0.50
W_CORR_BRK = 0.25
W_REL_PERF = 0.25

# ==========================================
# LOAD MASTER TICKERS
# ==========================================
TICKER_TYPES = {}
TICKER_NAMES = {}   # maps ticker -> human-readable name


def load_master_tickers():
    global TICKER_TYPES, TICKER_NAMES
    tickers = []

    if os.path.exists("ALL ETFs.csv"):
        df_etf = pd.read_csv("ALL ETFs.csv", header=None)
        etfs = df_etf[0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers += etfs
        for t in etfs:
            TICKER_TYPES[t] = "Pure ETF"
        if df_etf.shape[1] >= 2:
            for _, row in df_etf.iterrows():
                t = str(row.iloc[0]).strip().upper()
                TICKER_NAMES[t] = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""

    if os.path.exists("ALL STOCKS.csv"):
        df_stock = pd.read_csv("ALL STOCKS.csv", header=None)
        stocks = df_stock[0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers += stocks
        for t in stocks:
            TICKER_TYPES[t] = "Pure Stock"
        if df_stock.shape[1] >= 2:
            for _, row in df_stock.iterrows():
                t = str(row.iloc[0]).strip().upper()
                TICKER_NAMES[t] = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""

    tickers = list(set(tickers))
    tickers = [t for t in tickers if t not in ["", "NONE", "NAN", "SYMBOL", "TICKER"]]
    print(f"Loaded {len(tickers)} tickers.")
    return tickers


# ==========================================
# SAFE SAVE
# ==========================================
def safe_save(df):
    tmp = DATA_FILE + ".tmp"
    df.to_csv(tmp)
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    os.rename(tmp, DATA_FILE)


# ==========================================
# LOAD CACHE
# ==========================================
def load_cache():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
            df = df.loc[:, ~df.columns.duplicated()]
            print("Cache loaded.")
            return df
        except:
            pass
    return pd.DataFrame()


# ==========================================
# DOWNLOAD BATCH  (Close prices, or Volume)
# ==========================================
def download_batch(tickers, start_date, field="Close"):
    clean = [t.replace("/", "-") for t in tickers]
    max_retries = 3

    for attempt in range(max_retries):
        try:
            df = yf.download(
                clean,
                start=start_date,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                timeout=20
            )

            if df is None or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return pd.DataFrame()

            result = pd.DataFrame()
            for t in clean:
                try:
                    if len(clean) > 1:
                        if t in df.columns.levels[0]:
                            result[t] = df[t][field]
                    else:
                        if not df[field].empty:
                            result[t] = df[field]
                except:
                    continue

            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt+1} failed, retrying...")
                time.sleep(5)
            else:
                print(f"  Batch failed after {max_retries} attempts: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


# ==========================================
# BUILD DATASET
# ==========================================
def build_dataset(master):
    data = load_cache()

    if os.path.exists(DATA_FILE) and not data.empty:
        file_time = os.path.getmtime(DATA_FILE)
        last_update = datetime.fromtimestamp(file_time)
        hours_since_update = (datetime.now() - last_update).total_seconds() / 3600

        if hours_since_update < CACHE_UPDATE_COOLDOWN_HOURS:
            print(f"--- Cache is fresh ({round(hours_since_update, 2)}h old). Skipping download. ---")
            data = data[[c for c in data.columns if c in master]]
            return data

    existing = data.columns.tolist() if not data.empty else []
    missing = [t for t in master if t not in existing]

    if missing:
        total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Backfilling {len(missing)} tickers in {total_batches} batches...")
        start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

        for i, idx in enumerate(range(0, len(missing), BATCH_SIZE)):
            batch_num = i + 1
            batch = missing[idx: idx + BATCH_SIZE]
            print(f"[{batch_num}/{total_batches}] Downloading: {batch[0]}...{batch[-1] if len(batch) > 1 else ''}")
            batch_df = download_batch(batch, start)
            if not batch_df.empty:
                data = pd.concat([data, batch_df], axis=1)
                data = data.loc[:, ~data.columns.duplicated()]
                safe_save(data)
            time.sleep(COOLDOWN)

    if not data.empty:
        last_date = data.index.max()
        today = datetime.now().date()

        if last_date.date() < today:
            start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            tickers_to_update = data.columns.tolist()
            total_update_batches = (len(tickers_to_update) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Updating {len(tickers_to_update)} tickers ({total_update_batches} batches) starting from {start}...")

            for i, idx in enumerate(range(0, len(tickers_to_update), BATCH_SIZE)):
                batch_num = i + 1
                batch = tickers_to_update[idx: idx + BATCH_SIZE]
                print(f"[{batch_num}/{total_update_batches}] Updating: {batch[0]}...")
                batch_df = download_batch(batch, start)
                if not batch_df.empty:
                    data = pd.concat([data, batch_df], axis=0)
                    data = data[~data.index.duplicated(keep="last")].sort_index()
                    safe_save(data)
                time.sleep(COOLDOWN)

    data = data[[c for c in data.columns if c in master]]
    data = data.tail(LOOKBACK_DAYS)
    data = data.ffill().bfill()
    data = data.dropna(axis=1, thresh=len(data) * 0.2)
    safe_save(data)

    print(f"Final dataset ready: {len(data.columns)} tickers.")
    return data


# ==========================================
# BUILD EXTENDED CHART DATASET  (~5 years)
# ==========================================
def build_chart_dataset(master):
    """Downloads ~5 years of Close data for Z-score chart history.
    Uses its own cache file so it does not interfere with scoring data."""
    if os.path.exists(CHART_DATA_FILE):
        try:
            chart_data = pd.read_csv(CHART_DATA_FILE, index_col=0, parse_dates=True)
            chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()]
            file_time  = os.path.getmtime(CHART_DATA_FILE)
            hours_old  = (datetime.now() - datetime.fromtimestamp(file_time)).total_seconds() / 3600
            if hours_old < CACHE_UPDATE_COOLDOWN_HOURS:
                print(f"--- Chart cache fresh ({round(hours_old,2)}h). Skipping chart download. ---")
                chart_data = chart_data[[c for c in chart_data.columns if c in master]]
                return chart_data
        except:
            chart_data = pd.DataFrame()
    else:
        chart_data = pd.DataFrame()

    existing = chart_data.columns.tolist() if not chart_data.empty else []
    missing  = [t for t in master if t not in existing]
    start    = (datetime.now() - timedelta(days=CHART_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    if missing:
        total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Downloading {CHART_LOOKBACK_DAYS}-day chart history for {len(missing)} tickers ({total_batches} batches)...")
        for i, idx in enumerate(range(0, len(missing), BATCH_SIZE)):
            batch = missing[idx: idx + BATCH_SIZE]
            print(f"  [{i+1}/{total_batches}] {batch[0]}...")
            batch_df = download_batch(batch, start, field="Close")
            if not batch_df.empty:
                chart_data = pd.concat([chart_data, batch_df], axis=1)
                chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()]
                tmp = CHART_DATA_FILE + ".tmp"
                chart_data.to_csv(tmp)
                if os.path.exists(CHART_DATA_FILE): os.remove(CHART_DATA_FILE)
                os.rename(tmp, CHART_DATA_FILE)
            time.sleep(COOLDOWN)

    if not chart_data.empty:
        last_date = chart_data.index.max()
        if last_date.date() < datetime.now().date():
            upd_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            tickers_to_upd = chart_data.columns.tolist()
            total_upd = (len(tickers_to_upd) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Updating chart cache ({total_upd} batches) from {upd_start}...")
            for i, idx in enumerate(range(0, len(tickers_to_upd), BATCH_SIZE)):
                batch = tickers_to_upd[idx: idx + BATCH_SIZE]
                batch_df = download_batch(batch, upd_start, field="Close")
                if not batch_df.empty:
                    chart_data = pd.concat([chart_data, batch_df], axis=0)
                    chart_data = chart_data[~chart_data.index.duplicated(keep="last")].sort_index()
                    tmp = CHART_DATA_FILE + ".tmp"
                    chart_data.to_csv(tmp)
                    if os.path.exists(CHART_DATA_FILE): os.remove(CHART_DATA_FILE)
                    os.rename(tmp, CHART_DATA_FILE)
                time.sleep(COOLDOWN)

    chart_data = chart_data[[c for c in chart_data.columns if c in master]]
    chart_data = chart_data.ffill().bfill()
    print(f"Chart dataset ready: {len(chart_data.columns)} tickers, {len(chart_data)} days.")
    return chart_data


# ==========================================
# BUILD VOLUME DATASET
# Downloads Volume field, computes VOL_AVG_DAYS rolling mean.
# ==========================================
def build_volume_dataset(master):
    """Returns a dict {ticker: avg_volume} using a VOL_AVG_DAYS rolling average."""
    vol_avg = {}

    if os.path.exists(VOLUME_DATA_FILE):
        try:
            vol_df    = pd.read_csv(VOLUME_DATA_FILE, index_col=0, parse_dates=True)
            vol_df    = vol_df.loc[:, ~vol_df.columns.duplicated()]
            file_time = os.path.getmtime(VOLUME_DATA_FILE)
            hours_old = (datetime.now() - datetime.fromtimestamp(file_time)).total_seconds() / 3600
            if hours_old < CACHE_UPDATE_COOLDOWN_HOURS:
                print(f"--- Volume cache fresh ({round(hours_old,2)}h). Using cached volume. ---")
                for col in vol_df.columns:
                    series = vol_df[col].dropna()
                    if len(series) > 0:
                        vol_avg[col] = float(series.rolling(VOL_AVG_DAYS, min_periods=1).mean().iloc[-1])
                return vol_avg
        except:
            pass

    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    total_batches = (len(master) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Downloading volume data for {len(master)} tickers ({total_batches} batches)...")

    all_vol = pd.DataFrame()
    for i, idx in enumerate(range(0, len(master), BATCH_SIZE)):
        batch    = master[idx: idx + BATCH_SIZE]
        batch_df = download_batch(batch, start, field="Volume")
        if not batch_df.empty:
            all_vol = pd.concat([all_vol, batch_df], axis=1)
        time.sleep(COOLDOWN)

    if not all_vol.empty:
        all_vol = all_vol.loc[:, ~all_vol.columns.duplicated()]
        tmp = VOLUME_DATA_FILE + ".tmp"
        all_vol.to_csv(tmp)
        if os.path.exists(VOLUME_DATA_FILE): os.remove(VOLUME_DATA_FILE)
        os.rename(tmp, VOLUME_DATA_FILE)
        for col in all_vol.columns:
            series = all_vol[col].dropna()
            if len(series) > 0:
                vol_avg[col] = float(series.rolling(VOL_AVG_DAYS, min_periods=1).mean().iloc[-1])

    print(f"Volume data ready for {len(vol_avg)} tickers.")
    return vol_avg


# ==========================================
# ANALYZE PAIR
# ==========================================
def analyze_pair(pair):
    a, b = pair

    cl = corr_long.loc[a, b]
    if cl < MIN_CORR_FILTER:
        return None

    cs = corr_short.loc[a, b]
    corr_brk = cl - cs

    # Z-score on last Z_LENGTH days for scoring
    ratio = log_prices[a] - log_prices[b]
    mean  = ratio.mean()
    std   = ratio.std()

    if std == 0:
        return None

    z  = (ratio.iloc[-1] - mean) / std
    rp = perf[a] - perf[b]

    z_norm    = min(abs(z) / 3.0, 1.0)
    corr_norm = min(max(corr_brk, 0) / 0.5, 1.0)
    perf_norm = min(abs(rp) / 10.0, 1.0)

    score = W_ZSCORE * z_norm + W_CORR_BRK * corr_norm + W_REL_PERF * perf_norm

    type_a = TICKER_TYPES.get(a, "Unknown")
    type_b = TICKER_TYPES.get(b, "Unknown")

    if type_a == "Pure ETF" and type_b == "Pure ETF":
        pair_category = "Pure ETF"
    elif type_a == "Pure Stock" and type_b == "Pure Stock":
        pair_category = "Pure Stock"
    else:
        pair_category = "Mixed"

    if any(np.isnan(v) for v in [z, cl, corr_brk, rp, score]):
        return None

    return {
        "Pair":     f"{a}/{b}",
        "Category": pair_category,
        "Z":        round(z, 2),
        "Corr":     round(cl, 2),
        "CorrBrk":  round(corr_brk, 3),
        "PerfDiff": round(rp, 2),
        "Score":    round(score, 3),
    }


# ==========================================
# ROLLING Z-SCORE HISTORY FOR CHART
# ==========================================
def compute_z_history(a, b, price_data):
    """Rolling Z-score over all available data using Z_LENGTH-day window."""
    log_a = np.log(price_data[a].dropna())
    log_b = np.log(price_data[b].dropna())
    combined = pd.DataFrame({"a": log_a, "b": log_b}).dropna()
    spread   = combined["a"] - combined["b"]

    roll_mean = spread.rolling(Z_LENGTH).mean()
    roll_std  = spread.rolling(Z_LENGTH).std()
    z_series  = (spread - roll_mean) / roll_std
    z_series  = z_series.dropna()

    dates  = [d.strftime("%Y-%m-%d") for d in z_series.index]
    values = [round(float(v), 4) if not np.isnan(v) else None for v in z_series.values]
    return dates, values


# ==========================================
# BUILD SYMBOLS PAGE
# ==========================================
def build_symbols_page(valid_tickers):
    def read_csv_meta(path):
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, header=None)
        n_cols = min(df.shape[1], 5)
        df = df.iloc[:, :n_cols]
        col_names = ["Ticker", "Name", "Sector", "Industry", "Subindustry"][:n_cols]
        df.columns = col_names
        for col in ["Sector", "Industry", "Subindustry"]:
            if col not in df.columns:
                df[col] = "Other"
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        df = df[df["Ticker"].isin(valid_tickers)]
        df = df.fillna("Other")
        return df

    df_etf   = read_csv_meta("ALL ETFs.csv")
    df_stock = read_csv_meta("ALL STOCKS.csv")

    def build_section_html(df, accent_color):
        if df.empty:
            return "<p style='color:#64748b;'>No data found.</p>"
        html = ""
        for sector, sg in df.groupby("Sector"):
            html += f'<div class="sector-block"><div class="sector-header" style="color:{accent_color};">{sector}</div>'
            for industry, ig in sg.groupby("Industry"):
                html += f'<div class="industry-block"><div class="industry-label">{industry}</div>'
                for subindustry, sub_ig in ig.groupby("Subindustry"):
                    html += f'<div class="subindustry-block"><div class="subindustry-label">&#8627; {subindustry}</div><div class="ticker-grid">'
                    for _, row in sub_ig.sort_values("Ticker").iterrows():
                        html += (
                            f'<div class="ticker-card">'
                            f'<span class="ticker-sym" style="color:{accent_color};">{row["Ticker"]}</span>'
                            f'<span class="ticker-name">{row.get("Name","")}</span>'
                            f'</div>'
                        )
                    html += "</div></div>"
                html += "</div>"
            html += "</div>"
        return html

    etf_html   = build_section_html(df_etf,   "#38bdf8")
    stock_html = build_section_html(df_stock, "#f59e0b")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Symbol Reference | Pairs Scanner</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #08090d;
    --surface: #0e1117;
    --surface2:#151821;
    --border:  #1e2535;
    --text:    #cbd5e1;
    --muted:   #64748b;
    --cyan:    #38bdf8;
    --amber:   #f59e0b;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Syne', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; font-size: 16px; }}

  /* TOPBAR */
  .topbar {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 20px 40px; display: flex; align-items: center;
    justify-content: space-between; position: sticky; top: 0; z-index: 100;
  }}
  .topbar h1 {{ font-size: 24px; font-weight: 800; letter-spacing: 0.04em; color: white; }}
  .topbar a {{
    color: var(--cyan); text-decoration: none; font-size: 15px; font-weight: 600;
    border: 1px solid var(--cyan); padding: 9px 20px; border-radius: 5px;
    transition: background 0.2s;
  }}
  .topbar a:hover {{ background: rgba(56,189,248,0.1); }}

  /* STATS BAR */
  .stats-bar {{
    background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 13px 40px; font-family: var(--mono); font-size: 14px;
    color: var(--muted); display: flex; gap: 40px; flex-wrap: wrap;
  }}
  .stats-bar span {{ color: var(--text); font-weight: 600; }}

  /* SEARCH */
  .search-bar {{
    padding: 22px 40px; background: var(--surface); border-bottom: 1px solid var(--border);
  }}
  .search-bar input {{
    background: var(--surface2); border: 1px solid var(--border); color: white;
    padding: 13px 20px; border-radius: 7px; font-family: var(--mono);
    font-size: 15px; width: 400px; outline: none; transition: border 0.2s;
  }}
  .search-bar input:focus {{ border-color: var(--cyan); }}
  .search-bar input::placeholder {{ color: var(--muted); }}

  /* COLUMNS */
  .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
  .column {{ padding: 30px 40px; border-right: 1px solid var(--border); }}
  .column:last-child {{ border-right: none; }}

  .col-header {{
    font-size: 14px; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 26px; padding-bottom: 16px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px;
  }}
  .col-header .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}

  /* SECTOR */
  .sector-block {{ margin-bottom: 34px; }}
  .sector-header {{
    font-size: 16px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 14px; padding: 10px 16px; border-radius: 6px; border-left: 4px solid;
  }}
  .sector-header[style*="38bdf8"] {{ border-left-color: #38bdf8; background: rgba(56,189,248,0.06); }}
  .sector-header[style*="f59e0b"] {{ border-left-color: #f59e0b; background: rgba(245,158,11,0.06); }}

  /* INDUSTRY */
  .industry-block {{ margin: 16px 0 16px 18px; }}
  .industry-label {{
    font-size: 13px; font-weight: 700; color: #6b7f9a; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}

  /* SUBINDUSTRY */
  .subindustry-block {{ margin: 12px 0 12px 16px; }}
  .subindustry-label {{
    font-size: 12px; color: #4a5e72; font-family: var(--mono);
    margin-bottom: 10px; letter-spacing: 0.04em;
  }}

  /* TICKER CARDS */
  .ticker-grid {{ display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 12px; }}
  .ticker-card {{
    background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
    padding: 9px 14px; display: flex; flex-direction: column; min-width: 95px;
    transition: border-color 0.15s, background 0.15s; cursor: default;
  }}
  .ticker-card:hover {{ border-color: #334155; background: #1e2535; }}
  .ticker-sym  {{ font-family: var(--mono); font-size: 15px; font-weight: 700; line-height: 1.2; }}
  .ticker-name {{
    font-size: 12px; color: #4a5568; line-height: 1.4; margin-top: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px;
  }}

  @media (max-width: 900px) {{
    .columns {{ grid-template-columns: 1fr; }}
    .column {{ border-right: none; border-bottom: 1px solid var(--border); }}
  }}
</style>
</head>
<body>

<div class="topbar">
  <h1>Symbol Reference</h1>
  <a href="market_scanner.html">&#8592; Back to Dashboard</a>
</div>

<div class="stats-bar">
  <div>Total Active: <span>{len(valid_tickers)}</span></div>
  <div>ETFs: <span>{len(df_etf)}</span></div>
  <div>Stocks: <span>{len(df_stock)}</span></div>
  <div>Generated: <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span></div>
</div>

<div class="search-bar">
  <input type="text" id="searchBox" placeholder="Search ticker or name..." oninput="filterSymbols()" />
</div>

<div class="columns">
  <div class="column">
    <div class="col-header" style="color:#38bdf8;">
      <span class="dot" style="background:#38bdf8;"></span>Exchange Traded Funds
    </div>
    <div id="etf-section">{etf_html}</div>
  </div>
  <div class="column">
    <div class="col-header" style="color:#f59e0b;">
      <span class="dot" style="background:#f59e0b;"></span>Stocks
    </div>
    <div id="stock-section">{stock_html}</div>
  </div>
</div>

<script>
function filterSymbols() {{
  const q = document.getElementById("searchBox").value.toUpperCase().trim();
  document.querySelectorAll(".ticker-card").forEach(card => {{
    const sym  = card.querySelector(".ticker-sym")?.textContent || "";
    const name = card.querySelector(".ticker-name")?.textContent || "";
    card.style.display = (sym.includes(q) || name.toUpperCase().includes(q)) ? "" : "none";
  }});
  ["subindustry-block","industry-block","sector-block"].forEach(cls => {{
    document.querySelectorAll("." + cls).forEach(block => {{
      block.style.display = [...block.querySelectorAll(".ticker-card")].some(c => c.style.display !== "none") ? "" : "none";
    }});
  }});
}}
</script>
</body>
</html>"""

    with open("symbols.html", "w", encoding="utf-8") as f:
        f.write(page)
    print("symbols.html created.")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    TICKERS = load_master_tickers()
    data = build_dataset(TICKERS)

    if data is None or data.empty:
        print("\n[!] No data available. Check internet or ticker list.")
        exit()
    if len(data) < 2:
        print(f"\n[!] Not enough history ({len(data)} days).")
        exit()

    # Build extended chart history (~5 years, separate cache)
    print("Building extended chart dataset...")
    chart_data = build_chart_dataset(TICKERS)

    # Build volume data (avg daily volume per ticker)
    print("Building volume dataset...")
    vol_avg = build_volume_dataset(TICKERS)

    valid = list(data.columns)
    print(f"Computing matrices for {len(valid)} symbols...")

    returns    = data.pct_change().dropna(how="all")
    log_prices = np.log(data.tail(Z_LENGTH))

    corr_short = returns.tail(CORR_SHORT).corr()
    corr_long  = returns.tail(CORR_LONG).corr()

    perf_len = min(PERF_LENGTH, len(data) - 1)
    perf = (data.iloc[-1] / data.iloc[-(perf_len + 1)] - 1) * 100

    print("Building combinations...")
    combos = list(itertools.combinations(valid, 2))
    print(f"Total pairs: {len(combos):,}")

    results = []
    for pair in tqdm(combos):
        r = analyze_pair(pair)
        if r:
            results.append(r)

    results     = sorted(results, key=lambda x: x["Score"], reverse=True)
    top_results = results[:200]

    # Compute rolling Z-score histories for top pairs
    print("Computing Z-score chart histories for top pairs...")
    for r in tqdm(top_results):
        a, b = r["Pair"].split("/")
        try:
            src = chart_data if (not chart_data.empty and a in chart_data.columns and b in chart_data.columns) else data
            dates, z_vals = compute_z_history(a, b, src)
            r["ZDates"]   = dates
            r["ZHistory"] = z_vals
        except Exception:
            r["ZDates"]   = []
            r["ZHistory"] = []

    # Helper: format volume as human-readable string
    def fmt_vol(v):
        if v <= 0: return "—"
        if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
        if v >= 1_000:     return f"{v/1_000:.0f}K"
        return str(int(v))

    print("Generating symbols.html...")
    build_symbols_page(valid)

    # ==========================================
    # GENERATE MAIN DASHBOARD
    # ==========================================
    print("Generating market_scanner.html...")

    rows_html = ""
    for i, r in enumerate(top_results):
        z = r["Z"]
        a, b = r["Pair"].split("/")

        if any(np.isnan(v) for v in [z, r["Score"], r["Corr"], r["PerfDiff"]]):
            continue

        name_a = TICKER_NAMES.get(a, "")
        name_b = TICKER_NAMES.get(b, "")

        if z > Z_STRONG:
            sig_label, sig_class, sig_arrow = f"SHORT {a} \u00b7 LONG {b}",  "sig-strong-short", "\u25bc\u25bc"
        elif z > Z_THRESHOLD:
            sig_label, sig_class, sig_arrow = f"SHORT {a} \u00b7 LONG {b}",  "sig-short",        "\u25bc"
        elif z < -Z_STRONG:
            sig_label, sig_class, sig_arrow = f"LONG {a} \u00b7 SHORT {b}",  "sig-strong-long",  "\u25b2\u25b2"
        elif z < -Z_THRESHOLD:
            sig_label, sig_class, sig_arrow = f"LONG {a} \u00b7 SHORT {b}",  "sig-long",         "\u25b2"
        else:
            sig_label, sig_class, sig_arrow = "NEUTRAL", "sig-neutral", "\u2014"

        cat_class = {"Pure ETF": "cat-etf", "Pure Stock": "cat-stock"}.get(r["Category"], "cat-mixed")

        price_a    = round(data[a].iloc[-1], 2)
        price_b    = round(data[b].iloc[-1], 2)
        avgvol_a   = round(vol_avg.get(a, 0))
        avgvol_b   = round(vol_avg.get(b, 0))
        z_bar_pct  = min(max((abs(z) / 3.0) * 100, 0), 100)
        z_pos      = z >= 0
        score_pct  = round(min(max(r["Score"] * 100, 0), 100))

        chart_payload = json.dumps({
            "pair":    r["Pair"],
            "nameA":   name_a,
            "nameB":   name_b,
            "dates":   r.get("ZDates",   []),
            "z":       r.get("ZHistory", []),
            "zWindow": Z_LENGTH,
            "currentZ": r["Z"],
        })
        chart_payload_esc = chart_payload.replace("&", "&amp;").replace("'", "&#39;")

        rows_html += f"""
        <tr class="data-row" data-category="{r['Category']}" data-z="{z}"
            data-price-a="{price_a}" data-price-b="{price_b}"
            data-vol-a="{avgvol_a}" data-vol-b="{avgvol_b}">
          <td class="rank-cell">{i+1}</td>
          <td class="pair-cell">
            <div class="pair-names">
              <div class="pair-ticker-row">
                <span class="ticker-a">{a}</span>
                <span class="pair-sep">/</span>
                <span class="ticker-b">{b}</span>
                <span class="{cat_class} cat-badge">{r['Category'].replace('Pure ', '')}</span>
              </div>
              <div class="pair-fullnames">
                <div class="name-a">{name_a}</div>
                <div class="name-b">{name_b}</div>
              </div>
            </div>
          </td>
          <td class="z-cell">
            <div class="z-wrapper">
              <span class="z-value {'z-pos' if z_pos else 'z-neg'}">{z:+.2f}&sigma;</span>
              <div class="z-bar-track">
                <div class="z-bar-fill {'z-bar-pos' if z_pos else 'z-bar-neg'}" style="width:{z_bar_pct}%;"></div>
              </div>
            </div>
          </td>
          <td class="corr-cell">
            <span class="corr-value">{r['Corr']:.2f}</span>
            <span class="corr-brk">&Delta;{r['CorrBrk']:+.3f}</span>
          </td>
          <td class="perf-cell">
            <span class="{'perf-pos' if r['PerfDiff'] >= 0 else 'perf-neg'}">{r['PerfDiff']:+.1f}%</span>
          </td>
          <td class="score-cell">
            <div class="score-bar-wrap">
              <span class="score-num">{r['Score']:.3f}</span>
              <div class="score-bar-track">
                <div class="score-bar-fill" style="width:{score_pct}%;"></div>
              </div>
            </div>
          </td>
          <td class="sig-cell">
            <span class="signal-badge {sig_class}">{sig_arrow} {sig_label}</span>
          </td>
          <td class="chart-cell">
            <button class="chart-btn" onclick="openChart(this)" data-chart='{chart_payload_esc}'>&#9657; Chart</button>
          </td>
          <td class="shares-cell sharesA" data-price="{price_a}">
            <div class="share-price">${price_a:,.2f}</div>
            <div class="share-vol">{fmt_vol(avgvol_a)}</div>
          </td>
          <td class="shares-cell sharesB" data-price="{price_b}">
            <div class="share-price">${price_b:,.2f}</div>
            <div class="share-vol">{fmt_vol(avgvol_b)}</div>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pairs Trading Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:        #08090d;
    --surface:   #0d1117;
    --surface2:  #131720;
    --surface3:  #181f2e;
    --border:    #1c2333;
    --border2:   #242d40;
    --text:      #c9d1d9;
    --muted:     #4a5568;
    --faint:     #2d3748;
    --cyan:      #38bdf8;
    --cyan-dim:  rgba(56,189,248,0.12);
    --green:     #22c55e;
    --green-dim: rgba(34,197,94,0.12);
    --red:       #ef4444;
    --red-dim:   rgba(239,68,68,0.12);
    --amber:     #f59e0b;
    --orange:    #f97316;
    --purple:    #a78bfa;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Syne', sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; font-size: 14px; }}

  /* TOPBAR */
  .topbar {{
    position: sticky; top: 0; z-index: 200;
    background: rgba(8,9,13,0.92); backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 0 32px; height: 56px;
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
  }}
  .topbar-left {{ display: flex; align-items: center; gap: 16px; }}
  .brand {{ font-size: 16px; font-weight: 800; letter-spacing: 0.04em; color: white; }}
  .brand span {{ color: var(--cyan); }}
  .live-dot {{
    width: 7px; height: 7px; background: var(--green); border-radius: 50%;
    box-shadow: 0 0 6px var(--green); animation: pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1);}} 50%{{opacity:.5;transform:scale(.8);}} }}
  .topbar-meta {{ font-family: var(--mono); font-size: 11px; color: var(--muted); display: flex; gap: 20px; white-space: nowrap; }}
  .topbar-meta em {{ color: var(--text); font-style: normal; }}
  .nav-link {{
    font-size: 12px; font-weight: 600; color: var(--cyan); text-decoration: none;
    letter-spacing: 0.05em; padding: 6px 12px; border: 1px solid rgba(56,189,248,0.3);
    border-radius: 4px; transition: all 0.15s; white-space: nowrap;
  }}
  .nav-link:hover {{ background: var(--cyan-dim); border-color: var(--cyan); }}

  /* STATS ROW */
  .stats-row {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 12px 32px; display: flex; overflow-x: auto;
  }}
  .stat-item {{
    padding: 6px 28px 6px 0; margin-right: 28px;
    border-right: 1px solid var(--border); white-space: nowrap; flex-shrink: 0;
  }}
  .stat-item:last-child {{ border-right: none; }}
  .stat-label {{ font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }}
  .stat-value {{ font-family: var(--mono); font-size: 18px; font-weight: 600; color: white; }}
  .stat-value.cyan {{ color: var(--cyan); }}
  .stat-value.green {{ color: var(--green); }}
  .stat-value.amber {{ color: var(--amber); }}

  /* CONTROLS */
  .controls {{
    background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 14px 32px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
  }}
  .control-group {{
    display: flex; align-items: center; gap: 10px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 14px;
  }}
  .control-group label {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted); white-space: nowrap;
  }}
  .control-group select,
  .control-group input[type="number"],
  .control-group input[type="text"] {{
    background: transparent; border: none; outline: none;
    color: white; font-family: var(--mono); font-size: 13px; min-width: 0;
  }}
  .control-group select {{ cursor: pointer; }}
  .control-group select option {{ background: #0d1117; }}
  .control-group input[type="number"] {{ width: 90px; }}
  .control-group input[type="text"]   {{ width: 110px; }}

  /* TABLE */
  .table-wrapper {{ padding: 24px 32px; overflow-x: auto; }}
  table {{
    width: 100%; border-collapse: separate; border-spacing: 0;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden;
  }}
  thead tr {{ background: var(--surface2); }}
  th {{
    padding: 11px 14px; text-align: left; font-size: 10px; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border); white-space: nowrap;
    user-select: none; cursor: pointer; transition: color 0.15s;
  }}
  th:hover {{ color: var(--text); }}
  tbody tr {{ transition: background 0.12s; border-bottom: 1px solid var(--border); }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--surface3); }}
  tbody tr.row-hidden {{ display: none; }}
  td {{ padding: 10px 14px; vertical-align: middle; white-space: nowrap; }}

  /* PAIR CELL */
  .rank-cell {{ font-family: var(--mono); font-size: 11px; color: var(--muted); width: 38px; text-align: center; }}
  .pair-cell {{ min-width: 260px; }}
  .pair-names {{ display: flex; flex-direction: column; gap: 3px; }}
  .pair-ticker-row {{ display: flex; align-items: center; gap: 4px; }}
  .ticker-a {{ font-family: var(--mono); font-size: 14px; font-weight: 700; color: var(--cyan); }}
  .pair-sep  {{ color: var(--muted); margin: 0 2px; font-family: var(--mono); }}
  .ticker-b  {{ font-family: var(--mono); font-size: 14px; font-weight: 700; color: white; }}
  .pair-fullnames {{ display: flex; flex-direction: column; gap: 1px; margin-top: 2px; }}
  .name-a  {{ font-size: 10px; color: #4a8aaa; white-space: normal; line-height: 1.35; }}
  .name-b  {{ font-size: 10px; color: #6b7f9a; white-space: normal; line-height: 1.35; }}

  /* BADGES */
  .cat-badge {{
    display: inline-block; font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    padding: 2px 6px; border-radius: 3px; margin-left: 6px;
    text-transform: uppercase; vertical-align: middle;
  }}
  .cat-etf   {{ background: rgba(56,189,248,0.1);  color: var(--cyan);   border: 1px solid rgba(56,189,248,0.25); }}
  .cat-stock {{ background: rgba(245,158,11,0.1);  color: var(--amber);  border: 1px solid rgba(245,158,11,0.25); }}
  .cat-mixed {{ background: rgba(167,139,250,0.1); color: var(--purple); border: 1px solid rgba(167,139,250,0.25); }}

  /* Z-SCORE */
  .z-cell {{ min-width: 110px; }}
  .z-wrapper {{ display: flex; flex-direction: column; gap: 4px; }}
  .z-value {{ font-family: var(--mono); font-size: 14px; font-weight: 700; }}
  .z-pos {{ color: var(--red); }}
  .z-neg {{ color: var(--green); }}
  .z-bar-track {{ height: 3px; background: var(--faint); border-radius: 2px; overflow: hidden; width: 80px; }}
  .z-bar-fill  {{ height: 100%; border-radius: 2px; }}
  .z-bar-pos {{ background: var(--red); }}
  .z-bar-neg {{ background: var(--green); }}

  /* CORR / PERF / SCORE */
  .corr-cell  {{ min-width: 90px; }}
  .corr-value {{ font-family: var(--mono); font-size: 13px; color: white; display: block; }}
  .corr-brk   {{ font-family: var(--mono); font-size: 10px; color: var(--muted); }}
  .perf-cell  {{ min-width: 75px; }}
  .perf-pos {{ font-family: var(--mono); font-size: 13px; color: var(--green); font-weight: 500; }}
  .perf-neg {{ font-family: var(--mono); font-size: 13px; color: var(--red);   font-weight: 500; }}
  .score-cell {{ min-width: 100px; }}
  .score-bar-wrap {{ display: flex; flex-direction: column; gap: 3px; }}
  .score-num  {{ font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--amber); }}
  .score-bar-track {{ height: 3px; background: var(--faint); border-radius: 2px; width: 70px; overflow: hidden; }}
  .score-bar-fill  {{ height: 100%; background: linear-gradient(90deg, var(--amber), var(--orange)); border-radius: 2px; }}

  /* SIGNAL */
  .sig-cell {{ min-width: 220px; }}
  .signal-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 4px; font-size: 11px;
    font-weight: 700; letter-spacing: 0.05em; white-space: nowrap; font-family: var(--mono);
  }}
  .sig-strong-short {{ background: var(--red-dim);          color: var(--red);    border: 1px solid rgba(239,68,68,0.4); }}
  .sig-short        {{ background: rgba(249,115,22,0.1);    color: var(--orange); border: 1px solid rgba(249,115,22,0.4); }}
  .sig-strong-long  {{ background: var(--green-dim);        color: var(--green);  border: 1px solid rgba(34,197,94,0.4); }}
  .sig-long         {{ background: rgba(132,204,22,0.1);    color: #84cc16;       border: 1px solid rgba(132,204,22,0.4); }}
  .sig-neutral      {{ background: rgba(71,85,105,0.2);     color: var(--muted);  border: 1px solid var(--border); }}

  /* CHART BUTTON */
  .chart-cell {{ min-width: 80px; text-align: center; }}
  .chart-btn {{
    background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.25);
    color: var(--cyan); font-family: var(--mono); font-size: 11px; font-weight: 600;
    padding: 5px 11px; border-radius: 4px; cursor: pointer; letter-spacing: 0.05em;
    transition: background 0.15s, border-color 0.15s; white-space: nowrap;
  }}
  .chart-btn:hover {{ background: rgba(56,189,248,0.18); border-color: var(--cyan); }}

  /* SHARES */
  .shares-cell {{ font-family: var(--mono); font-size: 12px; color: var(--text); min-width: 90px; text-align: right; vertical-align: middle; }}
  .share-price {{ font-size: 11px; color: #64748b; }}
  .share-vol   {{ font-size: 10px; color: #3d4f62; letter-spacing: 0.03em; }}

  /* MODAL */
  .modal-overlay {{
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.78); backdrop-filter: blur(8px);
    align-items: center; justify-content: center;
  }}
  .modal-overlay.open {{ display: flex; }}
  .modal {{
    background: #0a0e17;
    border: 1px solid #242d40; border-radius: 14px;
    width: min(980px, 95vw); max-height: 90vh;
    display: flex; flex-direction: column;
    box-shadow: 0 40px 100px rgba(0,0,0,0.7), 0 0 0 1px rgba(56,189,248,0.06);
    overflow: hidden; animation: modal-in 0.2s ease;
  }}
  @keyframes modal-in {{ from{{opacity:0;transform:scale(0.96) translateY(12px);}} to{{opacity:1;transform:none;}} }}

  .modal-header {{
    padding: 22px 28px 18px;
    border-bottom: 1px solid #1c2333;
    display: flex; align-items: flex-start; justify-content: space-between; gap: 20px;
    background: linear-gradient(180deg, #0d1520 0%, #0a0e17 100%);
    flex-shrink: 0;
  }}
  .modal-title {{ display: flex; flex-direction: column; gap: 5px; }}
  .modal-pair  {{
    font-family: var(--mono); font-size: 24px; font-weight: 700;
    color: white; letter-spacing: -0.01em;
  }}
  .modal-pair .ma {{ color: var(--cyan); }}
  .modal-pair .mb {{ color: #e2e8f0; }}
  .modal-pair-names {{ font-size: 12px; color: #4a6080; font-family: var(--mono); }}

  .modal-stats {{ display: flex; gap: 28px; align-items: center; flex-shrink: 0; }}
  .mstat {{ text-align: right; }}
  .mstat-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 3px; }}
  .mstat-value {{ font-family: var(--mono); font-size: 20px; font-weight: 700; }}

  .modal-close {{
    background: none; border: none; color: var(--muted); font-size: 22px;
    cursor: pointer; padding: 0 4px; line-height: 1; transition: color 0.15s; flex-shrink: 0;
    margin-top: 2px;
  }}
  .modal-close:hover {{ color: white; }}

  .modal-body {{ padding: 20px 28px 22px; flex: 1; overflow: hidden; display: flex; flex-direction: column; }}
  .chart-legend {{ display: flex; gap: 22px; margin-bottom: 14px; flex-shrink: 0; flex-wrap: wrap; }}
  .leg-item {{ display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); font-family: var(--mono); }}
  .leg-line  {{ width: 24px; height: 2px; border-radius: 1px; flex-shrink: 0; }}

  .chart-container {{ position: relative; flex: 1; min-height: 340px; }}

  .modal-footer {{
    padding: 12px 28px;
    border-top: 1px solid #1c2333;
    font-family: var(--mono); font-size: 11px; color: var(--muted);
    flex-shrink: 0; display: flex; gap: 28px; flex-wrap: wrap;
    background: #080c14;
  }}
  .modal-footer em {{ color: #64748b; font-style: normal; }}

  /* FOOTER */
  .footer {{
    padding: 20px 32px; border-top: 1px solid var(--border);
    background: var(--surface); font-size: 11px; color: var(--muted);
    display: flex; justify-content: space-between; align-items: center; gap: 16px;
  }}
  .footer a {{ color: var(--cyan); text-decoration: none; }}
  .leg-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }}

  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <div class="topbar-left">
    <div class="live-dot"></div>
    <div class="brand">PAIRS <span>SCANNER</span></div>
  </div>
  <div class="topbar-meta">
    <span>Updated: <em>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></span>
    <span>Scanned: <em>{len(combos):,} pairs</em></span>
    <span>Setups: <em>{len(results):,}</em></span>
    <span>Showing: <em>Top {len(top_results)}</em></span>
  </div>
  <div><a href="symbols.html" class="nav-link">Symbol Reference &#8594;</a></div>
</div>

<!-- STATS ROW -->
<div class="stats-row">
  <div class="stat-item"><div class="stat-label">Pairs Scanned</div><div class="stat-value cyan">{len(combos):,}</div></div>
  <div class="stat-item"><div class="stat-label">Valid Setups</div><div class="stat-value green">{len(results):,}</div></div>
  <div class="stat-item"><div class="stat-label">Active Symbols</div><div class="stat-value">{len(valid)}</div></div>
  <div class="stat-item"><div class="stat-label">Z Threshold</div><div class="stat-value">{Z_THRESHOLD:.1f}&sigma; / {Z_STRONG:.1f}&sigma;</div></div>
  <div class="stat-item"><div class="stat-label">Min Correlation</div><div class="stat-value">{MIN_CORR_FILTER:.2f}</div></div>
  <div class="stat-item"><div class="stat-label">Corr Window</div><div class="stat-value">{CORR_SHORT}d / {CORR_LONG}d</div></div>
  <div class="stat-item"><div class="stat-label">Z Window</div><div class="stat-value">{Z_LENGTH}d</div></div>
  <div class="stat-item"><div class="stat-label">Perf Window</div><div class="stat-value amber">{PERF_LENGTH}d</div></div>
</div>

<!-- CONTROLS -->
<div class="controls">
  <div class="control-group">
    <label>Capital ($)</label>
    <input type="number" id="capitalInput" value="10000" min="0" step="1000" oninput="calcShares()">
  </div>
  <div class="control-group">
    <label>Pair Type</label>
    <select id="typeFilter" onchange="applyFilters()">
      <option value="All">All</option>
      <option value="Pure ETF">ETF / ETF</option>
      <option value="Pure Stock">Stock / Stock</option>
      <option value="Mixed">Mixed</option>
    </select>
  </div>
  <div class="control-group">
    <label>Signal</label>
    <select id="sigFilter" onchange="applyFilters()">
      <option value="All">All Signals</option>
      <option value="long">Long A</option>
      <option value="short">Short A</option>
      <option value="strong">Strong Only</option>
      <option value="neutral">Neutral</option>
    </select>
  </div>
  <div class="control-group">
    <label>Min |Z|</label>
    <input type="number" id="minZ" value="0" min="0" max="5" step="0.5" oninput="applyFilters()" style="width:55px;">
  </div>
  <div class="control-group">
    <label>Ticker</label>
    <input type="text" id="tickerSearch" placeholder="SPY, AAPL&hellip;" oninput="applyFilters()">
  </div>
  <div class="control-group">
    <label>Min Price ($)</label>
    <input type="number" id="minPrice" value="0" min="0" step="1" oninput="applyFilters()" style="width:70px;">
  </div>
  <div class="control-group">
    <label>Min Avg Vol</label>
    <select id="minVol" onchange="applyFilters()">
      <option value="0">Any</option>
      <option value="100000">&gt; 100K</option>
      <option value="500000">&gt; 500K</option>
      <option value="1000000">&gt; 1M</option>
      <option value="5000000">&gt; 5M</option>
      <option value="10000000">&gt; 10M</option>
    </select>
  </div>
  <div class="control-group">
    <label>Sort</label>
    <select id="sortBy" onchange="sortTable()">
      <option value="score">Score</option>
      <option value="z_abs">|Z-Score|</option>
      <option value="corr">Correlation</option>
      <option value="perf">Perf Diff</option>
    </select>
  </div>
</div>

<!-- TABLE -->
<div class="table-wrapper">
<table id="mainTable">
<thead>
<tr>
  <th>#</th>
  <th>Pair / Name</th>
  <th onclick="setSort('z_abs')">Z-Score &#8597;</th>
  <th onclick="setSort('corr')">Corr / &Delta; &#8597;</th>
  <th onclick="setSort('perf')">Perf Diff &#8597;</th>
  <th onclick="setSort('score')">Score &#8597;</th>
  <th>Signal</th>
  <th style="text-align:center;">Z-Chart</th>
  <th style="text-align:right;">Leg A &nbsp;Price / Vol</th>
  <th style="text-align:right;">Leg B &nbsp;Price / Vol</th>
</tr>
</thead>
<tbody id="tableBody">
{rows_html}
</tbody>
</table>
</div>

<!-- FOOTER -->
<div class="footer">
  <div>
    <span class="leg-dot" style="background:var(--red)"></span>Short A / Long B &nbsp;
    <span class="leg-dot" style="background:var(--green)"></span>Long A / Short B &nbsp;
    <span class="leg-dot" style="background:var(--muted)"></span>Neutral
  </div>
  <div>
    Score = {int(W_ZSCORE*100)}% |Z| + {int(W_CORR_BRK*100)}% Corr Break + {int(W_REL_PERF*100)}% Rel Perf
    &nbsp;&middot;&nbsp; 50/50 capital sizing
    &nbsp;&middot;&nbsp; <a href="symbols.html">Symbol Reference</a>
  </div>
</div>

<!-- Z-SCORE CHART MODAL -->
<div class="modal-overlay" id="chartModal" onclick="closeOnBg(event)">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title">
        <div class="modal-pair" id="modalPairLabel"></div>
        <div class="modal-pair-names" id="modalPairNames"></div>
      </div>
      <div class="modal-stats" id="modalStats"></div>
      <button class="modal-close" onclick="closeChart()">&#x2715;</button>
    </div>
    <div class="modal-body">
      <div class="chart-legend">
        <div class="leg-item"><div class="leg-line" style="background:#38bdf8;height:2px;"></div>Z-Score</div>
        <div class="leg-item"><div class="leg-line" style="background:#22c55e;opacity:.8;"></div>&plusmn;1&sigma;</div>
        <div class="leg-item"><div class="leg-line" style="background:#f59e0b;opacity:.8;"></div>&plusmn;2&sigma;</div>
        <div class="leg-item"><div class="leg-line" style="background:#ef4444;opacity:.9;"></div>&plusmn;3&sigma;</div>
        <div class="leg-item"><div class="leg-line" style="background:#94a3b8;opacity:.35;"></div>Zero</div>
      </div>
      <div class="chart-container">
        <canvas id="zChart"></canvas>
      </div>
    </div>
    <div class="modal-footer" id="modalFooter"></div>
  </div>
</div>

<script>
// ─── CHART ────────────────────────────────────────────────────────────────────
let activeChart = null;

// Load annotation plugin async
(function() {{
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js";
  s.onload = () => {{ Chart.register(window["chartjs-plugin-annotation"]); }};
  document.head.appendChild(s);
}})();

function openChart(btn) {{
  const d = JSON.parse(btn.dataset.chart);
  const {{ pair, nameA, nameB, dates, z, zWindow, currentZ }} = d;
  const [a, b] = pair.split("/");

  // Header
  document.getElementById("modalPairLabel").innerHTML =
    `<span class="ma">${{a}}</span><span style="color:#2d3748;margin:0 10px;">/</span><span class="mb">${{b}}</span>`;
  document.getElementById("modalPairNames").textContent =
    [nameA, nameB].filter(Boolean).join("  \u00b7  ");

  const czColor = currentZ >= 2 ? "#ef4444" : currentZ >= 1 ? "#f59e0b" :
                  currentZ <= -2 ? "#22c55e" : currentZ <= -1 ? "#84cc16" : "#94a3b8";
  document.getElementById("modalStats").innerHTML = `
    <div class="mstat">
      <div class="mstat-label">Current Z</div>
      <div class="mstat-value" style="color:${{czColor}};">${{currentZ >= 0 ? "+" : ""}}${{currentZ.toFixed(2)}}&sigma;</div>
    </div>
    <div class="mstat">
      <div class="mstat-label">History</div>
      <div class="mstat-value" style="color:#4a6080;">${{dates.length}} days</div>
    </div>`;

  document.getElementById("modalFooter").innerHTML =
    `<span>Rolling window: <em>${{zWindow}} days</em></span>` +
    `<span>First: <em>${{dates[0] || "—"}}</em></span>` +
    `<span>Last: <em>${{dates[dates.length-1] || "—"}}</em></span>` +
    `<span style="margin-left:auto;color:#2d3748;">Press ESC or click outside to close</span>`;

  if (activeChart) {{ activeChart.destroy(); activeChart = null; }}

  const ctx = document.getElementById("zChart").getContext("2d");

  // Gradient fill under the line
  const grad = ctx.createLinearGradient(0, 0, 0, 380);
  grad.addColorStop(0,   "rgba(56,189,248,0.20)");
  grad.addColorStop(0.45,"rgba(56,189,248,0.06)");
  grad.addColorStop(1,   "rgba(56,189,248,0.00)");

  // Per-point coloring based on Z level
  const ptColors = z.map(v => {{
    if (v === null) return "transparent";
    const av = Math.abs(v);
    if (av >= 3) return "#ef4444";
    if (av >= 2) return "#f59e0b";
    if (av >= 1) return "#38bdf8";
    return "rgba(148,163,184,0.5)";
  }});

  const hLine = (y, color, width, dash, lbl) => ({{
    type: "line", yMin: y, yMax: y,
    borderColor: color, borderWidth: width, borderDash: dash,
    label: {{ display: !!lbl, content: lbl, color, position: "end",
              font: {{ size: 10, family: "'JetBrains Mono',monospace", weight: "600" }},
              xAdjust: -10, yAdjust: y > 0 ? -10 : 8, backgroundColor: "transparent", borderWidth: 0 }},
  }});

  activeChart = new Chart(ctx, {{
    type: "line",
    data: {{
      labels: dates,
      datasets: [{{
        label: "Z-Score",
        data: z,
        borderColor: "#38bdf8",
        borderWidth: 1.8,
        pointRadius: dates.length > 250 ? 0 : 2.5,
        pointHoverRadius: 6,
        pointBackgroundColor: ptColors,
        pointBorderWidth: 0,
        fill: true,
        backgroundColor: grad,
        tension: 0.3,
        spanGaps: true,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: "#0d1520",
          borderColor: "#242d40",
          borderWidth: 1,
          titleColor: "#64748b",
          bodyColor: "#e2e8f0",
          titleFont: {{ family: "'JetBrains Mono',monospace", size: 11 }},
          bodyFont:  {{ family: "'JetBrains Mono',monospace", size: 14 }},
          padding: 14, caretSize: 5,
          callbacks: {{
            label: c => {{
              const v = c.raw;
              if (v === null) return " Z = —";
              const lv = Math.abs(v) >= 3 ? "EXTREME" : Math.abs(v) >= 2 ? "STRONG" : Math.abs(v) >= 1 ? "SIGNAL" : "neutral";
              return ` Z = ${{v >= 0 ? "+" : ""}}${{v.toFixed(3)}}\u03C3   [${{lv}}]`;
            }},
          }},
        }},
        annotation: {{
          annotations: {{
            zero: hLine(0,  "rgba(148,163,184,0.30)", 1,   [4,4], "0"),
            p1:   hLine(1,  "rgba(34,197,94,0.55)",   1,   [5,4], "+1\u03C3"),
            n1:   hLine(-1, "rgba(34,197,94,0.55)",   1,   [5,4], "-1\u03C3"),
            p2:   hLine(2,  "rgba(245,158,11,0.75)",  1.5, [5,3], "+2\u03C3"),
            n2:   hLine(-2, "rgba(245,158,11,0.75)",  1.5, [5,3], "-2\u03C3"),
            p3:   hLine(3,  "rgba(239,68,68,0.85)",   1.5, [],    "+3\u03C3"),
            n3:   hLine(-3, "rgba(239,68,68,0.85)",   1.5, [],    "-3\u03C3"),
          }},
        }},
      }},
      scales: {{
        x: {{
          ticks: {{
            color: "#374151",
            font: {{ family: "'JetBrains Mono',monospace", size: 10 }},
            maxRotation: 0, maxTicksLimit: 10, autoSkip: true,
          }},
          grid: {{ color: "rgba(28,35,51,0.7)" }},
          border: {{ color: "#1c2333" }},
        }},
        y: {{
          ticks: {{
            color: "#374151",
            font: {{ family: "'JetBrains Mono',monospace", size: 10 }},
            callback: v => (v >= 0 ? "+" : "") + v.toFixed(1) + "\u03C3",
          }},
          grid: {{ color: "rgba(28,35,51,0.6)" }},
          border: {{ color: "#1c2333" }},
        }},
      }},
    }},
  }});

  document.getElementById("chartModal").classList.add("open");
  document.body.style.overflow = "hidden";
}}

function closeChart() {{
  document.getElementById("chartModal").classList.remove("open");
  document.body.style.overflow = "";
  if (activeChart) {{ activeChart.destroy(); activeChart = null; }}
}}
function closeOnBg(e) {{ if (e.target.id === "chartModal") closeChart(); }}
document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeChart(); }});

// ─── SHARE CALCULATOR ─────────────────────────────────────────────────────────
function calcShares() {{
  const total = parseFloat(document.getElementById("capitalInput").value) || 0;
  const leg   = total / 2;
  document.querySelectorAll("tr.data-row:not(.row-hidden)").forEach(row => {{
    const cA = row.querySelector(".sharesA");
    const cB = row.querySelector(".sharesB");
    const pA = parseFloat(cA.dataset.price);
    const pB = parseFloat(cB.dataset.price);
    if (total > 0 && pA > 0 && pB > 0) {{
      const sA = Math.round(leg / pA);
      cA.textContent = sA.toLocaleString();
      cB.textContent = Math.round((sA * pA) / pB).toLocaleString();
    }} else {{ cA.textContent = cB.textContent = "\u2014"; }}
  }});
}}

// ─── FILTERS ──────────────────────────────────────────────────────────────────
function applyFilters() {{
  const catF      = document.getElementById("typeFilter").value;
  const sigF      = document.getElementById("sigFilter").value;
  const minZv     = parseFloat(document.getElementById("minZ").value) || 0;
  const searchV   = document.getElementById("tickerSearch").value.toUpperCase().trim();
  const minPriceV = parseFloat(document.getElementById("minPrice").value) || 0;
  const minVolV   = parseFloat(document.getElementById("minVol").value) || 0;

  document.querySelectorAll("tr.data-row").forEach(row => {{
    const z        = parseFloat(row.dataset.z);
    const cat      = row.dataset.category;
    const priceA   = parseFloat(row.dataset.priceA);
    const priceB   = parseFloat(row.dataset.priceB);
    const volA     = parseFloat(row.dataset.volA);
    const volB     = parseFloat(row.dataset.volB);
    const pairText = row.querySelector(".pair-cell").textContent.toUpperCase();
    const sigClass = row.querySelector(".signal-badge").className;

    let show = true;
    if (catF !== "All" && cat !== catF)          show = false;
    if (Math.abs(z) < minZv)                     show = false;
    if (searchV && !pairText.includes(searchV))  show = false;
    // Price filter: both legs must meet minimum price
    if (minPriceV > 0 && (priceA < minPriceV || priceB < minPriceV)) show = false;
    // Volume filter: both legs must meet minimum avg volume (0 = data unavailable, skip filter)
    if (minVolV > 0 && volA > 0 && volB > 0 && (volA < minVolV || volB < minVolV)) show = false;
    if (sigF === "long"    && !sigClass.includes("sig-long"))    show = false;
    else if (sigF === "short"   && !sigClass.includes("sig-short"))   show = false;
    else if (sigF === "strong"  && !sigClass.includes("sig-strong"))  show = false;
    else if (sigF === "neutral" && !sigClass.includes("sig-neutral")) show = false;

    row.classList.toggle("row-hidden", !show);
  }});
  calcShares();
}}

// ─── SORT ─────────────────────────────────────────────────────────────────────
let currentSort = {{ key: "score", asc: false }};
function setSort(key) {{
  currentSort.asc = currentSort.key === key ? !currentSort.asc : false;
  currentSort.key = key;
  sortTable();
}}
function sortTable() {{
  const key   = document.getElementById("sortBy").value;
  const tbody = document.getElementById("tableBody");
  const rows  = [...tbody.querySelectorAll("tr.data-row")];
  rows.sort((a, b) => {{
    if (key === "score") return parseFloat(b.querySelector(".score-num").textContent) - parseFloat(a.querySelector(".score-num").textContent);
    if (key === "z_abs") return Math.abs(parseFloat(b.dataset.z)) - Math.abs(parseFloat(a.dataset.z));
    if (key === "corr")  return parseFloat(b.querySelector(".corr-value").textContent) - parseFloat(a.querySelector(".corr-value").textContent);
    if (key === "perf")  return Math.abs(parseFloat(b.querySelector(".perf-cell span").textContent)) - Math.abs(parseFloat(a.querySelector(".perf-cell span").textContent));
    return 0;
  }});
  rows.forEach((r, i) => {{ r.querySelector(".rank-cell").textContent = i+1; tbody.appendChild(r); }});
  calcShares();
}}

window.addEventListener("DOMContentLoaded", () => {{
  calcShares();
  document.getElementById("sortBy").addEventListener("change", sortTable);
}});
</script>
</body>
</html>"""

    with open("market_scanner.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"market_scanner.html created. ({len(top_results)} pairs rendered)")
    print("\nDone. Open market_scanner.html in your browser.")

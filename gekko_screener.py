"""
gekko_screener.py
-----------------
Fetches the full Gekko screener_latest table from Supabase and generates
a CSV dataset for downstream scanners (gekko_screener.csv).

Usage:
    python gekko_screener.py
"""

import json
import math
import requests
import pandas as pd
from datetime import datetime

# ── API Config ──
BASE_URL = "https://kztjaygndfebxrpaiowt.supabase.co/rest/v1"
API_KEY  = "sb_publishable_07Cquoeqt4a63kFjMfb_Yg_EsFcoDX0"

HEADERS = {
    "apikey":         API_KEY,
    "Authorization":  f"Bearer {API_KEY}",
    "Accept":         "application/json",
    "Accept-Profile": "public",
}

OUTPUT_FILE = "gekko_screener.html"
OUTPUT_CSV_FILE = "gekko_screener.csv"
GENERATE_HTML = False
PAGE_SIZE   = 1000


# ── GI Score thresholds ──
def gi_tier(score):
    if score is None:
        return "none"
    if score >= 75:
        return "dark-green"
    if score >= 60:
        return "green"
    if score >= 43:
        return "yellow"
    if score >= 28:
        return "orange"
    return "red"


def gi_label(score):
    if score is None:
        return "N/A"
    if score >= 75:
        return "Strong Accumulation"
    if score >= 60:
        return "Accumulation"
    if score >= 43:
        return "Neutral"
    if score >= 28:
        return "Distribution"
    return "Heavy Distribution"


# ──────────────────────────────────────────
# DATA FETCH
# ──────────────────────────────────────────
def fetch_screener() -> pd.DataFrame:
    all_rows = []
    offset = 0
    while True:
        params = {"select": "*", "order": "gi_score.desc", "limit": PAGE_SIZE, "offset": offset}
        resp = requests.get(f"{BASE_URL}/screener_latest", headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        print(f"  Fetched {len(all_rows):,} rows...")
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.DataFrame(all_rows)


# ──────────────────────────────────────────
# HTML GENERATION
# ──────────────────────────────────────────
def fmt_mcap(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def fmt_vol(v):
    try:
        v = int(float(v))
    except (TypeError, ValueError):
        return "—"
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return str(v)


def build_rows_json(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        score = r.get("gi_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None

        chg = r.get("change_pct")
        try:
            chg = float(chg) if chg is not None else None
        except (TypeError, ValueError):
            chg = None

        close = r.get("close")
        try:
            close = float(close) if close is not None else None
        except (TypeError, ValueError):
            close = None

        mcap = r.get("market_cap")
        try:
            mcap = float(mcap) if mcap is not None else None
        except (TypeError, ValueError):
            mcap = None

        vol = r.get("volume")
        try:
            vol = int(float(vol)) if vol is not None else None
        except (TypeError, ValueError):
            vol = None

        rows.append({
            "ticker":   str(r.get("ticker") or ""),
            "name":     str(r.get("name") or ""),
            "sector":   str(r.get("sector") or ""),
            "score":    score,
            "tier":     gi_tier(score),
            "label":    gi_label(score),
            "close":    close,
            "chg":      chg,
            "mcap":     mcap,
            "mcap_fmt": fmt_mcap(mcap),
            "vol":      vol,
            "vol_fmt":  fmt_vol(vol),
            "date":     str(r.get("gi_date") or ""),
        })
    return json.dumps(rows)


def build_html(df: pd.DataFrame) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    total   = len(df)

    scores = df["gi_score"].dropna().astype(float)
    dark_green = int((scores >= 75).sum())
    green      = int(((scores >= 60) & (scores < 75)).sum())
    yellow     = int(((scores >= 43) & (scores < 60)).sum())
    orange     = int(((scores >= 28) & (scores < 43)).sum())
    red        = int((scores < 28).sum())
    avg_score  = scores.mean() if len(scores) else 0

    # unique sectors for filter
    sectors = sorted(df["sector"].dropna().unique().tolist())
    sector_opts = "\n".join(
        f'<option value="{s}">{s}</option>' for s in sectors if s.strip()
    )

    rows_json = build_rows_json(df)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gekko Screener</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
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
    --gi-dg:  #16a34a;
    --gi-g:   #22c55e;
    --gi-y:   #eab308;
    --gi-o:   #f97316;
    --gi-r:   #ef4444;
    --gi-dg-dim: rgba(22,163,74,0.18);
    --gi-g-dim:  rgba(34,197,94,0.13);
    --gi-y-dim:  rgba(234,179,8,0.13);
    --gi-o-dim:  rgba(249,115,22,0.13);
    --gi-r-dim:  rgba(239,68,68,0.13);
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; font-size: 14px; }}

  /* TOPBAR */
  .topbar {{
    position: sticky; top: 0; z-index: 100;
    background: rgba(8,9,13,0.94); backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 0 24px; height: 54px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  .topbar-left {{ display: flex; align-items: center; gap: 14px; }}
  .brand {{ font-size: 16px; font-weight: 800; color: white; letter-spacing: 0.04em; }}
  .brand span {{ color: var(--gi-g); }}
  .live-dot {{
    width: 7px; height: 7px; background: var(--green); border-radius: 50%;
    box-shadow: 0 0 6px var(--green); animation: pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1);}} 50%{{opacity:.5;transform:scale(.8);}} }}
  .topbar-meta {{ font-family: var(--mono); font-size: 11px; color: var(--muted); display:flex; gap:18px; }}
  .topbar-meta em {{ color: var(--text); font-style: normal; }}

  /* STATS ROW */
  .stats-row {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 5px 20px; display: flex; flex-wrap: wrap; gap: 0;
  }}
  .stat-item {{
    padding: 4px 14px 4px 0; margin-right: 14px;
    border-right: 1px solid var(--border); white-space: nowrap;
  }}
  .stat-item:last-child {{ border-right: none; }}
  .stat-label {{ font-size: 9px; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted); margin-bottom: 1px; }}
  .stat-value {{ font-family: var(--mono); font-size: 13px; font-weight: 600; color: white; }}
  .dg {{ color: var(--gi-dg); }} .g {{ color: var(--gi-g); }} .y {{ color: var(--gi-y); }}
  .og {{ color: var(--gi-o); }} .r {{ color: var(--gi-r); }} .cy {{ color: var(--cyan); }}

  /* LEGEND */
  .legend-row {{
    background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 5px 20px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap;
  }}
  .leg {{ display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 10px; color: var(--muted); }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

  /* CONTROLS */
  .controls {{
    background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 7px 16px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  }}
  .ctrl-input {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 5px;
    color: var(--text); font-family: var(--mono); font-size: 12px;
    padding: 6px 12px; outline: none; transition: border-color 0.15s;
  }}
  .ctrl-input:focus {{ border-color: var(--cyan); }}
  .ctrl-input::placeholder {{ color: var(--muted); }}
  select.ctrl-input {{
    cursor: pointer; -webkit-appearance: none; padding-right: 24px;
    background-image: url("data:image/svg+xml,%3Csvg width='10' height='6' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a5568'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 8px center;
  }}
  .ctrl-label {{ font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }}
  .ctrl-sep {{ width: 1px; height: 24px; background: var(--border); }}

  /* TABLE */
  .table-wrap {{ padding: 0 12px 16px; overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; min-width: 700px; }}
  thead th {{
    position: sticky; top: 54px; z-index: 50;
    background: var(--surface2); border-bottom: 2px solid var(--border);
    padding: 8px 10px; text-align: left; cursor: pointer; user-select: none;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); white-space: nowrap;
  }}
  thead th:hover {{ color: var(--cyan); }}
  thead th.sort-asc::after {{ content: ' ▲'; color: var(--cyan); }}
  thead th.sort-desc::after {{ content: ' ▼'; color: var(--cyan); }}
  tbody tr {{ border-bottom: 1px solid var(--border); transition: background 0.1s; }}
  tbody tr:hover {{ background: var(--surface2); }}
  tbody td {{ padding: 7px 10px; font-family: var(--mono); font-size: 12px; white-space: nowrap; }}

  .rank-cell {{ color: var(--muted); font-size: 11px; text-align: right; width: 44px; padding-right: 14px; }}
  .sym-cell  {{ font-weight: 700; color: white; min-width: 60px; }}
  .name-cell {{ color: var(--muted); font-size: 11px; max-width: 200px; overflow: hidden; text-overflow: ellipsis; }}
  .sect-cell {{ color: var(--muted); font-size: 11px; max-width: 140px; overflow: hidden; text-overflow: ellipsis; }}
  .price-cell {{ color: var(--text); }}
  .chg-pos {{ color: var(--green); }}
  .chg-neg {{ color: var(--red); }}

  /* GI SCORE BADGE */
  .gi-badge {{
    display: inline-flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-weight: 700; font-size: 13px;
    width: 52px; padding: 3px 0; border-radius: 5px; letter-spacing: 0;
  }}
  .gi-dark-green {{ background: var(--gi-dg-dim); color: var(--gi-dg); border: 1px solid rgba(22,163,74,0.35); }}
  .gi-green      {{ background: var(--gi-g-dim);  color: var(--gi-g);  border: 1px solid rgba(34,197,94,0.30); }}
  .gi-yellow     {{ background: var(--gi-y-dim);  color: var(--gi-y);  border: 1px solid rgba(234,179,8,0.30); }}
  .gi-orange     {{ background: var(--gi-o-dim);  color: var(--gi-o);  border: 1px solid rgba(249,115,22,0.30); }}
  .gi-red        {{ background: var(--gi-r-dim);  color: var(--gi-r);  border: 1px solid rgba(239,68,68,0.30); }}
  .gi-none       {{ background: var(--faint);     color: var(--muted); border: 1px solid var(--border); }}

  .label-cell {{ font-family: var(--mono); font-size: 10px; color: var(--muted); }}

  /* PAGINATION */
  .pagination {{
    display: flex; align-items: center; justify-content: center; gap: 8px;
    padding: 12px 12px 20px; font-family: var(--mono); font-size: 12px;
  }}
  .pagination button {{
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 6px 14px; border-radius: 4px; cursor: pointer; font-family: var(--mono); font-size: 12px;
  }}
  .pagination button:hover {{ border-color: var(--cyan); color: var(--cyan); }}
  .pagination button:disabled {{ opacity: 0.3; cursor: default; }}
  .page-info {{ color: var(--muted); }}
  .result-count {{ font-family: var(--mono); font-size: 11px; color: var(--muted); padding: 4px 12px 0; }}

  /* GI BAR */
  .gi-bar-wrap {{ width: 80px; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; }}
  .gi-bar {{ height: 100%; border-radius: 3px; }}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <div class="topbar-left">
    <div class="brand">Gekko<span>Screener</span></div>
    <div class="live-dot"></div>
  </div>
  <div class="topbar-meta">
    <span>Updated <em id="updatedAt">{now_str}</em></span>
    <span>Stocks <em>{total:,}</em></span>
    <span>Avg GI <em id="avgGI">{avg_score:.1f}</em></span>
  </div>
</div>

<!-- STATS ROW -->
<div class="stats-row">
  <div class="stat-item"><div class="stat-label">Strong Accum (75+)</div><div class="stat-value dg">{dark_green:,}</div></div>
  <div class="stat-item"><div class="stat-label">Accumulation (60-74)</div><div class="stat-value g">{green:,}</div></div>
  <div class="stat-item"><div class="stat-label">Neutral (43-59)</div><div class="stat-value y">{yellow:,}</div></div>
  <div class="stat-item"><div class="stat-label">Distribution (28-42)</div><div class="stat-value og">{orange:,}</div></div>
  <div class="stat-item"><div class="stat-label">Heavy Dist (&lt;28)</div><div class="stat-value r">{red:,}</div></div>
  <div class="stat-item"><div class="stat-label">Total</div><div class="stat-value cy">{total:,}</div></div>
</div>

<!-- LEGEND -->
<div class="legend-row">
  <div class="leg"><div class="leg-dot" style="background:var(--gi-dg)"></div>75+ Strong Accumulation</div>
  <div class="leg"><div class="leg-dot" style="background:var(--gi-g)"></div>60–74 Accumulation</div>
  <div class="leg"><div class="leg-dot" style="background:var(--gi-y)"></div>43–59 Neutral</div>
  <div class="leg"><div class="leg-dot" style="background:var(--gi-o)"></div>28–42 Distribution</div>
  <div class="leg"><div class="leg-dot" style="background:var(--gi-r)"></div>&lt;28 Heavy Distribution</div>
</div>

<!-- CONTROLS -->
<div class="controls">
  <input id="searchBox" class="ctrl-input" placeholder="Search ticker or name..." style="width:220px" oninput="applyFilters()">
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Tier</span>
  <select id="tierFilter" class="ctrl-input" onchange="applyFilters()">
    <option value="">All tiers</option>
    <option value="dark-green">Strong Accum (75+)</option>
    <option value="green">Accumulation (60+)</option>
    <option value="yellow">Neutral (43+)</option>
    <option value="orange">Distribution (28+)</option>
    <option value="red">Heavy Dist (&lt;28)</option>
  </select>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Sector</span>
  <select id="sectorFilter" class="ctrl-input" onchange="applyFilters()">
    <option value="">All sectors</option>
    {sector_opts}
  </select>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Min GI</span>
  <input id="minGI" class="ctrl-input" type="number" min="0" max="100" step="1" placeholder="0" style="width:72px" oninput="applyFilters()">
  <span class="ctrl-label">Max GI</span>
  <input id="maxGI" class="ctrl-input" type="number" min="0" max="100" step="1" placeholder="100" style="width:72px" oninput="applyFilters()">
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Rows/page</span>
  <select id="pageSize" class="ctrl-input" onchange="changePageSize()">
    <option value="50">50</option>
    <option value="100" selected>100</option>
    <option value="250">250</option>
    <option value="500">500</option>
  </select>
</div>

<div class="result-count" id="resultCount"></div>

<!-- TABLE -->
<div class="table-wrap">
<table id="mainTable">
<thead>
  <tr>
    <th onclick="sortBy('rank')"    data-col="rank"   class="sort-asc">#</th>
    <th onclick="sortBy('ticker')"  data-col="ticker">Ticker</th>
    <th onclick="sortBy('name')"    data-col="name">Name</th>
    <th onclick="sortBy('sector')"  data-col="sector">Sector</th>
    <th onclick="sortBy('score')"   data-col="score">GI Score</th>
    <th>Signal</th>
    <th onclick="sortBy('close')"   data-col="close">Price</th>
    <th onclick="sortBy('chg')"     data-col="chg">Chg %</th>
    <th onclick="sortBy('mcap')"    data-col="mcap">Mkt Cap</th>
    <th onclick="sortBy('vol')"     data-col="vol">Volume</th>
    <th onclick="sortBy('date')"    data-col="date">Date</th>
  </tr>
</thead>
<tbody id="tableBody"></tbody>
</table>
</div>

<div class="pagination" id="pagination"></div>

<script>
const ALL_ROWS = {rows_json};

// Add rank
ALL_ROWS.forEach((r, i) => r.rank = i + 1);

let filtered = [...ALL_ROWS];
let sortCol  = 'rank';
let sortDir  = 1;   // 1=asc, -1=desc
let curPage  = 1;
let pageSize = 100;

const GI_COLORS = {{
  'dark-green': '#16a34a',
  'green':      '#22c55e',
  'yellow':     '#eab308',
  'orange':     '#f97316',
  'red':        '#ef4444',
  'none':       '#4a5568',
}};

function fmtScore(r) {{
  if (r.score === null || r.score === undefined) return '<span class="gi-badge gi-none">—</span>';
  const cls = 'gi-' + r.tier;
  const col = GI_COLORS[r.tier] || '#4a5568';
  const pct = Math.round(r.score);
  const bar = `<div class="gi-bar-wrap" style="margin-left:6px"><div class="gi-bar" style="width:${{pct}}%;background:${{col}}"></div></div>`;
  return `<span class="gi-badge ${{cls}}">${{r.score.toFixed(1)}}</span>${{bar}}`;
}}

function renderTable() {{
  const tbody = document.getElementById('tableBody');
  const start = (curPage - 1) * pageSize;
  const page  = filtered.slice(start, start + pageSize);
  tbody.innerHTML = page.map(r => {{
    const chgCls = r.chg > 0 ? 'chg-pos' : r.chg < 0 ? 'chg-neg' : '';
    const chgStr = r.chg !== null && r.chg !== undefined
      ? `<span class="${{chgCls}}">${{r.chg > 0 ? '+' : ''}}${{r.chg.toFixed(2)}}%</span>` : '—';
    const priceStr = r.close !== null && r.close !== undefined
      ? `${{r.close.toFixed(2)}}` : '—';
    const labelCol = GI_COLORS[r.tier] || '#4a5568';
    return `<tr>
      <td class="rank-cell">${{r.rank}}</td>
      <td class="sym-cell">${{r.ticker}}</td>
      <td class="name-cell" title="${{r.name}}">${{r.name}}</td>
      <td class="sect-cell" title="${{r.sector}}">${{r.sector || '—'}}</td>
      <td>${{fmtScore(r)}}</td>
      <td class="label-cell" style="color:${{labelCol}}">${{r.label}}</td>
      <td class="price-cell">${{priceStr}}</td>
      <td>${{chgStr}}</td>
      <td class="price-cell">${{r.mcap_fmt}}</td>
      <td class="price-cell">${{r.vol_fmt}}</td>
      <td class="price-cell" style="color:var(--muted);font-size:11px">${{r.date}}</td>
    </tr>`;
  }}).join('');

  renderPagination();
  document.getElementById('resultCount').textContent =
    `Showing ${{(start+1).toLocaleString()}}–${{Math.min(start+pageSize, filtered.length).toLocaleString()}} of ${{filtered.length.toLocaleString()}} stocks`;
}}

function renderPagination() {{
  const total = Math.ceil(filtered.length / pageSize);
  const pg = document.getElementById('pagination');
  if (total <= 1) {{ pg.innerHTML = ''; return; }}
  let btns = `<button onclick="goPage(1)" ${{curPage===1?'disabled':''}}>«</button>
              <button onclick="goPage(${{curPage-1}})" ${{curPage===1?'disabled':''}}>‹</button>`;
  const lo = Math.max(1, curPage-2), hi = Math.min(total, curPage+2);
  for (let p = lo; p <= hi; p++) {{
    btns += `<button onclick="goPage(${{p}})" ${{p===curPage?'style="border-color:var(--cyan);color:var(--cyan)"':''}}">${{p}}</button>`;
  }}
  btns += `<button onclick="goPage(${{curPage+1}})" ${{curPage===total?'disabled':''}}>›</button>
           <button onclick="goPage(${{total}})" ${{curPage===total?'disabled':''}}>»</button>
           <span class="page-info">Page ${{curPage}} / ${{total}}</span>`;
  pg.innerHTML = btns;
}}

function goPage(p) {{
  const total = Math.ceil(filtered.length / pageSize);
  curPage = Math.max(1, Math.min(total, p));
  renderTable();
  window.scrollTo(0, 0);
}}

function applyFilters() {{
  const q      = document.getElementById('searchBox').value.trim().toLowerCase();
  const tier   = document.getElementById('tierFilter').value;
  const sector = document.getElementById('sectorFilter').value;
  const minGI  = parseFloat(document.getElementById('minGI').value);
  const maxGI  = parseFloat(document.getElementById('maxGI').value);

  filtered = ALL_ROWS.filter(r => {{
    if (q && !r.ticker.toLowerCase().includes(q) && !r.name.toLowerCase().includes(q)) return false;
    if (tier   && r.tier   !== tier)   return false;
    if (sector && r.sector !== sector) return false;
    if (!isNaN(minGI) && (r.score === null || r.score < minGI)) return false;
    if (!isNaN(maxGI) && (r.score === null || r.score > maxGI)) return false;
    return true;
  }});

  // re-sort
  doSort(false);
  curPage = 1;
  renderTable();
}}

function sortBy(col) {{
  if (sortCol === col) {{ sortDir *= -1; }}
  else {{ sortCol = col; sortDir = col === 'rank' ? 1 : -1; }}
  document.querySelectorAll('thead th').forEach(th => {{
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.col === col) th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
  }});
  doSort(true);
  curPage = 1;
  renderTable();
}}

function doSort(updateClass) {{
  const col = sortCol;
  const dir = sortDir;
  filtered.sort((a, b) => {{
    let av = a[col], bv = b[col];
    if (av === null || av === undefined) av = dir === 1 ? Infinity : -Infinity;
    if (bv === null || bv === undefined) bv = dir === 1 ? Infinity : -Infinity;
    if (typeof av === 'string') return av.localeCompare(bv) * dir;
    return (av - bv) * dir;
  }});
}}

function changePageSize() {{
  pageSize = parseInt(document.getElementById('pageSize').value);
  curPage  = 1;
  renderTable();
}}

// Initial render
renderTable();
</script>
</body>
</html>"""


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
def main():
    print("Fetching Gekko screener data...")
    df = fetch_screener()
    print(f"Total rows: {len(df):,}")

    if df.empty:
        print("No data returned.")
        return

    # Normalize types
    df["gi_score"] = pd.to_numeric(df["gi_score"], errors="coerce")
    df["close"]    = pd.to_numeric(df["close"],    errors="coerce")
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["volume"]   = pd.to_numeric(df["volume"],   errors="coerce")

    # Persist machine-readable output used by other scripts.
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Saved to {OUTPUT_CSV_FILE}  ({len(df):,} rows)")

    if not GENERATE_HTML:
      return

    print("Generating HTML...")
    html = build_html(df)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved to {OUTPUT_FILE}  ({len(html):,} bytes)")

    import webbrowser, os
    webbrowser.open(f"file:///{os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()


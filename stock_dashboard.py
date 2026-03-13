import pandas as pd
import numpy as np
import os
import sys
import time
import json
import gzip
import webbrowser
import importlib
from datetime import datetime, timedelta
from market_data_maintainer import ensure_shared_data

try:
  _tqdm = importlib.import_module("tqdm").tqdm
except Exception:
  class _NoOpTqdm:
    def __init__(self, iterable=None, total=None, **kwargs):
      self.iterable = iterable
    def __iter__(self):
      return iter(self.iterable) if self.iterable is not None else iter(())
    def update(self, n=1):
      return None
    def set_postfix(self, **kwargs):
      return None
    def close(self):
      return None
    def __enter__(self):
      return self
    def __exit__(self, exc_type, exc, tb):
      return False
  def _tqdm(iterable=None, total=None, **kwargs):
    return _NoOpTqdm(iterable=iterable, total=total, **kwargs)

tqdm = _tqdm

# ==========================================
# CONFIG
# ==========================================
STOCK_DATA_FILE = "stock_data.csv.gz"
ETF_DATA_FILE   = "etf_data.csv.gz"
CHART_OPEN_FILE = "chart_open_data.csv.gz"
CHART_HIGH_FILE = "chart_high_data.csv.gz"
CHART_LOW_FILE = "chart_low_data.csv.gz"
CHART_VOLUME_FILE = "chart_volume_data.csv.gz"
BATCH_SIZE      = 40
COOLDOWN        = 1.5
LOOKBACK_DAYS   = 400
OUTPUT_FILE     = "stock_dashboard.html"
COMPRESS_HTML_OUTPUT = True
REFRESH_DATA_BEFORE_SCAN = False  # Keep dashboard read-only unless explicitly enabled.

# Scan/filter tuning
MIN_HISTORY_DAYS_FOR_SCAN = 210
DEFAULT_MIN_PRICE_FILTER = 5
BIAS_BULL_THRESHOLD = 30
BIAS_STRONG_BULL_THRESHOLD = 60
BIAS_BEAR_THRESHOLD = -30

# Optional Gekko GI integration (from gekko_screener.py export)
GEKKO_SCREENER_FILE = "gekko_screener.csv"
INCLUDE_GEKKO_IN_SCORE = True
GEKKO_SCORE_WEIGHT = 0.35  # 0.0 = technical-only, 1.0 = GI-only

# ==========================================
# DATA HELPERS (same pattern as pairs_watchlist)
# ==========================================
TICKER_TYPES    = {}
TICKER_NAMES    = {}
TICKER_INDUSTRY = {}
TICKER_SUBIND   = {}
TICKER_SUBIND2  = {}
TICKER_EXCHANGES = {}
TICKER_CSV_MCAP  = {}
TICKER_CSV_VOL   = {}
ETF_LEV_TYPES   = {}
GEKKO_SCORE_MAP = {}


def _ticker_aliases(ticker):
  t = str(ticker or "").strip().upper()
  if not t:
    return []
  a = t.replace('.', '-')
  b = t.replace('-', '.')
  return [t, a, b]


def _gekko_label(gi_score):
  if gi_score is None:
    return "N/A"
  if gi_score >= 75:
    return "Strong Accumulation"
  if gi_score >= 60:
    return "Accumulation"
  if gi_score >= 43:
    return "Neutral"
  if gi_score >= 28:
    return "Distribution"
  return "Heavy Distribution"

def _parse_mcap_str(s):
    import re
    s = str(s).strip()
    m = re.match(r'\$?([\d.]+)\s*(T|B|M|K)\b', s, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        suffix = m.group(2).upper()
        if suffix == "T": val *= 1e12
        elif suffix == "B": val *= 1e9
        elif suffix == "M": val *= 1e6
        elif suffix == "K": val *= 1e3
        return int(val)
    raw = s.replace("$", "").replace(",", "").strip()
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return 0

def _parse_vol_str(s):
    s = str(s).strip().replace(",", "")
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0

_ETF_TYPE_MAP = {
    "etf": "normal", "etf, leveraged": "leveraged",
    "etf, inverse": "inverse", "etf, leveraged, inverse": "lev_inv",
    "etn": "etn", "etn, leveraged": "etn_lev", "etn, leveraged, inverse": "etn_lev_inv",
}

def load_master_tickers():
    global TICKER_TYPES, TICKER_NAMES, TICKER_INDUSTRY, TICKER_SUBIND, TICKER_SUBIND2
    global TICKER_EXCHANGES, TICKER_CSV_MCAP, TICKER_CSV_VOL, ETF_LEV_TYPES
    tickers = []

    if os.path.exists("ETFs.csv"):
        df_etf = pd.read_csv("ETFs.csv", header=None)
        etfs = df_etf[0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers += etfs
        ncols = df_etf.shape[1]
        for _, row in tqdm(
            df_etf.iterrows(),
            total=len(df_etf),
            desc="Loading ETF metadata",
            unit="row",
            ncols=88,
            file=sys.stdout,
        ):
            t = str(row.iloc[0]).strip().upper()
            if not t or t in ("", "NONE", "NAN", "SYMBOL", "TICKER"): continue
            TICKER_TYPES[t] = "Pure ETF"
            TICKER_EXCHANGES[t] = "AMEX"
            if ncols >= 2 and pd.notna(row.iloc[1]): TICKER_NAMES[t] = str(row.iloc[1]).strip()
            if ncols >= 3 and pd.notna(row.iloc[2]):
                ETF_LEV_TYPES[t] = _ETF_TYPE_MAP.get(str(row.iloc[2]).strip().lower(), "normal")
            else: ETF_LEV_TYPES[t] = "normal"
            if ncols >= 4 and pd.notna(row.iloc[3]): TICKER_INDUSTRY[t] = str(row.iloc[3]).strip()
            if ncols >= 5 and pd.notna(row.iloc[4]): TICKER_SUBIND[t] = str(row.iloc[4]).strip()
            if ncols >= 6 and pd.notna(row.iloc[5]): TICKER_CSV_MCAP[t] = _parse_mcap_str(row.iloc[5])
            if ncols >= 7 and pd.notna(row.iloc[6]): TICKER_CSV_VOL[t] = _parse_vol_str(row.iloc[6])
        print(f"ETFs.csv: {len(etfs)} tickers")

    if os.path.exists("STOCKS.csv"):
        df_stock = pd.read_csv("STOCKS.csv", header=None)
        stocks = df_stock[0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers += stocks
        ncols = df_stock.shape[1]
        for _, row in tqdm(
            df_stock.iterrows(),
            total=len(df_stock),
            desc="Loading stock metadata",
            unit="row",
            ncols=88,
            file=sys.stdout,
        ):
            t = str(row.iloc[0]).strip().upper()
            if not t or t in ("", "NONE", "NAN", "SYMBOL", "TICKER"): continue
            TICKER_TYPES[t] = "Pure Stock"
            if ncols >= 2 and pd.notna(row.iloc[1]): TICKER_NAMES[t] = str(row.iloc[1]).strip()
            if ncols >= 3 and pd.notna(row.iloc[2]): TICKER_INDUSTRY[t] = str(row.iloc[2]).strip()
            if ncols >= 4 and pd.notna(row.iloc[3]): TICKER_SUBIND[t] = str(row.iloc[3]).strip()
            if ncols >= 5 and pd.notna(row.iloc[4]): TICKER_SUBIND2[t] = str(row.iloc[4]).strip()
            if ncols >= 6 and pd.notna(row.iloc[5]):
                TICKER_EXCHANGES[t] = str(row.iloc[5]).strip().upper()
            else: TICKER_EXCHANGES[t] = "NASDAQ"
            if ncols >= 7 and pd.notna(row.iloc[6]): TICKER_CSV_MCAP[t] = _parse_mcap_str(row.iloc[6])
            if ncols >= 8 and pd.notna(row.iloc[7]): TICKER_CSV_VOL[t] = _parse_vol_str(row.iloc[7])

    tickers = list(set(tickers))
    tickers = [t for t in tickers if t not in ["", "NONE", "NAN", "SYMBOL", "TICKER"]]
    print(f"Loaded {len(tickers)} tickers total.")
    return tickers

def load_cache(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.loc[:, ~df.columns.duplicated()]
            return df
        except Exception:
            pass
    return pd.DataFrame()

def safe_save(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, compression='gzip')
    os.replace(tmp, path)

def write_dashboard_output(html, out_path):
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

  gz_path = None
  if COMPRESS_HTML_OUTPUT:
    gz_path = out_path + ".gz"
    with gzip.open(gz_path, "wb") as f:
      f.write(html.encode("utf-8"))
  return gz_path


def load_gekko_scores(path=GEKKO_SCREENER_FILE):
  """Load ticker -> GI score map from local Gekko export CSV."""
  if not os.path.exists(path):
    print(f"Gekko file not found: {path} (continuing without GI blend)")
    return {}

  try:
    df = pd.read_csv(path)
  except Exception as e:
    print(f"Failed to read {path}: {e} (continuing without GI blend)")
    return {}

  required = {"ticker", "gi_score"}
  if not required.issubset(df.columns):
    print(f"{path} missing required columns {required} (continuing without GI blend)")
    return {}

  out = {}
  for _, row in df.iterrows():
    t = str(row.get("ticker", "")).strip().upper()
    if not t:
      continue
    raw_g = row.get("gi_score")
    if raw_g is None:
      continue
    try:
      g = float(raw_g)
    except (TypeError, ValueError):
      continue
    if np.isnan(g):
      continue
    g = max(0.0, min(100.0, g))
    for alias in _ticker_aliases(t):
      out[alias] = g

  print(f"Loaded {len(out)} Gekko GI scores from {path}")
  return out


def trim_caches(master):
  """Trim cache files to only the data that's needed."""
  master_set = set(master)

  # â”€â”€ Price caches: keep LOOKBACK_DAYS + 20% buffer â”€â”€
  for cache_file in [STOCK_DATA_FILE, ETF_DATA_FILE]:
    if os.path.exists(cache_file):
      try:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        before = (df.shape[0], df.shape[1])
        keep_days = int(LOOKBACK_DAYS * 1.2)
        df = df[[c for c in df.columns if c in master_set]]
        df = df.tail(keep_days)
        if (df.shape[0], df.shape[1]) != before:
          safe_save(df, cache_file)
          print(f"  Trimmed {cache_file}: {before[0]}x{before[1]} â†’ {df.shape[0]}x{df.shape[1]}")
      except Exception:
        pass




# ==========================================
# TECHNICAL INDICATORS
# ==========================================

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d

def calc_bollinger(close, period=20, std_mult=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma * 100
    return sma, upper, lower, width

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = calc_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx, plus_di, minus_di

def calc_obv(close, volume):
    direction = np.sign(close.diff())
    return (volume * direction).fillna(0).cumsum()

def detect_golden_death_cross(ma50, ma200):
    if len(ma50) < 5 or len(ma200) < 5: return "NO", "NO"
    golden_recent = "NO"
    death_recent = "NO"
    for i in range(min(20, len(ma50)-1)):
        idx = -(i+1)
        prev_idx = idx - 1
        if abs(prev_idx) <= len(ma50):
            if ma50.iloc[idx] > ma200.iloc[idx] and ma50.iloc[prev_idx] <= ma200.iloc[prev_idx]:
                golden_recent = "YES"
            if ma50.iloc[idx] < ma200.iloc[idx] and ma50.iloc[prev_idx] >= ma200.iloc[prev_idx]:
                death_recent = "YES"
    # Also flag if currently above
    if ma50.iloc[-1] > ma200.iloc[-1] and golden_recent == "NO":
        golden_recent = "YES"
    return golden_recent, death_recent

def detect_higher_lows(close, lookback=60):
    if len(close) < lookback: return False, False
    prices = close.tail(lookback)
    vals = prices.values
    lows, highs = [], []
    for i in range(2, len(vals)-2):
        if vals[i] < vals[i-1] and vals[i] < vals[i-2] and vals[i] < vals[i+1] and vals[i] < vals[i+2]:
            lows.append(vals[i])
        if vals[i] > vals[i-1] and vals[i] > vals[i-2] and vals[i] > vals[i+1] and vals[i] > vals[i+2]:
            highs.append(vals[i])
    higher_lows = len(lows) >= 2 and all(lows[i] > lows[i-1] for i in range(1, len(lows)))
    lower_highs = len(highs) >= 2 and all(highs[i] < highs[i-1] for i in range(1, len(highs)))
    return higher_lows, lower_highs

def is_consolidating(close, lookback=20, threshold=0.05):
    recent = close.tail(lookback)
    return (recent.max() - recent.min()) / recent.mean() < threshold


# ==========================================
# FAST SCAN (cache-first, OHLCV-aware when available)
# ==========================================

def fast_scan_symbol(symbol, close_series, open_series=None, high_series=None, low_series=None, volume_series=None):
    """Score a symbol using cached prices; prefers OHLCV when available."""
    close = close_series.dropna()
    if len(close) < 210:
        return None

    close = close.astype(float)
    price = close.iloc[-1]
    prev_close = close.iloc[-2]
    pct_change = ((price - prev_close) / prev_close) * 100

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    rsi = calc_rsi(close)
    rsi_val = round(float(rsi.iloc[-1]), 1) if not np.isnan(rsi.iloc[-1]) else 50

    macd_line, signal_line, macd_hist = calc_macd(close)
    macd_hist_val = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0
    macd_hist_rising = macd_hist.iloc[-1] > macd_hist.iloc[-2] if len(macd_hist) > 1 else False

    bb_mid, bb_upper, bb_lower, bb_width = calc_bollinger(close)
    bb_width_val = float(bb_width.iloc[-1]) if not np.isnan(bb_width.iloc[-1]) else 0
    bb_width_avg = float(bb_width.tail(20).mean()) if not np.isnan(bb_width.tail(20).mean()) else 0
    bb_pct = float((price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 50

    open_aligned = open_series.reindex(close.index).astype(float) if open_series is not None else None
    high_aligned = high_series.reindex(close.index).astype(float) if high_series is not None else None
    low_aligned = low_series.reindex(close.index).astype(float) if low_series is not None else None
    vol_aligned = volume_series.reindex(close.index).astype(float) if volume_series is not None else None

    has_ohl = (
      high_aligned is not None
      and low_aligned is not None
      and high_aligned.notna().sum() >= 210
      and low_aligned.notna().sum() >= 210
    )

    if has_ohl:
      k_line, _ = calc_stochastic(high_aligned, low_aligned, close, k_period=14, d_period=3)
      stoch_k = float(k_line.iloc[-1]) if not np.isnan(k_line.iloc[-1]) else 50.0
    else:
      roll_high = close.rolling(14).max()
      roll_low = close.rolling(14).min()
      stoch_k = float(100 * (price - roll_low.iloc[-1]) / (roll_high.iloc[-1] - roll_low.iloc[-1])) if (roll_high.iloc[-1] - roll_low.iloc[-1]) > 0 else 50.0

    if has_ohl:
      atr_series = calc_atr(high_aligned, low_aligned, close, period=14)
      atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0
      atr_avg20 = float(atr_series.tail(20).mean()) if not np.isnan(atr_series.tail(20).mean()) else 0.0
      atr_is_proxy = False
    else:
      tr_proxy = close.diff().abs()
      atr_series = tr_proxy.rolling(14).mean()
      atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0
      atr_avg20 = float(atr_series.tail(20).mean()) if not np.isnan(atr_series.tail(20).mean()) else 0.0
      atr_is_proxy = True

    atr_proxy_pct = (atr_val / price * 100.0) if price > 0 else 0.0
    atr_expanding = atr_val > (atr_avg20 * 1.08) if atr_avg20 > 0 else False

    ret_pct = close.pct_change() * 100.0
    ret_vol20 = float(ret_pct.tail(20).std()) if len(ret_pct.dropna()) >= 20 else 0.0
    gap_proxy_thresh = max(1.0, ret_vol20 * 1.2)
    if open_aligned is not None and open_aligned.notna().sum() >= 2:
      today_open = float(open_aligned.iloc[-1]) if not np.isnan(open_aligned.iloc[-1]) else np.nan
      prev_close_val = float(close.iloc[-2]) if len(close) >= 2 else np.nan
      gap_pct = ((today_open - prev_close_val) / prev_close_val) * 100.0 if prev_close_val and not np.isnan(today_open) else np.nan
      gap_up_proxy = bool(not np.isnan(gap_pct) and gap_pct >= gap_proxy_thresh)
      gap_down_proxy = bool(not np.isnan(gap_pct) and gap_pct <= -gap_proxy_thresh)
      gap_is_proxy = False
    else:
      gap_pct = float(pct_change)
      gap_up_proxy = pct_change >= gap_proxy_thresh
      gap_down_proxy = pct_change <= -gap_proxy_thresh
      gap_is_proxy = True

    if vol_aligned is not None and vol_aligned.notna().sum() >= 30:
      vol_now = float(vol_aligned.iloc[-1]) if not np.isnan(vol_aligned.iloc[-1]) else 0.0
      vol_avg20 = float(vol_aligned.tail(20).mean()) if not np.isnan(vol_aligned.tail(20).mean()) else 0.0
      rvol = (vol_now / vol_avg20) if vol_avg20 > 0 else 0.0
      obv_series = calc_obv(close, vol_aligned.fillna(0))
      obv_up = bool(len(obv_series) > 2 and obv_series.iloc[-1] > obv_series.iloc[-2])
      volume_is_proxy = False
    else:
      vol_now = 0.0
      vol_avg20 = 0.0
      rvol = 0.0
      obv_up = False
      volume_is_proxy = True

    close_252 = close.tail(252)
    high_52w = close_252.max()
    low_52w = close_252.min()
    pct_from_52h = round(((price - high_52w) / high_52w) * 100, 1)

    above_ma20 = price > ma20.iloc[-1] if not np.isnan(ma20.iloc[-1]) else False
    above_ma50 = price > ma50.iloc[-1] if not np.isnan(ma50.iloc[-1]) else False
    above_ma200 = price > ma200.iloc[-1] if not np.isnan(ma200.iloc[-1]) else False

    golden_cross, death_cross = detect_golden_death_cross(ma50.dropna(), ma200.dropna())
    higher_lows, lower_highs = detect_higher_lows(close)
    in_squeeze = bb_width_val < bb_width_avg

    # Breakout score
    breakout = 0
    breakdown = 0
    if rsi_val > 60: breakout += 15
    if rsi_val > 70: breakout += 10
    if rsi_val < 40: breakdown += 15
    if rsi_val < 30: breakdown += 10
    if macd_hist_val > 0: breakout += 15
    if macd_hist_val > 0 and macd_hist_rising: breakout += 5
    if macd_hist_val < 0: breakdown += 15
    if macd_hist_val < 0 and not macd_hist_rising: breakdown += 5
    if stoch_k > 70: breakout += 10
    if stoch_k < 30: breakdown += 10
    if above_ma20: breakout += 8
    else: breakdown += 8
    if above_ma50: breakout += 8
    else: breakdown += 8
    if above_ma200: breakout += 8
    else: breakdown += 8
    if bb_pct > 95: breakout += 10
    if bb_pct < 5: breakdown += 10
    if higher_lows: breakout += 8
    if lower_highs: breakdown += 8
    if golden_cross == "YES": breakout += 8
    if death_cross == "YES": breakdown += 8
    if atr_expanding and pct_change > 0: breakout += 6
    if atr_expanding and pct_change < 0: breakdown += 6
    if gap_up_proxy: breakout += 6
    if gap_down_proxy: breakdown += 6

    breakout = min(100, breakout)
    breakdown = min(100, breakdown)
    net_bias_raw = breakout - breakdown

    gi_score = None
    for alias in _ticker_aliases(symbol):
      gi_score = GEKKO_SCORE_MAP.get(alias)
      if gi_score is not None:
        break
    gi_bias = ((gi_score - 50.0) * 2.0) if gi_score is not None else None

    if INCLUDE_GEKKO_IN_SCORE and gi_bias is not None:
      w = max(0.0, min(1.0, float(GEKKO_SCORE_WEIGHT)))
      net_bias = int(round((1.0 - w) * net_bias_raw + w * gi_bias))
    else:
      net_bias = int(round(net_bias_raw))

    net_bias = int(max(-100, min(100, net_bias)))

    chart_close = close.tail(252)
    chart_idx = chart_close.index
    ma20_chart = ma20.reindex(chart_idx)
    ma50_chart = ma50.reindex(chart_idx)
    ma200_chart = ma200.reindex(chart_idx)
    bb_upper_chart = bb_upper.reindex(chart_idx)
    bb_lower_chart = bb_lower.reindex(chart_idx)

    chart_dates = [d.strftime("%Y-%m-%d") for d in chart_idx]
    chart_prices = [round(float(v), 2) for v in chart_close.values]
    chart_ma20 = [round(float(v), 2) if pd.notna(v) else None for v in ma20_chart.values]
    chart_ma50 = [round(float(v), 2) if pd.notna(v) else None for v in ma50_chart.values]
    chart_ma200 = [round(float(v), 2) if pd.notna(v) else None for v in ma200_chart.values]
    chart_bb_upper = [round(float(v), 2) if pd.notna(v) else None for v in bb_upper_chart.values]
    chart_bb_lower = [round(float(v), 2) if pd.notna(v) else None for v in bb_lower_chart.values]

    breakout_level = round(float(chart_close.tail(20).max()), 2) if len(chart_close) >= 20 else round(float(chart_close.max()), 2)
    breakdown_level = round(float(chart_close.tail(20).min()), 2) if len(chart_close) >= 20 else round(float(chart_close.min()), 2)

    sector = TICKER_INDUSTRY.get(symbol, "â€”")
    name = TICKER_NAMES.get(symbol, symbol)
    mcap = TICKER_CSV_MCAP.get(symbol, 0)
    ttype = TICKER_TYPES.get(symbol, "â€”")

    return {
        "symbol": symbol,
        "name": name,
        "sector": sector,
        "type": ttype,
        "price": round(float(price), 2),
        "pct_change": round(float(pct_change), 2),
        "rsi": rsi_val,
        "macd_hist": round(macd_hist_val, 4),
        "macd_trend": "Bullish" if macd_hist_val > 0 else "Bearish",
        "macd_rising": bool(macd_hist_rising),
        "stoch_k": round(stoch_k),
        "bb_width": round(bb_width_val, 1),
        "bb_width_avg": round(bb_width_avg, 1),
        "bb_pct": round(bb_pct, 1),
        "atr_proxy": round(atr_val, 4),
        "atr_proxy_pct": round(atr_proxy_pct, 2),
        "atr_proxy_avg20": round(atr_avg20, 4),
        "atr_expanding": bool(atr_expanding),
        "gap_up_proxy": bool(gap_up_proxy),
        "gap_down_proxy": bool(gap_down_proxy),
        "gap_pct": round(float(gap_pct), 2) if not np.isnan(gap_pct) else 0.0,
        "gap_proxy_thresh": round(gap_proxy_thresh, 2),
        "atr_is_proxy": bool(atr_is_proxy),
        "gap_is_proxy": bool(gap_is_proxy),
        "volume_is_proxy": bool(volume_is_proxy),
        "rvol20": round(float(rvol), 2),
        "vol_now": int(vol_now) if vol_now > 0 else 0,
        "vol_avg20": int(vol_avg20) if vol_avg20 > 0 else 0,
        "obv_up": bool(obv_up),
        "pct_from_52h": pct_from_52h,
        "above_ma20": bool(above_ma20),
        "above_ma50": bool(above_ma50),
        "above_ma200": bool(above_ma200),
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "higher_lows": bool(higher_lows),
        "lower_highs": bool(lower_highs),
        "in_squeeze": bool(in_squeeze),
        "breakout": breakout,
        "breakdown": breakdown,
        "net_bias_raw": int(net_bias_raw),
        "net_bias": net_bias,
        "gi_score": round(float(gi_score), 1) if gi_score is not None else None,
        "gi_label": _gekko_label(gi_score),
        "mcap": mcap,
        "chart_dates": chart_dates,
        "chart_close": chart_prices,
        "chart_ma20": chart_ma20,
        "chart_ma50": chart_ma50,
        "chart_ma200": chart_ma200,
        "chart_bb_upper": chart_bb_upper,
        "chart_bb_lower": chart_bb_lower,
        "breakout_level": breakout_level,
        "breakdown_level": breakdown_level,
    }


# ==========================================
# HTML: MAIN TABLE + DETAIL OVERLAY
# ==========================================

def build_full_html(scan_results):
    """Build single HTML with ranking table + click-to-detail overlay."""
    # Sort by net_bias descending
    scan_results.sort(key=lambda x: x["net_bias"], reverse=True)

    # Add rank
    for i, r in enumerate(scan_results):
        r["rank"] = i + 1

    rows_json = json.dumps(scan_results)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(scan_results)
    bullish = sum(1 for r in scan_results if r["net_bias"] >= BIAS_BULL_THRESHOLD)
    bearish = sum(1 for r in scan_results if r["net_bias"] <= BIAS_BEAR_THRESHOLD)
    squeeze_count = sum(1 for r in scan_results if r["in_squeeze"])
    gi_scores = [r.get("gi_score") for r in scan_results if r.get("gi_score") is not None]
    gi_coverage = len(gi_scores)
    gi_avg = (sum(gi_scores) / gi_coverage) if gi_coverage else 0.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
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
    --detail-ink:   #e6edf5;
    --detail-soft:  #93a8ba;
    --detail-card:  #0f1622;
    --detail-card2: #121b29;
    --detail-line:  #253247;
    --detail-chip:  #111a28;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Syne', sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; font-size: 14px; }}

  /* â”€â”€ TOPBAR â”€â”€ */
  .topbar {{
    position: sticky; top: 0; z-index: 100;
    background: rgba(8,9,13,0.92); backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 0 32px; height: 56px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  .topbar-left {{ display: flex; align-items: center; gap: 16px; }}
  .brand {{ font-size: 16px; font-weight: 800; color: white; letter-spacing: 0.04em; }}
  .brand span {{ color: var(--cyan); }}
  .live-dot {{
    width: 7px; height: 7px; background: var(--green); border-radius: 50%;
    box-shadow: 0 0 6px var(--green); animation: pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1);}} 50%{{opacity:.5;transform:scale(.8);}} }}
  .topbar-meta {{ font-family: var(--mono); font-size: 11px; color: var(--muted); display: flex; gap: 20px; }}
  .topbar-meta em {{ color: var(--text); font-style: normal; }}

  /* â”€â”€ STATS ROW â”€â”€ */
  .stats-row {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 6px 20px; display: flex; flex-wrap: wrap;
  }}
  .stat-item {{
    padding: 4px 14px 4px 0; margin-right: 14px;
    border-right: 1px solid var(--border); white-space: nowrap;
  }}
  .stat-item:last-child {{ border-right: none; }}
  .stat-label {{ font-size: 9px; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted); margin-bottom: 1px; }}
  .stat-value {{ font-family: var(--mono); font-size: 14px; font-weight: 600; color: white; }}
  .stat-value.cyan {{ color: var(--cyan); }}
  .stat-value.green {{ color: var(--green); }}
  .stat-value.red {{ color: var(--red); }}
  .stat-value.amber {{ color: var(--amber); }}

  /* â”€â”€ CONTROLS â”€â”€ */
  .controls {{
    background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 8px 16px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  }}
  .ctrl-input {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 5px;
    color: var(--text); font-family: var(--mono); font-size: 12px;
    padding: 6px 12px; outline: none; transition: border-color 0.15s;
  }}
  .ctrl-input:focus {{ border-color: var(--cyan); }}
  .ctrl-input::placeholder {{ color: var(--muted); }}
  select.ctrl-input {{ cursor: pointer; -webkit-appearance: none; padding-right: 24px;
    background-image: url("data:image/svg+xml,%3Csvg width='10' height='6' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a5568'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 8px center; }}
  .ctrl-label {{ font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }}
  .mode-toggle {{
    background: var(--surface3); border: 1px solid var(--border2); border-radius: 5px;
    color: var(--text); font-family: var(--mono); font-size: 11px; font-weight: 600;
    padding: 6px 10px; cursor: pointer; letter-spacing: 0.04em;
  }}
  .mode-toggle:hover {{ border-color: var(--cyan); color: var(--cyan); }}
  .mode-toggle.active-tech {{ border-color: rgba(245,158,11,0.55); color: var(--amber); }}

  /* â”€â”€ TABLE â”€â”€ */
  .table-wrap {{ padding: 0 16px 16px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead th {{
    position: sticky; top: 56px; z-index: 50;
    background: var(--surface2); border-bottom: 2px solid var(--border);
    padding: 8px 10px; text-align: left; cursor: pointer; user-select: none;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); white-space: nowrap;
  }}
  thead th:hover {{ color: var(--cyan); }}
  thead th.sort-asc::after {{ content: ' â–²'; color: var(--cyan); }}
  thead th.sort-desc::after {{ content: ' â–¼'; color: var(--cyan); }}
  tbody tr {{
    border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.1s;
  }}
  tbody tr:hover {{ background: var(--surface2); }}
  tbody td {{
    padding: 7px 10px; font-family: var(--mono); font-size: 12px; white-space: nowrap;
  }}
  .rank-cell {{ color: var(--muted); font-size: 11px; text-align: center; width: 40px; }}
  .sym-cell {{ font-weight: 700; color: white; }}
  .name-cell {{ color: var(--muted); font-size: 11px; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }}
  .sector-cell {{ color: var(--muted); font-size: 11px; max-width: 140px; overflow: hidden; text-overflow: ellipsis; }}
  .price-cell {{ color: var(--text); }}
  .chg-pos {{ color: var(--green); }}
  .chg-neg {{ color: var(--red); }}
  .score-cell {{ font-weight: 700; text-align: center; }}
  .tag-sm {{
    font-family: var(--mono); font-size: 10px; padding: 2px 7px; border-radius: 3px;
    display: inline-block;
  }}
  .tag-sm.green {{ background: var(--green-dim); color: var(--green); }}
  .tag-sm.red {{ background: var(--red-dim); color: var(--red); }}
  .tag-sm.cyan {{ background: var(--cyan-dim); color: var(--cyan); }}
  .tag-sm.amber {{ background: rgba(245,158,11,0.12); color: var(--amber); }}
  .tag-sm.muted {{ background: rgba(74,85,104,0.12); color: var(--muted); }}
  .tag-sm.gi-strong {{ background: rgba(34,197,94,0.16); color: #22c55e; }}
  .tag-sm.gi-accum {{ background: rgba(74,222,128,0.16); color: #4ade80; }}
  .tag-sm.gi-neutral {{ background: rgba(245,158,11,0.16); color: #f59e0b; }}
  .tag-sm.gi-dist {{ background: rgba(249,115,22,0.16); color: #f97316; }}
  .tag-sm.gi-heavy {{ background: rgba(239,68,68,0.16); color: #ef4444; }}

  /* â”€â”€ PAGINATION â”€â”€ */
  .pagination {{
    display: flex; align-items: center; justify-content: center; gap: 8px;
    padding: 12px; font-family: var(--mono); font-size: 12px;
  }}
  .pagination button {{
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 6px 12px; border-radius: 4px; cursor: pointer; font-family: var(--mono); font-size: 12px;
  }}
  .pagination button:hover {{ border-color: var(--cyan); color: var(--cyan); }}
  .pagination button:disabled {{ opacity: 0.3; cursor: default; }}
  .pagination .page-info {{ color: var(--muted); }}

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     DETAIL OVERLAY
     â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  .overlay {{
    display: none; position: fixed; inset: 0; z-index: 500;
    background:
      radial-gradient(140% 120% at 50% -20%, rgba(56,189,248,0.14) 0%, rgba(56,189,248,0) 45%),
      var(--bg);
    overflow-y: auto;
  }}
  .overlay.open {{ display: block; }}
  .overlay-inner {{ max-width: 1160px; margin: 0 auto; padding: 20px 24px 40px; }}

  .detail-header {{
    display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;
  }}
  .detail-header h1 {{ font-size: 38px; line-height: 1.05; font-weight: 800; color: var(--detail-ink); letter-spacing: 0.01em; }}
  .detail-meta {{
    font-family: var(--mono); font-size: 11px; color: var(--detail-soft); margin-top: 8px;
  }}
  .detail-meta .v {{ color: var(--text); }}
  .detail-meta .g {{ color: var(--green); }}
  .detail-meta .r {{ color: var(--red); }}
  .detail-meta .c {{ color: var(--cyan); }}
  .close-btn {{
    font-family: var(--mono); font-size: 12px; letter-spacing: 0.06em;
    color: var(--muted); background: var(--surface2);
    border: 1px solid var(--border); padding: 9px 14px; border-radius: 6px; cursor: pointer;
  }}
  .close-btn:hover {{ color: var(--text); border-color: var(--border2); }}

  .scores {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; margin-bottom: 16px; }}
  .score-box {{
    background: linear-gradient(180deg, rgba(18,27,41,0.88) 0%, rgba(15,22,34,0.78) 100%);
    border: 1px solid var(--detail-line); border-radius: 10px;
    padding: 14px 16px; text-align: center;
  }}
  .score-label {{
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
  }}
  .score-val {{ font-family: var(--mono); font-size: 42px; line-height: 1; font-weight: 700; color: var(--detail-ink); }}

  .chart-section {{
    background:
      radial-gradient(120% 90% at 0% 0%, rgba(56,189,248,0.12) 0%, rgba(56,189,248,0) 58%),
      var(--detail-card);
    border: 1px solid var(--detail-line); border-radius: 10px;
    padding: 16px; margin-bottom: 16px; position: relative;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
  }}
  .chart-hdr {{
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
  }}
  .chart-title {{ font-family: var(--mono); font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: #a7bed3; }}
  .chart-legend {{ display: flex; gap: 12px; font-family: var(--mono); font-size: 9px; color: var(--muted); flex-wrap: wrap; }}
  .chart-legend span {{ display: flex; align-items: center; gap: 4px; }}
  .legend-line {{ width: 16px; height: 2px; border-radius: 1px; }}
  .chart-stats {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }}
  .chart-stat {{
    font-family: var(--mono); font-size: 9px; color: #b8ccdc;
    border: 1px solid var(--detail-line); background: var(--detail-chip);
    padding: 4px 8px; border-radius: 999px;
  }}
  .chart-stat .v {{ color: #f8fafc; font-weight: 600; }}
  .legend-dash {{ width: 16px; height: 0; border-top: 2px dashed; }}
  .ma-labels {{
    position: absolute; left: 24px; top: 60px; font-family: var(--mono); font-size: 11px;
  }}
  .ma-labels div {{ margin-bottom: 4px; }}
  canvas#priceChart {{ width: 100% !important; height: 340px !important; }}

  @media (max-width: 980px) {{
    .scores {{ grid-template-columns: 1fr; }}
    .ind-grid {{ grid-template-columns: 1fr; }}
    .comp-metrics {{ grid-template-columns: repeat(3, 1fr); }}
    .ai-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .detail-header {{ flex-direction: column; gap: 8px; }}
    .close-btn {{ width: 100%; }}
    canvas#priceChart {{ height: 280px !important; }}
    .ai-cell:nth-child(4n) {{ border-right: 1px solid rgba(51,65,85,0.35); }}
    .ai-cell:nth-child(2n) {{ border-right: 0; }}
  }}

  @media (max-width: 640px) {{
    .overlay-inner {{ padding: 16px; }}
    .detail-header h1 {{ font-size: 30px; }}
    .comp-metrics {{ grid-template-columns: repeat(2, 1fr); }}
    .ai-grid {{ grid-template-columns: 1fr; }}
    .chart-legend {{ gap: 8px; }}
    canvas#priceChart {{ height: 240px !important; }}
    .ai-cell {{ border-right: 0; }}
  }}

  .ind-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }}
  .ind-box {{ background: var(--detail-card); border: 1px solid var(--detail-line); border-radius: 8px; padding: 12px 14px; }}
  .ind-label {{ font-family: var(--mono); font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .tag {{
    font-family: var(--mono); font-size: 10px; font-weight: 500;
    padding: 4px 10px; border-radius: 5px; white-space: nowrap;
  }}
  .tag-green {{ background: var(--green-dim); color: var(--green); border: 1px solid rgba(34,197,94,0.25); }}
  .tag-red {{ background: var(--red-dim); color: var(--red); border: 1px solid rgba(239,68,68,0.25); }}
  .tag-cyan {{ background: var(--cyan-dim); color: var(--cyan); border: 1px solid rgba(56,189,248,0.25); }}
  .tag-amber {{ background: rgba(245,158,11,0.12); color: var(--amber); border: 1px solid rgba(245,158,11,0.25); }}
  .tag-muted {{ background: rgba(74,85,104,0.12); color: var(--muted); border: 1px solid rgba(74,85,104,0.25); }}

  .compression {{
    background: var(--detail-card); border: 1px solid var(--detail-line); border-radius: 8px;
    padding: 16px; margin-bottom: 16px;
  }}
  .comp-hdr {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
  .comp-icon {{
    width: 24px; height: 24px; background: var(--cyan-dim); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
  }}
  .comp-icon::after {{ content: ''; width: 8px; height: 8px; background: var(--cyan); border-radius: 50%; }}
  .comp-title {{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted); }}
  .squeeze-badge {{ font-family: var(--mono); font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 4px; }}
  .sq-fire {{ background: var(--red-dim); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }}
  .sq-on {{ background: rgba(245,158,11,0.15); color: var(--amber); border: 1px solid rgba(245,158,11,0.3); }}
  .sq-off {{ background: var(--green-dim); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }}
  .coil-label {{ font-size: 15px; font-weight: 700; color: white; }}
  .bias-label {{ font-family: var(--mono); font-size: 12px; color: var(--muted); }}
  .bias-val {{ color: var(--text); }}

  .comp-metrics {{
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px;
    background: var(--border); border-radius: 6px; overflow: hidden; margin-bottom: 16px;
  }}
  .cm {{ background: var(--surface2); padding: 10px 8px; }}
  .cm-label {{ font-family: var(--mono); font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }}
  .cm-val {{ font-family: var(--mono); font-size: 16px; font-weight: 600; color: white; }}
  .cm-sub {{ font-family: var(--mono); font-size: 9px; color: var(--muted); margin-top: 2px; }}

  .range-box {{
    background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
    padding: 12px 16px; display: inline-block; margin-bottom: 16px;
  }}
  .range-label {{ font-family: var(--mono); font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }}
  .range-value {{ font-family: var(--mono); font-size: 20px; font-weight: 600; color: white; }}
  .range-sub {{ font-family: var(--mono); font-size: 10px; color: var(--muted); }}

  .all-indicators {{
    background: var(--detail-card); border: 1px solid var(--detail-line); border-radius: 8px;
    padding: 16px; margin-top: 16px;
  }}
  .all-ind-title {{ font-family: var(--mono); font-size: 10px; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; font-weight: 600; }}
  .ai-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; }}
  .ai-cell {{
    padding: 8px 10px; border-bottom: 1px solid var(--border);
    border-right: 1px solid rgba(51,65,85,0.35);
    font-family: var(--mono); font-size: 10px; display: flex; align-items: center; gap: 6px;
    line-height: 1.35;
  }}
  .ai-cell:nth-child(4n) {{ border-right: 0; }}
  .ai-cell .val {{ color: white; font-weight: 600; }}
  .ai-cell .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
  .ai-cell .st {{ font-size: 10px; padding: 2px 8px; border-radius: 3px; }}

  /* Loading overlay */
  .loading {{
    display: none; position: fixed; inset: 0; z-index: 600;
    background: rgba(8,9,13,0.85); backdrop-filter: blur(8px);
    justify-content: center; align-items: center; flex-direction: column; gap: 16px;
  }}
  .loading.active {{ display: flex; }}
  .spinner {{
    width: 40px; height: 40px; border: 3px solid var(--border);
    border-top-color: var(--cyan); border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .loading-text {{ font-family: var(--mono); font-size: 13px; color: var(--muted); }}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <div class="topbar-left">
    <div class="brand"><span>STOCK</span> DASHBOARD</div>
    <div class="live-dot"></div>
    <div class="topbar-meta">
      <span>Updated <em>{now_str}</em></span>
    </div>
  </div>
  <div style="display:flex;gap:16px;align-items:center;">
    <a href="index.html" style="color:var(--cyan);text-decoration:none;font-size:12px;font-weight:600;font-family:var(--mono);border:1px solid rgba(56,189,248,0.3);padding:6px 14px;border-radius:5px;letter-spacing:0.05em;transition:all 0.15s;">&#8962; Home</a>
  </div>
</div>

<!-- STATS ROW -->
<div class="stats-row">
  <div class="stat-item"><div class="stat-label">Tickers</div><div class="stat-value cyan">{total}</div></div>
  <div class="stat-item"><div class="stat-label">Bullish ({BIAS_BULL_THRESHOLD}+)</div><div class="stat-value green">{bullish}</div></div>
  <div class="stat-item"><div class="stat-label">Bearish ({BIAS_BEAR_THRESHOLD})</div><div class="stat-value red">{bearish}</div></div>
  <div class="stat-item"><div class="stat-label">Gekko Avg GI</div><div class="stat-value amber">{gi_avg:.1f}</div></div>
  <div class="stat-item"><div class="stat-label">GI Coverage</div><div class="stat-value cyan">{gi_coverage}/{total}</div></div>
  <div class="stat-item"><div class="stat-label">In Squeeze</div><div class="stat-value amber">{squeeze_count}</div></div>
</div>

<!-- CONTROLS -->
<div class="controls">
  <span class="ctrl-label">Search</span>
  <input class="ctrl-input" id="searchBox" type="text" placeholder="Symbol or name..." style="width:180px;">
  <span class="ctrl-label">Type</span>
  <select class="ctrl-input" id="typeFilter">
    <option value="">All</option>
    <option value="Pure Stock">Stocks</option>
    <option value="Pure ETF">ETFs</option>
  </select>
  <span class="ctrl-label">Sector</span>
  <select class="ctrl-input" id="sectorFilter"><option value="">All</option></select>
  <span class="ctrl-label">Bias</span>
  <select class="ctrl-input" id="biasFilter">
    <option value="">All</option>
    <option value="bull">Bullish ({BIAS_BULL_THRESHOLD}+)</option>
    <option value="strong_bull">Strong Bull ({BIAS_STRONG_BULL_THRESHOLD}+)</option>
    <option value="bear">Bearish ({BIAS_BEAR_THRESHOLD})</option>
    <option value="neutral">Neutral</option>
    <option value="squeeze">In Squeeze</option>
  </select>
  <span class="ctrl-label">Min Price</span>
  <input class="ctrl-input" id="minPrice" type="number" value="{DEFAULT_MIN_PRICE_FILTER}" style="width:70px;" min="0">
  <span class="ctrl-label">Min GI</span>
  <input class="ctrl-input" id="minGI" type="number" value="0" style="width:70px;" min="0" max="100">
  <button class="mode-toggle" id="scoreModeBtn" type="button" onclick="toggleScoreMode()">Mode: Blended</button>
</div>

<!-- TABLE -->
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th data-col="rank">#</th>
        <th data-col="symbol">Symbol</th>
        <th data-col="name">Name</th>
        <th data-col="sector">Sector</th>
        <th data-col="price">Price</th>
        <th data-col="pct_change">Chg%</th>
        <th data-col="breakout">Breakout</th>
        <th data-col="breakdown">Breakdown</th>
        <th data-col="net_bias_raw">Tech Bias</th>
        <th data-col="gi_score">GI</th>
        <th data-col="net_bias" id="netBiasHeader">Net Bias (Blend)</th>
        <th data-col="rsi">RSI</th>
        <th data-col="macd_trend">MACD</th>
        <th data-col="stoch_k">Stoch</th>
        <th data-col="bb_pct">BB%</th>
        <th data-col="pct_from_52h">52H%</th>
        <th data-col="in_squeeze">Squeeze</th>
        <th data-col="higher_lows">Structure</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
</div>
<div class="pagination" id="pagination"></div>

<!-- DETAIL OVERLAY -->
<div class="overlay" id="detailOverlay">
  <div class="overlay-inner" id="detailContent">
    <!-- Populated by JS -->
  </div>
</div>

<!-- LOADING -->
<div class="loading" id="loadingOverlay">
  <div class="spinner"></div>
  <div class="loading-text" id="loadingText">Loading detail...</div>
</div>

<script>
const ALL_DATA = {rows_json};
let filtered = [...ALL_DATA];
let sortCol = 'net_bias', sortDir = 'desc';
let scoreMode = 'blended'; // 'blended' | 'technical'
let page = 0;
const PER_PAGE = 50;
const BIAS_BULL_THRESHOLD = {BIAS_BULL_THRESHOLD};
const BIAS_STRONG_BULL_THRESHOLD = {BIAS_STRONG_BULL_THRESHOLD};
const BIAS_BEAR_THRESHOLD = {BIAS_BEAR_THRESHOLD};

function modeBiasValue(r) {{
  return scoreMode === 'technical' ? r.net_bias_raw : r.net_bias;
}}

function updateScoreModeUI() {{
  const btn = document.getElementById('scoreModeBtn');
  const hdr = document.getElementById('netBiasHeader');
  if (scoreMode === 'technical') {{
    btn.textContent = 'Mode: Technical';
    btn.classList.add('active-tech');
    hdr.textContent = 'Net Bias (Tech)';
  }} else {{
    btn.textContent = 'Mode: Blended';
    btn.classList.remove('active-tech');
    hdr.textContent = 'Net Bias (Blend)';
  }}
}}

function toggleScoreMode() {{
  scoreMode = scoreMode === 'blended' ? 'technical' : 'blended';
  if (sortCol === 'net_bias') sortDir = 'desc';
  updateScoreModeUI();
  applyFilters();
}}

// â”€â”€ Populate sectors â”€â”€
const sectors = [...new Set(ALL_DATA.map(r => r.sector).filter(s => s && s !== 'â€”'))].sort();
const sf = document.getElementById('sectorFilter');
sectors.forEach(s => {{ const o = document.createElement('option'); o.value = s; o.textContent = s; sf.appendChild(o); }});

// â”€â”€ Filter + Sort â”€â”€
function applyFilters() {{
  const q = document.getElementById('searchBox').value.toLowerCase();
  const typ = document.getElementById('typeFilter').value;
  const sec = document.getElementById('sectorFilter').value;
  const bias = document.getElementById('biasFilter').value;
  const minP = parseFloat(document.getElementById('minPrice').value) || 0;
  const minGI = parseFloat(document.getElementById('minGI').value) || 0;

  filtered = ALL_DATA.filter(r => {{
    const activeBias = modeBiasValue(r);
    if (q && !r.symbol.toLowerCase().includes(q) && !r.name.toLowerCase().includes(q)) return false;
    if (typ && r.type !== typ) return false;
    if (sec && r.sector !== sec) return false;
    if (r.price < minP) return false;
    if (minGI > 0 && (r.gi_score === null || r.gi_score < minGI)) return false;
    if (bias === 'bull' && activeBias < BIAS_BULL_THRESHOLD) return false;
    if (bias === 'strong_bull' && activeBias < BIAS_STRONG_BULL_THRESHOLD) return false;
    if (bias === 'bear' && activeBias > BIAS_BEAR_THRESHOLD) return false;
    if (bias === 'neutral' && (activeBias >= BIAS_BULL_THRESHOLD || activeBias <= BIAS_BEAR_THRESHOLD)) return false;
    if (bias === 'squeeze' && !r.in_squeeze) return false;
    return true;
  }});
  doSort();
  page = 0;
  render();
}}

function doSort() {{
  filtered.sort((a, b) => {{
    let va = (sortCol === 'net_bias') ? modeBiasValue(a) : a[sortCol];
    let vb = (sortCol === 'net_bias') ? modeBiasValue(b) : b[sortCol];
    if (va === null || va === undefined) va = -Infinity;
    if (vb === null || vb === undefined) vb = -Infinity;
    if (typeof va === 'string') {{ va = va.toLowerCase(); vb = vb.toLowerCase(); }}
    if (typeof va === 'boolean') {{ va = va ? 1 : 0; vb = vb ? 1 : 0; }}
    if (va < vb) return sortDir === 'asc' ? -1 : 1;
    if (va > vb) return sortDir === 'asc' ? 1 : -1;
    return 0;
  }});
  // Re-rank
  filtered.forEach((r, i) => r._fRank = i + 1);
}}

function render() {{
  const tbody = document.getElementById('tableBody');
  const start = page * PER_PAGE;
  const slice = filtered.slice(start, start + PER_PAGE);

  tbody.innerHTML = slice.map(r => `
    <tr onclick="openDetail('${{r.symbol}}')">
      <td class="rank-cell">${{r._fRank}}</td>
      <td class="sym-cell">${{r.symbol}}</td>
      <td class="name-cell">${{r.name}}</td>
      <td class="sector-cell">${{r.sector}}</td>
      <td class="price-cell">${{r.price.toFixed(2)}}</td>
      <td class="${{r.pct_change >= 0 ? 'chg-pos' : 'chg-neg'}}">${{r.pct_change >= 0 ? '+' : ''}}${{r.pct_change.toFixed(2)}}%</td>
      <td class="score-cell" style="color:var(--green)">${{r.breakout}}</td>
      <td class="score-cell" style="color:${{r.breakdown > 0 ? 'var(--red)' : 'var(--muted)'}}">${{r.breakdown}}</td>
      <td class="score-cell" style="color:${{r.net_bias_raw > 0 ? 'var(--green)' : r.net_bias_raw < 0 ? 'var(--red)' : 'var(--muted)'}}">${{r.net_bias_raw > 0 ? '+' : ''}}${{r.net_bias_raw}}</td>
      <td>${{r.gi_score === null ? '<span class="tag-sm muted">N/A</span>' : `<span class="tag-sm ${{r.gi_score >= 75 ? 'gi-strong' : r.gi_score >= 60 ? 'gi-accum' : r.gi_score >= 43 ? 'gi-neutral' : r.gi_score >= 28 ? 'gi-dist' : 'gi-heavy'}}">${{r.gi_score.toFixed(1)}}</span>`}}</td>
      <td class="score-cell" style="color:${{modeBiasValue(r) > 0 ? 'var(--green)' : modeBiasValue(r) < 0 ? 'var(--red)' : 'var(--muted)'}}">${{modeBiasValue(r) > 0 ? '+' : ''}}${{modeBiasValue(r)}}</td>
      <td><span class="tag-sm ${{r.rsi >= 70 ? 'red' : r.rsi >= 60 ? 'green' : r.rsi <= 30 ? 'red' : 'muted'}}">${{r.rsi}}</span></td>
      <td><span class="tag-sm ${{r.macd_trend === 'Bullish' ? 'green' : 'red'}}">${{r.macd_trend === 'Bullish' ? 'Bull' : 'Bear'}}</span></td>
      <td><span class="tag-sm ${{r.stoch_k >= 70 ? 'green' : r.stoch_k <= 30 ? 'red' : 'muted'}}">${{r.stoch_k}}</span></td>
      <td><span class="tag-sm ${{r.bb_pct > 90 ? 'cyan' : r.bb_pct < 10 ? 'red' : 'muted'}}">${{r.bb_pct.toFixed(0)}}%</span></td>
      <td style="color:${{r.pct_from_52h > -5 ? 'var(--green)' : r.pct_from_52h > -15 ? 'var(--amber)' : 'var(--red)'}}">${{r.pct_from_52h}}%</td>
      <td><span class="tag-sm ${{r.in_squeeze ? 'amber' : 'muted'}}">${{r.in_squeeze ? 'YES' : 'â€”'}}</span></td>
      <td><span class="tag-sm ${{r.higher_lows ? 'green' : r.lower_highs ? 'red' : 'muted'}}">${{r.higher_lows ? 'HL â†—' : r.lower_highs ? 'LH â†˜' : 'â€”'}}</span></td>
    </tr>
  `).join('');

  // Pagination
  const totalPages = Math.ceil(filtered.length / PER_PAGE);
  const pg = document.getElementById('pagination');
  pg.innerHTML = `
    <button onclick="page=0;render();" ${{page===0?'disabled':''}}>&#171;</button>
    <button onclick="page--;render();" ${{page===0?'disabled':''}}>&#8249; Prev</button>
    <span class="page-info">Page ${{page+1}} of ${{totalPages}} &middot; ${{filtered.length}} results</span>
    <button onclick="page++;render();" ${{page>=totalPages-1?'disabled':''}}>Next &#8250;</button>
    <button onclick="page=${{totalPages-1}};render();" ${{page>=totalPages-1?'disabled':''}}>&#187;</button>
  `;

  // Update header sort indicators
  document.querySelectorAll('thead th').forEach(th => {{
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.col === sortCol) th.classList.add(sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
  }});
}}

// â”€â”€ Column sorting â”€â”€
document.querySelectorAll('thead th').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (sortCol === col) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    else {{ sortCol = col; sortDir = 'desc'; }}
    doSort();
    render();
  }});
}});

// â”€â”€ Filter listeners â”€â”€
document.getElementById('searchBox').addEventListener('input', applyFilters);
document.getElementById('typeFilter').addEventListener('change', applyFilters);
document.getElementById('sectorFilter').addEventListener('change', applyFilters);
document.getElementById('biasFilter').addEventListener('change', applyFilters);
document.getElementById('minPrice').addEventListener('change', applyFilters);
document.getElementById('minGI').addEventListener('change', applyFilters);

// â”€â”€ Initial render â”€â”€
updateScoreModeUI();
doSort();
render();


// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// DETAIL OVERLAY
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
let _detailChart = null;

function openDetail(symbol) {{
  const overlay = document.getElementById('detailOverlay');
  const loading = document.getElementById('loadingOverlay');
  const loadingText = document.getElementById('loadingText');

  // Check if we have pre-computed detail data embedded
  if (window._detailCache && window._detailCache[symbol]) {{
    renderDetail(window._detailCache[symbol]);
    return;
  }}

  // Show loading, fetch via yfinance proxy... but since this is static HTML,
  // we render from the scan data we already have (cache-backed indicators)
  const row = ALL_DATA.find(r => r.symbol === symbol);
  if (!row) return;

  // Render from cached scan payload (ATR/gap may be proxy if OHLC missing)
  renderDetailFromScan(row);
}}

function renderDetailFromScan(d) {{
  const overlay = document.getElementById('detailOverlay');
  const content = document.getElementById('detailContent');

  const rsiLabel = d.rsi >= 70 ? 'overbought' : d.rsi <= 30 ? 'oversold' : 'neutral';
  const rsiHot = d.rsi >= 65;
  const stochLabel = d.stoch_k >= 50 ? 'bull' : 'bear';
  const macdLabel = (d.macd_trend === 'Bullish' ? 'bull' : 'bear') + (d.macd_rising ? ' +hist' : ' -hist');
  const bbStatus = d.bb_pct > 95 ? 'upper break' : d.bb_pct < 5 ? 'lower break' : 'mid range';

  const maTags = [];
  if (d.above_ma20 && d.above_ma50 && d.above_ma200) maTags.push('>MA20+50+200');
  else if (d.above_ma20 && d.above_ma50) maTags.push('>MA20+50');
  else if (d.above_ma20) maTags.push('>MA20');
  else if (!d.above_ma20 && !d.above_ma50 && !d.above_ma200) maTags.push('<MA20+50+200');

  const biasScore = Math.round(d.net_bias / 2 + 50);
  const inSqueeze = d.in_squeeze;
  const netBias = d.net_bias;
  const coilType = inSqueeze ? (netBias > 0 ? 'BULLISH COIL' : netBias < 0 ? 'BEARISH COIL' : 'NEUTRAL COIL') : (netBias > 0 ? 'BULLISH EXPANSION' : 'BEARISH EXPANSION');

  const indexLabel = d.mcap > 10e9 ? 'S&P500' : d.mcap > 2e9 ? 'Mid-Cap' : 'Small-Cap';

  content.innerHTML = `
    <div class="detail-header">
      <div>
        <h1>${{d.symbol}}</h1>
        <div class="detail-meta">
          <span class="v">${{d.name}}</span> &middot;
          <span>${{d.sector}}</span> &middot;
          <span>${{indexLabel}}</span> &middot;
          <span class="v">${{d.price.toFixed(2)}}</span> &middot;
          <span class="${{d.pct_change >= 0 ? 'g' : 'r'}}">${{d.pct_change >= 0 ? '+' : ''}}${{d.pct_change.toFixed(2)}}%</span> &middot;
          <span>RSI <span class="c">${{d.rsi}}</span></span> &middot;
          <span>GI <span class="c">${{d.gi_score === null ? 'N/A' : d.gi_score.toFixed(1)}}</span></span>
        </div>
      </div>
      <button class="close-btn" onclick="closeDetail()">&#10005; Close</button>
    </div>

    <div class="scores">
      <div class="score-box"><div class="score-label">&#9650; Breakout</div><div class="score-val" style="color:var(--green)">${{d.breakout}}</div></div>
      <div class="score-box"><div class="score-label">&#9660; Breakdown</div><div class="score-val" style="color:${{d.breakdown > 0 ? 'var(--red)' : 'var(--muted)'}}">${{d.breakdown}}</div></div>
      <div class="score-box"><div class="score-label">Tech Bias</div><div class="score-val" style="color:${{d.net_bias_raw > 0 ? 'var(--green)' : d.net_bias_raw < 0 ? 'var(--red)' : 'var(--muted)'}}">${{d.net_bias_raw > 0 ? '+' : ''}}${{d.net_bias_raw}}</div></div>
      <div class="score-box"><div class="score-label">&#9830; Net Bias</div><div class="score-val" style="color:${{d.net_bias > 0 ? 'var(--green)' : d.net_bias < 0 ? 'var(--red)' : 'var(--muted)'}}">${{d.net_bias > 0 ? '+' : ''}}${{d.net_bias}}</div></div>
    </div>

    <div class="chart-section">
      <div class="chart-hdr">
        <div class="chart-title">Price Chart · Daily · 52 Weeks</div>
        <div class="chart-legend">
          <span><i class="legend-line" style="background:#38bdf8"></i>Close</span>
          <span><i class="legend-line" style="background:#22d3ee"></i>MA20</span>
          <span><i class="legend-line" style="background:#facc15"></i>MA50</span>
          <span><i class="legend-line" style="background:#a78bfa"></i>MA200</span>
          <span><i class="legend-line" style="background:rgba(56,189,248,0.35)"></i>BBands</span>
        </div>
      </div>
      <div class="chart-stats">
        <span class="chart-stat">Price <span class="v">${{d.price.toFixed(2)}}</span></span>
        <span class="chart-stat">52W High Dist <span class="v">${{d.pct_from_52h.toFixed(1)}}%</span></span>
        <span class="chart-stat">Breakout <span class="v">${{Number.isFinite(d.breakout_level) ? d.breakout_level.toFixed(2) : 'n/a'}}</span></span>
        <span class="chart-stat">Breakdown <span class="v">${{Number.isFinite(d.breakdown_level) ? d.breakdown_level.toFixed(2) : 'n/a'}}</span></span>
      </div>
      <canvas id="priceChart"></canvas>
    </div>

    <div class="ind-grid">
      <div class="ind-box"><div class="ind-label">Momentum</div><div class="tags">
        <span class="tag ${{rsiHot ? 'tag-green' : d.rsi < 35 ? 'tag-red' : 'tag-muted'}}">RSI ${{d.rsi}} ${{rsiHot ? '&mdash; hot' : d.rsi < 35 ? '&mdash; cold' : ''}}</span>
        <span class="tag ${{d.macd_trend === 'Bullish' ? 'tag-green' : 'tag-red'}}">MACD ${{macdLabel}}</span>
        <span class="tag ${{d.stoch_k >= 50 ? 'tag-green' : 'tag-red'}}">Stoch ${{d.stoch_k}} ${{stochLabel}}</span>
      </div></div>
      <div class="ind-box"><div class="ind-label">Volume</div><div class="tags">
        <span class="tag ${{d.volume_is_proxy ? 'tag-muted' : 'tag-cyan'}}">${{d.volume_is_proxy ? 'Volume unavailable' : ('RVOL ' + d.rvol20.toFixed(2) + 'x')}}</span>
        <span class="tag ${{d.volume_is_proxy ? 'tag-muted' : d.obv_up ? 'tag-green' : 'tag-red'}}">${{d.volume_is_proxy ? 'OBV n/a' : d.obv_up ? 'OBV rising' : 'OBV falling'}}</span>
      </div></div>
    </div>

    <div class="ind-grid">
      <div class="ind-box"><div class="ind-label">Technical</div><div class="tags">
        <span class="tag ${{d.pct_from_52h > -5 ? 'tag-green' : d.pct_from_52h > -15 ? 'tag-amber' : 'tag-red'}}">${{d.pct_from_52h}}% 52H</span>
        <span class="tag ${{d.above_ma20 && d.above_ma50 && d.above_ma200 ? 'tag-green' : 'tag-red'}}">${{maTags.join(' ') || '>MA20'}}</span>
        <span class="tag ${{d.golden_cross === 'YES' ? 'tag-green' : 'tag-muted'}}">${{d.golden_cross === 'YES' ? 'Golden Cross' : 'No Cross'}}</span>
      </div></div>
      <div class="ind-box"><div class="ind-label">Volatility</div><div class="tags">
        <span class="tag ${{d.bb_pct > 90 ? 'tag-cyan' : d.bb_pct < 10 ? 'tag-red' : 'tag-muted'}}">BB ${{bbStatus}}</span>
        <span class="tag ${{d.bb_width < d.bb_width_avg ? 'tag-amber' : 'tag-muted'}}">BB ${{d.bb_width < d.bb_width_avg ? 'contracting' : 'normal'}}</span>
        <span class="tag ${{d.atr_expanding ? 'tag-cyan' : 'tag-muted'}}">ATR ${{d.atr_is_proxy ? 'proxy ' : ''}}${{d.atr_expanding ? 'expanding' : 'stable'}}</span>
      </div></div>
    </div>

    <div class="ind-grid">
      <div class="ind-box"><div class="ind-label">Structure</div><div class="tags">
        <span class="tag ${{d.higher_lows ? 'tag-green' : d.lower_highs ? 'tag-red' : 'tag-muted'}}">${{d.higher_lows ? 'Higher lows &#8599;' : d.lower_highs ? 'Lower highs &#8600;' : 'Neutral'}}</span>
      </div></div>
      <div class="ind-box"><div class="ind-label">Price Action</div><div class="tags">
        <span class="tag ${{d.pct_change > 0 ? 'tag-green' : 'tag-red'}}">${{d.pct_change > 0 ? '+' : ''}}${{d.pct_change.toFixed(1)}}% today</span>
        <span class="tag ${{d.gap_up_proxy ? 'tag-green' : d.gap_down_proxy ? 'tag-red' : 'tag-muted'}}">${{d.gap_up_proxy ? 'Gap-up' : d.gap_down_proxy ? 'Gap-down' : 'No gap'}}${{d.gap_is_proxy ? ' (proxy)' : ''}}</span>
      </div></div>
    </div>

    <div class="compression">
      <div class="comp-hdr"><div class="comp-icon"></div><div class="comp-title">Compression Analysis</div></div>
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
        <span class="squeeze-badge ${{inSqueeze ? 'sq-on' : 'sq-off'}}">${{inSqueeze ? '&#9888; IN SQUEEZE' : '&#10003; NO SQUEEZE'}}</span>
        <span class="coil-label">&#9650; ${{coilType}}</span>
        <span class="bias-label">Bias score: <span class="bias-val">+${{biasScore}}</span> / 100</span>
      </div>
      <div class="comp-metrics">
        <div class="cm"><div class="cm-label">BB Width</div><div class="cm-val">${{d.bb_width}}%</div><div class="cm-sub">${{d.bb_width < d.bb_width_avg ? 'Tight' : 'Stable'}}</div></div>
        <div class="cm"><div class="cm-label">BB Width Avg</div><div class="cm-val">${{d.bb_width_avg}}%</div><div class="cm-sub">20-day avg</div></div>
        <div class="cm"><div class="cm-label">Stoch K</div><div class="cm-val">${{d.stoch_k}}</div><div class="cm-sub">${{d.stoch_k > 80 ? 'Overbought' : d.stoch_k < 20 ? 'Oversold' : '&mdash;'}}</div></div>
        <div class="cm"><div class="cm-label">RSI</div><div class="cm-val">${{d.rsi}}</div><div class="cm-sub">${{d.rsi > 70 ? 'Overbought' : d.rsi < 30 ? 'Oversold' : '&mdash;'}}</div></div>
        <div class="cm"><div class="cm-label">52H %</div><div class="cm-val">${{d.pct_from_52h}}%</div><div class="cm-sub">From 52w high</div></div>
        <div class="cm"><div class="cm-label">ATR %</div><div class="cm-val">${{d.atr_proxy_pct.toFixed(2)}}%</div><div class="cm-sub">${{d.atr_expanding ? 'Expanding' : 'Stable'}}${{d.atr_is_proxy ? ' (proxy)' : ''}}</div></div>
      </div>
      <div class="tags" style="margin-top:16px;">
        <span class="tag ${{d.above_ma20 ? 'tag-green' : 'tag-red'}}">${{d.above_ma20 ? 'Above' : 'Below'}} MA20</span>
        <span class="tag ${{d.above_ma50 ? 'tag-green' : 'tag-red'}}">${{d.above_ma50 ? 'Above' : 'Below'}} MA50</span>
        <span class="tag ${{d.above_ma200 ? 'tag-green' : 'tag-red'}}">${{d.above_ma200 ? 'Above' : 'Below'}} MA200</span>
        <span class="tag ${{d.higher_lows ? 'tag-green' : 'tag-muted'}}">${{d.higher_lows ? 'Higher lows in range &#8599;' : 'No structure'}}</span>
        <span class="tag ${{d.golden_cross === 'YES' ? 'tag-green' : 'tag-muted'}}">${{d.golden_cross === 'YES' ? 'Golden Cross' : 'No golden cross'}}</span>
        <span class="tag ${{inSqueeze ? 'tag-amber' : 'tag-muted'}}">${{inSqueeze ? 'BB in squeeze' : 'No squeeze'}}</span>
        <span class="tag ${{d.gap_up_proxy ? 'tag-green' : d.gap_down_proxy ? 'tag-red' : 'tag-muted'}}">Gap ${{d.gap_is_proxy ? 'proxy ' : ''}}threshold ${{d.gap_proxy_thresh.toFixed(2)}}%</span>
      </div>
    </div>

    <div class="all-indicators">
      <div class="all-ind-title">All Indicators</div>
      <div class="ai-grid">
        <div class="ai-cell">RSI (14) &nbsp; <span class="val">${{d.rsi}}</span> &nbsp; <span class="dot" style="background:${{d.rsi >= 70 ? 'var(--red)' : d.rsi <= 30 ? 'var(--green)' : 'var(--muted)'}}"></span> <span class="st" style="color:${{d.rsi >= 70 ? 'var(--red)' : d.rsi <= 30 ? 'var(--green)' : 'var(--muted)'}}">${{rsiLabel}}</span></div>
        <div class="ai-cell">MACD Hist &nbsp; <span class="val">${{d.macd_hist}}</span> &nbsp; <span class="dot" style="background:${{d.macd_hist > 0 ? 'var(--green)' : 'var(--red)'}}"></span> <span class="st" style="color:${{d.macd_hist > 0 ? 'var(--green)' : 'var(--red)'}}">${{d.macd_trend}}</span></div>
        <div class="ai-cell">Stoch K &nbsp; <span class="val">${{d.stoch_k}}</span> &nbsp; <span class="st" style="color:${{d.stoch_k > 80 ? 'var(--red)' : d.stoch_k < 20 ? 'var(--green)' : 'var(--muted)'}}">${{d.stoch_k > 80 ? 'Overbought' : d.stoch_k < 20 ? 'Oversold' : '&mdash;'}}</span></div>
        <div class="ai-cell">BB %B &nbsp; <span class="val">${{d.bb_pct.toFixed(1)}}</span> &nbsp; <span class="st" style="color:${{d.bb_pct > 80 ? 'var(--cyan)' : 'var(--muted)'}}">${{d.bb_pct > 95 ? 'Upper break' : d.bb_pct > 80 ? 'Upper' : d.bb_pct < 20 ? 'Lower' : 'Mid'}}</span></div>

        <div class="ai-cell">BB Width &nbsp; <span class="val">${{d.bb_width}}%</span> &nbsp; <span class="st" style="color:var(--muted)">${{d.bb_width < d.bb_width_avg ? 'Tight' : 'Normal'}}</span></div>
        <div class="ai-cell">Golden Cross &nbsp; <span class="val">${{d.golden_cross}}</span> &nbsp; <span class="dot" style="background:${{d.golden_cross === 'YES' ? 'var(--green)' : 'var(--muted)'}}"></span> <span class="st" style="color:${{d.golden_cross === 'YES' ? 'var(--green)' : 'var(--muted)'}}">${{d.golden_cross === 'YES' ? 'Bullish' : '&mdash;'}}</span></div>
        <div class="ai-cell">Death Cross &nbsp; <span class="val">${{d.death_cross}}</span> &nbsp; <span class="st" style="color:${{d.death_cross === 'YES' ? 'var(--red)' : 'var(--muted)'}}">${{d.death_cross === 'YES' ? 'Bearish' : '&mdash;'}}</span></div>
        <div class="ai-cell">52H % &nbsp; <span class="val">${{d.pct_from_52h}}%</span> &nbsp; <span class="st" style="color:${{d.pct_from_52h > -5 ? 'var(--green)' : 'var(--amber)'}}">${{d.pct_from_52h > -5 ? 'Near high' : 'Off high'}}</span></div>

        <div class="ai-cell">Higher Lows &nbsp; <span class="val">${{d.higher_lows ? 'YES' : 'NO'}}</span> &nbsp; <span class="dot" style="background:${{d.higher_lows ? 'var(--green)' : 'var(--muted)'}}"></span></div>
        <div class="ai-cell">Lower Highs &nbsp; <span class="val">${{d.lower_highs ? 'YES' : 'NO'}}</span> &nbsp; <span class="dot" style="background:${{d.lower_highs ? 'var(--red)' : 'var(--muted)'}}"></span></div>
        <div class="ai-cell">Squeeze &nbsp; <span class="val">${{inSqueeze ? 'ON' : 'OFF'}}</span> &nbsp; <span class="dot" style="background:${{inSqueeze ? 'var(--amber)' : 'var(--muted)'}}"></span></div>
        <div class="ai-cell">Net Bias &nbsp; <span class="val" style="color:${{d.net_bias > 0 ? 'var(--green)' : d.net_bias < 0 ? 'var(--red)' : 'var(--muted)'}}">${{d.net_bias > 0 ? '+' : ''}}${{d.net_bias}}</span></div>
        <div class="ai-cell">Gekko GI &nbsp; <span class="val">${{d.gi_score === null ? 'N/A' : d.gi_score.toFixed(1)}}</span> &nbsp; <span class="st" style="color:var(--muted)">${{d.gi_label || 'N/A'}}</span></div>

        <div class="ai-cell">ATR (14) &nbsp; <span class="val">${{d.atr_proxy_pct.toFixed(2)}}%</span> &nbsp; <span class="st" style="color:${{d.atr_expanding ? 'var(--cyan)' : 'var(--muted)'}}">${{d.atr_expanding ? 'Expanding' : 'Stable'}}${{d.atr_is_proxy ? ' proxy' : ''}}</span></div>
        <div class="ai-cell">Gap-Up &nbsp; <span class="val">${{d.gap_up_proxy ? 'YES' : 'NO'}}</span> &nbsp; <span class="dot" style="background:${{d.gap_up_proxy ? 'var(--green)' : 'var(--muted)'}}"></span></div>
        <div class="ai-cell">Gap-Down &nbsp; <span class="val">${{d.gap_down_proxy ? 'YES' : 'NO'}}</span> &nbsp; <span class="dot" style="background:${{d.gap_down_proxy ? 'var(--red)' : 'var(--muted)'}}"></span></div>
        <div class="ai-cell">Gap Threshold &nbsp; <span class="val">${{d.gap_proxy_thresh.toFixed(2)}}%</span> &nbsp; <span class="st" style="color:var(--muted)">20D vol-adjusted</span></div>
        <div class="ai-cell">Gap % &nbsp; <span class="val">${{d.gap_pct > 0 ? '+' : ''}}${{d.gap_pct.toFixed(2)}}%</span> &nbsp; <span class="st" style="color:${{d.gap_pct >= 0 ? 'var(--green)' : 'var(--red)'}}">${{d.gap_is_proxy ? 'proxy' : 'open vs prev close'}}</span></div>
        <div class="ai-cell">RVOL (20) &nbsp; <span class="val">${{d.rvol20.toFixed(2)}}x</span> &nbsp; <span class="st" style="color:${{d.volume_is_proxy ? 'var(--muted)' : d.rvol20 >= 1.5 ? 'var(--cyan)' : 'var(--muted)'}}">${{d.volume_is_proxy ? 'n/a' : d.rvol20 >= 1.5 ? 'High' : 'Normal'}}</span></div>
        <div class="ai-cell">OBV Trend &nbsp; <span class="val">${{d.volume_is_proxy ? 'N/A' : (d.obv_up ? 'UP' : 'DOWN')}}</span> &nbsp; <span class="dot" style="background:${{d.volume_is_proxy ? 'var(--muted)' : d.obv_up ? 'var(--green)' : 'var(--red)'}}"></span></div>
        <div class="ai-cell">Vol Today &nbsp; <span class="val">${{d.vol_now.toLocaleString()}}</span> &nbsp; <span class="st" style="color:var(--muted)">Avg20 ${{d.vol_avg20.toLocaleString()}}</span></div>
      </div>
    </div>
  `;

  renderPriceChart(d);

  overlay.classList.add('open');
  overlay.scrollTop = 0;
  document.body.style.overflow = 'hidden';
}}

function renderPriceChart(d) {{
  const canvas = document.getElementById('priceChart');
  if (!canvas || !window.Chart) return;
  const ctx = canvas.getContext('2d');

  const labels = Array.isArray(d.chart_dates) ? d.chart_dates : [];
  const close = Array.isArray(d.chart_close) ? d.chart_close : [];
  const ma20 = Array.isArray(d.chart_ma20) ? d.chart_ma20 : [];
  const ma50 = Array.isArray(d.chart_ma50) ? d.chart_ma50 : [];
  const ma200 = Array.isArray(d.chart_ma200) ? d.chart_ma200 : [];
  const bbUpper = Array.isArray(d.chart_bb_upper) ? d.chart_bb_upper : [];
  const bbLower = Array.isArray(d.chart_bb_lower) ? d.chart_bb_lower : [];

  if (!labels.length || !close.length) return;

  if (_detailChart) {{
    _detailChart.destroy();
    _detailChart = null;
  }}

  const hasBreakout = typeof d.breakout_level === 'number' && Number.isFinite(d.breakout_level);
  const hasBreakdown = typeof d.breakdown_level === 'number' && Number.isFinite(d.breakdown_level);
  const breakoutLine = hasBreakout ? labels.map(() => d.breakout_level) : [];
  const breakdownLine = hasBreakdown ? labels.map(() => d.breakdown_level) : [];

  const lineLabelPlugin = {{
    id: 'lineLabelPlugin',
    afterDatasetsDraw(chart) {{
      const chartArea = chart.chartArea;
      const yScale = chart.scales.y;
      if (!chartArea || !yScale) return;

      const c = chart.ctx;
      c.save();
      c.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      c.textBaseline = 'middle';

      const drawLineTag = (label, value, bg, fg, offsetY = 0) => {{
        if (!Number.isFinite(value)) return;
        const y = yScale.getPixelForValue(value) + offsetY;
        if (y < chartArea.top + 8 || y > chartArea.bottom - 8) return;

        const text = `${{label}} $${{value.toFixed(2)}}`;
        const tw = c.measureText(text).width;
        const padX = 6;
        const h = 16;
        const w = tw + padX * 2;
        const x = chartArea.right - w - 6;

        c.fillStyle = bg;
        c.fillRect(x, y - h / 2, w, h);
        c.strokeStyle = 'rgba(15,23,42,0.65)';
        c.strokeRect(x, y - h / 2, w, h);

        c.fillStyle = fg;
        c.fillText(text, x + padX, y);
      }};

      if (hasBreakout) drawLineTag('BO', d.breakout_level, 'rgba(34,197,94,0.20)', '#86efac', -10);
      if (hasBreakdown) drawLineTag('BD', d.breakdown_level, 'rgba(239,68,68,0.20)', '#fca5a5', 10);

      c.restore();
    }}
  }};

  const closeFill = ctx.createLinearGradient(0, 0, 0, canvas.height || 340);
  closeFill.addColorStop(0, 'rgba(56, 189, 248, 0.22)');
  closeFill.addColorStop(1, 'rgba(56, 189, 248, 0.01)');

  _detailChart = new Chart(ctx, {{
    type: 'line',
    plugins: [lineLabelPlugin],
    data: {{
      labels,
      datasets: [
        {{ label: 'BB Upper', data: bbUpper, borderColor: 'rgba(56, 189, 248, 0.20)', borderWidth: 1, pointRadius: 0, tension: 0.18 }},
        {{ label: 'BB Lower', data: bbLower, borderColor: 'rgba(56, 189, 248, 0.20)', backgroundColor: 'rgba(56, 189, 248, 0.07)', borderWidth: 1, pointRadius: 0, fill: '-1', tension: 0.18 }},
        {{
          label: 'Close',
          data: close,
          borderColor: '#38bdf8',
          backgroundColor: closeFill,
          borderWidth: 2.2,
          pointRadius: 0,
          pointHoverRadius: 0,
          fill: true,
          tension: 0.2
        }},
        {{ label: 'MA20', data: ma20, borderColor: '#22d3ee', borderWidth: 1.35, pointRadius: 0, pointHoverRadius: 0, tension: 0.18 }},
        {{ label: 'MA50', data: ma50, borderColor: '#facc15', borderWidth: 1.25, pointRadius: 0, pointHoverRadius: 0, tension: 0.18 }},
        {{ label: 'MA200', data: ma200, borderColor: '#a78bfa', borderWidth: 1.2, pointRadius: 0, pointHoverRadius: 0, tension: 0.18 }},
        {{ label: 'Breakout', data: breakoutLine, borderColor: 'rgba(34, 197, 94, 0.62)', borderDash: [5, 4], borderWidth: 1, pointRadius: 0, pointHoverRadius: 0, hidden: !hasBreakout }},
        {{ label: 'Breakdown', data: breakdownLine, borderColor: 'rgba(239, 68, 68, 0.62)', borderDash: [5, 4], borderWidth: 1, pointRadius: 0, pointHoverRadius: 0, hidden: !hasBreakdown }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: {{ duration: 260 }},
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#0b1220',
          borderColor: '#334155',
          borderWidth: 1,
          titleColor: '#f8fafc',
          bodyColor: '#e2e8f0',
          padding: 10,
          cornerRadius: 8,
          displayColors: false,
          callbacks: {{
            title: (items) => items.length ? `Date: ${{items[0].label}}` : '',
            label: (tipCtx) => `${{tipCtx.dataset.label}}: $${{Number(tipCtx.parsed.y).toFixed(2)}}`
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{
            color: '#94a3b8',
            maxTicksLimit: 8,
            callback: (val, idx) => {{
              const lbl = labels[idx] || '';
              return lbl.length >= 7 ? lbl.slice(2, 7) : lbl;
            }}
          }},
          border: {{ color: 'rgba(148, 163, 184, 0.18)' }},
          grid: {{ color: 'rgba(148, 163, 184, 0.06)', drawTicks: false }}
        }},
        y: {{
          position: 'right',
          ticks: {{ color: '#94a3b8', padding: 6, callback: (v) => `$${{Number(v).toFixed(0)}}` }},
          border: {{ color: 'rgba(148, 163, 184, 0.18)' }},
          grid: {{ color: 'rgba(148, 163, 184, 0.06)' }}
        }}
      }}
    }}
  }});
}}

function closeDetail() {{
  if (_detailChart) {{
    _detailChart.destroy();
    _detailChart = null;
  }}
  document.getElementById('detailOverlay').classList.remove('open');
  document.body.style.overflow = '';
}}

// ESC to close
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeDetail(); }});
</script>
</body>
</html>"""
    return html


# ==========================================
# MAIN
# ==========================================

def main():
    global GEKKO_SCORE_MAP
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_master_tickers()
    GEKKO_SCORE_MAP = load_gekko_scores(GEKKO_SCREENER_FILE)
    if REFRESH_DATA_BEFORE_SCAN:
        ensure_shared_data(
            lookback_days=LOOKBACK_DAYS,
            chart_years=3,
            batch_size=BATCH_SIZE,
            cooldown=COOLDOWN,
        )
    else:
        print("Shared-data refresh disabled (read-only dashboard mode).")

    mode = "table"
    symbol = None

    if len(sys.argv) >= 2:
        arg = sys.argv[1].strip().upper()
        if arg in ("--SCAN", "--TABLE", "-S"):
            mode = "table"
        else:
            mode = "single"
            symbol = arg

    if mode == "single":
      # Single symbol detail from shared cache.
      print("\n=== Loading price caches ===")
      with tqdm(total=7, desc="Single dashboard pipeline", unit="step", ncols=88, file=sys.stdout) as p:
        stock_data = load_cache(STOCK_DATA_FILE)
        p.update(1)
        etf_data = load_cache(ETF_DATA_FILE)
        p.update(1)
        chart_open = load_cache(CHART_OPEN_FILE)
        p.update(1)
        chart_high = load_cache(CHART_HIGH_FILE)
        p.update(1)
        chart_low = load_cache(CHART_LOW_FILE)
        p.update(1)
        chart_volume = load_cache(CHART_VOLUME_FILE)
        p.update(1)

        if not stock_data.empty and not etf_data.empty:
            all_data = pd.concat([stock_data, etf_data], axis=1)
            all_data = all_data.loc[:, ~all_data.columns.duplicated()]
        elif not stock_data.empty:
            all_data = stock_data
        elif not etf_data.empty:
            all_data = etf_data
        else:
            print("No cache data found. Run market_data_maintainer.py first.")
            return

        symbol = (symbol or "").upper().strip()
        if symbol not in all_data.columns:
            print(f"{symbol} not found in cache.")
            return

        open_s = chart_open[symbol] if (not chart_open.empty and symbol in chart_open.columns) else None
        high_s = chart_high[symbol] if (not chart_high.empty and symbol in chart_high.columns) else None
        low_s = chart_low[symbol] if (not chart_low.empty and symbol in chart_low.columns) else None
        vol_s = chart_volume[symbol] if (not chart_volume.empty and symbol in chart_volume.columns) else None

        row = fast_scan_symbol(symbol, all_data[symbol].tail(LOOKBACK_DAYS), open_s, high_s, low_s, vol_s)
        p.update(1)
        if row is None:
            print("Not enough cached history for this symbol.")
            return

        with tqdm(total=3, desc="Finalizing single dashboard", unit="step", ncols=88, file=sys.stdout) as p:
          html = build_full_html([row])
          p.update(1)
          out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stock_dashboard_{symbol}.html")
          gz_path = write_dashboard_output(html, out_path)
          p.update(1)
          webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")
          p.update(1)
        print(f"\nDashboard: {out_path}")
        if gz_path:
            print(f"Compressed dashboard: {gz_path}")
        return
    # â”€â”€ TABLE MODE: Batch scan all tickers from cache â”€â”€
    print("\n=== Loading price caches ===")
    with tqdm(total=6, desc="Loading caches", unit="file", ncols=88, file=sys.stdout) as p:
      stock_data = load_cache(STOCK_DATA_FILE)
      p.update(1)
      etf_data = load_cache(ETF_DATA_FILE)
      p.update(1)
      chart_open = load_cache(CHART_OPEN_FILE)
      p.update(1)
      chart_high = load_cache(CHART_HIGH_FILE)
      p.update(1)
      chart_low = load_cache(CHART_LOW_FILE)
      p.update(1)
      chart_volume = load_cache(CHART_VOLUME_FILE)
      p.update(1)

    # Merge
    if not stock_data.empty and not etf_data.empty:
        all_data = pd.concat([stock_data, etf_data], axis=1)
        all_data = all_data.loc[:, ~all_data.columns.duplicated()]
    elif not stock_data.empty:
        all_data = stock_data
    elif not etf_data.empty:
        all_data = etf_data
    else:
        print("No cache data found. Run pairs_finder.py first to build caches.")
        return

    all_data = all_data.tail(LOOKBACK_DAYS)
    all_data = all_data.ffill().bfill()

    # Filter to tickers with enough data
    valid_cols = [c for c in all_data.columns if all_data[c].notna().sum() >= MIN_HISTORY_DAYS_FOR_SCAN]
    print(f"Valid tickers with {MIN_HISTORY_DAYS_FOR_SCAN}+ days: {len(valid_cols)}")
    gi_hits = sum(1 for c in valid_cols if any(a in GEKKO_SCORE_MAP for a in _ticker_aliases(c)))
    print(f"GI coverage in scan universe: {gi_hits}/{len(valid_cols)}")

    # Scan all
    print(f"\n=== Scanning {len(valid_cols)} tickers ===")
    results = []
    scan_bar = tqdm(valid_cols, total=len(valid_cols), desc="Scanning tickers", unit="ticker", ncols=88, file=sys.stdout)
    for sym in scan_bar:
        try:
            open_s = chart_open[sym] if (not chart_open.empty and sym in chart_open.columns) else None
            high_s = chart_high[sym] if (not chart_high.empty and sym in chart_high.columns) else None
            low_s = chart_low[sym] if (not chart_low.empty and sym in chart_low.columns) else None
            vol_s = chart_volume[sym] if (not chart_volume.empty and sym in chart_volume.columns) else None

            r = fast_scan_symbol(sym, all_data[sym], open_s, high_s, low_s, vol_s)
            if r is not None:
                results.append(r)
                if len(results) % 250 == 0:
                    scan_bar.set_postfix(valid=len(results))
        except Exception:
            pass
    scan_bar.close()

    print(f"\nScan complete: {len(results)} tickers scored")

    with tqdm(total=3, desc="Finalizing dashboard", unit="step", ncols=88, file=sys.stdout) as p:
      html = build_full_html(results)
      p.update(1)
      out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
      gz_path = write_dashboard_output(html, out_path)
      p.update(1)
      webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")
      p.update(1)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Dashboard saved: {out_path} ({size_mb:.1f}MB)")
    if gz_path:
      size_gz_mb = os.path.getsize(gz_path) / 1e6
      print(f"Compressed: {gz_path} ({size_gz_mb:.1f}MB)")


if __name__ == "__main__":
    main()

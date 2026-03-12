import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import webbrowser
from datetime import datetime, timedelta

# ==========================================
# CONFIG
# ==========================================
STOCK_DATA_FILE = "stock_data.csv.gz"
ETF_DATA_FILE   = "etf_data.csv.gz"
BATCH_SIZE      = 40
COOLDOWN        = 1.5
LOOKBACK_DAYS   = 400
OUTPUT_FILE     = "stock_dashboard.html"

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
        for _, row in df_etf.iterrows():
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
        for _, row in df_stock.iterrows():
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

def download_batch(tickers, start_date, field="Close"):
    clean = [t.replace("/", "-") for t in tickers]
    for attempt in range(3):
        try:
            df = yf.download(clean, start=start_date, progress=False,
                             group_by="ticker", auto_adjust=True, threads=False, timeout=20)
            if df is None or df.empty:
                if attempt < 2: time.sleep(5); continue
                return pd.DataFrame()
            result = pd.DataFrame()
            for t in clean:
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        if t in df.columns.levels[0]: result[t] = df[t][field]
                    else:
                        if not df[field].empty: result[t] = df[field]
                except Exception: continue
            return result
        except Exception as e:
            if attempt < 2: time.sleep(5)
            else: return pd.DataFrame()
    return pd.DataFrame()

def get_ohlcv_data(symbol):
    symbol = symbol.upper().strip()
    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS + 50)).strftime("%Y-%m-%d")
    try:
        df = yf.download(symbol, start=start, progress=False, auto_adjust=True, timeout=20)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.tail(LOOKBACK_DAYS)
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
    return None


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
# FAST SCAN (close-only from cache)
# ==========================================

def fast_scan_symbol(symbol, close_series):
    """Score a symbol using only close prices (no OHLCV download needed)."""
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

    # Approximate stochastic from close only (use close for high/low approximation)
    roll_high = close.rolling(14).max()
    roll_low = close.rolling(14).min()
    stoch_k = float(100 * (price - roll_low.iloc[-1]) / (roll_high.iloc[-1] - roll_low.iloc[-1])) if (roll_high.iloc[-1] - roll_low.iloc[-1]) > 0 else 50

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

    breakout = min(100, breakout)
    breakdown = min(100, breakdown)
    net_bias = breakout - breakdown

    sector = TICKER_INDUSTRY.get(symbol, "—")
    name = TICKER_NAMES.get(symbol, symbol)
    mcap = TICKER_CSV_MCAP.get(symbol, 0)
    ttype = TICKER_TYPES.get(symbol, "—")

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
        "net_bias": net_bias,
        "mcap": mcap,
    }


# ==========================================
# FULL DETAIL ANALYSIS (needs OHLCV)
# ==========================================

def analyze_symbol(symbol):
    """Run full technical analysis on a symbol (downloads OHLCV)."""
    print(f"Analyzing {symbol}...")
    ohlcv = get_ohlcv_data(symbol)
    if ohlcv is None or len(ohlcv) < 200:
        print(f"Not enough data for {symbol}")
        return None

    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    volume = ohlcv["Volume"].astype(float)
    today_open = float(ohlcv["Open"].iloc[-1])

    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    pct_change = ((price - prev_close) / prev_close) * 100
    gap_pct = ((today_open - prev_close) / prev_close) * 100

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi = calc_rsi(close)
    rsi_val = round(float(rsi.iloc[-1]), 1)
    macd_line, signal_line, macd_hist = calc_macd(close)
    macd_hist_val = round(float(macd_hist.iloc[-1]), 4)
    macd_hist_rising = bool(macd_hist.iloc[-1] > macd_hist.iloc[-2])
    stoch_k, stoch_d = calc_stochastic(high, low, close)
    stoch_k_val = round(float(stoch_k.iloc[-1]))
    bb_mid, bb_upper, bb_lower, bb_width = calc_bollinger(close)
    bb_width_val = round(float(bb_width.iloc[-1]), 1)
    bb_width_avg = round(float(bb_width.tail(20).mean()), 1)
    bb_pct = round(float((price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100), 1) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 50.0
    atr = calc_atr(high, low, close)
    atr_val = round(float(atr.iloc[-1]), 2)
    atr_pct = round((atr_val / price) * 100, 2)
    atr_expanding = bool(atr.iloc[-1] > atr.iloc[-5]) if len(atr) > 5 else False
    adx, plus_di, minus_di = calc_adx(high, low, close)
    adx_val = int(round(float(adx.iloc[-1])))
    plus_di_val = int(round(float(plus_di.iloc[-1])))
    minus_di_val = int(round(float(minus_di.iloc[-1])))
    obv = calc_obv(close, volume)
    obv_trend = round(float(((obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) * 100) if obv.iloc[-20] != 0 else 0), 1)
    obv_surging = obv_trend > 15
    vol_avg_20 = float(volume.tail(20).mean())
    rvol = round(float(volume.iloc[-1]) / vol_avg_20, 2) if vol_avg_20 > 0 else 1.0

    close_252 = close.tail(252)
    high_52w = float(close_252.max())
    low_52w = float(close_252.min())
    pct_from_52h = round(((price - high_52w) / high_52w) * 100, 1)

    above_ma20 = bool(price > ma20.iloc[-1])
    above_ma50 = bool(price > ma50.iloc[-1])
    above_ma200 = bool(price > ma200.iloc[-1])
    golden_cross, death_cross = detect_golden_death_cross(ma50.dropna(), ma200.dropna())
    higher_lows, lower_highs = detect_higher_lows(close)
    consolidating = is_consolidating(close)

    # Accumulation
    changes = close.diff().tail(20)
    vols = volume.tail(20)
    up_vol = float(vols[changes > 0].sum())
    down_vol = float(vols[changes <= 0].sum())
    if down_vol == 0: acc_dist = "Accumulating"
    elif up_vol / down_vol > 1.3: acc_dist = "Accumulating"
    elif up_vol / down_vol < 0.7: acc_dist = "Distributing"
    else: acc_dist = "—"
    acc_dist_pct = round(((up_vol - down_vol) / float(vols.sum())) * 100, 1) if float(vols.sum()) > 0 else 0
    acc_days = sum(1 for i in range(min(20, len(close)-1)) if close.iloc[-(i+1)] > close.iloc[-(i+2)] and volume.iloc[-(i+1)] > vol_avg_20)

    # Squeeze
    in_squeeze = bb_width_val < bb_width_avg
    squeeze_firing = in_squeeze and macd_hist.iloc[-1] > macd_hist.iloc[-2] and macd_hist.iloc[-2] > macd_hist.iloc[-3]
    was_in_squeeze = float(bb_width.tail(5).mean()) < bb_width_avg
    squeeze_fired = was_in_squeeze and bb_width_val > bb_width_avg and macd_hist_val > 0

    range_high = float(high.tail(20).max())
    range_low = float(low.tail(20).min())
    range_pct = round(((range_high - range_low) / float(close.tail(20).mean())) * 100, 1)
    price_in_range = int(round(((price - range_low) / (range_high - range_low)) * 100)) if (range_high - range_low) > 0 else 50

    # Scores
    breakout = 0; breakdown = 0
    if rsi_val > 60: breakout += 15
    if rsi_val > 70: breakout += 10
    if rsi_val < 40: breakdown += 15
    if rsi_val < 30: breakdown += 10
    if macd_hist_val > 0: breakout += 15
    if macd_hist_val > 0 and macd_hist_rising: breakout += 5
    if macd_hist_val < 0: breakdown += 15
    if macd_hist_val < 0 and not macd_hist_rising: breakdown += 5
    if stoch_k_val > 70: breakout += 10
    if stoch_k_val < 30: breakdown += 10
    if above_ma20: breakout += 8
    else: breakdown += 8
    if above_ma50: breakout += 8
    else: breakdown += 8
    if above_ma200: breakout += 8
    else: breakdown += 8
    if bb_pct > 95: breakout += 10
    if bb_pct < 5: breakdown += 10
    if rvol > 2.0 and pct_change > 0: breakout += 6
    if rvol > 2.0 and pct_change < 0: breakdown += 6
    if rvol > 1.5 and pct_change > 0: breakout += 4
    if higher_lows: breakout += 8
    if lower_highs: breakdown += 8
    if golden_cross == "YES": breakout += 8
    if death_cross == "YES": breakdown += 8
    breakout = min(100, breakout); breakdown = min(100, breakdown)
    net_bias = breakout - breakdown
    bias_score = round((breakout - breakdown) / 2 + 50)

    if in_squeeze or squeeze_fired:
        coil_type = "BULLISH COIL" if net_bias > 0 else "BEARISH COIL" if net_bias < 0 else "NEUTRAL COIL"
    else:
        coil_type = "BULLISH EXPANSION" if net_bias > 0 else "BEARISH EXPANSION"

    # Chart data
    chart_close = close.tail(252)
    chart_ma20 = ma20.tail(252); chart_ma50 = ma50.tail(252); chart_ma200 = ma200.tail(252)
    chart_bb_upper = bb_upper.tail(252); chart_bb_lower = bb_lower.tail(252); chart_vol = volume.tail(252)
    chart_data = []
    for i in range(len(chart_close)):
        chart_data.append({
            "date": chart_close.index[i].strftime("%Y-%m-%d"),
            "close": round(float(chart_close.iloc[i]), 2),
            "ma20": round(float(chart_ma20.iloc[i]), 2) if not np.isnan(chart_ma20.iloc[i]) else None,
            "ma50": round(float(chart_ma50.iloc[i]), 2) if not np.isnan(chart_ma50.iloc[i]) else None,
            "ma200": round(float(chart_ma200.iloc[i]), 2) if not np.isnan(chart_ma200.iloc[i]) else None,
            "bb_upper": round(float(chart_bb_upper.iloc[i]), 2) if not np.isnan(chart_bb_upper.iloc[i]) else None,
            "bb_lower": round(float(chart_bb_lower.iloc[i]), 2) if not np.isnan(chart_bb_lower.iloc[i]) else None,
            "volume": int(chart_vol.iloc[i]) if not np.isnan(chart_vol.iloc[i]) else 0,
        })

    ma_tags = []
    if above_ma20 and above_ma50 and above_ma200: ma_tags.append(">MA20+50+200")
    elif above_ma20 and above_ma50: ma_tags.append(">MA20+50")
    elif above_ma20: ma_tags.append(">MA20")
    elif not above_ma20 and not above_ma50 and not above_ma200: ma_tags.append("<MA20+50+200")

    return {
        "symbol": symbol, "name": TICKER_NAMES.get(symbol, symbol),
        "sector": TICKER_INDUSTRY.get(symbol, "—"),
        "index": "S&P500" if TICKER_CSV_MCAP.get(symbol, 0) > 10e9 else "Mid-Cap" if TICKER_CSV_MCAP.get(symbol, 0) > 2e9 else "Small-Cap",
        "price": round(price, 2), "pct_change": round(pct_change, 2), "gap_pct": round(gap_pct, 2),
        "rsi": rsi_val, "rvol": rvol,
        "macd_hist": macd_hist_val, "macd_trend": "Bullish" if macd_hist_val > 0 else "Bearish",
        "macd_hist_rising": macd_hist_rising, "stoch_k": stoch_k_val,
        "bb_width": bb_width_val, "bb_width_avg": bb_width_avg, "bb_pct": bb_pct,
        "atr": atr_val, "atr_pct": atr_pct, "atr_expanding": atr_expanding,
        "adx": adx_val, "plus_di": plus_di_val, "minus_di": minus_di_val,
        "obv_trend": obv_trend, "obv_surging": obv_surging,
        "high_52w": round(high_52w, 2), "low_52w": round(low_52w, 2), "pct_from_52h": pct_from_52h,
        "golden_cross": golden_cross, "death_cross": death_cross,
        "higher_lows": higher_lows, "lower_highs": lower_highs,
        "consolidating": consolidating, "acc_dist": acc_dist, "acc_dist_pct": acc_dist_pct,
        "above_ma20": above_ma20, "above_ma50": above_ma50, "above_ma200": above_ma200,
        "breakout_score": breakout, "breakdown_score": breakdown, "net_bias": net_bias,
        "bias_score": bias_score, "in_squeeze": in_squeeze,
        "squeeze_firing": bool(squeeze_firing or squeeze_fired), "coil_type": coil_type,
        "range_pct": range_pct, "price_in_range": price_in_range, "acc_days": acc_days,
        "ma20_val": round(float(ma20.iloc[-1]), 2), "ma50_val": round(float(ma50.iloc[-1]), 2),
        "ma200_val": round(float(ma200.iloc[-1]), 2), "daily_volatility": atr_pct,
        "ma_tags": ma_tags, "chart_data": chart_data,
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
    bullish = sum(1 for r in scan_results if r["net_bias"] > 30)
    bearish = sum(1 for r in scan_results if r["net_bias"] < -30)
    squeeze_count = sum(1 for r in scan_results if r["in_squeeze"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Dashboard — Breakout Scanner</title>
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
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; font-size: 14px; }}

  /* ── TOPBAR ── */
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

  /* ── STATS ROW ── */
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

  /* ── CONTROLS ── */
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

  /* ── TABLE ── */
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
  thead th.sort-asc::after {{ content: ' ▲'; color: var(--cyan); }}
  thead th.sort-desc::after {{ content: ' ▼'; color: var(--cyan); }}
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

  /* ── PAGINATION ── */
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

  /* ═══════════════════════════════════════
     DETAIL OVERLAY
     ═══════════════════════════════════════ */
  .overlay {{
    display: none; position: fixed; inset: 0; z-index: 500;
    background: var(--bg); overflow-y: auto;
  }}
  .overlay.open {{ display: block; }}
  .overlay-inner {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}

  .detail-header {{
    display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;
  }}
  .detail-header h1 {{ font-size: 32px; font-weight: 800; color: white; }}
  .detail-meta {{
    font-family: var(--mono); font-size: 12px; color: var(--muted); margin-top: 4px;
  }}
  .detail-meta .v {{ color: var(--text); }}
  .detail-meta .g {{ color: var(--green); }}
  .detail-meta .r {{ color: var(--red); }}
  .detail-meta .c {{ color: var(--cyan); }}
  .close-btn {{
    font-family: var(--mono); font-size: 13px; color: var(--muted); background: var(--surface2);
    border: 1px solid var(--border); padding: 8px 16px; border-radius: 6px; cursor: pointer;
  }}
  .close-btn:hover {{ color: var(--text); border-color: var(--border2); }}

  .scores {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 20px; }}
  .score-box {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px 20px; text-align: center;
  }}
  .score-label {{
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
  }}
  .score-val {{ font-family: var(--mono); font-size: 36px; font-weight: 700; }}

  .chart-section {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin-bottom: 20px; position: relative;
  }}
  .chart-hdr {{
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;
  }}
  .chart-title {{ font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }}
  .chart-legend {{ display: flex; gap: 14px; font-family: var(--mono); font-size: 10px; color: var(--muted); flex-wrap: wrap; }}
  .chart-legend span {{ display: flex; align-items: center; gap: 4px; }}
  .legend-line {{ width: 16px; height: 2px; border-radius: 1px; }}
  .legend-dash {{ width: 16px; height: 0; border-top: 2px dashed; }}
  .ma-labels {{
    position: absolute; left: 24px; top: 60px; font-family: var(--mono); font-size: 11px;
  }}
  .ma-labels div {{ margin-bottom: 4px; }}
  canvas#priceChart {{ width: 100% !important; height: 340px !important; }}

  .ind-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }}
  .ind-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }}
  .ind-label {{ font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .tag {{
    font-family: var(--mono); font-size: 11px; font-weight: 500;
    padding: 5px 12px; border-radius: 5px; white-space: nowrap;
  }}
  .tag-green {{ background: var(--green-dim); color: var(--green); border: 1px solid rgba(34,197,94,0.25); }}
  .tag-red {{ background: var(--red-dim); color: var(--red); border: 1px solid rgba(239,68,68,0.25); }}
  .tag-cyan {{ background: var(--cyan-dim); color: var(--cyan); border: 1px solid rgba(56,189,248,0.25); }}
  .tag-amber {{ background: rgba(245,158,11,0.12); color: var(--amber); border: 1px solid rgba(245,158,11,0.25); }}
  .tag-muted {{ background: rgba(74,85,104,0.12); color: var(--muted); border: 1px solid rgba(74,85,104,0.25); }}

  .compression {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; margin-bottom: 20px;
  }}
  .comp-hdr {{ display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }}
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
  .coil-label {{ font-size: 16px; font-weight: 700; color: white; }}
  .bias-label {{ font-family: var(--mono); font-size: 12px; color: var(--muted); }}
  .bias-val {{ color: var(--text); }}

  .comp-metrics {{
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px;
    background: var(--border); border-radius: 6px; overflow: hidden; margin-bottom: 16px;
  }}
  .cm {{ background: var(--surface2); padding: 12px 10px; }}
  .cm-label {{ font-family: var(--mono); font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }}
  .cm-val {{ font-family: var(--mono); font-size: 18px; font-weight: 600; color: white; }}
  .cm-sub {{ font-family: var(--mono); font-size: 9px; color: var(--muted); margin-top: 2px; }}

  .range-box {{
    background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
    padding: 12px 16px; display: inline-block; margin-bottom: 16px;
  }}
  .range-label {{ font-family: var(--mono); font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }}
  .range-value {{ font-family: var(--mono); font-size: 20px; font-weight: 600; color: white; }}
  .range-sub {{ font-family: var(--mono); font-size: 10px; color: var(--muted); }}

  .all-indicators {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; margin-top: 20px;
  }}
  .all-ind-title {{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; font-weight: 600; }}
  .ai-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; }}
  .ai-cell {{
    padding: 8px 12px; border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 12px; display: flex; align-items: center; gap: 8px;
  }}
  .ai-cell .val {{ color: white; font-weight: 600; }}
  .ai-cell .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
  .ai-cell .st {{ font-size: 11px; padding: 2px 8px; border-radius: 3px; }}

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
    <div class="brand"><span>BREAKOUT</span> SCANNER</div>
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
  <div class="stat-item"><div class="stat-label">Bullish (30+)</div><div class="stat-value green">{bullish}</div></div>
  <div class="stat-item"><div class="stat-label">Bearish (-30)</div><div class="stat-value red">{bearish}</div></div>
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
    <option value="bull">Bullish (30+)</option>
    <option value="strong_bull">Strong Bull (60+)</option>
    <option value="bear">Bearish (-30)</option>
    <option value="neutral">Neutral</option>
    <option value="squeeze">In Squeeze</option>
  </select>
  <span class="ctrl-label">Min Price</span>
  <input class="ctrl-input" id="minPrice" type="number" value="5" style="width:70px;" min="0">
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
        <th data-col="net_bias">Net Bias</th>
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
let page = 0;
const PER_PAGE = 50;

// ── Populate sectors ──
const sectors = [...new Set(ALL_DATA.map(r => r.sector).filter(s => s && s !== '—'))].sort();
const sf = document.getElementById('sectorFilter');
sectors.forEach(s => {{ const o = document.createElement('option'); o.value = s; o.textContent = s; sf.appendChild(o); }});

// ── Filter + Sort ──
function applyFilters() {{
  const q = document.getElementById('searchBox').value.toLowerCase();
  const typ = document.getElementById('typeFilter').value;
  const sec = document.getElementById('sectorFilter').value;
  const bias = document.getElementById('biasFilter').value;
  const minP = parseFloat(document.getElementById('minPrice').value) || 0;

  filtered = ALL_DATA.filter(r => {{
    if (q && !r.symbol.toLowerCase().includes(q) && !r.name.toLowerCase().includes(q)) return false;
    if (typ && r.type !== typ) return false;
    if (sec && r.sector !== sec) return false;
    if (r.price < minP) return false;
    if (bias === 'bull' && r.net_bias < 30) return false;
    if (bias === 'strong_bull' && r.net_bias < 60) return false;
    if (bias === 'bear' && r.net_bias > -30) return false;
    if (bias === 'neutral' && (r.net_bias > 30 || r.net_bias < -30)) return false;
    if (bias === 'squeeze' && !r.in_squeeze) return false;
    return true;
  }});
  doSort();
  page = 0;
  render();
}}

function doSort() {{
  filtered.sort((a, b) => {{
    let va = a[sortCol], vb = b[sortCol];
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
      <td class="score-cell" style="color:${{r.net_bias > 0 ? 'var(--green)' : r.net_bias < 0 ? 'var(--red)' : 'var(--muted)'}}">${{r.net_bias > 0 ? '+' : ''}}${{r.net_bias}}</td>
      <td><span class="tag-sm ${{r.rsi >= 70 ? 'red' : r.rsi >= 60 ? 'green' : r.rsi <= 30 ? 'red' : 'muted'}}">${{r.rsi}}</span></td>
      <td><span class="tag-sm ${{r.macd_trend === 'Bullish' ? 'green' : 'red'}}">${{r.macd_trend === 'Bullish' ? 'Bull' : 'Bear'}}</span></td>
      <td><span class="tag-sm ${{r.stoch_k >= 70 ? 'green' : r.stoch_k <= 30 ? 'red' : 'muted'}}">${{r.stoch_k}}</span></td>
      <td><span class="tag-sm ${{r.bb_pct > 90 ? 'cyan' : r.bb_pct < 10 ? 'red' : 'muted'}}">${{r.bb_pct.toFixed(0)}}%</span></td>
      <td style="color:${{r.pct_from_52h > -5 ? 'var(--green)' : r.pct_from_52h > -15 ? 'var(--amber)' : 'var(--red)'}}">${{r.pct_from_52h}}%</td>
      <td><span class="tag-sm ${{r.in_squeeze ? 'amber' : 'muted'}}">${{r.in_squeeze ? 'YES' : '—'}}</span></td>
      <td><span class="tag-sm ${{r.higher_lows ? 'green' : r.lower_highs ? 'red' : 'muted'}}">${{r.higher_lows ? 'HL ↗' : r.lower_highs ? 'LH ↘' : '—'}}</span></td>
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

// ── Column sorting ──
document.querySelectorAll('thead th').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (sortCol === col) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    else {{ sortCol = col; sortDir = 'desc'; }}
    doSort();
    render();
  }});
}});

// ── Filter listeners ──
document.getElementById('searchBox').addEventListener('input', applyFilters);
document.getElementById('typeFilter').addEventListener('change', applyFilters);
document.getElementById('sectorFilter').addEventListener('change', applyFilters);
document.getElementById('biasFilter').addEventListener('change', applyFilters);
document.getElementById('minPrice').addEventListener('change', applyFilters);

// ── Initial render ──
doSort();
render();


// ═══════════════════════════════════════
// DETAIL OVERLAY
// ═══════════════════════════════════════
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
  // we render from the scan data we already have (close-only version)
  const row = ALL_DATA.find(r => r.symbol === symbol);
  if (!row) return;

  // Render a close-only detail (no OHLCV needed)
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

  const biasScore = Math.round((d.breakout - d.breakdown) / 2 + 50);
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
          <span>RSI <span class="c">${{d.rsi}}</span></span>
        </div>
      </div>
      <button class="close-btn" onclick="closeDetail()">&#10005; Close</button>
    </div>

    <div class="scores">
      <div class="score-box"><div class="score-label">&#9650; Breakout</div><div class="score-val" style="color:var(--green)">${{d.breakout}}</div></div>
      <div class="score-box"><div class="score-label">&#9660; Breakdown</div><div class="score-val" style="color:${{d.breakdown > 0 ? 'var(--red)' : 'var(--muted)'}}">${{d.breakdown}}</div></div>
      <div class="score-box"><div class="score-label">&#9830; Net Bias</div><div class="score-val" style="color:${{d.net_bias > 0 ? 'var(--green)' : d.net_bias < 0 ? 'var(--red)' : 'var(--muted)'}}">${{d.net_bias > 0 ? '+' : ''}}${{d.net_bias}}</div></div>
    </div>

    <div class="ind-grid">
      <div class="ind-box"><div class="ind-label">Momentum</div><div class="tags">
        <span class="tag ${{rsiHot ? 'tag-green' : d.rsi < 35 ? 'tag-red' : 'tag-muted'}}">RSI ${{d.rsi}} ${{rsiHot ? '— hot' : d.rsi < 35 ? '— cold' : ''}}</span>
        <span class="tag ${{d.macd_trend === 'Bullish' ? 'tag-green' : 'tag-red'}}">MACD ${{macdLabel}}</span>
        <span class="tag ${{d.stoch_k >= 50 ? 'tag-green' : 'tag-red'}}">Stoch ${{d.stoch_k}} ${{stochLabel}}</span>
      </div></div>
      <div class="ind-box"><div class="ind-label">Volume</div><div class="tags">
        <span class="tag tag-muted">Close-only scan</span>
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
      </div></div>
    </div>

    <div class="ind-grid">
      <div class="ind-box"><div class="ind-label">Structure</div><div class="tags">
        <span class="tag ${{d.higher_lows ? 'tag-green' : d.lower_highs ? 'tag-red' : 'tag-muted'}}">${{d.higher_lows ? 'Higher lows ↗' : d.lower_highs ? 'Lower highs ↘' : 'Neutral'}}</span>
      </div></div>
      <div class="ind-box"><div class="ind-label">Price Action</div><div class="tags">
        <span class="tag ${{d.pct_change > 0 ? 'tag-green' : 'tag-red'}}">${{d.pct_change > 0 ? '+' : ''}}${{d.pct_change.toFixed(1)}}% today</span>
      </div></div>
    </div>

    <div class="compression">
      <div class="comp-hdr"><div class="comp-icon"></div><div class="comp-title">Compression Analysis</div></div>
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
        <span class="squeeze-badge ${{inSqueeze ? 'sq-on' : 'sq-off'}}">${{inSqueeze ? '⚠ IN SQUEEZE' : '✓ NO SQUEEZE'}}</span>
        <span class="coil-label">▲ ${{coilType}}</span>
        <span class="bias-label">Bias score: <span class="bias-val">+${{biasScore}}</span> / 100</span>
      </div>
      <div class="comp-metrics">
        <div class="cm"><div class="cm-label">BB Width</div><div class="cm-val">${{d.bb_width}}%</div><div class="cm-sub">${{d.bb_width < d.bb_width_avg ? 'Tight' : 'Stable'}}</div></div>
        <div class="cm"><div class="cm-label">BB Width Avg</div><div class="cm-val">${{d.bb_width_avg}}%</div><div class="cm-sub">20-day avg</div></div>
        <div class="cm"><div class="cm-label">Stoch K</div><div class="cm-val">${{d.stoch_k}}</div><div class="cm-sub">${{d.stoch_k > 80 ? 'Overbought' : d.stoch_k < 20 ? 'Oversold' : '—'}}</div></div>
        <div class="cm"><div class="cm-label">RSI</div><div class="cm-val">${{d.rsi}}</div><div class="cm-sub">${{d.rsi > 70 ? 'Overbought' : d.rsi < 30 ? 'Oversold' : '—'}}</div></div>
        <div class="cm"><div class="cm-label">52H %</div><div class="cm-val">${{d.pct_from_52h}}%</div><div class="cm-sub">From 52w high</div></div>
        <div class="cm"><div class="cm-label">MACD</div><div class="cm-val">${{d.macd_hist}}</div><div class="cm-sub">${{d.macd_trend}}</div></div>
      </div>
      <div class="tags" style="margin-top:16px;">
        <span class="tag ${{d.above_ma20 ? 'tag-green' : 'tag-red'}}">${{d.above_ma20 ? 'Above' : 'Below'}} MA20</span>
        <span class="tag ${{d.above_ma50 ? 'tag-green' : 'tag-red'}}">${{d.above_ma50 ? 'Above' : 'Below'}} MA50</span>
        <span class="tag ${{d.above_ma200 ? 'tag-green' : 'tag-red'}}">${{d.above_ma200 ? 'Above' : 'Below'}} MA200</span>
        <span class="tag ${{d.higher_lows ? 'tag-green' : 'tag-muted'}}">${{d.higher_lows ? 'Higher lows in range ↗' : 'No structure'}}</span>
        <span class="tag ${{d.golden_cross === 'YES' ? 'tag-green' : 'tag-muted'}}">${{d.golden_cross === 'YES' ? 'Golden Cross ☀' : 'No golden cross'}}</span>
        <span class="tag ${{inSqueeze ? 'tag-amber' : 'tag-muted'}}">${{inSqueeze ? 'BB in squeeze' : 'No squeeze'}}</span>
      </div>
    </div>

    <div class="all-indicators">
      <div class="all-ind-title">All Indicators</div>
      <div class="ai-grid">
        <div class="ai-cell">RSI (14) &nbsp; <span class="val">${{d.rsi}}</span> &nbsp; <span class="dot" style="background:${{d.rsi >= 70 ? 'var(--red)' : d.rsi <= 30 ? 'var(--green)' : 'var(--muted)'}}"></span> <span class="st" style="color:${{d.rsi >= 70 ? 'var(--red)' : d.rsi <= 30 ? 'var(--green)' : 'var(--muted)'}}">${{rsiLabel}}</span></div>
        <div class="ai-cell">MACD Hist &nbsp; <span class="val">${{d.macd_hist}}</span> &nbsp; <span class="dot" style="background:${{d.macd_hist > 0 ? 'var(--green)' : 'var(--red)'}}"></span> <span class="st" style="color:${{d.macd_hist > 0 ? 'var(--green)' : 'var(--red)'}}">${{d.macd_trend}}</span></div>
        <div class="ai-cell">Stoch K &nbsp; <span class="val">${{d.stoch_k}}</span> &nbsp; <span class="st" style="color:${{d.stoch_k > 80 ? 'var(--red)' : d.stoch_k < 20 ? 'var(--green)' : 'var(--muted)'}}">${{d.stoch_k > 80 ? 'Overbought' : d.stoch_k < 20 ? 'Oversold' : '—'}}</span></div>
        <div class="ai-cell">BB %B &nbsp; <span class="val">${{d.bb_pct.toFixed(1)}}</span> &nbsp; <span class="st" style="color:${{d.bb_pct > 80 ? 'var(--cyan)' : 'var(--muted)'}}">${{d.bb_pct > 95 ? 'Upper break' : d.bb_pct > 80 ? 'Upper' : d.bb_pct < 20 ? 'Lower' : 'Mid'}}</span></div>

        <div class="ai-cell">BB Width &nbsp; <span class="val">${{d.bb_width}}%</span> &nbsp; <span class="st" style="color:var(--muted)">${{d.bb_width < d.bb_width_avg ? 'Tight' : 'Normal'}}</span></div>
        <div class="ai-cell">Golden Cross &nbsp; <span class="val">${{d.golden_cross}}</span> &nbsp; <span class="dot" style="background:${{d.golden_cross === 'YES' ? 'var(--green)' : 'var(--muted)'}}"></span> <span class="st" style="color:${{d.golden_cross === 'YES' ? 'var(--green)' : 'var(--muted)'}}">${{d.golden_cross === 'YES' ? '☀ Bullish' : '—'}}</span></div>
        <div class="ai-cell">Death Cross &nbsp; <span class="val">${{d.death_cross}}</span> &nbsp; <span class="st" style="color:var(--muted)">—</span></div>
        <div class="ai-cell">52H % &nbsp; <span class="val">${{d.pct_from_52h}}%</span> &nbsp; <span class="st" style="color:${{d.pct_from_52h > -5 ? 'var(--green)' : 'var(--amber)'}}">${{d.pct_from_52h > -5 ? 'Near high' : 'Off high'}}</span></div>

        <div class="ai-cell">Higher Lows &nbsp; <span class="val">${{d.higher_lows ? 'YES' : 'NO'}}</span> &nbsp; <span class="dot" style="background:${{d.higher_lows ? 'var(--green)' : 'var(--muted)'}}"></span></div>
        <div class="ai-cell">Lower Highs &nbsp; <span class="val">${{d.lower_highs ? 'YES' : 'NO'}}</span> &nbsp; <span class="dot" style="background:${{d.lower_highs ? 'var(--red)' : 'var(--muted)'}}"></span></div>
        <div class="ai-cell">Squeeze &nbsp; <span class="val">${{inSqueeze ? 'ON' : 'OFF'}}</span> &nbsp; <span class="dot" style="background:${{inSqueeze ? 'var(--amber)' : 'var(--muted)'}}"></span></div>
        <div class="ai-cell">Net Bias &nbsp; <span class="val" style="color:${{d.net_bias > 0 ? 'var(--green)' : d.net_bias < 0 ? 'var(--red)' : 'var(--muted)'}}">${{d.net_bias > 0 ? '+' : ''}}${{d.net_bias}}</span></div>
      </div>
    </div>
  `;

  overlay.classList.add('open');
  overlay.scrollTop = 0;
  document.body.style.overflow = 'hidden';
}}

function closeDetail() {{
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
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_master_tickers()

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
        # Single symbol detail (same as before, downloads OHLCV)
        data = analyze_symbol(symbol)
        if data is None:
            print("Analysis failed.")
            return
        # Wrap in a minimal page that auto-opens detail
        scan_row = {
            "symbol": data["symbol"], "name": data["name"], "sector": data["sector"],
            "type": TICKER_TYPES.get(symbol, "—"), "price": data["price"],
            "pct_change": data["pct_change"], "rsi": data["rsi"],
            "macd_hist": data["macd_hist"], "macd_trend": data["macd_trend"],
            "macd_rising": data["macd_hist_rising"], "stoch_k": data["stoch_k"],
            "bb_width": data["bb_width"], "bb_width_avg": data["bb_width_avg"],
            "bb_pct": data["bb_pct"], "pct_from_52h": data["pct_from_52h"],
            "above_ma20": data["above_ma20"], "above_ma50": data["above_ma50"],
            "above_ma200": data["above_ma200"], "golden_cross": data["golden_cross"],
            "death_cross": data["death_cross"], "higher_lows": data["higher_lows"],
            "lower_highs": data["lower_highs"], "in_squeeze": data["in_squeeze"],
            "breakout": data["breakout_score"], "breakdown": data["breakdown_score"],
            "net_bias": data["net_bias"], "mcap": TICKER_CSV_MCAP.get(symbol, 0),
        }
        html = build_full_html([scan_row])
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stock_dashboard_{symbol}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nDashboard: {out_path}")
        webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")
        return

    # ── TABLE MODE: Batch scan all tickers from cache ──
    print("\n=== Loading price caches ===")
    stock_data = load_cache(STOCK_DATA_FILE)
    etf_data = load_cache(ETF_DATA_FILE)

    # Merge
    if not stock_data.empty and not etf_data.empty:
        all_data = pd.concat([stock_data, etf_data], axis=1)
        all_data = all_data.loc[:, ~all_data.columns.duplicated()]
    elif not stock_data.empty:
        all_data = stock_data
    elif not etf_data.empty:
        all_data = etf_data
    else:
        print("No cache data found. Run pairs_watchlist.py first to build caches.")
        return

    all_data = all_data.tail(LOOKBACK_DAYS)
    all_data = all_data.ffill().bfill()

    # Filter to tickers with enough data
    valid_cols = [c for c in all_data.columns if all_data[c].notna().sum() >= 210]
    print(f"Valid tickers with 210+ days: {len(valid_cols)}")

    # Scan all
    print(f"\n=== Scanning {len(valid_cols)} tickers ===")
    results = []
    done = 0
    total = len(valid_cols)

    for sym in valid_cols:
        try:
            r = fast_scan_symbol(sym, all_data[sym])
            if r is not None:
                results.append(r)
        except Exception:
            pass
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{total} scanned... ({len(results)} valid)")

    print(f"\nScan complete: {len(results)} tickers scored")

    # Build HTML
    html = build_full_html(results)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Dashboard saved: {out_path} ({size_mb:.1f}MB)")
    webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()

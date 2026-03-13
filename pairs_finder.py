import itertools
import json
import multiprocessing as mp
import os
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from market_data_maintainer import _dropped_tickers, validate_and_repair


# ==========================================
# CONFIG
# ==========================================
STOCK_DATA_FILE = "stock_data.csv.gz"
ETF_DATA_FILE = "etf_data.csv.gz"
ETF_CSV_FILE = "etfs.csv"
STOCK_CSV_FILE = "stocks.csv"
OUTPUT_FILE = "pairs_finder.html"

LOOKBACK_DAYS = 504
CHART_LOOKBACK_DAYS = 504
MAX_CHART_PAIRS = 800

# Candidate prefilter mode:
# - all: analyze every possible pair
# - related_first: analyze same-sector/industry pairs first, then all others
# - related_only: only analyze same-sector/industry pairs (fastest)
PAIR_PREFILTER_MODE = "related_first"

MIN_PRICE = 1.00
MIN_AVG_VOLUME = 50000
MIN_CORR_FILTER = 0.50

ADF_CONFIDENCE = 0.90
ADF_MIN_DAYS = 252
MIN_COINT_YEARS = 1

NUM_WORKERS = max(1, (mp.cpu_count() or 2) - 0)


# ==========================================
# TICKER METADATA
# ==========================================
TICKER_TYPES = {}
TICKER_NAMES = {}
TICKER_SECTOR = {}
TICKER_INDUSTRY = {}
TICKER_SUBINDUSTRY = {}
TICKER_CSV_MCAP = {}
TICKER_CSV_VOL = {}


_EXCHANGE_CODES = {
    "NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "OTC", "CBOE", "NYSEARCA", "TSX", "TSXV"
}

_ETF_TYPE_MAP = {
    "etf": "normal",
    "etf, leveraged": "leveraged",
    "etf, inverse": "inverse",
    "etf, leveraged, inverse": "lev_inv",
    "etn": "etn",
    "etn, leveraged": "etn_lev",
    "etn, leveraged, inverse": "etn_lev_inv",
}


# ==========================================
# CSV HELPERS
# ==========================================
def _normalize_col_name(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _find_col(colmap, options):
    for opt in options:
        key = _normalize_col_name(opt)
        if key in colmap:
            return colmap[key]
    return None


def _is_ticker(s):
    return bool(re.match(r"^[A-Z]{1,6}$", str(s).strip().upper()))


def _is_exchange(s):
    return str(s).strip().upper() in _EXCHANGE_CODES


def _is_mcap(s):
    v = str(s).strip()
    return bool(re.match(r"\$?[\d,\.]+\s*(T|B|M|K)\b", v, re.IGNORECASE)) or v.startswith("$")


def _is_volume(s):
    v = str(s).strip().replace(",", "")
    return bool(re.match(r"^\d+(\.\d+)?$", v))


def _is_etf_type(s):
    v = str(s).lower()
    return ("etf" in v) or ("etn" in v) or ("leveraged" in v) or ("inverse" in v)


def _parse_mcap_str(s):
    txt = str(s).strip()
    m = re.match(r"\$?([\d.]+)\s*(T|B|M|K)\b", txt, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        suffix = m.group(2).upper()
        if suffix == "T":
            val *= 1e12
        elif suffix == "B":
            val *= 1e9
        elif suffix == "M":
            val *= 1e6
        elif suffix == "K":
            val *= 1e3
        return int(val)
    raw = txt.replace("$", "").replace(",", "").strip()
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return 0


def _parse_vol_str(s):
    raw = str(s).strip().replace(",", "")
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return 0
def _read_csv_guess_header(path):
    if not os.path.exists(path):
        return pd.DataFrame(), False
    df_raw = pd.read_csv(path, header=None)
    if df_raw.empty:
        return df_raw, False

    first_row = df_raw.iloc[0].astype(str).str.strip().str.lower().tolist()
    header_tokens = {
        "ticker", "symbol", "name", "sector", "industry", "subindustry",
        "exchange", "mcap", "market cap", "total assets", "type", "fund type",
        "avg volume", "average volume", "volume",
    }
    if any(cell in header_tokens for cell in first_row):
        return pd.read_csv(path), True
    return df_raw, False


def _col_ratio(series, fn):
    vals = series.astype(str).str.strip()
    if len(vals) == 0:
        return 0.0
    hits = sum(1 for v in vals if v and fn(v))
    return hits / len(vals)


def _avg_len(series):
    vals = series.astype(str).str.strip()
    lens = [len(v) for v in vals if v]
    return (sum(lens) / len(lens)) if lens else 0.0


def _build_colmap_by_header(df, kind):
    colmap = {_normalize_col_name(c): c for c in df.columns}
    if kind == "etf":
        return {
            "ticker": _find_col(colmap, ["ticker", "symbol"]),
            "name": _find_col(colmap, ["name", "fund name"]),
            "type": _find_col(colmap, ["type", "fund type", "etf type"]),
            "sector": _find_col(colmap, ["sector", "category", "fund category"]),
            "industry": _find_col(colmap, ["industry", "morningstar", "subindustry", "sub-industry"]),
            "mcap": _find_col(colmap, ["total assets", "assets", "aum", "market cap", "mcap"]),
            "volume": _find_col(colmap, ["average volume", "avg volume", "volume", "10 day avg volume", "avg vol"]),
        }

    return {
        "ticker": _find_col(colmap, ["ticker", "symbol"]),
        "name": _find_col(colmap, ["name", "company", "security name"]),
        "sector": _find_col(colmap, ["sector"]),
        "industry": _find_col(colmap, ["industry"]),
        "subindustry": _find_col(colmap, ["subindustry", "sub-industry", "sub industry"]),
        "exchange": _find_col(colmap, ["exchange", "exch"]),
        "mcap": _find_col(colmap, ["market cap", "mcap", "marketcap"]),
        "volume": _find_col(colmap, ["average volume", "avg volume", "volume", "10 day avg volume", "avg vol"]),
    }


def _build_colmap_heuristic(df, kind):
    str_df = df.fillna("").astype(str)
    cols = list(range(str_df.shape[1]))

    # Known local CSV layouts (no header):
    # STOCKS: ticker,name,sector,industry,subindustry,exchange,mcap,avg_volume
    # ETFs:   ticker,name,type,sector,industry,mcap,avg_volume
    if kind == "stock" and len(cols) >= 8:
        return {
            "ticker": 0,
            "name": 1,
            "sector": 2,
            "industry": 3,
            "subindustry": 4,
            "exchange": 5,
            "mcap": 6,
            "volume": 7,
        }
    if kind == "etf" and len(cols) >= 7:
        return {
            "ticker": 0,
            "name": 1,
            "type": 2,
            "sector": 3,
            "industry": 4,
            "mcap": 5,
            "volume": 6,
        }

    ratios = {}
    for i in cols:
        col = str_df.iloc[:, i]
        ratios[i] = {
            "ticker": _col_ratio(col, _is_ticker),
            "exchange": _col_ratio(col, _is_exchange),
            "mcap": _col_ratio(col, _is_mcap),
            "volume": _col_ratio(col, _is_volume),
            "type": _col_ratio(col, _is_etf_type),
            "avg_len": _avg_len(col),
        }

    def _best_idx(key):
        return max(cols, key=lambda idx: ratios[idx][key])

    chosen = set()
    ticker_idx = _best_idx("ticker")
    ticker_idx = ticker_idx if ratios[ticker_idx]["ticker"] >= 0.5 else None
    if ticker_idx is not None:
        chosen.add(ticker_idx)

    exchange_idx = None
    if kind == "stock":
        exchange_idx = _best_idx("exchange")
        if ratios[exchange_idx]["exchange"] < 0.4:
            exchange_idx = None
        if exchange_idx is not None:
            chosen.add(exchange_idx)

    mcap_idx = _best_idx("mcap")
    if ratios[mcap_idx]["mcap"] < 0.3:
        mcap_idx = None
    if mcap_idx is not None:
        chosen.add(mcap_idx)

    volume_idx = _best_idx("volume")
    if ratios[volume_idx]["volume"] < 0.3:
        volume_idx = None
    if volume_idx is not None:
        chosen.add(volume_idx)

    type_idx = None
    if kind == "etf":
        type_idx = _best_idx("type")
        if ratios[type_idx]["type"] < 0.4:
            type_idx = None
        if type_idx is not None:
            chosen.add(type_idx)

    remaining = [i for i in cols if i not in chosen]
    name_idx = max(remaining, key=lambda idx: ratios[idx]["avg_len"]) if remaining else None
    if name_idx is not None:
        chosen.add(name_idx)

    remaining = [i for i in cols if i not in chosen]
    remaining.sort()

    if kind == "etf":
        sector_idx = remaining[0] if len(remaining) > 0 else None
        industry_idx = remaining[1] if len(remaining) > 1 else None
        return {
            "ticker": ticker_idx,
            "name": name_idx,
            "type": type_idx,
            "sector": sector_idx,
            "industry": industry_idx,
            "mcap": mcap_idx,
            "volume": volume_idx,
        }

    sector_idx = remaining[0] if len(remaining) > 0 else None
    industry_idx = remaining[1] if len(remaining) > 1 else None
    subindustry_idx = remaining[2] if len(remaining) > 2 else None
    return {
        "ticker": ticker_idx,
        "name": name_idx,
        "sector": sector_idx,
        "industry": industry_idx,
        "subindustry": subindustry_idx,
        "exchange": exchange_idx,
        "mcap": mcap_idx,
        "volume": volume_idx,
    }


def _load_ticker_csv(path, kind):
    df, has_header = _read_csv_guess_header(path)
    if df.empty:
        return df, {}
    df = df.fillna("")
    colmap = _build_colmap_by_header(df, kind) if has_header else _build_colmap_heuristic(df, kind)
    return df, colmap


def _get_row_value(row, key, colmap):
    col = colmap.get(key)
    if col is None:
        return ""
    try:
        return row[col] if isinstance(col, str) else row.iloc[col]
    except Exception:
        return ""


def load_master_tickers():
    tickers = []

    df_etf, etf_cols = _load_ticker_csv(ETF_CSV_FILE, "etf")
    if df_etf.empty:
        print(f"[!] Missing or empty {ETF_CSV_FILE}.")
    else:
        etf_lev_types = {}
        for _, row in df_etf.iterrows():
            t = str(_get_row_value(row, "ticker", etf_cols)).strip().upper()
            if not t or t in {"", "NONE", "NAN", "SYMBOL", "TICKER"}:
                continue

            tickers.append(t)
            TICKER_TYPES[t] = "Pure ETF"

            name = str(_get_row_value(row, "name", etf_cols)).strip()
            if name:
                TICKER_NAMES[t] = name

            raw_type = str(_get_row_value(row, "type", etf_cols)).strip().lower()
            etf_lev_types[t] = _ETF_TYPE_MAP.get(raw_type, "normal") if raw_type else "normal"

            sector = str(_get_row_value(row, "sector", etf_cols)).strip()
            if sector:
                TICKER_SECTOR[t] = sector

            industry = str(_get_row_value(row, "industry", etf_cols)).strip()
            if industry:
                TICKER_INDUSTRY[t] = industry

            mcap = _parse_mcap_str(_get_row_value(row, "mcap", etf_cols))
            if mcap > 0:
                TICKER_CSV_MCAP[t] = mcap

            vol = _parse_vol_str(_get_row_value(row, "volume", etf_cols))
            if vol > 0:
                TICKER_CSV_VOL[t] = vol

        n_lev = sum(1 for v in etf_lev_types.values() if v == "leveraged")
        n_inv = sum(1 for v in etf_lev_types.values() if v == "inverse")
        n_levinv = sum(1 for v in etf_lev_types.values() if v == "lev_inv")
        n_etn = sum(1 for v in etf_lev_types.values() if v.startswith("etn"))
        print(f"{ETF_CSV_FILE}: {len(set(tickers))} tickers  |  leveraged={n_lev}  inverse={n_inv}  lev+inv={n_levinv}  etn={n_etn}")

    df_stock, stock_cols = _load_ticker_csv(STOCK_CSV_FILE, "stock")
    if df_stock.empty:
        print(f"[!] Missing or empty {STOCK_CSV_FILE}.")
    else:
        for _, row in df_stock.iterrows():
            t = str(_get_row_value(row, "ticker", stock_cols)).strip().upper()
            if not t or t in {"", "NONE", "NAN", "SYMBOL", "TICKER"}:
                continue

            tickers.append(t)
            TICKER_TYPES[t] = "Pure Stock"

            name = str(_get_row_value(row, "name", stock_cols)).strip()
            if name:
                TICKER_NAMES[t] = name

            sector = str(_get_row_value(row, "sector", stock_cols)).strip()
            if sector:
                TICKER_SECTOR[t] = sector

            industry = str(_get_row_value(row, "industry", stock_cols)).strip()
            if industry:
                TICKER_INDUSTRY[t] = industry

            subindustry = str(_get_row_value(row, "subindustry", stock_cols)).strip()
            if subindustry:
                TICKER_SUBINDUSTRY[t] = subindustry

            mcap = _parse_mcap_str(_get_row_value(row, "mcap", stock_cols))
            if mcap > 0:
                TICKER_CSV_MCAP[t] = mcap

            vol = _parse_vol_str(_get_row_value(row, "volume", stock_cols))
            if vol > 0:
                TICKER_CSV_VOL[t] = vol

    out = sorted(set(tickers))
    out = [t for t in out if t not in {"", "NONE", "NAN", "SYMBOL", "TICKER"}]
    print(f"Loaded {len(out)} tickers total.")
    return out


# ==========================================
# PRICE CACHE
# ==========================================
def _load_one_cache(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.loc[:, ~df.columns.duplicated()]
    except Exception:
        return pd.DataFrame()


def load_cache():
    stock = _load_one_cache(STOCK_DATA_FILE)
    etf = _load_one_cache(ETF_DATA_FILE)

    if stock.empty and etf.empty:
        return pd.DataFrame()
    if stock.empty:
        merged = etf
    elif etf.empty:
        merged = stock
    else:
        merged = pd.concat([stock, etf], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]

    print(f"Cache loaded: {STOCK_DATA_FILE} + {ETF_DATA_FILE}  ({len(merged.columns)} tickers)")
    return merged


def build_dataset(master):
    data = load_cache()
    if data.empty:
        return data

    avail = [t for t in master if t in data.columns]
    data = data[avail].copy()
    data = data.tail(LOOKBACK_DAYS)

    data = validate_and_repair(data, label="Price data", min_trading_days=252)

    # Filter by minimum close price.
    if MIN_PRICE > 0:
        keep = [c for c in data.columns if float(data[c].dropna().iloc[-1]) >= MIN_PRICE] if len(data) > 0 else []
        data = data[keep]

    # Filter by average volume if CSV has volume.
    if MIN_AVG_VOLUME > 0:
        keep = [c for c in data.columns if int(TICKER_CSV_VOL.get(c, 0)) >= MIN_AVG_VOLUME]
        data = data[keep]

    print(f"  Price data: {len(data.columns)} tickers, {len(data)} days ready.")
    return data


# ==========================================
# STATS
# ==========================================
def adf_pvalue(series, max_lag=None):
    """Approximate ADF p-value for spread stationarity (lower is better)."""
    try:
        y = np.asarray(series.dropna(), dtype=float)
        n = len(y)
        if n < 30:
            return 1.0

        if max_lag is None:
            max_lag = int(np.floor(12 * (n / 100) ** 0.25))
            max_lag = min(max_lag, n // 3)

        dy = np.diff(y)
        y_lag = y[:-1]
        nobs = len(dy) - max_lag
        if nobs < 10:
            return 1.0

        X_cols = [np.ones(nobs), y_lag[max_lag:]]
        for k in range(1, max_lag + 1):
            X_cols.append(dy[max_lag - k: -k if k < len(dy) else None][:nobs])
        X = np.column_stack(X_cols)
        Y = dy[max_lag:]

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            return 1.0

        coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        gamma = coef[1]

        Y_hat = X @ coef
        sse = np.sum((Y - Y_hat) ** 2)
        mse = sse / (nobs - X.shape[1])
        var_coef = mse * np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(var_coef[1, 1])

        if se_gamma <= 0:
            return 1.0

        t_stat = gamma / se_gamma

        if t_stat <= -4.32:
            return 0.001
        if t_stat <= -3.43:
            return 0.001 + (t_stat + 4.32) / (0.89) * (0.01 - 0.001)
        if t_stat <= -2.86:
            return 0.01 + (t_stat + 3.43) / (0.57) * (0.05 - 0.01)
        if t_stat <= -2.57:
            return 0.05 + (t_stat + 2.86) / (0.29) * (0.10 - 0.05)
        if t_stat <= -1.94:
            return 0.10 + (t_stat + 2.57) / (0.63) * (0.50 - 0.10)
        if t_stat <= -0.70:
            return 0.50 + (t_stat + 1.94) / (1.24) * (0.90 - 0.50)
        return 0.99
    except Exception:
        return 1.0


def _pair_category(a, b):
    ta = TICKER_TYPES.get(a, "Unknown")
    tb = TICKER_TYPES.get(b, "Unknown")
    if ta == "Pure ETF" and tb == "Pure ETF":
        return "Pure ETF"
    if ta == "Pure Stock" and tb == "Pure Stock":
        return "Pure Stock"
    return "Mixed"


def _pair_relation(a, b):
    sec_a = (TICKER_SECTOR.get(a, "") or "").strip()
    sec_b = (TICKER_SECTOR.get(b, "") or "").strip()
    ind_a = (TICKER_INDUSTRY.get(a, "") or "").strip()
    ind_b = (TICKER_INDUSTRY.get(b, "") or "").strip()
    sub_a = (TICKER_SUBINDUSTRY.get(a, "") or "").strip()
    sub_b = (TICKER_SUBINDUSTRY.get(b, "") or "").strip()

    relation = "mixed"
    if sub_a and sub_b and sub_a == sub_b:
        relation = "subindustry"
    elif ind_a and ind_b and ind_a == ind_b:
        relation = "industry"
    elif sec_a and sec_b and sec_a == sec_b:
        relation = "sector"

    return {
        "sector_a": sec_a,
        "sector_b": sec_b,
        "industry_a": ind_a,
        "industry_b": ind_b,
        "sub_a": sub_a,
        "sub_b": sub_b,
        "relation": relation,
    }


def _is_related_pair(a, b):
    """True when pair shares industry or sector (industry match implies sector-level relation)."""
    ind_a = (TICKER_INDUSTRY.get(a, "") or "").strip()
    ind_b = (TICKER_INDUSTRY.get(b, "") or "").strip()
    if ind_a and ind_b and ind_a == ind_b:
        return True

    sec_a = (TICKER_SECTOR.get(a, "") or "").strip()
    sec_b = (TICKER_SECTOR.get(b, "") or "").strip()
    if sec_a and sec_b and sec_a == sec_b:
        return True

    return False


# Worker state
_w_corr_long = None
_w_prices_raw = None


def _init_worker(corr_long, prices_raw):
    global _w_corr_long, _w_prices_raw
    _w_corr_long = corr_long
    _w_prices_raw = prices_raw


def _analyze_pair(pair):
    a, b = pair

    if _w_corr_long is None or _w_prices_raw is None:
        return None

    cl = float(_w_corr_long.loc[a, b])
    if np.isnan(cl) or cl < MIN_CORR_FILTER:
        return None

    spread = np.log(_w_prices_raw[a].clip(lower=1e-10)) - np.log(_w_prices_raw[b].clip(lower=1e-10))
    spread = spread.dropna()
    if len(spread) < ADF_MIN_DAYS:
        return None

    max_p = 1.0 - ADF_CONFIDENCE
    max_years = len(spread) // 252
    if max_years < 1:
        return None

    # Strict tradeability gate: if 1Y cointegration fails, skip the pair entirely.
    try:
        p1 = adf_pvalue(spread.iloc[-252:])
    except Exception:
        p1 = 1.0
    if p1 > max_p:
        return None

    coint_years = 1
    pass_years = [1]
    for yr in range(2, max_years + 1):
        days = yr * 252
        try:
            p = adf_pvalue(spread.iloc[-days:])
        except Exception:
            p = 1.0
        if p <= max_p:
            coint_years += 1
            pass_years.append(yr)

    if coint_years < MIN_COINT_YEARS:
        return None

    adf_p = adf_pvalue(spread.iloc[-252:])
    conf_pct = max(0.0, min(100.0, (1.0 - float(adf_p)) * 100.0))
    if conf_pct >= 99.0:
        conf_tier = "high"
    elif conf_pct >= 95.0:
        conf_tier = "medium"
    else:
        conf_tier = "low"
    rel = _pair_relation(a, b)

    mc_a = float(TICKER_CSV_MCAP.get(a, 0) or 0)
    mc_b = float(TICKER_CSV_MCAP.get(b, 0) or 0)
    mc_min = min(mc_a, mc_b) if mc_a > 0 and mc_b > 0 else 0.0

    return {
        "Pair": f"{a}/{b}",
        "TickerA": a,
        "TickerB": b,
        "Category": _pair_category(a, b),
        "Corr": round(cl, 4),
        "ADF_p": float(adf_p),
        "ConfPct": round(conf_pct, 1),
        "ConfTier": conf_tier,
        "CointYears": int(coint_years),
        "CointMaxYears": int(max_years),
        "CointByYear": ", ".join(f"{yr}Y" for yr in pass_years) if pass_years else "-",
        "AllYears": 1 if coint_years >= max_years else 0,
        "MinMcap": mc_min,
        "SectorA": rel["sector_a"],
        "SectorB": rel["sector_b"],
        "IndustryA": rel["industry_a"],
        "IndustryB": rel["industry_b"],
        "SubA": rel["sub_a"],
        "SubB": rel["sub_b"],
        "Relation": rel["relation"],
    }


# ==========================================
# UI OUTPUT
# ==========================================
def _safe_pair_meta(a, b, same_only=False):
    if same_only:
        return a if (a and b and a == b) else f"{a} / {b}"
    return a if a else (b if b else "")


def build_pair_chart_map(results, price_data):
    chart_map = {}
    subset = results[:MAX_CHART_PAIRS] if MAX_CHART_PAIRS > 0 else results

    for r in tqdm(subset, desc="Building Charts"):
        a = r.get("TickerA")
        b = r.get("TickerB")
        pair = r.get("Pair")
        if not a or not b or not pair:
            continue
        if a not in price_data.columns or b not in price_data.columns:
            continue

        coint_years = max(1, int(r.get("CointYears", 1) or 1))
        chart_days = max(CHART_LOOKBACK_DAYS, coint_years * 252)
        combined = price_data[[a, b]].dropna().tail(chart_days)
        if len(combined) < 20:
            continue

        base_a = float(combined[a].iloc[0])
        base_b = float(combined[b].iloc[0])
        if base_a <= 0 or base_b <= 0:
            continue

        dates = [d.strftime("%Y-%m-%d") for d in combined.index]
        norm_a = [round(float(v / base_a * 100.0), 4) for v in combined[a].values]
        norm_b = [round(float(v / base_b * 100.0), 4) for v in combined[b].values]

        chart_map[pair] = {
            "a": a,
            "b": b,
            "years": coint_years,
            "dates": dates,
            "aVals": norm_a,
            "bVals": norm_b,
        }

    return chart_map


def build_inventory_html(results, total_combos, chart_map):
    shown_by_cat = {"Pure ETF": 0, "Pure Stock": 0, "Mixed": 0}
    rows = []

    for r in results:
        shown_by_cat[r.get("Category", "Mixed")] += 1

        sec = _safe_pair_meta(r.get("SectorA", ""), r.get("SectorB", ""), same_only=True)
        ind = _safe_pair_meta(r.get("IndustryA", ""), r.get("IndustryB", ""), same_only=True)
        sub = _safe_pair_meta(r.get("SubA", ""), r.get("SubB", ""), same_only=True)
        all_years = "Yes" if int(r.get("AllYears", 0)) else "No"
        has_chart = r["Pair"] in chart_map
        chart_btn = (
            f"<button class='chart-btn' onclick=\"openPairChart('{r['Pair']}')\">Chart</button>"
            if has_chart else "-"
        )

        rows.append(
            f"<tr data-cat='{r['Category']}' data-rel='{r['Relation']}' data-mcap='{r['MinMcap']}' "
            f"data-cyears='{r['CointYears']}' data-all='{r['AllYears']}' data-conf='{r.get('ConfTier','low')}'>"
            f"<td>{r['Pair']}</td>"
            f"<td>{r['Category']}</td>"
            f"<td>{sec}</td>"
            f"<td>{ind}</td>"
            f"<td>{sub}</td>"
            f"<td>{r['MinMcap']:,.0f}</td>"
            f"<td>{r['Corr']:.2f}</td>"
            f"<td>{(1.0 - r['ADF_p']) * 100:.1f}%</td>"
            f"<td>{r['CointYears']}/{r['CointMaxYears']}</td>"
            f"<td>{r['CointByYear']}</td>"
            f"<td>{all_years}</td>"
            f"<td>{chart_btn}</td>"
            f"</tr>"
        )

    chart_payload = json.dumps(chart_map, separators=(",", ":"))

    page = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Pairs Finder</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #08090d;
            --surface: #0d1117;
            --surface2: #131720;
            --surface3: #181f2e;
            --line: #1c2333;
            --line2: #242d40;
            --ink: #c9d1d9;
            --muted: #94a3b8;
            --cyan: #38bdf8;
            --mono: 'JetBrains Mono', monospace;
            --sans: 'Syne', sans-serif;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: var(--sans); color: var(--ink); background: var(--bg); }}
        .wrap {{ max-width: 1500px; margin: 20px auto 30px; padding: 0 14px; }}
        .top {{ background: linear-gradient(180deg, #0d1520 0%, #0a0e17 100%); color: #f8fafc; border: 1px solid var(--line2); border-radius: 14px; padding: 18px 20px; box-shadow: 0 18px 40px rgba(0,0,0,0.45), 0 0 0 1px rgba(56,189,248,0.06); }}
        .top h1 {{ margin: 0; font-size: 26px; }}
        .top h1 span {{ color: var(--cyan); }}
        .sub {{ margin-top: 6px; color: var(--muted); font-size: 12px; font-family: var(--mono); }}
        .controls {{ margin-top: 14px; background: var(--surface2); border: 1px solid var(--line); border-radius: 12px; padding: 12px; display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }}
        .controls label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
        .controls input, .controls select {{ width: 100%; border: 1px solid var(--line2); border-radius: 8px; padding: 8px 10px; background: var(--surface); color: var(--ink); font-family: var(--mono); }}
        .controls select option {{ background: var(--surface); }}
        .table-card {{ margin-top: 14px; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--surface); box-shadow: 0 10px 28px rgba(0,0,0,0.3); }}
        .table-wrap {{ overflow-x: auto; max-height: 76vh; }}
        table {{ border-collapse: collapse; width: 100%; min-width: 1200px; }}
        th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; font-size: 12px; white-space: nowrap; }}
        th {{ position: sticky; top: 0; background: var(--surface2); color: var(--muted); z-index: 2; text-transform: uppercase; letter-spacing: 0.08em; font-size: 10px; }}
        tr:hover {{ background: var(--surface3); }}
        .count {{ padding: 10px 12px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--line); font-family: var(--mono); }}
        .pager {{ display: flex; align-items: center; justify-content: center; gap: 6px; padding: 10px 12px; border-top: 1px solid var(--line); background: var(--surface2); font-family: var(--mono); }}
        .pg-btn {{ border: 1px solid var(--line2); background: var(--surface); color: var(--muted); border-radius: 6px; padding: 4px 9px; cursor: pointer; font-size: 11px; }}
        .pg-btn:hover {{ border-color: var(--cyan); color: var(--cyan); }}
        .pg-btn:disabled {{ opacity: 0.35; cursor: default; }}
        .pg-info {{ font-size: 11px; color: var(--muted); margin: 0 8px; }}
        .chart-btn {{ border: 1px solid rgba(56,189,248,0.25); background: rgba(56,189,248,0.08); color: var(--cyan); border-radius: 999px; padding: 4px 10px; cursor: pointer; font-size: 11px; font-weight: 600; font-family: var(--mono); }}
        .chart-btn:hover {{ background: rgba(56,189,248,0.18); border-color: var(--cyan); }}
        .chart-overlay {{ position: fixed; inset: 0; background: rgba(0,0,0,0.78); display: none; align-items: center; justify-content: center; z-index: 9999; backdrop-filter: blur(8px); }}
        .chart-overlay.open {{ display: flex; }}
        .chart-modal {{ width: min(980px, 94vw); background: #0a0e17; border-radius: 14px; border: 1px solid var(--line2); box-shadow: 0 40px 100px rgba(0,0,0,0.7), 0 0 0 1px rgba(56,189,248,0.06); overflow: hidden; }}
        .chart-head {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-bottom: 1px solid var(--line); background: linear-gradient(180deg, #0d1520 0%, #0a0e17 100%); }}
        .chart-title {{ font-size: 15px; font-weight: 700; color: #e2e8f0; font-family: var(--mono); }}
        .chart-close {{ border: 1px solid var(--line2); background: var(--surface2); color: var(--ink); border-radius: 8px; padding: 4px 8px; cursor: pointer; font-family: var(--mono); }}
        .chart-wrap {{ padding: 8px 10px 14px; }}
        #pairChartCanvas {{ width: 100%; height: 450px; }}
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg); }}
        ::-webkit-scrollbar-thumb {{ background: var(--line2); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}
    </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"top\">
    <h1>PAIRS <span>FINDER</span></h1>
            <div class=\"sub\">Scanned: {total_combos:,} pairs | Listed: {len(rows):,} | Charts: {len(chart_map):,} | Stocks: {shown_by_cat['Pure Stock']} | ETFs: {shown_by_cat['Pure ETF']} | Mixed: {shown_by_cat['Mixed']} | Built: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class=\"controls\">
      <div><label>Search</label><input id=\"q\" placeholder=\"SPY/QQQ, sector, industry...\" oninput=\"applyFilters()\" /></div>
      <div><label>Pair Type</label><select id=\"cat\" onchange=\"applyFilters()\"><option value=\"all\">All</option><option>Pure ETF</option><option>Pure Stock</option><option>Mixed</option></select></div>
      <div><label>Sector Relationship</label><select id=\"rel\" onchange=\"applyFilters()\"><option value=\"all\">All</option><option value=\"sector\">Same Sector+</option><option value=\"industry\">Same Industry+</option><option value=\"subindustry\">Same Sub-industry</option></select></div>
      <div><label>Min MCap/Assets</label><select id=\"mcap\" onchange=\"applyFilters()\"><option value=\"0\">Any</option><option value=\"1000000\">1M+</option><option value=\"50000000\">50M+</option><option value=\"300000000\">300M+</option><option value=\"2000000000\">2B+</option><option value=\"10000000000\">10B+</option><option value=\"200000000000\">200B+</option></select></div>
      <div><label>Cointegration</label><select id=\"coint\" onchange=\"applyFilters()\"><option value=\"all\">All</option><option value=\"any\">Any Year</option><option value=\"all_years\">All Available Years</option><option value=\"1\">1Y+</option><option value=\"2\">2Y+</option><option value=\"3\">3Y+</option></select></div>
    <div><label>Confidence</label><select id=\"conf\" onchange=\"applyFilters()\"><option value=\"all\">All</option><option value=\"low\">Low+</option><option value=\"medium\">Medium+</option><option value=\"high\">High</option></select></div>
    </div>

    <div class=\"table-card\">
      <div class=\"table-wrap\">
        <table id=\"tbl\">
          <thead>
            <tr>
              <th>Pair</th><th>Type</th><th>Sector</th><th>Industry</th><th>Sub-industry</th>
                            <th>Min MCap/Assets</th><th>Corr</th><th>Coint Conf</th><th>Coint Years</th><th>Coint by Year</th><th>All Years</th><th>Chart</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
            <div class=\"count\" id=\"visibleCount\"></div>
            <div class=\"pager\" id=\"pager\"></div>
    </div>
  </div>

    <div class=\"chart-overlay\" id=\"pairChartModal\" onclick=\"if(event.target===this)closePairChart()\">
        <div class=\"chart-modal\">
            <div class=\"chart-head\">
                <div class=\"chart-title\" id=\"pairChartTitle\">Pair Chart</div>
                <button class=\"chart-close\" onclick=\"closePairChart()\">Close</button>
            </div>
            <div class=\"chart-wrap\">
                <canvas id=\"pairChartCanvas\"></canvas>
            </div>
        </div>
    </div>

<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js\"></script>
<script>
const chartData = {chart_payload};
let activePairChart = null;
const PER_PAGE = 25;
let currentPage = 1;

function relPass(rel, need) {{
    if (need === 'all') return true;
    if (need === 'sector') return rel === 'sector' || rel === 'industry' || rel === 'subindustry';
    if (need === 'industry') return rel === 'industry' || rel === 'subindustry';
    return rel === 'subindustry';
}}

function confPass(rowConf, need) {{
    if (need === 'all') return true;
    const order = ['low', 'medium', 'high'];
    const rowIdx = order.indexOf((rowConf || 'low').toLowerCase());
    const needIdx = order.indexOf(need);
    if (needIdx < 0) return true;
    return rowIdx >= needIdx;
}}

function applyFilters(resetPage = true) {{
    const q = (document.getElementById('q').value || '').toUpperCase();
    const cat = document.getElementById('cat').value;
    const rel = document.getElementById('rel').value;
    const mcap = parseFloat(document.getElementById('mcap').value || '0');
    const coint = document.getElementById('coint').value;
    const conf = document.getElementById('conf').value;
    if (resetPage) currentPage = 1;

    const rows = document.querySelectorAll('#tbl tbody tr');
    const matched = [];
    rows.forEach(tr => {{
        const text = (tr.textContent || '').toUpperCase();
        const tcat = tr.dataset.cat || '';
        const trel = tr.dataset.rel || 'mixed';
        const tmcap = parseFloat(tr.dataset.mcap || '0');
        const tcy = parseInt(tr.dataset.cyears || '0', 10);
        const tall = parseInt(tr.dataset.all || '0', 10);
        const tconf = (tr.dataset.conf || 'low').toLowerCase();

        let ok = true;
        if (q && !text.includes(q)) ok = false;
        if (cat !== 'all' && tcat !== cat) ok = false;
        if (!relPass(trel, rel)) ok = false;
        if (tmcap < mcap) ok = false;
        if (coint === 'any' && tcy < 1) ok = false;
        if (coint === 'all_years' && !tall) ok = false;
        if (!isNaN(parseInt(coint, 10)) && tcy < parseInt(coint, 10)) ok = false;
        if (!confPass(tconf, conf)) ok = false;

        if (ok) matched.push(tr);
        tr.style.display = 'none';
    }});

    const totalMatched = matched.length;
    const totalPages = Math.max(1, Math.ceil(totalMatched / PER_PAGE));
    if (currentPage > totalPages) currentPage = totalPages;
    const start = (currentPage - 1) * PER_PAGE;
    const end = start + PER_PAGE;

    matched.forEach((tr, idx) => {{
        tr.style.display = (idx >= start && idx < end) ? '' : 'none';
    }});

    const shown = totalMatched === 0 ? 0 : Math.min(PER_PAGE, Math.max(0, totalMatched - start));
    document.getElementById('visibleCount').textContent = 'Showing: ' + shown.toLocaleString() + ' of ' + totalMatched.toLocaleString() + ' matched (Total: ' + rows.length.toLocaleString() + ')';
    renderPager(totalPages);
}}

function renderPager(totalPages) {{
    const pager = document.getElementById('pager');
    if (!pager) return;
    if (totalPages <= 1) {{
        pager.innerHTML = '';
        return;
    }}

    const prevDisabled = currentPage <= 1 ? 'disabled' : '';
    const nextDisabled = currentPage >= totalPages ? 'disabled' : '';
    pager.innerHTML =
        `<button class=\"pg-btn\" ${{prevDisabled}} onclick=\"gotoPage(${{currentPage - 1}})\">Prev</button>` +
        `<span class=\"pg-info\">Page ${{currentPage}} / ${{totalPages}}</span>` +
        `<button class=\"pg-btn\" ${{nextDisabled}} onclick=\"gotoPage(${{currentPage + 1}})\">Next</button>`;
}}

function gotoPage(p) {{
    if (p < 1) return;
    currentPage = p;
    applyFilters(false);
}}

function openPairChart(pair) {{
    const p = chartData[pair];
    if (!p || !p.dates || p.dates.length === 0) {{
        alert('No chart data for this pair.');
        return;
    }}

    const yrs = parseInt(p.years || 1, 10);
    document.getElementById('pairChartTitle').textContent = pair + ' (' + yrs + 'Y chart, rebased to 100)';
    document.getElementById('pairChartModal').classList.add('open');

    if (activePairChart) {{
        activePairChart.destroy();
        activePairChart = null;
    }}

    const ctx = document.getElementById('pairChartCanvas').getContext('2d');
    activePairChart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: p.dates,
            datasets: [
                {{ label: p.a, data: p.aVals, borderColor: '#38bdf8', borderWidth: 2, pointRadius: 0, tension: 0.15 }},
                {{ label: p.b, data: p.bVals, borderColor: '#e2e8f0', borderWidth: 2, pointRadius: 0, tension: 0.15 }},
            ],
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            scales: {{
                x: {{ ticks: {{ maxTicksLimit: 10, color: '#94a3b8' }}, grid: {{ color: 'rgba(148,163,184,0.12)' }} }},
                y: {{ title: {{ display: true, text: 'Normalized Price (Base 100)', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(148,163,184,0.12)' }} }},
            }},
            plugins: {{ legend: {{ position: 'top', labels: {{ color: '#c9d1d9' }} }} }},
        }},
    }});
}}

function closePairChart() {{
    document.getElementById('pairChartModal').classList.remove('open');
    if (activePairChart) {{
        activePairChart.destroy();
        activePairChart = null;
    }}
}}

window.addEventListener('DOMContentLoaded', applyFilters);
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closePairChart(); }});
</script>
</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(page)


def write_no_data_page():
    page = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Pairs Finder</title>
    <link href=\"https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap\" rel=\"stylesheet\"> 
  <style>
        body { font-family: 'Syne', sans-serif; margin: 0; padding: 24px; background: #08090d; color: #c9d1d9; }
        .card { max-width: 920px; margin: 40px auto; background: #0d1117; border: 1px solid #1c2333; border-radius: 12px; padding: 22px; box-shadow: 0 18px 40px rgba(0,0,0,0.45); }
    h1 { margin-top: 0; }
    p { line-height: 1.5; }
        .muted { color: #94a3b8; }
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Pairs Finder</h1>
    <p>No price data is currently available, so no pairs could be built.</p>
    <p class=\"muted\">Run market_data_maintainer.py to refresh cache files, then rerun pairs_finder.py.</p>
  </div>
</body>
</html>"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(page)


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("Running Pairs Finder...")
    print("Using shared cache files from market_data_maintainer.py")

    tickers = load_master_tickers()
    data = build_dataset(tickers)

    if data is None or data.empty or len(data.columns) < 2:
        write_no_data_page()
        print(f"{OUTPUT_FILE} created (no-data fallback).")
        print("\n[!] No data available. Check internet or ticker list.")
        raise SystemExit(0)

    valid = [c for c in data.columns if c in tickers]
    if len(valid) < 2:
        write_no_data_page()
        print(f"{OUTPUT_FILE} created (no-data fallback).")
        print("\n[!] Not enough valid tickers to build pairs.")
        raise SystemExit(0)

    pre_filter_count = len(valid)
    print(f"Tickers: {pre_filter_count} ready for pair analysis")

    prices_raw = data[valid]
    returns = prices_raw.pct_change().dropna(how="all")
    corr_long = returns.corr()

    all_combos_count = len(valid) * (len(valid) - 1) // 2

    related_combos = []
    other_combos = []
    for a, b in itertools.combinations(valid, 2):
        if _is_related_pair(a, b):
            related_combos.append((a, b))
        else:
            other_combos.append((a, b))

    mode = (PAIR_PREFILTER_MODE or "all").strip().lower()
    if mode == "related_only":
        combos = related_combos
    elif mode == "related_first":
        combos = related_combos + other_combos
    else:
        combos = related_combos + other_combos

    total_combos = len(combos)
    print(
        f"Building combinations: {total_combos:,} selected / {all_combos_count:,} total "
        f"(mode={mode}, related={len(related_combos):,}, other={len(other_combos):,})"
    )

    chunksize = max(1, total_combos // max(1, NUM_WORKERS * 4))
    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(corr_long, prices_raw),
    ) as pool:
        results = [
            r
            for r in tqdm(
                pool.imap_unordered(_analyze_pair, combos, chunksize=chunksize),
                total=total_combos,
                desc="Analyzing Pairs",
            )
            if r is not None
        ]

    results = sorted(results, key=lambda r: r["Pair"])

    if not results:
        write_no_data_page()
        print(f"{OUTPUT_FILE} created (no-data fallback).")
        print("\n[!] No pairs passed correlation/cointegration criteria.")
        raise SystemExit(0)

    chart_map = build_pair_chart_map(results, prices_raw)
    build_inventory_html(results, total_combos, chart_map)
    print(f"{OUTPUT_FILE} created. ({len(results)} pairs rendered)")

    if _dropped_tickers:
        print(f"Dropped sparse tickers: {len(_dropped_tickers)}")

    print("\nDone.")


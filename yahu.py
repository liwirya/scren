#!/usr/bin/env python3
"""
SCREENER SAHAM  -  Fundamental + Teknikal + Trade Plan (Swing / Day Trade / Scalping)

Install:
    pip install yfinance pandas numpy
    pip install deep-translator      # opsional, untuk terjemah deskripsi bisnis ke Indonesia

Contoh pakai:
    python screener_saham.py BBCA                          # laporan lengkap, metode swing
    python screener_saham.py BBCA --method all             # swing + daytrade + scalping sekaligus
    python screener_saham.py BBCA --method daytrade --watch 60      # refresh tiap 60 detik
    python screener_saham.py --scan BBCA BBRI TLKM ASII --method swing --min-score 55
    python screener_saham.py --scan-file daftar.txt --method daytrade --top 15
    python screener_saham.py BBCA --capital 100000000 --risk-pct 1  # hitung jumlah lot
    python screener_saham.py AAPL --market global                  # saham luar (tanpa .JK)
    python screener_saham.py BBCA --json hasil.json

CATATAN PENTING
- Sumber data: Yahoo Finance (yfinance). Harga IDX umumnya delay ~10 menit; ini "near-realtime".
  Untuk benar-benar realtime (tick by tick) butuh data feed resmi (broker API / vendor IDX).
- Data fundamental Yahoo untuk emiten IDX kadang kosong/telat. Cek silang ke laporan keuangan
  di idx.co.id atau IR emiten sebelum ambil keputusan.
- Ini alat bantu analisis, BUKAN rekomendasi investasi. Semua keputusan dan risiko ada di Anda.
"""
import argparse
import json
import math
import os
import sys
import pickle
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# KONFIGURASI METODE
# --------------------------------------------------------------------------------------
METHODS = {
    "swing": dict(
        label="SWING (beberapa hari - beberapa minggu)",
        tf=("1d", "2y"), htf=("1wk", "5y"),
        atr_k=(1.0, 2.5), max_loss=0.07, fib_window=120, rr=(1.5, 2.5, 4.0),
        w_tech=0.60, w_fund=0.40,
        trail="Trailing: geser SL ke bawah EMA21/low candle harian terakhir setelah TP1 tercapai; "
              "exit jika close harian di bawah SL.",
    ),
    "daytrade": dict(
        label="DAY TRADE (intraday, tutup sebelum penutupan)",
        tf=("15m", "30d"), htf=("1h", "90d"),
        atr_k=(0.8, 1.8), max_loss=0.03, fib_window=100, rr=(1.2, 2.0, 3.0),
        w_tech=0.85, w_fund=0.15,
        trail="Trailing: setelah TP1, geser SL ke harga entry (break-even) lalu ikuti low swing 15m; "
              "jangan bawa posisi menginap.",
    ),
    "scalping": dict(
        label="SCALPING (menit - 1/2 jam)",
        tf=("5m", "5d"), htf=("15m", "30d"),
        atr_k=(0.5, 1.2), max_loss=0.015, fib_window=80, rr=(1.0, 1.5, 2.0),
        w_tech=1.00, w_fund=0.0,
        trail="Trailing: ambil sebagian di TP1, sisanya SL geser ke break-even. Disiplin cut loss; "
              "hindari saham dengan spread lebar/tidak likuid.",
    ),
}

LQ_UNIVERSE = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BRIS", "ARTO", "TLKM", "ISAT", "EXCL", "MTEL", "TOWR",
    "ASII", "UNTR", "UNVR", "ICBP", "INDF", "CPIN", "JPFA", "KLBF", "MAPI", "ACES", "ERAA",
    "ADRO", "PTBA", "ITMG", "HRUM", "MEDC", "PGAS", "AKRA", "ANTM", "MDKA", "INCO", "TINS",
    "AMMN", "BRPT", "TPIA", "BREN", "SMGR", "INTP", "CTRA", "BSDE", "SMRA", "PWON", "JSMR",
    "GOTO", "EMTK", "ESSA",
]


# --------------------------------------------------------------------------------------
# UTIL: TICK SIZE IDX, FORMAT
# --------------------------------------------------------------------------------------
def tick_size(price, idx=True):
    """Fraksi harga Bursa Efek Indonesia."""
    if not idx:
        return 0.01
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2000:
        return 5
    if price < 5000:
        return 10
    return 25


def rnd(price, idx=True, mode="nearest"):
    t = tick_size(price, idx)
    q = price / t
    if mode == "down":
        q = math.floor(q + 1e-9)
    elif mode == "up":
        q = math.ceil(q - 1e-9)
    else:
        q = round(q)
    val = q * t
    return float(round(val)) if idx else float(round(val, 2))


def px(x, idx=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.0f}" if idx else f"{x:,.2f}"


def big(x, idx=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    cur = "Rp " if idx else "$"
    a = abs(x)
    if a >= 1e12:
        return f"{cur}{x / 1e12:,.2f} T"
    if a >= 1e9:
        return f"{cur}{x / 1e9:,.2f} M" if idx else f"{cur}{x / 1e9:,.2f} B"
    if a >= 1e6:
        return f"{cur}{x / 1e6:,.2f} Jt" if idx else f"{cur}{x / 1e6:,.2f} M"
    return f"{cur}{x:,.0f}"


def pct(x, mult=100):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x * mult:.1f}%"


def num(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.{d}f}"


def norm_ticker(t, market):
    t = t.strip().upper()
    if market == "idx" and "." not in t and not t.startswith("^"):
        t += ".JK"
    return t


# --------------------------------------------------------------------------------------
# DATA HARGA
# --------------------------------------------------------------------------------------
def _hist_yahoo(sym, interval, period):
    import yfinance as yf
    df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False, actions=False)
    if df is None or df.empty:
        raise ValueError(f"Data harga kosong untuk {sym} ({interval}). Cek kode saham / koneksi.")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df = df[~df.index.duplicated(keep="last")]
    return df


# --------------------------------------------------------------------------------------
# SUMBER DATA: TradingView (utama, cepat) -> Yahoo (cadangan) + cache
# --------------------------------------------------------------------------------------
_SOURCE = "auto"                      # auto | tv | yahoo   (diset dari --source)
_HIST_CACHE = {}
_CACHE_LOCK = threading.RLock()
_TL = threading.local()
_WARNED = set()
_TTL = {"5m": 20, "15m": 30, "1h": 60, "1d": 60, "1wk": 300}       # detik
_TV_INTERVAL = {"5m": "in_5_minute", "15m": "in_15_minute", "1h": "in_1_hour",
                "1d": "in_daily", "1wk": "in_weekly"}
_TV_BARS = {"5m": 500, "15m": 700, "1h": 700, "1d": 520, "1wk": 260}
_FUND_FILE = os.path.join(tempfile.gettempdir(), "screener_fund_cache.pkl")


def _hist_tv(sym, interval, period):
    """OHLCV dari TradingView (bursa IDX) lewat library tvdatafeed."""
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        raise RuntimeError("tvdatafeed belum terpasang: pip install --upgrade --no-cache-dir "
                           "git+https://github.com/rongardF/tvdatafeed.git")
    tv = getattr(_TL, "tv", None)
    if tv is None:
        tv = _TL.tv = TvDatafeed()          # mode anonim; bisa login via TvDatafeed(user, pass)
    df = tv.get_hist(symbol=sym.replace(".JK", ""), exchange="IDX",
                     interval=getattr(Interval, _TV_INTERVAL[interval]), n_bars=_TV_BARS[interval])
    if df is None or len(df) == 0:
        raise ValueError(f"TradingView: data kosong untuk {sym} ({interval})")
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df[~df.index.duplicated(keep="last")]


def get_history(sym, interval, period):
    key = (sym, interval, period)
    ttl = _TTL.get(interval, 60)
    with _CACHE_LOCK:
        hit = _HIST_CACHE.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1].copy()
    df = None
    if _SOURCE in ("auto", "tv") and sym.endswith(".JK") and interval in _TV_INTERVAL:
        try:
            df = _hist_tv(sym, interval, period)
        except Exception as e:
            if _SOURCE == "tv":
                raise
            if "tv" not in _WARNED:
                _WARNED.add("tv")
                print(f"[info] TradingView gagal ({str(e)[:90]}) -> pakai Yahoo sebagai cadangan.")
    if df is None:
        df = _hist_yahoo(sym, interval, period)
    with _CACHE_LOCK:
        _HIST_CACHE[key] = (time.time(), df)
    return df.copy()


def get_fundamentals(sym, translate=False, ttl=12 * 3600):
    """Fundamental (Yahoo) + cache disk 12 jam: run berikutnya instan."""
    key = (sym, translate)

    def load():
        try:
            with open(_FUND_FILE, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return {}

    with _CACHE_LOCK:
        hit = load().get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    f = _get_fundamentals_yahoo(sym, translate)
    if f and f.get("score") is not None:
        with _CACHE_LOCK:
            c = load()
            c[key] = (time.time(), f)
            try:
                with open(_FUND_FILE, "wb") as fh:
                    pickle.dump(c, fh)
            except Exception:
                pass
    return f


# --------------------------------------------------------------------------------------
# INDIKATOR
# --------------------------------------------------------------------------------------
def calc_rsi(s, n=14):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(rd == 0) & ru.notna()] = 100
    return out


def add_indicators(df):
    df = df.copy()
    c = df["close"]
    for n in (5, 10, 20, 50, 200):
        df[f"sma{n}"] = c.rolling(n).mean()
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["rsi"] = calc_rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    pc = c.shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = mid + 2 * sd, mid - 2 * sd
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    # Stochastic (14,3)
    ll, hh = df["low"].rolling(14).min(), df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


# --------------------------------------------------------------------------------------
# SUPPORT / RESISTANCE, TREND, FIBONACCI
# --------------------------------------------------------------------------------------
def pivots(df, n=5):
    w = 2 * n + 1
    hi = df["high"][df["high"] == df["high"].rolling(w, center=True).max()].dropna()
    lo = df["low"][df["low"] == df["low"].rolling(w, center=True).min()].dropna()
    return hi, lo


def cluster_levels(vals, tol):
    vals = sorted(vals)
    groups = []
    for v in vals:
        if groups and abs(v - np.mean(groups[-1])) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [(float(np.mean(g)), len(g)) for g in groups]


def support_resistance(df, hi, lo, price, atr):
    tol = max(0.6 * atr, price * 0.003)
    levels = list(hi.tail(40).values) + list(lo.tail(40).values)
    cl = cluster_levels(levels, tol)
    sup = sorted([x for x in cl if x[0] < price - 0.1 * atr], key=lambda x: -x[0])[:3]
    res = sorted([x for x in cl if x[0] > price + 0.1 * atr], key=lambda x: x[0])[:3]
    return sup, res


def daily_pivot_points(df):
    """Classic pivot dari hari sebelumnya (khusus data intraday)."""
    try:
        days = df.groupby(df.index.date)
        keys = list(days.groups.keys())
        if len(keys) < 2:
            return None
        prev = days.get_group(keys[-2])
        h, l, c = prev["high"].max(), prev["low"].min(), prev["close"].iloc[-1]
        pp = (h + l + c) / 3
        return dict(PP=pp, R1=2 * pp - l, R2=pp + (h - l), S1=2 * pp - h, S2=pp - (h - l), PDH=h, PDL=l)
    except Exception:
        return None


def trend_info(df, hi, lo):
    l = df.iloc[-1]
    pts, notes = 0, []

    def chk(cond_up, cond_dn, up_txt, dn_txt):
        nonlocal pts
        if cond_up:
            pts += 1
            notes.append(up_txt)
        elif cond_dn:
            pts -= 1
            notes.append(dn_txt)

    chk(l.close > l.ema21, l.close < l.ema21, "Harga di atas EMA21", "Harga di bawah EMA21")
    if not np.isnan(l.sma50):
        chk(l.ema21 > l.sma50, l.ema21 < l.sma50, "EMA21 > SMA50", "EMA21 < SMA50")
        slope = l.sma50 - df["sma50"].iloc[-6] if len(df) > 6 and not np.isnan(df["sma50"].iloc[-6]) else 0
        chk(slope > 0, slope < 0, "SMA50 menanjak", "SMA50 menurun")
    if not np.isnan(l.sma200):
        chk(l.sma50 > l.sma200, l.sma50 < l.sma200, "SMA50 > SMA200 (golden alignment)", "SMA50 < SMA200 (death alignment)")

    structure = "n/a"
    if len(hi) >= 2 and len(lo) >= 2:
        hh, hl = hi.iloc[-1] > hi.iloc[-2], lo.iloc[-1] > lo.iloc[-2]
        if hh and hl:
            pts += 2
            structure = "Higher High - Higher Low"
        elif (not hh) and (not hl):
            pts -= 2
            structure = "Lower High - Lower Low"
        else:
            structure = "Campuran (transisi/sideways)"
        notes.append(f"Struktur harga: {structure}")

    label = "UPTREND" if pts >= 3 else "DOWNTREND" if pts <= -3 else "SIDEWAYS"
    return dict(label=label, points=pts, strength=min(abs(pts) / 6, 1.0), structure=structure, notes=notes)


def fibonacci(df, window):
    w = df.tail(window)
    hi, lo = float(w["high"].max()), float(w["low"].min())
    up = w["low"].idxmin() < w["high"].idxmax()
    rng = hi - lo
    ratios = (0.236, 0.382, 0.5, 0.618, 0.786)
    if up:
        ret = {r: hi - r * rng for r in ratios}
        ext = {e: lo + e * rng for e in (1.0, 1.272, 1.618)}
    else:
        ret = {r: lo + r * rng for r in ratios}
        ext = {e: hi - e * rng for e in (1.0, 1.272, 1.618)}
    price = float(df["close"].iloc[-1])
    zone = "di luar area retracement"
    if rng > 0:
        depth = (hi - price) / rng if up else (price - lo) / rng
        if 0.382 <= depth <= 0.618:
            zone = f"GOLDEN ZONE (retracement ~{depth:.1%})"
        elif 0 <= depth < 0.382:
            zone = f"retracement dangkal (~{depth:.1%})"
        elif 0.618 < depth <= 1:
            zone = f"retracement dalam (~{depth:.1%})"
    return dict(direction="UP-swing" if up else "DOWN-swing", high=hi, low=lo,
                retracement=ret, extension=ext, zone=zone)


# --------------------------------------------------------------------------------------
# SKOR TEKNIKAL
# --------------------------------------------------------------------------------------
def bias_score(df, trend, sup, res, atr, price):
    l, p = df.iloc[-1], df.iloc[-2]
    s, notes = trend["points"] * 5.0, []
    notes += trend["notes"]

    if l.macd > l.macd_sig:
        s += 10
        notes.append("MACD di atas signal (bullish)")
    else:
        s -= 10
        notes.append("MACD di bawah signal (bearish)")
    if p.macd <= p.macd_sig and l.macd > l.macd_sig:
        s += 5
        notes.append("MACD baru golden cross")
    if p.macd >= p.macd_sig and l.macd < l.macd_sig:
        s -= 5
        notes.append("MACD baru death cross")
    s += 5 if l.macd_hist > p.macd_hist else -5

    r = l.rsi if not np.isnan(l.rsi) else 50.0
    if r >= 75:
        s -= 10
        notes.append(f"RSI {r:.0f}: overbought, rawan koreksi")
    elif r >= 50:
        s += 10
        notes.append(f"RSI {r:.0f}: momentum bullish")
    elif r >= 30:
        s -= 10
        notes.append(f"RSI {r:.0f}: momentum lemah")
    else:
        s += 5
        notes.append(f"RSI {r:.0f}: oversold, potensi teknikal rebound (berisiko)")

    vr = l.vol_ratio if not np.isnan(l.vol_ratio) else 1.0
    if vr > 1.3:
        if l.close >= l.open:
            s += 10
            notes.append(f"Volume {vr:.1f}x rata-rata pada candle hijau (akumulasi)")
        else:
            s -= 10
            notes.append(f"Volume {vr:.1f}x rata-rata pada candle merah (distribusi)")

    if sup and (price - sup[0][0]) <= 1.0 * atr and trend["points"] >= 0:
        s += 10
        notes.append("Harga dekat support (area buy on weakness)")
    if res and (res[0][0] - price) <= 0.5 * atr:
        s -= 10
        notes.append("Harga menempel resistance (butuh breakout)")
    if not np.isnan(l.bb_up) and price > l.bb_up:
        s -= 5
        notes.append("Harga di atas Bollinger atas (extended)")
    return int(max(-100, min(100, round(s)))), notes


def rating_from_bias(b):
    if b >= 45:
        return "STRONG BUY SETUP"
    if b >= 20:
        return "BUY / AKUMULASI"
    if b > -20:
        return "NETRAL / WAIT"
    if b > -45:
        return "HINDARI"
    return "BEARISH KUAT"


# --------------------------------------------------------------------------------------
# TRADE PLAN
# --------------------------------------------------------------------------------------
def build_plan(df, price, atr, sup, res, fib, bias, cfg, idx, capital=None, risk_pct=None, htf_down=False):
    l = df.iloc[-1]
    kmin, kmax = cfg["atr_k"]
    plan = dict(side="NO TRADE", reason="", watch=[])

    if bias < 20 or htf_down and bias < 45:
        reason = "Skor teknikal belum cukup bullish" if bias < 20 else "Tren timeframe besar masih turun (lawan arah)"
        plan["reason"] = reason
        if res:
            plan["watch"].append(f"Tunggu breakout + close di atas resistance {px(res[0][0], idx)} dengan volume naik")
        if sup:
            plan["watch"].append(f"Atau tunggu pantulan valid (candle reversal) dari support {px(sup[0][0], idx)}")
        plan["watch"].append("Saham IDX tidak praktis untuk short-sell ritel: bias bearish = menunggu/menghindari.")
        return plan

    rsi = l.rsi if not np.isnan(l.rsi) else 50
    extended = rsi > 72 or (price - l.ema21) > 2.0 * atr
    cands = [x for x in (l.ema21, l.sma20) if not np.isnan(x)]
    if sup:
        cands.append(sup[0][0])
    if fib["direction"] == "UP-swing":
        cands += [fib["retracement"][0.382], fib["retracement"][0.5]]
    cands = [x for x in cands if price - 3 * atr < x < price - 0.3 * atr]
    pullback = max(cands) if cands else price - 1.0 * atr

    if extended:
        entry, status = rnd(pullback, idx), "TUNGGU PULLBACK (harga sudah extended)"
    else:
        entry, status = rnd(price, idx), "ENTRY SEKARANG / bertahap (limit di sekitar harga)"

    below = [s[0] for s in sup if s[0] < entry - 0.2 * atr]
    sl_struct = (below[0] - 0.25 * atr) if below else entry - kmax * atr
    risk = min(max(entry - sl_struct, kmin * atr), kmax * atr)
    if risk / entry > cfg["max_loss"]:
        risk = entry * cfg["max_loss"]
    sl = rnd(entry - risk, idx, "down")
    risk = entry - sl
    if risk <= 0:
        sl = entry - tick_size(entry, idx)
        risk = entry - sl
    cut = rnd(sl - 0.5 * atr, idx, "down")

    tps = [entry + k * risk for k in cfg["rr"]]
    if res and res[0][0] > entry + 1.0 * risk and res[0][0] < tps[0]:
        tps[0] = res[0][0]
    for i in range(1, 3):
        tps[i] = max(tps[i], tps[i - 1] + tick_size(tps[i - 1], idx))
    tps = [rnd(t, idx, "down") for t in tps]

    plan.update(
        side="BUY (LONG)", status=status, entry=entry, entry_pullback=rnd(pullback, idx),
        stop_loss=sl, cut_loss=cut, targets=tps,
        rr=[(t - entry) / risk for t in tps], risk_per_share=risk, risk_pct=risk / entry,
        fib_targets=[rnd(v, idx) for k, v in fib["extension"].items() if k > 1.0 and v > entry],
        trail=cfg["trail"],
    )
    if capital and risk_pct:
        shares_by_risk = (capital * risk_pct / 100) / risk
        lot_size = 100 if idx else 1
        lots = int(min(shares_by_risk, capital / entry) // lot_size)
        plan["position"] = dict(lots=lots, shares=lots * lot_size, value=lots * lot_size * entry,
                                max_loss=lots * lot_size * risk)
    return plan


# --------------------------------------------------------------------------------------
# ANALISIS TEKNIKAL PER METODE
# --------------------------------------------------------------------------------------
def analyze_method(sym, method, idx, capital=None, risk_pct=None):
    cfg = METHODS[method]
    df = add_indicators(get_history(sym, *cfg["tf"]))
    if len(df) < 60:
        raise ValueError(f"Data {cfg['tf'][0]} terlalu sedikit ({len(df)} bar) untuk analisis.")
    hi, lo = pivots(df, 5)
    l = df.iloc[-1]
    price, atr = float(l.close), float(l.atr)

    sup, res = support_resistance(df, hi, lo, price, atr)
    trend = trend_info(df, hi, lo)
    fib = fibonacci(df, cfg["fib_window"])

    htf_trend = None
    try:
        hdf = add_indicators(get_history(sym, *cfg["htf"]))
        hh, hl = pivots(hdf, 4)
        htf_trend = trend_info(hdf, hh, hl)
    except Exception:
        pass

    bias, notes = bias_score(df, trend, sup, res, atr, price)
    htf_down = bool(htf_trend and htf_trend["label"] == "DOWNTREND")
    if htf_trend:
        if htf_trend["label"] == "UPTREND":
            bias = min(100, bias + 10)
            notes.append(f"Timeframe besar ({cfg['htf'][0]}) UPTREND: searah")
        elif htf_down:
            bias = max(-100, bias - 15)
            notes.append(f"Timeframe besar ({cfg['htf'][0]}) DOWNTREND: melawan arus")
        else:
            notes.append(f"Timeframe besar ({cfg['htf'][0]}) SIDEWAYS")

    plan = build_plan(df, price, atr, sup, res, fib, bias, cfg, idx, capital, risk_pct, htf_down)
    avg_val = float((df["close"] * df["volume"]).tail(60).mean())
    age_min = None
    try:
        ts = df.index[-1]
        now = pd.Timestamp.now(tz=ts.tz) if ts.tzinfo else pd.Timestamp.now()
        age_min = (now - ts).total_seconds() / 60
    except Exception:
        pass

    dyn = {k: float(l[k]) for k in ("ema9", "ema21", "sma20", "sma50", "sma200") if not np.isnan(l[k])}
    return dict(
        method=method, label=cfg["label"], interval=cfg["tf"][0], htf=cfg["htf"][0],
        last_bar=str(df.index[-1]), bar_age_min=age_min, price=price, atr=atr,
        ohlc=dict(open=float(l.open), high=float(l.high), low=float(l.low), close=price, volume=float(l.volume)),
        trend=trend, htf_trend=htf_trend, support=sup, resistance=res, pivot_points=daily_pivot_points(df) if cfg["tf"][0] != "1d" else None,
        fib=fib, ma=dyn,
        rsi=float(l.rsi), stoch=(float(l.stoch_k), float(l.stoch_d)),
        macd=dict(macd=float(l.macd), signal=float(l.macd_sig), hist=float(l.macd_hist)),
        bb=(float(l.bb_lo), float(l.bb_up)), vol_ratio=float(l.vol_ratio),
        bias=bias, rating=rating_from_bias(bias), notes=notes, plan=plan, avg_value=avg_val,
    )


# --------------------------------------------------------------------------------------
# FUNDAMENTAL, SEKTOR, BISNIS, BERITA
# --------------------------------------------------------------------------------------
_FUND_CACHE = {}


def _get_fundamentals_yahoo(sym, translate=False, ttl=6 * 3600):
    if sym in _FUND_CACHE and time.time() - _FUND_CACHE[sym][0] < ttl:
        return _FUND_CACHE[sym][1]
    import yfinance as yf
    t = yf.Ticker(sym)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    dy = None
    if info.get("dividendRate") and price:
        dy = info["dividendRate"] / price
    f = dict(
        name=info.get("longName") or info.get("shortName") or sym,
        sector=info.get("sector"), industry=info.get("industry"),
        website=info.get("website"), employees=info.get("fullTimeEmployees"),
        summary=info.get("longBusinessSummary"),
        market_cap=info.get("marketCap"), beta=info.get("beta"),
        per=info.get("trailingPE"), fwd_per=info.get("forwardPE"), pbv=info.get("priceToBook"),
        ev_ebitda=info.get("enterpriseToEbitda"), eps=info.get("trailingEps"), bvps=info.get("bookValue"),
        roe=info.get("returnOnEquity"), roa=info.get("returnOnAssets"),
        gpm=info.get("grossMargins"), opm=info.get("operatingMargins"), npm=info.get("profitMargins"),
        rev_growth=info.get("revenueGrowth"), earn_growth=info.get("earningsGrowth"),
        der=info.get("debtToEquity"), current_ratio=info.get("currentRatio"), quick_ratio=info.get("quickRatio"),
        total_debt=info.get("totalDebt"), total_cash=info.get("totalCash"),
        fcf=info.get("freeCashflow"), ocf=info.get("operatingCashflow"),
        div_yield=dy, payout=info.get("payoutRatio"),
        hi52=info.get("fiftyTwoWeekHigh"), lo52=info.get("fiftyTwoWeekLow"),
        shares=info.get("sharesOutstanding"), float_shares=info.get("floatShares"),
        held_insiders=info.get("heldPercentInsiders"), held_inst=info.get("heldPercentInstitutions"),
        history=[], news=[],
    )
    try:
        inc = t.income_stmt
        if inc is not None and not inc.empty:
            for col in list(inc.columns)[:4]:
                f["history"].append(dict(
                    year=str(col)[:4],
                    revenue=inc.at["Total Revenue", col] if "Total Revenue" in inc.index else None,
                    net_income=inc.at["Net Income", col] if "Net Income" in inc.index else None,
                ))
    except Exception:
        pass
    try:
        for n in (t.news or [])[:6]:
            c = n.get("content", n)
            title = c.get("title")
            pub = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
            if title:
                f["news"].append(f"{title} ({pub or '-'})")
    except Exception:
        pass

    if translate and f["summary"]:
        try:
            from deep_translator import GoogleTranslator
            f["summary_id"] = GoogleTranslator(source="auto", target="id").translate(f["summary"][:4500])
        except Exception:
            pass
    f["score"], f["score_notes"] = score_fundamental(f)
    _FUND_CACHE[sym] = (time.time(), f)
    return f


def score_fundamental(f):
    fin = (f.get("sector") or "") == "Financial Services"
    items, notes = [], []

    def band(name, v, good, ok, lower_better=False, fmt=num):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return
        if lower_better:
            sc = 1.0 if 0 < v <= good else 0.5 if 0 < v <= ok else 0.0
        else:
            sc = 1.0 if v >= good else 0.5 if v >= ok else 0.0
        items.append(sc)
        notes.append(f"{name}: {fmt(v)} -> {'baik' if sc == 1 else 'cukup' if sc == 0.5 else 'lemah'}")

    band("PER", f["per"], 15, 25, lower_better=True)
    band("PBV", f["pbv"], 1.5, 3, lower_better=True)
    band("ROE", f["roe"], 0.15, 0.08, fmt=pct)
    band("Net margin", f["npm"], 0.10, 0.05, fmt=pct)
    band("Pertumbuhan pendapatan", f["rev_growth"], 0.10, 0.0, fmt=pct)
    band("Pertumbuhan laba", f["earn_growth"], 0.10, 0.0, fmt=pct)
    if not fin:
        band("DER (%)", f["der"], 50, 100, lower_better=True)
        band("Current ratio", f["current_ratio"], 1.5, 1.0)
    if f.get("fcf") is not None:
        items.append(1.0 if f["fcf"] > 0 else 0.0)
        notes.append(f"Free cash flow: {big(f['fcf'])} -> {'positif' if f['fcf'] > 0 else 'negatif'}")
    if not items:
        return None, ["Data fundamental tidak tersedia dari sumber."]
    return round(100 * sum(items) / len(items), 1), notes


# --------------------------------------------------------------------------------------
# LAPORAN
# --------------------------------------------------------------------------------------
LINE = "=" * 78


def print_fundamental(f, idx):
    print(f"\n{LINE}\nPROFIL & FUNDAMENTAL: {f['name']}\n{LINE}")
    print(f"Sektor / Industri : {f['sector'] or '-'} / {f['industry'] or '-'}")
    print(f"Website           : {f['website'] or '-'}   Karyawan: {f['employees'] or '-'}")
    summ = f.get("summary_id") or f.get("summary")
    if summ:
        print("\nBisnis & rencana usaha (ringkasan perusahaan):")
        print(summ if len(summ) < 1400 else summ[:1400] + " ...")
    if f["news"]:
        print("\nBerita/aksi korporasi terbaru (cek untuk rencana ekspansi/aksi korporasi):")
        for n in f["news"]:
            print(f"  - {n}")
    print("\nValuasi")
    print(f"  Market cap {big(f['market_cap'], idx)} | PER {num(f['per'])} | Fwd PER {num(f['fwd_per'])} | "
          f"PBV {num(f['pbv'])} | EV/EBITDA {num(f['ev_ebitda'])}")
    print(f"  EPS {num(f['eps'])} | BVPS {num(f['bvps'])} | Div. yield {pct(f['div_yield'])} | "
          f"Payout {pct(f['payout'])} | Beta {num(f['beta'])}")
    print("Profitabilitas & pertumbuhan")
    print(f"  ROE {pct(f['roe'])} | ROA {pct(f['roa'])} | Gross {pct(f['gpm'])} | Operating {pct(f['opm'])} | "
          f"Net {pct(f['npm'])}")
    print(f"  Pertumbuhan pendapatan {pct(f['rev_growth'])} | Pertumbuhan laba {pct(f['earn_growth'])}")
    print("Solvabilitas & arus kas")
    print(f"  DER {num(f['der'])}% | Current ratio {num(f['current_ratio'])} | Total utang {big(f['total_debt'], idx)} | "
          f"Kas {big(f['total_cash'], idx)}")
    print(f"  Operating CF {big(f['ocf'], idx)} | Free CF {big(f['fcf'], idx)}")
    print(f"Kepemilikan: insider {pct(f['held_insiders'])} | institusi {pct(f['held_inst'])} | "
          f"52w High {px(f['hi52'], idx)} | 52w Low {px(f['lo52'], idx)}")
    if f["history"]:
        print("Tren laporan tahunan (pendapatan / laba bersih):")
        for h in f["history"]:
            print(f"  {h['year']}: {big(h['revenue'], idx)} / {big(h['net_income'], idx)}")
    print(f"\nSKOR FUNDAMENTAL: {f['score'] if f['score'] is not None else 'n/a'} / 100")
    for n in f["score_notes"]:
        print(f"  * {n}")


def print_technical(a, idx):
    p = a["plan"]
    print(f"\n{LINE}\nTEKNIKAL - {a['label']}  [TF entry {a['interval']} | TF besar {a['htf']}]\n{LINE}")
    age = f" (usia data ~{a['bar_age_min']:.0f} menit)" if a["bar_age_min"] is not None else ""
    print(f"Bar terakhir: {a['last_bar']}{age}")
    o = a["ohlc"]
    print(f"Harga {px(a['price'], idx)} | O {px(o['open'], idx)} H {px(o['high'], idx)} L {px(o['low'], idx)} | "
          f"Vol {o['volume']:,.0f} | ATR {px(a['atr'], idx) if idx else num(a['atr'])} | "
          f"Nilai transaksi rata-rata/bar {big(a['avg_value'], idx)}")
    t = a["trend"]
    print(f"\nTREND: {t['label']} (kekuatan {t['strength']:.0%}) | struktur: {t['structure']}")
    if a["htf_trend"]:
        print(f"Trend TF besar ({a['htf']}): {a['htf_trend']['label']}")
    print("\nMoving average : " + " | ".join(f"{k.upper()} {px(v, idx)}" for k, v in a["ma"].items()))
    print(f"RSI(14) {a['rsi']:.1f} | Stoch %K/%D {a['stoch'][0]:.0f}/{a['stoch'][1]:.0f} | "
          f"MACD {a['macd']['macd']:.2f} sig {a['macd']['signal']:.2f} hist {a['macd']['hist']:.2f}")
    print(f"Bollinger: bawah {px(a['bb'][0], idx)} - atas {px(a['bb'][1], idx)} | Volume vs rata-rata: {a['vol_ratio']:.1f}x")
    print("\nSupport    : " + (", ".join(f"{px(v, idx)} ({n}x sentuh)" for v, n in a["support"]) or "-"))
    print("Resistance : " + (", ".join(f"{px(v, idx)} ({n}x sentuh)" for v, n in a["resistance"]) or "-"))
    if a["pivot_points"]:
        pp = a["pivot_points"]
        print("Pivot hari lalu: " + " | ".join(f"{k} {px(v, idx)}" for k, v in pp.items()))
    fb = a["fib"]
    print(f"\nFIBONACCI ({fb['direction']}: low {px(fb['low'], idx)} -> high {px(fb['high'], idx)}) | posisi: {fb['zone']}")
    print("  Retracement: " + " | ".join(f"{k:.1%}={px(v, idx)}" for k, v in fb["retracement"].items()))
    print("  Extension  : " + " | ".join(f"{k}={px(v, idx)}" for k, v in fb["extension"].items()))
    print("\nSinyal:")
    for n in a["notes"]:
        print(f"  * {n}")
    print(f"\nSKOR BIAS TEKNIKAL: {a['bias']:+d} (-100..+100)  =>  {a['rating']}")

    print(f"\n--- TRADE PLAN ({a['method'].upper()}) ---")
    if p["side"] == "NO TRADE":
        print(f"Keputusan : NO TRADE - {p['reason']}")
        for w in p["watch"]:
            print(f"  > {w}")
        return
    print(f"Keputusan : {p['side']} | {p['status']}")
    print(f"Entry     : {px(p['entry'], idx)}   (alternatif buy-on-pullback: {px(p['entry_pullback'], idx)})")
    print(f"Stop loss : {px(p['stop_loss'], idx)}   (risiko {p['risk_pct']:.2%} / {px(p['risk_per_share'], idx)} per saham)")
    print(f"Cut loss  : {px(p['cut_loss'], idx)}   (batas darurat: keluar tanpa menunggu bila tembus/gap down)")
    for i, (tp, rr) in enumerate(zip(p["targets"], p["rr"]), 1):
        print(f"TP{i}       : {px(tp, idx)}   (RR 1:{rr:.1f})")
    if p["fib_targets"]:
        print("Target ext. fibonacci: " + ", ".join(px(v, idx) for v in p["fib_targets"]))
    if p.get("position"):
        ps = p["position"]
        print(f"Ukuran posisi: {ps['lots']} lot ({ps['shares']:,} saham) | nilai {big(ps['value'], idx)} | "
              f"rugi maks jika SL {big(ps['max_loss'], idx)}")
    print(p["trail"])


def print_full(sym, f, analyses, idx):
    if f:
        print_fundamental(f, idx)
    for a in analyses:
        print_technical(a, idx)
    print(f"\n{LINE}\nKESIMPULAN\n{LINE}")
    for a in analyses:
        cfg = METHODS[a["method"]]
        tech_n = (a["bias"] + 100) / 2
        fs = f["score"] if f and f.get("score") is not None else None
        overall = tech_n if fs is None or cfg["w_fund"] == 0 else cfg["w_tech"] * tech_n + cfg["w_fund"] * fs
        a["overall"] = round(overall, 1)
        print(f"{a['method'].upper():9s}: skor total {a['overall']:.0f}/100 | {a['rating']} | "
              f"{a['plan']['side']}" + (f" @ {px(a['plan']['entry'], idx)} SL {px(a['plan']['stop_loss'], idx)}"
                                       if a['plan']['side'] != 'NO TRADE' else ""))
    print("\nDisclaimer: alat bantu analisis, bukan rekomendasi. Gunakan manajemen risiko; data bisa delay/tidak akurat.")


# --------------------------------------------------------------------------------------
# RUNNER
# --------------------------------------------------------------------------------------
def analyze_symbol(sym, methods, idx, with_fund, translate, capital, risk_pct):
    # fundamental + semua metode teknikal diambil paralel (jauh lebih cepat dari berurutan)
    with ThreadPoolExecutor(max_workers=len(methods) + 1) as ex:
        ff = ex.submit(get_fundamentals, sym, translate) if with_fund else None
        fa = [ex.submit(analyze_method, sym, m, idx, capital, risk_pct) for m in methods]
        analyses = [x.result() for x in fa]
        f = None
        if ff is not None:
            try:
                f = ff.result()
            except Exception as e:
                print(f"[warn] fundamental {sym} gagal: {str(e)[:80]}")
    return f, analyses


def run_single(args, idx):
    sym = norm_ticker(args.tickers[0], args.market)
    methods = list(METHODS) if args.method == "all" else [args.method]
    f, analyses = analyze_symbol(sym, methods, idx, not args.no_fund, args.translate, args.capital, args.risk_pct)
    print(f"\n#### {sym}  |  diproses {pd.Timestamp.now():%Y-%m-%d %H:%M:%S} ####")
    print_full(sym, f, analyses, idx)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(dict(symbol=sym, fundamental=f, technical=analyses), fh, indent=2, default=str, ensure_ascii=False)
        print(f"\nHasil disimpan ke {args.json}")


def run_scan(args, idx):
    tickers = list(args.scan or [])
    if args.scan_file:
        with open(args.scan_file) as fh:
            tickers += [x.strip() for x in fh.read().replace(",", "\n").split() if x.strip()]
    if not tickers:
        tickers = LQ_UNIVERSE
    tickers = [norm_ticker(t, args.market) for t in dict.fromkeys(tickers)]
    method = "swing" if args.method == "all" else args.method
    cfg = METHODS[method]

    def work(sym):
        try:
            f, an = analyze_symbol(sym, [method], idx, not args.no_fund and cfg["w_fund"] > 0, False,
                                   args.capital, args.risk_pct)
            a = an[0]
            tech_n = (a["bias"] + 100) / 2
            fs = f["score"] if f and f.get("score") is not None else None
            overall = tech_n if fs is None else cfg["w_tech"] * tech_n + cfg["w_fund"] * fs
            p = a["plan"]
            return dict(sym=sym, price=a["price"], trend=a["trend"]["label"], rsi=a["rsi"], bias=a["bias"],
                        fund=fs, score=overall, rating=a["rating"], side=p["side"],
                        entry=p.get("entry"), sl=p.get("stop_loss"), tp1=(p.get("targets") or [None])[0],
                        sector=(f or {}).get("sector"))
        except Exception as e:
            return dict(sym=sym, error=str(e)[:60])

    print(f"Scanning {len(tickers)} saham | metode {method} ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(work, tickers))
    ok = sorted([r for r in rows if "error" not in r], key=lambda r: -r["score"])
    ok = [r for r in ok if r["score"] >= args.min_score][: args.top]
    print(f"\n{'KODE':8s}{'SKOR':>6s}{'TEKN':>6s}{'FUND':>6s} {'TREND':10s}{'RSI':>5s} {'SINYAL':18s}"
          f"{'ENTRY':>9s}{'SL':>9s}{'TP1':>9s}  SEKTOR")
    for r in ok:
        fund_s = f"{r['fund']:.0f}" if r["fund"] is not None else "-"
        print(f"{r['sym'].replace('.JK', ''):8s}{r['score']:6.0f}{r['bias']:6d}"
              f"{fund_s:>6s} {r['trend']:10s}{r['rsi']:5.0f} "
              f"{r['rating']:18s}{px(r['entry'], idx):>9s}{px(r['sl'], idx):>9s}{px(r['tp1'], idx):>9s}  {r['sector'] or '-'}")
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\n{len(errs)} saham gagal diambil: " + ", ".join(f"{e['sym']}" for e in errs[:15]))
    print("\nDetail lengkap satu saham: python screener_saham.py KODE --method swing")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(ok, fh, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description="Screener saham: fundamental + teknikal + trade plan")
    ap.add_argument("tickers", nargs="*", help="kode saham, mis. BBCA (mode detail)")
    ap.add_argument("--method", choices=["swing", "daytrade", "scalping", "all"], default="swing")
    ap.add_argument("--market", choices=["idx", "global"], default="idx", help="idx: otomatis tambah .JK")
    ap.add_argument("--scan", nargs="*", help="mode screening banyak saham (kosong = universe default)")
    ap.add_argument("--scan-file", help="file berisi daftar kode saham")
    ap.add_argument("--min-score", type=float, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-fund", action="store_true", help="lewati data fundamental (lebih cepat)")
    ap.add_argument("--translate", action="store_true", help="terjemahkan deskripsi bisnis ke Indonesia")
    ap.add_argument("--capital", type=float, help="modal (Rp) untuk hitung lot")
    ap.add_argument("--risk-pct", type=float, help="risiko per trade (%% dari modal), mis. 1")
    ap.add_argument("--watch", type=int, help="refresh otomatis tiap N detik (near-realtime)")
    ap.add_argument("--json", help="simpan hasil ke file JSON")
    ap.add_argument("--source", choices=["auto", "tv", "yahoo"], default="auto",
                    help="sumber harga: auto = TradingView lalu Yahoo (default)")
    args = ap.parse_args()
    global _SOURCE
    _SOURCE = args.source

    idx = args.market == "idx"
    if not args.tickers and args.scan is None and not args.scan_file:
        ap.error("isi kode saham, atau gunakan --scan / --scan-file")
    scan_mode = args.scan is not None or args.scan_file

    def once():
        run_scan(args, idx) if scan_mode else run_single(args, idx)

    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                try:
                    once()
                except Exception as e:
                    print(f"Error: {e}")
                print(f"\n[watch] refresh tiap {args.watch} detik. Ctrl+C untuk berhenti.")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nBerhenti.")
    else:
        try:
            once()
        except Exception as e:
            sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()

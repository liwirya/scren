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
import hashlib
import pickle
import random
import re
import tempfile
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Kompatibilitas Python < 3.9: reportlab memanggil md5(usedforsecurity=False) yang belum didukung di sana.
try:
    hashlib.md5(usedforsecurity=False)
except TypeError:
    _md5_orig = hashlib.md5

    def _md5_compat(*a, **k):
        k.pop("usedforsecurity", None)
        return _md5_orig(*a, **k)

    hashlib.md5 = _md5_compat

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

# Perkiraan LQ45 (konstituen resmi BEI berganti tiap evaluasi; periode 3 Agu-30 Okt 2026: INDY & NCKL masuk,
# SMGR & TOWR keluar). Untuk daftar resmi/terbaru pakai --scan-file dengan daftar dari idx.co.id.
LQ_UNIVERSE = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BRIS", "TLKM", "ISAT", "EXCL", "MTEL",
    "ASII", "UNTR", "UNVR", "ICBP", "INDF", "CPIN", "JPFA", "KLBF", "MAPI", "ACES", "ERAA",
    "ADRO", "PTBA", "ITMG", "HRUM", "MEDC", "PGAS", "AKRA", "ANTM", "MDKA", "INCO", "TINS",
    "AMMN", "BRPT", "TPIA", "BREN", "INTP", "CTRA", "BSDE", "PWON", "JSMR",
    "GOTO", "EMTK", "ESSA", "INDY", "NCKL", "BUMI", "DSSA", "HEAL",
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


_FETCH_SEM = threading.Semaphore(3)   # maks request harga bersamaan (diset dari --fetch-limit)
_TRANSIENT = ("too many", "429", "rate limit", "rate-limit", "timed out", "timeout", "connection",
              "temporar", "max retries", "reset by peer")


def _fetch(fn, sym, interval, period, tries=3):
    """Panggil sumber data dgn batas konkurensi + retry exponential backoff utk error sementara."""
    last = None
    for i in range(tries):
        try:
            with _FETCH_SEM:
                return fn(sym, interval, period)
        except Exception as e:
            last = e
            if i == tries - 1 or not any(k in str(e).lower() for k in _TRANSIENT):
                break
            time.sleep(1.5 * (2 ** i) + random.random())      # tunggu di luar semaphore
    raise last


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
            df = _fetch(_hist_tv, sym, interval, period)
        except Exception as e:
            if _SOURCE == "tv":
                raise
            if "tv" not in _WARNED:
                _WARNED.add("tv")
                print(f"[info] TradingView gagal ({str(e)[:90]}) -> pakai Yahoo (lebih lambat & rawan rate-limit, "
                      f"terutama dari server cloud). Pasang tvdatafeed agar cepat.")
    if df is None:
        df = _fetch(_hist_yahoo, sym, interval, period)
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
        chart_df=df.tail(160).copy(),
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
# UNIVERSE: LQ45 (perkiraan) / SELURUH saham IDX (scanner TradingView)
# --------------------------------------------------------------------------------------
_UNI_FILE = os.path.join(tempfile.gettempdir(), "screener_universe.json")


def _scanner_all(ttl=6 * 3600):
    """Daftar semua saham IDX + harga/volume/market cap/sektor dari TradingView scanner (1 request)."""
    import urllib.request
    try:
        with open(_UNI_FILE, encoding="utf-8") as fh:
            c = json.load(fh)
        if time.time() - c["t"] < ttl and c["rows"]:
            return c["rows"]
    except Exception:
        pass
    col_sets = [
        ["name", "close", "volume", "market_cap_basic", "sector", "average_volume_30d_calc"],
        ["name", "close", "volume", "market_cap_basic", "sector"],
        ["name", "close", "volume"],
    ]
    last_err = None
    for cols in col_sets:
        payload = {"filter": [], "options": {"lang": "en"}, "markets": ["indonesia"],
                   "symbols": {"query": {"types": []}, "tickers": []}, "columns": cols,
                   "sort": {"sortBy": "volume", "sortOrder": "desc"}, "range": [0, 1500]}
        req = urllib.request.Request(
            "https://scanner.tradingview.com/indonesia/scan", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                js = json.loads(r.read().decode())
        except Exception as e:
            last_err = e
            continue
        rows = []
        for it in js.get("data", []):
            code = str(it.get("s", "")).split(":")[-1]
            if not re.fullmatch(r"[A-Z]{4}", code):      # buang waran/rights/kode non-saham
                continue
            d = dict(zip(cols, it.get("d", [])))
            rows.append(dict(code=code, close=d.get("close"), vol=d.get("volume"),
                             avgvol=d.get("average_volume_30d_calc"), mcap=d.get("market_cap_basic"),
                             sector=d.get("sector")))
        if rows:
            try:
                with open(_UNI_FILE, "w", encoding="utf-8") as fh:
                    json.dump({"t": time.time(), "rows": rows}, fh)
            except Exception:
                pass
            return rows
    raise RuntimeError(f"Scanner TradingView gagal ({last_err}). Alternatif: --scan-file daftar.txt")


def load_all_idx(args):
    rows = _scanner_all()
    for r in rows:
        r["value"] = (r["close"] or 0) * (r["avgvol"] or r["vol"] or 0)     # nilai transaksi/hari (Rp)
    keep = [r for r in rows if (r["close"] or 0) >= args.min_price and r["value"] >= args.min_value]
    keep.sort(key=lambda r: -r["value"])
    if args.max_stocks and args.max_stocks > 0:
        keep = keep[: args.max_stocks]
    note = (f"Seluruh saham IDX: {len(rows)} kode di scanner -> {len(keep)} lolos filter "
            f"(harga >= {args.min_price:,.0f}; nilai transaksi >= Rp {args.min_value:,.0f}/hari)")
    return [r["code"] for r in keep], note


# --------------------------------------------------------------------------------------
# RINGKASAN NARASI (Indonesia) + PDF
# --------------------------------------------------------------------------------------
def calc_overall(f, a):
    cfg = METHODS[a["method"]]
    tech_n = (a["bias"] + 100) / 2
    fs = f["score"] if f and f.get("score") is not None else None
    return tech_n if fs is None or cfg["w_fund"] == 0 else cfg["w_tech"] * tech_n + cfg["w_fund"] * fs


def _make_row(sym, f, a):
    p = a["plan"]
    return dict(sym=sym, price=a["price"], trend=a["trend"]["label"], rsi=a["rsi"], bias=a["bias"],
                fund=(f["score"] if f and f.get("score") is not None else None),
                score=calc_overall(f, a), rating=a["rating"], side=p["side"], entry=p.get("entry"),
                sl=p.get("stop_loss"), tp1=(p.get("targets") or [None])[0],
                sector=(f or {}).get("sector"), _f=f, _a=a)


def narrative(sym, f, analyses, idx=True):
    code = sym.replace(".JK", "")
    a = analyses[0]
    t, price = a["trend"], a["price"]
    out = []
    s = (f"{code} berada di {px(price, idx)} dan dalam {t['label']} pada timeframe {a['interval']} "
         f"(kekuatan {t['strength']:.0%}; struktur {t['structure']}).")
    if a.get("htf_trend"):
        same = a["htf_trend"]["label"] == t["label"]
        s += (f" Timeframe lebih besar ({a['htf']}) berstatus {a['htf_trend']['label']}"
              + (", sehingga searah." if same else ", jadi sinyal belum sepenuhnya searah."))
    out.append(s)

    rsi = a["rsi"]
    zone = ("jenuh beli (rawan koreksi)" if rsi > 70 else "kuat/bullish" if rsi >= 55 else "netral" if rsi >= 45
            else "lemah" if rsi >= 30 else "jenuh jual (potensi pantulan)")
    m = a["macd"]
    s = (f"Momentum: RSI {rsi:.0f} ({zone}); MACD "
         f"{'di atas' if m['macd'] > m['signal'] else 'di bawah'} garis sinyal "
         f"(histogram {'positif' if m['hist'] > 0 else 'negatif'}). ")
    vr = a["vol_ratio"]
    s += (f"Volume {vr:.1f}x rata-rata, " + ("mendukung pergerakan." if vr >= 1.3 else "relatif sepi." if vr < 0.7
                                                else "normal."))
    out.append(s)

    s = "Level kunci: "
    if a["support"]:
        v = a["support"][0][0]
        s += f"support terdekat {px(v, idx)} ({(v - price) / price:+.1%}) "
    if a["resistance"]:
        v = a["resistance"][0][0]
        s += f"dan resistance terdekat {px(v, idx)} ({(v - price) / price:+.1%}). "
    s += f"Fibonacci: {a['fib']['zone']} ({a['fib']['direction']})."
    out.append(s)

    if f and f.get("score") is not None:
        good = [n.split(" -> ")[0] for n in f["score_notes"] if n.endswith("baik") or n.endswith("positif")][:3]
        weak = [n.split(" -> ")[0] for n in f["score_notes"] if n.endswith("lemah") or n.endswith("negatif")][:3]
        s = f"Fundamental ({f.get('sector') or '-'}): skor {f['score']:.0f}/100."
        if good:
            s += " Kuat di: " + "; ".join(good) + "."
        if weak:
            s += " Lemah di: " + "; ".join(weak) + "."
        out.append(s)
    else:
        out.append("Fundamental: data tidak tersedia dari sumber, keputusan hanya berbasis teknikal.")

    for an in analyses:
        p = an["plan"]
        if p["side"] == "NO TRADE":
            out.append(f"{an['method'].upper()}: belum ada setup beli ({p['reason']}).")
        else:
            out.append(f"{an['method'].upper()}: {p['status'].lower()}; entry {px(p['entry'], idx)}, stop loss "
                       f"{px(p['stop_loss'], idx)}, cut loss {px(p['cut_loss'], idx)}, TP1 {px(p['targets'][0], idx)} "
                       f"(RR 1:{p['rr'][0]:.1f}), TP3 {px(p['targets'][2], idx)}.")
    risks = []
    if rsi > 70:
        risks.append("RSI jenuh beli")
    if a.get("htf_trend") and a["htf_trend"]["label"] == "DOWNTREND":
        risks.append("melawan tren timeframe besar")
    if a.get("avg_value") and a["avg_value"] < 1e9:
        risks.append("likuiditas rendah (slippage/spread lebar)")
    if f and f.get("score") is not None and f["score"] < 40:
        risks.append("fundamental lemah")
    if risks:
        out.append("Risiko utama: " + ", ".join(risks) + ".")
    return out


def _chart_png_impl(a, idx=True):
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    df = a.get("chart_df")
    if df is None or len(df) < 20:
        return None
    n = len(df)
    x = np.arange(n)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.4, 4.3), sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    col = np.where((df.close >= df.open).values, "#1b9e5a", "#d6403a")
    ax.vlines(x, df.low.values, df.high.values, colors=col, linewidth=0.7)
    ax.bar(x, np.maximum((df.close - df.open).abs().values, 1e-9), bottom=np.minimum(df.open, df.close).values,
           width=0.65, color=col)
    for c, lab, cc in (("ema9", "EMA9", "#f0a500"), ("ema21", "EMA21", "#1f77b4"),
                       ("sma50", "SMA50", "#7f3fbf"), ("sma200", "SMA200", "#444444")):
        if c in df and df[c].notna().sum() > 5:
            ax.plot(x, df[c].values, lw=0.9, color=cc, label=lab)
    lo, hi = float(df.low.min()), float(df.high.max())
    for v, _ in a["support"][:2]:
        ax.axhline(v, color="#1b9e5a", ls="--", lw=0.7, alpha=0.8)
    for v, _ in a["resistance"][:2]:
        ax.axhline(v, color="#d6403a", ls="--", lw=0.7, alpha=0.8)
    p = a["plan"]
    if p["side"] != "NO TRADE":
        ax.axhline(p["entry"], color="#1f4fd8", lw=0.9)
        ax.axhline(p["stop_loss"], color="#d6403a", lw=1.0)
        for tp in p["targets"]:
            ax.axhline(tp, color="#1b9e5a", ls=":", lw=0.9)
        ax.text(n - 1, p["entry"], " ENTRY", fontsize=6, color="#1f4fd8", va="bottom", ha="right")
        ax.text(n - 1, p["stop_loss"], " SL", fontsize=6, color="#d6403a", va="top", ha="right")
        lo, hi = min(lo, p["stop_loss"]), max(hi, p["targets"][0])
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=6, ncol=4, loc="upper left", frameon=False)
    axr.plot(x, df.rsi.values, color="#555", lw=0.8)
    axr.axhline(70, color="#d6403a", lw=0.5, ls="--")
    axr.axhline(30, color="#1b9e5a", lw=0.5, ls="--")
    axr.set_ylim(0, 100)
    axr.set_ylabel("RSI", fontsize=6)
    axr.tick_params(labelsize=6)
    axr.grid(alpha=0.15)
    ticks = np.linspace(0, n - 1, 6).astype(int)
    fmt = "%d %b %y" if a["interval"] in ("1d", "1wk") else "%d %b %H:%M"
    axr.set_xticks(ticks)
    axr.set_xticklabels([df.index[i].strftime(fmt) for i in ticks], fontsize=6)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_png(a, idx=True):
    try:
        return _chart_png_impl(a, idx)
    except Exception as e:
        if "chart" not in _WARNED:
            _WARNED.add("chart")
            print(f"[info] grafik dilewati ({type(e).__name__}: {str(e)[:80]}); PDF tetap dibuat tanpa grafik.")
        return None


def build_pdf(path, title, rows, details, meta, idx=True, charts=True):
    """rows: ringkasan semua saham terurut | details: [(sym, f, analyses)] | meta: dict info run."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                        TableStyle)
        from xml.sax.saxutils import escape
    except Exception as e:
        raise RuntimeError(f"reportlab gagal dimuat ({type(e).__name__}: {e}). Install: pip install reportlab "
                           f"(butuh Python 3.8+; disarankan 3.10+). Cek versi: python --version")

    def S(x):
        x = "-" if x is None else str(x)
        return escape(x.encode("cp1252", "replace").decode("cp1252"))

    base = getSampleStyleSheet()
    st = dict(
        h1=ParagraphStyle("h1", parent=base["Title"], fontSize=18, leading=22, alignment=0, spaceAfter=4),
        h2=ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#1f3a5f"),
                          spaceBefore=8, spaceAfter=3),
        body=ParagraphStyle("body", parent=base["Normal"], fontSize=8.5, leading=11.5),
        cell=ParagraphStyle("cell", parent=base["Normal"], fontSize=7.5, leading=9.3),
        small=ParagraphStyle("small", parent=base["Normal"], fontSize=7, leading=9, textColor=colors.HexColor("#555555")),
        white=ParagraphStyle("white", parent=base["Normal"], fontSize=10, leading=13, textColor=colors.white),
    )
    W = 523
    GRID = colors.HexColor("#cfd4da")
    LBL = colors.HexColor("#f1f3f5")

    def P(t, s="body"):
        return Paragraph(S(t), st[s])

    def kv(rows_, widths=(120, W - 120)):
        data = [[Paragraph(f"<b>{S(k)}</b>", st["cell"]), Paragraph(S(v), st["cell"])] for k, v in rows_]
        t = Table(data, colWidths=list(widths))
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, GRID), ("BACKGROUND", (0, 0), (0, -1), LBL),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return t

    def grid3(items):
        data = []
        for i in range(0, len(items), 3):
            chunk = items[i:i + 3] + [("", "")] * (3 - len(items[i:i + 3]))
            row = []
            for k, v in chunk:
                row += [Paragraph(f"<b>{S(k)}</b>", st["cell"]), Paragraph(S(v), st["cell"])]
            data.append(row)
        t = Table(data, colWidths=[64, 108, 64, 108, 64, 108][:6])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (0, -1), LBL), ("BACKGROUND", (2, 0), (2, -1), LBL),
                               ("BACKGROUND", (4, 0), (4, -1), LBL), ("TOPPADDING", (0, 0), (-1, -1), 2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return t

    def tone(score):
        return colors.HexColor("#1b7f4d") if score >= 65 else colors.HexColor("#b3392f") if score <= 35 \
            else colors.HexColor("#5b6670")

    def footer(c, doc):
        c.saveState()
        c.setFont("Helvetica", 6.5)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(36, 20, "Alat bantu analisis - BUKAN rekomendasi investasi. Data dapat tertunda/tidak akurat; "
                             "verifikasi ke sumber resmi & kelola risiko.")
        c.drawRightString(595 - 36, 20, f"Hal. {doc.page}")
        c.restoreState()

    story = [P(title, "h1"),
             kv([("Waktu laporan", meta.get("time")), ("Universe", meta.get("universe")),
                 ("Metode / ranking", meta.get("method")), ("Jumlah dianalisis", meta.get("count")),
                 ("Sumber data", meta.get("source")), ("Catatan", meta.get("note"))]),
             Spacer(1, 6),
             P("Cara membaca: Skor total = bobot skor teknikal (bias -100..+100 dinormalkan 0-100) dan skor fundamental "
               "(0-100) sesuai metode. BUY (LONG) hanya muncul bila setup teknikal cukup kuat dan tidak melawan tren "
               "timeframe besar. Stop loss = batas struktur/ATR; cut loss = batas darurat (keluar tanpa menunggu). "
               "Saham BEI tidak praktis untuk short-sell ritel, jadi sinyal bearish = menunggu/menghindari.", "small")]

    # ---- ringkasan
    story.append(P("Ringkasan Peringkat", "h2"))
    cnt = {}
    for r in rows:
        cnt[r["rating"]] = cnt.get(r["rating"], 0) + 1
    story.append(P("Distribusi sinyal: " + " | ".join(f"{k}: {v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])), "small"))
    story.append(Spacer(1, 3))
    head = ["#", "Kode", "Sektor", "Harga", "Trend", "RSI", "Tekn", "Fund", "Skor", "Sinyal", "Entry", "SL", "TP1"]
    data = [head]
    for i, r in enumerate(rows, 1):
        data.append([str(i), r["sym"].replace(".JK", ""), (r["sector"] or "-")[:13], px(r["price"], idx),
                     r["trend"][:9], f"{r['rsi']:.0f}", f"{r['bias']:+d}",
                     f"{r['fund']:.0f}" if r["fund"] is not None else "-", f"{r['score']:.0f}", r["rating"][:17],
                     px(r["entry"], idx) if r["entry"] else "-", px(r["sl"], idx) if r["sl"] else "-",
                     px(r["tp1"], idx) if r["tp1"] else "-"])
    cw = [20, 32, 58, 36, 54, 22, 26, 24, 26, 80, 40, 38, 38]
    t = Table(data, colWidths=cw, repeatRows=1)
    sty = [("FONTSIZE", (0, 0), (-1, -1), 6.6), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
           ("GRID", (0, 0), (-1, -1), 0.25, GRID), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5), ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
           ("ALIGN", (4, 0), (4, -1), "LEFT"), ("ALIGN", (9, 0), (9, -1), "LEFT")]
    for i, r in enumerate(rows, 1):
        if r["score"] >= 65:
            sty.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e6f4ec")))
        elif r["score"] <= 35:
            sty.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fbe9e7")))
    t.setStyle(TableStyle(sty))
    story.append(t)

    # ---- detail per saham
    for sym, f, analyses in details:
        code = sym.replace(".JK", "")
        story.append(PageBreak())
        a0 = analyses[0]
        ov = calc_overall(f, a0)
        name = (f or {}).get("name") or code
        banner = Table([[Paragraph(f"<b>{S(code)}</b> - {S(name)}", st["white"]),
                         Paragraph(f"<b>Skor {ov:.0f}/100</b> | {S(a0['rating'])}", st["white"])]],
                       colWidths=[330, 193])
        banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), tone(ov)), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("ALIGN", (1, 0), (1, 0), "RIGHT")]))
        story += [banner, Spacer(1, 4)]
        story.append(P("Ringkasan Analisis", "h2"))
        for para in narrative(sym, f, analyses, idx):
            story += [P(para), Spacer(1, 2)]

        if f:
            story.append(P("Profil Perusahaan", "h2"))
            story.append(P(f"Sektor / Industri: {f.get('sector') or '-'} / {f.get('industry') or '-'} | "
                           f"Website: {f.get('website') or '-'} | Karyawan: {f.get('employees') or '-'}", "small"))
            summ = f.get("summary_id") or f.get("summary")
            if summ:
                story.append(P(summ if len(summ) < 1100 else summ[:1100] + " ..."))
            if f.get("news"):
                story.append(P("Berita / aksi korporasi terbaru:", "small"))
                for n in f["news"][:5]:
                    story.append(P("- " + n, "small"))
            story.append(P("Fundamental", "h2"))
            items = [("Market cap", big(f["market_cap"], idx)), ("PER", num(f["per"])), ("Fwd PER", num(f["fwd_per"])),
                     ("PBV", num(f["pbv"])), ("EV/EBITDA", num(f["ev_ebitda"])), ("EPS", num(f["eps"])),
                     ("BVPS", num(f["bvps"])), ("Div. yield", pct(f["div_yield"])), ("Payout", pct(f["payout"])),
                     ("ROE", pct(f["roe"])), ("ROA", pct(f["roa"])), ("Beta", num(f["beta"])),
                     ("Gross margin", pct(f["gpm"])), ("Op. margin", pct(f["opm"])), ("Net margin", pct(f["npm"])),
                     ("Growth rev.", pct(f["rev_growth"])), ("Growth laba", pct(f["earn_growth"])),
                     ("DER", f"{num(f['der'])}%"), ("Current ratio", num(f["current_ratio"])),
                     ("Total utang", big(f["total_debt"], idx)), ("Kas", big(f["total_cash"], idx)),
                     ("Operating CF", big(f["ocf"], idx)), ("Free CF", big(f["fcf"], idx)),
                     ("Insider", pct(f["held_insiders"])), ("Institusi", pct(f["held_inst"])),
                     ("52w High", px(f["hi52"], idx)), ("52w Low", px(f["lo52"], idx)), ("", "")]
            story.append(grid3(items))
            if f.get("history"):
                story.append(Spacer(1, 3))
                story.append(P("Laporan tahunan (pendapatan / laba bersih): " + " | ".join(
                    f"{h['year']}: {big(h['revenue'], idx)} / {big(h['net_income'], idx)}" for h in f["history"]), "small"))
            story.append(Spacer(1, 3))
            story.append(P(f"Skor fundamental: {f['score'] if f['score'] is not None else 'n/a'} / 100", "body"))
            story.append(P(" | ".join(f["score_notes"]), "small"))

        for a in analyses:
            p = a["plan"]
            head_ = P(f"{a['label']}  [TF {a['interval']} | TF besar {a['htf']}]", "h2")
            png = _chart_png(a, idx) if charts else None
            if png:
                story.append(KeepTogether([head_, Image(png, width=W, height=W * 4.3 / 7.4)]))   # judul tidak terpisah dari grafik
            else:
                story.append(head_)
            t_ = a["trend"]
            age = f" (usia data ~{a['bar_age_min']:.0f} menit)" if a.get("bar_age_min") is not None else ""
            o = a["ohlc"]
            fb = a["fib"]
            rows_t = [
                ("Bar terakhir", f"{a['last_bar']}{age}"),
                ("Harga / OHLC", f"{px(a['price'], idx)} | O {px(o['open'], idx)} H {px(o['high'], idx)} "
                                 f"L {px(o['low'], idx)} | Vol {o['volume']:,.0f} | ATR {px(a['atr'], idx)} | "
                                 f"Nilai transaksi rata-rata/bar {big(a['avg_value'], idx)}"),
                ("Trend", f"{t_['label']} (kekuatan {t_['strength']:.0%}) | struktur {t_['structure']}"
                          + (f" | TF besar: {a['htf_trend']['label']}" if a.get("htf_trend") else "")),
                ("Moving average", " | ".join(f"{k.upper()} {px(v, idx)}" for k, v in a["ma"].items())),
                ("RSI / Stoch / MACD", f"RSI {a['rsi']:.1f} | Stoch %K/%D {a['stoch'][0]:.0f}/{a['stoch'][1]:.0f} | "
                                       f"MACD {a['macd']['macd']:.2f} sig {a['macd']['signal']:.2f} hist {a['macd']['hist']:.2f}"),
                ("Bollinger / Volume", f"{px(a['bb'][0], idx)} - {px(a['bb'][1], idx)} | volume {a['vol_ratio']:.1f}x rata-rata"),
                ("Support", ", ".join(f"{px(v, idx)} ({n}x)" for v, n in a["support"]) or "-"),
                ("Resistance", ", ".join(f"{px(v, idx)} ({n}x)" for v, n in a["resistance"]) or "-"),
            ]
            if a.get("pivot_points"):
                rows_t.append(("Pivot hari lalu", " | ".join(f"{k} {px(v, idx)}" for k, v in a["pivot_points"].items())))
            rows_t += [
                ("Fibonacci", f"{fb['direction']}: low {px(fb['low'], idx)} -> high {px(fb['high'], idx)} | {fb['zone']}"),
                ("Fib retracement", " | ".join(f"{k:.1%}={px(v, idx)}" for k, v in fb["retracement"].items())),
                ("Fib extension", " | ".join(f"{k}={px(v, idx)}" for k, v in fb["extension"].items())),
                ("Sinyal", " ; ".join(a["notes"])),
                ("Skor bias teknikal", f"{a['bias']:+d} (-100..+100) => {a['rating']} | skor total {calc_overall(f, a):.0f}/100"),
            ]
            story.append(kv(rows_t))
            story.append(Spacer(1, 4))
            if p["side"] == "NO TRADE":
                plan_rows = [("Trade plan", f"NO TRADE - {p['reason']}")] + [("Pantau", w) for w in p["watch"]]
            else:
                plan_rows = [("Keputusan", f"{p['side']} | {p['status']}"),
                             ("Entry", f"{px(p['entry'], idx)} (alternatif buy-on-pullback: {px(p['entry_pullback'], idx)})"),
                             ("Stop loss", f"{px(p['stop_loss'], idx)} (risiko {p['risk_pct']:.2%} / {px(p['risk_per_share'], idx)} per saham)"),
                             ("Cut loss", f"{px(p['cut_loss'], idx)} (batas darurat: keluar tanpa menunggu bila tembus/gap down)")]
                for i, (tp, rr) in enumerate(zip(p["targets"], p["rr"]), 1):
                    plan_rows.append((f"TP{i}", f"{px(tp, idx)} (RR 1:{rr:.1f})"))
                if p.get("fib_targets"):
                    plan_rows.append(("Target fib ext.", ", ".join(px(v, idx) for v in p["fib_targets"])))
                if p.get("position"):
                    ps = p["position"]
                    plan_rows.append(("Ukuran posisi", f"{ps['lots']} lot ({ps['shares']:,} saham) | nilai {big(ps['value'], idx)} | "
                                                       f"rugi maks jika SL {big(ps['max_loss'], idx)}"))
                plan_rows.append(("Trailing", p["trail"]))
            pt = kv(plan_rows)
            pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e3ecf7"))]))
            story.append(pt)

    story += [Spacer(1, 10), P("Disclaimer: laporan ini dihasilkan otomatis dari data publik (TradingView/Yahoo Finance) dan "
                               "rumus indikator. Bukan nasihat/rekomendasi investasi. Semua keputusan dan risiko ada pada "
                               "pengguna.", "small")]
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=34, bottomMargin=34,
                            title=title, author="screener_saham.py")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def _pdf_path(args):
    p = args.pdf
    if p in (None, ""):
        return None
    if p == "auto":
        p = f"laporan_saham_{datetime.now():%Y%m%d_%H%M}.pdf"
    return p if p.lower().endswith(".pdf") else p + ".pdf"


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


def _clean(a):
    return {k: v for k, v in a.items() if k != "chart_df"}


def run_single(args, idx):
    sym = norm_ticker(args.tickers[0], args.market)
    methods = list(METHODS) if args.method == "all" else [args.method]
    f, analyses = analyze_symbol(sym, methods, idx, not args.no_fund, args.translate, args.capital, args.risk_pct)
    print(f"\n#### {sym}  |  diproses {pd.Timestamp.now():%Y-%m-%d %H:%M:%S} ####")
    print_full(sym, f, analyses, idx)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(dict(symbol=sym, fundamental=f, technical=[_clean(a) for a in analyses]), fh, indent=2,
                      default=str, ensure_ascii=False)
        print(f"\nHasil disimpan ke {args.json}")
    path = _pdf_path(args)
    if path:
        meta = dict(time=f"{datetime.now():%Y-%m-%d %H:%M:%S} WIB", universe=sym, method=", ".join(methods),
                    count=1, source=_SOURCE + " (harga) + Yahoo (fundamental)", note="Laporan satu saham")
        build_pdf(path, f"Laporan Analisis Saham - {sym.replace('.JK', '')}", [_make_row(sym, f, analyses[0])],
                  [(sym, f, analyses)], meta, idx, charts=not args.no_chart)
        print(f"\nPDF disimpan: {os.path.abspath(path)}")


def run_scan(args, idx):
    tickers = list(args.scan or []) + (list(args.tickers) if len(args.tickers) > 1 else [])
    if args.scan_file:
        with open(args.scan_file) as fh:
            tickers += [x.strip() for x in fh.read().replace(",", "\n").split() if x.strip()]
    uni_note = "Daftar manual"
    if not tickers:
        if args.universe == "all":
            tickers, uni_note = load_all_idx(args)
        else:
            tickers, uni_note = list(LQ_UNIVERSE), "LQ45 (perkiraan; verifikasi daftar resmi di idx.co.id)"
    tickers = [norm_ticker(t, args.market) for t in dict.fromkeys(tickers)]
    method = "swing" if args.method == "all" else args.method
    cfg = METHODS[method]
    two_pass = args.universe == "all" or len(tickers) > 60
    n_head = max(args.detail_top, args.top)
    print(uni_note)

    def work(sym):
        try:
            f, an = analyze_symbol(sym, [method], idx, (not two_pass) and not args.no_fund and cfg["w_fund"] > 0,
                                   False, args.capital, args.risk_pct)
            return _make_row(sym, f, an[0])
        except Exception as e:
            return dict(sym=sym, error=f"{type(e).__name__}: {str(e)[:110]}".replace(sym, "<kode>").replace(sym.replace(".JK", ""), "<kode>"),
                        where=_where(e, 2))

    print(f"Scanning {len(tickers)} saham | metode {method}"
          f"{' | tahap 1: teknikal semua, tahap 2: fundamental utk top ' + str(n_head) if two_pass else ''} ...")
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in tickers]
        for fu in as_completed(futs):
            rows.append(fu.result())
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f"\r  progres {done}/{len(futs)}", end="", flush=True)
    print()
    valid = sorted([r for r in rows if "error" not in r], key=lambda r: -r["score"])

    if two_pass and not args.no_fund and cfg["w_fund"] > 0 and valid:
        head = valid[:n_head]

        def enrich(r):
            try:
                f = get_fundamentals(r["sym"], False)
                a = r["_a"]
                r.update(_make_row(r["sym"], f, a))
            except Exception:
                pass
            return r

        print(f"Tahap 2: mengambil fundamental {len(head)} saham teratas ...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            head = list(ex.map(enrich, head))
        valid = sorted(head, key=lambda r: -r["score"]) + valid[n_head:]

    ranked = [r for r in valid if r["score"] >= args.min_score]
    ok = ranked[: args.top]
    print(f"\n{'KODE':8s}{'SKOR':>6s}{'TEKN':>6s}{'FUND':>6s} {'TREND':10s}{'RSI':>5s} {'SINYAL':18s}"
          f"{'ENTRY':>9s}{'SL':>9s}{'TP1':>9s}  SEKTOR")
    for r in ok:
        fund_s = f"{r['fund']:.0f}" if r["fund"] is not None else "-"
        print(f"{r['sym'].replace('.JK', ''):8s}{r['score']:6.0f}{r['bias']:6d}"
              f"{fund_s:>6s} {r['trend']:10s}{r['rsi']:5.0f} "
              f"{r['rating']:18s}{px(r['entry'], idx):>9s}{px(r['sl'], idx):>9s}{px(r['tp1'], idx):>9s}  {r['sector'] or '-'}")
    errs = [r for r in rows if "error" in r]
    if errs:
        from collections import Counter
        print(f"\n{len(errs)} dari {len(tickers)} saham gagal diambil. Penyebab terbanyak:")
        for msg, n in Counter(e["error"] for e in errs).most_common(3):
            ex = [e["sym"].replace(".JK", "") for e in errs if e["error"] == msg][:4]
            wh = next(e.get("where") for e in errs if e["error"] == msg)
            print(f"  - {n}x {msg}\n      contoh: {', '.join(ex)} | lokasi: {wh}")
        if len(errs) > 0.3 * len(tickers):
            print("  Banyak yang gagal: biasanya rate-limit/blokir sumber data. Coba: pasang tvdatafeed (sumber TradingView), "
                  "kecilkan --workers 2 --fetch-limit 2, atau jalankan dari jaringan rumah (IP cloud sering diblokir Yahoo).")
    if not ranked:
        print("\nTidak ada saham yang berhasil dianalisis, PDF tidak dibuat.")
    print("\nDetail lengkap satu saham: python screener_saham.py KODE --method all --pdf")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in ok], fh, indent=2, default=str)

    path = _pdf_path(args)
    if path and ranked:
        sel = ranked[: args.detail_top]
        print(f"Menyusun PDF ({len(ranked)} saham di ringkasan, {len(sel)} detail lengkap) ...")
        if args.method == "all":
            def deep(r):
                try:
                    f, an = analyze_symbol(r["sym"], list(METHODS), idx, not args.no_fund, False, args.capital, args.risk_pct)
                except Exception as e:      # data intraday gagal -> pakai hasil swing yang sudah ada
                    print(f"  [warn] detail {r['sym']} hanya swing ({type(e).__name__}: {str(e)[:60]})")
                    f, an = r["_f"], [r["_a"]]
                return r["sym"], f, an
            with ThreadPoolExecutor(max_workers=max(1, args.workers // 2)) as ex:
                details = list(ex.map(deep, sel))
        else:
            details = [(r["sym"], r["_f"], [r["_a"]]) for r in sel]
        meta = dict(time=f"{datetime.now():%Y-%m-%d %H:%M:%S} WIB", universe=uni_note,
                    method=f"ranking pakai {method}" + (" | detail: swing + daytrade + scalping" if args.method == "all" else ""),
                    count=f"{len(valid)} berhasil dari {len(tickers)} (gagal: {len(errs)})",
                    source=f"{_SOURCE} (harga) + Yahoo (fundamental, top {n_head})" if two_pass else f"{_SOURCE} + Yahoo",
                    note=("Peringkat awal dari teknikal; fundamental ditambahkan untuk saham teratas." if two_pass else "-"))
        build_pdf(path, "Laporan Screening Saham", ranked, details, meta, idx, charts=not args.no_chart)
        print(f"PDF disimpan: {os.path.abspath(path)}")


def _where(e, n=3):
    try:
        tb = traceback.extract_tb(e.__traceback__)[-n:]
        return " <- ".join(f"{os.path.basename(t.filename)}:{t.lineno} {t.name}" for t in reversed(tb))
    except Exception:
        return "-"


def _report_error(e, debug=False):
    print(f"Error: {type(e).__name__}: {e}")
    print(f"  lokasi: {_where(e)}")
    if debug:
        traceback.print_exc()
    else:
        print("  (tambahkan --debug untuk traceback lengkap)")


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
    ap.add_argument("--universe", choices=["lq45", "all"], help="lq45 = LQ45 (perkiraan); all = SELURUH saham IDX (scanner TradingView)")
    ap.add_argument("--min-value", type=float, default=1e9, help="filter likuiditas universe 'all': nilai transaksi/hari minimal (Rp), default 1 miliar")
    ap.add_argument("--min-price", type=float, default=50, help="filter harga minimal universe 'all' (default 50)")
    ap.add_argument("--max-stocks", type=int, default=150, help="maks saham dianalisis dari universe 'all' (urut likuiditas; 0 = semua)")
    ap.add_argument("--pdf", nargs="?", const="auto", help="simpan laporan PDF (tanpa nama file = laporan_saham_TGL_JAM.pdf)")
    ap.add_argument("--detail-top", type=int, default=20, help="jumlah saham teratas yang dibuatkan halaman detail lengkap di PDF")
    ap.add_argument("--no-chart", action="store_true", help="PDF tanpa grafik")
    ap.add_argument("--debug", action="store_true", help="tampilkan traceback lengkap bila error")
    ap.add_argument("--fetch-limit", type=int, default=3, help="maks request harga bersamaan (default 3; kecilkan jika kena rate-limit)")
    ap.add_argument("--source", choices=["auto", "tv", "yahoo"], default="auto",
                    help="sumber harga: auto = TradingView lalu Yahoo (default)")
    args = ap.parse_args()
    global _SOURCE, _FETCH_SEM
    _SOURCE = args.source
    _FETCH_SEM = threading.Semaphore(max(1, args.fetch_limit))
    if sys.version_info < (3, 9):
        print(f"[peringatan] Python {sys.version_info[0]}.{sys.version_info[1]} terlalu lama; library terbaru sering error "
              f"(contoh: 'type' object is not subscriptable). Disarankan Python 3.10+ (cek: python --version).")

    idx = args.market == "idx"
    if not args.tickers and args.scan is None and not args.scan_file and not args.universe:
        ap.error("isi kode saham, atau gunakan --universe all / --scan / --scan-file")
    scan_mode = args.scan is not None or args.scan_file or args.universe or len(args.tickers) > 1

    def once():
        run_scan(args, idx) if scan_mode else run_single(args, idx)

    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                try:
                    once()
                except Exception as e:
                    _report_error(e, args.debug)
                print(f"\n[watch] refresh tiap {args.watch} detik. Ctrl+C untuk berhenti.")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nBerhenti.")
    else:
        try:
            once()
        except Exception as e:
            _report_error(e, args.debug)
            sys.exit(1)


if __name__ == "__main__":
    main()

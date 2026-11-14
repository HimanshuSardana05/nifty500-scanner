"""
Nifty 500 daily scanner.

Every run:
  1. Loads the Nifty 500 constituent list (NSE, with a cached copy as fallback).
  2. Downloads ~14 months of daily candles from Yahoo Finance (free, via yfinance).
  3. Checks the latest completed daily candle for candlestick patterns and
     technical setups.
  4. Draws a candlestick chart for the strongest signals.
  5. Publishes the report to reports/ (PUBLISH_BASE_URL set) for the Claude
     scheduled task to email, or emails it directly via Gmail SMTP.

Environment variables:
  PUBLISH_BASE_URL    raw.githubusercontent.com base of this repo (publish mode)
  GMAIL_USER / GMAIL_APP_PASSWORD / MAIL_TO   only for direct SMTP mode
Optional:
  MAX_CHARTS_PER_SIDE charts for bullish and for bearish lists (default 15)
  DRY_RUN=1           write email to out/email.html instead of sending
  FORCE=1             send even if this candle date was already reported
"""
from __future__ import annotations

import datetime as dt
import io
import os
import smtplib
import sys
import traceback
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplfinance as mpf  # noqa: E402

from levels import LevelIndicator  # noqa: E402

ROOT = Path(__file__).resolve().parent
UNIVERSE_CACHE = ROOT / "data" / "nifty500.csv"
STATE_FILE = ROOT / "data" / "last_reported.txt"
OUT_DIR = ROOT / "out"
IST = ZoneInfo("Asia/Kolkata")
NSE_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
MAX_CHARTS = int(os.getenv("MAX_CHARTS_PER_SIDE", "15"))

# Support/resistance zones + swing levels, tuned for daily candles (the class defaults
# are calibrated for 5-minute bars). One daily bar = one "session" here.
LEVELS = LevelIndicator()
LEVELS.ZONE_SESSIONS = 150           # ~7 months of daily bars searched for zone pivots
LEVELS.ZONE_PIVOT_BARS = 10          # ~2 trading weeks either side to count as a pivot
LEVELS.ZONE_SUPPRESSION_BARS = 10
LEVELS.ZONE_IN_PLAY_DAILY_ATR = 1.5  # a bit wider than default so charts aren't often empty


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_universe() -> pd.DataFrame:
    """Return DataFrame[Symbol, Company Name, Industry]. Refreshes cache from NSE."""
    import requests

    try:
        r = requests.get(
            NSE_LIST_URL,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            timeout=20,
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if len(df) >= 450 and "Symbol" in df.columns:
            UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(UNIVERSE_CACHE, index=False)
            print(f"Universe: {len(df)} symbols from NSE")
            return df
    except Exception as e:  # NSE sometimes blocks cloud IPs
        print(f"NSE list fetch failed ({e}); using cached list")
    if UNIVERSE_CACHE.exists():
        df = pd.read_csv(UNIVERSE_CACHE)
        print(f"Universe: {len(df)} symbols from cache")
        return df
    raise RuntimeError("Could not load Nifty 500 list from NSE and no cached copy exists.")


def download_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    tickers = [f"{s}.NS" for s in symbols]
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 100):  # batches keep Yahoo happy
        batch = tickers[i : i + 100]
        raw = yf.download(
            batch, period="14mo", interval="1d", group_by="ticker",
            auto_adjust=False, threads=True, progress=False,
        )
        for t in batch:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) >= 60:
                    out[t[:-3]] = d
            except KeyError:
                pass
    print(f"Prices: {len(out)} / {len(symbols)} symbols downloaded")
    return out


def drop_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    """If run during market hours, today's bar is still forming -> drop it."""
    now = dt.datetime.now(IST)
    last = pd.Timestamp(df.index[-1]).date()
    if last == now.date() and now.time() < dt.time(15, 45):
        return df.iloc[:-1]
    return df


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# --------------------------------------------------------------------------- #
# Support/resistance zones + swing levels
# --------------------------------------------------------------------------- #
def to_level_candles(d: pd.DataFrame) -> pd.DataFrame:
    """Adapt this scanner's daily OHLCV frame (DatetimeIndex) to what LevelIndicator
    expects: one row per session, since a daily bar IS its own session here."""
    idx = d.index
    return pd.DataFrame({
        "time": idx,
        "session": idx.date,
        "high": d["High"].to_numpy(dtype=float),
        "low": d["Low"].to_numpy(dtype=float),
        "close": d["Close"].to_numpy(dtype=float),
        "volume": d["Volume"].to_numpy(dtype=float),
    })


def stock_levels(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Support/resistance zones, swing levels, and which of them today's candle
    (its high/low range) touched - a 'test', regardless of where it closed."""
    candles = to_level_candles(d)
    zones_df = LEVELS.zones(candles)
    swings_df = LEVELS.swingLevels(candles)
    today_high, today_low = float(d["High"].iloc[-1]), float(d["Low"].iloc[-1])
    tested: list[dict] = []
    for _, z in zones_df.iterrows():
        if today_low <= z.high and today_high >= z.low:
            tested.append({
                "kind": "zone", "side": z.side, "score": float(z.score),
                "low": float(z.low), "high": float(z.high), "touches": int(z.touches),
            })
    for _, s in swings_df.iterrows():
        if today_low <= s.price <= today_high:
            tested.append({
                "kind": "swing", "side": s.side, "price": float(s.price),
                "confirmed": bool(s.confirmed), "broken": bool(s.broken),
            })
    return zones_df, swings_df, tested


# --------------------------------------------------------------------------- #
# Signal detection. Each returns list of (name, side, weight)
#   side: "bull" | "bear" | "neutral"
# --------------------------------------------------------------------------- #
def candlestick_signals(d: pd.DataFrame) -> list[tuple[str, str, int]]:
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    sig: list[tuple[str, str, int]] = []
    if len(c) < 25:
        return sig

    body = np.abs(c - o)
    rng = h - l
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    avg_body = pd.Series(body).rolling(10).mean().to_numpy()
    sma20 = pd.Series(c).rolling(20).mean().to_numpy()

    i, p, q = -1, -2, -3  # today, yesterday, day before
    bull = lambda k: c[k] > o[k]  # noqa: E731
    bear = lambda k: c[k] < o[k]  # noqa: E731
    # trend context from the close *before* the pattern, vs its 20DMA and 5 bars ago
    downtrend = c[p] < sma20[p] and c[p] < c[-7]
    uptrend = c[p] > sma20[p] and c[p] > c[-7]
    big = lambda k: body[k] > avg_body[k - 1]  # noqa: E731
    small = lambda k: body[k] < 0.5 * avg_body[k - 1]  # noqa: E731

    if rng[i] <= 0:
        return sig

    # --- single-candle ---
    is_doji = body[i] <= 0.1 * rng[i]
    hammer_shape = lower[i] >= 2 * body[i] and upper[i] <= 0.3 * body[i] + 0.1 * rng[i] and not is_doji
    inv_shape = upper[i] >= 2 * body[i] and lower[i] <= 0.3 * body[i] + 0.1 * rng[i] and not is_doji

    if hammer_shape and downtrend:
        sig.append(("Hammer", "bull", 2))
    if hammer_shape and uptrend:
        sig.append(("Hanging Man", "bear", 1))
    if inv_shape and downtrend:
        sig.append(("Inverted Hammer", "bull", 1))
    if inv_shape and uptrend:
        sig.append(("Shooting Star", "bear", 2))
    if is_doji:
        if lower[i] >= 0.7 * rng[i] and downtrend:
            sig.append(("Dragonfly Doji", "bull", 1))
        elif upper[i] >= 0.7 * rng[i] and uptrend:
            sig.append(("Gravestone Doji", "bear", 1))
        elif downtrend or uptrend:
            sig.append(("Doji (indecision)", "neutral", 0))
    if body[i] >= 0.9 * rng[i] and big(i) and body[i] > 1.5 * avg_body[p]:
        sig.append(("Bullish Marubozu", "bull", 2) if bull(i) else ("Bearish Marubozu", "bear", 2))

    # --- two-candle ---
    if bear(p) and bull(i) and o[i] <= c[p] and c[i] >= o[p] and body[i] > body[p] and downtrend:
        sig.append(("Bullish Engulfing", "bull", 3))
    if bull(p) and bear(i) and o[i] >= c[p] and c[i] <= o[p] and body[i] > body[p] and uptrend:
        sig.append(("Bearish Engulfing", "bear", 3))
    if bear(p) and big(p) and bull(i) and o[i] > c[p] and c[i] < o[p] and small(i) and downtrend:
        sig.append(("Bullish Harami", "bull", 1))
    if bull(p) and big(p) and bear(i) and o[i] < c[p] and c[i] > o[p] and small(i) and uptrend:
        sig.append(("Bearish Harami", "bear", 1))
    mid_p = (o[p] + c[p]) / 2
    if bear(p) and big(p) and bull(i) and o[i] < l[p] and mid_p < c[i] < o[p] and downtrend:
        sig.append(("Piercing Line", "bull", 2))
    if bull(p) and big(p) and bear(i) and o[i] > h[p] and o[p] < c[i] < mid_p and uptrend:
        sig.append(("Dark Cloud Cover", "bear", 2))

    # --- three-candle ---
    down_before = c[q] < sma20[q]
    up_before = c[q] > sma20[q]
    if (bear(q) and big(q) and small(p) and max(o[p], c[p]) < c[q] + 0.25 * body[q]
            and bull(i) and c[i] > (o[q] + c[q]) / 2 and down_before):
        sig.append(("Morning Star", "bull", 3))
    if (bull(q) and big(q) and small(p) and min(o[p], c[p]) > c[q] - 0.25 * body[q]
            and bear(i) and c[i] < (o[q] + c[q]) / 2 and up_before):
        sig.append(("Evening Star", "bear", 3))
    if (all(bull(k) for k in (q, p, i)) and c[q] < c[p] < c[i]
            and o[q] < o[p] <= c[q] and o[p] < o[i] <= c[p]
            and all(upper[k] < 0.3 * body[k] for k in (q, p, i)) and all(big(k) for k in (q, p, i))):
        sig.append(("Three White Soldiers", "bull", 3))
    if (all(bear(k) for k in (q, p, i)) and c[q] > c[p] > c[i]
            and o[q] > o[p] >= c[q] and o[p] > o[i] >= c[p]
            and all(lower[k] < 0.3 * body[k] for k in (q, p, i)) and all(big(k) for k in (q, p, i))):
        sig.append(("Three Black Crows", "bear", 3))
    return sig


def technical_signals(d: pd.DataFrame) -> list[tuple[str, str, int]]:
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    sig: list[tuple[str, str, int]] = []
    n = len(c)

    # 52-week breakout / breakdown (close vs prior 250-session high/low)
    if n >= 200:
        look = min(250, n - 1)
        prior_hi = h.iloc[-look - 1 : -1].max()
        prior_lo = l.iloc[-look - 1 : -1].min()
        if c.iloc[-1] > prior_hi:
            sig.append(("52-week high breakout", "bull", 3))
        elif c.iloc[-1] < prior_lo:
            sig.append(("52-week low breakdown", "bear", 3))

    # Moving-average crosses
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    if n >= 201:
        if s50.iloc[-2] <= s200.iloc[-2] and s50.iloc[-1] > s200.iloc[-1]:
            sig.append(("Golden Cross (50/200 DMA)", "bull", 3))
        if s50.iloc[-2] >= s200.iloc[-2] and s50.iloc[-1] < s200.iloc[-1]:
            sig.append(("Death Cross (50/200 DMA)", "bear", 3))
        if c.iloc[-2] <= s200.iloc[-2] and c.iloc[-1] > s200.iloc[-1]:
            sig.append(("Close above 200 DMA", "bull", 2))
        if c.iloc[-2] >= s200.iloc[-2] and c.iloc[-1] < s200.iloc[-1]:
            sig.append(("Close below 200 DMA", "bear", 2))

    # RSI(14)
    r = rsi(c)
    if r.iloc[-2] < 30 <= r.iloc[-1]:
        sig.append(("RSI up through 30", "bull", 2))
    if r.iloc[-2] > 70 >= r.iloc[-1]:
        sig.append(("RSI down through 70", "bear", 2))

    # MACD(12,26,9) signal-line cross
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sl = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-2] <= sl.iloc[-2] and macd.iloc[-1] > sl.iloc[-1]:
        sig.append(("MACD bullish cross", "bull", 1))
    if macd.iloc[-2] >= sl.iloc[-2] and macd.iloc[-1] < sl.iloc[-1]:
        sig.append(("MACD bearish cross", "bear", 1))

    # Volume spike with a real move
    vavg = v.iloc[-21:-1].mean()
    chg = c.iloc[-1] / c.iloc[-2] - 1
    if vavg > 0 and v.iloc[-1] >= 2 * vavg and abs(chg) >= 0.02:
        sig.append((f"Volume spike {v.iloc[-1] / vavg:.1f}x", "bull" if chg > 0 else "bear", 2))

    # Inside bar / NR7 (volatility contraction - direction neutral)
    if h.iloc[-1] < h.iloc[-2] and l.iloc[-1] > l.iloc[-2]:
        sig.append(("Inside Bar", "neutral", 0))
    rng = h - l
    if n >= 7 and rng.iloc[-1] == rng.iloc[-7:].min():
        sig.append(("NR7 (narrowest range in 7)", "neutral", 0))
    return sig


ZONES: dict[str, pd.DataFrame] = {}   # symbol -> in-play support/resistance zones, set in scan()
SWINGS: dict[str, pd.DataFrame] = {}  # symbol -> nearby swing levels, set in scan()


def scan(prices: dict[str, pd.DataFrame], names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for sym, d in prices.items():
        try:
            sigs = candlestick_signals(d) + technical_signals(d)
        except Exception as e:
            print(f"  {sym}: scan error {e}")
            continue
        try:
            zones_df, swings_df, tested = stock_levels(d)
            ZONES[sym], SWINGS[sym] = zones_df, swings_df
        except Exception as e:
            print(f"  {sym}: levels error {e}")
            zones_df, swings_df, tested = pd.DataFrame(), pd.DataFrame(), []
        if not sigs and not tested:
            continue
        bull = sum(w for _, s, w in sigs if s == "bull")
        bear = sum(w for _, s, w in sigs if s == "bear")
        c = d["Close"]
        vavg = d["Volume"].iloc[-21:-1].mean()
        rows.append({
            "Symbol": sym,
            "Company": names.get(sym, sym),
            "Close": round(float(c.iloc[-1]), 2),
            "High": round(float(d["High"].iloc[-1]), 2),
            "Low": round(float(d["Low"].iloc[-1]), 2),
            "Chg%": round(float((c.iloc[-1] / c.iloc[-2] - 1) * 100), 2),
            "VolX": round(float(d["Volume"].iloc[-1] / vavg), 1) if vavg else np.nan,
            "RSI": round(float(rsi(c).iloc[-1]), 0),
            "Bull": bull, "Bear": bear,
            "Side": "bull" if bull > bear else "bear" if bear > bull else "neutral",
            "Score": max(bull, bear),
            "Signals": sigs,
            "LevelTests": tested,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
STYLE = mpf.make_mpf_style(
    base_mpf_style="yahoo", gridstyle=":", facecolor="white",
    rc={"font.size": 8, "axes.titlesize": 10},
)


def make_chart(sym: str, d: pd.DataFrame, title: str,
                zones_df: pd.DataFrame | None = None, swings_df: pd.DataFrame | None = None) -> bytes:
    full = d.copy()
    for n in (20, 50, 200):
        full[f"SMA{n}"] = full["Close"].rolling(n).mean()
    plot = full.iloc[-120:]  # ~6 months
    aps = [
        mpf.make_addplot(plot["SMA20"], color="#1f77b4", width=0.8),
        mpf.make_addplot(plot["SMA50"], color="#ff7f0e", width=0.8),
        mpf.make_addplot(plot["SMA200"], color="#7f7f7f", width=0.8),
    ]
    buf = io.BytesIO()
    fig, axes = mpf.plot(
        plot, type="candle", style=STYLE, volume=True, addplot=aps,
        figsize=(9, 5), returnfig=True,
        vlines=dict(vlines=[plot.index[-1]], colors="#bbbbbb", linewidths=6, alpha=0.25),
    )
    ax = axes[0]
    # mplfinance's "yahoo" style puts the price axis + its tick labels on the RIGHT, so
    # put our own zone/swing price labels on the LEFT edge (x in axes-fraction, y in data
    # coords) instead of colliding with them.
    label = ax.get_yaxis_transform()
    extra_y: list[float] = []
    if zones_df is not None and not zones_df.empty:
        for _, z in zones_df.iterrows():
            ax.axhspan(z.low, z.high, color="#1a73e8", alpha=0.08 + 0.18 * float(z.score), zorder=0)
            ax.text(-0.01, z.mid, f"{z.mid:,.0f} ", transform=label, fontsize=7, color="#1a73e8",
                    va="center", ha="right", clip_on=False)
            extra_y += [float(z.low), float(z.high)]
    if swings_df is not None and not swings_df.empty:
        for _, s in swings_df.iterrows():
            color = "#0a7d32" if s.side == "low" else "#b3261e"
            ax.axhline(s.price, color=color, linewidth=0.9, linestyle="--" if s.broken else "-",
                       alpha=0.75, zorder=0)
            ax.text(-0.01, s.price, f"{s.price:,.0f} ", transform=label, fontsize=7, color=color,
                    va="center", ha="right", clip_on=False)
            extra_y.append(float(s.price))
    if extra_y:
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(min(ylo, *extra_y) * 0.998, max(yhi, *extra_y) * 1.002)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    fig.savefig(buf, dpi=80, format="png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Email
# --------------------------------------------------------------------------- #
def fmt_sigs(sigs) -> str:
    color = {"bull": "#0a7d32", "bear": "#b3261e", "neutral": "#666"}
    return ", ".join(f'<span style="color:{color[s]}">{n}</span>' for n, s, _ in sigs)


ALERT_TO = os.getenv("ALERT_TO", "REPLACE_WITH_YOUR_EMAIL")


def alert_mailto(sym: str, side: str, close: float, high: float, low: float, short: bool = False) -> str:
    """mailto: link that opens a pre-filled alert request. The Claude email task reads
    these emails every morning and checks them against the latest daily close."""
    default = f"close above {high:.2f}" if side != "bear" else f"close below {low:.2f}"
    body = (
        f"Alert me when: {default}\n\n"
        "Edit the line above if you want, then press Send. Checked on every day's closing data; "
        "a triggered alert appears at the top of the 8 AM Technical Analysis email.\n\n"
        "Examples you can write:\n"
        "  close above 1150  |  close below 1080  |  high above 1200  |  low below 1050\n"
        "  close crosses above 200 DMA  |  close below 50 DMA  |  RSI below 30  |  RSI above 70\n"
        "  volume above 2x  |  change above 5%  |  change below -4%  |  52-week high  |  golden cross\n"
        "  combine with 'and', e.g.  close above 1150 and volume above 1.5x\n\n"
        f"Reference: {sym} closed {close:.2f} (high {high:.2f}, low {low:.2f}).\n"
        f"To cancel later, send an email with subject: CANCEL ALERT {sym}"
    )
    if short:  # compact version for the email, which must stay small
        body = (f"Alert me when: {default}\n\n(Edit and send. e.g. close below 1100, RSI below 30, "
                f"close above 200 DMA, volume above 2x. Cancel: subject CANCEL ALERT {sym})")
    return f"mailto:{ALERT_TO}?subject={quote('ALERT ' + sym)}&body={quote(body)}"


def table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p style='color:#666'>None today.</p>"
    rows = []
    for _, r in df.iterrows():
        chg_color = "#0a7d32" if r["Chg%"] >= 0 else "#b3261e"
        rows.append(
            f"<tr><td><b>{r.Symbol}</b><br><span style='color:#888;font-size:11px'>{r.Company}</span></td>"
            f"<td align='right'>{r.Close:,.2f}</td>"
            f"<td align='right' style='color:{chg_color}'>{r['Chg%']:+.2f}%</td>"
            f"<td align='right'>{r.VolX}x</td><td align='right'>{r.RSI:.0f}</td>"
            f"<td>{fmt_sigs(r.Signals)}</td>"
            + (f"<td align='center'><a href='{alert_mailto(r.Symbol, r.Side, r.Close, r.High, r.Low, True)}' "
               f"style='text-decoration:none'>&#128276;</a></td></tr>"
               if r.Symbol in {c["sym"] for c in CHARTS} else "<td></td></tr>")
        )
    return (
        "<table cellpadding='5' cellspacing='0' border='1' "
        "style='border-collapse:collapse;border-color:#ddd;font-size:13px'>"
        "<tr style='background:#f4f4f4'><th align='left'>Stock</th><th>Close</th><th>Chg</th>"
        "<th>Vol vs 20d</th><th>RSI</th><th align='left'>Signals</th><th>Alert</th></tr>"
        + "".join(rows) + "</table>"
    )


def levels_tested_html(results: pd.DataFrame, chart_cid: dict[str, str]) -> str:
    """Every stock whose high/low range today traded into a support/resistance zone
    or through a swing level - the 'specific daily alert' for tested levels."""
    if results.empty or "LevelTests" not in results:
        return "<p style='color:#666'>None today.</p>"
    rows = []
    for _, r in results.iterrows():
        for t in r.LevelTests:
            if r.Symbol in chart_cid:
                chart_link, chart_label = f"cid:{chart_cid[r.Symbol]}", "chart"
            else:
                chart_link = f"https://www.tradingview.com/chart/?symbol=NSE:{quote(r.Symbol)}"
                chart_label = "TradingView"
            if t["kind"] == "zone":
                label = {"above": "Resistance zone", "below": "Support zone"}.get(t["side"], "Zone (price inside)")
                level_txt = f"{t['low']:,.2f}&ndash;{t['high']:,.2f}"
                extra = f"score {t['score']:.2f} &middot; {t['touches']} touches"
            else:
                label = "Swing high (resistance)" if t["side"] == "high" else "Swing low (support)"
                level_txt = f"{t['price']:,.2f}"
                extra = ("confirmed" if t["confirmed"] else "unconfirmed") + (", broken" if t["broken"] else "")
            rows.append(
                f"<tr><td><b>{r.Symbol}</b> <a href='{chart_link}' style='font-size:11px'>[{chart_label}]</a>"
                f"<br><span style='color:#888;font-size:11px'>{r.Company}</span></td>"
                f"<td>{label}</td><td align='right'>{level_txt}</td>"
                f"<td align='right'>{r.Close:,.2f}</td>"
                f"<td style='font-size:12px;color:#666'>{extra}</td></tr>"
            )
    if not rows:
        return "<p style='color:#666'>None today.</p>"
    return (
        "<table cellpadding='5' cellspacing='0' border='1' "
        "style='border-collapse:collapse;border-color:#ddd;font-size:13px;width:100%'>"
        "<tr style='background:#f4f4f4'><th align='left'>Stock</th><th align='left'>Level</th>"
        "<th>Price</th><th>Close</th><th align='left'>Detail</th></tr>"
        + "".join(rows) + "</table>"
    )


def neutral_html(df: pd.DataFrame) -> str:
    """Inside bars / NR7 / doji are common, so list them compactly by setup."""
    if df.empty:
        return "<p style='color:#666'>None today.</p>"
    groups: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        for n, _, _ in r.Signals:
            groups.setdefault(n, []).append(r.Symbol)
    return "".join(
        f"<p style='font-size:13px'><b>{k}</b> ({len(v)}): {', '.join(sorted(v))}</p>"
        for k, v in sorted(groups.items())
    )


CHARTS: list[dict] = []   # charts drawn this run, in email order
PRICES: dict = {}         # symbol -> DataFrame, set in main()


def stacked_page_md(day_label: str) -> str:
    """README.md for reports/<date>/: every chart stacked, each with an alert button."""
    out = [f"# Technical Analysis Alert on Nifty 500 - charts for {day_label}", "",
           "Scroll through all charts. Tap **Set alert** under any chart to get an email "
           "alert in the 8 AM report when your price/condition triggers on the daily close.", ""]
    for i, c in enumerate(CHARTS, 1):
        tag = {"bull": "BULLISH", "bear": "BEARISH"}.get(c["side"], "LEVEL TEST")
        tv = f"https://www.tradingview.com/chart/?symbol=NSE:{quote(c['sym'])}"
        mail = alert_mailto(c["sym"], c["side"], c["close"], c["high"], c["low"])
        out += [
            f"## {i}. {c['sym']} - {tag}",
            f"**{c['close']:,.2f}** ({c['chg']:+.2f}%) | vol {c['volx']}x | RSI {c['rsi']:.0f} | "
            f"H {c['high']:,.2f} / L {c['low']:,.2f}  ",
            f"{c['signals']}",
            "",
            f"![{c['sym']}]({c['cid']}.png)",
            "",
            f"[🔔 Set alert on {c['sym']}](https://claude.ai/artifact/Ewc5hHTPX6WkU1nVcDNQzq?s={quote(c['sym'])}&c={quote(('close above %.2f' % c['high']) if c['side'] == 'bull' else ('close below %.2f' % c['low']))}) &nbsp;|&nbsp; [📈 Live chart (TradingView)]({tv}) &nbsp;|&nbsp; [alert by email]({mail})",
            "", "---", "",
        ]
    return "\n".join(out)


def levels_json(day: str) -> dict:
    """Latest daily levels for every scanned stock, used to check alerts each morning."""
    out = {}
    for sym, d in PRICES.items():
        try:
            c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
            s20, s50, s200 = (c.rolling(n).mean() for n in (20, 50, 200))
            r = rsi(c)
            look = min(250, len(c) - 1)
            vavg = v.iloc[-21:-1].mean()
            f = lambda x: None if pd.isna(x) else round(float(x), 2)  # noqa: E731
            out[sym] = {
                "date": str(pd.Timestamp(d.index[-1]).date()),
                "open": f(d["Open"].iloc[-1]), "high": f(h.iloc[-1]), "low": f(l.iloc[-1]),
                "close": f(c.iloc[-1]), "prev_close": f(c.iloc[-2]),
                "change_pct": f((c.iloc[-1] / c.iloc[-2] - 1) * 100),
                "volume_x": f(v.iloc[-1] / vavg) if vavg else None,
                "rsi": f(r.iloc[-1]), "prev_rsi": f(r.iloc[-2]),
                "sma20": f(s20.iloc[-1]), "sma50": f(s50.iloc[-1]), "sma200": f(s200.iloc[-1]),
                "prev_sma20": f(s20.iloc[-2]), "prev_sma50": f(s50.iloc[-2]), "prev_sma200": f(s200.iloc[-2]),
                "prior_52w_high": f(h.iloc[-look - 1:-1].max()), "prior_52w_low": f(l.iloc[-look - 1:-1].min()),
            }
        except Exception as e:
            print(f"  levels failed {sym}: {e}")
    return {"as_of": day, "stocks": out}


def build_email(results: pd.DataFrame, prices, candle_date: dt.date, universe_n: int):
    bull = results[results.Side == "bull"].sort_values(["Score", "VolX"], ascending=False)
    bear = results[results.Side == "bear"].sort_values(["Score", "VolX"], ascending=False)
    neutral = results[results.Side == "neutral"]

    images: list[tuple[str, bytes]] = []
    chart_blocks = {"bull": [], "bear": []}
    chart_cid: dict[str, str] = {}
    for side, df in (("bull", bull), ("bear", bear)):
        for _, r in df.head(MAX_CHARTS).iterrows():
            names = ", ".join(n for n, _, _ in r.Signals)
            title = f"{r.Symbol}  {r.Close:,.2f} ({r['Chg%']:+.2f}%)\n{names}"
            try:
                png = make_chart(r.Symbol, prices[r.Symbol], title, ZONES.get(r.Symbol), SWINGS.get(r.Symbol))
            except Exception as e:
                print(f"  chart failed {r.Symbol}: {e}")
                continue
            cid = f"{side}_{r.Symbol}".replace("&", "and").replace("-", "_")
            images.append((cid, png))
            chart_cid[r.Symbol] = cid
            CHARTS.append({"sym": r.Symbol, "company": r.Company, "side": side, "cid": cid,
                           "close": r.Close, "chg": r["Chg%"], "volx": r.VolX, "rsi": r.RSI,
                           "high": r.High, "low": r.Low,
                           "signals": ", ".join(n for n, _, _ in r.Signals)})
            chart_blocks[side].append(
                f"<div style='margin:10px 0'><img src='cid:{cid}' width='720' "
                f"style='max-width:100%;border:1px solid #eee' alt='{r.Symbol}'></div>"
            )

    # A stock can test a level with no candlestick/technical signal (it won't be in the
    # bull/bear chart lists above) - still give it an annotated chart for the alert below.
    if "LevelTests" in results:
        for _, r in results[results["LevelTests"].map(bool)].iterrows():
            if r.Symbol in chart_cid:
                continue
            title = f"{r.Symbol}  {r.Close:,.2f} ({r['Chg%']:+.2f}%)\nLevel test"
            try:
                png = make_chart(r.Symbol, prices[r.Symbol], title, ZONES.get(r.Symbol), SWINGS.get(r.Symbol))
            except Exception as e:
                print(f"  chart failed {r.Symbol}: {e}")
                continue
            cid = f"level_{r.Symbol}".replace("&", "and").replace("-", "_")
            images.append((cid, png))
            chart_cid[r.Symbol] = cid
            CHARTS.append({"sym": r.Symbol, "company": r.Company, "side": "neutral", "cid": cid,
                           "close": r.Close, "chg": r["Chg%"], "volx": r.VolX, "rsi": r.RSI,
                           "high": r.High, "low": r.Low, "signals": "Level test"})

    day = candle_date.strftime("%a %d %b %Y")
    test_count = int(results["LevelTests"].map(len).sum()) if "LevelTests" in results else 0
    html = f"""
<div style="font-family:Arial,Helvetica,sans-serif;color:#222;max-width:900px">
<h2 style="margin-bottom:4px">Nifty 500 daily scan - candle of {day}</h2>
<p style="color:#666;margin-top:0">{len(prices)} of {universe_n} stocks scanned &middot;
<b style="color:#0a7d32">{len(bull)} bullish</b> &middot;
<b style="color:#b3261e">{len(bear)} bearish</b> &middot; {len(neutral)} neutral setups &middot;
<b style="color:#1a73e8">{test_count} level test(s)</b>.
Ranked by signal strength, then volume. Charts: last ~6 months, 20/50/200 SMA (blue/orange/grey),
support/resistance zones (blue bands) and swing levels (green = support, red = resistance, dashed = broken).</p>

<div style="border:2px solid #1a73e8;border-radius:8px;padding:10px 14px;margin:12px 0">
<h3 style="color:#1a73e8;margin:4px 0">&#127919; Levels tested today ({test_count})</h3>
{levels_tested_html(results, chart_cid)}
</div>

<h3 style="color:#0a7d32">Bullish</h3>{table_html(bull)}
<h3 style="color:#b3261e">Bearish</h3>{table_html(bear)}
<h3 style="color:#666">Neutral / volatility contraction</h3>{neutral_html(neutral)}

<h3 style="color:#0a7d32">Bullish charts (top {MAX_CHARTS})</h3>{''.join(chart_blocks['bull']) or '<p>None</p>'}
<h3 style="color:#b3261e">Bearish charts (top {MAX_CHARTS})</h3>{''.join(chart_blocks['bear']) or '<p>None</p>'}

<p style="color:#999;font-size:11px">Data: Yahoo Finance daily candles (free, unofficial, can lag or have gaps).
Patterns are rule-based and need trend context confirmation. Not investment advice.</p>
</div>"""
    subject = f"Nifty 500 scan {candle_date:%d-%b}: {len(bull)} bullish / {len(bear)} bearish"
    return subject, html, images


def publish_report(subject: str, html: str, images: list[tuple[str, bytes]]):
    """Write the report into reports/<date>/ with charts referenced by public URL,
    plus reports/latest.json, so another service (Claude + Gmail connector) can mail it.
    The repo must be public so Gmail can load the chart images."""
    import json
    import shutil

    day = dt.datetime.now(IST).strftime("%Y-%m-%d")
    rep_root = ROOT / "reports"
    folder = rep_root / day
    folder.mkdir(parents=True, exist_ok=True)
    base = os.environ["PUBLISH_BASE_URL"].rstrip("/") + f"/reports/{day}/"
    page = html
    for cid, png in images:
        (folder / f"{cid}.png").write_bytes(png)
        page = page.replace(f"cid:{cid}", base + f"{cid}.png")
    (folder / "email.html").write_text(page)
    candle_day = (next(iter(PRICES.values())).index[-1].strftime("%a %d %b %Y") if PRICES else day)
    (folder / "README.md").write_text(stacked_page_md(candle_day))
    # permanent "latest charts" page (bookmarkable, same URL every day)
    (rep_root / "README.md").write_text(stacked_page_md(candle_day)
                                        .replace("](bull_", f"]({day}/bull_").replace("](bear_", f"]({day}/bear_"))
    lv = levels_json(day)
    (rep_root / "levels.json").write_text(json.dumps(lv, separators=(",", ":")))
    repo = os.getenv("GITHUB_REPOSITORY", "HimanshuSardana05/nifty500-scanner")
    (rep_root / "latest.json").write_text(json.dumps({
        "run_date": day, "subject": subject, "html_url": base + "email.html",
        "charts_url": f"https://github.com/{repo}/blob/main/reports/{day}/README.md",
        "levels_url": os.environ["PUBLISH_BASE_URL"].rstrip("/") + "/reports/levels.json",
        "levels_as_of": next(iter(lv["stocks"].values()))["date"] if lv["stocks"] else None}))
    # keep the repo small: only the last 10 report folders
    olds = sorted(p for p in rep_root.iterdir() if p.is_dir())[:-10]
    for p in olds:
        shutil.rmtree(p)
    print(f"Published report {folder} ({len(images)} charts). Subject: {subject}")


def send_email(subject: str, html: str, images: list[tuple[str, bytes]]):
    user = os.environ.get("GMAIL_USER", "")
    to = [a.strip() for a in os.environ.get("MAIL_TO", user).split(",") if a.strip()]
    if os.getenv("PUBLISH_BASE_URL"):
        publish_report(subject, html, images)
        return
    if os.getenv("DRY_RUN") == "1" or not user:
        OUT_DIR.mkdir(exist_ok=True)
        page = html
        for cid, png in images:
            (OUT_DIR / f"{cid}.png").write_bytes(png)
            page = page.replace(f"cid:{cid}", f"{cid}.png")
        (OUT_DIR / "email.html").write_text(page)
        print(f"DRY RUN: wrote {OUT_DIR / 'email.html'} with {len(images)} charts. Subject: {subject}")
        return

    msg = MIMEMultipart("related")
    msg["Subject"], msg["From"], msg["To"] = subject, user, ", ".join(to)
    msg.attach(MIMEText(html, "html"))
    for cid, png in images:
        img = MIMEImage(png, "png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(img)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as s:
        s.login(user, os.environ["GMAIL_APP_PASSWORD"])
        s.sendmail(user, to, msg.as_string())
    print(f"Sent to {to}: {subject} ({len(images)} charts)")


# --------------------------------------------------------------------------- #
def main():
    uni = load_universe()
    names = dict(zip(uni["Symbol"], uni.get("Company Name", uni["Symbol"])))
    prices = download_prices(list(uni["Symbol"]))
    if len(prices) < 0.6 * len(uni):
        raise RuntimeError(f"Only {len(prices)} of {len(uni)} symbols downloaded - data source problem.")
    prices = {s: drop_incomplete_bar(d) for s, d in prices.items()}

    # Latest candle date = most common last date (ignores suspended stocks)
    last_dates = pd.Series([pd.Timestamp(d.index[-1]).date() for d in prices.values()])
    candle_date = last_dates.mode().iloc[0]
    prices = {s: d for s, d in prices.items() if pd.Timestamp(d.index[-1]).date() == candle_date}

    if STATE_FILE.exists() and STATE_FILE.read_text().strip() == str(candle_date) and os.getenv("FORCE") != "1":
        print(f"Candle {candle_date} already reported (market holiday?) - skipping email.")
        if os.getenv("PUBLISH_BASE_URL"):
            import json
            latest_path = ROOT / "reports" / "latest.json"
            prev = {}
            if latest_path.exists():
                try:
                    prev = json.loads(latest_path.read_text())
                except Exception:
                    prev = {}
            prev["run_date"] = dt.datetime.now(IST).strftime("%Y-%m-%d")
            prev["skipped"] = True
            prev["skip_reason"] = f"candle of {candle_date} already reported - no new trading session since (market holiday or data not yet updated)"
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(json.dumps(prev))
        return

    PRICES.update(prices)
    results = scan(prices, names)
    subject, html, images = build_email(results, prices, candle_date, len(uni))
    send_email(subject, html, images)
    if os.getenv("DRY_RUN") != "1":
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(str(candle_date))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        # Tell the user by email rather than failing silently
        if os.getenv("PUBLISH_BASE_URL"):
            import json
            (ROOT / "reports").mkdir(exist_ok=True)
            (ROOT / "reports" / "latest.json").write_text(json.dumps({
                "run_date": dt.datetime.now(IST).strftime("%Y-%m-%d"),
                "subject": "Nifty 500 scan FAILED today", "error": traceback.format_exc()[-1500:]}))
        elif os.environ.get("GMAIL_USER") and os.getenv("DRY_RUN") != "1":
            try:
                send_email("Nifty 500 scan FAILED today",
                           f"<pre>{traceback.format_exc()}</pre>", [])
            except Exception:
                traceback.print_exc()
        sys.exit(1)

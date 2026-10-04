"""Memoised indicator accessors shared by the worker families."""
from __future__ import annotations

import numpy as np

from lunatrade.indicators import ta


def ema(ctx, sym, n):
    return ctx.view.ind(sym, "ema", lambda c, n: ta.ema(c.close, n), n)


def sma(ctx, sym, n):
    return ctx.view.ind(sym, "sma", lambda c, n: ta.sma(c.close, n), n)


def atr(ctx, sym, n=14):
    return ctx.view.ind(sym, "atr", lambda c, n: ta.atr(c.high, c.low, c.close, n), n)


def rsi(ctx, sym, n=14):
    return ctx.view.ind(sym, "rsi", lambda c, n: ta.rsi(c.close, n), n)


def adx(ctx, sym, n=14):
    return ctx.view.ind(sym, "adx", lambda c, n: ta.adx(c.high, c.low, c.close, n), n)


def macd(ctx, sym, f=12, s=26, g=9):
    return ctx.view.ind(sym, "macd", lambda c, f, s, g: ta.macd(c.close, f, s, g), f, s, g)


def boll(ctx, sym, n=20, k=2.0):
    return ctx.view.ind(sym, "boll", lambda c, n, k: ta.bollinger(c.close, n, k), n, k)


def keltner(ctx, sym, n=20, m=1.5):
    return ctx.view.ind(sym, "kelt", lambda c, n, m: ta.keltner(c.high, c.low, c.close, n, m), n, m)


def vwap(ctx, sym, n):
    return ctx.view.ind(sym, "vwap", lambda c, n: ta.vwap(c.high, c.low, c.close, c.volume, n), n)


def stoch(ctx, sym, n=14):
    return ctx.view.ind(sym, "stoch", lambda c, n: ta.stochastic(c.high, c.low, c.close, n), n)


def zs(ctx, sym, n):
    return ctx.view.ind(sym, "zs", lambda c, n: ta.zscore(c.close, n), n)


def rvol(ctx, sym, n=30):
    return ctx.view.ind(sym, "rvol", lambda c, n: ta.realized_vol(c.close, n), n)


def volume_z(ctx, sym, n=20):
    return ctx.view.ind(sym, "volz", lambda c, n: ta.zscore(np.log1p(c.volume), n), n)


def pivots(ctx, sym, left=3):
    return ctx.view.ind(sym, "piv", lambda c, l: ta.pivots(c.high[-300:], c.low[-300:], l, l), left)


def er(ctx, sym, n=20):
    return ctx.view.ind(sym, "er", lambda c, n: ta.efficiency_ratio(c.close, n), n)


def rets(ctx, sym):
    return ctx.view.ind(sym, "ret", lambda c: ta.returns(c.close))


def last(x, default=0.0):
    v = ta.last(x, default=np.nan)
    return default if np.isnan(v) else v


def atr_pct(ctx, sym, n=14) -> float:
    c = ctx.view.candles(sym)
    a = last(atr(ctx, sym, n))
    return a / c.close[-1] if len(c) and c.close[-1] else 0.0


def fresh_cross(a, b, lookback=3) -> int:
    """+1 if a crossed above b within lookback bars, -1 if below, else 0."""
    if len(a) < lookback + 2:
        return 0
    d = np.sign(np.nan_to_num(a[-lookback - 1:] - b[-lookback - 1:]))
    if d[-1] > 0 and (d[:-1] <= 0).any():
        return 1
    if d[-1] < 0 and (d[:-1] >= 0).any():
        return -1
    return 0

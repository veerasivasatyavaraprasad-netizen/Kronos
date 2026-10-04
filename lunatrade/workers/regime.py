"""Regime-detection workers (10). Each casts weighted votes; the RegimeEngine tallies them."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.core.types import Regime as R
from lunatrade.indicators import ta
from lunatrade.intel.assets import base_of
from lunatrade.workers import common as K
from lunatrade.workers.base import clamp, spec

N = D.NEUTRAL


def _vote(w, ctx, sym, votes: dict, **extra):
    return ctx.signal(w, sym, N, 0.0, setup="REGIME_VOTE",
                      context={"votes": {k.value if hasattr(k, "value") else k: round(float(v), 3)
                                         for k, v in votes.items() if v > 0}, **extra})


def er_trend(w, ctx, sym):
    e = K.last(K.er(ctx, sym, 20), 0.3)
    return _vote(w, ctx, sym, {R.TRENDING: clamp((e - 0.3) * 3), R.RANGING: clamp((0.3 - e) * 4)}, efficiency=e)


def adx_trend(w, ctx, sym):
    a = K.last(K.adx(ctx, sym, 14)[0], 20)
    return _vote(w, ctx, sym, {R.TRENDING: clamp((a - 22) / 15), R.RANGING: clamp((20 - a) / 10)}, adx=a)


def atr_vol(w, ctx, sym):
    ap = ctx.view.ind(sym, "atrp", lambda c: ta.atr(c.high, c.low, c.close, 14) / c.close)
    p = K.last(ta.percentile_rank(np.nan_to_num(ap), min(200, len(ap))), 0.5)
    return _vote(w, ctx, sym, {R.HIGH_VOL: clamp((p - 0.7) * 3.3), R.LOW_VOL: clamp((0.3 - p) * 3.3)}, atr_pctile=p)


def rv_vol(w, ctx, sym):
    s, l = K.last(K.rvol(ctx, sym, 12)), K.last(K.rvol(ctx, sym, 96))
    ratio = s / l if l else 1
    return _vote(w, ctx, sym, {R.HIGH_VOL: clamp((ratio - 1.3)), R.LOW_VOL: clamp((0.7 - ratio) * 2)}, rv_ratio=ratio)


def bull_bear_ema(w, ctx, sym):
    c = ctx.view.candles(sym).close
    n = min(200, len(c) - 5)
    e = ta.ema(c, n)
    e50 = K.ema(ctx, sym, 50)
    if np.isnan(e[-1]) or np.isnan(e50[-1]):
        return None
    above = c[-1] > e[-1]
    slope = e50[-1] / e50[-10] - 1 if len(e50) > 10 else 0
    return _vote(w, ctx, sym, {R.BULL: 0.6 * above + clamp(slope * 50) * 0.4,
                               R.BEAR: 0.6 * (not above) + clamp(-slope * 50) * 0.4}, ema_slope=slope)


def bull_bear_drawdown(w, ctx, sym):
    c = ctx.view.candles(sym)
    hi = c.high[-200:].max()
    lo = c.low[-200:].min()
    dd = c.close[-1] / hi - 1
    up = c.close[-1] / lo - 1
    return _vote(w, ctx, sym, {R.BEAR: clamp(-dd * 4 - 0.2), R.BULL: clamp(up * 3 - 0.2) * (dd > -0.1)}, drawdown=dd)


def panic(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 30:
        return None
    a = K.atr_pct(ctx, sym) or 1e-6
    drop = c.close[-1] / c.high[-8:].max() - 1
    vz = K.last(K.volume_z(ctx, sym, 50))
    score = clamp((-drop / a - 3) / 4) * clamp((vz + 1) / 3)
    return _vote(w, ctx, sym, {R.PANIC: score}, fast_drop=drop, volume_z=vz)


def liquidity(w, ctx, sym):
    c = ctx.view.candles(sym)
    qv = c.quote_volume[-200:]
    if len(qv) < 50 or not qv.mean():
        return None
    p = (qv[-24:].mean()) / qv.mean()
    sp = ctx.view.spread_bps(sym)
    illiquid = clamp((0.4 - p) * 2.5) + (clamp((sp - 10) / 20) if sp else 0)
    return _vote(w, ctx, sym, {R.ILLIQUID: clamp(illiquid)}, volume_vs_avg=p, spread_bps=sp)


def event_driven(w, ctx, sym):
    big_news = [x for x in ctx.feeds.news() if x.impact >= 0.7 and x.decay(ctx.now) > 0.5 and
                (base_of(sym) in x.assets or x.market_wide)]
    events = ctx.feeds.upcoming_events(24)
    score = clamp(0.3 * len(big_news) + 0.5 * bool(events))
    return _vote(w, ctx, sym, {R.EVENT_DRIVEN: score}, high_impact_news=len(big_news), macro_events=len(events))


def hurst_regime(w, ctx, sym):
    c = ctx.view.candles(sym).close
    h = ctx.view.ind(sym, "hurst", lambda cc: ta.hurst(np.log(cc.close[-256:])))
    return _vote(w, ctx, sym, {R.TRENDING: clamp((h - 0.55) * 5), R.RANGING: clamp((0.45 - h) * 5)}, hurst=h)


def specs() -> list[dict]:
    G = Family.REGIME
    return [spec(G, "trend_vs_range_er", er_trend, min_bars=40),
            spec(G, "trend_vs_range_adx", adx_trend, min_bars=50),
            spec(G, "vol_regime_atr", atr_vol, min_bars=60),
            spec(G, "vol_regime_rv", rv_vol, min_bars=100),
            spec(G, "bull_bear_ema", bull_bear_ema, min_bars=80),
            spec(G, "bull_bear_drawdown", bull_bear_drawdown, min_bars=60),
            spec(G, "panic_detector", panic, min_bars=60),
            spec(G, "liquidity_regime", liquidity, min_bars=60),
            spec(G, "event_driven", event_driven, min_bars=1),
            spec(G, "hurst", hurst_regime, min_bars=100)]

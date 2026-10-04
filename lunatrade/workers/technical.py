"""Technical setup hunters (60 workers): trend, mean reversion, market structure and candlestick patterns."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.workers import common as K
from lunatrade.workers.base import clamp, spec

L, S = D.LONG, D.SHORT


def _move(ctx, sym, mult=2.0, direction=L):
    return direction.sign * mult * K.atr_pct(ctx, sym)


# ------------------------------------------------------------------------------------- trend

def ema_cross(w, ctx, sym):
    f, s = w.params["fast"], w.params["slow"]
    ef, es, a = K.ema(ctx, sym, f), K.ema(ctx, sym, s), K.last(K.atr(ctx, sym))
    if np.isnan(es[-1]) or not a:
        return None
    gap = (ef[-1] - es[-1]) / a
    d = L if gap > 0 else S
    cross = K.fresh_cross(ef, es, 3)
    conf = clamp(abs(gap) * 0.35 + (0.25 if cross == d.sign else 0))
    return ctx.signal(w, sym, d, conf, setup="EMA_TREND", expected_move=_move(ctx, sym, 2, d), time_horizon="1d",
                      evidence=[f"ema{f}{'>' if gap > 0 else '<'}ema{s}"] + (["fresh_cross"] if cross else []),
                      invalidations=["ema_recross"])


def vwap_trend(w, ctx, sym):
    n = w.params["n"]
    v, c = K.vwap(ctx, sym, n), ctx.view.candles(sym).close
    if len(v) < n + 5:
        return None
    slope = (v[-1] - v[-5]) / v[-5]
    above = c[-1] > v[-1]
    if above and slope > 0:
        d = L
    elif not above and slope < 0:
        d = S
    else:
        return None
    conf = clamp(abs(c[-1] / v[-1] - 1) * 40 + abs(slope) * 200)
    return ctx.signal(w, sym, d, conf, setup="VWAP_TREND", expected_move=_move(ctx, sym, 1.5, d),
                      evidence=["price_vs_vwap", "vwap_slope"], invalidations=["vwap_reclaim"])


def adx_trend(w, ctx, sym):
    n, thr = w.params["n"], w.params["threshold"]
    a, pdi, mdi = K.adx(ctx, sym, n)
    av = K.last(a)
    if av < thr:
        return None
    d = L if K.last(pdi) > K.last(mdi) else S
    return ctx.signal(w, sym, d, clamp((av - thr) / 25 + 0.2), setup="ADX_TREND", expected_move=_move(ctx, sym, 2, d),
                      evidence=[f"adx={av:.0f}"], invalidations=["adx_falling"])


def donchian_breakout(w, ctx, sym):
    n = w.params["n"]
    c = ctx.view.candles(sym)
    if len(c) < n + 2:
        return None
    hi, lo = c.high[-n - 1:-1].max(), c.low[-n - 1:-1].min()
    vz = K.last(K.volume_z(ctx, sym))
    if c.close[-1] > hi:
        d = L
    elif c.close[-1] < lo:
        d = S
    else:
        return None
    conf = clamp(0.35 + 0.15 * max(0, vz))
    return ctx.signal(w, sym, d, conf, setup="BREAKOUT", expected_move=_move(ctx, sym, 3, d), time_horizon="4h",
                      evidence=[f"donchian{n}_break"] + (["volume_expansion"] if vz > 1 else []),
                      invalidations=["breakout_failure", "volume_collapse"])


def structure_trend(w, ctx, sym):
    sh, sl = K.pivots(ctx, sym, w.params["left"])
    c = ctx.view.candles(sym).tail(300)
    if len(sh) < 2 or len(sl) < 2:
        return None
    hh, hl = c.high[sh[-1]] > c.high[sh[-2]], c.low[sl[-1]] > c.low[sl[-2]]
    if hh and hl:
        d, ev = L, ["higher_high", "higher_low"]
    elif not hh and not hl:
        d, ev = S, ["lower_high", "lower_low"]
    else:
        return None
    return ctx.signal(w, sym, d, 0.45, setup="MARKET_STRUCTURE", expected_move=_move(ctx, sym, 2, d),
                      time_horizon="1d", evidence=ev, invalidations=["structure_break"])


def macd_trend(w, ctx, sym):
    line, sig, hist = K.macd(ctx, sym, w.params["fast"], w.params["slow"], w.params["signal"])
    if np.isnan(hist[-1]):
        return None
    a = K.last(K.atr(ctx, sym)) or 1
    d = L if hist[-1] > 0 else S
    cross = K.fresh_cross(line, sig, 2)
    conf = clamp(abs(hist[-1]) / a * 1.5 + (0.2 if cross == d.sign else 0) + (0.1 if np.sign(line[-1]) == d.sign else 0))
    return ctx.signal(w, sym, d, conf, setup="MACD", expected_move=_move(ctx, sym, 1.5, d),
                      evidence=["macd_hist"] + (["macd_cross"] if cross else []), invalidations=["macd_flip"])


# ------------------------------------------------------------------------------ mean reversion

def bollinger_revert(w, ctx, sym):
    lo, mid, hi = K.boll(ctx, sym, w.params["n"], w.params["k"])
    c = ctx.view.candles(sym).close[-1]
    if np.isnan(mid[-1]) or hi[-1] == lo[-1]:
        return None
    if c < lo[-1]:
        d = L
    elif c > hi[-1]:
        d = S
    else:
        return None
    dist = abs(c - mid[-1]) / (hi[-1] - mid[-1])
    return ctx.signal(w, sym, d, clamp((dist - 1) * 1.5 + 0.3), setup="BOLLINGER_REVERSION",
                      expected_move=d.sign * abs(mid[-1] / c - 1), time_horizon="4h",
                      evidence=["outside_band"], invalidations=["band_walk"])


def rsi_extreme(w, ctx, sym):
    r = K.last(K.rsi(ctx, sym, w.params["n"]), 50)
    lo, hi = w.params["lo"], w.params["hi"]
    if r < lo:
        d, conf = L, (lo - r) / lo + 0.25
    elif r > hi:
        d, conf = S, (r - hi) / (100 - hi) + 0.25
    else:
        return None
    return ctx.signal(w, sym, d, clamp(conf), setup="RSI_EXTREME", expected_move=_move(ctx, sym, 1.2, d),
                      evidence=[f"rsi{w.params['n']}={r:.0f}"], invalidations=["rsi_divergence_fails"])


def vwap_deviation(w, ctx, sym):
    n, thr = w.params["n"], w.params["z"]
    c = ctx.view.candles(sym)
    v = K.vwap(ctx, sym, n)
    dev = (c.close - v)[-n:]
    sd = np.nanstd(dev)
    if not sd:
        return None
    z = dev[-1] / sd
    if abs(z) < thr:
        return None
    d = S if z > 0 else L
    return ctx.signal(w, sym, d, clamp((abs(z) - thr) * 0.4 + 0.3), setup="VWAP_DEVIATION",
                      expected_move=d.sign * abs(v[-1] / c.close[-1] - 1), evidence=[f"vwap_z={z:.2f}"],
                      invalidations=["trend_day"])


def zscore_revert(w, ctx, sym):
    z = K.last(K.zs(ctx, sym, w.params["n"]))
    thr = w.params["z"]
    if abs(z) < thr:
        return None
    d = S if z > 0 else L
    return ctx.signal(w, sym, d, clamp((abs(z) - thr) * 0.35 + 0.3), setup="ZSCORE_REVERSION",
                      expected_move=_move(ctx, sym, 1.0, d), evidence=[f"z{w.params['n']}={z:.2f}"],
                      invalidations=["regime_trending"])


def keltner_revert(w, ctx, sym):
    lo, mid, hi = K.keltner(ctx, sym, w.params["n"], w.params["mult"])
    c = ctx.view.candles(sym).close[-1]
    if np.isnan(mid[-1]):
        return None
    if c < lo[-1]:
        d = L
    elif c > hi[-1]:
        d = S
    else:
        return None
    return ctx.signal(w, sym, d, 0.4, setup="KELTNER_REVERSION", expected_move=d.sign * abs(mid[-1] / c - 1),
                      evidence=["outside_keltner"], invalidations=["channel_walk"])


def stoch_revert(w, ctx, sym):
    k, dd = K.stoch(ctx, sym, w.params["n"])
    if np.isnan(k[-1]) or np.isnan(dd[-1]) or len(k) < 3:
        return None
    if k[-1] < 20 and k[-1] > dd[-1] and k[-2] <= dd[-2]:
        d = L
    elif k[-1] > 80 and k[-1] < dd[-1] and k[-2] >= dd[-2]:
        d = S
    else:
        return None
    return ctx.signal(w, sym, d, 0.45, setup="STOCH_REVERSAL", expected_move=_move(ctx, sym, 1, d),
                      evidence=["stoch_cross_in_extreme"], invalidations=["stoch_reverse"])


# ---------------------------------------------------------------------------- market structure

def sr_bounce(w, ctx, sym):
    sh, sl = K.pivots(ctx, sym, w.params["left"])
    c = ctx.view.candles(sym).tail(300)
    a = K.last(K.atr(ctx, sym))
    if not a or not sl or not sh:
        return None
    price = c.close[-1]
    supports = [c.low[i] for i in sl if c.low[i] < price]
    resist = [c.high[i] for i in sh if c.high[i] > price]
    bull_candle = c.close[-1] > c.open[-1]
    if supports and price - max(supports) < 0.6 * a and bull_candle:
        return ctx.signal(w, sym, L, 0.45, setup="SUPPORT_BOUNCE", expected_move=_move(ctx, sym, 2, L),
                          evidence=["near_support", "bullish_candle"], invalidations=["support_break"])
    if resist and min(resist) - price < 0.6 * a and not bull_candle:
        return ctx.signal(w, sym, S, 0.45, setup="RESISTANCE_REJECT", expected_move=_move(ctx, sym, 2, S),
                          evidence=["near_resistance", "bearish_candle"], invalidations=["resistance_break"])
    return None


def failed_breakout(w, ctx, sym):
    n = w.params["n"]
    c = ctx.view.candles(sym)
    if len(c) < n + 3:
        return None
    hi, lo = c.high[-n - 2:-2].max(), c.low[-n - 2:-2].min()
    if c.high[-2] > hi and c.close[-1] < hi and c.close[-1] < c.open[-1]:
        return ctx.signal(w, sym, S, 0.5, setup="FAILED_BREAKOUT", expected_move=-abs(c.close[-1] / ((hi + lo) / 2) - 1),
                          evidence=["breakout_failed_back_inside"], invalidations=["new_high"])
    if c.low[-2] < lo and c.close[-1] > lo and c.close[-1] > c.open[-1]:
        return ctx.signal(w, sym, L, 0.5, setup="FAILED_BREAKDOWN", expected_move=abs(c.close[-1] / ((hi + lo) / 2) - 1),
                          evidence=["breakdown_failed_back_inside"], invalidations=["new_low"])
    return None


def sweep_reclaim(w, ctx, sym):
    n = w.params["n"]
    c = ctx.view.candles(sym)
    if len(c) < n + 2:
        return None
    lo, hi = c.low[-n - 1:-1].min(), c.high[-n - 1:-1].max()
    if c.low[-1] < lo and c.close[-1] > lo:
        wick = (min(c.open[-1], c.close[-1]) - c.low[-1]) / max(c.high[-1] - c.low[-1], 1e-12)
        return ctx.signal(w, sym, L, clamp(0.35 + wick * 0.4), setup="LIQUIDITY_SWEEP_RECLAIM",
                          expected_move=_move(ctx, sym, 2, L), evidence=["sweep_of_lows", "reclaim"],
                          invalidations=["close_below_sweep"])
    if c.high[-1] > hi and c.close[-1] < hi:
        wick = (c.high[-1] - max(c.open[-1], c.close[-1])) / max(c.high[-1] - c.low[-1], 1e-12)
        return ctx.signal(w, sym, S, clamp(0.35 + wick * 0.4), setup="LIQUIDITY_SWEEP_REJECT",
                          expected_move=_move(ctx, sym, 2, S), evidence=["sweep_of_highs", "rejection"],
                          invalidations=["close_above_sweep"])
    return None


def trend_pullback(w, ctx, sym):
    f, s = w.params["fast"], w.params["slow"]
    ef, es = K.ema(ctx, sym, f), K.ema(ctx, sym, s)
    c = ctx.view.candles(sym)
    if np.isnan(es[-1]) or len(c) < s + 5:
        return None
    up = ef[-1] > es[-1] and es[-1] > es[-5]
    dn = ef[-1] < es[-1] and es[-1] < es[-5]
    if up and c.low[-3:].min() <= ef[-1] and c.close[-1] > ef[-1] and c.close[-1] > c.open[-1]:
        return ctx.signal(w, sym, L, 0.55, setup="TREND_CONTINUATION", expected_move=_move(ctx, sym, 2.5, L),
                          evidence=["uptrend", "pullback_to_ema", "resumed"], invalidations=["close_below_slow_ema"])
    if dn and c.high[-3:].max() >= ef[-1] and c.close[-1] < ef[-1] and c.close[-1] < c.open[-1]:
        return ctx.signal(w, sym, S, 0.55, setup="TREND_CONTINUATION", expected_move=_move(ctx, sym, 2.5, S),
                          evidence=["downtrend", "pullback_to_ema", "resumed"], invalidations=["close_above_slow_ema"])
    return None


def inside_bar(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 4:
        return None
    mother_h, mother_l = c.high[-3], c.low[-3]
    if not (c.high[-2] < mother_h and c.low[-2] > mother_l):
        return None
    e = K.last(K.ema(ctx, sym, w.params["trend"]))
    trend = np.sign(c.close[-1] - e) if e else 0
    if c.close[-1] > mother_h and trend >= 0:
        return ctx.signal(w, sym, L, 0.45, setup="INSIDE_BAR_BREAK", expected_move=_move(ctx, sym, 1.5, L),
                          evidence=["inside_bar", "break_up"], invalidations=["back_inside"])
    if c.close[-1] < mother_l and trend <= 0:
        return ctx.signal(w, sym, S, 0.45, setup="INSIDE_BAR_BREAK", expected_move=_move(ctx, sym, 1.5, S),
                          evidence=["inside_bar", "break_down"], invalidations=["back_inside"])
    return None


# --------------------------------------------------------------------------- candle patterns

def _body(c, i):
    return abs(c.close[i] - c.open[i])


def _range(c, i):
    return max(c.high[i] - c.low[i], 1e-12)


def _prior_trend(c, n=10):
    return np.sign(c.close[-2] - c.close[-2 - n]) if len(c) > n + 2 else 0


def engulfing(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 15:
        return None
    pt = _prior_trend(c)
    bull = c.close[-1] > c.open[-1] and c.close[-2] < c.open[-2] and c.close[-1] >= c.open[-2] and c.open[-1] <= c.close[-2]
    bear = c.close[-1] < c.open[-1] and c.close[-2] > c.open[-2] and c.close[-1] <= c.open[-2] and c.open[-1] >= c.close[-2]
    if bull and pt < 0:
        return ctx.signal(w, sym, L, 0.4, setup="BULLISH_ENGULFING", expected_move=_move(ctx, sym, 1.5, L),
                          evidence=["engulfing_after_decline"], invalidations=["low_of_pattern"])
    if bear and pt > 0:
        return ctx.signal(w, sym, S, 0.4, setup="BEARISH_ENGULFING", expected_move=_move(ctx, sym, 1.5, S),
                          evidence=["engulfing_after_rally"], invalidations=["high_of_pattern"])
    return None


def hammer_star(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 15:
        return None
    i, rng = -1, _range(c, -1)
    lower = min(c.open[i], c.close[i]) - c.low[i]
    upper = c.high[i] - max(c.open[i], c.close[i])
    body = _body(c, i)
    pt = _prior_trend(c)
    if lower > 2 * body and upper < body and lower / rng > 0.55 and pt < 0:
        return ctx.signal(w, sym, L, 0.4, setup="HAMMER", expected_move=_move(ctx, sym, 1.2, L),
                          evidence=["long_lower_wick", "after_decline"], invalidations=["hammer_low"])
    if upper > 2 * body and lower < body and upper / rng > 0.55 and pt > 0:
        return ctx.signal(w, sym, S, 0.4, setup="SHOOTING_STAR", expected_move=_move(ctx, sym, 1.2, S),
                          evidence=["long_upper_wick", "after_rally"], invalidations=["star_high"])
    return None


def three_soldiers(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 5:
        return None
    ups = all(c.close[i] > c.open[i] and c.close[i] > c.close[i - 1] for i in (-3, -2, -1))
    dns = all(c.close[i] < c.open[i] and c.close[i] < c.close[i - 1] for i in (-3, -2, -1))
    strong = all(_body(c, i) / _range(c, i) > 0.5 for i in (-3, -2, -1))
    if ups and strong:
        return ctx.signal(w, sym, L, 0.4, setup="THREE_WHITE_SOLDIERS", expected_move=_move(ctx, sym, 1.5, L),
                          evidence=["three_strong_up_closes"], invalidations=["pattern_low"])
    if dns and strong:
        return ctx.signal(w, sym, S, 0.4, setup="THREE_BLACK_CROWS", expected_move=_move(ctx, sym, 1.5, S),
                          evidence=["three_strong_down_closes"], invalidations=["pattern_high"])
    return None


def doji_reversal(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 15:
        return None
    if _body(c, -2) / _range(c, -2) > 0.1:
        return None
    pt = _prior_trend(c)
    if pt < 0 and c.close[-1] > c.high[-2]:
        return ctx.signal(w, sym, L, 0.35, setup="DOJI_REVERSAL", expected_move=_move(ctx, sym, 1, L),
                          evidence=["doji", "confirmation_up"], invalidations=["doji_low"])
    if pt > 0 and c.close[-1] < c.low[-2]:
        return ctx.signal(w, sym, S, 0.35, setup="DOJI_REVERSAL", expected_move=_move(ctx, sym, 1, S),
                          evidence=["doji", "confirmation_down"], invalidations=["doji_high"])
    return None


def star_pattern(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 5:
        return None
    small = _body(c, -2) < 0.3 * _body(c, -3)
    if c.close[-3] < c.open[-3] and small and c.close[-1] > c.open[-1] and \
            c.close[-1] > (c.open[-3] + c.close[-3]) / 2:
        return ctx.signal(w, sym, L, 0.45, setup="MORNING_STAR", expected_move=_move(ctx, sym, 1.5, L),
                          evidence=["morning_star"], invalidations=["pattern_low"])
    if c.close[-3] > c.open[-3] and small and c.close[-1] < c.open[-1] and \
            c.close[-1] < (c.open[-3] + c.close[-3]) / 2:
        return ctx.signal(w, sym, S, 0.45, setup="EVENING_STAR", expected_move=_move(ctx, sym, 1.5, S),
                          evidence=["evening_star"], invalidations=["pattern_high"])
    return None


def double_top_bottom(w, ctx, sym):
    n = w.params["n"]
    c = ctx.view.candles(sym)
    if len(c) < n:
        return None
    a = K.last(K.atr(ctx, sym)) or 1
    seg = c.tail(n)
    half = n // 2
    l1, l2 = seg.low[:half].min(), seg.low[half:-1].min()
    h1, h2 = seg.high[:half].max(), seg.high[half:-1].max()
    neck_hi = seg.high[np.argmin(seg.low[:half]):half + int(np.argmin(seg.low[half:-1]))].max() \
        if abs(l1 - l2) < a else None
    if neck_hi and c.close[-1] > neck_hi:
        return ctx.signal(w, sym, L, 0.5, setup="DOUBLE_BOTTOM", expected_move=(neck_hi - min(l1, l2)) / c.close[-1],
                          evidence=["double_bottom", "neckline_break"], invalidations=["back_below_neckline"])
    if abs(h1 - h2) < a:
        lo_seg = seg.low[np.argmax(seg.high[:half]):half + int(np.argmax(seg.high[half:-1]))]
        neck_lo = lo_seg.min() if len(lo_seg) else None
        if neck_lo and c.close[-1] < neck_lo:
            return ctx.signal(w, sym, S, 0.5, setup="DOUBLE_TOP", expected_move=-(max(h1, h2) - neck_lo) / c.close[-1],
                              evidence=["double_top", "neckline_break"], invalidations=["back_above_neckline"])
    return None


def flag(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 40:
        return None
    pole = c.close[-15] / c.close[-30] - 1
    cons = c.close[-15:-1]
    rng = (cons.max() - cons.min()) / cons.mean()
    a = K.atr_pct(ctx, sym)
    if pole > 6 * a and rng < 3 * a and c.close[-1] > cons.max():
        return ctx.signal(w, sym, L, 0.5, setup="BULL_FLAG", expected_move=min(pole, 0.2),
                          evidence=["pole", "tight_flag", "break_up"], invalidations=["flag_low"])
    if pole < -6 * a and rng < 3 * a and c.close[-1] < cons.min():
        return ctx.signal(w, sym, S, 0.5, setup="BEAR_FLAG", expected_move=max(pole, -0.2),
                          evidence=["pole", "tight_flag", "break_down"], invalidations=["flag_high"])
    return None


def specs() -> list[dict]:
    T = Family.TECHNICAL
    out = []
    for f, s in [(9, 21), (12, 26), (20, 50), (50, 100), (50, 200), (21, 55)]:
        out.append(spec(T, f"ema_cross_{f}_{s}", ema_cross, params={"fast": f, "slow": s}, min_bars=s + 5))
    for n in (20, 50, 100):
        out.append(spec(T, f"vwap_trend_{n}", vwap_trend, params={"n": n}, min_bars=n + 10))
    for n, thr in [(14, 25), (14, 20), (28, 25)]:
        out.append(spec(T, f"adx_trend_{n}_{thr}", adx_trend, params={"n": n, "threshold": thr}, min_bars=3 * n))
    for n in (20, 55, 100, 10):
        out.append(spec(T, f"donchian_breakout_{n}", donchian_breakout, params={"n": n}, min_bars=n + 5))
    for left in (2, 3, 5):
        out.append(spec(T, f"structure_hh_hl_{left}", structure_trend, params={"left": left}))
    for f, s, g in [(12, 26, 9), (5, 35, 5), (8, 17, 9)]:
        out.append(spec(T, f"macd_{f}_{s}_{g}", macd_trend, params={"fast": f, "slow": s, "signal": g}, min_bars=s + g + 5))
    # mean reversion (16)
    for n, k in [(20, 2.0), (20, 2.5), (50, 2.0)]:
        out.append(spec(T, f"bollinger_{n}_{k}", bollinger_revert, params={"n": n, "k": k}, min_bars=n + 5))
    for n, lo, hi in [(7, 20, 80), (14, 30, 70), (21, 35, 65)]:
        out.append(spec(T, f"rsi_extreme_{n}", rsi_extreme, params={"n": n, "lo": lo, "hi": hi}))
    for n, z in [(20, 2.0), (50, 2.0), (100, 2.5)]:
        out.append(spec(T, f"vwap_deviation_{n}", vwap_deviation, params={"n": n, "z": z}, min_bars=n + 5))
    for n, z in [(20, 2.0), (50, 2.2), (100, 2.5)]:
        out.append(spec(T, f"zscore_{n}", zscore_revert, params={"n": n, "z": z}, min_bars=n + 5))
    for n, m in [(20, 1.5), (20, 2.0)]:
        out.append(spec(T, f"keltner_{n}_{m}", keltner_revert, params={"n": n, "mult": m}))
    for n in (14, 21):
        out.append(spec(T, f"stoch_{n}", stoch_revert, params={"n": n}))
    # market structure (14)
    for left in (3, 5, 8):
        out.append(spec(T, f"sr_bounce_{left}", sr_bounce, params={"left": left}))
    for n in (20, 50, 100):
        out.append(spec(T, f"failed_breakout_{n}", failed_breakout, params={"n": n}, min_bars=n + 5))
    for n in (10, 20, 50):
        out.append(spec(T, f"sweep_reclaim_{n}", sweep_reclaim, params={"n": n}, min_bars=n + 5))
    for f, s in [(20, 50), (9, 21), (50, 200)]:
        out.append(spec(T, f"trend_pullback_{f}_{s}", trend_pullback, params={"fast": f, "slow": s}, min_bars=s + 10))
    for t in (20, 50):
        out.append(spec(T, f"inside_bar_{t}", inside_bar, params={"trend": t}, min_bars=t + 5))
    # candlestick / chart patterns (8)
    out += [spec(T, "engulfing", engulfing), spec(T, "hammer_star", hammer_star),
            spec(T, "three_soldiers_crows", three_soldiers), spec(T, "doji_reversal", doji_reversal),
            spec(T, "morning_evening_star", star_pattern),
            spec(T, "double_top_bottom_40", double_top_bottom, params={"n": 40}),
            spec(T, "double_top_bottom_80", double_top_bottom, params={"n": 80}, min_bars=85),
            spec(T, "flag", flag)]
    return out

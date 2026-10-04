"""Momentum (13) and volatility (12) workers."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.indicators import ta
from lunatrade.workers import common as K
from lunatrade.workers.base import clamp, spec

L, S, N = D.LONG, D.SHORT, D.NEUTRAL


def volume_expansion(w, ctx, sym):
    vz = K.last(K.volume_z(ctx, sym, w.params["n"]))
    if vz < w.params["z"]:
        return None
    c = ctx.view.candles(sym)
    d = L if c.close[-1] > c.open[-1] else S
    return ctx.signal(w, sym, d, clamp((vz - w.params["z"]) * 0.25 + 0.3), setup="VOLUME_EXPANSION",
                      expected_move=d.sign * K.atr_pct(ctx, sym) * 1.5, time_horizon="4h",
                      evidence=[f"volume_z={vz:.1f}"], invalidations=["volume_collapse"])


def acceleration(w, ctx, sym):
    n = w.params["n"]
    c = ctx.view.candles(sym).close
    if len(c) < 2 * n + 2:
        return None
    r1, r2 = c[-1] / c[-n - 1] - 1, c[-n - 1] / c[-2 * n - 1] - 1
    acc = r1 - r2
    a = K.atr_pct(ctx, sym) or 1e-6
    if abs(acc) < a or np.sign(r1) != np.sign(acc):
        return None
    d = L if acc > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(acc) / a * 0.12 + 0.2), setup="PRICE_ACCELERATION",
                      expected_move=d.sign * abs(r1) * 0.5, evidence=[f"accel={acc:.4f}"], invalidations=["deceleration"])


def roc_momentum(w, ctx, sym):
    n = w.params["n"]
    r = K.last(ctx.view.ind(sym, "roc", lambda c, n: ta.roc(c.close, n), n))
    vol = K.last(K.rvol(ctx, sym, 30)) * np.sqrt(n) or 1e-6
    z = r / vol
    if abs(z) < 1:
        return None
    d = L if z > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(z) * 0.2), setup="ROC_MOMENTUM", expected_move=d.sign * abs(r) * 0.4,
                      evidence=[f"roc{n}_z={z:.2f}"], invalidations=["momentum_reversal"])


def breakout_confirmation(w, ctx, sym):
    n = w.params["n"]
    c = ctx.view.candles(sym)
    if len(c) < n + 3:
        return None
    hi, lo = c.high[-n - 2:-2].max(), c.low[-n - 2:-2].min()
    vz = K.last(K.volume_z(ctx, sym))
    if c.close[-2] > hi and c.close[-1] > c.close[-2] and vz > 0:
        return ctx.signal(w, sym, L, 0.55, setup="BREAKOUT_CONFIRMED", expected_move=K.atr_pct(ctx, sym) * 3,
                          evidence=["breakout", "follow_through", "volume"], invalidations=["back_into_range"])
    if c.close[-2] < lo and c.close[-1] < c.close[-2] and vz > 0:
        return ctx.signal(w, sym, S, 0.55, setup="BREAKDOWN_CONFIRMED", expected_move=-K.atr_pct(ctx, sym) * 3,
                          evidence=["breakdown", "follow_through", "volume"], invalidations=["back_into_range"])
    return None


def obv_trend(w, ctx, sym):
    n = w.params["n"]
    o = ctx.view.ind(sym, "obv", lambda c: ta.obv(c.close, c.volume))
    c = ctx.view.candles(sym).close
    if len(o) < n + 1:
        return None
    o_slope = (o[-1] - o[-n]) / (np.abs(o[-n:]).mean() + 1e-9)
    p_slope = c[-1] / c[-n] - 1
    if np.sign(o_slope) == np.sign(p_slope) and abs(o_slope) > 0.05:
        d = L if o_slope > 0 else S
        return ctx.signal(w, sym, d, clamp(abs(o_slope)), setup="OBV_CONFIRMS", expected_move=d.sign * abs(p_slope) * .3,
                          evidence=["obv_confirms_price"], invalidations=["obv_divergence"])
    if np.sign(o_slope) != np.sign(p_slope) and abs(o_slope) > 0.1:
        d = L if o_slope > 0 else S
        return ctx.signal(w, sym, d, 0.35, setup="OBV_DIVERGENCE", expected_move=d.sign * K.atr_pct(ctx, sym),
                          evidence=["obv_divergence"], invalidations=["price_confirms_trend"])
    return None


# ------------------------------------------------------------------------------- volatility

def vol_expansion(w, ctx, sym):
    a_fast = K.last(K.atr(ctx, sym, w.params["fast"]))
    a_slow = K.last(K.atr(ctx, sym, w.params["slow"]))
    if not a_slow:
        return None
    ratio = a_fast / a_slow
    if ratio < w.params["ratio"]:
        return None
    c = ctx.view.candles(sym)
    move = c.close[-1] / c.close[-w.params["fast"]] - 1
    d = L if move > 0 else S
    return ctx.signal(w, sym, d, clamp((ratio - 1) * 0.6), setup="VOLATILITY_EXPANSION",
                      expected_move=d.sign * K.atr_pct(ctx, sym) * 2,
                      evidence=[f"atr_ratio={ratio:.2f}"], invalidations=["vol_mean_revert"],
                      context={"atr_ratio": ratio})


def squeeze(w, ctx, sym):
    n = w.params["n"]
    lo, mid, hi = K.boll(ctx, sym, 20, 2.0)
    width = (hi - lo) / mid
    pr = ctx.view.ind(sym, "bbw_pr", lambda c, n: ta.percentile_rank(np.nan_to_num((hi - lo) / mid), n), n)
    p = K.last(pr, 0.5)
    if p > w.params["pct"]:
        return None
    c = ctx.view.candles(sym)
    e = K.last(K.ema(ctx, sym, 50))
    d = L if c.close[-1] > e else S
    return ctx.signal(w, sym, d, 0.3, setup="VOLATILITY_COMPRESSION", expected_move=d.sign * K.atr_pct(ctx, sym) * 3,
                      evidence=[f"bb_width_pctile={p:.2f}"], invalidations=["breakout_opposite_side"],
                      context={"bb_width": float(width[-1]) if len(width) else 0.0})


def atr_regime(w, ctx, sym):
    ap = ctx.view.ind(sym, "atrp", lambda c: ta.atr(c.high, c.low, c.close, 14) / c.close)
    pr = ctx.view.ind(sym, "atrp_pr", lambda c, n: ta.percentile_rank(np.nan_to_num(ap), n), w.params["n"])
    p = K.last(pr, 0.5)
    return ctx.signal(w, sym, N, 0.0, setup="ATR_REGIME", context={"atr_pct": K.last(ap), "atr_percentile": p,
                                                                   "high_vol": p > 0.8, "low_vol": p < 0.2})


def realized_vs_long(w, ctx, sym):
    short = K.last(K.rvol(ctx, sym, w.params["short"]))
    long = K.last(K.rvol(ctx, sym, w.params["long"]))
    if not long:
        return None
    ratio = short / long
    c = ctx.view.candles(sym).close
    trend = c[-1] / c[-w.params["short"]] - 1
    if ratio > 1.5 and trend < 0:
        return ctx.signal(w, sym, S, clamp((ratio - 1.5) * 0.5 + 0.25), setup="VOL_SPIKE_DOWN",
                          expected_move=trend * 0.5, evidence=[f"rv_ratio={ratio:.2f}"], context={"rv_ratio": ratio})
    if ratio < 0.6:
        return ctx.signal(w, sym, N, 0.0, setup="VOL_QUIET", context={"rv_ratio": ratio})
    return ctx.signal(w, sym, N, 0.0, setup="VOL_NORMAL", context={"rv_ratio": ratio})


def parkinson(w, ctx, sym):
    n = w.params["n"]
    pk = ctx.view.ind(sym, "park", lambda c, n: ta.parkinson_vol(c.high, c.low, n), n)
    cc = K.last(K.rvol(ctx, sym, n))
    p = K.last(pk)
    if not cc or not p:
        return None
    # intrabar range much larger than close-to-close vol -> indecision/wicks (reversal risk)
    ratio = p / cc
    return ctx.signal(w, sym, N, 0.0, setup="RANGE_VOL", context={"parkinson_vol": p, "range_to_close_vol": ratio})


def specs() -> list[dict]:
    M, V = Family.MOMENTUM, Family.VOLATILITY
    out = []
    for n, z in [(20, 1.5), (50, 2.0), (100, 2.5)]:
        out.append(spec(M, f"volume_expansion_{n}", volume_expansion, params={"n": n, "z": z}, min_bars=n + 5))
    for n in (4, 12, 24):
        out.append(spec(M, f"acceleration_{n}", acceleration, params={"n": n}))
    for n in (6, 24, 96):
        out.append(spec(M, f"roc_{n}", roc_momentum, params={"n": n}, min_bars=max(60, n + 35)))
    for n in (20, 50):
        out.append(spec(M, f"breakout_confirmation_{n}", breakout_confirmation, params={"n": n}, min_bars=n + 5))
    for n in (20, 50):
        out.append(spec(M, f"obv_trend_{n}", obv_trend, params={"n": n}, min_bars=n + 5))
    for f, s, r in [(5, 50, 1.6), (10, 100, 1.4), (14, 200, 1.3)]:
        out.append(spec(V, f"vol_expansion_{f}_{s}", vol_expansion, params={"fast": f, "slow": s, "ratio": r},
                        min_bars=s + 5))
    for n, p in [(100, 0.1), (200, 0.15), (50, 0.1)]:
        out.append(spec(V, f"squeeze_{n}", squeeze, params={"n": n, "pct": p}, min_bars=n + 25))
    for n in (100, 300):
        out.append(spec(V, f"atr_regime_{n}", atr_regime, params={"n": n}, min_bars=min(n + 20, 120)))
    for s, l in [(12, 96), (24, 240)]:
        out.append(spec(V, f"realized_vs_long_{s}_{l}", realized_vs_long, params={"short": s, "long": l}, min_bars=l + 5))
    for n in (20, 60):
        out.append(spec(V, f"parkinson_{n}", parkinson, params={"n": n}, min_bars=n + 5))
    return out

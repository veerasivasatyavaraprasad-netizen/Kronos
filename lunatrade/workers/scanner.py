"""Market scanner network (35 workers): cross-sectional scans, relative strength, derivatives positioning."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import MARKET_WIDE
from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.indicators import ta
from lunatrade.intel.assets import base_of
from lunatrade.workers import common as K
from lunatrade.workers.base import clamp, spec

L, S, N = D.LONG, D.SHORT, D.NEUTRAL

SECTORS = {
    "L1": {"BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "NEAR", "APT", "SUI", "TON", "TRX", "ATOM", "BNB"},
    "DEFI": {"UNI", "AAVE", "MKR", "LINK", "INJ", "LDO", "CRV", "SNX", "COMP", "DYDX", "JUP", "PENDLE"},
    "MEME": {"DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "TRUMP"},
    "L2": {"ARB", "OP", "MATIC", "POL", "STRK", "IMX", "MNT"},
    "AI": {"FET", "RENDER", "TAO", "WLD", "AGIX", "OCEAN", "ARKM"},
    "PAYMENTS": {"XRP", "XLM", "LTC", "BCH"},
}


def _ret(ctx, sym, n):
    c = ctx.view.candles(sym).close
    return c[-1] / c[-n - 1] - 1 if len(c) > n else None


def abnormal_volume(w, ctx, sym):
    n = w.params["n"]
    vz = K.last(K.volume_z(ctx, sym, n))
    if abs(vz) < 2.5:
        return None
    c = ctx.view.candles(sym)
    d = L if c.close[-1] > c.open[-1] else S
    return ctx.signal(w, sym, d, clamp(0.2 + (vz - 2.5) * 0.1), setup="ABNORMAL_VOLUME",
                      evidence=[f"volume_z{n}={vz:.1f}"], context={"volume_z": vz})


def relative_strength(w, ctx, sym):
    n = w.params["n"]
    m = ctx.market_symbol
    if sym == m:
        return None
    r, rm = _ret(ctx, sym, n), _ret(ctx, m, n)
    if r is None or rm is None:
        return None
    rel = r - rm
    vol = (K.last(K.rvol(ctx, sym, 30)) or 0.01) * np.sqrt(n)
    z = rel / vol
    if abs(z) < 0.75:
        return None
    d = L if z > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(z) * 0.2), setup="RELATIVE_STRENGTH", expected_move=d.sign * abs(rel) * 0.3,
                      evidence=[f"rs_vs_{base_of(m)}_{n}={rel:+.3f}"], context={"relative_return": rel})


def cross_sectional_rank(w, ctx, sym):
    """Rank of this symbol's n-bar return across the universe (momentum or reversal)."""
    n = w.params["n"]
    rets = {s: _ret(ctx, s, n) for s in ctx.symbols}
    rets = {s: r for s, r in rets.items() if r is not None}
    if sym not in rets or len(rets) < 4:
        return None
    ranked = sorted(rets, key=rets.get)
    pct = ranked.index(sym) / (len(ranked) - 1)
    mode = w.params["mode"]
    if mode == "momentum":
        d = L if pct >= 0.8 else S if pct <= 0.2 else None
    else:
        d = S if pct >= 0.9 else L if pct <= 0.1 else None
    if d is None:
        return None
    return ctx.signal(w, sym, d, 0.3, setup=f"XS_{mode.upper()}", context={"xs_percentile": pct},
                      evidence=[f"rank_pct={pct:.2f}"])


def new_highs_lows(w, ctx, sym):
    n = w.params["n"]
    c = ctx.view.candles(sym)
    if len(c) < n + 1:
        return None
    if c.high[-1] >= c.high[-n:].max():
        return ctx.signal(w, sym, L, 0.3, setup="NEW_HIGH", evidence=[f"{n}_bar_high"])
    if c.low[-1] <= c.low[-n:].min():
        return ctx.signal(w, sym, S, 0.3, setup="NEW_LOW", evidence=[f"{n}_bar_low"])
    return None


def correlation_monitor(w, ctx, sym):
    n = w.params["n"]
    m = ctx.market_symbol
    if sym == m:
        return None
    r1, r2 = K.rets(ctx, sym), K.rets(ctx, m)
    if r1 is None or r2 is None:
        return None
    rho = ta.corr(r1[-n:], r2[-n:])
    return ctx.signal(w, sym, N, 0.0, setup="CORRELATION", context={f"corr_to_market_{n}": rho, "corr_to_market": rho})


def market_dominance(w, ctx, sym):
    """BTC-dominance proxy: BTC vs. equal-weight alt basket. Rising dominance = risk-off for alts."""
    if sym is not None:
        return None
    n = w.params["n"]
    m = ctx.market_symbol
    rm = _ret(ctx, m, n)
    alts = [r for s in ctx.symbols if s != m and (r := _ret(ctx, s, n)) is not None]
    if rm is None or len(alts) < 3:
        return None
    spread = rm - float(np.mean(alts))
    out = [ctx.signal(w, MARKET_WIDE, N, 0.0, setup="DOMINANCE", context={"dominance_change": spread})]
    if abs(spread) > 0.01:
        for s in ctx.symbols:
            if s == m:
                continue
            d = S if spread > 0 else L
            out.append(ctx.signal(w, s, d, clamp(abs(spread) * 8), setup="DOMINANCE_ROTATION",
                                  evidence=[f"btc_minus_alts_{n}={spread:+.3f}"]))
    return out


def sector_rotation(w, ctx, sym):
    if sym is not None:
        return None
    n = w.params["n"]
    sector_rets = {}
    for name, members in SECTORS.items():
        rs = [r for s in ctx.symbols if base_of(s) in members and (r := _ret(ctx, s, n)) is not None]
        if rs:
            sector_rets[name] = float(np.mean(rs))
    if len(sector_rets) < 2:
        return None
    avg = float(np.mean(list(sector_rets.values())))
    out = [ctx.signal(w, MARKET_WIDE, N, 0.0, setup="SECTORS", context={"sector_returns": sector_rets})]
    for s in ctx.symbols:
        sec = next((k for k, v in SECTORS.items() if base_of(s) in v), None)
        if sec in sector_rets and abs(sector_rets[sec] - avg) > 0.01:
            d = L if sector_rets[sec] > avg else S
            out.append(ctx.signal(w, s, d, clamp(abs(sector_rets[sec] - avg) * 6), setup="SECTOR_ROTATION",
                                  evidence=[f"sector_{sec}_vs_avg={sector_rets[sec] - avg:+.3f}"]))
    return out


def spread_liquidity(w, ctx, sym):
    sp = ctx.view.spread_bps(sym)
    c = ctx.view.candles(sym)
    qv = float(c.quote_volume[-96:].sum()) if len(c) else 0.0
    return ctx.signal(w, sym, N, 0.0, setup="LIQUIDITY", context={"spread_bps": sp, "quote_volume_recent": qv})


def vol_percentile_scan(w, ctx, sym):
    rv = K.rvol(ctx, sym, 24)
    pr = ta.percentile_rank(np.nan_to_num(rv), min(w.params["n"], len(rv)))
    p = K.last(pr, 0.5)
    return ctx.signal(w, sym, N, 0.0, setup="VOL_SCAN", context={"vol_percentile": p})


def trend_strength_scan(w, ctx, sym):
    e = K.last(K.er(ctx, sym, w.params["n"]))
    c = ctx.view.candles(sym).close
    n = w.params["n"]
    if e < 0.45 or len(c) <= n:
        return None
    d = L if c[-1] > c[-n - 1] else S
    return ctx.signal(w, sym, d, clamp(e - 0.2), setup="EFFICIENT_TREND", evidence=[f"efficiency={e:.2f}"],
                      context={"efficiency_ratio": e})


def gap_scan(w, ctx, sym):
    c = ctx.view.candles(sym)
    if len(c) < 3:
        return None
    gap = c.open[-1] / c.close[-2] - 1
    a = K.atr_pct(ctx, sym) or 1e-6
    if abs(gap) < a:
        return None
    filled = (gap > 0 and c.low[-1] <= c.close[-2]) or (gap < 0 and c.high[-1] >= c.close[-2])
    d = (L if gap > 0 else S) if not filled else (S if gap > 0 else L)
    return ctx.signal(w, sym, d, 0.3, setup="GAP_FILLED" if filled else "GAP_HOLD", evidence=[f"gap={gap:+.4f}"])


# ------------------------------------------------------------------------- derivatives (read-only)

def _deriv(ctx, sym):
    return ctx.view.derivatives_history(sym, 100)


def funding_extreme(w, ctx, sym):
    h = _deriv(ctx, sym)
    if not h or "funding_rate" not in h[-1]:
        return None
    f = h[-1]["funding_rate"]
    thr = w.params["threshold"]
    if abs(f) < thr:
        return ctx.signal(w, sym, N, 0.0, setup="FUNDING", context={"funding_rate": f})
    # crowded longs pay high funding -> contrarian short tilt (and vice versa)
    d = S if f > 0 else L
    return ctx.signal(w, sym, d, clamp(abs(f) / thr * 0.2), setup="FUNDING_CROWDING",
                      evidence=[f"funding={f * 100:.3f}%"], context={"funding_rate": f})


def oi_change(w, ctx, sym):
    h = [x for x in _deriv(ctx, sym) if "open_interest" in x]
    n = w.params["n"]
    if len(h) < n + 1:
        return None
    oi_chg = h[-1]["open_interest"] / h[-n - 1]["open_interest"] - 1 if h[-n - 1]["open_interest"] else 0
    pr = _ret(ctx, sym, n) or 0.0
    ctxd = {"oi_change": oi_chg, "price_change": pr}
    if abs(oi_chg) < 0.03:
        return ctx.signal(w, sym, N, 0.0, setup="OI", context=ctxd)
    if oi_chg > 0 and pr > 0:
        d, setup = L, "OI_UP_PRICE_UP"           # new longs, trend supported
    elif oi_chg > 0 and pr < 0:
        d, setup = S, "OI_UP_PRICE_DOWN"         # new shorts
    elif oi_chg < 0 and pr > 0:
        d, setup = L, "SHORT_COVERING"
    else:
        d, setup = S, "LONG_LIQUIDATION"
    return ctx.signal(w, sym, d, clamp(abs(oi_chg) * 4), setup=setup, context=ctxd,
                      evidence=[f"oi_change={oi_chg:+.2%}"])


def long_short(w, ctx, sym):
    h = [x for x in _deriv(ctx, sym) if "long_short_ratio" in x]
    if not h:
        return None
    r = h[-1]["long_short_ratio"]
    if r > w.params["hi"]:
        return ctx.signal(w, sym, S, clamp((r - w.params["hi"]) * 0.5 + 0.2), setup="CROWDED_LONGS",
                          context={"long_short_ratio": r})
    if r < w.params["lo"]:
        return ctx.signal(w, sym, L, clamp((w.params["lo"] - r) * 0.8 + 0.2), setup="CROWDED_SHORTS",
                          context={"long_short_ratio": r})
    return ctx.signal(w, sym, N, 0.0, setup="POSITIONING", context={"long_short_ratio": r})


def liquidation_proxy(w, ctx, sym):
    """Price shock + OI drop = forced liquidations (cascade risk; often marks local extremes)."""
    h = [x for x in _deriv(ctx, sym) if "open_interest" in x]
    c = ctx.view.candles(sym)
    if len(h) < 3 or len(c) < 3:
        return None
    oi = h[-1]["open_interest"] / h[-3]["open_interest"] - 1 if h[-3]["open_interest"] else 0
    move = c.close[-1] / c.close[-3] - 1
    a = K.atr_pct(ctx, sym) or 1e-6
    if oi < -0.04 and abs(move) > 2 * a:
        d = L if move < 0 else S   # after a long-liquidation flush, mean reversion is likely
        return ctx.signal(w, sym, d, 0.35, setup="LIQUIDATION_FLUSH", context={"oi_drop": oi, "move": move},
                          evidence=["oi_flush", "price_shock"])
    return None


def specs() -> list[dict]:
    F = Family.SCANNER
    out = []
    for n in (20, 50, 100):
        out.append(spec(F, f"abnormal_volume_{n}", abnormal_volume, params={"n": n}, min_bars=n + 5))
    for n in (6, 12, 24, 48, 96):
        out.append(spec(F, f"relative_strength_{n}", relative_strength, params={"n": n}, min_bars=max(60, n + 35)))
    for n, mode in [(24, "momentum"), (96, "momentum"), (4, "reversal"), (12, "reversal")]:
        out.append(spec(F, f"xs_rank_{mode}_{n}", cross_sectional_rank, params={"n": n, "mode": mode},
                        min_bars=max(30, n + 2)))
    for n in (20, 55, 100):
        out.append(spec(F, f"new_highs_lows_{n}", new_highs_lows, params={"n": n}, min_bars=n + 2))
    for n in (24, 96):
        out.append(spec(F, f"correlation_{n}", correlation_monitor, params={"n": n}, min_bars=n + 2))
    for n in (24, 96):
        out.append(spec(F, f"dominance_{n}", market_dominance, params={"n": n}, per_symbol=False))
    for n in (24, 96):
        out.append(spec(F, f"sector_rotation_{n}", sector_rotation, params={"n": n}, per_symbol=False))
    out.append(spec(F, "spread_liquidity", spread_liquidity, min_bars=1))
    for n in (200,):
        out.append(spec(F, f"vol_percentile_{n}", vol_percentile_scan, params={"n": n}, min_bars=60))
    for n in (20, 50):
        out.append(spec(F, f"trend_strength_{n}", trend_strength_scan, params={"n": n}, min_bars=n + 2))
    out.append(spec(F, "gap_scan", gap_scan, min_bars=20))
    for thr in (0.0003, 0.0006, 0.001):
        out.append(spec(F, f"funding_extreme_{thr}", funding_extreme, params={"threshold": thr}, min_bars=1,
                        requires=("derivatives",)))
    for n in (3, 12, 24):
        out.append(spec(F, f"oi_change_{n}", oi_change, params={"n": n}, min_bars=n + 2, requires=("derivatives",)))
    out.append(spec(F, "long_short_2", long_short, params={"hi": 2.0, "lo": 0.8}, min_bars=1, requires=("derivatives",)))
    out.append(spec(F, "long_short_3", long_short, params={"hi": 3.0, "lo": 0.6}, min_bars=1, requires=("derivatives",)))
    out.append(spec(F, "liquidation_proxy", liquidation_proxy, min_bars=5, requires=("derivatives",)))
    return out

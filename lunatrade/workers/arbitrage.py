"""Arbitrage / relative-value workers (15). Signals only - LunaTrade does not auto-execute arbitrage."""
from __future__ import annotations

from lunatrade.core.types import MARKET_WIDE
from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.intel.assets import base_of
from lunatrade.workers.base import clamp, spec

L, S, N = D.LONG, D.SHORT, D.NEUTRAL


def _arb_avail(w, ctx):
    return (True, "") if ctx.feeds.hub.arb else (False, "no cross-exchange snapshot")


def cross_exchange(w, ctx, sym):
    snap = ctx.feeds.arb_latest()
    a, b = w.params["a"], w.params["b"]
    if not snap or sym not in snap.get(a, {}) or sym not in snap.get(b, {}):
        return None
    pa, pb = snap[a][sym], snap[b][sym]
    bps = (pa - pb) / pb * 1e4
    return ctx.signal(w, sym, N, 0.0, setup="CROSS_EXCHANGE_SPREAD",
                      context={"spread_bps": bps, "pair": f"{a}-{b}", "opportunity": abs(bps) > 25})


def coinbase_premium(w, ctx, sym):
    if base_of(sym) != w.params["asset"]:
        return None
    hist = [s for ts, s in ctx.feeds.hub.arb if ts <= ctx.feeds.as_of][-30:]
    prem = [(s["coinbase"][sym] / s["binance"][sym] - 1) * 1e4 for s in hist
            if sym in s.get("coinbase", {}) and sym in s.get("binance", {})]
    if len(prem) < 3:
        return None
    avg = sum(prem) / len(prem)
    if abs(avg) < 5:
        return None
    # persistent US-exchange premium = US spot demand
    return ctx.signal(w, sym, D.from_score(avg), clamp(abs(avg) / 40), setup="COINBASE_PREMIUM", time_horizon="1d",
                      context={"coinbase_premium_bps": avg})


def triangular(w, ctx, sym):
    if sym is not None:
        return None
    cross = w.params["cross"]          # e.g. ETHBTC
    base = cross[:-3]
    snap = ctx.feeds.arb_latest()
    bn = (snap or {}).get("binance", {})
    leg_a, leg_b = f"{base}USDT", "BTCUSDT"
    if cross not in bn or leg_a not in bn or leg_b not in bn:
        return None
    implied = bn[leg_a] / bn[leg_b]
    gap = (bn[cross] / implied - 1) * 1e4
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="TRIANGULAR", context={"triangle": cross, "gap_bps": gap,
                                                                          "opportunity": abs(gap) > 15})


def basis(w, ctx, sym):
    d = ctx.view.derivatives(sym)
    px = ctx.view.price(sym)
    if not d or not d.get("mark_price") or not px:
        return None
    b = (d["mark_price"] / px - 1) * 1e4
    thr = w.params["bps"]
    if abs(b) < thr:
        return ctx.signal(w, sym, N, 0.0, setup="BASIS", context={"basis_bps": b})
    # rich perp premium = leveraged longs chasing -> fade; deep discount = forced selling -> contrarian long
    return ctx.signal(w, sym, S if b > 0 else L, clamp(abs(b) / thr * 0.15), setup="BASIS_EXTREME",
                      context={"basis_bps": b})


def depeg(w, ctx, sym):
    if sym is not None:
        return None
    pair = w.params["pair"]
    snap = ctx.feeds.arb_latest()
    p = (snap or {}).get("binance", {}).get(pair)
    if not p:
        return None
    dev = (p - 1) * 1e4
    risk = abs(dev) > 50
    return ctx.signal(w, MARKET_WIDE, S if risk else N, 0.6 if risk else 0.0, setup="STABLECOIN_PEG",
                      context={"pair": pair, "deviation_bps": dev, "depeg_risk": risk})


def funding_carry(w, ctx, sym):
    d = ctx.view.derivatives(sym)
    if not d or "funding_rate" not in d:
        return None
    annual = d["funding_rate"] * 3 * 365
    return ctx.signal(w, sym, N, 0.0, setup="FUNDING_CARRY",
                      context={"funding_annualized": annual, "carry_opportunity": abs(annual) > w.params["annual"]})


def specs() -> list[dict]:
    A = Family.ARBITRAGE
    out = [spec(A, f"cross_{a}_{b}", cross_exchange, params={"a": a, "b": b}, min_bars=1, availability=_arb_avail)
           for a, b in [("binance", "coinbase"), ("binance", "kraken"), ("coinbase", "kraken")]]
    out += [spec(A, f"coinbase_premium_{x.lower()}", coinbase_premium, params={"asset": x}, min_bars=1,
                 availability=_arb_avail) for x in ("BTC", "ETH")]
    out += [spec(A, f"triangular_{c.lower()}", triangular, params={"cross": c}, per_symbol=False,
                 availability=_arb_avail) for c in ("ETHBTC", "BNBBTC", "SOLBTC")]
    out += [spec(A, f"basis_{b}", basis, params={"bps": b}, min_bars=1, requires=("derivatives",))
            for b in (10, 25, 50)]
    out += [spec(A, f"depeg_{p.lower()}", depeg, params={"pair": p}, per_symbol=False, availability=_arb_avail)
            for p in ("USDCUSDT", "FDUSDUSDT")]
    out += [spec(A, f"funding_carry_{int(a * 100)}", funding_carry, params={"annual": a}, min_bars=1,
                 requires=("derivatives",)) for a in (0.2, 0.5)]
    return out

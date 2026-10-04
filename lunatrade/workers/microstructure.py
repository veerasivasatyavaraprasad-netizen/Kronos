"""Order-book / microstructure workers (20). Book data comes from WebSocket depth; taker flow also from klines."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.workers import common as K
from lunatrade.workers.base import clamp, spec

L, S, N = D.LONG, D.SHORT, D.NEUTRAL


def _book(ctx, sym):
    b = ctx.view.book(sym)
    if not b or len(b["bids"]) == 0 or len(b["asks"]) == 0:
        return None
    return b


def book_imbalance(w, ctx, sym):
    b = _book(ctx, sym)
    if not b:
        return None
    k = w.params["levels"]
    bid = float((b["bids"][:k, 1] * b["bids"][:k, 0]).sum())
    ask = float((b["asks"][:k, 1] * b["asks"][:k, 0]).sum())
    if bid + ask == 0:
        return None
    imb = (bid - ask) / (bid + ask)
    if abs(imb) < 0.15:
        return ctx.signal(w, sym, N, 0.0, setup="BOOK_BALANCED", context={"imbalance": imb})
    d = L if imb > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(imb) * 0.8), setup="BOOK_IMBALANCE", time_horizon="15m",
                      evidence=[f"imbalance_{k}={imb:+.2f}"], context={"imbalance": imb})


def spread_z(w, ctx, sym):
    hist = ctx.view.ticker_history(sym, w.params["n"])
    if len(hist) < 10:
        return None
    sp = np.array([(t["ask"] - t["bid"]) / ((t["ask"] + t["bid"]) / 2) * 1e4 for t in hist if t["bid"] and t["ask"]])
    if len(sp) < 10 or sp.std() == 0:
        return ctx.signal(w, sym, N, 0.0, setup="SPREAD", context={"spread_bps": float(sp[-1]) if len(sp) else None})
    z = (sp[-1] - sp.mean()) / sp.std()
    return ctx.signal(w, sym, N, 0.0, setup="SPREAD_Z", context={"spread_bps": float(sp[-1]), "spread_z": float(z),
                                                                  "liquidity_stress": bool(z > 3)})


def depth_asymmetry(w, ctx, sym):
    b = _book(ctx, sym)
    if not b:
        return None
    mid = (b["bids"][0, 0] + b["asks"][0, 0]) / 2
    band = w.params["band_pct"]
    bid = b["bids"][b["bids"][:, 0] >= mid * (1 - band)]
    ask = b["asks"][b["asks"][:, 0] <= mid * (1 + band)]
    bv, av = float((bid[:, 0] * bid[:, 1]).sum()), float((ask[:, 0] * ask[:, 1]).sum())
    if bv + av == 0:
        return None
    asym = (bv - av) / (bv + av)
    if abs(asym) < 0.2:
        return None
    d = L if asym > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(asym) * 0.6), setup="DEPTH_ASYMMETRY", time_horizon="1h",
                      context={"depth_asymmetry": asym, "depth_usd": bv + av})


def taker_flow(w, ctx, sym):
    n = w.params["n"]
    c = ctx.view.candles(sym, n)
    vol = c.volume.sum()
    if len(c) < n or vol <= 0 or c.taker_buy_base.sum() == 0:
        return None
    buy_share = c.taker_buy_base.sum() / vol
    delta = buy_share - 0.5
    if abs(delta) < 0.03:
        return None
    d = L if delta > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(delta) * 6), setup="TAKER_FLOW", evidence=[f"taker_buy_share={buy_share:.2f}"],
                      context={"taker_buy_share": buy_share})


def large_trades(w, ctx, sym):
    tr = ctx.view.trades(sym, 2000)
    if len(tr) < 50:
        return None
    notional = np.array([t["p"] * t["q"] for t in tr])
    thr = np.percentile(notional, w.params["pct"])
    big = [t for t, n in zip(tr, notional) if n >= thr]
    buy = sum(t["p"] * t["q"] for t in big if not t["m"])     # m = buyer is maker -> aggressive sell
    sell = sum(t["p"] * t["q"] for t in big if t["m"])
    if buy + sell == 0:
        return None
    imb = (buy - sell) / (buy + sell)
    if abs(imb) < 0.2:
        return None
    d = L if imb > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(imb) * 0.6), setup="LARGE_TRADE_FLOW", time_horizon="1h",
                      context={"large_trade_imbalance": imb})


def microprice(w, ctx, sym):
    t = ctx.view.ticker(sym)
    if not t or not t.get("bid_qty") or not t.get("ask_qty"):
        return None
    bid, ask, bq, aq = t["bid"], t["ask"], t["bid_qty"], t["ask_qty"]
    mp = (bid * aq + ask * bq) / (bq + aq)
    mid = (bid + ask) / 2
    dev = (mp - mid) / (ask - bid) if ask > bid else 0
    if abs(dev) < w.params["thr"]:
        return None
    d = L if dev > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(dev)), setup="MICROPRICE", time_horizon="5m", context={"microprice_dev": dev})


def book_pressure_change(w, ctx, sym):
    hist = ctx.view.book_history(sym, w.params["n"])
    if len(hist) < 3:
        return None

    def imb(b):
        bv, av = float(b["bids"][:10, 1].sum()), float(b["asks"][:10, 1].sum())
        return (bv - av) / (bv + av) if bv + av else 0.0

    change = imb(hist[-1]) - imb(hist[0])
    if abs(change) < 0.2:
        return None
    d = L if change > 0 else S
    return ctx.signal(w, sym, d, clamp(abs(change) * 0.6), setup="BOOK_PRESSURE_SHIFT", time_horizon="15m",
                      context={"pressure_change": change})


def liquidity_vacuum(w, ctx, sym):
    b = _book(ctx, sym)
    if not b:
        return None
    mid = (b["bids"][0, 0] + b["asks"][0, 0]) / 2
    band = w.params["band_pct"]
    near = lambda side, up: side[(side[:, 0] <= mid * (1 + band)) if up else (side[:, 0] >= mid * (1 - band))]
    ask_depth = float(np.prod(near(b["asks"], True), axis=1).sum())
    bid_depth = float(np.prod(near(b["bids"], False), axis=1).sum())
    c = ctx.view.candles(sym, 96)
    typical = float(c.quote_volume.mean()) if len(c) else 0.0
    if not typical:
        return None
    thin_up, thin_down = ask_depth < 0.02 * typical, bid_depth < 0.02 * typical
    return ctx.signal(w, sym, N, 0.0, setup="LIQUIDITY_VACUUM",
                      context={"thin_asks": thin_up, "thin_bids": thin_down, "ask_depth": ask_depth,
                               "bid_depth": bid_depth, "slippage_risk": bool(thin_up or thin_down)})


def toxicity(w, ctx, sym):
    """VPIN-like: |buy - sell| volume share over volume buckets from klines."""
    n = w.params["n"]
    c = ctx.view.candles(sym, n)
    if len(c) < n or c.volume.sum() == 0 or c.taker_buy_base.sum() == 0:
        return None
    buy = c.taker_buy_base
    sell = c.volume - buy
    vpin = float(np.abs(buy - sell).sum() / c.volume.sum())
    return ctx.signal(w, sym, N, 0.0, setup="FLOW_TOXICITY", context={"vpin": vpin, "toxic_flow": vpin > 0.35})


def specs() -> list[dict]:
    M = Family.MICROSTRUCTURE
    out = []
    for k in (5, 10, 20):
        out.append(spec(M, f"book_imbalance_{k}", book_imbalance, params={"levels": k}, min_bars=1, requires=("book",)))
    for n in (50, 200):
        out.append(spec(M, f"spread_z_{n}", spread_z, params={"n": n}, min_bars=1, requires=("ticker",)))
    for b in (0.002, 0.005):
        out.append(spec(M, f"depth_asymmetry_{b}", depth_asymmetry, params={"band_pct": b}, min_bars=1,
                        requires=("book",)))
    for n in (4, 12, 48):
        out.append(spec(M, f"taker_flow_{n}", taker_flow, params={"n": n}, min_bars=n))
    for p in (95, 99):
        out.append(spec(M, f"large_trades_{p}", large_trades, params={"pct": p}, min_bars=1, requires=("trades",)))
    for t in (0.15, 0.3):
        out.append(spec(M, f"microprice_{t}", microprice, params={"thr": t}, min_bars=1, requires=("ticker",)))
    for n in (5, 20):
        out.append(spec(M, f"book_pressure_{n}", book_pressure_change, params={"n": n}, min_bars=1, requires=("book",)))
    for b in (0.003, 0.01):
        out.append(spec(M, f"liquidity_vacuum_{b}", liquidity_vacuum, params={"band_pct": b}, min_bars=1,
                        requires=("book",)))
    for n in (24, 96):
        out.append(spec(M, f"toxicity_{n}", toxicity, params={"n": n}, min_bars=n))
    return out

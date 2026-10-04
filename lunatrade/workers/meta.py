"""
Risk analysis (10), strategy evaluation (10), portfolio analysis (5) and adversarial / devil's-advocate (5)
workers. Strategy, portfolio and adversarial workers run in phase 2 and read the phase-1 signals.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from lunatrade.core.types import DIRECTIONAL_FAMILIES, MARKET_WIDE
from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.indicators import ta
from lunatrade.intel.assets import base_of
from lunatrade.workers import common as K
from lunatrade.workers.base import clamp, spec
from lunatrade.workers.scanner import SECTORS

N = D.NEUTRAL


def _risk(w, ctx, sym, **values):
    return ctx.signal(w, sym, N, 0.0, setup=w.name.upper(), context=values)


# ------------------------------------------------------------------------------- risk analysis

def var_hist(w, ctx, sym):
    r = K.rets(ctx, sym)[-w.params["n"]:]
    r = r[~np.isnan(r)]
    if len(r) < 50:
        return None
    var = -np.percentile(r, 100 - w.params["q"])
    tail = r[r <= -var]
    return _risk(w, ctx, sym, var=float(var), cvar=float(-tail.mean()) if len(tail) else float(var))


def realized_vol_ann(w, ctx, sym):
    bars_per_year = 365 * 86_400_000 / ctx.view.store.interval_ms
    v = K.last(K.rvol(ctx, sym, 96)) * np.sqrt(bars_per_year)
    return _risk(w, ctx, sym, annual_vol=float(v), high_vol=bool(v > 1.0))


def drawdown_speed(w, ctx, sym):
    c = ctx.view.candles(sym, 48).close
    if len(c) < 10:
        return None
    dd = c[-1] / c.max() - 1
    bars = len(c) - int(np.argmax(c))
    return _risk(w, ctx, sym, drawdown=float(dd), bars_since_high=bars, fast_drawdown=bool(dd < -0.08 and bars < 16))


def beta_to_market(w, ctx, sym):
    if sym == ctx.market_symbol:
        return _risk(w, ctx, sym, beta=1.0)
    a, b = K.rets(ctx, sym)[-200:], K.rets(ctx, ctx.market_symbol)[-200:]
    n = min(len(a), len(b))
    if n < 50:
        return None
    a, b = np.nan_to_num(a[-n:]), np.nan_to_num(b[-n:])
    var = b.var()
    return _risk(w, ctx, sym, beta=float(np.cov(a, b)[0, 1] / var) if var else 1.0)


def correlation_clusters(w, ctx, sym):
    syms = [s for s in ctx.symbols if len(ctx.view.candles(s)) > 100]
    if len(syms) < 2:
        return None
    rets = {s: np.nan_to_num(K.rets(ctx, s)[-100:]) for s in syms}
    thr = w.params["thr"]
    clusters, assigned = [], set()
    for s in syms:
        if s in assigned:
            continue
        group = [s] + [o for o in syms if o != s and o not in assigned and ta.corr(rets[s], rets[o]) >= thr]
        assigned.update(group)
        clusters.append(group)
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="CORRELATION_CLUSTERS",
                      context={"clusters": clusters, "threshold": thr})


def liquidity_risk(w, ctx, sym):
    c = ctx.view.candles(sym, 96)
    qv = float(c.quote_volume.mean()) if len(c) else 0.0
    sp = ctx.view.spread_bps(sym)
    est_slip = (sp or 2) / 2 + (10 if qv < 1e5 else 3 if qv < 1e6 else 1)
    return _risk(w, ctx, sym, avg_bar_quote_volume=qv, spread_bps=sp, expected_slippage_bps=float(est_slip))


def gap_risk(w, ctx, sym):
    r = K.rets(ctx, sym)[-500:]
    r = r[~np.isnan(r)]
    if len(r) < 50:
        return None
    return _risk(w, ctx, sym, max_bar_move=float(np.abs(r).max()), p99_bar_move=float(np.percentile(np.abs(r), 99)))


def stop_feasibility(w, ctx, sym):
    a = K.atr_pct(ctx, sym)
    stop = w.params["mult"] * a
    return _risk(w, ctx, sym, atr_stop_pct=float(stop), stop_too_wide=bool(stop > w.params["max_stop"]),
                 stop_too_tight=bool(stop < 0.002))


def tail_kurtosis(w, ctx, sym):
    r = K.rets(ctx, sym)[-500:]
    r = r[~np.isnan(r)]
    if len(r) < 100 or r.std() == 0:
        return None
    z = (r - r.mean()) / r.std()
    return _risk(w, ctx, sym, kurtosis=float((z ** 4).mean() - 3), skew=float((z ** 3).mean()))


# --------------------------------------------------------------------------- strategy evaluation

def strategy_eval(w, ctx, sym):
    group = w.params["group"]
    st = (ctx.performance.get("groups") or {}).get(group)
    if not st:
        return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="STRATEGY_EVAL", context={"group": group, "samples": 0})
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="STRATEGY_EVAL", context={
        "group": group, "samples": st.get("samples", 0), "hit_rate": st.get("hit_rate"),
        "recent_hit_rate": st.get("recent_hit_rate"), "expectancy": st.get("expectancy"),
        "weight": st.get("weight", 1.0), "degraded": bool(st.get("recent_hit_rate") is not None and
                                                           st.get("hit_rate") is not None and
                                                           st["recent_hit_rate"] < st["hit_rate"] - 0.1)})


def degradation_detector(w, ctx, sym):
    groups = ctx.performance.get("groups") or {}
    degraded = [g for g, st in groups.items() if st.get("recent_hit_rate") is not None and
                st.get("samples", 0) >= 20 and st["recent_hit_rate"] < 0.42]
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="DEGRADATION", context={"degraded_groups": degraded,
                                                                            "system_degraded": len(degraded) >= 4})


# --------------------------------------------------------------------------- portfolio analysis

def _positions(ctx):
    return ctx.portfolio.get("positions") or {}


def concentration(w, ctx, sym):
    pos = _positions(ctx)
    eq = ctx.portfolio.get("equity") or 0
    vals = [abs(p.get("notional", 0)) for p in pos.values()]
    hhi = sum((v / sum(vals)) ** 2 for v in vals) if vals and sum(vals) else 0.0
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="CONCENTRATION",
                      context={"hhi": hhi, "largest_pct": max(vals) / eq if vals and eq else 0.0})


def correlated_exposure(w, ctx, sym):
    pos = _positions(ctx)
    clusters = next((s.context["clusters"] for s in ctx.signals if s.setup == "CORRELATION_CLUSTERS"), [])
    eq = ctx.portfolio.get("equity") or 1
    expo = {",".join(c): sum(pos.get(s, {}).get("notional", 0) for s in c) / eq for c in clusters}
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="CLUSTER_EXPOSURE",
                      context={"cluster_exposure": expo, "max_cluster_pct": max(expo.values(), default=0.0)})


def sector_exposure(w, ctx, sym):
    pos = _positions(ctx)
    eq = ctx.portfolio.get("equity") or 1
    out = defaultdict(float)
    for s, p in pos.items():
        sec = next((k for k, v in SECTORS.items() if base_of(s) in v), "OTHER")
        out[sec] += p.get("notional", 0) / eq
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="SECTOR_EXPOSURE", context={"sector_exposure": dict(out)})


def beta_exposure(w, ctx, sym):
    pos = _positions(ctx)
    eq = ctx.portfolio.get("equity") or 1
    betas = {s.symbol: s.context.get("beta", 1.0) for s in ctx.signals if s.setup == "BETA_TO_MARKET"}
    beta_usd = sum(p.get("notional", 0) * betas.get(s, 1.0) for s, p in pos.items())
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="BETA_EXPOSURE", context={"beta_weighted_exposure": beta_usd / eq})


def cash_utilization(w, ctx, sym):
    eq = ctx.portfolio.get("equity") or 0
    cash = ctx.portfolio.get("cash") or 0
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="CASH", context={"cash_pct": cash / eq if eq else 1.0,
                                                                     "open_positions": len(_positions(ctx))})


# --------------------------------------------------------------------- adversarial (devil's advocate)

def _net_direction(ctx, sym):
    s = sum(x.score for x in ctx.signals if x.symbol in (sym, MARKET_WIDE) and x.family in DIRECTIONAL_FAMILIES)
    return D.from_score(s, 0.5)


def _adv(w, ctx, sym, points: float, reasons: list, **extra):
    if points <= 0:
        return None
    return ctx.signal(w, sym, N, 0.0, setup="RISK_FLAG", context={"risk_points": round(points, 2), "reasons": reasons,
                                                                 **extra})


def counter_signals(w, ctx, sym):
    d = _net_direction(ctx, sym)
    if d == N:
        return None
    opp = sorted((x for x in ctx.signals if x.symbol == sym and x.family in DIRECTIONAL_FAMILIES
                  and x.direction.sign == -d.sign), key=lambda x: -x.confidence)
    strong = [x for x in opp if x.confidence >= 0.5]
    return _adv(w, ctx, sym, min(3.0, 0.5 * len(strong) + sum(x.confidence for x in opp[:5]) * 0.3),
                [f"{x.worker_id} says {x.direction.value} ({x.confidence:.2f})" for x in opp[:4]], against=d.value)


def event_risk(w, ctx, sym):
    reasons, pts = [], 0.0
    for s in ctx.signals:
        if s.setup == "MACRO_EVENT":
            h = s.context.get("hours_to_event", 99)
            pts += 2.0 if abs(h) < 3 else 1.0
            reasons.append(f"{s.context.get('event')} in {h:.1f}h")
    for x in ctx.feeds.news():
        if x.impact >= 0.75 and x.decay(ctx.now) > 0.6 and (base_of(sym) in x.assets or x.market_wide):
            pts += 0.8
            reasons.append(f"high-impact news: {x.title[:70]}")
    return _adv(w, ctx, sym, min(4.0, pts), reasons[:5])


def crowding(w, ctx, sym):
    d = _net_direction(ctx, sym)
    reasons, pts = [], 0.0
    for s in ctx.signals:
        if s.symbol != sym:
            continue
        f = s.context.get("funding_rate")
        if f is not None and d != N and np.sign(f) == d.sign and abs(f) > 0.0005:
            pts += 1.2
            reasons.append(f"funding elevated {f * 100:.3f}% in trade direction")
        ls = s.context.get("long_short_ratio")
        if ls is not None and ((d == D.LONG and ls > 2.2) or (d == D.SHORT and ls < 0.7)):
            pts += 0.8
            reasons.append(f"positioning crowded (L/S {ls:.2f})")
        if s.setup == "OI_UP_PRICE_UP" and d == D.LONG and s.context.get("oi_change", 0) > 0.1:
            pts += 0.7
            reasons.append("open interest rising faster than spot")
        if s.context.get("hype_risk") or s.context.get("classification") == "COORDINATED_HYPE":
            pts += 0.8
            reasons.append("coordinated social hype")
    return _adv(w, ctx, sym, min(3.0, pts), sorted(set(reasons)))


def liquidity_slippage(w, ctx, sym):
    reasons, pts = [], 0.0
    for s in ctx.signals:
        if s.symbol != sym:
            continue
        if s.context.get("liquidity_stress"):
            pts += 1.0
            reasons.append("spread blown out")
        if s.context.get("slippage_risk"):
            pts += 0.8
            reasons.append("thin order book near price")
        slip = s.context.get("expected_slippage_bps")
        if slip and slip > 15:
            pts += 1.0
            reasons.append(f"expected slippage {slip:.0f}bps")
        if s.context.get("toxic_flow"):
            pts += 0.5
            reasons.append("toxic order flow")
    return _adv(w, ctx, sym, min(3.0, pts), sorted(set(reasons)))


def stop_check(w, ctx, sym):
    for s in ctx.signals:
        if s.symbol == sym and s.setup == "STOP_FEASIBILITY_2":
            if s.context.get("stop_too_wide"):
                return _adv(w, ctx, sym, 2.0, [f"ATR stop {s.context['atr_stop_pct']:.1%} too wide"])
            if s.context.get("stop_too_tight"):
                return _adv(w, ctx, sym, 1.0, ["stop inside noise"])
    return None


def specs() -> list[dict]:
    R, E, P, A = Family.RISK, Family.STRATEGY_EVAL, Family.PORTFOLIO, Family.ADVERSARIAL
    from lunatrade.workers.strategy_groups import GROUPS

    out = [spec(R, "var_95", var_hist, params={"n": 500, "q": 95}),
           spec(R, "var_99", var_hist, params={"n": 500, "q": 99}),
           spec(R, "realized_vol_annual", realized_vol_ann, min_bars=100),
           spec(R, "drawdown_speed", drawdown_speed),
           spec(R, "beta_to_market", beta_to_market),
           spec(R, "correlation_clusters", correlation_clusters, params={"thr": 0.7}, per_symbol=False),
           spec(R, "liquidity_risk", liquidity_risk, min_bars=1),
           spec(R, "gap_risk", gap_risk),
           spec(R, "stop_feasibility_2", stop_feasibility, params={"mult": 2.0, "max_stop": 0.08}),
           spec(R, "tail_kurtosis", tail_kurtosis, min_bars=100)]
    eval_groups = ["TREND", "MEAN_REVERSION", "MOMENTUM", "STRUCTURE", "PATTERN", "MICROSTRUCTURE", "POSITIONING",
                   "NEWS", "FORECAST"]
    assert set(eval_groups) <= set(GROUPS)
    out += [spec(E, f"eval_{g.lower()}", strategy_eval, params={"group": g}, per_symbol=False, phase=2)
            for g in eval_groups]
    out.append(spec(E, "degradation_detector", degradation_detector, per_symbol=False, phase=2))
    out += [spec(P, "concentration", concentration, per_symbol=False, phase=2),
            spec(P, "correlated_exposure", correlated_exposure, per_symbol=False, phase=2),
            spec(P, "sector_exposure", sector_exposure, per_symbol=False, phase=2),
            spec(P, "beta_exposure", beta_exposure, per_symbol=False, phase=2),
            spec(P, "cash_utilization", cash_utilization, per_symbol=False, phase=2)]
    out += [spec(A, "counter_signals", counter_signals, phase=2, min_bars=1),
            spec(A, "event_risk", event_risk, phase=2, min_bars=1),
            spec(A, "crowding", crowding, phase=2, min_bars=1),
            spec(A, "liquidity_slippage", liquidity_slippage, phase=2, min_bars=1),
            spec(A, "stop_feasibility_check", stop_check, phase=2, min_bars=1)]
    return out

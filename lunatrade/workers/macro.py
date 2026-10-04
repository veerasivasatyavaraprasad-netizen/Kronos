"""Macro workers (15): equities, dollar, yields, gold, VIX, credit, oil, risk-on/off, scheduled events."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import MARKET_WIDE
from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.workers.base import clamp, spec

L, S, N = D.LONG, D.SHORT, D.NEUTRAL


def _avail(name):
    def f(w, ctx):
        return (True, "") if ctx.feeds.has(f"macro_{name}") else (False, f"no macro data for {name}")
    return f


def _trend(vals: np.ndarray, n: int) -> float | None:
    if len(vals) <= n:
        return None
    return vals[-1] / vals[-n - 1] - 1


def asset_trend(w, ctx, sym):
    """Risk asset up = crypto tailwind; dollar/yields up = headwind (sign param)."""
    v = ctx.feeds.series(f"macro_{w.params['name']}")
    t = _trend(v, w.params["days"])
    if t is None:
        return None
    s = t * w.params["sign"]
    thr = w.params.get("thr", 0.02)
    if abs(s) < thr:
        return ctx.signal(w, MARKET_WIDE, N, 0.0, setup=f"MACRO_{w.params['name']}", context={"trend": t})
    return ctx.signal(w, MARKET_WIDE, D.from_score(s), clamp(abs(s) / thr * 0.15), setup=f"MACRO_{w.params['name']}",
                      time_horizon="1w", evidence=[f"{w.params['name']}_{w.params['days']}d={t:+.2%}"], context={"trend": t})


def vix_level(w, ctx, sym):
    v = ctx.feeds.series("macro_VIX")
    if not len(v):
        return None
    x = v[-1]
    if x > 30:
        return ctx.signal(w, MARKET_WIDE, S, clamp((x - 30) / 20 + 0.3), setup="VIX_STRESS", context={"vix": x,
                                                                                                   "risk_off": True})
    if x < 15:
        return ctx.signal(w, MARKET_WIDE, L, 0.2, setup="VIX_CALM", context={"vix": x})
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="VIX", context={"vix": x})


def vix_spike(w, ctx, sym):
    v = ctx.feeds.series("macro_VIX")
    t = _trend(v, 5)
    if t is None or t < 0.25:
        return None
    return ctx.signal(w, MARKET_WIDE, S, clamp(t), setup="VIX_SPIKE", context={"vix_5d_change": t, "risk_off": True})


def risk_composite(w, ctx, sym):
    n = w.params["days"]
    parts = {"SPY": 1, "QQQ": 1, "CREDIT": 1, "DOLLAR": -1, "US10Y": -0.5, "VIX": -1, "GOLD": 0.2}
    score, used = 0.0, 0
    for name, sign in parts.items():
        t = _trend(ctx.feeds.series(f"macro_{name}"), n)
        if t is None:
            continue
        score += np.tanh(t * 20) * sign
        used += 1
    if used < 3:
        return None
    score /= used
    state = "RISK_ON" if score > 0.2 else "RISK_OFF" if score < -0.2 else "MIXED"
    return ctx.signal(w, MARKET_WIDE, D.from_score(score, 0.2), clamp(abs(score)), setup=state, time_horizon="1w",
                      context={"risk_score": score, "risk_state": state, "inputs": used})


def event_proximity(w, ctx, sym):
    key = w.params["event"]
    evs = [e for e in ctx.feeds.upcoming_events(w.params["hours"]) if key.lower() in e["name"].lower()]
    if not evs:
        return None
    e = min(evs, key=lambda x: abs((x["time"] - ctx.now).total_seconds()))
    hours = (e["time"] - ctx.now).total_seconds() / 3600
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="MACRO_EVENT",
                      context={"event": e["name"], "hours_to_event": round(hours, 2), "event_risk": True},
                      invalidations=["event_surprise"])


def _calendar_avail(w, ctx):
    return (True, "") if ctx.feeds.hub.calendar else (False, "no macro calendar configured")


def specs() -> list[dict]:
    M = Family.MACRO
    out = [
        spec(M, "equities_spy", asset_trend, params={"name": "SPY", "days": 10, "sign": 1}, per_symbol=False,
             availability=_avail("SPY")),
        spec(M, "equities_qqq", asset_trend, params={"name": "QQQ", "days": 10, "sign": 1}, per_symbol=False,
             availability=_avail("QQQ")),
        spec(M, "dollar_strength", asset_trend, params={"name": "DOLLAR", "days": 10, "sign": -1, "thr": 0.01},
             per_symbol=False, availability=_avail("DOLLAR")),
        spec(M, "treasury_yields", asset_trend, params={"name": "US10Y", "days": 10, "sign": -1, "thr": 0.03},
             per_symbol=False, availability=_avail("US10Y")),
        spec(M, "gold", asset_trend, params={"name": "GOLD", "days": 20, "sign": 0.5}, per_symbol=False,
             availability=_avail("GOLD")),
        spec(M, "vix_level", vix_level, per_symbol=False, availability=_avail("VIX")),
        spec(M, "vix_spike", vix_spike, per_symbol=False, availability=_avail("VIX")),
        spec(M, "credit_risk", asset_trend, params={"name": "CREDIT", "days": 10, "sign": 1, "thr": 0.01},
             per_symbol=False, availability=_avail("CREDIT")),
        spec(M, "oil", asset_trend, params={"name": "OIL", "days": 10, "sign": -0.5, "thr": 0.05}, per_symbol=False,
             availability=_avail("OIL")),
        spec(M, "risk_on_off_20d", risk_composite, params={"days": 20}, per_symbol=False, availability=_avail("SPY")),
        spec(M, "risk_on_off_60d", risk_composite, params={"days": 60}, per_symbol=False, availability=_avail("SPY")),
    ]
    for ev, hours in [("FOMC", 48), ("CPI", 24), ("NFP", 24), ("PCE", 24)]:
        out.append(spec(M, f"event_{ev.lower()}", event_proximity, params={"event": ev, "hours": hours},
                        per_symbol=False, availability=_calendar_avail))
    return out

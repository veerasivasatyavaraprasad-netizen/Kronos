"""News intelligence workers (20): one specialist per event family plus velocity/divergence/composite."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.intel.assets import base_of
from lunatrade.workers.base import clamp, spec

N = D.NEUTRAL

CATEGORY_WORKERS = {
    "etf_flow": {"ETF_FLOW"}, "regulation": {"REGULATION"}, "lawsuit": {"LAWSUIT"},
    "hack_exploit": {"HACK_EXPLOIT"}, "listing": {"LISTING"}, "delisting": {"DELISTING"},
    "partnership": {"PARTNERSHIP"}, "token_unlock": {"TOKEN_UNLOCK"}, "central_bank": {"CENTRAL_BANK"},
    "macro_data": {"MACRO_DATA"}, "geopolitical": {"GEOPOLITICAL"}, "governance_tech": {"GOVERNANCE", "TECH_UPGRADE"},
    "exchange_news": {"EXCHANGE_NEWS"}, "institutional": {"INSTITUTIONAL"}, "stablecoin": {"STABLECOIN"},
    "mining_adoption": {"MINING", "ADOPTION"}, "market_chatter": {"MARKET_MOVE", "GENERAL"},
}
HORIZON = {"HACK_EXPLOIT": "1d", "LISTING": "4h", "DELISTING": "1d", "CENTRAL_BANK": "3d", "ETF_FLOW": "3d"}


def _has_news(w, ctx):
    return (True, "") if ctx.feeds.hub.news.items else (False, "no news ingested yet")


def category(w, ctx, sym):
    asset = base_of(sym)
    strength, items = ctx.feeds.news_signal(asset, set(w.params["types"]))
    if not items or abs(strength) < 0.05:
        return None
    top = max(items, key=lambda x: abs(x.signed_strength(ctx.now)))
    d = D.from_score(strength)
    return ctx.signal(w, sym, d, clamp(abs(strength)), setup=f"NEWS_{top.event_type}",
                      expected_move=d.sign * min(0.08, abs(strength) * 0.05), time_horizon=HORIZON.get(top.event_type, "1d"),
                      evidence=[f"{x.source}: {x.title[:90]}" for x in items[:3]],
                      invalidations=["news_retracted", "price_ignores_news"],
                      context={"event": top.event_type, "sentiment": round(top.sentiment, 3),
                               "source_quality": top.credibility, "novelty": top.novelty,
                               "items": len(items), "half_life_h": top.half_life_hours})


def breaking_velocity(w, ctx, sym):
    asset = base_of(sym)
    items = [x for x in ctx.feeds.news() if asset in x.assets]
    recent = [x for x in items if (ctx.now - x.ts).total_seconds() < 3600]
    if len(recent) < 3:
        return None
    s = float(np.mean([x.sentiment for x in recent]))
    return ctx.signal(w, sym, D.from_score(s, 0.1), clamp(len(recent) / 10 * abs(s) * 2), setup="NEWS_BURST",
                      time_horizon="4h", evidence=[f"{len(recent)} stories/hour"],
                      context={"stories_last_hour": len(recent), "burst": True})


def source_divergence(w, ctx, sym):
    asset = base_of(sym)
    items = [x for x in ctx.feeds.news() if asset in x.assets or x.market_wide]
    hi = [x.sentiment for x in items if x.credibility >= 0.85]
    lo = [x.sentiment for x in items if x.credibility < 0.7]
    if len(hi) < 2 or len(lo) < 2:
        return None
    gap = float(np.mean(hi) - np.mean(lo))
    # trust the credible sources when the crowd of weak sources disagrees
    if abs(gap) < 0.3:
        return None
    return ctx.signal(w, sym, D.from_score(np.mean(hi), 0.05), clamp(abs(gap) * 0.5), setup="CREDIBLE_VS_NOISE",
                      context={"credible_sentiment": float(np.mean(hi)), "low_cred_sentiment": float(np.mean(lo))})


def composite(w, ctx, sym):
    strength, items = ctx.feeds.news_signal(base_of(sym))
    if not items:
        return ctx.signal(w, sym, N, 0.0, setup="NEWS_NONE", context={"news_score": 0.0, "items": 0})
    return ctx.signal(w, sym, D.from_score(strength, 0.05), clamp(abs(strength)), setup="NEWS_COMPOSITE",
                      context={"news_score": strength, "items": len(items),
                               "high_impact": [x.title[:80] for x in items if x.impact >= 0.7][:3]})


def specs() -> list[dict]:
    F = Family.NEWS
    out = [spec(F, f"news_{name}", category, params={"types": sorted(types)}, availability=_has_news)
           for name, types in CATEGORY_WORKERS.items()]
    out += [spec(F, "news_breaking_velocity", breaking_velocity, availability=_has_news),
            spec(F, "news_source_divergence", source_divergence, availability=_has_news),
            spec(F, "news_composite", composite, availability=_has_news)]
    return out

"""Social sentiment workers (20). Low weight by design; hype and bot activity are filtered out."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import MARKET_WIDE
from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.intel import social as SA
from lunatrade.intel.assets import base_of
from lunatrade.workers import common as K
from lunatrade.workers.base import clamp, spec

L, S, N = D.LONG, D.SHORT, D.NEUTRAL


def _has_posts(w, ctx):
    return (True, "") if ctx.feeds.hub.posts else (False, "no social data")


def _has_fng(w, ctx):
    return (True, "") if ctx.feeds.has("fear_greed") else (False, "no fear & greed data")


def community_sentiment(w, ctx, sym):
    posts = [p for p in ctx.feeds.posts(24) if p.community.lower() == w.params["community"].lower()]
    r = SA.analyze(posts, base_of(sym), ctx.now, 12)
    if r.posts < 3 or r.classification in ("BOT_ACTIVITY", "COORDINATED_HYPE") or abs(r.sentiment) < 0.1:
        return None
    return ctx.signal(w, sym, D.from_score(r.sentiment), clamp(abs(r.sentiment) * r.confidence), setup="COMMUNITY_SENTIMENT",
                      context={"community": w.params["community"], "sentiment": r.sentiment, "posts": r.posts,
                               "classification": r.classification})


def mention_velocity(w, ctx, sym):
    r = SA.analyze(ctx.feeds.posts(72), base_of(sym), ctx.now, w.params["hours"])
    if r.posts < 5 or r.mention_velocity < 2:
        return None
    if r.classification in ("BOT_ACTIVITY", "COORDINATED_HYPE"):
        return ctx.signal(w, sym, N, 0.0, setup="SUSPICIOUS_BUZZ", context={"velocity": r.mention_velocity,
                                                                           "classification": r.classification})
    return ctx.signal(w, sym, D.from_score(r.sentiment, 0.05), clamp(np.log(r.mention_velocity) * 0.2 * r.confidence),
                      setup="MENTION_SURGE", context={"velocity": r.mention_velocity, "sentiment": r.sentiment})


def organic_vs_coordinated(w, ctx, sym):
    r = SA.analyze(ctx.feeds.posts(72), base_of(sym), ctx.now, w.params["hours"])
    if r.posts < 5:
        return None
    return ctx.signal(w, sym, N, 0.0, setup="DISCUSSION_QUALITY",
                      context={"classification": r.classification, "author_diversity": r.author_diversity,
                               "shill_ratio": r.shill_ratio, "hype_risk": r.classification == "COORDINATED_HYPE"})


def bot_detector(w, ctx, sym):
    r = SA.analyze(ctx.feeds.posts(48), base_of(sym) if w.params["per_asset"] else None, ctx.now, 12)
    if r.posts < 8:
        return None
    target = sym if w.params["per_asset"] else MARKET_WIDE
    return ctx.signal(w, target, N, 0.0, setup="BOT_SCAN",
                      context={"duplicate_ratio": r.duplicate_ratio, "bot_activity": r.classification == "BOT_ACTIVITY"})


def panic_detector(w, ctx, sym):
    r = SA.analyze(ctx.feeds.posts(48), base_of(sym) if w.params["per_asset"] else None, ctx.now, 6)
    if r.posts < 5 or r.classification not in ("PANIC", "CAPITULATION"):
        return None
    target = sym if w.params["per_asset"] else MARKET_WIDE
    # panic is bearish short-term but a contrarian tell at extremes -> low confidence, flagged for risk
    return ctx.signal(w, target, S, clamp(r.panic * 0.5), setup="SOCIAL_PANIC", time_horizon="4h",
                      context={"panic": r.panic, "classification": r.classification})


def capitulation(w, ctx, sym):
    r = SA.analyze(ctx.feeds.posts(48), base_of(sym), ctx.now, 6)
    c = ctx.view.candles(sym)
    if r.classification != "CAPITULATION" or len(c) < 50:
        return None
    dd = c.close[-1] / c.high[-50:].max() - 1
    if dd > -0.1:
        return None
    return ctx.signal(w, sym, L, 0.35, setup="CAPITULATION_CONTRARIAN", time_horizon="3d",
                      evidence=["social_capitulation", f"drawdown={dd:.1%}"])


def euphoria(w, ctx, sym):
    r = SA.analyze(ctx.feeds.posts(48), base_of(sym), ctx.now, 6)
    if r.classification not in ("EUPHORIA", "COORDINATED_HYPE"):
        return None
    return ctx.signal(w, sym, S, clamp(r.euphoria * 0.5), setup="EUPHORIA_CONTRARIAN", time_horizon="1d",
                      context={"euphoria": r.euphoria, "classification": r.classification})


def fear_greed(w, ctx, sym):
    s = ctx.feeds.series("fear_greed", 30)
    if not len(s):
        return None
    v = s[-1]
    lo, hi = w.params["lo"], w.params["hi"]
    if w.params["mode"] == "contrarian":
        if v <= lo:
            return ctx.signal(w, MARKET_WIDE, L, clamp((lo - v) / lo + 0.2), setup="EXTREME_FEAR", time_horizon="3d",
                              context={"fear_greed": v})
        if v >= hi:
            return ctx.signal(w, MARKET_WIDE, S, clamp((v - hi) / (100 - hi) + 0.2), setup="EXTREME_GREED",
                              time_horizon="3d", context={"fear_greed": v})
        return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="FEAR_GREED", context={"fear_greed": v})
    if len(s) < 8:
        return None
    trend = v - s[-8]
    if abs(trend) < 10:
        return None
    return ctx.signal(w, MARKET_WIDE, D.from_score(trend), clamp(abs(trend) / 40), setup="SENTIMENT_TREND",
                      context={"fear_greed": v, "fng_7d_change": trend})


def social_price_divergence(w, ctx, sym):
    r = SA.analyze(ctx.feeds.posts(72), base_of(sym), ctx.now, w.params["hours"])
    c = ctx.view.candles(sym)
    bars = max(2, int(w.params["hours"] * 3_600_000 / max(1, ctx.view.store.interval_ms)))
    if r.posts < 5 or len(c) <= bars:
        return None
    pr = c.close[-1] / c.close[-bars - 1] - 1
    a = K.atr_pct(ctx, sym) * np.sqrt(bars) or 1e-6
    if r.sentiment > 0.3 and pr < -a:
        return ctx.signal(w, sym, S, 0.3, setup="BULLISH_TALK_FALLING_PRICE", context={"sentiment": r.sentiment, "ret": pr})
    if r.sentiment < -0.3 and pr > a:
        return ctx.signal(w, sym, L, 0.3, setup="BEARISH_TALK_RISING_PRICE", context={"sentiment": r.sentiment, "ret": pr})
    return None


def specs() -> list[dict]:
    F = Family.SOCIAL
    out = []
    for comm in ("r/CryptoCurrency", "r/Bitcoin", "r/ethereum", "r/CryptoMarkets", "x"):
        out.append(spec(F, f"community_{comm.replace('r/', '').lower()}", community_sentiment,
                        params={"community": comm}, availability=_has_posts))
    for h in (1, 6, 24):
        out.append(spec(F, f"mention_velocity_{h}h", mention_velocity, params={"hours": h}, availability=_has_posts))
    for h in (6, 24):
        out.append(spec(F, f"organic_vs_coordinated_{h}h", organic_vs_coordinated, params={"hours": h},
                        availability=_has_posts))
    out.append(spec(F, "bot_detector_asset", bot_detector, params={"per_asset": True}, availability=_has_posts))
    out.append(spec(F, "bot_detector_market", bot_detector, params={"per_asset": False}, per_symbol=False,
                    availability=_has_posts))
    out.append(spec(F, "panic_asset", panic_detector, params={"per_asset": True}, availability=_has_posts))
    out.append(spec(F, "panic_market", panic_detector, params={"per_asset": False}, per_symbol=False,
                    availability=_has_posts))
    out.append(spec(F, "capitulation", capitulation, availability=_has_posts))
    out.append(spec(F, "euphoria", euphoria, availability=_has_posts))
    out.append(spec(F, "fear_greed_contrarian", fear_greed, params={"mode": "contrarian", "lo": 20, "hi": 80},
                    per_symbol=False, availability=_has_fng))
    out.append(spec(F, "fear_greed_trend", fear_greed, params={"mode": "trend", "lo": 0, "hi": 100},
                    per_symbol=False, availability=_has_fng))
    for h in (6, 24):
        out.append(spec(F, f"social_price_divergence_{h}h", social_price_divergence, params={"hours": h},
                        availability=_has_posts))
    return out

"""
Event memory: what happened to prices after each kind of news event. The measured reaction becomes a
bounded impact prior for the news pipeline (event types that actually move markets get more weight).
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

from lunatrade.intel.assets import base_of

CHECKPOINTS = {"ret_1h": 1, "ret_4h": 4, "ret_24h": 24}


class EventMemory:
    def __init__(self, repo=None, prior_bounds=(0.5, 1.6)):
        self.repo = repo
        self.pending: dict[str, dict] = {}          # news id -> {item, price0, done: set}
        self.reactions: dict[str, list] = defaultdict(list)   # event type -> [(sentiment, ret_4h)]
        self.lo, self.hi = prior_bounds

    def track(self, item, prices_by_base: dict) -> None:
        asset = item.assets[0] if item.assets else ("BTC" if item.market_wide else None)
        if not asset or item.id in self.pending or asset not in prices_by_base:
            return
        self.pending[item.id] = {"item": item, "asset": asset, "p0": prices_by_base[asset], "done": set()}

    def update(self, now: dt.datetime, prices_by_base: dict) -> int:
        n = 0
        for nid, rec in list(self.pending.items()):
            item, px = rec["item"], prices_by_base.get(rec["asset"])
            if not px:
                continue
            age_h = (now - item.ts).total_seconds() / 3600
            vals = {}
            for col, hours in CHECKPOINTS.items():
                if col not in rec["done"] and age_h >= hours:
                    vals[col] = px / rec["p0"] - 1
                    rec["done"].add(col)
                    if col == "ret_4h":
                        self.reactions[item.event_type].append((item.sentiment, vals[col]))
            if vals:
                n += 1
                if self.repo:
                    try:
                        self.repo.news_outcome(nid, **vals)
                    except Exception:
                        pass
            if len(rec["done"]) == len(CHECKPOINTS) or age_h > 48:
                self.pending.pop(nid, None)
        return n

    def impact_priors(self, min_samples: int = 10) -> dict[str, float]:
        """Mean |4h move| per event type relative to the all-event average, plus sign agreement."""
        all_moves = [abs(r) for v in self.reactions.values() for _, r in v]
        if len(all_moves) < min_samples:
            return {}
        base = sum(all_moves) / len(all_moves) or 1e-9
        out = {}
        for et, v in self.reactions.items():
            if len(v) < min_samples:
                continue
            mag = sum(abs(r) for _, r in v) / len(v) / base
            agree = sum(1 for s, r in v if s * r > 0) / len(v)
            out[et] = round(max(self.lo, min(self.hi, mag * (0.5 + agree))), 3)
        return out


def prices_by_base(prices: dict) -> dict:
    return {base_of(s): p for s, p in prices.items() if p}

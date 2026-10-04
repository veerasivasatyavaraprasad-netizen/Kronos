"""
Regime engine: tallies regime-worker votes into one state per symbol and for the market, and turns it
into strategy-group multipliers and an exposure multiplier.

    TRENDING -> trend/momentum weighted up       RANGING -> mean reversion weighted up
    PANIC    -> exposure cut hard                UNKNOWN -> reduced exposure
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from lunatrade.core.types import MARKET_WIDE, Family, Regime

GROUP_MULTIPLIERS = {
    Regime.TRENDING: {"TREND": 1.4, "MOMENTUM": 1.3, "MEAN_REVERSION": 0.6, "STRUCTURE": 1.0, "PATTERN": 0.9,
                      "FORECAST": 1.1, "ROTATION": 1.1},
    Regime.RANGING: {"TREND": 0.6, "MOMENTUM": 0.8, "MEAN_REVERSION": 1.4, "STRUCTURE": 1.2, "PATTERN": 1.1,
                     "MICROSTRUCTURE": 1.1},
    Regime.HIGH_VOL: {"MICROSTRUCTURE": 0.7, "PATTERN": 0.8, "MEAN_REVERSION": 0.8},
    Regime.PANIC: {"MEAN_REVERSION": 0.5, "PATTERN": 0.5, "SOCIAL": 0.5, "MOMENTUM": 0.8},
    Regime.EVENT_DRIVEN: {"NEWS": 1.5, "MACRO": 1.3, "TREND": 0.8, "PATTERN": 0.7},
    Regime.ILLIQUID: {"MICROSTRUCTURE": 0.5},
    Regime.LOW_VOL: {"VOLATILITY": 1.3, "MEAN_REVERSION": 1.1},
}
EXPOSURE = {Regime.PANIC: 0.3, Regime.ILLIQUID: 0.3, Regime.UNKNOWN: 0.5, Regime.HIGH_VOL: 0.7,
            Regime.EVENT_DRIVEN: 0.7}


@dataclass
class RegimeState:
    symbol: str
    primary: Regime = Regime.UNKNOWN
    scores: dict = field(default_factory=dict)      # regime -> 0..1
    trend_bias: str = "NEUTRAL"                     # BULL | BEAR | NEUTRAL
    voters: int = 0
    confidence: float = 0.0

    @property
    def flags(self) -> list[str]:
        return [r for r, v in self.scores.items() if v >= 0.5]

    def group_multiplier(self, group: str) -> float:
        m = GROUP_MULTIPLIERS.get(self.primary, {}).get(group, 1.0)
        for flag in (Regime.HIGH_VOL, Regime.EVENT_DRIVEN):
            if flag != self.primary and self.scores.get(flag.value, 0) >= 0.5:
                m *= GROUP_MULTIPLIERS[flag].get(group, 1.0)
        return m

    @property
    def exposure_multiplier(self) -> float:
        m = EXPOSURE.get(self.primary, 1.0)
        if self.scores.get(Regime.HIGH_VOL.value, 0) >= 0.5 and self.primary != Regime.HIGH_VOL:
            m *= EXPOSURE[Regime.HIGH_VOL]
        if self.scores.get(Regime.PANIC.value, 0) >= 0.5 and self.primary != Regime.PANIC:
            m = min(m, EXPOSURE[Regime.PANIC])
        return m

    def label(self) -> str:
        extra = [f for f in self.flags if f != self.primary.value]
        return " / ".join([self.primary.value, self.trend_bias] + extra)

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "primary": self.primary.value, "scores": self.scores, "trend_bias": self.trend_bias,
                "voters": self.voters, "confidence": self.confidence, "flags": self.flags,
                "exposure_multiplier": self.exposure_multiplier, "label": self.label()}


def classify(scores: dict, voters: int) -> tuple[Regime, float]:
    if voters < 3:
        return Regime.UNKNOWN, 0.0
    g = lambda r: scores.get(r.value, 0.0)
    for r, thr in ((Regime.PANIC, 0.5), (Regime.ILLIQUID, 0.6), (Regime.EVENT_DRIVEN, 0.7)):
        if g(r) >= thr:
            return r, g(r)
    t, rg = g(Regime.TRENDING), g(Regime.RANGING)
    if max(t, rg) < 0.2:
        if g(Regime.HIGH_VOL) >= 0.5:
            return Regime.HIGH_VOL, g(Regime.HIGH_VOL)
        return Regime.UNKNOWN, 0.0
    return (Regime.TRENDING, t) if t >= rg else (Regime.RANGING, rg)


class RegimeEngine:
    def compute(self, signals: list, symbols: list[str], market_symbol: str) -> dict[str, RegimeState]:
        votes: dict[str, dict] = defaultdict(lambda: defaultdict(list))
        voters: dict[str, set] = defaultdict(set)
        for s in signals:
            if s.family != Family.REGIME:
                continue
            voters[s.symbol].add(s.worker_id)
            for r, v in (s.context.get("votes") or {}).items():
                votes[s.symbol][r].append(v)
        out = {}
        for sym in symbols:
            n = len(voters.get(sym, ()))
            # average over every voter (a voter that didn't vote for a regime counts as 0)
            scores = {r: round(sum(v) / max(n, 1), 3) for r, v in votes.get(sym, {}).items()}
            primary, conf = classify(scores, n)
            bull, bear = scores.get(Regime.BULL.value, 0), scores.get(Regime.BEAR.value, 0)
            bias = "BULL" if bull > bear + 0.1 else "BEAR" if bear > bull + 0.1 else "NEUTRAL"
            out[sym] = RegimeState(sym, primary, scores, bias, n, round(conf, 3))
        m = out.get(market_symbol)
        out[MARKET_WIDE] = RegimeState(MARKET_WIDE, m.primary, dict(m.scores), m.trend_bias, m.voters, m.confidence) \
            if m else RegimeState(MARKET_WIDE)
        # a market-wide panic overrides calm-looking alts
        if out[MARKET_WIDE].primary == Regime.PANIC:
            for st in out.values():
                st.scores[Regime.PANIC.value] = max(st.scores.get(Regime.PANIC.value, 0), 0.5)
        return out

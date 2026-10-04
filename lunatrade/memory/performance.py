"""
Strategy performance memory + meta-learning.

Two feedback loops:
  * fast  - every directional signal is scored against the realised move over its horizon
            (hit rate and signed edge per strategy group, recent vs. long-run);
  * slow  - every closed trade is attributed to the strategy groups that supported it
            (win rate, expectancy, profit factor, Sharpe/Sortino, drawdown, per regime and asset).

The meta-learner turns these into bounded strategy weights and a bounded conviction threshold.
It may change weights, thresholds and multipliers - never code, never the risk limits.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from lunatrade.core.types import DIRECTIONAL_FAMILIES, MARKET_WIDE, Direction
from lunatrade.workers.strategy_groups import group_of

HORIZON_BARS = {"5m": 1, "15m": 1, "30m": 2, "1h": 4, "4h": 16, "1d": 96, "3d": 288, "1w": 672}


def horizon_bars(h: str, interval_ms: int) -> int:
    if h.endswith("bars"):
        try:
            return max(1, int(h[:-4]))
        except ValueError:
            return 8
    minutes = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    try:
        n = int(h[:-1]) * minutes[h[-1]]
    except (ValueError, KeyError):
        return 16
    return max(1, int(n * 60_000 / interval_ms))


@dataclass
class GroupStats:
    samples: int = 0
    hits: int = 0
    edge_sum: float = 0.0
    edge_sq: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=60))
    weight: float = 1.0
    trades: list = field(default_factory=list)

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.samples if self.samples else None

    @property
    def recent_hit_rate(self) -> float | None:
        return sum(self.recent) / len(self.recent) if self.recent else None

    @property
    def edge(self) -> float:
        if self.samples < 2:
            return 0.0
        mean = self.edge_sum / self.samples
        var = max(self.edge_sq / self.samples - mean ** 2, 1e-12)
        return mean / math.sqrt(var)       # information-ratio-like, per signal


def trade_stats(outcomes: list[dict]) -> dict:
    if not outcomes:
        return {"trades": 0}
    pnl = np.array([o.get("pnl", 0.0) for o in outcomes])
    rets = np.array([o.get("return_pct", 0.0) for o in outcomes])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0, eq])[1:]
    downside = rets[rets < 0]
    return {
        "trades": len(outcomes), "win_rate": float(len(wins) / len(pnl)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(pnl.mean()),
        "expectancy_r": float(np.mean([o.get("r_multiple", 0.0) for o in outcomes])),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        "sharpe": float(rets.mean() / rets.std() * math.sqrt(len(rets))) if rets.std() > 0 else 0.0,
        "sortino": float(rets.mean() / downside.std() * math.sqrt(len(rets))) if len(downside) > 1 and downside.std() > 0 else 0.0,
        "max_drawdown": float((peak - eq).max()) if len(eq) else 0.0,
        "avg_holding_bars": float(np.mean([o.get("bars_held", 0) for o in outcomes])),
    }


class StrategyMemory:
    def __init__(self, interval_ms: int, cfg: dict | None = None, min_confidence: float = 0.3):
        self.interval_ms = interval_ms
        self.cfg = cfg or {}
        self.min_conf = min_confidence
        self.groups: dict[str, GroupStats] = defaultdict(GroupStats)
        self.pending: deque = deque(maxlen=200_000)
        self._seen: set = set()
        self.outcomes: list[dict] = []
        self.threshold_offset = 0.0

    # ---------------------------------------------------------------- fast loop: signal scoring
    def record_signals(self, signals: list, prices: dict, bar_ms: int) -> None:
        for s in signals:
            if s.family not in DIRECTIONAL_FAMILIES or s.direction == Direction.NEUTRAL or s.confidence < self.min_conf:
                continue
            if s.symbol == MARKET_WIDE:
                continue
            px = prices.get(s.symbol)
            if not px:
                continue
            key = (s.worker_id, s.symbol, bar_ms)
            if key in self._seen:
                continue
            self._seen.add(key)
            due = bar_ms + horizon_bars(s.time_horizon, self.interval_ms) * self.interval_ms
            self.pending.append((due, group_of(s.worker_id), s.symbol, s.direction.sign, px))
        if len(self._seen) > 500_000:
            self._seen.clear()

    def evaluate_due(self, prices: dict, bar_ms: int) -> int:
        n, keep = 0, deque(maxlen=self.pending.maxlen)
        for item in self.pending:
            due, group, sym, sign, px0 = item
            if due > bar_ms or sym not in prices:
                keep.append(item)
                continue
            ret = prices[sym] / px0 - 1
            g = self.groups[group]
            g.samples += 1
            hit = 1 if sign * ret > 0 else 0
            g.hits += hit
            g.recent.append(hit)
            g.edge_sum += sign * ret
            g.edge_sq += ret * ret
            n += 1
        self.pending = keep
        return n

    # ---------------------------------------------------------------- slow loop: trade attribution
    def record_outcome(self, outcome: dict) -> None:
        self.outcomes.append(outcome)
        for g in {group_of(w) for w in outcome.get("supporting", [])}:
            self.groups[g].trades.append(outcome)

    # ---------------------------------------------------------------- meta-learning
    def update_weights(self) -> dict:
        mcfg = self.cfg
        lo, hi = float(mcfg.get("min_weight", 0.25)), float(mcfg.get("max_weight", 2.0))
        step = float(mcfg.get("max_step", 0.10))
        min_n = int(mcfg.get("min_samples", 20))
        changes = {}
        for name, g in self.groups.items():
            if g.samples < min_n:
                continue
            recent = g.recent_hit_rate if g.recent_hit_rate is not None else 0.5
            long_run = g.hit_rate or 0.5
            # blend long-run edge with recent hit rate (recent degradation pulls the weight down fast)
            target = 1.0 + 3.0 * g.edge + 2.0 * (0.6 * recent + 0.4 * long_run - 0.5)
            if len(g.trades) >= min_n:
                ts = trade_stats(g.trades[-100:])
                target += 0.5 * max(-1.0, min(1.0, ts["expectancy_r"]))
            target = max(lo, min(hi, target))
            new = g.weight * (1 + max(-step, min(step, target / g.weight - 1)))
            new = max(lo, min(hi, new))
            if abs(new - g.weight) > 1e-6:
                changes[name] = (round(g.weight, 3), round(new, 3))
            g.weight = new
        # conviction threshold: tighten after a losing streak of trades, relax slowly when trading well
        recent = self.outcomes[-30:]
        if len(recent) >= 15:
            ts = trade_stats(recent)
            if ts["expectancy_r"] < 0:
                self.threshold_offset = min(self.threshold_offset + 1.0, 15)
            elif ts["profit_factor"] > 1.5:
                self.threshold_offset = max(self.threshold_offset - 0.5, -5)
        return changes

    def min_conviction(self, base: float) -> float:
        lo, hi = self.cfg.get("conviction_threshold_bounds", [55, 80])
        return max(float(lo), min(float(hi), base + self.threshold_offset))

    def weights(self) -> dict[str, float]:
        return {k: round(v.weight, 4) for k, v in self.groups.items()}

    def calibration(self, conviction: float, min_samples: int = 20) -> float | None:
        """P(win) for trades opened near this conviction (+-5) - separate from conviction itself."""
        near = [o for o in self.outcomes if abs((o.get("conviction") or 0) - conviction) <= 5]
        if len(near) < min_samples:
            return None
        return round(sum(1 for o in near if o.get("pnl", 0) > 0) / len(near), 3)

    def summary(self) -> dict:
        out = {}
        for name, g in self.groups.items():
            st = {"samples": g.samples, "hit_rate": g.hit_rate, "recent_hit_rate": g.recent_hit_rate,
                  "signal_edge": round(g.edge, 4), "weight": round(g.weight, 4)}
            if g.trades:
                ts = trade_stats(g.trades)
                st.update(ts)
                by_regime = defaultdict(list)
                by_asset = defaultdict(list)
                for o in g.trades:
                    by_regime[o.get("regime") or "?"].append(o)
                    by_asset[o.get("symbol")].append(o)
                st["by_regime"] = {k: trade_stats(v)["expectancy"] for k, v in by_regime.items()}
                st["by_asset"] = {k: trade_stats(v)["expectancy"] for k, v in by_asset.items()}
            st["expectancy"] = st.get("expectancy", g.edge)
            out[name] = st
        return out

    def load(self, rows: list[dict]) -> None:
        """Restore weights from the strategy_performance table after a restart."""
        for r in rows:
            g = self.groups[r["strategy"]]
            g.weight = float(r.get("weight") or 1.0)
            g.samples = int(r.get("samples") or 0)
            if r.get("hit_rate") is not None:
                g.hits = int(round(g.samples * r["hit_rate"]))

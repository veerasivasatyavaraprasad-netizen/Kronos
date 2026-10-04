"""
Lead Trading Brain - the "fund manager".

It does not average opinions. For every symbol it:
  1. groups directional signals by family (technical, momentum, news, on-chain, ...),
  2. weights each signal by its strategy group's learned performance x the regime multiplier,
  3. scores each family in [-100, 100] (shrunk towards 0 when evidence is thin),
  4. combines families with configured category weights and a coverage factor,
  5. requires independent families to agree, penalises strong dissent,
  6. maps the result to conviction 0..100 (NOT a probability) and attaches a calibrated
     win probability from trade history when there is enough of it,
  7. optionally asks the LLM council for a bounded review (+5 / -15 conviction at most).

Spot is long-only: a LONG conviction opens/extends, a SHORT conviction closes an existing long.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from lunatrade.brains.regime_engine import RegimeState
from lunatrade.core.types import (DIRECTIONAL_FAMILIES, MARKET_WIDE, CategoryScore, Direction, TradeProposal)
from lunatrade.workers.strategy_groups import group_of

MARKET_WIDE_WEIGHT = 0.6   # market-wide evidence counts a bit less than symbol-specific evidence


@dataclass
class Assessment:
    symbol: str
    direction: Direction
    conviction: float
    signed_score: float
    category_scores: list
    families_agreeing: int
    families_opposing: int
    supporting: list = field(default_factory=list)
    opposing: list = field(default_factory=list)
    regime: str = "UNKNOWN"
    expected_move: float = 0.0
    horizon: str = "4h"

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "direction": self.direction.value, "conviction": round(self.conviction, 1),
                "signed_score": round(self.signed_score, 2), "families_agreeing": self.families_agreeing,
                "families_opposing": self.families_opposing, "regime": self.regime,
                "expected_move": self.expected_move, "horizon": self.horizon,
                "categories": [{"family": c.family, "score": round(c.score, 1), "n": c.n_signals,
                                "agreement": round(c.agreement, 2), "weight": c.weight} for c in self.category_scores],
                "supporting": self.supporting[:15], "opposing": self.opposing[:15]}


class LeadBrain:
    def __init__(self, cfg: dict, risk_cfg: dict, council=None, calibration=None):
        self.cfg = cfg
        self.risk_cfg = risk_cfg
        self.council = council
        self.calibration = calibration          # callable(conviction) -> win prob | None
        self.weights = dict(cfg.get("category_weights", {}))
        self.dead_zone = float(cfg.get("dead_zone", 4))

    # ------------------------------------------------------------------ scoring
    def assess(self, symbol: str, signals: list, regime: RegimeState | None, group_weights: dict | None = None,
               min_conviction: float | None = None) -> Assessment:
        gw = group_weights or {}
        by_family: dict[str, list] = defaultdict(list)
        for s in signals:
            if s.family not in DIRECTIONAL_FAMILIES or s.symbol not in (symbol, MARKET_WIDE):
                continue
            by_family[s.family.value].append(s)

        cats = []
        for fam, sigs in by_family.items():
            num = den = 0.0
            agree_pos = agree_neg = 0
            for s in sigs:
                g = group_of(s.worker_id)
                w = float(gw.get(g, 1.0)) * (regime.group_multiplier(g) if regime else 1.0)
                if s.symbol == MARKET_WIDE:
                    w *= MARKET_WIDE_WEIGHT
                num += w * s.score
                den += w
                if s.direction == Direction.LONG:
                    agree_pos += 1
                elif s.direction == Direction.SHORT:
                    agree_neg += 1
            if den <= 0:
                continue
            score = 100 * num / (den + 1.0)   # +1 shrinkage: one weak signal can't max a family out
            directional = agree_pos + agree_neg
            agreement = (max(agree_pos, agree_neg) / directional) if directional else 0.0
            cats.append(CategoryScore(fam, score, len(sigs), agreement, float(self.weights.get(fam, 0.05))))

        total_w = sum(self.weights.values()) or 1.0
        present_w = sum(c.weight for c in cats if abs(c.score) > 0)
        raw = sum(c.weight * c.score for c in cats) / present_w if present_w else 0.0
        coverage = present_w / total_w
        signed = raw * (0.6 + 0.4 * min(1.0, coverage))
        direction = Direction.from_score(signed, self.dead_zone / 2)
        agreeing = sum(1 for c in cats if c.score * direction.sign > self.dead_zone)
        opposing = [c for c in cats if c.score * direction.sign < -max(self.dead_zone * 3, 12)]
        conviction = 50 + abs(signed) / 2 - 3 * len(opposing)
        if agreeing < 2:
            conviction = min(conviction, 55)
        conviction = max(0.0, min(100.0, conviction))

        sup = [s for fam in by_family.values() for s in fam if s.direction.sign == direction.sign and direction.sign]
        opp = [s for fam in by_family.values() for s in fam if s.direction.sign == -direction.sign and direction.sign]
        sup.sort(key=lambda s: -s.confidence)
        opp.sort(key=lambda s: -s.confidence)
        moves = [s.expected_move for s in sup if s.expected_move]
        exp_move = float(sorted(moves)[len(moves) // 2]) if moves else 0.0
        horizons = [s.time_horizon for s in sup[:10]]
        horizon = max(set(horizons), key=horizons.count) if horizons else "4h"
        return Assessment(symbol, direction, conviction, signed, cats, agreeing, len(opposing),
                          [s.worker_id for s in sup], [s.worker_id for s in opp],
                          regime.label() if regime else "UNKNOWN", exp_move, horizon)

    # ------------------------------------------------------------------ decisions
    def propose(self, a: Assessment, price: float, atr_pct: float, equity: float, position_qty: float,
                regime: RegimeState | None, min_conviction: float | None = None) -> TradeProposal | None:
        min_conv = float(min_conviction if min_conviction is not None else self.cfg.get("min_conviction", 62))
        exit_conv = float(self.cfg.get("exit_conviction", 58))
        need_fam = int(self.cfg.get("min_families_agreeing", 3))
        r = self.risk_cfg
        stop_pct = min(float(r.get("max_stop_pct", 0.08)), max(0.003, float(r.get("stop_atr_mult", 2.0)) * atr_pct))
        if position_qty > 0 and a.direction == Direction.SHORT and a.conviction >= exit_conv:
            return self._proposal(a, price, stop_pct, 0.0, "CLOSE", regime,
                                  f"exit: bearish conviction {a.conviction:.0f} >= {exit_conv:.0f}")
        if a.direction != Direction.LONG or a.conviction < min_conv or a.families_agreeing < need_fam:
            return None
        scale = (a.conviction - 50) / 50
        requested = equity * float(r.get("max_position_pct", 0.1)) * max(0.25, min(1.0, scale * 2))
        why = (f"{a.families_agreeing} families agree, conviction {a.conviction:.0f}, regime {a.regime}; "
               f"top evidence: {', '.join(a.supporting[:4])}")
        return self._proposal(a, price, stop_pct, requested, "OPEN", regime, why)

    def _proposal(self, a, price, stop_pct, notional, action, regime, why) -> TradeProposal:
        target_r = float(self.risk_cfg.get("target_r_multiple", 2.0))
        p = TradeProposal(
            symbol=a.symbol, direction=a.direction if action == "OPEN" else Direction.SHORT,
            conviction=round(a.conviction, 1),
            win_probability=self.calibration(a.conviction) if self.calibration else None,
            expected_move=a.expected_move, time_horizon=a.horizon, entry_price=price,
            stop_price=price * (1 - stop_pct), target_price=price * (1 + stop_pct * target_r),
            requested_notional=round(notional, 2),
            category_scores=[{"family": c.family, "score": round(c.score, 1), "n": c.n_signals} for c in a.category_scores],
            supporting_signals=a.supporting[:25], opposing_signals=a.opposing[:25],
            regime=regime.primary.value if regime else "UNKNOWN", rationale=why, action=action)
        return p

    def llm_review(self, p: TradeProposal, assessment: Assessment) -> TradeProposal:
        """Bounded LLM adjustment. The LLM can make the system more cautious, barely more aggressive."""
        if not self.council or not self.cfg.get("llm_review", True) or p.action != "OPEN":
            return p
        res = self.council.ask_json("lead_review", "lead_review", {"proposal": p.to_dict(),
                                                                   "assessment": assessment.to_dict()})
        if not res:
            return p
        up, down = float(self.cfg.get("llm_max_up", 5)), float(self.cfg.get("llm_max_down", 15))
        try:
            adj = float(res.get("adjustment", 0))
        except (TypeError, ValueError):
            adj = 0.0
        adj = max(-down, min(up, adj))
        if res.get("veto") is True:
            adj = -down
        p.llm_review = {"adjustment": adj, "veto": bool(res.get("veto")), "concerns": res.get("concerns", [])[:6],
                        "reason": str(res.get("reason", ""))[:300], "providers": res.get("providers", [])}
        p.conviction = round(max(0.0, min(100.0, p.conviction + adj)), 1)
        return p

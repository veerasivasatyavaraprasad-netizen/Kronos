"""
"What can go wrong?" - mandatory veto layer before every opening trade.

PROPOSED TRADE -> risk factors, correlations, liquidity, news, event risk, stop feasibility, slippage,
counter-signals -> bull case vs bear case -> VETO / APPROVE (+ size reduction).
"""
from __future__ import annotations

from lunatrade.brains.regime_engine import RegimeState
from lunatrade.core.types import MARKET_WIDE, DevilsAdvocateReport, Family, Regime, TradeProposal

REGIME_PENALTY = {Regime.PANIC: 3.0, Regime.ILLIQUID: 2.5, Regime.UNKNOWN: 1.5, Regime.EVENT_DRIVEN: 1.0,
                  Regime.HIGH_VOL: 1.0}


class DevilsAdvocate:
    def __init__(self, cfg: dict, council=None):
        self.cfg = cfg
        self.council = council

    def review(self, p: TradeProposal, signals: list, regime: RegimeState | None,
               portfolio: dict | None = None) -> DevilsAdvocateReport:
        risks, bear = [], 0.0
        for s in signals:
            if s.symbol not in (p.symbol, MARKET_WIDE):
                continue
            if s.family == Family.ADVERSARIAL:
                bear += float(s.context.get("risk_points", 0))
                risks += s.context.get("reasons", [])
            if s.context.get("depeg_risk"):
                bear += 3
                risks.append(f"stablecoin depeg risk ({s.context.get('pair')})")
            if s.context.get("risk_off") and s.symbol == MARKET_WIDE:
                bear += 0.7
                risks.append("macro risk-off")
            if s.context.get("system_degraded"):
                bear += 1.5
                risks.append("several strategy groups degraded")
            if s.context.get("fast_drawdown") and s.symbol == p.symbol:
                bear += 1.0
                risks.append("fast drawdown in progress")

        if regime is not None:
            pen = REGIME_PENALTY.get(regime.primary, 0.0)
            if pen:
                bear += pen
                risks.append(f"regime {regime.primary.value}")
            if regime.trend_bias == "BEAR":
                bear += 0.8
                risks.append("bearish regime bias")
            if regime.scores.get(Regime.PANIC.value, 0) >= 0.5 and regime.primary != Regime.PANIC:
                bear += 2.0
                risks.append("market-wide panic")

        # correlated book: adding another risk-on bet to an already crowded cluster
        if portfolio:
            clusters = next((s.context.get("clusters") for s in signals if s.setup == "CORRELATION_CLUSTERS"), None) or []
            held = set((portfolio.get("positions") or {}).keys())
            for c in clusters:
                if p.symbol in c and len(held & set(c) - {p.symbol}) >= 2:
                    bear += 1.0
                    risks.append(f"correlated with {len(held & set(c) - {p.symbol})} open positions")

        bull = max(0.0, min(10.0, (p.conviction - 50) / 5))   # conviction 100 -> 10
        llm_note = ""
        if self.council and self.cfg.get("llm", True):
            res = self.council.ask_json("devils_advocate", "devils_advocate", {
                "trade": f"{p.symbol} {p.direction.value}", "conviction": p.conviction, "regime": p.regime,
                "supporting": p.supporting_signals[:10], "opposing": p.opposing_signals[:10],
                "hidden_risks": sorted(set(risks))[:15], "stop_pct": 1 - p.stop_price / p.entry_price})
            if res:
                try:
                    extra = max(0.0, min(2.0, float(res.get("extra_bear_points", 0))))
                except (TypeError, ValueError):
                    extra = 0.0
                bear += extra
                risks += [f"LLM: {r}" for r in (res.get("hidden_risks") or [])[:4]]
                llm_note = str(res.get("reason", ""))[:300]
                if res.get("veto") is True:
                    bear += 1.0
        bear = min(10.0, bear)
        margin = float(self.cfg.get("veto_margin", 2.5))
        veto = bear - bull >= margin or any("depeg" in r for r in risks)
        max_adj = abs(float(self.cfg.get("max_adjustment", -0.6)))
        adjustment = -min(max_adj, max(0.0, bear / 10 * max_adj * 1.2 - bull / 100))
        return DevilsAdvocateReport(p.id, round(bull, 2), round(bear, 2), sorted(set(risks))[:20], veto,
                                    round(adjustment, 3), llm_note)

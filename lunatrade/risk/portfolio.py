"""
Portfolio Brain - stops many agents from building one giant correlated bet.

BTC +$500, ETH +$400, SOL +$300, AVAX +$250 is effectively one large risk-on position. The portfolio
brain measures beta-weighted and cluster exposure and scales the new order down (or to zero).
"""
from __future__ import annotations

from lunatrade.core.types import RiskDecision, TradeProposal


class PortfolioBrain:
    def __init__(self, risk_cfg: dict):
        self.cfg = risk_cfg

    def cluster_of(self, symbol: str, clusters: list[list[str]]) -> list[str]:
        return next((c for c in clusters if symbol in c), [symbol])

    def adjust(self, p: TradeProposal, d: RiskDecision, equity: float, positions: dict,
               clusters: list[list[str]], betas: dict[str, float]) -> RiskDecision:
        if not d.approved or p.action != "OPEN" or d.approved_notional <= 0:
            return d
        eq = max(equity, 1e-9)
        size = d.approved_notional
        notes = []
        # 1) correlation cluster cap
        cluster = self.cluster_of(p.symbol, clusters)
        in_cluster = sum(abs(positions.get(s, {}).get("notional", 0)) for s in cluster)
        cap = eq * float(self.cfg.get("max_correlated_exposure_pct", 0.35))
        room = cap - in_cluster
        if size > room:
            notes.append(f"cluster {'/'.join(cluster[:4])} exposure {in_cluster / eq:.0%} -> capped")
            size = max(0.0, room)
        # 2) beta-weighted (effective risk-on) exposure cap
        beta_expo = sum(abs(x.get("notional", 0)) * betas.get(s, 1.0) for s, x in positions.items())
        b_new = betas.get(p.symbol, 1.0)
        b_cap = eq * float(self.cfg.get("max_portfolio_exposure_pct", 0.6))
        if beta_expo + size * b_new > b_cap:
            allowed = max(0.0, (b_cap - beta_expo) / max(b_new, 0.1))
            if allowed < size:
                notes.append(f"beta-weighted exposure {beta_expo / eq:.0%} -> capped")
                size = allowed
        d.checks["portfolio_correlation"] = "OK" if not notes else "WARNING"
        d.reasons += notes
        min_order = float(self.cfg.get("min_order_notional", 10))
        if size < min_order:
            d.approved = False
            d.checks["portfolio_correlation"] = "FAIL: correlated exposure limit"
            d.reasons.insert(0, "portfolio: correlated exposure limit reached")
            size = 0.0
        d.approved_notional = round(size, 2)
        return d

    @staticmethod
    def snapshot(equity: float, cash: float, positions: dict, prices: dict) -> dict:
        pos = {}
        for s, x in positions.items():
            px = prices.get(s) or x.get("entry", 0)
            pos[s] = {**x, "price": px, "notional": x.get("qty", 0) * px,
                      "unrealized": (px - x.get("entry", 0)) * x.get("qty", 0)}
        return {"equity": equity, "cash": cash, "positions": pos,
                "exposure": sum(p["notional"] for p in pos.values()),
                "exposure_pct": sum(p["notional"] for p in pos.values()) / equity if equity else 0.0}

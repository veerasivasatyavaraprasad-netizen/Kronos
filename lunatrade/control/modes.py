"""
Operating modes and the promotion pipeline.

    RESEARCH -> PAPER -> SHADOW -> APPROVAL -> LIVE
    (development -> unit tests -> backtest -> walk-forward -> paper -> shadow live -> human-approved live
     -> limited live -> full automation)

Promotion is never automatic: `check_promotion` reports whether the evidence gates are met and a human
must confirm the switch through the authenticated control API. Demotion (towards safety) is always allowed.
"""
from __future__ import annotations

import datetime as dt

from lunatrade.config import live_trading_allowed
from lunatrade.core.types import Mode
from lunatrade.memory.performance import trade_stats

ORDER = [Mode.RESEARCH, Mode.PAPER, Mode.SHADOW, Mode.APPROVAL, Mode.LIVE]


def check_promotion(current: Mode, target: Mode, outcomes: list[dict], first_trade_at: dt.datetime | None,
                    gates: dict, max_drawdown_pct: float) -> dict:
    if ORDER.index(target) <= ORDER.index(current):
        return {"allowed": True, "reasons": ["demotion / same mode is always allowed"], "gates": {}}
    if ORDER.index(target) - ORDER.index(current) > 1:
        return {"allowed": False, "reasons": ["promote one step at a time"], "gates": {}}
    ts = trade_stats(outcomes)
    days = (dt.datetime.utcnow() - first_trade_at).days if first_trade_at else 0
    checks = {
        "min_trades": (ts.get("trades", 0) >= gates.get("min_trades", 50), f"{ts.get('trades', 0)} trades"),
        "min_days": (days >= gates.get("min_days", 14), f"{days} days"),
        "profit_factor": (ts.get("profit_factor", 0) >= gates.get("min_profit_factor", 1.2),
                          f"PF {ts.get('profit_factor', 0):.2f}"),
        "expectancy_r": (ts.get("expectancy_r", -1) >= gates.get("min_expectancy_r", 0.05),
                         f"{ts.get('expectancy_r', 0):.3f}R"),
        "drawdown": (max_drawdown_pct <= gates.get("max_drawdown_pct", 0.10), f"max DD {max_drawdown_pct:.1%}"),
    }
    if target == Mode.PAPER:
        checks = {}          # research -> paper needs no track record
    if target == Mode.LIVE:
        checks["live_switch"] = (live_trading_allowed(), "LUNATRADE_ALLOW_LIVE=yes in environment")
    failed = [f"{k}: {v[1]}" for k, v in checks.items() if not v[0]]
    return {"allowed": not failed, "reasons": failed or ["all gates met - human confirmation required"],
            "gates": {k: {"ok": v[0], "value": v[1]} for k, v in checks.items()}}

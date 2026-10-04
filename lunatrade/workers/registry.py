"""
Builds the 300-worker roster.

    Market scanners        35     Social sentiment     20     Regime detection        10
    Technical / setups     60     On-chain             25     Forecast (Kronos+stat)   5
    Momentum / volatility  25     Whale tracking       20     Risk analysis           10
    Order book / micro     20     Macro                15     Strategy evaluation     10
    News                   20     Arbitrage            15     Portfolio analysis       5
                                                              Adversarial              5
                                                              -------------------------
                                                              TOTAL                  300
"""
from __future__ import annotations

from collections import Counter

from lunatrade.workers import (arbitrage, forecast, macro, meta, microstructure, momentum, news, onchain, regime,
                               scanner, social, technical)
from lunatrade.workers.base import FnWorker

EXPECTED = {"scanner": 35, "technical": 60, "momentum": 13, "volatility": 12, "microstructure": 20, "news": 20,
            "social": 20, "onchain": 25, "whale": 20, "macro": 15, "arbitrage": 15, "regime": 10, "forecast": 5,
            "risk": 10, "strategy_eval": 10, "portfolio": 5, "adversarial": 5}
TOTAL = 300


def all_specs(cfg=None) -> list[dict]:
    kronos_cfg = cfg.section("kronos") if cfg is not None else {}
    return (scanner.specs() + technical.specs() + momentum.specs() + microstructure.specs() + news.specs()
            + social.specs() + onchain.specs() + macro.specs() + arbitrage.specs() + regime.specs()
            + forecast.specs(kronos_cfg) + meta.specs())


def build_roster(cfg=None) -> list[FnWorker]:
    workers, seen = [], set()
    for s in all_specs(cfg):
        wid = f"{s['family'].value}.{s['name']}"
        if wid in seen:
            raise ValueError(f"duplicate worker id {wid}")
        seen.add(wid)
        workers.append(FnWorker(wid, s["name"], s["family"], s["fn"], params=s.get("params"),
                                per_symbol=s.get("per_symbol", True), min_bars=s.get("min_bars", 60),
                                phase=s.get("phase", 1), every_bars=s.get("every_bars", 1),
                                requires=s.get("requires", ()), availability=s.get("availability")))
    return workers


def roster_summary(workers) -> dict:
    return dict(Counter(w.family.value for w in workers))

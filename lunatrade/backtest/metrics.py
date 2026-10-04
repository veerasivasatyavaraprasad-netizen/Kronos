"""Performance metrics. The target is robust risk-adjusted expectancy, not a flattering win rate."""
from __future__ import annotations

import math

import numpy as np


def equity_metrics(equity: list[float], bars_per_year: float) -> dict:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                "volatility": 0.0}
    r = np.diff(eq) / eq[:-1]
    total = eq[-1] / eq[0] - 1
    years = len(r) / bars_per_year
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else -1.0
    vol = r.std() * math.sqrt(bars_per_year)
    sharpe = r.mean() / r.std() * math.sqrt(bars_per_year) if r.std() > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / down.std() * math.sqrt(bars_per_year) if len(down) > 1 and down.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max())
    return {"total_return": float(total), "cagr": float(cagr), "sharpe": float(sharpe), "sortino": float(sortino),
            "max_drawdown": dd, "calmar": float(cagr / dd) if dd > 0 else 0.0, "volatility": float(vol)}


def trade_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "avg_r": 0.0, "tail_loss": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "fees": 0.0, "avg_bars_held": 0.0}
    pnl = np.array([t["pnl"] for t in trades])
    rets = np.array([t["return_pct"] for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    q = np.percentile(rets, 5)
    return {
        "trades": len(trades), "win_rate": float(len(wins) / len(pnl)),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else (float("inf") if len(wins) else 0.0),
        "expectancy": float(pnl.mean()), "avg_r": float(np.mean([t.get("r_multiple", 0) for t in trades])),
        "tail_loss": float(rets[rets <= q].mean()),     # mean of the worst 5% trades (CVaR-style)
        "avg_win": float(wins.mean()) if len(wins) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "fees": float(sum(t.get("fees", 0) for t in trades)),
        "avg_bars_held": float(np.mean([t.get("bars_held", 0) for t in trades])),
    }


def full_report(equity: list[float], trades: list[dict], bars_per_year: float, orders: list[dict] | None = None,
                exposure: list[float] | None = None) -> dict:
    m = {**equity_metrics(equity, bars_per_year), **trade_metrics(trades)}
    if orders:
        filled = [o for o in orders if o.get("status") == "POSITION_UPDATED"]
        m["turnover"] = float(sum(o.get("filled_qty", 0) * o.get("avg_price", 0) for o in filled) / max(equity[0], 1e-9))
        slips = [o.get("slippage_bps", 0) for o in filled]
        m["avg_slippage_bps"] = float(np.mean(slips)) if slips else 0.0
    if exposure:
        m["time_in_market"] = float(np.mean([e > 0 for e in exposure]))
    return m

"""
Robustness suite: walk-forward, out-of-sample, Monte Carlo and stress scenarios.

    Historical data -> strategy -> backtest -> walk-forward -> out-of-sample -> Monte Carlo -> stress test
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lunatrade.backtest.engine import Backtester
from lunatrade.backtest.metrics import trade_metrics


def walk_forward(cfg, data: dict[str, pd.DataFrame], folds: int = 3, warmup: int = 200, train_frac: float = 0.6,
                 **kw) -> dict:
    """
    Anchored walk-forward: for each fold the meta-learner trains on the in-sample window, then its strategy
    weights are frozen and evaluated on the following out-of-sample window it has never seen.
    """
    n = min(len(df) for df in data.values())
    usable = n - warmup
    seg = usable // folds
    out = []
    for k in range(folds):
        start = warmup + k * seg
        end = start + seg
        split = start + int(seg * train_frac)
        if split - start < 10 or end - split < 10:
            continue
        ins = Backtester(cfg, data, warmup=warmup, start_index=start, end_index=split, learn=True, **kw).run()
        oos = Backtester(cfg, data, warmup=warmup, start_index=split, end_index=end, learn=False,
                         initial_weights=ins.weights, **kw).run()
        out.append({"fold": k, "in_sample": _brief(ins.metrics), "out_of_sample": _brief(oos.metrics),
                    "weights": ins.weights})
    oos_returns = [f["out_of_sample"]["total_return"] for f in out]
    return {"folds": out, "oos_mean_return": float(np.mean(oos_returns)) if oos_returns else 0.0,
            "oos_positive_folds": int(sum(r > 0 for r in oos_returns)),
            "degradation": float(np.mean([f["in_sample"]["sharpe"] - f["out_of_sample"]["sharpe"] for f in out]))
            if out else 0.0}


def _brief(m: dict) -> dict:
    keys = ("total_return", "sharpe", "sortino", "max_drawdown", "trades", "win_rate", "profit_factor", "expectancy",
            "avg_r", "tail_loss")
    return {k: m.get(k) for k in keys}


def monte_carlo(trades: list[dict], starting_equity: float = 10_000, runs: int = 2000, seed: int = 42) -> dict:
    """Bootstrap trade returns (with replacement) to see the spread of outcomes the same edge could produce."""
    if len(trades) < 5:
        return {"runs": 0, "note": "need at least 5 trades"}
    rng = np.random.default_rng(seed)
    pnl = np.array([t["pnl"] for t in trades])
    finals, dds = [], []
    for _ in range(runs):
        sample = rng.choice(pnl, size=len(pnl), replace=True)
        eq = starting_equity + np.cumsum(sample)
        peak = np.maximum.accumulate(np.r_[starting_equity, eq])[1:]
        finals.append(eq[-1])
        dds.append(float(((peak - eq) / peak).max()))
    finals, dds = np.array(finals), np.array(dds)
    pct = lambda a, q: float(np.percentile(a, q))
    return {"runs": runs, "final_equity_p5": pct(finals, 5), "final_equity_p50": pct(finals, 50),
            "final_equity_p95": pct(finals, 95), "prob_loss": float((finals < starting_equity).mean()),
            "max_drawdown_p50": pct(dds, 50), "max_drawdown_p95": pct(dds, 95), "max_drawdown_p99": pct(dds, 99)}


# ------------------------------------------------------------------------------------- stress

def _flash_crash(df: pd.DataFrame, at: float = 0.7, depth: float = 0.25) -> pd.DataFrame:
    df = df.copy()
    i = int(len(df) * at)
    for c in ("open", "high", "low", "close"):
        df.loc[df.index[i:], c] = df[c].iloc[i:] * (1 - depth * 0.6)   # partial recovery after the wick
    df.loc[df.index[i], "low"] = df["close"].iloc[i - 1] * (1 - depth)
    df.loc[df.index[i], "volume"] *= 15
    return df


def _scale_vol(df: pd.DataFrame, k: float) -> pd.DataFrame:
    df = df.copy()
    c = df["close"].to_numpy()
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    new = c[0] * np.exp(np.cumsum(r * k))
    scale = new / c
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].to_numpy() * scale
    df["high"] = np.maximum(df["high"], df[["open", "close"]].max(axis=1))
    df["low"] = np.minimum(df["low"], df[["open", "close"]].min(axis=1))
    return df


def _drift(df: pd.DataFrame, per_bar: float) -> pd.DataFrame:
    df = df.copy()
    f = np.exp(per_bar * np.arange(len(df)))
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].to_numpy() * f
    return df


def _thin(df: pd.DataFrame, k: float = 0.02) -> pd.DataFrame:
    df = df.copy()
    for col in ("volume", "quote_volume", "taker_buy_base", "trades"):
        if col in df:
            df[col] = df[col] * k
    return df


def _outage(df: pd.DataFrame, at: float = 0.6, bars: int = 30) -> pd.DataFrame:
    i = int(len(df) * at)
    return df.drop(df.index[i:i + bars]).reset_index(drop=True)


SCENARIOS = {
    "baseline": (lambda df: df, {}),
    "bull_market": (lambda df: _drift(df, 0.0008), {}),
    "bear_market": (lambda df: _drift(df, -0.0008), {}),
    "sideways": (lambda df: _drift(df, -np.log(df["close"].iloc[-1] / df["close"].iloc[0]) / len(df)), {}),
    "flash_crash": (_flash_crash, {}),
    "high_volatility": (lambda df: _scale_vol(df, 2.5), {}),
    "low_liquidity": (_thin, {}),
    "large_spread": (lambda df: df, {"extra_slippage_bps": 40}),
    "api_outage": (_outage, {}),
    "partial_fills": (lambda df: df, {"partial_fill_probability": 0.5}),
    "slippage_shock": (lambda df: df, {"extra_slippage_bps": 120}),
}


def stress_test(cfg, data: dict[str, pd.DataFrame], scenarios: list[str] | None = None, warmup: int = 200,
                **kw) -> dict:
    out = {}
    for name in scenarios or list(SCENARIOS):
        transform, broker = SCENARIOS[name]
        d = {s: transform(df) for s, df in data.items()}
        try:
            res = Backtester(cfg, d, warmup=warmup, broker_overrides=broker, **kw).run()
            out[name] = {**_brief(res.metrics), "kill_switch_events": res.metrics.get("kill_switch_events", 0)}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def summarize_trades(trades: list[dict]) -> dict:
    return trade_metrics(trades)

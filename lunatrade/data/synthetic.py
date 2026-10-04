"""Synthetic market generator for tests, demos and stress scenarios (regime-switching GBM)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lunatrade.data.market_store import INTERVAL_MS


def generate(n: int = 1000, interval: str = "15m", start_price: float = 100.0, seed: int = 7,
             regimes: list[tuple[int, float, float]] | None = None, start_ms: int = 1_700_000_000_000,
             base_volume: float = 1000.0) -> pd.DataFrame:
    """
    regimes: list of (bars, drift_per_bar, vol_per_bar). Default cycles trend-up, range, trend-down, panic.
    Returns a frame with open_time + candle fields, including taker_buy_base.
    """
    rng = np.random.default_rng(seed)
    if regimes is None:
        regimes = [(250, 0.0012, 0.006), (250, 0.0, 0.004), (250, -0.0010, 0.007), (250, 0.0003, 0.012)]
    drift = np.concatenate([np.full(b, d) for b, d, _ in regimes])
    vol = np.concatenate([np.full(b, v) for b, _, v in regimes])
    reps = int(np.ceil(n / len(drift)))
    drift, vol = np.tile(drift, reps)[:n], np.tile(vol, reps)[:n]
    rets = drift + vol * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.r_[start_price, close[:-1]]
    wick = np.abs(rng.standard_normal(n)) * vol * close * 0.6
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = base_volume * (1 + 3 * np.abs(rets) / vol.mean()) * rng.uniform(0.7, 1.3, n)
    buy_share = np.clip(0.5 + 8 * rets + rng.normal(0, 0.05, n), 0.05, 0.95)
    step = INTERVAL_MS[interval]
    return pd.DataFrame({
        "open_time": start_ms + np.arange(n, dtype=np.int64) * step,
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "quote_volume": volume * close, "trades": (volume / 2).round(), "taker_buy_base": volume * buy_share,
    })


def correlated_universe(symbols: list[str], n: int = 800, interval: str = "15m", seed: int = 11,
                        beta: float = 0.8) -> dict[str, pd.DataFrame]:
    """A market factor plus idiosyncratic noise, so correlation-aware components have something to see."""
    base = generate(n, interval, seed=seed)
    mret = np.diff(np.log(base["close"].to_numpy()), prepend=np.log(base["close"].iloc[0]))
    out = {}
    rng = np.random.default_rng(seed + 1)
    for i, sym in enumerate(symbols):
        idio = rng.normal(0, 0.004, n)
        r = beta * mret + idio if i else mret
        start = 100.0 * (i + 1)
        close = start * np.exp(np.cumsum(r))
        df = base.copy()
        scale = close / base["close"].to_numpy()
        for c in ("open", "high", "low", "close"):
            df[c] = base[c].to_numpy() * scale
        df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
        df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))
        df["quote_volume"] = df["volume"] * df["close"]
        out[sym] = df
    return out

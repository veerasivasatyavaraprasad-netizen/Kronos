"""
Forecast workers (5): the Kronos foundation model (this repo's model/) plus three statistical forecasters.

Kronos runs only when torch is installed and the weights can be loaded; otherwise those two workers
report OFFLINE and the statistical forecasters carry the family.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd

from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.data.market_store import from_ms
from lunatrade.workers import common as K
from lunatrade.workers.base import clamp, spec

log = logging.getLogger("lunatrade.kronos")
_PREDICTORS: dict = {}
_LOCK = threading.Lock()
_FAILED: dict = {}


def _kronos_available(w, ctx):
    if not w.params.get("enabled", True):
        return False, "disabled in config"
    key = w.params["model"]
    if key in _FAILED:
        return False, _FAILED[key]
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "torch not installed (pip install torch) - Kronos offline"
    return True, ""


def _predictor(key: str):
    with _LOCK:
        if key not in _PREDICTORS:
            try:
                from automation.forecaster import load_predictor

                _PREDICTORS[key] = load_predictor(key, "auto")
            except Exception as e:
                _FAILED[key] = f"model load failed: {e}"[:200]
                raise
        return _PREDICTORS[key]


def kronos(w, ctx, sym):
    from automation.forecaster import forecast

    c = ctx.view.candles(sym, w.params["lookback"])
    if len(c) < 64:
        return None
    df = pd.DataFrame({"timestamps": [from_ms(t) for t in c.open_time], "open": c.open, "high": c.high, "low": c.low,
                       "close": c.close, "volume": c.volume, "amount": c.quote_volume})
    pred = _predictor(w.params["model"])
    mean_df, paths = forecast(pred, df, len(df), w.params["pred_len"], sample_count=w.params["samples"], seed=7)
    last = c.close[-1]
    change = mean_df["close"].iloc[-1] / last - 1
    up_share = float((paths[:, -1, 3] > last).mean())
    agree = up_share if change > 0 else 1 - up_share
    a = K.atr_pct(ctx, sym) or 1e-6
    if abs(change) < 0.5 * a:
        return ctx.signal(w, sym, D.NEUTRAL, 0.0, setup="KRONOS_FLAT", context={"forecast_change": change})
    d = D.from_score(change)
    conf = clamp((agree - 0.5) * 2) * clamp(abs(change) / (2 * a))
    return ctx.signal(w, sym, d, conf, setup="KRONOS_FORECAST", expected_move=float(change),
                      time_horizon=f"{w.params['pred_len']}bars",
                      evidence=[f"kronos {w.params['model']} {change:+.2%}", f"{agree:.0%} paths agree"],
                      invalidations=["forecast_path_breaks"], context={"paths_agree": agree, "forecast_change": change})


def linear_trend(w, ctx, sym):
    n = w.params["n"]
    y = np.log(ctx.view.candles(sym, n).close)
    if len(y) < n:
        return None
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1 - resid.var() / (y.var() or 1)
    h = w.params["horizon"]
    change = float(np.exp(slope * h) - 1)
    if r2 < 0.3:
        return None
    d = D.from_score(change)
    return ctx.signal(w, sym, d, clamp(r2 * 0.6), setup="REGRESSION_TREND", expected_move=change,
                      time_horizon=f"{h}bars", context={"r2": r2, "slope": slope})


def holt(w, ctx, sym):
    a_, b_ = w.params["alpha"], w.params["beta"]
    y = ctx.view.candles(sym, 200).close
    if len(y) < 50:
        return None
    level, trend = y[0], y[1] - y[0]
    for v in y[1:]:
        prev = level
        level = a_ * v + (1 - a_) * (level + trend)
        trend = b_ * (level - prev) + (1 - b_) * trend
    h = w.params["horizon"]
    fc = level + h * trend
    change = fc / y[-1] - 1
    a = K.atr_pct(ctx, sym) or 1e-6
    if abs(change) < a:
        return None
    d = D.from_score(change)
    return ctx.signal(w, sym, d, clamp(abs(change) / (4 * a)), setup="HOLT_FORECAST", expected_move=float(change),
                      time_horizon=f"{h}bars", context={"holt_change": float(change)})


def knn_analog(w, ctx, sym):
    """Find the k most similar past return windows and average what happened next."""
    m, k, h = w.params["window"], w.params["k"], w.params["horizon"]
    c = ctx.view.candles(sym).close
    if len(c) < m + h + 100:
        return None
    r = np.diff(np.log(c))
    cur = r[-m:]
    cur = (cur - cur.mean()) / (cur.std() or 1)
    hist_end = len(r) - h   # windows whose outcome is fully known
    idx = np.arange(m, hist_end - m)
    if len(idx) < k * 3:
        return None
    wins = np.lib.stride_tricks.sliding_window_view(r[:hist_end], m)[: len(idx)]
    mu, sd = wins.mean(axis=1, keepdims=True), wins.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    dist = np.sqrt((((wins - mu) / sd - cur) ** 2).sum(axis=1))
    best = np.argsort(dist)[:k]
    outcomes = np.array([r[i + m:i + m + h].sum() for i in best])
    exp, hit = float(np.expm1(outcomes.mean())), float((np.sign(outcomes) == np.sign(outcomes.mean())).mean())
    if hit < 0.6:
        return None
    d = D.from_score(exp)
    return ctx.signal(w, sym, d, clamp((hit - 0.5) * 2), setup="PATTERN_ANALOG", expected_move=exp,
                      time_horizon=f"{h}bars", context={"analog_hit_rate": hit, "analogs": k})


def specs(kronos_cfg: dict | None = None) -> list[dict]:
    F = Family.FORECAST
    kc = kronos_cfg or {}
    models = (kc.get("models") or ["kronos-small", "kronos-mini"])[:2]
    while len(models) < 2:
        models.append("kronos-mini")
    out = [spec(F, f"kronos_{m.split('-')[-1]}", kronos,
                params={"model": m, "lookback": int(kc.get("lookback", 400)), "pred_len": int(kc.get("pred_len", 8)),
                        "samples": int(kc.get("sample_count", 5)), "enabled": kc.get("enabled", True)},
                min_bars=64, every_bars=int(kc.get("every_bars", 4)), availability=_kronos_available)
           for m in models]
    out += [spec(F, "linear_trend_48", linear_trend, params={"n": 48, "horizon": 8}),
            spec(F, "holt_es", holt, params={"alpha": 0.3, "beta": 0.1, "horizon": 8}),
            spec(F, "knn_analog", knn_analog, params={"window": 12, "k": 15, "horizon": 8}, min_bars=150)]
    return out

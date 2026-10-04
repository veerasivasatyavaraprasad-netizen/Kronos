"""
Vectorised technical indicators on numpy arrays (no TA-Lib dependency).

All functions return arrays aligned with the input; leading values that cannot be computed are NaN.
Every value at index i only uses inputs at indices <= i (no look-ahead).
"""
from __future__ import annotations

import numpy as np

NAN = np.nan


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sma(x, n: int) -> np.ndarray:
    x = _arr(x)
    out = np.full_like(x, NAN)
    if len(x) < n or n <= 0:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def ema(x, n: int) -> np.ndarray:
    x = _arr(x)
    out = np.full_like(x, NAN)
    if len(x) < n or n <= 0:
        return out
    alpha = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def rma(x, n: int) -> np.ndarray:
    """Wilder's smoothing."""
    x = _arr(x)
    out = np.full_like(x, NAN)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def rolling_std(x, n: int) -> np.ndarray:
    x = _arr(x)
    out = np.full_like(x, NAN)
    if len(x) < n:
        return out
    w = np.lib.stride_tricks.sliding_window_view(x, n)
    out[n - 1:] = w.std(axis=1)
    return out


def rolling_max(x, n: int) -> np.ndarray:
    x = _arr(x)
    out = np.full_like(x, NAN)
    if len(x) < n:
        return out
    out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).max(axis=1)
    return out


def rolling_min(x, n: int) -> np.ndarray:
    x = _arr(x)
    out = np.full_like(x, NAN)
    if len(x) < n:
        return out
    out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).min(axis=1)
    return out


def zscore(x, n: int) -> np.ndarray:
    m, s = sma(x, n), rolling_std(x, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, (_arr(x) - m) / s, 0.0)


def returns(close) -> np.ndarray:
    c = _arr(close)
    out = np.full_like(c, NAN)
    out[1:] = c[1:] / c[:-1] - 1
    return out


def log_returns(close) -> np.ndarray:
    c = _arr(close)
    out = np.full_like(c, NAN)
    out[1:] = np.log(c[1:] / c[:-1])
    return out


def rsi(close, n: int = 14) -> np.ndarray:
    c = _arr(close)
    d = np.diff(c, prepend=c[0])
    gain, loss = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = rma(gain[1:], n), rma(loss[1:], n)
    out = np.full_like(c, NAN)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(al > 0, ag / al, np.inf)
        out[1:] = np.where(np.isnan(ag), NAN, 100 - 100 / (1 + rs))
    return out


def true_range(high, low, close) -> np.ndarray:
    h, l, c = _arr(high), _arr(low), _arr(close)
    prev = np.roll(c, 1)
    prev[0] = c[0]
    return np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])


def atr(high, low, close, n: int = 14) -> np.ndarray:
    return rma(true_range(high, low, close), n)


def adx(high, low, close, n: int = 14):
    """Returns (adx, +DI, -DI)."""
    h, l = _arr(high), _arr(low)
    up = np.diff(h, prepend=h[0])
    down = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = rma(true_range(high, low, close), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * rma(plus_dm, n) / tr
        mdi = 100 * rma(minus_dm, n) / tr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    dx = np.nan_to_num(dx, nan=0.0)
    out = np.full_like(h, NAN)
    valid = ~np.isnan(tr)
    if valid.sum() >= n:
        first = np.argmax(valid)
        out[first:] = rma(dx[first:], n)
    return out, pdi, mdi


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    valid = ~np.isnan(line)
    sig = np.full_like(line, NAN)
    if valid.sum() >= signal:
        first = np.argmax(valid)
        sig[first:] = ema(line[first:], signal)
    return line, sig, line - sig


def bollinger(close, n: int = 20, k: float = 2.0):
    m, s = sma(close, n), rolling_std(close, n)
    return m - k * s, m, m + k * s


def keltner(high, low, close, n: int = 20, mult: float = 1.5):
    mid = ema(close, n)
    a = atr(high, low, close, n)
    return mid - mult * a, mid, mid + mult * a


def vwap(high, low, close, volume, n: int | None = None) -> np.ndarray:
    """Rolling VWAP over n bars (cumulative when n is None)."""
    tp = (_arr(high) + _arr(low) + _arr(close)) / 3
    v = _arr(volume)
    if n is None:
        cv = np.cumsum(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cv > 0, np.cumsum(tp * v) / cv, tp)
    pv, vv = sma(tp * v, n), sma(v, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vv > 0, pv / vv, tp)


def stochastic(high, low, close, n: int = 14, d: int = 3):
    hh, ll = rolling_max(high, n), rolling_min(low, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(hh > ll, 100 * (_arr(close) - ll) / (hh - ll), 50.0)
    k = np.where(np.isnan(hh), NAN, k)
    return k, sma(np.nan_to_num(k, nan=50.0), d)


def obv(close, volume) -> np.ndarray:
    c, v = _arr(close), _arr(volume)
    sign = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(sign * v)


def roc(close, n: int) -> np.ndarray:
    c = _arr(close)
    out = np.full_like(c, NAN)
    out[n:] = c[n:] / c[:-n] - 1
    return out


def efficiency_ratio(close, n: int = 20) -> np.ndarray:
    """Kaufman efficiency ratio: 1 = straight trend, 0 = pure noise."""
    c = _arr(close)
    out = np.full_like(c, NAN)
    if len(c) <= n:
        return out
    change = np.abs(c[n:] - c[:-n])
    vol = np.convolve(np.abs(np.diff(c)), np.ones(n), "valid")
    with np.errstate(divide="ignore", invalid="ignore"):
        out[n:] = np.where(vol > 0, change / vol, 0.0)
    return out


def realized_vol(close, n: int = 30) -> np.ndarray:
    return rolling_std(np.nan_to_num(log_returns(close)), n)


def parkinson_vol(high, low, n: int = 30) -> np.ndarray:
    hl = np.log(_arr(high) / _arr(low)) ** 2
    return np.sqrt(sma(hl, n) / (4 * np.log(2)))


def percentile_rank(x, n: int) -> np.ndarray:
    """Rank of the last value within its trailing window, 0..1."""
    x = _arr(x)
    out = np.full_like(x, NAN)
    if len(x) < n:
        return out
    w = np.lib.stride_tricks.sliding_window_view(x, n)
    out[n - 1:] = (w < w[:, -1:]).sum(axis=1) / (n - 1)
    return out


def hurst(x, max_lag: int = 20) -> float:
    """Hurst exponent of the series: <0.5 mean reverting, >0.5 trending."""
    x = _arr(x)
    x = x[~np.isnan(x)]
    if len(x) < max_lag * 4:
        return 0.5
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    tau = np.where(tau <= 0, 1e-12, tau)
    slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def pivots(high, low, left: int = 3, right: int = 3):
    """Confirmed swing highs/lows (confirmation needs `right` later bars, so no look-ahead at use time)."""
    h, l = _arr(high), _arr(low)
    sh, sl = [], []
    for i in range(left, len(h) - right):
        if h[i] == h[i - left:i + right + 1].max():
            sh.append(i)
        if l[i] == l[i - left:i + right + 1].min():
            sl.append(i)
    return sh, sl


def last(x, default: float = NAN) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return default
    v = x[-1]
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def corr(a, b) -> float:
    a, b = _arr(a), _arr(b)
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    a, b = a[-n:], b[-n:]
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])

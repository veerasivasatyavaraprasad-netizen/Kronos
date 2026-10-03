"""
Market data loaders that all return a normalized K-line DataFrame with columns:
    timestamps, open, high, low, close, volume, amount
"""
import time
from pathlib import Path

import pandas as pd
import requests

KLINE_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def normalize_kline(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names/types and sort by time."""
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]

    for alias in ("timestamp", "date", "datetime", "time"):
        if "timestamps" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "timestamps"})
    if "timestamps" not in df.columns:
        raise ValueError("Data needs a time column (timestamps/timestamp/date/datetime).")

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=False)
    if getattr(df["timestamps"].dt, "tz", None) is not None:
        df["timestamps"] = df["timestamps"].dt.tz_localize(None)

    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

    df = df[KLINE_COLUMNS].dropna()
    df = df.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)
    return df


def load_csv(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".feather":
        return normalize_kline(pd.read_feather(path))
    return normalize_kline(pd.read_csv(path, encoding="utf-8-sig"))


def fetch_yfinance(symbol: str, interval: str = "1d", period: str = "2y") -> pd.DataFrame:
    """Stocks, ETFs, indices, FX and crypto via Yahoo Finance (e.g. AAPL, ^GSPC, BTC-USD, RELIANCE.NS)."""
    import yfinance as yf

    raw = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} ({interval}, {period})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw = raw.rename(columns={raw.columns[0]: "timestamps"})
    return normalize_kline(raw)


BINANCE_HOSTS = ("https://api.binance.com", "https://api.binance.us")


def fetch_binance(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Crypto spot klines from Binance public API (no key). Falls back to binance.us (geo-restrictions)."""
    symbol = symbol.replace("/", "").replace("-", "").upper()
    rows, last_err = [], None
    for host in BINANCE_HOSTS:
        try:
            rows = _binance_klines(host, symbol, interval, limit)
            break
        except Exception as e:  # try next host
            last_err = e
    if not rows:
        raise RuntimeError(f"Binance fetch failed for {symbol}: {last_err}")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "tb_base", "tb_quote", "ignore",
    ])
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    df["amount"] = df["quote_volume"]
    return normalize_kline(df)


def _binance_klines(host, symbol, interval, limit):
    out, end_time = [], None
    while len(out) < limit:
        params = {"symbol": symbol, "interval": interval, "limit": min(1000, limit - len(out))}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get(f"{host}/api/v3/klines", params=params, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out = batch + out
        end_time = batch[0][0] - 1
        if len(batch) < params["limit"]:
            break
        time.sleep(0.2)
    return out


def load_source(spec: dict) -> pd.DataFrame:
    """Load data according to a symbol spec from the automation config."""
    source = spec.get("source", "yfinance").lower()
    if source == "csv":
        return load_csv(spec["path"])
    if source == "yfinance":
        return fetch_yfinance(spec["symbol"], spec.get("interval", "1d"), spec.get("period", "2y"))
    if source == "binance":
        return fetch_binance(spec["symbol"], spec.get("interval", "1h"), int(spec.get("limit", 1000)))
    raise ValueError(f"Unknown data source: {source}")

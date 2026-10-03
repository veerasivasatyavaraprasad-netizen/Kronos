from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automation import data_sources
from automation.data_sources import KLINE_COLUMNS, load_csv, normalize_kline
from automation.forecaster import future_timestamps

ROOT = Path(__file__).resolve().parent.parent


def test_load_bundled_sample():
    df = load_csv(ROOT / "data" / "sample_a_share_5min.csv")
    assert list(df.columns) == KLINE_COLUMNS
    assert len(df) == 2500
    assert df["timestamps"].is_monotonic_increasing


def test_load_csv_with_bom_and_other_column_order(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("﻿Date,Open,Close,High,Low\n2024/01/02,1,2,3,0.5\n2024/01/01,1,1,1,1\n", encoding="utf-8")
    df = load_csv(p)
    assert list(df.columns) == KLINE_COLUMNS
    assert df["timestamps"].iloc[0] == pd.Timestamp("2024-01-01")
    assert (df["volume"] == 0).all()


def test_missing_price_column_raises():
    with pytest.raises(ValueError):
        normalize_kline(pd.DataFrame({"date": ["2024-01-01"], "open": [1], "high": [1], "low": [1]}))


def test_future_timestamps_business_days_and_intraday():
    daily = pd.Series(pd.bdate_range("2024-01-01", periods=30))
    fut = future_timestamps(daily, 5)
    assert (fut.dayofweek < 5).all() and fut[0] > daily.iloc[-1]

    hourly = pd.Series(pd.date_range("2024-01-01", periods=50, freq="h"))
    fut = future_timestamps(hourly, 3)
    assert list(fut) == list(pd.date_range("2024-01-03 02:00", periods=3, freq="h"))


def test_fetch_binance_parses_klines(monkeypatch):
    start = 1_700_000_000_000
    rows = [[start + i * 3_600_000, "1", "2", "0.5", "1.5", "10", 0, "15", 1, "0", "0", "0"] for i in range(5)]

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return rows

    monkeypatch.setattr(data_sources.requests, "get", lambda *a, **k: Resp())
    df = data_sources.fetch_binance("BTC/USDT", "1h", limit=5)
    assert len(df) == 5 and list(df.columns) == KLINE_COLUMNS
    assert np.isclose(df["amount"].iloc[0], 15.0)


def test_fetch_yfinance_parses_multiindex(monkeypatch):
    import types

    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="Date")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL"]])
    raw = pd.DataFrame(np.ones((3, 6)), index=idx, columns=cols)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: raw))
    df = data_sources.fetch_yfinance("AAPL")
    assert len(df) == 3 and df["timestamps"].dt.tz is None

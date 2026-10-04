import numpy as np
import pandas as pd

from lunatrade.data.binance_ws import build_streams, handle_message
from lunatrade.data.market_store import MarketStore, from_ms
from lunatrade.data.universe import select_universe
from lunatrade.indicators import ta


def test_sma_ema_rsi_basic():
    x = np.arange(1, 51, dtype=float)
    assert np.isnan(ta.sma(x, 5)[3]) and ta.sma(x, 5)[4] == 3.0
    e = ta.ema(x, 10)
    assert np.isnan(e[8]) and abs(e[9] - 5.5) < 1e-9 and e[-1] < x[-1]
    r = ta.rsi(x, 14)
    assert r[-1] > 99          # monotonic rise -> RSI ~ 100
    r2 = ta.rsi(x[::-1], 14)
    assert r2[-1] < 1


def test_atr_adx_bollinger_shapes():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    h, l = c + 1, c - 1
    assert len(ta.atr(h, l, c)) == 300
    adx, pdi, mdi = ta.adx(h, l, c)
    assert np.nanmax(adx) <= 100
    lo, mid, hi = ta.bollinger(c, 20, 2)
    assert np.all((hi[19:] >= mid[19:]) & (mid[19:] >= lo[19:]))


def test_indicators_have_no_lookahead():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 1, 400))
    full = ta.ema(c, 20)
    part = ta.ema(c[:300], 20)
    assert np.allclose(full[:300], part, equal_nan=True)
    assert np.allclose(ta.rsi(c, 14)[:300], ta.rsi(c[:300], 14), equal_nan=True)


def test_market_view_hides_unclosed_candles(store):
    st = store.state("BTCUSDT")
    t_open = int(st.open_time[100])
    # exactly at candle 100's close -> 101 candles visible; 1ms earlier -> 100
    assert len(store.view(t_open + store.interval_ms).candles("BTCUSDT")) == 101
    assert len(store.view(t_open + store.interval_ms - 1).candles("BTCUSDT")) == 100


def test_snapshots_cut_by_receive_time():
    s = MarketStore("15m")
    s.add_ticker("BTCUSDT", 99, 101, received_ms=1000)
    s.add_ticker("BTCUSDT", 199, 201, received_ms=2000)
    assert s.view(1500).ticker("BTCUSDT")["bid"] == 99
    assert s.view(2500).ticker("BTCUSDT")["bid"] == 199
    assert s.view(500).ticker("BTCUSDT") is None


def test_websocket_messages_update_store():
    s = MarketStore("15m")
    closed = []
    k = {"t": 1_700_000_000_000, "s": "BTCUSDT", "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "q": "15",
         "n": 3, "V": "6", "x": True}
    handle_message(s, {"stream": "btcusdt@kline_15m", "data": {"e": "kline", "k": k}}, lambda *a: closed.append(a))
    handle_message(s, {"stream": "btcusdt@bookTicker", "data": {"u": 1, "s": "BTCUSDT", "b": "1.4", "B": "2",
                                                                "a": "1.6", "A": "3"}})
    handle_message(s, {"stream": "btcusdt@depth20@100ms", "data": {"bids": [["1.4", "2"]], "asks": [["1.6", "3"]]}})
    handle_message(s, {"stream": "btcusdt@aggTrade", "data": {"e": "aggTrade", "s": "BTCUSDT", "p": "1.5", "q": "1",
                                                              "m": False}})
    st = s.state("BTCUSDT")
    assert len(st.open_time) == 1 and st.cols["close"][-1] == 1.5 and closed
    assert st.ticker and st.book and st.trades
    k2 = dict(k, t=k["t"] + 900_000, x=False)
    handle_message(s, {"stream": "btcusdt@kline_15m", "data": {"e": "kline", "k": k2}})
    assert len(st.open_time) == 1           # unclosed candle never enters the store
    assert "btcusdt@kline_15m" in build_streams(["BTCUSDT"], "15m")


def test_universe_filters():
    info = {"symbols": [
        {"symbol": s, "status": "TRADING", "quoteAsset": "USDT", "baseAsset": s[:-4], "permissions": ["SPOT"]}
        for s in ("BTCUSDT", "ETHUSDT", "USDCUSDT", "BTCUPUSDT", "DOGEUSDT", "THINUSDT")]}
    tick = [{"symbol": "BTCUSDT", "quoteVolume": "1e9", "lastPrice": "60000"},
            {"symbol": "ETHUSDT", "quoteVolume": "5e8", "lastPrice": "3000"},
            {"symbol": "USDCUSDT", "quoteVolume": "9e9", "lastPrice": "1"},
            {"symbol": "BTCUPUSDT", "quoteVolume": "9e8", "lastPrice": "10"},
            {"symbol": "DOGEUSDT", "quoteVolume": "2e8", "lastPrice": "0.1"},
            {"symbol": "THINUSDT", "quoteVolume": "1e3", "lastPrice": "1"}]
    books = [{"symbol": "DOGEUSDT", "bidPrice": "0.09", "askPrice": "0.11"}]   # 2000 bps spread
    out = select_universe(tick, books, info, {"quote": "USDT", "exclude_bases": ["USDC"], "exclude_suffixes": ["UP"],
                                              "min_quote_volume_24h": 1e6, "max_spread_bps": 10, "max_symbols": 5,
                                              "always_include": ["BTCUSDT"]})
    syms = [x["symbol"] for x in out]
    assert syms == ["BTCUSDT", "ETHUSDT"]

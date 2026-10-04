import pandas as pd

from lunatrade.data.market_store import from_ms

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def last_close_time(store, sym="BTCUSDT"):
    st = store.state(sym)
    return from_ms(int(st.open_time[-1]) + store.interval_ms)


class FakePublic:
    """Offline stand-in for BinancePublic."""

    def klines(self, *a, **k):
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"])

    def derivatives_snapshot(self, sym):
        return {}

    def book_ticker(self, sym=None):
        raise RuntimeError("offline")

    def depth(self, sym, limit=20):
        raise RuntimeError("offline")

    def exchange_info(self, sym=None):
        return {"symbols": []}

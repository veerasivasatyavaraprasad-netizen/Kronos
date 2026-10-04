"""
In-memory market state shared by every worker.

The store keeps closed candles (plus order book, ticker and derivatives snapshots) per symbol.
Workers only ever see a `MarketView`, which is cut at `as_of`: a candle is visible only after it
has closed and a snapshot only after it was received. That is what makes backtests free of
look-ahead bias - the same worker code runs live and in simulation.
"""
from __future__ import annotations

import datetime as dt
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
               "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000}
CANDLE_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base")


def to_ms(t: dt.datetime) -> int:
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return int((t - dt.datetime(1970, 1, 1)).total_seconds() * 1000)


def from_ms(ms: int) -> dt.datetime:
    return dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=int(ms))


@dataclass
class Candles:
    """Column arrays of closed candles, oldest first."""
    open_time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    quote_volume: np.ndarray
    trades: np.ndarray
    taker_buy_base: np.ndarray

    def __len__(self) -> int:
        return len(self.close)

    def tail(self, n: int) -> "Candles":
        return Candles(*(getattr(self, f)[-n:] for f in ("open_time",) + CANDLE_FIELDS))

    @property
    def last_close(self) -> float:
        return float(self.close[-1]) if len(self.close) else float("nan")

    @classmethod
    def empty(cls) -> "Candles":
        z = np.zeros(0)
        return cls(np.zeros(0, dtype=np.int64), z, z, z, z, z, z, z, z)


@dataclass
class Snapshot:
    received_ms: int
    data: dict


@dataclass
class SymbolState:
    symbol: str
    interval_ms: int
    open_time: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    cols: dict = field(default_factory=lambda: {f: np.zeros(0) for f in CANDLE_FIELDS})
    book: deque = field(default_factory=lambda: deque(maxlen=50))          # order book snapshots
    ticker: deque = field(default_factory=lambda: deque(maxlen=200))       # best bid/ask
    derivatives: deque = field(default_factory=lambda: deque(maxlen=500))  # funding / OI / long-short
    trades: deque = field(default_factory=lambda: deque(maxlen=5000))      # aggregated trades
    last_update_ms: int = 0
    meta: dict = field(default_factory=dict)                               # exchange filters, 24h stats


class MarketStore:
    def __init__(self, interval: str = "15m", max_bars: int = 1000, view_cap: int | None = None):
        self.interval = interval
        self.view_cap = view_cap          # workers see at most this many bars (keeps backtests O(n))
        self.interval_ms = INTERVAL_MS[interval]
        self.max_bars = max_bars
        self._symbols: dict[str, SymbolState] = {}
        self._lock = threading.RLock()
        self._ind_cache: OrderedDict = OrderedDict()
        self._ind_cache_max = 50_000
        self.ws_connected = False

    # ------------------------------------------------------------------ writes
    def state(self, symbol: str) -> SymbolState:
        with self._lock:
            if symbol not in self._symbols:
                self._symbols[symbol] = SymbolState(symbol, self.interval_ms)
            return self._symbols[symbol]

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def load_frame(self, symbol: str, df: pd.DataFrame) -> None:
        """Bulk load candles. df needs open_time (ms or datetime) + CANDLE_FIELDS (missing ones -> 0)."""
        df = df.copy()
        if "open_time" not in df.columns and "timestamps" in df.columns:
            df["open_time"] = df["timestamps"]
        if not np.issubdtype(df["open_time"].dtype, np.integer):
            df["open_time"] = pd.to_datetime(df["open_time"]).astype("datetime64[ms]").astype("int64")
        if "quote_volume" not in df.columns:
            df["quote_volume"] = df.get("amount", df["volume"] * df["close"])
        for f in CANDLE_FIELDS:
            if f not in df.columns:
                df[f] = 0.0
        df = df.drop_duplicates("open_time").sort_values("open_time").tail(self.max_bars)
        st = self.state(symbol)
        with self._lock:
            st.open_time = df["open_time"].to_numpy(dtype=np.int64)
            st.cols = {f: df[f].to_numpy(dtype=float) for f in CANDLE_FIELDS}
            st.last_update_ms = int(st.open_time[-1] + self.interval_ms) if len(st.open_time) else 0
            self._invalidate(symbol)

    def upsert_candle(self, symbol: str, open_time_ms: int, values: dict, received_ms: int | None = None) -> None:
        """Insert/replace one *closed* candle."""
        st = self.state(symbol)
        with self._lock:
            ot = st.open_time
            if len(ot) and open_time_ms < ot[-1]:
                idx = np.searchsorted(ot, open_time_ms)
                if idx < len(ot) and ot[idx] == open_time_ms:
                    for f in CANDLE_FIELDS:
                        st.cols[f][idx] = float(values.get(f, 0.0))
                return
            if len(ot) and ot[-1] == open_time_ms:
                for f in CANDLE_FIELDS:
                    st.cols[f][-1] = float(values.get(f, 0.0))
            else:
                st.open_time = np.append(ot, np.int64(open_time_ms))[-self.max_bars:]
                for f in CANDLE_FIELDS:
                    st.cols[f] = np.append(st.cols[f], float(values.get(f, 0.0)))[-self.max_bars:]
            st.last_update_ms = received_ms or int(open_time_ms + self.interval_ms)
            self._invalidate(symbol)

    def add_book(self, symbol: str, bids: list, asks: list, received_ms: int) -> None:
        st = self.state(symbol)
        st.book.append(Snapshot(received_ms, {"bids": np.asarray(bids, dtype=float).reshape(-1, 2),
                                              "asks": np.asarray(asks, dtype=float).reshape(-1, 2)}))
        st.last_update_ms = max(st.last_update_ms, received_ms)

    def add_ticker(self, symbol: str, bid: float, ask: float, received_ms: int, bid_qty=0.0, ask_qty=0.0) -> None:
        st = self.state(symbol)
        st.ticker.append(Snapshot(received_ms, {"bid": bid, "ask": ask, "bid_qty": bid_qty, "ask_qty": ask_qty}))
        st.last_update_ms = max(st.last_update_ms, received_ms)

    def add_derivatives(self, symbol: str, data: dict, received_ms: int) -> None:
        self.state(symbol).derivatives.append(Snapshot(received_ms, dict(data)))

    def add_trade(self, symbol: str, price: float, qty: float, is_buyer_maker: bool, received_ms: int) -> None:
        self.state(symbol).trades.append(Snapshot(received_ms, {"p": price, "q": qty, "m": is_buyer_maker}))

    def set_meta(self, symbol: str, **meta) -> None:
        self.state(symbol).meta.update(meta)

    def _invalidate(self, symbol: str) -> None:
        # Cache keys include the last candle time, so stale entries are simply never hit again;
        # trimming here only bounds memory.
        while len(self._ind_cache) > self._ind_cache_max:
            self._ind_cache.popitem(last=False)

    # ------------------------------------------------------------------ reads
    def view(self, as_of: dt.datetime | int) -> "MarketView":
        return MarketView(self, as_of if isinstance(as_of, int) else to_ms(as_of))

    def staleness_seconds(self, symbol: str, now_ms: int) -> float:
        st = self._symbols.get(symbol)
        if not st or not st.last_update_ms:
            return float("inf")
        return max(0.0, (now_ms - st.last_update_ms) / 1000)


class MarketView:
    """Read-only, time-cut view of the store."""

    def __init__(self, store: MarketStore, as_of_ms: int):
        self.store = store
        self.as_of_ms = as_of_ms
        self._candles: dict[str, Candles] = {}

    @property
    def as_of(self) -> dt.datetime:
        return from_ms(self.as_of_ms)

    @property
    def symbols(self) -> list[str]:
        return self.store.symbols

    def candles(self, symbol: str, n: int | None = None) -> Candles:
        if symbol not in self._candles:
            st = self.store._symbols.get(symbol)
            if st is None or not len(st.open_time):
                self._candles[symbol] = Candles.empty()
            else:
                # visible iff open_time + interval <= as_of  (the candle has closed)
                end = int(np.searchsorted(st.open_time, self.as_of_ms - st.interval_ms, side="right"))
                start = max(0, end - self.store.view_cap) if self.store.view_cap else 0
                self._candles[symbol] = Candles(st.open_time[start:end],
                                                *(st.cols[f][start:end] for f in CANDLE_FIELDS))
        c = self._candles[symbol]
        return c.tail(n) if n else c

    def price(self, symbol: str) -> float:
        t = self.ticker(symbol)
        if t:
            return (t["bid"] + t["ask"]) / 2
        return self.candles(symbol).last_close

    def _latest(self, dq: deque) -> dict | None:
        for snap in reversed(dq):
            if snap.received_ms <= self.as_of_ms:
                return snap.data
        return None

    def _history(self, dq: deque, n: int) -> list[dict]:
        out = [s.data for s in dq if s.received_ms <= self.as_of_ms]
        return out[-n:]

    def book(self, symbol: str) -> dict | None:
        st = self.store._symbols.get(symbol)
        return self._latest(st.book) if st else None

    def book_history(self, symbol: str, n: int = 20) -> list[dict]:
        st = self.store._symbols.get(symbol)
        return self._history(st.book, n) if st else []

    def ticker(self, symbol: str) -> dict | None:
        st = self.store._symbols.get(symbol)
        return self._latest(st.ticker) if st else None

    def ticker_history(self, symbol: str, n: int = 100) -> list[dict]:
        st = self.store._symbols.get(symbol)
        return self._history(st.ticker, n) if st else []

    def derivatives(self, symbol: str) -> dict | None:
        st = self.store._symbols.get(symbol)
        return self._latest(st.derivatives) if st else None

    def derivatives_history(self, symbol: str, n: int = 100) -> list[dict]:
        st = self.store._symbols.get(symbol)
        return self._history(st.derivatives, n) if st else []

    def trades(self, symbol: str, n: int = 1000) -> list[dict]:
        st = self.store._symbols.get(symbol)
        return self._history(st.trades, n) if st else []

    def meta(self, symbol: str) -> dict:
        st = self.store._symbols.get(symbol)
        return dict(st.meta) if st else {}

    def spread_bps(self, symbol: str) -> float | None:
        t = self.ticker(symbol)
        if not t or not t["bid"] or not t["ask"]:
            return None
        mid = (t["bid"] + t["ask"]) / 2
        return (t["ask"] - t["bid"]) / mid * 10_000

    def ind(self, symbol: str, name: str, fn, *args, n: int | None = None):
        """Memoised indicator: computed once per (symbol, last visible candle, name, args)."""
        c = self.candles(symbol)
        if not len(c):
            return None
        key = (symbol, int(c.open_time[-1]), len(c), name, args)
        cache = self.store._ind_cache
        if key in cache:
            return cache[key]
        val = fn(c, *args)
        cache[key] = val
        return val

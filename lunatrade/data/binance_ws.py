"""
Binance spot WebSocket streams -> MarketStore.

Combined stream per connection: <sym>@kline_<interval>, <sym>@bookTicker, <sym>@depth20@100ms, <sym>@aggTrade.
Binance limits: max 1024 streams per connection, connections are dropped after 24h, and incoming
messages are limited per second - so we shard symbols over connections, reconnect with backoff, and
proactively reconnect before the 24h cut. The engine falls back to REST polling while disconnected.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

from lunatrade.data.market_store import MarketStore

log = logging.getLogger("lunatrade.ws")

WS_HOSTS = {"live": "wss://stream.binance.com:9443", "testnet": "wss://stream.testnet.binance.vision",
            "us": "wss://stream.binance.us:9443"}
MAX_STREAMS_PER_CONN = 200
RECONNECT_AFTER_S = 23 * 3600


def build_streams(symbols: list[str], interval: str, depth_levels: int = 20) -> list[str]:
    streams = []
    for s in symbols:
        s = s.lower()
        streams += [f"{s}@kline_{interval}", f"{s}@bookTicker", f"{s}@depth{depth_levels}@100ms", f"{s}@aggTrade"]
    return streams


def handle_message(store: MarketStore, msg: dict, on_candle_closed=None) -> None:
    """Apply one combined-stream message to the store (pure function -> unit-testable)."""
    stream, data = msg.get("stream", "").lower(), msg.get("data", msg)
    now = int(time.time() * 1000)
    if "@kline_" in stream or data.get("e") == "kline":
        k = data["k"]
        if k.get("x"):  # only closed candles enter the store
            sym = k["s"]
            store.upsert_candle(sym, int(k["t"]), {
                "open": float(k["o"]), "high": float(k["h"]), "low": float(k["l"]), "close": float(k["c"]),
                "volume": float(k["v"]), "quote_volume": float(k["q"]), "trades": float(k["n"]),
                "taker_buy_base": float(k["V"])}, received_ms=now)
            if on_candle_closed:
                on_candle_closed(sym, int(k["t"]))
        else:
            store.state(k["s"]).last_update_ms = now
    elif stream.endswith("@bookticker") or ("b" in data and "a" in data and "u" in data and "B" in data):
        store.add_ticker(data["s"], float(data["b"]), float(data["a"]), now, float(data["B"]), float(data["A"]))
    elif "@depth" in stream:
        sym = stream.split("@")[0].upper()
        store.add_book(sym, data.get("bids", []), data.get("asks", []), now)
    elif "@aggtrade" in stream or data.get("e") == "aggTrade":
        store.add_trade(data["s"], float(data["p"]), float(data["q"]), bool(data["m"]), now)


class BinanceStreams:
    def __init__(self, store: MarketStore, symbols: list[str], interval: str, environment: str = "live",
                 depth_levels: int = 20, on_candle_closed=None):
        self.store = store
        self.symbols = symbols
        self.interval = interval
        self.host = WS_HOSTS.get(environment, WS_HOSTS["live"])
        self.depth_levels = depth_levels
        self.on_candle_closed = on_candle_closed
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected_shards = 0
        self.reconnects = 0
        self.last_message_ms = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="binance-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        streams = build_streams(self.symbols, self.interval, self.depth_levels)
        shards = [streams[i:i + MAX_STREAMS_PER_CONN] for i in range(0, len(streams), MAX_STREAMS_PER_CONN)]
        await asyncio.gather(*(self._shard(s) for s in shards))

    async def _shard(self, streams: list[str]) -> None:
        import websockets  # optional at import time so tests/backtests need no network deps

        url = f"{self.host}/stream?streams={'/'.join(streams)}"
        backoff = 1
        while not self._stop.is_set():
            started = time.time()
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=60, max_size=2 ** 22) as ws:
                    self.connected_shards += 1
                    self.store.ws_connected = True
                    backoff = 1
                    while not self._stop.is_set() and time.time() - started < RECONNECT_AFTER_S:
                        raw = await asyncio.wait_for(ws.recv(), timeout=90)
                        self.last_message_ms = int(time.time() * 1000)
                        handle_message(self.store, json.loads(raw), self.on_candle_closed)
            except Exception as e:
                log.warning("websocket shard error: %s (reconnect in %ss)", e, backoff)
            finally:
                self.connected_shards = max(0, self.connected_shards - 1)
                self.store.ws_connected = self.connected_shards > 0
            if self._stop.is_set():
                break
            self.reconnects += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

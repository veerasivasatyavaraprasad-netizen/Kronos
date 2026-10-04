"""Binance public market data over REST (no API key): klines, depth, tickers, exchange info, derivatives."""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger("lunatrade.binance")

SPOT_HOSTS = ("https://api.binance.com", "https://api.binance.us")
FUTURES_HOST = "https://fapi.binance.com"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
              "taker_buy_base", "taker_buy_quote", "ignore"]


class BinancePublic:
    def __init__(self, base_url: str | None = None, futures_url: str = FUTURES_HOST, timeout: float = 15,
                 session: requests.Session | None = None):
        self.hosts = (base_url.rstrip("/"),) if base_url else SPOT_HOSTS
        self.futures_url = futures_url
        self.timeout = timeout
        self.s = session or requests.Session()
        self.weight_used = 0
        self._host = self.hosts[0]

    # -- plumbing
    def _get(self, path: str, params: dict | None = None, futures: bool = False):
        hosts = (self.futures_url,) if futures else (self._host,) + tuple(h for h in self.hosts if h != self._host)
        last = None
        for host in hosts:
            try:
                r = self.s.get(f"{host}{path}", params=params, timeout=self.timeout)
                self.weight_used = int(r.headers.get("x-mbx-used-weight-1m", self.weight_used) or 0)
                if r.status_code == 429 or r.status_code == 418:
                    retry = int(r.headers.get("Retry-After", "30"))
                    raise RuntimeError(f"Binance rate limit hit, retry after {retry}s")
                r.raise_for_status()
                if not futures:
                    self._host = host
                return r.json()
            except Exception as e:  # try the next host (geo restrictions -> binance.us)
                last = e
        raise RuntimeError(f"Binance GET {path} failed: {last}")

    # -- spot
    def klines(self, symbol: str, interval: str, limit: int = 500, end_time: int | None = None,
               closed_only: bool = True) -> pd.DataFrame:
        rows, end = [], end_time
        while len(rows) < limit:
            params = {"symbol": symbol, "interval": interval, "limit": min(1000, limit - len(rows))}
            if end is not None:
                params["endTime"] = end
            batch = self._get("/api/v3/klines", params)
            if not batch:
                break
            rows = batch + rows
            end = batch[0][0] - 1
            if len(batch) < params["limit"]:
                break
            time.sleep(0.1)
        df = pd.DataFrame(rows, columns=KLINE_COLS)
        for c in KLINE_COLS:
            if c not in ("open_time", "close_time", "trades", "ignore"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        df["trades"] = df["trades"].astype(float)
        if closed_only and len(df):
            df = df[df["close_time"] < int(time.time() * 1000)]
        return df.drop(columns=["ignore"]).reset_index(drop=True)

    def depth(self, symbol: str, limit: int = 20) -> dict:
        return self._get("/api/v3/depth", {"symbol": symbol, "limit": limit})

    def book_ticker(self, symbol: str | None = None):
        return self._get("/api/v3/ticker/bookTicker", {"symbol": symbol} if symbol else None)

    def ticker_24h(self, symbol: str | None = None):
        return self._get("/api/v3/ticker/24hr", {"symbol": symbol} if symbol else None)

    def exchange_info(self, symbol: str | None = None) -> dict:
        return self._get("/api/v3/exchangeInfo", {"symbol": symbol} if symbol else None)

    def server_time(self) -> int:
        return int(self._get("/api/v3/time")["serverTime"])

    # -- USD-M futures (read-only signals; nothing is ever traded there)
    def premium_index(self, symbol: str) -> dict:
        return self._get("/fapi/v1/premiumIndex", {"symbol": symbol}, futures=True)

    def open_interest(self, symbol: str) -> dict:
        return self._get("/fapi/v1/openInterest", {"symbol": symbol}, futures=True)

    def long_short_ratio(self, symbol: str, period: str = "1h") -> list:
        return self._get("/futures/data/globalLongShortAccountRatio",
                         {"symbol": symbol, "period": period, "limit": 30}, futures=True)

    def taker_ratio(self, symbol: str, period: str = "1h") -> list:
        return self._get("/futures/data/takerlongshortRatio", {"symbol": symbol, "period": period, "limit": 30},
                         futures=True)

    def derivatives_snapshot(self, symbol: str) -> dict:
        """Funding, open interest and positioning for one symbol; missing pieces are skipped."""
        out = {}
        try:
            p = self.premium_index(symbol)
            out["funding_rate"] = float(p.get("lastFundingRate") or 0)
            out["mark_price"] = float(p.get("markPrice") or 0)
            out["index_price"] = float(p.get("indexPrice") or 0)
        except Exception as e:
            log.debug("premium index %s: %s", symbol, e)
        try:
            out["open_interest"] = float(self.open_interest(symbol)["openInterest"])
        except Exception as e:
            log.debug("open interest %s: %s", symbol, e)
        try:
            ls = self.long_short_ratio(symbol)
            if ls:
                out["long_short_ratio"] = float(ls[-1]["longShortRatio"])
        except Exception as e:
            log.debug("long/short %s: %s", symbol, e)
        return out


def symbol_filters(info: dict) -> dict:
    """Extract precision/minimum rules from one exchangeInfo symbol entry."""
    f = {x["filterType"]: x for x in info.get("filters", [])}
    lot = f.get("LOT_SIZE", {})
    mlot = f.get("MARKET_LOT_SIZE", {})
    if float(mlot.get("stepSize", 0) or 0) > 0:
        market_step, market_min = float(mlot["stepSize"]), float(mlot.get("minQty", 0))
    else:
        market_step, market_min = float(lot.get("stepSize", 0) or 0), float(lot.get("minQty", 0) or 0)
    notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
    price = f.get("PRICE_FILTER", {})
    return {
        "symbol": info["symbol"], "base": info.get("baseAsset"), "quote": info.get("quoteAsset"),
        "status": info.get("status"),
        "tick_size": float(price.get("tickSize", 0) or 0),
        "step_size": float(lot.get("stepSize", 0) or 0), "min_qty": float(lot.get("minQty", 0) or 0),
        "max_qty": float(lot.get("maxQty", 0) or 0),
        "market_step_size": market_step, "market_min_qty": market_min,
        "min_notional": float(notional.get("minNotional", 0) or 0),
        "order_types": info.get("orderTypes", []),
        "permissions": info.get("permissions", []) or [p for ps in info.get("permissionSets", []) for p in ps],
    }

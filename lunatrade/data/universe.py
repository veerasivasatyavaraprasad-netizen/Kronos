"""Automatic discovery of liquid spot pairs with strict filters."""
from __future__ import annotations

import logging

log = logging.getLogger("lunatrade.universe")


def select_universe(tickers_24h: list[dict], book_tickers: list[dict], exchange_info: dict, cfg: dict) -> list[dict]:
    """
    Rank USDT pairs by 24h quote volume after removing stablecoins, leveraged tokens, non-trading
    symbols, wide spreads and dust prices. Returns [{symbol, quote_volume, spread_bps, ...}] best first.
    """
    quote = cfg.get("quote", "USDT")
    exclude_bases = set(cfg.get("exclude_bases", []))
    suffixes = tuple(cfg.get("exclude_suffixes", []))
    min_qv = float(cfg.get("min_quote_volume_24h", 0))
    max_spread = float(cfg.get("max_spread_bps", 1e9))
    min_price = float(cfg.get("min_price", 0))

    trading = {}
    for s in exchange_info.get("symbols", []):
        perms = s.get("permissions", []) or [p for ps in s.get("permissionSets", []) for p in ps]
        if s.get("status") == "TRADING" and s.get("quoteAsset") == quote and s.get("isSpotTradingAllowed", True) \
                and (not perms or "SPOT" in perms):
            trading[s["symbol"]] = s
    books = {b["symbol"]: b for b in book_tickers or []}

    picked = []
    for t in tickers_24h:
        sym = t.get("symbol")
        info = trading.get(sym)
        if not info:
            continue
        base = info.get("baseAsset", sym[: -len(quote)])
        if base in exclude_bases or (suffixes and base.endswith(suffixes) and len(base) > 4):
            continue
        qv, last = float(t.get("quoteVolume", 0)), float(t.get("lastPrice", 0))
        if qv < min_qv or last < min_price:
            continue
        b = books.get(sym)
        spread = None
        if b:
            bid, ask = float(b.get("bidPrice", 0)), float(b.get("askPrice", 0))
            if bid > 0 and ask > 0:
                spread = (ask - bid) / ((ask + bid) / 2) * 10_000
        if spread is not None and spread > max_spread:
            continue
        picked.append({"symbol": sym, "base": base, "quote_volume": qv, "spread_bps": spread,
                       "price_change_pct": float(t.get("priceChangePercent", 0)), "last_price": last})
    picked.sort(key=lambda x: x["quote_volume"], reverse=True)

    out, seen = [], set()
    for sym in cfg.get("always_include", []):
        match = next((p for p in picked if p["symbol"] == sym), None)
        if match or sym in trading:
            out.append(match or {"symbol": sym, "base": sym[: -len(quote)], "quote_volume": 0.0, "spread_bps": None})
            seen.add(sym)
    for p in picked:
        if len(out) >= int(cfg.get("max_symbols", 15)):
            break
        if p["symbol"] not in seen:
            out.append(p)
            seen.add(p["symbol"])
    return out


def discover(client, cfg: dict) -> list[str]:
    """Live discovery; falls back to the static list if Binance is unreachable."""
    if not cfg.get("discover", True):
        return list(cfg.get("static", []))
    try:
        ranked = select_universe(client.ticker_24h(), client.book_ticker(), client.exchange_info(), cfg)
        syms = [r["symbol"] for r in ranked]
        log.info("universe: %s", syms)
        return syms or list(cfg.get("static", []))
    except Exception as e:
        log.warning("universe discovery failed (%s) - using static list", e)
        return list(cfg.get("static", []))

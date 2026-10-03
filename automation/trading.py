"""
Turn forecasts into orders (long-only).

Brokers:
  paper  - local simulated account stored in <output_dir>/paper_account.json (default, no API, no money)
  alpaca - Alpaca Markets API. Uses the *paper* endpoint unless BOTH `live: true` is set in the
           config AND the env var KRONOS_ALLOW_LIVE_TRADING=yes is present.
           Keys: ALPACA_API_KEY, ALPACA_SECRET_KEY.

Only symbols with a `trade_symbol` in the config are ever traded.
"""
import csv
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import requests

DEFAULTS = {
    "enabled": False,
    "broker": "paper",
    "live": False,
    "dry_run": False,
    "buy_threshold_pct": 1.0,      # forecast change needed to open/add
    "sell_threshold_pct": -1.0,    # forecast change that closes the position
    "min_paths_agree_pct": 60,     # share of sampled paths that must agree with the direction
    "order_notional": 1000,        # money per buy order
    "max_position_notional": 5000, # cap per symbol
    "max_orders_per_run": 5,
    "max_data_age_hours": None,    # skip symbols whose last bar is older than this
    "starting_cash": 100000,       # paper broker only
}


def trading_config(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("trading") or {})}


def live_allowed(tcfg: dict) -> bool:
    return bool(tcfg.get("live")) and os.getenv("KRONOS_ALLOW_LIVE_TRADING", "").lower() == "yes"


def decide(result: dict, position_qty: float, price: float, tcfg: dict, now=None) -> dict:
    """Return {'action': 'buy'|'sell'|'hold', 'reason': str, 'notional'/'qty': ...}."""
    if "error" in result:
        return {"action": "hold", "reason": "forecast failed"}
    if tcfg.get("max_data_age_hours"):
        age = (now or dt.datetime.now()) - pd.Timestamp(result["last_timestamp"]).to_pydatetime()
        if age > dt.timedelta(hours=float(tcfg["max_data_age_hours"])):
            return {"action": "hold", "reason": f"data is stale ({age})"}

    change = result["forecast_change_pct"]
    up = result["paths_up_pct"]
    agree = float(tcfg["min_paths_agree_pct"])
    held = position_qty * price

    if change >= tcfg["buy_threshold_pct"] and up >= agree:
        room = float(tcfg["max_position_notional"]) - held
        notional = min(float(tcfg["order_notional"]), room)
        if notional < 1:
            return {"action": "hold", "reason": "position limit reached"}
        return {"action": "buy", "notional": round(notional, 2),
                "reason": f"forecast {change:+.2f}%, {up:.0f}% paths up"}
    if change <= tcfg["sell_threshold_pct"] and (100 - up) >= agree and position_qty > 0:
        return {"action": "sell", "qty": position_qty,
                "reason": f"forecast {change:+.2f}%, {100 - up:.0f}% paths down"}
    return {"action": "hold", "reason": f"no signal ({change:+.2f}%, {up:.0f}% paths up)"}


class PaperBroker:
    """Simulated account: fills instantly at the last close, fractional quantities."""
    mode = "paper"

    def __init__(self, state_path: Path, starting_cash: float):
        self.path = Path(state_path)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"cash": float(starting_cash), "positions": {}}

    def position(self, symbol):
        return float(self.state["positions"].get(symbol, 0.0))

    def buy(self, symbol, notional, price):
        notional = min(notional, self.state["cash"])
        if notional <= 0:
            raise RuntimeError("insufficient paper cash")
        qty = round(notional / price, 6)
        self.state["cash"] -= qty * price
        self.state["positions"][symbol] = self.position(symbol) + qty
        self._save()
        return qty, "filled"

    def sell(self, symbol, qty, price):
        qty = min(qty, self.position(symbol))
        self.state["cash"] += qty * price
        left = self.position(symbol) - qty
        if left <= 1e-9:
            self.state["positions"].pop(symbol, None)
        else:
            self.state["positions"][symbol] = left
        self._save()
        return qty, "filled"

    def equity(self, prices: dict):
        return self.state["cash"] + sum(q * prices.get(s, 0) for s, q in self.state["positions"].items())

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))


class AlpacaBroker:
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self, live: bool):
        key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to trade with Alpaca")
        self.mode = "live" if live else "paper"
        self.base = self.LIVE_URL if live else self.PAPER_URL
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _req(self, method, path, **kw):
        r = requests.request(method, self.base + path, headers=self.headers, timeout=30, **kw)
        return r

    def position(self, symbol):
        r = self._req("GET", f"/v2/positions/{symbol.replace('/', '')}")
        if r.status_code == 404:
            return 0.0
        r.raise_for_status()
        return float(r.json()["qty"])

    def buy(self, symbol, notional, price):
        tif = "gtc" if "/" in symbol else "day"  # crypto pairs need gtc
        r = self._req("POST", "/v2/orders", json={"symbol": symbol, "notional": str(notional), "side": "buy",
                                                  "type": "market", "time_in_force": tif})
        r.raise_for_status()
        return round(notional / price, 6), r.json().get("status", "submitted")

    def sell(self, symbol, qty, price):
        r = self._req("DELETE", f"/v2/positions/{symbol.replace('/', '')}")
        r.raise_for_status()
        status = "submitted"
        if r.content:
            try:
                status = r.json().get("status", status)
            except ValueError:
                pass
        return qty, status


def make_broker(tcfg: dict, output_dir: Path):
    if tcfg["broker"] == "paper":
        return PaperBroker(Path(output_dir) / "paper_account.json", tcfg["starting_cash"])
    if tcfg["broker"] == "alpaca":
        if tcfg.get("live") and not live_allowed(tcfg):
            print("live: true is set but KRONOS_ALLOW_LIVE_TRADING=yes is not -> using Alpaca PAPER account")
        return AlpacaBroker(live=live_allowed(tcfg))
    raise ValueError(f"Unknown broker: {tcfg['broker']}")


def execute(results: list, specs: list, cfg: dict, output_dir: Path) -> list:
    """Decide and place orders for every spec that has a trade_symbol. Returns executed/planned trades."""
    tcfg = trading_config(cfg)
    if not tcfg["enabled"]:
        return []
    tradable = {s["name"]: s["trade_symbol"] for s in specs if s.get("trade_symbol")}
    if not tradable:
        print("Trading enabled, but no symbol in the config has a trade_symbol.")
        return []

    broker = make_broker(tcfg, output_dir)
    print(f"\n== Trading ({broker.mode}{', dry run' if tcfg['dry_run'] else ''}) ==")
    trades, orders = [], 0
    for res in results:
        sym = tradable.get(res.get("name"))
        if not sym:
            continue
        price = res.get("last_close")
        try:
            pos = broker.position(sym)
            d = decide(res, pos, price or 0, tcfg)
        except Exception as e:
            d = {"action": "hold", "reason": f"broker error: {e}"}
        print(f"{sym}: {d['action'].upper()} - {d['reason']}")
        if d["action"] == "hold":
            continue
        if orders >= int(tcfg["max_orders_per_run"]):
            print(f"{sym}: skipped, max_orders_per_run reached")
            continue
        trade = {"time": dt.datetime.now().isoformat(timespec="seconds"), "mode": broker.mode,
                 "symbol": sym, "side": d["action"], "price": price, "reason": d["reason"]}
        try:
            if tcfg["dry_run"]:
                qty = d.get("qty") or round(d["notional"] / price, 6)
                status = "dry-run"
            elif d["action"] == "buy":
                qty, status = broker.buy(sym, d["notional"], price)
            else:
                qty, status = broker.sell(sym, d["qty"], price)
        except Exception as e:
            qty, status = 0, f"error: {e}"
        trade.update(qty=qty, status=status)
        trades.append(trade)
        orders += 1

    if trades:
        log = Path(output_dir) / "trades.csv"
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trades[0]))
            if new:
                w.writeheader()
            w.writerows(trades)
        print(f"Trade log: {log}")
    if isinstance(broker, PaperBroker):
        prices = {tradable[r["name"]]: r["last_close"] for r in results if r.get("name") in tradable and "last_close" in r}
        print(f"Paper account: cash {broker.state['cash']:.2f}, positions {broker.state['positions']}, "
              f"equity {broker.equity(prices):.2f}")
    return trades

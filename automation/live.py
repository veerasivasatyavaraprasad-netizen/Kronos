"""
Real-time trading loop.

Every bar (default 5 minutes) it waits for the candle to close, fetches fresh candles from the
broker, runs Kronos, applies risk rules and places orders on Binance and/or Alpaca.

Safety model
------------
* Secrets only come from environment variables / .env.
* Binance modes:  off | test | testnet | live
    test    - real market data, orders are validated with Binance's /order/test endpoint
              (nothing is bought or sold); fills are simulated in a local ledger.
    testnet - orders go to https://testnet.binance.vision with BINANCE_TESTNET_API_KEY/SECRET.
    live    - REAL MONEY. Requires env KRONOS_ALLOW_LIVE_TRADING=yes, otherwise the bot refuses.
* Alpaca modes: off | paper | live (live needs the same env switch; otherwise paper is used).
* The bot only ever sells what it bought itself (tracked in outputs/live/state_<broker>_<mode>.json),
  so coins or shares you already held are never touched.
* Long-only, per-symbol position caps, stop-loss / take-profit, daily loss limit, daily trade limit,
  cooldown between trades, and a kill switch: create the file outputs/live/STOP to block new buys.
"""
import datetime as dt
import hashlib
import hmac
import json
import logging
import math
import os
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml

from automation import notify, trading
from automation.data_sources import normalize_kline

ROOT = Path(__file__).resolve().parent.parent
LIVE_URLS = {"binance": "https://api.binance.com", "alpaca": "https://api.alpaca.markets"}
BINANCE_TESTNET = "https://testnet.binance.vision"
ALPACA_PAPER = "https://paper-api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"
INTERVAL_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
ALPACA_TIMEFRAME = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour", "1d": "1Day"}

RISK_DEFAULTS = {
    "buy_threshold_pct": 0.5,
    "sell_threshold_pct": -0.5,
    "min_paths_agree_pct": 60,
    "order_notional": 20,
    "max_position_notional": 100,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 3.0,
    "max_daily_loss": 20,          # realised loss (quote currency) after which buys stop for the day
    "max_trades_per_day": 20,
    "cooldown_bars": 3,            # bars to wait after a trade on the same symbol
}

log = logging.getLogger("kronos.live")


def live_switch_on() -> bool:
    return os.getenv("KRONOS_ALLOW_LIVE_TRADING", "").strip().lower() == "yes"


# ============================================================================ ledger

class Ledger:
    """What the bot itself holds, plus today's realised PnL and trade count, per broker+mode."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.state.setdefault("positions", {})
        self.state.setdefault("days", {})
        self.state.setdefault("last_trade_bar", {})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        os.replace(tmp, self.path)

    def position(self, symbol):
        return self.state["positions"].get(symbol, {"qty": 0.0, "entry": 0.0})

    def _today(self):
        day = dt.datetime.utcnow().strftime("%Y-%m-%d")
        return self.state["days"].setdefault(day, {"realized_pnl": 0.0, "trades": 0})

    def today(self):
        return dict(self._today())

    def record_buy(self, symbol, qty, price, bar_time):
        pos = self.position(symbol)
        new_qty = pos["qty"] + qty
        entry = (pos["qty"] * pos["entry"] + qty * price) / new_qty if new_qty else 0.0
        self.state["positions"][symbol] = {"qty": new_qty, "entry": entry,
                                           "opened": pos.get("opened") or dt.datetime.utcnow().isoformat()}
        self._today()["trades"] += 1
        self.state["last_trade_bar"][symbol] = str(bar_time)
        self.save()

    def record_sell(self, symbol, qty, price, bar_time):
        pos = self.position(symbol)
        qty = min(qty, pos["qty"])
        pnl = (price - pos["entry"]) * qty
        left = pos["qty"] - qty
        if left <= 1e-12:
            self.state["positions"].pop(symbol, None)
        else:
            self.state["positions"][symbol] = {**pos, "qty": left}
        day = self._today()
        day["realized_pnl"] += pnl
        day["trades"] += 1
        self.state["last_trade_bar"][symbol] = str(bar_time)
        self.save()
        return pnl


# ============================================================================ brokers

def _clean_secret(value):
    """Drop whitespace, quotes and invisible control characters picked up when pasting keys."""
    if not value:
        return value
    return "".join(c for c in value if ord(c) > 32 and ord(c) != 127).strip('"').strip("'")


def _floor(value, step):
    if step <= 0:
        return value
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(math.floor(value / step + 1e-9) * step, decimals)


class BinanceBroker:
    """Binance spot via signed REST calls."""
    name = "binance"

    def __init__(self, mode: str):
        if mode not in ("test", "testnet", "live"):
            raise ValueError(f"Unknown Binance mode {mode}")
        if mode == "live" and not live_switch_on():
            raise RuntimeError("Binance mode is 'live' but KRONOS_ALLOW_LIVE_TRADING=yes is not set. Refusing to start.")
        self.mode = mode
        if mode == "testnet":
            self.base = os.getenv("BINANCE_TESTNET_BASE_URL", BINANCE_TESTNET)
            self.key, self.secret = os.getenv("BINANCE_TESTNET_API_KEY"), os.getenv("BINANCE_TESTNET_SECRET_KEY")
        else:
            self.base = os.getenv("BINANCE_API_BASE_URL", LIVE_URLS["binance"]).rstrip("/")
            self.key, self.secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET_KEY")
        self.data_base = os.getenv("BINANCE_API_BASE_URL", LIVE_URLS["binance"]).rstrip("/")
        self.key, self.secret = _clean_secret(self.key), _clean_secret(self.secret)
        if not self.key or not self.secret:
            raise RuntimeError(f"Binance {mode}: API key/secret env vars are not set")
        self.session = requests.Session()
        self.session.headers["X-MBX-APIKEY"] = self.key
        self.time_offset = 0
        self._filters = {}
        self.sync_time()

    # -- low level
    def sign(self, params: dict) -> str:
        query = urlencode(params)
        sig = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    def sync_time(self):
        r = self.session.get(f"{self.base}/api/v3/time", timeout=10)
        r.raise_for_status()
        self.time_offset = r.json()["serverTime"] - int(time.time() * 1000)

    def signed(self, method, path, params=None):
        params = dict(params or {})
        params["recvWindow"] = 5000
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        r = self.session.request(method, f"{self.base}{path}?{self.sign(params)}", timeout=20)
        if r.status_code >= 400:
            raise RuntimeError(f"Binance {path} {r.status_code}: {r.text[:300]}")
        return r.json()

    # -- market info
    def filters(self, symbol):
        if symbol not in self._filters:
            r = self.session.get(f"{self.base}/api/v3/exchangeInfo", params={"symbol": symbol}, timeout=20)
            r.raise_for_status()
            info = r.json()["symbols"][0]
            f = {x["filterType"]: x for x in info["filters"]}
            lot = f.get("MARKET_LOT_SIZE") if float(f.get("MARKET_LOT_SIZE", {}).get("stepSize", 0) or 0) > 0 else f["LOT_SIZE"]
            notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
            self._filters[symbol] = {
                "base": info["baseAsset"], "quote": info["quoteAsset"],
                "step": float(lot["stepSize"]), "min_qty": float(lot["minQty"]),
                "min_notional": float(notional.get("minNotional", 0)),
            }
        return self._filters[symbol]

    def candles(self, symbol, interval, limit=500):
        r = self.session.get(f"{self.data_base}/api/v3/klines",
                             params={"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}, timeout=20)
        r.raise_for_status()
        rows = r.json()
        now_ms = int(time.time() * 1000) + self.time_offset
        rows = [k for k in rows if k[6] < now_ms]  # drop the candle that is still forming
        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                         "quote_volume", "trades", "tb_base", "tb_quote", "ignore"])
        df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
        df["amount"] = df["quote_volume"]
        return normalize_kline(df)

    def free_balance(self, asset):
        acct = self.signed("GET", "/api/v3/account")
        for b in acct["balances"]:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    def market_open(self, symbol):
        return True  # crypto trades 24/7

    # -- orders
    def buy(self, symbol, notional, price):
        f = self.filters(symbol)
        notional = math.floor(notional * 100) / 100
        if notional < f["min_notional"]:
            raise RuntimeError(f"order {notional} below Binance minimum notional {f['min_notional']}")
        params = {"symbol": symbol, "side": "BUY", "type": "MARKET", "quoteOrderQty": f"{notional:.2f}",
                  "newOrderRespType": "FULL"}
        if self.mode == "test":
            self.signed("POST", "/api/v3/order/test", params)
            return _floor(notional / price, f["step"]), price, "validated (test mode, not executed)"
        res = self.signed("POST", "/api/v3/order", params)
        qty = float(res.get("executedQty", 0))
        avg = float(res.get("cummulativeQuoteQty", 0)) / qty if qty else price
        return qty, avg, res.get("status", "?")

    def sell(self, symbol, qty, price):
        f = self.filters(symbol)
        if self.mode != "test":
            qty = min(qty, self.free_balance(f["base"]))  # never more than is actually there
        qty = _floor(qty, f["step"])
        if qty < f["min_qty"] or qty * price < f["min_notional"]:
            raise RuntimeError(f"sell size {qty} below Binance minimums (dust)")
        params = {"symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": f"{qty:.8f}".rstrip("0").rstrip("."),
                  "newOrderRespType": "FULL"}
        if self.mode == "test":
            self.signed("POST", "/api/v3/order/test", params)
            return qty, price, "validated (test mode, not executed)"
        res = self.signed("POST", "/api/v3/order", params)
        filled = float(res.get("executedQty", 0))
        avg = float(res.get("cummulativeQuoteQty", 0)) / filled if filled else price
        return filled, avg, res.get("status", "?")


class AlpacaLiveBroker:
    """Alpaca stocks: 5-minute bars from the data API, market clock, notional buys, qty sells."""
    name = "alpaca"

    def __init__(self, mode: str):
        if mode not in ("paper", "live"):
            raise ValueError(f"Unknown Alpaca mode {mode}")
        if mode == "live" and not live_switch_on():
            log.warning("Alpaca mode 'live' requested without KRONOS_ALLOW_LIVE_TRADING=yes -> using PAPER")
            mode = "paper"
        self.mode = mode
        base = os.getenv("ALPACA_API_BASE_URL", "").rstrip("/")
        if mode == "paper" or not base:
            # A paper key (PK...) only works against the paper endpoint, whatever base URL is configured.
            base = ALPACA_PAPER if mode == "paper" else LIVE_URLS["alpaca"]
        self.base = base
        self.data_base = os.getenv("ALPACA_DATA_BASE_URL", ALPACA_DATA).rstrip("/")
        key, secret = _clean_secret(os.getenv("ALPACA_API_KEY")), _clean_secret(os.getenv("ALPACA_SECRET_KEY"))
        if not key or not secret:
            raise RuntimeError("Alpaca: ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.feed = os.getenv("ALPACA_DATA_FEED", "iex")  # iex is free; sip needs a subscription

    def _req(self, method, url, **kw):
        r = self.session.request(method, url, timeout=20, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca {url.split('/v2')[-1]} {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else {}

    def market_open(self, symbol):
        return bool(self._req("GET", f"{self.base}/v2/clock").get("is_open"))

    def candles(self, symbol, interval, limit=500):
        tf = ALPACA_TIMEFRAME[interval]
        start = (dt.datetime.utcnow() - dt.timedelta(seconds=INTERVAL_SECONDS[interval] * limit * 4 + 86400 * 4))
        params = {"timeframe": tf, "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000,
                  "feed": self.feed, "adjustment": "raw", "sort": "asc"}
        bars, token = [], None
        while True:
            if token:
                params["page_token"] = token
            res = self._req("GET", f"{self.data_base}/v2/stocks/{symbol}/bars", params=params)
            bars += res.get("bars") or []
            token = res.get("next_page_token")
            if not token:
                break
        if not bars:
            raise RuntimeError(f"Alpaca returned no bars for {symbol}")
        df = pd.DataFrame(bars).rename(columns={"t": "timestamps", "o": "open", "h": "high", "l": "low",
                                                "c": "close", "v": "volume"})
        df["amount"] = df["volume"] * df.get("vw", df["close"])
        df = normalize_kline(df)
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(seconds=INTERVAL_SECONDS[interval])
        return df[df["timestamps"] <= cutoff].tail(limit).reset_index(drop=True)  # closed bars only

    def buy(self, symbol, notional, price):
        order = self._req("POST", f"{self.base}/v2/orders",
                          json={"symbol": symbol, "notional": f"{notional:.2f}", "side": "buy",
                                "type": "market", "time_in_force": "day"})
        return self._wait_fill(order, notional / price, price)

    def sell(self, symbol, qty, price):
        try:
            held = float(self._req("GET", f"{self.base}/v2/positions/{symbol}")["qty"])
        except RuntimeError:
            held = 0.0
        qty = min(qty, held)
        if qty <= 0:
            raise RuntimeError("no position to sell at the broker")
        order = self._req("POST", f"{self.base}/v2/orders",
                          json={"symbol": symbol, "qty": f"{qty:.9f}".rstrip("0").rstrip("."), "side": "sell",
                                "type": "market", "time_in_force": "day"})
        return self._wait_fill(order, qty, price)

    def _wait_fill(self, order, fallback_qty, price, tries=10):
        for _ in range(tries):
            if order.get("status") == "filled":
                break
            time.sleep(1)
            order = self._req("GET", f"{self.base}/v2/orders/{order['id']}")
        qty = float(order.get("filled_qty") or 0) or fallback_qty
        avg = float(order.get("filled_avg_price") or 0) or price
        return qty, avg, order.get("status", "?")


# ============================================================================ engine

class LiveTrader:
    def __init__(self, cfg: dict, predictor=None):
        self.cfg = cfg
        self.interval = cfg.get("interval", "5m")
        if self.interval not in INTERVAL_SECONDS:
            raise ValueError(f"Unsupported interval {self.interval}")
        self.out = ROOT / cfg.get("output_dir", "outputs/live")
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop_file = self.out / "STOP"
        self.alert_cfg = cfg.get("alerts") or {}
        self.brokers, self.ledgers = {}, {}
        for name, acc in (cfg.get("accounts") or {}).items():
            mode = acc.get("mode", "off")
            if mode == "off":
                continue
            try:
                broker = BinanceBroker(mode) if name == "binance" else AlpacaLiveBroker(mode)
            except Exception as e:
                if "KRONOS_ALLOW_LIVE_TRADING" in str(e):
                    raise  # never silently continue when real-money mode was refused
                log.error("%s not connected: %s", name, e)
                continue
            self.brokers[name] = broker
            self.ledgers[name] = Ledger(self.out / f"state_{name}_{broker.mode}.json")
            log.info("%s connected in %s mode (%s)", name, broker.mode.upper(), broker.base)
        self.symbols = [s for s in cfg.get("symbols", []) if s.get("enabled", True) and s.get("broker") in self.brokers]
        if not self.symbols:
            raise RuntimeError("No enabled symbols whose broker account is switched on")
        self.predictor = predictor

    def load_model(self):
        if self.predictor is None:
            from automation.forecaster import load_predictor
            self.predictor = load_predictor(self.cfg.get("model", "kronos-small"), self.cfg.get("device", "auto"),
                                            self.cfg.get("model_path"), self.cfg.get("tokenizer_path"))
        return self.predictor

    def risk(self, spec):
        return {**RISK_DEFAULTS, **(self.cfg.get("risk") or {}), **(spec.get("risk") or {})}

    # -- one symbol, one bar
    def step(self, spec):
        from automation.forecaster import forecast
        from automation.kronos_auto import summarize

        broker, ledger = self.brokers[spec["broker"]], self.ledgers[spec["broker"]]
        sym, risk = spec["symbol"], self.risk(spec)
        if not broker.market_open(sym):
            return {"symbol": sym, "action": "skip", "reason": "market closed"}

        lookback = int(self.cfg.get("lookback", 400))
        df = broker.candles(sym, self.interval, limit=lookback + 10)
        if len(df) < 64:
            return {"symbol": sym, "action": "skip", "reason": f"only {len(df)} candles"}
        bar_time = df["timestamps"].iloc[-1]
        age = (pd.Timestamp.utcnow().tz_localize(None) - bar_time).total_seconds()
        if age > INTERVAL_SECONDS[self.interval] * 3:
            return {"symbol": sym, "action": "skip", "reason": f"stale data (last bar {bar_time})"}

        pred, paths = forecast(self.predictor, df, lookback, int(self.cfg.get("pred_len", 12)),
                               temperature=self.cfg.get("temperature", 1.0), top_p=self.cfg.get("top_p", 0.9),
                               sample_count=int(self.cfg.get("sample_count", 5)))
        res = summarize(spec.get("name", sym), df, pred, paths)
        price = res["last_close"]
        pos = ledger.position(sym)
        decision = self.decide(sym, res, pos, price, risk, ledger, bar_time)
        out = {"symbol": sym, "price": price, "forecast_change_pct": res["forecast_change_pct"],
               "paths_up_pct": res["paths_up_pct"], **decision}
        if decision["action"] in ("buy", "sell"):
            out.update(self.execute(broker, ledger, sym, decision, price, bar_time))
        return out

    def decide(self, sym, res, pos, price, risk, ledger, bar_time):
        qty, entry = pos["qty"], pos["entry"]
        # Exits first: protective rules apply even when the kill switch is on.
        if qty > 0 and entry > 0:
            move = (price / entry - 1) * 100
            if move <= -risk["stop_loss_pct"]:
                return {"action": "sell", "qty": qty, "reason": f"stop-loss ({move:+.2f}% from entry {entry:.6g})"}
            if move >= risk["take_profit_pct"]:
                return {"action": "sell", "qty": qty, "reason": f"take-profit ({move:+.2f}% from entry {entry:.6g})"}

        d = trading.decide(res, qty, price, {**trading.DEFAULTS, **risk, "max_data_age_hours": None})
        if d["action"] != "buy":
            return d
        today = ledger.today()
        if self.stop_file.exists():
            return {"action": "hold", "reason": f"kill switch ({self.stop_file}) - no new buys"}
        if today["realized_pnl"] <= -abs(risk["max_daily_loss"]):
            return {"action": "hold", "reason": f"daily loss limit hit ({today['realized_pnl']:.2f})"}
        if today["trades"] >= risk["max_trades_per_day"]:
            return {"action": "hold", "reason": "max trades per day reached"}
        last = ledger.state["last_trade_bar"].get(sym)
        if last:
            bars_since = (bar_time - pd.Timestamp(last)).total_seconds() / INTERVAL_SECONDS[self.interval]
            if bars_since < risk["cooldown_bars"]:
                return {"action": "hold", "reason": f"cooldown ({bars_since:.0f}/{risk['cooldown_bars']} bars)"}
        return d

    def execute(self, broker, ledger, sym, d, price, bar_time):
        try:
            if d["action"] == "buy":
                qty, avg, status = broker.buy(sym, d["notional"], price)
                if qty > 0:
                    ledger.record_buy(sym, qty, avg, bar_time)
                pnl = None
            else:
                qty, avg, status = broker.sell(sym, d["qty"], price)
                pnl = ledger.record_sell(sym, qty, avg, bar_time) if qty > 0 else None
        except Exception as e:
            log.error("%s %s failed: %s", d["action"], sym, e)
            notify.send_text(f"⚠️ Kronos {broker.name} {d['action']} {sym} FAILED: {e}", self.alert_cfg)
            return {"status": f"error: {e}"}
        trade = {"time": dt.datetime.utcnow().isoformat(timespec="seconds"), "broker": broker.name,
                 "mode": broker.mode, "symbol": sym, "side": d["action"], "qty": qty, "price": avg,
                 "status": status, "pnl": pnl, "reason": d["reason"]}
        self.log_trade(trade)
        pnl_txt = f", PnL {pnl:+.4f}" if pnl is not None else ""
        notify.send_text(f"{'🟢' if d['action'] == 'buy' else '🔴'} Kronos {broker.name.upper()} [{broker.mode}] "
                         f"{d['action'].upper()} {qty:g} {sym} @ {avg:.6g} ({status}){pnl_txt}\n{d['reason']}",
                         self.alert_cfg)
        return {"qty": qty, "fill_price": avg, "status": status, "pnl": pnl}

    def log_trade(self, trade):
        path = self.out / "live_trades.csv"
        pd.DataFrame([trade]).to_csv(path, mode="a", header=not path.exists(), index=False)

    # -- loop
    def run_cycle(self):
        results = []
        for b in self.brokers.values():
            if hasattr(b, "sync_time"):
                try:
                    b.sync_time()  # PC clocks drift; Binance rejects requests >5s off
                except Exception as e:
                    log.warning("time sync failed: %s", e)
        for spec in self.symbols:
            try:
                r = self.step(spec)
            except Exception as e:
                log.error("%s: %s\n%s", spec.get("symbol"), e, traceback.format_exc())
                r = {"symbol": spec.get("symbol"), "action": "error", "reason": str(e)}
            log.info("%-10s %-5s %s", r.get("symbol"), r.get("action", "").upper(),
                     " | ".join(str(x) for x in [r.get("reason"), r.get("status")] if x))
            results.append(r)
        return results

    def seconds_to_next_bar(self, delay=5):
        step = INTERVAL_SECONDS[self.interval]
        now = time.time()
        return step - (now % step) + delay

    def run_forever(self, once=False):
        keep_awake()
        self.load_model()
        modes = ", ".join(f"{n}={b.mode}" for n, b in self.brokers.items())
        log.info("Kronos live trader started: %s, interval %s, symbols %s", modes, self.interval,
                 [s["symbol"] for s in self.symbols])
        notify.send_text(f"▶️ Kronos live trader started ({modes}, {self.interval}).", self.alert_cfg)
        errors_in_row = 0
        while True:
            if not once:
                wait = self.seconds_to_next_bar()
                log.info("next bar in %.0fs", wait)
                time.sleep(wait)
            results = self.run_cycle()
            errors_in_row = errors_in_row + 1 if all(r["action"] == "error" for r in results) else 0
            if errors_in_row == 3:
                notify.send_text("⚠️ Kronos live trader: 3 cycles in a row failed - check the log.", self.alert_cfg)
            if once:
                return results


def keep_awake():
    """On Windows, stop the PC from sleeping while the bot runs (the screen may still turn off)."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger("kronos")
    root.setLevel(logging.INFO)
    if not root.handlers:
        fh = RotatingFileHandler(out_dir / "live.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)


def main(config_path, once=False):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    setup_logging(ROOT / cfg.get("output_dir", "outputs/live"))
    try:
        trader = LiveTrader(cfg)
    except Exception as e:
        log.error("Configuration problem, not starting: %s", e)
        raise SystemExit(2)  # start_live.bat does not restart on configuration errors
    try:
        return trader.run_forever(once=once)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
        notify.send_text("⏹️ Kronos live trader stopped.", trader.alert_cfg)

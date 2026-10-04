"""Live trading engine tests against local fake Binance / Alpaca servers (no real accounts)."""
import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlparse

import numpy as np
import pandas as pd
import pytest

from automation import live

SECRET = "test-secret"


class FakeExchange:
    """State shared with the request handler."""

    def __init__(self):
        self.price = 100.0
        self.trend = 0.0           # per-bar drift used to build candles
        self.balances = {"BTC": 1.0, "USDT": 1000.0}   # the user already owns 1 BTC
        self.orders, self.test_orders = [], []
        self.alpaca_open = True
        self.alpaca_orders = []
        self.alpaca_position = 0.0

    def klines(self, n=500, step_ms=300_000):
        now = int(time.time() * 1000)
        start = (now // step_ms) * step_ms - (n - 1) * step_ms  # last candle is still forming
        rows = []
        for i in range(n):
            c = self.price * (1 + self.trend * (i - n + 1)) + np.sin(i / 5) * 0.2
            o = start + i * step_ms
            rows.append([o, f"{c:.4f}", f"{c + .3:.4f}", f"{c - .3:.4f}", f"{c:.4f}", "10", o + step_ms - 1,
                         f"{c * 10:.4f}", 5, "0", "0", "0"])
        return rows


def make_handler(ex: FakeExchange):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _signed_ok(self, query):
            q, _, sig = query.rpartition("&signature=")
            expected = hmac.new(SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
            return sig == expected and self.headers.get("X-MBX-APIKEY") == "test-key"

        def _route(self, method):
            u = urlparse(self.path)
            p = dict(parse_qsl(u.query))
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            # ---- Binance
            if u.path == "/api/v3/time":
                return self._send({"serverTime": int(time.time() * 1000)})
            if u.path == "/api/v3/exchangeInfo":
                return self._send({"symbols": [{"symbol": p["symbol"], "baseAsset": "BTC", "quoteAsset": "USDT",
                                                "filters": [
                                                    {"filterType": "LOT_SIZE", "stepSize": "0.00001000", "minQty": "0.00001000"},
                                                    {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.00000000", "minQty": "0"},
                                                    {"filterType": "NOTIONAL", "minNotional": "5.00000000"}]}]})
            if u.path == "/api/v3/klines":
                return self._send(ex.klines(int(p.get("limit", 500))))
            if u.path.startswith(("/api/v3/", "/sapi/")):
                if not self._signed_ok(u.query):
                    return self._send({"code": -1022, "msg": "bad signature"}, 401)
                if u.path == "/sapi/v1/account/apiRestrictions":
                    return self._send({"enableWithdrawals": False, "ipRestrict": True})
                if u.path == "/api/v3/account":
                    return self._send({"canTrade": True, "balances": [{"asset": a, "free": str(v), "locked": "0"} for a, v in ex.balances.items()]})
                if u.path == "/api/v3/order/test":
                    ex.test_orders.append(p)
                    return self._send({})
                if u.path == "/api/v3/order":
                    ex.orders.append(p)
                    if p["side"] == "BUY":
                        qty = round(float(p["quoteOrderQty"]) / ex.price, 5)
                        ex.balances["BTC"] += qty
                    else:
                        qty = float(p["quantity"])
                        ex.balances["BTC"] -= qty
                    return self._send({"status": "FILLED", "executedQty": str(qty),
                                       "cummulativeQuoteQty": str(qty * ex.price)})
            # ---- Alpaca
            if u.path == "/v2/clock":
                return self._send({"is_open": ex.alpaca_open})
            if u.path.startswith("/v2/stocks/") and u.path.endswith("/bars"):
                now = pd.Timestamp.utcnow().floor("5min")
                bars = [{"t": (now - pd.Timedelta(minutes=5 * (300 - i))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "o": ex.price, "h": ex.price + .2, "l": ex.price - .2,
                         "c": ex.price * (1 + ex.trend * (i - 299)), "v": 1000, "vw": ex.price} for i in range(300)]
                return self._send({"bars": bars, "next_page_token": None})
            if u.path == "/v2/orders" and method == "POST":
                ex.alpaca_orders.append(body)
                if body["side"] == "buy":
                    ex.alpaca_position += float(body["notional"]) / ex.price
                    qty = float(body["notional"]) / ex.price
                else:
                    ex.alpaca_position -= float(body["qty"])
                    qty = float(body["qty"])
                return self._send({"id": "o1", "status": "filled", "filled_qty": str(qty),
                                   "filled_avg_price": str(ex.price)})
            if u.path.startswith("/v2/positions/"):
                if ex.alpaca_position <= 0:
                    return self._send({"message": "position does not exist"}, 404)
                return self._send({"qty": str(ex.alpaca_position)})
            return self._send({"error": "not found", "path": u.path}, 404)

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

    return H


@pytest.fixture
def exchange(monkeypatch, tmp_path):
    ex = FakeExchange()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ex))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_SECRET_KEY", SECRET)
    monkeypatch.setenv("BINANCE_API_BASE_URL", url)
    monkeypatch.setenv("ALPACA_API_KEY", "PKTEST")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_DATA_BASE_URL", url)
    monkeypatch.delenv("KRONOS_ALLOW_LIVE_TRADING", raising=False)
    monkeypatch.setattr(live, "ALPACA_PAPER", url)  # paper endpoint -> fake server
    monkeypatch.setattr(live, "ROOT", tmp_path)
    yield ex
    srv.shutdown()


class FakePredictor:
    """Stands in for Kronos: every path ends `move` (fraction) away from the last close."""
    max_context = 512

    def __init__(self, move):
        self.move = move

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **kw):
        out = []
        for df, y in zip(df_list, y_timestamp_list):
            last = df["close"].iloc[-1]
            closes = np.linspace(last, last * (1 + self.move), pred_len + 1)[1:]
            out.append(pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                                     "volume": 1.0, "amount": 1.0}, index=y))
        return out


def config(binance_mode="test", alpaca_mode="off", **risk):
    return {"interval": "5m", "lookback": 200, "pred_len": 6, "sample_count": 3, "output_dir": "live",
            "accounts": {"binance": {"mode": binance_mode}, "alpaca": {"mode": alpaca_mode}},
            "risk": {"order_notional": 20, "max_position_notional": 60, **risk},
            "symbols": [{"symbol": "BTCUSDT", "broker": "binance"}, {"symbol": "AAPL", "broker": "alpaca"}]}


def test_binance_signature_matches_official_example(monkeypatch):
    # Example from Binance's API documentation (HMAC SHA256 signing).
    b = live.BinanceBroker.__new__(live.BinanceBroker)
    b.secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    params = {"symbol": "LTCBTC", "side": "BUY", "type": "LIMIT", "timeInForce": "GTC", "quantity": 1,
              "price": 0.1, "recvWindow": 5000, "timestamp": 1499827319559}
    assert b.sign(params).endswith("signature=c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71")


def test_live_mode_refused_without_switch(exchange):
    with pytest.raises(RuntimeError, match="KRONOS_ALLOW_LIVE_TRADING"):
        live.LiveTrader(config("live"))


def test_test_mode_validates_but_never_executes(exchange):
    t = live.LiveTrader(config("test"), predictor=FakePredictor(+0.02))
    r = t.run_cycle()[0]
    assert r["action"] == "buy" and "not executed" in r["status"]
    assert exchange.test_orders and not exchange.orders            # validated only
    assert exchange.test_orders[0]["quoteOrderQty"] == "20.00"
    assert t.ledgers["binance"].position("BTCUSDT")["qty"] > 0     # simulated position tracked


def test_closed_candle_only(exchange):
    t = live.LiveTrader(config("test"), predictor=FakePredictor(0))
    df = t.brokers["binance"].candles("BTCUSDT", "5m", 50)
    assert (pd.Timestamp.utcnow().tz_localize(None) - df["timestamps"].iloc[-1]).total_seconds() >= 300


def test_real_mode_buy_then_stop_loss_sells_only_bot_position(exchange, monkeypatch):
    monkeypatch.setenv("KRONOS_ALLOW_LIVE_TRADING", "yes")
    t = live.LiveTrader(config("live"), predictor=FakePredictor(+0.02))
    r = t.run_cycle()[0]
    assert r["action"] == "buy" and r["status"] == "FILLED"
    bot_qty = t.ledgers["binance"].position("BTCUSDT")["qty"]
    assert bot_qty == pytest.approx(0.2, abs=1e-3) and exchange.balances["BTC"] > 1.0

    exchange.price = 97.0           # -3% -> stop-loss, even though the forecast is still bullish
    r = t.run_cycle()[0]
    assert r["action"] == "sell" and "stop-loss" in r["reason"]
    sold = float(exchange.orders[-1]["quantity"])
    assert sold == pytest.approx(bot_qty, abs=1e-5)                # the user's own 1 BTC is untouched
    assert exchange.balances["BTC"] == pytest.approx(1.0, abs=1e-5)
    assert t.ledgers["binance"].today()["realized_pnl"] < 0
    assert (t.out / "live_trades.csv").read_text().count("\n") == 3


def test_kill_switch_blocks_buys(exchange):
    t = live.LiveTrader(config("test"), predictor=FakePredictor(+0.02))
    t.stop_file.write_text("stop")
    r = t.run_cycle()[0]
    assert r["action"] == "hold" and "kill switch" in r["reason"] and not exchange.test_orders


def test_daily_loss_and_cooldown(exchange):
    t = live.LiveTrader(config("test", max_daily_loss=1), predictor=FakePredictor(+0.02))
    t.ledgers["binance"]._today()["realized_pnl"] = -5
    assert "daily loss" in t.run_cycle()[0]["reason"]
    t.ledgers["binance"]._today()["realized_pnl"] = 0
    assert t.run_cycle()[0]["action"] == "buy"
    r = t.run_cycle()[0]                   # same bar again -> cooldown
    assert r["action"] == "hold" and "cooldown" in r["reason"]


def test_bearish_forecast_without_position_does_nothing(exchange):
    t = live.LiveTrader(config("test"), predictor=FakePredictor(-0.02))
    r = t.run_cycle()[0]
    assert r["action"] == "hold" and not exchange.test_orders


def test_alpaca_paper_respects_market_clock(exchange, monkeypatch):
    # Even if the live URL is configured, a paper-mode bot talks to the paper endpoint.
    monkeypatch.setenv("ALPACA_API_BASE_URL", "https://api.alpaca.markets")
    t = live.LiveTrader(config("off", "paper"), predictor=FakePredictor(+0.02))
    assert t.brokers["alpaca"].base == live.ALPACA_PAPER
    exchange.alpaca_open = False
    assert t.run_cycle()[0]["reason"] == "market closed"
    exchange.alpaca_open = True
    r = t.run_cycle()[0]
    assert r["action"] == "buy" and r["status"] == "filled"
    assert exchange.alpaca_orders[0]["notional"] == "20.00"


def test_alpaca_live_without_switch_falls_back_to_paper(exchange):
    t = live.LiveTrader(config("off", "live"), predictor=FakePredictor(0))
    assert t.brokers["alpaca"].mode == "paper"


def test_setup_wizard_writes_env_and_checks(exchange, tmp_path, monkeypatch):
    from automation import setup_wizard as w
    env = tmp_path / ".env"
    monkeypatch.setattr(w, "ENV_PATH", env)
    monkeypatch.setattr(w, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("# comment\nBINANCE_API_KEY=\nBINANCE_SECRET_KEY=\nKRONOS_SECRET_KEY=\n")
    answers = iter(["new-key", SECRET] + [""] * 20)
    monkeypatch.setattr(w.getpass, "getpass", lambda prompt="": next(answers))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(w, "CHECKS", [])
    assert w.configure() == 0
    text = env.read_text()
    assert "# comment" in text and "BINANCE_API_KEY=new-key" in text and f"BINANCE_SECRET_KEY={SECRET}" in text
    assert len(w.read_env_file(env)["KRONOS_SECRET_KEY"]) == 64

    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    status, msg = w.check_binance()
    assert status in ("ok", "warn") and "canTrade" in msg

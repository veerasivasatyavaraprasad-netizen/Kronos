import datetime as dt
import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlparse

import pytest

from lunatrade.core.types import Mode, OrderRequest
from lunatrade.execution.brokers.binance_spot import BinanceSpotBroker
from lunatrade.execution.brokers.paper import PaperBroker
from lunatrade.execution.gateway import ExecutionGateway
from lunatrade.execution.order_state import IllegalTransition, Order, OrderStatus
from lunatrade.execution.positions import PositionBook
from lunatrade.risk.kill_switch import KillSwitch

S = OrderStatus


def test_state_machine_blocks_illegal_transitions():
    o = Order("p", "BTCUSDT", "BUY", "paper", "PAPER")
    with pytest.raises(IllegalTransition):
        o.transition(S.FILLED)                       # CREATED -> FILLED skips validation/submission
    o.transition(S.VALIDATED)
    o.transition(S.SUBMITTED)
    o.transition(S.UNKNOWN, "timeout")
    o.transition(S.FILLED, "confirmed by query")
    o.transition(S.POSITION_UPDATED)
    with pytest.raises(IllegalTransition):
        o.transition(S.FILLED)
    assert [h["to"] for h in o.history] == ["VALIDATED", "SUBMITTED", "UNKNOWN", "FILLED", "POSITION_UPDATED"]


def gateway(tmp_path, cfg, mode=Mode.PAPER, cash=10_000, **broker_kw):
    broker = PaperBroker(cash, fee_bps=10, slippage_bps=5, **broker_kw)
    book = PositionBook(tmp_path / "book.json", cash)
    ks = KillSwitch(tmp_path / "ks", cfg.section("kill_switch"), cfg.section("risk"))
    return ExecutionGateway({"paper": broker}, book, mode, kill_switch=ks, reconcile_delay=0), book, broker, ks


def test_paper_round_trip_updates_book(tmp_path, cfg):
    gw, book, broker, _ = gateway(tmp_path, cfg)
    gw.authorize("p1", 1000)
    o = gw.execute(OrderRequest("p1", "BTCUSDT", "BUY", notional=1000, reference_price=100, broker="paper"),
                   stop=95, target=110)
    assert o.status == S.POSITION_UPDATED and book.qty("BTCUSDT") > 0
    assert o.avg_price == pytest.approx(100.05) and o.slippage_bps == pytest.approx(5)
    assert book.cash == pytest.approx(10_000 - 1000 - 1.0, rel=1e-6)
    s = gw.execute(OrderRequest("p1", "BTCUSDT", "SELL", quantity=book.qty("BTCUSDT"), reference_price=110,
                                broker="paper", reason="TAKE_PROFIT"))
    assert s.status == S.POSITION_UPDATED and book.qty("BTCUSDT") == 0
    out = book.closed[-1]
    assert out["pnl"] > 0 and out["reason"] == "TAKE_PROFIT" and out["r_multiple"] > 1


def test_gateway_validation_rules(tmp_path, cfg):
    gw, book, _, ks = gateway(tmp_path, cfg)
    r = gw.execute(OrderRequest("nope", "BTCUSDT", "BUY", notional=100, reference_price=100, broker="paper"))
    assert r.status == S.REJECTED and "approval" in r.error
    gw.authorize("p2", 50)
    o = gw.execute(OrderRequest("p2", "BTCUSDT", "BUY", notional=5000, reference_price=100, broker="paper"))
    assert o.filled_notional <= 50 * 1.01               # capped at the risk-approved notional
    dup = gw.execute(OrderRequest("p2", "BTCUSDT", "BUY", notional=50, reference_price=100, broker="paper"))
    assert dup.status == S.REJECTED and "duplicate" in dup.error
    sell = gw.execute(OrderRequest("x", "ETHUSDT", "SELL", quantity=1, reference_price=100, broker="paper"))
    assert sell.status == S.REJECTED                    # never sells what LunaTrade didn't buy
    ks.trip("MANUAL_STOP", "t")
    gw.authorize("p3", 100)
    blocked = gw.execute(OrderRequest("p3", "SOLUSDT", "BUY", notional=100, reference_price=10, broker="paper"))
    assert blocked.status == S.REJECTED and "kill switch" in blocked.error
    ok_exit = gw.execute(OrderRequest("p2", "BTCUSDT", "SELL", quantity=book.qty("BTCUSDT"), reference_price=100,
                                      broker="paper"))
    assert ok_exit.status == S.POSITION_UPDATED         # exits allowed with kill switch on
    gw.mode = Mode.RESEARCH
    gw.authorize("p4", 100)
    assert gw.execute(OrderRequest("p4", "SOLUSDT", "BUY", notional=100, reference_price=10,
                                   broker="paper")).status == S.REJECTED


def test_partial_fill_settles_filled_part(tmp_path, cfg):
    gw, book, _, _ = gateway(tmp_path, cfg, partial_fill_probability=1.0)
    gw.authorize("p", 1000)
    o = gw.execute(OrderRequest("p", "BTCUSDT", "BUY", notional=1000, reference_price=100, broker="paper"))
    assert o.status == S.POSITION_UPDATED and 0 < o.filled_qty < 10
    assert book.qty("BTCUSDT") == pytest.approx(o.filled_qty)
    assert any(h["to"] == "CANCELED" for h in o.history)


def test_position_exits_stop_trailing_time(tmp_path, cfg):
    book = PositionBook(None, 10_000)
    now = dt.datetime(2026, 1, 1)
    book.apply_buy("BTCUSDT", 1, 100, 0, now, stop=95, target=110)
    risk = cfg.section("risk")
    assert book.exit_signal("BTCUSDT", 99, 98, 100, 1.0, risk) is None
    assert book.exit_signal("BTCUSDT", 96, 94.5, 97, 1.0, risk)[0] == "STOP_LOSS"
    book.apply_buy("ETHUSDT", 1, 100, 0, now, stop=95, target=200)
    book.exit_signal("ETHUSDT", 106, 104, 106, 1.0, risk)        # +1.2R -> break-even
    assert book.positions["ETHUSDT"].stop == pytest.approx(100)
    book.exit_signal("ETHUSDT", 120, 118, 120, 1.0, risk)        # trail
    assert book.positions["ETHUSDT"].stop > 110
    book.apply_buy("SOLUSDT", 1, 100, 0, now, stop=90, target=200)
    book.positions["SOLUSDT"].bars_held = risk["time_stop_bars"]
    assert book.exit_signal("SOLUSDT", 100, 99, 101, 1.0, risk)[0] == "TIME_STOP"


# ------------------------------------------------------------------ fake Binance (signed REST)

SECRET = "test-secret"


class FakeBinance:
    def __init__(self):
        self.orders = {}
        self.slow_next_order = False


def make_handler(ex):
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

        def _check(self, query):
            q, _, sig = query.rpartition("&signature=")
            return sig == hmac.new(SECRET.encode(), q.encode(), hashlib.sha256).hexdigest() and \
                self.headers.get("X-MBX-APIKEY") == "test-key"

        def do_GET(self):
            u = urlparse(self.path)
            q = dict(parse_qsl(u.query))
            if u.path == "/api/v3/time":
                return self._send({"serverTime": int(time.time() * 1000)})
            if u.path == "/api/v3/exchangeInfo":
                return self._send({"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
                                                "quoteAsset": "USDT", "filters": [
                                                    {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001",
                                                     "maxQty": "9000"},
                                                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                                    {"filterType": "NOTIONAL", "minNotional": "5"}]}]})
            if not self._check(u.query):
                return self._send({"code": -1022, "msg": "bad signature"}, 400)
            if u.path == "/api/v3/order":
                o = ex.orders.get(q["origClientOrderId"])
                if not o:
                    return self._send({"code": -2013, "msg": "Order does not exist."}, 400)
                return self._send(o)
            if u.path == "/api/v3/account":
                return self._send({"balances": [{"asset": "USDT", "free": "1000", "locked": "0"},
                                                {"asset": "BTC", "free": "0.5", "locked": "0"}]})
            self._send({}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            q = dict(parse_qsl(u.query))
            if not self._check(u.query):
                return self._send({"code": -1022, "msg": "bad signature"}, 400)
            if u.path == "/api/v3/order/test":
                return self._send({})
            if u.path == "/api/v3/order":
                notional = float(q.get("quoteOrderQty", 0)) or float(q.get("quantity", 0)) * 100
                qty = notional / 100
                o = {"symbol": q["symbol"], "orderId": len(ex.orders) + 1, "clientOrderId": q["newClientOrderId"],
                     "status": "FILLED", "executedQty": f"{qty:.5f}", "cummulativeQuoteQty": f"{notional:.2f}",
                     "fills": [{"commission": "0.01", "commissionAsset": "USDT"}]}
                ex.orders[q["newClientOrderId"]] = o
                if ex.slow_next_order:
                    ex.slow_next_order = False
                    time.sleep(1.5)                     # client times out, but the order exists
                return self._send(o)
            self._send({}, 404)
    return H


@pytest.fixture
def fake_binance():
    ex = FakeBinance()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ex))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield ex, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_binance_broker_signed_order_and_reconcile_after_timeout(tmp_path, cfg, fake_binance):
    ex, url = fake_binance
    broker = BinanceSpotBroker("testnet", base_url=url, api_key="test-key", api_secret=SECRET, timeout=0.5)
    book = PositionBook(None, 1000)
    gw = ExecutionGateway({"binance": broker}, book, Mode.APPROVAL, reconcile_delay=0.05)
    gw.authorize("p1", 50)
    o = gw.execute(OrderRequest("p1", "BTCUSDT", "BUY", notional=50, reference_price=100, broker="binance"))
    assert o.status == S.POSITION_UPDATED and o.filled_qty == pytest.approx(0.5) and o.fee == pytest.approx(0.01)
    # timeout: the order reached the exchange but the response was lost -> UNKNOWN -> query -> FILLED
    ex.slow_next_order = True
    gw.authorize("p2", 20)
    o2 = gw.execute(OrderRequest("p2", "BTCUSDT", "BUY", notional=20, reference_price=100, broker="binance"))
    statuses = [h["to"] for h in o2.history]
    assert "UNKNOWN" in statuses and o2.status == S.POSITION_UPDATED
    assert len(ex.orders) == 2                     # reconciled, NOT re-sent
    assert book.qty("BTCUSDT") == pytest.approx(0.7)


def test_binance_query_missing_order_means_rejected(fake_binance):
    _, url = fake_binance
    broker = BinanceSpotBroker("testnet", base_url=url, api_key="test-key", api_secret=SECRET)
    res = broker.query(Order("p", "BTCUSDT", "BUY", "binance", "LIVE"))
    assert res.status == S.REJECTED


def test_binance_shadow_mode_validates_without_executing(fake_binance):
    ex, url = fake_binance
    broker = BinanceSpotBroker("testnet", validate_only=True, base_url=url, api_key="test-key", api_secret=SECRET)
    gw = ExecutionGateway({"binance": broker}, PositionBook(None, 1000), Mode.SHADOW, reconcile_delay=0)
    gw.authorize("p", 50)
    o = gw.execute(OrderRequest("p", "BTCUSDT", "BUY", notional=50, reference_price=100, broker="binance"))
    assert o.status == S.POSITION_UPDATED and not ex.orders


def test_live_requires_switch(monkeypatch):
    monkeypatch.delenv("LUNATRADE_ALLOW_LIVE", raising=False)
    monkeypatch.delenv("KRONOS_ALLOW_LIVE_TRADING", raising=False)
    with pytest.raises(RuntimeError, match="LUNATRADE_ALLOW_LIVE"):
        BinanceSpotBroker("live", api_key="k", api_secret="s")

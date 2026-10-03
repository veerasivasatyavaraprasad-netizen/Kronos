import io
import sys
from pathlib import Path

import pytest

from automation import notify, trading

ROOT = Path(__file__).resolve().parent.parent

RESULT_UP = {"name": "AAPL_1D", "last_close": 100.0, "forecast_close": 103.0, "forecast_change_pct": 3.0,
             "paths_up_pct": 80.0, "signal": "UP", "horizon_end": "2026-01-10", "last_timestamp": "2026-01-01"}
RESULT_DOWN = {**RESULT_UP, "forecast_close": 97.0, "forecast_change_pct": -3.0, "paths_up_pct": 10.0,
               "signal": "DOWN"}
RESULT_FLAT = {**RESULT_UP, "name": "FLAT", "forecast_change_pct": 0.1, "signal": "FLAT"}


# ----------------------------------------------------------------------------- alerts

def test_alert_filters_and_text():
    picked = notify.select_results([RESULT_UP, RESULT_FLAT, {"name": "X", "error": "boom"}],
                                   {"only_signals": ["UP", "DOWN"], "min_abs_change_pct": 1})
    assert [r["name"] for r in picked] == ["AAPL_1D", "X"]
    text = notify.format_text(picked, {"generated": "now", "model": "m"},
                              [{"mode": "paper", "side": "buy", "qty": 1, "symbol": "AAPL", "price": 100.0,
                                "status": "filled"}])
    assert "+3.00%" in text and "boom" in text and "PAPER BUY 1 AAPL" in text


def test_only_channels_with_env_are_used(monkeypatch):
    for v in sum(notify.REQUIRED_ENV.values(), ()):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/hook")
    assert notify.configured_channels({}) == ["discord"]
    assert notify.configured_channels({"channels": ["telegram"]}) == []


def test_send_alerts_all_channels(monkeypatch, tmp_path):
    env = {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1", "SMTP_HOST": "smtp.x", "SMTP_USER": "u",
           "SMTP_PASSWORD": "p", "ALERT_EMAIL_TO": "to@x", "DISCORD_WEBHOOK_URL": "https://d", "SLACK_WEBHOOK_URL": "https://s"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "AAPL_1D_forecast.png").write_bytes(b"\x89PNG")
    posts = []

    class Resp:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(notify.requests, "post", lambda url, **kw: posts.append(url) or Resp())
    sent = []

    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def starttls(self): pass
        def login(self, u, p): assert (u, p) == ("u", "p")
        def send_message(self, msg): sent.append(msg)

    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    status = notify.send_alerts([RESULT_UP], {"generated": "now", "model": "m"}, tmp_path, {"enabled": True})
    assert status == {"telegram": "ok", "email": "ok", "discord": "ok", "slack": "ok"}
    assert any("sendPhoto" in u for u in posts) and "https://d" in posts and "https://s" in posts
    assert sent[0]["To"] == "to@x" and list(sent[0].iter_attachments())


def test_disabled_alerts_send_nothing(monkeypatch):
    monkeypatch.setattr(notify.requests, "post", lambda *a, **k: pytest.fail("should not send"))
    assert notify.send_alerts([RESULT_UP], {}, ".", {"enabled": False}) == {}


# ----------------------------------------------------------------------------- trading

def test_decide_rules():
    t = trading.trading_config({})
    assert trading.decide(RESULT_UP, 0, 100, t)["action"] == "buy"
    assert trading.decide(RESULT_DOWN, 5, 100, t)["action"] == "sell"
    assert trading.decide(RESULT_DOWN, 0, 100, t)["action"] == "hold"            # long-only
    assert trading.decide(RESULT_FLAT, 0, 100, t)["action"] == "hold"
    assert trading.decide({**RESULT_UP, "paths_up_pct": 50}, 0, 100, t)["action"] == "hold"  # paths disagree
    assert trading.decide(RESULT_UP, 50, 100, t)["action"] == "hold"             # 5000 cap reached
    stale = trading.decide(RESULT_UP, 0, 100, {**t, "max_data_age_hours": 1})
    assert stale["action"] == "hold" and "stale" in stale["reason"]


def test_paper_trading_round_trip(tmp_path):
    cfg = {"trading": {"enabled": True, "broker": "paper"}}
    specs = [{"name": "AAPL_1D", "trade_symbol": "AAPL"}]
    trades = trading.execute([RESULT_UP], specs, cfg, tmp_path)
    assert trades[0]["side"] == "buy" and trades[0]["status"] == "filled" and trades[0]["qty"] == 10
    trades = trading.execute([RESULT_DOWN], specs, cfg, tmp_path)
    assert trades[0]["side"] == "sell" and trades[0]["qty"] == 10
    broker = trading.PaperBroker(tmp_path / "paper_account.json", 0)
    assert broker.state["positions"] == {} and broker.state["cash"] == pytest.approx(100000)
    assert (tmp_path / "trades.csv").read_text().count("\n") == 3


def test_dry_run_and_untradable_symbols(tmp_path):
    cfg = {"trading": {"enabled": True, "dry_run": True}}
    assert trading.execute([RESULT_UP], [{"name": "AAPL_1D"}], cfg, tmp_path) == []  # no trade_symbol
    trades = trading.execute([RESULT_UP], [{"name": "AAPL_1D", "trade_symbol": "AAPL"}], cfg, tmp_path)
    assert trades[0]["status"] == "dry-run"
    assert not (tmp_path / "paper_account.json").exists()


def test_live_trading_needs_double_opt_in(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.delenv("KRONOS_ALLOW_LIVE_TRADING", raising=False)
    tcfg = trading.trading_config({"trading": {"broker": "alpaca", "live": True}})
    assert trading.make_broker(tcfg, ".").base == trading.AlpacaBroker.PAPER_URL
    monkeypatch.setenv("KRONOS_ALLOW_LIVE_TRADING", "yes")
    assert trading.make_broker(tcfg, ".").base == trading.AlpacaBroker.LIVE_URL
    tcfg = trading.trading_config({"trading": {"broker": "alpaca", "live": False}})
    assert trading.make_broker(tcfg, ".").base == trading.AlpacaBroker.PAPER_URL


def test_alpaca_orders(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    calls = []

    class Resp:
        def __init__(self, code=200, body=None):
            self.status_code, self._body, self.content = code, body or {}, b"{}"
        def raise_for_status(self): pass
        def json(self): return self._body

    def fake(method, url, **kw):
        calls.append((method, url, kw.get("json")))
        if method == "GET":
            return Resp(404)
        return Resp(body={"status": "accepted"})

    monkeypatch.setattr(trading.requests, "request", fake)
    b = trading.AlpacaBroker(live=False)
    assert b.position("AAPL") == 0.0
    assert b.buy("BTC/USD", 500, 50000) == (0.01, "accepted")
    assert calls[-1][2]["time_in_force"] == "gtc" and calls[-1][2]["notional"] == "500"
    assert b.sell("AAPL", 3, 100) == (3, "accepted") and calls[-1][0] == "DELETE"


# ----------------------------------------------------------------------------- web UI login + upload

@pytest.fixture
def webapp(monkeypatch, tmp_path):
    monkeypatch.setenv("KRONOS_USERS", "alice:pw1")
    monkeypatch.setenv("KRONOS_SECRET_KEY", "test")
    monkeypatch.setenv("KRONOS_DATA_DIR", str(tmp_path))
    sys.path.insert(0, str(ROOT / "webui"))
    for mod in [m for m in sys.modules if m in ("webui.app", "app")]:
        del sys.modules[mod]
    from webui.app import app
    yield app.test_client()
    del sys.modules["webui.app"]


def test_login_required_and_upload(webapp):
    assert webapp.get("/healthz").status_code == 200
    assert webapp.get("/api/data-files").status_code == 401
    assert webapp.get("/").status_code == 302
    assert webapp.post("/login", data={"username": "alice", "password": "nope"}).status_code == 401
    r = webapp.post("/login?next=//evil.example", data={"username": "alice", "password": "pw1"})
    assert r.status_code == 302 and r.headers["Location"] == "/"
    csv = (ROOT / "data" / "sample_a_share_5min.csv").read_bytes()
    r = webapp.post("/api/upload-data", data={"file": (io.BytesIO(csv), "../x y.csv")},
                    content_type="multipart/form-data")
    assert r.get_json()["name"] == "x_y.csv"
    assert [f["name"] for f in webapp.get("/api/data-files").get_json()] == ["x_y.csv"]
    bad = webapp.post("/api/upload-data", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "bad.csv")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400

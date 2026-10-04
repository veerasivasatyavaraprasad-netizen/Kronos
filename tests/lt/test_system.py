"""End-to-end: engine cycles, approval flow, voice agent, API, backtests, robustness suite."""
import datetime as dt

import pytest
from fastapi.testclient import TestClient

from lunatrade.backtest.engine import Backtester
from lunatrade.backtest.robustness import monte_carlo, stress_test, walk_forward
from lunatrade.control.approvals import ApprovalQueue
from lunatrade.core.types import Direction, Mode, RiskDecision, TradeProposal
from lunatrade.voice.parser import parse
from lt.helpers import SYMS, FakePublic, last_close_time


@pytest.fixture
def runtime(cfg, data, monkeypatch):
    monkeypatch.setenv("LUNATRADE_API_TOKEN", "tok")
    monkeypatch.setenv("LUNATRADE_CONTROL_PIN", "4321")
    from lunatrade.orchestrator import Runtime

    cfg.set("data.websocket", False)
    rt = Runtime(cfg, start_threads=False, symbols=list(SYMS), public=FakePublic())
    for s, df in data.items():
        rt.store.load_frame(s, df)
    return rt


def test_engine_cycle_and_status(runtime):
    t = last_close_time(runtime.store)
    res = runtime.engine.cycle(t)
    assert res["signals"] > 50 and runtime.engine.cycle_count == 1
    st = runtime.engine.status()
    assert st["workers"]["total"] == 300 and st["market_regime"] and len(st["signals"]) == 3
    counts = runtime.repo.counts()
    assert counts["signals"] > 0 and counts["market_snapshots"] == 3 and counts["portfolio_snapshots"] == 1
    assert counts["agents"] == 300


def test_approval_mode_needs_human_yes(runtime):
    e = runtime.engine
    runtime.cfg.set("brokers.binance.environment", "paper")   # human approval on paper fills
    runtime.switch_mode(Mode.APPROVAL, "test")
    t = last_close_time(runtime.store)
    px = runtime.store.view(t).price("BTCUSDT")
    p = TradeProposal("BTCUSDT", Direction.LONG, 75, None, 0.02, "4h", px, px * 0.97, px * 1.06, 500)
    d = RiskDecision(p.id, True, 400, 500, {})
    req = e.approvals.request(p, d, None, t)
    assert e.approvals.pending() and e.book.qty("BTCUSDT") == 0
    assert "Reply /yes" in req.summary()
    e.approvals.decide(req.short_id, True, "owner", t)
    e.execute_approved({"BTCUSDT": px}, t)
    assert e.book.qty("BTCUSDT") > 0 and req.status == "EXECUTED"
    p2 = TradeProposal("ETHUSDT", Direction.LONG, 75, None, 0.02, "4h", 100, 97, 106, 500)
    r2 = e.approvals.request(p2, RiskDecision(p2.id, True, 400, 500, {}), None, t)
    e.approvals.decide(r2.short_id, False, "owner", t)
    e.execute_approved({"ETHUSDT": 100}, t)
    assert e.book.qty("ETHUSDT") == 0 and r2.status == "REJECTED"


def test_approval_expires():
    q = ApprovalQueue(timeout_minutes=1)
    now = dt.datetime(2026, 1, 1)
    p = TradeProposal("BTCUSDT", Direction.LONG, 70, None, 0, "4h", 100, 97, 106, 100)
    r = q.request(p, RiskDecision(p.id, True, 100, 100, {}), None, now)
    q.expire(now + dt.timedelta(minutes=2))
    assert r.status == "EXPIRED"
    assert q.decide(r.short_id, True, "late", now + dt.timedelta(minutes=3)).status == "EXPIRED"


def test_voice_parser_and_agent(runtime):
    assert parse("stop all new trades").name == "pause"
    assert parse("emergency close everything").sensitive
    assert parse("What's happening with ETH?").symbol == "ETHUSDT"
    assert parse("switch to live mode").args == {"mode": "LIVE"}
    runtime.engine.cycle(last_close_time(runtime.store))
    v = runtime.voice
    assert "regime" in v.handle_text("what's happening with BTC?")["answer"].lower()
    out = v.handle_text("stop all new trades")
    assert out["executed"] and runtime.engine.paused
    denied = v.handle_text("resume trading")
    assert not denied["executed"] and runtime.engine.paused
    assert v.handle_text("resume trading", pin="4321")["executed"] and not runtime.engine.paused
    assert not v.handle_text("emergency close", pin="0000")["executed"]


def test_api_auth_and_controls(runtime):
    runtime.engine.cycle(last_close_time(runtime.store))
    from lunatrade.api.app import create_app

    c = TestClient(create_app(runtime))
    assert c.get("/api/status").status_code == 401
    assert c.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    H = {"Authorization": "Bearer tok"}
    st = c.get("/api/status", headers=H).json()
    assert st["mode"] == "PAPER" and st["workers"]["total"] == 300
    assert c.get("/", headers=H).status_code == 200
    assert c.get("/health").status_code in (200, 503)
    assert "lunatrade_equity" in c.get("/metrics").text
    assert len(c.get("/api/agents", headers=H).json()["workers"]) == 300
    assert c.post("/api/control/pause", headers=H).json()["paused"]
    assert c.post("/api/control/resume", headers=H).status_code == 403          # PIN required
    assert c.post("/api/control/resume", headers={**H, "X-Control-Pin": "4321"}).json()["paused"] is False
    r = c.post("/api/voice/command", headers=H, json={"text": "what's my pnl"}).json()
    assert "Equity" in r["answer"]
    assert c.post("/api/control/emergency-close", headers=H).status_code == 403
    assert c.post("/api/control/emergency-close", headers={**H, "X-Control-Pin": "4321"}).status_code == 200
    assert runtime.engine.kill_switch.active                                       # emergency trips the switch
    m = c.post("/api/control/mode", headers={**H, "X-Control-Pin": "4321"}, json={"mode": "LIVE"}).json()
    assert "Cannot switch" in m["result"]                                          # promotion gates
    assert c.get("/api/promotion/SHADOW", headers=H).json()["allowed"] is False


def test_backtest_end_to_end_and_robustness(cfg, data):
    small = {s: df.iloc[:330] for s, df in data.items()}
    res = Backtester(cfg, small, warmup=200).run()
    m = res.metrics
    for k in ("total_return", "sharpe", "sortino", "max_drawdown", "profit_factor", "expectancy", "avg_r", "tail_loss",
              "turnover", "win_rate", "trades"):
        assert k in m
    assert len(res.equity_curve) == m["bars"] and m["final_equity"] > 0
    assert 0 <= m["max_drawdown"] < 1
    mc = monte_carlo([{"pnl": x} for x in (10, -5, 7, -3, 12, -8, 4)], 1000, runs=200)
    assert mc["final_equity_p5"] <= mc["final_equity_p50"] <= mc["final_equity_p95"]
    st = stress_test(cfg, {s: df.iloc[:300] for s, df in data.items()}, ["flash_crash", "api_outage", "partial_fills"],
                     warmup=200)
    assert set(st) == {"flash_crash", "api_outage", "partial_fills"}
    assert all("error" not in v for v in st.values()), st
    assert st["api_outage"]["kill_switch_events"] >= 1          # stale data halted new trades
    wf = walk_forward(cfg, {s: df.iloc[:360] for s, df in data.items()}, folds=2, warmup=200)
    assert wf["folds"] and "out_of_sample" in wf["folds"][0]

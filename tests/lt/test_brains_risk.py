import datetime as dt

import pytest

from lunatrade.brains.devils_advocate import DevilsAdvocate
from lunatrade.brains.lead_brain import LeadBrain
from lunatrade.brains.regime_engine import RegimeEngine, RegimeState, classify
from lunatrade.core.types import (DevilsAdvocateReport, Direction, Family, Regime, Signal, TradeProposal)
from lunatrade.risk.engine import AccountState, MarketConditions, RiskEngine
from lunatrade.risk.kill_switch import KillSwitch
from lunatrade.risk.portfolio import PortfolioBrain

L, S, N = Direction.LONG, Direction.SHORT, Direction.NEUTRAL


def sig(fam, name, d, conf, sym="BTCUSDT", **ctx):
    return Signal(f"{fam.value}.{name}", fam, sym, d, conf, context=ctx)


def bullish_signals():
    out = []
    for i, n in enumerate(["ema_cross_9_21", "macd_12_26_9", "donchian_breakout_20", "adx_trend_14_25"]):
        out.append(sig(Family.TECHNICAL, n, L, 0.7))
    out += [sig(Family.MOMENTUM, "roc_24", L, 0.7), sig(Family.MOMENTUM, "volume_expansion_20", L, 0.6),
            sig(Family.FORECAST, "linear_trend_48", L, 0.7), sig(Family.FORECAST, "holt_es", L, 0.6),
            sig(Family.ONCHAIN, "stablecoin_growth_7d", L, 0.6, sym="*"),
            sig(Family.WHALE, "net_exchange_flow_24h", L, 0.6), sig(Family.SCANNER, "relative_strength_24", L, 0.6),
            sig(Family.MICROSTRUCTURE, "taker_flow_12", L, 0.6), sig(Family.NEWS, "news_composite", L, 0.5)]
    return out


def regime(primary=Regime.TRENDING, **scores):
    return RegimeState("BTCUSDT", primary, {primary.value: 0.8, **scores}, "BULL", 10, 0.8)


def test_regime_classification_priorities():
    assert classify({"PANIC": 0.6, "TRENDING": 0.9}, 10)[0] == Regime.PANIC
    assert classify({"TRENDING": 0.5, "RANGING": 0.2}, 10)[0] == Regime.TRENDING
    assert classify({"TRENDING": 0.1, "RANGING": 0.6}, 10)[0] == Regime.RANGING
    assert classify({"TRENDING": 0.9}, 2)[0] == Regime.UNKNOWN
    votes = [Signal(f"regime.w{i}", Family.REGIME, "BTCUSDT", N, 0, context={"votes": {"TRENDING": 0.9}})
             for i in range(5)]
    st = RegimeEngine().compute(votes, ["BTCUSDT"], "BTCUSDT")
    assert st["BTCUSDT"].primary == Regime.TRENDING and st["*"].primary == Regime.TRENDING
    assert st["BTCUSDT"].group_multiplier("TREND") > 1 > st["BTCUSDT"].group_multiplier("MEAN_REVERSION")
    assert RegimeState("X", Regime.PANIC).exposure_multiplier < 0.5


def test_lead_brain_proposes_on_broad_agreement(cfg):
    lb = LeadBrain(cfg.section("lead_brain"), cfg.section("risk"))
    a = lb.assess("BTCUSDT", bullish_signals(), regime())
    assert a.direction == L and a.families_agreeing >= 5 and a.conviction >= 62
    p = lb.propose(a, 100.0, 0.01, 10_000, 0.0, regime())
    assert p and p.action == "OPEN" and p.stop_price < 100 < p.target_price
    assert p.win_probability is None              # no track record -> no fake probability
    assert 0 < p.requested_notional <= 10_000 * cfg.get("risk.max_position_pct")


def test_lead_brain_needs_independent_families(cfg):
    lb = LeadBrain(cfg.section("lead_brain"), cfg.section("risk"))
    only_tech = [sig(Family.TECHNICAL, f"ema_cross_{i}", L, 0.9) for i in range(20)]
    a = lb.assess("BTCUSDT", only_tech, regime())
    assert a.families_agreeing == 1 and a.conviction <= 55
    assert lb.propose(a, 100, 0.01, 10_000, 0, regime()) is None


def test_lead_brain_closes_long_on_bearish_conviction(cfg):
    lb = LeadBrain(cfg.section("lead_brain"), cfg.section("risk"))
    bear = [Signal(s.worker_id, s.family, s.symbol, S, s.confidence) for s in bullish_signals()]
    a = lb.assess("BTCUSDT", bear, regime())
    assert lb.propose(a, 100, 0.01, 10_000, 0.0, regime()) is None          # spot: never opens shorts
    p = lb.propose(a, 100, 0.01, 10_000, 1.5, regime())
    assert p and p.action == "CLOSE"


class FakeCouncil:
    def __init__(self, answer):
        self.answer = answer

    def ask_json(self, role, task, payload):
        return dict(self.answer)


def test_llm_review_is_bounded(cfg):
    lb = LeadBrain(cfg.section("lead_brain"), cfg.section("risk"), council=FakeCouncil({"adjustment": 50}))
    a = lb.assess("BTCUSDT", bullish_signals(), regime())
    p = lb.propose(a, 100, 0.01, 10_000, 0, regime())
    before = p.conviction
    lb.llm_review(p, a)
    assert p.conviction == pytest.approx(min(100, before + cfg.get("lead_brain.llm_max_up")))
    lb2 = LeadBrain(cfg.section("lead_brain"), cfg.section("risk"), council=FakeCouncil({"adjustment": 0, "veto": True}))
    p2 = lb2.propose(a, 100, 0.01, 10_000, 0, regime())
    b2 = p2.conviction
    lb2.llm_review(p2, a)
    assert p2.conviction == pytest.approx(b2 - cfg.get("lead_brain.llm_max_down"))


def _proposal(conv=75):
    return TradeProposal("BTCUSDT", L, conv, None, 0.03, "4h", 100, 96, 108, 1000, regime="TRENDING")


def test_devils_advocate_vetoes_when_bear_case_dominates(cfg):
    da = DevilsAdvocate(cfg.section("devils_advocate"))
    risky = [Signal("adversarial.event_risk", Family.ADVERSARIAL, "BTCUSDT", N, 0,
                    context={"risk_points": 4, "reasons": ["FOMC in 1.0h"]}),
             Signal("adversarial.crowding", Family.ADVERSARIAL, "BTCUSDT", N, 0,
                    context={"risk_points": 3, "reasons": ["funding elevated"]})]
    rep = da.review(_proposal(66), risky, RegimeState("BTCUSDT", Regime.PANIC, {"PANIC": 0.8}, "BEAR", 10))
    assert rep.veto and rep.bear_case > rep.bull_case and "FOMC in 1.0h" in rep.hidden_risks
    clean = da.review(_proposal(80), [], regime())
    assert not clean.veto and clean.risk_adjustment > -0.2


def test_devils_advocate_depeg_is_hard_veto(cfg):
    da = DevilsAdvocate(cfg.section("devils_advocate"))
    s = [Signal("arbitrage.depeg_usdcusdt", Family.ARBITRAGE, "*", S, 0.6, context={"depeg_risk": True, "pair": "USDCUSDT"})]
    assert da.review(_proposal(95), s, regime()).veto


def acct(equity=10_000, cash=10_000, positions=None, peak=None, day=None):
    return AccountState(equity, cash, peak or equity, day or equity, equity, positions or {})


def test_risk_engine_sizes_by_risk_budget(cfg):
    re_ = RiskEngine(cfg.section("risk"))
    d = re_.evaluate(_proposal(), None, acct(), MarketConditions(100, 0.01, spread_bps=2, quote_volume_24h=1e9,
                                                                 expected_slippage_bps=3))
    assert d.approved
    stop_pct = 0.04
    assert d.approved_notional <= 10_000 * 0.01 / stop_pct + 1e-6      # 1% equity at risk
    assert d.approved_notional <= 1000                                  # never more than requested
    assert all(not v.startswith("FAIL") for v in d.checks.values())


def test_risk_engine_rejects_and_reduces(cfg):
    re_ = RiskEngine(cfg.section("risk"))
    m = MarketConditions(100, 0.01, spread_bps=50)
    d = re_.evaluate(_proposal(), None, acct(), m)
    assert not d.approved and d.checks["spread"].startswith("FAIL")
    lost = acct(equity=9_600, day=10_000)
    d = re_.evaluate(_proposal(), None, lost, MarketConditions(100, 0.01))
    assert not d.approved and d.checks["daily_loss"].startswith("FAIL")
    veto = DevilsAdvocateReport("x", 5, 9, ["bad"], True, -0.5)
    assert not re_.evaluate(_proposal(), veto, acct(), MarketConditions(100, 0.01)).approved
    full = acct(positions={f"S{i}USDT": {"qty": 1, "notional": 100} for i in range(6)})
    assert re_.evaluate(_proposal(), None, full, MarketConditions(100, 0.01)).checks["open_positions"].startswith("FAIL")
    hv = re_.evaluate(_proposal(), None, acct(), MarketConditions(100, 0.01, high_vol=True))
    base = re_.evaluate(_proposal(), None, acct(), MarketConditions(100, 0.01))
    assert hv.approved_notional < base.approved_notional


def test_risk_engine_respects_kill_switch(cfg, tmp_path):
    ks = KillSwitch(tmp_path, cfg.section("kill_switch"), cfg.section("risk"))
    re_ = RiskEngine(cfg.section("risk"), ks)
    ks.trip("MANUAL_STOP", "test")
    d = re_.evaluate(_proposal(), None, acct(), MarketConditions(100, 0.01))
    assert not d.approved and d.checks["kill_switch"].startswith("FAIL")
    close = TradeProposal("BTCUSDT", S, 60, None, 0, "4h", 100, 96, 108, 0, action="CLOSE")
    assert re_.evaluate(close, None, acct(positions={"BTCUSDT": {"qty": 1, "notional": 100}}),
                        MarketConditions(100, 0.01)).approved      # risk-reducing exits still allowed


def test_kill_switch_hard_trips_persist_and_need_human(cfg, tmp_path):
    ks = KillSwitch(tmp_path, cfg.section("kill_switch"), cfg.section("risk"))
    ks.check_account(equity=8_000, peak_equity=10_000, day_start_equity=8_100, week_start_equity=8_100)
    assert "MAX_DRAWDOWN" in ks.trips and ks.active
    again = KillSwitch(tmp_path, cfg.section("kill_switch"), cfg.section("risk"))   # restart
    assert again.active
    again.reset("tester")
    assert not again.active
    ks2 = KillSwitch(tmp_path / "b", cfg.section("kill_switch"), cfg.section("risk"))
    ks2.check_data({"BTCUSDT": 10_000})
    assert ks2.active and "STALE_DATA" in ks2.halts
    ks2.check_data({"BTCUSDT": 5})
    assert not ks2.active                       # soft halt clears itself
    (tmp_path / "b" / "STOP").write_text("")
    assert ks2.active


def test_portfolio_brain_caps_correlated_cluster(cfg):
    pb = PortfolioBrain(cfg.section("risk"))
    from lunatrade.core.types import RiskDecision

    d = RiskDecision("p", True, 1000, 1000, {})
    positions = {"ETHUSDT": {"notional": 1800}, "SOLUSDT": {"notional": 1500}}
    clusters = [["BTCUSDT", "ETHUSDT", "SOLUSDT"]]
    out = pb.adjust(_proposal(), d, 10_000, positions, clusters, {})
    assert out.approved_notional <= 10_000 * 0.35 - 3300 + 1e-6
    d2 = RiskDecision("p", True, 1000, 1000, {})
    positions2 = {"ETHUSDT": {"notional": 2000}, "SOLUSDT": {"notional": 1500}}
    assert not pb.adjust(_proposal(), d2, 10_000, positions2, clusters, {}).approved

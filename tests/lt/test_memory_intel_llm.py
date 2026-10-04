import datetime as dt

import pytest

from lunatrade.core.events import Event, Topic
from lunatrade.core.types import Direction, Family, Signal, TradeProposal
from lunatrade.data.feeds import parse_rss
from lunatrade.intel import sentiment
from lunatrade.intel.assets import extract_assets
from lunatrade.intel.news_pipeline import NewsPipeline
from lunatrade.intel.social import Post, analyze
from lunatrade.intel.whale import Transfer, interpret
from lunatrade.llm.council import LLMCouncil, combine
from lunatrade.llm.providers import Provider, RuleProvider, parse_json
from lunatrade.memory.knowledge import EventMemory
from lunatrade.memory.performance import StrategyMemory, horizon_bars, trade_stats
from lunatrade.memory.repository import Repository
from lunatrade.memory.schema import TABLES, postgres_ddl

NOW = dt.datetime(2026, 10, 1, 12)


# ---------------------------------------------------------------- database
def test_schema_has_all_required_tables():
    required = {"agents", "agent_events", "market_data", "market_snapshots", "signals", "trade_proposals",
                "risk_decisions", "orders", "fills", "positions", "portfolio_snapshots", "news_events",
                "sentiment_events", "whale_events", "strategy_performance", "trade_outcomes", "system_alerts",
                "agent_health", "model_versions", "backtests"}
    assert required <= set(TABLES)
    ddl = postgres_ddl()
    assert "CREATE TABLE trade_proposals" in ddl and "JSON" in ddl


def test_repository_roundtrip(tmp_path):
    repo = Repository(f"sqlite:///{tmp_path / 'db.sqlite'}")
    p = TradeProposal("BTCUSDT", Direction.LONG, 70, None, 0.02, "4h", 100, 96, 108, 500, rationale="test")
    repo.proposal(p, "PENDING")
    repo.proposal_status(p.id, "EXECUTED")
    repo.event(Event(Topic.TRADE_PROPOSAL, {"x": 1}, correlation_id=p.id))
    repo.trade_outcome({"proposal_id": p.id, "symbol": "BTCUSDT", "pnl": 12.5, "return_pct": 0.02, "r_multiple": 0.6,
                        "supporting": ["technical.ema_cross_9_21"]}, "PAPER")
    repo.strategy("TREND", {"samples": 30, "hit_rate": 0.55, "weight": 1.1})
    ex = repo.explain(p.id)
    assert ex["proposal"]["status"] == "EXECUTED" and ex["outcome"][0]["pnl"] == 12.5
    assert repo.counts()["agent_events"] == 1 and repo.ping()
    repo.strategy("TREND", {"samples": 40, "hit_rate": 0.6, "weight": 1.2})     # upsert
    assert repo.strategies()[0]["samples"] == 40


# ---------------------------------------------------------------- strategy memory / meta-learning
def _sig(worker, d, sym="BTCUSDT", conf=0.6, h="4h"):
    return Signal(worker, Family.TECHNICAL, sym, d, conf, time_horizon=h)


def test_signal_scoring_and_bounded_weights():
    mem = StrategyMemory(900_000, {"min_samples": 5, "min_weight": 0.25, "max_weight": 2.0, "max_step": 0.1})
    t = 1_700_000_000_000
    for i in range(30):
        # trend signals are always right, mean-reversion always wrong
        mem.record_signals([_sig("technical.ema_cross_9_21", Direction.LONG, h="1bars"),
                            _sig("technical.bollinger_20_2.0", Direction.SHORT, h="1bars")],
                           {"BTCUSDT": 100 + i}, t + i * 900_000)
        mem.evaluate_due({"BTCUSDT": 100 + i + 1}, t + (i + 1) * 900_000)
    assert mem.groups["TREND"].hit_rate > 0.9 and mem.groups["MEAN_REVERSION"].hit_rate < 0.1
    for _ in range(3):
        mem.update_weights()
    w = mem.weights()
    assert w["TREND"] > 1.0 > w["MEAN_REVERSION"]
    assert w["TREND"] <= 1.1 ** 3 + 1e-9                      # max 10% per step
    for _ in range(100):
        mem.update_weights()
    assert 0.25 <= mem.weights()["MEAN_REVERSION"] and mem.weights()["TREND"] <= 2.0


def test_threshold_and_calibration():
    mem = StrategyMemory(900_000, {"conviction_threshold_bounds": [55, 80]})
    for i in range(20):
        mem.record_outcome({"pnl": -10, "return_pct": -0.01, "r_multiple": -1, "conviction": 70, "supporting": []})
    mem.update_weights()
    assert mem.min_conviction(62) > 62
    assert mem.calibration(70) == 0.0 and mem.calibration(90) is None
    assert horizon_bars("4h", 900_000) == 16 and horizon_bars("8bars", 900_000) == 8
    ts = trade_stats([{"pnl": 10, "return_pct": 0.01, "r_multiple": 1}, {"pnl": -5, "return_pct": -0.005, "r_multiple": -0.5}])
    assert ts["profit_factor"] == 2 and ts["win_rate"] == 0.5


# ---------------------------------------------------------------- news / social / whale
def test_entities_and_sentiment():
    assert set(extract_assets("Bitcoin and Ethereum rally while $SOL lags")) >= {"BTC", "ETH", "SOL"}
    assert "LINK" not in extract_assets("Click the link below")
    assert sentiment.score("Bitcoin surges to record high on ETF approval") > 0.3
    assert sentiment.score("Exchange hacked, funds stolen in massive exploit") < -0.3
    assert sentiment.score("SEC does not approve the ETF") < 0


def test_news_pipeline_stages():
    np_ = NewsPipeline({"coindesk": 0.9, "blog": 0.4})
    a = np_.process("coindesk", "Bitcoin ETF sees record inflows", "", ts=NOW - dt.timedelta(hours=1))
    assert a.event_type == "ETF_FLOW" and "BTC" in a.assets and a.sentiment > 0 and a.novelty == 1.0
    dup = np_.process("coindesk", "Bitcoin ETF sees record inflows", "", ts=NOW)
    assert dup is None                                          # exact duplicate dropped
    b = np_.process("blog", "Record inflows for Bitcoin ETF this week", "", ts=NOW - dt.timedelta(minutes=30))
    assert b.novelty < 1 and b.corroboration >= 2
    h = np_.process("coindesk", "DeFi protocol hacked, $80M drained in exploit", "", ts=NOW - dt.timedelta(minutes=5))
    assert h.event_type == "HACK_EXPLOIT" and h.sentiment < 0
    s_now, _ = np_.asset_signal("BTC", NOW)
    s_later, _ = np_.asset_signal("BTC", NOW + dt.timedelta(days=5))
    assert s_now > 0 and abs(s_later) < abs(s_now)              # time decay


def test_social_classification():
    now = NOW
    organic = [Post(now - dt.timedelta(minutes=i * 5), "r/Bitcoin", f"BTC adoption is growing, good news #{i}",
                    author=f"u{i}", score=20) for i in range(20)]
    assert analyze(organic, "BTC", now).classification == "ORGANIC"
    bots = [Post(now - dt.timedelta(minutes=i), "x", "BTC to the moon 100x buy now!!!", author=f"b{i % 2}")
            for i in range(20)]
    assert analyze(bots, "BTC", now).classification in ("BOT_ACTIVITY", "COORDINATED_HYPE")
    panic = [Post(now - dt.timedelta(minutes=i * 3), "r/CryptoCurrency", f"BTC crash, I'm rekt and selling everything {i}",
                  author=f"p{i}") for i in range(20)]
    assert analyze(panic, "BTC", now).classification in ("PANIC", "CAPITULATION")


def test_whale_interpretation_is_contextual():
    t = lambda a, f, to, usd=20e6, **k: Transfer(NOW, a, usd, f, to, **k)
    assert interpret(t("BTC", "unknown", "exchange")).bias < 0
    assert interpret(t("BTC", "exchange", "unknown")).bias > 0
    assert interpret(t("BTC", "exchange", "exchange", from_owner="binance", to_owner="binance")).bias == 0
    assert interpret(t("USDT", "unknown", "exchange")).bias > 0 and interpret(t("USDT", "unknown", "exchange")).asset == "*"
    assert interpret(t("USDC", "treasury", "unknown")).kind == "STABLE_MINT"
    assert interpret(t("BTC", "unknown", "exchange")).confidence < 0.5      # ambiguous by nature


def test_rss_parser():
    xml = """<rss><channel><item><title>Bitcoin hits new high</title><link>https://x/1</link>
             <description>&lt;p&gt;BTC rallies&lt;/p&gt;</description><pubDate>Wed, 01 Oct 2025 10:00:00 GMT</pubDate></item>
             </channel></rss>"""
    items = parse_rss(xml)
    assert items[0]["title"] == "Bitcoin hits new high" and items[0]["ts"].year == 2025


def test_event_memory_learns_impact_priors():
    em = EventMemory()
    np_ = NewsPipeline()
    for i in range(12):
        it = np_.process("s", f"Exchange hacked exploit number {i} drained funds", "", ts=NOW + dt.timedelta(minutes=i))
        em.track(it, {"BTC": 100.0})
        it2 = np_.process("s", f"Project announces governance proposal vote {i}", "", ts=NOW + dt.timedelta(minutes=i))
        it2.market_wide = True
        em.track(it2, {"BTC": 100.0})
    em.update(NOW + dt.timedelta(hours=5), {"BTC": 95.0})
    pri = em.impact_priors(min_samples=5)
    assert pri and all(0.5 <= v <= 1.6 for v in pri.values())


# ---------------------------------------------------------------- LLM council
def test_parse_json_tolerates_wrapping():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('blah {"veto": false, "x": {"y": 2}} trailing') == {"veto": False, "x": {"y": 2}}
    assert parse_json("no json") is None


def test_combine_median_majority_union():
    out = combine([{"adjustment": -10, "veto": True, "concerns": ["a"]},
                   {"adjustment": 0, "veto": False, "concerns": ["b"]},
                   {"adjustment": -4, "veto": True, "concerns": ["a", "c"]}])
    assert out["adjustment"] == -4 and out["veto"] is True and out["concerns"] == ["a", "b", "c"]


class Echo(Provider):
    kind = "echo"

    def __init__(self, name, text):
        super().__init__(name, "m")
        self.text = text

    def _complete(self, system, user):
        assert "untrusted data" in system
        return self.text


def test_council_uses_members_and_falls_back_to_rules():
    c = LLMCouncil({"a": Echo("a", '{"adjustment": -6}'), "b": Echo("b", '{"adjustment": -2}'),
                    "bad": Echo("bad", "not json")}, {"lead_review": ["a", "b", "bad"]}, council_size=3)
    res = c.ask_json("lead_review", "lead_review", {"proposal": {}})
    assert res["adjustment"] == -4 and set(res["providers"]) == {"a", "b"}
    empty = LLMCouncil({}, {"lead_review": []})
    assert empty.ask_json("lead_review", "lead_review", {"proposal": {"regime": "PANIC"}})["providers"] == ["rules"]
    budget = LLMCouncil({"a": Echo("a", '{"adjustment": 1}')}, {"r": ["a"]}, max_calls_per_hour=1)
    budget.ask_json("r", "lead_review", {})
    assert budget.ask_json("r", "lead_review", {"proposal": {}})["providers"] == ["rules"]

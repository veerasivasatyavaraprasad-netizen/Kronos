"""
LunaTrade command line.

    python -m lunatrade run                      # 24/7 engine + dashboard/API (mode from config, PAPER by default)
    python -m lunatrade cycle                    # bootstrap data and run one decision cycle, print the result
    python -m lunatrade backtest --symbols BTCUSDT,ETHUSDT --bars 1500 [--walk-forward] [--monte-carlo] [--stress]
    python -m lunatrade backtest --synthetic     # offline backtest on generated data
    python -m lunatrade agents                   # the 300-worker roster
    python -m lunatrade check                    # which keys/services are configured and reachable (no secrets shown)
    python -m lunatrade universe                 # liquid symbols discovered right now
    python -m lunatrade db init|sql|counts       # create tables / print PostgreSQL DDL / row counts
    python -m lunatrade ask "what's happening with BTC?"
"""
from __future__ import annotations

import argparse
import json
import sys


def _cfg(args):
    from lunatrade.config import load_config

    overrides = {}
    if getattr(args, "mode", None):
        overrides["mode"] = args.mode.upper()
    if getattr(args, "interval", None):
        overrides["interval"] = args.interval
    return load_config(getattr(args, "config", None), overrides)


def cmd_run(args):
    from lunatrade.orchestrator import Runtime

    Runtime(_cfg(args)).run_forever(api=not args.no_api)


def cmd_cycle(args):
    from lunatrade.orchestrator import Runtime

    rt = Runtime(_cfg(args))
    rt.cfg.set("data.websocket", False)
    res = rt.run_cycle()
    print(json.dumps({**res, "status": rt.control.status_sentence()}, indent=2, default=str))


def _load_data(args, cfg):
    from lunatrade.data.synthetic import correlated_universe

    syms = [s.strip().upper() for s in args.symbols.split(",")]
    if args.synthetic:
        return correlated_universe(syms, args.bars, args.interval or cfg.get("interval"))
    if args.csv:
        import pandas as pd

        from automation.data_sources import load_csv

        df = load_csv(args.csv).rename(columns={"timestamps": "open_time", "amount": "quote_volume"})
        return {syms[0]: df}
    from lunatrade.data.binance_public import BinancePublic

    pub = BinancePublic()
    return {s: pub.klines(s, args.interval or cfg.get("interval"), args.bars) for s in syms}


def cmd_backtest(args):
    from lunatrade.backtest.engine import Backtester
    from lunatrade.backtest.robustness import monte_carlo, stress_test, walk_forward

    cfg = _cfg(args)
    data = _load_data(args, cfg)
    out = {}
    res = Backtester(cfg, data, interval=args.interval, warmup=args.warmup, decision_every=args.every).run()
    out["backtest"] = res.metrics
    if args.monte_carlo:
        out["monte_carlo"] = monte_carlo(res.trades, res.metrics["starting_equity"])
    if args.walk_forward:
        out["walk_forward"] = walk_forward(cfg, data, folds=args.folds, warmup=args.warmup, interval=args.interval,
                                           decision_every=args.every)
    if args.stress:
        out["stress"] = stress_test(cfg, data, warmup=args.warmup, interval=args.interval, decision_every=args.every)
    try:
        from lunatrade.core.types import new_id
        from lunatrade.memory.repository import Repository

        Repository(cfg.database_url).backtest(new_id("bt_"), args.name, {"symbols": list(data), "bars": args.bars,
                                                                        "interval": args.interval}, out,
                                              [{"ts": t, "equity": e} for t, e in zip(res.timestamps, res.equity_curve)])
    except Exception as e:
        print(f"(backtest not stored: {e})", file=sys.stderr)
    print(json.dumps(out, indent=2, default=str))


def cmd_agents(args):
    from lunatrade.workers.registry import build_roster, roster_summary

    ws = build_roster(_cfg(args))
    if args.full:
        for w in ws:
            print(f"{w.id:55s} phase {w.phase}  {json.dumps(w.params, default=str)}")
    print(json.dumps({"total": len(ws), **roster_summary(ws)}, indent=2))


def cmd_check(args):
    import os

    import requests

    from lunatrade.config import live_trading_allowed, secret

    cfg = _cfg(args)
    rows = []

    def row(name, ok, detail=""):
        rows.append((name, "OK " if ok else "-- ", detail))

    for var in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY", "BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_SECRET_KEY",
                "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "VOICE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "WHALE_ALERT_API_KEY", "CRYPTOPANIC_API_KEY", "X_BEARER_TOKEN",
                "TELEGRAM_BOT_TOKEN", "LUNATRADE_API_TOKEN", "LUNATRADE_CONTROL_PIN"):
        v = secret(var)
        row(var, bool(v), f"set ({len(v)} chars)" if v else "not set")
    row("mode", True, cfg.get("mode"))
    row("live switch", True, "ON (real money possible)" if live_trading_allowed() else "off")
    try:
        r = requests.get("https://api.binance.com/api/v3/time", timeout=10)
        row("binance public", r.ok, f"HTTP {r.status_code}")
    except Exception as e:
        row("binance public", False, str(e)[:80])
    for env in ("testnet", "live"):
        k = "BINANCE_TESTNET_API_KEY" if env == "testnet" else "BINANCE_API_KEY"
        if not secret(k):
            continue
        try:
            from lunatrade.execution.brokers.binance_spot import BinanceSpotBroker

            b = BinanceSpotBroker(env, validate_only=True)
            bal = b.balances()
            row(f"binance {env} signed", True, f"{len(bal)} assets with balance")
        except Exception as e:
            row(f"binance {env} signed", False, str(e)[:120])
    if secret("ALPACA_API_KEY"):
        try:
            from lunatrade.execution.brokers.alpaca import AlpacaBroker

            a = AlpacaBroker(cfg.get("brokers.alpaca.environment", "paper"))
            row("alpaca", True, f"cash {a.balances().get('USD', 0):,.2f} ({a.mode})")
        except Exception as e:
            row("alpaca", False, str(e)[:120])
    from lunatrade.llm.council import LLMCouncil

    c = LLMCouncil.from_config(cfg)
    for p in c.status()["providers"]:
        row(f"llm {p['name']}", p["available"], p["model"])
    row("voice provider", bool(cfg.get("voice.provider")), cfg.get("voice.provider") or "VOICE_PROVIDER not set")
    try:
        import torch  # noqa: F401

        row("kronos (torch)", True, "available")
    except Exception:
        row("kronos (torch)", False, "torch not installed - Kronos forecasters offline")
    width = max(len(r[0]) for r in rows)
    for name, st, detail in rows:
        print(f"{st} {name:<{width}}  {detail}")


def cmd_universe(args):
    from lunatrade.data.binance_public import BinancePublic
    from lunatrade.data.universe import select_universe

    cfg = _cfg(args)
    pub = BinancePublic()
    ranked = select_universe(pub.ticker_24h(), pub.book_ticker(), pub.exchange_info(), cfg.section("universe"))
    print(json.dumps(ranked, indent=2, default=str))


def cmd_db(args):
    from lunatrade.memory.repository import Repository
    from lunatrade.memory.schema import postgres_ddl

    cfg = _cfg(args)
    if args.action == "sql":
        print(postgres_ddl())
        return
    repo = Repository(cfg.database_url)
    if args.action == "init":
        print(f"tables ready at {cfg.database_url.split('@')[-1]}")
    else:
        print(json.dumps(repo.counts(), indent=2))


def cmd_ask(args):
    from lunatrade.voice.parser import parse

    i = parse(args.text)
    print(json.dumps({"intent": i.name, "symbol": i.symbol, "args": i.args, "sensitive": i.sensitive}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lunatrade", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to lunatrade.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--mode")
    p.add_argument("--interval")
    p.add_argument("--no-api", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("cycle")
    p.add_argument("--mode")
    p.add_argument("--interval")
    p.set_defaults(func=cmd_cycle)

    p = sub.add_parser("backtest")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--interval", default=None)
    p.add_argument("--bars", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--every", type=int, default=1, help="decide every N bars")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--csv")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--monte-carlo", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--name", default="cli")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("agents")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_agents)

    sub.add_parser("check").set_defaults(func=cmd_check)
    sub.add_parser("universe").set_defaults(func=cmd_universe)

    p = sub.add_parser("db")
    p.add_argument("action", choices=["init", "sql", "counts"])
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("ask")
    p.add_argument("text")
    p.set_defaults(func=cmd_ask)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

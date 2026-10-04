"""
Database schema (SQLAlchemy Core). PostgreSQL in production, SQLite for local/paper/tests.
`python -m lunatrade db sql` prints the PostgreSQL DDL (also committed as migrations/0001_initial.sql).
"""
from __future__ import annotations

from sqlalchemy import (JSON, BigInteger, Boolean, Column, DateTime, Float, Index, Integer, MetaData, String, Table,
                        Text)

metadata = MetaData()


def _id():
    return Column("id", String(40), primary_key=True)


agents = Table("agents", metadata, Column("id", String(80), primary_key=True), Column("name", String(80)),
               Column("family", String(30), index=True), Column("phase", Integer), Column("kind", String(20)),
               Column("params", JSON), Column("status", String(20)), Column("updated_at", DateTime))

agent_events = Table("agent_events", metadata, _id(), Column("ts", DateTime, index=True),
                     Column("topic", String(60), index=True), Column("source", String(60)),
                     Column("correlation_id", String(60), index=True), Column("payload", JSON))

market_data = Table("market_data", metadata, Column("symbol", String(30), primary_key=True),
                    Column("interval", String(5), primary_key=True), Column("open_time", BigInteger, primary_key=True),
                    Column("open", Float), Column("high", Float), Column("low", Float), Column("close", Float),
                    Column("volume", Float), Column("quote_volume", Float), Column("trades", Float),
                    Column("taker_buy_base", Float))

market_snapshots = Table("market_snapshots", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                         Column("ts", DateTime, index=True), Column("symbol", String(30), index=True),
                         Column("price", Float), Column("spread_bps", Float), Column("regime", String(80)),
                         Column("assessment", JSON))

signals = Table("signals", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                Column("ts", DateTime, index=True), Column("cycle_id", String(40), index=True),
                Column("worker_id", String(80), index=True), Column("family", String(30)),
                Column("symbol", String(30), index=True), Column("direction", String(8)), Column("confidence", Float),
                Column("setup", String(60)), Column("expected_move", Float), Column("horizon", String(12)),
                Column("evidence", JSON), Column("context", JSON), Column("price", Float),
                Column("outcome_return", Float), Column("evaluated_at", DateTime))

trade_proposals = Table("trade_proposals", metadata, _id(), Column("ts", DateTime, index=True),
                        Column("symbol", String(30), index=True), Column("direction", String(8)),
                        Column("action", String(10)), Column("conviction", Float), Column("win_probability", Float),
                        Column("entry", Float), Column("stop", Float), Column("target", Float),
                        Column("requested_notional", Float), Column("regime", String(30)), Column("rationale", Text),
                        Column("status", String(20), index=True), Column("data", JSON))

risk_decisions = Table("risk_decisions", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                       Column("proposal_id", String(40), index=True), Column("ts", DateTime),
                       Column("approved", Boolean), Column("approved_notional", Float),
                       Column("requested_notional", Float), Column("checks", JSON), Column("reasons", JSON),
                       Column("devils_advocate", JSON))

approvals = Table("approvals", metadata, Column("proposal_id", String(40), primary_key=True),
                  Column("requested_at", DateTime), Column("expires_at", DateTime), Column("status", String(12)),
                  Column("decided_by", String(60)), Column("decided_at", DateTime), Column("summary", Text))

orders = Table("orders", metadata, Column("client_order_id", String(40), primary_key=True),
               Column("proposal_id", String(40), index=True), Column("created_at", DateTime, index=True),
               Column("updated_at", DateTime), Column("symbol", String(30), index=True), Column("side", String(4)),
               Column("order_type", String(10)), Column("broker", String(20)), Column("mode", String(10)),
               Column("status", String(20), index=True), Column("requested_qty", Float),
               Column("requested_notional", Float), Column("filled_qty", Float), Column("avg_price", Float),
               Column("fee", Float), Column("reference_price", Float), Column("slippage_bps", Float),
               Column("exchange_order_id", String(60)), Column("error", Text), Column("history", JSON))

fills = Table("fills", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
              Column("client_order_id", String(40), index=True), Column("ts", DateTime),
              Column("symbol", String(30)), Column("side", String(4)), Column("qty", Float), Column("price", Float),
              Column("fee", Float))

positions = Table("positions", metadata, Column("symbol", String(30), primary_key=True), Column("qty", Float),
                  Column("entry", Float), Column("stop", Float), Column("target", Float), Column("opened_at", String(40)),
                  Column("proposal_id", String(40)), Column("updated_at", DateTime))

portfolio_snapshots = Table("portfolio_snapshots", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                            Column("ts", DateTime, index=True), Column("equity", Float), Column("cash", Float),
                            Column("exposure", Float), Column("drawdown", Float), Column("mode", String(10)),
                            Column("data", JSON))

news_events = Table("news_events", metadata, Column("id", String(40), primary_key=True),
                    Column("ts", DateTime, index=True), Column("source", String(40)), Column("title", Text),
                    Column("url", Text), Column("assets", JSON), Column("event_type", String(30), index=True),
                    Column("sentiment", Float), Column("novelty", Float), Column("credibility", Float),
                    Column("impact", Float), Column("data", JSON), Column("ret_1h", Float), Column("ret_4h", Float),
                    Column("ret_24h", Float))

sentiment_events = Table("sentiment_events", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                         Column("ts", DateTime, index=True), Column("asset", String(20)), Column("community", String(40)),
                         Column("sentiment", Float), Column("classification", String(30)), Column("posts", Integer),
                         Column("data", JSON))

whale_events = Table("whale_events", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                     Column("ts", DateTime, index=True), Column("asset", String(20)), Column("amount_usd", Float),
                     Column("from_type", String(20)), Column("to_type", String(20)), Column("kind", String(30)),
                     Column("bias", Float), Column("tx_hash", String(120)))

strategy_performance = Table("strategy_performance", metadata, Column("strategy", String(40), primary_key=True),
                             Column("updated_at", DateTime), Column("samples", Integer), Column("hit_rate", Float),
                             Column("recent_hit_rate", Float), Column("signal_edge", Float), Column("weight", Float),
                             Column("trades", Integer), Column("win_rate", Float), Column("avg_win", Float),
                             Column("avg_loss", Float), Column("expectancy", Float), Column("profit_factor", Float),
                             Column("sharpe", Float), Column("sortino", Float), Column("max_drawdown", Float),
                             Column("avg_holding_bars", Float), Column("avg_slippage_bps", Float),
                             Column("by_regime", JSON), Column("by_asset", JSON))

trade_outcomes = Table("trade_outcomes", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                       Column("proposal_id", String(40), index=True), Column("symbol", String(30), index=True),
                       Column("opened_at", String(40)), Column("closed_at", String(40)), Column("entry", Float),
                       Column("exit", Float), Column("qty", Float), Column("pnl", Float), Column("return_pct", Float),
                       Column("r_multiple", Float), Column("reason", String(30)), Column("bars_held", Integer),
                       Column("conviction", Float), Column("regime", String(30)), Column("supporting", JSON),
                       Column("review", JSON), Column("mode", String(10)))

system_alerts = Table("system_alerts", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                      Column("ts", DateTime, index=True), Column("level", String(10)), Column("source", String(40)),
                      Column("message", Text), Column("data", JSON))

agent_health = Table("agent_health", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                     Column("ts", DateTime, index=True), Column("worker_id", String(80), index=True),
                     Column("status", String(12)), Column("runs", Integer), Column("errors", Integer),
                     Column("avg_latency_ms", Float), Column("reason", Text))

model_versions = Table("model_versions", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                       Column("ts", DateTime), Column("component", String(40)), Column("version", String(40)),
                       Column("params", JSON), Column("notes", Text))

backtests = Table("backtests", metadata, _id(), Column("ts", DateTime), Column("name", String(80)),
                  Column("config", JSON), Column("metrics", JSON), Column("equity_curve", JSON))

knowledge = Table("knowledge", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                  Column("ts", DateTime), Column("kind", String(30), index=True), Column("title", String(200)),
                  Column("body", Text), Column("tags", JSON))

Index("ix_signals_eval", signals.c.evaluated_at, signals.c.ts)

TABLES = sorted(metadata.tables)


def postgres_ddl() -> str:
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable

    dialect = postgresql.dialect()
    parts = ["-- LunaTrade initial schema (PostgreSQL). Generated by `python -m lunatrade db sql`.\n"]
    for t in metadata.sorted_tables:
        parts.append(str(CreateTable(t).compile(dialect=dialect)).strip() + ";\n")
        for ix in sorted(t.indexes, key=lambda i: i.name):
            parts.append(str(CreateIndex(ix).compile(dialect=dialect)).strip() + ";\n")
    return "\n".join(parts)

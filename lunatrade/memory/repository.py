"""Repository: every write/read of the trading journal goes through here (one place to audit)."""
from __future__ import annotations

import datetime as dt
import logging
import threading

from sqlalchemy import create_engine, delete, desc, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from lunatrade.core.types import _jsonable
from lunatrade.memory import schema as T

log = logging.getLogger("lunatrade.db")


def _now():
    return dt.datetime.utcnow()


class Repository:
    def __init__(self, url: str):
        kw = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, future=True, **kw)
        self.url = url
        self._lock = threading.RLock()
        T.metadata.create_all(self.engine)

    def _upsert(self, table, row: dict, keys: list[str]):
        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            stmt = pg_insert(table).values(**row)
            stmt = stmt.on_conflict_do_update(index_elements=keys, set_={k: v for k, v in row.items() if k not in keys})
        elif dialect == "sqlite":
            stmt = sqlite_insert(table).values(**row)
            stmt = stmt.on_conflict_do_update(index_elements=keys, set_={k: v for k, v in row.items() if k not in keys})
        else:
            with self.engine.begin() as c:
                c.execute(delete(table).where(*[table.c[k] == row[k] for k in keys]))
                c.execute(insert(table).values(**row))
            return
        with self.engine.begin() as c:
            c.execute(stmt)

    def _insert(self, table, rows):
        rows = rows if isinstance(rows, list) else [rows]
        if not rows:
            return
        with self._lock, self.engine.begin() as c:
            c.execute(insert(table), rows)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(select(1))
            return True
        except Exception as e:
            log.error("database ping failed: %s", e)
            return False

    # ------------------------------------------------------------------ writes
    def save_agents(self, workers) -> None:
        for w in workers:
            self._upsert(T.agents, {"id": w.id, "name": w.name, "family": w.family.value, "phase": w.phase,
                                    "kind": "worker", "params": _jsonable(w.params), "status": w.health.status,
                                    "updated_at": _now()}, ["id"])

    def event(self, e) -> None:
        self._insert(T.agent_events, {"id": e.id, "ts": e.ts, "topic": e.topic, "source": e.source,
                                      "correlation_id": e.correlation_id, "payload": _jsonable(e.payload)})

    def candles(self, symbol: str, interval: str, df) -> None:
        rows = [{"symbol": symbol, "interval": interval, "open_time": int(r.open_time), "open": r.open, "high": r.high,
                 "low": r.low, "close": r.close, "volume": r.volume, "quote_volume": getattr(r, "quote_volume", 0.0),
                 "trades": getattr(r, "trades", 0.0), "taker_buy_base": getattr(r, "taker_buy_base", 0.0)}
                for r in df.itertuples()]
        for row in rows:
            self._upsert(T.market_data, row, ["symbol", "interval", "open_time"])

    def signals(self, cycle_id: str, sigs: list, prices: dict) -> None:
        self._insert(T.signals, [{"ts": s.ts, "cycle_id": cycle_id, "worker_id": s.worker_id, "family": s.family.value,
                                  "symbol": s.symbol, "direction": s.direction.value, "confidence": s.confidence,
                                  "setup": s.setup, "expected_move": s.expected_move, "horizon": s.time_horizon,
                                  "evidence": _jsonable(s.evidence[:5]), "context": _jsonable(s.context),
                                  "price": prices.get(s.symbol)} for s in sigs])

    def snapshot(self, ts, symbol, price, spread, regime, assessment) -> None:
        self._insert(T.market_snapshots, {"ts": ts, "symbol": symbol, "price": price, "spread_bps": spread,
                                          "regime": regime, "assessment": _jsonable(assessment)})

    def proposal(self, p, status: str) -> None:
        self._upsert(T.trade_proposals, {"id": p.id, "ts": p.ts, "symbol": p.symbol, "direction": p.direction.value,
                                         "action": p.action, "conviction": p.conviction,
                                         "win_probability": p.win_probability, "entry": p.entry_price,
                                         "stop": p.stop_price, "target": p.target_price,
                                         "requested_notional": p.requested_notional, "regime": p.regime,
                                         "rationale": p.rationale, "status": status, "data": p.to_dict()}, ["id"])

    def proposal_status(self, pid: str, status: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(T.trade_proposals).where(T.trade_proposals.c.id == pid).values(status=status))

    def risk_decision(self, d, da=None) -> None:
        self._insert(T.risk_decisions, {"proposal_id": d.proposal_id, "ts": d.ts, "approved": d.approved,
                                        "approved_notional": d.approved_notional,
                                        "requested_notional": d.requested_notional, "checks": d.checks,
                                        "reasons": d.reasons, "devils_advocate": da.to_dict() if da else None})

    def approval(self, row: dict) -> None:
        self._upsert(T.approvals, row, ["proposal_id"])

    def order(self, o: dict) -> None:
        row = {k: o.get(k) for k in ("client_order_id", "proposal_id", "symbol", "side", "order_type", "broker", "mode",
                                     "status", "requested_qty", "requested_notional", "filled_qty", "avg_price", "fee",
                                     "reference_price", "slippage_bps", "exchange_order_id", "error", "history")}
        row["created_at"] = dt.datetime.fromisoformat(o["created_at"])
        row["updated_at"] = dt.datetime.fromisoformat(o["updated_at"])
        self._upsert(T.orders, row, ["client_order_id"])
        if o.get("status") == "POSITION_UPDATED" and o.get("filled_qty"):
            self._insert(T.fills, {"client_order_id": o["client_order_id"], "ts": row["updated_at"], "symbol": o["symbol"],
                                   "side": o["side"], "qty": o["filled_qty"], "price": o["avg_price"], "fee": o["fee"]})

    def positions(self, snapshot: dict) -> None:
        with self.engine.begin() as c:
            c.execute(delete(T.positions))
            for s, p in snapshot.get("positions", {}).items():
                c.execute(insert(T.positions).values(symbol=s, qty=p["qty"], entry=p["entry"], stop=p["stop"],
                                                     target=p["target"], opened_at=p.get("opened_at"),
                                                     proposal_id=p.get("proposal_id"), updated_at=_now()))

    def portfolio_snapshot(self, ts, snap: dict, mode: str) -> None:
        self._insert(T.portfolio_snapshots, {"ts": ts, "equity": snap["equity"], "cash": snap["cash"],
                                             "exposure": snap["exposure"], "drawdown": snap.get("drawdown", 0.0),
                                             "mode": mode, "data": _jsonable(snap)})

    def news(self, item) -> None:
        self._upsert(T.news_events, {"id": item.id, "ts": item.ts, "source": item.source, "title": item.title,
                                     "url": item.url, "assets": item.assets, "event_type": item.event_type,
                                     "sentiment": item.sentiment, "novelty": item.novelty,
                                     "credibility": item.credibility, "impact": item.impact,
                                     "data": _jsonable(item.to_dict())}, ["id"])

    def news_outcome(self, item_id: str, **rets) -> None:
        with self.engine.begin() as c:
            c.execute(update(T.news_events).where(T.news_events.c.id == item_id).values(**rets))

    def sentiment(self, ts, reading, community="*") -> None:
        self._insert(T.sentiment_events, {"ts": ts, "asset": reading.asset, "community": community,
                                          "sentiment": reading.sentiment, "classification": reading.classification,
                                          "posts": reading.posts, "data": _jsonable(reading.__dict__)})

    def whale(self, t, interp) -> None:
        self._insert(T.whale_events, {"ts": t.ts, "asset": t.asset, "amount_usd": t.amount_usd, "from_type": t.from_type,
                                      "to_type": t.to_type, "kind": interp.kind, "bias": interp.bias,
                                      "tx_hash": t.tx_hash})

    def strategy(self, name: str, stats: dict) -> None:
        cols = {c.name for c in T.strategy_performance.columns}
        row = _jsonable({k: v for k, v in stats.items() if k in cols})
        row = {k: (None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v)
               for k, v in row.items()}
        row.update(strategy=name, updated_at=_now())
        self._upsert(T.strategy_performance, row, ["strategy"])

    def trade_outcome(self, o: dict, mode: str) -> None:
        cols = {c.name for c in T.trade_outcomes.columns} - {"id"}
        row = {k: o.get(k) for k in cols if k in o}
        row["mode"] = mode
        self._insert(T.trade_outcomes, _jsonable(row))

    def alert(self, level: str, source: str, message: str, data: dict | None = None) -> None:
        self._insert(T.system_alerts, {"ts": _now(), "level": level, "source": source, "message": message[:2000],
                                       "data": _jsonable(data or {})})

    def health(self, workers) -> None:
        now = _now()
        self._insert(T.agent_health, [{"ts": now, "worker_id": w.id, "status": w.health.status, "runs": w.health.runs,
                                       "errors": w.health.errors, "avg_latency_ms": w.health.avg_latency_ms,
                                       "reason": w.health.reason or w.health.last_error} for w in workers])

    def model_version(self, component: str, version: str, params: dict, notes: str = "") -> None:
        self._insert(T.model_versions, {"ts": _now(), "component": component, "version": version,
                                        "params": _jsonable(params), "notes": notes})

    def backtest(self, bid: str, name: str, config: dict, metrics: dict, curve: list) -> None:
        self._upsert(T.backtests, {"id": bid, "ts": _now(), "name": name, "config": _jsonable(config),
                                   "metrics": _jsonable(metrics), "equity_curve": _jsonable(curve[-2000:])}, ["id"])

    def knowledge(self, kind: str, title: str, body: str, tags: list | None = None) -> None:
        self._insert(T.knowledge, {"ts": _now(), "kind": kind, "title": title[:200], "body": body, "tags": tags or []})

    # ------------------------------------------------------------------ reads
    def _rows(self, stmt) -> list[dict]:
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(stmt)]

    def recent(self, table: str, limit: int = 50, order_col: str = "ts") -> list[dict]:
        t = T.metadata.tables[table]
        col = t.c[order_col] if order_col in t.c else list(t.primary_key.columns)[0]
        return self._rows(select(t).order_by(desc(col)).limit(limit))

    def proposal_by_id(self, pid: str) -> dict | None:
        rows = self._rows(select(T.trade_proposals).where(T.trade_proposals.c.id == pid))
        return rows[0] if rows else None

    def explain(self, pid: str) -> dict | None:
        p = self.proposal_by_id(pid)
        if not p:
            return None
        return {"proposal": p,
                "risk": self._rows(select(T.risk_decisions).where(T.risk_decisions.c.proposal_id == pid)),
                "orders": self._rows(select(T.orders).where(T.orders.c.proposal_id == pid)),
                "outcome": self._rows(select(T.trade_outcomes).where(T.trade_outcomes.c.proposal_id == pid))}

    def outcomes(self, limit: int = 1000) -> list[dict]:
        return self._rows(select(T.trade_outcomes).order_by(desc(T.trade_outcomes.c.id)).limit(limit))[::-1]

    def strategies(self) -> list[dict]:
        return self._rows(select(T.strategy_performance))

    def counts(self) -> dict:
        out = {}
        with self.engine.connect() as c:
            for name, t in T.metadata.tables.items():
                out[name] = c.execute(select(func.count()).select_from(t)).scalar()
        return out

    def search_knowledge(self, q: str, limit: int = 20) -> list[dict]:
        like = f"%{q}%"
        return self._rows(select(T.knowledge).where(T.knowledge.c.title.ilike(like) | T.knowledge.c.body.ilike(like))
                          .order_by(desc(T.knowledge.c.ts)).limit(limit))

"""
24/7 supervisor. Watches feeds, WebSockets, database, workers, brokers, LLMs, latency, CPU/memory,
order failures, risk-engine heartbeat, PnL/drawdown and data freshness.

    market feed dead -> STOP NEW TRADES -> attempt reconnect -> validate data -> resume after N healthy checks
"""
from __future__ import annotations

import logging
import os
import time
from collections import Counter

log = logging.getLogger("lunatrade.supervisor")


def system_load() -> dict:
    try:
        import psutil

        return {"cpu_pct": psutil.cpu_percent(interval=None), "memory_pct": psutil.virtual_memory().percent}
    except Exception:
        pass
    out = {}
    try:
        load1 = os.getloadavg()[0]
        out["cpu_pct"] = min(100.0, load1 / (os.cpu_count() or 1) * 100)
    except Exception:
        out["cpu_pct"] = None
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        total = float(info["MemTotal"].split()[0])
        avail = float(info["MemAvailable"].split()[0])
        out["memory_pct"] = (1 - avail / total) * 100
    except Exception:
        out["memory_pct"] = None
    return out


class Supervisor:
    def __init__(self, engine, cfg: dict, reconnect=None):
        self.engine = engine
        self.cfg = cfg
        self.reconnect = reconnect          # callable(component) -> None
        self.healthy_streak = 0
        self.last: dict = {}
        self.checks_run = 0

    def check(self, now_ms: int | None = None) -> dict:
        e = self.engine
        now_ms = now_ms or int(time.time() * 1000)
        problems, warnings = [], []
        # data freshness (a closed candle is expected every interval)
        stale = {s: e.store.staleness_seconds(s, now_ms) for s in e.symbols}
        limit = float(e.cfg.get("kill_switch.max_data_staleness_seconds", 300)) + e.store.interval_ms / 1000
        bad = {s: v for s, v in stale.items() if v > limit}
        if bad:
            problems.append(f"stale data: {', '.join(list(bad)[:5])}")
            e.kill_switch.halt("STALE_DATA", ", ".join(f"{s} {v:.0f}s" for s, v in list(bad.items())[:5]))
            if self.reconnect:
                self.reconnect("market_data")
        # brokers
        for name, b in e.gateway.brokers.items():
            if not b.healthy():
                problems.append(f"broker {name} unhealthy")
                e.kill_switch.halt("EXCHANGE_UNHEALTHY", name)
                if self.reconnect:
                    self.reconnect(f"broker:{name}")
        # database
        db_ok = e.repo.ping() if e.repo else True
        if not db_ok:
            problems.append("database unreachable")
            e.kill_switch.halt("DATABASE", "unreachable")
        # risk engine heartbeat (it must have evaluated recently if the engine is cycling)
        hb_age = time.time() - e.risk.heartbeat if e.risk.heartbeat else None
        if e.last_cycle_at and hb_age is not None and hb_age > max(3 * e.store.interval_ms / 1000, 900) \
                and e.proposals_seen:
            warnings.append(f"risk engine idle {hb_age:.0f}s")
        # workers
        status = Counter(w.health.status for w in e.workers)
        if status.get("DEGRADED", 0) + sum(1 for w in e.workers if w.health.consecutive_errors >= 3) > 30:
            warnings.append(f"{status.get('DEGRADED', 0)} degraded workers")
        # order failures
        if e.gateway.consecutive_failures:
            warnings.append(f"{e.gateway.consecutive_failures} consecutive order failures")
        load = system_load()
        sc = self.cfg
        if load.get("cpu_pct") and load["cpu_pct"] > sc.get("max_cpu_pct", 95):
            warnings.append(f"cpu {load['cpu_pct']:.0f}%")
        if load.get("memory_pct") and load["memory_pct"] > sc.get("max_memory_pct", 92):
            problems.append(f"memory {load['memory_pct']:.0f}%")
        # cycle latency
        if e.last_cycle_ms and e.last_cycle_ms > e.store.interval_ms * 0.5:
            warnings.append(f"cycle took {e.last_cycle_ms:.0f}ms")

        if problems:
            self.healthy_streak = 0
        else:
            self.healthy_streak += 1
            if self.healthy_streak >= int(sc.get("healthy_checks_to_resume", 3)):
                for src in ("STALE_DATA", "EXCHANGE_UNHEALTHY", "DATABASE"):
                    e.kill_switch.clear_halt(src)
        self.checks_run += 1
        self.last = {
            "healthy": not problems, "problems": problems, "warnings": warnings, "healthy_streak": self.healthy_streak,
            "data_staleness_s": {s: (round(v, 1) if v != float("inf") else None) for s, v in stale.items()},
            "websocket": e.store.ws_connected, "database": db_ok, "workers": dict(status),
            "brokers": {n: b.healthy() for n, b in e.gateway.brokers.items()},
            "feeds": e.feeds.status, "llm": e.council.status() if e.council else {},
            "system": load, "cycle_ms": e.last_cycle_ms, "kill_switch": e.kill_switch.status(),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if problems:
            log.warning("supervisor problems: %s", problems)
        return self.last

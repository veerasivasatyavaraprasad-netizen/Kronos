"""
Agent communication bus.

Agents never call each other directly: they publish events and subscribe to topics. The in-process
bus is synchronous and deterministic (identical in backtests and live). Mirrors copy every event to
Redis Streams or Kafka so other processes (dashboard, analytics, extra workers) can consume them.
"""
from __future__ import annotations

import fnmatch
import logging
import threading
from collections import defaultdict, deque
from typing import Callable

from lunatrade.core.events import Event

log = logging.getLogger("lunatrade.bus")
Handler = Callable[[Event], None]


class EventBus:
    def __init__(self, history: int = 2000):
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._mirrors: list = []
        self._lock = threading.RLock()
        self.history: deque[Event] = deque(maxlen=history)
        self.counts: dict[str, int] = defaultdict(int)
        self.handler_errors = 0

    def subscribe(self, pattern: str, handler: Handler) -> None:
        """pattern supports wildcards: 'order.*', '*'."""
        with self._lock:
            self._subs[pattern].append(handler)

    def add_mirror(self, mirror) -> None:
        self._mirrors.append(mirror)

    def publish(self, event: Event) -> Event:
        with self._lock:
            self.history.append(event)
            self.counts[event.topic] += 1
            handlers = [h for p, hs in self._subs.items() if fnmatch.fnmatchcase(event.topic, p) for h in hs]
        for h in handlers:
            try:
                h(event)
            except Exception:  # one broken subscriber must not break the pipeline
                self.handler_errors += 1
                log.exception("bus handler failed for %s", event.topic)
        for m in self._mirrors:
            try:
                m.send(event)
            except Exception as e:
                log.warning("bus mirror %s failed: %s", type(m).__name__, e)
        return event

    def emit(self, topic: str, payload: dict, source: str = "system", correlation_id: str = "") -> Event:
        return self.publish(Event(topic=topic, payload=payload, source=source, correlation_id=correlation_id))

    def recent(self, pattern: str = "*", limit: int = 100) -> list[Event]:
        with self._lock:
            items = [e for e in self.history if fnmatch.fnmatchcase(e.topic, pattern)]
        return items[-limit:]


class RedisStreamMirror:
    """XADD every event to `lunatrade:<topic>` (capped). Requires `pip install redis`."""

    def __init__(self, url: str, maxlen: int = 100_000):
        import redis  # optional dependency

        self.r = redis.Redis.from_url(url)
        self.maxlen = maxlen

    def send(self, event: Event) -> None:
        self.r.xadd(f"lunatrade:{event.topic}", {"event": event.to_json()}, maxlen=self.maxlen, approximate=True)


class KafkaMirror:
    """Produce every event to Kafka topic `lunatrade.<topic>`. Requires `pip install kafka-python`."""

    def __init__(self, bootstrap: str):
        from kafka import KafkaProducer  # optional dependency

        self.p = KafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda v: v.encode())

    def send(self, event: Event) -> None:
        self.p.send(f"lunatrade.{event.topic}", event.to_json())


def build_mirrors(redis_url: str | None, kafka_bootstrap: str | None) -> list:
    mirrors = []
    if redis_url:
        try:
            mirrors.append(RedisStreamMirror(redis_url))
        except Exception as e:
            log.warning("Redis mirror disabled: %s", e)
    if kafka_bootstrap:
        try:
            mirrors.append(KafkaMirror(kafka_bootstrap))
        except Exception as e:
            log.warning("Kafka mirror disabled: %s", e)
    return mirrors

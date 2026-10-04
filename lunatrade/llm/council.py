"""
LLM council: several models answer the same structured question and their answers are combined.

Numbers are combined with the median, booleans by majority, lists by union. A budget guard caps calls
per hour. If no provider is configured (or all fail) the deterministic rule provider answers.
"""
from __future__ import annotations

import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from lunatrade.llm.providers import Provider, RuleProvider, build_providers


class LLMCouncil:
    def __init__(self, providers: dict[str, Provider], roles: dict[str, list], council_size: int = 3,
                 max_calls_per_hour: int = 120):
        self.providers = providers
        self.providers.setdefault("rules", RuleProvider())
        self.roles = roles or {}
        self.council_size = council_size
        self.max_calls = max_calls_per_hour
        self._calls: deque = deque()
        self._lock = threading.Lock()
        self.decisions = deque(maxlen=200)

    @classmethod
    def from_config(cls, cfg) -> "LLMCouncil":
        llm = cfg.section("llm")
        return cls(build_providers(llm.get("providers", []), float(llm.get("timeout_seconds", 45))),
                   llm.get("roles", {}), int(llm.get("council_size", 3)), int(llm.get("max_calls_per_hour", 120)))

    def members(self, role: str) -> list[Provider]:
        names = self.roles.get(role) or [n for n in self.providers if n != "rules"]
        live = [self.providers[n] for n in names if n in self.providers and self.providers[n].available]
        return live[: self.council_size]

    def _budget_ok(self, n: int) -> bool:
        with self._lock:
            now = time.time()
            while self._calls and now - self._calls[0] > 3600:
                self._calls.popleft()
            if len(self._calls) + n > self.max_calls:
                return False
            self._calls.extend([now] * n)
            return True

    def ask_all(self, role: str, task: str, payload: dict) -> list[dict]:
        members = self.members(role)
        if not members or not self._budget_ok(len(members)):
            ans = self.providers["rules"].complete_json(task, payload)
            return [ans] if ans else []
        with ThreadPoolExecutor(max_workers=len(members)) as ex:
            answers = [a for a in ex.map(lambda p: p.complete_json(task, payload), members) if a]
        if not answers:
            ans = self.providers["rules"].complete_json(task, payload)
            answers = [ans] if ans else []
        return answers

    def ask_json(self, role: str, task: str, payload: dict) -> dict | None:
        answers = self.ask_all(role, task, payload)
        if not answers:
            return None
        combined = combine(answers)
        combined["providers"] = [a.get("provider") for a in answers]
        self.decisions.append({"ts": time.time(), "role": role, "task": task, "answer": combined})
        return combined

    def status(self) -> dict:
        return {"providers": [p.status() for p in self.providers.values()],
                "roles": {r: [p.name for p in self.members(r)] or ["rules"] for r in self.roles},
                "calls_last_hour": len(self._calls), "budget_per_hour": self.max_calls}


def combine(answers: list[dict]) -> dict:
    if len(answers) == 1:
        return dict(answers[0])
    keys = {k for a in answers for k in a if k != "provider"}
    out = {}
    for k in keys:
        vals = [a[k] for a in answers if k in a]
        if all(isinstance(v, bool) for v in vals):
            out[k] = sum(vals) > len(vals) / 2
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = statistics.median(vals)
        elif all(isinstance(v, list) for v in vals):
            seen, merged = set(), []
            for v in vals:
                for x in v:
                    if str(x) not in seen:
                        seen.add(str(x))
                        merged.append(x)
            out[k] = merged
        else:
            # strings / mixed: most common, falling back to the first answer's value
            strs = [str(v) for v in vals]
            out[k] = max(set(strs), key=strs.count) if len(set(strs)) < len(strs) else vals[0]
    return out

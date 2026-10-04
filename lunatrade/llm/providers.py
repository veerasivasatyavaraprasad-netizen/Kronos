"""
LLM providers. Several models can serve the same role; the council combines their answers.

    anthropic          Claude via the official Anthropic SDK (ANTHROPIC_API_KEY)
    openai_compatible  OpenAI, DeepSeek, Groq, Together, OpenRouter, ... (chat/completions)
    gemini             Google Gemini REST (GEMINI_API_KEY)
    ollama             local models (OLLAMA_HOST, e.g. http://localhost:11434)
    rules              deterministic fallback so the system works with no LLM at all
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

from lunatrade.config import secret
from lunatrade.llm.prompts import SYSTEM_BASE, build_user_prompt

log = logging.getLogger("lunatrade.llm")


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from model text (tolerates code fences / preambles)."""
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
        start = text.find("{", start + 1) if not candidates else -1
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class Provider:
    kind = "base"

    def __init__(self, name: str, model: str, env_key: str | None = None, timeout: float = 45, **opts):
        self.name, self.model, self.env_key, self.timeout, self.opts = name, model, env_key, timeout, opts
        self.calls = 0
        self.failures = 0
        self.last_error = ""
        self.last_latency_ms = 0.0

    @property
    def available(self) -> bool:
        return bool(secret(self.env_key)) if self.env_key else True

    def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def complete_json(self, task: str, payload: dict) -> dict | None:
        t0 = time.perf_counter()
        self.calls += 1
        try:
            text = self._complete(SYSTEM_BASE, build_user_prompt(task, payload))
            out = parse_json(text)
            if out is None:
                raise ValueError(f"no JSON in reply: {text[:120]!r}")
            out["provider"] = self.name
            return out
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            log.warning("LLM %s failed on %s: %s", self.name, task, self.last_error)
            return None
        finally:
            self.last_latency_ms = (time.perf_counter() - t0) * 1000

    def status(self) -> dict:
        return {"name": self.name, "kind": self.kind, "model": self.model, "available": self.available,
                "calls": self.calls, "failures": self.failures, "last_error": self.last_error,
                "last_latency_ms": round(self.last_latency_ms, 1)}


class AnthropicProvider(Provider):
    kind = "anthropic"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._client = None

    def _client_(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=secret(self.env_key or "ANTHROPIC_API_KEY"),
                                               timeout=self.timeout, max_retries=2)
        return self._client

    def _complete(self, system: str, user: str) -> str:
        client = self._client_()
        kwargs = dict(model=self.model, max_tokens=4000, system=system,
                      messages=[{"role": "user", "content": user}],
                      output_config={"effort": self.opts.get("effort", "medium")})
        # Server-side refusal fallback (routes a declined request to another model inside the same call).
        try:
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except TypeError:   # older SDK without the typed parameter
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                               extra_body={"fallbacks": "default"}, **kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            raise RuntimeError("model declined the request")
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAICompatibleProvider(Provider):
    kind = "openai_compatible"

    def _complete(self, system: str, user: str) -> str:
        base = self.opts.get("base_url", "https://api.openai.com/v1").rstrip("/")
        r = requests.post(f"{base}/chat/completions", timeout=self.timeout,
                          headers={"Authorization": f"Bearer {secret(self.env_key)}"},
                          json={"model": self.model, "temperature": 0.2,
                                "response_format": {"type": "json_object"},
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class GeminiProvider(Provider):
    kind = "gemini"

    def _complete(self, system: str, user: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        r = requests.post(url, timeout=self.timeout, headers={"x-goog-api-key": secret(self.env_key) or ""},
                          json={"systemInstruction": {"parts": [{"text": system}]},
                                "contents": [{"role": "user", "parts": [{"text": user}]}],
                                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}})
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


class OllamaProvider(Provider):
    kind = "ollama"

    @property
    def available(self) -> bool:
        return bool(os.getenv(self.env_key or "OLLAMA_HOST"))

    def _complete(self, system: str, user: str) -> str:
        host = os.getenv(self.env_key or "OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        r = requests.post(f"{host}/api/chat", timeout=self.timeout,
                          json={"model": self.model, "stream": False, "format": "json",
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"]


class RuleProvider(Provider):
    """Deterministic answers - keeps every role functional without any API key."""
    kind = "rules"

    def __init__(self):
        super().__init__("rules", "deterministic")

    def complete_json(self, task: str, payload: dict) -> dict | None:
        self.calls += 1
        if task == "lead_review":
            p = payload.get("proposal", {})
            concerns, adj = [], 0
            if p.get("regime") in ("UNKNOWN", "PANIC", "ILLIQUID"):
                concerns.append(f"regime {p.get('regime')}")
                adj -= 5
            if len(p.get("opposing_signals", [])) > len(p.get("supporting_signals", [])) * 0.6:
                concerns.append("many opposing signals")
                adj -= 4
            return {"adjustment": adj, "veto": False, "concerns": concerns, "reason": "rule-based review",
                    "provider": self.name}
        if task == "devils_advocate":
            risks = payload.get("hidden_risks", [])
            return {"extra_bear_points": min(2.0, 0.25 * len(risks)), "hidden_risks": [], "veto": False,
                    "reason": "rule-based", "provider": self.name}
        if task == "classify_news":
            return {"event_type": payload.get("rule_event_type"), "sentiment": payload.get("rule_sentiment", 0.0),
                    "reason": "rule-based", "provider": self.name}
        if task == "voice_answer":
            return {"answer": payload.get("fallback_answer", "I don't have that information."), "provider": self.name}
        if task == "post_trade":
            o = payload.get("outcome", {})
            win = (o.get("pnl") or 0) > 0
            return {"summary": f"{'Win' if win else 'Loss'} of {o.get('pnl', 0):.2f} on {o.get('symbol')}.",
                    "lesson": "Keep following the process." if win else "Review the signals that supported this entry.",
                    "tags": ["win" if win else "loss"], "provider": self.name}
        return None


KINDS = {"anthropic": AnthropicProvider, "openai_compatible": OpenAICompatibleProvider, "gemini": GeminiProvider,
         "ollama": OllamaProvider}


def build_providers(cfg_list: list[dict], timeout: float = 45) -> dict[str, Provider]:
    out = {}
    for p in cfg_list or []:
        cls = KINDS.get(p.get("kind"))
        if not cls:
            continue
        opts = {k: v for k, v in p.items() if k not in ("name", "kind", "model", "env_key")}
        out[p["name"]] = cls(p["name"], p["model"], p.get("env_key"), timeout=timeout, **opts)
    out["rules"] = RuleProvider()
    return out

"""
Configuration: default.yaml <- lunatrade.yaml (repo root or LUNATRADE_CONFIG) <- environment overrides.

Secrets are read from the environment only (python-dotenv style .env is loaded by automation.env).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from automation.env import load_env

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = Path(__file__).with_name("default.yaml")

# Environment variable -> dotted config key
ENV_OVERRIDES = {
    "LUNATRADE_MODE": "mode",
    "LUNATRADE_INTERVAL": "interval",
    "LUNATRADE_DATABASE_URL": "storage.database_url",
    "LUNATRADE_REDIS_URL": "bus.redis_url",
    "LUNATRADE_KAFKA_BOOTSTRAP": "bus.kafka_bootstrap",
    "LUNATRADE_API_PORT": "api.port",
    "LUNATRADE_BINANCE_ENV": "brokers.binance.environment",
    "LUNATRADE_ALPACA_ENV": "brokers.alpaca.environment",
    "VOICE_PROVIDER": "voice.provider",
    "VOICE_TTS_PROVIDER": "voice.tts_provider",
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(old: Any, value: str) -> Any:
    if isinstance(old, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(value)
    if isinstance(old, float):
        return float(value)
    return value


class Config:
    def __init__(self, data: dict):
        self.data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self.data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    def section(self, name: str) -> dict:
        return self.data.get(name) or {}

    @property
    def output_dir(self) -> Path:
        p = Path(self.get("output_dir", "outputs/lunatrade"))
        return p if p.is_absolute() else ROOT / p

    @property
    def database_url(self) -> str:
        url = self.get("storage.database_url")
        if url:
            return url
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.output_dir / 'lunatrade.db'}"


def load_config(path: str | Path | None = None, overrides: dict | None = None, use_env: bool = True) -> Config:
    if use_env:
        load_env()
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    user = Path(path) if path else Path(os.getenv("LUNATRADE_CONFIG", ROOT / "lunatrade.yaml"))
    if user.is_file():
        data = deep_merge(data, yaml.safe_load(user.read_text(encoding="utf-8")) or {})
    cfg = Config(data)
    if use_env:
        for env, key in ENV_OVERRIDES.items():
            if os.getenv(env):
                cfg.set(key, _coerce(cfg.get(key), os.environ[env]))
    if overrides:
        cfg.data = deep_merge(cfg.data, overrides)
    cfg.set("mode", str(cfg.get("mode", "PAPER")).upper())
    return cfg


def secret(name: str) -> str | None:
    """Read a secret from the environment, dropping whitespace/quotes picked up when pasting keys."""
    value = os.getenv(name)
    if not value:
        return None
    return "".join(c for c in value if ord(c) > 32 and ord(c) != 127).strip('"').strip("'") or None


def live_trading_allowed() -> bool:
    """Real-money switch. Either name works so existing Kronos .env files stay valid."""
    return any(os.getenv(k, "").strip().lower() == "yes" for k in ("LUNATRADE_ALLOW_LIVE", "KRONOS_ALLOW_LIVE_TRADING"))

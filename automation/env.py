"""Minimal .env loader: KEY=VALUE lines; real environment variables always win."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path=None) -> None:
    path = Path(path) if path else ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)

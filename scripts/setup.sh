#!/usr/bin/env bash
# One-shot setup: venv + dependencies + model download + smoke test.
#   ./scripts/setup.sh            # auto (CPU torch if no NVIDIA GPU)
#   ./scripts/setup.sh --gpu      # force default (CUDA) torch wheels
#   ./scripts/setup.sh --no-test  # skip the smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
FORCE_GPU=0; RUN_TEST=1
for a in "$@"; do
  case $a in --gpu) FORCE_GPU=1 ;; --no-test) RUN_TEST=0 ;; esac
done

$PY -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

if [ ! -d .venv ]; then
  echo ">> Creating virtualenv .venv"
  $PY -m venv .venv
fi
. .venv/bin/activate
python -m pip install -q --upgrade pip

if ! python -c 'import torch' 2>/dev/null; then
  if [ $FORCE_GPU = 0 ] && ! command -v nvidia-smi >/dev/null 2>&1 && [ "$(uname)" = "Linux" ]; then
    echo ">> Installing CPU-only PyTorch"
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu || pip install -q torch
  else
    echo ">> Installing PyTorch"
    pip install -q torch
  fi
fi

echo ">> Installing app requirements"
pip install -q -r requirements-app.txt

echo ">> Downloading Kronos models from Hugging Face"
python -m automation.kronos_auto setup-models --model kronos-small
python -m automation.kronos_auto setup-models --model kronos-mini

if [ $RUN_TEST = 1 ]; then
  echo ">> Smoke test: forecasting bundled sample data"
  python -m automation.kronos_auto run --no-backtest
fi

cat <<'MSG'

Setup complete. Next steps:
  source .venv/bin/activate
  make serve       # Web UI at http://localhost:7070
  make run         # forecast every symbol in automation/config.yaml -> outputs/latest/
  make test        # regression tests
MSG

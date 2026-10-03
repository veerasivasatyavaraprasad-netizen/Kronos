#!/usr/bin/env bash
# End-to-end Qlib A-share pipeline: download data -> preprocess -> fine-tune tokenizer
# -> fine-tune predictor -> backtest.  Works on GPU (NCCL) or CPU (gloo).
#
#   ./finetune/run_pipeline.sh                 # full run, all GPUs found (or 1 CPU process)
#   NUM_GPUS=2 ./finetune/run_pipeline.sh
#   SKIP_DOWNLOAD=1 ./finetune/run_pipeline.sh # Qlib data already at $QLIB_DATA_PATH
#   STEPS="train_predictor backtest" ./finetune/run_pipeline.sh
#
# Settings live in finetune/config.py; the common ones can be overridden with env vars
# (QLIB_DATA_PATH, KRONOS_FT_EPOCHS, KRONOS_FT_BATCH_SIZE, KRONOS_FT_INSTRUMENT, ...).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

export QLIB_DATA_PATH=${QLIB_DATA_PATH:-$HOME/.qlib/qlib_data/cn_data}
if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$($PY -c 'import torch; print(max(torch.cuda.device_count(), 1))')
fi
STEPS=${STEPS:-"download preprocess train_tokenizer train_predictor backtest"}

$PY -c 'import qlib' 2>/dev/null || { echo "Installing pyqlib"; $PY -m pip install -q -r requirements-finetune.txt; }

for step in $STEPS; do
  echo "================ $step ================"
  case $step in
    download)
      if [ "${SKIP_DOWNLOAD:-0}" = 1 ] || [ -d "$QLIB_DATA_PATH/features" ]; then
        echo "Qlib data present at $QLIB_DATA_PATH (or SKIP_DOWNLOAD=1), skipping download"
      else
        $PY -m qlib.run.get_data qlib_data --target_dir "$QLIB_DATA_PATH" --region cn
      fi ;;
    preprocess)      $PY finetune/qlib_data_preprocess.py ;;
    train_tokenizer) $PY -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" finetune/train_tokenizer.py ;;
    train_predictor) $PY -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" finetune/train_predictor.py ;;
    backtest)        $PY finetune/qlib_test.py ;;
    *) echo "Unknown step: $step"; exit 1 ;;
  esac
done
echo "Done. Checkpoints: finetune/outputs/models, backtest results: finetune/outputs/backtest_results"

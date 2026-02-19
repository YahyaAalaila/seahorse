#!/usr/bin/env bash
set -euo pipefail

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"${PYTHON_BIN}" train.py \
  --preset deep_stpp \
  --data hawkes \
  --n_train 8 \
  --n_val 2 \
  --n_epochs 1 \
  --batch_size 4 \
  --no_save_metrics

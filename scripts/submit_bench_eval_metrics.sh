#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_bench_eval_metrics.sh <bench_root>

Submits one `evaluate metrics` job per cell listed in <bench_root>/cell_index.json.

Useful environment overrides:
  PARTITION=...
  GPUS=0|1
  CPUS_PER_TASK=8
  MEM=32G
  TIME_LIMIT=12:00:00
  SPLIT=test
  METRIC_PROFILE=core|nll|predictive|generative|surface|full
  ARTIFACT_MODE=load_or_compute|load_only
  DEVICE=cpu|cuda|auto
  EVAL_SEED=0
  K_PRED=32
  K_GEN=20
  EXACT_TIME_BINS=8
  EXACT_SPATIAL_BINS=8
  RUN_ROOT=...
  EXCLUDE=serv-3307
  FORCE_RUN=1

Examples:
  scripts/submit_bench_eval_metrics.sh runs/exp1/covid-stpp/bench/covid-stpp__gen__04231151
  METRIC_PROFILE=nll GPUS=0 DEVICE=cpu scripts/submit_bench_eval_metrics.sh runs/exp1/covid-stpp/bench/covid-stpp__fact__04231151
  METRIC_PROFILE=predictive GPUS=1 DEVICE=cuda TIME_LIMIT=24:00:00 scripts/submit_bench_eval_metrics.sh runs/exp1/covid-stpp/bench/covid-stpp__neural_cond_gmm__04231151
EOF
}

if [ "$#" -ne 1 ]; then
  usage
  exit 1
fi

BENCH_ROOT_INPUT="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$ROOT/scripts/pegasus_eval_metrics.sbatch"
TARGET_SCRIPT="$ROOT/scripts/resolve_bench_eval_targets.py"
PYTHON_BIN_LOCAL="${UNIFIED_STPP_PYTHON_LOCAL:-python}"

BENCH_ROOT="$(cd "$(dirname "$BENCH_ROOT_INPUT")" && pwd)/$(basename "$BENCH_ROOT_INPUT")"
BENCH_ROOT="$(python - <<'PY' "$BENCH_ROOT"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

GPUS="${GPUS:-0}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SPLIT="${SPLIT:-test}"
METRIC_PROFILE="${METRIC_PROFILE:-core}"
ARTIFACT_MODE="${ARTIFACT_MODE:-load_or_compute}"
DEVICE="${DEVICE:-}"
if [ -z "$DEVICE" ]; then
  if [ "$GPUS" = "0" ]; then
    DEVICE="cpu"
  else
    DEVICE="cuda"
  fi
fi

RUN_ROOT="${RUN_ROOT:-$ROOT/runs/exp1}"
REPO_ROOT="${REPO_ROOT:-$ROOT}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$REPO_ROOT}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$REPO_ROOT:$REPO_ROOT,/netscratch/aalaila:/netscratch/aalaila}"

declare -a SBATCH_ARGS
SBATCH_ARGS+=(--gpus="$GPUS")
SBATCH_ARGS+=(--cpus-per-task="$CPUS_PER_TASK")
SBATCH_ARGS+=(--mem="$MEM")
SBATCH_ARGS+=(--time="$TIME_LIMIT")
SBATCH_ARGS+=(--output="$ROOT/logs/%x_%j.out")
if [ -n "${PARTITION:-}" ]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi
if [ -n "${EXCLUDE:-}" ]; then
  SBATCH_ARGS+=(--exclude="$EXCLUDE")
fi

mkdir -p "$ROOT/logs" "$BENCH_ROOT/evaluate"
LEDGER="$BENCH_ROOT/evaluate/submissions.csv"
if [ ! -f "$LEDGER" ]; then
  printf 'submitted_at,job_id,job_name,bench_root,bench_id,dataset_id,preset,seed,split,metric_profile,partition,gpus,cpus_per_task,mem,time_limit,run_dir,data_path,ground_truth_intensity_path,ground_truth_params_path,out_dir\n' > "$LEDGER"
fi

printf '[submit-eval] bench_root=%s\n' "$BENCH_ROOT"
printf '[submit-eval] metric_profile=%s split=%s device=%s\n' "$METRIC_PROFILE" "$SPLIT" "$DEVICE"

while IFS=$'\x1f' read -r BENCH_ID DATASET_ID PRESET SEED RUN_DIR DATA_PATH TRAIN_DATA TARGET_SPLIT DATASET_REF DATASET_REVISION GT_INTENSITY_PATH GT_PARAMS_PATH; do
  JOB_NAME="${BENCH_ID}__eval__${PRESET}__s${SEED}"
  OUT_DIR="${RUN_DIR}/evaluate/metrics/${METRIC_PROFILE}_${TARGET_SPLIT}"
  CAMPAIGN_ID="${BENCH_ID}__eval_metrics"

  printf '[submit-eval] job=%s run=%s\n' "$JOB_NAME" "$RUN_DIR"

  ENV_ARGS=()
  declare -a ENV_ARGS
  ENV_ARGS+=(
    REPO_ROOT="$REPO_ROOT"
    RUN_ROOT="$RUN_ROOT"
    DATA_ROOT="$DATA_ROOT"
    CAMPAIGN_ID="$CAMPAIGN_ID"
    RUN_DIR="$RUN_DIR"
    TRAIN_DATA="$TRAIN_DATA"
    SPLIT="$TARGET_SPLIT"
    METRIC_PROFILE="$METRIC_PROFILE"
    ARTIFACT_MODE="$ARTIFACT_MODE"
    DEVICE="$DEVICE"
    EVAL_SEED="${EVAL_SEED:-0}"
    K_PRED="${K_PRED:-32}"
    K_GEN="${K_GEN:-20}"
    EXACT_TIME_BINS="${EXACT_TIME_BINS:-8}"
    EXACT_SPATIAL_BINS="${EXACT_SPATIAL_BINS:-8}"
    BENCHMARK_ID="$BENCH_ID"
    OUT_DIR="$OUT_DIR"
    FORCE_RUN="${FORCE_RUN:-0}"
    CONTAINER_IMAGE="$CONTAINER_IMAGE"
    CONTAINER_WORKDIR="$CONTAINER_WORKDIR"
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS"
  )
  if [ -n "$DATASET_REF" ]; then
    ENV_ARGS+=( DATASET="$DATASET_REF" )
    if [ -n "$DATASET_REVISION" ]; then
      ENV_ARGS+=( DATASET_REVISION="$DATASET_REVISION" )
    fi
  else
    ENV_ARGS+=( DATA_PATH="$DATA_PATH" )
  fi
  if [ -n "${GT_INTENSITY_PATH:-}" ]; then
    ENV_ARGS+=( GROUND_TRUTH_INTENSITY="$GT_INTENSITY_PATH" )
  fi
  if [ -n "${GT_PARAMS_PATH:-}" ]; then
    ENV_ARGS+=( GROUND_TRUTH_PARAMS="$GT_PARAMS_PATH" )
  fi

  declare -a SCRIPT_ARGS
  SCRIPT_ARGS=(
    "${GT_INTENSITY_PATH:-}"
    "${GT_PARAMS_PATH:-}"
  )

  sbatch_output="$(
    env \
      "${ENV_ARGS[@]}" \
      sbatch --job-name="$JOB_NAME" "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${SCRIPT_ARGS[@]}"
  )"

  printf '%s\n' "$sbatch_output"
  JOB_ID="$(printf '%s\n' "$sbatch_output" | awk '/Submitted batch job/ {print $4}')"
  if [ -z "$JOB_ID" ]; then
    echo "Failed to parse job id from sbatch output" >&2
    exit 1
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$JOB_ID" \
    "$JOB_NAME" \
    "$BENCH_ROOT" \
    "$BENCH_ID" \
    "$DATASET_ID" \
    "$PRESET" \
    "$SEED" \
    "$TARGET_SPLIT" \
    "$METRIC_PROFILE" \
    "${PARTITION:-}" \
    "$GPUS" \
    "$CPUS_PER_TASK" \
    "$MEM" \
    "$TIME_LIMIT" \
    "$RUN_DIR" \
    "$DATA_PATH" \
    "${GT_INTENSITY_PATH:-}" \
    "${GT_PARAMS_PATH:-}" \
    "$OUT_DIR" >> "$LEDGER"

  printf '[submit-eval] recorded %s in %s\n' "$JOB_ID" "$LEDGER"
done < <("$PYTHON_BIN_LOCAL" "$TARGET_SCRIPT" --bench-root "$BENCH_ROOT" --split "$SPLIT" --format usv)

#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_hawkesnest_predictive_kde.sh <campaign-root-or-parent> [more roots...]

Submit HawkesNest predictive-KDE evaluation jobs for the presets currently
supported by the predictive rollout stack:
  auto_stpp deep_stpp smash diffusion_stpp

The helper discovers campaign manifests under the given roots and submits one
job per eligible campaign/preset pair.

Defaults:
  SPLIT=test
  SEQ_IDX=0
  START_EVENT_IDX=20
  ROLLOUT_MODE=teacher_forced
  N_FRAMES=2
  HORIZON=1.0
  STEP_SIZE=1.0
  N_ROLLOUTS=16
  GRID_SIZE=32
  WITH_RENDERS=0

Useful overrides:
  PARTITION=...
  EXCLUDE=serv-3307
  FORCE_RUN=1
  WITH_RENDERS=1
  PLOT_STYLE=both
  GIF=1
  DEVICE=cuda|cpu
EOF
}

if [[ "$#" -lt 1 ]]; then
  usage
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN_LOCAL="${UNIFIED_STPP_PYTHON_LOCAL:-python3}"
TARGET_SCRIPT="$ROOT/scripts/resolve_hawkesnest_predictive_kde_targets.py"
SBATCH_SCRIPT="$ROOT/scripts/pegasus_hawkesnest_predictive_kde.sbatch"

resource_profile() {
  case "$1" in
    smash|diffusion_stpp)
      printf '%s\x1f%s\x1f%s\x1f%s\n' "${GPU_CPUS_PER_TASK:-8}" "${GPU_MEM:-48G}" "${GPU_TIME_LIMIT:-48:00:00}" "${GPUS:-1}"
      ;;
    auto_stpp|deep_stpp)
      printf '%s\x1f%s\x1f%s\x1f%s\n' "${GPU_CPUS_PER_TASK:-8}" "${GPU_MEM:-32G}" "${GPU_TIME_LIMIT:-24:00:00}" "${GPUS:-1}"
      ;;
    *)
      printf '%s\x1f%s\x1f%s\x1f%s\n' "${CPU_CPUS_PER_TASK:-8}" "${CPU_MEM:-32G}" "${CPU_TIME_LIMIT:-12:00:00}" "0"
      ;;
  esac
}

mkdir -p "$ROOT/logs"
LEDGER="$ROOT/runs/hawkesnest_campaigns/predictive_kde_submissions.csv"
mkdir -p "$(dirname "$LEDGER")"
if [[ ! -f "$LEDGER" ]]; then
  printf 'submitted_at,job_id,job_name,campaign_root,suite,preset,split,seq_idx,start_event_idx,rollout_mode,n_frames,horizon,step_size,n_rollouts,grid_size,out_dir\n' > "$LEDGER"
fi

while IFS=$'\x1f' read -r CAMPAIGN_ID CAMPAIGN_ROOT SUITE SUITE_PATH PRESET; do
  [[ -n "$CAMPAIGN_ROOT" && -n "$PRESET" ]] || continue
  SPLITS_DIR="$SUITE_PATH/jsonl"
  if [[ ! -d "$SPLITS_DIR" ]]; then
    echo "Skipping $CAMPAIGN_ROOT $PRESET: missing splits dir $SPLITS_DIR" >&2
    continue
  fi

  IFS=$'\x1f' read -r CPUS MEM TIME_LIMIT GPUS_VALUE <<< "$(resource_profile "$PRESET")"
  DEVICE_VALUE="${DEVICE:-cuda}"
  if [[ "$GPUS_VALUE" == "0" ]]; then
    DEVICE_VALUE="${DEVICE:-cpu}"
  fi

  OUT_DIR="$CAMPAIGN_ROOT/evaluate/predictive_kde/$PRESET"
  JOB_NAME="${CAMPAIGN_ID}__${PRESET}__pkde"

  declare -a SBATCH_ARGS
  SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --gpus="$GPUS_VALUE"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME_LIMIT"
    --output="$ROOT/logs/%x_%j.out"
  )
  if [[ -n "${PARTITION:-}" ]]; then
    SBATCH_ARGS+=( --partition="$PARTITION" )
  fi
  if [[ -n "${EXCLUDE:-}" ]]; then
    SBATCH_ARGS+=( --exclude="$EXCLUDE" )
  fi

  declare -a ENV_ARGS
  ENV_ARGS=(
    REPO_ROOT="$ROOT"
    RUN_ROOT="$ROOT/runs"
    DATA_ROOT="$ROOT/data"
    CAMPAIGN_ID="${CAMPAIGN_ID}__predictive_kde"
    CAMPAIGN_ROOT="$CAMPAIGN_ROOT"
    PRESET="$PRESET"
    SPLITS_DIR="$SPLITS_DIR"
    OUT_DIR="$OUT_DIR"
    SPLIT="${SPLIT:-test}"
    SEQ_IDX="${SEQ_IDX:-0}"
    START_EVENT_IDX="${START_EVENT_IDX:-20}"
    HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
    ROLLOUT_MODE="${ROLLOUT_MODE:-teacher_forced}"
    N_FRAMES="${N_FRAMES:-2}"
    HORIZON="${HORIZON:-1.0}"
    STEP_SIZE="${STEP_SIZE:-1.0}"
    N_ROLLOUTS="${N_ROLLOUTS:-16}"
    GRID_SIZE="${GRID_SIZE:-32}"
    BANDWIDTH="${BANDWIDTH:-}"
    LAMBDA_BAR="${LAMBDA_BAR:-10.0}"
    MAX_EVENTS_PER_WINDOW="${MAX_EVENTS_PER_WINDOW:-64}"
    BRIDGE_RETRIES="${BRIDGE_RETRIES:-64}"
    ADAPTIVE_THINNING="${ADAPTIVE_THINNING:-1}"
    EXACT_PROPOSAL="${EXACT_PROPOSAL:-coarse}"
    EXACT_TIME_BINS="${EXACT_TIME_BINS:-12}"
    EXACT_SPATIAL_BINS="${EXACT_SPATIAL_BINS:-12}"
    EXACT_SAFETY="${EXACT_SAFETY:-2.0}"
    COLOR_PERCENTILE="${COLOR_PERCENTILE:-99.0}"
    EVAL_SEED="${EVAL_SEED:-0}"
    DEVICE="$DEVICE_VALUE"
    WITH_RENDERS="${WITH_RENDERS:-0}"
    PLOT_STYLE="${PLOT_STYLE:-both}"
    GIF="${GIF:-0}"
    FPS="${FPS:-2.0}"
    PLOT_SUMMARY="${PLOT_SUMMARY:-1}"
    FORCE_RUN="${FORCE_RUN:-0}"
    CONTAINER_IMAGE="${CONTAINER_IMAGE:-/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh}"
    CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$ROOT}"
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$ROOT:$ROOT,/netscratch/aalaila:/netscratch/aalaila}"
  )

  sbatch_output="$(
    env \
      "${ENV_ARGS[@]}" \
      sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT"
  )"
  printf '%s\n' "$sbatch_output"
  JOB_ID="$(printf '%s\n' "$sbatch_output" | awk '/Submitted batch job/ {print $4}')"
  if [[ -z "$JOB_ID" ]]; then
    echo "Failed to parse job id from sbatch output: $sbatch_output" >&2
    exit 1
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$JOB_ID" \
    "$JOB_NAME" \
    "$CAMPAIGN_ROOT" \
    "$SUITE" \
    "$PRESET" \
    "${SPLIT:-test}" \
    "${SEQ_IDX:-0}" \
    "${START_EVENT_IDX:-20}" \
    "${ROLLOUT_MODE:-teacher_forced}" \
    "${N_FRAMES:-2}" \
    "${HORIZON:-1.0}" \
    "${STEP_SIZE:-1.0}" \
    "${N_ROLLOUTS:-16}" \
    "${GRID_SIZE:-32}" \
    "$OUT_DIR" >> "$LEDGER"
done < <("$PYTHON_BIN_LOCAL" "$TARGET_SCRIPT" "$@" --format usv)

printf '[submit] recorded submissions in %s\n' "$LEDGER"

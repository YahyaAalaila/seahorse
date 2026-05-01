#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_hawkesnest_campaign_eval.sh <campaign-root-or-parent> [more roots...]

This submits one evaluation job per discovered synthetic campaign run.

Defaults:
  METRIC_PROFILES=predictive
  WITH_SURFACE_VIZ=0
  SPLIT=test

Useful overrides:
  PARTITION=...
  EXCLUDE=serv-3307
  FORCE_RUN=1

  CPU_CPUS_PER_TASK=8
  CPU_MEM=32G
  CPU_TIME_LIMIT=12:00:00

  GPU_CPUS_PER_TASK=8
  GPU_MEM=48G
  GPU_TIME_LIMIT=24:00:00

  METRIC_PROFILES=predictive,surface
  METRIC_PROFILES=autoregressive
  WITH_SURFACE_VIZ=1
  SURFACE_SEQ_IDX=0
  SURFACE_HISTORY_LENGTH=0
  FUTURE_HORIZON=...

Examples:
  scripts/submit_hawkesnest_campaign_eval.sh runs/hawkesnest_campaigns
  METRIC_PROFILES=predictive,surface WITH_SURFACE_VIZ=1 scripts/submit_hawkesnest_campaign_eval.sh runs/hawkesnest_campaigns/suite3_entanglement/s3ent_v2__deep_stpp__04251144
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN_LOCAL="${UNIFIED_STPP_PYTHON_LOCAL:-python}"
TARGET_SCRIPT="$ROOT/scripts/resolve_hawkesnest_campaign_eval_targets.py"
METRICS_SBATCH="$ROOT/scripts/pegasus_eval_metrics.sbatch"
SURFACE_SBATCH="$ROOT/scripts/pegasus_eval_surface.sbatch"

SPLIT="${SPLIT:-test}"
METRIC_PROFILES_CSV="${METRIC_PROFILES:-predictive}"
WITH_SURFACE_VIZ="${WITH_SURFACE_VIZ:-0}"

CPU_CPUS_PER_TASK="${CPU_CPUS_PER_TASK:-8}"
CPU_MEM="${CPU_MEM:-32G}"
CPU_TIME_LIMIT="${CPU_TIME_LIMIT:-12:00:00}"

GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-8}"
GPU_MEM="${GPU_MEM:-48G}"
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-24:00:00}"

SURFACE_SEQ_IDX="${SURFACE_SEQ_IDX:-0}"
SURFACE_HISTORY_LENGTH="${SURFACE_HISTORY_LENGTH:-0}"

declare -a METRIC_PROFILES
IFS=',' read -r -a METRIC_PROFILES <<< "$METRIC_PROFILES_CSV"

resolve_path() {
  "$PYTHON_BIN_LOCAL" - <<'PY' "$1"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

discover_campaign_roots() {
  local input
  local resolved
  for input in "$@"; do
    resolved="$(resolve_path "$input")"
    if [ -f "$resolved/manifests/campaign_manifest.json" ]; then
      printf '%s\n' "$resolved"
    elif [ -d "$resolved" ]; then
      find "$resolved" -path '*/manifests/campaign_manifest.json' -print | sed 's#/manifests/campaign_manifest.json##'
    else
      echo "Skipping missing path: $resolved" >&2
    fi
  done | awk 'NF' | sort -u
}

is_gpu_preset() {
  case "$1" in
    auto_stpp|deep_stpp|smash|diffusion_stpp|neural_attncnf|neural_jumpcnf|neural_cond_gmm)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

surface_profile_for_preset() {
  case "$1" in
    auto_stpp|deep_stpp)
      printf 'history_frame\n'
      ;;
    neural_attncnf|neural_jumpcnf|neural_cond_gmm)
      printf 'future_exact\n'
      ;;
    *)
      printf '\n'
      ;;
  esac
}

submit_metrics_jobs_for_campaign() {
  local campaign_root="$1"
  local campaign_id
  local metrics_ledger

  campaign_id="$(basename "$campaign_root")"
  metrics_ledger="$campaign_root/evaluate/metrics_submissions.csv"
  mkdir -p "$(dirname "$metrics_ledger")" "$ROOT/logs"
  if [ ! -f "$metrics_ledger" ]; then
    printf 'submitted_at,job_id,job_name,campaign_root,suite,config_id,preset,seed,metric_profile,split,run_dir,test_path,ground_truth_intensity_path,ground_truth_params_path,out_dir\n' > "$metrics_ledger"
  fi

  while IFS=$'\x1f' read -r TARGET_CAMPAIGN_ID SUITE SUITE_PATH CONFIG_ID LEVEL_INDEX PRESET SEED RUN_DIR TEST_PATH TRAIN_PATH GT_INTENSITY_PATH GT_PARAMS_PATH; do
    if [ -z "$RUN_DIR" ] || [ -z "$PRESET" ]; then
      continue
    fi
    local use_gpu
    use_gpu=0
    if is_gpu_preset "$PRESET"; then
      use_gpu=1
    fi

    local profile
    for profile in "${METRIC_PROFILES[@]}"; do
      [ -n "$profile" ] || continue
      local cpus mem time_limit gpus device job_name out_dir sbatch_output job_id
      if [ "$use_gpu" = "1" ]; then
        cpus="$GPU_CPUS_PER_TASK"
        mem="$GPU_MEM"
        time_limit="$GPU_TIME_LIMIT"
        gpus=1
        device="${DEVICE:-cuda}"
      else
        cpus="$CPU_CPUS_PER_TASK"
        mem="$CPU_MEM"
        time_limit="$CPU_TIME_LIMIT"
        gpus=0
        device="${DEVICE:-cpu}"
      fi

      out_dir="$RUN_DIR/evaluate/metrics/${profile}_${SPLIT}"
      job_name="${TARGET_CAMPAIGN_ID}__${CONFIG_ID}__${PRESET}__s${SEED}__${profile}"

      declare -a SBATCH_ARGS
      SBATCH_ARGS=(
        --job-name="$job_name"
        --gpus="$gpus"
        --cpus-per-task="$cpus"
        --mem="$mem"
        --time="$time_limit"
        --output="$ROOT/logs/%x_%j.out"
      )
      local export_spec
      export_spec="ALL"
      if [ -n "$GT_INTENSITY_PATH" ]; then
        export_spec="${export_spec},GROUND_TRUTH_INTENSITY=$GT_INTENSITY_PATH"
      fi
      if [ -n "$GT_PARAMS_PATH" ]; then
        export_spec="${export_spec},GROUND_TRUTH_PARAMS=$GT_PARAMS_PATH"
      fi
      SBATCH_ARGS+=( --export="$export_spec" )
      if [ -n "${PARTITION:-}" ]; then
        SBATCH_ARGS+=( --partition="$PARTITION" )
      fi
      if [ -n "${EXCLUDE:-}" ]; then
        SBATCH_ARGS+=( --exclude="$EXCLUDE" )
      fi

      declare -a ENV_ARGS
      ENV_ARGS=(
        REPO_ROOT="$ROOT"
        RUN_ROOT="$ROOT/runs"
        DATA_ROOT="$ROOT/data"
        CAMPAIGN_ID="${TARGET_CAMPAIGN_ID}__eval_${profile}"
        RUN_DIR="$RUN_DIR"
        DATA_PATH="$TEST_PATH"
        TRAIN_DATA="$TRAIN_PATH"
        SPLIT="$SPLIT"
        METRIC_PROFILE="$profile"
        ARTIFACT_MODE="${ARTIFACT_MODE:-load_or_compute}"
        DEVICE="$device"
        EVAL_SEED="${EVAL_SEED:-0}"
        K_PRED="${K_PRED:-64}"
        K_GEN="${K_GEN:-20}"
        N_CONTEXT_EVENTS="${N_CONTEXT_EVENTS:-50}"
        EXACT_TIME_BINS="${EXACT_TIME_BINS:-8}"
        EXACT_SPATIAL_BINS="${EXACT_SPATIAL_BINS:-8}"
        BENCHMARK_ID="${TARGET_CAMPAIGN_ID}__${CONFIG_ID}__${PRESET}__s${SEED}"
        OUT_DIR="$out_dir"
        FORCE_RUN="${FORCE_RUN:-0}"
        CONTAINER_IMAGE="${CONTAINER_IMAGE:-/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh}"
        CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$ROOT}"
        CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$ROOT:$ROOT,/netscratch/aalaila:/netscratch/aalaila}"
      )
      if [ -n "$GT_INTENSITY_PATH" ]; then
        ENV_ARGS+=( GROUND_TRUTH_INTENSITY="$GT_INTENSITY_PATH" )
      fi
      if [ -n "$GT_PARAMS_PATH" ]; then
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
          sbatch "${SBATCH_ARGS[@]}" "$METRICS_SBATCH" "${SCRIPT_ARGS[@]}"
      )"
      printf '%s\n' "$sbatch_output"
      job_id="$(printf '%s\n' "$sbatch_output" | awk '/Submitted batch job/ {print $4}')"
      if [ -z "$job_id" ]; then
        echo "Failed to parse job id from sbatch output: $sbatch_output" >&2
        exit 1
      fi
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" \
        "$job_id" \
        "$job_name" \
        "$campaign_root" \
        "$SUITE" \
        "$CONFIG_ID" \
        "$PRESET" \
        "$SEED" \
        "$profile" \
        "$SPLIT" \
        "$RUN_DIR" \
        "$TEST_PATH" \
        "$GT_INTENSITY_PATH" \
        "$GT_PARAMS_PATH" \
        "$out_dir" >> "$metrics_ledger"
    done
  done < <("$PYTHON_BIN_LOCAL" "$TARGET_SCRIPT" --campaign-root "$campaign_root" --split "$SPLIT" --group predictive --format usv)
}

submit_surface_jobs_for_campaign() {
  local campaign_root="$1"
  local surface_ledger
  surface_ledger="$campaign_root/evaluate/surface_submissions.csv"
  mkdir -p "$(dirname "$surface_ledger")" "$ROOT/logs"
  if [ ! -f "$surface_ledger" ]; then
    printf 'submitted_at,job_id,job_name,campaign_root,suite,config_id,preset,seed,surface_profile,split,run_dir,test_path,out_dir\n' > "$surface_ledger"
  fi

  while IFS=$'\x1f' read -r TARGET_CAMPAIGN_ID SUITE SUITE_PATH CONFIG_ID LEVEL_INDEX PRESET SEED RUN_DIR TEST_PATH TRAIN_PATH GT_INTENSITY_PATH GT_PARAMS_PATH; do
    if [ -z "$RUN_DIR" ] || [ -z "$PRESET" ]; then
      continue
    fi
    local surface_profile
    surface_profile="$(surface_profile_for_preset "$PRESET")"
    if [ -z "$surface_profile" ]; then
      continue
    fi

    local cpus mem time_limit gpus device job_name out_dir sbatch_output job_id
    cpus="$GPU_CPUS_PER_TASK"
    mem="$GPU_MEM"
    time_limit="$GPU_TIME_LIMIT"
    gpus=1
    device="${DEVICE:-cuda}"
    printf -v out_dir '%s/evaluate/surface/%s_%s_seq%03d' "$RUN_DIR" "$surface_profile" "$SPLIT" "$SURFACE_SEQ_IDX"
    job_name="${TARGET_CAMPAIGN_ID}__${CONFIG_ID}__${PRESET}__s${SEED}__${surface_profile}"

    declare -a SBATCH_ARGS
    SBATCH_ARGS=(
      --job-name="$job_name"
      --gpus="$gpus"
      --cpus-per-task="$cpus"
      --mem="$mem"
      --time="$time_limit"
      --output="$ROOT/logs/%x_%j.out"
    )
    if [ -n "${PARTITION:-}" ]; then
      SBATCH_ARGS+=( --partition="$PARTITION" )
    fi
    if [ -n "${EXCLUDE:-}" ]; then
      SBATCH_ARGS+=( --exclude="$EXCLUDE" )
    fi

    declare -a ENV_ARGS
    ENV_ARGS=(
      REPO_ROOT="$ROOT"
      RUN_DIR="$RUN_DIR"
      HISTORY_PATH="$TEST_PATH"
      OUT_DIR="$out_dir"
      PROFILE="$surface_profile"
      DEVICE="$device"
      SPLIT="$SPLIT"
      SEQ_IDX="$SURFACE_SEQ_IDX"
      HISTORY_LENGTH="$SURFACE_HISTORY_LENGTH"
      FORCE_RUN="${FORCE_RUN:-0}"
      CONTAINER_IMAGE="${CONTAINER_IMAGE:-/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh}"
      CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$ROOT}"
      CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$ROOT:$ROOT,/netscratch/aalaila:/netscratch/aalaila}"
    )
    if [ -n "${FUTURE_HORIZON:-}" ]; then
      ENV_ARGS+=( FUTURE_HORIZON="${FUTURE_HORIZON}" )
    fi

    sbatch_output="$(
      env \
        "${ENV_ARGS[@]}" \
        sbatch "${SBATCH_ARGS[@]}" "$SURFACE_SBATCH"
    )"
    printf '%s\n' "$sbatch_output"
    job_id="$(printf '%s\n' "$sbatch_output" | awk '/Submitted batch job/ {print $4}')"
    if [ -z "$job_id" ]; then
      echo "Failed to parse job id from sbatch output: $sbatch_output" >&2
      exit 1
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "$job_id" \
      "$job_name" \
      "$campaign_root" \
      "$SUITE" \
      "$CONFIG_ID" \
      "$PRESET" \
      "$SEED" \
      "$surface_profile" \
      "$SPLIT" \
      "$RUN_DIR" \
      "$TEST_PATH" \
      "$out_dir" >> "$surface_ledger"
  done < <("$PYTHON_BIN_LOCAL" "$TARGET_SCRIPT" --campaign-root "$campaign_root" --split "$SPLIT" --group all --format usv)
}

while IFS= read -r campaign_root; do
  [ -n "$campaign_root" ] || continue
  printf '[submit-hawkesnest-eval] campaign_root=%s metrics=%s\n' "$campaign_root" "$METRIC_PROFILES_CSV"
  submit_metrics_jobs_for_campaign "$campaign_root"
  if [ "$WITH_SURFACE_VIZ" = "1" ]; then
    printf '[submit-hawkesnest-eval] campaign_root=%s with_surface_viz=1\n' "$campaign_root"
    submit_surface_jobs_for_campaign "$campaign_root"
  fi
done < <(discover_campaign_roots "$@")

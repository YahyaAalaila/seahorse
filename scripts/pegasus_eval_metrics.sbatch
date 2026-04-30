#!/usr/bin/env bash
#SBATCH --job-name=ustpp-eval-metrics
#SBATCH --output=%x-%j.out
set -euo pipefail

# Slurm export behavior can vary across cluster wrappers.  Accepting these as
# positional arguments makes synthetic GT handoff independent of environment
# propagation from the submit shell.
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  GROUND_TRUTH_INTENSITY="$1"
fi
if [[ $# -ge 2 && -n "${2:-}" ]]; then
  GROUND_TRUTH_PARAMS="$2"
fi
if [[ $# -ge 3 && -n "${3:-}" ]]; then
  DOMAIN_MASK_PATH="$3"
fi

if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/scripts/pegasus_campaign_env.sh" ]]; then
  source "${REPO_ROOT}/scripts/pegasus_campaign_env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/pegasus_campaign_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/pegasus_campaign_env.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "${SCRIPT_DIR}/pegasus_campaign_env.sh"
fi

for name in REPO_ROOT RUN_DIR OUT_DIR; do
  require_var "$name"
done
if [[ -z "${DATA_PATH:-}" && -z "${DATASET:-}" ]]; then
  echo "Missing required environment variable: DATA_PATH or DATASET" >&2
  exit 2
fi

PYTHON_BIN="${UNIFIED_STPP_PYTHON:-python}"
mkdir -p "${OUT_DIR}"
register_job_summary_trap "evaluate:metrics:${METRIC_PROFILE:-core}" "${OUT_DIR}"

if [[ "${FORCE_RUN:-0}" != "1" ]] && job_complete "${OUT_DIR}" metrics.json evaluation_manifest.json artifacts.json; then
  echo "[eval-metrics] output already complete: ${OUT_DIR}"
  exit 0
fi

CMD=(
  "${PYTHON_BIN}" scripts/run_eval_metrics.py
  --run "${RUN_DIR}"
  --metric-profile "${METRIC_PROFILE:-core}"
  --artifact-mode "${ARTIFACT_MODE:-load_or_compute}"
  --device "${DEVICE:-cpu}"
  --seed "${EVAL_SEED:-0}"
  --k-pred "${K_PRED:-32}"
  --k-gen "${K_GEN:-20}"
  --exact-time-bins "${EXACT_TIME_BINS:-8}"
  --exact-spatial-bins "${EXACT_SPATIAL_BINS:-8}"
  --out "${OUT_DIR}"
)

if [[ -n "${DATA_PATH:-}" ]]; then
  CMD+=( --data "${DATA_PATH}" )
else
  CMD+=( --dataset "${DATASET}" )
  if [[ -n "${DATASET_REVISION:-}" ]]; then
    CMD+=( --dataset-revision "${DATASET_REVISION}" )
  fi
fi

if [[ -n "${SPLIT:-}" ]]; then
  CMD+=( --split "${SPLIT}" )
fi
if [[ -n "${TRAIN_DATA:-}" ]]; then
  CMD+=( --train-data "${TRAIN_DATA}" )
fi
if [[ -n "${GROUND_TRUTH_INTENSITY:-}" ]]; then
  CMD+=( --ground-truth-intensity "${GROUND_TRUTH_INTENSITY}" )
fi
if [[ -n "${GROUND_TRUTH_PARAMS:-}" ]]; then
  CMD+=( --ground-truth-params "${GROUND_TRUTH_PARAMS}" )
fi
if [[ -n "${DOMAIN_MASK_PATH:-}" ]]; then
  CMD+=( --domain-mask "${DOMAIN_MASK_PATH}" )
fi
if [[ -n "${SEQ_SHARD:-}" ]]; then
  CMD+=( --seq-shard "${SEQ_SHARD}" )
fi
if [[ -n "${BENCHMARK_ID:-}" ]]; then
  CMD+=( --benchmark-id "${BENCHMARK_ID}" )
fi

cd "${REPO_ROOT}"
print_context
printf '[eval-metrics]'; printf ' %q' "${CMD[@]}"; printf '\n'
run_cmd "${CMD[@]}"

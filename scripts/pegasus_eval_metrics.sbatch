#!/usr/bin/env bash
#SBATCH --job-name=ustpp-eval-metrics
#SBATCH --output=%x-%j.out
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pegasus_campaign_env.sh"

for name in REPO_ROOT RUN_DIR DATA_PATH OUT_DIR; do
  require_var "$name"
done

PYTHON_BIN="${UNIFIED_STPP_PYTHON:-python}"
mkdir -p "${OUT_DIR}"

if [[ "${FORCE_RUN:-0}" != "1" ]] && job_complete "${OUT_DIR}" metrics.json evaluation_manifest.json artifacts.json; then
  echo "[eval-metrics] output already complete: ${OUT_DIR}"
  exit 0
fi

CMD=(
  "${PYTHON_BIN}" -m unified_stpp evaluate metrics
  --run "${RUN_DIR}"
  --data "${DATA_PATH}"
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

if [[ -n "${SPLIT:-}" ]]; then
  CMD+=( --split "${SPLIT}" )
fi
if [[ -n "${TRAIN_DATA:-}" ]]; then
  CMD+=( --train-data "${TRAIN_DATA}" )
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

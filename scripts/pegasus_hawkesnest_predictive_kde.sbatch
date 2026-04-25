#!/usr/bin/env bash
#SBATCH --job-name=ustpp-hawkesnest-kde
#SBATCH --output=%x-%j.out
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/scripts/pegasus_campaign_env.sh" ]]; then
  source "${REPO_ROOT}/scripts/pegasus_campaign_env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/pegasus_campaign_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/pegasus_campaign_env.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "${SCRIPT_DIR}/pegasus_campaign_env.sh"
fi

for name in REPO_ROOT CAMPAIGN_ROOT PRESET SPLITS_DIR OUT_DIR; do
  require_var "$name"
done

PYTHON_BIN="${UNIFIED_STPP_PYTHON:-python}"
mkdir -p "${OUT_DIR}"

if [[ "${FORCE_RUN:-0}" != "1" ]] && job_complete "${OUT_DIR}" metrics.json metrics_by_run.csv metrics_by_family_level.csv; then
  echo "[hawkesnest-predictive-kde] output already complete: ${OUT_DIR}"
  exit 0
fi

CMD=(
  "${PYTHON_BIN}" scripts/hawkesnest_predictive_family_metrics.py
  --bench-run-root "${CAMPAIGN_ROOT}"
  --preset "${PRESET}"
  --splits-dir "${SPLITS_DIR}"
  --out-dir "${OUT_DIR}"
  --split "${SPLIT:-test}"
  --seq-idx "${SEQ_IDX:-0}"
  --start-event-idx "${START_EVENT_IDX:-20}"
  --history-length "${HISTORY_LENGTH:-0}"
  --rollout-mode "${ROLLOUT_MODE:-teacher_forced}"
  --n-frames "${N_FRAMES:-2}"
  --horizon "${HORIZON:-1.0}"
  --step-size "${STEP_SIZE:-1.0}"
  --n-rollouts "${N_ROLLOUTS:-16}"
  --grid-size "${GRID_SIZE:-32}"
  --lambda-bar "${LAMBDA_BAR:-10.0}"
  --max-events-per-window "${MAX_EVENTS_PER_WINDOW:-64}"
  --bridge-retries "${BRIDGE_RETRIES:-64}"
  --exact-proposal "${EXACT_PROPOSAL:-coarse}"
  --exact-time-bins "${EXACT_TIME_BINS:-12}"
  --exact-spatial-bins "${EXACT_SPATIAL_BINS:-12}"
  --exact-safety "${EXACT_SAFETY:-2.0}"
  --color-percentile "${COLOR_PERCENTILE:-99.0}"
  --eval-seed "${EVAL_SEED:-0}"
  --device "${DEVICE:-cuda}"
  --keep-going
)

if [[ -n "${BANDWIDTH:-}" ]]; then
  CMD+=( --bandwidth "${BANDWIDTH}" )
fi
if [[ "${ADAPTIVE_THINNING:-1}" != "1" ]]; then
  CMD+=( --no-adaptive-thinning )
fi
if [[ "${PLOT_SUMMARY:-1}" != "1" ]]; then
  CMD+=( --no-plot )
fi
if [[ "${WITH_RENDERS:-0}" == "1" ]]; then
  CMD+=( --with-renders --plot-style "${PLOT_STYLE:-both}" --fps "${FPS:-2.0}" )
  if [[ "${GIF:-0}" == "1" ]]; then
    CMD+=( --gif )
  fi
fi

cd "${REPO_ROOT}"
print_context
printf '[hawkesnest-predictive-kde]'; printf ' %q' "${CMD[@]}"; printf '\n'
run_cmd "${CMD[@]}"

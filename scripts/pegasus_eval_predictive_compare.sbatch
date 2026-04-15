#!/usr/bin/env bash
#SBATCH --job-name=ustpp-eval-predictive
#SBATCH --output=%x-%j.out
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pegasus_campaign_env.sh"

for name in REPO_ROOT RUNS HISTORY_PATH HORIZON OUT_DIR; do
  require_var "$name"
done

PYTHON_BIN="${UNIFIED_STPP_PYTHON:-python}"
mkdir -p "${OUT_DIR}"

if [[ "${FORCE_RUN:-0}" != "1" ]] && job_complete "${OUT_DIR}" summary.json artifacts.json derived_surfaces.npz; then
  echo "[eval-predictive] output already complete: ${OUT_DIR}"
  exit 0
fi

CMD=(
  "${PYTHON_BIN}" -m unified_stpp evaluate predictive-compare
  --history "${HISTORY_PATH}"
  --horizon "${HORIZON}"
  --n-rollouts "${N_ROLLOUTS:-128}"
  --n-frames "${N_FRAMES:-6}"
  --rollout-mode "${ROLLOUT_MODE:-teacher_forced}"
  --seed "${EVAL_SEED:-0}"
  --device "${DEVICE:-auto}"
  --plot-style "${PLOT_STYLE:-2d}"
  --out "${OUT_DIR}"
)

read -r -a RUN_ARRAY <<< "${RUNS}"
for run_dir in "${RUN_ARRAY[@]}"; do
  CMD+=( --run "${run_dir}" )
done

if [[ -n "${LABELS:-}" ]]; then
  read -r -a LABEL_ARRAY <<< "${LABELS}"
  for label in "${LABEL_ARRAY[@]}"; do
    CMD+=( --label "${label}" )
  done
fi
if [[ -n "${SPLIT:-}" ]]; then
  CMD+=( --split "${SPLIT}" )
fi
if [[ -n "${SEQ_IDX:-}" ]]; then
  CMD+=( --seq-idx "${SEQ_IDX}" )
fi
if [[ -n "${START_EVENT_IDX:-}" ]]; then
  CMD+=( --start-event-idx "${START_EVENT_IDX}" )
fi
if [[ -n "${HISTORY_LENGTH:-}" ]]; then
  CMD+=( --history-length "${HISTORY_LENGTH}" )
fi

cd "${REPO_ROOT}"
print_context
printf '[eval-predictive]'; printf ' %q' "${CMD[@]}"; printf '\n'
run_cmd "${CMD[@]}"

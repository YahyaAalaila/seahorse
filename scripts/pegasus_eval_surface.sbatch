#!/usr/bin/env bash
#SBATCH --job-name=ustpp-eval-surface
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

for name in REPO_ROOT RUN_DIR HISTORY_PATH OUT_DIR; do
  require_var "$name"
done

PYTHON_BIN="${UNIFIED_STPP_PYTHON:-python}"
mkdir -p "${OUT_DIR}"
register_job_summary_trap "evaluate:surface:${PROFILE:-history_frame}" "${OUT_DIR}"

if [[ "${FORCE_RUN:-0}" != "1" ]] && job_complete "${OUT_DIR}" summary.json data.npz artifacts.json; then
  echo "[eval-surface] output already complete: ${OUT_DIR}"
  exit 0
fi

CMD=(
  "${PYTHON_BIN}" -m unified_stpp evaluate surface
  --run "${RUN_DIR}"
  --history "${HISTORY_PATH}"
  --profile "${PROFILE:-history_frame}"
  --device "${DEVICE:-cpu}"
  --out "${OUT_DIR}"
)

if [[ -n "${SPLIT:-}" ]]; then
  CMD+=( --split "${SPLIT}" )
fi
if [[ -n "${SEQ_IDX:-}" ]]; then
  CMD+=( --seq-idx "${SEQ_IDX}" )
fi
if [[ -n "${HISTORY_LENGTH:-}" ]]; then
  CMD+=( --history-length "${HISTORY_LENGTH}" )
fi
if [[ -n "${FUTURE_HORIZON:-}" ]]; then
  CMD+=( --future-horizon "${FUTURE_HORIZON}" )
fi

cd "${REPO_ROOT}"
print_context
printf '[eval-surface]'; printf ' %q' "${CMD[@]}"; printf '\n'
run_cmd "${CMD[@]}"

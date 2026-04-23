#!/usr/bin/env bash
#SBATCH --job-name=ustpp-tune
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

for name in REPO_ROOT RUN_ROOT CAMPAIGN_ID PRESET; do
  require_var "$name"
done

PYTHON_BIN="${UNIFIED_STPP_PYTHON:-python}"
OUT_PATH="${OUT_PATH:-$(campaign_hpo_dir)/${PRESET}_best.yaml}"
MANIFEST_PATH="${OUT_PATH%.yaml}.hpo_manifest.json"

mkdir -p "$(dirname "${OUT_PATH}")" "$(campaign_logs_dir)"

if [[ "${FORCE_RUN:-0}" != "1" ]] && job_complete "$(dirname "${OUT_PATH}")" "$(basename "${OUT_PATH}")" "$(basename "${MANIFEST_PATH}")"; then
  echo "[tune] output already complete: ${OUT_PATH}"
  exit 0
fi

CMD=( "${PYTHON_BIN}" -m unified_stpp tune --preset "${PRESET}" --out "${OUT_PATH}" )

if [[ -n "${DATASET:-}" ]]; then
  CMD+=( --dataset "${DATASET}" )
  if [[ -n "${DATASET_REVISION:-}" ]]; then
    CMD+=( --dataset-revision "${DATASET_REVISION}" )
  fi
else
  for name in TRAIN_PATH VAL_PATH; do
    require_var "$name"
  done
  CMD+=( --train "${TRAIN_PATH}" --val "${VAL_PATH}" )
fi

if [[ -n "${N_TRIALS:-}" ]]; then
  CMD+=( --n_trials "${N_TRIALS}" )
fi
if [[ -n "${SEARCH_ALG:-}" ]]; then
  CMD+=( --search-alg "${SEARCH_ALG}" )
fi
if [[ -n "${SCHEDULER:-}" ]]; then
  CMD+=( --scheduler "${SCHEDULER}" )
fi
if [[ -n "${TUNE_SEED:-}" ]]; then
  CMD+=( --seed "${TUNE_SEED}" )
fi
if [[ -n "${MAX_CONCURRENT_TRIALS:-}" ]]; then
  CMD+=( --max-concurrent-trials "${MAX_CONCURRENT_TRIALS}" )
fi
if [[ "${FAIL_FAST:-0}" == "1" ]]; then
  CMD+=( --fail-fast )
fi

cd "${REPO_ROOT}"
print_context
printf '[tune]'; printf ' %q' "${CMD[@]}"; printf '\n'
run_cmd "${CMD[@]}"

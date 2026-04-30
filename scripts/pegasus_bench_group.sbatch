#!/usr/bin/env bash
#SBATCH --job-name=ustpp-bench
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

for name in REPO_ROOT RUN_ROOT CAMPAIGN_ID GROUP_NAME; do
  require_var "$name"
done

PYTHON_BIN="${UNIFIED_STPP_PYTHON:-python}"
BENCH_OUT="${BENCH_OUT:-$(campaign_bench_dir "${GROUP_NAME}")}"

mkdir -p "${BENCH_OUT}" "$(campaign_logs_dir)"
register_job_summary_trap "train:bench" "${BENCH_OUT}"

if [[ "${FORCE_RUN:-0}" != "1" ]] && job_complete "${BENCH_OUT}" bench_meta.json results.json report.html; then
  echo "[bench] output already complete: ${BENCH_OUT}"
  exit 0
fi

CMD=( "${PYTHON_BIN}" -m unified_stpp bench --out "${BENCH_OUT}" --n_workers "${N_WORKERS:-1}" --no-normalize )

if [[ -n "${PRESET:-}" ]]; then
  CMD+=( --preset "${PRESET}" )
else
  require_var PRESETS
  read -r -a PRESET_ARRAY <<< "${PRESETS}"
  CMD+=( --presets "${PRESET_ARRAY[@]}" )
fi

if [[ -n "${DATASET:-}" ]]; then
  CMD+=( --dataset "${DATASET}" )
  if [[ -n "${DATASET_REVISION:-}" ]]; then
    CMD+=( --dataset-revision "${DATASET_REVISION}" )
  fi
else
  require_var SPLITS_DIR
  CMD+=( --splits_dir "${SPLITS_DIR}" )
  if [[ -n "${DATASETS:-}" ]]; then
    read -r -a DATASET_ARRAY <<< "${DATASETS}"
    CMD+=( --datasets "${DATASET_ARRAY[@]}" )
  fi
fi

read -r -a SEED_ARRAY <<< "${SEEDS:-42}"
CMD+=( --seeds "${SEED_ARRAY[@]}" )

if [[ "${USE_HPO:-0}" == "1" ]]; then
  HPO_DIR="${HPO_CONFIGS_DIR:-$(campaign_hpo_dir)}"
  if [[ -d "${HPO_DIR}" ]]; then
    CMD+=( --hpo_configs_dir "${HPO_DIR}" )
  fi
fi

if [[ -n "${OVERRIDES:-}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  CMD+=( --override "${OVERRIDE_ARRAY[@]}" )
fi

cd "${REPO_ROOT}"
print_context
printf '[bench]'; printf ' %q' "${CMD[@]}"; printf '\n'
run_cmd "${CMD[@]}"

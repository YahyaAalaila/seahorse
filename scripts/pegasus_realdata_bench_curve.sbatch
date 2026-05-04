#!/usr/bin/env bash
#SBATCH --job-name=ustpp-curve-bench
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

for name in REPO_ROOT RUN_ROOT DATASET PRESET SEED BENCH_OUT; do
  require_var "$name"
done

PYTHON_BIN="${UNIFIED_STPP_PYTHON:-python}"
mkdir -p "${BENCH_OUT}" "$(campaign_logs_dir)"
register_job_summary_trap "train:bench_curve" "${BENCH_OUT}"

CMD=(
  "${PYTHON_BIN}" scripts/run_realdata_bench_with_curve.py
  --preset "${PRESET}"
  --dataset "${DATASET}"
  --seed "${SEED}"
  --out "${BENCH_OUT}"
  --curve-step "${CURVE_STEP:-0.05}"
  --no-normalize
)

if [[ -n "${DATASET_REVISION:-}" ]]; then
  CMD+=( --dataset-revision "${DATASET_REVISION}" )
fi
if [[ -n "${SUITE:-}" ]]; then
  CMD+=( --suite "${SUITE}" )
fi
if [[ -n "${CONFIG_ID:-}" ]]; then
  CMD+=( --config-id "${CONFIG_ID}" )
fi
if [[ -n "${OVERRIDES:-}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  CMD+=( --override "${OVERRIDE_ARRAY[@]}" )
fi

cd "${REPO_ROOT}"
print_context
printf '[curve-bench]'; printf ' %q' "${CMD[@]}"; printf '\n'
run_cmd "${CMD[@]}"

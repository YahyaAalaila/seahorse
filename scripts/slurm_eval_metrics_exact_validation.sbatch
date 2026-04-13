#!/bin/bash
# Packaged-path validation only. This is not a benchmark-valid rerun by itself.
# Target: exact/intensity model, CPU thinning-backed predictive metrics.
# Submit from the repo root, or pass REPO_ROOT=/path/to/unified_stpp to sbatch.
#SBATCH --job-name=eval_metrics_exact
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=eval_metrics_exact_%j.out
#SBATCH --error=eval_metrics_exact_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/unified_stpp_matplotlib_${SLURM_JOB_ID:-local}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/unified_stpp_cache_${SLURM_JOB_ID:-local}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

RUN="runs/bch_hawkes_famv1_5seeds_auto_stpp/fit/auto_stpp/20260410_213027_b80c1ebf"
DATA="data/exports/unified_stpp_hawkesnest_hard_v2/echo_E0/test.jsonl"
OUT_ROOT="${OUT_ROOT:-runs/eval_metrics_packaged_validation}"
ART="${ART_ROOT:-$OUT_ROOT/artifacts}/auto_stpp_echo_E0"

mkdir -p "$OUT_ROOT" "$ART"

run_step() {
  local name="$1"
  shift
  echo "BEGIN ${name} $(date -Is)"
  local start=$SECONDS
  set +e
  "$@"
  local status=$?
  set -e
  local elapsed=$((SECONDS - start))
  echo "END ${name} status=${status} elapsed_sec=${elapsed} $(date -Is)"
  return "$status"
}

echo "scope=packaged_path_validation_not_benchmark"
echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "python=$PYTHON_BIN"
echo "run=$RUN"
echo "data=$DATA"
echo "artifact_dir=$ART"

run_step core "$PYTHON_BIN" -m unified_stpp evaluate metrics \
  --run "$RUN" \
  --data "$DATA" \
  --split test \
  --metric-profile core \
  --max-seqs 10 \
  --max-events 100 \
  --device cpu \
  --artifact-dir "$ART" \
  --artifact-mode load_or_compute \
  --benchmark-id packaged_path_validation_not_benchmark_exact \
  --out "$OUT_ROOT/auto_stpp_echo_E0_core"

run_step predictive_compute "$PYTHON_BIN" -m unified_stpp evaluate metrics \
  --run "$RUN" \
  --data "$DATA" \
  --split test \
  --metric-profile predictive \
  --max-seqs 10 \
  --max-events 100 \
  --k-pred 32 \
  --device cpu \
  --artifact-dir "$ART" \
  --artifact-mode load_or_compute \
  --benchmark-id packaged_path_validation_not_benchmark_exact \
  --out "$OUT_ROOT/auto_stpp_echo_E0_predictive_compute"

run_step predictive_load "$PYTHON_BIN" -m unified_stpp evaluate metrics \
  --run "$RUN" \
  --data "$DATA" \
  --split test \
  --metric-profile predictive \
  --max-seqs 10 \
  --max-events 100 \
  --k-pred 32 \
  --device cpu \
  --artifact-dir "$ART" \
  --artifact-mode load_only \
  --benchmark-id packaged_path_validation_not_benchmark_exact \
  --out "$OUT_ROOT/auto_stpp_echo_E0_predictive_load"

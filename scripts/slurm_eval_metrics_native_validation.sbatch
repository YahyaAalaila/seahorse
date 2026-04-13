#!/bin/bash
# Packaged-path validation only. This is not a benchmark-valid post-freeze run:
# the selected diffusion_stpp run is a legacy native-sampler run.
# Target: approximate/native-sampler model, GPU-backed predictive metrics.
# Submit from the repo root, or pass REPO_ROOT=/path/to/unified_stpp to sbatch.
#SBATCH --job-name=eval_metrics_native
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=eval_metrics_native_%j.out
#SBATCH --error=eval_metrics_native_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/unified_stpp_matplotlib_${SLURM_JOB_ID:-local}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/unified_stpp_cache_${SLURM_JOB_ID:-local}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

RUN="runs/runtotryvis2/fit/diffusion_stpp/20260330_181445_eb9498c2"
DATA="data/sthp0/test.jsonl"
OUT_ROOT="${OUT_ROOT:-runs/eval_metrics_packaged_validation}"
ART="${ART_ROOT:-$OUT_ROOT/artifacts}/diffusion_stpp_sthp0_legacy"

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

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
run_step cuda_probe "$PYTHON_BIN" -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('cuda_device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); assert torch.cuda.is_available()"

run_step core "$PYTHON_BIN" -m unified_stpp evaluate metrics \
  --run "$RUN" \
  --data "$DATA" \
  --split test \
  --metric-profile core \
  --max-seqs 5 \
  --max-events 100 \
  --device cuda \
  --artifact-dir "$ART" \
  --artifact-mode load_or_compute \
  --benchmark-id packaged_path_validation_not_benchmark_native_legacy \
  --out "$OUT_ROOT/diffusion_stpp_sthp0_core"

run_step predictive_compute "$PYTHON_BIN" -m unified_stpp evaluate metrics \
  --run "$RUN" \
  --data "$DATA" \
  --split test \
  --metric-profile predictive \
  --max-seqs 5 \
  --max-events 100 \
  --k-pred 64 \
  --device cuda \
  --artifact-dir "$ART" \
  --artifact-mode load_or_compute \
  --benchmark-id packaged_path_validation_not_benchmark_native_legacy \
  --out "$OUT_ROOT/diffusion_stpp_sthp0_predictive_compute"

run_step predictive_load "$PYTHON_BIN" -m unified_stpp evaluate metrics \
  --run "$RUN" \
  --data "$DATA" \
  --split test \
  --metric-profile predictive \
  --max-seqs 5 \
  --max-events 100 \
  --k-pred 64 \
  --device cuda \
  --artifact-dir "$ART" \
  --artifact-mode load_only \
  --benchmark-id packaged_path_validation_not_benchmark_native_legacy \
  --out "$OUT_ROOT/diffusion_stpp_sthp0_predictive_load"

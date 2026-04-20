#!/bin/bash
#SBATCH --job-name=factgmm-tune
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch tune_factorized_gmm_cpu.sbatch <preset> [preset ...]"
  echo "Good for: poisson_gmm hawkes_gmm selfcorrecting_gmm"
  exit 1
fi

DATASET="${DATASET:-yahya021/covid-stpp}"
DATASET_REVISION="${DATASET_REVISION:-}"
SEED="${SEED:-42}"
N_TRIALS="${N_TRIALS:-30}"
MAX_CONCURRENT_TRIALS="${MAX_CONCURRENT_TRIALS:-8}"

ROOT="${ROOT:-/home/aalaila/projects/uni-stpp}"
cd "$ROOT"

DATASET_SLUG="$(basename "$DATASET")"
STAMP="$(date +%Y%m%d_%H%M%S)"

for PRESET in "$@"; do
  OUT_DIR="runs/exp1/${DATASET_SLUG}/tune/${PRESET}"
  mkdir -p "$OUT_DIR"

  CMD=(python -m unified_stpp tune
    --config "unified_stpp/configs/${PRESET}_hpo.yaml"
    --dataset "$DATASET"
    --seed "$SEED"
    --n_trials "$N_TRIALS"
    --max-concurrent-trials "$MAX_CONCURRENT_TRIALS"
    --out "$OUT_DIR/${PRESET}_best_${STAMP}.yaml")

  if [ -n "$DATASET_REVISION" ]; then
    CMD+=(--dataset-revision "$DATASET_REVISION")
  fi

  printf '[tune:%s]' "$PRESET"
  printf ' %q' "${CMD[@]}"
  printf '\n'

  printf -v CMD_STRING '%q ' "${CMD[@]}"
  srun -K \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh \
    --container-workdir="$ROOT" \
    --container-mounts="$ROOT:$ROOT,/netscratch/aalaila:/netscratch/aalaila" \
    bash -lc "
      set -euo pipefail
      python -m pip install --break-system-packages -e '.[hpo]'
      ${CMD_STRING}
    "
done

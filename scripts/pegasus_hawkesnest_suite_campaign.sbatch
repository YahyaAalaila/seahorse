#!/bin/bash
#SBATCH --job-name=hawkesnest-campaign
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch pegasus_hawkesnest_suite_campaign.sbatch <suite_name> <preset1> [preset2 ...]"
  echo "Example: sbatch pegasus_hawkesnest_suite_campaign.sbatch suite4_heterogeneity smash diffusion_stpp"
  exit 1
fi

SUITE_NAME="$1"
shift
PRESETS=("$@")

ROOT="/home/aalaila/projects/uni-stpp"
SUITE_ROOT="${SUITE_ROOT:-$ROOT/data/hawkesnest_suitesv2}"
SUITE_PATH="${SUITE_PATH:-$SUITE_ROOT/$SUITE_NAME}"
SEEDS="${SEEDS:-42}"
HPO_SEED="${HPO_SEED:-42}"
HPO_CONFIG_DIR="${HPO_CONFIG_DIR:-unified_stpp/configs}"
DEVICE="${DEVICE:-cuda}"
STAGE="${STAGE:-all}"
CURVE_STEP="${CURVE_STEP:-0.1}"

PRESET_SLUG="$(printf '%s_' "${PRESETS[@]}")"
PRESET_SLUG="${PRESET_SLUG%_}"
PRESETS_STR="${PRESETS[*]}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$PRESET_SLUG}"
OUT_ROOT="${OUT_ROOT:-runs/hawkesnest_campaigns/${SUITE_NAME}/${CAMPAIGN_TAG}}"

cd "$ROOT"
mkdir -p logs "$OUT_ROOT"

srun -K \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh \
  --container-workdir="$ROOT" \
  --container-mounts="$ROOT:$ROOT,/netscratch/aalaila:/netscratch/aalaila" \
  bash -lc "
    set -euo pipefail
    mkdir -p /tmp/mpl
    export MPLCONFIGDIR=/tmp/mpl
    pip install -e \".[hpo]\"
    read -r -a PRESET_ARRAY <<< \"$PRESETS_STR\"
    read -r -a SEED_ARRAY <<< \"$SEEDS\"

    CMD=(
      python scripts/synthetic_suite_campaign.py
      --suite-path \"$SUITE_PATH\"
      --presets \"\${PRESET_ARRAY[@]}\"
      --seeds \"\${SEED_ARRAY[@]}\"
      --out \"$OUT_ROOT\"
      --stage \"$STAGE\"
      --hpo-config-dir \"$HPO_CONFIG_DIR\"
      --curve-step \"$CURVE_STEP\"
      --hpo-seed \"$HPO_SEED\"
      --device \"$DEVICE\"
    )

    if [ \"\${FORCE_RUN:-0}\" = \"1\" ]; then
      CMD+=(--force)
    else
      CMD+=(--resume)
    fi

    printf '[campaign]'
    printf ' %q' \"\${CMD[@]}\"
    printf '\n'
    \"\${CMD[@]}\"
  "

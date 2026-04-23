#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_hawkesnest_suite_family.sh <suite_name> <family>
  scripts/submit_hawkesnest_suite_family.sh <suite_name> custom <preset1> [preset2 ...]

Families:
  neural      -> neural_attncnf neural_jumpcnf neural_cond_gmm
  gen         -> smash diffusion_stpp
  rest        -> auto_stpp deep_stpp nsmpp rmtpp_gmm thp_gmm
  factorized  -> poisson_gmm hawkes_gmm selfcorrecting_gmm poisson_cnf hawkes_cnf selfcorrecting_cnf poisson_tvcnf hawkes_tvcnf selfcorrecting_tvcnf
  custom      -> presets passed explicitly after the family name

Useful environment overrides:
  PARTITION=...
  GPUS=1
  CPUS_PER_TASK=8
  MEM=32G
  TIME_LIMIT=24:00:00
  SEEDS="42 3 555"
  HPO_POLICY=skip-heavy|generative-thin|all-thin
  STAGE=all
  RUN_BATCH_SIZE=...
  DEVICE=cuda|cpu
  SUITE_ROOT=...
  EXCLUDE=serv-3307
  TAG_SUFFIX=...
  CAMPAIGN_TAG=...
  OUT_ROOT=...
  FORCE_RUN=1

Examples:
  scripts/submit_hawkesnest_suite_family.sh suite4_heterogeneity neural
  HPO_POLICY=all-thin TIME_LIMIT=48:00:00 scripts/submit_hawkesnest_suite_family.sh suite4_heterogeneity neural
  HPO_POLICY=generative-thin TIME_LIMIT=48:00:00 scripts/submit_hawkesnest_suite_family.sh suite4_heterogeneity gen
  PARTITION=A100-80GB scripts/submit_hawkesnest_suite_family.sh suite4_heterogeneity gen
  GPUS=0 DEVICE=cpu scripts/submit_hawkesnest_suite_family.sh suite4_heterogeneity factorized
EOF
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

SUITE_NAME="$1"
FAMILY="$2"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$ROOT/scripts/pegasus_hawkesnest_suite_campaign.sbatch"

suite_alias() {
  case "$1" in
    suite1_branching) echo "s1br" ;;
    suite2_training_size) echo "s2size" ;;
    suite3_entanglement) echo "s3ent" ;;
    suite4_heterogeneity) echo "s4het" ;;
    suite5_topology) echo "s5top" ;;
    suite6_corner) echo "s6cor" ;;
    *) echo "$1" ;;
  esac
}

family_alias() {
  case "$1" in
    neural) echo "neural" ;;
    gen|generative) echo "gen" ;;
    rest) echo "rest" ;;
    factorized) echo "fact" ;;
    custom) echo "custom" ;;
    *) echo "$1" ;;
  esac
}

declare -a PRESETS
case "$FAMILY" in
  neural)
    PRESETS=(neural_attncnf neural_jumpcnf neural_cond_gmm)
    ;;
  gen|generative)
    PRESETS=(smash diffusion_stpp)
    FAMILY="gen"
    ;;
  rest)
    PRESETS=(auto_stpp deep_stpp nsmpp rmtpp_gmm thp_gmm)
    ;;
  factorized)
    PRESETS=(
      poisson_gmm hawkes_gmm selfcorrecting_gmm
      poisson_cnf hawkes_cnf selfcorrecting_cnf
      poisson_tvcnf hawkes_tvcnf selfcorrecting_tvcnf
    )
    ;;
  custom)
    if [ "$#" -lt 1 ]; then
      echo "custom family requires at least one preset" >&2
      exit 1
    fi
    PRESETS=("$@")
    ;;
  *)
    echo "Unknown family: $FAMILY" >&2
    usage
    exit 1
    ;;
esac

SUITE_TAG="$(suite_alias "$SUITE_NAME")"
FAMILY_TAG="$(family_alias "$FAMILY")"
if [ "$FAMILY_TAG" = "custom" ] && [ "${#PRESETS[@]}" -eq 1 ]; then
  FAMILY_TAG="${PRESETS[0]}"
fi
TAG_SUFFIX="${TAG_SUFFIX:-$(date +%m%d%H%M)}"
JOB_NAME="${JOB_NAME:-${SUITE_TAG}__${FAMILY_TAG}__${TAG_SUFFIX}}"

SEEDS="${SEEDS:-42 3 555}"
HPO_SEED="${HPO_SEED:-42}"
HPO_CONFIG_DIR="${HPO_CONFIG_DIR:-unified_stpp/configs}"
HPO_POLICY="${HPO_POLICY:-skip-heavy}"
SUITE_ROOT="${SUITE_ROOT:-$ROOT/data/hawkesnest_suitesv2}"
RUN_BATCH_SIZE="${RUN_BATCH_SIZE:-}"
STAGE="${STAGE:-all}"
CURVE_STEP="${CURVE_STEP:-0.1}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
DEVICE="${DEVICE:-}"
if [ -z "$DEVICE" ]; then
  if [ "$GPUS" = "0" ]; then
    DEVICE="cpu"
  else
    DEVICE="cuda"
  fi
fi

CAMPAIGN_TAG="${CAMPAIGN_TAG:-$JOB_NAME}"
OUT_ROOT="${OUT_ROOT:-runs/hawkesnest_campaigns/${SUITE_NAME}/${CAMPAIGN_TAG}}"

declare -a SBATCH_ARGS
SBATCH_ARGS+=(--job-name="$JOB_NAME")
SBATCH_ARGS+=(--gpus="$GPUS")
SBATCH_ARGS+=(--cpus-per-task="$CPUS_PER_TASK")
SBATCH_ARGS+=(--mem="$MEM")
SBATCH_ARGS+=(--time="$TIME_LIMIT")
if [ -n "${PARTITION:-}" ]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi
if [ -n "${EXCLUDE:-}" ]; then
  SBATCH_ARGS+=(--exclude="$EXCLUDE")
fi

mkdir -p "$ROOT/runs/hawkesnest_campaigns"
LEDGER="$ROOT/runs/hawkesnest_campaigns/submissions.csv"
if [ ! -f "$LEDGER" ]; then
  printf 'submitted_at,job_id,job_name,suite,family,presets,seeds,partition,gpus,cpus_per_task,mem,time_limit,stage,device,out_root\n' > "$LEDGER"
fi

printf '[submit] suite=%s family=%s job=%s\n' "$SUITE_NAME" "$FAMILY" "$JOB_NAME"
printf '[submit] presets=%s\n' "${PRESETS[*]}"
printf '[submit] hpo_policy=%s\n' "$HPO_POLICY"
printf '[submit] out_root=%s\n' "$OUT_ROOT"

sbatch_output="$(
  env \
    SEEDS="$SEEDS" \
    HPO_SEED="$HPO_SEED" \
    HPO_CONFIG_DIR="$HPO_CONFIG_DIR" \
    HPO_POLICY="$HPO_POLICY" \
    SUITE_ROOT="$SUITE_ROOT" \
    DEVICE="$DEVICE" \
    RUN_BATCH_SIZE="$RUN_BATCH_SIZE" \
    STAGE="$STAGE" \
    CURVE_STEP="$CURVE_STEP" \
    CAMPAIGN_TAG="$CAMPAIGN_TAG" \
    OUT_ROOT="$OUT_ROOT" \
    FORCE_RUN="${FORCE_RUN:-0}" \
    sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "$SUITE_NAME" "${PRESETS[@]}"
)"

printf '%s\n' "$sbatch_output"
JOB_ID="$(printf '%s\n' "$sbatch_output" | awk '/Submitted batch job/ {print $4}')"
if [ -z "$JOB_ID" ]; then
  echo "Failed to parse job id from sbatch output" >&2
  exit 1
fi

printf '%s,%s,%s,%s,%s,"%s","%s",%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$JOB_ID" \
  "$JOB_NAME" \
  "$SUITE_NAME" \
  "$FAMILY" \
  "${PRESETS[*]}" \
  "$SEEDS" \
  "${PARTITION:-}" \
  "$GPUS" \
  "$CPUS_PER_TASK" \
  "$MEM" \
  "$TIME_LIMIT" \
  "$STAGE" \
  "$DEVICE" \
  "$OUT_ROOT" >> "$LEDGER"

printf '[submit] recorded %s in %s\n' "$JOB_ID" "$LEDGER"

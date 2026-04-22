#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_realdata_family.sh <dataset> <family>
  scripts/submit_realdata_family.sh <dataset> custom <preset1> [preset2 ...]

Datasets:
  pass a full HF dataset id (e.g. yahya021/covid-stpp), or one of:
    covid        -> yahya021/covid-stpp
    earthquake   -> yahya021/earthquakes-stpp
    bold         -> yahya021/bold5000-stpp
    citibike     -> yahya021/citibike-stpp

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
  N_WORKERS=1
  USE_HPO=0|1        (default: 0)
  RUN_ROOT=/home/aalaila/projects/uni-stpp/runs/exp1
  DATASET_REVISION=...
  OVERRIDES="training.device=cuda data.num_workers=0"
  HPO_CONFIGS_DIR=...
  CAMPAIGN_ID=...
  GROUP_NAME=...
  BENCH_OUT=...
  FORCE_RUN=1

Examples:
  scripts/submit_realdata_family.sh covid neural
  PARTITION=A100-80GB scripts/submit_realdata_family.sh yahya021/earthquakes-stpp gen
  USE_HPO=1 scripts/submit_realdata_family.sh covid rest
  GPUS=0 OVERRIDES=\"training.device=cpu\" scripts/submit_realdata_family.sh citibike factorized
EOF
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

DATASET_INPUT="$1"
FAMILY="$2"
shift 2

dataset_id() {
  case "$1" in
    covid) echo "yahya021/covid-stpp" ;;
    earthquake) echo "yahya021/earthquakes-stpp" ;;
    bold) echo "yahya021/bold5000-stpp" ;;
    citibike) echo "yahya021/citibike-stpp" ;;
    *) echo "$1" ;;
  esac
}

dataset_slug() {
  local ds="$1"
  printf '%s\n' "${ds##*/}"
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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$ROOT/scripts/pegasus_bench_group.sbatch"
DATASET="$(dataset_id "$DATASET_INPUT")"
DATASET_SLUG="$(dataset_slug "$DATASET")"
FAMILY_TAG="$(family_alias "$FAMILY")"
TAG_SUFFIX="${TAG_SUFFIX:-$(date +%m%d%H%M)}"
JOB_NAME="${JOB_NAME:-${DATASET_SLUG}__${FAMILY_TAG}__${TAG_SUFFIX}}"

GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
SEEDS="${SEEDS:-42 3 555}"
N_WORKERS="${N_WORKERS:-1}"
USE_HPO="${USE_HPO:-0}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/exp1}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$DATASET_SLUG}"
GROUP_NAME="${GROUP_NAME:-$JOB_NAME}"
BENCH_OUT="${BENCH_OUT:-$RUN_ROOT/${DATASET_SLUG}/bench/${GROUP_NAME}}"
HPO_CONFIGS_DIR="${HPO_CONFIGS_DIR:-$RUN_ROOT/${DATASET_SLUG}/tune}"
REPO_ROOT="${REPO_ROOT:-$ROOT}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$REPO_ROOT}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$REPO_ROOT:$REPO_ROOT,/netscratch/aalaila:/netscratch/aalaila}"

declare -a SBATCH_ARGS
SBATCH_ARGS+=(--job-name="$JOB_NAME")
SBATCH_ARGS+=(--gpus="$GPUS")
SBATCH_ARGS+=(--cpus-per-task="$CPUS_PER_TASK")
SBATCH_ARGS+=(--mem="$MEM")
SBATCH_ARGS+=(--time="$TIME_LIMIT")
SBATCH_ARGS+=(--output="$ROOT/logs/%x_%j.out")
if [ -n "${PARTITION:-}" ]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

mkdir -p "$ROOT/logs" "$RUN_ROOT/${DATASET_SLUG}/bench"
LEDGER="$RUN_ROOT/${DATASET_SLUG}/bench/submissions.csv"
if [ ! -f "$LEDGER" ]; then
  printf 'submitted_at,job_id,job_name,dataset,family,presets,seeds,partition,gpus,cpus_per_task,mem,time_limit,bench_out,hpo_configs_dir\n' > "$LEDGER"
fi

printf '[submit] dataset=%s family=%s job=%s\n' "$DATASET" "$FAMILY" "$JOB_NAME"
printf '[submit] presets=%s\n' "${PRESETS[*]}"
printf '[submit] bench_out=%s\n' "$BENCH_OUT"
printf '[submit] use_hpo=%s\n' "$USE_HPO"

sbatch_output="$(
  env \
    REPO_ROOT="$REPO_ROOT" \
    RUN_ROOT="$RUN_ROOT" \
    DATA_ROOT="$DATA_ROOT" \
    CAMPAIGN_ID="$CAMPAIGN_ID" \
    GROUP_NAME="$GROUP_NAME" \
    PRESETS="${PRESETS[*]}" \
    DATASET="$DATASET" \
    DATASET_REVISION="${DATASET_REVISION:-}" \
    SEEDS="$SEEDS" \
    N_WORKERS="$N_WORKERS" \
    USE_HPO="$USE_HPO" \
    HPO_CONFIGS_DIR="$HPO_CONFIGS_DIR" \
    BENCH_OUT="$BENCH_OUT" \
    OVERRIDES="${OVERRIDES:-}" \
    FORCE_RUN="${FORCE_RUN:-0}" \
    CONTAINER_IMAGE="$CONTAINER_IMAGE" \
    CONTAINER_WORKDIR="$CONTAINER_WORKDIR" \
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS" \
    sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT"
)"

printf '%s\n' "$sbatch_output"
JOB_ID="$(printf '%s\n' "$sbatch_output" | awk '/Submitted batch job/ {print $4}')"
if [ -z "$JOB_ID" ]; then
  echo "Failed to parse job id from sbatch output" >&2
  exit 1
fi

printf '%s,%s,%s,%s,%s,"%s","%s",%s,%s,%s,%s,%s,%s,%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$JOB_ID" \
  "$JOB_NAME" \
  "$DATASET" \
  "$FAMILY" \
  "${PRESETS[*]}" \
  "$SEEDS" \
  "${PARTITION:-}" \
  "$GPUS" \
  "$CPUS_PER_TASK" \
  "$MEM" \
  "$TIME_LIMIT" \
  "$BENCH_OUT" \
  "$HPO_CONFIGS_DIR" >> "$LEDGER"

printf '[submit] recorded %s in %s\n' "$JOB_ID" "$LEDGER"

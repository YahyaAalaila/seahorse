#!/usr/bin/env bash
set -euo pipefail

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
}

campaign_root() {
  require_var RUN_ROOT
  require_var CAMPAIGN_ID
  printf '%s\n' "${RUN_ROOT%/}/${CAMPAIGN_ID}"
}

campaign_logs_dir() {
  printf '%s\n' "$(campaign_root)/logs"
}

campaign_hpo_dir() {
  printf '%s\n' "$(campaign_root)/hpo"
}

campaign_bench_dir() {
  local group_name="${1:?group_name is required}"
  printf '%s\n' "$(campaign_root)/bench/${group_name}"
}

job_complete() {
  local root="$1"
  shift
  local rel
  for rel in "$@"; do
    if [[ ! -e "${root%/}/${rel}" ]]; then
      return 1
    fi
  done
  return 0
}

run_cmd() {
  local -a cmd=( "$@" )
  if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    local -a srun_cmd=( srun "--container-image=${CONTAINER_IMAGE}" )
    if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
      srun_cmd+=( "--container-mounts=${CONTAINER_MOUNTS}" )
    fi
    if [[ -n "${CONTAINER_WORKDIR:-}" ]]; then
      srun_cmd+=( "--container-workdir=${CONTAINER_WORKDIR}" )
    fi
    "${srun_cmd[@]}" "${cmd[@]}"
    return
  fi
  "${cmd[@]}"
}

print_context() {
  echo "[campaign] REPO_ROOT=${REPO_ROOT:-}"
  echo "[campaign] RUN_ROOT=${RUN_ROOT:-}"
  echo "[campaign] DATA_ROOT=${DATA_ROOT:-}"
  echo "[campaign] CAMPAIGN_ID=${CAMPAIGN_ID:-}"
  echo "[campaign] CONTAINER_IMAGE=${CONTAINER_IMAGE:-<none>}"
}

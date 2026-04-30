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
    local quoted_cmd
    printf -v quoted_cmd '%q ' "${cmd[@]}"
    local setup_cmd=""
    if [[ "${INSTALL_PROJECT:-1}" == "1" ]]; then
      local python_bin="${UNIFIED_STPP_PYTHON:-python}"
      local quoted_root
      local quoted_python
      printf -v quoted_root '%q' "${REPO_ROOT}"
      printf -v quoted_python '%q' "${python_bin}"
      setup_cmd="cd ${quoted_root} && ${quoted_python} -m pip install -e '.[hpo]' && "
    fi
    if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
      srun_cmd+=( "--container-mounts=${CONTAINER_MOUNTS}" )
    fi
    if [[ -n "${CONTAINER_WORKDIR:-}" ]]; then
      srun_cmd+=( "--container-workdir=${CONTAINER_WORKDIR}" )
    fi
    "${srun_cmd[@]}" bash -lc "${setup_cmd}${quoted_cmd}"
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

write_job_summary() {
  local exit_code="$1"
  if [[ "${UNIFIED_STPP_JOB_NOTIFY:-1}" == "0" ]]; then
    return 0
  fi
  if [[ -z "${REPO_ROOT:-}" || ! -f "${REPO_ROOT}/scripts/slurm_job_summary.py" ]]; then
    return 0
  fi
  local python_bin="${UNIFIED_STPP_NOTIFY_PYTHON:-python3}"
  set +e
  "${python_bin}" "${REPO_ROOT}/scripts/slurm_job_summary.py" --exit-code "${exit_code}"
  set -e
  return 0
}

register_job_summary_trap() {
  export UNIFIED_STPP_JOB_PIPELINE="${1:-unknown}"
  if [[ $# -ge 2 && -n "${2:-}" ]]; then
    export UNIFIED_STPP_JOB_RESULT_ROOT="$2"
  fi
  __USTPP_JOB_SUMMARY_WRITTEN=0
  _ustpp_write_job_summary_once() {
    local exit_code="$1"
    if [[ "${__USTPP_JOB_SUMMARY_WRITTEN:-0}" == "1" ]]; then
      return 0
    fi
    __USTPP_JOB_SUMMARY_WRITTEN=1
    write_job_summary "${exit_code}"
  }
  trap 'ustpp_exit_code=$?; _ustpp_write_job_summary_once "${ustpp_exit_code}"; exit "${ustpp_exit_code}"' EXIT
  trap '_ustpp_write_job_summary_once 143; trap - TERM; exit 143' TERM
  trap '_ustpp_write_job_summary_once 130; trap - INT; exit 130' INT
}

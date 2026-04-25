#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/print_hawkesnest_easy_hard_submit_commands.sh [suite1 suite2 ...]

Print copy-pasteable cluster submission commands for the HawkesNest easy-hard
suite folders under data/hawkesnest_easy_hard.

Defaults:
  suites: combined sweep_E sweep_H

The commands reuse scripts/submit_hawkesnest_suite_family.sh and are intended
to be run one by one on the cluster, mirroring the v2 workflow.

Environment knobs baked into the printed commands:
  SUITE_ROOT=/home/aalaila/projects/uni-stpp/data/hawkesnest_easy_hard
  SUITE_DATA_TAG=easyhard
  EXCLUDE=serv-3307

Family resources:
  factorized: CPU, 24h
  rest:       CPU, 24h
  gen:        GPU, 48h
  neural:     GPU, 72h

Example:
  scripts/print_hawkesnest_easy_hard_submit_commands.sh
  scripts/print_hawkesnest_easy_hard_submit_commands.sh combined sweep_E
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE_ROOT_CLUSTER="/home/aalaila/projects/uni-stpp/data/hawkesnest_easy_hard"
SUBMIT_SCRIPT="scripts/submit_hawkesnest_suite_family.sh"

if [ "$#" -gt 0 ]; then
  SUITES=("$@")
else
  SUITES=(combined sweep_E sweep_H)
fi

for suite in "${SUITES[@]}"; do
  if [ ! -d "$ROOT/data/hawkesnest_easy_hard/$suite/jsonl" ]; then
    echo "# skipping unknown suite: $suite" >&2
    continue
  fi

  echo "# $suite"
  echo "cd ~/projects/uni-stpp"
  echo "SUITE_ROOT=$SUITE_ROOT_CLUSTER SUITE_DATA_TAG=easyhard GPUS=0 DEVICE=cpu TIME_LIMIT=24:00:00 EXCLUDE=serv-3307 $SUBMIT_SCRIPT $suite factorized"
  echo "SUITE_ROOT=$SUITE_ROOT_CLUSTER SUITE_DATA_TAG=easyhard GPUS=0 DEVICE=cpu TIME_LIMIT=24:00:00 EXCLUDE=serv-3307 $SUBMIT_SCRIPT $suite rest"
  echo "SUITE_ROOT=$SUITE_ROOT_CLUSTER SUITE_DATA_TAG=easyhard GPUS=1 DEVICE=cuda TIME_LIMIT=48:00:00 EXCLUDE=serv-3307 $SUBMIT_SCRIPT $suite gen"
  echo "SUITE_ROOT=$SUITE_ROOT_CLUSTER SUITE_DATA_TAG=easyhard GPUS=1 DEVICE=cuda TIME_LIMIT=72:00:00 EXCLUDE=serv-3307 $SUBMIT_SCRIPT $suite neural"
  echo
done

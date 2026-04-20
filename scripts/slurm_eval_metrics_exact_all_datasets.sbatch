#!/bin/bash
#SBATCH --job-name=eval_metrics_exact_all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_metrics_exact_all_%j.out
#SBATCH --error=logs/eval_metrics_exact_all_%j.err

# Usage:
#   sbatch ... slurm_eval_metrics_exact_all_datasets.sbatch                                    # all 24
#   sbatch ... slurm_eval_metrics_exact_all_datasets.sbatch echo                              # family: echo_E0..E5
#   sbatch ... slurm_eval_metrics_exact_all_datasets.sbatch pulse                             # family: pulse_P0..P5
#   sbatch ... slurm_eval_metrics_exact_all_datasets.sbatch regime                            # family: regime_R0..R5
#   sbatch ... slurm_eval_metrics_exact_all_datasets.sbatch topology                          # family: topology_T0..T5
#   sbatch ... slurm_eval_metrics_exact_all_datasets.sbatch echo_E2 20260410_213356_b80c1ebf  # single dataset

ARG1="${1:-}"
ARG2="${2:-}"

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh \
  --container-mounts=/home/aalaila:/home/aalaila \
  --container-workdir=/home/aalaila/projects/uni-stpp \
  bash -lc '
    set -e
    cd /home/aalaila/projects/uni-stpp

    ARG1="$1"
    ARG2="$2"

    python -m pip install --upgrade pip --quiet &&
    python -m pip install -r requirements.txt --quiet &&
    python -m pip install "pydantic>=2.0" "pandas>=2.0" "jinja2>=3.0" "joblib>=1.3" "ray[tune]>=2.0" --quiet

    BENCH=runs/bch_hawkes_famv1_5seeds_auto_stpp/fit/auto_stpp
    DATA=data/exports/unified_stpp_hawkesnest_hard_v2
    ARTIFACTS=runs/eval_metrics_exact_all/artifacts
    OUT=runs/eval_metrics_exact_all

    run_dataset() {
        local dataset="$1"
        local run_id="$2"

        echo ""
        echo "============================================================"
        echo "DATASET: ${dataset}   RUN: ${run_id}"
        echo "============================================================"

        echo "--- core ---"
        python -m unified_stpp evaluate metrics \
            --run "${BENCH}/${run_id}" \
            --data "${DATA}/${dataset}/test.jsonl" \
            --split test \
            --metric-profile core \
            --exact-spatial-bins 8 \
            --exact-time-bins 8 \
            --artifact-mode load_or_compute \
            --artifact-dir "${ARTIFACTS}/${dataset}" \
            --device cpu \
            --benchmark-id exact_all_datasets \
            --out "${OUT}/${dataset}_core"

        echo "--- predictive (compute) ---"
        time python -m unified_stpp evaluate metrics \
            --run "${BENCH}/${run_id}" \
            --data "${DATA}/${dataset}/test.jsonl" \
            --split test \
            --metric-profile predictive \
            --k-pred 32 \
            --exact-spatial-bins 8 \
            --exact-time-bins 8 \
            --artifact-mode load_or_compute \
            --artifact-dir "${ARTIFACTS}/${dataset}" \
            --device cpu \
            --benchmark-id exact_all_datasets \
            --out "${OUT}/${dataset}_predictive_compute"

        echo "--- predictive (load-only) ---"
        time python -m unified_stpp evaluate metrics \
            --run "${BENCH}/${run_id}" \
            --data "${DATA}/${dataset}/test.jsonl" \
            --split test \
            --metric-profile predictive \
            --k-pred 32 \
            --exact-spatial-bins 8 \
            --exact-time-bins 8 \
            --artifact-mode load_only \
            --artifact-dir "${ARTIFACTS}/${dataset}" \
            --device cpu \
            --benchmark-id exact_all_datasets \
            --out "${OUT}/${dataset}_predictive_load"
    }

    run_echo()     {
        run_dataset echo_E0 20260410_213027_b80c1ebf
        run_dataset echo_E1 20260410_213212_b80c1ebf
        run_dataset echo_E2 20260410_213356_b80c1ebf
        run_dataset echo_E3 20260410_213543_b80c1ebf
        run_dataset echo_E4 20260410_213737_b80c1ebf
        run_dataset echo_E5 20260410_213941_b80c1ebf
    }

    run_pulse()    {
        run_dataset pulse_P0 20260410_214152_b80c1ebf
        run_dataset pulse_P1 20260410_214338_b80c1ebf
        run_dataset pulse_P2 20260410_214525_b80c1ebf
        run_dataset pulse_P3 20260410_214711_b80c1ebf
        run_dataset pulse_P4 20260410_214856_b80c1ebf
        run_dataset pulse_P5 20260410_215040_b80c1ebf
    }

    run_regime()   {
        run_dataset regime_R0 20260410_215225_b80c1ebf
        run_dataset regime_R1 20260410_215411_b80c1ebf
        run_dataset regime_R2 20260410_215603_b80c1ebf
        run_dataset regime_R3 20260410_215756_b80c1ebf
        run_dataset regime_R4 20260410_215953_b80c1ebf
        run_dataset regime_R5 20260410_220155_b80c1ebf
    }

    run_topology() {
        run_dataset topology_T0 20260410_220400_b80c1ebf
        run_dataset topology_T1 20260410_220550_b80c1ebf
        run_dataset topology_T2 20260410_220736_b80c1ebf
        run_dataset topology_T3 20260410_220923_b80c1ebf
        run_dataset topology_T4 20260410_221110_b80c1ebf
        run_dataset topology_T5 20260410_221257_b80c1ebf
    }

    if [ -z "$ARG1" ]; then
        echo "=== Running all 24 datasets ==="
        run_echo && run_pulse && run_regime && run_topology
        echo "=== All 24 datasets complete ==="
    elif [ "$ARG1" = "echo" ];     then run_echo
    elif [ "$ARG1" = "pulse" ];    then run_pulse
    elif [ "$ARG1" = "regime" ];   then run_regime
    elif [ "$ARG1" = "topology" ]; then run_topology
    else
        # single dataset: ARG1=dataset ARG2=run_id
        if [ -z "$ARG2" ]; then
            echo "ERROR: single-dataset mode requires both dataset and run_id" >&2
            echo "Usage: sbatch ... slurm_eval_metrics_exact_all_datasets.sbatch <dataset> <run_id>" >&2
            exit 1
        fi
        run_dataset "$ARG1" "$ARG2"
    fi
  ' -- "$ARG1" "$ARG2"

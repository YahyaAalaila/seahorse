#!/bin/bash
#SBATCH --job-name=eval_metrics_batched
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_metrics_batched_%j.out
#SBATCH --error=logs/eval_metrics_batched_%j.err

# Validates the batched-thinning performance-engineering changes:
#   - _build_exact_proposal_cache: one spatial-grid batch per sampled time slice
#   - _thinning_k_chains_batched: K serial chains -> 1 batched call per round
#   - New defaults: time_bins=8, spatial_bins=8, k_pred=32
#
# Three passes: core (no sampling), predictive (compute artifacts), predictive (load artifacts).
# The load-only pass must complete in seconds -- proves artifact caching works end-to-end.

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh \
  --container-mounts=/home/aalaila:/home/aalaila \
  --container-workdir=/home/aalaila/projects/uni-stpp \
  bash -lc '
    set -e

    cd /home/aalaila/projects/uni-stpp

    python -m pip install --upgrade pip --quiet &&
    python -m pip install -r requirements.txt --quiet &&
    python -m pip install "pydantic>=2.0" "pandas>=2.0" "jinja2>=3.0" "joblib>=1.3" "ray[tune]>=2.0" --quiet &&

    echo "=== Pass 1: core (no sampling) ===" &&
    time python -m unified_stpp evaluate metrics \
        --run runs/bch_hawkes_famv1_5seeds_auto_stpp/fit/auto_stpp/20260410_213027_b80c1ebf \
        --data data/exports/unified_stpp_hawkesnest_hard_v2/echo_E0/test.jsonl \
        --split test \
        --metric-profile core \
        --max-seqs 10 \
        --max-events 20 \
        --artifact-mode load_or_compute \
        --artifact-dir runs/eval_metrics_batched_validation/artifacts/auto_stpp_echo_E0 \
        --device cpu \
        --benchmark-id batched_thinning_validation \
        --out runs/eval_metrics_batched_validation/auto_stpp_echo_E0_core &&

    echo "=== Pass 2: predictive (compute + cache artifacts) ===" &&
    time python -m unified_stpp evaluate metrics \
        --run runs/bch_hawkes_famv1_5seeds_auto_stpp/fit/auto_stpp/20260410_213027_b80c1ebf \
        --data data/exports/unified_stpp_hawkesnest_hard_v2/echo_E0/test.jsonl \
        --split test \
        --metric-profile predictive \
        --max-seqs 10 \
        --max-events 20 \
        --artifact-mode load_or_compute \
        --artifact-dir runs/eval_metrics_batched_validation/artifacts/auto_stpp_echo_E0 \
        --device cpu \
        --benchmark-id batched_thinning_validation \
        --out runs/eval_metrics_batched_validation/auto_stpp_echo_E0_predictive_compute &&

    echo "=== Pass 3: predictive (load-only, must be fast) ===" &&
    time python -m unified_stpp evaluate metrics \
        --run runs/bch_hawkes_famv1_5seeds_auto_stpp/fit/auto_stpp/20260410_213027_b80c1ebf \
        --data data/exports/unified_stpp_hawkesnest_hard_v2/echo_E0/test.jsonl \
        --split test \
        --metric-profile predictive \
        --max-seqs 10 \
        --max-events 20 \
        --artifact-mode load_only \
        --artifact-dir runs/eval_metrics_batched_validation/artifacts/auto_stpp_echo_E0 \
        --device cpu \
        --benchmark-id batched_thinning_validation \
        --out runs/eval_metrics_batched_validation/auto_stpp_echo_E0_predictive_load

    echo "=== All passes complete ==="
  '

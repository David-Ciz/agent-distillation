#!/usr/bin/env bash
# =============================================================================
# submit_tta_sweep.sh — Submit one TTA job per model (Phase A sweep)
#
# Each job runs 07_tta_experiment.py for one model with the configured N
# values, temperatures and aggregation methods.
#
# Usage:
#   bash task1/scripts/slurm/submit_tta_sweep.sh
#   bash task1/scripts/slurm/submit_tta_sweep.sh --dry-run
#   bash task1/scripts/slurm/submit_tta_sweep.sh --phase a   # default
#   bash task1/scripts/slurm/submit_tta_sweep.sh --phase b   # temp sweep
#
# After all jobs finish:
#   1. Sync results locally:
#        bash task1/scripts/slurm/sync_results.sh
#   2. Inspect comparison CSV:
#        open task1/outputs/evaluations/tta_run_<timestamp>/tta_comparison_summary.csv
#   3. View MLflow UI:
#        mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
TTA_SCRIPT="${SCRIPT_DIR}/tta_lumi.sh"

DRY_RUN=false
PHASE="a"

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --phase)   shift; PHASE="${1:-a}" ;;
        --phase=*) PHASE="${arg#--phase=}" ;;
    esac
done

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------
case "${PHASE,,}" in
    a)
        # Phase A — quick signal: smallest/most-over-abstaining models
        # N ∈ {1,3,5}, T=0.7, all aggregations
        N_VALUES="1,3,5"
        TEMPS="0.7"
        AGGS="majority_vote,centroid,oracle"
        WALL_TIME="04:00:00"

        declare -a MODELS=(
            "${REPO_DIR}/task1/outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32"
            "${REPO_DIR}/task1/outputs/google_gemma-3-1b-it-lora-final,gemma-3-1b-it-lora,lora,32"
        )
        ;;
    b)
        # Phase B — temperature sweep (run only after Phase A shows improvement)
        N_VALUES="1,3,5"
        TEMPS="0.5,1.0"
        AGGS="majority_vote,centroid,oracle"
        WALL_TIME="06:00:00"

        declare -a MODELS=(
            "${REPO_DIR}/task1/outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32"
        )
        ;;
    c)
        # Phase C — scaling check (does TTA help larger models?)
        N_VALUES="1,5"
        TEMPS="0.7"
        AGGS="majority_vote,centroid,oracle"
        WALL_TIME="04:00:00"

        declare -a MODELS=(
            "${REPO_DIR}/task1/outputs/Qwen_Qwen2.5-1.5B-Instruct-lora-final,Qwen2.5-1.5B-Instruct-lora,lora,32"
        )
        ;;
    sanity)
        # Sanity check: N=1, T=0 must reproduce 05 baseline ±0.002 F1
        N_VALUES="1"
        TEMPS="0.0"
        AGGS="majority_vote"
        WALL_TIME="02:00:00"

        declare -a MODELS=(
            "${REPO_DIR}/task1/outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32"
        )
        ;;
    *)
        echo "Unknown phase: ${PHASE}. Use: a, b, c, sanity" >&2
        exit 1
        ;;
esac

echo "=================================================="
echo "TTA sweep — Phase ${PHASE^^} — $(date)"
echo "N values:     ${N_VALUES}"
echo "Temperatures: ${TEMPS}"
echo "Aggregations: ${AGGS}"
echo "Wall time:    ${WALL_TIME}"
echo "Models to submit: ${#MODELS[@]}"
echo "=================================================="
for m in "${MODELS[@]}"; do
    echo "  ${m%%,*}  (${m#*,})"
done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would submit ${#MODELS[@]} TTA jobs."
    exit 0
fi

# ---------------------------------------------------------------------------
# Submit one job per model
# ---------------------------------------------------------------------------
for model_cfg in "${MODELS[@]}"; do
    model_name="${model_cfg%%,*}"
    model_name="${model_name##*/}"   # basename

    JOB_ID=$(TTA_MODELS="--models '${model_cfg}'" \
              TTA_N_VALUES="${N_VALUES}" \
              TTA_TEMPS="${TEMPS}" \
              TTA_AGGS="${AGGS}" \
              sbatch --parsable \
                  --job-name="tta-${model_name:0:20}" \
                  --time="${WALL_TIME}" \
                  --export=ALL \
                  "${TTA_SCRIPT}")

    echo "Submitted job ${JOB_ID} for model: ${model_name}"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ~/agent-distillation/logs/tta_<JOB_ID>.out"
echo ""
echo "After all jobs finish, sync results:"
echo "  bash task1/scripts/slurm/sync_results.sh"


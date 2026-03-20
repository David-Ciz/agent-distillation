#!/usr/bin/env bash
# =============================================================================
# submit_qwen35_train_sweep.sh — Submit the Qwen3.5 LoRA training sweep
#
# Uses the standard / slow-path environment for the whole Qwen3.5 family to
# keep training and evaluation workflows uniform.
#
# Usage:
#   bash task1/scripts/slurm/submit_qwen35_train_sweep.sh
#   bash task1/scripts/slurm/submit_qwen35_train_sweep.sh --dry-run
#
# Format: "MODEL_NAME BATCH_SIZE GRAD_ACCUM TIME_LIMIT"
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_lora_lumi.sh"
QWEN35_OVERRIDES="${QWEN35_OVERRIDES:-${HOME}/agent-distillation/py-overrides}"
SBATCH_EXPORT="ALL,CONTAINER_PYTHON_OVERRIDES=${QWEN35_OVERRIDES}"
QWEN35_ATTN_IMPL="${QWEN35_ATTN_IMPL:-sdpa}"
SBATCH_EXPORT+=",TRAIN_ATTN_IMPL=${QWEN35_ATTN_IMPL}"
if [[ -n "${CONTAINER_VENV:-}" ]]; then
    SBATCH_EXPORT+=",CONTAINER_VENV=${CONTAINER_VENV}"
fi

# ---------------------------------------------------------------------------
# Qwen3.5 models to train in the requested order using the standard override
# path. These are the post-trained checkpoints, not the -Base variants.
# ---------------------------------------------------------------------------
MODELS=(
    "Qwen/Qwen3.5-0.8B       4  2   10:00:00"
    "Qwen/Qwen3.5-2B         2  4   16:00:00"
    "Qwen/Qwen3.5-4B         1  8   24:00:00"
    "Qwen/Qwen3.5-9B         1  16  36:00:00"
)

echo "=================================================="
echo "Qwen3.5 training sweep — $(date)"
echo "Overrides path: $QWEN35_OVERRIDES"
echo "Attention impl: $QWEN35_ATTN_IMPL"
echo "=================================================="

SUBMITTED=()

for entry in "${MODELS[@]}"; do
    read -r model batch accum timelimit <<< "$entry"

    job_name="lora-$(echo "$model" | sed 's|.*/||')"

    echo ""
    echo "Model:      $model"
    echo "Batch:      $batch  GradAccum: $accum"
    echo "Time limit: $timelimit"

    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: MODEL_NAME=$model BATCH_SIZE=$batch GRAD_ACCUM=$accum"
        echo "[DRY RUN] sbatch --export=$SBATCH_EXPORT ..."
    else
        JOB_ID=$(MODEL_NAME="$model" \
                 BATCH_SIZE="$batch" \
                 GRAD_ACCUM="$accum" \
                 sbatch --time="$timelimit" \
                        --export="$SBATCH_EXPORT" \
                        --job-name="$job_name" \
                        --parsable \
                        "$TRAIN_SCRIPT")
        echo "Submitted job: $JOB_ID"
        SUBMITTED+=("$JOB_ID:$model")
    fi
done

echo ""
echo "=================================================="
if $DRY_RUN; then
    echo "Dry run complete — no jobs submitted."
else
    echo "Submitted ${#SUBMITTED[@]} jobs:"
    for entry in "${SUBMITTED[@]}"; do
        echo "  $entry"
    done
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "After all finish, run: bash task1/scripts/slurm/submit_qwen35_eval_sweep.sh"
fi
echo "=================================================="

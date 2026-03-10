#!/usr/bin/env bash
# =============================================================================
# submit_train_sweep.sh — Submit a full training sweep across models
#
# Submits one sbatch job per model. Each job runs train_lora_lumi.sh with
# the model's hyperparameters passed as environment variables.
#
# Usage:
#   bash task1/scripts/slurm/submit_train_sweep.sh
#   bash task1/scripts/slurm/submit_train_sweep.sh --dry-run   # print without submitting
#
# To add a model: add a line to the MODELS array below.
# Format: "MODEL_NAME BATCH_SIZE GRAD_ACCUM TIME_LIMIT"
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_lora_lumi.sh"

# ---------------------------------------------------------------------------
# Models to train — edit this list for your sweep
# Columns: MODEL_NAME  BATCH_SIZE  GRAD_ACCUM  TIME_LIMIT
# ---------------------------------------------------------------------------
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct  4  2  02:00:00"
    "Qwen/Qwen2.5-1.5B-Instruct  2  4  03:00:00"
    "Qwen/Qwen2.5-3B-Instruct    2  4  04:00:00"
    "Qwen/Qwen2.5-7B-Instruct    1  8  08:00:00"
)

echo "=================================================="
echo "Training sweep — $(date)"
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
    else
        JOB_ID=$(MODEL_NAME="$model" \
                 BATCH_SIZE="$batch" \
                 GRAD_ACCUM="$accum" \
                 sbatch --time="$timelimit" \
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
    echo "After all finish, run: bash task1/scripts/slurm/submit_eval_sweep.sh"
fi
echo "=================================================="


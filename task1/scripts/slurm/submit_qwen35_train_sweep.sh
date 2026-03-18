#!/usr/bin/env bash
# =============================================================================
# submit_qwen35_train_sweep.sh — Submit the Qwen3.5 LoRA training sweep
#
# Keeps the baseline Qwen2.5 sweep stable while adding a dedicated Phase 6
# sweep for the selected Qwen3.5 checkpoints.
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

# ---------------------------------------------------------------------------
# Qwen3.5 models to train
# Conservative first-pass settings to minimize OOM risk on the first sweep.
# ---------------------------------------------------------------------------
MODELS=(
    "Qwen/Qwen3.5-0.8B  4  2  06:00:00"
    "Qwen/Qwen3.5-2B    2  4  12:00:00"
    "Qwen/Qwen3.5-4B    1  8  20:00:00"
)

echo "=================================================="
echo "Qwen3.5 training sweep — $(date)"
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
    echo "After all finish, run: bash task1/scripts/slurm/submit_qwen35_eval_sweep.sh"
fi
echo "=================================================="

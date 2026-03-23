#!/usr/bin/env bash
# =============================================================================
# submit_eval_sweep.sh — Submit the Qwen2.5 LoRA evaluation sweep
#
# Evaluates the trained Qwen2.5 LoRA final adapters in a single evaluation job.
# This wrapper delegates to eval_lumi.sh so the same container/env path used by
# training is also used for evaluation.
#
# Usage:
#   bash task1/scripts/slurm/submit_eval_sweep.sh
#   bash task1/scripts/slurm/submit_eval_sweep.sh --dry-run
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_lumi.sh"
SBATCH_EXPORT="ALL"

if [[ -n "${CONTAINER_PYTHON_OVERRIDES:-}" ]]; then
    SBATCH_EXPORT+=",CONTAINER_PYTHON_OVERRIDES=${CONTAINER_PYTHON_OVERRIDES}"
fi
if [[ -n "${CONTAINER_VENV:-}" ]]; then
    SBATCH_EXPORT+=",CONTAINER_VENV=${CONTAINER_VENV}"
fi

# Model weights live on scratch, not home
SCRATCH_DIR="/scratch/project_465002758/${USER:-daciz}/agent-distillation/task1/outputs"

# ---------------------------------------------------------------------------
# Qwen2.5 LoRA finals to evaluate
# Format per entry: 'ADAPTER_PATH_OR_HF_ID,NAME,TYPE,GEN_BATCH'
# TYPE: lora | full_finetune | base
# ---------------------------------------------------------------------------
MODELS_ARGS=(
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-1.5B-Instruct-lora-final,Qwen2.5-1.5B-Instruct-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-7B-Instruct-lora-final,Qwen2.5-7B-Instruct-lora,lora,16'"
)

MODELS_STR="${MODELS_ARGS[*]}"

echo "=================================================="
echo "Qwen2.5 evaluation sweep — $(date)"
echo "Models to evaluate: ${#MODELS_ARGS[@]}"
echo "=================================================="
for m in "${MODELS_ARGS[@]}"; do echo "  $m"; done
if [[ -n "${CONTAINER_PYTHON_OVERRIDES:-}" ]]; then
    echo "Python overrides: $CONTAINER_PYTHON_OVERRIDES"
fi
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would submit eval job with the above models."
    exit 0
fi

JOB_ID=$(EVAL_MODELS="$MODELS_STR" \
    sbatch --parsable \
        --job-name="agent-distill-qwen25-eval" \
        --time=08:00:00 \
        --export="$SBATCH_EXPORT" \
        "$EVAL_SCRIPT")

echo "Submitted eval sweep job: $JOB_ID"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ~/agent-distillation/logs/eval_${JOB_ID}.out"
echo ""
echo "After completion, sync results locally:"
echo "  bash task1/scripts/slurm/sync_results.sh"

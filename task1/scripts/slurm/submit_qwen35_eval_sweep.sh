#!/usr/bin/env bash
# =============================================================================
# submit_qwen35_eval_sweep.sh — Submit the Qwen3.5 evaluation sweep
#
# Evaluates the trained Qwen3.5 LoRA final adapters in a single evaluation job.
# This wrapper uses the same container override path as the Qwen3.5 training
# sweep so evaluation runs under the same Python environment family.
#
# Usage:
#   bash task1/scripts/slurm/submit_qwen35_eval_sweep.sh
#   bash task1/scripts/slurm/submit_qwen35_eval_sweep.sh --dry-run
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_lumi.sh"
QWEN35_OVERRIDES="${QWEN35_OVERRIDES:-${HOME}/agent-distillation/py-overrides}"
SBATCH_EXPORT="ALL,CONTAINER_PYTHON_OVERRIDES=${QWEN35_OVERRIDES}"

if [[ -n "${CONTAINER_VENV:-}" ]]; then
    SBATCH_EXPORT+=",CONTAINER_VENV=${CONTAINER_VENV}"
fi

# Model weights live on scratch, not home
SCRATCH_DIR="/scratch/project_465002758/${USER:-daciz}/agent-distillation/task1/outputs"

# ---------------------------------------------------------------------------
# Qwen3.5 LoRA finals to evaluate
# Format per entry: 'ADAPTER_PATH_OR_HF_ID,NAME,TYPE,GEN_BATCH'
# TYPE: lora | full_finetune | base
# ---------------------------------------------------------------------------
MODELS_ARGS=(
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-0.8B-lora-final,Qwen3.5-0.8B-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-2B-lora-final,Qwen3.5-2B-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-4B-lora-final,Qwen3.5-4B-lora,lora,16'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-9B-lora-final,Qwen3.5-9B-lora,lora,8'"
)

MODELS_STR="${MODELS_ARGS[*]}"

echo "=================================================="
echo "Qwen3.5 evaluation sweep — $(date)"
echo "Models to evaluate: ${#MODELS_ARGS[@]}"
echo "=================================================="
for m in "${MODELS_ARGS[@]}"; do echo "  $m"; done
echo "Python overrides: $QWEN35_OVERRIDES"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would submit eval job with the above models."
    exit 0
fi

JOB_ID=$(EVAL_MODELS="$MODELS_STR" \
    sbatch --parsable \
        --job-name="agent-distill-qwen35-eval" \
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

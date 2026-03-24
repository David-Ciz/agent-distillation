#!/usr/bin/env bash
# =============================================================================
# submit_collapse_eval_sweep.sh — Submit the Qwen2.5 collapse benchmark sweep
#
# Benchmarks the trained Qwen2.5 LoRA final adapters with the Golden Five suite.
#
# Usage:
#   bash task1/scripts/slurm/submit_collapse_eval_sweep.sh
#   bash task1/scripts/slurm/submit_collapse_eval_sweep.sh --dry-run
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/collapse_eval_lumi.sh"
SBATCH_EXPORT="ALL"

if [[ -n "${CONTAINER_PYTHON_OVERRIDES:-}" ]]; then
    SBATCH_EXPORT+=",CONTAINER_PYTHON_OVERRIDES=${CONTAINER_PYTHON_OVERRIDES}"
fi
if [[ -n "${CONTAINER_VENV:-}" ]]; then
    SBATCH_EXPORT+=",CONTAINER_VENV=${CONTAINER_VENV}"
fi

SCRATCH_DIR="/scratch/project_465002758/${USER:-daciz}/agent-distillation/task1/outputs"

MODELS_ARGS=(
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,8'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-1.5B-Instruct-lora-final,Qwen2.5-1.5B-Instruct-lora,lora,4'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora,2'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-7B-Instruct-lora-final,Qwen2.5-7B-Instruct-lora,lora,1'"
)

MODELS_STR="${MODELS_ARGS[*]}"
EXTRA_ARGS="${COLLAPSE_EVAL_EXTRA_ARGS:---mlflow-experiment collapse-benchmarks}"

echo "=================================================="
echo "Qwen2.5 collapse benchmark sweep — $(date)"
echo "Models to benchmark: ${#MODELS_ARGS[@]}"
echo "=================================================="
for m in "${MODELS_ARGS[@]}"; do echo "  $m"; done
if [[ -n "${CONTAINER_PYTHON_OVERRIDES:-}" ]]; then
    echo "Python overrides: $CONTAINER_PYTHON_OVERRIDES"
fi
echo "Extra args: $EXTRA_ARGS"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would submit collapse benchmark job with the above models."
    exit 0
fi

JOB_ID=$(COLLAPSE_EVAL_MODELS="$MODELS_STR" \
    COLLAPSE_EVAL_EXTRA_ARGS="$EXTRA_ARGS" \
    sbatch --parsable \
        --job-name="agent-distill-qwen25-collapse" \
        --time=16:00:00 \
        --export="$SBATCH_EXPORT" \
        "$EVAL_SCRIPT")

echo "Submitted collapse benchmark sweep job: $JOB_ID"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ~/agent-distillation/logs/collapse_${JOB_ID}.out"
echo ""
echo "After completion, sync results locally:"
echo "  bash task1/scripts/slurm/sync_results.sh --results-only"

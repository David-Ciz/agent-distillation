#!/usr/bin/env bash
# =============================================================================
# submit_qwen35_collapse_sweep.sh — Submit the Qwen3.5 collapse benchmark sweep
#
# Benchmarks the trained Qwen3.5 LoRA final adapters with the Golden Five suite.
# This wrapper uses the same override path convention as the Qwen3.5 training
# and standard evaluation sweeps.
#
# Usage:
#   bash task1/scripts/slurm/submit_qwen35_collapse_sweep.sh
#   bash task1/scripts/slurm/submit_qwen35_collapse_sweep.sh --dry-run
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/collapse_eval_lumi.sh"
QWEN35_OVERRIDES="${QWEN35_OVERRIDES:-${HOME}/agent-distillation/py-overrides}"
SBATCH_EXPORT="ALL,CONTAINER_PYTHON_OVERRIDES=${QWEN35_OVERRIDES}"

if [[ -n "${CONTAINER_VENV:-}" ]]; then
    SBATCH_EXPORT+=",CONTAINER_VENV=${CONTAINER_VENV}"
fi

SCRATCH_DIR="/scratch/project_465002758/${USER:-daciz}/agent-distillation/task1/outputs"

MODELS_ARGS=(
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-0.8B-lora-final,Qwen3.5-0.8B-lora,lora,8'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-2B-lora-final,Qwen3.5-2B-lora,lora,4'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-4B-lora-final,Qwen3.5-4B-lora,lora,2'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-9B-lora-final,Qwen3.5-9B-lora,lora,1'"
)

MODELS_STR="${MODELS_ARGS[*]}"
EXTRA_ARGS="${COLLAPSE_EVAL_EXTRA_ARGS:---mlflow-experiment collapse-benchmarks}"

echo "=================================================="
echo "Qwen3.5 collapse benchmark sweep — $(date)"
echo "Models to benchmark: ${#MODELS_ARGS[@]}"
echo "=================================================="
for m in "${MODELS_ARGS[@]}"; do echo "  $m"; done
echo "Python overrides: $QWEN35_OVERRIDES"
echo "Extra args: $EXTRA_ARGS"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would submit collapse benchmark job with the above models."
    exit 0
fi

JOB_ID=$(COLLAPSE_EVAL_MODELS="$MODELS_STR" \
    COLLAPSE_EVAL_EXTRA_ARGS="$EXTRA_ARGS" \
    sbatch --parsable \
        --job-name="agent-distill-qwen35-collapse" \
        --time=24:00:00 \
        --export="$SBATCH_EXPORT" \
        "$EVAL_SCRIPT")

echo "Submitted collapse benchmark sweep job: $JOB_ID"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ~/agent-distillation/logs/collapse_${JOB_ID}.out"
echo ""
echo "After completion, sync results locally:"
echo "  bash task1/scripts/slurm/sync_results.sh --results-only"

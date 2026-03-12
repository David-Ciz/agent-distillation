#!/usr/bin/env bash
# =============================================================================
# submit_qwen35_eval_sweep.sh — Submit the Qwen3.5 evaluation sweep
#
# Evaluates the Qwen3.5 LoRA adapters from scratch plus their base models.
# This keeps the baseline evaluation sweep unchanged.
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
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"

# Model weights live on scratch, not home
SCRATCH_DIR="/scratch/project_465002758/${USER:-daciz}/agent-distillation/task1/outputs"

# ---------------------------------------------------------------------------
# Qwen3.5 models to evaluate
# Format per entry: 'ADAPTER_PATH_OR_HF_ID,NAME,TYPE,GEN_BATCH'
# TYPE: lora | full_finetune | base
# ---------------------------------------------------------------------------
MODELS_ARGS=(
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-0.8B-lora-final,Qwen3.5-0.8B-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-2B-lora-final,Qwen3.5-2B-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen3.5-4B-lora-final,Qwen3.5-4B-lora,lora,16'"
    "--models 'Qwen/Qwen3.5-0.8B,Qwen3.5-0.8B-base,base,32'"
    "--models 'Qwen/Qwen3.5-2B,Qwen3.5-2B-base,base,32'"
    "--models 'Qwen/Qwen3.5-4B,Qwen3.5-4B-base,base,16'"
)

MODELS_STR="${MODELS_ARGS[*]}"

echo "=================================================="
echo "Qwen3.5 evaluation sweep — $(date)"
echo "Models to evaluate: ${#MODELS_ARGS[@]}"
echo "=================================================="
for m in "${MODELS_ARGS[@]}"; do echo "  $m"; done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would submit eval job with the above models."
    exit 0
fi

JOB_ID=$(sbatch --parsable \
    --job-name="agent-distill-qwen35-eval" \
    --time=08:00:00 \
    --export=ALL \
    --wrap="
        source ${SCRIPT_DIR}/container.env
        export MLFLOW_TRACKING_URI=\"sqlite:////users/\${USER}/mlflow/mlflow.db\"
        export MLFLOW_ARTIFACT_ROOT=\"/scratch/project_465002758/\${USER}/mlruns\"
        export SCRATCH_OUTPUT_DIR=\"/scratch/project_465002758/\${USER}/agent-distillation/task1/outputs\"
        [ -n \"\${CONTAINER_VENV}\" ] && VENV_CMD=\"source \${CONTAINER_VENV}/bin/activate &&\" || VENV_CMD=\"\"
        srun singularity run \"\$SIF\" bash -c \"
            \${VENV_CMD}
            cd ${REPO_DIR}
            python task1/scripts/05_model_evaluation.py ${MODELS_STR}
        \"
    ")

echo "Submitted eval sweep job: $JOB_ID"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ~/agent-distillation/logs/eval_${JOB_ID}.out"
echo ""
echo "After completion, sync results locally:"
echo "  bash task1/scripts/slurm/sync_results.sh"

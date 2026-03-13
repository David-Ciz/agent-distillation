#!/usr/bin/env bash
# =============================================================================
# submit_eval_sweep.sh — Submit a single evaluation job covering all models
#
# Runs 05_model_evaluation.py with one --models flag per model config.
# Evaluates both the trained adapter and the base model for comparison.
#
# Usage:
#   bash task1/scripts/slurm/submit_eval_sweep.sh
#   bash task1/scripts/slurm/submit_eval_sweep.sh --dry-run
#
# Prerequisites:
#   - All training jobs from submit_train_sweep.sh must be complete
#   - Adapter directories must exist under task1/outputs/
#
# After this job finishes:
#   1. rsync results locally:
#        bash task1/scripts/slurm/sync_results.sh
#   2. Run analysis locally:
#        .venv/bin/python task1/scripts/06_analyse_visualize_results.py \
#            --eval-run-dir task1/outputs/evaluations/eval_run_<timestamp> \
#            --output-dir task1/outputs/analysis/<run_name>/
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_lumi.sh"

# Model weights live on scratch, not home
SCRATCH_DIR="/scratch/project_465002758/${USER:-daciz}/agent-distillation/task1/outputs"

# ---------------------------------------------------------------------------
# Models to evaluate — add --models lines here for each trained model
# Format per entry: 'ADAPTER_PATH_OR_HF_ID,NAME,TYPE,GEN_BATCH'
# TYPE: lora | full_finetune | base
# ---------------------------------------------------------------------------
MODELS_ARGS=(
    # LoRA adapters (trained) — weights on scratch
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-1.5B-Instruct-lora-final,Qwen2.5-1.5B-Instruct-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora,32'"
    "--models '${SCRATCH_DIR}/Qwen_Qwen2.5-7B-Instruct-lora-final,Qwen2.5-7B-Instruct-lora,lora,16'"
    # Base models (sanity check — HuggingFace, no local path)
    "--models 'Qwen/Qwen2.5-0.5B-Instruct,Qwen2.5-0.5B-Instruct-base,base,32'"
    "--models 'Qwen/Qwen2.5-3B-Instruct,Qwen2.5-3B-Instruct-base,base,32'"
    "--models 'Qwen/Qwen2.5-7B-Instruct,Qwen2.5-7B-Instruct-base,base,16'"
)

MODELS_STR="${MODELS_ARGS[*]}"

echo "=================================================="
echo "Evaluation sweep — $(date)"
echo "Models to evaluate: ${#MODELS_ARGS[@]}"
echo "=================================================="
for m in "${MODELS_ARGS[@]}"; do echo "  $m"; done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would submit eval job with the above models."
    exit 0
fi

JOB_ID=$(sbatch --parsable \
    --job-name="agent-distill-eval-sweep" \
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
            [ -n \"\${CONTAINER_PYTHON_OVERRIDES}\" ] && [ -d \"\${CONTAINER_PYTHON_OVERRIDES}\" ] && export PYTHONPATH=\"\${CONTAINER_PYTHON_OVERRIDES}\${PYTHONPATH:+:\${PYTHONPATH}}\"
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

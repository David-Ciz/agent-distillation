#!/bin/bash
# =============================================================================
# audit_lumi.sh — Run qualitative output audits on LUMI
#
# Can be used directly:
#   sbatch task1/scripts/slurm/audit_lumi.sh
#
# Or driven by setting env vars before sbatch:
#   AUDIT_MODELS="--models 'path,name,type' --models '...'" \
#   AUDIT_SAMPLE_IDS="0,1,2,11,21" \
#       sbatch task1/scripts/slurm/audit_lumi.sh
# =============================================================================
#SBATCH --job-name=agent-distill-audit
#SBATCH --account=project_465002758
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/users/%u/agent-distillation/logs/audit_%j.out
#SBATCH --error=/users/%u/agent-distillation/logs/audit_%j.err

set -euo pipefail
ulimit -c 0

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${HOME}/agent-distillation"
source "${REPO_DIR}/task1/scripts/slurm/container.env"

mkdir -p "${REPO_DIR}/logs"

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPUs: 1"

export MLFLOW_TRACKING_URI="sqlite:////users/${USER}/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/${USER}/mlruns"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
if [[ -n "${CONTAINER_PYTHON_OVERRIDES:-}" && -d "${CONTAINER_PYTHON_OVERRIDES}" ]]; then
    echo "Python overrides:    $CONTAINER_PYTHON_OVERRIDES"
fi

SCRATCH_OUTPUT_DIR="/scratch/project_465002758/${USER}/agent-distillation/task1/outputs"

# Default to a simple control-vs-stressed audit set.
AUDIT_MODELS="${AUDIT_MODELS:---models '${SCRATCH_OUTPUT_DIR}/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora' --models '${SCRATCH_OUTPUT_DIR}/Qwen_Qwen3.5-4B-lora-final,Qwen3.5-4B-lora,lora'}"
AUDIT_SAMPLE_IDS="${AUDIT_SAMPLE_IDS:-0,1,2,5,11,21,22,25,29,339}"
AUDIT_NUM_SAMPLES="${AUDIT_NUM_SAMPLES:-10}"
AUDIT_MAX_NEW_TOKENS="${AUDIT_MAX_NEW_TOKENS:-256}"
AUDIT_EXTRA_ARGS="${AUDIT_EXTRA_ARGS:-}"

echo "Audit models:    ${AUDIT_MODELS}"
echo "Audit sample ids:${AUDIT_SAMPLE_IDS}"
echo "Audit num samp.: ${AUDIT_NUM_SAMPLES}"
echo "Max new tokens:  ${AUDIT_MAX_NEW_TOKENS}"
if [[ -n "${AUDIT_EXTRA_ARGS}" ]]; then
    echo "Extra args:      ${AUDIT_EXTRA_ARGS}"
fi
echo "Container venv: ${CONTAINER_VENV:-<none>}"

srun singularity run "$SIF" \
    bash -c "
        ulimit -c 0
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        [ -n \"${CONTAINER_PYTHON_OVERRIDES:-}\" ] && [ -d \"${CONTAINER_PYTHON_OVERRIDES}\" ] && export PYTHONPATH=\"${CONTAINER_PYTHON_OVERRIDES}\${PYTHONPATH:+:\${PYTHONPATH}}\"
        cd ${REPO_DIR}
        python task1/scripts/15_audit_model_outputs.py \
            ${AUDIT_MODELS} \
            --sample-ids '${AUDIT_SAMPLE_IDS}' \
            --num-samples ${AUDIT_NUM_SAMPLES} \
            --max-new-tokens ${AUDIT_MAX_NEW_TOKENS} \
            ${AUDIT_EXTRA_ARGS}
    "

echo "Audit complete."

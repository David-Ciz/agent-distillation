#!/bin/bash
# =============================================================================
# eval_lumi.sh — Evaluate fine-tuned models on LUMI
#
# Can be used in two ways:
#
# 1. Direct submit (quick single eval):
#      sbatch task1/scripts/slurm/eval_lumi.sh
#    Uses the EVAL_MODELS default below.
#
# 2. Driven by submit_eval_sweep.sh (full sweep):
#    The sweep script passes --models args via the EVAL_MODELS env var.
#
# Override models on the command line:
#   EVAL_MODELS="--models 'path,name,type,batch' --models '...'" \
#       sbatch task1/scripts/slurm/eval_lumi.sh
#
# Monitor: squeue -u $USER
# Logs:    tail -f logs/eval_<JOB_ID>.out
# =============================================================================
#SBATCH --job-name=agent-distill-eval
#SBATCH --account=project_465002758
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/users/%u/agent-distillation/logs/eval_%j.out
#SBATCH --error=/users/%u/agent-distillation/logs/eval_%j.err

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${HOME}/agent-distillation"
source "${REPO_DIR}/task1/scripts/slurm/container.env"

mkdir -p "${REPO_DIR}/logs"

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPUs: 1"

# MLflow storage — backend (metadata) in home, artifacts on scratch
export MLFLOW_TRACKING_URI="sqlite:////users/${USER}/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/${USER}/mlruns"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
if [[ -n "${CONTAINER_PYTHON_OVERRIDES:-}" && -d "${CONTAINER_PYTHON_OVERRIDES}" ]]; then
    echo "Python overrides:    $CONTAINER_PYTHON_OVERRIDES"
fi

# Model weights live on scratch, not home
SCRATCH_OUTPUT_DIR="/scratch/project_465002758/${USER}/agent-distillation/task1/outputs"

# Default model list — override by setting EVAL_MODELS before sbatch
EVAL_MODELS="${EVAL_MODELS:---models '${SCRATCH_OUTPUT_DIR}/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora,32' --models 'Qwen/Qwen2.5-3B-Instruct,Qwen2.5-3B-Instruct-base,base,32'}"

echo "Evaluating: ${EVAL_MODELS}"

srun singularity run "$SIF" \
    bash -c "
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        cd ${REPO_DIR}
        python task1/scripts/05_model_evaluation.py ${EVAL_MODELS}
    "

echo "Evaluation complete."

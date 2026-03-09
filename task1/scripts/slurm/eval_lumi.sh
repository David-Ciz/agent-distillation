#!/bin/bash
# =============================================================================
# eval_lumi.sh — Evaluate fine-tuned models on LUMI
#
# Runs 05_model_evaluation.py for one or more model configs and logs all
# metrics to MLflow (nested runs under a single parent eval job run).
#
# Requires only 1 GPU — uses small-g partition.
#
# Before running:
#   1. Training jobs must be complete (adapter_config.json must exist)
#   2. task1_eval_dataset.csv must be present in task1/data/
#
# Submit: sbatch task1/scripts/slurm/eval_lumi.sh
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

srun singularity run "$SIF" \
    bash -c "
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        cd ${REPO_DIR}
        python task1/scripts/05_model_evaluation.py \
            --models '${REPO_DIR}/task1/outputs/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora,32' \
            --models 'Qwen/Qwen2.5-3B-Instruct,Qwen2.5-3B-Instruct-base,base,32'
    "

echo "Evaluation complete."


#!/bin/bash
# =============================================================================
# train_lora_lumi.sh — Phase 3: LoRA training on LUMI
#
# Before running:
#   1. Complete Phase 2 (lumi_env_check.py passes all required packages)
#   2. Recreate venv at ~/agent-distillation/my-env with mlflow + click installed
#   3. Set your project ID and scratch path below
#
# Submit: sbatch task1/scripts/slurm/train_lora_lumi.sh
# Monitor: squeue -u $USER
# =============================================================================
#SBATCH --job-name=agent-distill-lora
#SBATCH --account=project_465002758
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=/users/%u/agent-distillation/logs/train_lora_%j.out
#SBATCH --error=/users/%u/agent-distillation/logs/train_lora_%j.err

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Use absolute path — SLURM copies the script to a spool dir, so relative paths break
REPO_DIR="${HOME}/agent-distillation"
source "${REPO_DIR}/task1/scripts/slurm/container.env"

mkdir -p "${REPO_DIR}/logs"

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPUs: 8"

# MLflow storage — backend (metadata) in home, artifacts on scratch
export MLFLOW_TRACKING_URI="sqlite:////users/${USER}/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/${USER}/mlruns"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"

srun singularity run "$SIF" \
    bash -c "
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        python -m torch.distributed.run \
            --nproc_per_node=8 \
            --standalone \
        task1/scripts/02_train_model_lora.py \
            --model-name 'Qwen/Qwen2.5-0.3B-Instruct' \
            --num-epochs 3 \
            --batch-size 2 \
            --gradient-accumulation-steps 4
    "

echo "Training complete."

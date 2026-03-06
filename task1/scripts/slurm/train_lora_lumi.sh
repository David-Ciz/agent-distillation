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
#SBATCH --ntasks-per-node=8
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

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
export MASTER_PORT=29500

# MLflow storage — backend (metadata) in home, artifacts on scratch
export MLFLOW_TRACKING_URI="sqlite:////users/${USER}/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/${USER}/mlruns"

echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_NNODES  GPUs/node: 8  Master: $MASTER_ADDR"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo "MLflow artifact root: $MLFLOW_ARTIFACT_ROOT"

srun singularity run "$SIF" \
    bash -c "
        source ~/agent-distillation/my-env/bin/activate
        python -m torch.distributed.run \
            --nproc_per_node=8 \
            --nnodes=${SLURM_NNODES} \
            --node_rank=${SLURM_NODEID} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
        task1/scripts/02_train_model_lora.py \
            --model-name 'Qwen/Qwen2.5-0.5B-Instruct' \
            --num-epochs 1 \
            --batch-size 4 \
            --gradient-accumulation-steps 2
    "

echo "Training complete."

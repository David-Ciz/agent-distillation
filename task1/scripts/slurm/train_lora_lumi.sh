#!/bin/bash
# =============================================================================
# train_lora_lumi.sh -- Phase 3: LoRA training on LUMI
#
# Before running:
#   1. Complete Phase 1 (hello_world.sh) and Phase 2 (lumi_env_check.py)
#   2. Add MLflow integration to 02_train_model_lora.py
#   3. Set MLFLOW_TRACKING_URI and your project ID below
#
# Submit: sbatch task1/scripts/slurm/train_lora_lumi.sh
# =============================================================================
#SBATCH --job-name=agent-distill-lora
#SBATCH --account=<your-project-id>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_lora_%j.out
#SBATCH --error=logs/train_lora_%j.err

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/container.env"

mkdir -p logs

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
export MASTER_PORT=29500
# Set MLFLOW_TRACKING_URI to a path on shared FS or a remote URI
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# If extra packages were installed in a venv, activate it here:
# source /path/to/my-env/bin/activate

echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_NNODES  GPUs/node: 8  Master: $MASTER_ADDR"

srun singularity run "$SIF" \
    python -m torch.distributed.run \
        --nproc_per_node=8 \
        --nnodes="$SLURM_NNODES" \
        --node_rank="$SLURM_NODEID" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
    task1/scripts/02_train_model_lora.py \
        --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
        --num_epochs 1 \
        --batch_size 4

echo "Training complete."

#!/bin/bash
# =============================================================================
# train_full_finetune_lumi.sh — Full finetune training on LUMI
#
# Full finetune is memory-intensive (no LoRA adapter, all weights updated).
# Uses 8 MI250X GCDs with torch.distributed.run --standalone.
#
# Submit: sbatch task1/scripts/slurm/train_full_finetune_lumi.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/train_full_finetune_<JOB_ID>.out
# =============================================================================
#SBATCH --job-name=agent-distill-full-ft
#SBATCH --account=project_465002758
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=10:00:00
#SBATCH --output=/users/%u/agent-distillation/logs/train_full_finetune_%j.out
#SBATCH --error=/users/%u/agent-distillation/logs/train_full_finetune_%j.err

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${HOME}/agent-distillation"
source "${REPO_DIR}/task1/scripts/slurm/container.env"

mkdir -p "${REPO_DIR}/logs"

# ------------------------------------------------------------------
# MLflow
# ------------------------------------------------------------------
export MLFLOW_TRACKING_URI="sqlite:////users/${USER}/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/${USER}/mlruns"

# Model checkpoints and final weights go to scratch — NOT home dir.
# Home quota is 20 GB; a single 7B checkpoint can be >14 GB.
export SCRATCH_OUTPUT_DIR="/scratch/project_465002758/${USER}/agent-distillation/task1/outputs"
mkdir -p "${SCRATCH_OUTPUT_DIR}"

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPUs: 8"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo "Model output dir:    $SCRATCH_OUTPUT_DIR"

# ------------------------------------------------------------------
# Background GPU monitor (every 30 s)
# ------------------------------------------------------------------
singularity run "$SIF" bash -c "
    while true; do
        echo '--- '\$(date)' ---' >> ${REPO_DIR}/logs/gpu_stats_${SLURM_JOB_ID}.log
        rocm-smi --showuse --showmeminfo vram >> ${REPO_DIR}/logs/gpu_stats_${SLURM_JOB_ID}.log 2>&1
        sleep 30
    done
" &
GPU_MONITOR_PID=$!

# ------------------------------------------------------------------
# Launch — single srun task, torch.distributed.run --standalone spawns
# 8 worker processes internally (one per GCD).
# batch_size=1 + grad_accum=8 → effective batch = 8 per GPU × 8 GPUs = 64
# ------------------------------------------------------------------
srun singularity run "$SIF" \
    bash -c "
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        python -m torch.distributed.run \
            --nproc_per_node=8 \
            --standalone \
        task1/scripts/03_train_model_full_finetune.py \
            --model-name 'Qwen/Qwen2.5-3B-Instruct' \
            --num-epochs 3 \
            --batch-size 1 \
            --gradient-accumulation-steps 8
    "

kill $GPU_MONITOR_PID 2>/dev/null || true
echo "Full finetune training complete."


#!/bin/bash
# =============================================================================
# train_lora_lumi.sh — Phase 3: LoRA training on LUMI
#
# GPU utilisation note:
#   All 8 MI250X GCDs are used correctly by torch.distributed.run --standalone.
#   The occasional 0% reading in rocm-smi is normal — it's sampled during the
#   brief DataLoader prefetch gap between training steps, not a real problem.
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
ulimit -c 0

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Use absolute path — SLURM copies the script to a spool dir, so relative paths break
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
if [[ -n "${CONTAINER_PYTHON_OVERRIDES:-}" ]]; then
    echo "Python overrides:    $CONTAINER_PYTHON_OVERRIDES"
fi

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
# 8 worker processes internally (one per GCD). No port conflicts.
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Model / training hyperparams — override via env before sbatch, e.g.:
#   MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct BATCH_SIZE=4 GRAD_ACCUM=2 \
#       sbatch task1/scripts/slurm/train_lora_lumi.sh
#   CONTAINER_PYTHON_OVERRIDES=/scratch/project_465002758/$USER/qwen35-test-overrides \
#       MODEL_NAME=Qwen/Qwen3.5-2B sbatch task1/scripts/slurm/train_lora_lumi.sh
# ------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
TRAIN_ATTN_IMPL="${TRAIN_ATTN_IMPL:-auto}"

echo "Model: $MODEL_NAME  Epochs: $NUM_EPOCHS  Batch: $BATCH_SIZE  GradAccum: $GRAD_ACCUM  Attn: $TRAIN_ATTN_IMPL"

srun singularity run "$SIF" \
    bash -c "
        ulimit -c 0
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        [ -n \"${CONTAINER_PYTHON_OVERRIDES:-}\" ] && export PYTHONPATH=\"${CONTAINER_PYTHON_OVERRIDES}\${PYTHONPATH:+:\${PYTHONPATH}}\"
        python -m torch.distributed.run \
            --nproc_per_node=8 \
            --standalone \
        task1/scripts/02_train_model_lora.py \
            --model-name '${MODEL_NAME}' \
            --num-epochs ${NUM_EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --gradient-accumulation-steps ${GRAD_ACCUM} \
            --attn-implementation ${TRAIN_ATTN_IMPL}
    "

kill $GPU_MONITOR_PID 2>/dev/null || true
echo "Training complete."

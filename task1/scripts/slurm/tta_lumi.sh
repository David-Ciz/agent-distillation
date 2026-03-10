#!/bin/bash
# =============================================================================
# tta_lumi.sh — TTA / Self-Consistency experiment on LUMI
#
# Can be used in two ways:
#
# 1. Direct submit (quick single run):
#      sbatch task1/scripts/slurm/tta_lumi.sh
#    Uses the TTA_MODELS default below.
#
# 2. Driven by submit_tta_sweep.sh (per-model jobs):
#    The sweep script sets TTA_MODELS (and optionally TTA_N_VALUES,
#    TTA_TEMPS, TTA_AGGS, TTA_SAMPLES) before calling sbatch.
#
# Override any env var before sbatch:
#   TTA_MODELS="--models 'path,name,type,batch'" \
#   TTA_N_VALUES="1,3,5" TTA_TEMPS="0.7" \
#       sbatch task1/scripts/slurm/tta_lumi.sh
#
# Monitor: squeue -u $USER
# Logs:    tail -f ~/agent-distillation/logs/tta_<JOB_ID>.out
# =============================================================================
#SBATCH --job-name=agent-distill-tta
#SBATCH --account=project_465002758
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/users/%u/agent-distillation/logs/tta_%j.out
#SBATCH --error=/users/%u/agent-distillation/logs/tta_%j.err

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

# ---------------------------------------------------------------------------
# TTA parameters — all overridable via env vars before sbatch
# ---------------------------------------------------------------------------
# Default: Qwen 0.5B LoRA (smallest/most-over-abstaining model — Phase A)
TTA_MODELS="${TTA_MODELS:---models '${REPO_DIR}/task1/outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32'}"

TTA_N_VALUES="${TTA_N_VALUES:-1,3,5}"
TTA_TEMPS="${TTA_TEMPS:-0.7}"
TTA_AGGS="${TTA_AGGS:-majority_vote,centroid,oracle}"
TTA_SAMPLES="${TTA_SAMPLES:-0}"
TTA_TIME="${TTA_TIME:-08:00:00}"   # informational only; set via --time in sbatch

echo "TTA models:      ${TTA_MODELS}"
echo "N values:        ${TTA_N_VALUES}"
echo "Temperatures:    ${TTA_TEMPS}"
echo "Aggregations:    ${TTA_AGGS}"
echo "Samples:         ${TTA_SAMPLES}"

srun singularity run "$SIF" \
    bash -c "
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        cd ${REPO_DIR}
        python task1/scripts/07_tta_experiment.py \
            ${TTA_MODELS} \
            --n-values '${TTA_N_VALUES}' \
            --temperatures '${TTA_TEMPS}' \
            --aggregations '${TTA_AGGS}' \
            --num-samples ${TTA_SAMPLES} \
            --gen-batch-size 32
    "

echo "TTA experiment complete."


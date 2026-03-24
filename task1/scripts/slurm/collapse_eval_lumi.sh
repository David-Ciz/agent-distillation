#!/bin/bash
# =============================================================================
# collapse_eval_lumi.sh — Run final-checkpoint collapse benchmarks on LUMI
#
# Can be used directly:
#   sbatch task1/scripts/slurm/collapse_eval_lumi.sh
#
# Or driven by a sweep wrapper:
#   COLLAPSE_EVAL_MODELS="--models 'path,name,type,batch' ..." \
#       sbatch task1/scripts/slurm/collapse_eval_lumi.sh
# =============================================================================
#SBATCH --job-name=agent-distill-collapse
#SBATCH --account=project_465002758
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/users/%u/agent-distillation/logs/collapse_%j.out
#SBATCH --error=/users/%u/agent-distillation/logs/collapse_%j.err

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
COLLAPSE_EVAL_MODELS="${COLLAPSE_EVAL_MODELS:---models '${SCRATCH_OUTPUT_DIR}/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora,4'}"
COLLAPSE_EVAL_EXTRA_ARGS="${COLLAPSE_EVAL_EXTRA_ARGS:-}"

echo "Benchmarking models: ${COLLAPSE_EVAL_MODELS}"
if [[ -n "${COLLAPSE_EVAL_EXTRA_ARGS}" ]]; then
    echo "Extra args:          ${COLLAPSE_EVAL_EXTRA_ARGS}"
fi
echo "Container venv: ${CONTAINER_VENV:-<none>}"

srun singularity run "$SIF" \
    bash -c "
        ulimit -c 0
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        [ -n \"${CONTAINER_PYTHON_OVERRIDES:-}\" ] && [ -d \"${CONTAINER_PYTHON_OVERRIDES}\" ] && export PYTHONPATH=\"${CONTAINER_PYTHON_OVERRIDES}\${PYTHONPATH:+:\${PYTHONPATH}}\"
        cd ${REPO_DIR}
        python task1/scripts/13_run_collapse_benchmarks.py ${COLLAPSE_EVAL_MODELS} ${COLLAPSE_EVAL_EXTRA_ARGS}
        python task1/scripts/14_aggregate_collapse_benchmarks.py
    "

echo "Collapse benchmark evaluation complete."

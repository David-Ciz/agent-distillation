#!/bin/bash
#SBATCH --job-name=agent-distill-hello
#SBATCH --account=<your-project-id>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/hello_world_%j.out
#SBATCH --error=logs/hello_world_%j.err
set -euo pipefail
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
source "$(dirname "$0")/container.env"
mkdir -p logs
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  SIF: $SIF"
srun singularity run "$SIF" python task1/scripts/lumi_hello_world.py
echo "Phase 1 complete."

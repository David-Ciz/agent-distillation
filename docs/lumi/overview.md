# LUMI Overview

LUMI is the EuroHPC pre-exascale supercomputer at CSC (Finland). The project runs training and evaluation on LUMI's GPU partition using AMD MI250X GPUs.

---

## Hardware

| Resource | Spec |
|----------|------|
| Partition | `small-g` (up to 8 GCDs per node) |
| GPU | AMD MI250X — each card has 2 GCDs (128 GB HBM2e total per card) |
| Project account | `project_465002758` |

Training jobs use **8 GCDs** (4× MI250X cards, one full node).  
Evaluation and TTA jobs use **1 GCD** (`small-g`, 1× GPU).

---

## MLflow Storage Layout

| What | Path |
|------|------|
| Metadata DB | `sqlite:////users/$USER/mlflow/mlflow.db` (home dir) |
| Artifacts | `/scratch/project_465002758/$USER/mlruns/` (scratch) |

Set these before submitting any job:

```bash
export MLFLOW_TRACKING_URI="sqlite:////users/${USER}/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/${USER}/mlruns"
```

All SLURM scripts set these automatically.

---

## Container Setup

Jobs run inside a Singularity container. The container image path and optional venv are defined in `container.env`:

```bash
# task1/scripts/slurm/container.env
SIF=/path/to/pytorch.sif
CONTAINER_VENV=/path/to/venv   # optional — activated inside container
```

All SLURM scripts `source container.env` automatically.

---

## Typical End-to-End Workflow

```
[LUMI] bash task1/scripts/slurm/submit_train_sweep.sh
[LUMI] bash task1/scripts/slurm/submit_eval_sweep.sh
[LUMI] bash task1/scripts/slurm/submit_tta_sweep.sh --phase a
[local] bash task1/scripts/slurm/sync_results.sh
[local] python task1/scripts/06_analyse_visualize_results.py ...
[local] mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```


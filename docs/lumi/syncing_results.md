# Syncing Results from LUMI

After any evaluation or TTA job finishes, pull the lightweight result files (CSVs, JSONs, MLflow DB) to your local machine — no model weights needed.

---

## Quick Sync

```bash
bash task1/scripts/slurm/sync_results.sh
```

This syncs both the MLflow database and all evaluation/TTA result files.

## Options

```bash
bash task1/scripts/slurm/sync_results.sh --mlflow-only    # only the MLflow DB
bash task1/scripts/slurm/sync_results.sh --results-only   # only CSVs/JSONs
```

## What Gets Synced

| File pattern | From | To |
|---|---|---|
| `mlflow.db` | `~/mlflow/mlflow.db` | `mlflow/mlflow.db` |
| `*_detailed_results.csv` | eval runs | `task1/outputs/evaluations/` |
| `*_N*_detailed_results.csv` | TTA runs | same |
| `*_summary.json` | eval + TTA | same |
| `model_comparison_summary.csv` | eval runs | same |
| `tta_comparison_summary.csv` | TTA runs | same |
| `evaluation.log`, `tta_run.log` | all runs | same |

Model weights are **not** synced (too large).

## Configure Remote

Set these in your shell or `~/.config/fish/config.fish`:

```fish
set -x LUMI_USER daciz
set -x LUMI_HOST lumi.csc.fi
set -x LUMI_REPO ~/agent-distillation
```

## After Syncing

```bash
# View results in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000

# Run analysis on a standard eval run
python task1/scripts/06_analyse_visualize_results.py \
    --eval-run-dir task1/outputs/evaluations/eval_run_<timestamp> \
    --output-dir task1/outputs/analysis/run1/

# Inspect TTA comparison table
open task1/outputs/evaluations/tta_run_<timestamp>/tta_comparison_summary.csv
```


#!/usr/bin/env bash
# =============================================================================
# sync_results.sh — Sync evaluation results and MLflow DB from LUMI to local
#
# Run this locally after your LUMI eval job finishes.
#
# Related tools (all in this directory):
#   submit_train_sweep.sh  — submit one training job per model in parallel
#   submit_eval_sweep.sh   — submit one eval job covering all trained models
#   eval_lumi.sh           — single eval job (used by submit_eval_sweep.sh or standalone)
#   train_lora_lumi.sh     — single LoRA training job (used by submit_train_sweep.sh)
#   train_full_finetune_lumi.sh — single full finetune job
#
# Typical end-to-end flow:
#   1. [LUMI] bash submit_train_sweep.sh       # submit all training jobs
#   2. [LUMI] bash submit_eval_sweep.sh        # submit eval after training finishes
#   3. [local] bash sync_results.sh            # pull CSVs + mlflow.db
#   4. [local] .venv/bin/python 06_analyse_visualize_results.py ...
#   5. [local] mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
#
# Usage:
#   bash task1/scripts/slurm/sync_results.sh
#   bash task1/scripts/slurm/sync_results.sh --mlflow-only
#   bash task1/scripts/slurm/sync_results.sh --results-only
#
# After syncing, run the analysis:
#   .venv/bin/python task1/scripts/06_analyse_visualize_results.py \
#       --eval-run-dir task1/outputs/evaluations/<eval_run_dir> \
#       --output-dir task1/outputs/analysis/<run_name>/
# =============================================================================

set -euo pipefail

LUMI_USER="${LUMI_USER:-daciz}"
LUMI_HOST="${LUMI_HOST:-lumi.csc.fi}"
LUMI_REPO="${LUMI_REPO:-~/agent-distillation}"

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

SYNC_MLFLOW=true
SYNC_RESULTS=true

for arg in "$@"; do
    case $arg in
        --mlflow-only)  SYNC_RESULTS=false ;;
        --results-only) SYNC_MLFLOW=false ;;
    esac
done

echo "=================================================="
echo "Syncing from ${LUMI_USER}@${LUMI_HOST}"
echo "Local repo:  ${LOCAL_REPO}"
echo "=================================================="

# ---------------------------------------------------------------------------
# 1. Sync MLflow DB
# ---------------------------------------------------------------------------
if $SYNC_MLFLOW; then
    echo ""
    echo "Syncing MLflow DB..."
    mkdir -p "${LOCAL_REPO}/mlflow"
    rsync -av --progress \
        "${LUMI_USER}@${LUMI_HOST}:~/mlflow/mlflow.db" \
        "${LOCAL_REPO}/mlflow/mlflow.db"
    echo "MLflow DB synced. Start UI with:"
    echo "  mlflow ui --backend-store-uri sqlite:///${LOCAL_REPO}/mlflow/mlflow.db --port 5000"
fi

# ---------------------------------------------------------------------------
# 2. Sync evaluation CSVs (lightweight — no model weights)
# ---------------------------------------------------------------------------
if $SYNC_RESULTS; then
    echo ""
    echo "Syncing evaluation results..."
    mkdir -p "${LOCAL_REPO}/task1/outputs/evaluations"
    rsync -av --progress \
        --include="*/" \
        --include="*_detailed_results.csv" \
        --include="*_N*_detailed_results.csv" \
        --include="*_summary.json" \
        --include="model_comparison_summary.csv" \
        --include="tta_comparison_summary.csv" \
        --include="evaluation.log" \
        --include="tta_run.log" \
        --exclude="*" \
        "${LUMI_USER}@${LUMI_HOST}:${LUMI_REPO}/task1/outputs/evaluations/" \
        "${LOCAL_REPO}/task1/outputs/evaluations/"
    echo "Results synced to: ${LOCAL_REPO}/task1/outputs/evaluations/"
fi

echo ""
echo "=================================================="
echo "Sync complete."
echo ""
echo "Next steps:"
echo "  1. Check new eval / TTA runs:"
echo "     ls task1/outputs/evaluations/"
echo ""
echo "  2. Run analysis on a standard eval run:"
echo "     .venv/bin/python task1/scripts/06_analyse_visualize_results.py \\"
echo "         --eval-run-dir task1/outputs/evaluations/<eval_run_dir> \\"
echo "         --output-dir task1/outputs/analysis/<run_name>/"
echo ""
echo "  3. Inspect TTA comparison table:"
echo "     open task1/outputs/evaluations/tta_run_<timestamp>/tta_comparison_summary.csv"
echo ""
echo "  4. View MLflow:"
echo "     mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000"
echo "=================================================="


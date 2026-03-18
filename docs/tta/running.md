# Running the TTA Experiment

---

## Quick Start (local / single GPU)

```bash
python task1/scripts/07_tta_experiment.py \
    --models 'task1/outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32' \
    --n-values 1,3,5 \
    --temperatures 0.7 \
    --aggregations majority_vote,centroid,oracle
```

## Sanity Check (N=1, T=0 must reproduce the 05 baseline)

```bash
python task1/scripts/07_tta_experiment.py \
    --models 'task1/outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32' \
    --n-values 1 \
    --temperatures 0.0 \
    --aggregations majority_vote
```

Compare the `abstain_f1` in `tta_comparison_summary.csv` against the `05` baseline. **Must be within ±0.002.**

## Deployment-Style Latency Benchmark

Use this when you want to answer the product question directly: how much
quality do we gain for the extra wall-clock latency of `N=3` or `N=5`?

```bash
python task1/scripts/07_tta_experiment.py \
    --models 'task1/outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32' \
    --n-values 1,3,5 \
    --temperatures 0.0,0.7 \
    --aggregations majority_vote \
    --latency-benchmark \
    --latency-num-samples 10 \
    --latency-warmup-samples 1
```

This adds per-query latency columns to `tta_comparison_summary.csv`, including:

- `latency_total_mean_ms`
- `latency_total_p50_ms`
- `latency_total_p95_ms`
- `latency_multiplier_vs_n1_t0`
- `delta_abstain_f1_vs_n1_t0`

Interpretation:

- `N=1, T=0.0` is the real baseline row.
- `delta_abstain_f1_vs_n1_t0` tells you the quality gain/loss.
- `latency_total_mean_ms` tells you the average end-to-end per-query cost.
- `latency_multiplier_vs_n1_t0` tells you how much slower the TTA variant is.

---

## CLI Reference

```
python task1/scripts/07_tta_experiment.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--models` | *(required)* | `'path,name,type[,batch]'` — repeat for multiple models |
| `--eval-dataset` | `task1/data/task1_eval_dataset.csv` | Evaluation dataset CSV |
| `--output-dir` | `task1/outputs/evaluations` | Base output directory |
| `--n-values` | `1,3,5` | Comma-separated N values to sweep |
| `--temperatures` | `0.7` | Comma-separated temperatures |
| `--aggregations` | `majority_vote,centroid,oracle` | Methods to run |
| `--num-samples` | `0` (all) | Limit samples for quick testing |
| `--gen-batch-size` | `8` | Default generation batch size |
| `--max-new-tokens` | `256` | Max tokens per generation pass |
| `--embedding-model` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model |
| `--mlflow-experiment` | `tta_experiment` | MLflow experiment name |
| `--latency-benchmark` | `False` | Measure per-query latency on a small eval subset |
| `--latency-num-samples` | `10` | Number of samples to use for latency benchmarking |
| `--latency-warmup-samples` | `1` | Warmup samples excluded from latency stats |

Model config format: `path,name,type[,gen_batch_size]`  
Types: `lora` | `full_finetune` | `base`

---

## Phased Approach on LUMI

### Phase A — Quick signal (~2h per model on small-g)

Targets the two most over-abstaining models:

```bash
bash task1/scripts/slurm/submit_tta_sweep.sh --phase a
```

This submits one `tta_lumi.sh` job per model with `N ∈ {1,3,5}`, `T=0.7`.

### Phase B — Temperature sweep (only if Phase A improves F1)

```bash
bash task1/scripts/slurm/submit_tta_sweep.sh --phase b
```

Runs `T ∈ {0.5, 1.0}` on the best Phase A model.

### Phase C — Scaling check (does TTA help larger models?)

```bash
bash task1/scripts/slurm/submit_tta_sweep.sh --phase c
```

Runs `N ∈ {1,5}`, `T=0.7` on Qwen 1.5B LoRA.

### Dry-run (see what would be submitted without submitting)

```bash
bash task1/scripts/slurm/submit_tta_sweep.sh --phase a --dry-run
```

---

## Running a Single Job Manually

Override any parameter via environment variables before `sbatch`:

```bash
TTA_MODELS="--models '/path/to/model,my-model,lora,32'" \
TTA_N_VALUES="1,3,5,10" \
TTA_TEMPS="0.5,0.7" \
TTA_TIME="10:00:00" \
    sbatch task1/scripts/slurm/tta_lumi.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `TTA_MODELS` | Qwen 0.5B LoRA | `--models` argument(s) |
| `TTA_N_VALUES` | `1,3,5` | Comma-separated N values |
| `TTA_TEMPS` | `0.7` | Comma-separated temperatures |
| `TTA_AGGS` | `majority_vote,centroid,oracle` | Aggregation methods |
| `TTA_SAMPLES` | `0` | Max samples (0 = all) |

---

## After Jobs Complete

```bash
# 1. Sync results to local machine
bash task1/scripts/slurm/sync_results.sh

# 2. Inspect the flat comparison table
open task1/outputs/evaluations/tta_run_<timestamp>/tta_comparison_summary.csv

# 3. Feed a single TTA CSV to the standard analysis script
python task1/scripts/06_analyse_visualize_results.py \
    --eval-run-dir task1/outputs/evaluations/tta_run_<timestamp>/Qwen2.5-0.5B-Instruct-lora/N5_T07 \
    --output-dir task1/outputs/analysis/tta_N5_T07/

# 4. View in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```

---

## GPU Budget

| Phase | Models | N values | Temps | Est. GPU-hours |
|-------|--------|----------|-------|---------------|
| A | Gemma 270M + Qwen 0.5B | 1,3,5 | 0.7 | ~4h total |
| B | Best Phase A model | 1,3,5 | 0.5, 1.0 | ~6h per temp |
| C | Qwen 1.5B | 1,5 | 0.7 | ~3h |
| D | Best model | 10 | best | +1.5h |

!!! warning "Get budget approval before Phase B+"
    The full sweep (all phases, all temps) is ~50 GPU-hours on LUMI small-g.

Per-model generation time on 1× MI250X GCD (`gen_batch_size=32`):

| Model | Time per pass (1,886 samples) |
|-------|-------------------------------|
| Gemma 270M | ~5 min |
| Qwen 0.5B | ~6 min |
| Qwen 1.5B | ~12 min |
| Embedding (any) | ~1 min |

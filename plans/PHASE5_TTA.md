# Phase 5 — TTA / Self-Consistency Experiment

> **Status**: Ready to implement  
> **Date created**: 2026-03-09  
> **Prerequisite**: Phase 4 complete — baseline reproduced on LUMI, eval pipeline working end-to-end.  
> **Goal**: Determine whether running each eval sample N times (temperature > 0) and aggregating answers improves performance, especially for the smallest models (Gemma 270M, Qwen 0.5B).  
> **Reference**: Self-Consistency (Wang et al. 2022)

---

## Background

The hypothesis is that small models are uncertain — their single-pass outputs are noisy. Running N passes and taking a majority vote on the abstain decision and centroid-nearest selection on the answer text should smooth out that noise, effectively trading inference time for accuracy. The oracle aggregation (pick the best of N, requires ground truth) gives the theoretical ceiling.

From the existing baseline, small models over-abstain significantly:
- Qwen 0.5B LoRA: student abstain rate 87% vs teacher 70% → majority vote may correct this
- Gemma 270M LoRA: similar over-abstention pattern

---

## Step 0 — Refactor shared utilities (do this first)

Before writing `07`, extract shared code from `05_model_evaluation.py` into a new module:

**Create `task1/scripts/eval_utils.py`** containing:
- `ABSTAIN_PATTERNS`
- `extract_answer()`
- `is_abstain()`
- `normalize_text()`, `normalize_for_comparison()`
- `EvalDataset`
- `load_model_and_tokenizer()`
- `compute_state_counts()`
- `load_qwen_embedding_model()`
- `compute_qwen_embeddings_batch()`
- `compute_embedding_similarities()`
- `compute_token_overlap()`
- `summarize_numeric()`
- `setup_logging()`
- `utc_timestamp()`

Then update `05_model_evaluation.py` to `from eval_utils import ...` and verify it still works:
```bash
.venv/bin/python task1/scripts/05_model_evaluation.py --help
```

This avoids the `importlib` hack that would otherwise be needed to import a file starting with `07_`.

---

## Step 1 — Script Architecture: `task1/scripts/07_tta_experiment.py`

### 1.1 New core functions

**`generate_n_answers(model, tokenizer, dataset, device, n, temperature, max_new_tokens, batch_size) → List[List[str]]`**

- Runs `n` generation passes over the full dataset
- For `n=1` and `temperature=0`: greedy (`do_sample=False`) — reproduces the `05` baseline exactly
- For `n>1` or `temperature>0`: `do_sample=True, temperature=temperature`
- Returns `raw_outputs[sample_idx][pass_idx]` — list-of-lists of raw decoded strings
- Loads the model **once**, loops `n` times — no reloading

**`select_centroid_answer(answer_texts: List[str], embedding_model_tuple) → Tuple[str, int]`**

- Takes a list of non-empty answer strings and the pre-loaded `(model, tokenizer, device)` tuple
- Embeds all texts with `compute_qwen_embeddings_batch()`
- Computes centroid = mean of all embeddings
- Returns `(best_text, best_idx)` — answer with highest cosine similarity to centroid

**`aggregate_majority_vote(outputs_n, embedding_model_tuple) → List[Dict]`**

Per sample:
1. Call `is_abstain(extract_answer(out))` for all N outputs
2. If `abstain_count > n / 2`: mark abstain, use first abstain output as representative
3. Else: call `select_centroid_answer()` on non-abstain answers, use that as representative
4. Returns dict with: `student_output`, `student_answer`, `student_abstain`, `vote_abstain_count`, `vote_answer_count`, `aggregation_method="majority_vote"`

**`aggregate_centroid(outputs_n, embedding_model_tuple) → List[Dict]`**

- Same abstain logic as majority_vote
- For the answer: always calls `select_centroid_answer()` (even if abstain_count == 0)
- `aggregation_method="centroid"`

**`aggregate_oracle(outputs_n, teacher_answers: List[str], embedding_model_tuple) → List[Dict]`**

- For each sample, pick the output with **highest embedding similarity to the teacher answer**
- For abstain cases: if teacher abstains, pick first output that also abstains; if none do, pick the one with highest similarity to a canonical abstain phrase
- `aggregation_method="oracle"`
- **Tag as analysis-only upper bound** — not usable in deployment (requires ground truth)

**`build_results_from_aggregated(aggregated, dataset, teacher_info, tta_n, tta_temperature) → List[Dict]`**

Assembles the final results list in the exact same schema as `05_model_evaluation.py`'s `generate_answers_batched()`, plus TTA-extra columns. The first 17 standard columns are identical — CSVs are drop-in compatible with `06_analyse_visualize_results.py`.

### 1.2 Click CLI

```python
@click.command()
@click.option("--models", multiple=True, required=True,
    help="Model config: 'path,name,type[,gen_batch_size]'. Repeat for multiple models.")
@click.option("--eval-dataset", default=..., type=click.Path(exists=True))
@click.option("--output-dir", default=EVAL_OUTPUT_DIR)
@click.option("--n-values", default="1,3,5",
    help="Comma-separated N values to sweep, e.g. '1,3,5,10'")
@click.option("--temperatures", default="0.7",
    help="Comma-separated temperatures, e.g. '0.5,0.7,1.0'")
@click.option("--aggregations", default="majority_vote,centroid,oracle",
    help="Comma-separated aggregation methods")
@click.option("--num-samples", default=0, type=int,
    help="Max samples (0 = all)")
@click.option("--gen-batch-size", default=8, type=int)
@click.option("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
@click.option("--mlflow-experiment", default="tta_experiment")
```

Inner loop order (most efficient — minimises model loads):
```
for model in models:
    load model once
    for temperature in temperatures:
        generate n=max(n_values) passes (reuse lower-N subsets)
        for n in n_values:
            for aggregation in aggregations:
                aggregate from first-n outputs
                compute metrics
                save CSV
                log to MLflow child run
    unload model
```

> Key optimisation: generate `max(N)` passes once per temperature, then derive N=3 and N=5 as subsets of N=10. This avoids redundant generation.

---

## Step 2 — Output File Structure

```
task1/outputs/evaluations/
└── tta_run_{timestamp}/
    ├── tta_run.log
    ├── tta_comparison_summary.csv       ← flat table for easy analysis
    │
    └── {model_name}/
        └── N{n}_T{temp}/
            ├── {model_name}_N{n}_T{temp}_majority_vote_detailed_results.csv
            ├── {model_name}_N{n}_T{temp}_centroid_detailed_results.csv
            ├── {model_name}_N{n}_T{temp}_oracle_detailed_results.csv
            └── {model_name}_N{n}_T{temp}_summary.json
```

### `detailed_results.csv` schema

Standard columns (identical to `05` — compatible with `06`):
`idx, data_source, teacher_id, query, search_index, llm_input, teacher_output, student_output, teacher_answer, student_answer, teacher_abstain, student_abstain, answer_state, exact_match_score, embedding_similarity, embedding_similarity_adjusted, token_overlap, decision_label`

TTA-extra columns (ignored by `06`, used by TTA analysis):
`tta_n, tta_temperature, tta_aggregation, vote_abstain_count, vote_answer_count`

### `tta_comparison_summary.csv` schema

Flat table, one row per (model × N × temperature × aggregation):
`model_name, tta_n, tta_temperature, aggregation, abstain_f1, abstain_precision, abstain_recall, abstain_accuracy, embedding_similarity_adjusted_avg, exact_match_avg, student_abstain_rate, teacher_abstain_rate`

### Model naming convention

Pass `{base_name}-tta-N{n}-T{temp_str}-{agg}` as the model name in the CSV — e.g. `Qwen2.5-0.5B-Instruct-lora-tta-N5-T07-centroid`. The `-lora` substring is preserved so `06`'s `get_train_type()` labels it correctly.

---

## Step 3 — MLflow Run Structure

```
Experiment: "tta_experiment"
└── Parent: "tta-job-{timestamp}"
    │  params: eval_dataset_sha256, models, n_values, temperatures, aggregations
    │
    └── Child: "tta-{model_name}-N{n}-T{temp}"  (one per model × N × temp)
        params:  model_name, model_path, model_type, n, temperature,
                 gen_batch_size, embedding_model, num_samples, eval_dataset_sha256
        metrics: abstain_f1_{agg}, abstain_precision_{agg}, abstain_recall_{agg},
                 abstain_accuracy_{agg}, embedding_similarity_adjusted_avg_{agg},
                 student_abstain_rate_{agg}, vote_abstain_rate_mean
                 (one set per aggregation method, suffixed)
        tags:    oracle_is_upper_bound=true
        artifacts: all CSVs under artifact_path="{model_name}/N{n}_T{temp}"
```

---

## Step 4 — SLURM Scripts

### `task1/scripts/slurm/tta_lumi.sh`

```
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00   (override via TTA_TIME env var)
```

Environment variables (all overridable before `sbatch`):
- `TTA_MODELS` — `--models` args string
- `TTA_N_VALUES` — default `"1,3,5"`
- `TTA_TEMPS` — default `"0.7"`
- `TTA_AGGS` — default `"majority_vote,centroid,oracle"`
- `TTA_SAMPLES` — default `0` (all)

### `task1/scripts/slurm/submit_tta_sweep.sh`

Submits one job per model with appropriate wall times. Supports `--dry-run`.

**Also update `sync_results.sh`** to include TTA files:
- `--include="*_N*_detailed_results.csv"`
- `--include="tta_comparison_summary.csv"`

---

## Step 5 — Sanity Check Gate

Before trusting any TTA results, verify that `N=1, temperature=0` (greedy) reproduces the `05` baseline:

> `N=1` Abstain F1 must match `eval_run_20260124_052629/` within **±0.002** for the same model.

If it doesn't match, something is wrong with the refactored utilities — fix before proceeding.

---

## Step 6 — Analysis

### Using `06` directly (free standard plots)

Feed any single TTA `detailed_results.csv` to `06` to get the full standard plot suite:
```bash
.venv/bin/python task1/scripts/06_analyse_visualize_results.py \
    --eval-run-dir task1/outputs/evaluations/tta_run_{timestamp}/{model_name}/N5_T07 \
    --output-dir task1/outputs/analysis/tta_N5_T07/
```

### TTA-specific plots (new, in `07` or a standalone `08_tta_analysis.py`)

**Plot A — F1 vs N (line chart)**
- X: N ∈ {1, 3, 5, 10}; Y: Abstain F1
- Lines: one per (model, aggregation)
- Shows where the curve flattens → practical sweet spot

**Plot B — Oracle vs practical methods (grouped bar at N=5)**
- Groups: baseline (N=1), majority_vote, centroid, oracle
- X: model; Y: Abstain F1
- Shows gap between ceiling and what's achievable

**Plot C — Temperature sensitivity**
- X: temperature; Y: Abstain F1 per aggregation
- Shows optimal temperature per model

**Plot D — Vote distribution (histogram)**
- X: `vote_abstain_count` (0..N); Y: frequency
- Faceted by model + N
- Confident models cluster near 0 or N; uncertain models cluster near N/2

**Plot E — Oracle gap**
- Oracle F1 − centroid F1 per model = remaining headroom
- If large: worth researching better selection methods (e.g. verifier model)

---

## Parameter Sweep and GPU Budget

### Recommended phased approach

**Phase A — Quick signal** (submit first, ~4 GPU-hours total):

| Model | N values | Temp | Est. wall time |
|-------|----------|------|----------------|
| Gemma 270M LoRA | 1, 3, 5 | 0.7 | ~2.5h |
| Qwen 0.5B LoRA | 1, 3, 5 | 0.7 | ~3h |

**Phase B — Temperature sweep** (only if Phase A shows improvement):

| Model | N values | Temps | Est. wall time |
|-------|----------|-------|----------------|
| Best Phase A model | 1, 3, 5 | 0.5, 1.0 | ~3h per temp |

**Phase C — Scaling check** (does TTA help larger models?):

| Model | N values | Temp | Est. wall time |
|-------|----------|------|----------------|
| Qwen 1.5B LoRA | 1, 5 | 0.7 | ~2.5h |

**Phase D — N=10** (only if curve hasn't flattened at N=5):

| Model | N | Temp | Est. wall time |
|-------|---|------|----------------|
| Best model | 10 | best temp | +1.5h |

> Get budget approval before Phase B+ (~50 GPU-hours for full sweep).

### GPU time estimates (1× MI250X GCD, `gen_batch_size=32`)

| Model | Time per generation pass (1886 samples) |
|-------|----------------------------------------|
| Gemma 270M | ~5 min |
| Qwen 0.5B | ~6 min |
| Qwen 1.5B | ~12 min |
| Embedding computation (any model) | ~1 min |

Phase A total: (5+6) × 5 passes × 1 temp + embedding overhead ≈ **~2h per model** on small-g.

---

## Expected Outcomes and Interpretation

| Outcome | Interpretation |
|---------|----------------|
| F1 improves with N for small models | TTA compensates for model capacity — use N>1 in deployment |
| Gain flattens by N=5 | N=5 is practical sweet spot; N=10 not worth compute cost |
| Oracle >> centroid | Outputs contain signal but centroid selection wastes it → research reranking |
| Centroid ≈ majority_vote | Embedding-based selection adds no value over simpler vote |
| No improvement for Qwen 1.5B | TTA benefit is size-dependent; larger models already confident |
| Student abstain rate shifts toward 70% | Majority vote corrects systematic over-abstention |

**Key hypothesis to falsify**: *"TTA helps most for models with the highest over-abstention rates"* (Qwen 0.5B abstains at 87% vs teacher 70%).

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Script runs end-to-end on 1886 samples | ✅ |
| All aggregations produce schema-valid `detailed_results.csv` | ✅ |
| `06_analyse_visualize_results.py` accepts TTA CSVs without modification | ✅ |
| MLflow shows parent + child runs with all params/metrics | ✅ |
| `N=1, T=0` reproduces `05` baseline within ±0.002 F1 | ✅ sanity check |
| Qwen 0.5B centroid (N=5, T=0.7) Abstain F1 > 0.815 | improvement confirmed |
| Oracle F1 > centroid F1 | upper bound confirmed |
| SLURM job completes within wall-time budget | ✅ |

---

## Files to Create / Modify

| Action | File |
|--------|------|
| **Create** | `task1/scripts/eval_utils.py` (Step 0 refactor) |
| **Modify** | `task1/scripts/05_model_evaluation.py` → `from eval_utils import ...` |
| **Create** | `task1/scripts/07_tta_experiment.py` |
| **Create** | `task1/scripts/slurm/tta_lumi.sh` |
| **Create** | `task1/scripts/slurm/submit_tta_sweep.sh` |
| **Modify** | `task1/scripts/slurm/sync_results.sh` → add TTA file patterns |
| **Update** | `PLAN.md` → mark 4.2 in progress |
| **Update** | `task1/RESULTS.md` → add TTA results section once numbers are in |

---

## Tools Note

If this project scales to dozens of models × datasets with more collaborators, consider replacing the SLURM scripts + DVC + MLflow combination with a purpose-built HPC sweep tool:

- **[Metaflow](https://metaflow.org/)** (Netflix) — Python-native workflow tool with built-in support for AWS Batch, Kubernetes, and HPC schedulers. Steps are Python functions decorated with `@step`; the scheduler is pluggable. Good fit if the team moves to a cloud-HPC hybrid.
- **[Hydra + submitit](https://github.com/facebookresearch/submitit)** (Facebook Research) — Hydra handles config composition and parameter sweeps (`--multirun`); submitit is a thin SLURM/LSF wrapper that makes `sbatch` blocking and returns futures. This is the closest to "DVC repro but for HPC" and is widely used in NLP research. Would replace `submit_train_sweep.sh` entirely.


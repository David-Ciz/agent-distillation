# Phase 6 — New Research Directions

> **Status**: Planned — not yet started  
> **Date created**: 2026-03-10  
> **Prerequisites**: Phase 5 (TTA) in progress — `07_tta_experiment.py` implemented, Phase A jobs pending.  
> **Origin**: Four directions requested by project stakeholders after Phase 5 implementation.  
> **Read before acting**: `plans/PHASE5_TTA.md`, `task1/RESULTS.md`, `plans/PLAN.md`

---

## Direction 1 — Newest Qwen Models

**Goal**: Evaluate whether the latest Qwen model generations (Qwen3 series) deliver better distillation performance than the Qwen 2.5 baseline, and at what parameter count the gains plateau.

### Context

The baseline uses Qwen **2.5** (0.5B, 1.5B, 3B, 7B). Qwen3 models are now available on HuggingFace. The key question is whether a Qwen3-0.6B outperforms Qwen2.5-3B at a fraction of the compute cost — which would directly lower the inference cost of the deployed student.

### What to do

1. **Identify the relevant Qwen3 models** — check HuggingFace for the latest instruct-tuned Qwen3 checkpoints. Likely candidates at time of writing:
   - `Qwen/Qwen3-0.6B-Instruct` (or equivalent)
   - `Qwen/Qwen3-1.7B-Instruct`
   - `Qwen/Qwen3-4B-Instruct`
   - `Qwen/Qwen3-8B-Instruct`
   - Verify exact model IDs on HuggingFace before running.

2. **Training**: Use the existing LoRA pipeline (`02_train_model_lora.py` + `train_lora_lumi.sh`). Training config should be identical to the Qwen 2.5 runs — 3 epochs, same hyperparameters — so results are directly comparable.

3. **Evaluation**: Run `05_model_evaluation.py` with the same eval dataset (`task1_eval_dataset.csv`, 1886 samples). Log to a new MLflow experiment: `"qwen3_evaluation"`.

4. **Comparison**: Add Qwen3 rows to the comparison table in `task1/RESULTS.md`, alongside Qwen 2.5 equivalents at similar parameter counts. Plot `Abstain F1 vs parameter count` for both generations to visualise the efficiency gain (or lack thereof).

5. **SLURM**: Add new model entries to `submit_train_sweep.sh` and `submit_eval_sweep.sh`. Consider a new `submit_qwen3_sweep.sh` to keep the existing sweep scripts stable.

### Success Criteria

| Criterion | Target |
|-----------|--------|
| At least 2 Qwen3 models trained and evaluated | ✅ |
| Results logged in MLflow under `qwen3_evaluation` | ✅ |
| Comparison table updated in `task1/RESULTS.md` | ✅ |
| `Abstain F1 vs params` plot comparing Qwen2.5 and Qwen3 | ✅ |

---

## Direction 2 — Model Collapse Detection

**Goal**: Determine whether any trained models have collapsed — i.e. degenerated into a mode where they output the same (or near-identical) text regardless of input — and build a routine check into the evaluation pipeline.

### Context

Model collapse can be subtle: a model that always outputs the canonical abstain phrase ("I cannot answer based on the provided evidence.") will score a high student abstain rate but near-zero `both_answer` rate. It may also show suspiciously low perplexity variance across samples.

From the baseline, Qwen 0.5B LoRA already abstains at 87% vs the teacher's 70%, which is a yellow flag. A collapsed model would push this even higher (95–100%).

### Signals to check

| Signal | Healthy range | Collapse indicator |
|--------|--------------|-------------------|
| `student_abstain_rate` | 60–80% | > 90% |
| `both_answer_rate` | > 20% | < 5% |
| Unique student outputs (dedup ratio) | > 80% unique | < 20% unique |
| Output length variance (std dev of token count) | high | near-zero |
| Perplexity across samples (std dev) | > 1.0 | < 0.1 |

### What to do

1. **Add a collapse detection function** to `eval_utils.py`:

   ```python
   def detect_collapse(results: List[Dict], threshold_abstain: float = 0.90,
                       threshold_unique: float = 0.20) -> Dict:
       """
       Returns a dict with:
         - is_collapsed: bool
         - abstain_rate: float
         - unique_output_ratio: float  (len(set(outputs)) / len(outputs))
         - output_length_std: float
         - warnings: List[str]
       """
   ```

2. **Call it inside `evaluate_model()`** in `05_model_evaluation.py` and log the result as an MLflow tag: `collapse_detected=true/false`.

3. **Surface it in `06_analyse_visualize_results.py`**: print a prominent warning in the console and add a `collapse_detected` column to `model_comparison_summary.csv`.

4. **Retroactively check all existing eval runs**: write a small one-off script `task1/scripts/check_collapse.py` that reads `*_detailed_results.csv` files and reports any models that trip the thresholds. Run it against `task1/outputs/evaluations/` immediately.

5. **Add a collapse section to `task1/RESULTS.md`** with the findings.

### Files to modify / create

| Action | File |
|--------|------|
| **Modify** | `task1/scripts/eval_utils.py` — add `detect_collapse()` |
| **Modify** | `task1/scripts/05_model_evaluation.py` — call it, log to MLflow |
| **Modify** | `task1/scripts/06_analyse_visualize_results.py` — warn + add column |
| **Create** | `task1/scripts/check_collapse.py` — one-off retroactive check |
| **Update** | `task1/RESULTS.md` — collapse findings section |
| **Update** | `docs/task1/evaluation_metrics.md` — document collapse signals |

---

## Direction 3 — TTA Results

**Goal**: Execute Phase 5 (TTA) on LUMI, collect results, and write up the findings.

### Context

`07_tta_experiment.py` is fully implemented (Phase 5). The Phase A SLURM jobs have not yet been submitted. This direction is a reminder to:

1. **Submit Phase A** on LUMI once a GPU allocation is available:
   ```bash
   bash task1/scripts/slurm/submit_tta_sweep.sh --phase a
   ```
   Targets: Qwen 0.5B LoRA + Gemma 270M LoRA, N ∈ {1,3,5}, T=0.7, all aggregations.

2. **Run the sanity check first** (N=1, T=0 must reproduce `05` baseline within ±0.002 F1):
   ```bash
   bash task1/scripts/slurm/submit_tta_sweep.sh --phase sanity
   ```

3. **Sync and analyse** after jobs finish:
   ```bash
   bash task1/scripts/slurm/sync_results.sh
   # Open tta_comparison_summary.csv, check F1 vs N trend
   ```

4. **Decide on Phase B** (temperature sweep) based on Phase A results. Only proceed if Phase A shows improvement in Abstain F1 over the single-pass baseline. See `plans/PHASE5_TTA.md § Parameter Sweep` for the full decision tree.

5. **Write up results** in `task1/RESULTS.md` under the TTA section (currently says "Pending Phase A completion").

6. **Update docs**: fill in `docs/task1/results.md` TTA row and add actual numbers to `docs/tta/interpreting_results.md`.

### Key hypothesis to confirm or falsify

> *TTA helps most for models with the highest over-abstention rates (Qwen 0.5B at 87% → target: closer to 70% teacher rate with N=5, T=0.7).*

See `plans/PHASE5_TTA.md § Expected Outcomes` for the full interpretation guide.

---

## Direction 4 — New Datasets from Traces

**Goal**: Extend the training and evaluation data by generating additional datasets from teacher traces on new domains or retrieval indices, then measure whether broader coverage improves student generalisation.

### Context

The current `task1_dataset.csv` is built from traces over three source datasets:

| Source | Type |
|--------|------|
| `causalqa` | Causal QA — why/how questions |
| `msmarco` | Web search queries |
| `quasart` | Open-domain factoid QA |

All traces were generated by running the agentic teacher (GPT-4o with retrieval) over these datasets. The teacher traces are in `task1/data/synthetic_traces/`.

New datasets can come from two directions:
- **New domains**: run the teacher over a new QA benchmark (e.g. NaturalQuestions, TriviaQA, BioASQ) and add those traces.
- **New trace format**: the detailed traces in `task1/data/synthetic_traces/agentic_search_detailed_traces/` include intermediate reasoning steps — these have not been used for training yet. A dataset built from detailed traces might produce students that reason more transparently.

### What to do

1. **Audit existing traces**: check `task1/data/synthetic_traces/` for any trace files that haven't been converted into training samples yet (especially `agentic_search_detailed_traces/`).

2. **Extend `01_create_train_dataset.py`** to support a `--source` flag for selecting which trace directories to include, and a `--trace-format` flag (`simple` vs `detailed`).

3. **Generate a `detailed_traces` dataset variant**: run `01_create_train_dataset.py --trace-format detailed` and produce `task1_dataset_detailed.csv`. Compare training on this dataset vs the existing `task1_dataset.csv` — does reasoning chain supervision help?

4. **New domain traces** (if GPU budget allows): write a `00_generate_teacher_traces.py` script that runs the teacher agent over a new QA benchmark. Target: 5k–10k new samples per domain. Start with one domain to validate the pipeline before scaling.

5. **Train and evaluate** using the new datasets: use the same LoRA pipeline but point `--train_data` at the new CSV. Evaluate on the existing `task1_eval_dataset.csv` (keep eval fixed for comparability) **and** on a held-out split of the new domain.

6. **Track everything in MLflow** under a new experiment: `"dataset_ablation"`.

### Suggested priority order

```
Step 1: audit existing traces (< 1h, no GPU)
Step 2: detailed traces dataset + training (use existing LUMI allocation)
Step 3: new domain traces (requires new teacher API calls — check budget)
```

### Files to modify / create

| Action | File |
|--------|------|
| **Modify** | `task1/scripts/01_create_train_dataset.py` — `--trace-format`, `--source` flags |
| **Create** | `task1/scripts/00_generate_teacher_traces.py` — new domain trace generation |
| **Update** | `task1/RESULTS.md` — dataset ablation section |
| **Update** | `docs/task1/pipeline.md` — document new step 0 and dataset variants |

---

## Recommended Execution Order

Given GPU budget constraints and dependencies, the suggested order is:

```
1. Direction 3 (TTA)         — submit Phase A immediately, no new code needed
2. Direction 2 (Collapse)    — add detect_collapse() before any new training run
3. Direction 4 (Datasets)    — audit traces first (free), then detailed-trace training
4. Direction 1 (Qwen3)       — run after baseline is stable and collapse checks pass
```

Directions 1, 2, and 4 can proceed in parallel on LUMI once the collapse check is wired in.

---

## Open Questions

- What is the GPU budget for Direction 4 (teacher trace generation)? GPT-4o API costs need sign-off.
- Are there any existing detailed-trace training experiments in the literature to benchmark against for Direction 4?
- Should Qwen3 evaluation (Direction 1) also include TTA, or single-pass only first?
- Is model collapse defined as an absolute threshold or relative to the teacher abstain rate?


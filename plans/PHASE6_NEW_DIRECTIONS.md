# Phase 6 — New Research Directions

> **Status**: Planned — not yet started  
> **Date created**: 2026-03-10  
> **Prerequisites**: Phase 5 (TTA) in progress — `07_tta_experiment.py` implemented, Phase A jobs pending.  
> **Origin**: Four directions requested by project stakeholders after Phase 5 implementation.  
> **Read before acting**: `plans/PHASE5_TTA.md`, `task1/RESULTS.md`, `plans/PLAN.md`

---

## Direction 1 — Newest Qwen Models

**Goal**: Evaluate whether the latest Qwen model generations (Qwen3/Qwen3.5 series) deliver better distillation performance than the Qwen 2.5 baseline, and at what parameter count the gains plateau.

### Context

The baseline uses Qwen **2.5** (0.5B, 1.5B, 3B, 7B). Newer Qwen3-family models are now available on HuggingFace. The key question is whether a Qwen3.5-0.8B outperforms Qwen2.5-3B at a fraction of the compute cost — which would directly lower the inference cost of the deployed student.

### What to do

1. **Identify the relevant Qwen3-family models** — check HuggingFace for the latest checkpoints that fit the existing text pipeline. Selected candidates as of 2026-03-12:
   - `Qwen/Qwen3.5-0.8B`
   - `Qwen/Qwen3.5-2B`
   - `Qwen/Qwen3.5-4B`
   - Defer `Qwen/Qwen3.5-9B` until after the first pass; it is better used as a later ceiling model.
   - `Qwen/Qwen3.5-35B-A3B` is a low-priority future experiment because it
     requires sharded training / inference rather than the current DDP pipeline.

2. **Training**: Use the existing LoRA pipeline (`02_train_model_lora.py` + `train_lora_lumi.sh`). Training config should be identical to the Qwen 2.5 runs — 3 epochs, same hyperparameters — so results are directly comparable.

3. **Evaluation**: Run `05_model_evaluation.py` with the same eval dataset (`task1_eval_dataset.csv`, 1886 samples). Log to a new MLflow experiment: `"qwen3_evaluation"`.

4. **Comparison**: Add Qwen3 rows to the comparison table in `task1/RESULTS.md`, alongside Qwen 2.5 equivalents at similar parameter counts. Plot `Abstain F1 vs parameter count` for both generations to visualise the efficiency gain (or lack thereof).

5. **SLURM**: Use dedicated Qwen3.5 sweep scripts to keep the baseline sweep stable:
   - `task1/scripts/slurm/submit_qwen35_train_sweep.sh`
   - `task1/scripts/slurm/submit_qwen35_eval_sweep.sh`

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

**Goal**: As new batches of teacher traces are received, build and track new training dataset variants from them, then measure whether different trace sources / formats improve student generalisation.

### Context

Traces are **received externally** — a separate team runs the teacher agent (GPT-4o with retrieval) against QA benchmarks using their own tooling and drops the trace files into `task1/data/synthetic_traces/`. We do not generate traces ourselves and there is no need for a trace-generation script here.

The current `task1_dataset.csv` is built from three source directories:

| Source directory | Domain type |
|-----------------|-------------|
| `agentic_search/agentic_teacher_causalqa*` | Causal QA — why/how questions |
| `agentic_search/agentic_teacher_msmarco*` | Web search queries |
| `agentic_search/agentic_teacher_quasart*` | Open-domain factoid QA |

Two additional trace formats are already present but **not yet used for training**:

| Directory | What it contains |
|-----------|-----------------|
| `agentic_search_detailed_traces/` | Same queries but with intermediate reasoning steps included in the output — richer supervision signal |
| `causalqa/`, `msmarco/`, `quasart/` | Simpler non-agentic traces (no retrieval loop) — potentially a cleaner signal for the abstain decision |

When new trace batches arrive they will land in new subdirectories under `task1/data/synthetic_traces/`. The workflow is: receive traces → run `01_create_train_dataset.py` with the appropriate flags → train → evaluate → compare.

### What to do

1. **Audit existing traces**: document exactly which subdirectories of `task1/data/synthetic_traces/` exist, what format they use, and which have already been included in `task1_dataset.csv`. Add a `task1/data/TRACES.md` manifest file that lists each directory, its domain, trace format, sample count, and whether it has been used.

2. **Extend `01_create_train_dataset.py`** with two new flags:
   - `--source` — one or more trace subdirectory paths to include (repeatable, like `--models` in eval). Defaults to the current set.
   - `--trace-format` — `simple` (current behaviour) or `detailed` (includes reasoning chain in the target output). Selects the parser branch inside the script.
   - The output CSV filename should encode the source selection, e.g. `task1_dataset_detailed_causalqa_msmarco.csv`, so different variants are clearly distinguishable on disk.

3. **Build a `detailed` dataset variant** first (no new traces needed, uses existing `agentic_search_detailed_traces/`):
   ```bash
   python task1/scripts/01_create_train_dataset.py \
       --source task1/data/synthetic_traces/agentic_search_detailed_traces \
       --trace-format detailed \
       --output task1/data/task1_dataset_detailed.csv
   ```
   Train with the same LoRA config and evaluate on `task1_eval_dataset.csv`. Does reasoning chain supervision improve Abstain F1?

4. **When new trace batches arrive**: drop them into `task1/data/synthetic_traces/<new_dir>/`, update `task1/data/TRACES.md`, then build a new dataset variant and add a row to the ablation table.

5. **Track everything in MLflow** under a new experiment: `"dataset_ablation"`. Log the source directories and trace format as MLflow params so every training run is fully reproducible from the DB alone.

6. **Keep `task1_eval_dataset.csv` fixed** across all variants so all results are directly comparable.

### Files to modify / create

| Action | File |
|--------|------|
| **Create** | `task1/data/TRACES.md` — manifest of all trace directories |
| **Modify** | `task1/scripts/01_create_train_dataset.py` — `--source`, `--trace-format`, `--output` flags |
| **Update** | `task1/RESULTS.md` — dataset ablation section |
| **Update** | `docs/task1/pipeline.md` — document the new flags and TRACES.md workflow |

---

## Recommended Execution Order

Given GPU budget constraints and dependencies, the suggested order is:

```
1. Direction 3 (TTA)         — submit Phase A immediately, no new code needed
2. Direction 2 (Collapse)    — add detect_collapse() before any new training run
3. Direction 4 (Datasets)    — audit traces + write TRACES.md (free, no GPU),
                               then build detailed variant + train
4. Direction 1 (Qwen3)       — run after baseline is stable and collapse checks pass
```

Directions 1, 2, and 4 can proceed in parallel on LUMI once the collapse check is wired in.

---

## Open Questions

- Are there any existing detailed-trace training experiments in the literature to benchmark against for Direction 4?
- Should Qwen3 evaluation (Direction 1) also include TTA, or single-pass only first?
- Is model collapse defined as an absolute threshold or relative to the teacher abstain rate?
- What cadence will new trace batches arrive on, and which domains are planned?

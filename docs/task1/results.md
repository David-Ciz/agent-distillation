# Results

All models evaluated on **1,886 samples** from the evaluation dataset.  
Teacher model: **GPT-4o** (abstain rate: 70.0%).

---

## LUMI Reproduction Run (March 2026)

Re-trained and re-evaluated on LUMI using the full MLflow-tracked pipeline.

| Model | Train Type | Abstain F1 | Abstain Acc | Embed Sim Adj | Student Abstain Rate |
|-------|------------|-----------|-------------|---------------|----------------------|
| Qwen 2.5 3B | **LoRA** | **0.875** | 0.815 | **0.802** | 0.781 |
| Qwen 2.5 3B | Base | 0.820 | 0.752 | 0.737 | 0.677 |

Delta vs original baseline (Qwen 3B LoRA): −0.006 F1, −0.008 Embed Sim → **within expected variance ✅**

---

## Current Comparison

There are now two relevant synced evaluation checkpoints:

- `task1/outputs/evaluations/eval_run_20260324_173233`: broad family comparison across current `Qwen2.5` and `Qwen3.5`
- `task1/outputs/evaluations/eval_run_20260326_154836`: targeted rerun after restoring the legacy `SFTTrainer(formatting_func=...)` path and evaluating with a larger generation cap

### Targeted Rerun (March 26, 2026)

This rerun is the strongest current evidence for the training rollback.

| Model | Prior Current Run | March 26 Rerun | Reading |
|-------|-------------------|----------------|---------|
| Qwen 2.5 0.5B LoRA | `0.436 / 0.207 / 0.315` | `0.453 / 0.292 / 0.368` | Better than the broken March 24 run, but still weak |
| Qwen 2.5 3B LoRA | `0.828 / 0.664 / 0.814` | `0.819 / 0.662 / 0.807` | Essentially flat; good control result |
| Qwen 3.5 4B LoRA | `0.491 / 0.199 / 0.413` | `0.886 / 0.673 / 0.857` | Major recovery; now the strongest current rerun |

Values are `Abstain Agreement / Exact Match / Embed Sim Adj`.

What this says:

- **Qwen 3.5 4B LoRA recovered strongly** after the training rollback and corrected evaluation settings
- **Qwen 2.5 3B LoRA stayed stable**, which makes it a useful control
- **Qwen 2.5 0.5B LoRA improved, but is still not a strong candidate**

### Broad Family Snapshot (March 24, 2026)

Latest full-family synced evaluation run: `task1/outputs/evaluations/eval_run_20260324_173233`.

This is the side-by-side comparison that matters:

- archived pre-LUMI `Qwen2.5`
- current LUMI `Qwen2.5`
- current LUMI `Qwen3.5`

Values are `Abstain Agreement / Exact Match / Embed Sim Adj`.

Note: `Qwen3.5` does not match `Qwen2.5` sizes exactly, so the comparison is by nearest size tier: `0.5B~0.8B`, `1.5B~2B`, `3B~4B`, `7B~9B`.

| Size Tier | Archived Qwen 2.5 LoRA | Current Qwen 2.5 LoRA | Current Qwen 3.5 LoRA | Reading |
|-----------|--------------------------|------------------------|------------------------|---------|
| 0.5B / 0.8B | Qwen 2.5 0.5B: `0.710 / 0.642 / 0.701` | Qwen 2.5 0.5B: `0.436 / 0.207 / 0.315` | Qwen 3.5 0.8B: `0.805 / 0.655 / 0.792` | Current `Qwen3.5` clearly wins; current `Qwen2.5-0.5B` is broken / unreliable |
| 1.5B / 2B | Qwen 2.5 1.5B: `0.768 / 0.645 / 0.757` | Qwen 2.5 1.5B: `0.690 / 0.615 / 0.675` | Qwen 3.5 2B: `0.853 / 0.676 / 0.839` | In the March 24 full sweep, `Qwen3.5-2B` was the strongest clean result |
| 3B / 4B | Qwen 2.5 3B: `0.823 / 0.674 / 0.810` | Qwen 2.5 3B: `0.828 / 0.664 / 0.814` | Qwen 3.5 4B: `0.491 / 0.199 / 0.413` | This row is outdated as a verdict; March 26 rerun recovered `Qwen3.5-4B` substantially |
| 7B / 9B | Qwen 2.5 7B: `0.865 / 0.676 / 0.851` | Qwen 2.5 7B: `0.849 / 0.669 / 0.836` | Qwen 3.5 9B: `0.484 / 0.230 / 0.398` | `Qwen2.5-7B` remains strong; current `Qwen3.5-9B` is still suspect |

### What this says

- **Best current rerun so far**: Qwen 3.5 4B LoRA
- **Best current Qwen 2.5 model**: Qwen 2.5 7B LoRA
- **Best archived Qwen 2.5 model**: Qwen 2.5 7B LoRA
- **Most consistent family in the March 24 broad sweep**: Qwen 2.5

So the answer to "is Qwen 2.5 still best?" is:

- **Against the archived baselines, most current Qwen 2.5 reruns are still worse**
- **In the last broad family sweep, Qwen 2.5 was still more consistent than Qwen 3.5**
- **In the latest targeted rerun, Qwen 3.5 4B LoRA is the best current result**

---

## Full Baseline (all models)

| Model | Train Type | Abstain F1 | Abstain Acc | Embed Sim Adj | Student Abstain Rate |
|-------|------------|-----------|-------------|---------------|----------------------|
| Qwen 2.5 7B | LoRA | **0.891** | 0.832 | 0.821 | 0.774 |
| Qwen 2.5 3B | LoRA | 0.881 | 0.823 | 0.810 | 0.781 |
| Qwen 2.5 3B | Full Finetune | 0.862 | 0.808 | 0.793 | 0.764 |
| Qwen 2.5 1.5B | LoRA | 0.855 | 0.801 | 0.789 | 0.768 |
| Qwen 2.5 0.5B | LoRA | 0.815 | 0.763 | 0.756 | **0.870** |
| Qwen 2.5 0.5B | Full Finetune | 0.802 | 0.749 | 0.741 | 0.847 |
| Gemma 270M | LoRA | 0.773 | 0.719 | 0.701 | 0.823 |
| Qwen 2.5 7B | Base | 0.698 | 0.631 | 0.612 | 0.521 |
| Qwen 2.5 3B | Base | 0.656 | 0.597 | 0.578 | 0.498 |
| Qwen 2.5 0.5B | Base | 0.452 | 0.391 | 0.372 | 0.213 |
| Gemma 270M | Base | 0.401 | 0.347 | 0.329 | 0.178 |

---

## Key Findings

### Training dramatically improves performance

Average Abstain F1 improvement after training:

- Small models (270M–0.5B): **+70–100%** over base
- Larger models (3B–7B): **+25–35%** over base

### LoRA consistently beats Full Finetune

For both Qwen 0.5B and 3B, LoRA achieves higher Abstain F1 than full fine-tuning, likely because the smaller adapter overfits less on the limited training data.

### Small models over-abstain systematically

| Model | Teacher abstain rate | Student abstain rate | Over-abstention |
|-------|---------------------|---------------------|-----------------|
| Qwen 0.5B LoRA | 70% | **87%** | +17pp |
| Gemma 270M LoRA | 70% | **82%** | +12pp |
| Qwen 1.5B LoRA | 70% | 77% | +7pp |
| Qwen 3B LoRA | 70% | 78% | +8pp |

This is the key motivation for the [TTA experiment](../tta/what_is_tta.md) — majority voting across N passes should pull the abstain rate back towards 70%.

### Failure-mode taxonomy is useful

Using the archived pre-LUMI evaluation run, every sample can be placed into one of four categories:

- `agreement_answer`
- `agreement_abstain`
- `false_abstention`
- `false_confidence`

This split is informative rather than cosmetic:

- Small **base** models show much higher `false_confidence` than trained models.
- Training sharply reduces `false_confidence`, especially for the Qwen 3B and 7B models.
- The tradeoff is `false_abstention`: smaller trained models become more conservative and over-abstain.

Examples from the pre-LUMI archive:

| Model | False Confidence | False Abstention |
|-------|------------------|------------------|
| Qwen 2.5 0.5B Base | **53.5%** | 4.7% |
| Qwen 2.5 0.5B LoRA | 5.9% | **23.1%** |
| Qwen 2.5 3B Base | 16.3% | 9.2% |
| Qwen 2.5 3B LoRA | **4.7%** | 13.0% |
| Qwen 2.5 7B LoRA | **4.6%** | 8.9% |

Interpretation:

- `false_confidence` is the dangerous failure mode because it corresponds to answering when the teacher abstains.
- `false_abstention` is a capacity/conservatism problem: the model refuses too often, but it is safer.
- The strongest current balance is in the Qwen 3B LoRA and Qwen 7B LoRA models.

### QA signal validation passes the sanity check

We also joined the failure-mode labels with the per-sample evaluation metrics to test whether the categories track answer quality.

What holds across all 12 archived pre-LUMI models:

- `agreement_answer` has much higher quality than `false_confidence`
- the separation is statistically overwhelming under a one-sided Mann-Whitney U test
- this supports the core routing intuition: the failure-mode signal is not random noise

This now holds not only for `embedding_similarity_adjusted`, but also for the more independent metrics:

- raw embedding similarity
- token overlap
- exact match score

Observed result:

- `false_confidence` is worse than `agreement_answer` on raw embedding similarity for all 12 models
- `false_confidence` is worse than `agreement_answer` on token overlap for all 12 models
- `false_confidence` is worse on exact match for 11 of 12 models

Important caveat:

- the exact-match signal is weak for the weakest model (`gemma-3-270m-it-base`) because both categories are effectively at zero there
- `embedding_similarity_adjusted` still should not be treated as the primary evidence, because abstain mismatches are built into that metric

So the current evidence is materially stronger than the first sanity check: the failure-mode taxonomy appears to predict degraded answer quality even under metrics that do not directly encode abstain mismatches.

The next refinement, if needed, is descriptive rather than foundational:

- answer length by category
- per-category plots for the strongest routing candidates
- validation on the fresh LUMI runs as they arrive

### Routing simulation is positive on the archived run

We then simulated five routing strategies over the archived pre-LUMI Qwen LoRA family (`0.5B -> 1.5B -> 3B -> 7B`).

| Strategy | Avg Cost (B params) | Embed Sim Adj | Abstain F1 |
|----------|---------------------|---------------|------------|
| always_small | 0.50 | 0.701 | 0.815 |
| always_large | 7.00 | **0.851** | **0.907** |
| blind_cascade | 8.66 | 0.768 | 0.841 |
| qa_routing | **0.68** | 0.757 | 0.856 |
| qa_ensemble | 12.00 | 0.830 | 0.894 |

Interpretation:

- `qa_routing` is substantially better than `always_small` while staying far cheaper than `always_large`
- `qa_routing` also beats the naive abstention-based `blind_cascade` on cost by a large margin
- On the archived run, `qa_routing` sits above the line between `always_small` and `always_large`, which is the positive result the routing paper needs

Important caveat:

- the current `qa_routing` simulation is still an **oracle upper bound**
- it uses the labelled `false_confidence` category directly, not a deployable learned classifier

So this result supports the routing direction, but the next step is still required:

- compare against a random-routing baseline
- replace the oracle decision with a thresholded or learned routing signal

### Random-routing ablation supports the signal claim

We also replaced the QA-based escalation decision with a random escalation decision that preserves the same stage-wise escalation architecture and approximately the same average cost.

Result over 20 random seeds:

| Metric | Random Routing Mean ± Std | Oracle QA Routing |
|--------|----------------------------|-------------------|
| Avg cost | 0.687 ± 0.025 | **0.682** |
| Embed Sim Adj | 0.706 ± 0.003 | **0.757** |
| Raw Embed Sim | 0.827 ± 0.001 | **0.851** |
| Token Overlap | 0.570 ± 0.002 | **0.621** |
| Exact Match | 0.643 ± 0.002 | **0.698** |
| Abstain F1 | 0.818 ± 0.002 | **0.856** |

Interpretation:

- the random baseline uses essentially the same budget as oracle QA routing
- despite that, oracle QA routing wins by a wide margin on every quality metric
- this is the strongest current evidence that the gain comes from the QA signal itself, not just from adding a routing structure

### A first non-oracle routing score works

We then trained a simple false-confidence classifier for the Qwen LoRA routing chain using only model-side features available at inference time:

- student answer text
- abstain flag
- answer length features

The classifier was evaluated using out-of-fold predictions on the archived pre-LUMI run.

Stage-wise classifier quality:

| Model | ROC AUC | Average Precision |
|-------|---------|-------------------|
| Qwen 0.5B LoRA | **0.969** | 0.534 |
| Qwen 1.5B LoRA | 0.943 | 0.412 |
| Qwen 3B LoRA | 0.917 | 0.245 |

Threshold sweep result:

- the sweep shows a broad stable region rather than a brittle single-point optimum
- the best quality is around threshold `0.6`
- at threshold `0.6`, average cost is `1.16` B params, adjusted similarity is `0.738`, and Abstain F1 is `0.844`
- at threshold `0.85`, average cost is `0.697` B params, adjusted similarity is `0.732`, and Abstain F1 is `0.839`

Interpretation:

- the learned router does not match the oracle router yet
- but it still produces a non-oracle cost-quality tradeoff that is comfortably above the `always_small` baseline
- the threshold sweep is fairly flat across a useful range, which is a good sign for robustness

Important caveat:

- these scores are out-of-fold on the archived pre-LUMI split, not a fresh held-out LUMI validation set
- so this is a strong prototype result, but not yet the final deployable claim

---

## Visualisations

All plots are committed under `task1/outputs/analysis/`.

### Answer State Distribution
Shows the proportion of `both_abstain`, `both_answer`, and mismatch states per model.  
![Answer State Distribution](../assets/answer_state_distribution.png)

### Metrics Heatmap
All models vs all metrics — useful for spotting outliers at a glance.  
![Metrics Heatmap](../assets/metrics_heatmap.png)

### Improvement Rate by Training Method
Percentage improvement from base model after LoRA / full fine-tune.  
![Improvement Rate](../assets/improvement_rate_by_training.png)

---

## TTA Results

> Pending Phase A completion. See [Interpreting TTA Results](../tta/interpreting_results.md) for what to expect.

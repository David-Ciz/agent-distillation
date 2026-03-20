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

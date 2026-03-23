# Task 1: Answer-Abstain QA — Results Analysis

This document provides a detailed analysis of the evaluation results for Task 1 (Answer-Abstain QA). All models were evaluated on 1,886 samples from the evaluation dataset.

---

## LUMI Reproduction Run (March 2026)

New training + evaluation runs on LUMI using the MLflow-tracked pipeline. Models trained from scratch using `02_train_model_lora.py` (3 epochs, 8× MI250X), evaluated with `05_model_evaluation.py`.

| Model | Train Type | Abstain F1 | Abstain Acc | Embed Sim Adj | Student Abstain Rate | MLflow Run |
|-------|------------|------------|-------------|---------------|----------------------|------------|
| Qwen 2.5 3B Instruct | **LoRA** | **0.875** | 0.815 | **0.802** | 0.781 | eval-job-20260309_114742 |
| Qwen 2.5 3B Instruct | Base | 0.820 | 0.752 | 0.737 | 0.677 | eval-job-20260309_114742 |

**Teacher abstain rate**: 70.0% (699/1886 samples)

**Comparison to original baseline** (Qwen 3B LoRA):
- Original: Abstain F1 = 0.881, Embed Sim = 0.810
- LUMI run:  Abstain F1 = 0.875, Embed Sim = 0.802
- Delta: −0.006 F1, −0.008 Embed Sim → **within expected variance ✅**

**Pending**: Qwen 2.5 0.5B LoRA (Step 4.4) and Qwen 2.5 3B Full Finetune (Step 4.5).

---

## Overview

We trained and evaluated 12 model configurations across 5 model families:

| Model Family | Parameters | Training Methods |
|--------------|------------|------------------|
| Gemma 3 270M IT | 270M | Base, LoRA |
| Qwen 2.5 0.5B Instruct | 0.5B | Base, LoRA, Full Finetune |
| Qwen 2.5 1.5B Instruct | 1.5B | Base, LoRA |
| Qwen 2.5 3B Instruct | 3B | Base, LoRA, Full Finetune |
| Qwen 2.5 7B Instruct | 7B | Base, LoRA |

**Teacher model**: GPT-4o (abstain rate: 70.0%)

---

## Key Findings

### 1. Training Significantly Improves Performance

All trained models (LoRA and Full Finetune) show substantial improvements over their base counterparts:

- **Abstain F1**: Average improvement of +20-125% after training
- **Embedding Similarity**: Average improvement of +10-113% after training
- **Exact Match Rate**: Improvements up to +283% for smaller models

### 2. LoRA vs Full Finetune

For models where both methods were tested (Qwen 0.5B and 3B):

| Model | LoRA F1 | Full Finetune F1 | Winner |
|-------|---------|------------------|--------|
| Qwen 2.5 0.5B | 0.815 | 0.802 | LoRA |
| Qwen 2.5 3B | 0.881 | 0.862 | LoRA |

**Observation**: LoRA slightly outperforms Full Finetune while being more memory-efficient.

### 3. Model Size Matters

Larger models consistently achieve better performance:

| Model | Base F1 | LoRA F1 | Improvement |
|-------|---------|---------|-------------|
| Gemma 270M | 0.675 | 0.814 | +20.6% |
| Qwen 0.5B | 0.362 | 0.815 | +125.5% |
| Qwen 1.5B | 0.800 | 0.845 | +5.6% |
| Qwen 3B | 0.808 | 0.881 | +9.0% |
| Qwen 7B | 0.882 | 0.907 | +2.8% |

**Observation**: Smaller models benefit more from training (higher relative improvement), but larger models achieve higher absolute performance.

### 4. Best Performing Model

**Qwen 2.5 7B Instruct LoRA** achieves the best overall performance:
- Abstain F1: **0.907**
- Abstain Accuracy: **86.5%**
- Embedding Similarity: **0.851**
- Exact Match Rate: **67.6%**

### 5. Failure-Mode Taxonomy Exposes the Real Tradeoff

The archived pre-LUMI evaluation run was relabelled with a four-way taxonomy:

- `agreement_answer`
- `agreement_abstain`
- `false_abstention`
- `false_confidence`

This gives a better view of model behaviour than Abstain F1 alone.

| Model | False Confidence | False Abstention | Reading |
|-------|------------------|------------------|---------|
| Qwen 2.5 0.5B Base | **53.5%** | 4.7% | Unacceptably over-confident |
| Qwen 2.5 0.5B LoRA | 5.9% | **23.1%** | Safer, but heavily over-abstains |
| Qwen 2.5 3B Base | 16.3% | 9.2% | Still too willing to answer |
| Qwen 2.5 3B LoRA | **4.7%** | 13.0% | Strong balance |
| Qwen 2.5 7B Base | **2.4%** | 15.7% | Very conservative |
| Qwen 2.5 7B LoRA | **4.6%** | 8.9% | Strong balance, better answer agreement |

What this means:

- **Training reduces dangerous false confidence** very strongly, especially for 0.5B, 1.5B, and 3B models.
- **Smaller trained models trade false confidence for false abstention** — they become safer but more conservative.
- **Qwen 2.5 3B LoRA and Qwen 2.5 7B LoRA currently look like the best routing candidates** because they keep false confidence low without collapsing into excessive abstention.

### 6. QA Signal Validation Passes, With an Important Caveat

We joined the failure-mode labels with the archived per-sample evaluation metrics and compared quality by category for all 12 pre-LUMI models.

Immediate result:

- For every model, `agreement_answer` scores much higher than `false_confidence`
- The separation is statistically overwhelming under a one-sided Mann-Whitney U test
- This supports the routing hypothesis that the abstain/answer failure mode is a meaningful quality signal

The stronger rerun shows that this separation is not limited to the adjusted metric:

- On **raw `embedding_similarity`**, `false_confidence` is worse than `agreement_answer` for all 12 models
- On **`token_overlap`**, `false_confidence` is worse than `agreement_answer` for all 12 models
- On **`exact_match_score`**, `false_confidence` is worse for 11 of 12 models

That substantially strengthens the argument that the failure-mode taxonomy is tracking real answer-quality degradation, not just the abstain mismatch itself.

Important caveat:

- The weakest model (`gemma-3-270m-it-base`) shows no useful exact-match separation because both categories are effectively at zero there
- `embedding_similarity_adjusted` still should not be treated as the main evidence because it assigns:
- `1.0` when both teacher and student abstain
- `0.0` when exactly one side abstains

That means all `false_confidence` rows have mean adjusted similarity `0.0` by construction. So:

- **The adjusted metric remains a sanity check**
- **The raw similarity / overlap / exact-match results are the more important independent evidence**

The next refinement is now downstream rather than foundational:

- answer length by category
- per-category plots for the strongest routing candidates
- validation on the fresh LUMI runs as they arrive

### 7. Routing Simulation Produces a Positive Phase 3 Result

We simulated five routing strategies over the archived pre-LUMI Qwen LoRA family:

- Qwen 2.5 0.5B LoRA
- Qwen 2.5 1.5B LoRA
- Qwen 2.5 3B LoRA
- Qwen 2.5 7B LoRA

Summary result:

| Strategy | Avg Cost (B params) | Embed Sim Adj | Embed Sim | Abstain F1 |
|----------|---------------------|---------------|-----------|------------|
| `always_small` | 0.50 | 0.7006 | 0.8243 | 0.8154 |
| `always_large` | 7.00 | **0.8511** | **0.8997** | **0.9067** |
| `blind_cascade` | 8.66 | 0.7683 | 0.8562 | 0.8406 |
| `qa_routing` | **0.6824** | 0.7568 | 0.8510 | 0.8563 |
| `qa_ensemble` | 12.00 | 0.8299 | 0.8903 | 0.8936 |

What this means:

- `qa_routing` is **much cheaper than `always_large`** while still recovering a substantial fraction of the quality gap
- `qa_routing` is **materially better than `always_small`** at only a modest cost increase
- `qa_routing` is **far cheaper than `blind_cascade`** and still outperforms the small-model baseline comfortably
- `qa_ensemble` is strong but too expensive to be the main cost-saving story

Most importantly:

- On the archived run, `qa_routing` lies **above the line** between `always_small` and `always_large`
- That is the positive Phase 3 outcome required by the research plan

Important caveat:

- The current `qa_routing` experiment is still an **oracle routing upper bound**
- It routes using the known `false_confidence` label directly rather than a learned classifier or score

So the result is promising but not yet deployable. The next step is to test whether the gain survives stronger baselines and more realistic routing rules:

- random-routing ablation
- threshold sensitivity
- eventually, classifier-based routing on fresh LUMI results

### 8. Random-Routing Ablation Confirms the Signal Matters

We ran a random-routing baseline over the same Qwen LoRA family and routing architecture.

Design:

- keep the same staged routing structure
- replace the oracle QA escalation decision with random escalation
- match the stage-wise escalation probabilities of the oracle QA router
- run 20 random seeds

Result:

| Metric | Random Routing Mean ± Std | Oracle QA Routing | Oracle - Random Mean |
|--------|----------------------------|-------------------|----------------------|
| Avg cost | 0.687 ± 0.025 | **0.682** | -0.0048 |
| Embed Sim Adj | 0.7057 ± 0.0027 | **0.7568** | +0.0511 |
| Raw Embed Sim | 0.8268 ± 0.0013 | **0.8510** | +0.0242 |
| Token Overlap | 0.5705 ± 0.0020 | **0.6210** | +0.0505 |
| Exact Match | 0.6426 ± 0.0019 | **0.6978** | +0.0552 |
| Abstain F1 | 0.8181 ± 0.0017 | **0.8563** | +0.0382 |

Interpretation:

- The random baseline operates at essentially the **same average cost** as oracle QA routing
- Oracle QA routing still wins by a large margin on every quality metric
- This is strong evidence that the routing gain is coming from the **signal**, not merely from the multi-stage architecture

This is the strongest result so far in favour of the routing direction.

Remaining caveat:

- The router is still oracle-labelled rather than learned
- So the random ablation validates the usefulness of the signal, but not yet the deployability of the method

What remains for Phase 4:

- classifier-based or score-based routing on the fresh LUMI runs
- comparison of the learned router against fresh post-LUMI evaluations as they arrive

### 9. A First Non-Oracle Router Is Feasible

We trained a simple false-confidence classifier for the Qwen LoRA routing chain using only features available from the current model output:

- student answer text
- abstain flag
- answer length / output length

Method:

- logistic regression with TF-IDF text features + numeric features
- evaluated with out-of-fold predictions on the archived pre-LUMI run
- routed stage-by-stage using the predicted false-confidence probability

Stage-wise classifier quality:

| Model | Positive Rate | ROC AUC | Average Precision |
|-------|---------------|---------|-------------------|
| Qwen 2.5 0.5B LoRA | 5.9% | **0.969** | 0.534 |
| Qwen 2.5 1.5B LoRA | 6.7% | 0.943 | 0.412 |
| Qwen 2.5 3B LoRA | 4.7% | 0.917 | 0.245 |

These are strong enough to justify a threshold sweep.

Threshold sensitivity result:

- The sweep is not brittle; there is a fairly broad flat region
- Peak adjusted quality occurs around threshold `0.60`
- A cost-matched operating point to the oracle router appears around threshold `0.85`

Selected operating points:

| Threshold | Avg Cost (B params) | Embed Sim Adj | Embed Sim | Abstain F1 |
|-----------|---------------------|---------------|-----------|------------|
| 0.60 | 1.157 | **0.7377** | **0.8442** | **0.8436** |
| 0.85 | 0.697 | 0.7325 | 0.8406 | 0.8390 |
| Oracle QA routing | 0.682 | **0.7568** | **0.8510** | **0.8563** |
| Always small | 0.500 | 0.7006 | 0.8243 | 0.8154 |

Interpretation:

- The learned router does **not** match the oracle router yet
- But it does recover a substantial part of the gain with a real scalar score
- Around threshold `0.85`, it operates at almost the same cost as the oracle router while clearly beating the `always_small` baseline
- The flat region from roughly `0.45` to `0.70` suggests the method is not hypersensitive to threshold choice

Important caveat:

- This is still evaluated on the archived pre-LUMI split using out-of-fold predictions
- It is therefore a credible non-oracle prototype, but not yet the final held-out deployment claim

This is enough to justify taking the learned routing setup forward onto the fresh LUMI evaluations as they complete.

---

## Detailed Visualizations

### Answer State Distribution

This visualization shows how student models agree or disagree with the teacher on answer vs abstain decisions.

![Answer State Distribution](outputs/pre-lumi/analysis/answer_state_distribution.png)

**States explained**:
- **Both Abstain** (green): Teacher and student both abstained — ideal agreement
- **Both Answer** (blue): Teacher and student both provided answers — ideal agreement
- **Teacher Answers, Student Abstains** (orange): Student is too conservative
- **Teacher Abstains, Student Answers** (red): Student is hallucinating/over-answering

**Key observations**:
- Base models (especially Qwen 0.5B) have high "Teacher Abstains, Student Answers" — they over-answer
- Trained models shift toward "Both Abstain" — they learn to abstain appropriately
- Gemma 270M LoRA has high "Teacher Answers, Student Abstains" — it's too conservative after training

---

### Abstain Detection Metrics

Classification metrics treating teacher abstain as ground truth.

![Abstain Metrics Comparison](outputs/pre-lumi/analysis/abstain_metrics_comparison.png)

**Metrics explained**:
- **Accuracy**: Overall correctness of abstain/answer decisions
- **F1 Score**: Harmonic mean of precision and recall (best single metric)
- **Precision**: When student abstains, how often is it correct?
- **Recall**: Of all teacher abstains, how many did student catch?

**Key observations**:
- Qwen 7B LoRA achieves the highest F1 (0.907) and best balance of precision/recall
- Qwen 0.5B Base has very low recall (0.24) — it rarely abstains when it should
- All trained models achieve recall > 0.88 — they learn to abstain appropriately

---

### Student vs Teacher Abstain Rates

Comparison of abstain rates between student models and the teacher baseline.

![Abstain Rates Comparison](outputs/pre-lumi/analysis/abstain_rates_comparison.png)

**Teacher abstain rate**: 70.0% (purple dashed line)

**Key observations**:
- Base models vary widely: Qwen 0.5B Base abstains only 21%, while Qwen 7B Base abstains 83%
- Trained models tend to over-abstain (most are above the teacher line)
- Qwen 3B LoRA (78%) and Qwen 7B LoRA (74%) are closest to the teacher rate

---

### Embedding Similarity Distribution

Violin plots showing the distribution of semantic similarity between teacher and student answers.

#### Grouped by Model Family

![Embedding Similarity Grouped](outputs/pre-lumi/analysis/violin_embedding-similarity-adjusted_grouped.png)

#### All Models

![Embedding Similarity All](outputs/pre-lumi/analysis/violin_embedding-similarity-adjusted_all.png)

**Metric explained**:
- Uses Qwen3-Embedding-0.6B for semantic similarity
- Adjusted similarity: 1.0 if both abstain, 0.0 if one abstains, actual similarity if both answer

**Key observations**:
- Base models have bimodal distributions (peaks at 0 and 1)
- Trained models concentrate mass at 1.0 (high similarity)
- Qwen 7B LoRA has the tightest distribution around 1.0

---

### Similarity Metrics Comparison

Comparison of embedding similarity, token overlap, and exact match rate.

![Similarity Metrics Comparison](outputs/pre-lumi/analysis/similarity_metrics_comparison.png)

**Metrics explained**:
- **Embedding Similarity**: Semantic similarity using neural embeddings
- **Token Overlap**: Jaccard similarity of answer tokens
- **Exact Match Rate**: Percentage of identical answers

**Key observations**:
- All three metrics correlate strongly
- Trained models consistently outperform base models
- Larger models achieve higher similarity scores

---

### Model Performance Heatmap

Comprehensive comparison of all models across key metrics.

![Metrics Heatmap](outputs/pre-lumi/analysis/metrics_heatmap.png)

**Color scale**: Red (low) → Yellow (medium) → Green (high)

**Key observations**:
- Qwen 7B LoRA is consistently green across all metrics
- Qwen 0.5B Base is consistently red — worst performer
- Training (LoRA/Full Finetune) consistently improves all metrics

---

### Training Effect by Model Size

Comparison of Base, LoRA, and Full Finetune performance across model sizes.

![Training Effect by Size](outputs/pre-lumi/analysis/training_effect_by_size.png)

**Key observations**:
- LoRA (red) consistently outperforms or matches Full Finetune (purple)
- Base models (green) are always the lowest
- The gap between Base and trained models is largest for smaller models

---

### Performance Improvement Rate

Percentage improvement from base model after training.

![Improvement Rate](outputs/pre-lumi/analysis/improvement_rate_by_training.png)

**Note**: Models marked with `*` only have LoRA training (no Full Finetune data).

**Key observations**:
- Qwen 0.5B shows the largest improvements (+125% F1, +113% embedding similarity)
- Larger models (7B) show smaller relative improvements but start from a higher baseline
- LoRA and Full Finetune achieve similar improvement rates

---

## Summary Table

| Model | Train Type | Abstain F1 | Abstain Acc | Embed Sim | Exact Match |
|-------|------------|------------|-------------|-----------|-------------|
| Gemma 270M | Base | 0.675 | 57.4% | 0.544 | 44.1% |
| Gemma 270M | LoRA | 0.814 | 70.5% | 0.692 | 64.6% |
| Qwen 0.5B | Base | 0.362 | 41.8% | 0.368 | 16.8% |
| Qwen 0.5B | LoRA | 0.815 | 71.0% | 0.701 | 64.2% |
| Qwen 0.5B | Full FT | 0.802 | 69.4% | 0.685 | 62.5% |
| Qwen 1.5B | Base | 0.800 | 72.1% | 0.700 | 56.8% |
| Qwen 1.5B | LoRA | 0.845 | 76.8% | 0.757 | 64.5% |
| Qwen 3B | Base | 0.808 | 74.4% | 0.728 | 56.2% |
| Qwen 3B | LoRA | **0.881** | 82.3% | 0.810 | 67.4% |
| Qwen 3B | Full FT | 0.862 | 79.0% | 0.779 | 67.5% |
| Qwen 7B | Base | 0.882 | 81.9% | 0.811 | 70.7% |
| Qwen 7B | LoRA | **0.907** | **86.5%** | **0.851** | 67.6% |

**Best performers highlighted in bold.**

---

## Conclusions

1. **Training is essential**: All base models underperform compared to their trained counterparts, especially smaller models.

2. **LoRA is sufficient**: LoRA achieves comparable or better results than Full Finetune while being more memory-efficient.

3. **Scale helps but isn't everything**: Larger models perform better, but a well-trained small model (Qwen 0.5B LoRA) can match or exceed a larger base model (Qwen 1.5B Base).

4. **Abstain behavior is learnable**: Models successfully learn to abstain when evidence is insufficient, with recall rates exceeding 88% for all trained models.

5. **Recommended model**: For production use, **Qwen 2.5 7B LoRA** offers the best performance. For resource-constrained environments, **Qwen 2.5 3B LoRA** provides an excellent balance of performance and efficiency.

---

## Files

All analysis outputs are in `outputs/analysis/`:

| File | Description |
|------|-------------|
| `metrics_summary.json` | Full metrics for all models (JSON) |
| `metrics_summary.csv` | Full metrics for all models (CSV) |
| `answer_state_distribution.png` | Answer state stacked bar chart |
| `abstain_metrics_comparison.png` | Abstain F1/Precision/Recall/Accuracy |
| `abstain_rates_comparison.png` | Student vs teacher abstain rates |
| `similarity_metrics_comparison.png` | Embedding/token/exact match comparison |
| `violin_embedding-similarity-adjusted_*.png` | Similarity distribution plots |
| `metrics_heatmap.png` | Model comparison heatmap |
| `training_effect_by_size.png` | Training method comparison |
| `improvement_rate_by_training.png` | Improvement from base models |

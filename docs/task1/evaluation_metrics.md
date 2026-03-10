# Evaluation Metrics

Complete reference for all metrics, output files, and column schemas produced by `05_model_evaluation.py` and `07_tta_experiment.py`.

---

## Answer States

Every sample is classified into one of four *answer states* based on whether the teacher and student abstained:

| State | Meaning |
|-------|---------|
| `both_abstain` | Both teacher and student refused to answer ✅ |
| `both_answer` | Both provided answers ✅ |
| `teacher_answer_student_abstain` | Teacher answered, student over-abstained ❌ |
| `teacher_abstain_student_answer` | Teacher abstained, student hallucinated ❌ |

A well-calibrated model maximises the top two states.

---

## Abstain Classification Metrics

The abstain decision is treated as a binary classification problem:

- **Positive class**: *abstain*
- **True Positive (TP)**: both teacher and student abstain (`both_abstain`)
- **True Negative (TN)**: both answer (`both_answer`)
- **False Positive (FP)**: student abstains, teacher answers (`teacher_answer_student_abstain`)
- **False Negative (FN)**: student answers, teacher abstains (`teacher_abstain_student_answer`)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Of all student abstentions, how many were correct? |
| **Recall** | TP / (TP + FN) | Of all teacher abstentions, how many did the student catch? |
| **F1** | 2 · P · R / (P + R) | Harmonic mean — primary ranking metric |
| **Accuracy** | (TP + TN) / N | Overall agreement rate on the binary decision |

---

## Similarity Metrics

### Embedding Similarity

Cosine similarity between teacher and student answer embeddings, computed using **Qwen3-Embedding-0.6B**.

| Variant | Description |
|---------|-------------|
| `embedding_similarity` | Raw cosine similarity for all pairs with non-empty answers |
| `embedding_similarity_adjusted` | Same, but forced to **1.0** if both abstain, **0.0** if only one abstains |

The *adjusted* variant is the primary metric because it penalises abstain disagreement.

### Token Overlap

Jaccard similarity of answer token sets:

```
overlap = |teacher_tokens ∩ student_tokens| / |teacher_tokens ∪ student_tokens|
```

Ranges 0–1. Complements embedding similarity by catching surface-level matches.

### Exact Match

`1.0` if normalised teacher and student answers are identical, `0.0` otherwise.
Normalisation: lower-case, strip punctuation, collapse whitespace.
Special case: both abstain → `1.0`.

---

## Loss & Perplexity

| Metric | Description |
|--------|-------------|
| `loss` | Cross-entropy loss on the eval set (teacher outputs as targets) |
| `perplexity` | `exp(loss)` — how "surprised" the model is by teacher outputs |

Lower is better. Perplexity < 10 indicates good in-distribution fit.

---

## Output File Reference

### `{model}_detailed_results.csv`

One row per evaluation sample.

| Column | Type | Description |
|--------|------|-------------|
| `idx` | int | Sample index |
| `data_source` | str | Source dataset (causalqa, msmarco, quasart) |
| `teacher_id` | str | Teacher model identifier |
| `query` | str | User query |
| `search_index` | str | Retrieval index used |
| `llm_input` | str | Full prompt sent to the model |
| `teacher_output` | str | Raw teacher output |
| `student_output` | str | Raw student output |
| `teacher_answer` | str | Extracted answer from teacher |
| `student_answer` | str | Extracted answer from student |
| `teacher_abstain` | bool | Teacher abstained? |
| `student_abstain` | bool | Student abstained? |
| `answer_state` | str | One of the four states above |
| `exact_match_score` | float | 0.0 or 1.0 |
| `embedding_similarity` | float | Raw cosine similarity |
| `embedding_similarity_adjusted` | float | Adjusted similarity |
| `token_overlap` | float | Jaccard token overlap |
| `decision_label` | str | Original label from dataset |

### TTA extra columns (07 only)

| Column | Description |
|--------|-------------|
| `tta_n` | Number of passes used |
| `tta_temperature` | Sampling temperature used |
| `tta_aggregation` | `majority_vote`, `centroid`, or `oracle` |
| `vote_abstain_count` | How many of the N passes chose to abstain |
| `vote_answer_count` | How many of the N passes gave an answer |

### `{model}_summary.json`

```json
{
  "timestamp_utc": "...",
  "model_name": "...",
  "model_type": "lora | full_finetune | base",
  "num_samples": 1886,
  "loss_perplexity": { "loss": 1.23, "perplexity": 3.42 },
  "answer_states": {
    "state_counts": { "both_abstain": 900, "both_answer": 750, ... },
    "state_percentages": { ... }
  },
  "teacher_stats": { "abstain_count": 1320, "abstain_rate": 0.70 },
  "student_stats": { "abstain_count": 1100, "abstain_rate": 0.58 },
  "agreement": {
    "abstain_agreement_rate": 0.87,
    "both_abstain_rate": 0.52,
    "both_answer_rate": 0.35
  },
  "exact_match": { "avg_score": 0.42, "stats": { ... } },
  "embedding_similarity": {
    "all_pairs": { "mean": 0.78, ... },
    "adjusted_for_abstain": { "mean": 0.81, ... }
  },
  "token_overlap": { "mean": 0.45, ... }
}
```

### `model_comparison_summary.csv`

One row per model, flat table of all scalar metrics — useful for cross-model analysis in pandas or the MLflow UI.


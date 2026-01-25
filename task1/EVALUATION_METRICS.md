# Model Evaluation Metrics Guide

This document explains all metrics, statistics, output files, and columns produced by the evaluation script (`05_model_evaluation.py`).

## Overview

The evaluation script compares a fine-tuned student model against teacher outputs from the evaluation dataset. It measures:
1. **Language modeling quality** (loss, perplexity)
2. **Answer agreement** (abstain detection, state tracking)
3. **Answer similarity** (exact match, embedding similarity, token overlap)

---

## Output Files

For each evaluated model, the script produces:

### 1. `{model_name}_detailed_results.csv`

Per-sample results with all metrics. Contains one row per evaluation sample.

| Column | Description |
|--------|-------------|
| `idx` | Sample index in the evaluation dataset |
| `data_source` | Source dataset (e.g., causalqa, msmarco) |
| `teacher_id` | Teacher model identifier |
| `query` | Original query/question |
| `search_index` | Search index used for retrieval |
| `llm_input` | Full input prompt sent to the model |
| `teacher_output` | Complete teacher model output |
| `student_output` | Complete student model output |
| `teacher_answer` | Extracted answer from teacher output |
| `student_answer` | Extracted answer from student output |
| `teacher_abstain` | `True` if teacher abstained from answering |
| `student_abstain` | `True` if student abstained from answering |
| `answer_state` | Agreement state (see Answer States below) |
| `exact_match_score` | 1.0 if answers match exactly, 0.0 otherwise |
| `embedding_similarity` | Cosine similarity of answer embeddings (0-1) |
| `embedding_similarity_adjusted` | Embedding similarity with abstain penalty |
| `token_overlap` | Jaccard similarity of answer tokens (0-1) |
| `decision_label` | Original decision label from dataset |

### 2. `{model_name}_summary.json`

Comprehensive statistics for the model evaluation.

```json
{
  "timestamp_utc": "20260124_120000_UTC",
  "model_name": "model-name",
  "model_path": "/path/to/model",
  "model_type": "full_finetune",
  "num_samples": 1000,
  "config": { ... },
  "loss_perplexity": { ... },
  "answer_states": { ... },
  "teacher_stats": { ... },
  "student_stats": { ... },
  "agreement": { ... },
  "exact_match": { ... },
  "embedding_similarity": { ... },
  "token_overlap": { ... }
}
```

### 3. `model_comparison_summary.csv`

Summary CSV comparing all evaluated models side-by-side.

---

## Answer States

The `answer_state` column tracks the agreement between teacher and student:

| State | Description |
|-------|-------------|
| `both_abstain` | Both teacher and student abstained (refused to answer) |
| `teacher_answer_student_abstain` | Teacher provided answer, student abstained |
| `teacher_abstain_student_answer` | Teacher abstained, student provided answer |
| `both_answer` | Both teacher and student provided answers |

### State Interpretation

- **High `both_abstain` + `both_answer`**: Good agreement on when to answer
- **High `teacher_answer_student_abstain`**: Student is too conservative (over-abstaining)
- **High `teacher_abstain_student_answer`**: Student is too aggressive (under-abstaining)

---

## Metrics Explained

### Loss & Perplexity

| Metric | Description | Good Value |
|--------|-------------|------------|
| `loss` | Cross-entropy loss on eval set | Lower is better |
| `perplexity` | exp(loss) - how "surprised" the model is | Lower is better (< 10 is good) |

### Abstain Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `teacher_abstain_rate` | % of samples where teacher abstained | 0-1 |
| `student_abstain_rate` | % of samples where student abstained | 0-1 |
| `abstain_agreement_rate` | % where teacher and student agree on abstain decision | 0-1 (higher is better) |
| `both_abstain_rate` | % where both abstained | 0-1 |
| `both_answer_rate` | % where both answered | 0-1 |

### Similarity Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `exact_match_score` | 1 if normalized answers are identical | 0 or 1 |
| `exact_match_avg` | Average exact match across all samples | 0-1 (higher is better) |
| `embedding_similarity` | Cosine similarity of answer embeddings | 0-1 (higher is better) |
| `embedding_similarity_adjusted` | Same, but 0 if abstain mismatch, 1 if both abstain | 0-1 |
| `token_overlap` | Jaccard similarity of answer tokens | 0-1 (higher is better) |

### Embedding Similarity Details

Two versions are computed:

1. **`embedding_similarity`** (all pairs)
   - Computes cosine similarity for all pairs where both have non-empty answers
   - `None` for pairs where either answer is empty

2. **`embedding_similarity_adjusted`** (abstain-aware)
   - If both abstain: similarity = 1.0 (perfect agreement)
   - If one abstains, other answers: similarity = 0.0 (complete disagreement)
   - If both answer: compute actual embedding similarity

---

## Abstain Detection

The script uses robust pattern matching to detect abstention. An answer is considered an abstain if it:

1. Is empty or very short (< 3 characters)
2. Contains phrases like:
   - "I cannot answer"
   - "cannot answer based on the provided evidence"
   - "not enough information/evidence"
   - "insufficient information/evidence"
   - "unable to answer/determine/provide"
   - "I don't have enough"
3. Exactly matches: "n/a", "none", "unknown", "no answer", "abstain"

---

## Answer Extraction

Answers are extracted from model outputs using:

1. Look for `ANSWER:` section (case-insensitive)
2. Also matches `Answer:`, `final answer:`, `my answer:`
3. Removes trailing sections (EVIDENCE, RATIONALE, etc.)
4. Falls back to full output if no section found

---

## Summary Statistics

For numeric metrics, the summary includes:

| Statistic | Description |
|-----------|-------------|
| `count` | Number of valid values |
| `mean` | Average value |
| `min` | Minimum value |
| `median` | Median value |
| `max` | Maximum value |
| `std` | Standard deviation |

---

## Usage Examples

### Basic Evaluation

```bash
cd task1
python scripts/05_model_evaluation.py \
    --model_paths outputs/Qwen-Qwen2.5-0.5B-Instruct-full-finetune-final
```

### Compare Multiple Models

```bash
python scripts/05_model_evaluation.py \
    --model_paths \
        Qwen/Qwen2.5-0.5B-Instruct \
        outputs/Qwen-Qwen2.5-0.5B-Instruct-full-finetune-final \
    --model_names "base" "finetuned"
```

### Quick Test (Limited Samples)

```bash
python scripts/05_model_evaluation.py \
    --model_paths outputs/model-final \
    --num_samples 100
```

### Multi-GPU with Accelerate

```bash
accelerate launch --multi_gpu --num_processes=4 \
    scripts/05_model_evaluation.py \
    --model_paths outputs/model-final
```

### LoRA Model Evaluation

```bash
python scripts/05_model_evaluation.py \
    --model_paths outputs/model-lora-final \
    --lora
```

### Custom Embedding Model

```bash
python scripts/05_model_evaluation.py \
    --model_paths outputs/model-final \
    --embedding_model "sentence-transformers/all-mpnet-base-v2"
```

---

## Interpreting Results

### Good Fine-tuning Indicators

1. **Lower loss/perplexity** than base model
2. **High abstain_agreement_rate** (> 0.8) - student follows teacher's abstain decisions
3. **High exact_match_avg** for `both_answer` cases
4. **High embedding_similarity_adjusted** (> 0.7) - semantically similar answers
5. **Balanced state distribution** - not over/under-abstaining

### Warning Signs

1. **High `teacher_answer_student_abstain`** - student too conservative
2. **High `teacher_abstain_student_answer`** - student hallucinating answers
3. **Very different abstain rates** between teacher and student

---

## Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_paths` | Required | Path(s) to model directories |
| `--eval_dataset` | `data/task1_eval_dataset.csv` | Evaluation dataset |
| `--num_samples` | `-1` (all) | Number of samples (-1 for all) |
| `--batch_size` | `4` | Batch size for loss computation |
| `--gen_batch_size` | `8` | Batch size for generation |
| `--output_dir` | `outputs/evaluations` | Output directory |
| `--lora` | Flag | Models are LoRA adapters |
| `--loss_only` | Flag | Skip generation metrics |
| `--model_names` | Auto | Custom names for models |
| `--embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |

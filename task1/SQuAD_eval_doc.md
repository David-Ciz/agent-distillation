# SQuAD v2.0 Evaluation — External Benchmark for Task 1 Models

## Table of Contents

1. [Experiment Overview](#1-experiment-overview)
2. [Dataset](#2-dataset)
3. [Evaluation Methodology](#3-evaluation-methodology)
4. [Models Evaluated](#4-models-evaluated)
5. [Results](#5-results)
6. [Key Findings](#6-key-findings)
7. [How to Reproduce](#7-how-to-reproduce)
8. [Directory Structure](#8-directory-structure)

---

## 1. Experiment Overview

### Purpose

This experiment evaluates Task 1 fine-tuned models (Base, LoRA, Full Fine-Tune) on **SQuAD v2.0** — a well-known, external reading comprehension benchmark — to measure how well our distilled models generalize beyond the proprietary training dataset.

SQuAD (Stanford Question Answering Dataset) v2.0 contains both **answerable** and **unanswerable** questions over Wikipedia passages. By formatting SQuAD samples into the **same prompt template** used in our Task 1 training/evaluation pipeline, we get a standardized comparison between:

1. **Base pre-trained models** (no fine-tuning)
2. **LoRA-adapted models** (Task 1 LoRA)
3. **Full fine-tuned models** (where available)
4. **GPT-4o teacher** (upper bound reference)

### Key Difference from Task 1 Internal Evaluation

In the internal Task 1 evaluation, models are compared against GPT-4o teacher outputs on our proprietary dataset. Here, models are compared against **SQuAD gold answers** (human-annotated spans from Wikipedia passages), providing an independent measure of QA quality.

### Script

`scripts/05.2_model_evaluation_SQuAD.py`

---

## 2. Dataset

### SQuAD v2.0

| Property | Value |
|----------|-------|
| Dataset | SQuAD v2.0 (Stanford Question Answering Dataset) |
| Source | `rajpurkar/squad_v2` (HuggingFace) / official JSON |
| Split used | Validation |
| Full validation set | 11,873 QA pairs |
| Answerable questions | 5,928 |
| Unanswerable questions | 5,945 |

SQuAD v2.0 extends SQuAD v1.1 by adding **unanswerable questions** — questions that look plausible but cannot be answered from the given passage. This tests a model's ability to abstain when evidence is insufficient, which directly aligns with our Task 1 abstain-detection requirement.

### Sampling

For this evaluation run, **200 samples** were sub-sampled with stratified sampling:

| Property | Value |
|----------|-------|
| Total samples | 200 |
| Answerable | 140 (70%) |
| Unanswerable | 60 (30%) |
| Minimum unanswerable ratio | 30% (enforced) |
| Random seed | 42 |

### Prompt Template

SQuAD samples are formatted into the **exact same prompt template** used in our Task 1 pipeline:

```
You are a question-answering system with NO ACCESS to external knowledge.

IMPORTANT: First, explain your internal reasoning about how you're using the evidence,
then provide your answer.
Format:
REASONING: [Your thought process...]
ANSWER: [Your final answer]

CRITICAL RULES - FAILURE TO FOLLOW WILL RESULT IN REJECTION:
1. Use ONLY information explicitly stated in the evidence passages below
2. Do NOT use any internal knowledge, common sense, or world knowledge
3. Do NOT make inferences beyond what is directly stated
4. If the evidence doesn't contain a complete answer, respond with:
   "I cannot answer based on the provided evidence."
5. Keep the answer concise (2-3 sentences maximum)
6. Do NOT add explanations, context, or citations in your answer

Query: {question}

Evidence Passages (your ONLY source of information):
[1] {context}
```

This ensures that any performance differences reflect the model's learned QA ability under our pipeline's constraints, not prompt format differences.

---

## 3. Evaluation Methodology

### Two-Phase Evaluation

The evaluation runs in two phases:

1. **Phase 1 — Teacher (GPT-4o)**: GPT-4o is evaluated on all 200 SQuAD samples via the OpenAI API (temperature=0, max_tokens=512). Teacher responses are cached for resumability.
2. **Phase 2 — Student models**: Each student model generates answers for the same 200 samples using batched inference (batch_size=8, max_new_tokens=256, greedy decoding).

### Metrics Computed

Each student model is evaluated against **two references**: SQuAD gold answers and the GPT-4o teacher.

#### vs. SQuAD Gold Answers

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | Official SQuAD EM: normalized string equality after lowercasing, removing articles/punctuation/extra whitespace. For unanswerable questions, EM=1.0 if the model correctly abstains. Max over all gold answers. |
| **F1 Score** | Official SQuAD token-level F1: precision/recall over normalized tokens. For unanswerable questions, F1=1.0 if model abstains. Max over all gold answers. |
| **Token Overlap (Jaccard)** | Word-level Jaccard similarity between prediction and first gold answer |
| **Embedding Similarity (vs Gold)** | Cosine similarity of `Qwen/Qwen3-Embedding-0.6B` embeddings between student answer and first gold answer (non-abstaining pairs only) |

#### vs. GPT-4o Teacher

| Metric | Description |
|--------|-------------|
| **Exact Match (vs Teacher)** | Normalized string equality between student and teacher answers. Both-abstain counts as 1.0. |
| **Token Overlap (vs Teacher)** | Jaccard similarity between student and teacher answers |
| **Embedding Similarity (vs Teacher, Adjusted)** | Cosine similarity with abstain adjustments: both abstain = 1.0, abstain mismatch = 0.0 |
| **Abstain Agreement Rate** | Rate at which student and teacher agree on abstaining vs. answering |

#### Abstain & Unanswerable Evaluation

| Metric | Description |
|--------|-------------|
| **Student Abstain Rate** | Fraction of all samples where the student abstains |
| **Correct Abstain Rate** | Fraction of unanswerable questions where the student correctly abstains |
| **False Abstain Rate** | Fraction of answerable questions where the student incorrectly abstains |

Abstention is detected using the same robust regex patterns as the Task 1 internal evaluation (matching phrases like "I cannot answer", "insufficient evidence", etc.).

### Answer Extraction

The same `extract_answer()` function from the Task 1 pipeline is used: it searches for an `ANSWER:` section in the model output, with fallback heuristics for short outputs or last-paragraph extraction.

---

## 4. Models Evaluated

**13 models total**: 1 teacher + 12 student models (5 base + 5 LoRA + 2 full fine-tune).

| Model | Type | HuggingFace ID / Path |
|-------|------|----------------------|
| **GPT-4o** | Teacher | `gpt-4o` (OpenAI API) |
| gemma-270m-base | Base | `google/gemma-3-270m-it` |
| gemma-270m-lora | LoRA | `outputs/models/google_gemma-3-270m-it-lora-final` |
| qwen-0.5B-base | Base | `Qwen/Qwen2.5-0.5B-Instruct` |
| qwen-0.5B-lora | LoRA | `outputs/models/Qwen2.5-0.5B-Instruct-lora-final` |
| qwen-0.5B-full | Full FT | `outputs/models/Qwen-Qwen2.5-0.5B-Instruct-full-finetune-final` |
| qwen-1.5B-base | Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| qwen-1.5B-lora | LoRA | `outputs/models/Qwen2.5-1.5B-Instruct-lora-final` |
| qwen-3B-base | Base | `Qwen/Qwen2.5-3B-Instruct` |
| qwen-3B-lora | LoRA | `outputs/models/Qwen2.5-3B-Instruct-lora-final` |
| qwen-3B-full | Full FT | `outputs/models/Qwen-Qwen2.5-3B-Instruct-full-finetune-final` |
| qwen-7B-base | Base | `Qwen/Qwen2.5-7B-Instruct` |
| qwen-7B-lora | LoRA | `outputs/models/Qwen2.5-7B-Instruct-lora-final` |

Note: Gemma-1B models were not included in this SQuAD evaluation run.

---

## 5. Results

All results are from evaluation run `squad_eval_run_20260208_145212` (200 SQuAD v2.0 samples).

### 5.1 Overall Performance vs. SQuAD Gold Answers

| Model | Type | EM (Gold) | F1 (Gold) | Token Overlap (Gold) | Embed Sim (Gold) |
|-------|------|-----------|-----------|---------------------|-----------------|
| GPT-4o | Teacher | 0.235 | 0.474 | 0.132 | — |
| qwen-7B-base | Base | 0.235 | 0.424 | 0.091 | 0.724 |
| qwen-7B-lora | LoRA | 0.240 | 0.425 | 0.080 | 0.714 |
| qwen-3B-base | Base | 0.245 | 0.421 | 0.140 | 0.747 |
| qwen-3B-lora | LoRA | 0.250 | 0.400 | 0.073 | 0.719 |
| qwen-3B-full | Full FT | 0.240 | 0.394 | 0.074 | 0.719 |
| qwen-0.5B-full | Full FT | 0.285 | 0.298 | 0.011 | 0.695 |
| gemma-270m-lora | LoRA | 0.275 | 0.286 | 0.012 | 0.663 |
| qwen-0.5B-lora | LoRA | 0.270 | 0.297 | 0.017 | 0.664 |
| qwen-1.5B-lora | LoRA | 0.265 | 0.332 | 0.035 | 0.710 |
| qwen-1.5B-base | Base | 0.115 | 0.248 | 0.064 | 0.694 |
| gemma-270m-base | Base | 0.120 | 0.168 | 0.025 | 0.613 |
| qwen-0.5B-base | Base | 0.095 | 0.176 | 0.049 | 0.655 |

### 5.2 Answerable vs. Unanswerable Breakdown

| Model | Type | EM (Answerable) | F1 (Answerable) | EM (Unanswerable) |
|-------|------|----------------|-----------------|-------------------|
| GPT-4o | Teacher | 0.036 | 0.378 | 0.700 |
| qwen-7B-base | Base | 0.000 | 0.271 | 0.783 |
| qwen-7B-lora | LoRA | 0.000 | 0.264 | 0.800 |
| qwen-3B-base | Base | 0.093 | 0.344 | 0.600 |
| qwen-3B-lora | LoRA | 0.000 | 0.214 | 0.833 |
| qwen-3B-full | Full FT | 0.000 | 0.221 | 0.800 |
| qwen-1.5B-lora | LoRA | 0.000 | 0.095 | 0.883 |
| qwen-1.5B-base | Base | 0.014 | 0.205 | 0.350 |
| qwen-0.5B-full | Full FT | 0.000 | 0.018 | 0.950 |
| qwen-0.5B-lora | LoRA | 0.000 | 0.038 | 0.900 |
| gemma-270m-lora | LoRA | 0.000 | 0.016 | 0.917 |
| qwen-0.5B-base | Base | 0.021 | 0.137 | 0.267 |
| gemma-270m-base | Base | 0.000 | 0.068 | 0.400 |

### 5.3 Abstain Behavior

| Model | Type | Student Abstain Rate | Correct Abstain Rate | False Abstain Rate |
|-------|------|---------------------|---------------------|-------------------|
| GPT-4o | Teacher | 25.0% | 70.0% | 5.7% |
| qwen-0.5B-full | Full FT | 95.0% | 95.0% | 95.0% |
| gemma-270m-lora | LoRA | 92.5% | 91.7% | 92.9% |
| qwen-0.5B-lora | LoRA | 88.5% | 90.0% | 87.9% |
| qwen-1.5B-lora | LoRA | 77.0% | 88.3% | 72.1% |
| qwen-3B-lora | LoRA | 48.0% | 83.3% | 32.9% |
| qwen-3B-full | Full FT | 47.5% | 80.0% | 33.6% |
| qwen-7B-lora | LoRA | 39.0% | 80.0% | 21.4% |
| qwen-7B-base | Base | 38.5% | 78.3% | 21.4% |
| gemma-270m-base | Base | 33.5% | 40.0% | 30.7% |
| qwen-3B-base | Base | 28.0% | 60.0% | 14.3% |
| qwen-1.5B-base | Base | 25.5% | 35.0% | 21.4% |
| qwen-0.5B-base | Base | 16.5% | 26.7% | 12.1% |

### 5.4 vs. GPT-4o Teacher Comparison

| Model | Type | EM (vs Teacher) | Embed Sim (Teacher Adj) | Abstain Agreement |
|-------|------|----------------|------------------------|-------------------|
| qwen-7B-base | Base | 0.365 | 0.775 | 79.5% |
| qwen-7B-lora | LoRA | 0.325 | 0.764 | 79.0% |
| qwen-3B-base | Base | 0.290 | 0.747 | 79.0% |
| qwen-3B-lora | LoRA | 0.305 | 0.665 | 69.0% |
| qwen-3B-full | Full FT | 0.285 | 0.653 | 67.5% |
| qwen-1.5B-base | Base | 0.165 | 0.652 | 69.5% |
| qwen-0.5B-base | Base | 0.090 | 0.611 | 70.5% |
| gemma-270m-base | Base | 0.085 | 0.491 | 57.5% |
| qwen-1.5B-lora | LoRA | 0.270 | 0.429 | 44.0% |
| qwen-0.5B-lora | LoRA | 0.245 | 0.319 | 32.5% |
| qwen-0.5B-full | Full FT | 0.255 | 0.278 | 28.0% |
| gemma-270m-lora | LoRA | 0.230 | 0.276 | 28.5% |

---

## 6. Key Findings

### Finding 1: Fine-tuned models over-abstain on SQuAD

The most striking result is that **LoRA and Full FT models abstain far too aggressively** on SQuAD:

| Model Group | Avg Abstain Rate | Avg False Abstain Rate |
|-------------|-----------------|----------------------|
| Base models | 16.5%–38.5% | 12.1%–30.7% |
| LoRA models | 39.0%–92.5% | 21.4%–92.9% |
| Full FT models | 47.5%–95.0% | 33.6%–95.0% |
| GPT-4o (teacher) | 25.0% | 5.7% |

Our Task 1 training data teaches models to abstain when evidence is insufficient. On SQuAD — where evidence passages are short Wikipedia paragraphs rather than our pipeline's multi-passage evidence format — fine-tuned models interpret the unfamiliar format as insufficient evidence and abstain excessively. Smaller fine-tuned models (gemma-270m-lora: 92.5%, qwen-0.5B-full: 95.0%) are worst affected.

### Finding 2: EM on answerable questions drops to 0.0 for most fine-tuned models

Nearly all LoRA and Full FT models achieve **EM (Answerable) = 0.000**, meaning they never produce an exact span match on answerable questions. This is because:
1. They abstain on most answerable questions (false abstain)
2. When they do answer, they generate in our pipeline's verbose REASONING+ANSWER format rather than producing concise SQuAD-style span answers

Base models (especially qwen-3B-base at 0.093) occasionally produce span-like answers because they haven't been trained to follow our specific output format.

### Finding 3: Overall EM is inflated by correct abstention on unanswerable questions

Models with high overall EM (e.g., qwen-0.5B-full at 0.285) achieve this primarily through **correct abstention on unanswerable questions** (EM Unanswerable = 0.950), not through answering correctly. This is a side effect of the over-abstain problem: abstaining on everything yields high unanswerable EM but zero answerable EM.

GPT-4o achieves EM = 0.235 overall with a balanced approach: 70.0% correct abstain on unanswerable, 5.7% false abstain on answerable, and F1 = 0.378 on answerable questions.

### Finding 4: F1 (Gold) is the most informative metric

F1 score accounts for partial token overlap and is less sensitive to the format mismatch. The ranking by F1 (Gold) better reflects actual QA capability:

| Rank | Model | F1 (Gold) |
|------|-------|-----------|
| 1 | GPT-4o | 0.474 |
| 2 | qwen-7B-lora | 0.425 |
| 3 | qwen-7B-base | 0.424 |
| 4 | qwen-3B-base | 0.421 |
| 5 | qwen-3B-lora | 0.400 |
| 6 | qwen-3B-full | 0.394 |
| 7 | qwen-1.5B-lora | 0.332 |
| 8 | qwen-0.5B-full | 0.298 |
| 9 | qwen-0.5B-lora | 0.297 |
| 10 | gemma-270m-lora | 0.286 |

Larger models (3B, 7B) perform closer to GPT-4o. LoRA models generally match or slightly exceed their base counterparts on F1, despite the abstain problem.

### Finding 5: Larger base models approach GPT-4o without fine-tuning

On SQuAD, **qwen-7B-base** (F1=0.424) and **qwen-3B-base** (F1=0.421) nearly match GPT-4o (F1=0.474) without any fine-tuning. This suggests that larger pre-trained models already have strong reading comprehension, and the gap to GPT-4o is relatively small on this benchmark.

### Finding 6: LoRA vs Full FT show similar SQuAD performance

For the two model sizes where both LoRA and Full FT were evaluated:

| Model | LoRA F1 | Full FT F1 | LoRA Abstain | Full FT Abstain |
|-------|---------|-----------|-------------|----------------|
| Qwen 0.5B | 0.297 | 0.298 | 88.5% | 95.0% |
| Qwen 3B | 0.400 | 0.394 | 48.0% | 47.5% |

LoRA and Full FT achieve nearly identical F1 scores. Full FT tends to abstain slightly more aggressively than LoRA.

### Implications

1. **Over-abstention is the primary failure mode** when applying Task 1 fine-tuned models to external benchmarks. The models learn the abstain behavior too strongly from our training data.
2. **SQuAD EM is not a good metric** for evaluating our models because our pipeline produces verbose REASONING+ANSWER outputs, not concise span answers. F1 is more appropriate.
3. **Future work**: Consider calibrating abstain thresholds or adding SQuAD-like data to training to reduce false abstention on external benchmarks.

---

## 7. How to Reproduce

### Prerequisites

- Trained Task 1 models (LoRA and/or Full FT) in `outputs/models/`
- OpenAI API key set in `.env` file (for GPT-4o teacher evaluation)
- Python dependencies: `torch`, `transformers`, `peft`, `datasets`, `openai`, `numpy`, `pandas`, `tqdm`

### Run Command

```bash
cd /root/DeKIS/agent-distillation/task1

python3 scripts/05.2_model_evaluation_SQuAD.py \
    --num_samples 200 \
    --seed 42 \
    --teacher_model gpt-4o \
    --gen_batch_size 8 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --models \
    "google/gemma-3-270m-it,gemma-270m-base,base" \
    "outputs/models/google_gemma-3-270m-it-lora-final,gemma-270m-lora,lora" \
    "Qwen/Qwen2.5-0.5B-Instruct,qwen-0.5B-base,base" \
    "outputs/models/Qwen-Qwen2.5-0.5B-Instruct-full-finetune-final,qwen-0.5B-full,full_finetune" \
    "outputs/models/Qwen2.5-0.5B-Instruct-lora-final,qwen-0.5B-lora,lora" \
    "Qwen/Qwen2.5-1.5B-Instruct,qwen-1.5B-base,base" \
    "outputs/models/Qwen2.5-1.5B-Instruct-lora-final,qwen-1.5B-lora,lora" \
    "Qwen/Qwen2.5-3B-Instruct,qwen-3B-base,base" \
    "outputs/models/Qwen-Qwen2.5-3B-Instruct-full-finetune-final,qwen-3B-full,full_finetune" \
    "outputs/models/Qwen2.5-3B-Instruct-lora-final,qwen-3B-lora,lora" \
    "Qwen/Qwen2.5-7B-Instruct,qwen-7B-base,base" \
    "outputs/models/Qwen2.5-7B-Instruct-lora-final,qwen-7B-lora,lora"
```

### Model Config Format

Each `--models` argument follows the format: `path,name,type[,gen_batch_size]`

- **path**: HuggingFace model ID or local path to model/adapter
- **name**: Short label for output files
- **type**: `base`, `lora`, or `full_finetune`
- **gen_batch_size** (optional): Override default batch size

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_samples` | -1 (all) | Number of SQuAD samples (-1 = full 11,873) |
| `--seed` | 42 | Random seed for sub-sampling |
| `--teacher_model` | gpt-4o | OpenAI model for teacher evaluation |
| `--gen_batch_size` | 8 | Batch size for student generation |
| `--embedding_model` | Qwen/Qwen3-Embedding-0.6B | Model for embedding similarity |

---

## 8. Directory Structure

```
task1/outputs/evaluations_squad/
|-- squad_eval_run_20260208_145212/
|   |-- squad_model_comparison_summary.csv   # All models comparison table
|   |-- squad_evaluation.log                 # Full evaluation log
|   |-- gpt-4o/                              # Teacher results
|   |   |-- gpt-4o_cache.json               # Cached API responses
|   |   |-- gpt-4o_squad_detailed_results.csv
|   |   |-- gpt-4o_squad_summary.json
|   |-- gemma-270m-base/                     # Per-model results
|   |   |-- gemma-270m-base_squad_detailed_results.csv
|   |   |-- gemma-270m-base_squad_summary.json
|   |-- gemma-270m-lora/
|   |-- qwen-0.5B-base/
|   |-- qwen-0.5B-lora/
|   |-- qwen-0.5B-full/
|   |-- qwen-1.5B-base/
|   |-- qwen-1.5B-lora/
|   |-- qwen-3B-base/
|   |-- qwen-3B-lora/
|   |-- qwen-3B-full/
|   |-- qwen-7B-base/
|   |-- qwen-7B-lora/
```

Each per-model directory contains:
- `*_squad_detailed_results.csv` — Per-sample predictions, gold answers, all metrics
- `*_squad_summary.json` — Aggregated statistics (overall, answerable, unanswerable splits)

---

*Document generated from evaluation run `squad_eval_run_20260208_145212`.*
*200 SQuAD v2.0 samples (140 answerable + 60 unanswerable). 13 models evaluated (1 teacher + 12 students).*
*Teacher: GPT-4o. Embedding model: Qwen/Qwen3-Embedding-0.6B.*
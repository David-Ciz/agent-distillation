# Agent Distillation

Research project for distilling knowledge from large language model (LLM) agents into smaller, efficient models. The goal is to train compact student models that can replicate the behavior of powerful teacher models on specific agentic tasks.

## Overview

This project implements a multi-task distillation pipeline:

1. **Trace Collection**: Collect behavioral traces from a teacher model (e.g., GPT-4o) performing agentic tasks
2. **Dataset Creation**: Process traces into supervised training datasets
3. **Model Training**: Fine-tune smaller models (LoRA or full fine-tuning) on the collected data
4. **Evaluation**: Comprehensively evaluate student models against teacher outputs

Currently implemented tasks:
- **Task 1**: Answer-Abstain QA

---

## Task 1: Answer-Abstain QA

### Task Description

Given a user query and a set of retrieved evidence passages, the model must decide whether the evidence is sufficient to answer the query:
- **If sufficient**: Generate a document-grounded answer
- **If insufficient or unsupported**: Abstain with "I cannot answer based on the provided evidence."

This task tests the model's ability to:
- Ground answers strictly in provided evidence
- Recognize when evidence is insufficient
- Avoid hallucination by abstaining appropriately

### Data Format

| Component | Description |
|-----------|-------------|
| **Input** | Query + top-k retrieved evidence passages (max 10) |
| **Teacher Output** | Answer or abstention decision with reasoning |
| **Student Output** | Answer or abstention decision |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Embedding Similarity** | Cosine similarity between teacher and student answers (using Qwen3-Embedding-0.6B) |
| **Abstain Accuracy** | Accuracy of student abstain decisions vs teacher |
| **Abstain F1/Precision/Recall** | Classification metrics for abstain detection |
| **Exact Match Rate** | Percentage of exact answer matches |
| **Token Overlap** | Jaccard similarity of answer tokens |

---

## Project Structure

```
agent-distillation/
├── task1/                              # Task 1: Answer-Abstain QA
│   ├── scripts/
│   │   ├── 01_create_train_dataset.py      # Create training dataset from traces
│   │   ├── 02_train_model_lora.py          # LoRA fine-tuning
│   │   ├── 03_train_model_full_finetune.py # Full fine-tuning
│   │   ├── 04_create_eval_dataset.py       # Create evaluation dataset
│   │   ├── 05_model_evaluation.py          # Evaluate trained models
│   │   └── 06_analyse_visualize_results.py # Generate analysis plots
│   │   
│   ├── data/
│   │   ├── task1_dataset.csv           # Training dataset (~94MB)
│   │   ├── task1_eval_dataset.csv      # Evaluation dataset (~8MB)
│   │   ├── eval/                       # Evaluation trace data
│   │   └── synthetic_traces/           # Raw traces (gitignored, ~large)
│   ├── outputs/                        # Training & evaluation outputs (gitignored)
│   │   ├── models/                     # Trained model checkpoints
│   │   ├── evaluations/                # Evaluation results
│   │   └── analysis/                   # Visualization outputs (tracked)
│   ├── EVALUATION_METRICS.md           # Detailed metrics documentation
│   └── requirements.txt                # Python dependencies
└── README.md
```

### What's Included in the Repository

| Item | Included | Notes |
|------|----------|-------|
| Training dataset (`task1_dataset.csv`) | ✅ | ~94MB, processed from traces |
| Evaluation dataset (`task1_eval_dataset.csv`) | ✅ | ~8MB |
| All scripts | ✅ | Complete pipeline |
| Analysis outputs | ✅ | Plots and visualizations |
| Evaluation results | ✅ | All evaluation results |
| Raw traces (`synthetic_traces/`) | ❌ | Too large, regenerate with scripts |
| Trained models | ❌ | Too large, retrain with scripts |

---

## Step-by-Step Guide

### Prerequisites

```bash
cd task1
pip install -r requirements.txt
```

Required: Python 3.8+, CUDA-capable GPU

---

### Step 1: Create Training Dataset

> **Note**: The processed dataset `task1_dataset.csv` is already included. Only run this if you have raw traces.

```bash
cd task1/scripts
python 01_create_train_dataset.py
```

**Input**: Raw traces in `data/synthetic_traces/`  
**Output**: `data/task1_dataset.csv`

---

### Step 2: Train Models

#### Option A: LoRA Fine-tuning (Memory Efficient)

```bash
# Single GPU
python scripts/02_train_model_lora.py --model_name "Qwen/Qwen2.5-0.5B-Instruct"

# Multi-GPU
accelerate launch --multi_gpu --num_processes=4 scripts/02_train_model_lora.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct"
```

#### Option B: Full Fine-tuning (Better Quality)

```bash
# Single GPU
python scripts/03_train_model_full_finetune.py --model_name "Qwen/Qwen2.5-0.5B-Instruct"

# Multi-GPU
accelerate launch --multi_gpu --num_processes=4 scripts/03_train_model_full_finetune.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct"
```

**Training Arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model |
| `--num_epochs` | `3` | Training epochs |
| `--batch_size` | `2` | Per-device batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |

**Output**: `outputs/models/{model}-{lora|full-finetune}-final/`

---

### Step 3: Create Evaluation Dataset

> **Note**: The evaluation dataset `task1_eval_dataset.csv` is already included.

```bash
python scripts/04_create_eval_dataset.py
```

**Output**: `data/task1_eval_dataset.csv`

---

### Step 4: Evaluate Models

Evaluate multiple models in a single run:

```bash
accelerate launch --multi_gpu --num_processes=4 scripts/05_model_evaluation.py \
    --models \
        "outputs/models/Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-lora,lora,32" \
        "outputs/models/Qwen2.5-0.5B-Instruct-full-finetune-final,Qwen2.5-0.5B-full,full_finetune,32" \
        "Qwen/Qwen2.5-0.5B-Instruct,Qwen2.5-0.5B-base,base,32"
```

**Model config format**: `path,name,type,batch_size`

**Output**:
- `outputs/evaluations/eval_run_{timestamp}/` — Per-model results
- `{model}_detailed_results.csv` — Per-sample metrics
- `{model}_summary.json` — Aggregate statistics
- `model_comparison_summary.csv` — Cross-model comparison

---

### Step 5: Analyze and Visualize Results

```bash
python scripts/06_analyse_visualize_results.py \
    --eval_run_dir outputs/evaluations/eval_run_YYYYMMDD_HHMMSS \
    --output_dir outputs/analysis
```

**Generated Plots**:
- `answer_state_distribution.png` — Stacked bar chart of answer states
- `abstain_metrics_comparison.png` — F1, Precision, Recall, Accuracy
- `abstain_rates_comparison.png` — Student vs teacher abstain rates
- `violin_embedding-similarity-adjusted_*.png` — Distribution plots
- `heatmap_metrics.png` — Model comparison heatmap
- `training_effect_by_size.png` — Training method comparison
- `improvement_rate_by_training.png` — Improvement from base models

---

### Optional: Update Embedding Metrics

If you need to recompute embedding similarity with a different model:

```bash
python scripts/update_embedding_metrics.py \
    --eval_dir outputs/evaluations/eval_run_YYYYMMDD_HHMMSS \
    --batch_size 32
```

---

## Models Trained

The following models have been trained and evaluated:

| Model | Parameters | Training Methods |
|-------|------------|------------------|
| Gemma 3 270M IT | 270M | Base, LoRA |
| Qwen 2.5 0.5B Instruct | 0.5B | Base, LoRA, Full Finetune |
| Qwen 2.5 1.5B Instruct | 1.5B | Base, LoRA |
| Qwen 2.5 3B Instruct | 3B | Base, LoRA, Full Finetune |
| Qwen 2.5 7B Instruct | 7B | Base, LoRA |

---

## Task 1 Results and Visualizations

The evaluation results for Task 1 are available in `task1/outputs/analysis/`. Below is a summary of the key visualizations.

For detailed analysis and interpretation, see **[Task 1 Results Analysis](task1/RESULTS.md)**.

### Answer State Distribution

Shows how student models agree/disagree with teacher on answer vs abstain decisions.

![Answer State Distribution](task1/outputs/analysis/answer_state_distribution.png)

### Abstain Detection Metrics

Classification metrics (Accuracy, F1, Precision, Recall) for abstain detection.

![Abstain Metrics Comparison](task1/outputs/analysis/abstain_metrics_comparison.png)

### Embedding Similarity Distribution

Violin plots showing the distribution of semantic similarity between teacher and student answers.

![Embedding Similarity Grouped](task1/outputs/analysis/violin_embedding-similarity-adjusted_grouped.png)

### Model Performance Heatmap

Comprehensive comparison of all models across key metrics.

![Metrics Heatmap](task1/outputs/analysis/metrics_heatmap.png)

### Training Effect by Model Size

Comparison of Base, LoRA, and Full Finetune performance across model sizes.

![Training Effect by Size](task1/outputs/analysis/training_effect_by_size.png)

### Performance Improvement Rate

Percentage improvement from base model after training.

![Improvement Rate](task1/outputs/analysis/improvement_rate_by_training.png)

---

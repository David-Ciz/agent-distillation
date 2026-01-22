# Agent Distillation

Research project for distilling knowledge from LLM agents into smaller, efficient models.

## Project Structure

```
agent-distillation/
├── task1/                      # Task 1: Dataset Creation & Model Training
│   ├── scripts/
│   │   ├── 01_create_train_dataset.py   # Create task1_dataset.csv
│   │   ├── 02_train_model_lora.py       # LoRA fine-tuning
│   │   ├── 03_train_model_full_finetune.py # Full fine-tuning
│   │   └── evaluate_model.py            # Model evaluation
│   ├── data/
│   │   └── task1_dataset.csv   # Processed supervised dataset
│   ├── outputs/                # Training outputs (gitignored)
│   └── requirements.txt
└── README.md
```

## Task 1: Dataset Creation

### Description

Creates a supervised dataset from synthetic agent traces for model distillation. The script scans trace directories, extracts supervised training examples, and parses REASONING/ANSWER sections from LLM outputs.

### Input

Raw trace data in `data/synthetic_traces/` (not included in repo due to size). Each sample directory contains:
- `config.json` — Run metadata (run_uuid, teacher_id, query, search_index)
- `supervised.jsonl` — Training examples (input, output, tool, decision_label, latency_ms, tokens)

### Output

`task1_dataset.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `data_source` | Source dataset (e.g., causalqa, msmarco) |
| `run_uuid` | Unique run identifier |
| `teacher_id` | Teacher model identifier |
| `query` | Input query |
| `search_index` | Search index used |
| `llm_input` | Input prompt to the LLM |
| `llm_output` | Raw LLM output |
| `tool` | Tool called by the agent |
| `decision_label` | Decision classification |
| `latency_ms` | Response latency in milliseconds |
| `token_count` | Number of tokens |
| `llm_reasoning` | Extracted REASONING section |
| `llm_answer` | Extracted ANSWER section |

### Usage

```bash
cd task1/scripts
python 01_create_train_dataset.py
```

### Requirements

- Python 3.8+
- No external dependencies (uses only standard library)

## Model Training

### Setup

```bash
cd task1
pip install -r requirements.txt
```

---

### LoRA Fine-tuning (`02_train_model_lora.py`)

Memory-efficient fine-tuning using Low-Rank Adaptation. Trains only adapter weights.

#### Without Accelerate (auto GPU selection)
```bash
python scripts/02_train_model_lora.py
```

#### With Accelerate (multi-GPU)
```bash
accelerate launch --multi_gpu --num_processes=4 scripts/02_train_model_lora.py
```

**Default model**: `google/gemma-3-270m-it` (edit script to change)

---

### Full Fine-tuning (`03_train_model_full_finetune.py`)

Trains all model parameters. Better quality but requires more VRAM.

#### Without Accelerate (auto GPU selection)
```bash
python scripts/03_train_model_full_finetune.py
```

#### With Accelerate (multi-GPU)
```bash
accelerate launch --multi_gpu --num_processes=4 scripts/03_train_model_full_finetune.py
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model name |
| `--dataset_path` | `data/task1_dataset.csv` | Path to training CSV |
| `--output_dir` | Auto | Checkpoint directory |
| `--final_model_dir` | Auto | Final model directory |
| `--num_epochs` | `3` | Training epochs |
| `--batch_size` | `2` | Per-device batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--stats_only` | Flag | Print model stats only |

#### Examples

```bash
# Train with default model (Qwen 3B)
python scripts/03_train_model_full_finetune.py

# Train a specific model
python scripts/03_train_model_full_finetune.py --model_name "Qwen/Qwen2.5-1.5B-Instruct"
python scripts/03_train_model_full_finetune.py --model_name "Qwen/Qwen2.5-7B-Instruct"

# Custom hyperparameters
python scripts/03_train_model_full_finetune.py --num_epochs 5 --batch_size 4 --learning_rate 1e-5

# Multi-GPU with accelerate + custom model
accelerate launch --multi_gpu --num_processes=4 scripts/03_train_model_full_finetune.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct"

# Check model stats without training
python scripts/03_train_model_full_finetune.py --model_name "Qwen/Qwen2.5-7B-Instruct" --stats_only
```

---

### Outputs

All outputs saved to `task1/outputs/`:
- `training.log` / `training_full_finetune.log` — Logs
- `{model}-lora-checkpoints/` or `{model}-full-finetune-checkpoints/` — Checkpoints
- `{model}-lora-final/` or `{model}-full-finetune-final/` — Final model

### Requirements

- Python 3.8+
- CUDA-capable GPU
- See `requirements.txt` for dependencies

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/padas-lab-de/agent-distillation.git
   cd agent-distillation
   ```

2. For Task 1, the processed dataset is already included. To regenerate from raw traces:
   ```bash
   # Place raw traces in task1/data/synthetic_traces/
   cd task1/scripts
   python 01_create_train_dataset.py
   ```

3. To train a model (see [Model Training](#model-training) for details):
   ```bash
   cd task1
   pip install -r requirements.txt
   python scripts/03_train_model_full_finetune.py
   ```

## License

[Add your license]

## Citation

[Add citation information]

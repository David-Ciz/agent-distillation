# Agent Distillation

Research project for distilling knowledge from LLM agents into smaller, efficient models.

## Project Structure

```
agent-distillation/
├── task1/                      # Task 1: Dataset Creation & Model Training
│   ├── scripts/
│   │   ├── create_dataset.py   # Script to create task1_dataset.csv
│   │   └── train_model.py      # Script to fine-tune models with LoRA
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
python create_dataset.py
```

### Requirements

- Python 3.8+
- No external dependencies (uses only standard library)

## Model Training

### Description

Fine-tunes a language model using LoRA (Low-Rank Adaptation) on the supervised dataset. Supports multi-GPU training with automatic GPU selection based on available memory.

### Usage

```bash
cd task1
pip install -r requirements.txt
python scripts/train_model.py
```

### Configuration

Edit `train_model.py` to change:
- **Model**: Default is `google/gemma-3-270m-it` (other options commented in script)
- **GPU requirements**: `min_free_gb=40` (minimum free VRAM per GPU)
- **Training hyperparameters**: batch size, learning rate, epochs, etc.

### Outputs

All outputs are saved to `task1/outputs/`:
- `training.log` — Training logs
- `{model}-lora-checkpoints/` — Intermediate checkpoints
- `{model}-lora-final/` — Final trained model

### Requirements

- Python 3.8+
- CUDA-capable GPU
- See `requirements.txt` for Python dependencies

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
   python create_dataset.py
   ```

3. To train a model:
   ```bash
   # Install dependencies
   cd task1
   pip install -r requirements.txt
   
   # Run training
   python scripts/train_model.py
   ```

   To train using accelerate, run:
   ```bash
   accelerate launch --multi_gpu --num_processes=4 scripts/train_model.py
   ```

## License

[Add your license]

## Citation

[Add citation information]

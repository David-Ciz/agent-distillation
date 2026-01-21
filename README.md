# Agent Distillation

Research project for distilling knowledge from LLM agents into smaller, efficient models.

## Project Structure

```
agent-distillation/
├── task1/                      # Task 1: Dataset Creation
│   ├── scripts/
│   │   └── create_dataset.py   # Script to create task1_dataset.csv
│   ├── data/
│   │   └── task1_dataset.csv   # Processed supervised dataset
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

## License

[Add your license]

## Citation

[Add citation information]

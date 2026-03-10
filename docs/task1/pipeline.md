# Pipeline

End-to-end walkthrough of the Task 1 pipeline from raw traces to evaluation results.

---

## Step 1 — Create Training Dataset

> The processed `task1_dataset.csv` is already committed. Only run this if you have new raw traces.

```bash
python task1/scripts/01_create_train_dataset.py
```

**Input**: `task1/data/synthetic_traces/`  
**Output**: `task1/data/task1_dataset.csv` (~94 MB, ~50 k samples)

---

## Step 2 — Train Models

### LoRA (memory-efficient, recommended)

```bash
# Single GPU
python task1/scripts/02_train_model_lora.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct"

# Multi-GPU
accelerate launch --multi_gpu --num_processes=4 \
    task1/scripts/02_train_model_lora.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct"
```

### Full Fine-tune

```bash
python task1/scripts/03_train_model_full_finetune.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct"
```

**Output**: `task1/outputs/{model}-{lora|full-finetune}-final/`

| Key argument | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID |
| `--num_epochs` | `3` | Training epochs |
| `--batch_size` | `2` | Per-device batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |

---

## Step 3 — Create Evaluation Dataset

> `task1_eval_dataset.csv` is already committed.

```bash
python task1/scripts/04_create_eval_dataset.py
```

**Output**: `task1/data/task1_eval_dataset.csv` (1,886 samples)

---

## Step 4 — Evaluate Models

Single command evaluates multiple models and produces per-sample CSVs:

```bash
python task1/scripts/05_model_evaluation.py \
    --models "task1/outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-lora,lora,32" \
    --models "Qwen/Qwen2.5-0.5B-Instruct,Qwen2.5-0.5B-base,base,32"
```

Model config format: `path,name,type[,gen_batch_size]`  
Types: `lora` | `full_finetune` | `base`

**Output** under `task1/outputs/evaluations/eval_run_{timestamp}/`:

```
eval_run_20260309_120000/
├── evaluation.log
├── model_comparison_summary.csv
└── {model_name}/
    ├── {model_name}_detailed_results.csv
    └── {model_name}_summary.json
```

---

## Step 5 — Analyse & Visualise

```bash
python task1/scripts/06_analyse_visualize_results.py \
    --eval-run-dir task1/outputs/evaluations/eval_run_20260309_120000 \
    --output-dir task1/outputs/analysis/run1/
```

Produces: answer-state distribution, abstain F1 comparison, violin plots,
metrics heatmap, training-effect charts.

---

## Step 6 — TTA (optional)

See [Running the TTA experiment](../tta/running.md) for the N-pass aggregation step that can improve small-model performance without retraining.

---

## MLflow Tracking

Every training and evaluation run is logged automatically. To view:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```

Then open <http://localhost:5000>.


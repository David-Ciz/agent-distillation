# Phase 3 — MLflow Integration Plan

> **Status**: ✅ First training run successful (2026-03-06)  
> **Prerequisite**: Phase 2 complete — `lumi_env_check.py` passes all 12 required packages (including `mlflow`)  
> **Goal**: Every training and evaluation run on LUMI is fully tracked in MLflow from day one — parameters, metrics, artifacts.

---

## Run Log

| Date | Job ID | Model | Epochs | GPUs | Epoch time | Status | Notes |
|------|--------|-------|--------|------|------------|--------|-------|
| 2026-03-06 | 16513488 | Qwen2.5-0.5B-Instruct | 1 | 8× MI250X | ~27 min | ✅ Complete | First successful run. Slow due to `dataloader_num_workers=4`. Fixed. |
| 2026-03-06 | 16519490 | Qwen2.5-3B-Instruct | 3 | 8× MI250X | — | ❌ Failed | `SFTTrainer` in container's TRL 0.27.1 does not support `dataset_kwargs`. Removed. |
| 2026-03-06 | (next) | Qwen2.5-3B-Instruct | 3 | 8× MI250X | — | ⏳ Pending | Ready to submit. GPU utilisation was confirmed fine (100% across all 8 GCDs when sampled mid-step). Occasional 0% readings are normal rocm-smi sampling during DataLoader prefetch gaps. |

### GPU utilisation — confirmed working

The uneven readings (some GPUs at 0%) were a **sampling artefact**: `rocm-smi` happened to be called during the brief pause between training steps while the DataLoader prefetches the next batch. A second sample taken mid-step showed all 8 GCDs at 100%. The `ntasks-per-node=1` + `torch.distributed.run --standalone` pattern is correct and should not be changed.

### Performance notes — job 16513488

- **Tokenization took ~7 min per rank** — each rank processed the full 16K dataset independently with 4 DataLoader workers. Fix: `dataloader_num_workers=1`. Applied to script. (`dataset_kwargs={"num_proc":1}` was also tried but is not supported by TRL 0.27.1 in the container — removed.)
- **Actual training**: 257 steps × 3.75 s/it = ~16 min. This is the real GPU compute time and is reasonable for 0.5B on 8 GPUs.
- **Expected next run**: ~20–25 min total.

---

## Table of Contents

1. [Storage Layout on LUMI](#1-storage-layout-on-lumi)
2. [What to Track](#2-what-to-track)
3. [Step-by-Step Implementation](#3-step-by-step-implementation)
   - [Step 3.1 — Shared MLflow config helper](#step-31--shared-mlflow-config-helper)
   - [Step 3.2 — Integrate MLflow into `02_train_model_lora.py`](#step-32--integrate-mlflow-into-02_train_model_lorapy)
   - [Step 3.3 — Integrate MLflow into `03_train_model_full_finetune.py`](#step-33--integrate-mlflow-into-03_train_model_full_finetunepy)
   - [Step 3.4 — Integrate MLflow into `05_model_evaluation.py`](#step-34--integrate-mlflow-into-05_model_evaluationpy)
   - [Step 3.5 — Update SLURM job script](#step-35--update-slurm-job-script)
4. [Running & Verifying](#4-running--verifying)
5. [Viewing Results](#5-viewing-results)
6. [Checklist](#6-checklist)

---

## 1. Storage Layout on LUMI

MLflow has two separate storage concerns with very different size characteristics:

| Component | What it stores | Size | Where to put it on LUMI |
|-----------|---------------|------|------------------------|
| **Backend store** | Run metadata, params, metrics (SQLite) | Tiny (~MB) | `$HOME/mlflow/mlflow.db` — **per user** |
| **Artifact store** | Model checkpoints, CSVs, plots | Large (GB) | `/scratch/project_465002758/$USER/mlruns/` — **per user** |

**Why per-user SQLite instead of a shared DB?**

SQLite uses file-level locking — a shared DB between two users running concurrent SLURM jobs will cause lock contention and potential corruption. Each user keeps their own DB:

```
/users/daciz/mlflow/mlflow.db        ← your runs
/users/colleague/mlflow/mlflow.db    ← their runs
```

**To compare results together**, copy both DBs locally and view them side by side (see Section 5).

**Why this split?**
- Home (`$HOME`, ~50 GB quota) is fine for the SQLite DB — it's just text rows.
- `/scratch` is LUMI's large Lustre parallel filesystem — right for big sequential writes (checkpoints).
- `/flash` (NVMe) is reserved for hot training data I/O — don't put MLflow there.

**Set these environment variables** (add to your `~/.bashrc` on LUMI or source before each job):

```bash
export MLFLOW_TRACKING_URI="sqlite:////users/daciz/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/daciz/mlruns"
mkdir -p ~/mlflow
mkdir -p /scratch/project_465002758/daciz/mlruns
```

> ⚠️ Note the **four slashes** in `sqlite:////` — that's three slashes for the `sqlite:///` scheme plus one for the absolute path.

> 💡 **Future upgrade path**: if you later want always-on shared results, point both `MLFLOW_TRACKING_URI` env vars at a small external server (e.g. Hetzner VPS running `mlflow server --backend-store-uri postgresql://...`). No code changes needed — only the env var changes.

---

> 📝 **CLI note**: all scripts use **Click** for argument parsing (not argparse). Arguments use `--kebab-case` (e.g. `--model-name`, `--num-epochs`). Run any script with `--help` to see all options.

---

## 2. What to Track

### Training runs (`02` / `03`)

```
Experiment name: "lora-training"  /  "full-finetune-training"
```

**Parameters** (logged once at run start):

| Parameter | Example value | Source |
|-----------|--------------|--------|
| `model_name` | `"Qwen/Qwen2.5-0.5B-Instruct"` | script arg |
| `method` | `"lora"` or `"full_finetune"` | script name |
| `lora_rank` | `16` | LoraConfig |
| `lora_alpha` | `32` | LoraConfig |
| `lora_dropout` | `0.05` | LoraConfig |
| `lora_target_modules` | `"q_proj,k_proj,v_proj,..."` | LoraConfig |
| `num_epochs` | `3` | TrainingArguments |
| `per_device_train_batch_size` | `4` | TrainingArguments |
| `gradient_accumulation_steps` | `2` | TrainingArguments |
| `learning_rate` | `2e-4` | TrainingArguments |
| `lr_scheduler_type` | `"cosine"` | TrainingArguments |
| `warmup_steps` | `100` | TrainingArguments |
| `bf16` | `True` | TrainingArguments |
| `optim` | `"adamw_torch_fused"` | TrainingArguments |
| `max_seq_length` | `2048` | SFTTrainer |
| `dataset_path` | `"task1/data/task1_dataset.csv"` | resolved path |
| `dataset_sha256` | `"a3f9..."` | hash at load time |
| `train_samples` | `8000` | after split |
| `val_samples` | `2000` | after split |
| `val_split` | `0.2` | constant |
| `num_gpus` | `1` | `torch.cuda.device_count()` |
| `gpu_type` | `"AMD Instinct MI250X"` | `torch.cuda.get_device_name(0)` |
| `container_sif` | `"/appl/local/laifs/..."` | `$SIF` env var |
| `slurm_job_id` | `"16510235"` | `$SLURM_JOB_ID` env var |

**Metrics** (logged per step/epoch via a custom callback):

| Metric | When |
|--------|------|
| `train/loss` | every `logging_steps` |
| `train/learning_rate` | every `logging_steps` |
| `eval/loss` | end of each epoch (if eval set provided) |
| `epoch_duration_sec` | end of each epoch |
| `total_train_duration_sec` | end of run |

**Artifacts** (logged at end of run):

- `training_args.json` — full `TrainingArguments` dump
- `lora_config.json` — full `LoraConfig` dump  
- Final model checkpoint directory (logged as artifact folder)

---

### Evaluation runs (`05`)

```
Experiment name: "evaluation"
```

**Parameters**:

| Parameter | Example value |
|-----------|--------------|
| `model_path` | `"outputs/models/Qwen2.5-0.5B-Instruct-lora-final"` |
| `model_label` | `"Qwen2.5-0.5B-lora"` |
| `model_type` | `"lora"` / `"full_finetune"` / `"base"` |
| `eval_dataset_path` | `"task1/data/task1_eval_dataset.csv"` |
| `eval_dataset_sha256` | `"b7c2..."` |
| `num_eval_samples` | `1886` |
| `batch_size` | `32` |
| `max_new_tokens` | `256` |
| `linked_train_run_id` | MLflow run ID of the training run that produced this model |

**Metrics**:

| Metric | Description |
|--------|-------------|
| `abstain_f1` | Main metric |
| `abstain_accuracy` | |
| `abstain_precision` | |
| `abstain_recall` | |
| `embedding_similarity_adjusted` | Semantic similarity (non-abstain pairs) |
| `exact_match_avg` | |
| `token_overlap_avg` | |
| `both_answer_rate` | Fraction of samples where both models answered |
| `both_abstain_rate` | |
| `teacher_abstain_rate` | Should be ~70% |
| `student_abstain_rate` | |

**Artifacts**:
- `detailed_results.csv` — per-sample scores
- `summary.json` — aggregated metrics

---

## 3. Step-by-Step Implementation

### Step 3.1 — Shared MLflow config helper

Create `task1/scripts/mlflow_utils.py` — a small module imported by all scripts so MLflow setup is in one place.

```python
# task1/scripts/mlflow_utils.py
import os
import hashlib
import mlflow

def setup_mlflow(experiment_name: str) -> None:
    """Configure MLflow tracking URI and set/create the experiment."""
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "sqlite:////" + os.path.expanduser("~/mlflow/mlflow.db")
    )
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT", None)

    mlflow.set_tracking_uri(tracking_uri)

    # If artifact root is set and no experiment exists yet, create with custom artifact location
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None and artifact_root:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=os.path.join(artifact_root, experiment_name)
        )

    mlflow.set_experiment(experiment_name)


def hash_file(path: str, algo: str = "sha256") -> str:
    """Return hex digest of a file (used to fingerprint datasets)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

---

### Step 3.2 — Integrate MLflow into `02_train_model_lora.py`

**Changes needed** (in order):

1. **Remove NVIDIA-smi GPU selection** — the `get_gpus_with_free_memory()` function and the `CUDA_VISIBLE_DEVICES` override block at the top of `main()`. On LUMI, SLURM allocates GPUs; setting `CUDA_VISIBLE_DEVICES` manually breaks distributed training.

2. **Replace hardcoded values with a Click CLI** — `model_name`, `num_epochs`, `batch_size`, `learning_rate`, LoRA hyperparams, `val_split`, and `mlflow_experiment` are all Click options with sensible defaults. Usage:
   ```bash
   python 02_train_model_lora.py --help
   python 02_train_model_lora.py --model-name Qwen/Qwen2.5-0.5B-Instruct --num-epochs 1
   ```

3. **Add MLflow setup** at the top of `main()`:

```python
from mlflow_utils import setup_mlflow, hash_file
import mlflow

setup_mlflow("lora-training")

with mlflow.start_run(run_name=f"{model_name_safe}-lora") as run:
    # log params
    mlflow.log_params({
        "model_name": model_name,
        "method": "lora",
        "lora_rank": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "lora_target_modules": ",".join(peft_config.target_modules),
        "num_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "bf16": training_args.bf16,
        "optim": training_args.optim,
        "dataset_path": csv_path,
        "dataset_sha256": hash_file(csv_path),
        "train_samples": len(train_df),
        "val_samples": len(test_df),
        "val_split": 0.2,
        "num_gpus": torch.cuda.device_count(),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "container_sif": os.environ.get("SIF", "local"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    })

    # ... rest of training ...

    # After trainer.train():
    trainer.save_model(output_path)

    # Log artifacts
    mlflow.log_artifact(os.path.join(output_path))       # model checkpoint
    mlflow.log_dict(training_args.to_dict(), "training_args.json")
```

4. **Add MLflow callback** so per-step loss is streamed live (not just at the end):

```python
from transformers import TrainerCallback

class MLflowMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and mlflow.active_run():
            # only log scalar values
            scalar_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(scalar_logs, step=state.global_step)
```

Pass it to `SFTTrainer(callbacks=[MLflowMetricsCallback()])`.

5. **Guard with `LOCAL_RANK`** — in distributed training, only rank 0 should log to MLflow:

```python
is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
if is_main_process:
    setup_mlflow("lora-training")
    mlflow.start_run(...)
```

---

### Step 3.3 — Integrate MLflow into `03_train_model_full_finetune.py`

Same changes as Step 3.2, with these differences:
- Experiment name: `"full-finetune-training"`
- No `peft_config` block — log `method: "full_finetune"` instead
- Also remove the `select_gpus_before_torch()` / nvidia-smi block (same LUMI reason)
- Replace `attn_implementation="flash_attention_2"` with a ROCm-safe fallback:

```python
# ROCm-safe attention implementation
try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
except ImportError:
    attn_impl = "sdpa"  # PyTorch scaled dot-product attention — works on ROCm

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    trust_remote_code=True,
)
```

---

### Step 3.4 — Integrate MLflow into `05_model_evaluation.py`

This script already has a `--models` CLI interface. Changes:

1. Import and setup:
```python
from mlflow_utils import setup_mlflow, hash_file
import mlflow
setup_mlflow("evaluation")
```

2. For each model being evaluated, wrap in its own MLflow run:
```python
with mlflow.start_run(run_name=model_label):
    mlflow.log_params({
        "model_path": model_path,
        "model_label": model_label,
        "model_type": model_type,
        "eval_dataset_path": eval_csv_path,
        "eval_dataset_sha256": hash_file(eval_csv_path),
        "num_eval_samples": len(eval_df),
        "batch_size": batch_size,
    })

    # ... run evaluation ...

    mlflow.log_metrics({
        "abstain_f1": results["abstain_f1"],
        "abstain_accuracy": results["abstain_accuracy"],
        "abstain_precision": results["abstain_precision"],
        "abstain_recall": results["abstain_recall"],
        "embedding_similarity_adjusted": results["embedding_similarity_adjusted"],
        "exact_match_avg": results["exact_match_avg"],
        "token_overlap_avg": results["token_overlap_avg"],
        "teacher_abstain_rate": results["teacher_abstain_rate"],
        "student_abstain_rate": results["student_abstain_rate"],
        "both_answer_rate": results["both_answer_rate"],
        "both_abstain_rate": results["both_abstain_rate"],
    })

    mlflow.log_artifact(detailed_results_csv_path)
    mlflow.log_artifact(summary_json_path)
```

---

### Step 3.5 — Update SLURM job script

Edit `task1/scripts/slurm/train_lora_lumi.sh` to:
1. Set the correct `MLFLOW_TRACKING_URI` and `MLFLOW_ARTIFACT_ROOT`
2. Activate the venv before running the container
3. Pass `--model_name`, `--num_epochs`, `--batch_size` as CLI args

```bash
export MLFLOW_TRACKING_URI="sqlite:////users/daciz/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/daciz/mlruns"

srun singularity run "$SIF" \
    bash -c 'source ~/agent-distillation/my-env/bin/activate && \
    python -m torch.distributed.run \
        --nproc_per_node=8 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
    task1/scripts/02_train_model_lora.py \
        --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
        --num-epochs 1 \
        --batch-size 4 \
        --gradient-accumulation-steps 2'
```

---

## 4. Running & Verifying

### Smoke test (1 GPU, 1 epoch, small model) before full run

```bash
# On LUMI — quick interactive test first, not a full sbatch job
srun -A project_465002758 -p small-g -n 1 --gpus-per-task=1 \
    singularity run $SIF \
    bash -c 'source ~/agent-distillation/my-env/bin/activate && \
    MLFLOW_TRACKING_URI="sqlite:////users/daciz/mlflow/mlflow.db" \
    MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/daciz/mlruns" \
    python task1/scripts/02_train_model_lora.py \
        --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
        --num_epochs 1 \
        --batch_size 4'
```

**Success criteria:**
- [ ] Job runs to completion without OOM or ROCm errors
- [ ] `~/mlflow/mlflow.db` exists and has content (check with `sqlite3 ~/mlflow/mlflow.db "SELECT run_uuid, status FROM runs;"`)
- [ ] Checkpoint directory created under `/scratch/.../mlruns/`
- [ ] MLflow run appears in UI (see Section 5)

---

## 5. Viewing Results

### Option A — Copy DB locally and view in UI (easiest, single user)

From your **local machine**:

```bash
scp daciz@lumi.csc.fi:~/mlflow/mlflow.db ./mlflow_lumi.db
MLFLOW_TRACKING_URI="sqlite:////$(pwd)/mlflow_lumi.db" mlflow ui --port 5001
# Open http://localhost:5001
```

> Artifacts won't be browseable this way (they're on LUMI scratch), but all params and metrics will be.

---

### Option B — Compare your runs + colleague's runs together

Copy both DBs locally and merge them with a small script:

```bash
# Fetch both DBs
scp daciz@lumi.csc.fi:~/mlflow/mlflow.db ./david.db
scp colleague@lumi.csc.fi:~/mlflow/mlflow.db ./colleague.db
```

```python
# merge_mlflow_dbs.py  — run locally, one-off
import sqlite3, shutil, os

shutil.copy("david.db", "merged.db")
conn = sqlite3.connect("merged.db")
conn.execute("ATTACH DATABASE 'colleague.db' AS src")

# Tables that need merging (MLflow schema)
tables = [
    "experiments", "runs", "params", "metrics",
    "tags", "latest_metrics", "run_data",
]
for table in tables:
    try:
        conn.execute(f"INSERT OR IGNORE INTO main.{table} SELECT * FROM src.{table}")
    except Exception as e:
        print(f"  {table}: {e}")

conn.commit()
conn.close()
print("Done → merged.db")
```

```bash
MLFLOW_TRACKING_URI="sqlite:////$(pwd)/merged.db" mlflow ui --port 5001
# Both your runs and your colleague's runs appear in one UI
```

Run `merge_mlflow_dbs.py` whenever you want a fresh combined view — it's idempotent (`INSERT OR IGNORE`).

---

### Option C — SSH port forward (live view during training)

```bash
# On LUMI login node — start the MLflow server
mlflow server \
    --backend-store-uri sqlite:////users/daciz/mlflow/mlflow.db \
    --default-artifact-root /scratch/project_465002758/daciz/mlruns \
    --host 0.0.0.0 --port 5000

# On local machine — open tunnel
ssh -L 5000:localhost:5000 daciz@lumi.csc.fi
# Open http://localhost:5000
```

> ⚠️ LUMI login nodes are shared — don't run the server for long periods. Use it during active training sessions only.

---

## 6. Checklist

### Implementation
- [x] Create `task1/scripts/mlflow_utils.py` (Step 3.1)
- [x] Update `02_train_model_lora.py` — remove nvidia-smi, add Click CLI, add MLflow (Step 3.2)
- [x] Update `03_train_model_full_finetune.py` — same + ROCm attn fallback (Step 3.3)
- [ ] Update `05_model_evaluation.py` — add MLflow per-model run (Step 3.4)
- [x] Update `task1/scripts/slurm/train_lora_lumi.sh` — add MLflow env vars, venv activation (Step 3.5)

### Infrastructure on LUMI
- [x] Create MLflow DB directory: `mkdir -p ~/mlflow`
- [x] Create artifact directory: `mkdir -p /scratch/project_465002758/daciz/mlruns`
- [x] Install missing packages into venv: `pip install mlflow click`
- [x] Verify imports: `python -c "import mlflow, click; print(mlflow.__version__, click.__version__)"`

### Validation
- [x] Smoke test passes — job 16513488 completed 1 epoch successfully
- [x] `~/mlflow/mlflow.db` created and populated (MLflow DB initialised on first run)
- [ ] MLflow UI verified locally with correct params + loss curve
- [ ] Checkpoint artifact visible in `/scratch/.../mlruns/`
- [x] Performance fix applied: `dataloader_num_workers=1` (container's TRL 0.27.1 does not support `dataset_kwargs` — removed)


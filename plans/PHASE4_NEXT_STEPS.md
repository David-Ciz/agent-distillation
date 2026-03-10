# Phase 4 — Next Steps: Reproduce, Evaluate, Analyse

> **Status**: Ready to execute  
> **Date created**: 2026-03-06  
> **Prerequisite**: Phase 3 complete — first training job (job 16513488, Qwen2.5-0.5B LoRA, 1 epoch) ran successfully on LUMI with MLflow tracking.  
> **Goal**: Train a full 3-epoch LoRA run, evaluate it, confirm analysis pipeline works end-to-end, then run full finetune and build out the comparison table.

---

## Workflow (How Experiments Are Run)

```
On LUMI:
  1. bash submit_train_sweep.sh      # submits one sbatch job per model, all in parallel
  2. [wait for all training jobs]
  3. bash submit_eval_sweep.sh       # submits one eval job covering all models at once
  4. [wait for eval job]

Locally:
  5. bash sync_results.sh            # rsync CSVs + mlflow.db from LUMI
  6. .venv/bin/python 06_analyse_visualize_results.py --eval-run-dir ... --output-dir ...
  7. mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

**Tool responsibilities:**
| Tool | Does |
|------|------|
| SLURM scripts | Run training/eval on LUMI GPUs |
| MLflow | Track all params, metrics, artifacts per run |
| DVC | Version `task1_dataset.csv` + `task1_eval_dataset.csv` only |
| `06` script | Publication-quality comparison plots, run locally after rsync |

---

## Context & Current State

### What we have
- ✅ Training pipeline working on LUMI (8× MI250X, Singularity container, MLflow tracking)
- ✅ `task1_dataset.csv` on LUMI at `/users/daciz/agent-distillation/task1/data/`
- ✅ `task1_eval_dataset.csv` on LUMI (needed for step 4.3)
- ✅ Baseline results from original paper runs (see `task1/RESULTS.md`)
- ✅ `05_model_evaluation.py` — converted to Click + MLflow ✅
- ✅ `06_analyse_visualize_results.py` — converted to Click ✅
- ✅ `task1/scripts/slurm/eval_lumi.sh` — created ✅
- ✅ `task1/scripts/slurm/train_full_finetune_lumi.sh` — created ✅
- ✅ `Qwen2.5-3B-Instruct` LoRA training — 3 epochs complete, adapter at `task1/outputs/Qwen_Qwen2.5-3B-Instruct-lora-final/`
- ⏳ Evaluation job running (SLURM job 16593112) — Qwen2.5-3B-Instruct LoRA + base

### Baseline numbers to reproduce
| Model | Method | Abstain F1 | Embed Sim Adjusted |
|-------|--------|-----------|-------------------|
| Qwen 2.5 0.5B | LoRA | 0.815 | — |
| Qwen 2.5 3B | LoRA | 0.881 | — |
| Qwen 2.5 0.5B | Full FT | 0.802 | — |
| Qwen 2.5 3B | Full FT | — | — |

---

## GPU Utilisation Monitoring on LUMI

Before submitting the next big job, set up monitoring to confirm all 8 GPUs are actually being used.

### Option A — `rocm-smi` inside a running job (simplest)

SSH into the allocated node while the job runs:

```bash
# Find which node your job is on
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"

# SSH into it (LUMI allows this while your job holds the allocation)
ssh nid00XXXX

# Watch GPU utilisation in real time (refreshes every 2s)
watch -n 2 rocm-smi
```

Key columns to check:
- **GPU use%** — should be 80–100% during forward/backward passes
- **GPU Memory** — should be 50–60 GB / 64 GB per GCD for 3B model
- **Temp** — informational

### Option B — Log GPU stats to a file during the job

Add this to `train_lora_lumi.sh` right before the `srun` line:

```bash
# Background GPU monitor — writes to logs/gpu_stats_$SLURM_JOB_ID.log every 30s
singularity run "$SIF" bash -c "
    while true; do
        echo \"--- \$(date) ---\" >> ${REPO_DIR}/logs/gpu_stats_${SLURM_JOB_ID}.log
        rocm-smi --showuse --showmeminfo vram >> ${REPO_DIR}/logs/gpu_stats_${SLURM_JOB_ID}.log 2>&1
        sleep 30
    done
" &
GPU_MONITOR_PID=$!
```

And after the `srun` line:

```bash
kill $GPU_MONITOR_PID 2>/dev/null || true
```

### Option C — LUMI web portal

Go to [https://www.lumi.csc.fi/](https://www.lumi.csc.fi/) → **My Jobs** → click your job ID → **Efficiency** tab. Shows CPU/GPU utilisation after the job completes. Useful for post-hoc analysis but not real-time.

### What to look for
- GPU use consistently **< 50%** → batch size too small, increase `--batch-size` or `--gradient-accumulation-steps`
- GPU memory **< 30 GB** → model fits comfortably, can increase batch size
- GPU use **spiky** (0% → 100% → 0%) → DataLoader is the bottleneck (CPU preprocessing between batches)

---

## Step-by-Step Plan

---

### Step 4.1 — Wait for / submit 3-epoch Qwen 3B LoRA training

The `train_lora_lumi.sh` is already configured for this (attached context shows `--model-name Qwen/Qwen2.5-3B-Instruct --num-epochs 3`).

**If not yet submitted:**
```bash
cd ~/agent-distillation
sbatch task1/scripts/slurm/train_lora_lumi.sh
```

**Monitor:**
```bash
squeue -u $USER
tail -f logs/train_lora_<JOB_ID>.out
```

**Expected output:**
- `task1/outputs/Qwen_Qwen2.5-3B-Instruct-lora-final/` — saved LoRA adapter
- MLflow run in `~/mlflow/mlflow.db` with train loss curve
- Estimated wall time: ~2–3 hours (3 epochs × ~45 min/epoch for 3B on 8 GPUs)

**Success criteria:**
- Job exits with code 0
- `task1/outputs/Qwen_Qwen2.5-3B-Instruct-lora-final/adapter_config.json` exists
- MLflow shows `status=FINISHED`

---

### Step 4.2 — Convert `05_model_evaluation.py` to Click + add MLflow

> ⚠️ Do this **before** running evaluation so results are tracked from the start.

The script currently uses `argparse`. Convert to Click (same pattern as `02` and `03`) and add MLflow logging.

**Model configs to evaluate** (mirrors the original baseline run):

```
Qwen2.5-3B-Instruct LoRA (newly trained)     — primary result
Qwen2.5-3B-Instruct base                      — sanity check: should be much lower
```

**Click interface to implement:**

```python
@click.command()
@click.option("--models", multiple=True, required=True,
    help="Model config: 'path,name,type,gen_batch_size'. Repeat for multiple models.")
@click.option("--eval-dataset",
    default=os.path.join(DATA_DIR, "task1_eval_dataset.csv"), show_default=True,
    type=click.Path(exists=True))
@click.option("--output-dir", default=EVAL_OUTPUT_DIR, show_default=True)
@click.option("--num-samples", default=0, show_default=True,
    help="Max samples to evaluate (0 = all).")
@click.option("--batch-size", default=8, show_default=True)
@click.option("--gen-batch-size", default=32, show_default=True)
@click.option("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B", show_default=True)
@click.option("--loss-only", is_flag=True)
@click.option("--mlflow-experiment", default="evaluation", show_default=True)
```

**MLflow to add** (per model evaluated, in a nested run):

```python
with mlflow.start_run(run_name=f"eval-{cfg['name']}"):
    mlflow.log_params({
        "model_path": cfg['path'],
        "model_name": cfg['name'],
        "model_type": cfg['type'],
        "eval_dataset_sha256": hash_file(eval_dataset_path),
        "num_eval_samples": len(eval_df),
        "embedding_model": embedding_model,
    })
    # ... run evaluation ...
    mlflow.log_metrics({
        "abstain_f1": metrics['abstain_f1'],
        "abstain_accuracy": metrics['abstain_accuracy'],
        "abstain_precision": metrics['abstain_precision'],
        "abstain_recall": metrics['abstain_recall'],
        "embedding_similarity_adjusted": metrics['embedding_similarity_adjusted_avg'],
        "exact_match_avg": metrics['exact_match_avg'],
        "token_overlap_avg": metrics['token_overlap_avg'],
        "teacher_abstain_rate": metrics['teacher_abstain_rate'],
        "student_abstain_rate": metrics['student_abstain_rate'],
    })
    mlflow.log_artifact(detailed_results_csv_path)
    mlflow.log_artifact(summary_json_path)
```

**SLURM script to create:** `task1/scripts/slurm/eval_lumi.sh`

```bash
#!/bin/bash
#SBATCH --job-name=agent-distill-eval
#SBATCH --account=project_465002758
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1          # evaluation only needs 1 GPU
#SBATCH --time=02:00:00
#SBATCH --output=/users/%u/agent-distillation/logs/eval_%j.out
#SBATCH --error=/users/%u/agent-distillation/logs/eval_%j.err

REPO_DIR="${HOME}/agent-distillation"
source "${REPO_DIR}/task1/scripts/slurm/container.env"
export MLFLOW_TRACKING_URI="sqlite:////users/${USER}/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="/scratch/project_465002758/${USER}/mlruns"

srun singularity run "$SIF" \
    bash -c "
        [ -n \"${CONTAINER_VENV}\" ] && source \"${CONTAINER_VENV}/bin/activate\"
        python task1/scripts/05_model_evaluation.py \
            --models 'task1/outputs/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-lora,lora,32' \
            --models 'Qwen/Qwen2.5-3B-Instruct,Qwen2.5-3B-base,base,32'
    "
```

---

### Step 4.3 — Verify analysis pipeline (`06_analyse_visualize_results.py`)

Run `06` on the new eval output and confirm the plots match the structure of existing ones in `task1/outputs/analysis/`.

```bash
# On LUMI (or locally if you rsync the eval results down)
python task1/scripts/06_analyse_visualize_results.py \
    --results_dirs task1/outputs/evaluations/eval_run_<timestamp>/ \
    --output_dir task1/outputs/analysis/lumi_3B_lora/
```

**Checks:**
- [ ] `metrics_summary.csv` produced with correct columns
- [ ] All 9 plot types generated (violin, heatmap, abstain rates, etc.)
- [ ] Abstain F1 for Qwen 3B LoRA is in range 0.870–0.900 (baseline: 0.881)
- [ ] If numbers differ significantly from baseline, investigate: different split? different eval dataset?

> The `06` script uses `MODEL_GROUPS` and `get_train_type()` to group models — make sure the model names passed to `05` match the naming convention expected by `06`. Convention: `Qwen2.5-3B-Instruct-lora`, `Qwen2.5-3B-Instruct-base` etc.

---

### Step 4.4 — Train Qwen 2.5 0.5B LoRA (3 epochs) for direct baseline comparison

The original baseline used 0.5B for both LoRA and full finetune. Training it on LUMI with our new pipeline verifies the numbers are reproducible and gives us a properly MLflow-tracked reference point.

Update `train_lora_lumi.sh` and submit:

```bash
--model-name 'Qwen/Qwen2.5-0.5B-Instruct' \
--num-epochs 3 \
--batch-size 4 \
--gradient-accumulation-steps 2
```

Expected wall time: ~1 hour. Target F1: ~0.815.

---

### Step 4.5 — Train Qwen 2.5 3B Full Finetune

Use `03_train_model_full_finetune.py` via a new SLURM script `train_full_finetune_lumi.sh` (copy `train_lora_lumi.sh`, point at `03_train_model_full_finetune.py`).

```bash
python task1/scripts/03_train_model_full_finetune.py \
    --model-name 'Qwen/Qwen2.5-3B-Instruct' \
    --num-epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 8
```

> ⚠️ 3B full finetune is memory-intensive. With `batch_size=1, grad_accum=8` → effective batch = 8.
> Gradient checkpointing is already enabled in the script.
> Expected GPU memory: ~50–55 GB / 64 GB per GCD.
> Expected wall time: ~6–8 hours — request `--time=10:00:00`.

---

### Step 4.6 — Evaluate all models together and run full analysis

Once steps 4.1–4.5 are complete, run a single evaluation job covering all 4 models:

```bash
python task1/scripts/05_model_evaluation.py \
            --models 'task1/outputs/Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora,32' \
    --models 'task1/outputs/Qwen_Qwen2.5-3B-Instruct_full-finetune-final,Qwen2.5-3B-Instruct-full-finetune,full_finetune,32' \
    --models 'task1/outputs/Qwen_Qwen2.5-0.5B-Instruct_lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32' \
    --models 'Qwen/Qwen2.5-3B-Instruct,Qwen2.5-3B-Instruct-base,base,32' \
    --models 'Qwen/Qwen2.5-0.5B-Instruct,Qwen2.5-0.5B-Instruct-base,base,32'
```

Then run `06` to produce the full comparison plots.

**Expected deliverable**: a `metrics_summary.csv` and full plot set comparable to `task1/outputs/analysis/` that can be directly appended to `RESULTS.md`.

---

## Parallel / Opportunistic Tasks

These don't block the main flow and can be done while jobs are running:

| Task | Who | Effort | Notes |
|------|-----|--------|-------|
| ~~Convert `06_analyse_visualize_results.py` to Click~~ | ~~Agent~~ | ~~1h~~ | ✅ Done |
| ~~Write `task1/scripts/slurm/eval_lumi.sh`~~ | ~~Agent~~ | ~~30m~~ | ✅ Done |
| ~~Write `task1/scripts/slurm/train_full_finetune_lumi.sh`~~ | ~~Agent~~ | ~~15m~~ | ✅ Done |
| Pull MLflow DB locally and verify UI | Human | 15m | `scp daciz@lumi.csc.fi:~/mlflow/mlflow.db .` then `mlflow ui` |
| ~~Add GPU monitor to SLURM script~~ | ~~Agent~~ | ~~15m~~ | ✅ Done (in train_lora_lumi.sh) |

---

## Success Criteria for Phase 4

- [x] Qwen 3B LoRA 3-epoch training completes on LUMI
- [x] `05_model_evaluation.py` converted to Click + MLflow
- [x] Evaluation run produces `detailed_results.csv` + `summary.json` for all models
- [x] `06_analyse_visualize_results.py` converted to Click
- [ ] `06_analyse_visualize_results.py` produces plots successfully ← **next**
- [x] Abstain F1 for Qwen 3B LoRA within ±0.01 of baseline (got 0.875 vs 0.881 ✅)
- [ ] All runs visible in MLflow UI
- [ ] `RESULTS.md` updated with new LUMI run numbers ← done for 3B, pending 0.5B + full FT

---

## What Comes After Phase 4

Once the baseline is reproduced on LUMI with clean MLflow tracking:

1. **TTA / Self-Consistency experiment** (`07_tta_experiment.py`) — see `plans/PLAN.md` Section 4.2
2. **Sweep smaller models** — Gemma 270M, Qwen 0.5B with both methods
3. **Larger model** — Qwen 7B LoRA (the current best at 0.907 F1)
4. **DVC setup** — by this point datasets will be stable and collaboration will benefit from it


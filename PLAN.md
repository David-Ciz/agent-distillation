# Agent Distillation — Project Plan

> **Last updated**: March 4, 2026  
> **Status**: Active  
> **Goal**: Distill GPT-4o behavior into a model small enough to run in-browser (target: ≤ 500M params), and investigate Test-Time Augmentation (TTA/self-consistency) as a way to boost inference-time performance without additional training.

---

## Table of Contents

1. [Project Understanding](#1-project-understanding)
2. [Immediate Next Steps — Get a Baseline](#2-immediate-next-steps--get-a-baseline)
3. [Tooling & Infrastructure Setup](#3-tooling--infrastructure-setup)
4. [Experiment Roadmap](#4-experiment-roadmap)
5. [LUMI HPC / AMD GPU Optimisation](#5-lumi-hpc--amd-gpu-optimisation)
6. [Open Questions & Notes from Meeting](#6-open-questions--notes-from-meeting)
7. [Backlog / Future Work](#7-backlog--future-work)

---

## 1. Project Understanding

### What the pipeline does

```
Raw traces (GPT-4o outputs)
        │
        ▼  01_create_train_dataset.py
task1_dataset.csv  (94 MB, ~training pairs)
        │
        ▼  02 / 03  train_model_lora.py / train_model_full_finetune.py
Trained student checkpoint
        │
        ▼  05_model_evaluation.py
Per-sample CSV + summary JSON
        │
        ▼  06_analyse_visualize_results.py
Plots + metrics_summary.csv
```

### Current results snapshot (baseline, already run)

| Model | Params | Method | Abstain F1 | Embed Sim |
|-------|--------|--------|-----------|-----------|
| Gemma 3 270M | 270M | LoRA | 0.814 | — |
| Qwen 2.5 0.5B | 0.5B | LoRA | 0.815 | — |
| Qwen 2.5 0.5B | 0.5B | Full FT | 0.802 | — |
| Qwen 2.5 1.5B | 1.5B | LoRA | 0.845 | — |
| Qwen 2.5 3B | 3B | LoRA | 0.881 | — |
| Qwen 2.5 7B | 7B | LoRA | **0.907** | **0.851** |

Teacher abstain rate: **70 %**. All models evaluated on **1 886 samples**.

---

## 2. Immediate Next Steps — Get a Baseline

These steps must be completed **before** any new training, so we have a clean reference point. The order matters — do not skip phases.

### 2.1 Environment Setup (local)

```bash
cd /Users/davidciz/Work/agent-distillation

# Install project (editable, via pyproject.toml)
pip install -e ".[dev]"

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 2.2 Install & Initialise DVC

> See [Section 3.1](#31-dvc--dataset-versioning) for rationale.

```bash
pip install dvc dvc-s3   # or dvc-gdrive / dvc-ssh depending on remote
dvc init
dvc add task1/data/task1_dataset.csv
dvc add task1/data/task1_eval_dataset.csv
git add task1/data/*.dvc .dvcignore
git commit -m "chore: track datasets with DVC"
```

### 2.3 Start MLflow Tracking Server (local)

> See [Section 3.2](#32-mlflow-experiment-tracking) for rationale.

```bash
pip install mlflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
# Open http://localhost:5000
```

### 2.4 Reproduce Evaluation Baseline

Run evaluation on the **already-trained models** (from `outputs/evaluations/eval_run_20260124_052629/`) through the **unmodified** script. This logs the baseline into MLflow so every future run is comparable.

```bash
# From task1/ directory
python scripts/05_model_evaluation.py \
    --models \
        "outputs/models/gemma-3-270m-it-lora-final,gemma-3-270m-it-lora,lora,32" \
        "outputs/models/Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-lora,lora,32" \
        "outputs/models/Qwen2.5-0.5B-Instruct-full-finetune-final,Qwen2.5-0.5B-full,full_finetune,32" \
        "Qwen/Qwen2.5-0.5B-Instruct,Qwen2.5-0.5B-base,base,32"
```

Expected output: `outputs/evaluations/eval_run_<timestamp>/`

---

## 3. Tooling & Infrastructure Setup

### 3.1 DVC — Dataset Versioning

**Why**: The training (`task1_dataset.csv`, 94 MB) and evaluation (`task1_eval_dataset.csv`, 8 MB) datasets are the ground truth for all experiments. DVC lets you:
- Version them like code (`.dvc` pointer files in git).
- Push/pull from a shared remote (S3, SSH, etc.) so collaborators and HPC nodes get the exact same data.
- Reproduce a historical experiment by checking out the right git commit + `dvc checkout`.

**Where to apply**:

| File/Directory | DVC tracked? | Notes |
|---|---|---|
| `task1/data/task1_dataset.csv` | ✅ `dvc add` | Main training set |
| `task1/data/task1_eval_dataset.csv` | ✅ `dvc add` | Eval set — never modify |
| `task1/data/synthetic_traces/` | ✅ `dvc add` | Raw teacher traces (large) |
| `task1/outputs/evaluations/` | ✅ `dvc add` | Evaluation results |
| `task1/outputs/models/` | ✅ `dvc add` | Trained checkpoints |
| `task1/outputs/analysis/` | ❌ git-tracked directly | Small plots, already in repo |

**Remote setup** (pick one):
```bash
# LUMI object storage (recommended for HPC)
dvc remote add lumi s3://your-bucket/agent-distillation
dvc remote modify lumi endpointurl https://your-lumi-s3-endpoint

# SSH fallback
dvc remote add origin ssh://user@hpc-host/path/to/dvc-store
```

**Typical workflow**:
```bash
dvc pull          # get data on a new machine / CI node
# ... run experiment ...
dvc push          # upload new model checkpoints / eval results
git add *.dvc && git commit -m "experiment: ..."
```

### 3.2 MLflow — Experiment Tracking

**Why**: Every training run and evaluation produces hyperparameters, metrics, and artifacts. MLflow gives:
- A searchable UI at `localhost:5000`.
- Automatic parameter + metric logging.
- Artifact storage (model checkpoints, plots, CSVs).
- Run comparison across model sizes, training methods, and TTA configurations.

**Decision**: MLflow integration will be added **before the first LUMI training run** (see Section 5 Phase 3). Since there will be significant code changes anyway to support the Singularity container environment, it makes sense to bundle MLflow in rather than do two rounds of script changes. This is a deliberate deviation from the original "baseline first, then tooling" approach — justified because we have no existing LUMI baseline to protect.

**Full implementation plan**: see [`plans/PHASE3_MLFLOW_INTEGRATION.md`](plans/PHASE3_MLFLOW_INTEGRATION.md)

**TODO summary** (Phase 3 checklist):

- [ ] Create `task1/scripts/mlflow_utils.py` — shared setup + dataset hashing helper
- [ ] Update `02_train_model_lora.py` — remove nvidia-smi GPU selection, add argparse, add MLflow logging
- [ ] Update `03_train_model_full_finetune.py` — same + ROCm-safe flash-attn fallback
- [ ] Update `05_model_evaluation.py` — add MLflow per-model run with all eval metrics
- [ ] Update `task1/scripts/slurm/train_lora_lumi.sh` — add MLflow env vars, venv activation
- [ ] **Task: Add MLflow to TTA experiment** (new script, see Section 4.2)
  - Log: `n_samples` (how many TTA passes), `temperature`, `aggregation_method`
  - Compare single-pass vs TTA metrics

### 3.3 pyproject.toml

The project root has a `pyproject.toml` (see repo root). It defines:
- Package metadata
- All dependencies (replacing the per-task `requirements.txt`)
- Optional dependency groups: `[dev]`, `[train]`, `[eval]`, `[lumi]`

### 3.4 Click CLI Interface

**Why**: The current scripts use `argparse`. Click provides a cleaner, more composable interface with better help text, type validation, and testability.

**TODO — Migrate scripts to Click** (do not implement yet, after LUMI bring-up is stable):

- [ ] `01_create_train_dataset.py` → `agent-distill create-dataset`
- [ ] `02_train_model_lora.py` → `agent-distill train --method lora`
- [ ] `03_train_model_full_finetune.py` → `agent-distill train --method full`
- [ ] `04_create_eval_dataset.py` → `agent-distill create-eval-dataset`
- [ ] `05_model_evaluation.py` → `agent-distill evaluate`
- [ ] `06_analyse_visualize_results.py` → `agent-distill analyze`

All commands will be registered under a single entry point `agent-distill` defined in `pyproject.toml`.

---

## 4. Experiment Roadmap


### 4.1 Reproduce & Verify Existing Results

**Goal**: Confirm the numbers in `RESULTS.md` match what the scripts produce today.  
**Success criterion**: Abstain F1 for Qwen 2.5 7B LoRA ≥ 0.900.

- [ ] Run `05_model_evaluation.py` on the existing checkpoints
- [ ] Compare output CSV to `outputs/evaluations/eval_run_20260124_052629/`
- [ ] Log baseline run to MLflow (experiment name: `baseline`)

### 4.2 Test-Time Augmentation / Self-Consistency

**Goal**: Does running the same query N times (with temperature > 0) and aggregating answers improve performance?

This is directly analogous to **Self-Consistency** (Wang et al. 2022). The hypothesis is that for the smallest models (Gemma 270M, Qwen 0.5B), TTA might compensate for model capacity.

**Approach**:
1. For each evaluation sample, generate N responses (e.g. N ∈ {3, 5, 10}) at `temperature=0.7`.
2. Aggregate:
   - **Abstain decision**: majority vote (> 50% abstain → abstain)
   - **Answer text**: embed all non-abstain answers, pick the one closest to the centroid, OR concatenate and re-summarise (requires a second pass)
3. Compare against single-pass baseline.

**New script**: `task1/scripts/07_tta_experiment.py`

**Parameters to sweep**:

| Parameter | Values |
|-----------|--------|
| Model | Gemma 270M LoRA, Qwen 0.5B LoRA |
| N (passes) | 1 (baseline), 3, 5, 10 |
| Temperature | 0.5, 0.7, 1.0 |
| Aggregation | majority_vote, centroid, oracle (best of N) |

**Expected outcome**: Oracle upper bound is the most interesting — it tells us the theoretical ceiling if we could pick the best answer.

### 4.3 Minimum Viable Browser Model

**Goal**: What is the smallest model that achieves acceptable performance (Abstain F1 ≥ 0.80)?

From current results:
- Gemma 3 270M LoRA: **0.814** ✅ — this is already above threshold
- Qwen 0.5B LoRA: **0.815** ✅

**Next steps**:
- Evaluate both on the full 1 886-sample eval set with the existing checkpoints
- Test inference speed on CPU (relevant for browser)
- Investigate ONNX / GGUF export for WebAssembly/WebGPU deployment

### 4.4 Agentic / Multi-Step Query Loop (Future)

The meeting discussed extending Task 1 to a multi-step agentic loop where the model decides:
- "I have enough evidence → answer"
- "I need more evidence → issue another query"

This is **Task 2** (not yet implemented). Key design questions:
- How to represent the state between steps (context window as state vector)
- When to stop querying (max steps, confidence threshold, Bayesian stopping rule)
- How to evaluate: does the final answer quality improve with more steps?

This is tracked as future work; do not start until Task 1 is stable.

---

## 5. LUMI HPC / AMD GPU Optimisation

### 5.1 Overview — Singularity Container Approach

LUMI requires all AI workloads to run inside **Singularity containers** provided by the LUMI AI Factory. There is no bare-metal Python environment. The canonical container for PyTorch workloads is:

```
/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/
    lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
```

The `lumi-multitorch-full-*` image already includes: ROCm, MPICH, PyTorch, Bitsandbytes, DeepSpeed, Flash Attention, Megatron-LM, vLLM. It is very likely all our training dependencies are already covered.

**Key module for network/filesystem access**:
```bash
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
```
This module injects the Slingshot network bindings and working-directory mounts that RCCL and MPI need for multi-GPU/multi-node performance.

**Adding extra pip packages** (when the container is missing something):
```bash
export SIF=<path-to-.sif>
singularity shell $SIF
# Inside the container:
python -m venv ~/my-env --system-site-packages
source ~/my-env/bin/activate
pip install <package>
# Then run jobs with:
singularity run $SIF bash -c 'source ~/my-env/bin/activate && python my-script.py'
```
> ⚠️ Installing into a venv on Lustre creates thousands of small files. If the package list grows, build a custom container instead (see LUMI AI Factory GitHub).

### 5.2 Phased Bring-up Plan

Work through these phases **in order**. Do not skip ahead.

---

#### Phase 1 — Hello World (verify Singularity + GPU access)

**Goal**: Confirm we can submit a job, the container runs, and GPUs are visible.

```bash
# On LUMI login node:
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif

srun -A <your-project-id> -p small-g -n 1 --gpus-per-task=1 \
    singularity run $SIF \
    python -c "import torch; print('GPUs:', torch.cuda.device_count()); print('ROCm:', torch.version.hip)"
```

**Success criterion**: prints `GPUs: 1` and a ROCm version string.

- [ ] Submit hello-world job and confirm GPU is visible
- [ ] Check `torch.cuda.is_available()` returns `True`
- [ ] Optionally: run a tiny matrix multiply to confirm compute works

---

#### Phase 2 — Environment Compatibility Check

**Goal**: Verify that all packages our codebase needs are present in the container, and identify any gaps.

**Step 1 — List packages in container**:
```bash
singularity run $SIF pip list | grep -iE "transformers|peft|trl|accelerate|bitsandbytes|datasets|pandas|numpy|mlflow|sentencepiece|tokenizers"
```

**Step 2 — Dry-run import check**: create `task1/scripts/lumi_env_check.py`:
```python
# lumi_env_check.py — run inside the container to check imports
import sys
packages = [
    "torch", "transformers", "peft", "trl", "accelerate",
    "bitsandbytes", "datasets", "pandas", "numpy",
    "sentencepiece", "tokenizers", "mlflow",
]
ok, missing = [], []
for pkg in packages:
    try:
        __import__(pkg)
        ok.append(pkg)
    except ImportError as e:
        missing.append((pkg, str(e)))

print(f"OK ({len(ok)}): {ok}")
if missing:
    print(f"MISSING ({len(missing)}): {missing}")
    sys.exit(1)
else:
    print("All imports OK.")
```

```bash
srun -A <project> -p small-g -n 1 --gpus-per-task=1 \
    singularity run $SIF python task1/scripts/lumi_env_check.py
```

**Step 3 — If packages are missing**: install them into a venv (see Section 5.1) or build a custom container.

- [ ] Run `pip list` in container and record versions
- [ ] Run `lumi_env_check.py` — all imports must succeed
- [ ] If `mlflow` is missing: install into venv (it's small enough)
- [ ] If `trl` or `peft` versions are incompatible: pin versions or build custom container

---

#### Phase 3 — Add MLflow + Run First Training Job

**Goal**: Run a real (but short) training job — e.g. 1 epoch of LoRA on Qwen 0.5B — with MLflow logging, and confirm metrics are recorded.

**Before writing the job script**, integrate MLflow into the training scripts (see Section 3.2). This is the right moment because:
- We're already making code changes for LUMI compatibility
- We want every LUMI run tracked from day one
- The scripts will need to handle `LOCAL_RANK` / ROCm anyway — one round of changes

**Job script template** (`task1/scripts/slurm/train_lora_lumi.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=agent-distill-lora
#SBATCH --account=<your-project-id>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_lora_%j.out
#SBATCH --error=logs/train_lora_%j.err

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
export MASTER_PORT=29500
export MLFLOW_TRACKING_URI=<your-mlflow-uri>   # or sqlite:///mlflow.db on shared FS

srun singularity run $SIF \
    python -m torch.distributed.run \
        --nproc_per_node=8 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
    task1/scripts/02_train_model_lora.py \
        --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
        --num_epochs 1 \
        --batch_size 4
```

**ROCm-specific code changes** (to be done in Phase 3 before submitting real training):
- Remove the `select_gpus_before_torch()` NVIDIA-smi logic from `03_train_model_full_finetune.py` — not applicable on LUMI (GPU allocation is handled by SLURM)
- Replace any `flash-attn` hard dependency with a fallback:
  ```python
  attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
  model = AutoModelForCausalLM.from_pretrained(..., attn_implementation=attn_impl)
  ```
- Change `optim="adamw_torch"` to `optim="adamw_torch_fused"` for better ROCm performance (if supported)

**Success criteria for Phase 3**:
- [ ] Training job runs to completion (1 epoch) without OOM or ROCm errors
- [ ] MLflow records the run with correct hyperparameters and loss curve
- [ ] Checkpoint is saved to a LUMI-accessible path

---

#### Phase 4 — Full Training Runs

Only after Phase 3 is green. Scale up to full experiments:

- [ ] LoRA on Qwen 2.5 3B — 3 epochs
- [ ] LoRA on Qwen 2.5 7B — 3 epochs (reproducing the best local result)
- [ ] Full fine-tune on Qwen 2.5 3B (multi-node if needed)
- [ ] TTA experiment on smallest models (see Section 4.2)

### 5.3 What Needs to Change for ROCm

| Area | Current | LUMI / ROCm |
|------|---------|-------------|
| CUDA backend | `torch.cuda.*` | ROCm uses same `torch.cuda.*` API — usually transparent |
| GPU selection | `nvidia-smi` in script | Remove entirely — SLURM allocates GPUs |
| Flash Attention | `flash-attn` (NVIDIA only) | Use `flash-attn` ROCm build (included in container) OR `sdpa` fallback |
| BF16 / FP16 | Works on A100+ | MI250X supports BF16 natively — should work |
| Multi-node | `accelerate` DDP | Switch to `torchrun` with RCCL for multi-node on LUMI |
| Batch size | Tuned for A100 80GB | MI250X has 2×64GB HBM (128GB total per GCD pair) — can increase significantly |
| Optimiser | `adamw_torch` | Consider `adamw_torch_fused` for ROCm |

### 5.4 Container Version Management

The container path encodes a timestamp (`20260124_092648`). When LUMI AI Factory releases a new container:
1. Check the GitHub releases for changelog / new package versions
2. Update the `SIF` path in all SLURM scripts
3. Re-run Phase 2 (env compatibility check) before any training

Pin the exact container path in a shared config file (e.g., `task1/scripts/slurm/container.env`) so all job scripts reference one place:
```bash
# container.env — source this in every SLURM script
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
```

---

## 6. Open Questions & Notes from Meeting

| Question | Status | Notes |
|----------|--------|-------|
| Model that will run in a browser | 🟡 In progress | Gemma 270M and Qwen 0.5B are candidates; need ONNX/GGUF export |
| GT = teacher model output? | ✅ Confirmed | No human annotation; student is trained to match GPT-4o behavior (behavioral cloning) |
| How do models agree on an answer? | ✅ Documented | `answer_state` column: `both_answer`, `both_abstain`, etc. See `EVALUATION_METRICS.md` |
| Annotation of agreement? | ✅ Automated | Pattern matching + embedding similarity, no human annotation |
| Run several times → TTA | 🟡 Planned | Section 4.2 — new script `07_tta_experiment.py` |
| How to train/tune for a task | ✅ Done | LoRA and full fine-tune on behavioral cloning traces |
| Prompt + state vector | 🔴 Future (Task 2) | Agentic loop — context window = state |
| Query → decide whether to query again | 🔴 Future (Task 2) | Stopping criterion design needed |
| API adapters | 🔴 Future (Task 2) | Thin adapter layer per retrieval backend |
| Run all traces on HPC | 🟡 In progress | Section 5 — LUMI Singularity bring-up (Phases 1–4) |
| Bayesian abstain model | 🔴 Research | Could replace hard pattern-match with calibrated confidence score |

---

## 7. Backlog / Future Work

### High Priority

- [ ] **LUMI Phase 1** — Hello-world Singularity job: verify GPU visible (Section 5.2)
- [ ] **LUMI Phase 2** — Environment compatibility check: run `lumi_env_check.py` (Section 5.2)
- [ ] **MLflow integration** — Add to training + eval scripts before first LUMI training run (Section 3.2)
- [ ] **LUMI Phase 3** — First real training job with MLflow tracking (Section 5.2)
- [ ] **Reproduce baseline** — Section 2.4 — run locally first to confirm eval script works
- [ ] **DVC remote setup** — pick storage backend, configure, push existing data

### Medium Priority

- [ ] **LUMI Phase 4** — Full training runs on LUMI (3B, 7B models)
- [ ] **TTA experiment** — write `07_tta_experiment.py`, run on Gemma 270M + Qwen 0.5B
- [ ] **Click CLI migration** — migrate scripts one by one, after LUMI bring-up is stable
- [ ] **ONNX/GGUF export** — for browser deployment candidates (Gemma 270M, Qwen 0.5B)
- [ ] **Calibration analysis** — is the model's abstain decision well-calibrated? (Bayesian framing)

### Low Priority / Research

- [ ] **Task 2: Agentic loop** — multi-step retrieval with learned stopping criterion
- [ ] **Posterior confidence** — replace hard abstain threshold with a learnable confidence head
- [ ] **Cross-dataset generalisation** — does the model trained on causalqa+msmarco+quasart generalise to new domains?
- [ ] **Teacher diversity** — what happens if we mix GPT-4o and DeepSeek traces? (traces exist in `agentic_teacher_*_deepseek/`)
- [ ] **Custom container build** — if pip venv approach becomes unwieldy, build a custom `.sif` on top of the LUMI AI Factory base

---

## Appendix: Key File Reference

| File | Purpose |
|------|---------|
| `task1/data/task1_dataset.csv` | Training set (94 MB, DVC-tracked) |
| `task1/data/task1_eval_dataset.csv` | Eval set (8 MB, DVC-tracked, never modify) |
| `task1/scripts/01_create_train_dataset.py` | Processes raw traces → CSV |
| `task1/scripts/02_train_model_lora.py` | LoRA fine-tuning |
| `task1/scripts/03_train_model_full_finetune.py` | Full fine-tuning |
| `task1/scripts/04_create_eval_dataset.py` | Processes eval traces → CSV |
| `task1/scripts/05_model_evaluation.py` | Runs all metrics |
| `task1/scripts/06_analyse_visualize_results.py` | Generates plots |
| `task1/scripts/lumi_env_check.py` | Import smoke-test for LUMI container (Phase 2) |
| `task1/scripts/slurm/container.env` | Shared container path — source in all SLURM scripts |
| `task1/scripts/slurm/train_lora_lumi.sh` | SLURM job script for LoRA training |
| `task1/EVALUATION_METRICS.md` | Full metric definitions |
| `task1/RESULTS.md` | Baseline numbers and visualisations |
| `pyproject.toml` | Project dependencies and CLI entry points |
| `PLAN.md` | This document |


# Running Jobs on LUMI

---

## Training

If you need a newer Hugging Face stack than the container provides, install it
into `~/agent-distillation/py-overrides` and keep `CONTAINER_PYTHON_OVERRIDES`
set in [container.env](/Users/davidciz/Work/agent-distillation/task1/scripts/slurm/container.env).
The train, eval, and TTA wrappers will prepend that directory to `PYTHONPATH`
automatically.
For this setup, leave `CONTAINER_VENV` empty unless you have explicitly verified
that the venv can still import the container's `torch` stack.

### Submit all model training jobs in parallel

```bash
bash task1/scripts/slurm/submit_train_sweep.sh
bash task1/scripts/slurm/submit_train_sweep.sh --dry-run   # preview
```

This submits one `train_lora_lumi.sh` job per model in the configured list.

### Submit the Qwen3.5 training sweep

```bash
bash task1/scripts/slurm/submit_qwen35_train_sweep.sh
bash task1/scripts/slurm/submit_qwen35_train_sweep.sh --dry-run
```

This runs the selected Qwen3.5 checkpoints: `Qwen3.5-0.8B`, `Qwen3.5-2B`, and `Qwen3.5-4B`.

### Qwen3.5 notes

- The Qwen3.5 LoRA runs on LUMI are launched with 8 distributed ranks. Repeated
  `Starting LoRA training script`, dataset load, and model load lines in the
  SLURM log are usually one copy per rank, not eight independent jobs.
- The logged `epoch` value during training is fractional progress through the
  configured total epochs. With `--num-epochs 3`, training ends around
  `epoch: 3.0`.
- The periodic training dictionaries are emitted every `logging_steps=50`, so a
  small number of rows does not mean only a small number of epochs.
- Mid-training Hugging Face `config.json` HEAD requests can appear when
  checkpoint saving triggers `save_pretrained`-related Hub metadata checks. This
  is noisy, but it is not a full model re-download.
- The Qwen3.5-2B run observed on 2026-03-17 was killed by the SLURM time limit,
  not by a Python exception. At roughly `39 s/step`, a 3-epoch run is closer to
  `8+ hours` than `3.5 hours`.
- If the log says `The fast path is not available because one of the required
  library is not installed`, the model is running a slower torch fallback. For
  Qwen3.5 this likely points at the `flash-linear-attention` /
  `causal-conv1d` kernel stack.

### Qwen3.5 overrides

When using `~/agent-distillation/py-overrides` for Qwen3.5:

- It is safe to layer pure Python packages such as `transformers` or `mlflow`
  there.
- Be very careful with compiled packages. Do not allow pip to pull in a second
  `torch`, `triton`, `nvidia_*`, or `cuda_*` stack into `py-overrides`, because
  that can shadow the container's ROCm build and break imports.
- If testing `flash-linear-attention`, prefer `--no-deps` first and inspect the
  target directory before running jobs.

### Single LoRA job

```bash
sbatch task1/scripts/slurm/train_lora_lumi.sh
```

Override model via env var:

```bash
TRAIN_MODEL="Qwen/Qwen2.5-1.5B-Instruct" sbatch task1/scripts/slurm/train_lora_lumi.sh
```

---

## Evaluation

### Submit evaluation sweep (all trained models)

```bash
bash task1/scripts/slurm/submit_eval_sweep.sh
bash task1/scripts/slurm/submit_eval_sweep.sh --dry-run
```

### Submit the Qwen3.5 evaluation sweep

```bash
bash task1/scripts/slurm/submit_qwen35_eval_sweep.sh
bash task1/scripts/slurm/submit_qwen35_eval_sweep.sh --dry-run
```

### Single eval job

```bash
sbatch task1/scripts/slurm/eval_lumi.sh
```

Override models:

```bash
EVAL_MODELS="--models 'path,name,type,batch'" sbatch task1/scripts/slurm/eval_lumi.sh
```

---

## TTA Experiment

See the full guide at [Running the TTA Experiment](../tta/running.md).

```bash
# Phase A (smallest models, N={1,3,5}, T=0.7)
bash task1/scripts/slurm/submit_tta_sweep.sh --phase a

# Sanity check
bash task1/scripts/slurm/submit_tta_sweep.sh --phase sanity
```

---

## Monitoring

```bash
# Watch your jobs
squeue -u $USER

# Live log tail
tail -f ~/agent-distillation/logs/tta_<JOB_ID>.out

# Cancel a job
scancel <JOB_ID>
```

---

## SLURM Script Reference

| Script | Purpose |
|--------|---------|
| `train_lora_lumi.sh` | Single LoRA training job |
| `train_full_finetune_lumi.sh` | Single full fine-tune job |
| `submit_train_sweep.sh` | Submit all training jobs |
| `submit_qwen35_train_sweep.sh` | Submit the Qwen3.5 LoRA training sweep |
| `eval_lumi.sh` | Single evaluation job |
| `submit_eval_sweep.sh` | Submit eval job covering all models |
| `submit_qwen35_eval_sweep.sh` | Submit the Qwen3.5 evaluation sweep |
| `tta_lumi.sh` | Single TTA job |
| `submit_tta_sweep.sh` | Submit one TTA job per model (phases A–D) |
| `sync_results.sh` | Pull results from LUMI to local machine |
| `container.env` | Container image path and venv config |

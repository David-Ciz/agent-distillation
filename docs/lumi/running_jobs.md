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

This runs the Qwen3.5 queue in the standard / slow-path environment in the
requested order: `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.5-2B`.

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
- As of 2026-03-20, the project standard is to run the full Qwen3.5 family via
  the standard / slow-path environment rather than maintaining a separate
  fast-path workflow for only some checkpoints.

### Qwen3.5 fast-path experiment

We attempted a separate fast-path setup for Qwen3.5 on 2026-03-18 to
2026-03-20.

What was tried:

- Installed a newer `transformers` plus `mlflow` into `py-overrides` because
  the stock container stack was too old for Qwen3.5.
- Added `flash-linear-attention` from source in a scratch override directory so
  the model would stop falling back to the slow torch implementation.
- Verified in an interactive GPU session that:
  - the Qwen3.5 model imported cleanly,
  - the earlier `fla.modules` import error was gone,
  - forward pass and generation worked on GPU with the scratch override dir.

What failed:

- A plain pip install of `flash-linear-attention` tried to pull in a separate
  CUDA / NVIDIA `torch` stack, which is unsafe on the ROCm container. This had
  to be avoided with a more careful install process.
- The fast-path setup introduced a second environment to reason about
  (`~/agent-distillation/py-overrides` vs a tested scratch override dir),
  increasing operational overhead for both training and evaluation.
- Most importantly, the larger Qwen3.5 checkpoint (`Qwen3.5-4B`) crashed during
  training with ROCm GPU memory access faults at step 0 even though interactive
  inference succeeded. This strongly suggested a training-kernel instability in
  the fast-path stack rather than a simple launcher problem.

Why we kept the slower route:

- The slow path is operationally simpler: one environment, one launch path, one
  evaluation path.
- The fast-path setup only looked promising for the smaller checkpoints and
  still required custom handling and extra verification.
- The project priority is comparability and repeatability across the whole
  Qwen3.5 family, not maximizing speed for only a subset of models.

Decision:

- Keep the entire Qwen3.5 family on the standard / slow-path environment unless
  a future container or officially supported ROCm kernel stack makes the
  fast-path route straightforward and stable.
- Do not queue `Qwen3.5-35B-A3B` with the current DDP-based training/eval
  scripts. Models in this range need sharded training / inference
  (FSDP/ZeRO/tensor parallel or equivalent). This is a low-priority future
  experiment rather than part of the current sweep.

### Qwen3.5 overrides

When using `~/agent-distillation/py-overrides` for Qwen3.5:

- It is safe to layer pure Python packages such as `transformers` or `mlflow`
  there.
- Be very careful with compiled packages. Do not allow pip to pull in a second
  `torch`, `triton`, `nvidia_*`, or `cuda_*` stack into `py-overrides`, because
  that can shadow the container's ROCm build and break imports.
- If testing `flash-linear-attention`, prefer `--no-deps` first and inspect the
  target directory before running jobs.

The Qwen3.5 sweep script uses `~/agent-distillation/py-overrides` by default so
the whole family stays on the same environment. If you need to override that
for a one-off run, you can still do so explicitly:

```bash
QWEN35_OVERRIDES=/path/to/overrides bash task1/scripts/slurm/submit_qwen35_train_sweep.sh
```

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

### Submit collapse benchmark sweeps (final checkpoints only)

Qwen2.5 family:

```bash
bash task1/scripts/slurm/submit_collapse_eval_sweep.sh
bash task1/scripts/slurm/submit_collapse_eval_sweep.sh --dry-run
```

Qwen3.5 family:

```bash
bash task1/scripts/slurm/submit_qwen35_collapse_sweep.sh
bash task1/scripts/slurm/submit_qwen35_collapse_sweep.sh --dry-run
```

These launch the Golden Five `lm-eval` suite (`MMLU`, `GSM8K`,
`ARC-Challenge`, `HumanEval`, `TruthfulQA`) against the final LoRA adapters and
refresh the aggregate CSVs afterward.

### Single eval job

```bash
sbatch task1/scripts/slurm/eval_lumi.sh
```

Override models:

```bash
EVAL_MODELS="--models 'path,name,type,batch'" sbatch task1/scripts/slurm/eval_lumi.sh
```

### Single collapse benchmark job

```bash
sbatch task1/scripts/slurm/collapse_eval_lumi.sh
```

Override models or pass extra script args:

```bash
COLLAPSE_EVAL_MODELS="--models 'path,name,type,batch'" \
COLLAPSE_EVAL_EXTRA_ARGS="--limit 20 --mlflow-experiment collapse-benchmarks" \
sbatch task1/scripts/slurm/collapse_eval_lumi.sh
```

Local aggregation after syncing:

```bash
python task1/scripts/14_aggregate_collapse_benchmarks.py
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
| `collapse_eval_lumi.sh` | Single Golden Five collapse benchmark job |
| `submit_collapse_eval_sweep.sh` | Submit the Qwen2.5 collapse benchmark sweep |
| `submit_qwen35_collapse_sweep.sh` | Submit the Qwen3.5 collapse benchmark sweep |
| `tta_lumi.sh` | Single TTA job |
| `submit_tta_sweep.sh` | Submit one TTA job per model (phases A–D) |
| `sync_results.sh` | Pull results from LUMI to local machine |
| `container.env` | Container image path and venv config |

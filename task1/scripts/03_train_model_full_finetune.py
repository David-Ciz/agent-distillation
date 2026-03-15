import os
import sys
import logging
import time

import click
import numpy as np
import pandas as pd
import torch
import mlflow
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTTrainer

from mlflow_utils import hash_file, setup_mlflow
from training_args_compat import make_training_arguments

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
OUTPUT_DIR = os.path.join(TASK1_DIR, "outputs")

# On LUMI, model checkpoints and final weights go to scratch (large, not home-quota-safe).
# Set SCRATCH_OUTPUT_DIR in the SLURM job to override, e.g.:
#   /scratch/project_465002758/$USER/agent-distillation/task1/outputs
# Falls back to the repo outputs/ dir for local runs.
SCRATCH_OUTPUT_DIR = os.environ.get("SCRATCH_OUTPUT_DIR", OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def get_rocm_safe_attn_impl() -> str:
    """Return flash_attention_2 if available, else fall back to sdpa (works on ROCm)."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------
class MLflowMetricsCallback(TrainerCallback):
    def __init__(self):
        self._epoch_start: float = 0.0

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._epoch_start = time.time()

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if mlflow.active_run():
            mlflow.log_metric(
                "epoch_duration_sec",
                time.time() - self._epoch_start,
                step=state.global_step,
            )

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and mlflow.active_run():
            scalars = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if scalars:
                mlflow.log_metrics(scalars, step=state.global_step)


# ---------------------------------------------------------------------------
# Parameter stats (kept from original — useful for full fine-tune bookkeeping)
# ---------------------------------------------------------------------------
def print_model_parameter_stats(model, model_name: str) -> dict:
    total = trainable = frozen = 0
    for _, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        else:
            frozen += n

    report = (
        f"\n{'='*70}\n"
        f"  MODEL PARAMETER STATISTICS — {model_name}\n"
        f"{'='*70}\n"
        f"  Total:     {total:>20,}  ({total * 2 / 1024**3:.2f} GB bf16)\n"
        f"  Trainable: {trainable:>20,}  ({100 * trainable / total:.2f}%)\n"
        f"  Frozen:    {frozen:>20,}  ({100 * frozen / total:.2f}%)\n"
        f"  Training memory estimate (model+grads+Adam): "
        f"~{total * 2 * 4 / 1024**3:.2f} GB\n"
        f"{'='*70}"
    )
    logging.info(report)

    stats_path = os.path.join(OUTPUT_DIR, f"model_stats_{model_name.replace('/', '_')}.txt")
    with open(stats_path, "w") as f:
        f.write(report)

    return {"total_params": total, "trainable_params": trainable, "frozen_params": frozen}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--model-name",
    default="Qwen/Qwen2.5-3B-Instruct",
    show_default=True,
    help="HuggingFace model ID or local path.",
)
@click.option(
    "--dataset-path",
    default=os.path.join(DATA_DIR, "task1_dataset.csv"),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the training CSV file (must have llm_input / llm_output columns).",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory for checkpoints. Defaults to outputs/<model>-full-finetune-checkpoints.",
)
@click.option(
    "--final-model-dir",
    default=None,
    help="Directory for the saved final model. Defaults to outputs/<model>-full-finetune-final.",
)
@click.option("--num-epochs", default=3, show_default=True, type=int, help="Training epochs.")
@click.option("--batch-size", default=2, show_default=True, type=int, help="Per-device batch size.")
@click.option(
    "--learning-rate", default=2e-5, show_default=True, type=float,
    help="Learning rate (lower than LoRA for full fine-tuning).",
)
@click.option(
    "--gradient-accumulation-steps", default=4, show_default=True, type=int,
    help="Gradient accumulation steps.",
)
@click.option("--val-split", default=0.2, show_default=True, type=float, help="Validation split fraction.")
@click.option(
    "--stats-only",
    is_flag=True,
    default=False,
    help="Print model parameter statistics and exit without training.",
)
@click.option(
    "--mlflow-experiment",
    default="full-finetune-training",
    show_default=True,
    help="MLflow experiment name.",
)
def main(
    model_name,
    dataset_path,
    output_dir,
    final_model_dir,
    num_epochs,
    batch_size,
    learning_rate,
    gradient_accumulation_steps,
    val_split,
    stats_only,
    mlflow_experiment,
):
    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    handlers = [logging.StreamHandler(sys.stdout)]
    if is_main_process():
        handlers.append(
            logging.FileHandler(os.path.join(OUTPUT_DIR, "training_full_finetune.log"))
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    model_name_safe = model_name.replace("/", "_")
    checkpoint_dir = output_dir or os.path.join(SCRATCH_OUTPUT_DIR, f"{model_name_safe}-full-finetune-checkpoints")
    final_dir = final_model_dir or os.path.join(SCRATCH_OUTPUT_DIR, f"{model_name_safe}-full-finetune-final")

    if is_main_process():
        logging.info("=" * 70)
        logging.info("FULL FINE-TUNING SCRIPT")
        logging.info(f"  Model:   {model_name}")
        logging.info(f"  Dataset: {dataset_path}")
        logging.info("=" * 70)

    # ------------------------------------------------------------------
    # MLflow setup (main process only)
    # ------------------------------------------------------------------
    if is_main_process():
        setup_mlflow(mlflow_experiment)
        mlflow.start_run(run_name=f"{model_name_safe}-full-finetune")

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    logging.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    attn_impl = get_rocm_safe_attn_impl()
    logging.info(f"Using attention implementation: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Full fine-tuning — all params trainable
    for param in model.parameters():
        param.requires_grad = True

    if is_main_process():
        stats = print_model_parameter_stats(model, model_name)

    if stats_only:
        if is_main_process():
            logging.info("--stats-only: exiting without training.")
            if mlflow.active_run():
                mlflow.end_run(status="KILLED")
        return

    # ------------------------------------------------------------------
    # Load & filter data
    # ------------------------------------------------------------------
    csv_path = dataset_path
    df = pd.read_csv(csv_path, usecols=["llm_input", "llm_output"])
    if is_main_process():
        logging.info(f"Loaded {len(df)} rows from {csv_path}.")

    df = df.dropna(subset=["llm_input", "llm_output"])
    df = df[df["llm_input"].apply(lambda x: isinstance(x, str) and len(x) > 5)]
    df = df[df["llm_output"].apply(lambda x: isinstance(x, str) and len(x) > 0)]

    if is_main_process():
        logging.info(f"Filtered to {len(df)} rows.")

    if len(df) == 0:
        logging.error("No valid training data found after filtering.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * (1 - val_split))
    train_df = df.iloc[indices[:split_idx]]
    val_df = df.iloc[indices[split_idx:]]

    if is_main_process():
        logging.info(f"Train: {len(train_df)} rows  |  Val: {len(val_df)} rows.")
        val_df.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)

    dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(val_df, preserve_index=False)

    # ------------------------------------------------------------------
    # Formatting function
    # ------------------------------------------------------------------
    def formatting_prompts_func(example):
        if isinstance(example["llm_input"], list):
            return [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": inp},
                        {"role": "assistant", "content": out},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for inp, out in zip(example["llm_input"], example["llm_output"])
            ]
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["llm_input"]},
                {"role": "assistant", "content": example["llm_output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    # ------------------------------------------------------------------
    # Log params to MLflow
    # ------------------------------------------------------------------
    if is_main_process():
        mlflow.log_params({
            "model_name": model_name,
            "method": "full_finetune",
            "attn_implementation": attn_impl,
            "num_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "bf16": True,
            "dataset_path": csv_path,
            "dataset_sha256": hash_file(csv_path),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "val_split": val_split,
            "total_params": stats["total_params"],
            "num_gpus": torch.cuda.device_count(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "container_sif": os.environ.get("SIF", "local"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        })

    # ------------------------------------------------------------------
    # TrainingArguments
    # ------------------------------------------------------------------
    training_args = make_training_arguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=50,
        num_train_epochs=num_epochs,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        eval_strategy="epoch",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[MLflowMetricsCallback()] if is_main_process() else [],
    )

    if is_main_process():
        logging.info("Starting full fine-tuning...")

    t0 = time.time()
    trainer.train()
    total_duration = time.time() - t0

    # ------------------------------------------------------------------
    # Save model & log artifacts
    # ------------------------------------------------------------------
    if is_main_process():
        logging.info("Saving model...")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        mlflow.log_metric("total_train_duration_sec", total_duration)
        mlflow.log_dict(training_args.to_dict(), "training_args.json")
        mlflow.log_artifacts(final_dir, artifact_path="final_model")
        mlflow.end_run()

        logging.info(f"Training complete. Total time: {total_duration:.1f}s")
        logging.info(f"Model saved to: {final_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

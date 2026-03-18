import os
import re
import sys
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
import mlflow
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTTrainer

from mlflow_utils import hash_file, setup_mlflow

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

MAX_SEQ_LENGTH = 2048
_TOKENIZER = None


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


def format_chat_example(llm_input: str, llm_output: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": llm_input},
            {"role": "assistant", "content": llm_output},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_batch(examples):
    encoded = _TOKENIZER(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
    return encoded


def make_tokenized_cache_path(csv_path: str, model_name: str, split_name: str) -> str:
    dataset_hash = hash_file(csv_path)[:12]
    model_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    cache_dir = os.path.join(SCRATCH_OUTPUT_DIR, "tokenized_datasets")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"{model_safe}-{dataset_hash}-{split_name}-msl{MAX_SEQ_LENGTH}.arrow",
    )


_TRAINING_ARGUMENT_FIELDS = getattr(TrainingArguments, "__dataclass_fields__", {})


if "push_to_hub_token" in _TRAINING_ARGUMENT_FIELDS:
    CompatTrainingArguments = TrainingArguments
else:
    @dataclass
    class CompatTrainingArguments(TrainingArguments):
        # TRL releases used on LUMI still expect this legacy field to exist when
        # they convert TrainingArguments into an SFTConfig.
        push_to_hub_token: Optional[str] = field(default=None, repr=False)


def make_training_arguments(**kwargs) -> TrainingArguments:
    if "push_to_hub_token" in getattr(CompatTrainingArguments, "__dataclass_fields__", {}):
        kwargs.setdefault("push_to_hub_token", None)
    return CompatTrainingArguments(**kwargs)


# ---------------------------------------------------------------------------
# MLflow callback — streams per-step metrics live during training
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

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if mlflow.active_run() and state.log_history:
            # Derive total training time from first/last log entry if available
            pass  # total duration is logged in main() after trainer.train() returns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--model-name",
    default="google/gemma-3-270m-it",
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
    help="Directory for checkpoints. Defaults to outputs/<model>-lora-checkpoints.",
)
@click.option("--num-epochs", default=3, show_default=True, type=int, help="Training epochs.")
@click.option("--batch-size", default=4, show_default=True, type=int, help="Per-device batch size.")
@click.option("--learning-rate", default=2e-4, show_default=True, type=float, help="Learning rate.")
@click.option(
    "--gradient-accumulation-steps", default=2, show_default=True, type=int,
    help="Gradient accumulation steps.",
)
@click.option("--lora-rank", default=16, show_default=True, type=int, help="LoRA rank (r).")
@click.option("--lora-alpha", default=32, show_default=True, type=int, help="LoRA alpha.")
@click.option("--lora-dropout", default=0.05, show_default=True, type=float, help="LoRA dropout.")
@click.option(
    "--lora-target-modules",
    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    show_default=True,
    help="Comma-separated list of modules to apply LoRA to.",
)
@click.option("--val-split", default=0.2, show_default=True, type=float, help="Validation split fraction.")
@click.option(
    "--mlflow-experiment",
    default="lora-training",
    show_default=True,
    help="MLflow experiment name.",
)
def main(
    model_name,
    dataset_path,
    output_dir,
    num_epochs,
    batch_size,
    learning_rate,
    gradient_accumulation_steps,
    lora_rank,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    val_split,
    mlflow_experiment,
):
    # ------------------------------------------------------------------
    # Logging — configure once, only stream to stdout on worker ranks
    # ------------------------------------------------------------------
    handlers = [logging.StreamHandler(sys.stdout)]
    if is_main_process():
        handlers.append(logging.FileHandler(os.path.join(OUTPUT_DIR, "training_lora.log")))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    if is_main_process():
        logging.info("Starting LoRA training script...")


    # ------------------------------------------------------------------
    # MLflow setup (main process only)
    # ------------------------------------------------------------------
    if is_main_process():
        setup_mlflow(mlflow_experiment)
        run = mlflow.start_run(run_name=f"{model_name.replace('/', '_')}-lora")

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
    if is_main_process():
        val_df.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    if is_main_process():
        logging.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    attn_impl = get_rocm_safe_attn_impl()
    if is_main_process():
        logging.info(f"Using attention implementation: {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    target_modules = [m.strip() for m in lora_target_modules.split(",")]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    # Note: do NOT call get_peft_model() here — SFTTrainer applies peft_config internally (TRL 0.15+).
    # Passing an already-wrapped PeftModel together with peft_config raises a ValueError.

    # ------------------------------------------------------------------
    # TrainingArguments
    # ------------------------------------------------------------------
    model_name_safe = model_name.replace("/", "_")
    checkpoint_dir = output_dir or os.path.join(SCRATCH_OUTPUT_DIR, f"{model_name_safe}-lora-checkpoints")
    final_model_dir = os.path.join(SCRATCH_OUTPUT_DIR, f"{model_name_safe}-lora-final")

    training_args = make_training_arguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=50,
        num_train_epochs=num_epochs,
        save_steps=200,
        bf16=True,
        optim="adamw_torch",
        report_to="none",  # MLflow logging done via our callback
        ddp_find_unused_parameters=False,
        dataloader_num_workers=1,   # container recommends max=1; higher values caused slowdowns
        dataloader_pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Pre-tokenize once and reuse cached Arrow shards across ranks/reruns
    # ------------------------------------------------------------------
    global _TOKENIZER
    _TOKENIZER = tokenizer

    formatted_dataset = Dataset.from_dict(
        {
            "text": [
                format_chat_example(llm_input, llm_output, tokenizer)
                for llm_input, llm_output in zip(train_df["llm_input"], train_df["llm_output"])
            ]
        }
    )
    train_cache_file = make_tokenized_cache_path(csv_path, model_name, "train")

    with training_args.main_process_first(desc="tokenize train dataset"):
        if is_main_process():
            logging.info(f"Tokenizing train dataset (cache: {train_cache_file})")
        dataset = formatted_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=True,
            cache_file_name=train_cache_file,
            desc="Tokenizing train dataset",
        )

    # ------------------------------------------------------------------
    # Log params to MLflow (before training starts)
    # ------------------------------------------------------------------
    if is_main_process():
        mlflow.log_params({
            "model_name": model_name,
            "method": "lora",
            "attn_implementation": attn_impl,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "num_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "bf16": True,
            "optim": training_args.optim,
            "dataset_path": csv_path,
            "dataset_sha256": hash_file(csv_path),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "val_split": val_split,
            "max_seq_length": MAX_SEQ_LENGTH,
            "num_gpus": torch.cuda.device_count(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "container_sif": os.environ.get("SIF", "local"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        })

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[MLflowMetricsCallback()] if is_main_process() else [],
    )

    if is_main_process():
        logging.info("Starting training...")
    t0 = time.time()
    trainer.train()
    total_duration = time.time() - t0

    # ------------------------------------------------------------------
    # Save model & log artifacts
    # ------------------------------------------------------------------
    if is_main_process():
        logging.info("Saving model...")
        trainer.save_model(final_model_dir)

        mlflow.log_metric("total_train_duration_sec", total_duration)
        mlflow.log_dict(training_args.to_dict(), "training_args.json")
        mlflow.log_dict(
            {
                "r": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": target_modules,
            },
            "lora_config.json",
        )
        mlflow.log_artifacts(final_model_dir, artifact_path="final_model")
        mlflow.end_run()

        logging.info(f"Training complete. Total time: {total_duration:.1f}s")
        logging.info(f"Model saved to: {final_model_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

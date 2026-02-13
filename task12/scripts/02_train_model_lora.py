#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Task12: Multi-task (QA + Next-Action Prediction)

Features:
- LoRA fine-tuning on combined Task1+Task2 dataset
- GPU selection before torch import (when not using accelerate)
- Timing and memory metrics tracking
- Support for multi-GPU training with Accelerate
"""

# ============================================================================
# GPU SELECTION - MUST HAPPEN BEFORE IMPORTING TORCH
# ============================================================================
import os
import sys
import subprocess


def select_gpus_before_torch(min_free_gb=40, max_gpus=4):
    """
    Select GPUs with at least min_free_gb of free memory BEFORE torch is imported.
    Uses nvidia-smi to query GPU memory without initializing CUDA.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )

        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 3:
                gpu_id = int(parts[0].strip())
                free_gb = float(parts[1].strip()) / 1024
                print(f"GPU {gpu_id}: {free_gb:.2f} GB free")

                if free_gb >= min_free_gb:
                    gpu_info.append((gpu_id, free_gb))

        gpu_info.sort(key=lambda x: x[1], reverse=True)
        selected = [g[0] for g in gpu_info[:max_gpus]]

        if selected:
            print(f"Selected GPUs: {selected}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))
        else:
            print(f"WARNING: No GPUs found with >= {min_free_gb} GB free memory.")

        return selected

    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return []
    except FileNotFoundError:
        print("nvidia-smi not found.")
        return []


def is_launched_by_accelerate():
    """Check if script is launched via accelerate launch."""
    return any(var in os.environ for var in ['ACCELERATE_LAUNCHED', 'LOCAL_RANK', 'WORLD_SIZE'])


# Select GPUs BEFORE importing torch (only if NOT using accelerate)
if __name__ == "__main__" and not is_launched_by_accelerate():
    select_gpus_before_torch(min_free_gb=40, max_gpus=4)

# ============================================================================
# NOW IMPORT TORCH AND OTHER LIBRARIES
# ============================================================================
import time
import argparse
import pandas as pd
import numpy as np
import torch
import logging
import json
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK12_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK12_DIR / "data"
OUTPUT_DIR = TASK12_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"

# Create output directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "training_lora.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3)
        }
    return {"allocated_gb": 0, "reserved_gb": 0}


def get_model_size(model_path: Path) -> float:
    """Get total size of model files in GB."""
    total = 0
    if model_path.exists():
        for f in model_path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024**3)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Task12 (multi-task)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to training CSV (default: data/combined_train_dataset.csv)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--output_model_dir", type=str, default=None,
                        help="Custom output dir for the final model")
    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = str(DATA_DIR / "combined_train_dataset.csv")

    model_name_safe = args.model_name.replace("/", "-")

    if args.output_model_dir:
        final_model_dir = Path(args.output_model_dir)
    else:
        final_model_dir = MODELS_DIR / f"{model_name_safe}-task12-lora-final"

    checkpoint_dir = MODELS_DIR / f"{model_name_safe}-task12-lora-checkpoints"

    # Metrics tracking
    metrics = {
        "model_name": args.model_name,
        "training_type": "lora",
        "task": "task12",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {k: str(v) for k, v in vars(args).items()},
        "timing": {},
        "memory": {},
    }

    logger.info("=" * 60)
    logger.info("Task12: LoRA Fine-tuning (Multi-task)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {final_model_dir}")
    logger.info(f"Epochs: {args.num_epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
    logger.info(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")

    training_start = time.time()

    # Load data
    logger.info("Loading dataset...")
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path, usecols=['llm_input', 'llm_output'])
    logger.info(f"Loaded {len(df)} rows")

    # Filter valid rows
    df = df.dropna(subset=['llm_input', 'llm_output'])
    df = df[df['llm_input'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 5)]
    df = df[df['llm_output'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    logger.info(f"After filtering: {len(df)} valid rows")

    if len(df) == 0:
        logger.error("No valid training data found.")
        sys.exit(1)

    metrics["dataset_size"] = len(df)

    # Convert to HF Dataset
    dataset = Dataset.from_pandas(df[['llm_input', 'llm_output']], preserve_index=False)

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    mem_after_load = get_gpu_memory_usage()
    metrics["memory"]["after_model_load"] = mem_after_load
    logger.info(f"GPU memory after model load: {mem_after_load['allocated_gb']:.2f} GB")

    # LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    mem_after_lora = get_gpu_memory_usage()
    metrics["memory"]["after_lora_setup"] = mem_after_lora

    # Formatting function - uses chat template
    def formatting_prompts_func(example):
        if isinstance(example['llm_input'], list):
            output_texts = []
            for i in range(len(example['llm_input'])):
                messages = [
                    {"role": "user", "content": example['llm_input'][i]},
                    {"role": "assistant", "content": example['llm_output'][i]}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                output_texts.append(text)
            return output_texts
        else:
            messages = [
                {"role": "user", "content": example['llm_input']},
                {"role": "assistant", "content": example['llm_output']}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        num_train_epochs=args.num_epochs,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=training_args,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    training_end = time.time()
    total_time = training_end - training_start

    metrics["timing"]["total_training_time_s"] = total_time
    metrics["timing"]["total_training_time_min"] = total_time / 60
    metrics["timing"]["avg_epoch_time_s"] = total_time / args.num_epochs

    logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} min)")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        metrics["memory"]["peak_training_memory_gb"] = peak
        logger.info(f"Peak GPU memory: {peak:.2f} GB")

    # Save model
    logger.info(f"Saving model to: {final_model_dir}")
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    model_size = get_model_size(final_model_dir)
    metrics["memory"]["model_file_size_gb"] = model_size
    logger.info(f"Model file size: {model_size:.4f} GB ({model_size * 1024:.2f} MB)")

    # Save metrics
    metrics_path = final_model_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Training metrics saved to: {metrics_path}")

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {final_model_dir}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    logger.info(f"Peak memory: {metrics['memory'].get('peak_training_memory_gb', 'N/A')} GB")
    logger.info(f"Model size: {model_size:.4f} GB")


if __name__ == "__main__":
    main()

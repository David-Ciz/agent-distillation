#!/usr/bin/env python3
"""
Full Fine-tuning Script for Task 2: Next-Action Prediction

Features:
- Full parameter fine-tuning (all weights trainable)
- Timing metrics: per-epoch time, full training time
- Memory metrics: GPU memory usage, model file size
- Gradient checkpointing for memory efficiency
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
                free_mb = float(parts[1].strip())
                free_gb = free_mb / 1024
                print(f"GPU {gpu_id}: {free_gb:.2f} GB free")
                
                if free_gb >= min_free_gb:
                    gpu_info.append((gpu_id, free_gb))
        
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        selected_gpus = [gpu_id for gpu_id, _ in gpu_info[:max_gpus]]
        
        if selected_gpus:
            print(f"Selected GPUs: {selected_gpus}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        else:
            print(f"WARNING: No GPUs found with >= {min_free_gb} GB free memory.")
        
        return selected_gpus
        
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
from trl import SFTTrainer

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK2_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK2_DIR / "data"
OUTPUT_DIR = TASK2_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"

# Create output directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "training_full_finetune.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Instruction suffix for prompts
INSTRUCTION_SUFFIX = "\nWhat is the best next action? Choose from the available_actions above."


def get_local_rank():
    """Get local rank for distributed training."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_local_rank() == 0


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return {"allocated_gb": allocated, "reserved_gb": reserved}
    return {"allocated_gb": 0, "reserved_gb": 0}


def get_model_size(model_path: Path) -> float:
    """Get total size of model files in GB."""
    total_size = 0
    if model_path.exists():
        for f in model_path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
    return total_size / (1024**3)


def add_instruction_suffix(text: str) -> str:
    """Add instruction suffix only if not already present."""
    if INSTRUCTION_SUFFIX.strip() not in text:
        return text + INSTRUCTION_SUFFIX
    return text


def print_model_stats(model, model_name: str):
    """Print model parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"Model size (bf16): {total_params * 2 / 1024**3:.2f} GB")
    logger.info("=" * 60)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_bf16_gb": total_params * 2 / 1024**3
    }


def main():
    parser = argparse.ArgumentParser(description="Full fine-tuning for Task 2")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to training CSV (default: data/task2_dataset.csv)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only print model stats without training")
    
    args = parser.parse_args()
    
    if args.dataset_path is None:
        args.dataset_path = DATA_DIR / "task2_dataset.csv"
    else:
        args.dataset_path = Path(args.dataset_path)
    
    model_name_safe = args.model_name.replace("/", "-")
    checkpoint_dir = MODELS_DIR / f"{model_name_safe}-full-finetune-checkpoints"
    final_model_dir = MODELS_DIR / f"{model_name_safe}-full-finetune-final"
    
    # Metrics tracking
    metrics = {
        "model_name": args.model_name,
        "training_type": "full_finetune",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": vars(args),
        "timing": {},
        "memory": {},
    }
    
    if is_main_process():
        logger.info("=" * 60)
        logger.info("Task 2: Full Fine-tuning")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Dataset: {args.dataset_path}")
    
    # Record start time
    training_start = time.time()
    
    # Load model and tokenizer
    if is_main_process():
        logger.info(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Ensure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # Print model stats
    if is_main_process():
        model_stats = print_model_stats(model, args.model_name)
        metrics["model_stats"] = model_stats
    
    if args.stats_only:
        if is_main_process():
            logger.info("Stats-only mode. Exiting without training.")
        return
    
    # Record memory after model load
    mem_after_load = get_gpu_memory_usage()
    metrics["memory"]["after_model_load"] = mem_after_load
    if is_main_process():
        logger.info(f"GPU memory after model load: {mem_after_load['allocated_gb']:.2f} GB")
    
    # Load data
    if is_main_process():
        logger.info("Loading dataset...")
    
    if not args.dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset_path}")
        sys.exit(1)
    
    df = pd.read_csv(args.dataset_path, usecols=['llm_input', 'llm_output'])
    if is_main_process():
        logger.info(f"Loaded {len(df)} rows")
    
    # Filter valid rows
    df = df.dropna(subset=['llm_input', 'llm_output'])
    df = df[df['llm_input'].apply(lambda x: isinstance(x, str) and len(x) > 5)]
    df = df[df['llm_output'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    
    if is_main_process():
        logger.info(f"Filtered to {len(df)} valid rows")
    
    if len(df) == 0:
        logger.error("No valid training data found.")
        sys.exit(1)
    
    # Split data (80/20)
    np.random.seed(42)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[indices[:split_idx]]
    test_df = df.iloc[indices[split_idx:]]
    
    if is_main_process():
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # Save test split
        test_split_path = DATA_DIR / "test_split.csv"
        test_df.to_csv(test_split_path, index=False)
        logger.info(f"Test split saved to: {test_split_path}")
    
    # Convert to HF Dataset
    dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    
    # Formatting function
    def formatting_prompts_func(example):
        if isinstance(example['llm_input'], list):
            output_texts = []
            for i in range(len(example['llm_input'])):
                user_content = add_instruction_suffix(example['llm_input'][i])
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": example['llm_output'][i]}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                output_texts.append(text)
            return output_texts
        else:
            user_content = add_instruction_suffix(example['llm_input'])
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": example['llm_output']}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=50,
        num_train_epochs=args.num_epochs,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        eval_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
        processing_class=tokenizer,
    )
    
    # Training
    if is_main_process():
        logger.info("Starting training...")
        logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps} per GPU")
    
    trainer.train()
    
    training_end = time.time()
    total_training_time = training_end - training_start
    
    # Record timing metrics
    metrics["timing"]["total_training_time"] = total_training_time
    metrics["timing"]["avg_epoch_time"] = total_training_time / args.num_epochs
    
    if is_main_process():
        logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    
    # Record peak memory
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        metrics["memory"]["peak_training_memory_gb"] = peak_memory
        if is_main_process():
            logger.info(f"Peak GPU memory during training: {peak_memory:.2f} GB")
    
    # Save model
    if is_main_process():
        logger.info(f"Saving model to: {final_model_dir}")
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        
        # Record model size
        model_size = get_model_size(final_model_dir)
        metrics["memory"]["model_file_size_gb"] = model_size
        logger.info(f"Model file size: {model_size:.2f} GB")
        
        # Save metrics
        metrics_path = final_model_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Training metrics saved to: {metrics_path}")
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {final_model_dir}")
        logger.info(f"Total training time: {total_training_time:.2f}s")
        logger.info(f"Peak memory: {peak_memory:.2f} GB")
        logger.info(f"Model size: {model_size:.2f} GB")
    
    # Clean up distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

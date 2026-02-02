#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Task 2: Next-Action Prediction

Features:
- LoRA (Low-Rank Adaptation) fine-tuning for memory efficiency
- Timing metrics: per-epoch time, full training time
- Memory metrics: GPU memory usage, model file size
- Automatic GPU selection based on available memory
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
from peft import LoraConfig, get_peft_model, TaskType
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
        logging.FileHandler(OUTPUT_DIR / "training_lora.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Instruction suffix for prompts
INSTRUCTION_SUFFIX = "\nWhat is the best next action? Choose from the available_actions above."


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


class TimingCallback:
    """Callback to track per-epoch timing."""
    def __init__(self):
        self.epoch_times = []
        self.epoch_start = None
    
    def on_epoch_begin(self):
        self.epoch_start = time.time()
    
    def on_epoch_end(self):
        if self.epoch_start:
            elapsed = time.time() - self.epoch_start
            self.epoch_times.append(elapsed)
            logger.info(f"Epoch completed in {elapsed:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Task 2")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to training CSV (default: data/task2_dataset.csv)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    
    if args.dataset_path is None:
        args.dataset_path = DATA_DIR / "task2_dataset.csv"
    else:
        args.dataset_path = Path(args.dataset_path)
    
    model_name_safe = args.model_name.replace("/", "-")
    checkpoint_dir = MODELS_DIR / f"{model_name_safe}-lora-checkpoints"
    final_model_dir = MODELS_DIR / f"{model_name_safe}-lora-final"
    
    # Metrics tracking
    metrics = {
        "model_name": args.model_name,
        "training_type": "lora",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": vars(args),
        "timing": {},
        "memory": {},
    }
    
    logger.info("=" * 60)
    logger.info("Task 2: LoRA Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset_path}")
    
    # Record start time
    training_start = time.time()
    
    # Load data
    logger.info("Loading dataset...")
    if not args.dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset_path}")
        sys.exit(1)
    
    df = pd.read_csv(args.dataset_path, usecols=['llm_input', 'llm_output'])
    logger.info(f"Loaded {len(df)} rows")
    
    # Filter valid rows
    df = df.dropna(subset=['llm_input', 'llm_output'])
    df = df[df['llm_input'].apply(lambda x: isinstance(x, str) and len(x) > 5)]
    df = df[df['llm_output'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
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
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Save test split
    test_split_path = DATA_DIR / "test_split.csv"
    test_df.to_csv(test_split_path, index=False)
    logger.info(f"Test split saved to: {test_split_path}")
    
    # Convert to HF Dataset
    dataset = Dataset.from_pandas(train_df, preserve_index=False)
    
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
    
    # Record memory after model load
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
    
    # Record memory after LoRA
    mem_after_lora = get_gpu_memory_usage()
    metrics["memory"]["after_lora_setup"] = mem_after_lora
    
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
    
    # Training with timing
    logger.info("Starting training...")
    epoch_times = []
    
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        if epoch == 0:
            trainer.train()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} completed in {epoch_time:.2f}s")
    
    training_end = time.time()
    total_training_time = training_end - training_start
    
    # Record timing metrics
    metrics["timing"]["epoch_times"] = epoch_times
    metrics["timing"]["total_training_time"] = total_training_time
    metrics["timing"]["avg_epoch_time"] = np.mean(epoch_times) if epoch_times else total_training_time / args.num_epochs
    
    logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    
    # Record peak memory
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        metrics["memory"]["peak_training_memory_gb"] = peak_memory
        logger.info(f"Peak GPU memory during training: {peak_memory:.2f} GB")
    
    # Save model
    logger.info(f"Saving model to: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Record model size
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
    logger.info(f"Total training time: {total_training_time:.2f}s")
    logger.info(f"Peak memory: {metrics['memory'].get('peak_training_memory_gb', 'N/A')} GB")
    logger.info(f"Model size: {model_size:.4f} GB")


if __name__ == "__main__":
    main()

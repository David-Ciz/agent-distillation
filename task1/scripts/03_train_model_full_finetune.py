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
    Sets CUDA_VISIBLE_DEVICES environment variable.
    """
    try:
        # Use nvidia-smi to get GPU memory info without importing torch
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
                total_mb = float(parts[2].strip())
                free_gb = free_mb / 1024
                total_gb = total_mb / 1024
                print(f"GPU {gpu_id}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
                
                if free_gb >= min_free_gb:
                    gpu_info.append((gpu_id, free_gb))
        
        # Sort by free memory (descending) and take up to max_gpus
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        selected_gpus = [gpu_id for gpu_id, _ in gpu_info[:max_gpus]]
        
        if selected_gpus:
            print(f"Selected GPUs: {selected_gpus} (each with >= {min_free_gb} GB free)")
            # Set CUDA_VISIBLE_DEVICES BEFORE torch is imported
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
            print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            print(f"ERROR: No GPUs found with >= {min_free_gb} GB free memory.")
            sys.exit(1)
        
        return selected_gpus
        
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        sys.exit(1)

def is_launched_by_accelerate():
    """Check if script is launched via accelerate launch."""
    return any(var in os.environ for var in ['ACCELERATE_LAUNCHED', 'LOCAL_RANK', 'WORLD_SIZE'])

# Select GPUs BEFORE importing torch (only if NOT using accelerate)
if __name__ == "__main__" and not is_launched_by_accelerate():
    selected_gpus = select_gpus_before_torch(min_free_gb=40, max_gpus=4)
    print(f"Will use {len(selected_gpus)} GPUs: {selected_gpus}")
elif is_launched_by_accelerate():
    print("Launched via accelerate - GPU selection handled by accelerate")

# ============================================================================
# NOW IMPORT TORCH AND OTHER LIBRARIES
# ============================================================================
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer
import logging
import numpy as np
import argparse

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
OUTPUT_DIR = os.path.join(TASK1_DIR, "outputs")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "training_full_finetune.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_local_rank():
    """Get local rank for distributed training."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_local_rank() == 0


def print_model_parameter_stats(model, model_name):
    """
    Print detailed parameter statistics for the model.
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    # Parameter breakdown by module type
    embedding_params = 0
    attention_params = 0
    mlp_params = 0
    norm_params = 0
    lm_head_params = 0
    other_params = 0
    
    # Trainable breakdown
    trainable_embedding = 0
    trainable_attention = 0
    trainable_mlp = 0
    trainable_norm = 0
    trainable_lm_head = 0
    trainable_other = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params
        
        # Categorize by module type
        name_lower = name.lower()
        if 'embed' in name_lower:
            embedding_params += num_params
            if param.requires_grad:
                trainable_embedding += num_params
        elif any(x in name_lower for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention']):
            attention_params += num_params
            if param.requires_grad:
                trainable_attention += num_params
        elif any(x in name_lower for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
            mlp_params += num_params
            if param.requires_grad:
                trainable_mlp += num_params
        elif any(x in name_lower for x in ['norm', 'layernorm', 'rmsnorm']):
            norm_params += num_params
            if param.requires_grad:
                trainable_norm += num_params
        elif 'lm_head' in name_lower:
            lm_head_params += num_params
            if param.requires_grad:
                trainable_lm_head += num_params
        else:
            other_params += num_params
            if param.requires_grad:
                trainable_other += num_params
    
    # Format numbers with commas
    def fmt(n):
        return f"{n:,}"
    
    def fmt_pct(n, total):
        if total == 0:
            return "0.00%"
        return f"{100 * n / total:.2f}%"
    
    def fmt_size(n):
        """Format parameter count as memory size (assuming float32 = 4 bytes, bfloat16 = 2 bytes)"""
        bytes_fp32 = n * 4
        bytes_bf16 = n * 2
        if bytes_bf16 >= 1024**3:
            return f"{bytes_bf16 / 1024**3:.2f} GB (bf16)"
        elif bytes_bf16 >= 1024**2:
            return f"{bytes_bf16 / 1024**2:.2f} MB (bf16)"
        else:
            return f"{bytes_bf16 / 1024:.2f} KB (bf16)"
    
    stats_report = f"""
================================================================================
                    MODEL PARAMETER STATISTICS
================================================================================
Model: {model_name}
--------------------------------------------------------------------------------

OVERALL SUMMARY
---------------
Total Parameters:      {fmt(total_params):>20}  ({fmt_size(total_params)})
Trainable Parameters:  {fmt(trainable_params):>20}  ({fmt_pct(trainable_params, total_params)})
Frozen Parameters:     {fmt(frozen_params):>20}  ({fmt_pct(frozen_params, total_params)})

PARAMETER BREAKDOWN BY MODULE TYPE
----------------------------------
{"Module Type":<20} {"Total Params":>18} {"Trainable":>18} {"% of Total":>12}
{"-"*70}
{"Embeddings":<20} {fmt(embedding_params):>18} {fmt(trainable_embedding):>18} {fmt_pct(embedding_params, total_params):>12}
{"Attention":<20} {fmt(attention_params):>18} {fmt(trainable_attention):>18} {fmt_pct(attention_params, total_params):>12}
{"MLP/FFN":<20} {fmt(mlp_params):>18} {fmt(trainable_mlp):>18} {fmt_pct(mlp_params, total_params):>12}
{"Normalization":<20} {fmt(norm_params):>18} {fmt(trainable_norm):>18} {fmt_pct(norm_params, total_params):>12}
{"LM Head":<20} {fmt(lm_head_params):>18} {fmt(trainable_lm_head):>18} {fmt_pct(lm_head_params, total_params):>12}
{"Other":<20} {fmt(other_params):>18} {fmt(trainable_other):>18} {fmt_pct(other_params, total_params):>12}
{"-"*70}

MEMORY ESTIMATES (approximate)
------------------------------
Full Precision (FP32):  {total_params * 4 / 1024**3:.2f} GB
Half Precision (BF16):  {total_params * 2 / 1024**3:.2f} GB
Training Memory (BF16 + Adam states): ~{total_params * 2 * 4 / 1024**3:.2f} GB (model + gradients + optimizer)

================================================================================
"""
    
    logging.info(stats_report)
    
    # Also save to file
    stats_filename = os.path.join(OUTPUT_DIR, f"model_stats_{model_name.replace('/', '_')}.txt")
    with open(stats_filename, 'w') as f:
        f.write(stats_report)
    logging.info(f"Parameter statistics saved to: {stats_filename}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'embedding_params': embedding_params,
        'attention_params': attention_params,
        'mlp_params': mlp_params,
        'norm_params': norm_params,
        'lm_head_params': lm_head_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Full fine-tuning script for causal language models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name or path (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(DATA_DIR, "task1_dataset.csv"),
        help="Path to the training CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: ./<model_name>-full-finetune-checkpoints)"
    )
    parser.add_argument(
        "--final_model_dir",
        type=str,
        default=None,
        help="Directory to save final model (default: ./<model_name>-full-finetune-final)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2, smaller than LoRA due to more memory usage)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5, lower than LoRA for full fine-tuning)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only print model statistics without training"
    )
    
    args = parser.parse_args()
    
    model_name = args.model_name
    model_name_safe = model_name.replace("/", "-")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(OUTPUT_DIR, f"{model_name_safe}-full-finetune-checkpoints")
    if args.final_model_dir is None:
        args.final_model_dir = os.path.join(OUTPUT_DIR, f"{model_name_safe}-full-finetune-final")
    
    # Only log from main process to avoid duplicate logs
    if is_main_process():
        logging.info("=" * 80)
        logging.info("FULL FINE-TUNING SCRIPT")
        logging.info("=" * 80)
        logging.info(f"Model: {model_name}")
        logging.info(f"Dataset: {args.dataset_path}")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"Final model directory: {args.final_model_dir}")
        logging.info("=" * 80)

    # 1. Load Model and Tokenizer
    if is_main_process():
        logging.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model for FULL fine-tuning (no PEFT/LoRA)
    # Use low_cpu_mem_usage to avoid loading full model on each process
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bf16 for A100s
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Important for multi-GPU to reduce CPU memory
    )
    
    # Ensure all parameters are trainable (full fine-tuning)
    for param in model.parameters():
        param.requires_grad = True
    
    # 2. Print detailed parameter statistics (only on main process)
    if is_main_process():
        logging.info("\n" + "=" * 80)
        logging.info("COMPUTING MODEL PARAMETER STATISTICS...")
        logging.info("=" * 80)
        stats = print_model_parameter_stats(model, model_name)
    
    if args.stats_only:
        if is_main_process():
            logging.info("Stats-only mode. Exiting without training.")
        return
    
    # 3. Load Data
    csv_path = args.dataset_path
    if not os.path.exists(csv_path):
        logging.error(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path, usecols=['llm_input', 'llm_output'])
    if is_main_process():
        logging.info(f"Loaded {len(df)} rows from CSV.")

    # Filter for rows with valid llm_input and llm_output
    df = df.dropna(subset=['llm_input', 'llm_output'])
    # Ensure they are strings and meet length requirements
    df = df[df['llm_input'].apply(lambda x: isinstance(x, str) and len(x) > 5)]
    df = df[df['llm_output'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    
    if is_main_process():
        logging.info(f"Filtered to {len(df)} rows.")
    
    if len(df) == 0:
        logging.error("No valid training data found.")
        sys.exit(1)

    # 4. Split Data (80/20 train/test)
    np.random.seed(42)  # Ensure reproducibility
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    split_idx = int(len(df) * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    
    if is_main_process():
        logging.info(f"Training on {len(train_df)} rows.")
        logging.info(f"Saving {len(test_df)} rows to test_split.csv for evaluation.")
        # Save test split for evaluation script (only on main process)
        test_df.to_csv(os.path.join(DATA_DIR, 'test_split.csv'), index=False)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    # 5. Formatting Function for SFTTrainer
    def formatting_prompts_func(example):
        output_texts = []
        # Check if we have a batch (list) or single example (string)
        if isinstance(example['llm_input'], list):
            for i in range(len(example['llm_input'])):
                # Construct chat format
                messages = [
                    {"role": "user", "content": example['llm_input'][i]},
                    {"role": "assistant", "content": example['llm_output'][i]}
                ]
                # Apply template
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                output_texts.append(text)
            return output_texts
        else:
            # Single example
            messages = [
                {"role": "user", "content": example['llm_input']},
                {"role": "assistant", "content": example['llm_output']}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            # Return the string directly, not a list, because SFTTrainer calls this with batched=False
            return text

    # 6. Training Arguments (optimized for full fine-tuning)
    # Note: Full fine-tuning requires more memory, so we use smaller batch sizes
    # and potentially gradient checkpointing
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,  
        gradient_accumulation_steps=args.gradient_accumulation_steps,  
        learning_rate=args.learning_rate,  # Lower LR for full fine-tuning
        logging_steps=100,
        num_train_epochs=args.num_epochs,
        bf16=True,  # Use bf16 for A100s (better than fp16)
        optim="adamw_torch",
        report_to="none",
        # Save only best and last checkpoints
        save_strategy="epoch",
        save_total_limit=2,  # Keep only last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        eval_strategy="epoch",  # Required for load_best_model_at_end
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Gradient checkpointing to save memory (important for full fine-tuning)
        gradient_checkpointing=True,
        # Warmup for stability
        warmup_ratio=0.1,
    )

    # 7. Trainer (no peft_config for full fine-tuning)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
        processing_class=tokenizer,
    )

    if is_main_process():
        logging.info("Starting FULL fine-tuning...")
        logging.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps} per GPU")
    
    trainer.train()
    
    if is_main_process():
        logging.info("Saving model...")
        trainer.save_model(args.final_model_dir)
        tokenizer.save_pretrained(args.final_model_dir)
        logging.info(f"Full fine-tuned model saved to: {args.final_model_dir}")
        logging.info("Training complete.")
    
    # Clean up distributed process group to avoid warning
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # GPU selection already happened at module load time (before torch import)
    main()

import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import sys
import logging
import numpy as np

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
        logging.FileHandler(os.path.join(OUTPUT_DIR, "training.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_gpus_with_free_memory(min_free_gb=70, max_gpus=4):
    """
    Returns a list of GPU indices that have at least min_free_gb of free memory.
    Returns up to max_gpus GPUs, sorted by free memory (most free first).
    """
    if not torch.cuda.is_available():
        logging.info("No CUDA devices available.")
        return []
    
    gpu_info = []
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)
            logging.info(f"GPU {i}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
            
            if free_gb >= min_free_gb:
                gpu_info.append((i, free_gb))
        except Exception as e:
            logging.error(f"Error getting info for GPU {i}: {e}")
            continue
    
    # Sort by free memory (descending) and take up to max_gpus
    gpu_info.sort(key=lambda x: x[1], reverse=True)
    selected_gpus = [gpu_id for gpu_id, _ in gpu_info[:max_gpus]]
    
    if selected_gpus:
        logging.info(f"Selected GPUs: {selected_gpus} (each with >= {min_free_gb} GB free)")
    else:
        logging.warning(f"No GPUs found with >= {min_free_gb} GB free memory.")
    
    return selected_gpus

def main():
    logging.info("Starting training script...")
    
    # 1. Select GPUs with at least 40GB free VRAM (up to 4 GPUs)
    selected_gpus = get_gpus_with_free_memory(min_free_gb=40, max_gpus=4)
    
    if not selected_gpus:
        logging.error("No GPUs with sufficient free memory. Exiting.")
        sys.exit(1)
    
    # Set CUDA_VISIBLE_DEVICES to selected GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    num_gpus = len(selected_gpus)
    logging.info(f"Using {num_gpus} GPU(s): {selected_gpus}")

    # 2. Load Data
    csv_path = os.path.join(DATA_DIR, "task1_dataset.csv")
    if not os.path.exists(csv_path):
        logging.error(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path, usecols=['llm_input', 'llm_output'])
    logging.info(f"Loaded {len(df)} rows from CSV.")

    # Filter for rows with valid llm_input and llm_output
    df = df.dropna(subset=['llm_input', 'llm_output'])
    # Ensure they are strings and meet length requirements
    df = df[df['llm_input'].apply(lambda x: isinstance(x, str) and len(x) > 5)]
    df = df[df['llm_output'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    
    logging.info(f"Filtered to {len(df)} rows.")
    
    if len(df) == 0:
        logging.error("No valid training data found.")
        sys.exit(1)

    # 3. Split Data (80/20 train/test)
    np.random.seed(42)  # Ensure reproducibility
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    split_idx = int(len(df) * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    
    logging.info(f"Training on {len(train_df)} rows.")
    logging.info(f"Saving {len(test_df)} rows to test_split.csv for evaluation.")
    
    # Save test split for evaluation script
    test_df.to_csv(os.path.join(DATA_DIR, 'test_split.csv'), index=False)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(train_df, preserve_index=False)

    # 3. Model and Tokenizer
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "google/gemma-3-270m-it"
    model_name = "google/gemma-3-1b-it"
    
    logging.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    # For multi-GPU DDP training, we load without device_map and let Accelerate handle distribution
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bf16 for A100s
        trust_remote_code=True
    )

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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

    # Sanitize model name for directory paths
    model_name_safe = model_name.replace("/", "_")
    
    # 6. Training Arguments (optimized for multi-GPU)
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"{model_name_safe}-lora-checkpoints"),
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=2,  
        learning_rate=2e-4,
        logging_steps=100,
        num_train_epochs=3,
        save_steps=50,
        bf16=True,  # Use bf16 for A100s (better than fp16)
        optim="adamw_torch",
        report_to="none",
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=training_args,
        processing_class=tokenizer,
    )

    logging.info("Starting training...")
    trainer.train()
    
    logging.info("Saving model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, f"{model_name_safe}-lora-final"))
    logging.info("Training complete.")

if __name__ == "__main__":
    main()

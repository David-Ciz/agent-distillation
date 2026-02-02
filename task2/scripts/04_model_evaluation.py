#!/usr/bin/env python3
"""
Model Evaluation Script for Task 2: Next-Action Prediction

Evaluates trained models with comprehensive classification metrics:
- Overall accuracy
- Macro/Weighted F1, Precision, Recall
- Per-class F1, Precision, Recall, Support
- Confusion matrix
- Timing: evaluation time, inference time per sample
- Memory: inference memory usage

Supports:
- Multiple model evaluation (base, LoRA, full fine-tuned)
- Accelerate for multi-GPU inference
- Batch processing for efficiency
- Summary comparison across models

Usage:
    # With accelerate (recommended for speed):
    accelerate launch --multi_gpu --num_processes=4 scripts/04_model_evaluation.py --models "path,name,type,batch_size" ...
    
    # Without accelerate:
    python3 04_model_evaluation.py --models "path,name,type,batch_size" ...
"""

import os
import sys
import argparse
import json
import logging
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def is_launched_by_accelerate():
    """Check if script is launched via accelerate launch."""
    return any(var in os.environ for var in ['ACCELERATE_LAUNCHED', 'LOCAL_RANK', 'WORLD_SIZE'])

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK2_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK2_DIR / "data"
OUTPUT_DIR = TASK2_DIR / "outputs"
EVAL_DIR = OUTPUT_DIR / "evaluations"

# Create output directories
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Action vocabulary
ACTION_VOCAB = ['query_formulator', 'chatnoir_retriever', 'opensearch_retriever',
                'reranker', 'deduplicator', 'answer_drafter', 'finish']
ACTION_TO_ID = {action: i for i, action in enumerate(ACTION_VOCAB)}
ID_TO_ACTION = {i: action for i, action in enumerate(ACTION_VOCAB)}

# Instruction suffix for prompts
INSTRUCTION_SUFFIX = "\nWhat is the best next action? Choose from the available_actions above."


def add_instruction_suffix(text: str) -> str:
    """Add instruction suffix only if not already present."""
    if INSTRUCTION_SUFFIX.strip() not in text:
        return text + INSTRUCTION_SUFFIX
    return text


def extract_action_from_generation(generated_text: str) -> Optional[str]:
    """
    Extract action name from model generation.
    Handles various formats the model might output.
    """
    if not isinstance(generated_text, str):
        return None
    
    generated_text = generated_text.strip().lower()
    
    # Try exact pattern first
    match = re.search(r'next action:\s*(\w+)', generated_text)
    if match:
        action = match.group(1)
        if action in ACTION_VOCAB:
            return action
    
    # Try to find any action name in the text (prioritize earlier matches)
    for action in ACTION_VOCAB:
        if action in generated_text:
            return action
    
    # Check if the entire output is just an action name
    clean_text = generated_text.strip().strip('"\'')
    if clean_text in ACTION_VOCAB:
        return clean_text
    
    return None


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return {"allocated_gb": allocated, "reserved_gb": reserved}
    return {"allocated_gb": 0, "reserved_gb": 0}


class EvaluationDataset(Dataset):
    """Dataset for evaluation."""
    
    def __init__(self, df: pd.DataFrame):
        self.data = []
        
        for _, row in df.iterrows():
            llm_input = row.get('llm_input', row.get('task_input', ''))
            llm_output = row.get('llm_output', row.get('task_output', ''))
            
            if not isinstance(llm_input, str) or not isinstance(llm_output, str):
                continue
            
            # Extract ground truth action
            action = llm_output.strip().lower()
            if action not in ACTION_VOCAB:
                # Try to extract from "Next action: X" format
                match = re.search(r'next action:\s*(\w+)', action)
                if match:
                    action = match.group(1)
            
            if action not in ACTION_VOCAB:
                continue
            
            self.data.append({
                'llm_input': llm_input,
                'ground_truth': action,
                'ground_truth_id': ACTION_TO_ID[action]
            })
        
        logger.info(f"Loaded {len(self.data)} valid samples for evaluation")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_model(model_path: str, model_type: str):
    """
    Load model based on type (base, lora, full_finetune).
    Uses device_map="auto" for multi-GPU support with accelerate.
    
    Args:
        model_path: Path to model or HuggingFace model name
        model_type: One of 'base', 'lora', 'full_finetune'
    
    Returns:
        model, tokenizer
    """
    model_path_obj = Path(model_path) if not model_path.startswith(("Qwen/", "google/", "meta-llama/")) else None
    
    if model_type == "lora":
        # For LoRA, we need to find the base model
        # Check if there's a config file that specifies the base model
        if model_path_obj is not None and model_path_obj.exists():
            adapter_config_path = model_path_obj / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
            else:
                # Try to infer from path name
                path_str = str(model_path)
                if "Qwen2.5-0.5B" in path_str:
                    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                elif "Qwen2.5-1.5B" in path_str:
                    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
                elif "Qwen2.5-3B" in path_str:
                    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
                elif "Qwen2.5-7B" in path_str:
                    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
                elif "Qwen3-4B" in path_str:
                    base_model_name = "Qwen/Qwen3-4B-Instruct-2507"
                elif "gemma-3-1b" in path_str.lower():
                    base_model_name = "google/gemma-3-1b-it"
                elif "gemma" in path_str.lower():
                    base_model_name = "google/gemma-3-270m-it"
                else:
                    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        else:
            base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        logger.info(f"Loading LoRA model with base: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model = model.merge_and_unload()
        
    elif model_type == "full_finetune":
        logger.info(f"Loading full fine-tuned model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
    else:  # base
        logger.info(f"Loading base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    dataset: EvaluationDataset,
    model_name: str,
    batch_size: int = 8,
    max_new_tokens: int = 20
) -> Dict[str, Any]:
    """
    Evaluate a model on the dataset using batched inference.
    
    Returns dict with metrics, predictions, and timing info.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name} (batch_size={batch_size})")
    logger.info(f"{'='*60}")
    
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Ensure tokenizer uses left padding for batched generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    predictions = []
    ground_truths = []
    
    # Track memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    eval_start = time.time()
    total_samples = len(dataset)
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, total_samples, batch_size), desc=f"Evaluating {model_name}"):
            end_idx = min(start_idx + batch_size, total_samples)
            
            # Collect batch data
            batch_prompts = []
            batch_metadata = []
            
            for idx in range(start_idx, end_idx):
                item = dataset[idx]
                user_content = add_instruction_suffix(item['llm_input'])
                messages = [{"role": "user", "content": user_content}]
                prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                batch_prompts.append(prompt)
                batch_metadata.append(item)
            
            # Tokenize batch with padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_lengths = inputs['attention_mask'].sum(dim=1)
            
            # Generate for entire batch
            batch_start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            batch_time = time.time() - batch_start
            per_sample_time = batch_time / len(batch_prompts)
            
            # Decode each output
            for i, (output_ids, input_len, meta) in enumerate(zip(outputs, input_lengths, batch_metadata)):
                # Extract only generated tokens (after input)
                generated_ids = output_ids[input_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract predicted action
                pred_action = extract_action_from_generation(generated_text)
                
                predictions.append({
                    'llm_input': meta['llm_input'],
                    'ground_truth': meta['ground_truth'],
                    'generated_text': generated_text,
                    'predicted_action': pred_action,
                    'inference_time': per_sample_time
                })
                ground_truths.append(meta['ground_truth'])
    
    # Restore padding side
    tokenizer.padding_side = 'right'
    
    eval_time = time.time() - eval_start
    
    # Compute metrics
    valid_preds = []
    valid_gts = []
    invalid_count = 0
    
    for pred, gt in zip(predictions, ground_truths):
        if pred['predicted_action'] is not None:
            valid_preds.append(pred['predicted_action'])
            valid_gts.append(gt)
        else:
            invalid_count += 1
    
    metrics = {}
    
    if len(valid_preds) > 0:
        # Convert to IDs
        pred_ids = [ACTION_TO_ID.get(p, -1) for p in valid_preds]
        gt_ids = [ACTION_TO_ID.get(g, -1) for g in valid_gts]
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(gt_ids, pred_ids)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_ids, pred_ids, labels=list(range(len(ACTION_VOCAB))), zero_division=0
        )
        
        # Macro averages
        metrics['macro_precision'] = float(np.mean(precision))
        metrics['macro_recall'] = float(np.mean(recall))
        metrics['macro_f1'] = float(np.mean(f1))
        
        # Weighted averages
        w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
            gt_ids, pred_ids, average='weighted', zero_division=0
        )
        metrics['weighted_precision'] = float(w_precision)
        metrics['weighted_recall'] = float(w_recall)
        metrics['weighted_f1'] = float(w_f1)
        
        # Per-class metrics
        metrics['per_class'] = {}
        for i, action in enumerate(ACTION_VOCAB):
            metrics['per_class'][action] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Confusion matrix
        cm = confusion_matrix(gt_ids, pred_ids, labels=list(range(len(ACTION_VOCAB))))
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            gt_ids, pred_ids,
            labels=list(range(len(ACTION_VOCAB))),
            target_names=ACTION_VOCAB,
            zero_division=0
        )
    
    # Timing metrics - extract from predictions
    inference_times = [p['inference_time'] for p in predictions]
    metrics['timing'] = {
        'total_eval_time': eval_time,
        'avg_inference_time_per_sample': np.mean(inference_times) if inference_times else 0,
        'std_inference_time': np.std(inference_times) if inference_times else 0,
        'min_inference_time': np.min(inference_times) if inference_times else 0,
        'max_inference_time': np.max(inference_times) if inference_times else 0,
    }
    
    # Memory metrics
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        metrics['memory'] = {
            'peak_inference_memory_gb': peak_memory
        }
    
    # Counts
    metrics['valid_predictions'] = len(valid_preds)
    metrics['invalid_predictions'] = invalid_count
    metrics['total_samples'] = len(predictions)
    
    # Log summary
    logger.info(f"\nResults for {model_name}:")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0)*100:.2f}%")
    logger.info(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
    logger.info(f"  Weighted F1: {metrics.get('weighted_f1', 0):.4f}")
    logger.info(f"  Valid predictions: {metrics['valid_predictions']}/{metrics['total_samples']}")
    logger.info(f"  Eval time: {eval_time:.2f}s")
    logger.info(f"  Avg inference time: {metrics['timing']['avg_inference_time_per_sample']*1000:.2f}ms")
    
    if 'per_class' in metrics:
        logger.info("\n  Per-class F1:")
        for action in ACTION_VOCAB:
            pc = metrics['per_class'][action]
            if pc['support'] > 0:
                logger.info(f"    {action:20s}: F1={pc['f1']:.3f}, P={pc['precision']:.3f}, R={pc['recall']:.3f}, N={pc['support']}")
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 models")
    parser.add_argument("--models", nargs="+", type=str, required=True,
                        help="Model configs: 'path,name,type,batch_size' (type: base/lora/full_finetune)")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test CSV (default: data/test_split.csv)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--gen_batch_size", type=int, default=8,
                        help="Default batch size for generation (can be overridden per model)")
    
    args = parser.parse_args()
    
    if args.test_data is None:
        args.test_data = DATA_DIR / "test_split.csv"
    else:
        args.test_data = Path(args.test_data)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        eval_run_dir = EVAL_DIR / f"eval_run_{timestamp}"
    else:
        eval_run_dir = Path(args.output_dir)
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Task 2: Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output dir: {eval_run_dir}")
    logger.info(f"Accelerate launched: {is_launched_by_accelerate()}")
    
    # Load test data
    if not args.test_data.exists():
        logger.error(f"Test data not found: {args.test_data}")
        sys.exit(1)
    
    df = pd.read_csv(args.test_data)
    logger.info(f"Loaded {len(df)} rows from test data")
    
    dataset = EvaluationDataset(df)
    
    # Action distribution
    action_counts = {}
    for item in dataset.data:
        action = item['ground_truth']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    logger.info("\nTest set action distribution:")
    for action in ACTION_VOCAB:
        count = action_counts.get(action, 0)
        pct = 100 * count / len(dataset) if len(dataset) > 0 else 0
        logger.info(f"  {action}: {count} ({pct:.1f}%)")
    
    # Evaluate each model
    all_results = []
    
    for model_config in args.models:
        parts = model_config.split(",")
        if len(parts) < 3:
            logger.warning(f"Invalid model config: {model_config}")
            continue
        
        model_path = parts[0]
        model_name = parts[1]
        model_type = parts[2]
        batch_size = int(parts[3]) if len(parts) > 3 else args.gen_batch_size
        
        try:
            # Load model (uses device_map="auto")
            model, tokenizer = load_model(model_path, model_type)
            
            # Evaluate with batch processing
            result = evaluate_model(model, tokenizer, dataset, model_name, batch_size=batch_size)
            result['model_path'] = model_path
            result['model_type'] = model_type
            all_results.append(result)
            
            # Save detailed results
            preds_df = pd.DataFrame(result['predictions'])
            preds_path = eval_run_dir / f"{model_name}_detailed_results.csv"
            preds_df.to_csv(preds_path, index=False)
            logger.info(f"Saved detailed results to: {preds_path}")
            
            # Save summary
            summary = {
                'model_name': model_name,
                'model_path': model_path,
                'model_type': model_type,
                **result['metrics']
            }
            summary_path = eval_run_dir / f"{model_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Saved summary to: {summary_path}")
            
            # Free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison summary
    if all_results:
        comparison_data = []
        for result in all_results:
            m = result['metrics']
            comparison_data.append({
                'model_name': result['model_name'],
                'model_type': result['model_type'],
                'accuracy': m.get('accuracy', 0),
                'macro_f1': m.get('macro_f1', 0),
                'macro_precision': m.get('macro_precision', 0),
                'macro_recall': m.get('macro_recall', 0),
                'weighted_f1': m.get('weighted_f1', 0),
                'valid_predictions': m.get('valid_predictions', 0),
                'total_samples': m.get('total_samples', 0),
                'avg_inference_time_ms': m.get('timing', {}).get('avg_inference_time_per_sample', 0) * 1000,
                'total_eval_time_s': m.get('timing', {}).get('total_eval_time', 0),
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = eval_run_dir / "model_comparison_summary.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nSaved comparison summary to: {comparison_path}")
        
        # Print comparison table
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 80)
        logger.info(comparison_df.to_string(index=False))
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

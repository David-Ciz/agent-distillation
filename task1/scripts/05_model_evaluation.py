#!/usr/bin/env python3
"""
Model Evaluation Script for Agent Distillation

Evaluates fine-tuned models on the eval dataset with comprehensive metrics:
- Loss and Perplexity
- Abstain Detection and Agreement Metrics
- Answer Embedding Similarity (semantic similarity between teacher and student answers)
- Exact Match and Token Overlap
- Per-sample detailed results with all metrics

Supports:
- Multiple model evaluation (base, LoRA, full fine-tuned)
- Accelerate for multi-GPU inference
- Batch processing for efficiency
- Summary comparison across models
- Detailed per-sample CSV output
- JSON summary with comprehensive statistics
"""

import os
import sys
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import click
import mlflow

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mlflow_utils import hash_file, setup_mlflow

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
OUTPUT_DIR = os.path.join(TASK1_DIR, "outputs")
EVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluations")

# Create output directories
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# Robust abstain detection patterns
ABSTAIN_PATTERNS = [
    r"\bI\s+cannot\s+answer\b",
    r"\bI\s+can't\s+answer\b",
    r"\bcannot\s+answer\s+based\s+on\s+the\s+provided\s+evidence\b",
    r"\bcannot\s+answer\s+based\s+on\s+the\s+evidence\b",
    r"\bnot\s+enough\s+information\b",
    r"\bnot\s+enough\s+evidence\b",
    r"\binsufficient\s+information\b",
    r"\binsufficient\s+evidence\b",
    r"\bprovided\s+evidence\s+does\s+not\b",
    r"\bevidence\s+does\s+not\s+(contain|provide|support)\b",
    r"\bno\s+relevant\s+information\b",
    r"\bno\s+information\s+(is\s+)?provided\b",
    r"\bunable\s+to\s+(answer|determine|provide)\b",
    r"\bI\s+don'?t\s+have\s+(enough|sufficient)\b",
    r"\bI\s+cannot\s+provide\s+an?\s+answer\b",
    r"\bI\s+cannot\s+determine\b",
]


def is_launched_by_accelerate():
    """Check if script is launched via accelerate launch."""
    return any(var in os.environ for var in ['ACCELERATE_LAUNCHED', 'LOCAL_RANK', 'WORLD_SIZE'])


def setup_logging(log_file: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def utc_timestamp() -> str:
    """Get UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def normalize_text(text: str) -> str:
    """Normalize text by stripping and collapsing whitespace."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def normalize_for_comparison(text: str) -> str:
    """Normalize text for exact match comparison."""
    t = normalize_text(text)
    t = t.lower()
    t = t.strip(" \t\n\r\f\v\"'`""''")
    return t


def extract_answer(text: str) -> str:
    """
    Extract the ANSWER section from model output.
    Uses multiple patterns for robust extraction.
    """
    if text is None:
        return ""
    
    t = text.strip()
    if not t:
        return ""
    
    # Patterns to match ANSWER section (case-insensitive)
    patterns = [
        r"(?is)\bANSWER\b\s*[:\-\]]\s*(.*)$",
        r"(?is)\bAnswer\b\s*[:\-\]]\s*(.*)$",
        r"(?is)\bfinal\s+answer\b\s*[:\-\]]\s*(.*)$",
        r"(?is)\bmy\s+answer\b\s*[:\-\]]\s*(.*)$",
    ]
    
    for p in patterns:
        m = re.search(p, t)
        if m:
            ans = m.group(1).strip()
            # Remove any trailing sections like EVIDENCE, RATIONALE, etc.
            ans = re.split(r"(?is)\n\s*(EVIDENCE|RATIONALE|EXPLANATION|SOURCES|REASONING)\b\s*[:\-\]]", ans)[0]
            return ans.strip()
    
    # If no ANSWER section found, check if the whole text is short enough to be an answer
    # (for cases where model outputs just the answer without formatting)
    if len(t) < 500 and '\n' not in t[:100]:
        return t
    
    # Try to extract from the last paragraph if it looks like an answer
    paragraphs = t.split('\n\n')
    if paragraphs:
        last_para = paragraphs[-1].strip()
        if len(last_para) < 300:
            return last_para
    
    return t


def is_abstain(answer_text: str) -> bool:
    """
    Check if an answer indicates abstention using robust pattern matching.
    """
    if not answer_text or not isinstance(answer_text, str):
        return True
    
    t = normalize_for_comparison(answer_text)
    if not t:
        return True
    
    # Check against abstain patterns
    for p in ABSTAIN_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            return True
    
    # Check for exact matches of common abstain phrases
    if re.fullmatch(r"(n/?a|none|unknown|no\s*answer|abstain)", t):
        return True
    
    # Very short answers that are just punctuation or single words
    if len(t) < 3:
        return True
    
    return False


class EvalDataset(Dataset):
    """Dataset for evaluation with full row data."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 2048):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'llm_input': str(row['llm_input']),
            'llm_output': str(row['llm_output']),
            'llm_answer': str(row.get('llm_answer', '')),
            'decision_label': str(row.get('decision_label', '')),
            'data_source': str(row.get('data_source', '')),
            'teacher_id': str(row.get('teacher_id', '')),
            'query': str(row.get('query', '')),
            'search_index': str(row.get('search_index', '')),
            'idx': idx
        }


def load_model_and_tokenizer(model_path: str, is_lora: bool = False):
    """Load model and tokenizer from path."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    logging.info(f"Loading model from: {model_path}")
    
    if is_lora:
        # For LoRA models, we need to load base model first then apply adapter
        # Try to find adapter_config.json to get base model name
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-3B-Instruct')
        else:
            base_model_name = 'Qwen/Qwen2.5-3B-Instruct'
        
        logging.info(f"Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        logging.info(f"Applying LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def compute_loss_and_perplexity(
    model, 
    tokenizer, 
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = -1
) -> Tuple[float, float]:
    """Compute average loss and perplexity on the dataset."""
    total_loss = 0.0
    total_tokens = 0
    num_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing loss/perplexity"):
            if max_samples > 0 and num_samples >= max_samples:
                break
            
            for i in range(len(batch['llm_input'])):
                prompt = batch['llm_input'][i]
                target = batch['llm_output'][i]
                
                # Format as chat
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                
                total_loss += outputs.loss.item() * inputs['input_ids'].shape[1]
                total_tokens += inputs['input_ids'].shape[1]
                num_samples += 1
                
                if max_samples > 0 and num_samples >= max_samples:
                    break
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return avg_loss, perplexity


def generate_answers_batched(
    model,
    tokenizer,
    dataset: EvalDataset,
    device: torch.device,
    max_new_tokens: int = 256,
    max_samples: int = -1,
    batch_size: int = 8
) -> List[Dict]:
    """
    Generate answers for all samples using batched inference.
    
    Returns list of dicts with all metadata for detailed results.
    """
    results = []
    
    # Determine number of samples
    total_samples = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    
    # Ensure tokenizer uses left padding for batched generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Process in batches
    with torch.no_grad():
        for start_idx in tqdm(range(0, total_samples, batch_size), desc="Generating answers"):
            end_idx = min(start_idx + batch_size, total_samples)
            
            # Collect batch data
            batch_prompts = []
            batch_metadata = []
            
            for idx in range(start_idx, end_idx):
                item = dataset[idx]
                prompt = item['llm_input']
                
                # Format as chat
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(text)
                
                # Store all metadata including original columns
                batch_metadata.append({
                    'idx': item['idx'],
                    'llm_input': item['llm_input'],
                    'teacher_output': item['llm_output'],
                    'teacher_answer_from_dataset': item['llm_answer'],
                    'decision_label': item['decision_label'],
                    'data_source': item['data_source'],
                    'teacher_id': item['teacher_id'],
                    'query': item['query'],
                    'search_index': item['search_index'],
                })
            
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Decode each output
            for i, (output_ids, input_len, meta) in enumerate(zip(outputs, input_lengths, batch_metadata)):
                # Extract only generated tokens (after input)
                generated_ids = output_ids[input_len:]
                student_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract answers using robust extraction
                teacher_answer = extract_answer(meta['teacher_output'])
                student_answer = extract_answer(student_output)
                
                # Detect abstain
                teacher_abstain = is_abstain(teacher_answer)
                student_abstain = is_abstain(student_answer)
                
                # Determine state
                if teacher_abstain and student_abstain:
                    state = "both_abstain"
                elif not teacher_abstain and student_abstain:
                    state = "teacher_answer_student_abstain"
                elif teacher_abstain and not student_abstain:
                    state = "teacher_abstain_student_answer"
                else:
                    state = "both_answer"
                
                # Compute exact match score
                if teacher_abstain and student_abstain:
                    exact_score = 1.0
                else:
                    exact_score = 1.0 if normalize_for_comparison(teacher_answer) == normalize_for_comparison(student_answer) else 0.0
                
                results.append({
                    'idx': meta['idx'],
                    'data_source': meta['data_source'],
                    'teacher_id': meta['teacher_id'],
                    'query': meta['query'],
                    'search_index': meta['search_index'],
                    'llm_input': meta['llm_input'],
                    'teacher_output': meta['teacher_output'],
                    'student_output': student_output,
                    'teacher_answer': teacher_answer,
                    'student_answer': student_answer,
                    'teacher_abstain': teacher_abstain,
                    'student_abstain': student_abstain,
                    'answer_state': state,
                    'exact_match_score': exact_score,
                    'decision_label': meta['decision_label'],
                })
    
    # Restore padding side
    tokenizer.padding_side = 'right'
    
    return results


def summarize_numeric(values: List[float]) -> Dict[str, Any]:
    """Compute summary statistics for numeric values."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "mean": None, "min": None, "median": None, "max": None, "std": None}
    
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def compute_state_counts(results: List[Dict]) -> Dict[str, int]:
    """
    Compute state counts for answer agreement analysis.
    
    States:
        - both_abstain: Both teacher and student abstained
        - teacher_answer_student_abstain: Teacher answered, student abstained
        - teacher_abstain_student_answer: Teacher abstained, student answered
        - both_answer: Both provided answers
    """
    state_counts = {
        "both_abstain": 0,
        "teacher_answer_student_abstain": 0,
        "teacher_abstain_student_answer": 0,
        "both_answer": 0,
    }
    
    for r in results:
        state = r.get('answer_state', '')
        if state in state_counts:
            state_counts[state] += 1
    
    return state_counts


def load_qwen_embedding_model(model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = None):
    """Load the Qwen embedding model and tokenizer."""
    from transformers import AutoModel, AutoTokenizer
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Loading embedding model: {model_name}")
    logging.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


def compute_qwen_embeddings_batch(
    texts: List[str], 
    model, 
    tokenizer, 
    device: str,
    batch_size: int = 32,
    max_length: int = 512
) -> np.ndarray:
    """Compute embeddings for a batch of texts using Qwen embedding model."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Handle empty or None texts
        batch_texts = [t if t and isinstance(t, str) else "" for t in batch_texts]
        
        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            
            # Use last hidden state mean pooling
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.last_hidden_state
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def compute_embedding_similarities(
    results: List[Dict],
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 32,
    embedding_model=None
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Compute cosine similarity between teacher and student answers using embeddings.
    
    Uses Qwen3-Embedding-0.6B for high-quality semantic similarity.
    
    Returns two lists:
        - similarities_all: Embedding similarity for all pairs (None if either is empty)
        - similarities_adjusted: Same but 0 if abstain mismatch, 1 if both abstain
    """
    # Load model if not provided
    if embedding_model is None:
        model, tokenizer, device = load_qwen_embedding_model(embedding_model_name)
    else:
        model, tokenizer, device = embedding_model
    
    similarities_all = []
    similarities_adjusted = []
    
    # Collect valid pairs for batch embedding
    valid_indices = []
    teacher_texts = []
    student_texts = []
    
    for i, r in enumerate(results):
        teacher_ans = r['teacher_answer']
        student_ans = r['student_answer']
        t_abstain = r.get('teacher_abstain', is_abstain(teacher_ans))
        s_abstain = r.get('student_abstain', is_abstain(student_ans))
        
        # For adjusted similarity
        if t_abstain and s_abstain:
            similarities_adjusted.append(1.0)
        elif t_abstain != s_abstain:
            similarities_adjusted.append(0.0)
        else:
            similarities_adjusted.append(None)  # Will compute
        
        # For all similarity - compute if both have non-empty answers
        if teacher_ans and student_ans and len(teacher_ans.strip()) > 0 and len(student_ans.strip()) > 0:
            valid_indices.append(i)
            teacher_texts.append(teacher_ans)
            student_texts.append(student_ans)
            similarities_all.append(None)  # Will compute
        else:
            similarities_all.append(None)
    
    # Batch encode using Qwen model
    if valid_indices:
        logging.info(f"Computing embeddings for {len(valid_indices)} valid answer pairs...")
        teacher_embeddings = compute_qwen_embeddings_batch(
            teacher_texts, model, tokenizer, device, batch_size
        )
        student_embeddings = compute_qwen_embeddings_batch(
            student_texts, model, tokenizer, device, batch_size
        )
        
        # Compute cosine similarities (embeddings are already normalized)
        cosine_sims = np.sum(teacher_embeddings * student_embeddings, axis=1)
        
        for idx, sim in enumerate(cosine_sims):
            orig_idx = valid_indices[idx]
            similarities_all[orig_idx] = float(sim)
            
            # Also update adjusted if it was None (both have answers)
            if similarities_adjusted[orig_idx] is None:
                similarities_adjusted[orig_idx] = float(sim)
    
    return similarities_all, similarities_adjusted


def compute_token_overlap(teacher: str, student: str) -> float:
    """Compute Jaccard token overlap between two strings."""
    teacher_tokens = set(teacher.strip().lower().split())
    student_tokens = set(student.strip().lower().split())
    
    if not teacher_tokens and not student_tokens:
        return 1.0
    if not teacher_tokens or not student_tokens:
        return 0.0
    
    intersection = len(teacher_tokens & student_tokens)
    union = len(teacher_tokens | student_tokens)
    return intersection / union if union > 0 else 0.0


def evaluate_model(
    model_path: str,
    eval_df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    is_lora: bool = False,
    max_samples: int = -1,
    batch_size: int = 4,
    gen_batch_size: int = 8,
    compute_generation: bool = True,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
) -> Dict:
    """
    Evaluate a single model and save comprehensive results.
    
    Outputs:
        - {model_name}_detailed_results.csv: Per-sample results with all metrics
        - {model_name}_summary.json: Comprehensive summary statistics
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Evaluating model: {model_name}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Max samples: {max_samples if max_samples > 0 else 'all'}")
    logging.info(f"Generation batch size: {gen_batch_size}")
    logging.info(f"{'='*80}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path, is_lora=is_lora)
    device = next(model.parameters()).device
    
    # Prepare dataset
    if max_samples > 0:
        eval_df_subset = eval_df.head(max_samples)
    else:
        eval_df_subset = eval_df
    
    dataset = EvalDataset(eval_df_subset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize summary structure
    summary = {
        "timestamp_utc": utc_timestamp(),
        "model_name": model_name,
        "model_path": model_path,
        "model_type": "lora" if is_lora else "full_finetune",
        "num_samples": len(eval_df_subset),
        "config": {
            "batch_size": batch_size,
            "gen_batch_size": gen_batch_size,
            "embedding_model": embedding_model,
        }
    }
    
    # 1. Compute Loss and Perplexity
    logging.info("Computing loss and perplexity...")
    avg_loss, perplexity = compute_loss_and_perplexity(
        model, tokenizer, dataloader, device, max_samples
    )
    summary['loss_perplexity'] = {
        'loss': avg_loss,
        'perplexity': perplexity
    }
    logging.info(f"Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    
    results = None
    
    if compute_generation:
        # 2. Generate answers using batched inference
        logging.info(f"Generating student answers (batch_size={gen_batch_size})...")
        results = generate_answers_batched(
            model, tokenizer, dataset, device,
            max_samples=max_samples,
            batch_size=gen_batch_size
        )
        
        # Free up model memory before embedding computation
        del model
        torch.cuda.empty_cache()
        
        # 3. Compute embedding similarities
        logging.info("Computing embedding similarities...")
        sims_all, sims_adjusted = compute_embedding_similarities(
            results, embedding_model_name=embedding_model
        )
        
        # Add similarities to results
        for i, r in enumerate(results):
            r['embedding_similarity'] = sims_all[i]
            r['embedding_similarity_adjusted'] = sims_adjusted[i]
            # Also compute token overlap per sample
            r['token_overlap'] = compute_token_overlap(r['teacher_answer'], r['student_answer'])
        
        # 4. Compute aggregate statistics
        state_counts = compute_state_counts(results)
        
        # Exact match scores
        exact_scores = [r['exact_match_score'] for r in results]
        
        # Teacher statistics
        teacher_abstain_count = sum(1 for r in results if r['teacher_abstain'])
        teacher_answer_count = len(results) - teacher_abstain_count
        
        # Student statistics
        student_abstain_count = sum(1 for r in results if r['student_abstain'])
        student_answer_count = len(results) - student_abstain_count
        
        # Agreement rate
        agreement_count = sum(1 for r in results if r['teacher_abstain'] == r['student_abstain'])
        
        # Build summary
        summary['answer_states'] = {
            'state_counts': state_counts,
            'state_percentages': {k: v / len(results) * 100 for k, v in state_counts.items()}
        }
        
        summary['teacher_stats'] = {
            'abstain_count': teacher_abstain_count,
            'answer_count': teacher_answer_count,
            'abstain_rate': teacher_abstain_count / len(results)
        }
        
        summary['student_stats'] = {
            'abstain_count': student_abstain_count,
            'answer_count': student_answer_count,
            'abstain_rate': student_abstain_count / len(results)
        }
        
        summary['agreement'] = {
            'abstain_agreement_count': agreement_count,
            'abstain_agreement_rate': agreement_count / len(results),
            'both_abstain_rate': state_counts['both_abstain'] / len(results),
            'both_answer_rate': state_counts['both_answer'] / len(results)
        }
        
        summary['exact_match'] = {
            'avg_score': float(np.mean(exact_scores)),
            'stats': summarize_numeric(exact_scores)
        }
        
        # Embedding similarity stats
        valid_sims_all = [s for s in sims_all if s is not None]
        valid_sims_adj = [s for s in sims_adjusted if s is not None]
        
        summary['embedding_similarity'] = {
            'all_pairs': summarize_numeric(valid_sims_all),
            'adjusted_for_abstain': summarize_numeric(valid_sims_adj)
        }
        
        # Token overlap stats
        token_overlaps = [r['token_overlap'] for r in results]
        summary['token_overlap'] = summarize_numeric(token_overlaps)
        
        # Save detailed results CSV
        results_df = pd.DataFrame(results)
        
        # Reorder columns for clarity
        column_order = [
            'idx', 'data_source', 'teacher_id', 'query', 'search_index',
            'llm_input', 'teacher_output', 'student_output',
            'teacher_answer', 'student_answer',
            'teacher_abstain', 'student_abstain', 'answer_state',
            'exact_match_score', 'embedding_similarity', 'embedding_similarity_adjusted',
            'token_overlap', 'decision_label'
        ]
        columns = [c for c in column_order if c in results_df.columns]
        results_df = results_df[columns]
        
        results_path = os.path.join(output_dir, f"{model_name}_detailed_results.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"Detailed results saved to: {results_path}")
    else:
        del model
        torch.cuda.empty_cache()
    
    # Save summary JSON
    summary_path = os.path.join(output_dir, f"{model_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved to: {summary_path}")
    
    # Return flat metrics for comparison CSV
    flat_metrics = {
        'model_name': model_name,
        'model_path': model_path,
        'num_samples': len(eval_df_subset),
        'loss': avg_loss,
        'perplexity': perplexity,
        'timestamp': summary['timestamp_utc']
    }
    
    if compute_generation and results:
        flat_metrics.update({
            'teacher_abstain_rate': summary['teacher_stats']['abstain_rate'],
            'student_abstain_rate': summary['student_stats']['abstain_rate'],
            'abstain_agreement_rate': summary['agreement']['abstain_agreement_rate'],
            'both_abstain_rate': summary['agreement']['both_abstain_rate'],
            'both_answer_rate': summary['agreement']['both_answer_rate'],
            'exact_match_avg': summary['exact_match']['avg_score'],
            'embedding_similarity_avg': summary['embedding_similarity']['all_pairs'].get('mean', 0),
            'embedding_similarity_adjusted_avg': summary['embedding_similarity']['adjusted_for_abstain'].get('mean', 0),
            'token_overlap_avg': summary['token_overlap'].get('mean', 0),
        })
    
    return flat_metrics


def create_comparison_summary(all_metrics: List[Dict], output_path: str):
    """Create a summary CSV comparing all evaluated models."""
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns for readability
    column_order = [
        'model_name', 'num_samples', 'loss', 'perplexity',
        'teacher_abstain_rate', 'student_abstain_rate',
        'abstain_agreement_rate', 'both_abstain_rate', 'both_answer_rate',
        'exact_match_avg', 'embedding_similarity_avg', 'embedding_similarity_adjusted_avg',
        'token_overlap_avg', 'model_path', 'timestamp'
    ]
    
    # Only include columns that exist
    columns = [c for c in column_order if c in df.columns]
    df = df[columns]
    
    df.to_csv(output_path, index=False)
    logging.info(f"Comparison summary saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*120)
    print("MODEL COMPARISON SUMMARY")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)


def parse_model_config(config_str: str) -> Dict:
    """
    Parse a model config string in format: path,name,type[,gen_batch_size]
    
    Types: lora, full_finetune, base
    gen_batch_size is optional (defaults to global setting)
    
    Examples:
        "outputs/model-lora,my-model,lora,16"
        "Qwen/Qwen2.5-0.5B,qwen-base,base"
    """
    parts = [p.strip() for p in config_str.split(',')]
    if len(parts) < 3:
        raise ValueError(f"Invalid model config: {config_str}. Expected: path,name,type[,gen_batch_size]")
    
    model_type = parts[2].lower()
    if model_type not in ('lora', 'full_finetune', 'base'):
        raise ValueError(f"Invalid model type: {model_type}. Must be: lora, full_finetune, or base")
    
    config = {
        'path': parts[0],
        'name': parts[1],
        'type': model_type,
        'is_lora': model_type == 'lora',
        'gen_batch_size': int(parts[3]) if len(parts) > 3 else None
    }
    return config


@click.command()
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Model config: 'path,name,type,gen_batch_size'. Repeat for multiple models. "
         "Types: lora, full_finetune, base.",
)
@click.option(
    "--eval-dataset",
    default=os.path.join(DATA_DIR, "task1_eval_dataset.csv"),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to evaluation dataset CSV.",
)
@click.option(
    "--output-dir",
    default=EVAL_OUTPUT_DIR,
    show_default=True,
    help="Base directory for evaluation results.",
)
@click.option(
    "--num-samples",
    default=0,
    show_default=True,
    type=int,
    help="Max samples to evaluate (0 = all).",
)
@click.option(
    "--batch-size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size for loss/perplexity computation.",
)
@click.option(
    "--gen-batch-size",
    default=8,
    show_default=True,
    type=int,
    help="Default generation batch size (overridable per-model via the config string).",
)
@click.option(
    "--embedding-model",
    default="Qwen/Qwen3-Embedding-0.6B",
    show_default=True,
    help="Embedding model for semantic similarity.",
)
@click.option(
    "--loss-only",
    is_flag=True,
    help="Only compute loss/perplexity — skip generation-based metrics.",
)
@click.option(
    "--mlflow-experiment",
    default="evaluation",
    show_default=True,
    help="MLflow experiment name.",
)
def main(
    models,
    eval_dataset,
    output_dir,
    num_samples,
    batch_size,
    gen_batch_size,
    embedding_model,
    loss_only,
    mlflow_experiment,
):
    """Evaluate fine-tuned models on the agent-distillation eval dataset.

    \b
    Examples:
      python 05_model_evaluation.py \\
          --models 'outputs/Qwen_lora-final,Qwen2.5-3B-lora,lora,32' \\
          --models 'Qwen/Qwen2.5-3B-Instruct,Qwen2.5-3B-base,base,32'
    """
    # ------------------------------------------------------------------
    # Parse model configs
    # ------------------------------------------------------------------
    model_configs = []
    for config_str in models:
        try:
            config = parse_model_config(config_str)
            if config['gen_batch_size'] is None:
                config['gen_batch_size'] = gen_batch_size
            model_configs.append(config)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--models")

    # Translate num_samples=0 → -1 (internal convention)
    max_samples = num_samples if num_samples > 0 else -1

    # ------------------------------------------------------------------
    # Output directory + logging
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"eval_run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    setup_logging(os.path.join(run_output_dir, "evaluation.log"))

    logging.info("=" * 80)
    logging.info("MODEL EVALUATION")
    logging.info("=" * 80)
    for cfg in model_configs:
        logging.info(f"  - {cfg['name']}: {cfg['path']} (type={cfg['type']}, batch={cfg['gen_batch_size']})")
    logging.info(f"Eval dataset:  {eval_dataset}")
    logging.info(f"Num samples:   {max_samples if max_samples > 0 else 'all'}")
    logging.info(f"Output dir:    {run_output_dir}")

    # ------------------------------------------------------------------
    # Load eval dataset
    # ------------------------------------------------------------------
    eval_df = pd.read_csv(eval_dataset)
    logging.info(f"Loaded {len(eval_df)} samples from eval dataset")

    eval_dataset_sha256 = hash_file(eval_dataset)

    # ------------------------------------------------------------------
    # MLflow: one parent run for the whole evaluation job
    # ------------------------------------------------------------------
    setup_mlflow(mlflow_experiment)

    with mlflow.start_run(run_name=f"eval-job-{timestamp}") as parent_run:
        mlflow.log_params({
            "eval_dataset": eval_dataset,
            "eval_dataset_sha256": eval_dataset_sha256,
            "num_eval_samples_requested": num_samples,
            "num_eval_samples_loaded": len(eval_df),
            "embedding_model": embedding_model,
            "loss_only": loss_only,
            "models": "; ".join(models),
        })

        # --------------------------------------------------------------
        # Evaluate each model in a nested MLflow run
        # --------------------------------------------------------------
        all_metrics = []

        for cfg in model_configs:
            model_output_dir = os.path.join(run_output_dir, cfg['name'])
            os.makedirs(model_output_dir, exist_ok=True)

            logging.info(f"\n{'='*80}")
            logging.info(f"Evaluating: {cfg['name']} (type={cfg['type']}, batch_size={cfg['gen_batch_size']})")
            logging.info(f"{'='*80}")

            with mlflow.start_run(run_name=f"eval-{cfg['name']}", nested=True):
                mlflow.log_params({
                    "model_path": cfg['path'],
                    "model_name": cfg['name'],
                    "model_type": cfg['type'],
                    "eval_dataset_sha256": eval_dataset_sha256,
                    "num_eval_samples": len(eval_df) if max_samples <= 0 else min(max_samples, len(eval_df)),
                    "embedding_model": embedding_model,
                    "gen_batch_size": cfg['gen_batch_size'],
                    "batch_size": batch_size,
                })

                try:
                    metrics = evaluate_model(
                        model_path=cfg['path'],
                        eval_df=eval_df,
                        output_dir=model_output_dir,
                        model_name=cfg['name'],
                        is_lora=cfg['is_lora'],
                        max_samples=max_samples,
                        batch_size=batch_size,
                        gen_batch_size=cfg['gen_batch_size'],
                        compute_generation=not loss_only,
                        embedding_model=embedding_model,
                    )
                    metrics['model_type'] = cfg['type']
                    all_metrics.append(metrics)

                    # Log scalar metrics to MLflow
                    mlflow_metrics = {}
                    if metrics.get('loss') is not None:
                        mlflow_metrics['loss'] = metrics['loss']
                    if metrics.get('perplexity') is not None:
                        mlflow_metrics['perplexity'] = metrics['perplexity']
                    if not loss_only:
                        for key in (
                            'teacher_abstain_rate', 'student_abstain_rate',
                            'abstain_agreement_rate', 'both_abstain_rate', 'both_answer_rate',
                            'exact_match_avg', 'embedding_similarity_avg',
                            'embedding_similarity_adjusted_avg', 'token_overlap_avg',
                        ):
                            if metrics.get(key) is not None:
                                mlflow_metrics[key] = metrics[key]

                    if mlflow_metrics:
                        mlflow.log_metrics(mlflow_metrics)

                    # Derive abstain precision/recall/F1/accuracy and log explicitly
                    # (primary metrics referenced in PHASE4_NEXT_STEPS.md)
                    ba = metrics.get('both_abstain_rate')       # both abstain  (TP / N)
                    ta = metrics.get('teacher_abstain_rate')    # teacher abstain rate
                    sa = metrics.get('student_abstain_rate')    # student abstain rate
                    ba_answer = metrics.get('both_answer_rate') # both answer   (TN / N)
                    if all(v is not None for v in (ba, ta, sa, ba_answer)):
                        precision = ba / sa if sa > 0 else 0.0
                        recall    = ba / ta if ta > 0 else 0.0
                        f1 = (2 * precision * recall / (precision + recall)
                              if (precision + recall) > 0 else 0.0)
                        accuracy  = ba + ba_answer  # (TP + TN) / N
                        abstain_extra = {
                            "abstain_precision": precision,
                            "abstain_recall":    recall,
                            "abstain_f1":        f1,
                            "abstain_accuracy":  accuracy,
                        }
                        mlflow.log_metrics(abstain_extra)

                    # Log output artifacts
                    for fname in Path(model_output_dir).glob("*.csv"):
                        mlflow.log_artifact(str(fname), artifact_path=cfg['name'])
                    for fname in Path(model_output_dir).glob("*.json"):
                        mlflow.log_artifact(str(fname), artifact_path=cfg['name'])

                except Exception as e:
                    logging.error(f"Error evaluating {cfg['name']}: {e}")
                    import traceback
                    traceback.print_exc()
                    mlflow.set_tag("status", "FAILED")
                    mlflow.set_tag("error", str(e))
                    continue

        # --------------------------------------------------------------
        # Comparison summary (rebuild from summary.json files for safety)
        # --------------------------------------------------------------
        import time
        time.sleep(2)

        all_summaries = []
        for model_dir in Path(run_output_dir).iterdir():
            if model_dir.is_dir():
                for sf in model_dir.glob("*_summary.json"):
                    try:
                        with open(sf, 'r') as f:
                            summary = json.load(f)
                        all_summaries.append({
                            'model_name': summary.get('model_name', ''),
                            'num_samples': summary.get('num_samples', 0),
                            'loss': summary.get('loss_perplexity', {}).get('loss'),
                            'perplexity': summary.get('loss_perplexity', {}).get('perplexity'),
                            'teacher_abstain_rate': summary.get('teacher_stats', {}).get('abstain_rate'),
                            'student_abstain_rate': summary.get('student_stats', {}).get('abstain_rate'),
                            'abstain_agreement_rate': summary.get('agreement', {}).get('abstain_agreement_rate'),
                            'both_abstain_rate': summary.get('agreement', {}).get('both_abstain_rate'),
                            'both_answer_rate': summary.get('agreement', {}).get('both_answer_rate'),
                            'exact_match_avg': summary.get('exact_match', {}).get('avg_score'),
                            'embedding_similarity_avg': summary.get('embedding_similarity', {}).get('all_pairs', {}).get('mean'),
                            'embedding_similarity_adjusted_avg': summary.get('embedding_similarity', {}).get('adjusted_for_abstain', {}).get('mean'),
                            'token_overlap_avg': summary.get('token_overlap', {}).get('mean'),
                            'model_path': summary.get('model_path', ''),
                            'timestamp': summary.get('timestamp_utc', ''),
                            'model_type': summary.get('model_type', ''),
                        })
                    except Exception as e:
                        logging.warning(f"Could not read {sf}: {e}")

        if all_summaries:
            summary_path = os.path.join(run_output_dir, "model_comparison_summary.csv")
            create_comparison_summary(all_summaries, summary_path)
            mlflow.log_artifact(summary_path, artifact_path="summary")

    logging.info("\nEvaluation complete!")
    logging.info(f"Results saved to: {run_output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Model Evaluation Script for Task12: Multi-task Evaluation

Evaluates models on BOTH Task1 (QA) and Task2 (Next-Action Prediction) test sets.

Task1 metrics:
- Loss and Perplexity
- Answer generation: abstain detection, embedding similarity, exact match, token overlap

Task2 metrics:
- Accuracy, Macro/Weighted F1, Precision, Recall
- Per-class metrics, confusion matrix

Usage:
    python3 scripts/04_model_evaluation.py \\
        --models "path,name,type,batch_size" \\
        --task1_test data/task1_eval_dataset.csv \\
        --task2_test data/task2_test_split.csv

    # With accelerate:
    accelerate launch --multi_gpu --num_processes=4 scripts/04_model_evaluation.py \\
        --models "path,name,type,batch_size" ...
"""

import os
import sys
import argparse
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK12_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK12_DIR / "data"
OUTPUT_DIR = TASK12_DIR / "outputs"
EVAL_DIR = OUTPUT_DIR / "evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TASK2 CONSTANTS
# ============================================================================
ACTION_VOCAB = ['continue', 'finish']
ACTION_TO_ID = {action: i for i, action in enumerate(ACTION_VOCAB)}

# ============================================================================
# TASK1 CONSTANTS - Abstain patterns
# ============================================================================
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
    return any(var in os.environ for var in ['ACCELERATE_LAUNCHED', 'LOCAL_RANK', 'WORLD_SIZE'])


def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


# ============================================================================
# TASK1 HELPER FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def normalize_for_comparison(text: str) -> str:
    t = normalize_text(text).lower()
    t = t.strip(" \t\n\r\f\v\"'`\u201c\u201d\u2018\u2019")
    return t


def extract_answer(text: str) -> str:
    """Extract the ANSWER section from model output."""
    if text is None:
        return ""
    t = text.strip()
    if not t:
        return ""

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
            ans = re.split(r"(?is)\n\s*(EVIDENCE|RATIONALE|EXPLANATION|SOURCES|REASONING)\b\s*[:\-\]]", ans)[0]
            return ans.strip()

    if len(t) < 500 and '\n' not in t[:100]:
        return t

    paragraphs = t.split('\n\n')
    if paragraphs:
        last_para = paragraphs[-1].strip()
        if len(last_para) < 300:
            return last_para
    return t


def is_abstain(answer_text: str) -> bool:
    """Check if answer indicates abstention."""
    if not answer_text or not isinstance(answer_text, str):
        return True
    t = normalize_for_comparison(answer_text)
    if not t:
        return True

    for p in ABSTAIN_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            return True

    if re.fullmatch(r"(n/?a|none|unknown|no\s*answer|abstain)", t):
        return True

    if len(t) < 3:
        return True
    return False


def compute_token_overlap(teacher: str, student: str) -> float:
    """Compute Jaccard token overlap."""
    t_tokens = set(teacher.strip().lower().split())
    s_tokens = set(student.strip().lower().split())
    if not t_tokens and not s_tokens:
        return 1.0
    if not t_tokens or not s_tokens:
        return 0.0
    intersection = len(t_tokens & s_tokens)
    union = len(t_tokens | s_tokens)
    return intersection / union if union > 0 else 0.0


# ============================================================================
# TASK2 HELPER FUNCTIONS
# ============================================================================

def extract_action_from_generation(generated_text: str) -> Optional[str]:
    """Extract action name from model generation for binary classification."""
    if not isinstance(generated_text, str):
        return None

    text = generated_text.strip().lower()
    text = re.sub(r'^(the\s+)?(best\s+)?(next\s+)?action(\s+is)?[:\s]*', '', text)
    text = re.sub(r'^output[:\s]*', '', text)
    text = re.sub(r'^answer[:\s]*', '', text)
    text = text.strip('"\'\.\,\!\?')

    if text in ACTION_VOCAB:
        return text

    first_word = text.split()[0] if text.split() else ''
    first_word = first_word.strip('"\'\.\,\!\?')
    if first_word in ACTION_VOCAB:
        return first_word

    first_line = text.split('\n')[0].strip().strip('"\'\.\,\!\?')
    if first_line in ACTION_VOCAB:
        return first_line

    for action in ACTION_VOCAB:
        pattern = r'\b' + re.escape(action) + r'\b'
        if re.search(pattern, text):
            return action

    finish_synonyms = ['finish', 'done', 'complete', 'stop', 'end', 'finished']
    continue_synonyms = ['continue', 'proceed', 'next', 'go', 'more', 'keep']
    for word in finish_synonyms:
        if word in text:
            return 'finish'
    for word in continue_synonyms:
        if word in text:
            return 'continue'

    return None


# ============================================================================
# EMBEDDING SIMILARITY (for Task1)
# ============================================================================

def load_embedding_model(model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = None):
    """Load embedding model for semantic similarity."""
    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Loading embedding model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def compute_embeddings_batch(texts: List[str], model, tokenizer, device: str,
                              batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    """Compute normalized embeddings for a batch of texts."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t and isinstance(t, str) else "" for t in texts[i:i + batch_size]]
        with torch.no_grad():
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            outputs = model(**inputs)
            mask = inputs['attention_mask']
            hidden = outputs.last_hidden_state
            mask_expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask_expanded, 1)
            count = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = summed / count
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def compute_embedding_similarities(results: List[Dict],
                                    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                                    batch_size: int = 32):
    """Compute cosine similarity between teacher and student answers."""
    model, tokenizer, device = load_embedding_model(embedding_model_name)

    sims_all = [None] * len(results)
    sims_adjusted = [None] * len(results)

    valid_indices = []
    teacher_texts = []
    student_texts = []

    for i, r in enumerate(results):
        t_ans = r['teacher_answer']
        s_ans = r['student_answer']
        t_abst = r.get('teacher_abstain', is_abstain(t_ans))
        s_abst = r.get('student_abstain', is_abstain(s_ans))

        if t_abst and s_abst:
            sims_adjusted[i] = 1.0
        elif t_abst != s_abst:
            sims_adjusted[i] = 0.0

        if t_ans and s_ans and len(t_ans.strip()) > 0 and len(s_ans.strip()) > 0:
            valid_indices.append(i)
            teacher_texts.append(t_ans)
            student_texts.append(s_ans)

    if valid_indices:
        logging.info(f"Computing embeddings for {len(valid_indices)} answer pairs...")
        t_emb = compute_embeddings_batch(teacher_texts, model, tokenizer, device, batch_size)
        s_emb = compute_embeddings_batch(student_texts, model, tokenizer, device, batch_size)
        cosine_sims = np.sum(t_emb * s_emb, axis=1)

        for idx, sim in enumerate(cosine_sims):
            orig_idx = valid_indices[idx]
            sims_all[orig_idx] = float(sim)
            if sims_adjusted[orig_idx] is None:
                sims_adjusted[orig_idx] = float(sim)

    # Free embedding model
    del model
    torch.cuda.empty_cache()

    return sims_all, sims_adjusted


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(model_path: str, model_type: str = "base"):
    """Load model and tokenizer. Supports base, lora, full_finetune types."""
    model_path_obj = Path(model_path)
    is_hf_hub = "/" in model_path and not model_path_obj.exists()

    if model_type == "lora":
        # Find base model from adapter config
        adapter_config_path = model_path_obj / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        else:
            # Infer from path name
            path_str = str(model_path).lower()
            if "qwen2.5-0.5b" in path_str:
                base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            elif "qwen2.5-1.5b" in path_str:
                base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            elif "qwen2.5-3b" in path_str:
                base_model_name = "Qwen/Qwen2.5-3B-Instruct"
            elif "qwen2.5-7b" in path_str:
                base_model_name = "Qwen/Qwen2.5-7B-Instruct"
            elif "gemma-3-1b" in path_str:
                base_model_name = "google/gemma-3-1b-it"
            elif "gemma" in path_str:
                base_model_name = "google/gemma-3-270m-it"
            else:
                base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        logging.info(f"Loading LoRA model: base={base_model_name}, adapter={model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model = model.merge_and_unload()

    elif model_type == "full_finetune":
        logging.info(f"Loading full fine-tuned model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto"
        )
    else:  # base
        logging.info(f"Loading base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ============================================================================
# TASK1 EVALUATION
# ============================================================================

def evaluate_task1(model, tokenizer, eval_df: pd.DataFrame,
                   max_samples: int = -1, batch_size: int = 8,
                   max_new_tokens: int = 256,
                   compute_embeddings: bool = True,
                   embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> Dict:
    """
    Evaluate model on Task1 (QA) test data.
    Returns dict with metrics and per-sample results.
    """
    logging.info(f"  Task1 evaluation: {len(eval_df)} samples (max={max_samples})")

    device = next(model.parameters()).device

    if max_samples > 0:
        eval_df = eval_df.head(max_samples)

    # Ensure required columns exist
    required_cols = ['llm_input', 'llm_output']
    for col in required_cols:
        if col not in eval_df.columns:
            logging.error(f"Task1 eval data missing column: {col}")
            return {"error": f"missing column {col}"}

    has_answer_col = 'llm_answer' in eval_df.columns

    # Generate answers
    tokenizer.padding_side = 'left'
    model.eval()
    results = []

    total = len(eval_df)
    with torch.no_grad():
        for start_idx in tqdm(range(0, total, batch_size), desc="Task1 generation"):
            end_idx = min(start_idx + batch_size, total)
            batch_df = eval_df.iloc[start_idx:end_idx]

            batch_prompts = []
            batch_meta = []
            for _, row in batch_df.iterrows():
                prompt = str(row['llm_input'])
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(text)
                batch_meta.append({
                    'llm_input': prompt,
                    'teacher_output': str(row['llm_output']),
                    'teacher_answer_from_dataset': str(row.get('llm_answer', '')),
                    'data_source': str(row.get('data_source', '')),
                    'teacher_id': str(row.get('teacher_id', '')),
                    'query': str(row.get('query', '')),
                })

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_lengths = inputs['attention_mask'].sum(dim=1)

            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True
            )

            for i, (out_ids, in_len, meta) in enumerate(zip(outputs, input_lengths, batch_meta)):
                gen_ids = out_ids[in_len:]
                student_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

                teacher_answer = extract_answer(meta['teacher_output'])
                student_answer = extract_answer(student_output)

                teacher_abstain = is_abstain(teacher_answer)
                student_abstain = is_abstain(student_answer)

                if teacher_abstain and student_abstain:
                    state = "both_abstain"
                elif not teacher_abstain and student_abstain:
                    state = "teacher_answer_student_abstain"
                elif teacher_abstain and not student_abstain:
                    state = "teacher_abstain_student_answer"
                else:
                    state = "both_answer"

                if teacher_abstain and student_abstain:
                    exact_score = 1.0
                else:
                    exact_score = 1.0 if normalize_for_comparison(teacher_answer) == normalize_for_comparison(student_answer) else 0.0

                results.append({
                    'data_source': meta['data_source'],
                    'teacher_id': meta['teacher_id'],
                    'query': meta['query'],
                    'llm_input': meta['llm_input'],
                    'teacher_output': meta['teacher_output'],
                    'student_output': student_output,
                    'teacher_answer': teacher_answer,
                    'student_answer': student_answer,
                    'teacher_abstain': teacher_abstain,
                    'student_abstain': student_abstain,
                    'answer_state': state,
                    'exact_match_score': exact_score,
                    'token_overlap': compute_token_overlap(teacher_answer, student_answer),
                })

    tokenizer.padding_side = 'right'

    # Compute embedding similarities (after freeing main model from caller)
    # We store results and compute embeddings later

    # Aggregate metrics
    n = len(results)
    if n == 0:
        return {"error": "no results", "num_samples": 0}

    exact_scores = [r['exact_match_score'] for r in results]
    token_overlaps = [r['token_overlap'] for r in results]

    teacher_abstain_count = sum(1 for r in results if r['teacher_abstain'])
    student_abstain_count = sum(1 for r in results if r['student_abstain'])
    agreement_count = sum(1 for r in results if r['teacher_abstain'] == r['student_abstain'])

    state_counts = {}
    for r in results:
        s = r['answer_state']
        state_counts[s] = state_counts.get(s, 0) + 1

    metrics = {
        'num_samples': n,
        'exact_match_avg': float(np.mean(exact_scores)),
        'token_overlap_avg': float(np.mean(token_overlaps)),
        'teacher_abstain_rate': teacher_abstain_count / n,
        'student_abstain_rate': student_abstain_count / n,
        'abstain_agreement_rate': agreement_count / n,
        'state_counts': state_counts,
        'state_percentages': {k: v / n * 100 for k, v in state_counts.items()},
    }

    return {'metrics': metrics, 'results': results, 'compute_embeddings': compute_embeddings,
            'embedding_model_name': embedding_model_name}


def finalize_task1_embeddings(task1_output: Dict) -> Dict:
    """Compute embedding similarities for task1 results (call after freeing eval model)."""
    if 'error' in task1_output:
        return task1_output

    results = task1_output['results']
    metrics = task1_output['metrics']

    if task1_output.get('compute_embeddings', False) and results:
        logging.info("Computing Task1 embedding similarities...")
        sims_all, sims_adjusted = compute_embedding_similarities(
            results, embedding_model_name=task1_output.get('embedding_model_name', "Qwen/Qwen3-Embedding-0.6B")
        )
        for i, r in enumerate(results):
            r['embedding_similarity'] = sims_all[i]
            r['embedding_similarity_adjusted'] = sims_adjusted[i]

        valid_sims_all = [s for s in sims_all if s is not None]
        valid_sims_adj = [s for s in sims_adjusted if s is not None]

        metrics['embedding_similarity_avg'] = float(np.mean(valid_sims_all)) if valid_sims_all else None
        metrics['embedding_similarity_adjusted_avg'] = float(np.mean(valid_sims_adj)) if valid_sims_adj else None

    task1_output['metrics'] = metrics
    task1_output['results'] = results
    return task1_output


# ============================================================================
# TASK2 EVALUATION
# ============================================================================

def evaluate_task2(model, tokenizer, eval_df: pd.DataFrame,
                   max_samples: int = -1, batch_size: int = 8,
                   max_new_tokens: int = 20) -> Dict:
    """
    Evaluate model on Task2 (Next-Action Prediction) test data.
    Returns dict with metrics and per-sample results.
    """
    logging.info(f"  Task2 evaluation: {len(eval_df)} samples (max={max_samples})")

    device = next(model.parameters()).device

    # Filter to valid samples
    valid_data = []
    for _, row in eval_df.iterrows():
        llm_input = row.get('llm_input', '')
        llm_output = row.get('llm_output', '')
        if not isinstance(llm_input, str) or not isinstance(llm_output, str):
            continue
        action = llm_output.strip().lower()
        if action not in ACTION_VOCAB:
            continue
        valid_data.append({'llm_input': llm_input, 'ground_truth': action})

    # Balance classes
    from collections import Counter
    class_counts = Counter(item['ground_truth'] for item in valid_data)
    logging.info(f"  Task2 class distribution: {dict(class_counts)}")
    min_count = min(class_counts.values()) if class_counts else 0
    if min_count > 0 and len(set(class_counts.values())) > 1:
        np.random.seed(42)
        balanced = []
        for cls in class_counts:
            cls_items = [item for item in valid_data if item['ground_truth'] == cls]
            indices = np.random.choice(len(cls_items), size=min_count, replace=False)
            balanced.extend([cls_items[i] for i in indices])
        valid_data = balanced
        np.random.shuffle(valid_data)

    if max_samples > 0:
        valid_data = valid_data[:max_samples]

    total = len(valid_data)
    logging.info(f"  Task2 eval samples (balanced): {total}")

    if total == 0:
        return {"error": "no valid task2 samples", "num_samples": 0}

    # Generate predictions
    tokenizer.padding_side = 'left'
    model.eval()
    predictions = []

    with torch.no_grad():
        for start_idx in tqdm(range(0, total, batch_size), desc="Task2 generation"):
            end_idx = min(start_idx + batch_size, total)

            batch_prompts = []
            batch_meta = []
            for idx in range(start_idx, end_idx):
                item = valid_data[idx]
                messages = [{"role": "user", "content": item['llm_input']}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(prompt)
                batch_meta.append(item)

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_lengths = inputs['attention_mask'].sum(dim=1)

            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True
            )

            for i, (out_ids, in_len, meta) in enumerate(zip(outputs, input_lengths, batch_meta)):
                gen_ids = out_ids[in_len:]
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred_action = extract_action_from_generation(generated_text)
                predictions.append({
                    'llm_input': meta['llm_input'],
                    'ground_truth': meta['ground_truth'],
                    'generated_text': generated_text,
                    'predicted_action': pred_action,
                })

    tokenizer.padding_side = 'right'

    # Compute metrics
    valid_preds = [(p['predicted_action'], p['ground_truth']) for p in predictions if p['predicted_action'] is not None]
    invalid_count = sum(1 for p in predictions if p['predicted_action'] is None)

    metrics = {
        'num_samples': total,
        'valid_predictions': len(valid_preds),
        'invalid_predictions': invalid_count,
    }

    if valid_preds:
        pred_labels = [p[0] for p in valid_preds]
        gt_labels = [p[1] for p in valid_preds]
        pred_ids = [ACTION_TO_ID.get(p, -1) for p in pred_labels]
        gt_ids = [ACTION_TO_ID.get(g, -1) for g in gt_labels]

        metrics['accuracy'] = float(accuracy_score(gt_ids, pred_ids))

        precision, recall, f1, support = precision_recall_fscore_support(
            gt_ids, pred_ids, labels=list(range(len(ACTION_VOCAB))), zero_division=0
        )
        metrics['macro_precision'] = float(np.mean(precision))
        metrics['macro_recall'] = float(np.mean(recall))
        metrics['macro_f1'] = float(np.mean(f1))

        w_p, w_r, w_f1, _ = precision_recall_fscore_support(
            gt_ids, pred_ids, average='weighted', zero_division=0
        )
        metrics['weighted_precision'] = float(w_p)
        metrics['weighted_recall'] = float(w_r)
        metrics['weighted_f1'] = float(w_f1)

        metrics['per_class'] = {}
        for i, action in enumerate(ACTION_VOCAB):
            metrics['per_class'][action] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        cm = confusion_matrix(gt_ids, pred_ids, labels=list(range(len(ACTION_VOCAB))))
        metrics['confusion_matrix'] = cm.tolist()

    return {'metrics': metrics, 'predictions': predictions}


# ============================================================================
# MAIN
# ============================================================================

def parse_model_config(config_str: str) -> Dict:
    """Parse model config: path,name,type[,batch_size]"""
    parts = [p.strip() for p in config_str.split(',')]
    if len(parts) < 3:
        raise ValueError(f"Invalid model config: {config_str}. Expected: path,name,type[,batch_size]")
    model_type = parts[2].lower()
    if model_type not in ('lora', 'full_finetune', 'base'):
        raise ValueError(f"Invalid model type: {model_type}")
    return {
        'path': parts[0],
        'name': parts[1],
        'type': model_type,
        'is_lora': model_type == 'lora',
        'gen_batch_size': int(parts[3]) if len(parts) > 3 else None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on Task1 and Task2 test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", type=str, required=True,
                        help="Model configs: 'path,name,type[,batch_size]'")
    parser.add_argument("--task1_test", type=str,
                        default=str(DATA_DIR / "task1_eval_dataset.csv"),
                        help="Path to Task1 evaluation CSV")
    parser.add_argument("--task2_test", type=str,
                        default=str(DATA_DIR / "task2_test_split.csv"),
                        help="Path to Task2 test CSV")
    parser.add_argument("--task1_samples", type=int, default=-1,
                        help="Max Task1 eval samples (-1 for all)")
    parser.add_argument("--task2_samples", type=int, default=-1,
                        help="Max Task2 eval samples (-1 for all)")
    parser.add_argument("--gen_batch_size", type=int, default=8,
                        help="Default generation batch size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="Skip embedding similarity computation for Task1")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Embedding model for Task1 semantic similarity")

    args = parser.parse_args()

    # Parse model configs
    model_configs = []
    for cfg_str in args.models:
        try:
            cfg = parse_model_config(cfg_str)
            if cfg['gen_batch_size'] is None:
                cfg['gen_batch_size'] = args.gen_batch_size
            model_configs.append(cfg)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = EVAL_DIR / f"eval_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(run_dir / "evaluation.log"))

    logging.info("=" * 80)
    logging.info("TASK12: MULTI-TASK MODEL EVALUATION")
    logging.info("=" * 80)
    logging.info(f"Models: {len(model_configs)}")
    for cfg in model_configs:
        logging.info(f"  - {cfg['name']}: {cfg['path']} (type={cfg['type']}, batch={cfg['gen_batch_size']})")
    logging.info(f"Task1 test: {args.task1_test}")
    logging.info(f"Task2 test: {args.task2_test}")
    logging.info(f"Output: {run_dir}")

    # Load test datasets
    task1_df = None
    task2_df = None

    if os.path.exists(args.task1_test):
        task1_df = pd.read_csv(args.task1_test)
        logging.info(f"Task1 test data: {len(task1_df)} rows")
    else:
        logging.warning(f"Task1 test data not found: {args.task1_test}")

    if os.path.exists(args.task2_test):
        task2_df = pd.read_csv(args.task2_test)
        logging.info(f"Task2 test data: {len(task2_df)} rows")
    else:
        logging.warning(f"Task2 test data not found: {args.task2_test}")

    if task1_df is None and task2_df is None:
        logging.error("No test data found for either task!")
        sys.exit(1)

    # Evaluate each model
    all_comparison = []

    for cfg in model_configs:
        model_dir = run_dir / cfg['name']
        model_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n{'=' * 80}")
        logging.info(f"Evaluating: {cfg['name']} ({cfg['type']})")
        logging.info(f"{'=' * 80}")

        eval_start = time.time()

        try:
            # Load model
            model, tokenizer = load_model_and_tokenizer(cfg['path'], cfg['type'])

            # --- Task1 evaluation ---
            task1_output = None
            if task1_df is not None:
                logging.info("--- Task1 (QA) Evaluation ---")
                task1_output = evaluate_task1(
                    model, tokenizer, task1_df,
                    max_samples=args.task1_samples,
                    batch_size=cfg['gen_batch_size'],
                    compute_embeddings=not args.skip_embeddings,
                    embedding_model_name=args.embedding_model
                )

            # --- Task2 evaluation ---
            task2_output = None
            if task2_df is not None:
                logging.info("--- Task2 (Next-Action) Evaluation ---")
                task2_output = evaluate_task2(
                    model, tokenizer, task2_df,
                    max_samples=args.task2_samples,
                    batch_size=cfg['gen_batch_size']
                )

            # Free model before embedding computation
            del model
            torch.cuda.empty_cache()

            # Compute Task1 embeddings (needs GPU, so do after freeing eval model)
            if task1_output and 'error' not in task1_output:
                task1_output = finalize_task1_embeddings(task1_output)

            eval_time = time.time() - eval_start

            # --- Save results ---
            summary = {
                'model_name': cfg['name'],
                'model_path': cfg['path'],
                'model_type': cfg['type'],
                'eval_time_s': eval_time,
                'timestamp': utc_timestamp(),
            }

            # Task1 results
            if task1_output and 'error' not in task1_output:
                summary['task1'] = task1_output['metrics']

                # Save detailed CSV
                t1_df = pd.DataFrame(task1_output['results'])
                t1_df.to_csv(model_dir / f"{cfg['name']}_task1_detailed_results.csv", index=False)

                logging.info(f"\n  Task1 Results:")
                logging.info(f"    Exact match: {task1_output['metrics']['exact_match_avg']:.4f}")
                logging.info(f"    Token overlap: {task1_output['metrics']['token_overlap_avg']:.4f}")
                logging.info(f"    Abstain agreement: {task1_output['metrics']['abstain_agreement_rate']:.4f}")
                if task1_output['metrics'].get('embedding_similarity_adjusted_avg') is not None:
                    logging.info(f"    Embedding sim (adj): {task1_output['metrics']['embedding_similarity_adjusted_avg']:.4f}")

            # Task2 results
            if task2_output and 'error' not in task2_output:
                summary['task2'] = task2_output['metrics']

                # Save detailed CSV
                t2_df = pd.DataFrame(task2_output['predictions'])
                t2_df.to_csv(model_dir / f"{cfg['name']}_task2_detailed_results.csv", index=False)

                logging.info(f"\n  Task2 Results:")
                logging.info(f"    Accuracy: {task2_output['metrics'].get('accuracy', 0)*100:.2f}%")
                logging.info(f"    Macro F1: {task2_output['metrics'].get('macro_f1', 0):.4f}")
                logging.info(f"    Valid preds: {task2_output['metrics']['valid_predictions']}/{task2_output['metrics']['num_samples']}")

            # Save summary JSON
            with open(model_dir / f"{cfg['name']}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            # Build comparison row
            comp_row = {
                'model_name': cfg['name'],
                'model_type': cfg['type'],
                'eval_time_s': eval_time,
            }
            if task1_output and 'error' not in task1_output:
                t1m = task1_output['metrics']
                comp_row['t1_exact_match'] = t1m.get('exact_match_avg')
                comp_row['t1_token_overlap'] = t1m.get('token_overlap_avg')
                comp_row['t1_abstain_agreement'] = t1m.get('abstain_agreement_rate')
                comp_row['t1_teacher_abstain_rate'] = t1m.get('teacher_abstain_rate')
                comp_row['t1_student_abstain_rate'] = t1m.get('student_abstain_rate')
                comp_row['t1_embedding_sim_adj'] = t1m.get('embedding_similarity_adjusted_avg')
            if task2_output and 'error' not in task2_output:
                t2m = task2_output['metrics']
                comp_row['t2_accuracy'] = t2m.get('accuracy')
                comp_row['t2_macro_f1'] = t2m.get('macro_f1')
                comp_row['t2_weighted_f1'] = t2m.get('weighted_f1')
                comp_row['t2_macro_precision'] = t2m.get('macro_precision')
                comp_row['t2_macro_recall'] = t2m.get('macro_recall')
            all_comparison.append(comp_row)

            logging.info(f"\n  Eval time: {eval_time:.1f}s")

        except Exception as e:
            logging.error(f"Error evaluating {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            # Ensure GPU cleanup even on error
            torch.cuda.empty_cache()
            continue

    # Save comparison CSV
    if all_comparison:
        comp_df = pd.DataFrame(all_comparison)
        comp_path = run_dir / "model_comparison_summary.csv"
        comp_df.to_csv(comp_path, index=False)
        logging.info(f"\nComparison summary saved to: {comp_path}")
        logging.info("\n" + "=" * 100)
        logging.info("MODEL COMPARISON SUMMARY")
        logging.info("=" * 100)
        logging.info("\n" + comp_df.to_string(index=False))

    logging.info(f"\n{'=' * 80}")
    logging.info("Evaluation complete!")
    logging.info(f"Results: {run_dir}")
    logging.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()

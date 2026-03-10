"""
eval_utils.py — Shared evaluation utilities for agent-distillation task1.

Extracted from 05_model_evaluation.py so that 07_tta_experiment.py and future
scripts can import without resorting to importlib hacks.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Abstain detection patterns
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


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
    t = t.strip(" \t\n\r\f\v\"'`\u201c\u201d\u2018\u2019")
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
            ans = re.split(
                r"(?is)\n\s*(EVIDENCE|RATIONALE|EXPLANATION|SOURCES|REASONING)\b\s*[:\-\]]",
                ans,
            )[0]
            return ans.strip()

    if len(t) < 500 and "\n" not in t[:100]:
        return t

    paragraphs = t.split("\n\n")
    if paragraphs:
        last_para = paragraphs[-1].strip()
        if len(last_para) < 300:
            return last_para

    return t


def is_abstain(answer_text: str) -> bool:
    """Check if an answer indicates abstention using robust pattern matching."""
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EvalDataset(Dataset):
    """Dataset for evaluation with full row data."""

    def __init__(self, df, tokenizer, max_length: int = 2048):
        import pandas as pd  # local import to avoid hard dep at module level

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "llm_input": str(row["llm_input"]),
            "llm_output": str(row["llm_output"]),
            "llm_answer": str(row.get("llm_answer", "")),
            "decision_label": str(row.get("decision_label", "")),
            "data_source": str(row.get("data_source", "")),
            "teacher_id": str(row.get("teacher_id", "")),
            "query": str(row.get("query", "")),
            "search_index": str(row.get("search_index", "")),
            "idx": idx,
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_path: str, is_lora: bool = False):
    """Load model and tokenizer from path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.info(f"Loading model from: {model_path}")

    if is_lora:
        from peft import PeftModel

        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get(
                "base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct"
            )
        else:
            base_model_name = "Qwen/Qwen2.5-3B-Instruct"

        logging.info(f"Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        logging.info(f"Applying LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------


def load_qwen_embedding_model(
    model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = None
):
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
    max_length: int = 512,
) -> np.ndarray:
    """Compute embeddings for a batch of texts using Qwen embedding model."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_texts = [t if t and isinstance(t, str) else "" for t in batch_texts]

        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)

            attention_mask = inputs["attention_mask"]
            last_hidden = outputs.last_hidden_state

            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def compute_embedding_similarities(
    results: List[Dict],
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 32,
    embedding_model=None,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Compute cosine similarity between teacher and student answers using embeddings.

    Returns two lists:
        - similarities_all: Embedding similarity for all pairs (None if either empty)
        - similarities_adjusted: Same but 0 if abstain mismatch, 1 if both abstain
    """
    if embedding_model is None:
        model, tokenizer, device = load_qwen_embedding_model(embedding_model_name)
    else:
        model, tokenizer, device = embedding_model

    similarities_all = []
    similarities_adjusted = []

    valid_indices = []
    teacher_texts = []
    student_texts = []

    for i, r in enumerate(results):
        teacher_ans = r["teacher_answer"]
        student_ans = r["student_answer"]
        t_abstain = r.get("teacher_abstain", is_abstain(teacher_ans))
        s_abstain = r.get("student_abstain", is_abstain(student_ans))

        if t_abstain and s_abstain:
            similarities_adjusted.append(1.0)
        elif t_abstain != s_abstain:
            similarities_adjusted.append(0.0)
        else:
            similarities_adjusted.append(None)

        if (
            teacher_ans
            and student_ans
            and len(teacher_ans.strip()) > 0
            and len(student_ans.strip()) > 0
        ):
            valid_indices.append(i)
            teacher_texts.append(teacher_ans)
            student_texts.append(student_ans)
            similarities_all.append(None)
        else:
            similarities_all.append(None)

    if valid_indices:
        logging.info(
            f"Computing embeddings for {len(valid_indices)} valid answer pairs..."
        )
        teacher_embeddings = compute_qwen_embeddings_batch(
            teacher_texts, model, tokenizer, device, batch_size
        )
        student_embeddings = compute_qwen_embeddings_batch(
            student_texts, model, tokenizer, device, batch_size
        )

        cosine_sims = np.sum(teacher_embeddings * student_embeddings, axis=1)

        for idx, sim in enumerate(cosine_sims):
            orig_idx = valid_indices[idx]
            similarities_all[orig_idx] = float(sim)
            if similarities_adjusted[orig_idx] is None:
                similarities_adjusted[orig_idx] = float(sim)

    return similarities_all, similarities_adjusted


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


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


def compute_state_counts(results: List[Dict]) -> Dict[str, int]:
    """Compute state counts for answer agreement analysis."""
    state_counts = {
        "both_abstain": 0,
        "teacher_answer_student_abstain": 0,
        "teacher_abstain_student_answer": 0,
        "both_answer": 0,
    }
    for r in results:
        state = r.get("answer_state", "")
        if state in state_counts:
            state_counts[state] += 1
    return state_counts


def summarize_numeric(values: List[float]) -> Dict[str, Any]:
    """Compute summary statistics for numeric values."""
    arr = np.array(
        [v for v in values if v is not None and np.isfinite(v)], dtype=np.float32
    )
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "median": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


# ---------------------------------------------------------------------------
# Logging / timestamp helpers
# ---------------------------------------------------------------------------


def setup_logging(log_file: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def utc_timestamp() -> str:
    """Get UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


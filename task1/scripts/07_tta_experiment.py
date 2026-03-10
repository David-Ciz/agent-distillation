#!/usr/bin/env python3
"""
TTA (Test-Time Aggregation / Self-Consistency) Experiment
==========================================================
Implements the experiment described in plans/PHASE5_TTA.md.

Hypothesis: Running each eval sample N times with temperature > 0 and
aggregating via majority vote / centroid selection improves performance,
especially for small over-abstaining models (Qwen 0.5B, Gemma 270M).

Three aggregation methods:
  majority_vote — Abstain if > N/2 outputs abstain; else centroid of answers
  centroid      — Same abstain logic; answer is always centroid of all answers
  oracle        — Pick the output with highest similarity to teacher answer
                  (analysis-only upper bound, requires ground truth)

Sanity check: N=1, T=0 must reproduce the 05_model_evaluation.py baseline
within ±0.002 Abstain F1.

Output file structure
---------------------
task1/outputs/evaluations/
└── tta_run_{timestamp}/
    ├── tta_run.log
    ├── tta_comparison_summary.csv
    └── {model_name}/
        └── N{n}_T{temp_str}/
            ├── {model_name}_N{n}_T{temp_str}_{agg}_detailed_results.csv
            └── {model_name}_N{n}_T{temp_str}_summary.json

Usage
-----
python 07_tta_experiment.py \\
    --models 'outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32' \\
    --n-values 1,3,5 \\
    --temperatures 0.7 \\
    --aggregations majority_vote,centroid,oracle
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import mlflow
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from eval_utils import (
    EvalDataset,
    compute_embedding_similarities,
    compute_qwen_embeddings_batch,
    compute_state_counts,
    compute_token_overlap,
    extract_answer,
    is_abstain,
    load_model_and_tokenizer,
    load_qwen_embedding_model,
    normalize_for_comparison,
    setup_logging,
    summarize_numeric,
    utc_timestamp,
)
from mlflow_utils import hash_file, setup_mlflow

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
OUTPUT_DIR = os.path.join(TASK1_DIR, "outputs")
EVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluations")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# Canonical abstain phrase for oracle matching
_CANONICAL_ABSTAIN = "I cannot answer based on the provided evidence."


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_n_answers(
    model,
    tokenizer,
    dataset: EvalDataset,
    device,
    n: int,
    temperature: float,
    max_new_tokens: int = 256,
    max_samples: int = -1,
    batch_size: int = 8,
) -> List[List[str]]:
    """
    Run N generation passes over the dataset.

    Returns raw_outputs[sample_idx][pass_idx] — list-of-lists of raw decoded
    strings.  For N=1, temperature=0 this is greedy and reproduces the 05
    baseline exactly.

    The model is loaded *once* and looped N times — no reloading.
    """
    total = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))

    do_sample = (temperature > 0) or (n > 1)
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # raw_outputs[sample_idx] = list of n raw strings (appended pass by pass)
    raw_outputs: List[List[str]] = [[] for _ in range(total)]

    for pass_idx in range(n):
        logging.info(f"  Generation pass {pass_idx + 1}/{n} (T={temperature}, do_sample={do_sample})")
        with torch.no_grad():
            for start in tqdm(
                range(0, total, batch_size),
                desc=f"Pass {pass_idx + 1}/{n}",
                leave=False,
            ):
                end = min(start + batch_size, total)
                batch_prompts = []
                for idx in range(start, end):
                    item = dataset[idx]
                    msgs = [{"role": "user", "content": item["llm_input"]}]
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    batch_prompts.append(text)

                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_lengths = inputs["attention_mask"].sum(dim=1)

                outputs = model.generate(**inputs, **gen_kwargs)

                for i, (out_ids, in_len) in enumerate(zip(outputs, input_lengths)):
                    generated_ids = out_ids[in_len:]
                    text_out = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    raw_outputs[start + i].append(text_out)

    tokenizer.padding_side = "right"
    return raw_outputs


# ---------------------------------------------------------------------------
# Centroid selection
# ---------------------------------------------------------------------------


def select_centroid_answer(
    answer_texts: List[str],
    embedding_model_tuple: Tuple,
) -> Tuple[str, int]:
    """
    Given a list of non-empty answer strings, return (best_text, best_idx)
    where best_idx is the index with the highest cosine similarity to the
    centroid of all embeddings.
    """
    model, tokenizer, device = embedding_model_tuple
    embeddings = compute_qwen_embeddings_batch(
        answer_texts, model, tokenizer, device, batch_size=len(answer_texts)
    )
    centroid = embeddings.mean(axis=0, keepdims=True)
    # Normalize centroid
    norm = np.linalg.norm(centroid) + 1e-9
    centroid = centroid / norm
    sims = (embeddings * centroid).sum(axis=1)
    best_idx = int(np.argmax(sims))
    return answer_texts[best_idx], best_idx


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------


def _classify_outputs(raw_outputs_for_sample: List[str]) -> Tuple[List[str], List[str]]:
    """Split raw outputs into abstain texts and answer texts."""
    abstain_texts: List[str] = []
    answer_texts: List[str] = []
    for raw in raw_outputs_for_sample:
        ans = extract_answer(raw)
        if is_abstain(ans):
            abstain_texts.append(raw)
        else:
            answer_texts.append(raw)
    return abstain_texts, answer_texts


def aggregate_majority_vote(
    raw_outputs: List[List[str]],
    embedding_model_tuple: Tuple,
) -> List[Dict]:
    """
    Per sample:
      - Count abstain vs answer outputs across N passes.
      - If abstain_count > N/2: mark abstain, use first abstain raw output.
      - Else: centroid of answer texts is the representative.
    """
    results = []
    for sample_outputs in tqdm(raw_outputs, desc="Agg: majority_vote", leave=False):
        n = len(sample_outputs)
        abstain_texts, answer_texts = _classify_outputs(sample_outputs)
        abstain_count = len(abstain_texts)
        answer_count = len(answer_texts)

        if abstain_count > n / 2:
            rep_output = abstain_texts[0]
            rep_answer = extract_answer(rep_output)
            student_abstain = True
        elif answer_texts:
            if len(answer_texts) == 1:
                rep_output = answer_texts[0]
            else:
                extracted = [extract_answer(t) for t in answer_texts]
                _, best_idx = select_centroid_answer(extracted, embedding_model_tuple)
                rep_output = answer_texts[best_idx]
            rep_answer = extract_answer(rep_output)
            student_abstain = is_abstain(rep_answer)
        else:
            # All outputs are abstains (abstain_count == n/2 == 0 edge case)
            rep_output = sample_outputs[0]
            rep_answer = extract_answer(rep_output)
            student_abstain = True

        results.append(
            {
                "student_output": rep_output,
                "student_answer": rep_answer,
                "student_abstain": student_abstain,
                "vote_abstain_count": abstain_count,
                "vote_answer_count": answer_count,
                "aggregation_method": "majority_vote",
            }
        )
    return results


def aggregate_centroid(
    raw_outputs: List[List[str]],
    embedding_model_tuple: Tuple,
) -> List[Dict]:
    """
    Per sample: same abstain logic as majority_vote, but always calls
    select_centroid_answer even if all/none abstain.
    """
    results = []
    for sample_outputs in tqdm(raw_outputs, desc="Agg: centroid", leave=False):
        n = len(sample_outputs)
        abstain_texts, answer_texts = _classify_outputs(sample_outputs)
        abstain_count = len(abstain_texts)
        answer_count = len(answer_texts)

        if abstain_count > n / 2 or not answer_texts:
            # Majority said abstain — centroid over abstain texts
            all_texts = abstain_texts if abstain_texts else sample_outputs
            extracted = [extract_answer(t) for t in all_texts]
            if len(extracted) == 1:
                rep_output = all_texts[0]
            else:
                _, best_idx = select_centroid_answer(extracted, embedding_model_tuple)
                rep_output = all_texts[best_idx]
            rep_answer = extract_answer(rep_output)
            student_abstain = True
        else:
            extracted = [extract_answer(t) for t in answer_texts]
            if len(extracted) == 1:
                rep_output = answer_texts[0]
            else:
                _, best_idx = select_centroid_answer(extracted, embedding_model_tuple)
                rep_output = answer_texts[best_idx]
            rep_answer = extract_answer(rep_output)
            student_abstain = is_abstain(rep_answer)

        results.append(
            {
                "student_output": rep_output,
                "student_answer": rep_answer,
                "student_abstain": student_abstain,
                "vote_abstain_count": abstain_count,
                "vote_answer_count": answer_count,
                "aggregation_method": "centroid",
            }
        )
    return results


def aggregate_oracle(
    raw_outputs: List[List[str]],
    teacher_answers: List[str],
    embedding_model_tuple: Tuple,
) -> List[Dict]:
    """
    Oracle (analysis-only upper bound): pick the output with highest
    embedding similarity to the teacher answer.

    For abstain cases: if teacher abstains, prefer a matching abstain output.
    """
    model, tokenizer, device = embedding_model_tuple
    results = []

    for sample_outputs, teacher_ans in tqdm(
        zip(raw_outputs, teacher_answers), desc="Agg: oracle", total=len(raw_outputs), leave=False
    ):
        n = len(sample_outputs)
        abstain_texts, answer_texts = _classify_outputs(sample_outputs)
        teacher_abstains = is_abstain(teacher_ans)

        if teacher_abstains:
            # Prefer an output that also abstains
            if abstain_texts:
                rep_output = abstain_texts[0]
            else:
                # No matching abstain — pick output most similar to canonical abstain phrase
                embeddings = compute_qwen_embeddings_batch(
                    [extract_answer(t) for t in sample_outputs],
                    model, tokenizer, device, batch_size=n,
                )
                canon_emb = compute_qwen_embeddings_batch(
                    [_CANONICAL_ABSTAIN], model, tokenizer, device, batch_size=1
                )
                sims = (embeddings * canon_emb).sum(axis=1)
                rep_output = sample_outputs[int(np.argmax(sims))]
        else:
            # Pick the output most similar to the teacher answer
            candidate_answers = [extract_answer(t) for t in sample_outputs]
            # Filter empties
            valid = [(i, a) for i, a in enumerate(candidate_answers) if a.strip()]
            if not valid:
                rep_output = sample_outputs[0]
            elif len(valid) == 1:
                rep_output = sample_outputs[valid[0][0]]
            else:
                indices, texts = zip(*valid)
                cand_embs = compute_qwen_embeddings_batch(
                    list(texts), model, tokenizer, device, batch_size=len(texts)
                )
                teacher_emb = compute_qwen_embeddings_batch(
                    [teacher_ans], model, tokenizer, device, batch_size=1
                )
                sims = (cand_embs * teacher_emb).sum(axis=1)
                best_local = int(np.argmax(sims))
                rep_output = sample_outputs[indices[best_local]]

        rep_answer = extract_answer(rep_output)
        student_abstain = is_abstain(rep_answer)

        results.append(
            {
                "student_output": rep_output,
                "student_answer": rep_answer,
                "student_abstain": student_abstain,
                "vote_abstain_count": len(abstain_texts),
                "vote_answer_count": len(answer_texts),
                "aggregation_method": "oracle",
            }
        )
    return results


# ---------------------------------------------------------------------------
# Build results in 05-compatible schema
# ---------------------------------------------------------------------------


def build_results_from_aggregated(
    aggregated: List[Dict],
    dataset: EvalDataset,
    tta_n: int,
    tta_temperature: float,
) -> List[Dict]:
    """
    Combine aggregated student outputs with dataset metadata to produce a
    results list that is schema-compatible with 05_model_evaluation.py.

    Standard columns (identical to 05) + TTA-extra columns.
    """
    results = []
    for i, agg in enumerate(aggregated):
        item = dataset[i]
        teacher_output = item["llm_output"]
        teacher_answer = extract_answer(teacher_output)
        teacher_abstain = is_abstain(teacher_answer)
        student_answer = agg["student_answer"]
        student_abstain = agg["student_abstain"]

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
            exact_score = (
                1.0
                if normalize_for_comparison(teacher_answer)
                == normalize_for_comparison(student_answer)
                else 0.0
            )

        results.append(
            {
                # --- standard columns (compatible with 06) ---
                "idx": item["idx"],
                "data_source": item["data_source"],
                "teacher_id": item["teacher_id"],
                "query": item["query"],
                "search_index": item["search_index"],
                "llm_input": item["llm_input"],
                "teacher_output": teacher_output,
                "student_output": agg["student_output"],
                "teacher_answer": teacher_answer,
                "student_answer": student_answer,
                "teacher_abstain": teacher_abstain,
                "student_abstain": student_abstain,
                "answer_state": state,
                "exact_match_score": exact_score,
                "embedding_similarity": None,       # filled later
                "embedding_similarity_adjusted": None,
                "token_overlap": compute_token_overlap(teacher_answer, student_answer),
                "decision_label": item["decision_label"],
                # --- TTA-extra columns ---
                "tta_n": tta_n,
                "tta_temperature": tta_temperature,
                "tta_aggregation": agg["aggregation_method"],
                "vote_abstain_count": agg["vote_abstain_count"],
                "vote_answer_count": agg["vote_answer_count"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Metrics helpers (mirrors 05 evaluate_model())
# ---------------------------------------------------------------------------

STANDARD_COLUMN_ORDER = [
    "idx", "data_source", "teacher_id", "query", "search_index",
    "llm_input", "teacher_output", "student_output",
    "teacher_answer", "student_answer",
    "teacher_abstain", "student_abstain", "answer_state",
    "exact_match_score", "embedding_similarity", "embedding_similarity_adjusted",
    "token_overlap", "decision_label",
    "tta_n", "tta_temperature", "tta_aggregation",
    "vote_abstain_count", "vote_answer_count",
]


def compute_flat_metrics(results: List[Dict], model_name: str, tta_n: int,
                         tta_temperature: float, aggregation: str) -> Dict:
    """Return a flat dict of scalar metrics (one row for the comparison CSV)."""
    n = len(results)
    state_counts = compute_state_counts(results)

    teacher_abstain_count = sum(1 for r in results if r["teacher_abstain"])
    student_abstain_count = sum(1 for r in results if r["student_abstain"])
    agreement_count = sum(1 for r in results if r["teacher_abstain"] == r["student_abstain"])

    ta = teacher_abstain_count / n if n else 0.0
    sa = student_abstain_count / n if n else 0.0
    ba = state_counts["both_abstain"] / n if n else 0.0
    ba_ans = state_counts["both_answer"] / n if n else 0.0

    precision = ba / sa if sa > 0 else 0.0
    recall = ba / ta if ta > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = ba + ba_ans

    exact_scores = [r["exact_match_score"] for r in results]
    sims_adj = [r["embedding_similarity_adjusted"] for r in results if r["embedding_similarity_adjusted"] is not None]

    return {
        "model_name": model_name,
        "tta_n": tta_n,
        "tta_temperature": tta_temperature,
        "aggregation": aggregation,
        "num_samples": n,
        "teacher_abstain_rate": ta,
        "student_abstain_rate": sa,
        "abstain_agreement_rate": agreement_count / n if n else 0.0,
        "both_abstain_rate": ba,
        "both_answer_rate": ba_ans,
        "abstain_precision": precision,
        "abstain_recall": recall,
        "abstain_f1": f1,
        "abstain_accuracy": accuracy,
        "exact_match_avg": float(np.mean(exact_scores)) if exact_scores else 0.0,
        "embedding_similarity_adjusted_avg": float(np.mean(sims_adj)) if sims_adj else 0.0,
        "vote_abstain_rate_mean": float(
            np.mean([r["vote_abstain_count"] / max(r["tta_n"], 1) for r in results])
        ),
    }


# ---------------------------------------------------------------------------
# Parse model config (same format as 05)
# ---------------------------------------------------------------------------


def parse_model_config(config_str: str) -> Dict:
    parts = [p.strip() for p in config_str.split(",")]
    if len(parts) < 3:
        raise ValueError(
            f"Invalid model config: {config_str!r}. Expected: path,name,type[,gen_batch_size]"
        )
    model_type = parts[2].lower()
    if model_type not in ("lora", "full_finetune", "base"):
        raise ValueError(f"Invalid model type: {model_type}")
    return {
        "path": parts[0],
        "name": parts[1],
        "type": model_type,
        "is_lora": model_type == "lora",
        "gen_batch_size": int(parts[3]) if len(parts) > 3 else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Model config: 'path,name,type[,gen_batch_size]'. Repeat for multiple models.",
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
    help="Base directory for TTA results.",
)
@click.option(
    "--n-values",
    default="1,3,5",
    show_default=True,
    help="Comma-separated N values to sweep, e.g. '1,3,5,10'.",
)
@click.option(
    "--temperatures",
    default="0.7",
    show_default=True,
    help="Comma-separated temperatures, e.g. '0.5,0.7,1.0'.",
)
@click.option(
    "--aggregations",
    default="majority_vote,centroid,oracle",
    show_default=True,
    help="Comma-separated aggregation methods.",
)
@click.option(
    "--num-samples",
    default=0,
    show_default=True,
    type=int,
    help="Max samples (0 = all).",
)
@click.option(
    "--gen-batch-size",
    default=8,
    show_default=True,
    type=int,
    help="Default generation batch size.",
)
@click.option(
    "--max-new-tokens",
    default=256,
    show_default=True,
    type=int,
    help="Max new tokens per generation.",
)
@click.option(
    "--embedding-model",
    default="Qwen/Qwen3-Embedding-0.6B",
    show_default=True,
    help="Embedding model for similarity and centroid selection.",
)
@click.option(
    "--mlflow-experiment",
    default="tta_experiment",
    show_default=True,
    help="MLflow experiment name.",
)
def main(
    models,
    eval_dataset,
    output_dir,
    n_values,
    temperatures,
    aggregations,
    num_samples,
    gen_batch_size,
    max_new_tokens,
    embedding_model,
    mlflow_experiment,
):
    """TTA / Self-Consistency experiment for agent-distillation task1.

    \b
    Examples:
      # Phase A — quick signal
      python 07_tta_experiment.py \\
          --models 'outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32' \\
          --n-values 1,3,5 --temperatures 0.7

      # Sanity check (should reproduce 05 baseline ±0.002 F1)
      python 07_tta_experiment.py \\
          --models 'outputs/Qwen_Qwen2.5-0.5B-Instruct-lora-final,Qwen2.5-0.5B-Instruct-lora,lora,32' \\
          --n-values 1 --temperatures 0.0 --aggregations majority_vote
    """
    # ------------------------------------------------------------------
    # Parse inputs
    # ------------------------------------------------------------------
    n_list = sorted(set(int(x.strip()) for x in n_values.split(",")))
    temp_list = sorted(set(float(x.strip()) for x in temperatures.split(",")))
    agg_list = [a.strip() for a in aggregations.split(",")]

    valid_aggs = {"majority_vote", "centroid", "oracle"}
    for a in agg_list:
        if a not in valid_aggs:
            raise click.BadParameter(f"Unknown aggregation: {a!r}. Must be one of {valid_aggs}")

    model_configs = []
    for cfg_str in models:
        try:
            cfg = parse_model_config(cfg_str)
            if cfg["gen_batch_size"] is None:
                cfg["gen_batch_size"] = gen_batch_size
            model_configs.append(cfg)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--models")

    max_samples = num_samples if num_samples > 0 else -1
    max_n = max(n_list)

    # ------------------------------------------------------------------
    # Output dir + logging
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"tta_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(os.path.join(run_dir, "tta_run.log"))

    logging.info("=" * 80)
    logging.info("TTA EXPERIMENT")
    logging.info("=" * 80)
    logging.info(f"Models:       {[c['name'] for c in model_configs]}")
    logging.info(f"N values:     {n_list}  (max_n={max_n})")
    logging.info(f"Temperatures: {temp_list}")
    logging.info(f"Aggregations: {agg_list}")
    logging.info(f"Output dir:   {run_dir}")

    # ------------------------------------------------------------------
    # Load eval dataset
    # ------------------------------------------------------------------
    eval_df = pd.read_csv(eval_dataset)
    logging.info(f"Loaded {len(eval_df)} samples from {eval_dataset}")
    eval_dataset_sha256 = hash_file(eval_dataset)

    # ------------------------------------------------------------------
    # MLflow parent run
    # ------------------------------------------------------------------
    setup_mlflow(mlflow_experiment)

    all_flat_metrics: List[Dict] = []

    with mlflow.start_run(run_name=f"tta-job-{timestamp}") as _parent:
        mlflow.log_params(
            {
                "eval_dataset": eval_dataset,
                "eval_dataset_sha256": eval_dataset_sha256,
                "n_values": n_values,
                "temperatures": temperatures,
                "aggregations": aggregations,
                "num_samples_requested": num_samples,
                "num_samples_loaded": len(eval_df),
                "embedding_model": embedding_model,
                "models": "; ".join(models),
            }
        )
        mlflow.set_tag("oracle_is_upper_bound", "true")

        # ==================================================================
        # Outer loop: model
        # ==================================================================
        for cfg in model_configs:
            logging.info(f"\n{'='*80}")
            logging.info(f"Model: {cfg['name']}  ({cfg['path']})")
            logging.info(f"{'='*80}")

            model, tokenizer = load_model_and_tokenizer(cfg["path"], is_lora=cfg["is_lora"])
            device = next(model.parameters()).device

            if max_samples > 0:
                df_subset = eval_df.head(max_samples)
            else:
                df_subset = eval_df
            dataset = EvalDataset(df_subset, tokenizer)
            total_samples = len(dataset)

            # Precompute teacher answers (needed for oracle)
            teacher_answers = [
                extract_answer(dataset[i]["llm_output"]) for i in range(total_samples)
            ]

            # Load embedding model once per outer model (kept alive for centroid/oracle)
            logging.info(f"Loading embedding model: {embedding_model}")
            emb_model, emb_tokenizer, emb_device = load_qwen_embedding_model(embedding_model)
            emb_tuple = (emb_model, emb_tokenizer, emb_device)

            model_dir = os.path.join(run_dir, cfg["name"])

            # ==============================================================
            # Middle loop: temperature
            # ==============================================================
            for temperature in temp_list:
                temp_str = str(temperature).replace(".", "")  # e.g. "07" for 0.7
                logging.info(f"\n  Temperature: {temperature}")

                # Generate max_n passes (subset for lower N values is free)
                logging.info(f"  Generating {max_n} passes (T={temperature})...")
                raw_outputs = generate_n_answers(
                    model,
                    tokenizer,
                    dataset,
                    device,
                    n=max_n,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    max_samples=max_samples,
                    batch_size=cfg["gen_batch_size"],
                )

                # ==========================================================
                # Inner loop: N
                # ==========================================================
                for n in n_list:
                    # Take only the first n passes
                    raw_n = [sample_outs[:n] for sample_outs in raw_outputs]

                    nt_dir = os.path.join(model_dir, f"N{n}_T{temp_str}")
                    os.makedirs(nt_dir, exist_ok=True)

                    summary_json: Dict[str, Any] = {
                        "timestamp_utc": utc_timestamp(),
                        "model_name": cfg["name"],
                        "model_path": cfg["path"],
                        "model_type": cfg["type"],
                        "tta_n": n,
                        "tta_temperature": temperature,
                        "num_samples": total_samples,
                        "aggregations": {},
                    }

                    # ======================================================
                    # Innermost loop: aggregation method
                    # ======================================================
                    mlflow_child_metrics: Dict[str, float] = {}

                    for agg in agg_list:
                        logging.info(f"    Aggregating: {agg}  (N={n}, T={temperature})")

                        if agg == "majority_vote":
                            agg_results = aggregate_majority_vote(raw_n, emb_tuple)
                        elif agg == "centroid":
                            agg_results = aggregate_centroid(raw_n, emb_tuple)
                        elif agg == "oracle":
                            agg_results = aggregate_oracle(raw_n, teacher_answers, emb_tuple)

                        results = build_results_from_aggregated(
                            agg_results, dataset, tta_n=n, tta_temperature=temperature
                        )

                        # Compute embedding similarities
                        sims_all, sims_adj = compute_embedding_similarities(
                            results, embedding_model_name=embedding_model, embedding_model=emb_tuple
                        )
                        for j, r in enumerate(results):
                            r["embedding_similarity"] = sims_all[j]
                            r["embedding_similarity_adjusted"] = sims_adj[j]

                        # Save detailed CSV
                        csv_name = f"{cfg['name']}_N{n}_T{temp_str}_{agg}_detailed_results.csv"
                        results_df = pd.DataFrame(results)
                        cols = [c for c in STANDARD_COLUMN_ORDER if c in results_df.columns]
                        results_df = results_df[cols]
                        csv_path = os.path.join(nt_dir, csv_name)
                        results_df.to_csv(csv_path, index=False)
                        logging.info(f"      Saved: {csv_path}")

                        # Flat metrics
                        flat = compute_flat_metrics(
                            results, cfg["name"], n, temperature, agg
                        )
                        all_flat_metrics.append(flat)

                        # Accumulate for MLflow child
                        for k, v in flat.items():
                            if isinstance(v, float):
                                mlflow_child_metrics[f"{k}_{agg}"] = v

                        # Store in summary JSON
                        token_overlaps = [r["token_overlap"] for r in results]
                        summary_json["aggregations"][agg] = {
                            "teacher_abstain_rate": flat["teacher_abstain_rate"],
                            "student_abstain_rate": flat["student_abstain_rate"],
                            "abstain_precision": flat["abstain_precision"],
                            "abstain_recall": flat["abstain_recall"],
                            "abstain_f1": flat["abstain_f1"],
                            "abstain_accuracy": flat["abstain_accuracy"],
                            "exact_match_avg": flat["exact_match_avg"],
                            "embedding_similarity_adjusted_avg": flat["embedding_similarity_adjusted_avg"],
                            "token_overlap": summarize_numeric(token_overlaps),
                        }

                    # Save summary JSON
                    json_path = os.path.join(nt_dir, f"{cfg['name']}_N{n}_T{temp_str}_summary.json")
                    with open(json_path, "w") as f:
                        json.dump(summary_json, f, indent=2)
                    logging.info(f"    Summary JSON saved: {json_path}")

                    # MLflow nested run: one per (model × N × temperature)
                    run_name = f"tta-{cfg['name']}-N{n}-T{temp_str}"
                    with mlflow.start_run(run_name=run_name, nested=True):
                        mlflow.log_params(
                            {
                                "model_name": cfg["name"],
                                "model_path": cfg["path"],
                                "model_type": cfg["type"],
                                "tta_n": n,
                                "tta_temperature": temperature,
                                "gen_batch_size": cfg["gen_batch_size"],
                                "embedding_model": embedding_model,
                                "num_samples": total_samples,
                                "eval_dataset_sha256": eval_dataset_sha256,
                                "aggregations": aggregations,
                            }
                        )
                        mlflow.set_tag("oracle_is_upper_bound", "true")
                        if mlflow_child_metrics:
                            mlflow.log_metrics(mlflow_child_metrics)
                        # Artifacts
                        for fpath in Path(nt_dir).glob("*.csv"):
                            mlflow.log_artifact(
                                str(fpath),
                                artifact_path=f"{cfg['name']}/N{n}_T{temp_str}",
                            )
                        for fpath in Path(nt_dir).glob("*.json"):
                            mlflow.log_artifact(
                                str(fpath),
                                artifact_path=f"{cfg['name']}/N{n}_T{temp_str}",
                            )

            # Unload gen model + embedding model before next model
            del model, emb_model
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # TTA comparison summary CSV
        # ------------------------------------------------------------------
        if all_flat_metrics:
            summary_df = pd.DataFrame(all_flat_metrics)
            summary_cols = [
                "model_name", "tta_n", "tta_temperature", "aggregation",
                "num_samples",
                "abstain_f1", "abstain_precision", "abstain_recall", "abstain_accuracy",
                "embedding_similarity_adjusted_avg", "exact_match_avg",
                "student_abstain_rate", "teacher_abstain_rate",
                "both_abstain_rate", "both_answer_rate",
                "abstain_agreement_rate", "vote_abstain_rate_mean",
            ]
            summary_df = summary_df[[c for c in summary_cols if c in summary_df.columns]]
            summary_path = os.path.join(run_dir, "tta_comparison_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            logging.info(f"\nComparison summary saved: {summary_path}")
            mlflow.log_artifact(summary_path, artifact_path="summary")

            print("\n" + "=" * 120)
            print("TTA COMPARISON SUMMARY")
            print("=" * 120)
            print(summary_df.to_string(index=False))
            print("=" * 120)

    logging.info("\nTTA experiment complete!")
    logging.info(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()


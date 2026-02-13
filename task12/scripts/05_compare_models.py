#!/usr/bin/env python3
"""
Compare Task12 (multi-task) models against single-task models and base models.

Loads evaluation summary JSONs from the eval run directory and creates
side-by-side comparison tables showing:
- Task1 metrics: exact match, token overlap, abstain agreement, embedding similarity
- Task2 metrics: accuracy, macro F1, weighted F1, precision, recall
- Improvement/degradation from base -> single-task LoRA -> multi-task LoRA

Usage:
    python3 scripts/05_compare_models.py --eval_dir outputs/evaluations/eval_run_XXXXXX
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK12_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = TASK12_DIR / "outputs"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_model_family(model_name: str) -> str:
    """Infer model family from name."""
    name_lower = model_name.lower()
    if "gemma" in name_lower:
        if "1b" in name_lower:
            return "Gemma 1B"
        return "Gemma 270M"
    elif "0.5b" in name_lower:
        return "Qwen 0.5B"
    elif "1.5b" in name_lower:
        return "Qwen 1.5B"
    elif "3b" in name_lower:
        return "Qwen 3B"
    elif "7b" in name_lower:
        return "Qwen 7B"
    return "Other"


def get_training_type(model_name: str) -> str:
    """Infer training type from model name."""
    name_lower = model_name.lower()
    if "task12" in name_lower:
        return "Task12 LoRA"
    elif "task1" in name_lower and "lora" in name_lower:
        return "Task1 LoRA"
    elif "task2" in name_lower and "lora" in name_lower:
        return "Task2 LoRA"
    elif "full" in name_lower and ("finetune" in name_lower or "ft" in name_lower):
        return "Full Finetune"
    elif "lora" in name_lower:
        return "LoRA"
    elif "base" in name_lower:
        return "Base"
    return "Unknown"


def load_summaries(eval_dir: Path) -> List[Dict]:
    """Load all model summary JSONs from evaluation directory."""
    summaries = []

    for item in eval_dir.iterdir():
        if item.is_dir():
            for summary_file in item.glob("*_summary.json"):
                try:
                    with open(summary_file) as f:
                        data = json.load(f)
                    data['_source_file'] = str(summary_file)
                    summaries.append(data)
                except Exception as e:
                    logger.warning(f"Could not load {summary_file}: {e}")

    # Also check root level
    for summary_file in eval_dir.glob("*_summary.json"):
        if summary_file.name != "model_comparison_summary.csv":
            try:
                with open(summary_file) as f:
                    data = json.load(f)
                data['_source_file'] = str(summary_file)
                summaries.append(data)
            except Exception as e:
                logger.warning(f"Could not load {summary_file}: {e}")

    return summaries


def build_comparison_table(summaries: List[Dict]) -> pd.DataFrame:
    """Build a comprehensive comparison table from summaries."""
    rows = []

    for s in summaries:
        model_name = s.get('model_name', 'unknown')
        model_type = s.get('model_type', 'unknown')

        row = {
            'model_name': model_name,
            'model_type': model_type,
            'model_family': get_model_family(model_name),
            'training_type': get_training_type(model_name),
        }

        # Task1 metrics
        t1 = s.get('task1', {})
        if t1 and 'error' not in t1:
            row['t1_num_samples'] = t1.get('num_samples', 0)
            row['t1_exact_match'] = t1.get('exact_match_avg')
            row['t1_token_overlap'] = t1.get('token_overlap_avg')
            row['t1_abstain_agreement'] = t1.get('abstain_agreement_rate')
            row['t1_teacher_abstain_rate'] = t1.get('teacher_abstain_rate')
            row['t1_student_abstain_rate'] = t1.get('student_abstain_rate')
            row['t1_embedding_sim'] = t1.get('embedding_similarity_avg')
            row['t1_embedding_sim_adj'] = t1.get('embedding_similarity_adjusted_avg')

        # Task2 metrics
        t2 = s.get('task2', {})
        if t2 and 'error' not in t2:
            row['t2_num_samples'] = t2.get('num_samples', 0)
            row['t2_accuracy'] = t2.get('accuracy')
            row['t2_macro_f1'] = t2.get('macro_f1')
            row['t2_weighted_f1'] = t2.get('weighted_f1')
            row['t2_macro_precision'] = t2.get('macro_precision')
            row['t2_macro_recall'] = t2.get('macro_recall')
            row['t2_valid_predictions'] = t2.get('valid_predictions', 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compute_improvements(df: pd.DataFrame) -> pd.DataFrame:
    """Compute improvement from base -> single-task LoRA -> task12 LoRA per model family."""
    improvement_rows = []

    families = df['model_family'].unique()

    for family in families:
        family_df = df[df['model_family'] == family]

        base = family_df[family_df['training_type'] == 'Base']
        task1_lora = family_df[family_df['training_type'] == 'Task1 LoRA']
        task12_lora = family_df[family_df['training_type'] == 'Task12 LoRA']
        task2_lora = family_df[family_df['training_type'] == 'Task2 LoRA']

        metrics_task1 = ['t1_exact_match', 't1_token_overlap', 't1_abstain_agreement', 't1_embedding_sim_adj']
        metrics_task2 = ['t2_accuracy', 't2_macro_f1', 't2_weighted_f1']

        for metric in metrics_task1 + metrics_task2:
            row = {'model_family': family, 'metric': metric}

            base_val = base[metric].values[0] if len(base) > 0 and metric in base.columns and pd.notna(base[metric].values[0]) else None
            t1_val = task1_lora[metric].values[0] if len(task1_lora) > 0 and metric in task1_lora.columns and pd.notna(task1_lora[metric].values[0]) else None
            t12_val = task12_lora[metric].values[0] if len(task12_lora) > 0 and metric in task12_lora.columns and pd.notna(task12_lora[metric].values[0]) else None
            t2_val = task2_lora[metric].values[0] if len(task2_lora) > 0 and metric in task2_lora.columns and pd.notna(task2_lora[metric].values[0]) else None

            row['base'] = base_val
            row['task1_lora'] = t1_val
            row['task12_lora'] = t12_val
            row['task2_lora'] = t2_val

            # Improvements
            if base_val is not None and t1_val is not None:
                row['t1_lora_vs_base'] = t1_val - base_val
            if base_val is not None and t12_val is not None:
                row['t12_lora_vs_base'] = t12_val - base_val
            if t1_val is not None and t12_val is not None:
                row['t12_vs_t1_lora'] = t12_val - t1_val

            improvement_rows.append(row)

    return pd.DataFrame(improvement_rows)


def main():
    parser = argparse.ArgumentParser(description="Compare Task12 models against baselines")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Path to evaluation run directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as eval_dir)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        logger.error(f"Eval directory not found: {eval_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Task12: Model Comparison")
    logger.info("=" * 60)
    logger.info(f"Input: {eval_dir}")
    logger.info(f"Output: {output_dir}")

    # Load all summaries
    summaries = load_summaries(eval_dir)
    logger.info(f"Loaded {len(summaries)} model summaries")

    if not summaries:
        logger.error("No summaries found!")
        sys.exit(1)

    # Build comparison table
    comparison_df = build_comparison_table(summaries)

    # Sort by family and training type
    family_order = ["Gemma 270M", "Gemma 1B", "Qwen 0.5B", "Qwen 1.5B", "Qwen 3B", "Qwen 7B"]
    type_order = ["Base", "Task1 LoRA", "Task2 LoRA", "Task12 LoRA", "LoRA", "Full Finetune"]

    def sort_key(row):
        f_idx = family_order.index(row['model_family']) if row['model_family'] in family_order else 99
        t_idx = type_order.index(row['training_type']) if row['training_type'] in type_order else 99
        return (f_idx, t_idx)

    comparison_df['_sort'] = comparison_df.apply(sort_key, axis=1)
    comparison_df = comparison_df.sort_values('_sort').drop(columns=['_sort'])

    # Save comparison table
    comp_path = output_dir / "full_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    logger.info(f"Full comparison saved to: {comp_path}")

    # Print summary
    logger.info("\n" + "=" * 120)
    logger.info("FULL MODEL COMPARISON")
    logger.info("=" * 120)

    # Task1 summary
    t1_cols = ['model_name', 'training_type', 't1_exact_match', 't1_token_overlap',
               't1_abstain_agreement', 't1_embedding_sim_adj']
    t1_available = [c for c in t1_cols if c in comparison_df.columns]
    if len(t1_available) > 2:
        logger.info("\n--- Task1 (QA) Metrics ---")
        logger.info(comparison_df[t1_available].to_string(index=False))

    # Task2 summary
    t2_cols = ['model_name', 'training_type', 't2_accuracy', 't2_macro_f1',
               't2_weighted_f1', 't2_macro_precision', 't2_macro_recall']
    t2_available = [c for c in t2_cols if c in comparison_df.columns]
    if len(t2_available) > 2:
        logger.info("\n--- Task2 (Next-Action) Metrics ---")
        logger.info(comparison_df[t2_available].to_string(index=False))

    # Compute improvements
    logger.info("\n--- Improvements ---")
    improvements_df = compute_improvements(comparison_df)
    if len(improvements_df) > 0:
        imp_path = output_dir / "improvements.csv"
        improvements_df.to_csv(imp_path, index=False)
        logger.info(f"Improvements saved to: {imp_path}")

        # Print key improvements
        for family in improvements_df['model_family'].unique():
            fam_df = improvements_df[improvements_df['model_family'] == family]
            logger.info(f"\n  {family}:")
            for _, row in fam_df.iterrows():
                parts = [f"    {row['metric']}:"]
                if pd.notna(row.get('base')):
                    parts.append(f"base={row['base']:.4f}")
                if pd.notna(row.get('task1_lora')):
                    parts.append(f"t1_lora={row['task1_lora']:.4f}")
                if pd.notna(row.get('task12_lora')):
                    parts.append(f"t12_lora={row['task12_lora']:.4f}")
                if pd.notna(row.get('t12_vs_t1_lora')):
                    delta = row['t12_vs_t1_lora']
                    sign = "+" if delta >= 0 else ""
                    parts.append(f"(t12 vs t1: {sign}{delta:.4f})")
                logger.info(" ".join(parts))

    logger.info("\n" + "=" * 60)
    logger.info("Comparison complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

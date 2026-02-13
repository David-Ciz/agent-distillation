#!/usr/bin/env python3
"""
Prepare combined dataset for Task12: Multi-task training.

Merges Task1 (QA) and Task2 (Next-Action Prediction) training data
into a unified training dataset.

Task1: Free-form QA - llm_input is a QA prompt with evidence, llm_output contains REASONING + ANSWER
Task2: Binary classification - llm_input is state description, llm_output is "continue" or "finish"

Input:
    - data/task1_dataset.csv (columns include: llm_input, llm_output, ...)
    - data/task2_train_split.csv (columns include: llm_input, llm_output, ...)

Output:
    - data/combined_train_dataset.csv (columns: llm_input, llm_output, task_type)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Script paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK12_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK12_DIR / "data"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_and_filter(path: str, label: str) -> pd.DataFrame:
    """Load CSV and filter to valid llm_input/llm_output rows."""
    logger.info(f"Loading {label} data: {path}")
    df = pd.read_csv(path, usecols=['llm_input', 'llm_output'])
    before = len(df)

    df = df.dropna(subset=['llm_input', 'llm_output'])
    df = df[df['llm_input'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 5)]
    df = df[df['llm_output'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    df = df.reset_index(drop=True)

    logger.info(f"  {label}: {before} -> {len(df)} after filtering ({before - len(df)} removed)")
    return df


def balance_task2_classes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Balance Task2 binary classes (continue/finish) by undersampling majority."""
    class_counts = df['llm_output'].value_counts()
    logger.info(f"  Task2 class distribution before balancing: {class_counts.to_dict()}")

    min_count = class_counts.min()
    balanced_parts = []
    for cls in class_counts.index:
        cls_df = df[df['llm_output'] == cls]
        if len(cls_df) > min_count:
            balanced_parts.append(cls_df.sample(n=min_count, random_state=seed))
        else:
            balanced_parts.append(cls_df)

    result = pd.concat(balanced_parts, ignore_index=True)
    logger.info(f"  Task2 after class balancing: {len(result)} rows")
    logger.info(f"  Task2 balanced distribution: {result['llm_output'].value_counts().to_dict()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare combined Task12 dataset")
    parser.add_argument("--task1_data", type=str,
                        default=str(DATA_DIR / "task1_dataset.csv"),
                        help="Path to task1 training CSV")
    parser.add_argument("--task2_data", type=str,
                        default=str(DATA_DIR / "task2_train_split.csv"),
                        help="Path to task2 training CSV")
    parser.add_argument("--output", type=str,
                        default=str(DATA_DIR / "combined_train_dataset.csv"),
                        help="Output path for combined dataset")
    parser.add_argument("--balance_strategy", type=str, default="match_smaller",
                        choices=["match_smaller", "all", "ratio"],
                        help="Strategy to balance task1 vs task2 sample counts")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Task1:Task2 ratio when balance_strategy=ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("Task12: Preparing Combined Dataset")
    logger.info("=" * 60)

    # Load and filter both datasets
    df1 = load_and_filter(args.task1_data, "Task1")
    df2 = load_and_filter(args.task2_data, "Task2")

    # Add task_type column
    df1['task_type'] = 'task1'
    df2['task_type'] = 'task2'

    # Balance Task2 classes (continue/finish)
    logger.info("\nBalancing Task2 classes...")
    df2 = balance_task2_classes(df2, seed=args.seed)

    # Balance Task1 vs Task2 proportions
    logger.info(f"\nBalancing tasks (strategy={args.balance_strategy})...")
    logger.info(f"  Before: Task1={len(df1)}, Task2={len(df2)}")

    if args.balance_strategy == "match_smaller":
        target = min(len(df1), len(df2))
        if len(df1) > target:
            df1 = df1.sample(n=target, random_state=args.seed)
        if len(df2) > target:
            df2 = df2.sample(n=target, random_state=args.seed)
        logger.info(f"  Matched to {target} rows per task")

    elif args.balance_strategy == "ratio":
        task2_size = len(df2)
        task1_target = int(task2_size * args.ratio)
        if len(df1) > task1_target:
            df1 = df1.sample(n=task1_target, random_state=args.seed)
        logger.info(f"  Task1: {len(df1)}, Task2: {len(df2)} (ratio={args.ratio})")

    else:  # "all"
        logger.info(f"  Using all data: Task1={len(df1)}, Task2={len(df2)}")

    logger.info(f"  After: Task1={len(df1)}, Task2={len(df2)}")

    # Combine and shuffle
    combined = pd.concat(
        [df1[['llm_input', 'llm_output', 'task_type']],
         df2[['llm_input', 'llm_output', 'task_type']]],
        ignore_index=True
    )
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    combined.to_csv(args.output, index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Combined dataset saved to: {args.output}")
    logger.info(f"Total rows: {len(combined)}")
    logger.info(f"Task distribution: {combined['task_type'].value_counts().to_dict()}")

    # Show Task2 output distribution in combined set
    task2_subset = combined[combined['task_type'] == 'task2']
    if len(task2_subset) > 0:
        logger.info(f"Task2 action distribution: {task2_subset['llm_output'].value_counts().to_dict()}")

    logger.info("Dataset preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

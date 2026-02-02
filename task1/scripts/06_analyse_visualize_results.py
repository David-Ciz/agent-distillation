#!/usr/bin/env python3
"""
Analyze and visualize model evaluation results.

This script:
- Loads detailed_results.csv files from multiple models
- Calculates comprehensive metrics (abstain F1, accuracy, precision, recall, etc.)
- Generates visualizations comparing models grouped by family/size
- Outputs metrics summary and publication-ready plots

Usage:
    python 06_analyse_visualize_results.py --results_dirs dir1 dir2 ... --output_dir plots/
    python 06_analyse_visualize_results.py --results_files file1.csv file2.csv ... --model_names name1 name2
"""

import argparse
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, bootstrap

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Model groupings for visualization
MODEL_GROUPS = {
    "Gemma 270M": ["gemma-3-270m-it-base", "gemma-3-270m-it-lora"],
    "Qwen 2.5 0.5B": ["Qwen2.5-0.5B-Instruct-base", "Qwen2.5-0.5B-Instruct-lora", "Qwen2.5-0.5B-Instruct-full-finetune"],
    "Qwen 2.5 1.5B": ["Qwen2.5-1.5B-Instruct-base", "Qwen2.5-1.5B-Instruct-lora"],
    "Qwen 2.5 3B": ["Qwen2.5-3B-Instruct-base", "Qwen2.5-3B-Instruct-lora", "Qwen2.5-3B-Instruct-full-finetune"],
    "Qwen 2.5 7B": ["Qwen2.5-7B-Instruct-base", "Qwen2.5-7B-Instruct-lora"],
}

# Training type colors - clearly distinguishable
TRAIN_TYPE_COLORS = {
    "Base": "#2ECC71",       # Green
    "LoRA": "#E74C3C",        # Red
    "Full Finetune": "#9B59B6",  # Purple
}

# Answer state colors
STATE_COLORS = {
    "both_abstain": "#2ecc71",
    "both_answer": "#3498db",
    "teacher_answer_student_abstain": "#e74c3c",
    "teacher_abstain_student_answer": "#f39c12",
}

# Teacher colors for per-teacher analysis
TEACHER_COLORS = {
    "gpt-4o": "#1f77b4",
    "deepseek-v3": "#ff7f0e",
    "mistral-large": "#2ca02c",
    "claude": "#d62728",
    "gemini": "#9467bd",
}


def format_model_name(name: str) -> str:
    """Convert model name to readable format without underscores."""
    name = name.replace("_", " ").replace("-", " ")
    # Clean up common patterns
    name = name.replace("Instruct", "").replace("instruct", "")
    name = name.replace("  ", " ").strip()
    return name


def get_train_type(model_name: str) -> str:
    """Extract training type from model name."""
    name_lower = model_name.lower()
    if "full-finetune" in name_lower or "full_finetune" in name_lower:
        return "Full Finetune"
    elif "lora" in name_lower:
        return "LoRA"
    else:
        return "Base"


def get_model_group(model_name: str) -> str:
    """Get the model group/family for a model."""
    for group_name, models in MODEL_GROUPS.items():
        if model_name in models:
            return group_name
    # Fallback: try to infer from name
    name_lower = model_name.lower()
    if "gemma" in name_lower:
        return "Gemma 270M"
    elif "0.5b" in name_lower:
        return "Qwen 2.5 0.5B"
    elif "1.5b" in name_lower:
        return "Qwen 2.5 1.5B"
    elif "3b" in name_lower:
        return "Qwen 2.5 3B"
    elif "7b" in name_lower:
        return "Qwen 2.5 7B"
    return "Other"


def load_results(results_path: str, model_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Load results from a CSV file or directory."""
    path = Path(results_path)
    
    if path.is_dir():
        # Find the detailed_results.csv file
        csv_files = list(path.glob("*_detailed_results.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No *_detailed_results.csv found in {path}")
        csv_path = csv_files[0]
        if model_name is None:
            model_name = csv_path.stem.replace("_detailed_results", "")
    else:
        csv_path = path
        if model_name is None:
            model_name = path.stem.replace("_detailed_results", "")
    
    df = pd.read_csv(csv_path)
    return df, model_name


def calculate_abstain_metrics(df: pd.DataFrame) -> Dict:
    """Calculate abstain-related metrics treating teacher as ground truth."""
    # Teacher abstain = positive class (1), Teacher answer = negative class (0)
    y_true = df['teacher_abstain'].astype(int).values
    y_pred = df['student_abstain'].astype(int).values
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For abstain detection (teacher abstain = positive)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    
    return {
        'abstain_accuracy': accuracy,
        'abstain_precision': precision,
        'abstain_recall': recall,
        'abstain_f1': f1,
        'abstain_specificity': specificity,
        'true_positives': int(tp),  # Both abstain
        'true_negatives': int(tn),  # Both answer
        'false_positives': int(fp),  # Teacher answers, student abstains
        'false_negatives': int(fn),  # Teacher abstains, student answers
    }


def calculate_similarity_metrics(df: pd.DataFrame) -> Dict:
    """Calculate similarity metrics between teacher and student answers."""
    metrics = {}
    
    # Embedding similarity
    if 'embedding_similarity' in df.columns:
        valid_sims = df['embedding_similarity'].dropna()
        if len(valid_sims) > 0:
            metrics['embedding_similarity_mean'] = valid_sims.mean()
            metrics['embedding_similarity_std'] = valid_sims.std()
            metrics['embedding_similarity_median'] = valid_sims.median()
    
    # Adjusted embedding similarity
    if 'embedding_similarity_adjusted' in df.columns:
        valid_sims = df['embedding_similarity_adjusted'].dropna()
        if len(valid_sims) > 0:
            metrics['embedding_similarity_adjusted_mean'] = valid_sims.mean()
            metrics['embedding_similarity_adjusted_std'] = valid_sims.std()
    
    # Token overlap
    if 'token_overlap' in df.columns:
        valid_overlaps = df['token_overlap'].dropna()
        if len(valid_overlaps) > 0:
            metrics['token_overlap_mean'] = valid_overlaps.mean()
            metrics['token_overlap_std'] = valid_overlaps.std()
    
    # Exact match
    if 'exact_match_score' in df.columns:
        metrics['exact_match_rate'] = df['exact_match_score'].mean()
    
    return metrics


def calculate_state_distribution(df: pd.DataFrame) -> Dict:
    """Calculate answer state distribution."""
    if 'answer_state' in df.columns:
        state_counts = df['answer_state'].value_counts().to_dict()
    else:
        # Calculate from abstain columns
        state_counts = {
            'both_abstain': ((df['teacher_abstain']) & (df['student_abstain'])).sum(),
            'both_answer': ((~df['teacher_abstain']) & (~df['student_abstain'])).sum(),
            'teacher_answer_student_abstain': ((~df['teacher_abstain']) & (df['student_abstain'])).sum(),
            'teacher_abstain_student_answer': ((df['teacher_abstain']) & (~df['student_abstain'])).sum(),
        }
    
    total = len(df)
    state_percentages = {k: v / total * 100 for k, v in state_counts.items()}
    
    return {
        'state_counts': state_counts,
        'state_percentages': state_percentages,
        'teacher_abstain_rate': df['teacher_abstain'].mean(),
        'student_abstain_rate': df['student_abstain'].mean(),
    }


def calculate_all_metrics(df: pd.DataFrame, model_name: str) -> Dict:
    """Calculate all metrics for a model."""
    metrics = {
        'model_name': model_name,
        'model_group': get_model_group(model_name),
        'train_type': get_train_type(model_name),
        'num_samples': len(df),
    }
    
    # Abstain metrics
    metrics.update(calculate_abstain_metrics(df))
    
    # Similarity metrics
    metrics.update(calculate_similarity_metrics(df))
    
    # State distribution
    state_info = calculate_state_distribution(df)
    metrics['teacher_abstain_rate'] = state_info['teacher_abstain_rate']
    metrics['student_abstain_rate'] = state_info['student_abstain_rate']
    metrics['state_counts'] = state_info['state_counts']
    metrics['state_percentages'] = state_info['state_percentages']
    
    return metrics


def calculate_per_teacher_metrics(df: pd.DataFrame) -> Dict:
    """Calculate metrics per unique teacher (teacher_id column)."""
    if 'teacher_id' not in df.columns:
        return {}
    
    per_teacher = {}
    teachers = df['teacher_id'].unique()
    
    for teacher in teachers:
        teacher_df = df[df['teacher_id'] == teacher]
        teacher_metrics = {
            'count': len(teacher_df),
            'teacher_abstain_rate': teacher_df['teacher_abstain'].mean(),
            'teacher_abstain_count': teacher_df['teacher_abstain'].sum(),
            'student_abstain_rate': teacher_df['student_abstain'].mean(),
            'student_abstain_count': teacher_df['student_abstain'].sum(),
        }
        
        # Abstain agreement metrics for this teacher
        y_true = teacher_df['teacher_abstain'].astype(int).values
        y_pred = teacher_df['student_abstain'].astype(int).values
        
        teacher_metrics['abstain_accuracy'] = accuracy_score(y_true, y_pred)
        teacher_metrics['abstain_f1'] = f1_score(y_true, y_pred, zero_division=0)
        teacher_metrics['abstain_precision'] = precision_score(y_true, y_pred, zero_division=0)
        teacher_metrics['abstain_recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        # Similarity metrics for this teacher
        if 'embedding_similarity_adjusted' in teacher_df.columns:
            valid_sims = teacher_df['embedding_similarity_adjusted'].dropna()
            if len(valid_sims) > 0:
                teacher_metrics['embedding_similarity_mean'] = valid_sims.mean()
                teacher_metrics['embedding_similarity_std'] = valid_sims.std()
        
        if 'token_overlap' in teacher_df.columns:
            valid_overlaps = teacher_df['token_overlap'].dropna()
            if len(valid_overlaps) > 0:
                teacher_metrics['token_overlap_mean'] = valid_overlaps.mean()
                teacher_metrics['token_overlap_std'] = valid_overlaps.std()
        
        if 'exact_match_score' in teacher_df.columns:
            teacher_metrics['exact_match_rate'] = teacher_df['exact_match_score'].mean()
        
        per_teacher[teacher] = teacher_metrics
    
    return per_teacher


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate mean and 95% confidence interval using bootstrap."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    
    mean = np.mean(data)
    n = len(data)
    
    if n < 2:
        return mean, mean, mean
    
    # Use bootstrap for confidence interval
    try:
        rng = np.random.default_rng(42)
        res = bootstrap((data,), np.mean, confidence_level=confidence, random_state=rng, n_resamples=1000)
        ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
    except Exception:
        # Fallback to t-distribution based CI
        se = stats.sem(data)
        ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_low, ci_high = mean - ci, mean + ci
    
    return mean, ci_low, ci_high


def calculate_metrics_with_ci(df: pd.DataFrame, model_name: str) -> Dict:
    """Calculate metrics with 95% confidence intervals."""
    metrics = calculate_all_metrics(df, model_name)
    
    # Add confidence intervals for key metrics
    # Abstain accuracy CI (using bootstrap on per-sample correctness)
    y_true = df['teacher_abstain'].astype(int).values
    y_pred = df['student_abstain'].astype(int).values
    correct = (y_true == y_pred).astype(float)
    
    mean, ci_low, ci_high = calculate_confidence_interval(correct)
    metrics['abstain_accuracy_ci_low'] = ci_low
    metrics['abstain_accuracy_ci_high'] = ci_high
    
    # Embedding similarity CI
    if 'embedding_similarity_adjusted' in df.columns:
        valid_sims = df['embedding_similarity_adjusted'].dropna().values
        if len(valid_sims) > 0:
            mean, ci_low, ci_high = calculate_confidence_interval(valid_sims)
            metrics['embedding_similarity_ci_low'] = ci_low
            metrics['embedding_similarity_ci_high'] = ci_high
    
    # Token overlap CI
    if 'token_overlap' in df.columns:
        valid_overlaps = df['token_overlap'].dropna().values
        if len(valid_overlaps) > 0:
            mean, ci_low, ci_high = calculate_confidence_interval(valid_overlaps)
            metrics['token_overlap_ci_low'] = ci_low
            metrics['token_overlap_ci_high'] = ci_high
    
    # Per-teacher metrics
    metrics['per_teacher'] = calculate_per_teacher_metrics(df)
    
    return metrics


def run_statistical_tests(all_data: Dict[str, pd.DataFrame], all_metrics: List[Dict]) -> Dict:
    """Run t-tests for key comparisons."""
    results = {
        'base_vs_lora': [],
        'lora_vs_full_ft': [],
        'cross_size': [],
        'gemma_vs_qwen': [],
    }
    
    # Helper to get metric values for a model
    def get_metric_values(model_name: str, metric_col: str) -> np.ndarray:
        if model_name not in all_data:
            return np.array([])
        df = all_data[model_name]
        if metric_col == 'abstain_correct':
            y_true = df['teacher_abstain'].astype(int).values
            y_pred = df['student_abstain'].astype(int).values
            return (y_true == y_pred).astype(float)
        elif metric_col in df.columns:
            return df[metric_col].dropna().values
        return np.array([])
    
    # Group models by model_group
    group_models = defaultdict(list)
    for m in all_metrics:
        group_models[m['model_group']].append(m)
    
    metrics_to_test = ['abstain_correct', 'embedding_similarity_adjusted', 'token_overlap']
    metric_labels = ['Abstain Accuracy', 'Embedding Similarity', 'Token Overlap']
    
    # 1. Base vs LoRA for each model size
    for group_name, models in group_models.items():
        base_model = next((m for m in models if m['train_type'] == 'Base'), None)
        lora_model = next((m for m in models if m['train_type'] == 'LoRA'), None)
        
        if base_model and lora_model:
            for metric_col, metric_label in zip(metrics_to_test, metric_labels):
                base_vals = get_metric_values(base_model['model_name'], metric_col)
                lora_vals = get_metric_values(lora_model['model_name'], metric_col)
                
                if len(base_vals) > 0 and len(lora_vals) > 0:
                    # Use independent t-test (samples may differ)
                    t_stat, p_value = ttest_ind(base_vals, lora_vals)
                    results['base_vs_lora'].append({
                        'group': group_name,
                        'metric': metric_label,
                        'base_mean': np.mean(base_vals),
                        'lora_mean': np.mean(lora_vals),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                    })
    
    # 2. LoRA vs Full Finetune for 0.5B and 3B
    for group_name in ['Qwen 2.5 0.5B', 'Qwen 2.5 3B']:
        if group_name not in group_models:
            continue
        models = group_models[group_name]
        lora_model = next((m for m in models if m['train_type'] == 'LoRA'), None)
        full_ft_model = next((m for m in models if m['train_type'] == 'Full Finetune'), None)
        
        if lora_model and full_ft_model:
            for metric_col, metric_label in zip(metrics_to_test, metric_labels):
                lora_vals = get_metric_values(lora_model['model_name'], metric_col)
                full_ft_vals = get_metric_values(full_ft_model['model_name'], metric_col)
                
                if len(lora_vals) > 0 and len(full_ft_vals) > 0:
                    t_stat, p_value = ttest_ind(lora_vals, full_ft_vals)
                    results['lora_vs_full_ft'].append({
                        'group': group_name,
                        'metric': metric_label,
                        'lora_mean': np.mean(lora_vals),
                        'full_ft_mean': np.mean(full_ft_vals),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                    })
    
    # 3. Cross-size comparisons (0.5B LoRA vs 1.5B Base)
    cross_size_pairs = [
        ('Qwen 2.5 0.5B', 'LoRA', 'Qwen 2.5 1.5B', 'Base'),
        ('Qwen 2.5 1.5B', 'LoRA', 'Qwen 2.5 3B', 'Base'),
        ('Qwen 2.5 0.5B', 'Full Finetune', 'Qwen 2.5 1.5B', 'Base'),
    ]
    
    for group1, type1, group2, type2 in cross_size_pairs:
        if group1 not in group_models or group2 not in group_models:
            continue
        
        model1 = next((m for m in group_models[group1] if m['train_type'] == type1), None)
        model2 = next((m for m in group_models[group2] if m['train_type'] == type2), None)
        
        if model1 and model2:
            for metric_col, metric_label in zip(metrics_to_test, metric_labels):
                vals1 = get_metric_values(model1['model_name'], metric_col)
                vals2 = get_metric_values(model2['model_name'], metric_col)
                
                if len(vals1) > 0 and len(vals2) > 0:
                    t_stat, p_value = ttest_ind(vals1, vals2)
                    results['cross_size'].append({
                        'comparison': f"{group1} {type1} vs {group2} {type2}",
                        'metric': metric_label,
                        'model1_mean': np.mean(vals1),
                        'model2_mean': np.mean(vals2),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                    })
    
    # 4. Gemma vs Qwen (similar sizes - Gemma 270M vs Qwen 0.5B)
    gemma_models = [m for m in all_metrics if 'gemma' in m['model_name'].lower()]
    qwen_05b_models = [m for m in all_metrics if m['model_group'] == 'Qwen 2.5 0.5B']
    
    for train_type in ['Base', 'LoRA']:
        gemma_model = next((m for m in gemma_models if m['train_type'] == train_type), None)
        qwen_model = next((m for m in qwen_05b_models if m['train_type'] == train_type), None)
        
        if gemma_model and qwen_model:
            for metric_col, metric_label in zip(metrics_to_test, metric_labels):
                gemma_vals = get_metric_values(gemma_model['model_name'], metric_col)
                qwen_vals = get_metric_values(qwen_model['model_name'], metric_col)
                
                if len(gemma_vals) > 0 and len(qwen_vals) > 0:
                    t_stat, p_value = ttest_ind(gemma_vals, qwen_vals)
                    results['gemma_vs_qwen'].append({
                        'comparison': f"Gemma 270M {train_type} vs Qwen 0.5B {train_type}",
                        'metric': metric_label,
                        'gemma_mean': np.mean(gemma_vals),
                        'qwen_mean': np.mean(qwen_vals),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                    })
    
    return results


def calculate_ensemble_vs_individual_abstain(all_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Calculate ensemble (majority vote) abstain rate vs individual teacher abstain rates.
    This helps verify if the ~70% abstain rate is ensemble or per-teacher.
    """
    # Get any dataframe to analyze teachers
    sample_df = next(iter(all_data.values()))
    
    if 'teacher_id' not in sample_df.columns:
        return {}
    
    results = {
        'per_teacher_abstain_rates': {},
        'ensemble_abstain_rate': None,
        'total_samples': len(sample_df),
    }
    
    # Per-teacher abstain rates
    teachers = sample_df['teacher_id'].unique()
    for teacher in teachers:
        teacher_df = sample_df[sample_df['teacher_id'] == teacher]
        results['per_teacher_abstain_rates'][teacher] = {
            'abstain_rate': teacher_df['teacher_abstain'].mean(),
            'abstain_count': int(teacher_df['teacher_abstain'].sum()),
            'total_count': len(teacher_df),
        }
    
    # Overall teacher abstain rate (this is what we see in the data)
    results['overall_teacher_abstain_rate'] = sample_df['teacher_abstain'].mean()
    
    return results


def plot_answer_state_distribution(all_metrics: List[Dict], output_dir: str, teacher_abstain_rate: float):
    """Plot stacked bar chart of answer state distribution grouped by model family."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sort models by group then by train type
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    model_names = [format_model_name(m['model_name']) for m in sorted_metrics]
    x = np.arange(len(model_names))
    width = 0.7
    
    # Stack the states
    states = ['both_abstain', 'both_answer', 'teacher_answer_student_abstain', 'teacher_abstain_student_answer']
    state_labels = ['Both Abstain', 'Both Answer', 'Teacher Answers, Student Abstains', 'Teacher Abstains, Student Answers']
    
    bottom = np.zeros(len(sorted_metrics))
    
    for state, label in zip(states, state_labels):
        values = [m['state_percentages'].get(state, 0) for m in sorted_metrics]
        bars = ax.bar(x, values, width, bottom=bottom, label=label, color=STATE_COLORS[state])
        bottom += values
    
    # Add teacher abstain baseline
    ax.axhline(y=teacher_abstain_rate * 100, color='purple', linestyle='--', linewidth=2, 
               label=f'Teacher Abstain Rate ({teacher_abstain_rate*100:.1f}%)')
    
    # Add group separators
    current_group = None
    for i, m in enumerate(sorted_metrics):
        if current_group is not None and m['model_group'] != current_group:
            ax.axvline(x=i - 0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        current_group = m['model_group']
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 105)
    
    # Legend as one line under the title
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='upper center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.97))
    
    # Title above legend
    fig.suptitle('Answer State Distribution by Model\n(Grouped by Model Family)', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, 'answer_state_distribution.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'answer_state_distribution.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: answer_state_distribution.png")


def plot_violin_similarity_grouped(all_data: Dict[str, pd.DataFrame], all_metrics: List[Dict], 
                                    output_dir: str, metric_col: str = 'embedding_similarity_adjusted',
                                    title: str = 'Embedding Similarity Distribution'):
    """Plot violin plots grouped by model family in a 2x3 grid."""
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, group_name in enumerate(group_order):
        ax = axes[idx]
        
        # Get models in this group
        group_models = [m for m in all_metrics if m['model_group'] == group_name]
        if not group_models:
            ax.set_visible(False)
            continue
        
        # Sort by train type
        type_order = ["Base", "LoRA", "Full Finetune"]
        group_models = sorted(group_models, key=lambda x: type_order.index(x['train_type']) if x['train_type'] in type_order else 99)
        
        # Prepare data for violin plot
        plot_data = []
        labels = []
        colors = []
        
        for m in group_models:
            model_name = m['model_name']
            if model_name in all_data:
                df = all_data[model_name]
                if metric_col in df.columns:
                    values = df[metric_col].dropna().values
                    plot_data.append(values)
                    labels.append(m['train_type'])
                    colors.append(TRAIN_TYPE_COLORS.get(m['train_type'], '#888888'))
        
        if plot_data:
            # Use violin plot without internal stats
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                                  showmeans=False, showmedians=False, showextrema=False)
            
            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
            
            # Add boxplots on top - transparent with only outlines
            bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.08,
                           patch_artist=True, showfliers=False)
            for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
                box.set_facecolor('none')  # Transparent fill
                box.set_edgecolor('black')
                box.set_linewidth(1.2)
                median.set_color('red')
                median.set_linewidth(1.5)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylabel('Similarity Score', fontsize=10)
            ax.set_title(group_name, fontsize=12, fontweight='bold')
            ax.set_ylim(-0.1, 1.1)
    
    # Hide unused subplot and use it for legend
    axes[5].set_visible(False)
    
    # Add legend in the position of the hidden subplot (bottom right)
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.7) for t, c in TRAIN_TYPE_COLORS.items()]
    axes[5].legend(handles=legend_patches, loc='center', fontsize=12, frameon=True)
    axes[5].set_visible(True)
    axes[5].axis('off')
    
    fig.suptitle(f'{title}\n(Grouped by Model Family)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = metric_col.replace('_', '-')
    plt.savefig(os.path.join(output_dir, f'violin_{filename}_grouped.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'violin_{filename}_grouped.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: violin_{filename}_grouped.png")


def plot_violin_all_models(all_data: Dict[str, pd.DataFrame], all_metrics: List[Dict],
                           output_dir: str, metric_col: str = 'embedding_similarity_adjusted',
                           title: str = 'Embedding Similarity Distribution'):
    """Plot violin plots for all models in a single figure."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sort models by group then by train type
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    plot_data = []
    labels = []
    colors = []
    
    for m in sorted_metrics:
        model_name = m['model_name']
        if model_name in all_data:
            df = all_data[model_name]
            if metric_col in df.columns:
                values = df[metric_col].dropna().values
                plot_data.append(values)
                labels.append(format_model_name(model_name))
                colors.append(TRAIN_TYPE_COLORS.get(m['train_type'], '#888888'))
    
    if plot_data:
        # Use violin plot without internal stats
        parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                              showmeans=False, showmedians=False, showextrema=False)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        
        # Add boxplots on top - transparent with only outlines
        bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.08,
                       patch_artist=True, showfliers=False)
        for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
            box.set_facecolor('none')  # Transparent fill
            box.set_edgecolor('black')
            box.set_linewidth(1.2)
            median.set_color('red')
            median.set_linewidth(1.5)
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1)
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
    
    # Add legend outside plot on top right
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.7) for t, c in TRAIN_TYPE_COLORS.items()]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, borderaxespad=0)
    
    plt.tight_layout()
    filename = metric_col.replace('_', '-')
    plt.savefig(os.path.join(output_dir, f'violin_{filename}_all.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'violin_{filename}_all.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: violin_{filename}_all.png")


def plot_abstain_metrics_comparison(all_metrics: List[Dict], output_dir: str, teacher_abstain_rate: float):
    """Plot abstain metrics (F1, Precision, Recall, Accuracy) comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics_to_plot = [
        ('abstain_accuracy', 'Abstain Accuracy', axes[0, 0]),
        ('abstain_f1', 'Abstain F1 Score', axes[0, 1]),
        ('abstain_precision', 'Abstain Precision', axes[1, 0]),
        ('abstain_recall', 'Abstain Recall', axes[1, 1]),
    ]
    
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    model_names = [format_model_name(m['model_name']) for m in sorted_metrics]
    colors = [TRAIN_TYPE_COLORS.get(m['train_type'], '#888888') for m in sorted_metrics]
    x = np.arange(len(model_names))
    
    for metric_key, metric_title, ax in metrics_to_plot:
        values = [m.get(metric_key, 0) for m in sorted_metrics]
        bars = ax.bar(x, values, color=colors, alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.8) for t, c in TRAIN_TYPE_COLORS.items()]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle('Abstain Detection Metrics\n(Teacher Abstain as Ground Truth)', fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'abstain_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'abstain_metrics_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: abstain_metrics_comparison.png")


def plot_abstain_rates_comparison(all_metrics: List[Dict], output_dir: str, teacher_abstain_rate: float):
    """Plot student vs teacher abstain rates."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    model_names = [format_model_name(m['model_name']) for m in sorted_metrics]
    student_rates = [m['student_abstain_rate'] * 100 for m in sorted_metrics]
    colors = [TRAIN_TYPE_COLORS.get(m['train_type'], '#888888') for m in sorted_metrics]
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, student_rates, color=colors, alpha=0.8)
    
    # Teacher baseline
    ax.axhline(y=teacher_abstain_rate * 100, color='purple', linestyle='--', linewidth=2,
               label=f'Teacher Abstain Rate ({teacher_abstain_rate*100:.1f}%)')
    
    ax.set_ylabel('Abstain Rate (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    
    # Add value labels
    for bar, val in zip(bars, student_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Legend as one line under the title
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.8) for t, c in TRAIN_TYPE_COLORS.items()]
    legend_patches.append(plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=2, label='Teacher Baseline'))
    fig.legend(handles=legend_patches, loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.98))
    
    # Title above legend
    fig.suptitle('Student Abstain Rate vs Teacher Baseline', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'abstain_rates_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'abstain_rates_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: abstain_rates_comparison.png")


def plot_similarity_metrics_comparison(all_metrics: List[Dict], output_dir: str):
    """Plot similarity metrics comparison (embedding similarity, token overlap, exact match)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics_to_plot = [
        ('embedding_similarity_adjusted_mean', 'Embedding Similarity (Adjusted)', axes[0]),
        ('token_overlap_mean', 'Token Overlap', axes[1]),
        ('exact_match_rate', 'Exact Match Rate', axes[2]),
    ]
    
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    model_names = [format_model_name(m['model_name']) for m in sorted_metrics]
    colors = [TRAIN_TYPE_COLORS.get(m['train_type'], '#888888') for m in sorted_metrics]
    x = np.arange(len(model_names))
    
    for metric_key, metric_title, ax in metrics_to_plot:
        values = [m.get(metric_key, 0) for m in sorted_metrics]
        bars = ax.bar(x, values, color=colors, alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Legend
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.8) for t, c in TRAIN_TYPE_COLORS.items()]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle('Teacher-Student Answer Similarity Metrics', fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'similarity_metrics_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: similarity_metrics_comparison.png")


def plot_heatmap_metrics(all_metrics: List[Dict], output_dir: str):
    """Plot heatmap of key metrics across models."""
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    # Select key metrics
    metric_keys = [
        'abstain_accuracy', 'abstain_f1', 'abstain_precision', 'abstain_recall',
        'embedding_similarity_adjusted_mean', 'token_overlap_mean', 'exact_match_rate'
    ]
    metric_labels = [
        'Abstain Accuracy', 'Abstain F1', 'Abstain Precision', 'Abstain Recall',
        'Embedding Similarity', 'Token Overlap', 'Exact Match'
    ]
    
    model_names = [format_model_name(m['model_name']) for m in sorted_metrics]
    
    # Build data matrix
    data = []
    for m in sorted_metrics:
        row = [m.get(k, 0) for k in metric_keys]
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(model_names, fontsize=9)
    
    # Remove gridlines
    ax.grid(False)
    
    # Add text annotations - bolder and larger
    for i in range(len(model_names)):
        for j in range(len(metric_labels)):
            text = ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', 
                          color='black' if 0.3 < data[i, j] < 0.7 else 'white', 
                          fontsize=10, fontweight='bold')
    
    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Score', rotation=-90, va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_heatmap.png")


def plot_training_effect_by_size(all_metrics: List[Dict], output_dir: str):
    """Plot the effect of training (Base vs LoRA vs Full Finetune) by model size."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics_to_plot = [
        ('abstain_f1', 'Abstain F1 Score'),
        ('embedding_similarity_adjusted_mean', 'Embedding Similarity'),
        ('exact_match_rate', 'Exact Match Rate'),
        ('abstain_accuracy', 'Abstain Accuracy'),
    ]
    
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    for (metric_key, metric_title), ax in zip(metrics_to_plot, axes.flatten()):
        # Group data by model group and train type
        group_data = defaultdict(dict)
        for m in all_metrics:
            group_data[m['model_group']][m['train_type']] = m.get(metric_key, 0)
        
        x = np.arange(len(group_order))
        width = 0.25
        
        for i, train_type in enumerate(type_order):
            values = [group_data.get(g, {}).get(train_type, 0) for g in group_order]
            # Only plot if there are non-zero values
            if any(v > 0 for v in values):
                offset = (i - 1) * width
                bars = ax.bar(x + offset, values, width, label=train_type, 
                             color=TRAIN_TYPE_COLORS.get(train_type, '#888888'), alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_order, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
    
    fig.suptitle('Training Effect by Model Size', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_effect_by_size.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'training_effect_by_size.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_effect_by_size.png")


def plot_per_teacher_abstain_rates(all_data: Dict[str, pd.DataFrame], output_dir: str):
    """Plot per-teacher abstain rates to show individual teacher behavior."""
    # Get any dataframe to analyze teachers
    sample_df = next(iter(all_data.values()))
    
    if 'teacher_id' not in sample_df.columns:
        print("  Skipping per-teacher plots: no teacher_id column found")
        return
    
    teachers = sorted(sample_df['teacher_id'].unique())
    
    # Calculate per-teacher abstain rates
    teacher_abstain_rates = {}
    teacher_counts = {}
    for teacher in teachers:
        teacher_df = sample_df[sample_df['teacher_id'] == teacher]
        teacher_abstain_rates[teacher] = teacher_df['teacher_abstain'].mean() * 100
        teacher_counts[teacher] = len(teacher_df)
    
    # Overall abstain rate
    overall_rate = sample_df['teacher_abstain'].mean() * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(teachers))
    colors = [TEACHER_COLORS.get(t, '#888888') for t in teachers]
    
    bars = ax.bar(x, [teacher_abstain_rates[t] for t in teachers], color=colors, alpha=0.8)
    
    # Add overall baseline
    ax.axhline(y=overall_rate, color='red', linestyle='--', linewidth=2,
               label=f'Overall Abstain Rate ({overall_rate:.1f}%)')
    
    # Add value labels
    for bar, teacher in zip(bars, teachers):
        height = bar.get_height()
        count = teacher_counts[teacher]
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
               f'{height:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Abstain Rate (%)', fontsize=12)
    ax.set_xlabel('Teacher Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(teachers, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_title('Per-Teacher Abstain Rates\n(Individual Teacher Behavior)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_teacher_abstain_rates.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'per_teacher_abstain_rates.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_teacher_abstain_rates.png")


def plot_per_teacher_student_similarity(all_data: Dict[str, pd.DataFrame], all_metrics: List[Dict], output_dir: str):
    """Plot student similarity to each teacher across different models."""
    # Get teachers from any dataframe
    sample_df = next(iter(all_data.values()))
    
    if 'teacher_id' not in sample_df.columns:
        return
    
    teachers = sorted(sample_df['teacher_id'].unique())
    
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    # Create figure with subplots for each teacher
    n_teachers = len(teachers)
    fig, axes = plt.subplots(1, n_teachers, figsize=(6 * n_teachers, 6), sharey=True)
    if n_teachers == 1:
        axes = [axes]
    
    for ax, teacher in zip(axes, teachers):
        model_names = []
        similarities = []
        colors = []
        
        for m in sorted_metrics:
            model_name = m['model_name']
            if model_name in all_data:
                df = all_data[model_name]
                teacher_df = df[df['teacher_id'] == teacher]
                
                if 'embedding_similarity_adjusted' in teacher_df.columns:
                    valid_sims = teacher_df['embedding_similarity_adjusted'].dropna()
                    if len(valid_sims) > 0:
                        model_names.append(format_model_name(model_name))
                        similarities.append(valid_sims.mean())
                        colors.append(TRAIN_TYPE_COLORS.get(m['train_type'], '#888888'))
        
        if model_names:
            x = np.arange(len(model_names))
            bars = ax.bar(x, similarities, color=colors, alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            ax.set_title(f'Teacher: {teacher}', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)
            
            if ax == axes[0]:
                ax.set_ylabel('Embedding Similarity', fontsize=11)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.8) for t, c in TRAIN_TYPE_COLORS.items()]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle('Student-Teacher Similarity by Teacher Model', fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_teacher_student_similarity.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'per_teacher_student_similarity.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_teacher_student_similarity.png")


def plot_per_teacher_abstain_agreement(all_data: Dict[str, pd.DataFrame], all_metrics: List[Dict], output_dir: str):
    """Plot student abstain agreement with each teacher across different models."""
    sample_df = next(iter(all_data.values()))
    
    if 'teacher_id' not in sample_df.columns:
        return
    
    teachers = sorted(sample_df['teacher_id'].unique())
    
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    # Create heatmap data: models x teachers
    model_names = [format_model_name(m['model_name']) for m in sorted_metrics]
    
    # Calculate abstain F1 for each model-teacher pair
    data = np.zeros((len(sorted_metrics), len(teachers)))
    
    for i, m in enumerate(sorted_metrics):
        model_name = m['model_name']
        if model_name in all_data:
            df = all_data[model_name]
            for j, teacher in enumerate(teachers):
                teacher_df = df[df['teacher_id'] == teacher]
                if len(teacher_df) > 0:
                    y_true = teacher_df['teacher_abstain'].astype(int).values
                    y_pred = teacher_df['student_abstain'].astype(int).values
                    data[i, j] = f1_score(y_true, y_pred, zero_division=0)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(teachers)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(teachers, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(model_names, fontsize=9)
    
    ax.grid(False)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(teachers)):
            text = ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                          color='black' if 0.3 < data[i, j] < 0.7 else 'white',
                          fontsize=9, fontweight='bold')
    
    ax.set_title('Abstain F1 Score by Model and Teacher', fontsize=14, fontweight='bold')
    ax.set_xlabel('Teacher Model', fontsize=12)
    ax.set_ylabel('Student Model', fontsize=12)
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Abstain F1 Score', rotation=-90, va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_teacher_abstain_agreement_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'per_teacher_abstain_agreement_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_teacher_abstain_agreement_heatmap.png")


def plot_per_teacher_metrics_grouped(all_data: Dict[str, pd.DataFrame], all_metrics: List[Dict], output_dir: str):
    """Plot per-teacher metrics in a grouped bar chart format."""
    sample_df = next(iter(all_data.values()))
    
    if 'teacher_id' not in sample_df.columns:
        return
    
    teachers = sorted(sample_df['teacher_id'].unique())
    
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    # Create 2x2 subplot for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics_to_plot = [
        ('abstain_f1', 'Abstain F1 Score', axes[0, 0]),
        ('abstain_accuracy', 'Abstain Accuracy', axes[0, 1]),
        ('embedding_similarity', 'Embedding Similarity', axes[1, 0]),
        ('token_overlap', 'Token Overlap', axes[1, 1]),
    ]
    
    for metric_key, metric_title, ax in metrics_to_plot:
        x = np.arange(len(teachers))
        width = 0.8 / len(sorted_metrics)
        
        for i, m in enumerate(sorted_metrics):
            model_name = m['model_name']
            if model_name not in all_data:
                continue
            
            df = all_data[model_name]
            values = []
            
            for teacher in teachers:
                teacher_df = df[df['teacher_id'] == teacher]
                if len(teacher_df) == 0:
                    values.append(0)
                    continue
                
                if metric_key == 'abstain_f1':
                    y_true = teacher_df['teacher_abstain'].astype(int).values
                    y_pred = teacher_df['student_abstain'].astype(int).values
                    values.append(f1_score(y_true, y_pred, zero_division=0))
                elif metric_key == 'abstain_accuracy':
                    y_true = teacher_df['teacher_abstain'].astype(int).values
                    y_pred = teacher_df['student_abstain'].astype(int).values
                    values.append(accuracy_score(y_true, y_pred))
                elif metric_key == 'embedding_similarity':
                    if 'embedding_similarity_adjusted' in teacher_df.columns:
                        valid = teacher_df['embedding_similarity_adjusted'].dropna()
                        values.append(valid.mean() if len(valid) > 0 else 0)
                    else:
                        values.append(0)
                elif metric_key == 'token_overlap':
                    if 'token_overlap' in teacher_df.columns:
                        valid = teacher_df['token_overlap'].dropna()
                        values.append(valid.mean() if len(valid) > 0 else 0)
                    else:
                        values.append(0)
            
            offset = (i - len(sorted_metrics)/2 + 0.5) * width
            color = TRAIN_TYPE_COLORS.get(m['train_type'], '#888888')
            ax.bar(x + offset, values, width, label=format_model_name(model_name) if metric_key == 'abstain_f1' else '',
                   color=color, alpha=0.7)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(teachers, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.05)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.7) for t, c in TRAIN_TYPE_COLORS.items()]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle('Per-Teacher Performance Metrics by Student Model', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_teacher_metrics_grouped.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'per_teacher_metrics_grouped.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_teacher_metrics_grouped.png")


def plot_metrics_with_confidence_intervals(all_metrics: List[Dict], output_dir: str):
    """Plot key metrics with 95% confidence intervals."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    model_names = [format_model_name(m['model_name']) for m in sorted_metrics]
    colors = [TRAIN_TYPE_COLORS.get(m['train_type'], '#888888') for m in sorted_metrics]
    x = np.arange(len(model_names))
    
    metrics_to_plot = [
        ('abstain_accuracy', 'abstain_accuracy_ci_low', 'abstain_accuracy_ci_high', 'Abstain Accuracy', axes[0]),
        ('embedding_similarity_adjusted_mean', 'embedding_similarity_ci_low', 'embedding_similarity_ci_high', 'Embedding Similarity', axes[1]),
        ('token_overlap_mean', 'token_overlap_ci_low', 'token_overlap_ci_high', 'Token Overlap', axes[2]),
    ]
    
    for metric_key, ci_low_key, ci_high_key, metric_title, ax in metrics_to_plot:
        values = [m.get(metric_key, 0) for m in sorted_metrics]
        ci_lows = [m.get(ci_low_key, m.get(metric_key, 0)) for m in sorted_metrics]
        ci_highs = [m.get(ci_high_key, m.get(metric_key, 0)) for m in sorted_metrics]
        
        # Calculate error bars
        yerr_low = [v - l for v, l in zip(values, ci_lows)]
        yerr_high = [h - v for v, h in zip(values, ci_highs)]
        
        bars = ax.bar(x, values, color=colors, alpha=0.8)
        ax.errorbar(x, values, yerr=[yerr_low, yerr_high], fmt='none', color='black', capsize=3, capthick=1)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{metric_title}\n(with 95% CI)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
    
    # Legend
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.8) for t, c in TRAIN_TYPE_COLORS.items()]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle('Key Metrics with 95% Confidence Intervals', fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_with_confidence_intervals.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_with_confidence_intervals.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_with_confidence_intervals.png")


def plot_abstain_rates_with_teacher_baselines(all_metrics: List[Dict], per_teacher_abstain: Dict, output_dir: str):
    """Plot student abstain rates with per-teacher baselines."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sort models
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    sorted_metrics = sorted(all_metrics, key=lambda x: (
        group_order.index(x['model_group']) if x['model_group'] in group_order else 99,
        type_order.index(x['train_type']) if x['train_type'] in type_order else 99
    ))
    
    model_names = [format_model_name(m['model_name']) for m in sorted_metrics]
    student_rates = [m['student_abstain_rate'] * 100 for m in sorted_metrics]
    colors = [TRAIN_TYPE_COLORS.get(m['train_type'], '#888888') for m in sorted_metrics]
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, student_rates, color=colors, alpha=0.8)
    
    # Add per-teacher baselines
    if 'per_teacher_abstain_rates' in per_teacher_abstain:
        teacher_rates = per_teacher_abstain['per_teacher_abstain_rates']
        linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
        
        for i, (teacher, info) in enumerate(sorted(teacher_rates.items())):
            rate = info['abstain_rate'] * 100
            color = TEACHER_COLORS.get(teacher, f'C{i}')
            ls = linestyles[i % len(linestyles)]
            ax.axhline(y=rate, color=color, linestyle=ls, linewidth=2,
                       label=f'{teacher}: {rate:.1f}%')
    
    # Overall baseline
    overall_rate = per_teacher_abstain.get('overall_teacher_abstain_rate', 0) * 100
    ax.axhline(y=overall_rate, color='black', linestyle='-', linewidth=3,
               label=f'Overall Teacher: {overall_rate:.1f}%')
    
    ax.set_ylabel('Abstain Rate (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars, student_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Legend
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    
    ax.set_title('Student Abstain Rate vs Per-Teacher Baselines', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'abstain_rates_with_teacher_baselines.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'abstain_rates_with_teacher_baselines.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: abstain_rates_with_teacher_baselines.png")


def save_statistical_tests_report(test_results: Dict, output_dir: str):
    """Save statistical test results to a markdown report."""
    report_path = os.path.join(output_dir, 'statistical_tests_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Statistical Tests Report\n\n")
        f.write("All tests use independent samples t-test with α = 0.05\n\n")
        
        # Base vs LoRA
        f.write("## 1. Base vs LoRA (by Model Size)\n\n")
        if test_results['base_vs_lora']:
            f.write("| Model Group | Metric | Base Mean | LoRA Mean | t-statistic | p-value | Significant |\n")
            f.write("|-------------|--------|-----------|-----------|-------------|---------|-------------|\n")
            for r in test_results['base_vs_lora']:
                sig = "✓" if r['significant'] else ""
                f.write(f"| {r['group']} | {r['metric']} | {r['base_mean']:.4f} | {r['lora_mean']:.4f} | {r['t_statistic']:.3f} | {r['p_value']:.4f} | {sig} |\n")
        else:
            f.write("No data available for this comparison.\n")
        f.write("\n")
        
        # LoRA vs Full Finetune
        f.write("## 2. LoRA vs Full Finetune (0.5B and 3B)\n\n")
        if test_results['lora_vs_full_ft']:
            f.write("| Model Group | Metric | LoRA Mean | Full FT Mean | t-statistic | p-value | Significant |\n")
            f.write("|-------------|--------|-----------|--------------|-------------|---------|-------------|\n")
            for r in test_results['lora_vs_full_ft']:
                sig = "✓" if r['significant'] else ""
                f.write(f"| {r['group']} | {r['metric']} | {r['lora_mean']:.4f} | {r['full_ft_mean']:.4f} | {r['t_statistic']:.3f} | {r['p_value']:.4f} | {sig} |\n")
        else:
            f.write("No data available for this comparison.\n")
        f.write("\n")
        
        # Cross-size comparisons
        f.write("## 3. Cross-Size Comparisons\n\n")
        if test_results['cross_size']:
            f.write("| Comparison | Metric | Model 1 Mean | Model 2 Mean | t-statistic | p-value | Significant |\n")
            f.write("|------------|--------|--------------|--------------|-------------|---------|-------------|\n")
            for r in test_results['cross_size']:
                sig = "✓" if r['significant'] else ""
                f.write(f"| {r['comparison']} | {r['metric']} | {r['model1_mean']:.4f} | {r['model2_mean']:.4f} | {r['t_statistic']:.3f} | {r['p_value']:.4f} | {sig} |\n")
        else:
            f.write("No data available for this comparison.\n")
        f.write("\n")
        
        # Gemma vs Qwen
        f.write("## 4. Gemma vs Qwen (Family Comparison)\n\n")
        if test_results['gemma_vs_qwen']:
            f.write("| Comparison | Metric | Gemma Mean | Qwen Mean | t-statistic | p-value | Significant |\n")
            f.write("|------------|--------|------------|-----------|-------------|---------|-------------|\n")
            for r in test_results['gemma_vs_qwen']:
                sig = "✓" if r['significant'] else ""
                f.write(f"| {r['comparison']} | {r['metric']} | {r['gemma_mean']:.4f} | {r['qwen_mean']:.4f} | {r['t_statistic']:.3f} | {r['p_value']:.4f} | {sig} |\n")
        else:
            f.write("No data available for this comparison.\n")
        f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        total_tests = sum(len(v) for v in test_results.values())
        significant_tests = sum(1 for v in test_results.values() for r in v if r.get('significant', False))
        f.write(f"- Total tests conducted: {total_tests}\n")
        f.write(f"- Significant results (p < 0.05): {significant_tests}\n")
        f.write(f"- Non-significant results: {total_tests - significant_tests}\n")
    
    print(f"  Saved: statistical_tests_report.md")
    
    # Also save as JSON for programmatic access
    json_path = os.path.join(output_dir, 'statistical_tests_results.json')
    with open(json_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"  Saved: statistical_tests_results.json")


def save_per_teacher_metrics_report(all_metrics: List[Dict], per_teacher_abstain: Dict, output_dir: str):
    """Save per-teacher metrics to a detailed report."""
    report_path = os.path.join(output_dir, 'per_teacher_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Per-Teacher Analysis Report\n\n")
        
        # Per-teacher abstain rates
        f.write("## 1. Per-Teacher Abstain Rates\n\n")
        if 'per_teacher_abstain_rates' in per_teacher_abstain:
            f.write("| Teacher | Abstain Rate | Abstain Count | Total Count |\n")
            f.write("|---------|--------------|---------------|-------------|\n")
            for teacher, info in sorted(per_teacher_abstain['per_teacher_abstain_rates'].items()):
                f.write(f"| {teacher} | {info['abstain_rate']*100:.1f}% | {info['abstain_count']} | {info['total_count']} |\n")
            f.write(f"\n**Overall Teacher Abstain Rate:** {per_teacher_abstain.get('overall_teacher_abstain_rate', 0)*100:.1f}%\n\n")
        
        # Per-teacher metrics for each model
        f.write("## 2. Per-Teacher Performance by Student Model\n\n")
        
        for m in all_metrics:
            if 'per_teacher' not in m:
                continue
            
            f.write(f"### {m['model_name']} ({m['train_type']})\n\n")
            f.write("| Teacher | Abstain F1 | Abstain Acc | Emb Sim | Token Overlap | Exact Match |\n")
            f.write("|---------|------------|-------------|---------|---------------|-------------|\n")
            
            for teacher, tm in sorted(m['per_teacher'].items()):
                f.write(f"| {teacher} | {tm.get('abstain_f1', 0):.3f} | {tm.get('abstain_accuracy', 0):.3f} | ")
                f.write(f"{tm.get('embedding_similarity_mean', 0):.3f} | {tm.get('token_overlap_mean', 0):.3f} | ")
                f.write(f"{tm.get('exact_match_rate', 0):.3f} |\n")
            f.write("\n")
    
    print(f"  Saved: per_teacher_analysis_report.md")
    
    # Save per-teacher data as CSV
    rows = []
    for m in all_metrics:
        if 'per_teacher' not in m:
            continue
        for teacher, tm in m['per_teacher'].items():
            row = {
                'model_name': m['model_name'],
                'model_group': m['model_group'],
                'train_type': m['train_type'],
                'teacher': teacher,
                **tm
            }
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'per_teacher_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Saved: per_teacher_metrics.csv")


def plot_improvement_rate_by_training(all_metrics: List[Dict], output_dir: str):
    """
    Plot the performance improvement rate from Base to LoRA/Full Finetune.
    Shows how much each training method improved performance relative to base.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics_to_plot = [
        ('abstain_f1', 'Abstain F1 Score'),
        ('embedding_similarity_adjusted_mean', 'Embedding Similarity'),
        ('exact_match_rate', 'Exact Match Rate'),
        ('abstain_accuracy', 'Abstain Accuracy'),
    ]
    
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    train_types_to_compare = ["LoRA", "Full Finetune"]
    colors = {"LoRA": "#E74C3C", "Full Finetune": "#9B59B6"}  # Use consistent colors
    
    # Determine which groups have both training methods
    groups_with_both = set()
    group_train_types = defaultdict(set)
    for m in all_metrics:
        group_train_types[m['model_group']].add(m['train_type'])
    for g, types in group_train_types.items():
        if 'LoRA' in types and 'Full Finetune' in types:
            groups_with_both.add(g)
    
    for (metric_key, metric_title), ax in zip(metrics_to_plot, axes.flatten()):
        # Group data by model group and train type
        group_data = defaultdict(dict)
        for m in all_metrics:
            group_data[m['model_group']][m['train_type']] = m.get(metric_key, 0)
        
        x = np.arange(len(group_order))
        width = 0.35
        
        for i, train_type in enumerate(train_types_to_compare):
            improvements = []
            has_data = []
            for g in group_order:
                base_val = group_data.get(g, {}).get('Base', 0)
                trained_val = group_data.get(g, {}).get(train_type, 0)
                
                if base_val > 0 and trained_val > 0:
                    # Calculate percentage improvement
                    improvement = ((trained_val - base_val) / base_val) * 100
                    has_data.append(True)
                else:
                    improvement = 0
                    has_data.append(False)
                improvements.append(improvement)
            
            # Only plot if there are non-zero values
            if any(v != 0 for v in improvements):
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, improvements, width, label=train_type,
                             color=colors.get(train_type, '#888888'), alpha=0.8)
                
                # Add value labels - positioned at top-right corner of bar (or bottom-right if negative)
                for j, (bar, val, has) in enumerate(zip(bars, improvements, has_data)):
                    if val != 0:
                        # Position at right edge of bar, just above/below the top
                        x_pos = bar.get_x() + bar.get_width()  # Right edge
                        if val >= 0:
                            y_pos = bar.get_height() + 1
                            va = 'bottom'
                        else:
                            y_pos = bar.get_height() - 1
                            va = 'top'
                        ax.text(x_pos, y_pos, f'{val:+.1f}%', ha='left', va=va, fontsize=7, fontweight='bold')
        
        # Add markers for models with only one training method (LoRA only)
        for j, g in enumerate(group_order):
            if g not in groups_with_both:
                # Add a small annotation or marker
                ax.annotate('*', xy=(j, 0), xytext=(j, ax.get_ylim()[0] + 2),
                           fontsize=12, ha='center', color='gray', fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Improvement from Base (%)', fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_order, rotation=30, ha='right', fontsize=9)
        ax.legend(fontsize=9, loc='best')
    
    # Add note about models with only one training method
    note_text = "* Models with only LoRA training (no Full Finetune): Gemma 270M, Qwen 2.5 1.5B, Qwen 2.5 7B"
    fig.text(0.5, 0.01, note_text, ha='center', fontsize=10, style='italic', color='gray')
    
    fig.suptitle('Performance Improvement Rate by Training Method\n(Relative to Base Model)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, 'improvement_rate_by_training.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'improvement_rate_by_training.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: improvement_rate_by_training.png")


def save_metrics_summary(all_metrics: List[Dict], output_dir: str):
    """Save metrics summary to CSV and JSON."""
    # Flatten state_counts for CSV, exclude nested dicts like per_teacher
    flat_metrics = []
    for m in all_metrics:
        flat_m = {k: v for k, v in m.items() if k not in ['state_counts', 'state_percentages', 'per_teacher']}
        if 'state_counts' in m:
            for state, count in m['state_counts'].items():
                flat_m[f'count_{state}'] = count
        if 'state_percentages' in m:
            for state, pct in m['state_percentages'].items():
                flat_m[f'pct_{state}'] = pct
        flat_metrics.append(flat_m)
    
    df = pd.DataFrame(flat_metrics)
    
    # Reorder columns
    priority_cols = ['model_name', 'model_group', 'train_type', 'num_samples',
                     'abstain_accuracy', 'abstain_accuracy_ci_low', 'abstain_accuracy_ci_high',
                     'abstain_f1', 'abstain_precision', 'abstain_recall',
                     'embedding_similarity_adjusted_mean', 'embedding_similarity_ci_low', 'embedding_similarity_ci_high',
                     'token_overlap_mean', 'token_overlap_ci_low', 'token_overlap_ci_high',
                     'exact_match_rate', 'teacher_abstain_rate', 'student_abstain_rate']
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]
    
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: metrics_summary.csv")
    
    # Save JSON
    json_path = os.path.join(output_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  Saved: metrics_summary.json")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize model evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results from directories
  python 06_analyse_visualize_results.py --results_dirs dir1 dir2 dir3 --output_dir plots/

  # Analyze specific CSV files with custom names
  python 06_analyse_visualize_results.py --results_files file1.csv file2.csv --model_names Model1 Model2

  # Analyze all models from an eval run
  python 06_analyse_visualize_results.py --eval_run_dir outputs/evaluations/eval_run_XXXXX --output_dir analysis/
        """
    )
    
    parser.add_argument('--results_dirs', type=str, nargs='+', 
                        help='Directories containing *_detailed_results.csv files')
    parser.add_argument('--results_files', type=str, nargs='+',
                        help='Direct paths to detailed_results.csv files')
    parser.add_argument('--model_names', type=str, nargs='+',
                        help='Custom model names (must match number of results_files)')
    parser.add_argument('--eval_run_dir', type=str,
                        help='Path to eval run directory (will find all model subdirs)')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                        help='Directory to save plots and metrics')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Only calculate metrics, skip plot generation')
    
    args = parser.parse_args()
    
    # Collect results paths
    results_paths = []
    model_names = []
    
    if args.eval_run_dir:
        # Find all model directories in the eval run
        eval_dir = Path(args.eval_run_dir)
        for subdir in sorted(eval_dir.iterdir()):
            if subdir.is_dir():
                csv_files = list(subdir.glob("*_detailed_results.csv"))
                if csv_files:
                    results_paths.append(str(subdir))
                    model_names.append(subdir.name)
    
    if args.results_dirs:
        for d in args.results_dirs:
            results_paths.append(d)
            model_names.append(Path(d).name)
    
    if args.results_files:
        results_paths.extend(args.results_files)
        if args.model_names:
            if len(args.model_names) != len(args.results_files):
                print("Error: Number of model_names must match number of results_files")
                sys.exit(1)
            model_names.extend(args.model_names)
        else:
            for f in args.results_files:
                model_names.append(Path(f).stem.replace("_detailed_results", ""))
    
    if not results_paths:
        print("Error: Must provide --results_dirs, --results_files, or --eval_run_dir")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("MODEL EVALUATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Found {len(results_paths)} models to analyze")
    print(f"Output directory: {args.output_dir}")
    
    # Load all results
    print("\nLoading results...")
    all_data = {}
    all_metrics = []
    
    for path, name in zip(results_paths, model_names):
        try:
            df, model_name = load_results(path, name)
            all_data[model_name] = df
            # Use calculate_metrics_with_ci for confidence intervals and per-teacher metrics
            metrics = calculate_metrics_with_ci(df, model_name)
            all_metrics.append(metrics)
            print(f"  Loaded: {model_name} ({len(df)} samples)")
        except Exception as e:
            print(f"  Error loading {path}: {e}")
    
    if not all_metrics:
        print("Error: No valid results loaded")
        sys.exit(1)
    
    # Get teacher abstain rate (should be same for all)
    teacher_abstain_rate = all_metrics[0]['teacher_abstain_rate']
    
    # Calculate per-teacher abstain rates (ensemble vs individual)
    print("\nCalculating per-teacher abstain rates...")
    per_teacher_abstain = calculate_ensemble_vs_individual_abstain(all_data)
    if per_teacher_abstain and 'per_teacher_abstain_rates' in per_teacher_abstain:
        print("  Per-teacher abstain rates:")
        for teacher, info in sorted(per_teacher_abstain['per_teacher_abstain_rates'].items()):
            print(f"    {teacher}: {info['abstain_rate']*100:.1f}% ({info['abstain_count']}/{info['total_count']})")
        print(f"  Overall teacher abstain rate: {per_teacher_abstain.get('overall_teacher_abstain_rate', 0)*100:.1f}%")
    
    # Run statistical tests
    print("\nRunning statistical tests...")
    test_results = run_statistical_tests(all_data, all_metrics)
    n_tests = sum(len(v) for v in test_results.values())
    n_significant = sum(1 for v in test_results.values() for r in v if r.get('significant', False))
    print(f"  Conducted {n_tests} t-tests, {n_significant} significant (p < 0.05)")
    
    # Save metrics summary
    print("\nSaving metrics summary...")
    metrics_df = save_metrics_summary(all_metrics, args.output_dir)
    
    # Save statistical tests report
    print("\nSaving statistical tests report...")
    save_statistical_tests_report(test_results, args.output_dir)
    
    # Save per-teacher metrics report
    print("\nSaving per-teacher analysis report...")
    save_per_teacher_metrics_report(all_metrics, per_teacher_abstain, args.output_dir)
    
    # Print summary table
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    summary_cols = ['model_name', 'abstain_accuracy', 'abstain_f1', 
                    'embedding_similarity_adjusted_mean', 'exact_match_rate']
    print(metrics_df[[c for c in summary_cols if c in metrics_df.columns]].to_string(index=False))
    
    # Print confidence intervals
    print("\n" + "="*80)
    print("95% CONFIDENCE INTERVALS")
    print("="*80)
    ci_cols = ['model_name', 'abstain_accuracy', 'abstain_accuracy_ci_low', 'abstain_accuracy_ci_high']
    ci_df = metrics_df[[c for c in ci_cols if c in metrics_df.columns]]
    if len(ci_df.columns) > 1:
        print(ci_df.to_string(index=False))
    
    if args.skip_plots:
        print("\nSkipping plot generation (--skip_plots)")
    else:
        print("\nGenerating visualizations...")
        
        # ===== ORIGINAL PLOTS (kept as-is) =====
        # 1. Answer state distribution
        plot_answer_state_distribution(all_metrics, args.output_dir, teacher_abstain_rate)
        
        # 2. Violin plots - grouped by model family
        plot_violin_similarity_grouped(all_data, all_metrics, args.output_dir,
                                       'embedding_similarity_adjusted', 
                                       'Embedding Similarity Distribution (Adjusted)')
        
        # 3. Violin plots - all models
        plot_violin_all_models(all_data, all_metrics, args.output_dir,
                              'embedding_similarity_adjusted',
                              'Embedding Similarity Distribution by Model')
        
        # 4. Abstain metrics comparison
        plot_abstain_metrics_comparison(all_metrics, args.output_dir, teacher_abstain_rate)
        
        # 5. Abstain rates comparison
        plot_abstain_rates_comparison(all_metrics, args.output_dir, teacher_abstain_rate)
        
        # 6. Similarity metrics comparison
        plot_similarity_metrics_comparison(all_metrics, args.output_dir)
        
        # 7. Metrics heatmap
        plot_heatmap_metrics(all_metrics, args.output_dir)
        
        # 8. Training effect by size
        plot_training_effect_by_size(all_metrics, args.output_dir)
        
        # 9. Improvement rate by training
        plot_improvement_rate_by_training(all_metrics, args.output_dir)
        
        # ===== NEW PLOTS (per-teacher analysis, confidence intervals, etc.) =====
        print("\n  Generating per-teacher analysis plots...")
        
        # 10. Per-teacher abstain rates
        plot_per_teacher_abstain_rates(all_data, args.output_dir)
        
        # 11. Per-teacher student similarity
        plot_per_teacher_student_similarity(all_data, all_metrics, args.output_dir)
        
        # 12. Per-teacher abstain agreement heatmap
        plot_per_teacher_abstain_agreement(all_data, all_metrics, args.output_dir)
        
        # 13. Per-teacher metrics grouped
        plot_per_teacher_metrics_grouped(all_data, all_metrics, args.output_dir)
        
        # 14. Metrics with confidence intervals
        plot_metrics_with_confidence_intervals(all_metrics, args.output_dir)
        
        # 15. Abstain rates with per-teacher baselines
        plot_abstain_rates_with_teacher_baselines(all_metrics, per_teacher_abstain, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    # Print summary of new outputs
    print("\nNew outputs generated:")
    print("  - per_teacher_abstain_rates.png: Individual teacher abstain rates")
    print("  - per_teacher_student_similarity.png: Student similarity to each teacher")
    print("  - per_teacher_abstain_agreement_heatmap.png: Abstain F1 by model and teacher")
    print("  - per_teacher_metrics_grouped.png: Per-teacher metrics comparison")
    print("  - metrics_with_confidence_intervals.png: Key metrics with 95% CI")
    print("  - abstain_rates_with_teacher_baselines.png: Student rates vs per-teacher baselines")
    print("  - statistical_tests_report.md: T-test results (Base vs LoRA, etc.)")
    print("  - per_teacher_analysis_report.md: Detailed per-teacher metrics")
    print("  - per_teacher_metrics.csv: Per-teacher metrics in CSV format")


if __name__ == "__main__":
    main()

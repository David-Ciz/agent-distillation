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

import os
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
    # Flatten state_counts for CSV
    flat_metrics = []
    for m in all_metrics:
        flat_m = {k: v for k, v in m.items() if k not in ['state_counts', 'state_percentages']}
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
                     'abstain_accuracy', 'abstain_f1', 'abstain_precision', 'abstain_recall',
                     'embedding_similarity_adjusted_mean', 'token_overlap_mean', 'exact_match_rate',
                     'teacher_abstain_rate', 'student_abstain_rate']
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


import click


@click.command()
@click.option("--results-dirs", multiple=True,
              help="Directories containing *_detailed_results.csv files.")
@click.option("--results-files", multiple=True,
              help="Direct paths to detailed_results.csv files.")
@click.option("--model-names", multiple=True,
              help="Custom model names (must match number of --results-files).")
@click.option("--eval-run-dir", default=None,
              help="Path to an eval run directory — finds all model subdirs automatically.")
@click.option("--output-dir", default="analysis_output", show_default=True,
              help="Directory to save plots and metrics.")
@click.option("--skip-plots", is_flag=True,
              help="Only calculate metrics, skip plot generation.")
def main(results_dirs, results_files, model_names, eval_run_dir, output_dir, skip_plots):
    """Analyze and visualize model evaluation results.

    \b
    Examples:
      python 06_analyse_visualize_results.py --results-dirs dir1 dir2 --output-dir plots/
      python 06_analyse_visualize_results.py --results-files f1.csv f2.csv --model-names M1 M2
      python 06_analyse_visualize_results.py --eval-run-dir outputs/evaluations/eval_run_XXX --output-dir analysis/
    """
    # Collect results paths
    results_paths = []
    names = list(model_names)

    if eval_run_dir:
        eval_dir = Path(eval_run_dir)
        for subdir in sorted(eval_dir.iterdir()):
            if subdir.is_dir():
                csv_files = list(subdir.glob("*_detailed_results.csv"))
                if csv_files:
                    results_paths.append(str(subdir))
                    names.append(subdir.name)

    if results_dirs:
        for d in results_dirs:
            results_paths.append(d)
            names.append(Path(d).name)

    if results_files:
        results_paths.extend(results_files)
        if model_names:
            if len(model_names) != len(results_files):
                raise click.UsageError("Number of --model-names must match number of --results-files")
        else:
            for f in results_files:
                names.append(Path(f).stem.replace("_detailed_results", ""))

    if not results_paths:
        raise click.UsageError("Must provide --results-dirs, --results-files, or --eval-run-dir")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("MODEL EVALUATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Found {len(results_paths)} models to analyze")
    print(f"Output directory: {output_dir}")

    print("\nLoading results...")
    all_data = {}
    all_metrics = []

    for path, name in zip(results_paths, names):
        try:
            df, model_name = load_results(path, name)
            all_data[model_name] = df
            metrics = calculate_all_metrics(df, model_name)
            all_metrics.append(metrics)
            print(f"  Loaded: {model_name} ({len(df)} samples)")
        except Exception as e:
            print(f"  Error loading {path}: {e}")

    if not all_metrics:
        raise click.ClickException("No valid results loaded")

    teacher_abstain_rate = all_metrics[0]['teacher_abstain_rate']

    print("\nSaving metrics summary...")
    metrics_df = save_metrics_summary(all_metrics, output_dir)

    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    summary_cols = ['model_name', 'abstain_accuracy', 'abstain_f1',
                    'embedding_similarity_adjusted_mean', 'exact_match_rate']
    print(metrics_df[[c for c in summary_cols if c in metrics_df.columns]].to_string(index=False))

    if skip_plots:
        print("\nSkipping plot generation (--skip-plots)")
    else:
        print("\nGenerating visualizations...")

        plot_answer_state_distribution(all_metrics, output_dir, teacher_abstain_rate)
        plot_violin_similarity_grouped(all_data, all_metrics, output_dir,
                                       'embedding_similarity_adjusted',
                                       'Embedding Similarity Distribution (Adjusted)')
        plot_violin_all_models(all_data, all_metrics, output_dir,
                               'embedding_similarity_adjusted',
                               'Embedding Similarity Distribution by Model')
        plot_abstain_metrics_comparison(all_metrics, output_dir, teacher_abstain_rate)
        plot_abstain_rates_comparison(all_metrics, output_dir, teacher_abstain_rate)
        plot_similarity_metrics_comparison(all_metrics, output_dir)
        plot_heatmap_metrics(all_metrics, output_dir)
        plot_training_effect_by_size(all_metrics, output_dir)
        plot_improvement_rate_by_training(all_metrics, output_dir)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

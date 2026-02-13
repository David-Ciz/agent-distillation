#!/usr/bin/env python3
"""
Analyze and Visualize Task12 Multi-task Evaluation Results.

Generates visualizations comparing models across both tasks:
- Task1 metrics grouped bar charts (exact match, token overlap, abstain agreement, embedding sim)
- Task2 metrics grouped bar charts (accuracy, F1, precision, recall)
- Combined heatmap of all metrics
- Training effect plots (base -> task1 LoRA -> task12 LoRA)
- Per-model-family comparison

Usage:
    python3 scripts/06_analyse_visualize_results.py --eval_dir outputs/evaluations/eval_run_XXXXXX
"""

import argparse
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK12_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = TASK12_DIR / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Colors
TRAIN_TYPE_COLORS = {
    "Base": "#2ECC71",
    "Task1 LoRA": "#3498DB",
    "Task2 LoRA": "#F39C12",
    "Task12 LoRA": "#E74C3C",
    "LoRA": "#E74C3C",
    "Full Finetune": "#9B59B6",
}

FAMILY_ORDER = ["Gemma 270M", "Gemma 1B", "Qwen 0.5B", "Qwen 1.5B", "Qwen 3B", "Qwen 7B"]
TYPE_ORDER = ["Base", "Task1 LoRA", "Task2 LoRA", "Task12 LoRA"]


def get_model_family(model_name: str) -> str:
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
            for sf in item.glob("*_summary.json"):
                try:
                    with open(sf) as f:
                        data = json.load(f)
                    data['model_family'] = get_model_family(data.get('model_name', ''))
                    data['training_type'] = get_training_type(data.get('model_name', ''))
                    summaries.append(data)
                except Exception as e:
                    print(f"Warning: Could not load {sf}: {e}")
    return summaries


def sort_summaries(summaries: List[Dict]) -> List[Dict]:
    """Sort summaries by family and training type."""
    def sort_key(s):
        f_idx = FAMILY_ORDER.index(s['model_family']) if s['model_family'] in FAMILY_ORDER else 99
        t_idx = TYPE_ORDER.index(s['training_type']) if s['training_type'] in TYPE_ORDER else 99
        return (f_idx, t_idx)
    return sorted(summaries, key=sort_key)


def plot_task1_metrics_comparison(summaries: List[Dict], output_dir: Path):
    """Bar chart comparing Task1 metrics across all models."""
    sorted_s = sort_summaries(summaries)

    metrics = [
        ('exact_match_avg', 'Exact Match'),
        ('token_overlap_avg', 'Token Overlap'),
        ('abstain_agreement_rate', 'Abstain Agreement'),
        ('embedding_similarity_adjusted_avg', 'Embedding Sim (Adj)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        model_names = []
        values = []
        colors = []

        for s in sorted_s:
            t1 = s.get('task1', {})
            if not t1 or 'error' in t1:
                continue
            val = t1.get(metric_key)
            if val is None:
                continue
            model_names.append(s['model_name'])
            values.append(val)
            colors.append(TRAIN_TYPE_COLORS.get(s['training_type'], '#888888'))

        if not model_names:
            ax.set_visible(False)
            continue

        x = np.arange(len(model_names))
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.3)

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Task1: {metric_title}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, min(1.15, max(values) * 1.3) if values else 1.0)
        ax.grid(axis='y', alpha=0.3)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.85) for t, c in TRAIN_TYPE_COLORS.items()
                      if any(s['training_type'] == t for s in sorted_s)]
    fig.legend(handles=legend_patches, loc='upper center', ncol=len(legend_patches),
               fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle('Task1 (QA) Metrics Comparison', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'task1_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'task1_metrics_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: task1_metrics_comparison.png")


def plot_task2_metrics_comparison(summaries: List[Dict], output_dir: Path):
    """Bar chart comparing Task2 metrics across all models."""
    sorted_s = sort_summaries(summaries)

    metrics = [
        ('accuracy', 'Accuracy'),
        ('macro_f1', 'Macro F1'),
        ('macro_precision', 'Macro Precision'),
        ('macro_recall', 'Macro Recall'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        model_names = []
        values = []
        colors = []

        for s in sorted_s:
            t2 = s.get('task2', {})
            if not t2 or 'error' in t2:
                continue
            val = t2.get(metric_key)
            if val is None:
                continue
            model_names.append(s['model_name'])
            values.append(val)
            colors.append(TRAIN_TYPE_COLORS.get(s['training_type'], '#888888'))

        if not model_names:
            ax.set_visible(False)
            continue

        x = np.arange(len(model_names))
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Task2: {metric_title}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, min(1.15, max(values) * 1.3) if values else 1.0)
        ax.grid(axis='y', alpha=0.3)

    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.85) for t, c in TRAIN_TYPE_COLORS.items()
                      if any(s['training_type'] == t for s in sorted_s)]
    fig.legend(handles=legend_patches, loc='upper center', ncol=len(legend_patches),
               fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle('Task2 (Next-Action) Metrics Comparison', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'task2_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'task2_metrics_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: task2_metrics_comparison.png")


def plot_combined_heatmap(summaries: List[Dict], output_dir: Path):
    """Heatmap of all metrics for all models."""
    sorted_s = sort_summaries(summaries)

    metric_keys = [
        ('task1', 'exact_match_avg', 'T1: Exact Match'),
        ('task1', 'token_overlap_avg', 'T1: Token Overlap'),
        ('task1', 'abstain_agreement_rate', 'T1: Abstain Agr.'),
        ('task1', 'embedding_similarity_adjusted_avg', 'T1: Embed Sim'),
        ('task2', 'accuracy', 'T2: Accuracy'),
        ('task2', 'macro_f1', 'T2: Macro F1'),
        ('task2', 'weighted_f1', 'T2: Weighted F1'),
    ]

    model_names = [s['model_name'] for s in sorted_s]
    metric_labels = [mk[2] for mk in metric_keys]

    data = np.full((len(sorted_s), len(metric_keys)), np.nan)
    for i, s in enumerate(sorted_s):
        for j, (task, key, _) in enumerate(metric_keys):
            task_data = s.get(task, {})
            if task_data and 'error' not in task_data:
                val = task_data.get(key)
                if val is not None:
                    data[i, j] = val

    fig, ax = plt.subplots(figsize=(12, max(6, len(model_names) * 0.55)))

    # Mask NaN values
    mask = np.isnan(data)
    im = ax.imshow(np.where(mask, 0, data), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(model_names, fontsize=9)

    for i in range(len(model_names)):
        for j in range(len(metric_labels)):
            if not mask[i, j]:
                text_color = 'white' if data[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                        color=text_color, fontsize=8, fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', color='gray', fontsize=8)

    ax.set_title('Task12 Multi-task Performance Heatmap', fontsize=14, fontweight='bold')
    ax.grid(False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'combined_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: combined_heatmap.png")


def plot_training_effect(summaries: List[Dict], output_dir: Path):
    """Show improvement from base -> task1 LoRA -> task12 LoRA per model family."""
    # Organize by family
    family_data = defaultdict(dict)
    for s in summaries:
        family_data[s['model_family']][s['training_type']] = s

    # Only plot families that have data
    families = [f for f in FAMILY_ORDER if f in family_data]
    if not families:
        return

    # Plot for key metrics
    plot_configs = [
        ('task1', 'abstain_agreement_rate', 'T1: Abstain Agreement'),
        ('task1', 'exact_match_avg', 'T1: Exact Match'),
        ('task2', 'accuracy', 'T2: Accuracy'),
        ('task2', 'macro_f1', 'T2: Macro F1'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (task, metric, title) in zip(axes, plot_configs):
        x = np.arange(len(families))
        width = 0.2
        offsets = {'Base': -1.5, 'Task1 LoRA': -0.5, 'Task2 LoRA': 0.5, 'Task12 LoRA': 1.5}

        for train_type, offset in offsets.items():
            values = []
            for family in families:
                s = family_data.get(family, {}).get(train_type, {})
                task_data = s.get(task, {}) if isinstance(s, dict) else {}
                if task_data and 'error' not in task_data:
                    val = task_data.get(metric, 0) or 0
                else:
                    val = 0
                values.append(val)

            color = TRAIN_TYPE_COLORS.get(train_type, '#888888')
            bars = ax.bar(x + offset * width, values, width, label=train_type,
                          color=color, alpha=0.85, edgecolor='black', linewidth=0.3)

            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Training Effect: Base vs Task1 LoRA vs Task12 LoRA',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_effect.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'training_effect.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: training_effect.png")


def plot_per_family_radar(summaries: List[Dict], output_dir: Path):
    """Radar/spider chart per model family showing all metrics."""
    family_data = defaultdict(dict)
    for s in summaries:
        family_data[s['model_family']][s['training_type']] = s

    families = [f for f in FAMILY_ORDER if f in family_data]

    metric_keys = [
        ('task1', 'exact_match_avg', 'T1 Exact'),
        ('task1', 'token_overlap_avg', 'T1 Overlap'),
        ('task1', 'abstain_agreement_rate', 'T1 Abstain'),
        ('task2', 'accuracy', 'T2 Acc'),
        ('task2', 'macro_f1', 'T2 F1'),
    ]

    n_cols = min(3, len(families))
    n_rows = (len(families) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                              subplot_kw=dict(polar=True))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]

    for idx, family in enumerate(families):
        ax = axes[idx]
        ax.set_title(family, fontsize=11, fontweight='bold', pad=20)

        for train_type in TYPE_ORDER:
            s = family_data.get(family, {}).get(train_type, None)
            if s is None:
                continue

            values = []
            for task, metric, _ in metric_keys:
                task_data = s.get(task, {})
                if task_data and 'error' not in task_data:
                    val = task_data.get(metric, 0) or 0
                else:
                    val = 0
                values.append(val)
            values += values[:1]

            color = TRAIN_TYPE_COLORS.get(train_type, '#888888')
            ax.plot(angles, values, 'o-', linewidth=1.5, label=train_type, color=color, markersize=4)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([mk[2] for mk in metric_keys], fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    for idx in range(len(families), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Per-Family Multi-task Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_family_radar.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'per_family_radar.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: per_family_radar.png")


def save_metrics_summary(summaries: List[Dict], output_dir: Path):
    """Save metrics summary as CSV and JSON."""
    rows = []
    for s in sort_summaries(summaries):
        row = {
            'model_name': s.get('model_name', ''),
            'model_family': s.get('model_family', ''),
            'training_type': s.get('training_type', ''),
            'model_type': s.get('model_type', ''),
        }
        t1 = s.get('task1', {})
        if t1 and 'error' not in t1:
            for k in ['num_samples', 'exact_match_avg', 'token_overlap_avg',
                       'abstain_agreement_rate', 'teacher_abstain_rate',
                       'student_abstain_rate', 'embedding_similarity_avg',
                       'embedding_similarity_adjusted_avg']:
                row[f't1_{k}'] = t1.get(k)

        t2 = s.get('task2', {})
        if t2 and 'error' not in t2:
            for k in ['num_samples', 'accuracy', 'macro_f1', 'weighted_f1',
                       'macro_precision', 'macro_recall', 'valid_predictions']:
                row[f't2_{k}'] = t2.get(k)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'analysis_metrics_summary.csv', index=False)

    with open(output_dir / 'analysis_metrics_summary.json', 'w') as f:
        json.dump(summaries, f, indent=2, default=str)

    print(f"  Saved: analysis_metrics_summary.csv/.json")


def main():
    parser = argparse.ArgumentParser(description="Analyze Task12 evaluation results")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Path to evaluation run directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: outputs/analysis)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Directory not found: {eval_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Task12: Results Analysis and Visualization")
    print("=" * 60)
    print(f"Input: {eval_dir}")
    print(f"Output: {output_dir}")

    summaries = load_summaries(eval_dir)
    print(f"Loaded {len(summaries)} model results")

    if not summaries:
        print("No results found!")
        sys.exit(1)

    print("\nGenerating visualizations...")

    print("  - Task1 metrics comparison...")
    plot_task1_metrics_comparison(summaries, output_dir)

    print("  - Task2 metrics comparison...")
    plot_task2_metrics_comparison(summaries, output_dir)

    print("  - Combined heatmap...")
    plot_combined_heatmap(summaries, output_dir)

    print("  - Training effect plot...")
    plot_training_effect(summaries, output_dir)

    print("  - Per-family radar charts...")
    plot_per_family_radar(summaries, output_dir)

    print("  - Saving metrics summary...")
    save_metrics_summary(summaries, output_dir)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

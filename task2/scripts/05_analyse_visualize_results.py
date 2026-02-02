#!/usr/bin/env python3
"""
Analyze and Visualize Model Evaluation Results for Task 2: Next-Action Prediction

Generates visualizations:
- Confusion matrix heatmaps (per model)
- Per-class F1 comparison bar chart
- Accuracy comparison across models
- Per-action accuracy breakdown
- Training effect by model (gain from training)
- Metrics heatmap

Usage:
    python 05_analyse_visualize_results.py --eval_run_dir outputs/evaluations/eval_run_XXXXXX
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK2_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = TASK2_DIR / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# Create output directory
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Action vocabulary
ACTION_VOCAB = ['query_formulator', 'chatnoir_retriever', 'opensearch_retriever',
                'reranker', 'deduplicator', 'answer_drafter', 'finish']

# Model groupings for visualization
MODEL_GROUPS = {
    "Gemma 270M": ["gemma-3-270m-it-base", "gemma-3-270m-it-lora"],
    "Qwen 2.5 0.5B": ["Qwen2.5-0.5B-Instruct-base", "Qwen2.5-0.5B-Instruct-lora", "Qwen2.5-0.5B-Instruct-full-finetune"],
    "Qwen 2.5 1.5B": ["Qwen2.5-1.5B-Instruct-base", "Qwen2.5-1.5B-Instruct-lora"],
    "Qwen 2.5 3B": ["Qwen2.5-3B-Instruct-base", "Qwen2.5-3B-Instruct-lora", "Qwen2.5-3B-Instruct-full-finetune"],
    "Qwen 2.5 7B": ["Qwen2.5-7B-Instruct-base", "Qwen2.5-7B-Instruct-lora"],
}

# Training type colors
TRAIN_TYPE_COLORS = {
    "Base": "#2ECC71",        # Green
    "LoRA": "#E74C3C",        # Red
    "Full Finetune": "#9B59B6",  # Purple
}

# Action colors for per-class plots
ACTION_COLORS = {
    'query_formulator': '#3498db',
    'chatnoir_retriever': '#2ecc71',
    'opensearch_retriever': '#27ae60',
    'reranker': '#e74c3c',
    'deduplicator': '#f39c12',
    'answer_drafter': '#9b59b6',
    'finish': '#1abc9c',
}


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
    # Fallback: infer from name
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


def load_results(eval_run_dir: Path) -> List[Dict]:
    """Load all model results from evaluation run directory."""
    results = []
    
    # Find all summary JSON files
    for summary_file in eval_run_dir.glob("*_summary.json"):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            
            # Add derived fields
            data['model_group'] = get_model_group(data['model_name'])
            data['train_type'] = get_train_type(data['model_name'])
            
            results.append(data)
        except Exception as e:
            print(f"Error loading {summary_file}: {e}")
    
    return results


def plot_confusion_matrix(result: Dict, output_dir: Path):
    """Plot confusion matrix heatmap for a single model."""
    if 'confusion_matrix' not in result:
        return
    
    cm = np.array(result['confusion_matrix'])
    model_name = result['model_name']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=ACTION_VOCAB, yticklabels=ACTION_VOCAB,
                ax=ax, vmin=0, vmax=1)
    
    ax.set_xlabel('Predicted Action', fontsize=12)
    ax.set_ylabel('True Action', fontsize=12)
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    safe_name = model_name.replace("/", "-").replace(" ", "_")
    plt.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f"confusion_matrix_{safe_name}.pdf", bbox_inches='tight')
    plt.close()


def plot_per_class_f1_comparison(results: List[Dict], output_dir: Path):
    """Plot per-class F1 scores comparison across models."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    model_names = [r['model_name'] for r in results]
    x = np.arange(len(ACTION_VOCAB))
    width = 0.8 / len(results)
    
    for i, result in enumerate(results):
        if 'per_class' not in result:
            continue
        
        f1_scores = [result['per_class'].get(action, {}).get('f1', 0) for action in ACTION_VOCAB]
        train_type = result['train_type']
        color = TRAIN_TYPE_COLORS.get(train_type, '#888888')
        
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_scores, width, label=result['model_name'],
                     color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_VOCAB, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_f1_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "per_class_f1_comparison.pdf", bbox_inches='tight')
    plt.close()


def plot_accuracy_comparison(results: List[Dict], output_dir: Path):
    """Plot accuracy comparison across models grouped by training type."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by model group
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    type_order = ["Base", "LoRA", "Full Finetune"]
    
    # Organize data
    group_data = defaultdict(dict)
    for r in results:
        group_data[r['model_group']][r['train_type']] = r.get('accuracy', 0)
    
    x = np.arange(len(group_order))
    width = 0.25
    
    for i, train_type in enumerate(type_order):
        values = [group_data.get(g, {}).get(train_type, 0) for g in group_order]
        if any(v > 0 for v in values):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=train_type,
                         color=TRAIN_TYPE_COLORS.get(train_type, '#888888'), alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison by Model and Training Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "accuracy_comparison.pdf", bbox_inches='tight')
    plt.close()


def plot_per_action_accuracy(results: List[Dict], output_dir: Path):
    """Plot accuracy breakdown by action type."""
    # ACTION_VOCAB has 7 actions, so use 3x3 grid (9 subplots, 2 will be empty)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, action in enumerate(ACTION_VOCAB):
        ax = axes[idx]
        
        model_names = []
        accuracies = []
        colors = []
        
        for r in results:
            if 'per_class' not in r:
                continue
            pc = r['per_class'].get(action, {})
            if pc.get('support', 0) > 0:
                # Accuracy for this class = recall (TP / (TP + FN))
                recall = pc.get('recall', 0)
                model_names.append(r['model_name'].replace('-Instruct', '').replace('Qwen2.5-', 'Q'))
                accuracies.append(recall)
                colors.append(TRAIN_TYPE_COLORS.get(r['train_type'], '#888888'))
        
        if model_names:
            bars = ax.barh(range(len(model_names)), accuracies, color=colors, alpha=0.8)
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names, fontsize=8)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_title(f'{action}', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(ACTION_VOCAB), len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.8) for t, c in TRAIN_TYPE_COLORS.items()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Per-Action Recall by Model', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "per_action_accuracy.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "per_action_accuracy.pdf", bbox_inches='tight')
    plt.close()


def plot_training_effect_by_model(results: List[Dict], output_dir: Path):
    """Plot the gain from training (LoRA/Full FT improvement over Base)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics_to_plot = [
        ('accuracy', 'Accuracy Improvement'),
        ('macro_f1', 'Macro F1 Improvement'),
    ]
    
    group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
    
    # Organize data by group and type
    group_data = defaultdict(dict)
    for r in results:
        group_data[r['model_group']][r['train_type']] = r
    
    for ax, (metric_key, metric_title) in zip(axes, metrics_to_plot):
        x = np.arange(len(group_order))
        width = 0.35
        
        lora_improvements = []
        full_ft_improvements = []
        has_lora = []
        has_full_ft = []
        
        for group in group_order:
            base_val = group_data.get(group, {}).get('Base', {}).get(metric_key, 0)
            lora_val = group_data.get(group, {}).get('LoRA', {}).get(metric_key, 0)
            full_ft_val = group_data.get(group, {}).get('Full Finetune', {}).get(metric_key, 0)
            
            # Calculate improvement (percentage points)
            if base_val > 0:
                lora_imp = (lora_val - base_val) * 100 if lora_val > 0 else 0
                full_ft_imp = (full_ft_val - base_val) * 100 if full_ft_val > 0 else 0
            else:
                lora_imp = 0
                full_ft_imp = 0
            
            lora_improvements.append(lora_imp)
            full_ft_improvements.append(full_ft_imp)
            has_lora.append(lora_val > 0)
            has_full_ft.append(full_ft_val > 0)
        
        # Plot bars
        bars1 = ax.bar(x - width/2, lora_improvements, width, label='LoRA',
                      color=TRAIN_TYPE_COLORS['LoRA'], alpha=0.8)
        bars2 = ax.bar(x + width/2, full_ft_improvements, width, label='Full Finetune',
                      color=TRAIN_TYPE_COLORS['Full Finetune'], alpha=0.8)
        
        # Add value labels
        for bar, val, has_data in zip(bars1, lora_improvements, has_lora):
            if has_data and val != 0:
                y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 1.5
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                       f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
        
        for bar, val, has_data in zip(bars2, full_ft_improvements, has_full_ft):
            if has_data and val != 0:
                y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 1.5
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                       f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Improvement (percentage points)', fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_order, rotation=30, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Training Effect: Improvement Over Base Model', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "training_effect_by_model.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "training_effect_by_model.pdf", bbox_inches='tight')
    plt.close()


def plot_metrics_heatmap(results: List[Dict], output_dir: Path):
    """Plot heatmap of all models vs metrics."""
    metrics_to_show = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'weighted_f1']
    metric_labels = ['Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Weighted F1']
    
    # Sort results by model group and train type
    def sort_key(r):
        group_order = ["Gemma 270M", "Qwen 2.5 0.5B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Qwen 2.5 7B"]
        type_order = ["Base", "LoRA", "Full Finetune"]
        g_idx = group_order.index(r['model_group']) if r['model_group'] in group_order else 99
        t_idx = type_order.index(r['train_type']) if r['train_type'] in type_order else 99
        return (g_idx, t_idx)
    
    sorted_results = sorted(results, key=sort_key)
    
    model_names = [r['model_name'].replace('-Instruct', '').replace('Qwen2.5-', 'Q') for r in sorted_results]
    
    # Build data matrix
    data = np.zeros((len(sorted_results), len(metrics_to_show)))
    for i, r in enumerate(sorted_results):
        for j, m in enumerate(metrics_to_show):
            data[i, j] = r.get(m, 0)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(model_names) * 0.5)))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(model_names, fontsize=9)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(metric_labels)):
            text_color = 'white' if data[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
    
    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    ax.grid(False)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "metrics_heatmap.pdf", bbox_inches='tight')
    plt.close()


def save_metrics_summary(results: List[Dict], output_dir: Path):
    """Save metrics summary as CSV and JSON."""
    # CSV summary
    summary_data = []
    for r in results:
        row = {
            'model_name': r['model_name'],
            'model_group': r['model_group'],
            'train_type': r['train_type'],
            'accuracy': r.get('accuracy', 0),
            'macro_f1': r.get('macro_f1', 0),
            'macro_precision': r.get('macro_precision', 0),
            'macro_recall': r.get('macro_recall', 0),
            'weighted_f1': r.get('weighted_f1', 0),
            'valid_predictions': r.get('valid_predictions', 0),
            'total_samples': r.get('total_samples', 0),
        }
        
        # Add per-class F1
        if 'per_class' in r:
            for action in ACTION_VOCAB:
                row[f'f1_{action}'] = r['per_class'].get(action, {}).get('f1', 0)
        
        # Add timing
        if 'timing' in r:
            row['avg_inference_time_ms'] = r['timing'].get('avg_inference_time_per_sample', 0) * 1000
            row['total_eval_time_s'] = r['timing'].get('total_eval_time', 0)
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / "metrics_summary.csv", index=False)
    
    # JSON summary
    with open(output_dir / "metrics_summary.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved metrics summary to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Task 2 evaluation results")
    parser.add_argument("--eval_run_dir", type=str, required=True,
                        help="Path to evaluation run directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: outputs/analysis)")
    
    args = parser.parse_args()
    
    eval_run_dir = Path(args.eval_run_dir)
    if not eval_run_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_run_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Task 2: Results Analysis and Visualization")
    print("=" * 60)
    print(f"Input: {eval_run_dir}")
    print(f"Output: {output_dir}")
    
    # Load results
    results = load_results(eval_run_dir)
    print(f"Loaded {len(results)} model results")
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # 1. Confusion matrices (per model)
    print("  - Confusion matrices...")
    for r in results:
        plot_confusion_matrix(r, output_dir)
    
    # 2. Per-class F1 comparison
    print("  - Per-class F1 comparison...")
    plot_per_class_f1_comparison(results, output_dir)
    
    # 3. Accuracy comparison
    print("  - Accuracy comparison...")
    plot_accuracy_comparison(results, output_dir)
    
    # 4. Per-action accuracy
    print("  - Per-action accuracy...")
    plot_per_action_accuracy(results, output_dir)
    
    # 5. Training effect by model
    print("  - Training effect by model...")
    plot_training_effect_by_model(results, output_dir)
    
    # 6. Metrics heatmap
    print("  - Metrics heatmap...")
    plot_metrics_heatmap(results, output_dir)
    
    # 7. Save summary
    print("  - Saving metrics summary...")
    save_metrics_summary(results, output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

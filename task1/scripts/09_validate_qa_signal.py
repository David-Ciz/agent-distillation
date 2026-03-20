#!/usr/bin/env python3
"""
Validate whether failure-mode categories predict answer quality.

Inputs:
- failure-mode outputs from 08_label_failure_modes.py
- detailed evaluation CSVs from 05_model_evaluation.py

Outputs:
- per-model merged QA-signal tables
- root-level summary tables
- cross-model plot of embedding similarity by failure-mode category
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "agent-distillation-mpl"))

import click
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu


CATEGORY_ORDER = [
    "agreement_answer",
    "agreement_abstain",
    "false_abstention",
    "false_confidence",
]

CATEGORY_COLORS = {
    "agreement_answer": "#2E8B57",
    "agreement_abstain": "#4C9F70",
    "false_abstention": "#E6A23C",
    "false_confidence": "#C0392B",
}

QUALITY_METRICS = {
    "embedding_similarity_adjusted": {
        "higher_is_better": True,
        "include_in_significance": True,
    },
    "embedding_similarity": {
        "higher_is_better": True,
        "include_in_significance": True,
    },
    "token_overlap": {
        "higher_is_better": True,
        "include_in_significance": True,
    },
    "exact_match_score": {
        "higher_is_better": True,
        "include_in_significance": True,
    },
    "student_answer_length": {
        "higher_is_better": None,
        "include_in_significance": False,
    },
}


def build_detailed_results_path(eval_run_dir: str, model_name: str) -> Path:
    return Path(eval_run_dir) / model_name / f"{model_name}_detailed_results.csv"


def discover_failure_mode_files(failure_modes_dir: str) -> List[Path]:
    files = sorted(Path(failure_modes_dir).glob("*/failure_modes.csv"))
    if not files:
        raise click.ClickException(
            f"No per-model failure_modes.csv files found under: {failure_modes_dir}"
        )
    return files


def load_joined_model_frame(failure_modes_csv: Path, eval_run_dir: str | None) -> pd.DataFrame:
    failure_df = pd.read_csv(failure_modes_csv)
    if failure_df.empty:
        raise click.ClickException(f"No rows found in: {failure_modes_csv}")

    model_name = str(failure_df["model_name"].iloc[0])
    source_csv = Path(str(failure_df["source_results_csv"].iloc[0]))

    if source_csv.exists():
        detailed_csv = source_csv
    elif eval_run_dir:
        detailed_csv = build_detailed_results_path(eval_run_dir, model_name)
    else:
        raise click.ClickException(
            f"Detailed results CSV not found for {model_name}. "
            "Provide --eval-run-dir so it can be reconstructed."
        )

    detailed_df = pd.read_csv(detailed_csv)
    if "idx" not in detailed_df.columns:
        raise click.ClickException(f"Missing 'idx' column in {detailed_csv}")

    detailed_subset = detailed_df[
        [
            "idx",
            "embedding_similarity_adjusted",
            "embedding_similarity",
            "token_overlap",
            "exact_match_score",
            "student_answer",
            "teacher_answer",
        ]
    ].copy()
    detailed_subset = detailed_subset.rename(columns={"idx": "sample_id"})
    detailed_subset["student_answer_length"] = (
        detailed_subset["student_answer"].fillna("").astype(str).str.len()
    )
    detailed_subset["teacher_answer_length"] = (
        detailed_subset["teacher_answer"].fillna("").astype(str).str.len()
    )

    merged = failure_df.merge(detailed_subset, on="sample_id", how="left", validate="one_to_one")
    merged["category"] = pd.Categorical(merged["category"], categories=CATEGORY_ORDER, ordered=True)
    return merged.sort_values("sample_id").reset_index(drop=True)


def append_metric_summary_aggs() -> Dict[str, tuple[str, str]]:
    aggs: Dict[str, tuple[str, str]] = {"num_samples": ("sample_id", "count")}
    for metric in QUALITY_METRICS:
        aggs[f"{metric}_mean"] = (metric, "mean")
        aggs[f"{metric}_median"] = (metric, "median")
        aggs[f"{metric}_std"] = (metric, "std")
    return aggs


def metric_comparison_stats(frame: pd.DataFrame, metric_name: str) -> Dict[str, object]:
    agreement_answer = frame.loc[
        frame["category"] == "agreement_answer", metric_name
    ].dropna()
    false_confidence = frame.loc[
        frame["category"] == "false_confidence", metric_name
    ].dropna()

    prefix = metric_name
    row: Dict[str, object] = {
        f"{prefix}_agreement_answer_n": int(len(agreement_answer)),
        f"{prefix}_false_confidence_n": int(len(false_confidence)),
        f"{prefix}_agreement_answer_mean": float(agreement_answer.mean()) if len(agreement_answer) else None,
        f"{prefix}_false_confidence_mean": float(false_confidence.mean()) if len(false_confidence) else None,
    }

    spec = QUALITY_METRICS[metric_name]
    if len(agreement_answer) > 0 and len(false_confidence) > 0:
        if spec["include_in_significance"]:
            alternative = "greater" if spec["higher_is_better"] else "less"
            stat, p_value = mannwhitneyu(
                agreement_answer,
                false_confidence,
                alternative=alternative,
            )
            row[f"{prefix}_mannwhitney_u_stat"] = float(stat)
            row[f"{prefix}_mannwhitney_u_pvalue"] = float(p_value)
            row[f"{prefix}_agreement_better_than_false_confidence"] = bool(
                agreement_answer.mean() > false_confidence.mean()
            )
        else:
            row[f"{prefix}_mannwhitney_u_stat"] = None
            row[f"{prefix}_mannwhitney_u_pvalue"] = None
            row[f"{prefix}_agreement_better_than_false_confidence"] = None
    else:
        row[f"{prefix}_mannwhitney_u_stat"] = None
        row[f"{prefix}_mannwhitney_u_pvalue"] = None
        row[f"{prefix}_agreement_better_than_false_confidence"] = None

    category_groups = [
        values.dropna().tolist()
        for _, values in frame.groupby("category", observed=True)[metric_name]
        if len(values.dropna()) > 0
    ]
    if len(category_groups) >= 2 and spec["include_in_significance"]:
        kw_stat, kw_pvalue = kruskal(*category_groups)
        row[f"{prefix}_kruskal_stat"] = float(kw_stat)
        row[f"{prefix}_kruskal_pvalue"] = float(kw_pvalue)
    else:
        row[f"{prefix}_kruskal_stat"] = None
        row[f"{prefix}_kruskal_pvalue"] = None

    return row


def build_summary_tables(merged_frames: List[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.concat(merged_frames, ignore_index=True)

    summary = (
        merged.groupby(["model_name", "training_type", "model_size_B", "category"], observed=True)
        .agg(**append_metric_summary_aggs())
        .reset_index()
        .sort_values(["model_name", "category"])
    )

    significance_rows: List[Dict] = []
    for model_name, frame in merged.groupby("model_name"):
        row: Dict[str, object] = {"model_name": model_name}
        for metric_name in QUALITY_METRICS:
            row.update(metric_comparison_stats(frame, metric_name))

        significance_rows.append(row)

    significance = pd.DataFrame(significance_rows).sort_values("model_name").reset_index(drop=True)
    return summary, significance


def write_per_model_outputs(merged_frames: List[pd.DataFrame], output_dir: str) -> None:
    for frame in merged_frames:
        model_name = str(frame["model_name"].iloc[0])
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        frame.to_csv(os.path.join(model_dir, "qa_signal_validation.csv"), index=False)


def plot_validation_figure(merged_frames: List[pd.DataFrame], output_dir: str) -> str:
    merged = pd.concat(merged_frames, ignore_index=True)
    model_order = (
        merged[["model_name", "model_size_B", "training_type"]]
        .drop_duplicates()
        .sort_values(["model_size_B", "model_name"])
    )
    ordered_models = model_order["model_name"].tolist()

    n_models = len(ordered_models)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows), sharey=True)
    axes = axes.flatten()

    for ax, model_name in zip(axes, ordered_models):
        subset = merged[merged["model_name"] == model_name].copy()
        subset = subset.dropna(subset=["embedding_similarity_adjusted"])
        sns.boxplot(
            data=subset,
            x="category",
            y="embedding_similarity_adjusted",
            order=CATEGORY_ORDER,
            palette=CATEGORY_COLORS,
            ax=ax,
            fliersize=1.5,
            linewidth=1.0,
        )
        ax.set_title(model_name)
        ax.set_xlabel("")
        ax.set_ylabel("Embedding Similarity Adjusted")
        ax.tick_params(axis="x", rotation=35)

    for ax in axes[n_models:]:
        ax.axis("off")

    fig.suptitle("QA Signal Validation by Failure Mode Category", fontsize=16, y=0.995)
    fig.tight_layout()

    output_path = os.path.join(output_dir, "qa_signal_validation.png")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def log_to_mlflow(
    source_label: str,
    output_dir: str,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    mlflow_experiment: str,
) -> None:
    from mlflow_utils import setup_mlflow

    setup_mlflow(mlflow_experiment)

    with mlflow.start_run(run_name=f"qa-signal-{source_label}"):
        mlflow.log_params(
            {
                "source_label": source_label,
                "num_models": summary["model_name"].nunique(),
                "output_dir": output_dir,
            }
        )

        mlflow.log_artifact(os.path.join(output_dir, "qa_signal_summary_by_model_category.csv"), artifact_path=source_label)
        mlflow.log_artifact(os.path.join(output_dir, "qa_signal_significance_by_model.csv"), artifact_path=source_label)
        plot_path = os.path.join(output_dir, "qa_signal_validation.png")
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, artifact_path=source_label)

        for model_name, model_summary in summary.groupby("model_name"):
            with mlflow.start_run(run_name=f"qa-signal-{model_name}", nested=True):
                row0 = model_summary.iloc[0]
                mlflow.log_params(
                    {
                        "source_label": source_label,
                        "model_name": model_name,
                        "training_type": row0["training_type"],
                        "model_size_B": float(row0["model_size_B"]),
                    }
                )

                metrics: Dict[str, float] = {}
                for _, row in model_summary.iterrows():
                    category = row["category"]
                    for metric_name in QUALITY_METRICS:
                        mean_key = f"{metric_name}_mean"
                        metrics[f"{category}_{mean_key}"] = float(row[mean_key])

                sig_row = significance[significance["model_name"] == model_name]
                if not sig_row.empty:
                    sig = sig_row.iloc[0]
                    for metric_name, spec in QUALITY_METRICS.items():
                        for suffix in (
                            "agreement_answer_mean",
                            "false_confidence_mean",
                            "mannwhitney_u_stat",
                            "mannwhitney_u_pvalue",
                            "kruskal_stat",
                            "kruskal_pvalue",
                        ):
                            key = f"{metric_name}_{suffix}"
                            value = sig[key]
                            if pd.notna(value):
                                metrics[key] = float(value)
                mlflow.log_metrics(metrics)


@click.command()
@click.option(
    "--failure-modes-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing per-model failure_modes.csv outputs from 08_label_failure_modes.py.",
)
@click.option(
    "--eval-run-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Eval run directory used as a fallback when source detailed-results paths need reconstruction.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Output directory for QA signal validation artifacts.",
)
@click.option(
    "--source-label",
    default=None,
    help="Human-readable label for this validation run. Defaults to the failure-modes directory name.",
)
@click.option(
    "--mlflow-experiment",
    default=None,
    help="If set, log summary metrics and artifacts to this MLflow experiment.",
)
@click.option(
    "--skip-plot",
    is_flag=True,
    help="Skip local PNG plot generation and only write tables / MLflow metrics.",
)
def main(
    failure_modes_dir: str,
    eval_run_dir: str | None,
    output_dir: str,
    source_label: str | None,
    mlflow_experiment: str | None,
    skip_plot: bool,
) -> None:
    """Validate that failure-mode labels correlate with answer quality."""
    if source_label is None:
        source_label = Path(failure_modes_dir).name

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    failure_mode_files = discover_failure_mode_files(failure_modes_dir)
    merged_frames = [
        load_joined_model_frame(path, eval_run_dir=eval_run_dir)
        for path in failure_mode_files
    ]

    write_per_model_outputs(merged_frames, output_dir)
    summary, significance = build_summary_tables(merged_frames)

    summary_path = os.path.join(output_dir, "qa_signal_summary_by_model_category.csv")
    significance_path = os.path.join(output_dir, "qa_signal_significance_by_model.csv")
    all_models_path = os.path.join(output_dir, "qa_signal_validation_all_models.csv")

    summary.to_csv(summary_path, index=False)
    significance.to_csv(significance_path, index=False)
    pd.concat(merged_frames, ignore_index=True).to_csv(all_models_path, index=False)

    plot_path = None
    if not skip_plot:
        plot_path = plot_validation_figure(merged_frames, output_dir)

    if mlflow_experiment:
        log_to_mlflow(
            source_label=source_label,
            output_dir=output_dir,
            summary=summary,
            significance=significance,
            mlflow_experiment=mlflow_experiment,
        )

    click.echo(f"Validated {len(merged_frames)} model(s).")
    click.echo(f"Saved directory: {output_dir}")
    click.echo(f"  - {summary_path}")
    click.echo(f"  - {significance_path}")
    click.echo(f"  - {all_models_path}")
    if plot_path:
        click.echo(f"  - {plot_path}")
    if mlflow_experiment:
        click.echo(f"Logged to MLflow experiment: {mlflow_experiment}")

    preview = significance[
        [
            "model_name",
            "embedding_similarity_agreement_answer_mean",
            "embedding_similarity_false_confidence_mean",
            "embedding_similarity_agreement_better_than_false_confidence",
            "embedding_similarity_mannwhitney_u_pvalue",
            "token_overlap_agreement_answer_mean",
            "token_overlap_false_confidence_mean",
            "token_overlap_agreement_better_than_false_confidence",
            "token_overlap_mannwhitney_u_pvalue",
            "exact_match_score_agreement_answer_mean",
            "exact_match_score_false_confidence_mean",
            "exact_match_score_agreement_better_than_false_confidence",
            "exact_match_score_mannwhitney_u_pvalue",
        ]
    ]
    click.echo("")
    click.echo(preview.to_string(index=False))


if __name__ == "__main__":
    main()

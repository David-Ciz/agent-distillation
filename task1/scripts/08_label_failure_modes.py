#!/usr/bin/env python3
"""
Label eval samples with the QA-guided routing failure-mode taxonomy.

This script is post-processing only. It reads one or more
`*_detailed_results.csv` files produced by `05_model_evaluation.py` and writes
one combined `failure_modes.csv` table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import click
import mlflow
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(TASK1_DIR, "outputs", "analysis")


def infer_training_type(model_name: str) -> str:
    name_lower = model_name.lower()
    if "full-finetune" in name_lower or "full_finetune" in name_lower:
        return "full_finetune"
    if "lora" in name_lower:
        return "lora"
    return "base"


def infer_model_size_b(model_name: str) -> float | None:
    name_lower = model_name.lower()
    if "270m" in name_lower:
        return 0.27
    if "0.5b" in name_lower:
        return 0.5
    if "0.8b" in name_lower:
        return 0.8
    if "1b" in name_lower:
        return 1.0
    if "1.5b" in name_lower:
        return 1.5
    if "2b" in name_lower:
        return 2.0
    if "3b" in name_lower:
        return 3.0
    if "4b" in name_lower:
        return 4.0
    if "7b" in name_lower:
        return 7.0
    if "9b" in name_lower:
        return 9.0
    return None


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if normalized.isna().any():
        bad_values = sorted(series[normalized.isna()].astype(str).unique())
        raise click.ClickException(
            f"Could not parse boolean values in column '{series.name}': {bad_values}"
        )
    return normalized


def categorize_failure_mode(teacher_abstains: bool, student_abstains: bool) -> str:
    if not teacher_abstains and not student_abstains:
        return "agreement_answer"
    if teacher_abstains and student_abstains:
        return "agreement_abstain"
    if not teacher_abstains and student_abstains:
        return "false_abstention"
    return "false_confidence"


def discover_results_files(eval_run_dir: str) -> List[Path]:
    run_dir = Path(eval_run_dir)
    if not run_dir.is_dir():
        raise click.ClickException(f"Eval run dir does not exist: {eval_run_dir}")

    files = sorted(run_dir.glob("*/*_detailed_results.csv"))
    if not files:
        raise click.ClickException(
            f"No *_detailed_results.csv files found under: {eval_run_dir}"
        )
    return files


def label_results_file(csv_path: Path, eval_run_name: str | None = None) -> pd.DataFrame:
    model_name = csv_path.stem.replace("_detailed_results", "")
    df = pd.read_csv(csv_path)

    required_columns = {"teacher_abstain", "student_abstain"}
    missing = required_columns - set(df.columns)
    if missing:
        raise click.ClickException(
            f"Missing required columns in {csv_path}: {sorted(missing)}"
        )

    teacher_abstains = parse_bool_series(df["teacher_abstain"])
    student_abstains = parse_bool_series(df["student_abstain"])

    sample_ids = df["idx"] if "idx" in df.columns else pd.Series(range(len(df)))
    categories = [
        categorize_failure_mode(bool(t), bool(s))
        for t, s in zip(teacher_abstains.tolist(), student_abstains.tolist())
    ]

    labeled = pd.DataFrame(
        {
            "eval_run": eval_run_name,
            "model_name": model_name,
            "training_type": infer_training_type(model_name),
            "model_size_B": infer_model_size_b(model_name),
            "sample_id": sample_ids,
            "category": categories,
            "teacher_abstains": teacher_abstains,
            "student_abstains": student_abstains,
            "answer_state": df["answer_state"] if "answer_state" in df.columns else None,
            "decision_label": df["decision_label"] if "decision_label" in df.columns else None,
            "source_results_csv": str(csv_path),
        }
    )

    return labeled


def build_labeled_table(results_files: Iterable[Path], eval_run_name: str | None) -> pd.DataFrame:
    frames = [label_results_file(path, eval_run_name=eval_run_name) for path in results_files]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["model_name", "sample_id"]).reset_index(drop=True)


def build_summary_table(labeled: pd.DataFrame) -> pd.DataFrame:
    counts = (
        labeled.groupby(["eval_run", "model_name", "training_type", "model_size_B", "category"])
        .size()
        .rename("count")
        .reset_index()
    )

    totals = (
        labeled.groupby(["eval_run", "model_name"])
        .size()
        .rename("num_samples")
        .reset_index()
    )

    summary = counts.merge(totals, on=["eval_run", "model_name"], how="left")
    summary["rate"] = summary["count"] / summary["num_samples"]
    return summary.sort_values(["model_name", "category"]).reset_index(drop=True)


def log_to_mlflow(
    labeled: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str,
    source_label: str,
    mlflow_experiment: str,
) -> None:
    from mlflow_utils import setup_mlflow

    setup_mlflow(mlflow_experiment)

    with mlflow.start_run(run_name=f"failure-modes-{source_label}") as parent_run:
        mlflow.log_params(
            {
                "source_label": source_label,
                "eval_run": labeled["eval_run"].dropna().iloc[0] if labeled["eval_run"].notna().any() else "",
                "num_models": labeled["model_name"].nunique(),
                "num_rows": len(labeled),
                "output_dir": output_dir,
            }
        )

        mlflow.log_artifact(os.path.join(output_dir, "failure_modes.csv"), artifact_path=source_label)
        mlflow.log_artifact(
            os.path.join(output_dir, "failure_mode_summary_by_model.csv"),
            artifact_path=source_label,
        )

        for model_name, model_summary in summary.groupby("model_name"):
            with mlflow.start_run(run_name=f"failure-modes-{model_name}", nested=True):
                row0 = model_summary.iloc[0]
                mlflow.log_params(
                    {
                        "source_label": source_label,
                        "model_name": model_name,
                        "training_type": row0["training_type"],
                        "model_size_B": "" if pd.isna(row0["model_size_B"]) else float(row0["model_size_B"]),
                        "num_samples": int(row0["num_samples"]),
                    }
                )

                metrics = {}
                for _, row in model_summary.iterrows():
                    category = row["category"]
                    metrics[f"{category}_count"] = int(row["count"])
                    metrics[f"{category}_rate"] = float(row["rate"])
                mlflow.log_metrics(metrics)


def write_per_model_outputs(
    labeled: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str,
) -> List[str]:
    written_dirs: List[str] = []
    for model_name, model_labeled in labeled.groupby("model_name"):
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model_labeled_path = os.path.join(model_dir, "failure_modes.csv")
        model_summary_path = os.path.join(model_dir, "failure_mode_summary.csv")

        model_labeled.sort_values("sample_id").to_csv(model_labeled_path, index=False)
        summary[summary["model_name"] == model_name].to_csv(model_summary_path, index=False)
        written_dirs.append(model_dir)

    return sorted(written_dirs)


def rebuild_aggregate_summary(output_dir: str) -> pd.DataFrame:
    summary_files = sorted(Path(output_dir).glob("*/failure_mode_summary.csv"))
    if not summary_files:
        return pd.DataFrame()

    frames = [pd.read_csv(path) for path in summary_files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["model_name", "category"]).reset_index(drop=True)
    combined.to_csv(os.path.join(output_dir, "failure_mode_summary_by_model.csv"), index=False)
    return combined


@click.command()
@click.option(
    "--eval-run-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Eval run directory containing per-model *_detailed_results.csv files.",
)
@click.option(
    "--results-file",
    "results_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Explicit *_detailed_results.csv path. Repeat for multiple models.",
)
@click.option(
    "--output-dir",
    default=None,
    help="Output directory. Defaults to <eval-run-dir>/failure_modes/ or task1/outputs/analysis/failure_modes/.",
)
@click.option(
    "--source-label",
    default=None,
    help="Human-readable label for this labeling run. Defaults to the eval run directory name.",
)
@click.option(
    "--mlflow-experiment",
    default=None,
    help="If set, log per-model failure-mode counts and rates to this MLflow experiment.",
)
def main(
    eval_run_dir: str | None,
    results_files: tuple[str, ...],
    output_dir: str | None,
    source_label: str | None,
    mlflow_experiment: str | None,
) -> None:
    """Label samples with agreement / false-abstention / false-confidence categories."""
    if not eval_run_dir and not results_files:
        raise click.ClickException("Provide either --eval-run-dir or at least one --results-file.")

    if eval_run_dir:
        discovered = discover_results_files(eval_run_dir)
        eval_run_name = Path(eval_run_dir).name
    else:
        discovered = [Path(path) for path in results_files]
        eval_run_name = None

    if source_label is None:
        source_label = eval_run_name or "manual_results_files"

    if results_files:
        explicit = [Path(path) for path in results_files]
        if eval_run_dir:
            discovered_set = {path.resolve() for path in discovered}
            explicit_set = {path.resolve() for path in explicit}
            discovered = sorted(Path(path) for path in discovered_set | explicit_set)
        else:
            discovered = explicit

    if output_dir is None:
        if eval_run_dir:
            output_dir = os.path.join(eval_run_dir, "failure_modes")
        else:
            output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "failure_modes")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    labeled = build_labeled_table(discovered, eval_run_name=eval_run_name)
    summary = build_summary_table(labeled)
    written_dirs = write_per_model_outputs(labeled, summary, output_dir)
    aggregate_summary = rebuild_aggregate_summary(output_dir)
    all_labeled_output_path = os.path.join(output_dir, "failure_modes_all_models.csv")
    labeled.to_csv(all_labeled_output_path, index=False)

    if mlflow_experiment:
        log_to_mlflow(
            labeled=labeled,
            summary=summary,
            output_dir=output_dir,
            source_label=source_label,
            mlflow_experiment=mlflow_experiment,
        )

    terminal_summary = summary[["model_name", "category", "count", "rate"]]

    click.echo(f"Labeled {len(labeled)} rows across {labeled['model_name'].nunique()} model(s).")
    click.echo(f"Saved directory: {output_dir}")
    click.echo(f"  - {all_labeled_output_path}")
    if not aggregate_summary.empty:
        click.echo(f"  - {os.path.join(output_dir, 'failure_mode_summary_by_model.csv')}")
    for model_dir in written_dirs:
        click.echo(f"  - {model_dir}/")
    if mlflow_experiment:
        click.echo(f"Logged to MLflow experiment: {mlflow_experiment}")
    click.echo("")
    click.echo(terminal_summary.to_string(index=False))


if __name__ == "__main__":
    main()

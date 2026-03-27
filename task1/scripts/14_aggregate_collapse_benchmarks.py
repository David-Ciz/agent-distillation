#!/usr/bin/env python3
"""
Aggregate normalized collapse benchmark outputs across one or more runs.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(TASK1_DIR, "outputs")
COLLAPSE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "collapse_benchmarks")


def infer_model_family(model_name: str) -> str:
    if "Qwen2.5" in model_name:
        return "Qwen2.5"
    if "Qwen3.5" in model_name:
        return "Qwen3.5"
    if "Qwen" in model_name:
        return "Qwen"
    if "Gemma" in model_name:
        return "Gemma"
    return "Other"


def extract_model_size(model_name: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+)M", model_name)
    if match:
        return float(match.group(1)) / 1000.0
    return None


def model_sort_key(model_name: str) -> Tuple[str, float, str]:
    size = extract_model_size(model_name)
    return (infer_model_family(model_name), size if size is not None else 9999.0, model_name)


def discover_summary_csvs(input_dirs: Tuple[str, ...]) -> List[Path]:
    summary_files: List[Path] = []
    for input_dir in input_dirs:
        root = Path(input_dir)
        if not root.exists():
            continue
        summary_files.extend(sorted(root.rglob("collapse_benchmark_summary.csv")))
    return summary_files


def print_latest_summary(latest_wide_df: pd.DataFrame) -> None:
    if latest_wide_df.empty:
        return
    print("\n" + "=" * 120)
    print("LATEST COLLAPSE BENCHMARK RESULTS")
    print("=" * 120)
    print(latest_wide_df.to_string(index=False))
    print("=" * 120)


@click.command()
@click.option(
    "--input-dir",
    "input_dirs",
    multiple=True,
    default=(COLLAPSE_OUTPUT_DIR,),
    show_default=True,
    help="Root directory to scan for collapse_benchmark_summary.csv files. Repeatable.",
)
@click.option(
    "--output-dir",
    default=os.path.join(COLLAPSE_OUTPUT_DIR, "aggregated"),
    show_default=True,
    help="Directory for aggregate CSV outputs.",
)
def main(input_dirs: Tuple[str, ...], output_dir: str) -> None:
    """Aggregate normalized collapse benchmark outputs across runs."""
    os.makedirs(output_dir, exist_ok=True)

    summary_files = discover_summary_csvs(input_dirs)
    if not summary_files:
        raise click.ClickException(
            f"No collapse_benchmark_summary.csv files found under: {', '.join(input_dirs)}"
        )

    frames = []
    for summary_file in summary_files:
        df = pd.read_csv(summary_file)
        if df.empty:
            continue
        df["source_summary_csv"] = str(summary_file)
        df["source_run_dir"] = str(summary_file.parent)
        df["model_family"] = df["model_name"].map(infer_model_family)
        df["model_size_b"] = df["model_name"].map(extract_model_size)
        frames.append(df)

    if not frames:
        raise click.ClickException("Discovered summary CSVs, but all of them were empty.")

    all_runs_df = pd.concat(frames, ignore_index=True)
    all_runs_df = all_runs_df.sort_values(
        ["model_family", "model_name", "benchmark_name", "evaluated_at", "source_run_dir"]
    ).reset_index(drop=True)

    latest_df = (
        all_runs_df.sort_values(["evaluated_at", "source_run_dir"])
        .drop_duplicates(subset=["model_name", "benchmark_task"], keep="last")
        .sort_values(
            by="model_name",
            key=lambda series: series.map(model_sort_key),
        )
        .reset_index(drop=True)
    )

    latest_wide_df = (
        latest_df.pivot(
            index=["model_family", "model_name", "model_type"],
            columns="benchmark_name",
            values="primary_metric_value",
        )
        .reset_index()
        .sort_values(
            by="model_name",
            key=lambda series: series.map(model_sort_key),
        )
        .reset_index(drop=True)
    )

    benchmark_counts_df = (
        all_runs_df.groupby(["model_family", "benchmark_name"], as_index=False)
        .agg(
            runs=("run_name", "nunique"),
            models=("model_name", "nunique"),
            latest_score_mean=("primary_metric_value", "mean"),
        )
        .sort_values(["model_family", "benchmark_name"])
        .reset_index(drop=True)
    )

    all_runs_path = os.path.join(output_dir, "collapse_benchmark_aggregate_all_runs.csv")
    latest_path = os.path.join(output_dir, "collapse_benchmark_aggregate_latest_long.csv")
    latest_wide_path = os.path.join(output_dir, "collapse_benchmark_aggregate_latest_wide.csv")
    benchmark_counts_path = os.path.join(output_dir, "collapse_benchmark_family_summary.csv")

    all_runs_df.to_csv(all_runs_path, index=False)
    latest_df.to_csv(latest_path, index=False)
    latest_wide_df.to_csv(latest_wide_path, index=False)
    benchmark_counts_df.to_csv(benchmark_counts_path, index=False)

    print_latest_summary(latest_wide_df)
    print(f"Saved aggregate outputs to: {output_dir}")


if __name__ == "__main__":
    main()

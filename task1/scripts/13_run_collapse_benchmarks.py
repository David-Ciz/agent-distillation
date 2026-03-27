#!/usr/bin/env python3
"""
Run final-checkpoint model-collapse benchmarks via lm-evaluation-harness.

This script is intentionally limited to final checkpoints / final LoRA adapters.
It shells out to `python -m lm_eval`, stores the raw harness outputs, then
normalizes them into stable CSV/JSON artifacts for downstream aggregation.
"""

import json
import importlib.util
import logging
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click
import mlflow
import pandas as pd

from eval_utils import setup_logging, utc_timestamp
from mlflow_utils import setup_mlflow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(TASK1_DIR, "outputs")
COLLAPSE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "collapse_benchmarks")

os.makedirs(COLLAPSE_OUTPUT_DIR, exist_ok=True)

DEFAULT_TASKS: Tuple[str, ...] = (
    "mmlu",
    "gsm8k",
    "arc_challenge",
    "humaneval",
    "truthfulqa_mc1",
)

TASK_DISPLAY_NAMES: Dict[str, str] = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "arc_challenge": "ARC-Challenge",
    "humaneval": "HumanEval",
    "truthfulqa_mc1": "TruthfulQA",
}

PRIMARY_METRIC_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "mmlu": ("acc,none", "acc_norm,none"),
    "gsm8k": ("exact_match,strict-match", "exact_match,flexible-extract", "acc,none"),
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "humaneval": ("pass@1,create_test", "pass@1,none"),
    "truthfulqa_mc1": ("acc,none", "mc1,none"),
}


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parse_model_config(config_str: str) -> Dict[str, object]:
    """
    Parse a model config string in format: path,name,type[,batch_size]

    Types: lora, full_finetune, base
    """
    parts = [p.strip() for p in config_str.split(",")]
    if len(parts) < 3:
        raise ValueError(
            f"Invalid model config: {config_str}. Expected: path,name,type[,batch_size]"
        )

    model_type = parts[2].lower()
    if model_type not in ("lora", "full_finetune", "base"):
        raise ValueError(
            f"Invalid model type: {model_type}. Must be one of: lora, full_finetune, base"
        )

    return {
        "path": parts[0],
        "name": parts[1],
        "type": model_type,
        "batch_size": int(parts[3]) if len(parts) > 3 and parts[3] else None,
    }


def resolve_tasks(tasks: Iterable[str]) -> List[str]:
    resolved = [task.strip() for task in tasks if task.strip()]
    return resolved if resolved else list(DEFAULT_TASKS)


def resolve_base_model_for_lora(adapter_path: str) -> str:
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"LoRA adapter config not found at {adapter_config_path}. "
            "Expected a final PEFT adapter directory."
        )

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"'base_model_name_or_path' missing from {adapter_config_path}. "
            "Cannot resolve the underlying base checkpoint for lm-eval."
        )
    return str(base_model)


def build_model_args(model_cfg: Dict[str, object], dtype: str) -> str:
    model_path = str(model_cfg["path"])
    model_type = str(model_cfg["type"])

    if model_type == "lora":
        base_model = resolve_base_model_for_lora(model_path)
        model_args = {
            "pretrained": base_model,
            "peft": model_path,
            "dtype": dtype,
            "trust_remote_code": "True",
        }
    else:
        model_args = {
            "pretrained": model_path,
            "dtype": dtype,
            "trust_remote_code": "True",
        }

    return ",".join(f"{key}={value}" for key, value in model_args.items())


def build_lm_eval_command(
    model_cfg: Dict[str, object],
    tasks: List[str],
    output_dir: str,
    batch_size: int,
    device: str,
    dtype: str,
    limit: int,
    apply_chat_template: bool,
    confirm_run_unsafe_code: bool,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        build_model_args(model_cfg, dtype=dtype),
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--output_path",
        output_dir,
    ]

    if limit > 0:
        command.extend(["--limit", str(limit)])
    if apply_chat_template:
        command.append("--apply_chat_template")
    if confirm_run_unsafe_code and any(task == "humaneval" for task in tasks):
        command.append("--confirm_run_unsafe_code")

    return command


def ensure_lm_eval_available() -> None:
    if importlib.util.find_spec("lm_eval") is not None:
        return

    raise RuntimeError(
        "The active Python environment cannot import `lm_eval`. "
        "For LUMI jobs, set `CONTAINER_VENV` to a venv created with "
        "`python -m venv --system-site-packages` inside the container and install "
        "`lm-evaluation-harness` there, or otherwise ensure `python -m lm_eval` "
        f"works for `{sys.executable}` before submitting the collapse benchmark job."
    )


def find_results_json(output_dir: str) -> str:
    candidates = sorted(
        Path(output_dir).rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.name.endswith(".json") and "results" in candidate.name:
            return str(candidate)
    for candidate in candidates:
        return str(candidate)
    raise FileNotFoundError(f"No JSON results found under {output_dir}")


def is_numeric_metric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def select_primary_metric(task_name: str, metric_map: Dict[str, object]) -> Tuple[str, float]:
    for candidate in PRIMARY_METRIC_CANDIDATES.get(task_name, ()):
        value = metric_map.get(candidate)
        if is_numeric_metric(value):
            return candidate, float(value)

    numeric_items = [
        (metric_name, float(metric_value))
        for metric_name, metric_value in metric_map.items()
        if is_numeric_metric(metric_value)
        and "stderr" not in metric_name
        and metric_name != "alias"
    ]
    if not numeric_items:
        raise ValueError(f"No numeric metrics found for task '{task_name}'")
    return numeric_items[0]


def normalize_results(
    model_cfg: Dict[str, object],
    tasks: List[str],
    raw_results_path: str,
    run_name: str,
    evaluated_at: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    with open(raw_results_path, "r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    raw_results = raw_payload.get("results", {})
    primary_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    for task_name in tasks:
        task_results = raw_results.get(task_name)
        if not isinstance(task_results, dict):
            raise ValueError(
                f"Task '{task_name}' missing from {raw_results_path}. "
                f"Available tasks: {sorted(raw_results.keys())}"
            )

        primary_metric_name, primary_metric_value = select_primary_metric(task_name, task_results)
        display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)

        primary_rows.append(
            {
                "run_name": run_name,
                "evaluated_at": evaluated_at,
                "model_name": model_cfg["name"],
                "model_type": model_cfg["type"],
                "model_path": model_cfg["path"],
                "benchmark_task": task_name,
                "benchmark_name": display_name,
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
                "raw_results_path": raw_results_path,
            }
        )

        for metric_name, metric_value in task_results.items():
            if not is_numeric_metric(metric_value):
                continue
            metric_rows.append(
                {
                    "run_name": run_name,
                    "evaluated_at": evaluated_at,
                    "model_name": model_cfg["name"],
                    "model_type": model_cfg["type"],
                    "model_path": model_cfg["path"],
                    "benchmark_task": task_name,
                    "benchmark_name": display_name,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "raw_results_path": raw_results_path,
                }
            )

    summary_payload = {
        "run_name": run_name,
        "evaluated_at": evaluated_at,
        "model_name": model_cfg["name"],
        "model_type": model_cfg["type"],
        "model_path": model_cfg["path"],
        "tasks": tasks,
        "primary_results": primary_rows,
    }
    return primary_rows, metric_rows, summary_payload


def print_summary_table(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    pivot = (
        summary_df.pivot(
            index="model_name",
            columns="benchmark_name",
            values="primary_metric_value",
        )
        .reset_index()
        .fillna("")
    )
    print("\n" + "=" * 100)
    print("COLLAPSE BENCHMARK SUMMARY")
    print("=" * 100)
    print(pivot.to_string(index=False))
    print("=" * 100)


@click.command()
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Model config: 'path,name,type,batch_size'. Repeat for multiple models.",
)
@click.option(
    "--task",
    "tasks",
    multiple=True,
    help="Benchmark task to run. Repeat to override the default Golden Five suite.",
)
@click.option(
    "--output-dir",
    default=COLLAPSE_OUTPUT_DIR,
    show_default=True,
    help="Base directory for collapse benchmark runs.",
)
@click.option(
    "--batch-size",
    default=4,
    show_default=True,
    type=int,
    help="Default lm-eval batch size if not provided in the model config.",
)
@click.option(
    "--device",
    default="cuda:0",
    show_default=True,
    help="Device passed through to lm-eval.",
)
@click.option(
    "--dtype",
    default="bfloat16",
    show_default=True,
    help="Model dtype passed through to lm-eval model_args.",
)
@click.option(
    "--limit",
    default=0,
    show_default=True,
    type=int,
    help="Optional sample limit per benchmark for smoke tests (0 = full benchmark).",
)
@click.option(
    "--apply-chat-template/--no-apply-chat-template",
    default=True,
    show_default=True,
    help="Pass --apply_chat_template to lm-eval.",
)
@click.option(
    "--confirm-run-unsafe-code/--no-confirm-run-unsafe-code",
    default=True,
    show_default=True,
    help="Pass --confirm_run_unsafe_code when HumanEval is included.",
)
@click.option(
    "--mlflow-experiment",
    default="collapse-benchmarks",
    show_default=True,
    help="MLflow experiment name.",
)
def main(
    models: Tuple[str, ...],
    tasks: Tuple[str, ...],
    output_dir: str,
    batch_size: int,
    device: str,
    dtype: str,
    limit: int,
    apply_chat_template: bool,
    confirm_run_unsafe_code: bool,
    mlflow_experiment: str,
) -> None:
    """Run Golden Five collapse benchmarks on final checkpoints."""
    model_configs = []
    for config_str in models:
        try:
            config = parse_model_config(config_str)
            if config["batch_size"] is None:
                config["batch_size"] = batch_size
            model_configs.append(config)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--models")

    resolved_tasks = resolve_tasks(tasks)
    ensure_lm_eval_available()
    if confirm_run_unsafe_code and any(task == "humaneval" for task in resolved_tasks):
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"collapse_benchmark_run_{timestamp}"
    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    setup_logging(os.path.join(run_output_dir, "collapse_benchmark.log"))
    logging.info("=" * 80)
    logging.info("COLLAPSE BENCHMARK EVALUATION")
    logging.info("=" * 80)
    logging.info("Run name: %s", run_name)
    logging.info("Benchmarks: %s", ", ".join(resolved_tasks))
    logging.info("Output dir: %s", run_output_dir)

    setup_mlflow(mlflow_experiment)

    all_primary_rows: List[Dict[str, object]] = []
    all_metric_rows: List[Dict[str, object]] = []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "benchmark_suite": "golden_five" if tuple(resolved_tasks) == DEFAULT_TASKS else "custom",
                "tasks": ",".join(resolved_tasks),
                "num_models": len(model_configs),
                "device": device,
                "dtype": dtype,
                "limit": limit,
                "apply_chat_template": apply_chat_template,
                "confirm_run_unsafe_code": confirm_run_unsafe_code,
                "models": "; ".join(models),
            }
        )
        mlflow.set_tag("evaluation_scope", "final_checkpoints_only")

        for model_cfg in model_configs:
            model_name = str(model_cfg["name"])
            model_slug = slugify(model_name)
            model_output_dir = os.path.join(run_output_dir, model_slug)
            raw_output_dir = os.path.join(model_output_dir, "raw")
            os.makedirs(raw_output_dir, exist_ok=True)

            logging.info("-" * 80)
            logging.info(
                "Running lm-eval for %s (%s) with batch_size=%s",
                model_name,
                model_cfg["type"],
                model_cfg["batch_size"],
            )

            command = build_lm_eval_command(
                model_cfg=model_cfg,
                tasks=resolved_tasks,
                output_dir=raw_output_dir,
                batch_size=int(model_cfg["batch_size"]),
                device=device,
                dtype=dtype,
                limit=limit,
                apply_chat_template=apply_chat_template,
                confirm_run_unsafe_code=confirm_run_unsafe_code,
            )
            logging.info("Command: %s", shlex.join(command))
            if confirm_run_unsafe_code and any(task == "humaneval" for task in resolved_tasks):
                logging.info("HF_ALLOW_CODE_EVAL=%s", os.environ.get("HF_ALLOW_CODE_EVAL", ""))

            try:
                subprocess.run(command, check=True)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Failed to execute lm-eval. Ensure the environment can run "
                    "`python -m lm_eval`."
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"lm-eval failed for {model_name} with exit code {exc.returncode}"
                ) from exc

            raw_results_path = find_results_json(raw_output_dir)
            evaluated_at = utc_timestamp()
            primary_rows, metric_rows, summary_payload = normalize_results(
                model_cfg=model_cfg,
                tasks=resolved_tasks,
                raw_results_path=raw_results_path,
                run_name=run_name,
                evaluated_at=evaluated_at,
            )

            model_primary_df = pd.DataFrame(primary_rows)
            model_metrics_df = pd.DataFrame(metric_rows)
            model_summary_path = os.path.join(model_output_dir, f"{model_slug}_benchmark_summary.json")
            model_primary_path = os.path.join(model_output_dir, f"{model_slug}_primary_metrics.csv")
            model_metrics_path = os.path.join(model_output_dir, f"{model_slug}_all_metrics.csv")

            with open(model_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
            model_primary_df.to_csv(model_primary_path, index=False)
            model_metrics_df.to_csv(model_metrics_path, index=False)

            all_primary_rows.extend(primary_rows)
            all_metric_rows.extend(metric_rows)

            with mlflow.start_run(run_name=f"collapse-{model_name}", nested=True):
                mlflow.log_params(
                    {
                        "model_name": model_name,
                        "model_path": model_cfg["path"],
                        "model_type": model_cfg["type"],
                        "lm_eval_batch_size": model_cfg["batch_size"],
                        "tasks": ",".join(resolved_tasks),
                    }
                )

                primary_metric_logs = {}
                for row in primary_rows:
                    metric_key = f"{slugify(str(row['benchmark_name']).lower())}"
                    primary_metric_logs[metric_key] = float(row["primary_metric_value"])
                if primary_metric_logs:
                    mlflow.log_metrics(primary_metric_logs)

                mlflow.log_artifact(raw_results_path, artifact_path=model_slug)
                mlflow.log_artifact(model_summary_path, artifact_path=model_slug)
                mlflow.log_artifact(model_primary_path, artifact_path=model_slug)
                mlflow.log_artifact(model_metrics_path, artifact_path=model_slug)

        summary_df = pd.DataFrame(all_primary_rows)
        metrics_df = pd.DataFrame(all_metric_rows)
        summary_path = os.path.join(run_output_dir, "collapse_benchmark_summary.csv")
        metrics_path = os.path.join(run_output_dir, "collapse_benchmark_raw_metrics.csv")
        metadata_path = os.path.join(run_output_dir, "collapse_benchmark_metadata.json")

        if not summary_df.empty:
            summary_df = summary_df.sort_values(["model_name", "benchmark_name"]).reset_index(drop=True)
            summary_df.to_csv(summary_path, index=False)
        if not metrics_df.empty:
            metrics_df = metrics_df.sort_values(
                ["model_name", "benchmark_name", "metric_name"]
            ).reset_index(drop=True)
            metrics_df.to_csv(metrics_path, index=False)

        metadata = {
            "run_name": run_name,
            "created_at": utc_timestamp(),
            "tasks": resolved_tasks,
            "models": model_configs,
            "summary_csv": summary_path,
            "raw_metrics_csv": metrics_path,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if os.path.exists(summary_path):
            mlflow.log_artifact(summary_path, artifact_path="summary")
        if os.path.exists(metrics_path):
            mlflow.log_artifact(metrics_path, artifact_path="summary")
        mlflow.log_artifact(metadata_path, artifact_path="summary")

    if not summary_df.empty:
        print_summary_table(summary_df)
        logging.info("Saved summary CSV to %s", summary_path)
    else:
        logging.warning("No summary rows were produced.")


if __name__ == "__main__":
    main()

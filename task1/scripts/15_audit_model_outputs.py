#!/usr/bin/env python3
"""
Audit a small fixed set of eval prompts across one or more models.

This is intended for fast qualitative inspection after training finishes:
- same samples for every model
- raw student output
- extracted student answer
- abstain flags for teacher and student
- a flag for "abstain signal appears somewhere in raw output"

Example:
  python task1/scripts/15_audit_model_outputs.py \
    --models '/scratch/.../Qwen_Qwen2.5-3B-Instruct-lora-final,Qwen2.5-3B-Instruct-lora,lora' \
    --models '/scratch/.../Qwen_Qwen3.5-4B-lora-final,Qwen3.5-4B-lora,lora' \
    --sample-ids 0,1,2,11,21
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import click
import pandas as pd
import torch

from eval_utils import (
    ABSTAIN_PATTERNS,
    extract_answer,
    is_abstain,
    load_model_and_tokenizer,
    utc_timestamp,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
OUTPUT_DIR = os.path.join(TASK1_DIR, "outputs")
AUDIT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "audits")

os.makedirs(AUDIT_OUTPUT_DIR, exist_ok=True)


def parse_model_spec(spec: str) -> Dict[str, str]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 3:
        raise click.ClickException(
            f"Invalid --models entry '{spec}'. Expected 'path,name,type[,batch]'."
        )
    return {
        "model_path": parts[0],
        "model_name": parts[1],
        "model_type": parts[2],
    }


def parse_sample_ids(sample_ids: str | None, num_samples: int, seed: int, df: pd.DataFrame) -> List[int]:
    if sample_ids:
        parsed = []
        for chunk in sample_ids.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                parsed.append(int(chunk))
            except ValueError as exc:
                raise click.ClickException(f"Invalid sample id '{chunk}'.") from exc
        if not parsed:
            raise click.ClickException("--sample-ids was provided but no valid ids were parsed.")
        bad = [idx for idx in parsed if idx < 0 or idx >= len(df)]
        if bad:
            raise click.ClickException(f"Sample ids out of range: {bad}")
        return parsed

    if num_samples <= 0:
        raise click.ClickException("--num-samples must be positive when --sample-ids is not used.")

    rng = pd.Series(range(len(df))).sample(
        n=min(num_samples, len(df)),
        random_state=seed,
        replace=False,
    )
    return sorted(rng.tolist())


def raw_output_contains_abstain(text: str) -> bool:
    if not text:
        return False
    for pattern in ABSTAIN_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def generate_one(
    model,
    tokenizer,
    llm_input: str,
    max_new_tokens: int,
) -> str:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [{"role": "user", "content": llm_input}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_width = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_ids = outputs[0][prompt_width:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def audit_model(
    model_spec: Dict[str, str],
    sample_rows: Sequence[Tuple[int, pd.Series]],
    max_new_tokens: int,
) -> List[Dict]:
    is_lora = model_spec["model_type"].lower() == "lora"
    model, tokenizer = load_model_and_tokenizer(model_spec["model_path"], is_lora=is_lora)
    results: List[Dict] = []

    try:
        for sample_id, row in sample_rows:
            student_output = generate_one(
                model=model,
                tokenizer=tokenizer,
                llm_input=str(row["llm_input"]),
                max_new_tokens=max_new_tokens,
            )
            teacher_answer = extract_answer(str(row["llm_output"]))
            student_answer = extract_answer(student_output)

            results.append(
                {
                    "sample_id": sample_id,
                    "query": str(row.get("query", "")),
                    "data_source": str(row.get("data_source", "")),
                    "decision_label": str(row.get("decision_label", "")),
                    "teacher_output": str(row["llm_output"]),
                    "teacher_answer": teacher_answer,
                    "teacher_abstain": is_abstain(teacher_answer),
                    "student_output": student_output,
                    "student_answer": student_answer,
                    "student_abstain": is_abstain(student_answer),
                    "raw_contains_abstain": raw_output_contains_abstain(student_output),
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def write_markdown_report(
    output_path: Path,
    run_label: str,
    sample_ids: Sequence[int],
    model_results: Dict[str, List[Dict]],
) -> None:
    lines: List[str] = []
    lines.append(f"# Model Output Audit — {run_label}")
    lines.append("")
    lines.append(f"Sample ids: {', '.join(str(i) for i in sample_ids)}")
    lines.append("")

    for model_name, rows in model_results.items():
        lines.append(f"## {model_name}")
        lines.append("")

        for row in rows:
            lines.append(
                f"### Sample {row['sample_id']} — {row['query'] or '<no query>'}"
            )
            lines.append("")
            lines.append(f"- Data source: `{row['data_source']}`")
            lines.append(f"- Decision label: `{row['decision_label']}`")
            lines.append(f"- Teacher abstain: `{row['teacher_abstain']}`")
            lines.append(f"- Student abstain: `{row['student_abstain']}`")
            lines.append(f"- Raw contains abstain phrase: `{row['raw_contains_abstain']}`")
            lines.append("")
            lines.append("Teacher answer:")
            lines.append("")
            lines.append("```text")
            lines.append(row["teacher_answer"] or "<empty>")
            lines.append("```")
            lines.append("")
            lines.append("Student answer:")
            lines.append("")
            lines.append("```text")
            lines.append(row["student_answer"] or "<empty>")
            lines.append("```")
            lines.append("")
            lines.append("Student raw output:")
            lines.append("")
            lines.append("```text")
            lines.append(row["student_output"] or "<empty>")
            lines.append("```")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


@click.command()
@click.option(
    "--models",
    "model_specs",
    multiple=True,
    required=True,
    help="Model spec: 'path,name,type[,batch]' where type is lora|base|full_finetune.",
)
@click.option(
    "--eval-dataset",
    default=os.path.join(DATA_DIR, "task1_eval_dataset.csv"),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Evaluation dataset CSV used for prompt sampling.",
)
@click.option(
    "--sample-ids",
    default=None,
    help="Comma-separated fixed dataset row ids to audit.",
)
@click.option(
    "--num-samples",
    default=10,
    show_default=True,
    type=int,
    help="Number of deterministic random samples when --sample-ids is not provided.",
)
@click.option("--seed", default=42, show_default=True, type=int, help="Sampling seed.")
@click.option(
    "--max-new-tokens",
    default=256,
    show_default=True,
    type=int,
    help="Generation length limit.",
)
@click.option(
    "--output-dir",
    default=AUDIT_OUTPUT_DIR,
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory for audit outputs.",
)
def main(
    model_specs: Sequence[str],
    eval_dataset: str,
    sample_ids: str | None,
    num_samples: int,
    seed: int,
    max_new_tokens: int,
    output_dir: str,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = pd.read_csv(eval_dataset)
    required = {"llm_input", "llm_output"}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(f"Eval dataset missing required columns: {sorted(missing)}")

    chosen_sample_ids = parse_sample_ids(sample_ids, num_samples, seed, df)
    sample_rows = [(idx, df.iloc[idx]) for idx in chosen_sample_ids]

    parsed_models = [parse_model_spec(spec) for spec in model_specs]
    run_label = utc_timestamp()
    run_dir = Path(output_dir) / f"audit_run_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_label": run_label,
        "eval_dataset": eval_dataset,
        "sample_ids": chosen_sample_ids,
        "models": parsed_models,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    model_results: Dict[str, List[Dict]] = {}
    for model_spec in parsed_models:
        model_name = model_spec["model_name"]
        logging.info("Auditing model: %s", model_name)
        rows = audit_model(model_spec, sample_rows, max_new_tokens=max_new_tokens)
        model_results[model_name] = rows
        (run_dir / f"{model_name}_audit.json").write_text(
            json.dumps(rows, indent=2),
            encoding="utf-8",
        )

    report_path = run_dir / "audit_report.md"
    write_markdown_report(report_path, run_label, chosen_sample_ids, model_results)

    logging.info("Audit complete.")
    logging.info("Report: %s", report_path)


if __name__ == "__main__":
    main()

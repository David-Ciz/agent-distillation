#!/usr/bin/env python3
"""
Script to create task1_eval_dataset.csv from eval traces.

This script follows the same format as 01_create_train_dataset.py to ensure
compatibility with training and evaluation pipelines.

The eval directory structure is:
  data/eval/exploratory_{dataset}_test/{run_uuid}/{dataset}_eval/sample_XXX/

Input:
    - data/eval/ directory containing eval trace subdirectories
    - Each sample directory should contain:
        - config.json: Contains run_uuid, teacher_id, query, search_index
        - supervised.jsonl: Contains input, output, tool, decision_label, latency_ms, tokens

Output:
    - data/task1_eval_dataset.csv with columns matching task1_dataset.csv:
        data_source, run_uuid, teacher_id, query, search_index,
        llm_input, llm_output, tool, decision_label, latency_ms, token_count,
        llm_reasoning, llm_answer
"""

import os
import json
import csv
import re
from pathlib import Path

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")


def natural_sort_key(name):
    """Sort strings with embedded numbers numerically."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', name)]


def find_sample_directories(base_dir):
    """
    Find all sample_xxx directories recursively in the eval directory.
    Returns list of tuples: (data_source, run_uuid, sample_dir_path)
    
    Structure: eval/exploratory_{dataset}_test/{run_uuid}/{dataset}_eval/sample_XXX/
    """
    sample_dirs = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory does not exist: {base_dir}")
        return sample_dirs
    
    # Iterate through exploratory directories (exploratory_msmarco_test, etc.)
    for exploratory_dir in sorted(base_path.iterdir(), key=lambda x: x.name):
        if not exploratory_dir.is_dir():
            continue
        
        # Extract data_source from exploratory name (e.g., exploratory_msmarco_test -> msmarco)
        exploratory_name = exploratory_dir.name
        data_source_match = re.match(r'exploratory_(.+?)_test', exploratory_name)
        if data_source_match:
            data_source = data_source_match.group(1)
        else:
            data_source = exploratory_name
        
        # Iterate through run_uuid directories
        for run_uuid_dir in sorted(exploratory_dir.iterdir(), key=lambda x: x.name):
            if not run_uuid_dir.is_dir():
                continue
            run_uuid = run_uuid_dir.name
            
            # Iterate through dataset_eval directories (msmarco_eval, etc.)
            for dataset_eval_dir in sorted(run_uuid_dir.iterdir(), key=lambda x: x.name):
                if not dataset_eval_dir.is_dir():
                    continue
                
                # Find sample_xxx directories
                for sample_dir in sorted(dataset_eval_dir.iterdir(), key=lambda x: natural_sort_key(x.name)):
                    if sample_dir.is_dir() and re.match(r'^sample_\d+$', sample_dir.name):
                        # Verify it has the expected files
                        if (sample_dir / 'config.json').exists():
                            sample_dirs.append((data_source, run_uuid, sample_dir))
    
    return sample_dirs


def load_json_file(filepath):
    """Load a JSON file and return its contents, or None if it doesn't exist."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def load_jsonl_file(filepath):
    """Load a JSONL file and return list of dicts, or empty list if it doesn't exist."""
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line in {filepath}: {e}")
    except FileNotFoundError:
        pass
    return lines


def extract_reasoning_and_answer(llm_output):
    """
    Extract REASONING and ANSWER parts from llm_output.
    Returns tuple (reasoning, answer).
    """
    if llm_output is None or not isinstance(llm_output, str):
        return '', ''
    
    reasoning = ''
    answer = ''
    
    # Pattern to extract REASONING section
    reasoning_match = re.search(
        r'REASONING:\s*(.*?)(?=\nANSWER:|$)',
        llm_output,
        re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Pattern to extract ANSWER section
    answer_match = re.search(
        r'ANSWER:\s*(.*?)$',
        llm_output,
        re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning, answer


def create_eval_dataset(sample_dirs, output_path):
    """
    Create task1_eval_dataset.csv with the same format as task1_dataset.csv.
    This ensures compatibility with training and evaluation pipelines.
    """
    fieldnames = [
        'data_source',
        'run_uuid',
        'teacher_id',
        'query',
        'search_index',
        'llm_input',
        'llm_output',
        'tool',
        'decision_label',
        'latency_ms',
        'token_count',
        'llm_reasoning',
        'llm_answer'
    ]
    
    rows = []
    has_reasoning_count = 0
    has_answer_count = 0
    
    for data_source, run_uuid, sample_path in sample_dirs:
        config = load_json_file(sample_path / 'config.json')
        
        if config is None:
            print(f"Skipping {sample_path}: missing config.json")
            continue
        
        # Load supervised.jsonl
        supervised_path = sample_path / 'supervised.jsonl'
        supervised_lines = load_jsonl_file(supervised_path)
        
        if not supervised_lines:
            continue
        
        # Extract common fields from config
        teacher_id = config.get('teacher_id', '')
        query = config.get('query', '')
        search_index = config.get('search_index', '')
        
        # Create a row for each line in supervised.jsonl
        for line_data in supervised_lines:
            llm_output = line_data.get('output', '')
            
            # Extract REASONING and ANSWER from llm_output
            reasoning, answer = extract_reasoning_and_answer(llm_output)
            
            if reasoning:
                has_reasoning_count += 1
            if answer:
                has_answer_count += 1
            
            row = {
                'data_source': data_source,
                'run_uuid': run_uuid,
                'teacher_id': teacher_id,
                'query': query,
                'search_index': search_index,
                'llm_input': line_data.get('input', ''),
                'llm_output': llm_output,
                'tool': line_data.get('tool', ''),
                'decision_label': line_data.get('decision_label', ''),
                'latency_ms': line_data.get('latency_ms', ''),
                'token_count': line_data.get('tokens', ''),
                'llm_reasoning': reasoning,
                'llm_answer': answer
            }
            rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Created {output_path} with {len(rows)} rows")
    if rows:
        print(f"Rows with REASONING: {has_reasoning_count} ({100*has_reasoning_count/len(rows):.1f}%)")
        print(f"Rows with ANSWER: {has_answer_count} ({100*has_answer_count/len(rows):.1f}%)")
    
    return len(rows)


def main():
    # Configuration - use relative paths based on script location
    eval_data_dir = os.path.join(DATA_DIR, "eval")
    output_path = os.path.join(DATA_DIR, "task1_eval_dataset.csv")
    
    print(f"Scanning for sample directories in: {eval_data_dir}")
    sample_dirs = find_sample_directories(eval_data_dir)
    print(f"Found {len(sample_dirs)} sample directories")
    
    if not sample_dirs:
        print("No sample directories found. Exiting.")
        print(f"Expected directory structure: {eval_data_dir}/exploratory_{{dataset}}_test/{{run_uuid}}/{{dataset}}_eval/sample_XXX/")
        return
    
    # Create eval dataset (matching training dataset format)
    print("\n--- Creating task1_eval_dataset.csv ---")
    row_count = create_eval_dataset(sample_dirs, output_path)
    
    print(f"\n=== Summary ===")
    print(f"Total sample directories processed: {len(sample_dirs)}")
    print(f"Total rows in task1_eval_dataset.csv: {row_count}")


if __name__ == "__main__":
    main()

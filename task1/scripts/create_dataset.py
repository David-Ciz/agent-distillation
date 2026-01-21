#!/usr/bin/env python3
"""
End-to-end script to create task1_dataset.csv from synthetic traces.

This script combines the functionality of:
- create_datasets.py: Scans sample directories and extracts supervised data
- process_supervised_dataset.py: Extracts REASONING and ANSWER from llm_output

Input:
    - data/synthetic_traces/ directory containing sample_xxx subdirectories
    - Each sample directory should contain:
        - config.json: Contains run_uuid, teacher_id, query, search_index
        - supervised.jsonl: Contains input, output, tool, decision_label, latency_ms, tokens

Output:
    - data/task1_dataset.csv with columns:
        data_source, run_uuid, teacher_id, query, search_index,
        llm_input, llm_output, tool, decision_label, latency_ms, token_count,
        llm_reasoning, llm_answer
"""

import os
import json
import csv
import re
from pathlib import Path


def natural_sort_key(name):
    """Sort strings with embedded numbers numerically."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', name)]


def find_sample_directories(base_dir):
    """
    Find all sample_xxx directories recursively.
    Returns list of tuples: (data_source, sample_dir_path)
    """
    sample_dirs = []
    base_path = Path(base_dir)
    
    # Iterate through data sources (causalqa, msmarco, quasart, etc.)
    for data_source_dir in sorted(base_path.iterdir(), key=lambda x: x.name):
        if not data_source_dir.is_dir():
            continue
        data_source = data_source_dir.name
        
        # Walk through all subdirectories to find sample_xxx folders
        for root, dirs, files in os.walk(data_source_dir):
            # Sort dirs for consistent ordering
            dirs.sort(key=natural_sort_key)
            
            for d in dirs:
                if re.match(r'^sample_\d+$', d):
                    sample_path = Path(root) / d
                    # Verify it has the expected files
                    if (sample_path / 'config.json').exists():
                        sample_dirs.append((data_source, sample_path))
    
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
    # Matches "REASONING:" followed by content until "ANSWER:" or end of string
    reasoning_match = re.search(
        r'REASONING:\s*(.*?)(?=\nANSWER:|$)',
        llm_output,
        re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Pattern to extract ANSWER section
    # Matches "ANSWER:" followed by content until end of string
    answer_match = re.search(
        r'ANSWER:\s*(.*?)$',
        llm_output,
        re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning, answer


def create_task1_dataset(sample_dirs, output_path):
    """
    Create task1_dataset.csv directly from sample directories.
    Combines supervised data extraction and REASONING/ANSWER parsing.
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
    
    for data_source, sample_path in sample_dirs:
        config = load_json_file(sample_path / 'config.json')
        
        if config is None:
            print(f"Skipping {sample_path}: missing config.json")
            continue
        
        # Load supervised.jsonl
        supervised_path = sample_path / 'supervised.jsonl'
        supervised_lines = load_jsonl_file(supervised_path)
        
        if not supervised_lines:
            # Skip samples with empty supervised.jsonl
            continue
        
        # Extract common fields from config
        run_uuid = config.get('run_uuid', '')
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
    print(f"Rows with REASONING: {has_reasoning_count} ({100*has_reasoning_count/len(rows):.1f}%)" if rows else "No rows")
    print(f"Rows with ANSWER: {has_answer_count} ({100*has_answer_count/len(rows):.1f}%)" if rows else "No rows")
    
    return len(rows)


def main():
    # Configuration
    base_dir = "data/synthetic_traces"
    output_path = "data/task1_dataset.csv"
    
    print(f"Scanning for sample directories in: {base_dir}")
    sample_dirs = find_sample_directories(base_dir)
    print(f"Found {len(sample_dirs)} sample directories")
    
    if not sample_dirs:
        print("No sample directories found. Exiting.")
        return
    
    # Create task1 dataset directly
    print("\n--- Creating task1_dataset.csv ---")
    row_count = create_task1_dataset(sample_dirs, output_path)
    
    print(f"\n=== Summary ===")
    print(f"Total sample directories processed: {len(sample_dirs)}")
    print(f"Total rows in task1_dataset.csv: {row_count}")


if __name__ == "__main__":
    main()

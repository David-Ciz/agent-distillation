#!/usr/bin/env python3
"""
Dataset Builder Script for Task 2: Next-Action Prediction

This script processes traces.jsonl and config.json file pairs from the raw data
directory and builds a consolidated dataset CSV file.

Output columns:
- tool, evidence_count, query, iterations, has_draft, available_actions
- next_tool, created_at, teacher_id, run_uuid, path, final_answer, step
- llm_input: Formatted input prompt
- llm_output: Just the action name (e.g., "reranker")
- llm_answer: Same as llm_output
"""

import os
import json
import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TASK2_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK2_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data" / "agentic_search"
OUTPUT_CSV = DATA_DIR / "task2_dataset.csv"
LOG_FILE = DATA_DIR / "dataset_builder.log"

AVAILABLE_ACTIONS = "['query_formulator', 'chatnoir_retriever', 'opensearch_retriever', 'reranker', 'deduplicator', 'answer_drafter', 'finish']"

# Patterns to detect "cannot answer" responses
CANNOT_ANSWER_PATTERNS = [
    r"i cannot answer",
    r"cannot answer based on",
    r"unable to answer",
    r"cannot provide an answer",
    r"no answer can be provided",
    r"insufficient evidence",
    r"not enough information",
]


def setup_logging() -> logging.Logger:
    """Setup logging with timestamp to both file and console."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("dataset_builder")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def is_cannot_answer_response(answer: str) -> bool:
    """Check if the answer indicates inability to answer based on evidence."""
    if not answer:
        return True
    answer_lower = answer.lower()
    for pattern in CANNOT_ANSWER_PATTERNS:
        if re.search(pattern, answer_lower):
            return True
    return False


def get_tool_from_line(line_data: Dict[str, Any]) -> Optional[str]:
    """Extract tool name from a trace line."""
    try:
        return line_data.get("action", {}).get("tool")
    except (AttributeError, TypeError):
        return None


def get_evidence_count(line_data: Dict[str, Any]) -> Optional[int]:
    """Extract evidence_count from a trace line."""
    return line_data.get("evidence_count")


def get_tool_output_answer(line_data: Dict[str, Any]) -> Optional[str]:
    """Extract tool_output.answer from a trace line."""
    try:
        tool_output = line_data.get("tool_output")
        if tool_output and isinstance(tool_output, dict):
            return tool_output.get("answer")
    except (AttributeError, TypeError):
        pass
    return None


def find_file_pairs(base_dir: Path, logger: logging.Logger) -> List[Tuple[Path, Path]]:
    """Find all traces.jsonl and config.json pairs in the directory tree."""
    pairs = []
    
    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)
        traces_file = root_path / "traces.jsonl"
        config_file = root_path / "config.json"
        
        if traces_file.exists() and config_file.exists():
            pairs.append((traces_file, config_file))
    
    logger.info(f"Found {len(pairs)} traces.jsonl/config.json pairs")
    return pairs


def load_config(config_path: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Load and parse config.json file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {config_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading {config_path}: {e}")
    return None


def load_traces(traces_path: Path, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """Load and parse traces.jsonl file."""
    traces = []
    try:
        with open(traces_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                    traces.append(trace)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in {traces_path} line {line_num}: {e}")
        return traces
    except Exception as e:
        logger.error(f"Error reading {traces_path}: {e}")
    return None


def has_answer_drafter(traces: List[Dict[str, Any]]) -> bool:
    """Check if any line in traces has tool='answer_drafter'."""
    for trace in traces:
        if get_tool_from_line(trace) == "answer_drafter":
            return True
    return False


def get_final_answer(traces: List[Dict[str, Any]]) -> Optional[str]:
    """Get the answer from the last answer_drafter line."""
    last_answer = None
    for trace in traces:
        if get_tool_from_line(trace) == "answer_drafter":
            answer = get_tool_output_answer(trace)
            if answer is not None:
                last_answer = answer
    return last_answer


def count_previous_answer_drafters(traces: List[Dict[str, Any]], current_idx: int) -> int:
    """Count how many answer_drafter tools appear before current index."""
    count = 0
    for i in range(current_idx):
        if get_tool_from_line(traces[i]) == "answer_drafter":
            count += 1
    return count


def check_has_draft(traces: List[Dict[str, Any]], current_idx: int) -> bool:
    """
    Determine has_draft value:
    - False if no previous answer_drafter
    - False if all previous answer_drafter have "cannot answer" responses
    - True if at least one previous answer_drafter has a valid answer
    """
    previous_drafters = []
    for i in range(current_idx):
        if get_tool_from_line(traces[i]) == "answer_drafter":
            answer = get_tool_output_answer(traces[i])
            previous_drafters.append(answer)
    
    if not previous_drafters:
        return False
    
    # Check if at least one has a valid answer (not "cannot answer")
    for answer in previous_drafters:
        if answer and not is_cannot_answer_response(answer):
            return True
    
    return False


def process_file_pair(
    traces_path: Path,
    config_path: Path,
    logger: logging.Logger
) -> Tuple[List[Dict[str, Any]], int, int, bool]:
    """
    Process a single traces.jsonl/config.json pair.
    
    Returns:
        - List of row dictionaries for the dataset
        - Number of ignored lines
        - Total lines in file
        - Whether the entire file was ignored
    """
    rows = []
    ignored_lines = 0
    
    # Load config
    config = load_config(config_path, logger)
    if config is None:
        logger.warning(f"Skipping {traces_path} due to config load error")
        return [], 0, 0, True
    
    # Load traces
    traces = load_traces(traces_path, logger)
    if traces is None:
        logger.warning(f"Skipping {traces_path} due to traces load error")
        return [], 0, 0, True
    
    total_lines = len(traces)
    
    # Check if file has any answer_drafter
    if not has_answer_drafter(traces):
        logger.debug(f"Ignoring {traces_path}: no answer_drafter found")
        return [], 0, total_lines, True
    
    # Extract config fields
    query = config.get("query", "")
    created_at = config.get("created_at", "")
    teacher_id = config.get("teacher_id", "")
    run_uuid = config.get("run_uuid", "")
    
    # Get final answer
    final_answer = get_final_answer(traces)
    
    step_counter = 0
    
    for idx, trace in enumerate(traces):
        tool = get_tool_from_line(trace)
        
        if tool is None:
            logger.warning(f"No tool found in {traces_path} line {idx + 1}")
            ignored_lines += 1
            continue
        
        # Check if current line is reranker and next line is also reranker
        if tool == "reranker" and idx + 1 < len(traces):
            next_tool_check = get_tool_from_line(traces[idx + 1])
            if next_tool_check == "reranker":
                ignored_lines += 1
                continue
        
        # Determine iterations (count of previous answer_drafter)
        if idx == 0:
            iterations = 0
        else:
            iterations = count_previous_answer_drafters(traces, idx)
        
        # Determine has_draft
        has_draft = check_has_draft(traces, idx)
        
        # Determine next_tool
        if idx == len(traces) - 1:
            next_tool = "finish"
        else:
            next_tool = get_tool_from_line(traces[idx + 1])
            if next_tool is None:
                next_tool = "finish"
        
        # Increment step counter
        step_counter += 1
        
        # Build llm_input (formatted prompt)
        llm_input = (
            f"QUERY: {query}\n"
            "CURRENT STATE:{\n"
            f"    evidence_count: {get_evidence_count(trace)}\n"
            f"    has_draft: {has_draft}\n"
            f"    iterations: {iterations}\n"
            "}\n"
            f"available_actions: {AVAILABLE_ACTIONS}"
        )
        
        # llm_output is just the action name (no prefix)
        llm_output = next_tool
        
        # Build row
        row = {
            "tool": tool,
            "evidence_count": get_evidence_count(trace),
            "query": query,
            "iterations": iterations,
            "has_draft": has_draft,
            "available_actions": AVAILABLE_ACTIONS,
            "next_tool": next_tool,
            "created_at": created_at,
            "teacher_id": teacher_id,
            "run_uuid": run_uuid,
            "path": str(traces_path),
            "final_answer": final_answer,
            "step": step_counter,
            "llm_input": llm_input,
            "llm_output": llm_output,
            "llm_answer": llm_output,  # Same as llm_output
        }
        
        rows.append(row)
    
    return rows, ignored_lines, total_lines, False


def main():
    """Main entry point for the dataset builder."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Task 2 Dataset Builder Started")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if raw data directory exists
    if not RAW_DATA_DIR.exists():
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        logger.error("Please ensure raw data is in: data/raw_data/agentic_search/")
        return
    
    # Find all file pairs
    file_pairs = find_file_pairs(RAW_DATA_DIR, logger)
    
    if not file_pairs:
        logger.error(f"No traces.jsonl/config.json pairs found in {RAW_DATA_DIR}")
        return
    
    # Process all pairs
    all_rows = []
    ignored_files_count = 0
    total_ignored_lines = 0
    
    for traces_path, config_path in file_pairs:
        logger.debug(f"Processing: {traces_path}")
        
        rows, ignored_count, total_lines, file_ignored = process_file_pair(
            traces_path, config_path, logger
        )
        
        if file_ignored:
            ignored_files_count += 1
        else:
            all_rows.extend(rows)
            total_ignored_lines += ignored_count
    
    logger.info(f"Total rows collected: {len(all_rows)}")
    logger.info(f"Total files ignored: {ignored_files_count}")
    logger.info(f"Total lines ignored: {total_ignored_lines}")
    
    # Write CSV
    if all_rows:
        fieldnames = [
            "tool", "evidence_count", "query", "iterations", "has_draft",
            "available_actions", "next_tool", "created_at", "teacher_id",
            "run_uuid", "path", "final_answer", "step", "llm_input", 
            "llm_output", "llm_answer"
        ]
        
        try:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            logger.info(f"Dataset written to: {OUTPUT_CSV}")
        except Exception as e:
            logger.error(f"Error writing CSV: {e}")
    else:
        logger.warning("No rows to write to CSV")
    
    # Print action distribution
    action_counts = {}
    for row in all_rows:
        action = row['llm_output']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    logger.info("\nAction distribution in dataset:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_rows) if all_rows else 0
        logger.info(f"  {action}: {count} ({pct:.1f}%)")
    
    logger.info("=" * 60)
    logger.info("Dataset Builder Completed")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

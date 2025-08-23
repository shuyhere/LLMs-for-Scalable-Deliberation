#!/usr/bin/env python3
"""
Process Polis-style datasets to filtered JSON outputs.

- For each dataset folder (bowling-green, operation, wage):
  - Read `summary.csv` to extract the conversation description as `question`.
  - Find the comments CSV (any .csv except `summary.csv`).
  - Keep rows where agrees + disagrees >= 50.
  - Produce JSON with structure:
      {
        "question": <str>,
        "comments": { "0": <comment-body>, "1": <comment-body>, ... }
      }
  - Write to datasets/<folder>.json

No external dependencies; uses Python stdlib only.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
TARGET_FOLDERS = ["bowling-green", "operation", "wage"]
OUTPUT_FILENAME_FMT = "{folder}.json"


def read_summary_question(summary_csv_path: Path) -> str:
    """Read key-value pairs from summary.csv and return conversation-description as question.

    The file is expected to be two-column CSV without header: key,value per line.
    """
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv_path}")

    question: str = ""
    with summary_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = (row[0] or "").strip()
            value = (row[1] if len(row) > 1 else "").strip()
            if key == "conversation-description":
                question = value
                break

    if not question:
        raise ValueError(
            f"conversation-description not found in {summary_csv_path}"
        )
    return question


def find_comments_csv(folder_path: Path) -> Path:
    """Return the first .csv file that is not named summary.csv.

    This accommodates different naming like comments.csv or wage.csv.
    """
    candidates: List[Path] = []
    for child in sorted(folder_path.iterdir()):
        if child.is_file() and child.suffix.lower() == ".csv" and child.name != "summary.csv":
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(
            f"No comments CSV found in {folder_path} (expected a .csv other than summary.csv)"
        )
    # Prefer a file literally named comments.csv if present
    for p in candidates:
        if p.name == "comments.csv":
            return p
    # Otherwise fall back to the first candidate (e.g., wage.csv)
    return candidates[0]


def load_and_filter_comments(comments_csv_path: Path) -> List[str]:
    """Load comments and filter by agrees + disagrees >= 50.

    Returns a list of comment-body strings, in file order, after filtering.
    """
    if not comments_csv_path.exists():
        raise FileNotFoundError(f"comments CSV not found: {comments_csv_path}")

    filtered_comments: List[str] = []
    with comments_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"agrees", "disagrees", "comment-body"}
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in {comments_csv_path}"
            )
        for row in reader:
            try:
                agrees_val = int((row.get("agrees") or "0").strip())
                disagrees_val = int((row.get("disagrees") or "0").strip())
            except ValueError:
                # Skip rows with non-numeric votes
                continue
            if agrees_val + disagrees_val < 50:
                continue
            comment_text = (row.get("comment-body") or "").strip()
            if comment_text:
                filtered_comments.append(comment_text)
    return filtered_comments


def build_output_json(question: str, comments: List[str]) -> Dict[str, object]:
    """Build the target JSON structure.

    comments: a list of dicts with keys {"index", "comment"}
    """
    comments_list: List[Dict[str, object]] = [
        {"index": i, "comment": c} for i, c in enumerate(comments)
    ]
    return {
        "question": question,
        "comments": comments_list,
    }


def process_folder(folder_name: str) -> Tuple[Path, int]:
    folder_path = DATASETS_DIR / folder_name
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

    summary_csv_path = folder_path / "summary.csv"
    question = read_summary_question(summary_csv_path)

    comments_csv_path = find_comments_csv(folder_path)
    comments = load_and_filter_comments(comments_csv_path)

    output_path = DATASETS_DIR / OUTPUT_FILENAME_FMT.format(folder=folder_name)
    output_data = build_output_json(question, comments)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_path, len(comments)


def main() -> None:
    results: List[Tuple[str, Path, int]] = []
    for folder in TARGET_FOLDERS:
        out_path, kept = process_folder(folder)
        results.append((folder, out_path, kept))

    # Print a concise summary for the operator
    for folder, out_path, kept in results:
        print(f"{folder}: wrote {kept} comments -> {out_path}")


if __name__ == "__main__":
    main()

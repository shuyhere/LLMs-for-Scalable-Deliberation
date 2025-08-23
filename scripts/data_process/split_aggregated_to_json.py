#!/usr/bin/env python3
"""
Split aggregated deliberation comments into per-question JSON files.

Input: datasets/deliberation_comments_aggregated.csv with columns:
  - comment, question, id
Assumption: No dirty data; we include all rows.

Output: Two JSON files under datasets/ with structure:
  {
    "question": <str>,
    "comments": [ {"index": <int>, "comment": <str>}, ... ]
  }

Filenames: derive from short slugs for the two questions:
  - protest.json
  - gun_use.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
INPUT_CSV = DATASETS_DIR / "deliberation_comments_aggregated.csv"

# Map canonical question text substrings to output filename slugs
QUESTION_TO_SLUG = {
    # Protest question has distinctive substring "peaceful protests"
    "peaceful protests": "protest",
    # Gun use question likely has substring "gun"
    "gun": "gun_use",
}


def split_by_question() -> Dict[str, Dict[str, object]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    # Accumulate per full question text
    question_to_comments: Dict[str, List[str]] = {}
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question_text = (row.get("question") or "").strip()
            comment_text = (row.get("comment") or "").strip()
            if not question_text or not comment_text:
                continue
            question_to_comments.setdefault(question_text, []).append(comment_text)

    # Build outputs per question
    outputs: Dict[str, Dict[str, object]] = {}
    for question_text, comments in question_to_comments.items():
        comments_list = [{"index": i, "comment": c} for i, c in enumerate(comments)]
        outputs[question_text] = {
            "question": question_text,
            "comments": comments_list,
        }
    return outputs


def slug_for_question(question_text: str) -> str:
    lower_q = question_text.lower()
    for key_substring, slug in QUESTION_TO_SLUG.items():
        if key_substring in lower_q:
            return slug
    # Fallback: generic slug
    return "question"


def write_outputs(outputs: Dict[str, Dict[str, object]]) -> List[Path]:
    written: List[Path] = []
    for question_text, data in outputs.items():
        slug = slug_for_question(question_text)
        out_path = DATASETS_DIR / f"{slug}.json"
        # If multiple different questions map to same slug (shouldn't happen here),
        # append numeric suffix.
        if out_path.exists():
            suffix = 2
            while True:
                candidate = DATASETS_DIR / f"{slug}_{suffix}.json"
                if not candidate.exists():
                    out_path = candidate
                    break
                suffix += 1
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        written.append(out_path)
    return written


def main() -> None:
    outputs = split_by_question()
    paths = write_outputs(outputs)
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

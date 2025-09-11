#!/usr/bin/env python3
"""
Extracts specific fields from summary-rating annotation JSONL files into a compact JSONL
suited for SFT-style consumption. For each triplet, it emits one JSON object containing:

- displayed_text: HTML summary text from the rating record
- answer_text: free-text answer from the question record
- perspective: value of "To what extent is your perspective represented in this response?" (accepts any scale_* present) from the rating record

Input is expected to be a JSONL with records such as *_rating, *_question, *_comparison.
Only *_rating and *_question are used.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import re
import html as html_lib
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fields from summary-rating annotations JSONL")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file or a directory to scan recursively",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file",
    )
    return parser.parse_args()


def strip_html(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return text
    # Remove tags
    no_tags = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities
    unescaped = html_lib.unescape(no_tags)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", unescaped).strip()
    return cleaned


def get_triplet_key(record_id: str) -> str:
    # Example ids: triplet_1382_rating, triplet_1382_question
    # The triplet key is everything up to the last underscore
    parts = record_id.split("_")
    if len(parts) < 3:
        return record_id
    return "_".join(parts[:2])


def extract_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    rec_id: str = record.get("id", "")
    if rec_id.endswith("_rating"):
        extracted["displayed_text"] = strip_html(record.get("displayed_text"))
        label_annotations = record.get("label_annotations", {}) or {}
        
        # Define the four rating questions in order
        rating_questions = [
            "To what extent is your perspective represented in this response?",
            "How informative is this summary?",
            "Do you think this summary presents a neutral and balanced view of the issue?",
            "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
        ]
        
        # Extract ratings for all four dimensions
        rating_scores = []
        for question in rating_questions:
            question_data = label_annotations.get(question, {}) or {}
            # Extract the rating value (1-5 scale)
            score = None
            if isinstance(question_data, dict):
                # Look for any scale_* key and extract the numeric value
                for k in sorted(question_data.keys()):
                    if k.startswith("scale_") and question_data.get(k):
                        try:
                            score = int(question_data[k])
                            break
                        except (ValueError, TypeError):
                            continue
            rating_scores.append(score)
        
        extracted["rating_scores"] = rating_scores
    elif rec_id.endswith("_question"):
        label_annotations = record.get("label_annotations", {}) or {}
        answer = label_annotations.get("answer", {}) or {}
        extracted["answer_text"] = answer.get("text_box")
        # Also store the question text (HTML stripped) from displayed_text
        extracted["question"] = strip_html(record.get("displayed_text"))
    elif rec_id.endswith("_comparison"):
        # Skip comparison records as per requirement
        pass
    return extracted


def merge_triplet(acc: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer not to overwrite non-empty values
    for k, v in new.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if acc.get(k) in (None, ""):
            acc[k] = v
    return acc


def validate(item: Dict[str, Any]) -> Optional[str]:
    # Include record only if it has rating_scores (4-element list for the four dimensions)
    rating_scores = item.get("rating_scores")
    
    if not rating_scores:
        return "missing fields: rating_scores"
    
    # Validate that it's a 4-element list with valid rating values (1-5 or None)
    if not isinstance(rating_scores, list) or len(rating_scores) != 4:
        return "invalid rating_scores: must be 4-element list"
    
    # Check that all scores are either 1-5 or None
    for i, score in enumerate(rating_scores):
        if score is not None and (not isinstance(score, int) or score < 1 or score > 5):
            return f"invalid rating score at position {i}: {score}, must be 1-5 or None"
    
    return None


def process_file(path: Path, triplet_data: Dict[str, Dict[str, Any]]) -> None:
    # Folder id is the parent directory name under the "full" directory
    folder_id = path.parent.name
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line in {path}: {e}", file=sys.stderr)
                continue

            rec_id = obj.get("id", "")
            if not rec_id:
                continue
            key = get_triplet_key(rec_id)
            fields = extract_fields(obj)
            # Attach metadata fields
            fields.setdefault("folder_id", folder_id)
            fields.setdefault("source", "summary-rating")
            current = triplet_data.get(key, {})
            triplet_data[key] = merge_triplet(current, fields)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    triplet_data: Dict[str, Dict[str, Any]] = {}

    if input_path.is_dir():
        # Recursively find all JSONL files; prefer files named annotated_instances.jsonl first
        candidate_files = []
        for p in input_path.rglob("*.jsonl"):
            candidate_files.append(p)
        # Sort to ensure deterministic order; prioritize files named annotated_instances.jsonl
        candidate_files.sort(key=lambda p: (p.name != "annotated_instances.jsonl", str(p)))
        if not candidate_files:
            print(f"No JSONL files found under directory: {input_path}", file=sys.stderr)
            sys.exit(1)
        for p in candidate_files:
            process_file(p, triplet_data)
    else:
        # Single file mode
        process_file(input_path, triplet_data)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for key, item in triplet_data.items():
        err = validate(item)
        if err:
            continue
        payload = {
            "triplet_key": key,
            "displayed_text": item.get("displayed_text"),
            "answer_text": item.get("answer_text"),
            "question": item.get("question"),
            "rating_scores": item.get("rating_scores"),  # 4-element list for regression training
            "folder_id": item.get("folder_id"),
            "source": item.get("source"),
        }
        items.append(payload)

    # If output ends with .json, write a single JSON array; otherwise, JSONL
    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as out_f:
            json.dump(items, out_f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(items)} records (JSON array) to {output_path}")
    else:
        written = 0
        with output_path.open("w", encoding="utf-8") as out_f:
            for obj in items:
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
        print(f"Wrote {written} records (JSONL) to {output_path}")

    # Print basic statistics
    total = len(items)
    question_counter = Counter(
        x.get("question") for x in items if x.get("question")
    )
    
    # Statistics for rating_scores (4-tuple)
    rating_dimension_names = [
        "Perspective Representation",
        "Informativeness", 
        "Neutrality & Balance",
        "Policy Approval"
    ]
    
    rating_stats = {}
    for i, dim_name in enumerate(rating_dimension_names):
        dim_scores = [x.get("rating_scores", [None]*4)[i] for x in items if x.get("rating_scores")]
        dim_scores = [s for s in dim_scores if s is not None]
        if dim_scores:
            rating_stats[dim_name] = {
                'count': len(dim_scores),
                'distribution': Counter(dim_scores),
                'mean': sum(dim_scores) / len(dim_scores)
            }

    print("\n==== Dataset Statistics ====")
    print(f"Total records: {total}")
    print("Question occurrence counts:")
    for q, cnt in sorted(question_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cnt} - {q}")
    
    print("\nRating Scores Statistics (4-tuple for regression):")
    for dim_name, stats in rating_stats.items():
        print(f"  {dim_name}:")
        print(f"    Count: {stats['count']}, Mean: {stats['mean']:.2f}")
        print(f"    Distribution: {dict(sorted(stats['distribution'].items()))}")


if __name__ == "__main__":
    main()



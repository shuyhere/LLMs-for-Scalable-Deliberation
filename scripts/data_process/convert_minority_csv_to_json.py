#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Target JSON schema example (Binary-Health-Care-Policy.json):
# {
#   "question": "...",
#   "comments": [
#       {"index": 0, "comment": "..."},
#       ...
#   ]
# }
# This script outputs the same schema, with an extra field per comment:
#   {"index": i, "comment": "...", "is_minority": true|false}


CANDIDATE_TEXT_COLUMNS = [
    "comment",
    "text",
    "content",
    "response",
    "answer",
    "Comment",
    "Text",
    "Content",
]

DEFAULT_MINORITY_TRUE_PATTERNS = [
    "yes, my opinion differs",
    "yes my opinion differs",
    "differs from the majority",
    "minority",
]

DEFAULT_MINORITY_FALSE_PATTERNS = [
    "no, my opinion aligns with the majority",
    "no my opinion aligns with the majority",
    "aligns with the majority",
    "majority",
]


def normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def parse_mapping(values_csv: str) -> List[str]:
    return [normalize(v) for v in values_csv.split(",") if v.strip()]


def detect_is_minority(value: str, true_keys: List[str], false_keys: List[str], default: Optional[bool]) -> Optional[bool]:
    v = normalize(value)
    for k in true_keys:
        if k in v:
            return True
    for k in false_keys:
        if k in v:
            return False
    return default


def read_comments_and_labels_from_csv(
    csv_path: Path,
    comment_column: Optional[str],
    minority_column: Optional[str],
    true_keys: List[str],
    false_keys: List[str],
    default_label: Optional[bool],
) -> Tuple[List[str], List[Optional[bool]]]:
    comments: List[str] = []
    labels: List[Optional[bool]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]

        # Comment column detection
        comment_col = None
        if comment_column:
            if comment_column not in headers:
                raise ValueError(f"Specified comment column '{comment_column}' not found in CSV headers: {headers}")
            comment_col = comment_column
        else:
            for cand in CANDIDATE_TEXT_COLUMNS:
                if cand in headers:
                    comment_col = cand
                    break
        if comment_col is None:
            raise ValueError(f"Could not auto-detect comment text column. Headers: {headers}. Use --comment-column to specify.")

        # Minority column optional
        minority_col = None
        if minority_column:
            if minority_column not in headers:
                raise ValueError(f"Specified minority column '{minority_column}' not found in CSV headers: {headers}")
            minority_col = minority_column

        for row in reader:
            raw_comment = row.get(comment_col, "")
            if not isinstance(raw_comment, str):
                continue
            comment_text = raw_comment.strip()
            if not comment_text:
                continue

            comments.append(comment_text)

            # Label
            label: Optional[bool] = None
            if minority_col is not None:
                raw_label = row.get(minority_col, "")
                if isinstance(raw_label, str):
                    label = detect_is_minority(raw_label, true_keys, false_keys, default_label)
                else:
                    label = default_label
            else:
                label = default_label

            labels.append(label)

    return comments, labels


def write_json(output_path: Path, question: str, comments: List[str], labels: List[Optional[bool]]) -> None:
    output_comments = []
    for i, c in enumerate(comments):
        obj = {"index": i, "comment": c}
        lbl = labels[i] if i < len(labels) else None
        if lbl is not None:
            obj["is_minority"] = bool(lbl)
        else:
            obj["is_minority"] = None
        output_comments.append(obj)

    output = {
        "question": question,
        "comments": output_comments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def infer_default_output(input_csv: Path, output_dir: Path) -> Path:
    # Keep the base name but change extension to .json
    return output_dir / (input_csv.stem + ".json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert minority CSV to target JSON schema with is_minority flag (per-row if column provided).")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="", help="Path to output JSON (defaults to datasets/minority/<name>.json)")
    parser.add_argument("--question", type=str, required=True, help="Question/topic string to store in JSON")
    parser.add_argument("--comment-column", type=str, default="", help="CSV column containing comment text (auto-detect if omitted)")
    parser.add_argument("--minority-column", type=str, default="", help="CSV column containing self-assessed majority/minority selection")
    parser.add_argument("--minority-true-values", type=str, default=",".join(DEFAULT_MINORITY_TRUE_PATTERNS), help="Comma-separated substrings to detect as minority=true (case-insensitive)")
    parser.add_argument("--minority-false-values", type=str, default=",".join(DEFAULT_MINORITY_FALSE_PATTERNS), help="Comma-separated substrings to detect as minority=false (case-insensitive)")
    parser.add_argument("--default-minority", type=str, default="", choices=["", "true", "false"], help="Default label if detection fails (empty means None)")
    parser.add_argument("--all-minority", action="store_true", help="Override: mark all as minority=true")
    parser.add_argument("--all-not-minority", action="store_true", help="Override: mark all as minority=false")
    args = parser.parse_args()

    input_csv = Path(args.input).resolve()
    if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}")
        sys.exit(1)

    # Determine output path
    default_out_dir = Path("/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/minority").resolve()
    output_path = Path(args.output).resolve() if args.output else infer_default_output(input_csv, default_out_dir)

    # Determine default label
    default_label: Optional[bool]
    if args.all_minority:
        default_label = True
    elif args.all_not_minority:
        default_label = False
    else:
        if args.default_minority == "true":
            default_label = True
        elif args.default_minority == "false":
            default_label = False
        else:
            default_label = None

    # Parse mappings
    true_keys = parse_mapping(args.minority_true_values)
    false_keys = parse_mapping(args.minority_false_values)

    # Read comments and labels
    comments, labels = read_comments_and_labels_from_csv(
        csv_path=input_csv,
        comment_column=(args.comment_column or None),
        minority_column=(args.minority_column or None),
        true_keys=true_keys,
        false_keys=false_keys,
        default_label=default_label,
    )

    if not comments:
        print("No comments found in CSV. Nothing to write.")
        sys.exit(1)

    # Write JSON
    write_json(output_path, args.question, comments, labels)
    print(f"Wrote {len(comments)} comments to {output_path}")


if __name__ == "__main__":
    main()

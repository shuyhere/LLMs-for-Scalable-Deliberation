#!/usr/bin/env python3
"""
Process Polis-style JSON datasets under datasets/polis/* into filtered JSON outputs.

- For each subfolder in datasets/polis/ that contains conversation.json and comments.json:
  - Read conversation.json to get the description as "question"; fallback to topic if description missing.
  - Read comments.json (array of objects) and keep those where agree_count + disagree_count >= 50.
  - Build output with structure:
      {
        "question": <str>,
        "comments": [ {"index": <int>, "comment": <str>}, ... ]
      }
  - Write to datasets/<folder>.json

No external dependencies; uses Python stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
POLIS_ROOT = REPO_ROOT / "datasets" / "polis"
OUTPUT_DIR = REPO_ROOT / "datasets"

REQUIRED_FILES = ("conversation.json", "comments.json")


def read_question(conversation_path: Path) -> str:
    if not conversation_path.exists():
        raise FileNotFoundError(f"conversation.json not found: {conversation_path}")
    with conversation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Prefer description; fallback to topic
    question = (data.get("description") or data.get("topic") or "").strip()
    if not question:
        raise ValueError(f"Missing 'description' and 'topic' in {conversation_path}")
    return question


def load_and_filter_comments(comments_path: Path) -> List[str]:
    if not comments_path.exists():
        raise FileNotFoundError(f"comments.json not found: {comments_path}")
    with comments_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"comments.json must be a list: {comments_path}")

    kept: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        agrees = int(item.get("agree_count") or 0)
        disagrees = int(item.get("disagree_count") or 0)
        if agrees + disagrees < 15:
            continue
        text = (item.get("txt") or item.get("text") or item.get("comment") or "").strip()
        if text:
            kept.append(text)
    return kept


def build_output_json(question: str, comments: List[str]) -> Dict[str, object]:
    return {
        "question": question,
        "comments": [{"index": i, "comment": c} for i, c in enumerate(comments)],
    }


def process_folder(folder_path: Path) -> Tuple[Path, int]:
    conversation_path = folder_path / "conversation.json"
    comments_path = folder_path / "comments.json"

    question = read_question(conversation_path)
    comments = load_and_filter_comments(comments_path)

    out_path = OUTPUT_DIR / f"{folder_path.name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(build_output_json(question, comments), f, ensure_ascii=False, indent=2)

    return out_path, len(comments)


def main() -> None:
    if not POLIS_ROOT.exists():
        raise FileNotFoundError(f"Polis root not found: {POLIS_ROOT}")

    results: List[Tuple[str, Path, int]] = []
    for child in sorted(POLIS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if all((child / name).exists() for name in REQUIRED_FILES):
            out_path, kept = process_folder(child)
            results.append((child.name, out_path, kept))

    for folder, out_path, kept in results:
        print(f"{folder}: wrote {kept} comments -> {out_path}")


if __name__ == "__main__":
    main()
